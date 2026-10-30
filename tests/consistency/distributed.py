"""torchrun --standalone --nproc-per-node=8 -m tests.consistency.distributed."""
import argparse
import datetime
import json
import os
import time
import torch
import torch.distributed as dist
from flash_kda.continuation import cp_consistent_fwd, Continuation, rebase_continuations
from .helpers import make_case, run, select, assert_bits


def check(cp, group, rank, h, lengths):
    case = make_case(lengths, h)
    expected, expected_state = run(case)
    spans = [(n * rank // cp, n * (rank + 1) // cp) for n in lengths]
    local, lens = select(case, lengths, spans)
    out = torch.empty_like(local["q"])
    state = torch.empty_like(local["state_in"])
    cp_consistent_fwd(**local, out=out, state_out=state, group=group)
    selected, _ = select(dict(case, q=expected), lengths, spans)
    assert_bits(out, selected["q"], label=f"CP={cp} rank={rank} output", lengths=lens)
    prefix, _ = select(case, lengths, [(0, end) for _, end in spans])
    _, prefix_state = run(prefix)
    assert_bits(state, prefix_state, label=f"CP={cp} rank={rank} boundary")
    # Exercise the actual PD/CP boundary around a captured local graph.
    from flash_kda import consistent_fwd
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        consistent_fwd(**local, out=out, state_out=state)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        consistent_fwd(**local, out=out, state_out=state)
    local["state_in"].copy_(case["state_in"])
    if rank:
        dist.recv(local["state_in"], src=rank-1, group=group)
    graph.replay()
    if rank + 1 < cp:
        dist.send(state, dst=rank+1, group=group)
    assert_bits(out, selected["q"], label=f"CP={cp} graph output", lengths=lens)
    assert_bits(state, prefix_state, label=f"CP={cp} graph boundary")
    if rank == cp - 1:
        assert_bits(state, expected_state, label=f"CP={cp} final")
        order = [3, 1, 0, 2]
        context = Continuation(tuple(range(4)), tuple(lengths), state)
        rebased = rebase_continuations([context], order, torch.empty_like(state))
        decode = make_case([1]*4, h, seed=2026)
        decode["state_in"] = expected_state
        expected_decode, expected_decode_state = run(decode)
        moved, _ = select(decode, [1]*4, [(0, 1)]*4, order, state=rebased.state)
        actual_decode, actual_state = run(moved)
        assert_bits(actual_decode, expected_decode[:, order], label="CP to PD decode", lengths=[1]*4)
        assert_bits(actual_state, expected_decode_state[order], label="CP to PD state")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", nargs="+", type=int, default=[12, 16, 32])
    parser.add_argument("--cp-sizes", nargs="+", type=int, choices=[1, 2, 4, 8], default=[1, 2, 4, 8])
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init = {}
    if os.getenv("FLASH_KDA_FILE_STORE"):
        init = dict(init_method="file://" + os.environ["FLASH_KDA_FILE_STORE"],
                    rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=5), device_id=torch.device("cuda", torch.cuda.current_device()), **init)
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        for cp in args.cp_sizes:
            if cp > world:
                raise RuntimeError(f"CP={cp} needs at least {cp} ranks; got {world}. Full acceptance requires CP=8.")
            group = dist.new_group(list(range(cp)), backend="nccl")
            for h in args.heads:
                for lengths in ([0, 1, 17, 129], [0, 0, 0, 0], [1, 2, 0, 0]):
                    start = time.monotonic()
                    if rank < cp:
                        check(cp, group, rank, h, lengths)
                    dist.barrier()
                    if rank == 0:
                        print(json.dumps(dict(cp=cp, h=h, lengths=lengths, status="PASS",
                                              seconds=time.monotonic()-start)), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
