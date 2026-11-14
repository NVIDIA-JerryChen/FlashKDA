"""Time the complete prepared FP32 API with fixed-address CUDA Graph replay."""

import argparse
import itertools
import json
import statistics

import torch
from flash_kda import prepare_fp32_decode, prepare_fp32_prefill


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["decode", "prefill"], default="prefill")
    parser.add_argument("--lengths", default="0,128,1024,4096")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--query-length", type=int, choices=[1, 8], default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--workspace-mib", type=int, default=256)
    args = parser.parse_args()
    lengths = (
        [int(s) for s in args.lengths.split(",")]
        if args.mode == "prefill"
        else [args.query_length] * args.batch
    )
    if (
        not lengths
        or min(lengths) < 0
        or sum(lengths) == 0
        or args.heads < 1
        or args.samples < 1
        or args.workspace_mib < 1
    ):
        parser.error(
            "require nonnegative lengths, positive total/heads/samples/workspace"
        )
    torch.manual_seed(7)
    b, h, total = len(lengths), args.heads, sum(lengths)
    shape = (
        (1, total, h, 128) if args.mode == "prefill" else (b, args.query_length, h, 128)
    )
    xs = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    beta = torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16)
    a, bias = torch.zeros(h, device="cuda"), torch.zeros(h, 128, device="cuda")
    initial = torch.randn(b, h, 128, 128, device="cuda", dtype=torch.float32) * 0.1
    output = torch.empty_like(xs[0])
    saved = initial.clone()
    if args.mode == "prefill":
        terminal = torch.empty_like(initial)
        cu = torch.tensor(
            [0] + list(itertools.accumulate(lengths)), device="cuda", dtype=torch.int32
        )
        plan = prepare_fp32_prefill(
            *xs,
            beta,
            a,
            bias,
            initial,
            output,
            terminal,
            cu_seqlens=cu,
            workspace_limit_bytes=args.workspace_mib * 1024**2,
        )
        written = [output, terminal]
    elif args.query_length == 1:
        terminal = torch.empty_like(initial)
        plan = prepare_fp32_decode(*xs, beta, a, bias, initial, output, terminal)
        written = [output, terminal]
    else:
        checkpoints = torch.empty(b, 8, h, 128, 128, device="cuda", dtype=torch.float32)
        plan = prepare_fp32_decode(
            *xs, beta, a, bias, initial, output, checkpoint_states=checkpoints
        )
        written = [output, checkpoints]
    for _ in range(3):
        plan.run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run()
    samples = []
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    for i in range(args.samples + 3):
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        if i >= 3:
            samples.append(start.elapsed_time(end) * 100)
    assert torch.equal(initial.view(torch.int32), saved.view(torch.int32))
    assert all(torch.isfinite(t).all() for t in written)
    print(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                mode=args.mode,
                lengths=lengths,
                heads=h,
                state_dtype="float32",
                workspace_bytes=plan.workspace_bytes,
                scope="complete prepared GPU API; fixed-address Graph; prepare/prefix/acceptance excluded",
                median_us=statistics.median(samples),
                samples_us=samples,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
