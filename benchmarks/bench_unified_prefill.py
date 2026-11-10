"""Paired complete-API Graph timing for explicit packed varlen lengths."""

import argparse
import itertools
import json
import statistics
import torch
from flash_kda import prepare_unified_prefill


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", default="0,128,1024,4096")
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--workspace-mib", type=int, default=256)
    parser.add_argument(
        "--flashkda",
        action="store_true",
        help="Also time the installed original FlashKDA extension",
    )
    parser.add_argument("--samples", type=int, default=9)
    args = parser.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    if (
        not lengths
        or min(lengths) < 0
        or sum(lengths) == 0
        or args.heads < 1
        or args.samples < 1
    ):
        parser.error(
            "require nonnegative lengths with positive total, heads and samples"
        )
    torch.manual_seed(7)
    total, h = sum(lengths), args.heads
    xs = [
        torch.randn(1, total, h, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(4)
    ]
    beta = torch.randn(1, total, h, device="cuda", dtype=torch.bfloat16)
    a, bias = torch.zeros(h, device="cuda"), torch.zeros(h, 128, device="cuda")
    state = torch.zeros(len(lengths), h, 128, 128, device="cuda", dtype=torch.bfloat16)
    out, final = torch.empty_like(xs[0]), torch.empty_like(state)
    cu = torch.tensor(
        [0] + list(itertools.accumulate(lengths)), device="cuda", dtype=torch.int32
    )
    plan = prepare_unified_prefill(
        *xs,
        beta,
        a,
        bias,
        state,
        out,
        final,
        cu_seqlens=cu,
        workspace_limit_bytes=args.workspace_mib * 1024**2
    )
    calls = {"recurrent": plan.run}
    outputs = [(out, final)]
    if args.flashkda:
        from flash_kda_C import fwd, get_workspace_size

        other, other_state = torch.empty_like(out), torch.empty_like(final)
        workspace = torch.empty(
            get_workspace_size(total, h, len(lengths)), device="cuda", dtype=torch.uint8
        )
        cu64 = cu.long()

        def original():
            fwd(
                *xs,
                beta,
                128**-0.5,
                other,
                workspace,
                a,
                bias,
                -5.0,
                initial_state=state,
                final_state=other_state,
                cu_seqlens=cu64
            )

        calls["flashkda"] = original
        outputs.append((other, other_state))
    graphs = {}
    for name, fn in calls.items():
        fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graphs[name] = graph
        for _ in range(3):
            graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in graphs}
    names = list(graphs)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    for i in range(args.samples):
        for name in names[i % len(names) :] + names[: i % len(names)]:
            start.record()
            for _ in range(3):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / 3)
    for pair in outputs:
        assert all(bool(torch.isfinite(t).all()) for t in pair)
    print(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                lengths=lengths,
                heads=h,
                dtype="bf16",
                initial_state="zero; no historical prefix generation timed",
                timing="complete GPU API, hot CUDA Graph; preparation excluded",
                workspace_bytes=plan.workspace_bytes,
                samples_us=samples,
                median_us={name: statistics.median(v) for name, v in samples.items()},
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
