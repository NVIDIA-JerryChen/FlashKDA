"""Paired prepared dense-decode API and pinned FlashInfer T=1 kernel timing."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
import torch
from flash_kda import prepare_consistent_decode
from tests.consistency.flashinfer_reference import compiled_reference


def measure(calls, reset, samples, repeats):
    for name, call in calls.items():
        reset(name)
        for _ in range(3):
            call()
    torch.cuda.synchronize()
    values = {n: dict(gpu_us=[], wall_us=[]) for n in calls}
    begin, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    for sample in range(samples):
        order = list(calls) if sample % 2 == 0 else list(calls)[::-1]
        for name in order:
            reset(name)
            torch.cuda.synchronize()
            start = time.perf_counter()
            begin.record()
            for _ in range(repeats):
                calls[name]()
            end.record()
            end.synchronize()
            values[name]["gpu_us"].append(
                begin.elapsed_time(end) * 1000 / (repeats * 64)
            )
            values[name]["wall_us"].append(
                (time.perf_counter() - start) * 1e6 / (repeats * 64)
            )
    return {
        n: {
            k: dict(median=statistics.median(x), min=min(x), max=max(x), samples=x)
            for k, x in v.items()
        }
        for n, v in values.items()
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--cases", default="6:1,6:64,12:16,12:32,12:64,12:128,16:1,16:64,32:1,32:64"
    )
    p.add_argument("--samples", type=int, default=9)
    p.add_argument("--repeats", type=int, default=25)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20261105)
    fi = compiled_reference(
        torch.cuda.current_device(), torch.cuda.get_device_capability()
    )
    dummy = torch.zeros(1, device="cuda", dtype=torch.int32)
    results = []
    for h, b in [tuple(map(int, x.split(":"))) for x in args.cases.split(",")]:
        print(
            json.dumps(dict(phase="case_start", h=h, b=b, wall=time.time())), flush=True
        )

        def rand(shape, dtype=torch.bfloat16):
            return torch.randn(shape, device="cuda", dtype=dtype)

        xs = [
            [*[rand((b, 1, h, 128)) for _ in range(4)], rand((b, 1, h))]
            for _ in range(64)
        ]
        a = torch.zeros(h, device="cuda")
        bias = rand((h, 128), torch.float32)
        initial = (rand((b, h, 128, 128), torch.float32) * 0.1).bfloat16()
        states = {n: initial.clone() for n in ("decode", "flashinfer")}
        outputs = {n: [torch.empty_like(xs[0][0]) for _ in xs] for n in states}
        plans = [
            prepare_consistent_decode(
                *x, a, bias, states["decode"], outputs["decode"][i], states["decode"]
            )
            for i, x in enumerate(xs)
        ]
        fiargs = [
            (
                *x,
                states["flashinfer"],
                states["flashinfer"],
                dummy,
                outputs["flashinfer"][i],
                a,
                bias.view(-1),
                dummy,
                dummy,
                dummy,
                128**-0.5,
                1e-6,
                -5.0,
            )
            for i, x in enumerate(xs)
        ]
        for i in range(64):
            plans[i].run()
            fi(*fiargs[i])
            for actual, expected in [
                (outputs["decode"][i], outputs["flashinfer"][i]),
                (states["decode"], states["flashinfer"]),
            ]:
                assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
                assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))

        def decode(plans=plans):
            for plan in plans:
                plan.run()

        def baseline(fiargs=fiargs):
            for values in fiargs:
                fi(*values)

        calls = dict(decode=decode, flashinfer=baseline)
        graphs = {}
        audits = {}
        for name, call in calls.items():
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                call()
            graph.instantiate()
            path = args.output / f"h{h}-b{b}-{name}.dot"
            graph.debug_dump(str(path))
            dot = path.read_text()
            audit = dict(
                kernel_nodes=len(re.findall(r'label="\{\s*KERNEL\b', dot)),
                memcpy_nodes=len(re.findall(r'label="\{\s*MEMCPY\b', dot)),
            )
            assert audit == dict(kernel_nodes=64, memcpy_nodes=0), audit
            graphs[name] = graph
            audits[name] = audit

        def reset(name, states=states, initial=initial):
            states[name].copy_(initial)

        graph_times = measure(
            {n: g.replay for n, g in graphs.items()}, reset, args.samples, args.repeats
        )
        eager_times = measure(calls, reset, args.samples, 5)
        result = dict(
            h=h,
            b=b,
            graph=graph_times,
            eager=eager_times,
            audit=audits,
            bitwise_steps=64,
            ratio=graph_times["decode"]["gpu_us"]["median"]
            / graph_times["flashinfer"]["gpu_us"]["median"],
        )
        (args.output / f"h{h}-b{b}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        results.append(result)
        print(
            json.dumps(
                dict(
                    phase="case_complete",
                    h=h,
                    b=b,
                    ratio=result["ratio"],
                    times={n: v["gpu_us"]["median"] for n, v in graph_times.items()},
                    wall=time.time(),
                )
            ),
            flush=True,
        )
        del graphs, plans, fiargs, xs, outputs, states
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    import importlib.metadata, socket

    provenance = dict(
        gpu=torch.cuda.get_device_name(),
        host=socket.gethostname(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cutlass=importlib.metadata.version("nvidia-cutlass-dsl"),
        reference="c3af0c13328fd23b256c53b9b4f94d9e160d118c",
        kernel_sha256=hashlib.sha256(
            (
                Path(__file__).resolve().parents[1]
                / "flash_kda/_consistent_decode_kernel.py"
            ).read_bytes()
        ).hexdigest(),
    )
    (args.output / "summary.json").write_text(
        json.dumps(
            dict(
                environment=provenance,
                records=results,
                protocol="Fixed-address warm-cache medians, alternating order; 64 T1 API calls/cycle, no cross-token fusion. Prevalidated bound buffers and compilation outside timing. No cold-L2 claim. Eager compares prepared plan.run with precompiled FI kernel, not FI package dispatcher.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
