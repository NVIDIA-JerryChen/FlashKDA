"""Wall time of one graph-captured decode step of the KDA layer chain: six batch-invariant linears, three short convolutions and the CF-BI decode call.

usage: python benchmarks/bench_layer_chain.py [--shapes N ...] [--dm D] [--heads H] [--impl N] [--trials T] [--calls C] [--profile-calls C] [--calls-scale S]
           [--cells SUBSTR ...] [--expect-compute N] [--expect-aux NAME,NAME] [--json PATH] [--label TEXT] [--list-cells]

The step is the layer of tests/cfbi/test_layer_chain.py (q / k / v / low-rank g / beta projections, the three depthwise convolutions with SiLU and the decode
core) on a pre-allocated path: every output buffer exists before the capture and no fp32 side output is requested, so the graph holds no zero-fill or other
auxiliary node. The tool lists the nodes of one replay from the CUDA profiler, splits them into compute kernels (the CF-BI kernels: 6 linear + 3 convolution
+ the decode kernels, 11 with the two-launch decode and 10 with the single-launch decode) and auxiliary nodes (memsets, copies, anything else), records both
lists and asserts them with ``--expect-compute`` and ``--expect-aux`` (the explicit list of the auxiliary nodes that may remain; none by default).

The graph is replayed in loops (the position of the core state advances with the replays, so a trial covers all 64 positions equally). One cell per shape; a
trial takes ``gpu_graph_us`` (CUDA events around the loop of replays), ``cpu_graph_us`` (CPU clock around the loop plus the final synchronisation) and one
profiled replay for the kernel sum ``total_us`` (the same records and gating as bench_decode.py; the record declares per-process-median samples).
Whole-step graph time is the metric that sees a shared-memory carve-out switch between two kernels of the chain, which an isolated decode call cannot show.

The tool measures the flash_kda of its own source tree; CFBI_BENCH_SRC=<dir> measures the flash_kda of another tree (see bench_decode.py).
"""
import argparse
import datetime
import gc
import os
import socket
import sys
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("CFBI_BENCH_SRC") or TOOL_ROOT).resolve()
for _p in (TOOL_ROOT / "tests" / "cfbi", TOOL_ROOT / "benchmarks", ROOT, ROOT / "tests" / "cfbi"):
    sys.path.insert(0, str(_p))

import bench_decode as bd  # noqa: E402  (loop_trial, device_info, trial_wall, defaults)
import bench_records as records  # noqa: E402

HD = 128   # head dimension (K = V)
W = 4      # convolution width
LB = -5.0
SHAPES = (1, 64)
DEFAULT_DM = 2048
DEFAULT_H = 12
DEFAULTS = {"trials": 5, "calls": 1536, "profile_calls": 16, "calls_scale": 1}


# ---- cells and node lists (pure Python, no torch) ---------------------------------------------------------------------------------------------

def cell_specs(shapes=SHAPES, dm=DEFAULT_DM, heads=DEFAULT_H):
    return [{"N": n, "H": heads, "Dm": dm, "key": records.cell_key("chain", N=n, H=heads, Dm=dm, regime="lockstep")} for n in sorted(shapes)]


def select_cells(specs, substrings):
    return [s for s in specs if not substrings or any(sub in s["key"] for sub in substrings)]


def calls_per_trial(opts):
    """Replays of a trial: a whole number of 64-position cycles of the core state; the re-run scale multiplies them."""
    return -(-(opts["calls"] * opts["calls_scale"]) // 64) * 64


def split_nodes(counts):
    """``(compute, aux)`` from ``{raw kernel name: launches per replay}``: compute kernels are the CF-BI kernels (``{base name: count}``), everything else
    (memsets, copies, ATen kernels) is auxiliary."""
    compute, aux = {}, {}
    for name, n in counts.items():
        if records.is_helper_kernel(name):
            aux[name.strip()] = aux.get(name.strip(), 0) + n
        else:
            base = records.kernel_base(name)
            compute[base] = compute.get(base, 0) + n
    return compute, aux


def check_nodes(compute, aux, expect_compute, expect_aux):
    """None when the replay holds the expected nodes, otherwise the message: the number of compute kernels (``expect_compute``, None = no assertion) and the exact
    set of auxiliary node names (``expect_aux``: a list, None = no assertion; the empty list asserts that there is none)."""
    n = int(round(sum(compute.values())))
    if expect_compute is not None and n != expect_compute:
        return "the step launches %d compute kernels %s, expected %d" % (n, dict(sorted(compute.items())), expect_compute)
    if expect_aux is not None and sorted(aux) != sorted(expect_aux):
        return "the graph holds the auxiliary nodes %s, expected %s" % (sorted(aux), sorted(expect_aux))
    return None


def run_meta(a, started, opts, device, torch_mod):
    commit, dirty, tree = records.tree_identity(ROOT)
    return {"commit": commit, "dirty": dirty, "tree": tree, "gpu": device["name"], "sm_count": device["sm_count"], "cc": device["cc"], "host": socket.gethostname(),
            "torch": torch_mod.__version__, "cuda": torch_mod.version.cuda, "timestamp": started, "kind": "chain-wall", "sample_stat": records.PROCESS_MEDIAN,
            "impl": a.impl, "env_impl": os.environ.get("FLASH_KDA_CFBI_IMPL"), "dm": a.dm, "heads": a.heads, "trials": opts["trials"], "calls_per_trial": calls_per_trial(opts),
            "profile_calls": opts["profile_calls"], "calls_scale": opts["calls_scale"], "cells_filter": a.cells, "label": a.label, "tag": os.environ.get("CFBI_BENCH_TAG"),
            "build_dir": os.environ.get("FLASH_KDA_CFBI_BUILD_DIR"), "src": str(ROOT), "argv": sys.argv[1:]}


# ---- measurement (needs a GPU) ----------------------------------------------------------------------------------------------------------------

class ChainEnv:
    """The layer weights, the input rows, the pre-allocated outputs and the states of one shape; ``step`` is one decode step of the layer."""

    def __init__(self, N, Dm, H, impl):
        import math

        import torch
        from flash_kda.cfbi import alloc_conv_state, alloc_state, bi_linear, decode, short_conv

        self.torch, self.N, self.H, self.impl = torch, N, H, impl
        self._bi_linear, self._decode, self._short_conv = bi_linear, decode, short_conv
        kd = H * HD
        g = torch.Generator(device="cuda")
        g.manual_seed(0)

        def rb(*shape, scale):
            return (torch.randn(*shape, device="cuda", generator=g) * scale).to(torch.bfloat16)

        s = 1.0 / math.sqrt(Dm)
        self.wq, self.wk, self.wv = rb(kd, Dm, scale=s), rb(kd, Dm, scale=s), rb(kd, Dm, scale=s)
        self.f1, self.f2, self.bw = rb(HD, Dm, scale=s), rb(kd, HD, scale=1.0 / math.sqrt(HD)), rb(H, Dm, scale=s)
        self.cq, self.ck, self.cv = (torch.randn(kd, W, device="cuda", generator=g) * 0.6 for _ in range(3))
        self.A_log, self.dt_bias, self.scale = torch.rand(H, device="cuda", generator=g), torch.rand(H, HD, device="cuda", generator=g), 1.0 / math.sqrt(HD)
        self.x = torch.randn(N, Dm, device="cuda", generator=g).to(torch.bfloat16)
        bf = torch.bfloat16
        self.pq, self.pk, self.pv = (torch.empty(N, kd, dtype=bf, device="cuda") for _ in range(3))
        self.f1o, self.go, self.beta = torch.empty(N, HD, dtype=bf, device="cuda"), torch.empty(N, kd, dtype=bf, device="cuda"), torch.empty(N, H, dtype=bf, device="cuda")
        self.q, self.k, self.v = (torch.empty(N, 1, kd, dtype=bf, device="cuda") for _ in range(3))
        self.out = torch.empty(N, 1, H, HD, dtype=bf, device="cuda")
        self.conv = [alloc_conv_state(N, kd, W) for _ in range(3)]
        self.core = alloc_state(N, H)
        self.graph = None

    def step(self):
        N, H = self.N, self.H
        lin, conv = self._bi_linear, self._short_conv
        lin(self.x, self.wq, out=self.pq)
        lin(self.x, self.wk, out=self.pk)
        lin(self.x, self.wv, out=self.pv)
        lin(self.x, self.f1, out=self.f1o)
        lin(self.f1o, self.f2, out=self.go)
        lin(self.x, self.bw, out=self.beta)
        conv(self.pq.view(N, 1, -1), self.cq, None, self.conv[0], "silu", None, out=self.q)
        conv(self.pk.view(N, 1, -1), self.ck, None, self.conv[1], "silu", None, out=self.k)
        conv(self.pv.view(N, 1, -1), self.cv, None, self.conv[2], "silu", None, out=self.v)
        self._decode(self.q.view(N, 1, H, HD), self.k.view(N, 1, H, HD), self.v.view(N, 1, H, HD), self.go.view(N, 1, H, HD), self.beta.view(N, 1, H), self.scale,
                     self.out, self.A_log, self.dt_bias, LB, self.core, _impl=self.impl)

    def capture(self):
        torch = self.torch
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.step()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.step()

    def replay(self, i=0):
        self.graph.replay()


def measure_cell(spec, env, opts, expect_compute, expect_aux):
    from bench_breakdown import kernel_times

    torch = env.torch
    calls = calls_per_trial(opts)
    for i in range(100):
        env.replay(i)
    torch.cuda.synchronize()
    runs, walls, counts = [], [], {}
    gc.collect()
    gc.disable()
    try:
        for _ in range(opts["trials"]):
            gpu_g, cpu_g = bd.loop_trial(torch, env.replay, calls)
            times, counts = kernel_times(env.replay, iters=opts["profile_calls"], warm=1)
            runs.append(times)
            walls.append({"gpu_graph_us": gpu_g, "cpu_graph_us": cpu_g})
    finally:
        gc.enable()
    compute, aux = split_nodes(counts)
    problem = check_nodes(compute, aux, expect_compute, expect_aux)
    if problem:
        raise SystemExit("cell %s: %s" % (spec["key"], problem))
    args = {"N": spec["N"], "H": spec["H"], "Dm": spec["Dm"], "regime": "lockstep", "impl": env.impl, "calls_per_trial": calls, "profile_calls": opts["profile_calls"],
            "compute_nodes": {k: round(v, 3) for k, v in sorted(compute.items())}, "compute_kernels": int(round(sum(compute.values()))),
            "aux_nodes": {k: round(v, 3) for k, v in sorted(aux.items())}}
    return records.make_cell("chain", spec["key"], args, runs, drop_helpers=False, wall=walls)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", nargs="+", type=int, default=list(SHAPES), help="numbers of sequences N (default 1 64)")
    ap.add_argument("--dm", type=int, default=DEFAULT_DM, help="model dimension of the layer (default 2048)")
    ap.add_argument("--heads", type=int, default=DEFAULT_H)
    ap.add_argument("--impl", type=int, default=0)
    ap.add_argument("--trials", type=int, default=DEFAULTS["trials"])
    ap.add_argument("--calls", type=int, default=DEFAULTS["calls"])
    ap.add_argument("--profile-calls", type=int, default=DEFAULTS["profile_calls"])
    ap.add_argument("--calls-scale", type=int, default=DEFAULTS["calls_scale"])
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--expect-compute", type=int, default=None, help="number of compute kernels one step launches (11 with the two-launch decode, 10 with the single launch)")
    ap.add_argument("--expect-aux", default=None, help="comma-separated names of the auxiliary nodes that may remain ('' asserts that there is none)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--list-cells", action="store_true")
    a = ap.parse_args(argv)
    specs = select_cells(cell_specs(a.shapes, a.dm, a.heads), a.cells)
    if a.list_cells:
        for s in specs:
            print(s["key"])
        return 0
    if not specs:
        print("no cell selected", file=sys.stderr)
        return 2
    opts = {"trials": a.trials, "calls": a.calls, "profile_calls": a.profile_calls, "calls_scale": a.calls_scale}
    expect_aux = None if a.expect_aux is None else [x for x in a.expect_aux.split(",") if x]

    import torch

    device = bd.device_info(torch)
    started = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    print("device %s  impl=%s  %d cell(s)  trials=%d  replays/trial %d" % (device["name"], a.impl, len(specs), opts["trials"], calls_per_trial(opts)), flush=True)
    cells = []
    for spec in specs:
        torch.cuda.empty_cache()
        env = ChainEnv(spec["N"], spec["Dm"], spec["H"], a.impl)
        env.capture()
        cell = measure_cell(spec, env, opts, a.expect_compute, expect_aux)
        cells.append(cell)
        print(bd.cell_line(spec, cell) + "   compute kernels %d, auxiliary nodes %d" % (cell["args"]["compute_kernels"], len(cell["args"]["aux_nodes"])), flush=True)
        del env
    if a.json:
        records.write_record(a.json, run_meta(a, started, opts, device, torch), cells)
        print("record written to %s" % a.json, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
