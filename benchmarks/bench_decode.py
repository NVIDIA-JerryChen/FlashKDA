"""Wall time of the CF-BI decode call: CUDA-graph replay and eager loop, in four regimes.

usage: python benchmarks/bench_decode.py [--impl N] [--shapes N,H ...] [--regimes lockstep staggered cold probe] [--probe-shapes N,H ...] [--positions P ...]
           [--trials T] [--calls C] [--cold-calls C] [--probe-calls C] [--profile-calls C] [--calls-scale S] [--cells SUBSTR ...]
           [--expect-compute MAPPING | --expect-kernels NAME,NAME] [--json PATH] [--label TEXT] [--list-cells]

Regimes (one cell per shape, the probe regime one cell per shape and position):
  lockstep   all sequences at the same ``pos``; the calls of a trial advance ``pos`` one by one, so a trial covers all 64 positions equally
  staggered  sequence n starts at ``pos0 = n mod 64`` (one in 64 sequences is at a chunk end in every call) and advances with the calls
  cold       lock-step positions, L2 evicted before every call by writing a buffer of at least twice the L2 size
  probe      every sequence at ``pos`` p before every call (positions 0, 1, 15, 16, 31, 32, 47, 48, 62, 63 at N = 1 and N = 64 by default)

One trial of a cell measures, with the same inputs, an eager loop (the call as a user makes it) and a replay of a CUDA graph that holds one call, and
takes one profiled pass for the kernel sum (one warm-up call, then ``profile_calls`` calls, a whole number of 64-position cycles in the regimes whose position
advances with the calls, so that every position phase enters the kernel sum equally often). ``--calls-scale S`` multiplies the (rounded) calls per trial AND
the profiled calls, so that a re-run has exactly S times the calls of the first campaign in both; the cell records the effective counts. Per trial and cell the sample carries
  gpu_eager_us, gpu_graph_us   GPU-elapsed time per call from CUDA events. lockstep / staggered: one pair of events around a loop of calls (no helper runs in
                               between). cold / probe: one pair of events around every single call; the helper (eviction write, ``pos`` rewrite) is enqueued and
                               synchronised before the first event, so it is not in the metric.
  cpu_eager_us, cpu_graph_us   CPU monotonic-clock time per call of the loop including the synchronisation at its end (host work included); lockstep / staggered only.
  total_us                     sum of the CF-BI kernel durations of one call from the CUDA profiler; the helper kernels are recognised by name and left out.
The record (schema in benchmarks/bench_records.py) declares ``sample_stat = process_median``: the comparison takes the median of the trials of every cell of every
record file as one sample, so at least five trials per cell are needed and at least three record files per build (interleaved A / B runs of this tool).

The tool measures the flash_kda of its own source tree. CFBI_BENCH_SRC=<dir> measures the flash_kda of another tree (a git archive snapshot of a baseline
commit, for example) with this tool; the record then carries that tree's commit.

Launch assertion: every trial's profiled pass yields the exact mapping ``{kernel: launches per call}`` of the CF-BI kernels (the helper kernels of a regime, an eviction write or
a ``pos`` rewrite, are listed apart as auxiliary nodes), the same split and check as the layer chain's (``bench_records.split_nodes`` / ``check_nodes``). The mapping is recorded in
every sample (``nodes``) and in the cell (``compute_nodes``, ``aux_nodes``, ``compute_kernels``), every trial must carry the cell's mapping, and with ``--expect-compute`` every trial must
equal the asserted one (``single-launch``: ``rows_kernel`` only; ``two-launch``: ``plan_kernel`` and ``rows_kernel``; ``reference``: ``plan_kernel`` and ``ref::rows_kernel``; or
``NAME=COUNT;NAME=COUNT``): a side of an A/B that silently runs another path in any trial fails instead of measuring it. ``--expect-kernels a,b`` is the shorthand for "every name once".
The names carry no template arguments (the profiler pass strips them), so the two paths are told apart by the plan kernel. ``cell_node_problems`` checks a recorded cell.
"""
import argparse
import datetime
import gc
import os
import socket
import statistics
import sys
import time
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parents[1]  # the tree that holds this script and bench_records.py
ROOT = Path(os.environ.get("CFBI_BENCH_SRC") or TOOL_ROOT).resolve()  # the tree whose flash_kda is measured
for _p in (TOOL_ROOT / "tests" / "cfbi", TOOL_ROOT / "benchmarks", ROOT, ROOT / "tests" / "cfbi"):
    sys.path.insert(0, str(_p))

import bench_records as records  # noqa: E402

LB = -5.0
STEPS = 64
PROFILE_WARM = 1                                   # calls before the profiled pass; the positions are set right before them, so the first profiled call is at phase 1
MATRIX_D = ((1, 12), (4, 12), (16, 12), (64, 12), (128, 12), (256, 12))
PROBE_SHAPES = ((1, 12), (64, 12))
PROBE_POSITIONS = (0, 1, 15, 16, 31, 32, 47, 48, 62, 63)
REGIMES = ("lockstep", "staggered", "cold", "probe")
LOOP_REGIMES = ("lockstep", "staggered")           # no helper between the calls: loop bracketing, CPU clock available
DEFAULTS = {"trials": 5, "calls": 1536, "cold_calls": 256, "probe_calls": 256, "profile_calls": 64, "calls_scale": 1}
CANONICAL_COMPUTE = {"single-launch": {"rows_kernel": 1}, "two-launch": {"plan_kernel": 1, "rows_kernel": 1}, "reference": {"plan_kernel": 1, "ref::rows_kernel": 1}}   # one decode call's launches


# ---- cells and trials (pure Python, no torch) ------------------------------------------------------------------------------------------------

def cell_specs(shapes=MATRIX_D, regimes=REGIMES, probe_shapes=PROBE_SHAPES, positions=PROBE_POSITIONS):
    """The cells of a run in measurement order (grouped by shape): dicts with ``N``, ``H``, ``regime``, ``pos`` (probe only) and the ``key``."""
    for r in regimes:
        if r not in REGIMES:
            raise ValueError("unknown regime %r (choose from %s)" % (r, ", ".join(REGIMES)))
    specs = []
    for N, H in sorted(set(tuple(s) for s in list(shapes) + (list(probe_shapes) if "probe" in regimes else []))):
        for regime in REGIMES:
            if regime not in regimes:
                continue
            if regime == "probe":
                if (N, H) in [tuple(s) for s in probe_shapes]:
                    for p in positions:
                        specs.append({"N": N, "H": H, "regime": regime, "pos": int(p), "key": records.cell_key("decode", N=N, H=H, regime=regime, pos=int(p))})
            elif (N, H) in [tuple(s) for s in shapes]:
                specs.append({"N": N, "H": H, "regime": regime, "pos": None, "key": records.cell_key("decode", N=N, H=H, regime=regime)})
    return specs


def select_cells(specs, substrings):
    """The cells whose key contains one of ``substrings`` (all of them without a filter); keys end every field with a delimiter, so ``N=1;`` does not match N=16."""
    return [s for s in specs if not substrings or any(sub in s["key"] for sub in substrings)]


def cycles_up(n):
    """``n`` rounded up to a whole number of 64-position cycles."""
    return -(-int(n) // STEPS) * STEPS


def calls_per_trial(regime, opts):
    """Calls of one trial: a loop trial covers a whole number of 64-position cycles, the probe regime (one position per call) any number. The re-run scale multiplies
    the rounded calls, so that a re-run has exactly ``calls_scale`` times the calls of the first campaign (it never changes the number of trials)."""
    base = opts["calls"] if regime in LOOP_REGIMES else (opts["cold_calls"] if regime == "cold" else opts["probe_calls"])
    return (base if regime == "probe" else cycles_up(base)) * opts["calls_scale"]


def profile_calls_of(regime, opts):
    """Calls of the profiled pass of a trial (the kernel sum), after the warm-up call: the profile calls rounded up to whole 64-position cycles (the probe regime
    holds one position per call: any number) times the re-run scale. The effective count is what the cell records."""
    base = opts["profile_calls"]
    return (base if regime == "probe" else cycles_up(base)) * opts["calls_scale"]


def parse_shape(text):
    n, h = text.split(",")
    return int(n), int(h)


def expected_mapping(expect_compute, expect_kernels):
    """The exact mapping ``{kernel: launches per call}`` that every trial of a run must show (None: no assertion). ``expect_compute``: a preset of ``CANONICAL_COMPUTE`` or
    ``NAME=COUNT;NAME=COUNT``; ``expect_kernels``: the shorthand ``NAME,NAME`` (every name once). Giving both is an error: they are two spellings of one assertion."""
    if expect_compute and expect_kernels:
        raise ValueError("--expect-compute and --expect-kernels are two spellings of one assertion: give one of them")
    if expect_compute:
        return records.parse_mapping(expect_compute, CANONICAL_COMPUTE)
    names = [x.strip() for x in (expect_kernels or "").split(",") if x.strip()]
    return records.parse_mapping(";".join("%s=1" % n for n in names)) if names else None


def trial_wall(gpu_eager, gpu_graph, cpu_eager=None, cpu_graph=None):
    """The wall metrics of one trial (microseconds per call); the CPU metrics only exist for loop trials."""
    wall = {"gpu_eager_us": gpu_eager, "gpu_graph_us": gpu_graph}
    if cpu_eager is not None:
        wall["cpu_eager_us"] = cpu_eager
    if cpu_graph is not None:
        wall["cpu_graph_us"] = cpu_graph
    return wall


def cell_line(spec, cell):
    """One printed line per cell: the median over the trials of every metric."""
    s = cell["samples"]
    parts = []
    for m in records.WALL_METRICS + ("total_us",):
        vals = [x[m] for x in s if m in x]
        if vals:
            parts.append("%s %8.2f" % (m.replace("_us", ""), statistics.median(vals)))
    return "%-44s %s   (%d trials, %d calls, %d profiled)" % (spec["key"], "  ".join(parts), len(s), cell["args"]["calls_per_trial"], cell["args"]["profile_calls"])


def run_meta(a, started, opts, device, torch_mod):
    commit, dirty, tree = records.tree_identity(ROOT)
    return {"commit": commit, "dirty": dirty, "tree": tree, "gpu": device["name"], "sm_count": device["sm_count"], "cc": device["cc"], "host": socket.gethostname(),
            "torch": torch_mod.__version__, "cuda": torch_mod.version.cuda, "timestamp": started, "kind": "decode-wall", "sample_stat": records.PROCESS_MEDIAN,
            "impl": a.impl, "env_impl": os.environ.get("FLASH_KDA_CFBI_IMPL"), "regimes": list(a.regimes), "trials": opts["trials"],
            "calls_per_trial": {r: calls_per_trial(r, opts) for r in REGIMES}, "profile_calls": {r: profile_calls_of(r, opts) for r in REGIMES},
            "profile_calls_option": opts["profile_calls"], "calls_scale": opts["calls_scale"], "l2_cache_bytes": device["l2_bytes"], "eviction_bytes": device["eviction_bytes"],
            "cells_filter": a.cells, "label": a.label, "tag": os.environ.get("CFBI_BENCH_TAG"), "build_dir": os.environ.get("FLASH_KDA_CFBI_BUILD_DIR"),
            "src": str(ROOT), "argv": sys.argv[1:]}


# ---- measurement (needs a GPU) --------------------------------------------------------------------------------------------------------------

class ShapeEnv:
    """Inputs, output, state, the eager call and the captured graph of one (N, H) shape (the state advances with the calls)."""

    def __init__(self, N, H, impl):
        import torch
        from flash_kda.cfbi import alloc_state, decode
        from helpers import make_case

        self.torch, self.N, self.H, self.impl = torch, N, H, impl
        self._decode = decode
        case = make_case(N * STEPS, H, "authors", seed=0)
        self.case = case
        self.per_step = [{key: torch.stack([case[key][0, n * STEPS + s] for n in range(N)], dim=0).unsqueeze(1).contiguous() for key in ("q", "k", "v", "g", "beta")}
                         for s in range(STEPS)]
        self.out = torch.zeros_like(self.per_step[0]["q"])
        self.state = alloc_state(N, H)
        self.graph = None

    def call(self, d):
        c = self.case
        self._decode(d["q"], d["k"], d["v"], d["g"], d["beta"], c["scale"], self.out, c["A_log"], c["dt_bias"], LB, self.state, _impl=self.impl)

    def eager(self, i):
        self.call(self.per_step[i % STEPS])

    def capture(self):
        torch = self.torch
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.call(self.per_step[0])
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.call(self.per_step[0])

    def replay(self, i):
        self.graph.replay()

    def set_positions(self, regime):
        torch = self.torch
        if regime == "staggered":
            self.state.pos.copy_(torch.arange(self.N, dtype=torch.int32, device=self.state.pos.device) % STEPS)
        else:
            self.state.pos.zero_()


def loop_trial(torch, run, calls):
    """``(gpu_us, cpu_us)`` per call of ``calls`` consecutive calls: one pair of CUDA events around the loop, the CPU clock around the loop and the
    synchronisation that ends it."""
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter_ns()
    ev0.record()
    for i in range(calls):
        run(i)
    ev1.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    return ev0.elapsed_time(ev1) * 1000.0 / calls, (t1 - t0) / 1000.0 / calls


def single_call_trial(torch, run, helper, calls):
    """Mean GPU-elapsed microseconds of ``calls`` calls, each bracketed by its own pair of events; ``helper(i)`` is enqueued and synchronised before the first event."""
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(calls)]
    for i in range(calls):
        helper(i)
        torch.cuda.synchronize()
        pairs[i][0].record()
        run(i)
        pairs[i][1].record()
    torch.cuda.synchronize()
    return statistics.mean(a.elapsed_time(b) * 1000.0 for a, b in pairs)


def device_info(torch):
    prop = torch.cuda.get_device_properties(0)
    l2 = getattr(prop, "L2_cache_size", None)
    if not l2:
        raise RuntimeError("this torch does not report the L2 cache size of the device: the cold regime needs it (eviction buffer of at least twice the size)")
    return {"name": prop.name, "sm_count": prop.multi_processor_count, "cc": "%d.%d" % (prop.major, prop.minor), "l2_bytes": int(l2), "eviction_bytes": 2 * int(l2) + (1 << 20)}


def measure_cell(spec, env, opts, device, expect_compute, evict_buf):
    """One cell: ``opts["trials"]`` trials of (eager, graph, one profiled pass); returns the cell for the record. The launch mapping of the profiled pass is checked and recorded in EVERY
    trial: whole launches per call, equal to ``expect_compute`` when that is given (None: no assertion) and equal to the first trial's in any case."""
    torch, regime = env.torch, spec["regime"]
    from bench_breakdown import kernel_times

    env.set_positions(regime)
    calls, profile_calls = calls_per_trial(regime, opts), profile_calls_of(regime, opts)
    if regime == "cold":
        def helper(i):
            evict_buf.fill_(float(i % 251 + 1))
    elif regime == "probe":
        def helper(i):
            env.state.pos.fill_(spec["pos"])
    else:
        helper = None
    for i in range(200):                                            # warm-up of the eager path
        if helper:
            helper(i)
        env.eager(i)
    for i in range(50):
        if helper:
            helper(i)
        env.replay(i)
    torch.cuda.synchronize()
    counter = [0]

    def profiled():
        if helper:
            helper(counter[0])
        env.eager(counter[0])
        counter[0] += 1

    runs, walls, nodes = [], [], []
    gc.collect()
    gc.disable()
    try:
        for trial in range(opts["trials"]):
            env.set_positions(regime)
            if regime in LOOP_REGIMES:
                gpu_e, cpu_e = loop_trial(torch, env.eager, calls)
                env.set_positions(regime)
                gpu_g, cpu_g = loop_trial(torch, env.replay, calls)
                wall = trial_wall(gpu_e, gpu_g, cpu_e, cpu_g)
            else:
                gpu_e = single_call_trial(torch, env.eager, helper, calls)
                gpu_g = single_call_trial(torch, env.replay, helper, calls)
                wall = trial_wall(gpu_e, gpu_g)
            env.set_positions(regime)
            counter[0] = 0                                          # the data step of a call follows its position: both start at 0 in every trial
            times, counts = kernel_times(profiled, iters=profile_calls, warm=PROFILE_WARM)
            compute, aux = records.split_nodes(counts)
            problem = records.check_nodes(compute, aux, expect_compute, None)
            if not problem and nodes and (compute, aux) != nodes[0]:
                problem = "the nodes of this trial differ from those of the first trial (%s / %s vs %s / %s)" % (compute, aux, nodes[0][0], nodes[0][1])
            if problem:
                raise SystemExit("cell %s, trial %d: %s" % (spec["key"], trial, problem))
            runs.append(times)
            walls.append(wall)
            nodes.append((compute, aux))
    finally:
        gc.enable()
    compute, aux = nodes[0]
    loop = regime != "probe"                                        # the positions of a loop regime advance with the calls: the profiled pass covers whole cycles
    args = {"N": spec["N"], "H": spec["H"], "regime": regime, "pos": spec["pos"], "impl": env.impl, "calls_per_trial": calls, "profile_calls": profile_calls,
            "profile_cycles": profile_calls // STEPS if loop else None, "profile_start_phase": PROFILE_WARM if loop else None, "calls_scale": opts["calls_scale"],
            "kernels": sorted(compute), "compute_nodes": whole(compute), "compute_kernels": int(round(sum(compute.values()))), "aux_nodes": whole(aux), "expect_compute": expect_compute}
    cell = records.make_cell("decode", spec["key"], args, runs, drop_helpers=True, wall=walls)
    for sample, (c, a) in zip(cell["samples"], nodes):
        sample["nodes"] = {"compute": whole(c), "aux": whole(a)}
    return cell


def whole(mapping):
    """``{name: launches per call}`` with whole numbers (the profiler reports them as quotients; ``check_nodes`` has already refused a fractional one), sorted by name."""
    return {k: int(round(v)) for k, v in sorted(mapping.items())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", type=int, default=0, help="implementation bits of the decode call (default 0 = the default path)")
    ap.add_argument("--shapes", nargs="+", default=None, help='"N,H" shapes (default: the decode matrix)')
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=REGIMES)
    ap.add_argument("--probe-shapes", nargs="+", default=None, help='"N,H" shapes of the per-position probe (default 1,12 64,12)')
    ap.add_argument("--positions", nargs="+", type=int, default=list(PROBE_POSITIONS))
    ap.add_argument("--trials", type=int, default=DEFAULTS["trials"], help="trials per cell and process (at least 5 for a usable record)")
    ap.add_argument("--calls", type=int, default=DEFAULTS["calls"], help="calls of a lock-step / staggered trial (rounded up to whole 64-position cycles)")
    ap.add_argument("--cold-calls", type=int, default=DEFAULTS["cold_calls"])
    ap.add_argument("--probe-calls", type=int, default=DEFAULTS["probe_calls"])
    ap.add_argument("--profile-calls", type=int, default=DEFAULTS["profile_calls"],
                    help="calls of the profiled pass of a trial (kernel sum), rounded up to whole 64-position cycles (the probe regime excepted)")
    ap.add_argument("--calls-scale", type=int, default=DEFAULTS["calls_scale"],
                    help="multiplier of the calls per trial AND of the profiled calls (3 for the re-run of unstable cells)")
    ap.add_argument("--cells", nargs="*", default=None, help="only cells whose key contains one of these substrings")
    ap.add_argument("--expect-compute", default=None, help="the exact mapping of the CF-BI kernels one decode call launches, asserted in EVERY trial: single-launch, two-launch, reference "
                                                           "or NAME=COUNT;NAME=COUNT")
    ap.add_argument("--expect-kernels", default=None, help="shorthand of --expect-compute: comma-separated base names of the CF-BI kernels one decode call must launch (each once)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--list-cells", action="store_true")
    a = ap.parse_args(argv)
    for name in ("trials", "calls", "cold_calls", "probe_calls", "profile_calls", "calls_scale"):
        if getattr(a, name) < 1:
            ap.error("--%s must be at least 1" % name.replace("_", "-"))
    try:
        expect_compute = expected_mapping(a.expect_compute, a.expect_kernels)
    except ValueError as e:
        ap.error(str(e))
    shapes = [parse_shape(s) for s in a.shapes] if a.shapes else list(MATRIX_D)
    probe_shapes = [parse_shape(s) for s in a.probe_shapes] if a.probe_shapes else list(PROBE_SHAPES)
    specs = select_cells(cell_specs(shapes, a.regimes, probe_shapes, a.positions), a.cells)
    if a.list_cells:
        for s in specs:
            print(s["key"])
        return 0
    if not specs:
        print("no cell selected", file=sys.stderr)
        return 2
    opts = {"trials": a.trials, "calls": a.calls, "cold_calls": a.cold_calls, "probe_calls": a.probe_calls, "profile_calls": a.profile_calls, "calls_scale": a.calls_scale}

    import torch

    device = device_info(torch)
    started = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    print("device %s  impl=%s  %d cell(s)  trials=%d  calls/trial: lockstep %d, cold %d, probe %d" % (
        device["name"], a.impl, len(specs), opts["trials"], calls_per_trial("lockstep", opts), calls_per_trial("cold", opts), calls_per_trial("probe", opts)), flush=True)
    evict_buf = torch.empty(device["eviction_bytes"] // 4, dtype=torch.float32, device="cuda") if any(s["regime"] == "cold" for s in specs) else None
    cells, env, env_shape = [], None, None
    for spec in specs:
        if env_shape != (spec["N"], spec["H"]):
            env = None
            torch.cuda.empty_cache()
            env = ShapeEnv(spec["N"], spec["H"], a.impl)
            env.capture()
            env_shape = (spec["N"], spec["H"])
        cell = measure_cell(spec, env, opts, device, expect_compute, evict_buf)
        cells.append(cell)
        print(cell_line(spec, cell), flush=True)
    if a.json:
        records.write_record(a.json, run_meta(a, started, opts, device, torch), cells)
        print("record written to %s" % a.json, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
