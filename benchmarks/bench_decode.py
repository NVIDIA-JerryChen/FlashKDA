"""Wall time of the CF-BI decode call: CUDA-graph replay and eager loop, in four regimes.

usage: python benchmarks/bench_decode.py [--impl N] [--shapes N,H ...] [--regimes lockstep staggered cold probe] [--probe-shapes N,H ...] [--positions P ...]
           [--trials T] [--calls C] [--cold-calls C] [--probe-calls C] [--profile-calls C] [--calls-scale S] [--cells SUBSTR ...] [--boundary [SMS]] [--single-force]
           [--expect-compute MAPPING | --expect-kernels NAME,NAME | --expect-route spec] [--graph-nodes] [--json PATH] [--label TEXT] [--list-cells]

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

Campaign cells: ``--boundary [SMS]`` replaces the default cells by those of the single-launch campaign (``campaign_cell_specs``): matrix D plus the grids that straddle the limit of the single
launch (``boundary_shapes``: for 6, 12 and 33 heads, the largest N whose grid of N * H CTAs holds at most x CTAs and the next larger N, for x at half the limit, at the limit and at twice the limit)
in the four regimes, the probe included. The limit is ``DECODE_SINGLE_MAX_CTAS_PER_SM`` of the MEASURED tree (its csrc/cfbi/cfbi_policy.h) times ``SMS``; without a value ``SMS`` is the device's SM
count, a value fixes the cell set so that devices with another SM count measure the same shapes. ``--cells`` still narrows the set; ``--shapes``, ``--probe-shapes``, ``--regimes`` and ``--positions``
name cell sets of their own and are not combined with it. The default cells (``cell_specs()``) are unchanged.

Calibration control: ``--single-force`` switches the hidden control of the measured tree's extension (``decode_single_force``: every eligible decode call is one launch whatever the size of the grid and
the device, never with ``--impl 64`` or on the reference row engine) on before the first call and before any graph is captured (a graph bakes the route it was captured with), and restores its previous
state when the run ends, however it ends. The meta and every cell record it (``single_force``); the meta also records the limit constant of the measured tree (``single_per_sm``).

Route specification: ``--expect-route spec`` asserts in every trial of a cell the launches that the route specification (``expected_route`` of tests/cfbi/decode_policy.py: the device's SM count and
compute capability, the limit of the measured tree, the state of the control) gives for THAT cell, instead of one mapping for the whole run, so that one process can cross the limit. Before the
first call the extension's own answer (``decode_single_launch``) is compared with the specification for every shape, and a disagreement ends the run: a mapping that was only measured never vouches
for the route it is supposed to confirm. It is not combined with ``--expect-compute`` or ``--expect-kernels`` (two spellings of one assertion); the cells record ``expect_route`` = ``"spec"`` and,
as ``expect_compute``, the mapping that each of them was asserted against.

Graph replay: the profiled pass of a trial is an eager one, and the wall metric that gates most comparisons is the graph replay. ``--graph-nodes`` profiles ONE replay of the cell's graph in every trial, after
the eager pass (nothing of it enters a metric, and the default run does not take it), and requires its compute kernels to equal the eager pass's of the same trial and ``--expect-compute`` / the specification
of the cell when there is one; the replay's mapping is recorded in the sample as ``graph_nodes`` (``compute`` and ``aux`` as in ``nodes``, which stays the eager pass's) and the meta as ``graph_nodes``

Order of the cells: ``--route-order`` measures the cells whose default route is the two-launch path first and those whose default route is the single launch after them (``route_ordered``), the same
order on both sides of an A/B. The cells that both sides run with the same kernels (the default routing above the limit and the explicit two-launch path) are then measured in a process that has no
single-launch shape in its history: a process in which the default side had run the single launch for the small grids first measured the same two kernels differently from the side that had not, and
identical kernels differed per shape by up to 1.3 % (a process that measured one shape, or only the cells above the limit, did not show it). Recorded as the meta ``route_order``..
"""
import argparse
import contextlib
import datetime
import functools
import gc
import importlib.util
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
BOUNDARY_HEADS = (6, 12, 33)
BOUNDARY_FRACTIONS = ((1, 2), (1, 1), (2, 1))      # (numerator, denominator) of the limit: half the limit, the limit, twice the limit
FROM_DEVICE = 0                                    # the value of ``--boundary`` without an SM count: the cells are made for the SM count of the device that runs them


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


def boundary_shapes(per_sm, sms, heads=BOUNDARY_HEADS, fractions=BOUNDARY_FRACTIONS):
    """The ``(N, H)`` shapes that straddle the limit of the single launch, sorted: for every head count and every x = ``per_sm * sms * fraction`` (the limit is ``per_sm`` CTAs on each of ``sms``
    SMs; ``fractions`` are ``(numerator, denominator)`` pairs, by default half the limit, the limit and twice the limit), the largest N whose grid of N * H CTAs holds at most x CTAs and the next
    larger N, which does not (a shape with fewer than one sequence is no decode call and is left out). Pure: the same shapes for the benchmark, the checker and the tests."""
    if per_sm < 1 or sms < 1:
        raise ValueError("the limit needs a positive number of CTAs per SM and of SMs, got %r and %r" % (per_sm, sms))
    shapes = set()
    for H in heads:
        for numerator, denominator in fractions:
            x = per_sm * sms * numerator // denominator
            largest = x // H
            shapes.update((N, H) for N in (largest, largest + 1) if N >= 1)
    return sorted(shapes)


def campaign_cell_specs(per_sm, sms):
    """The cells of the single-launch campaign: matrix D plus ``boundary_shapes(per_sm, sms)`` in every regime, and the probe (92 cells for 148 SMs and 8 CTAs per SM). The default cells of
    ``cell_specs()`` are not changed by this: baselines that were recorded on them stay comparable."""
    return cell_specs(shapes=tuple(MATRIX_D) + tuple(boundary_shapes(per_sm, sms)), regimes=REGIMES)


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


@functools.lru_cache(maxsize=None)
def policy_spec():
    """The route specification (tests/cfbi/decode_policy.py) of the tree that holds THIS script, loaded by its path. The measured tree (``CFBI_BENCH_SRC``) may be an older snapshot whose copy of
    the module knows neither ``expected_route`` nor ``header_per_sm``, and ``sys.path`` puts that tree's ``tests/cfbi`` before this one's: an import by name could pick it."""
    path = TOOL_ROOT / "tests" / "cfbi" / "decode_policy.py"
    spec = importlib.util.spec_from_file_location("decode_policy_of_the_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tree_per_sm(root):
    """The limit of the single launch, in CTAs per SM, that the policy header of the source tree ``root`` defines (the tree that is measured): None when it cannot be read, a tree from before
    the single-launch decode has no such header."""
    try:
        return policy_spec().header_per_sm(root)
    except (OSError, ValueError):
        return None


def device_cc(device):
    """The compute capability of a device as the policy takes it (major * 10 + minor): ``"10.3"`` is 103."""
    major, minor = (int(x) for x in device["cc"].split("."))
    return policy_spec().cc_number((major, minor))


def route_expectations(specs, impl, device, per_sm, force):
    """``{cell key: preset}``: the launches that the route specification (``decode_policy.expected_route``) gives for each cell on ``device`` (its SM count and compute capability), for the
    limit ``per_sm`` of the measured tree and the state ``force`` of the calibration control. The presets are keys of ``CANONICAL_COMPUTE``."""
    sms, cc, expected_route = device["sm_count"], device_cc(device), policy_spec().expected_route
    return {s["key"]: expected_route(impl, s["N"], s["H"], sms, cc, per_sm, force) for s in specs}


def route_ordered(specs, device, per_sm):
    """``specs`` in the order of ``--route-order``: the cells whose default route (``expected_route`` for impl 0 with the control off, on ``device`` and the limit ``per_sm`` of the measured
    tree) is the two-launch path first, then those whose default route is the single launch; a stable sort, so that the order of the cells within a class and of the regimes within a shape stays.
    The route is the one of the DEFAULT path on both sides of an A/B (side A pins bit 64 and launches two kernels everywhere): one order for both."""
    spec = policy_spec()
    sms, cc = device["sm_count"], device_cc(device)
    return sorted(specs, key=lambda s: spec.expected_route(0, s["N"], s["H"], sms, cc, per_sm, False) == spec.SINGLE_LAUNCH)


def route_binding_problems(ext, specs, impl, routes):
    """Why the extension of the measured tree does not agree with the route specification (an empty list: it does, for every shape of ``specs``). The extension is asked what the benchmark's
    decode call asks it (``decode_single_launch`` with the resolved bits of ``impl``, on the current device, under the current state of the control) and ``routes`` is ``route_expectations``."""
    spec = policy_spec()
    resolved = ext.resolve_impl(impl, -1)
    if resolved & spec.IMPL_REF_ROWS:
        return ["impl %d resolves to %d on this device, the reference row engine, whose launches are the preset 'reference': the route specification does not cover it" % (impl, resolved)]
    out, seen = [], set()
    for s in specs:
        if (s["N"], s["H"]) in seen:
            continue
        seen.add((s["N"], s["H"]))
        want = routes[s["key"]] == spec.SINGLE_LAUNCH
        got = bool(ext.decode_single_launch(1, resolved, s["N"], s["H"]))
        if got != want:
            out.append("N=%d H=%d: the extension says %s, the specification %s" % (s["N"], s["H"], _launches_text(got), _launches_text(want)))
    return out


def _launches_text(single):
    """The route of one decode call in words: ``"one launch"`` (``single``) or ``"two launches"``."""
    return "one launch" if single else "two launches"


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
    """The metadata of a record. ``single_force`` (the calibration control was on for the whole process) and ``single_per_sm`` (the limit constant of the measured tree: None for a tree that has
    none) are what the comparison of two sides binds; ``boundary_sms`` is the SM count that the cells of ``--boundary`` were made for (None without it), ``graph_nodes`` whether one graph replay
    per trial was profiled, ``route_order`` whether the cells were measured in the order of ``--route-order``, ``cuda_module_loading`` the module loading mode of the process (the environment variable CUDA_MODULE_LOADING, None when unset): with the default lazy loading the
    machine code of a kernel is placed when it is first launched, so two builds that launch different kernels earlier in a long process get another placement of the kernels that both run
    (graph replays of identical kernels differed by 1 % per shape); two sides of an A/B must share the mode."""
    commit, dirty, tree = records.tree_identity(ROOT)
    boundary = getattr(a, "boundary", None)
    return {"commit": commit, "dirty": dirty, "tree": tree, "gpu": device["name"], "sm_count": device["sm_count"], "cc": device["cc"], "host": socket.gethostname(),
            "torch": torch_mod.__version__, "cuda": torch_mod.version.cuda, "timestamp": started, "kind": "decode-wall", "sample_stat": records.PROCESS_MEDIAN,
            "impl": a.impl, "env_impl": os.environ.get("FLASH_KDA_CFBI_IMPL"), "regimes": list(a.regimes), "trials": opts["trials"],
            "calls_per_trial": {r: calls_per_trial(r, opts) for r in REGIMES}, "profile_calls": {r: profile_calls_of(r, opts) for r in REGIMES},
            "profile_calls_option": opts["profile_calls"], "calls_scale": opts["calls_scale"], "l2_cache_bytes": device["l2_bytes"], "eviction_bytes": device["eviction_bytes"],
            "single_force": bool(getattr(a, "single_force", False)), "single_per_sm": tree_per_sm(ROOT), "boundary_sms": None if boundary is None else (boundary or device["sm_count"]),
            "graph_nodes": bool(getattr(a, "graph_nodes", False)), "route_order": bool(getattr(a, "route_order", False)),
            "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING"),
            "cells_filter": a.cells, "label": a.label, "tag": os.environ.get("CFBI_BENCH_TAG"), "build_dir": os.environ.get("FLASH_KDA_CFBI_BUILD_DIR"),
            "src": str(ROOT), "argv": sys.argv[1:]}


# ---- measurement (needs a GPU) --------------------------------------------------------------------------------------------------------------

def load_binding():
    """The extension module of the measured tree, the one that its decode calls use (``flash_kda.cfbi.load_ext`` builds or loads it once per process)."""
    from flash_kda.cfbi import load_ext

    return load_ext()


@contextlib.contextmanager
def single_launch_control(ext, on):
    """The hidden calibration control of the extension, ``on`` for the whole block and restored to its previous state when the block ends, by an exception too (a trial whose launches are not the
    asserted ones ends the run with ``SystemExit``). Enter it before the first call and before any graph is captured: a graph bakes the route that it was captured with. Off: nothing is touched."""
    if not on:
        yield
        return
    switch = getattr(ext, "decode_single_force", None)
    if switch is None:
        raise SystemExit("--single-force: the extension of the measured tree (%s) has no hidden calibration control (decode_single_force)" % ROOT)
    previous = switch(True)
    try:
        yield
    finally:
        switch(previous)


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


def graph_problem(graph_compute, graph_aux, compute, expect_compute):
    """None when one replay of the graph launched exactly the compute kernels of the eager pass of its trial (``compute``, whole launches per call) and ``expect_compute`` when that is given;
    otherwise the message. The auxiliary nodes of the replay are recorded, not asserted: the route is the compute kernels."""
    problem = records.check_nodes(graph_compute, graph_aux, expect_compute, None)
    if problem:
        return problem
    if whole(graph_compute) != whole(compute):
        return "the graph replay launches the compute kernels %s, the eager pass of the same trial %s" % (whole(graph_compute), whole(compute))
    return None


def measure_cell(spec, env, opts, device, expect_compute, evict_buf, single_force=False, expect_route=None, graph_nodes=False):
    """One cell: ``opts["trials"]`` trials of (eager, graph, one profiled pass); returns the cell for the record. The launch mapping of the profiled pass is checked and recorded in EVERY
    trial: whole launches per call, equal to ``expect_compute`` when that is given (None: no assertion) and equal to the first trial's in any case. ``single_force`` (the process runs under the
    calibration control) and ``expect_route`` (``"spec"``: ``expect_compute`` is the mapping that the route specification gives for this cell) are recorded in the arguments of the cell.
    ``graph_nodes``: after the eager pass of every trial ONE replay of the graph is profiled (it is not timed and enters no metric) and its compute kernels must be the eager pass's of that trial and
    ``expect_compute`` when that is given (``graph_problem``); the replay's mapping is recorded in the sample as ``graph_nodes``, next to the eager ``nodes``."""
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

    def profiled_replay():
        if helper:
            helper(counter[0])
        env.replay(counter[0])

    runs, walls, nodes, graphs = [], [], [], []
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
            graph = None
            if graph_nodes:
                _, graph_counts = kernel_times(profiled_replay, iters=1, warm=0)
                graph_compute, graph_aux = records.split_nodes(graph_counts)
                problem = graph_problem(graph_compute, graph_aux, compute, expect_compute)
                if problem:
                    raise SystemExit("cell %s, trial %d: graph replay: %s" % (spec["key"], trial, problem))
                graph = {"compute": whole(graph_compute), "aux": whole(graph_aux)}
            runs.append(times)
            walls.append(wall)
            nodes.append((compute, aux))
            graphs.append(graph)
    finally:
        gc.enable()
    compute, aux = nodes[0]
    loop = regime != "probe"                                        # the positions of a loop regime advance with the calls: the profiled pass covers whole cycles
    args = {"N": spec["N"], "H": spec["H"], "regime": regime, "pos": spec["pos"], "impl": env.impl, "calls_per_trial": calls, "profile_calls": profile_calls,
            "profile_cycles": profile_calls // STEPS if loop else None, "profile_start_phase": PROFILE_WARM if loop else None, "calls_scale": opts["calls_scale"],
            "kernels": sorted(compute), "compute_nodes": whole(compute), "compute_kernels": int(round(sum(compute.values()))), "aux_nodes": whole(aux), "expect_compute": expect_compute,
            "single_force": bool(single_force), "expect_route": expect_route}
    cell = records.make_cell("decode", spec["key"], args, runs, drop_helpers=True, wall=walls)
    for sample, (c, a), graph in zip(cell["samples"], nodes, graphs):
        sample["nodes"] = {"compute": whole(c), "aux": whole(a)}
        if graph is not None:
            sample["graph_nodes"] = graph
    return cell


def whole(mapping):
    """``{name: launches per call}`` with whole numbers (the profiler reports them as quotients; ``check_nodes`` has already refused a fractional one), sorted by name."""
    return {k: int(round(v)) for k, v in sorted(mapping.items())}


def run_cells(specs, a, opts, device, torch_mod, expect_compute=None, routes=None, env_factory=None):
    """Measure ``specs`` in order (one set of inputs, state and graph per shape) and return the cells. ``a``: the options of the run (``impl``, ``single_force``, ``expect_route``,
    ``graph_nodes``). ``expect_compute``: the one mapping that every trial of every cell must show (None: no assertion). ``routes`` (``route_expectations``, with ``--expect-route spec``): the
    preset of each cell, which every trial of that cell must show instead. The calibration control, when it is wanted, is on before this is called (the graph of a shape is captured here).
    ``env_factory(N, H, impl)`` builds the inputs, state and graph holder of a shape (``ShapeEnv``)."""
    make_env = env_factory or ShapeEnv
    evict_buf = torch_mod.empty(device["eviction_bytes"] // 4, dtype=torch_mod.float32, device="cuda") if any(s["regime"] == "cold" for s in specs) else None
    cells, env, env_shape = [], None, None
    for spec in specs:
        if env_shape != (spec["N"], spec["H"]):
            env = None
            torch_mod.cuda.empty_cache()
            env = make_env(spec["N"], spec["H"], a.impl)
            env.capture()
            env_shape = (spec["N"], spec["H"])
        expect = dict(CANONICAL_COMPUTE[routes[spec["key"]]]) if routes is not None else expect_compute
        cell = measure_cell(spec, env, opts, device, expect, evict_buf, single_force=a.single_force, expect_route=a.expect_route, graph_nodes=a.graph_nodes)
        cells.append(cell)
        print(cell_line(spec, cell), flush=True)
    return cells


def sm_count(text):
    """The argument of ``--boundary``: a positive number of SMs."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("%r is not a positive SM count" % text)
    return value


def build_parser():
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
    ap.add_argument("--boundary", nargs="?", const=FROM_DEVICE, default=None, type=sm_count, metavar="SMS",
                    help="measure the cells of the single-launch campaign (matrix D, the grids that straddle the single-launch limit of the measured tree, the probe; four regimes) instead of the "
                         "default ones; SMS fixes the SM count that the cells are made for (default: this device's: give it where devices with another SM count must measure the same shapes)")
    ap.add_argument("--single-force", action="store_true",
                    help="switch the hidden calibration control of the extension on before the first call and the graph capture (every eligible decode call is one launch, whatever the grid and "
                         "the device) and restore it at the end; recorded in the meta and in every cell")
    ap.add_argument("--expect-compute", default=None, help="the exact mapping of the CF-BI kernels one decode call launches, asserted in EVERY trial: single-launch, two-launch, reference "
                                                           "or NAME=COUNT;NAME=COUNT")
    ap.add_argument("--expect-kernels", default=None, help="shorthand of --expect-compute: comma-separated base names of the CF-BI kernels one decode call must launch (each once)")
    ap.add_argument("--expect-route", default=None, choices=("spec",),
                    help="assert in every trial of a cell the launches that the route specification (tests/cfbi/decode_policy.py) gives for THAT cell on this device, with the limit of the measured "
                         "tree and the state of the control; the extension's own answer is checked against it first. Not combined with --expect-compute or --expect-kernels")
    ap.add_argument("--graph-nodes", action="store_true",
                    help="profile one replay of the graph in every trial (after the eager pass, not timed) and assert its compute kernels like the eager pass's; recorded as graph_nodes in the samples")
    ap.add_argument("--route-order", action="store_true",
                    help="measure the cells whose DEFAULT route (impl 0, control off) is the two-launch path before those whose default route is the single launch, keeping the order of the cells "
                         "within each class: the same order on both sides of an A/B, so that a process has no single-launch shape in its history when it measures the cells that both sides run "
                         "with the same kernels; recorded as route_order")
    ap.add_argument("--json", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--list-cells", action="store_true")
    return ap


def check_args(ap, a):
    """The rules of the options (``ap.error``: exit status 2): a count of at least 1, and the spellings that exclude each other."""
    for name in ("trials", "calls", "cold_calls", "probe_calls", "profile_calls", "calls_scale"):
        if getattr(a, name) < 1:
            ap.error("--%s must be at least 1" % name.replace("_", "-"))
    if a.expect_route and (a.expect_compute or a.expect_kernels):
        ap.error("--expect-route spec derives the expected launches of every cell itself: it cannot be combined with --expect-compute or --expect-kernels (two spellings of one assertion)")
    if a.boundary is not None:
        clashes = [flag for flag, given in (("--shapes", a.shapes), ("--probe-shapes", a.probe_shapes), ("--regimes", list(a.regimes) != list(REGIMES)),
                                            ("--positions", list(a.positions) != list(PROBE_POSITIONS))) if given]
        if clashes:
            ap.error("--boundary fixes the cells (matrix D, the straddling grids, the probe, four regimes): it cannot be combined with %s; narrow the set with --cells" % ", ".join(clashes))
        if a.boundary == FROM_DEVICE and a.list_cells:
            ap.error("--list-cells asks no device, so --boundary needs the SM count that the cells are made for (--boundary 148)")


def choose_cells(a, per_sm, device_sms=None):
    """The cells of a run before the ``--cells`` filter: with ``--boundary`` the campaign cells for the SM count of the option (``device_sms`` when it has none), otherwise the cells of the
    shapes, regimes and probe options."""
    if a.boundary is not None:
        return campaign_cell_specs(per_sm, a.boundary or device_sms)
    shapes = [parse_shape(s) for s in a.shapes] if a.shapes else list(MATRIX_D)
    probe_shapes = [parse_shape(s) for s in a.probe_shapes] if a.probe_shapes else list(PROBE_SHAPES)
    return cell_specs(shapes, a.regimes, probe_shapes, a.positions)


def start_cuda():
    """``(torch, device)``: torch is imported here and nowhere earlier, so that listing the cells needs neither torch nor a GPU."""
    import torch

    return torch, device_info(torch)


def run_notes(a, per_sm, device):
    """What the options of the campaign ask for, as the words that the header line of a run names (empty for a default run)."""
    notes = []
    if a.boundary is not None:
        notes.append("boundary cells for %d SMs" % (a.boundary or device["sm_count"]))
    if a.single_force:
        notes.append("single-force")
    if a.expect_route:
        notes.append("expect-route spec, %d CTAs per SM" % per_sm)
    if a.graph_nodes:
        notes.append("graph-nodes")
    if a.route_order:
        notes.append("route-order")
    return notes


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    check_args(ap, a)
    try:
        expect_compute = expected_mapping(a.expect_compute, a.expect_kernels)
    except ValueError as e:
        ap.error(str(e))
    per_sm = tree_per_sm(ROOT)
    if per_sm is None and (a.boundary is not None or a.expect_route or a.route_order):
        ap.error("--boundary, --expect-route and --route-order need the single-launch limit of the measured tree, which cannot be read: %s must define DECODE_SINGLE_MAX_CTAS_PER_SM as an integer literal "
                 "(and the tests/cfbi/decode_policy.py of this tool must exist)" % (ROOT / "csrc" / "cfbi" / "cfbi_policy.h"))
    torch = device = None
    if a.boundary == FROM_DEVICE:                                   # the cells depend on the device: its SM count is read before they are chosen
        torch, device = start_cuda()
    specs = select_cells(choose_cells(a, per_sm, device["sm_count"] if device else None), a.cells)
    if a.list_cells:
        for s in specs:
            print(s["key"])
        return 0
    if not specs:
        print("no cell selected", file=sys.stderr)
        return 2
    opts = {"trials": a.trials, "calls": a.calls, "cold_calls": a.cold_calls, "probe_calls": a.probe_calls, "profile_calls": a.profile_calls, "calls_scale": a.calls_scale}

    if device is None:
        torch, device = start_cuda()
    if a.route_order:
        specs = route_ordered(specs, device, per_sm)
    started = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    notes = run_notes(a, per_sm, device)
    print("device %s  impl=%s  %d cell(s)  trials=%d  calls/trial: lockstep %d, cold %d, probe %d%s" % (
        device["name"], a.impl, len(specs), opts["trials"], calls_per_trial("lockstep", opts), calls_per_trial("cold", opts), calls_per_trial("probe", opts),
        "  [%s]" % "; ".join(notes) if notes else ""), flush=True)
    ext = load_binding() if (a.single_force or a.expect_route) else None
    with single_launch_control(ext, a.single_force):
        routes = None
        if a.expect_route:
            routes = route_expectations(specs, a.impl, device, per_sm, a.single_force)
            problems = route_binding_problems(ext, specs, a.impl, routes)
            if problems:
                raise SystemExit("--expect-route spec: the extension and the route specification disagree (device %s, %d SMs, compute capability %s, %d CTAs per SM, control %s): %s" % (
                    device["name"], device["sm_count"], device["cc"], per_sm, "on" if a.single_force else "off", "; ".join(problems[:6]) + (" ..." if len(problems) > 6 else "")))
        cells = run_cells(specs, a, opts, device, torch, expect_compute, routes)
    if a.json:
        records.write_record(a.json, run_meta(a, started, opts, device, torch), cells)
        print("record written to %s" % a.json, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
