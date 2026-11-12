"""Benchmark records and the interleaved A/B comparison logic of the CF-BI benchmarks (plain Python, no torch).

``bench_breakdown.py --json`` and ``bench_decode.py --json`` write one record file per process run (one build, one GPU, several cells).
``compare_bench.py`` compares the record files of two builds that were measured interleaved in the same session on the same node: a baseline
build (A) and a candidate build (B).

A record file::

    {"schema": "cfbi-bench-v1",
     "meta":  {"commit": ..., "dirty": ..., "tree": "snapshot|worktree", "gpu": ..., "sm_count": ..., "host": ..., "timestamp": ..., ...},
     "cells": [{"kind": "prefill|varlen|train|decode|chain", "key": "prefill B=8 L=8192 H=6 pos0=0", "args": {...},
                "samples": [{"kernels": {"k1_kernel": us, ...}, "total_us": ..., "k1_us": ..., "k2_stage_us": ..., "k2_main_us": ...,
                             "gpu_graph_us": ..., "gpu_eager_us": ..., "cpu_graph_us": ..., "cpu_eager_us": ...}, ...]}]}

Kernel metrics are microseconds per call (CUDA profiler kernel durations, no launch overhead); helper kernels that a benchmark runs around the
measured call are left out of the kernel sum. The wall metrics are optional per sample: ``gpu_*`` is the GPU-elapsed time per decode call taken with
CUDA events (a loop of calls bracketed by two events, or one call bracketed by two events when a helper has to run between calls), ``cpu_*`` the CPU
monotonic-clock time per call of a batch of launches with a synchronisation at its end (the host work is included).

How samples are pooled: by default a cell's ``samples`` are in-process repeats and the comparison pools the samples of all record files of a build
(the spread of a cell is ``(max - min) / median`` over the pooled samples). A record whose meta says ``"sample_stat": "process_median"`` has one
*sample per process*: the median of the cell's trials (at least ``MIN_TRIALS``) in that record file; the comparison then takes one value per record
file and the spread is taken over those values (at least three interleaved record files per build).

Rules enforced by ``compare``: a baseline record must carry the expected commit, a clean tree and the GPU / SM metadata; every record of a
comparison comes from the same GPU model, SM count and host; each build has at least ``min_repeats`` record files that start in the required order (strict
alternation A B A B ..., or blocks of four that are ABBA or BAAB; the start stamps are parsed, never compared as text, and a stamp that is missing, does not parse or is
shared by two records, or one record file that is listed on both builds, leaves the order undefined and the comparison INVALID); a record can be required to carry the
launch mapping of every trial (``require_nodes``), and ``route_problems`` checks those mappings against an independent specification;
a metric that is absent or zero in a cell (a decode cell has no K2 stage) makes the comparison INVALID. The samples are validated per compared cell and metric,
never by the number of files: cell keys are unique inside a file, every file of a side holds the same cells (the campaign cell set is not inferred from the
first file), the two builds hold the same cells, every raw sample of the metric is finite (and positive for a derived or wall metric), a per-process-median
record has at least ``MIN_TRIALS`` such trials per cell, and every cell has at least ``min_repeats`` distinct process samples per side; anything else is INVALID.
A baseline cell whose spread exceeds 0.3 % is not accepted as a gate baseline (``invalid``), is judged on its worst repeat (``worst``) or, in the widening mode,
needs a re-run: the gate is INVALID until re-run records prove three times the calls per trial (and profiled calls) of the first campaign for that cell on both
sides from the same snapshot and configuration, after which the cell is judged on the re-run and, if it stays above 0.3 %, as noise-limited with the tolerance
widened by the larger of the two spreads; the rows of a first campaign without re-run stay available as a diagnostic (``needs_rerun``) for the runner.
"""
import datetime
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
from collections.abc import Mapping

SCHEMA = "cfbi-bench-v1"
UNSTABLE_SPREAD = 0.003  # 0.3 %: repeat-to-repeat spread above which a cell is unstable
REQUIRED_META = ("commit", "dirty", "gpu", "sm_count", "host")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{7,40}$")  # a commit hash or an abbreviation of at least seven hexadecimal characters
DERIVED_METRICS = ("total_us", "k1_us", "k2_stage_us", "k2_main_us")
WALL_METRICS = ("gpu_graph_us", "gpu_eager_us", "cpu_graph_us", "cpu_eager_us")
SUMMARY_METRICS = ("k1_us", "k2_stage_us", "total_us")  # default metrics of a per-build summary
PROCESS_MEDIAN = "process_median"
MIN_TRIALS = 5  # trials per cell and process of a process-median record
UNSTABLE_POLICIES = ("invalid", "worst", "widen")
RERUN_SCALE = 3  # calls per trial (and profiled calls) of a re-run relative to the first campaign
# meta fields that are one value for all record files of a side (the last three: whether the single-launch control was on, the limit constant of the measured tree, and whether one graph
# replay per trial was profiled)
SIDE_META = ("commit", "tree", "src", "impl", "env_impl", "torch", "cuda", "cc", "trials", "sample_stat", "kind", "single_force", "single_per_sm", "graph_nodes")
RERUN_SAME_CONFIG = SIDE_META + ("gpu", "sm_count", "host")  # meta fields a re-run shares with its first campaign
AB_META = ("trials", "torch", "cuda", "cc", "kind", "dm", "heads", "l2_cache_bytes", "eviction_bytes")  # the baseline and the candidate are measured under one contract and one toolchain
AB_FLAGS = ("graph_nodes",)  # boolean options of the benchmark that both builds must share (a record of before the option counts as False): profiling a replay is no neutral act for the next trial
# cell arguments that describe a build, which A and B may differ in (the last two: the single-launch control, and whether the expected launches came from a per-cell specification)
BUILD_ARGS = ("impl", "kernels", "compute_nodes", "aux_nodes", "compute_kernels", "expect_compute", "expect_aux", "single_force", "expect_route")
SCALED_ARGS = ("calls_per_trial", "profile_calls", "profile_cycles", "calls_scale")  # cell arguments that a re-run multiplies
ORDERS = ("alternate", "abba")  # the start orders of the record files of the two builds: strict alternation A B A B ..., or blocks of four that are ABBA or BAAB
ABBA_BLOCKS = ("ABBA", "BAAB")  # the blocks of the abba order: the outer two record files of a block belong to one build, the inner two to the other
STAMP_FORMATS = ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ")  # the start stamps of the record files (UTC): with a fraction of a second, as the benchmarks write them, and without


# ---- kernel names and stages ---------------------------------------------------------------------------------------------------------------

def kernel_base(name):
    """``void cfbi::k2_kernel<16>(cfbi::Params)`` -> ``k2_kernel`` (``ref::`` of the frozen reference kernels is kept)."""
    name = name.split("(")[0].split("<")[0].strip()
    if name.startswith("void "):
        name = name[5:]
    return name.replace("cfbi::", "", 1)


def is_helper_kernel(name):
    """True for a kernel that a benchmark launches around the measured call (L2 eviction writes, ``pos`` rewrites, copies, memsets): every CF-BI kernel
    lives in the namespace ``cfbi`` (``void cfbi::rows_kernel(cfbi::Params)``, ``void cfbi::ref::rows_kernel(...)``), everything else is helper work."""
    return "cfbi::" not in name


def _without_ref(base):
    """``ref::k2_kernel`` -> ``k2_kernel``: the frozen reference kernels belong to the same stages as the optimised ones."""
    return base[len("ref::"):] if base.startswith("ref::") else base


def kernel_stage(base):
    """``k1`` | ``k2`` | ``other``: K2 stage = the K2 kernels (any implementation), the schedule builder / reset kernels named ``k2*``, memsets."""
    b = _without_ref(base)
    if b.startswith("k1"):
        return "k1"
    if b.startswith(("k2", "Memset")):
        return "k2"
    return "other"


def is_k2_main(base):
    """The K2 kernels proper (``k2_kernel*``, ``k2u_kernel``, the new schedule's K2 kernel), without builder / reset kernels."""
    return _without_ref(base).startswith(("k2_kernel", "k2u_kernel", "k2_dyn_kernel"))


def derive(kernels):
    """Per-sample derived metrics from a ``{kernel base name: us}`` dict."""
    total = sum(kernels.values())
    k1 = sum(v for k, v in kernels.items() if kernel_stage(k) == "k1")
    k2 = sum(v for k, v in kernels.items() if kernel_stage(k) == "k2")
    main = sum(v for k, v in kernels.items() if is_k2_main(k))
    return {"kernels": dict(kernels), "total_us": total, "k1_us": k1, "k2_stage_us": k2, "k2_main_us": main}


def make_cell(kind, key, args, runs, drop_helpers=False, wall=None):
    """A cell from ``runs``: a list (one entry per in-process repeat or trial) of ``{raw kernel name: us per call}`` dicts.

    ``drop_helpers``: leave the helper kernels (``is_helper_kernel``) out of the kernel sum and keep their time as ``helper_us`` of the sample.
    ``wall``: optional list, one dict of wall metrics (``WALL_METRICS``) per run, merged into the samples."""
    if wall is not None and len(wall) != len(runs):
        raise ValueError("wall needs one entry per run (%d != %d)" % (len(wall), len(runs)))
    samples = []
    for i, run in enumerate(runs):
        folded, helper = {}, 0.0
        for name, us in run.items():
            if drop_helpers and is_helper_kernel(name):
                helper += float(us)
                continue
            base = kernel_base(name)
            folded[base] = folded.get(base, 0.0) + float(us)
        sample = derive(folded)
        if drop_helpers:
            sample["helper_us"] = helper
        if wall is not None:
            for m, v in wall[i].items():
                if m not in WALL_METRICS:
                    raise ValueError("unknown wall metric %r" % m)
                if v is not None:
                    sample[m] = float(v)
        samples.append(sample)
    return {"kind": kind, "key": key, "args": args, "samples": samples}


def cell_key(kind, **fields):
    """A cell key that ends every field with a delimiter, so that a filter on ``N=1;`` does not match ``N=16;``: ``decode;N=1;H=12;regime=lockstep;``."""
    return kind + ";" + "".join("%s=%s;" % (k, v) for k, v in fields.items())


# ---- the kernels that one call (or one graph replay) launches: the exact mapping of every trial ---------------------------------------------------------------

def split_nodes(counts):
    """``(compute, aux)`` from ``{raw kernel name: launches per call or replay}``: compute kernels are the CF-BI kernels (``{base name: count}``), everything else
    (memsets, copies, ATen kernels) is auxiliary."""
    compute, aux = {}, {}
    for name, n in counts.items():
        if is_helper_kernel(name):
            aux[name.strip()] = aux.get(name.strip(), 0) + n
        else:
            base = kernel_base(name)
            compute[base] = compute.get(base, 0) + n
    return compute, aux


def parse_mapping(text, presets=None):
    """``{name: count}`` from ``NAME=COUNT;NAME=COUNT`` (``none`` or the empty string: the empty mapping; a preset name: that mapping)."""
    text = text.strip()
    if presets and text in presets:
        return dict(presets[text])
    if text in ("", "none"):
        return {}
    out = {}
    for item in text.split(";"):
        name, sep, count = item.rpartition("=")
        if not sep or not name.strip() or not count.strip().isdigit() or int(count) < 1 or name.strip() in out:
            raise ValueError("%r is not NAME=COUNT (a positive count, every name once)" % item)
        out[name.strip()] = int(count)
    return out


def check_nodes(compute, aux, expect_compute, expect_aux):
    """None when one call (or replay) holds exactly the expected nodes, otherwise the message. ``compute`` / ``aux``: ``{name: launches per call}`` from the profiler counts of
    one trial; a fractional count (a launch that is missing from some calls) is a defect of its own. ``expect_compute``: the exact mapping of the compute kernels
    (None = no assertion); ``expect_aux``: the exact mapping of the auxiliary nodes that may remain (``{}`` asserts that there is none, None = no assertion)."""
    for what, got in (("compute", compute), ("auxiliary", aux)):
        fractional = {k: round(v, 3) for k, v in got.items() if abs(v - round(v)) > 1e-6}
        if fractional:
            return "%s nodes are not launched a whole number of times per call: %s" % (what, dict(sorted(fractional.items())))
    if expect_compute is not None:
        got = {k: int(round(v)) for k, v in compute.items()}
        if got != dict(expect_compute):
            return "the step launches the compute kernels %s, expected %s" % (dict(sorted(got.items())), dict(sorted(expect_compute.items())))
    if expect_aux is not None:
        got = {k: int(round(v)) for k, v in aux.items()}
        if got != dict(expect_aux):
            return "the graph holds the auxiliary nodes %s, expected %s" % (dict(sorted(got.items())), dict(sorted(expect_aux.items())))
    return None


def cell_node_problems(cell, expect_compute=None, expect_aux=None):
    """Why the per-trial node mappings of a recorded cell (a decode or a chain cell) are not the asserted ones (an empty list means every trial carries them): the cell records its
    ``compute_nodes`` / ``aux_nodes``, they equal the given expectation when there is one, and the ``nodes`` of EVERY sample equal them."""
    out, args, key = [], cell.get("args", {}), cell.get("key")
    samples = cell.get("samples") or []
    for name in ("compute_nodes", "aux_nodes"):
        if not isinstance(args.get(name), dict):
            out.append("cell %r records no %s" % (key, name))
    if not samples:
        out.append("cell %r has no samples" % key)
    if out:
        return out
    for what, want, claimed in (("compute", expect_compute, args["compute_nodes"]), ("auxiliary", expect_aux, args["aux_nodes"])):
        if want is not None and dict(want) != dict(claimed):
            out.append("cell %r: the recorded %s nodes %s are not the expected %s" % (key, what, dict(sorted(claimed.items())), dict(sorted(want.items()))))
    for i, s in enumerate(samples):
        nodes = s.get("nodes")
        if not isinstance(nodes, dict) or not isinstance(nodes.get("compute"), dict) or not isinstance(nodes.get("aux"), dict):
            out.append("cell %r: trial %d carries no node mapping" % (key, i))
            continue
        problem = check_nodes(nodes["compute"], nodes["aux"], args["compute_nodes"], args["aux_nodes"])
        if problem:
            out.append("cell %r: trial %d: %s" % (key, i, problem))
    return out


def route_problems(records, expected_by_key, side="side"):
    """Why the launches that the record files ``records`` of one build (named ``side`` in the messages) recorded are not the ones that an independent specification expects (an empty
    list means every trial of every cell launched exactly the expected kernels). ``expected_by_key``: ``{cell key: {kernel: launches per call}}``, the mapping that a cell must show
    whatever the record claims about itself. Every cell of every record needs a specification (a missing key is a problem, not a skipped cell) and the per-trial launch mappings (a
    record of before they were recorded is a problem), and every trial must equal the specification (``cell_node_problems``): a cell that was measured on the other launch path is a
    problem. A specification for a cell that no record holds is not examined here (which cells a campaign holds is the business of the cell-set checks). Without records there is
    nothing to check."""
    out = []
    for rec in records:
        where = "%s record %s" % (side, _path_of(rec))
        cells = rec.get("cells") if isinstance(rec, dict) else None
        if not isinstance(cells, list) or not cells:
            out.append("%s: no cells" % where)
            continue
        bad = []
        for cell in cells:
            key = cell.get("key") if isinstance(cell, dict) else None
            want = expected_by_key.get(key) if isinstance(expected_by_key, Mapping) and isinstance(key, str) else None
            if not isinstance(key, str):
                bad.append("a cell without a key")
            elif not isinstance(want, Mapping):
                bad.append("cell %r has no expected launch mapping" % key)
            else:
                try:
                    found = cell_node_problems(cell, want)
                except (AttributeError, KeyError, TypeError, ValueError) as e:      # a count that is no number, a sample that is no object: a problem, not an exception
                    found = ["cell %r: the recorded launch mapping cannot be read (%s: %s)" % (key, type(e).__name__, e)]
                if found:
                    bad.append(found[0] + (" (and %d more problem(s) in this cell)" % (len(found) - 1) if len(found) > 1 else ""))
        out.extend("%s: %s" % (where, p) for p in bad[:6])
        if len(bad) > 6:
            out.append("%s: ... and %d more cell(s) whose launches are not the expected ones" % (where, len(bad) - 6))
    return out


# ---- identity of the measured tree ---------------------------------------------------------------------------------------------------------

def tree_identity(root):
    """``(commit, dirty, tree)`` of the source tree ``root``.

    Priority: the environment (``CFBI_BENCH_COMMIT``, ``CFBI_BENCH_DIRTY``, ``CFBI_BENCH_TREE``, set by the node driver, which has git) > git
    (untracked paths under ``.humanize/`` do not make a tree dirty) > the ``SNAPSHOT_COMMIT`` file of a ``git archive`` snapshot.
    """
    commit = os.environ.get("CFBI_BENCH_COMMIT")
    if commit:
        return commit, os.environ.get("CFBI_BENCH_DIRTY", "0") not in ("0", "", "false", "False"), os.environ.get("CFBI_BENCH_TREE", "worktree")
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.STDOUT).decode().strip()
        status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], stderr=subprocess.STDOUT).decode().splitlines()
        dirty = any(not line.startswith("?? .humanize/") for line in status)
        return commit, dirty, "worktree"
    except (OSError, subprocess.CalledProcessError):
        pass
    marker = os.path.join(str(root), "SNAPSHOT_COMMIT")
    if os.path.exists(marker):
        with open(marker) as f:
            return f.read().strip(), False, "snapshot"
    return None, None, None


# ---- records ---------------------------------------------------------------------------------------------------------------------------------

def write_record(path, meta, cells):
    with open(path, "w") as f:
        json.dump({"schema": SCHEMA, "meta": meta, "cells": cells}, f, indent=1, sort_keys=True)
        f.write("\n")


def load_record(path):
    with open(path) as f:
        rec = json.load(f)
    if not isinstance(rec, dict):
        raise ValueError("%s: a record must be a JSON object" % path)
    rec["_path"] = str(path)
    return rec


def sample_stat_of(rec):
    """``"pooled"`` (default) or ``PROCESS_MEDIAN``: how the samples of a record file enter a comparison."""
    meta = rec.get("meta") if isinstance(rec, dict) else None
    return (meta or {}).get("sample_stat", "pooled")


def structure_problems(rec, require_nodes=False):
    """Why ``rec`` cannot be used at all (wrong schema, missing or malformed parts); an empty list means it can be compared. A cell whose samples carry per-trial launch mappings
    (``nodes``, see ``cell_node_problems``) must carry them consistently: every trial equal to the cell's own mapping. A cell without any (a record of before the mappings were
    recorded) stays valid, unless the caller requires them (``require_nodes``: the records of a campaign whose launch path is asserted)."""
    if not isinstance(rec, dict):
        return ["record is not a JSON object"]
    where = "record %s" % rec.get("_path", "?")
    if rec.get("schema") != SCHEMA:
        return ["%s: schema %r, expected %r" % (where, rec.get("schema"), SCHEMA)]
    problems = []
    if not isinstance(rec.get("meta"), dict):
        problems.append("%s: no metadata object" % where)
    cells = rec.get("cells")
    if not isinstance(cells, list) or not cells:
        problems.append("%s: no cells" % where)
        return problems
    for i, cell in enumerate(cells):
        samples = cell.get("samples") if isinstance(cell, dict) else None
        if not isinstance(cell, dict) or not isinstance(cell.get("key"), str) or not isinstance(samples, list) or not samples:
            problems.append("%s: cell %d has no key or no samples" % (where, i))
            continue
        for s in samples:
            if (not isinstance(s, dict) or not isinstance(s.get("kernels"), dict) or not all(isinstance(s.get(m), (int, float)) for m in DERIVED_METRICS)
                    or not all(isinstance(s[m], (int, float)) for m in WALL_METRICS if m in s)):
                problems.append("%s: cell %r has a malformed sample" % (where, cell["key"]))
                break
        else:
            if require_nodes or any("nodes" in s for s in samples):
                problems.extend("%s: %s" % (where, p) for p in cell_node_problems(cell))
    return problems


def known_metric(records, metric):
    """True when ``metric`` is a derived metric, or a wall metric or the name of a kernel that appears in at least one sample (records must be well formed)."""
    if metric in DERIVED_METRICS:
        return True
    if metric in WALL_METRICS:
        return any(metric in s for rec in records for cell in rec["cells"] for s in cell["samples"])
    return any(metric in s["kernels"] for rec in records for cell in rec["cells"] for s in cell["samples"])


def identity_problems(meta, where, expect_commit=None, require_clean=True, required=REQUIRED_META):
    """Problems of the identity fields of a record's metadata: commit (hexadecimal, at least 7 characters, the expected one when given), dirty (a boolean, false when a clean tree is
    required), GPU and host (non-empty strings) and SM count (a positive integer). One validation for every kind of record (``where`` names the record in the messages)."""
    problems = []
    for field in required:
        if meta.get(field) is None:
            problems.append("%s: metadata field %r is missing" % (where, field))
    commit = meta.get("commit")
    if commit is not None:
        if not (isinstance(commit, str) and COMMIT_PATTERN.match(commit)):
            problems.append("%s: commit %r is not a hexadecimal commit hash of at least 7 characters" % (where, commit))
        elif expect_commit:
            if not (isinstance(expect_commit, str) and COMMIT_PATTERN.match(expect_commit)):
                problems.append("the expected commit %r is not a hexadecimal hash of at least 7 characters" % (expect_commit,))
            elif not commit.startswith(expect_commit) and not expect_commit.startswith(commit):
                problems.append("%s: commit %s is not the expected %s" % (where, commit, expect_commit))
    dirty = meta.get("dirty")
    if dirty is not None and type(dirty) is not bool:
        problems.append("%s: metadata field 'dirty' must be a boolean, got %r" % (where, dirty))
    elif require_clean and dirty is True:
        problems.append("%s: the measured tree was dirty" % where)
    for field in ("gpu", "host"):
        if meta.get(field) is not None and not (isinstance(meta[field], str) and meta[field].strip()):
            problems.append("%s: metadata field %r must be a non-empty string, got %r" % (where, field, meta[field]))
    if meta.get("sm_count") is not None and not _positive_int(meta["sm_count"]):
        problems.append("%s: metadata field 'sm_count' must be a positive integer, got %r" % (where, meta["sm_count"]))
    return problems


def validate_record(rec, role="baseline", expect_commit=None, require_clean=True, require_nodes=False):
    """Problems of one record (an empty list means valid). ``role`` is only used in the messages."""
    structural = structure_problems(rec, require_nodes)
    if structural:
        return ["%s: %s" % (role, p) for p in structural]
    where = "%s record %s" % (role, rec.get("_path", "?"))
    meta = rec["meta"]
    problems = identity_problems(meta, where, expect_commit, require_clean)
    stat = sample_stat_of(rec)
    if stat not in ("pooled", PROCESS_MEDIAN):
        problems.append("%s: unknown sample statistic %r" % (where, stat))
    elif stat == PROCESS_MEDIAN:
        declared = meta.get("trials")
        if not _positive_int(declared):
            problems.append("%s: a per-process-median record must declare its trials per cell (meta.trials, a positive integer), got %r" % (where, declared))
        for cell in rec["cells"]:
            n = len(cell["samples"])
            if n < MIN_TRIALS:
                problems.append("%s: cell %r has %d trial(s), at least %d are required for a per-process median" % (where, cell["key"], n, MIN_TRIALS))
            if _positive_int(declared) and n != declared:
                problems.append("%s: cell %r holds %d trial(s) but the record declares %d" % (where, cell["key"], n, declared))
    return problems


# ---- statistics ------------------------------------------------------------------------------------------------------------------------------

def metric_value(sample, metric):
    """The value of ``metric`` in one sample; None for a wall metric that the sample does not carry."""
    if metric in DERIVED_METRICS:
        return sample[metric]
    if metric in WALL_METRICS:
        return sample.get(metric)
    return sample["kernels"].get(metric, 0.0)


def pooled_values(records, key, metric):
    """The values of cell ``key`` that enter a comparison: every sample of every record (pooled records) or one median per record (process-median records)."""
    vals = []
    for rec in records:
        per = [metric_value(s, metric) for cell in rec["cells"] if cell["key"] == key for s in cell["samples"]]
        per = [v for v in per if v is not None]
        if not per:
            continue
        if sample_stat_of(rec) == PROCESS_MEDIAN:
            vals.append(statistics.median(per))
        else:
            vals.extend(per)
    return vals


def cell_kind(records, key):
    for rec in records:
        for cell in rec["cells"]:
            if cell["key"] == key:
                return cell.get("kind", "?")
    return "?"


def spread_of(vals):
    """``(max - min) / median`` of the values (0 for fewer than two values)."""
    if len(vals) < 2:
        return 0.0
    med = statistics.median(vals)
    return (max(vals) - min(vals)) / med if med else 0.0


def summarize(records, metrics=SUMMARY_METRICS):
    """Per-cell statistics over the pooled samples of the records of ONE build: a list of row dicts (``stable`` = spread <= 0.3 %)."""
    keys = []
    for rec in records:
        for cell in rec["cells"]:
            if cell["key"] not in keys:
                keys.append(cell["key"])
    rows = []
    for key in keys:
        for metric in metrics:
            vals = pooled_values(records, key, metric)
            if not vals:
                continue
            sp = spread_of(vals)
            rows.append({"key": key, "metric": metric, "n": len(vals), "median": statistics.median(vals), "min": min(vals), "max": max(vals),
                         "spread_pct": 100 * sp, "stable": sp <= UNSTABLE_SPREAD})
    return rows


def format_summary(rows):
    lines = ["%-44s %-12s %3s %12s %8s  %s" % ("cell", "metric", "n", "median us", "sprd %", "status")]
    for r in rows:
        lines.append("%-44s %-12s %3d %12.1f %8.2f  %s" % (r["key"][:44], r["metric"], r["n"], r["median"], r["spread_pct"],
                                                         "stable" if r["stable"] else "UNSTABLE (spread > %.1f %%)" % (100 * UNSTABLE_SPREAD)))
    return "\n".join(lines)


# ---- the start order of the record files of the two builds -----------------------------------------------------------------------------------

def parse_stamp(text):
    """The ``datetime`` of a record's start stamp (``STAMP_FORMATS``), or None when it is not a string or does not parse. Stamps are compared as instants, never as text:
    ``...:00.5Z`` is later than ``...:00Z`` although it sorts before it as a string."""
    if not isinstance(text, str):
        return None
    for fmt in STAMP_FORMATS:
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _path_of(rec):
    return rec.get("_path", "?") if isinstance(rec, dict) else "?"


def _start_order(base, cand):
    """``(labels, [])`` or ``(None, why)``: the label string of the record files in start order (``"A"`` for a record of ``base``, ``"B"`` for one of ``cand``), or the reasons why the
    order is not defined: a record without a start stamp, a stamp that does not parse, or two records that started at the same instant (whichever sides they are on)."""
    why, stamped, seen = [], [], {}
    for label, side, recs in (("A", "baseline", base), ("B", "candidate", cand)):
        for rec in recs:
            meta = rec.get("meta") if isinstance(rec, dict) else None
            raw = meta.get("timestamp") if isinstance(meta, dict) else None
            when = parse_stamp(raw)
            who = "%s record %s" % (side, _path_of(rec))
            if raw is None:
                why.append("%s has no timestamp" % who)
            elif when is None:
                why.append("%s has the timestamp %r, which is not a UTC start stamp" % (who, raw))
            else:
                stamped.append((when, label))
                seen.setdefault(when, []).append(who)
    why.extend("%s start at the same instant, %s" % (" and ".join(who), when.isoformat()) for when, who in sorted(seen.items()) if len(who) > 1)
    if why:
        return None, why
    return "".join(label for _, label in sorted(stamped)), []


def run_labels(base, cand):
    """The label string of the record files of ``base`` (A) and ``cand`` (B) in start order, e.g. ``"ABBA"``, from ``meta.timestamp`` parsed as an instant; None when the order is not
    defined (a stamp is missing or does not parse, or two records share one)."""
    return _start_order(base, cand)[0]


def run_plan(samples, order="alternate"):
    """The runs of one campaign in start order, ``[(side, index within the side)]`` with indices from 1. ``alternate``: A B A B ... (the baseline first); ``abba``: blocks of four, ABBA
    for the even blocks and BAAB for the odd ones, so that the two builds start at the same mean position (for an even number of samples) and, for whole pairs of blocks (a number of
    samples that is a multiple of four), at the same mean squared position. It needs an even number of samples of at least four."""
    if order not in ORDERS:
        raise ValueError("order must be one of %s, got %r" % (", ".join(ORDERS), order))
    if not _positive_int(samples):
        raise ValueError("samples must be a positive integer, got %r" % (samples,))
    if order == "alternate":
        return [(side, i) for i in range(1, samples + 1) for side in ("A", "B")]
    if samples < 4 or samples % 2:
        raise ValueError("the abba order needs an even number of samples of at least 4, got %d" % samples)
    plan, taken = [], {"A": 0, "B": 0}
    for j in range(samples // 2):
        for side in ABBA_BLOCKS[j % 2]:
            taken[side] += 1
            plan.append((side, taken[side]))
    return plan


def _switches(labels):
    """The number of places where two neighbouring record files of the label string belong to different builds."""
    return sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])


def _file_id(rec):
    """``(device, inode)`` of the file that a loaded record came from; None when ``_path`` names no existing file (a record that was built in memory)."""
    path = rec.get("_path") if isinstance(rec, dict) else None
    if not isinstance(path, str):
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_dev, st.st_ino)


def _digest(rec):
    """SHA-256 of the content of a record, without the path that it was loaded from."""
    body = dict((k, v) for k, v in rec.items() if k != "_path") if isinstance(rec, dict) else rec
    try:
        text = json.dumps(body, sort_keys=True, default=repr)
    except (TypeError, ValueError):                                    # a record that was built in memory with keys that cannot be sorted
        text = repr(body)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shared_record_problems(base, cand, label):
    """Why ``base`` and ``cand``, the record files of the two builds of the campaign ``label``, are not two sets of measurements: a record file that is listed on both sides, whether
    it is named by the same path (the same file on disk, however it is spelled) or by the same content (a copy). One file measured once cannot be both builds."""
    by_file, by_digest = {}, {}
    for rec in base:
        file_id = _file_id(rec)
        if file_id is not None:
            by_file.setdefault(file_id, rec)
        by_digest.setdefault(_digest(rec), rec)
    out = []
    for rec in cand:
        file_id, digest = _file_id(rec), _digest(rec)
        if file_id is not None and file_id in by_file:
            out.append("the baseline record %s and the candidate record %s of the %s are the same file" % (_path_of(by_file[file_id]), _path_of(rec), label))
        elif digest in by_digest:
            out.append("the baseline record %s and the candidate record %s of the %s have the same content" % (_path_of(by_digest[digest]), _path_of(rec), label))
    return out[:4] + (["... and %d more record(s) of the %s are listed on both sides" % (len(out) - 4, label)] if len(out) > 4 else [])


def _abba_problem(labels, label):
    """None when the label string is a whole number of blocks of ``ABBA_BLOCKS`` with as many A as B, otherwise the message naming what breaks the grammar (the first violated rule: unequal
    counts, a length that is not a multiple of four, the first block that is neither ABBA nor BAAB), the label string and its number of variant switches."""
    n_a, n_b = labels.count("A"), labels.count("B")
    reason = None
    if n_a != n_b:
        reason = "%d baseline and %d candidate record files, the same number is needed on both sides" % (n_a, n_b)
    elif len(labels) % 4:
        reason = "%d record files are not a whole number of blocks of four" % len(labels)
    else:
        for i in range(0, len(labels), 4):
            if labels[i:i + 4] not in ABBA_BLOCKS:
                reason = "the block of record files %d to %d is %s" % (i + 1, i + 4, labels[i:i + 4])
                break
    if reason is None:
        return None
    return "baseline and candidate records of the %s are not in blocks of %s in time (start order %s, %d variant switches): %s" % (
        label, " or ".join(ABBA_BLOCKS), labels, _switches(labels), reason)


def order_problems(base, cand, order, label):
    """Why the record files of the two builds of one campaign (``label``: ``first campaign`` or ``re-run``) did not start in the required ``order`` (an empty list means they did).

    Every order needs a defined start order (a parsed stamp on every record, no two records at one instant) and two disjoint sets of record files (``shared_record_problems``).
    ``alternate``: the files alternate strictly, i.e. the label string has at least ``2 * min(len(base), len(cand)) - 1`` variant switches. ``abba``: the label string is a whole number of
    blocks of four, every aligned block is ABBA or BAAB, and both builds have the same number of files (a switch count cannot tell the valid strings from the others: ABBAABBA has four
    switches and ABBABAAB five, both are valid, and AABBABBA, which is not, has four as well)."""
    out = shared_record_problems(base, cand, label)
    labels, why = _start_order(base, cand)
    if labels is None:
        return out + ["the start order of the %s records is not defined (%s)" % (label, "; ".join(why[:4]) + (" ..." if len(why) > 4 else ""))]
    if order == "alternate":
        need = 2 * min(len(base), len(cand)) - 1
        got = _switches(labels)
        if got < need:
            out.append("baseline and candidate records of the %s are not interleaved in time (%d variant switches, %d needed)" % (label, got, need))
    elif order == "abba":
        problem = _abba_problem(labels, label)
        if problem:
            out.append(problem)
    else:
        out.append("order must be one of %s, got %r" % (", ".join(ORDERS), order))
    return out


# ---- the comparison --------------------------------------------------------------------------------------------------------------------------

def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def raw_sample_problems(rec, key, metric):
    """Why the raw samples of cell ``key`` in record ``rec`` cannot enter a comparison of ``metric`` (an empty list means valid): every sample must carry the
    metric, finite, and positive for a derived or wall metric (a kernel metric may be zero: the kernel may not exist in a build). Nothing is dropped silently;
    one message per kind of defect with the number of samples that have it."""
    strict = metric in DERIVED_METRICS or metric in WALL_METRICS
    n = absent = bad = zero = 0
    for cell in rec["cells"]:
        if cell["key"] != key:
            continue
        for s in cell["samples"]:
            n += 1
            v = metric_value(s, metric)
            if v is None:
                absent += 1
            elif not _finite(v) or v < 0:
                bad += 1
            elif strict and v == 0:
                zero += 1
    out = []
    if absent:
        out.append("%d of %d sample(s) carry no %s" % (absent, n, metric))
    if bad:
        out.append("%d of %d sample(s) have a non-finite or negative %s" % (bad, n, metric))
    if zero:
        out.append("%d of %d sample(s) have a zero %s (not defined for this kind of cell, or lost in this build)" % (zero, n, metric))
    return out


def campaign_cell_problems(records, role):
    """``(problems, union of the cell keys, keys that not every file holds)`` of the record files of one side: unique keys inside every file and the same cells in every file."""
    problems, sets = [], []
    for rec in records:
        keys = [c["key"] for c in rec["cells"]]
        dup = sorted(set(k for k in keys if keys.count(k) > 1))
        if dup:
            problems.append("%s record %s repeats the cell key(s) %s" % (role, rec.get("_path", "?"), dup))
        sets.append(set(keys))
    union = set().union(*sets) if sets else set()
    for rec, keys in zip(records, sets):
        if keys != union:
            problems.append("%s record %s lacks the cell(s) %s that other records of the same side hold" % (role, rec.get("_path", "?"), sorted(union - keys)[:4]))
    return problems, union, union - (set.intersection(*sets) if sets else set())


def uses_profiler(metric):
    """True when the compared metric comes from the profiled pass of a trial (the kernel sum, a derived kernel metric or one kernel); only the GPU-elapsed and the CPU
    wall metrics are measured without it. The effective profiler calls are part of the proof of a re-run for the former and not for the latter."""
    return metric not in WALL_METRICS


def count_names(metric):
    """The recorded counts that a comparison of ``metric`` depends on: the calls per trial, and for a profiled metric the profiled calls too."""
    return ("calls_per_trial", "profile_calls") if uses_profiler(metric) else ("calls_per_trial",)


def _positive_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _arg_values(records, key, name):
    """The distinct values of the cell argument ``name`` over the records (a missing argument is ``None``), in a fixed order."""
    vals = {}
    for r in records:
        for c in r["cells"]:
            if c["key"] == key:
                v = (c.get("args") or {}).get(name)
                vals[json.dumps(v, sort_keys=True)] = v
    return [vals[k] for k in sorted(vals)]


def rerun_problems(first, rerun, side, key, metric, same_config=RERUN_SAME_CONFIG):
    """Why the re-run records of ``side`` do not prove a three-times-calls re-run of cell ``key`` of the first campaign for the compared ``metric`` (an empty list means
    they do). The counts of ``count_names(metric)`` must each be recorded as one strictly positive integer in the first campaign and in the re-run, and the re-run's must
    be exactly three times the first campaign's: a count that is absent on both sides is not read as "there was no such measurement". ``same_config``: the meta fields that the
    two record lists must share (a confirmation campaign on another node shares all of them but the host)."""
    out = []
    for name in count_names(metric):
        f, r = _arg_values(first, key, name), _arg_values(rerun, key, name)
        if len(f) != 1 or len(r) != 1 or not _positive_int(f[0]) or not _positive_int(r[0]):
            out.append("%s %s of cell %r is not recorded as one positive integer in the first campaign and in the re-run (first campaign %s, re-run %s)" % (side, name, key, f, r))
        elif r[0] != RERUN_SCALE * f[0]:
            out.append("%s re-run of cell %r has %s = %s, not %d times the first campaign's %s" % (side, key, name, r[0], RERUN_SCALE, f[0]))
    for field, a, b in config_differences(first, rerun, same_config):
        out.append("%s re-run does not come from the same snapshot / configuration as its first campaign (meta %s: %s vs %s)" % (side, field, a, b))
    return out


def _meta_values(records, field):
    return sorted(set(json.dumps((r.get("meta") or {}).get(field), sort_keys=True) for r in records))


def _flag_values(records, field):
    """The sorted values that a boolean meta field takes over the records; a record without the field (of before the option) counts as False."""
    return sorted(set(bool((r.get("meta") or {}).get(field)) for r in records))


def config_differences(first, other, fields):
    """``[(field, values of first, values of other)]`` for the meta ``fields`` that do not hold one value that both record lists share (the values are the sorted JSON texts of the field in
    each list): the configuration that two campaigns must share (the snapshot, the routing, the toolchain, the statistic, the device) to measure the same thing."""
    out = []
    for field in fields:
        a, b = _meta_values(first, field), _meta_values(other, field)
        if len(a) != 1 or a != b:
            out.append((field, a, b))
    return out


def side_config_problems(records, role):
    """Why the record files of one side (a first campaign, or a re-run) are not one fixed configuration: every meta field of ``SIDE_META`` is one value over the files
    (one commit, tree, source directory, implementation, toolchain and statistic: three candidate processes of different commits are not one candidate), and every cell
    has the same arguments (workload, counts, launched kernels) in every file. This holds for a stable first campaign as well as for a re-run."""
    out = []
    for field in SIDE_META:
        values = _meta_values(records, field)
        if len(values) > 1:
            out.append("%s records disagree on meta %s: %s" % (role, field, ", ".join(values)))
    cells = {}
    for rec in records:
        for cell in rec["cells"]:
            cells.setdefault(cell["key"], set()).add(json.dumps(cell.get("args") or {}, sort_keys=True))
    for key in sorted(cells):
        if len(cells[key]) > 1:
            out.append("%s records disagree on the arguments of cell %r: %s" % (role, key, " | ".join(sorted(cells[key]))[:300]))
    return out


def workload_problems(records_a, records_b, name_a, name_b, ignore):
    """Why two record lists do not measure the same workload: for every cell that both hold, the cell arguments except those named in ``ignore`` are equal."""
    cells_a = dict((c["key"], c) for r in records_a for c in r["cells"])
    cells_b = dict((c["key"], c) for r in records_b for c in r["cells"])
    out = []
    for key in sorted(set(cells_a) & set(cells_b)):
        a = dict((k, v) for k, v in (cells_a[key].get("args") or {}).items() if k not in ignore)
        b = dict((k, v) for k, v in (cells_b[key].get("args") or {}).items() if k not in ignore)
        if a != b:
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            out.append("cell %r: %s and %s differ in %s (%s vs %s)" % (key, name_a, name_b, ", ".join(diff), [a.get(k) for k in diff], [b.get(k) for k in diff]))
    return out


def ab_problems(base, cand, name):
    """Why the baseline and the candidate records of one campaign (``name``) were not measured under the same contract: the same trials, toolchain versions and cell
    workload / call / profiled-call counts. Their commit, source directory and implementation may differ, that is what is compared."""
    out = ["the baseline and the candidate %s differ in meta %s (%s vs %s)" % (name, field, ", ".join(_meta_values(base, field)), ", ".join(_meta_values(cand, field)))
           for field in AB_META if _meta_values(base, field) != _meta_values(cand, field)]
    out += ["the baseline and the candidate %s differ in meta %s (%s vs %s)" % (name, field, ", ".join(map(str, _flag_values(base, field))), ", ".join(map(str, _flag_values(cand, field))))
            for field in AB_FLAGS if _flag_values(base, field) != _flag_values(cand, field)]
    return out + workload_problems(base, cand, "the baseline %s" % name, "the candidate %s" % name, BUILD_ARGS)


def compare(base, cand, metric="k2_stage_us", tol_pct=1.0, mode="regress", expect_base_commit=None, require_clean_cand=False,
            min_repeats=3, require_interleaved=True, cells=None, baseline_unstable="invalid", base_rerun=None, cand_rerun=None, order="alternate", require_nodes=False):
    """Compare the record lists ``base`` (A) and ``cand`` (B).

    ``mode`` ``regress``: a cell FAILs when the candidate is more than ``tol_pct`` percent slower than the baseline median;
    ``within``: a cell FAILs when the candidate differs by more than ``tol_pct`` percent in either direction.

    ``baseline_unstable`` says what a cell whose spread exceeds 0.3 % means: ``invalid`` (default; an unstable baseline cell is not accepted as a gate
    baseline, which is the rule that establishes a baseline), ``worst`` (the rule of a campaign comparison: the cell is judged on its worst repeat, i.e.
    the baseline is referenced at its fastest repeat and an unstable candidate at its slowest, the most adverse pairing) or ``widen`` (the cell is
    judged on the medians as *noise-limited*: the tolerance is widened by the larger of the two spreads; such a cell can fail a gross regression but is
    never the evidence of a gain).

    ``base_rerun`` / ``cand_rerun``: the record lists of a second campaign over some cells only (more calls per trial); the cells that both lists
    contain are judged on them (``judged_on`` = ``rerun``), the others on the first campaign. A re-run is judged as it is and is never repeated.

    ``order``: the start order that the record files of the two builds must have (``ORDERS``; ``order_problems``): strict alternation (default) or ABBA blocks. The first campaign and
    the re-run are judged by the same order; an unknown order is INVALID. Whatever the order, a start stamp that is missing, does not parse or is shared by two records, and a record
    file that is listed on both sides, make the comparison INVALID (``require_interleaved=False``, a diagnostic that no checker uses, skips the order itself and keeps the check of the
    shared files).
    ``require_nodes``: every record of both builds and of the re-runs must carry the launch mapping of every trial of every cell (``validate_record``); a record of before the mappings
    were recorded is INVALID. Which mapping a cell must show is the business of ``route_problems``.

    Returns ``{"verdict": "PASS" | "FAIL" | "INVALID", "problems": [...], "rows": [...], "order": ...}``. INVALID wins over FAIL: a comparison on a
    wrong, dirty, unstable or incomplete baseline is not a measurement.
    """
    base_rerun, cand_rerun = list(base_rerun or []), list(cand_rerun or [])
    problems = []
    for rec in base + base_rerun:
        problems.extend(validate_record(rec, "baseline", expect_base_commit, True, require_nodes))
    for rec in cand + cand_rerun:
        problems.extend(validate_record(rec, "candidate", None, require_clean_cand, require_nodes))
    if mode not in ("regress", "within"):
        problems.append("mode must be regress or within, got %r" % mode)
    if baseline_unstable not in UNSTABLE_POLICIES:
        problems.append("baseline_unstable must be one of %s, got %r" % (", ".join(UNSTABLE_POLICIES), baseline_unstable))
    if order not in ORDERS:
        problems.append("order must be one of %s, got %r" % (", ".join(ORDERS), order))
    invalid = {"verdict": "INVALID", "problems": problems, "rows": [], "metric": metric, "tol_pct": tol_pct, "mode": mode, "unstable_policy": baseline_unstable, "order": order,
               "require_nodes": bool(require_nodes)}
    if not base or not cand:
        problems.append("both a baseline and a candidate record list are required")
        return invalid
    if bool(base_rerun) != bool(cand_rerun):
        problems.append("re-run records are required for both sides or for neither")
    everything = list(base) + list(cand) + base_rerun + cand_rerun
    if any(structure_problems(r) for r in everything):
        return invalid  # the validation messages above already say what is wrong; nothing can be compared
    if len(set(sample_stat_of(r) for r in everything)) > 1:
        problems.append("the records mix sample statistics (%s): pooled and per-process-median records cannot be compared" %
                        ", ".join(sorted(set(sample_stat_of(r) for r in everything))))
        return invalid
    if not known_metric(everything, metric):
        problems.append("metric %r is neither a derived metric (%s) nor a wall metric (%s) nor a kernel that appears in the records" %
                        (metric, ", ".join(DERIVED_METRICS), ", ".join(WALL_METRICS)))
        return invalid
    ident = set()
    for rec in everything:
        m = rec.get("meta") or {}
        ident.add((m.get("gpu"), m.get("sm_count"), m.get("host")))
    if len(ident) > 1:
        problems.append("records come from different GPU / SM count / host combinations: %s" % sorted(str(i) for i in ident))
    for name, recs in (("baseline", base), ("candidate", cand), ("baseline re-run", base_rerun), ("candidate re-run", cand_rerun)):
        if recs:
            problems.extend(side_config_problems(recs, name))
    problems.extend(ab_problems(base, cand, "first campaigns"))
    if base_rerun and cand_rerun:
        problems.extend(ab_problems(base_rerun, cand_rerun, "re-runs"))
        for name, first, rerun in (("baseline", base, base_rerun), ("candidate", cand, cand_rerun)):
            problems.extend(workload_problems(first, rerun, "the %s first campaign" % name, "the %s re-run" % name, SCALED_ARGS))
    for name, recs in (("baseline", base), ("candidate", cand), ("baseline re-run", base_rerun), ("candidate re-run", cand_rerun)):
        if recs and len(recs) < min_repeats:
            problems.append("%s has %d record file(s), at least %d interleaved repeats are required" % (name, len(recs), min_repeats))
    for label, b, c in (("first campaign", base, cand), ("re-run", base_rerun, cand_rerun)):    # the re-run is judged by the order of its first campaign
        if not b or not c:
            continue
        if require_interleaved and order in ORDERS:
            problems.extend(order_problems(b, c, order, label))
        else:
            problems.extend(shared_record_problems(b, c, label))

    # ---- the campaign: unique keys, the same cells in every file of a side, the same cells in both builds, a consistent re-run set -----------------------------
    sides, ragged = {}, set()
    for role, recs in (("baseline", base), ("candidate", cand), ("baseline re-run", base_rerun), ("candidate re-run", cand_rerun)):
        if recs:
            probs, union, partial = campaign_cell_problems(recs, role)
            problems.extend(probs)
            sides[role] = union
            ragged |= partial                                          # a cell that not every file of a side holds has no valid sample set: it gets no row

    def chosen(keys):
        return set(k for k in keys if not cells or any(sub in k for sub in cells))

    base_set, cand_set = chosen(sides["baseline"]), chosen(sides["candidate"])
    if base_set - cand_set:
        problems.append("cell(s) %s missing in the candidate records" % sorted(base_set - cand_set)[:4])
    if cand_set - base_set:
        problems.append("cell(s) %s missing in the baseline records" % sorted(cand_set - base_set)[:4])
    rerun_keys = set()
    if base_rerun and cand_rerun:
        rb, rc = chosen(sides["baseline re-run"]), chosen(sides["candidate re-run"])
        for key in sorted(rb ^ rc):
            problems.append("cell %r is in the re-run records of one side only" % key)
        for key in sorted((rb | rc) - base_set):
            problems.append("cell %r is in the re-run records but not in the first campaign" % key)
        rerun_keys = rb & rc & base_set
    first_order = [c["key"] for c in base[0]["cells"]]
    rows = []
    unstable_base = False
    failed = False
    for key in first_order:
        if key not in base_set or key not in cand_set or key in ragged:
            continue
        judged_on = "rerun" if key in rerun_keys else "first"
        base_k, cand_k = (base_rerun, cand_rerun) if judged_on == "rerun" else (base, cand)
        kind = cell_kind(base_k, key)
        cell_problems = []
        sets = [("baseline", base), ("candidate", cand)]
        if judged_on == "rerun":                           # the first campaign of a re-run cell is evidence as well (it decided that the re-run was needed)
            sets += [("baseline re-run", base_rerun), ("candidate re-run", cand_rerun)]
        for role, recs in sets:
            for rec in recs:
                cell_problems += ["%s record %s, cell %r (kind %s): %s" % (role, rec.get("_path", "?"), key, kind, p) for p in raw_sample_problems(rec, key, metric)]
            stamps = [(r.get("meta") or {}).get("timestamp") for r in recs]
            if None in stamps or len(set(stamps)) < min_repeats:
                cell_problems.append("%s records of cell %r are fewer than %d distinct processes (timestamps %s)" % (role, key, min_repeats, sorted(str(x) for x in set(stamps))))
        if cell_problems:
            problems.extend(cell_problems[:6] + (["... and %d more problem(s) in cell %r" % (len(cell_problems) - 6, key)] if len(cell_problems) > 6 else []))
            continue
        if judged_on == "rerun":
            for role, first, rerun in (("baseline", base, base_rerun), ("candidate", cand, cand_rerun)):
                problems.extend(rerun_problems(first, rerun, role, key, metric))
            for name in count_names(metric):
                fb, fc = _arg_values(base, key, name), _arg_values(cand, key, name)
                rb, rc = _arg_values(base_rerun, key, name), _arg_values(cand_rerun, key, name)
                if fb != fc or rb != rc:
                    problems.append("cell %r: the baseline and the candidate used different %s (first %s vs %s, re-run %s vs %s)" % (key, name, fb, fc, rb, rc))
        bv, cv = pooled_values(base_k, key, metric), pooled_values(cand_k, key, metric)
        med_b, med_c = statistics.median(bv), statistics.median(cv)
        if med_b == 0 or (med_c == 0 and (metric in DERIVED_METRICS or metric in WALL_METRICS)):
            problems.append("metric %r is zero in cell %r (kind %s): it is not defined for this kind of cell, name another metric" % (metric, key, kind))
            continue
        sp_b, sp_c = spread_of(bv), spread_of(cv)
        ref_b, value_c = med_b, med_c
        notes = []
        noise_limited = needs_rerun = False
        tol_eff = tol_pct
        if sp_b > UNSTABLE_SPREAD and baseline_unstable == "invalid":
            status = "UNSTABLE_BASELINE"
            notes.append("baseline spread %.2f %% exceeds %.1f %%: not accepted as a gate baseline" % (100 * sp_b, 100 * UNSTABLE_SPREAD))
            unstable_base = True
        elif baseline_unstable == "widen" and judged_on == "first" and max(sp_b, sp_c) > UNSTABLE_SPREAD:
            status = "NEEDS_RERUN"                          # the widened tolerance is only for a cell that stays unstable with three times the calls
            needs_rerun = True
            tol_eff = tol_pct + 100 * max(sp_b, sp_c)
            problems.append("cell %r is above %.1f %% spread in the first campaign (%.2f %% / %.2f %%) and has no valid re-run with %d times the calls per trial" %
                            (key, 100 * UNSTABLE_SPREAD, 100 * sp_b, 100 * sp_c, RERUN_SCALE))
            notes.append("needs a re-run with %d times the calls per trial" % RERUN_SCALE)
        else:
            if baseline_unstable == "widen":
                if max(sp_b, sp_c) > UNSTABLE_SPREAD:      # judged on the re-run and still above 0.3 %
                    noise_limited = True
                    tol_eff = tol_pct + 100 * max(sp_b, sp_c)
                    notes.append("noise-limited: spread %.2f %% (baseline) / %.2f %% (candidate), tolerance widened to %.2f %%" % (100 * sp_b, 100 * sp_c, tol_eff))
            else:
                if sp_b > UNSTABLE_SPREAD:
                    ref_b = min(bv)
                    notes.append("baseline spread %.2f %%: referenced at its fastest repeat" % (100 * sp_b))
                if sp_c > UNSTABLE_SPREAD:
                    value_c = max(cv)
                    notes.append("candidate spread %.2f %%: judged on its worst repeat" % (100 * sp_c))
            delta = 100.0 * (value_c / ref_b - 1.0)
            bad = delta > tol_eff if mode == "regress" else abs(delta) > tol_eff
            status = "FAIL" if bad else "PASS"
            failed = failed or bad
        if judged_on == "rerun":
            notes.append("judged on the re-run")
        rows.append({"key": key, "kind": kind, "metric": metric, "base_median": med_b, "base_reference": ref_b, "base_spread_pct": 100 * sp_b, "cand_median": med_c,
                     "cand_spread_pct": 100 * sp_c, "cand_judged": value_c, "delta_pct": 100.0 * (value_c / ref_b - 1.0),
                     "median_delta_pct": 100.0 * (med_c / med_b - 1.0), "status": status, "note": "; ".join(notes),
                     "n_base": len(bv), "n_cand": len(cv), "noise_limited": noise_limited, "needs_rerun": needs_rerun, "tol_pct_effective": tol_eff,
                     "judged_on": judged_on})
    if not rows:
        problems.append("no comparable cells")
    if problems or unstable_base:
        verdict = "INVALID"
    elif failed:
        verdict = "FAIL"
    else:
        verdict = "PASS"
    return {"verdict": verdict, "problems": problems, "rows": rows, "metric": metric, "tol_pct": tol_pct, "mode": mode, "unstable_policy": baseline_unstable, "order": order,
            "require_nodes": bool(require_nodes)}


def classify_rows(result, tol_pct=None):
    """Which cells are slower, faster or noise-limited, read from the rows of a comparison (the ``--json`` rows of ``compare_bench.py``).

    A cell is *slower* when its median delta is above the tolerance and *faster* when it is at or below minus the tolerance; the plain tolerance is
    used, not the widened one. A noise-limited cell (spread above 0.3 % after the re-run) is never *faster*: it cannot be the evidence of a gain. A cell that
    still needs its re-run (``needs_rerun``) is only listed as such: nothing is read from an unstable first campaign."""
    tol = result["tol_pct"] if tol_pct is None else tol_pct
    out = {"slower": [], "faster": [], "noise_limited": [], "neutral": [], "needs_rerun": []}
    for r in result["rows"]:
        d = r.get("median_delta_pct", r.get("delta_pct"))
        if d is None:
            continue
        if r.get("needs_rerun"):
            out["needs_rerun"].append(r["key"])
            continue
        if r.get("noise_limited"):
            out["noise_limited"].append(r["key"])
        if d > tol:
            out["slower"].append(r["key"])
        elif d <= -tol and not r.get("noise_limited"):
            out["faster"].append(r["key"])
        elif not r.get("noise_limited"):
            out["neutral"].append(r["key"])
    return out


def format_table(result):
    order = result.get("order", "alternate")                                   # the header names the start order only where it is not the strict alternation of every older result
    lines = ["metric %s, tolerance %.2f %% (%s mode%s%s)" % (result["metric"], result["tol_pct"], result["mode"],
                                                            ", widening of noise-limited cells" if result.get("unstable_policy") == "widen" else "",
                                                            "" if order == "alternate" else ", %s start order" % order)]
    lines.append("%-44s %12s %7s %12s %7s %8s  %s" % ("cell", "base median", "sprd %", "cand median", "sprd %", "delta %", "status"))
    for r in result["rows"]:
        delta = "%+.2f" % r["delta_pct"] if r["delta_pct"] is not None else "n/a"
        lines.append("%-44s %12.1f %7.2f %12.1f %7.2f %8s  %s%s" % (r["key"][:44], r["base_median"], r["base_spread_pct"], r["cand_median"],
                                                                 r["cand_spread_pct"], delta, r["status"], ("  (%s)" % r["note"]) if r["note"] else ""))
    for p in result["problems"]:
        lines.append("PROBLEM: %s" % p)
    lines.append("VERDICT: %s" % result["verdict"])
    return "\n".join(lines)
