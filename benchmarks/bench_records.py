"""Benchmark records and the interleaved A/B comparison logic of the CF-BI benchmarks (plain Python, no torch).

``bench_breakdown.py --json`` writes one record file per process run (one build, one GPU, several cells). ``compare_bench.py`` compares the record
files of two builds that were measured interleaved in the same session on the same node: a baseline build (A) and a candidate build (B).

A record file::

    {"schema": "cfbi-bench-v1",
     "meta":  {"commit": ..., "dirty": ..., "tree": "snapshot|worktree", "gpu": ..., "sm_count": ..., "host": ..., "timestamp": ..., ...},
     "cells": [{"kind": "prefill|varlen|train|decode", "key": "prefill B=8 L=8192 H=6 pos0=0", "args": {...},
                "samples": [{"kernels": {"k1_kernel": us, ...}, "total_us": ..., "k1_us": ..., "k2_stage_us": ..., "k2_main_us": ...}, ...]}]}

Times are microseconds per call (CUDA profiler kernel durations, no launch overhead). A cell's ``samples`` are the in-process repeats; the
comparison pools the samples of all record files of a build. The spread of a cell is ``(max - min) / median`` over the pooled samples.

Rules enforced by ``compare``: a baseline record must carry the expected commit, a clean tree and the GPU / SM metadata; every record of a
comparison comes from the same GPU model, SM count and host; each build has at least ``min_repeats`` record files that are interleaved in time;
a baseline cell whose spread exceeds 0.3 % is not accepted as a gate baseline; a candidate cell above that spread is judged on its worst repeat.
"""
import json
import os
import statistics
import subprocess

SCHEMA = "cfbi-bench-v1"
UNSTABLE_SPREAD = 0.003  # 0.3 %: repeat-to-repeat spread above which a cell is unstable
REQUIRED_META = ("commit", "dirty", "gpu", "sm_count")
DERIVED_METRICS = ("total_us", "k1_us", "k2_stage_us", "k2_main_us")
SUMMARY_METRICS = ("k1_us", "k2_stage_us", "total_us")  # default metrics of a per-build summary


# ---- kernel names and stages ---------------------------------------------------------------------------------------------------------------

def kernel_base(name):
    """``void cfbi::k2_kernel<16>(cfbi::Params)`` -> ``k2_kernel`` (``ref::`` of the frozen reference kernels is kept)."""
    name = name.split("(")[0].split("<")[0].strip()
    if name.startswith("void "):
        name = name[5:]
    return name.replace("cfbi::", "", 1)


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


def make_cell(kind, key, args, runs):
    """A cell from ``runs``: a list (one entry per in-process repeat) of ``{raw kernel name: us per call}`` dicts."""
    samples = []
    for run in runs:
        folded = {}
        for name, us in run.items():
            base = kernel_base(name)
            folded[base] = folded.get(base, 0.0) + float(us)
        samples.append(derive(folded))
    return {"kind": kind, "key": key, "args": args, "samples": samples}


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


def structure_problems(rec):
    """Why ``rec`` cannot be used at all (wrong schema, missing or malformed parts); an empty list means it can be compared."""
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
            if not isinstance(s, dict) or not isinstance(s.get("kernels"), dict) or not all(isinstance(s.get(m), (int, float)) for m in DERIVED_METRICS):
                problems.append("%s: cell %r has a malformed sample" % (where, cell["key"]))
                break
    return problems


def known_metric(records, metric):
    """True when ``metric`` is a derived metric or the name of a kernel that appears in at least one sample (records must be well formed)."""
    if metric in DERIVED_METRICS:
        return True
    return any(metric in s["kernels"] for rec in records for cell in rec["cells"] for s in cell["samples"])


def validate_record(rec, role="baseline", expect_commit=None, require_clean=True):
    """Problems of one record (an empty list means valid). ``role`` is only used in the messages."""
    structural = structure_problems(rec)
    if structural:
        return ["%s: %s" % (role, p) for p in structural]
    where = "%s record %s" % (role, rec.get("_path", "?"))
    problems = []
    meta = rec["meta"]
    for field in REQUIRED_META:
        if meta.get(field) is None:
            problems.append("%s: metadata field %r is missing" % (where, field))
    commit = meta.get("commit")
    if expect_commit and commit is not None and not str(commit).startswith(expect_commit) and not str(expect_commit).startswith(str(commit)):
        problems.append("%s: commit %s is not the expected %s" % (where, commit, expect_commit))
    if require_clean and meta.get("dirty") is True:
        problems.append("%s: the measured tree was dirty" % where)
    return problems


# ---- statistics ------------------------------------------------------------------------------------------------------------------------------

def metric_value(sample, metric):
    if metric in DERIVED_METRICS:
        return sample[metric]
    return sample["kernels"].get(metric, 0.0)


def pooled_values(records, key, metric):
    vals = []
    for rec in records:
        for cell in rec["cells"]:
            if cell["key"] == key:
                vals.extend(metric_value(s, metric) for s in cell["samples"])
    return vals


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


def _interleave_switches(base, cand):
    seq = sorted([(r["meta"].get("timestamp", ""), "A") for r in base] + [(r["meta"].get("timestamp", ""), "B") for r in cand])
    return sum(1 for i in range(1, len(seq)) if seq[i][1] != seq[i - 1][1])


# ---- the comparison --------------------------------------------------------------------------------------------------------------------------

def compare(base, cand, metric="k2_stage_us", tol_pct=1.0, mode="regress", expect_base_commit=None, require_clean_cand=False,
            min_repeats=3, require_interleaved=True, cells=None, baseline_unstable="invalid"):
    """Compare the record lists ``base`` (A) and ``cand`` (B).

    ``mode`` ``regress``: a cell FAILs when the candidate is more than ``tol_pct`` percent slower than the baseline median;
    ``within``: a cell FAILs when the candidate differs by more than ``tol_pct`` percent in either direction.

    ``baseline_unstable`` says what an unstable baseline cell (spread above 0.3 %) means: ``invalid`` (default; the cell is not accepted as a
    gate baseline, which is the rule that establishes a baseline) or ``worst`` (the rule of a campaign comparison: the cell is judged on its
    worst repeat, i.e. the baseline is referenced at its fastest repeat and an unstable candidate at its slowest, the most adverse pairing).

    Returns ``{"verdict": "PASS" | "FAIL" | "INVALID", "problems": [...], "rows": [...]}``. INVALID wins over FAIL: a comparison on a
    wrong, dirty, unstable or incomplete baseline is not a measurement.
    """
    problems = []
    for rec in base:
        problems.extend(validate_record(rec, "baseline", expect_base_commit, True))
    for rec in cand:
        problems.extend(validate_record(rec, "candidate", None, require_clean_cand))
    if mode not in ("regress", "within"):
        problems.append("mode must be regress or within, got %r" % mode)
    invalid = {"verdict": "INVALID", "problems": problems, "rows": [], "metric": metric, "tol_pct": tol_pct, "mode": mode}
    if not base or not cand:
        problems.append("both a baseline and a candidate record list are required")
        return invalid
    if any(structure_problems(r) for r in list(base) + list(cand)):
        return invalid  # the validation messages above already say what is wrong; nothing can be compared
    if not known_metric(list(base) + list(cand), metric):
        problems.append("metric %r is neither a derived metric (%s) nor a kernel that appears in the records" % (metric, ", ".join(DERIVED_METRICS)))
        return invalid
    ident = set()
    for rec in list(base) + list(cand):
        m = rec.get("meta") or {}
        ident.add((m.get("gpu"), m.get("sm_count"), m.get("host")))
    if len(ident) > 1:
        problems.append("records come from different GPU / SM count / host combinations: %s" % sorted(str(i) for i in ident))
    for name, recs in (("baseline", base), ("candidate", cand)):
        if len(recs) < min_repeats:
            problems.append("%s has %d record file(s), at least %d interleaved repeats are required" % (name, len(recs), min_repeats))
    if require_interleaved:
        need = 2 * min(len(base), len(cand)) - 1
        got = _interleave_switches(base, cand)
        if got < need:
            problems.append("baseline and candidate records are not interleaved in time (%d variant switches, %d needed)" % (got, need))

    base_keys = [c["key"] for c in base[0]["cells"]]
    cand_keys = set(c["key"] for rec in cand for c in rec["cells"])
    rows = []
    unstable_base = False
    failed = False
    for key in base_keys:
        if cells and not any(s in key for s in cells):
            continue
        if key not in cand_keys:
            problems.append("cell %r is missing in the candidate records" % key)
            continue
        bv, cv = pooled_values(base, key, metric), pooled_values(cand, key, metric)
        med_b, med_c = statistics.median(bv), statistics.median(cv)
        sp_b, sp_c = spread_of(bv), spread_of(cv)
        ref_b, value_c = med_b, med_c
        notes = []
        if sp_b > UNSTABLE_SPREAD and baseline_unstable != "worst":
            status = "UNSTABLE_BASELINE"
            notes.append("baseline spread %.2f %% exceeds %.1f %%: not accepted as a gate baseline" % (100 * sp_b, 100 * UNSTABLE_SPREAD))
            unstable_base = True
        else:
            if sp_b > UNSTABLE_SPREAD:
                ref_b = min(bv)
                notes.append("baseline spread %.2f %%: referenced at its fastest repeat" % (100 * sp_b))
            if sp_c > UNSTABLE_SPREAD:
                value_c = max(cv)
                notes.append("candidate spread %.2f %%: judged on its worst repeat" % (100 * sp_c))
            delta = 100.0 * (value_c / ref_b - 1.0) if ref_b else float("inf")
            bad = delta > tol_pct if mode == "regress" else abs(delta) > tol_pct
            status = "FAIL" if bad else "PASS"
            failed = failed or bad
        rows.append({"key": key, "metric": metric, "base_median": med_b, "base_reference": ref_b, "base_spread_pct": 100 * sp_b, "cand_median": med_c,
                     "cand_spread_pct": 100 * sp_c, "cand_judged": value_c,
                     "delta_pct": (100.0 * (value_c / ref_b - 1.0)) if ref_b else None, "status": status, "note": "; ".join(notes),
                     "n_base": len(bv), "n_cand": len(cv)})
    if not rows:
        problems.append("no comparable cells")
    if problems or unstable_base:
        verdict = "INVALID"
    elif failed:
        verdict = "FAIL"
    else:
        verdict = "PASS"
    return {"verdict": verdict, "problems": problems, "rows": rows, "metric": metric, "tol_pct": tol_pct, "mode": mode}


def format_table(result):
    lines = ["metric %s, tolerance %.2f %% (%s mode)" % (result["metric"], result["tol_pct"], result["mode"])]
    lines.append("%-44s %12s %7s %12s %7s %8s  %s" % ("cell", "base median", "sprd %", "cand median", "sprd %", "delta %", "status"))
    for r in result["rows"]:
        delta = "%+.2f" % r["delta_pct"] if r["delta_pct"] is not None else "n/a"
        lines.append("%-44s %12.1f %7.2f %12.1f %7.2f %8s  %s%s" % (r["key"][:44], r["base_median"], r["base_spread_pct"], r["cand_median"],
                                                                 r["cand_spread_pct"], delta, r["status"], ("  (%s)" % r["note"]) if r["note"] else ""))
    for p in result["problems"]:
        lines.append("PROBLEM: %s" % p)
    lines.append("VERDICT: %s" % result["verdict"])
    return "\n".join(lines)
