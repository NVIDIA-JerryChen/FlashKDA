#!/usr/bin/env python3
"""The decision and the ledger row of a row-engine step, derived from its raw evidence.

A row-engine step ends adopted or not adopted by fixed rules (the step gate protocol). This tool recomputes everything from the files that the evidence runs wrote and holds nothing that a
run only claims:

  evidence manifest (JSON, ``cfbi-step-evidence-v1``)
    step, commit (the commit of the step: the measured candidate and the commit of the dumps), previous (the previous accepted state), base, markers, tolerance_pct;
    devices[<b200|b300>]: baseline (a record with the device's ``gpu`` and ``sm_count``), previous and base (each the entry of one A/B campaign: allocation, report, records_first,
      records_rerun, and the commits of its sides);
    codegen: allow (the allow-list of the comparison) and, per toolchain (conda, container_b200, container_b300), the base, candidate and previous-state dumps (and the occupancy record of
      a container toolchain);
    suites[<device>][gate|S]: baseline (the ``cfbi-suite-v1`` record of the previous accepted state) and candidate (the step's), made by ``suite_record.py`` from the JUnit files of the runs.

  decision: two campaigns per device, both recomputed from their raw records through the checker's own admission (``record_checker.campaign_evidence_problems``: the Method M records of the
    benchmark's own matrix, the commits of the sides, the GPU, one host, the comparison again and the report row by row) and judged on the KERNEL SUM (``total_us``, the gating metric of the
    row-engine steps; the wall metrics are measured and reported, they do not gate). Against the previous accepted state a step is adopted only with no confirmed slower cell and at least one
    confirmed faster cell, on each of the devices (a faster cell is stable, judged on the re-run, faster by the tolerance or more on the re-run and in the first campaign: a re-run that finds
    a gain where the first campaign found none has reproduced nothing); against the base (cumulative: a no-regression check, because the cumulative comparison also holds the gain of the
    steps before this one) with no confirmed slower cell. A comparison that is missing, not a measurement (INVALID) or made under other parameters leaves the outcome undecided: that is a
    repair, never "not adopted".

  exactness: Gate G and Suite S of the step have, for every file of the previous accepted state's record, at least as many passed tests and no failed or errored test
    (``suite_record.check``), on both devices, and are complete records of their device.

  row: the ledger row (``record_checker.step_problems``) built from the dumps and the decision: the comparison entries in change mode with the previous accepted dump and the allow-list,
    the resource figures of the row engine, the A/B medians and the noise-limited cells of the comparison against the previous accepted state, and the timing results.

usage: step_record.py decide|row|check --evidence MANIFEST [--out FILE] [--repo DIR]
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_records as br  # noqa: E402
import codegen_gate as cg  # noqa: E402
import record_checker as rc  # noqa: E402
import suite_record as sr  # noqa: E402

SCHEMA = "cfbi-step-evidence-v1"
GATING = rc.DEVICES                                          # the devices that decide
GATE_METRIC = "total_us"                                     # the kernel sum: the gating metric of the row-engine steps
KINDS = ("previous", "base")                                 # the two campaigns of a device: against the previous accepted state and against the base
AGAINST = {"previous": "the previous accepted state", "base": "the base"}
TOL_PCT = rc.TOLERANCE_PCT


def figure(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else float("nan")


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---- the manifest ---------------------------------------------------------------------------------------------------------------------------------------------
def manifest_problems(m):
    if not isinstance(m, dict) or m.get("schema") != SCHEMA:
        return ["not a %s record" % SCHEMA]
    out = []
    for field in ("step", "commit", "previous", "base"):
        if not isinstance(m.get(field), str) or not m[field]:
            out.append("the manifest names no %s" % field)
    for field in ("commit", "previous", "base", "suite_commit"):
        if isinstance(m.get(field), str) and not rc.hexcommit(m[field]):
            out.append("%s %r is not a hexadecimal commit hash of at least 7 characters" % (field, m[field]))
    if m.get("suite_commit") is not None and not isinstance(m["suite_commit"], str):
        out.append("suite_commit must be a commit hash or left out")
    if m.get("step") not in rc.STEP_IDS:
        out.append("the step %r is none of %s" % (m.get("step"), ", ".join(rc.STEP_IDS)))
    if not (isinstance(m.get("markers"), list) and m["markers"] and all(isinstance(x, str) for x in m["markers"])):
        out.append("markers must be a non-empty list of identifiers")
    devices = m.get("devices")
    if not isinstance(devices, dict) or sorted(devices) != sorted(GATING):
        out.append("devices must be exactly %s" % list(GATING))
    else:
        for dev in GATING:
            d = devices[dev]
            for field in ("baseline",) + KINDS:
                if not isinstance(d, dict) or not isinstance(d.get(field), str):
                    out.append("devices.%s.%s must be the path of a record" % (dev, field))
    suites = m.get("suites")
    for dev in GATING:
        for name in ("gate", "S"):
            e = ((suites or {}).get(dev) or {}).get(name) if isinstance(suites, dict) else None
            if not (isinstance(e, dict) and isinstance(e.get("baseline"), str) and isinstance(e.get("candidate"), str)):
                out.append("suites.%s.%s must name the baseline and the candidate record of the suite" % (dev, name))
    cg_ = m.get("codegen")
    if not isinstance(cg_, dict) or not isinstance(cg_.get("allow"), str) or any(k not in cg_ for k in rc.COMPARISONS):
        out.append("codegen must name the allow-list and one comparison per toolchain (%s)" % ", ".join(rc.COMPARISONS))
    else:
        for key in rc.COMPARISONS:
            e = cg_[key]
            need = ("base", "cand", "prev") + (("occupancy",) if key.startswith("container_") else ())
            out.extend("codegen.%s.%s must be the path of a record" % (key, f) for f in need if not (isinstance(e, dict) and isinstance(e.get(f), str)))
    return out


# ---- exactness ----------------------------------------------------------------------------------------------------------------------------------------------
KERNEL_SOURCES = ("csrc", "flash_kda", "setup.py")           # what the kernels are built from: a step changes it, a repair of the tests after the step must not


def default_repo():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def source_problems(repo, step, later):
    """Why the suites cannot have run on ``later`` for the step ``step``: ``later`` is not a descendant of the step's commit, or its kernel sources (``KERNEL_SOURCES``) are not the step's. A
    suite that ran on a later commit proves the step only when that commit changed tests and tools and nothing that is compiled."""
    full = {}
    for what, rev in (("the step", step), ("the commit of the suites", later)):
        full[what] = rc.resolve(repo, rev)
        if full[what] is None:
            return ["the repository %s does not have %s (%s)" % (repo, what, rev[:12])]
    if not rc.is_ancestor(repo, full["the step"], full["the commit of the suites"]):
        return ["the commit of the suites (%s) is not a descendant of the step (%s)" % (later[:10], step[:10])]
    status, out, err = rc.git(repo, "diff", "--name-only", full["the step"], full["the commit of the suites"], "--", *KERNEL_SOURCES)
    if status != 0:
        return ["cannot compare the kernel sources of %s and %s: %s" % (step[:10], later[:10], err.strip() or "git failed")]
    changed = [p for p in out.split("\n") if p.strip()]
    return ["the suites ran on %s, whose kernel sources differ from the step's %s: %s" % (later[:10], step[:10], ", ".join(changed[:4]) + (" ..." if len(changed) > 4 else ""))] if changed else []


def exactness_problems(m, repo=None):
    """Why the exactness gate (Gate G and Suite S on both devices) is not shown by the suite records of the manifest: a record that cannot be read, that is of another commit than the one the
    suites ran on (the step's commit or, after a repair of the tests, a later commit with the same kernel sources: ``suite_commit``), of another device, or that fails the comparison with
    the record of the baseline (``suite_record.check``)."""
    out = []
    ran_on = m.get("suite_commit") or m["commit"]
    if not rc.commit_matches(ran_on, m["commit"]):
        out.extend(source_problems(repo or default_repo(), m["commit"], ran_on))
    for dev in GATING:
        for name in ("gate", "S"):
            spec = m["suites"][dev][name]
            try:
                base, cand = load_json(spec["baseline"]), load_json(spec["candidate"])
            except (OSError, ValueError) as e:
                out.append("%s, suite %s: a record cannot be read: %s" % (dev, name, e))
                continue
            where = "%s, suite %s" % (dev, name)
            if (cand.get("meta") or {}).get("suite") != name:
                out.append("%s: the candidate record is of the suite %r" % (where, (cand.get("meta") or {}).get("suite")))
            if not rc.commit_matches(str((cand.get("meta") or {}).get("commit")), ran_on):
                out.append("%s: the candidate record is of commit %s, the suites ran on %s" % (where, (cand.get("meta") or {}).get("commit"), ran_on[:10]))
            if rc.DEVICE_GPU[dev] not in str((cand.get("meta") or {}).get("gpu")):
                out.append("%s: the candidate record is of the GPU %r" % (where, (cand.get("meta") or {}).get("gpu")))
            out.extend("%s: %s" % (where, p) for p in sr.check(base, cand))
    return out


# ---- the campaigns --------------------------------------------------------------------------------------------------------------------------------------------
def campaign_infos(m):
    """``({device: {kind: (problems, info)}})``: every campaign of the manifest recomputed from its raw records (``record_checker.campaign_evidence_problems``, the benchmark's own decode
    matrix); ``info`` holds the recomputed rows by ``(metric, cell)`` and the first-campaign records by side."""
    out = {}
    for dev in GATING:
        out[dev] = {}
        baseline = load_json(m["devices"][dev]["baseline"])
        for kind in KINDS:
            entry = load_json(m["devices"][dev][kind])
            a_commit = m["previous"] if kind == "previous" else m["base"]
            where = "%s, %s, against %s" % (dev, m["step"], AGAINST[kind])
            problems, info = rc.campaign_evidence_problems(entry, where, "decode", dev, baseline, a_commit, a_commit, m["commit"])
            for field, want in (("a_commit", a_commit), ("b_commit", m["commit"])):
                if entry.get(field) is not None and not rc.commit_matches(entry.get(field), want):
                    problems.append("%s: the entry says %s = %s, the manifest %s" % (where, field, entry.get(field), want))
            if not problems:
                problems.extend("%s: %s" % (where, p) for p in rc.default_route_problems(info["first"], where))
            out[dev][kind] = (problems, info)
    return out


def first_deltas(info, metric, commit):
    """The median difference of every cell in the first campaign alone (the comparison without its re-run), by cell key."""
    if not info.get("first"):
        return {}
    res = br.compare(info["first"]["A"], info["first"]["B"], **rc.compare_kwargs(metric, TOL_PCT, commit, rc.metric_cells("decode", metric)))
    return dict((r["key"], r["median_delta_pct"]) for r in res["rows"])


def confirmed_gain(row, first, tol_pct):
    """A row of the kernel-sum comparison that is a confirmed gain: PASS, judged on the re-run, stable (not noise-limited, not in need of another), faster by the tolerance or more on the re-run
    and in the first campaign (the re-run is a confirmation: where the first campaign found no gain it reproduced nothing, and the cell is noise)."""
    return (row.get("status") == "PASS" and row.get("judged_on") == "rerun" and not row.get("noise_limited") and not row.get("needs_rerun") and is_number(row.get("median_delta_pct"))
            and figure(row["median_delta_pct"]) <= -tol_pct and is_number(first) and figure(first) <= -tol_pct)


def decide(m, infos, tol_pct=TOL_PCT):
    """``{"outcome": adopted | not_adopted | undecided, "reasons": [...], "per_device": {...}, "timing": {"vs_previous": ..., "vs_base": ...}}`` from the campaigns of ``campaign_infos``."""
    invalid, regress, nogain, per_device = [], [], [], {}
    for dev in GATING:
        info_dev = {"previous": {"slower": 0, "faster": 0, "cells": 0}, "base": {"slower": 0, "cells": 0}, "reported": {}}
        per_device[dev] = info_dev
        for kind in KINDS:
            problems, info = infos[dev][kind]
            if problems:
                invalid.extend(problems[:3] + (["... %d more problem(s) of %s %s" % (len(problems) - 3, dev, kind)] if len(problems) > 3 else []))
                continue
            rows = dict((key, row) for (metric, key), row in info["rows"].items() if metric == GATE_METRIC)
            if not rows:
                invalid.append("%s, against %s: no cell of the kernel sum was judged" % (dev, AGAINST[kind]))
                continue
            firsts = first_deltas(info, GATE_METRIC, m["previous"] if kind == "previous" else m["base"])
            info_dev[kind]["cells"] = len(rows)
            for key in sorted(rows):
                row = rows[key]
                if row["status"] == "FAIL":
                    info_dev[kind]["slower"] += 1
                    regress.append("%s, against %s, %s: B is %.2f %% slower than A in the kernel sum, the tolerance of the cell is %.2f %%" % (
                        dev, AGAINST[kind], key, figure(row["median_delta_pct"]), figure(row.get("tol_pct_effective", tol_pct))))
                elif kind == "previous" and confirmed_gain(row, firsts.get(key), tol_pct):
                    info_dev["previous"]["faster"] += 1
            wall = {}
            for metric in br.WALL_METRICS:                                          # reported, never gating: the confirmed slower and faster cells of the wall metrics
                wrows = [row for (mm, key), row in info["rows"].items() if mm == metric]
                wall[metric] = {"slower": sum(1 for r in wrows if r["status"] == "FAIL"), "cells": len(wrows)}
            info_dev["reported"][kind] = wall
        if infos[dev]["previous"][0] == [] and info_dev["previous"]["cells"] and not info_dev["previous"]["faster"]:
            nogain.append("%s: no confirmed gain against the previous accepted state: no stable cell of the kernel sum, judged on the re-run, is faster by %.2f %% or more on the re-run and in its "
                          "first campaign" % (dev, tol_pct))
    if invalid:
        outcome, reasons = "undecided", invalid
    elif regress or nogain:
        outcome, reasons = "not_adopted", regress + nogain
    else:
        outcome = "adopted"
        reasons = ["no confirmed slower cell of the kernel sum against the previous accepted state or the base on %s" % ", ".join(GATING)] + [
            "%s: %d confirmed faster cell(s) of the kernel sum against the previous accepted state" % (dev, per_device[dev]["previous"]["faster"]) for dev in GATING]
    timing = {"vs_previous": "pass" if not any(per_device[d]["previous"]["slower"] for d in GATING) and all(per_device[d]["previous"]["faster"] for d in GATING) else "fail",
              "vs_base": "pass" if not any(per_device[d]["base"]["slower"] for d in GATING) else "fail", "chain": "not_applicable"}
    return {"outcome": outcome, "reasons": reasons, "per_device": per_device, "timing": timing}


# ---- the row ----------------------------------------------------------------------------------------------------------------------------------------------------
def ab_figures(infos):
    """``{device: {"medians": {"<metric>|<cell>": {"base", "cand"}}, "noise_limited": [...]}}`` of the comparisons against the previous accepted state."""
    out = {}
    for dev in GATING:
        info = infos[dev]["previous"][1]
        medians, noisy = {}, []
        for (metric, key), row in sorted(info["rows"].items()):
            medians["%s|%s" % (metric, key)] = {"base": figure(row["base_median"]), "cand": figure(row["cand_median"])}
            if row.get("noise_limited"):
                noisy.append("%s|%s" % (metric, key))
        out[dev] = {"medians": medians, "noise_limited": noisy}
    return out


def codegen_entries(m):
    """``(codegen_gate, resource_gate)`` of the row from the dumps of the manifest: one comparison in change mode per toolchain (the previous accepted dump, the allow-list, the occupancy record of a
    device), the verdict as ``codegen_gate.compare_dumps`` gives it (and, on a device, the occupancy verdict with it), and the resource figures of the row engine."""
    codegen, resources = {}, {}
    cgm = m["codegen"]
    for key in rc.COMPARISONS:
        spec = cgm[key]
        base, cand = cg.load_dump(spec["base"]), cg.load_dump(spec["cand"])
        report = cg.compare_dumps(base, cand, cg.load_allow(cgm["allow"]), "change", cg.load_dump(spec["prev"]))
        owned = dict((arch, sorted(k for k in cand["kernels"][arch] if cg.family(k) in cg.DEFAULT_OWNED)) for arch in rc.COMPARISON_ARCHS[key])
        entry = {"base_dump": spec["base"], "cand_dump": spec["cand"], "prev_dump": spec["prev"], "allow": cgm["allow"], "mode": "change", "compared": report["compared"],
                 "differences": len(report["differences"]), "instantiations": next(iter(owned.values()))}
        passed = report["exit"] == 0
        if key.startswith("container_"):
            entry["occupancy"] = spec["occupancy"]
            import decision_record as dr
            failures = dr.occupancy_gate_failures(load_json(spec["occupancy"]), key.split("_", 1)[1])
            passed = passed and not failures
        entry["verdict"] = "PASS" if passed else "FAIL"
        codegen[key] = entry
        resources[key] = dict((arch, dict((i, dict((f, cand["kernels"][arch][i][f]) for f in rc.RESOURCE_FIGURES)) for i in owned[arch])) for arch in owned)
    return codegen, resources


def build_row(m, decision, infos, exactness=None):
    """The ledger row of the step for an ``adopted`` or ``not_adopted`` decision that reached the timing phase (a row that failed earlier has no timing and is written by hand from the phase that
    failed, with ``not_run`` figures: this tool is for the campaigns). ``exactness``: the problems of ``exactness_problems`` when the caller has them already."""
    codegen, resources = codegen_entries(m)
    exact = exactness_problems(m) if exactness is None else exactness
    reached = {"exactness": not exact, "codegen": all(e["verdict"] == "PASS" for e in codegen.values())}
    adopted = decision["outcome"] == "adopted"
    row = {"id": m["step"], "outcome": decision["outcome"], "commit": m["commit"], "revert_commit": m.get("revert_commit"), "markers": list(m["markers"]),
           "failed_at": None if adopted else "timing", "reason": "" if adopted else "; ".join(decision["reasons"][:3]), "gates": reached, "timing": decision["timing"],
           "codegen_gate": codegen, "resource_gate": resources, "ab": ab_figures(infos)}
    return row


def row_problems(m, row, repo=None):
    """The problems that the record checker finds in the row: its own checks (the dumps, the occupancy records, the timing results of an adopted row), and the marker check against the tree."""
    meta = {"base": m["base"]}
    return rc.step_problems(row, None, meta, repo)


# ---- the command line ------------------------------------------------------------------------------------------------------------------------------------------
def _report(problems):
    for p in problems:
        print("PROBLEM: %s" % p)
    print("VERDICT: %s" % ("PASS" if not problems else "FAIL"))
    return 0 if not problems else 1


def _print_decision(decision):
    print("OUTCOME: %s" % decision["outcome"])
    for r in decision["reasons"]:
        print("  - %s" % r)
    for dev, info in decision["per_device"].items():
        print("  %s: against the previous accepted state: %d cell(s), %d slower, %d confirmed faster (kernel sum); against the base: %d cell(s), %d slower; reported wall metrics: %s" % (
            dev, info["previous"]["cells"], info["previous"]["slower"], info["previous"]["faster"], info["base"]["cells"], info["base"]["slower"], json.dumps(info["reported"], sort_keys=True)))
    print("  timing: %s" % json.dumps(decision["timing"], sort_keys=True))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("decide", "row", "check"))
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--repo", default=None)
    a = ap.parse_args(argv)
    try:
        m = load_json(a.evidence)
        bad = manifest_problems(m)
        if bad:
            return _report(bad)
        infos = campaign_infos(m)
        decision = decide(m, infos)
        _print_decision(decision)
        if a.cmd == "decide":
            return 0 if decision["outcome"] != "undecided" else 2
        if decision["outcome"] == "undecided":
            return 2
        exact = exactness_problems(m, a.repo)
        row = build_row(m, decision, infos, exact)
        if a.cmd == "row":
            with open(a.out or "-", "w") if a.out else sys.stdout as f:
                json.dump(row, f, indent=1, sort_keys=True)
                f.write("\n")
        return _report(["exactness: %s" % p for p in exact] + row_problems(m, row, a.repo))
    except (OSError, ValueError, KeyError, cg.Unusable) as e:
        print("unusable input: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
