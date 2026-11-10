#!/usr/bin/env python3
"""Compare the benchmark records of a baseline build (A) and a candidate build (B) that were measured interleaved on one node.

usage: python benchmarks/compare_bench.py --summary R1.json R2.json R3.json [--metric ...] [--expect-base-commit HASH]
       python benchmarks/compare_bench.py --base A1.json A2.json A3.json --cand B1.json B2.json B3.json
           [--metric total_us|k1_us|k2_stage_us|k2_main_us|<kernel name> ...] [--tol PERCENT] [--mode regress|within]
           [--expect-base-commit HASH] [--require-clean-cand] [--min-repeats N] [--no-interleave-check] [--unstable-baseline invalid|worst]
           [--cells SUBSTR ...] [--json OUT]

``--mode regress`` (default): FAIL when a cell of the candidate is more than ``--tol`` percent slower than the baseline median (no-regression gate).
``--mode within``: FAIL when it differs by more than ``--tol`` percent in either direction (protection of an unchanged code path).
The records come from ``bench_breakdown.py --json``; see ``bench_records.py`` for the rules (expected commit, clean tree, GPU / SM metadata, same
hardware, interleaving, 0.3 % spread).

exit status: 0 PASS, 1 FAIL, 2 INVALID (wrong, dirty or unstable baseline, missing data, not interleaved, mixed hardware, unusable input)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_records as br  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", nargs="+", default=None, metavar="FILE",
                    help="no comparison: per-cell median and spread of the records of ONE build; exit 1 when a cell is unstable (spread > 0.3 %%)")
    ap.add_argument("--base", nargs="+", default=None, help="record files of the baseline build (A)")
    ap.add_argument("--cand", nargs="+", default=None, help="record files of the candidate build (B)")
    ap.add_argument("--metric", action="append",
                    help="metric to compare or summarise (repeatable); default k2_stage_us for a comparison, k1_us, k2_stage_us and total_us for --summary")
    ap.add_argument("--tol", type=float, default=1.0, help="tolerance in percent (default 1.0)")
    ap.add_argument("--mode", choices=("regress", "within"), default="regress")
    ap.add_argument("--expect-base-commit", default=None, help="commit (prefix) every baseline record must carry")
    ap.add_argument("--require-clean-cand", action="store_true", help="refuse candidate records of a dirty tree")
    ap.add_argument("--min-repeats", type=int, default=3, help="minimum number of record files per build (default 3)")
    ap.add_argument("--no-interleave-check", action="store_true", help="do not require the records of A and B to alternate in time")
    ap.add_argument("--unstable-baseline", choices=("invalid", "worst"), default="invalid",
                    help="unstable baseline cell (spread > 0.3 %%): invalid = not accepted as a gate baseline (default); "
                         "worst = judged on its worst repeat (baseline at its fastest repeat, unstable candidate at its slowest)")
    ap.add_argument("--cells", nargs="*", default=None, help="only cells whose key contains one of these substrings")
    ap.add_argument("--json", default=None, help="write the results to this file")
    a = ap.parse_args(argv)

    if a.summary:
        try:
            recs = [br.load_record(p) for p in a.summary]
        except (OSError, ValueError) as e:
            print("cannot read the records: %s" % e, file=sys.stderr)
            return 2
        problems = []
        for r in recs:
            problems.extend(br.validate_record(r, "summary", a.expect_base_commit, True))
        metrics = a.metric or br.SUMMARY_METRICS
        rows = []
        if not any(br.structure_problems(r) for r in recs):
            problems.extend("metric %r is neither a derived metric nor a kernel that appears in the records" % m
                            for m in metrics if not br.known_metric(recs, m))
            rows = br.summarize(recs, [m for m in metrics if br.known_metric(recs, m)])
        print(br.format_summary(rows))
        for p in problems:
            print("PROBLEM: %s" % p)
        unstable = [r for r in rows if not r["stable"]]
        print("%d record file(s), %d cell metrics, %d unstable%s" % (len(recs), len(rows), len(unstable), "; INVALID" if problems else ""))
        if problems:
            return 2
        return 1 if unstable else 0
    if not a.base or not a.cand:
        print("--base and --cand (or --summary) are required", file=sys.stderr)
        return 2

    try:
        base = [br.load_record(p) for p in a.base]
        cand = [br.load_record(p) for p in a.cand]
    except (OSError, ValueError) as e:
        print("cannot read the records: %s" % e, file=sys.stderr)
        return 2

    results = []
    for metric in (a.metric or ["k2_stage_us"]):
        try:
            res = br.compare(base, cand, metric=metric, tol_pct=a.tol, mode=a.mode, expect_base_commit=a.expect_base_commit,
                             require_clean_cand=a.require_clean_cand, min_repeats=a.min_repeats,
                             require_interleaved=not a.no_interleave_check, cells=a.cells, baseline_unstable=a.unstable_baseline)
        except (KeyError, TypeError, ValueError, IndexError) as e:  # an unforeseen record shape is unusable input, never a performance FAIL
            print("cannot compare the records: %s: %s" % (type(e).__name__, e), file=sys.stderr)
            return 2
        print(br.format_table(res))
        print()
        results.append(res)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1, sort_keys=True)
            f.write("\n")
    verdicts = [r["verdict"] for r in results]
    if "INVALID" in verdicts:
        return 2
    if "FAIL" in verdicts:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
