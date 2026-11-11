#!/usr/bin/env python3
"""Per-test-file counts of a pytest run (JUnit XML) as a ``cfbi-suite-v1`` record (plain Python, no torch).

usage: python benchmarks/suite_record.py --junit FILE --suite gate|S|Sd --out FILE --commit HASH [--dirty 0|1] [--tree TEXT] [--probe FILE]
           [--gpu NAME --sm-count N] [--env NAME=VALUE ...] [--ignored PATH ...] [--label TEXT]
       python benchmarks/suite_record.py --check BASELINE.json CANDIDATE.json [--allow-failures FILE]

The record::

    {"schema": "cfbi-suite-v1",
     "meta":   {"suite": "gate|S|Sd", "commit": ..., "dirty": ..., "tree": ..., "gpu": ..., "sm_count": ..., "env": {...}, "ignored": [...], ...},
     "files":  {"tests/cfbi/test_vs_ref.py": {"passed": n, "failed": n, "skipped": n, "errors": n, "total": n}, ...},
     "totals": {"passed": n, ...}, "failed": [test ids], "skipped": [{"id": ..., "reason": ...}]}

``--check``: the candidate must have, for every file of the baseline, at least as many passed tests, and no failed or errored test other than the ids
listed in ``--allow-failures`` (one id per line: the failures that are by design on a device); files that only the candidate has are allowed (new tests).

exit status: 0 ok, 1 the check fails, 2 unusable input
"""
import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

SCHEMA = "cfbi-suite-v1"
COUNTS = ("passed", "failed", "skipped", "errors", "total")


def classname_to_file(classname):
    """``tests.cfbi.test_arch`` -> ``tests/cfbi/test_arch.py``; ``tests.cfbi.test_x.TestFoo`` (a test class) -> ``tests/cfbi/test_x.py``."""
    parts = classname.split(".")
    idx = None
    for i, p in enumerate(parts):
        if p.startswith("test_") or p.endswith("_test") or p == "conftest":
            idx = i
    if idx is None:
        idx = len(parts) - 1
    return "/".join(parts[:idx + 1]) + ".py"


def parse_junit(path):
    """``{"files": {file: counts}, "failed": [ids], "skipped": [{"id", "reason"}]}`` of a pytest ``--junitxml`` file."""
    root = ET.parse(path).getroot()
    files, failed, skipped = {}, [], []
    for case in root.iter("testcase"):
        f = classname_to_file(case.get("classname", ""))
        counts = files.setdefault(f, dict.fromkeys(COUNTS, 0))
        tid = "%s::%s" % (f, case.get("name", ""))
        counts["total"] += 1
        if case.find("error") is not None:
            counts["errors"] += 1
            failed.append(tid)
        elif case.find("failure") is not None:
            counts["failed"] += 1
            failed.append(tid)
        elif case.find("skipped") is not None:
            counts["skipped"] += 1
            sk = case.find("skipped")
            skipped.append({"id": tid, "reason": (sk.get("message") or "")[:200]})
        else:
            counts["passed"] += 1
    return {"files": files, "failed": failed, "skipped": skipped}


def parse_probe(text):
    """GPU name, SM count, python, torch and CUDA versions from the output of the node mode ``probe`` (the line ``<python> <torch> <cuda> <gpu name> <sms>``)."""
    for line in text.splitlines():
        m = re.match(r"^(\d+\.\d+\.\d+)\s+(\S+)\s+(\d+\.\d+)\s+(.+?)\s+(\d+)\s*$", line.strip())
        if m:
            return {"python": m.group(1), "torch": m.group(2), "cuda": m.group(3), "gpu": m.group(4), "sm_count": int(m.group(5))}
    return {}


def build_record(parsed, meta):
    totals = dict.fromkeys(COUNTS, 0)
    for c in parsed["files"].values():
        for k in COUNTS:
            totals[k] += c[k]
    return {"schema": SCHEMA, "meta": meta, "files": parsed["files"], "totals": totals, "failed": parsed["failed"], "skipped": parsed["skipped"]}


def structure_problems(rec):
    if not isinstance(rec, dict) or rec.get("schema") != SCHEMA:
        return ["not a %s record" % SCHEMA]
    problems = []
    if not isinstance(rec.get("meta"), dict):
        problems.append("no metadata")
    else:
        for field in ("suite", "commit", "dirty", "gpu", "sm_count"):
            if rec["meta"].get(field) is None:
                problems.append("metadata field %r is missing" % field)
    files = rec.get("files")
    if not isinstance(files, dict) or not files:
        problems.append("no files")
    else:
        for f, c in files.items():
            if not isinstance(c, dict) or any(not isinstance(c.get(k), int) for k in COUNTS):
                problems.append("file %s: malformed counts" % f)
            elif c["passed"] + c["failed"] + c["skipped"] + c["errors"] != c["total"]:
                problems.append("file %s: the counts do not add up" % f)
    return problems


def check(baseline, candidate, allowed_failures=()):
    """Problems of ``candidate`` against ``baseline`` (an empty list means the check passes)."""
    problems = ["baseline: %s" % p for p in structure_problems(baseline)] + ["candidate: %s" % p for p in structure_problems(candidate)]
    if problems:
        return problems
    if baseline["meta"]["suite"] != candidate["meta"]["suite"]:
        problems.append("the records are of different suites (%s, %s)" % (baseline["meta"]["suite"], candidate["meta"]["suite"]))
    for f, c in sorted(baseline["files"].items()):
        got = candidate["files"].get(f)
        if got is None:
            problems.append("file %s (%d passed in the baseline) is missing in the candidate" % (f, c["passed"]))
        elif got["passed"] < c["passed"]:
            problems.append("file %s: %d passed, the baseline has %d" % (f, got["passed"], c["passed"]))
    allowed = set(allowed_failures)
    for tid in candidate["failed"]:
        if tid not in allowed:
            problems.append("failed or errored test %s" % tid)
    return problems


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--junit")
    ap.add_argument("--suite", choices=("gate", "S", "Sd"))
    ap.add_argument("--out")
    ap.add_argument("--commit")
    ap.add_argument("--dirty", default="0")
    ap.add_argument("--tree", default=None)
    ap.add_argument("--probe", default=None)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--sm-count", type=int, default=None)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--ignored", nargs="*", default=[])
    ap.add_argument("--label", default="")
    ap.add_argument("--check", nargs=2, metavar=("BASELINE", "CANDIDATE"))
    ap.add_argument("--allow-failures", default=None)
    a = ap.parse_args(argv)
    try:
        if a.check:
            allowed = []
            if a.allow_failures:
                with open(a.allow_failures, encoding="utf-8") as f:
                    allowed = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
            problems = check(_load(a.check[0]), _load(a.check[1]), allowed)
            for p in problems:
                print("PROBLEM: %s" % p)
            print("VERDICT: %s" % ("PASS" if not problems else "FAIL"))
            return 0 if not problems else 1
        if not (a.junit and a.suite and a.out and a.commit):
            print("--junit, --suite, --out and --commit are required", file=sys.stderr)
            return 2
        meta = {"suite": a.suite, "commit": a.commit, "dirty": a.dirty not in ("0", "", "false", "False"), "tree": a.tree, "gpu": a.gpu, "sm_count": a.sm_count,
                "env": dict(e.split("=", 1) for e in a.env), "ignored": list(a.ignored), "label": a.label, "junit": os.path.basename(a.junit)}
        if a.probe:
            with open(a.probe, encoding="utf-8", errors="replace") as f:
                probe = parse_probe(f.read())
            for k in ("gpu", "sm_count"):
                if meta[k] is None:
                    meta[k] = probe.get(k)
            meta.update({k: probe.get(k) for k in ("python", "torch", "cuda")})
        rec = build_record(parse_junit(a.junit), meta)
        problems = structure_problems(rec)
        if problems:
            print("unusable: %s" % "; ".join(problems), file=sys.stderr)
            return 2
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1, sort_keys=True)
            f.write("\n")
        t = rec["totals"]
        print("%s: %d files, %d passed, %d failed, %d errors, %d skipped" % (a.suite, len(rec["files"]), t["passed"], t["failed"], t["errors"], t["skipped"]))
        return 0
    except (OSError, ValueError, ET.ParseError) as e:
        print("unusable input: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
