#!/usr/bin/env python3
"""Per-test-file counts of a pytest run (JUnit XML) as a ``cfbi-suite-v1`` record (plain Python, no torch).

usage: python benchmarks/suite_record.py --junit FILE --suite gate|S|Sd --out FILE --commit HASH [--dirty 0|1] [--tree TEXT] [--probe FILE]
           [--gpu NAME --sm-count N] [--env NAME=VALUE ...] [--ignored PATH ...] [--label TEXT] [--subset]
       python benchmarks/suite_record.py --merge PART.json PART.json ... --out FILE [--label TEXT] [--expect-files-from TREE] [--expect-tests FILE]
       python benchmarks/suite_record.py --complete RECORD.json [--expect-files-from TREE] [--expect-tests FILE]
       python benchmarks/suite_record.py --check BASELINE.json CANDIDATE.json [--allow-failures FILE]

The record::

    {"schema": "cfbi-suite-v1",
     "meta":   {"suite": "gate|S|Sd", "commit": ..., "dirty": ..., "tree": ..., "gpu": ..., "sm_count": ..., "env": {...}, "ignored": [...], ...},
     "files":  {"tests/cfbi/test_vs_ref.py": {"passed": n, "failed": n, "skipped": n, "errors": n, "total": n}, ...},
     "tests":  {"tests/cfbi/test_vs_ref.py::test_x[1-2]": "passed|failed|errors|skipped", ...},
     "totals": {"passed": n, ...}, "failed": [test ids], "skipped": [{"id": ..., "reason": ...}]}

``--merge`` joins records of the same suite that were run separately (the part of a suite that cannot run in one session, per-case runs of a long test) into one
record that says it is a merge (``kind``, ``parts`` with the hash of every part). A merge is only valid when every part has per-test outcomes, a clean tree and the same
commit, tree, GPU, SM count, environment and toolchain versions, no test is in two parts, every file that a part ignored is covered by another part and, when given,
the union is exactly the expected files (the ``tests/cfbi/test_*.py`` of a source tree) and the expected test ids (a pytest collection of that tree). A record that
leaves files out of its suite says so with ``--subset``; ``--complete`` tells whether a record certifies the whole suite.

``--check``: the candidate must have, for every file of the baseline, at least as many passed tests, and no failed or errored test other than the ids
listed in ``--allow-failures`` (one id per line: the failures that are by design on a device); files that only the candidate has are allowed (new tests).
A Suite Sd candidate (Suite S run with ``FLASH_KDA_CFBI_IMPL=64``) has no baseline of its own and is judged against the Suite S baseline, provided its environment is the
baseline's plus exactly that variable.

exit status: 0 ok, 1 the check fails, 2 unusable input
"""
import argparse
import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

SCHEMA = "cfbi-suite-v1"
COUNTS = ("passed", "failed", "skipped", "errors", "total")
OUTCOMES = ("passed", "failed", "errors", "skipped")
MERGE_SAME = ("suite", "commit", "dirty", "tree", "gpu", "sm_count", "env", "python", "torch", "cuda")   # what the parts of a merge must agree on
SUITES = ("gate", "S", "Sd")
SD_ENV = ("FLASH_KDA_CFBI_IMPL", "64")             # what Suite Sd adds to the environment of Suite S: calls that pass no _impl are then the request for the two-launch decode
KINDS = ("single", "merge")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{7,40}$")   # a commit hash or an abbreviation of at least seven lowercase hexadecimal characters
TOOLCHAIN = ("python", "torch", "cuda")            # versions that a baseline and the run compared with it must share


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
    """``{"files": {file: counts}, "tests": {id: outcome}, "failed": [ids], "skipped": [{"id", "reason"}], "duplicates": [ids], "session": {...}}`` of a pytest
    ``--junitxml`` file. ``duplicates`` are test ids that occur more than once (a record built from such a file would count a test twice)."""
    root = ET.parse(path).getroot()
    files, failed, skipped, tests, duplicates = {}, [], [], {}, []
    for case in root.iter("testcase"):
        f = classname_to_file(case.get("classname", ""))
        counts = files.setdefault(f, dict.fromkeys(COUNTS, 0))
        tid = "%s::%s" % (f, case.get("name", ""))
        skip = case.find("skipped")
        if case.find("error") is not None:
            outcome = "errors"
            failed.append(tid)
        elif case.find("failure") is not None:
            outcome = "failed"
            failed.append(tid)
        elif skip is not None:
            outcome = "skipped"
            skipped.append({"id": tid, "reason": (skip.get("message") or "")[:200]})
        else:
            outcome = "passed"
        counts[outcome] += 1
        counts["total"] += 1
        if tid in tests:
            duplicates.append(tid)
        tests[tid] = outcome
    suite = root if root.tag == "testsuite" else next(root.iter("testsuite"), None)
    session = {k: suite.get(k) for k in ("hostname", "timestamp", "time")} if suite is not None else {}
    return {"files": files, "tests": tests, "failed": failed, "skipped": skipped, "duplicates": duplicates, "session": session}


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
    return {"schema": SCHEMA, "meta": meta, "files": parsed["files"], "tests": parsed["tests"], "totals": totals, "failed": parsed["failed"], "skipped": parsed["skipped"]}


def _count_ok(x):
    return type(x) is int and x >= 0          # a boolean is not a count


def _str_list(x):
    return isinstance(x, list) and all(isinstance(e, str) for e in x)


def metadata_problems(meta):
    """Types and values of the metadata of a suite record (a field that is left out or None is only a problem for the required ones)."""
    problems = []
    for field in ("suite", "commit", "dirty", "gpu", "sm_count"):
        if meta.get(field) is None:
            problems.append("metadata field %r is missing" % field)
    suite, commit, dirty, gpu, sms = (meta.get(f) for f in ("suite", "commit", "dirty", "gpu", "sm_count"))
    if suite is not None and suite not in SUITES:
        problems.append("metadata field 'suite' must be one of %s, got %r" % ("/".join(SUITES), suite))
    if commit is not None and not (isinstance(commit, str) and COMMIT_PATTERN.match(commit)):
        problems.append("metadata field 'commit' must be a lowercase hexadecimal commit hash of at least 7 characters, got %r" % (commit,))
    if dirty is not None and type(dirty) is not bool:
        problems.append("metadata field 'dirty' must be a boolean, got %r" % (dirty,))
    if gpu is not None and not (isinstance(gpu, str) and gpu.strip()):
        problems.append("metadata field 'gpu' must be a non-empty string, got %r" % (gpu,))
    if sms is not None and not (type(sms) is int and sms > 0):
        problems.append("metadata field 'sm_count' must be a positive integer, got %r" % (sms,))
    for field in ("tree", "host", "label", "junit") + TOOLCHAIN:
        if meta.get(field) is not None and not isinstance(meta[field], str):
            problems.append("metadata field %r must be a string, got %r" % (field, meta[field]))
    env = meta.get("env")
    if env is not None and not (isinstance(env, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())):
        problems.append("metadata field 'env' must map names to strings, got %r" % (env,))
    for field in ("ignored", "hosts"):
        if meta.get(field) is not None and not _str_list(meta[field]):
            problems.append("metadata field %r must be a list of strings, got %r" % (field, meta[field]))
    if meta.get("subset") is not None and type(meta["subset"]) is not bool:
        problems.append("metadata field 'subset' must be a boolean, got %r" % (meta["subset"],))
    if meta.get("kind") is not None and meta["kind"] not in KINDS:
        problems.append("metadata field 'kind' must be one of %s, got %r" % ("/".join(KINDS), meta["kind"]))
    return problems


def selection_problems(rec):
    """The selection contract of a record: which files its run left out (``ignored``), whether it says so (``subset``), and what a merge is made of."""
    meta, files = rec.get("meta") or {}, rec.get("files")
    ignored = meta.get("ignored") if _str_list(meta.get("ignored")) else []
    problems = []
    if meta.get("kind") == "merge":
        if meta.get("subset") is not False:
            problems.append("a merge is not a subset record (subset must be false)")
        if ignored:
            problems.append("a merge ignores nothing, this one ignores %s" % ", ".join(ignored))
        parts = meta.get("parts")
        if not isinstance(parts, list) or len(parts) < 2 or any(not isinstance(p, dict) for p in parts):
            problems.append("a merge names at least two parts")
        else:
            for i, p in enumerate(parts):
                h = p.get("sha256")
                if not (isinstance(h, str) and re.match(r"^[0-9a-f]{64}$", h)):
                    problems.append("part %d has no sha256 of its record" % i)
            counts = [p.get("tests") for p in parts]
            if all(_count_ok(c) for c in counts) and isinstance(rec.get("tests"), dict) and sum(counts) != len(rec["tests"]):
                problems.append("the parts hold %d tests, the record %d" % (sum(counts), len(rec["tests"])))
        expected = meta.get("expected")
        if isinstance(expected, dict):
            if _count_ok(expected.get("files")) and isinstance(files, dict) and expected["files"] != len(files):
                problems.append("the record expected files %d but holds %d" % (expected["files"], len(files)))
            if _count_ok(expected.get("tests")) and isinstance(rec.get("tests"), dict) and expected["tests"] != len(rec["tests"]):
                problems.append("the record expected tests %d but holds %d" % (expected["tests"], len(rec["tests"])))
    elif ignored and meta.get("subset") is not True:
        problems.append("a record that ignores files must say it is a subset (subset: true)")
    if isinstance(files, dict):
        for f in ignored:
            if f in files:
                problems.append("the file %s is ignored and has results" % f)
    return problems


def structure_problems(rec):
    if not isinstance(rec, dict) or rec.get("schema") != SCHEMA:
        return ["not a %s record" % SCHEMA]
    problems = []
    if not isinstance(rec.get("meta"), dict):
        problems.append("no metadata")
    else:
        problems.extend(metadata_problems(rec["meta"]))
    files = rec.get("files")
    files_ok = isinstance(files, dict) and bool(files)
    if not files_ok:
        problems.append("no files")
    else:
        for f, c in files.items():
            if not isinstance(c, dict) or any(not _count_ok(c.get(k)) for k in COUNTS):
                problems.append("file %s: malformed counts (non-negative integers expected)" % f)
                files_ok = False
            elif c["passed"] + c["failed"] + c["skipped"] + c["errors"] != c["total"]:
                problems.append("file %s: the counts do not add up" % f)
    totals = rec.get("totals")
    if not isinstance(totals, dict) or sorted(totals) != sorted(COUNTS) or any(not _count_ok(v) for v in totals.values()):
        problems.append("malformed totals (the counts %s as non-negative integers expected)" % ", ".join(COUNTS))
        totals = None
    elif files_ok and totals != dict((k, sum(c[k] for c in files.values())) for k in COUNTS):
        problems.append("the totals do not match the per-file counts")
    failed = rec.get("failed")
    if not _str_list(failed) or len(set(failed)) != len(failed):
        problems.append("malformed list of failed tests (unique ids expected)")
    elif totals is not None and len(failed) != totals["failed"] + totals["errors"]:
        problems.append("the list of failed tests has %d entries, the totals say %d" % (len(failed), totals["failed"] + totals["errors"]))
    skipped = rec.get("skipped")
    skipped_ids = None
    if not isinstance(skipped, list) or any(not (isinstance(e, dict) and isinstance(e.get("id"), str) and isinstance(e.get("reason"), str)) for e in skipped):
        problems.append("malformed list of skipped tests (an id and a reason each)")
    else:
        skipped_ids = [e["id"] for e in skipped]
        if len(set(skipped_ids)) != len(skipped_ids):
            problems.append("the list of skipped tests holds an id twice")
        elif totals is not None and len(skipped_ids) != totals["skipped"]:
            problems.append("the list of skipped tests has %d entries, the totals say %d" % (len(skipped_ids), totals["skipped"]))
        if files_ok:
            per_file = {}
            for tid in skipped_ids:
                per_file[tid.split("::", 1)[0]] = per_file.get(tid.split("::", 1)[0], 0) + 1
            for f in sorted(set(per_file) | set(files)):
                if per_file.get(f, 0) != files.get(f, {}).get("skipped", 0):
                    problems.append("file %s: the skipped list has %d entries, the counts say %d" % (f, per_file.get(f, 0), files.get(f, {}).get("skipped", 0)))
    tests = rec.get("tests")
    if tests is not None:                                    # per-test outcomes must say what the per-file counts say
        if not isinstance(tests, dict) or any(v not in OUTCOMES for v in tests.values()):
            problems.append("malformed per-test outcomes")
        elif isinstance(files, dict) and files:
            derived = {}
            for tid, outcome in tests.items():
                c = derived.setdefault(tid.split("::", 1)[0], dict.fromkeys(COUNTS, 0))
                c[outcome] += 1
                c["total"] += 1
            if derived != files:
                problems.append("the per-file counts do not match the per-test outcomes")
            if _str_list(failed) and sorted(failed) != sorted(t for t, o in tests.items() if o in ("failed", "errors")):
                problems.append("the list of failed tests does not match the per-test outcomes")
            if skipped_ids is not None and sorted(skipped_ids) != sorted(t for t, o in tests.items() if o == "skipped"):
                problems.append("the skipped list does not match the per-test outcomes")
    if isinstance(rec.get("meta"), dict):
        problems.extend(selection_problems(rec))
    return problems


def comparability_problems(baseline, candidate):
    """Why two structurally valid records are not runs of the same suite on the same device with the same toolchain and environment (empty: comparable)."""
    b, c = baseline["meta"], candidate["meta"]
    problems = []
    sd_against_s = b["suite"] == "S" and c["suite"] == "Sd"
    if b["suite"] != c["suite"] and not sd_against_s:
        problems.append("the records are of different suites (%s, %s)" % (b["suite"], c["suite"]))
    for name, m in (("baseline", b), ("candidate", c)):
        if m["dirty"] is not False:
            problems.append("the %s was run on a dirty tree" % name)
    for field in ("gpu", "sm_count"):
        if b[field] != c[field]:
            problems.append("the records are not comparable: %s %r against %r" % (field, b[field], c[field]))
    if sd_against_s:
        if (c.get("env") or {}) != dict(b.get("env") or {}, **dict([SD_ENV])):
            problems.append("the records are not comparable: Suite Sd must run in the environment of Suite S plus %s=%s (%r against %r)" % (SD_ENV + (b.get("env") or {}, c.get("env") or {})))
    elif (b.get("env") or {}) != (c.get("env") or {}):
        problems.append("the records are not comparable: env %r against %r" % (b.get("env") or {}, c.get("env") or {}))
    for field in TOOLCHAIN:
        bv, cv = b.get(field), c.get(field)
        if (bv is None) != (cv is None):
            problems.append("only one of the records says which %s it ran with (%r, %r)" % (field, bv, cv))
        elif bv != cv:
            problems.append("the records are not comparable: %s %r against %r" % (field, bv, cv))
    return problems


def check(baseline, candidate, allowed_failures=()):
    """Problems of ``candidate`` against ``baseline`` (an empty list means the check passes)."""
    problems = ["baseline: %s" % p for p in structure_problems(baseline)] + ["candidate: %s" % p for p in structure_problems(candidate)]
    if problems:
        return problems
    problems.extend(comparability_problems(baseline, candidate))
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


def completeness_problems(rec, expected_files=None, expected_tests=None):
    """Why ``rec`` does not certify the whole suite (an empty list means it does): it is no subset record, ignores no file, holds every expected file and exactly the
    expected test ids. Whether the tests passed is a different question (``check``)."""
    problems = structure_problems(rec)
    if problems:
        return problems
    meta = rec["meta"]
    if meta.get("subset"):
        problems.append("the record is a subset record: it leaves files of its suite out")
    if meta.get("ignored"):
        problems.append("the record ignores %s" % ", ".join(meta["ignored"]))
    if expected_files is not None:
        missing = sorted(set(expected_files) - set(rec["files"]))
        if missing:
            problems.append("%d expected file(s) are missing: %s" % (len(missing), ", ".join(missing[:5])))
    if expected_tests is not None:
        if rec.get("tests") is None:
            problems.append("the record has no per-test outcomes to compare with the expected test ids")
        else:
            missing, extra = sorted(set(expected_tests) - set(rec["tests"])), sorted(set(rec["tests"]) - set(expected_tests))
            if missing:
                problems.append("%d expected test(s) are missing: %s" % (len(missing), ", ".join(missing[:5])))
            if extra:
                problems.append("%d test(s) are not expected: %s" % (len(extra), ", ".join(extra[:5])))
    return problems


def merge_problems(parts, expected_files=None, expected_tests=None):
    """Why ``parts`` (suite records) are not a valid merge (an empty list means they are); see the module docstring for the rules."""
    problems = []
    if len(parts) < 2:
        problems.append("a merge needs at least two records, got %d" % len(parts))
    for i, p in enumerate(parts):
        problems.extend("part %d: %s" % (i, q) for q in structure_problems(p))
        if isinstance(p, dict) and isinstance(p.get("meta"), dict):
            if p.get("tests") is None:
                problems.append("part %d has no per-test outcomes (build it again from its JUnit file)" % i)
            if p["meta"].get("dirty") is not False:
                problems.append("part %d was run on a dirty or unidentified tree" % i)
    if problems:
        return problems
    for field in MERGE_SAME:
        values = sorted(set(json.dumps(p["meta"].get(field), sort_keys=True) for p in parts))
        if len(values) > 1:
            problems.append("the parts differ in %s: %s" % (field, ", ".join(values)))
    owner = {}
    for i, p in enumerate(parts):
        for tid in p["tests"]:
            if tid in owner:
                problems.append("test %s is in part %d and in part %d" % (tid, owner[tid], i))
            owner.setdefault(tid, i)
    covered = set(tid.split("::", 1)[0] for tid in owner)
    for i, p in enumerate(parts):
        for f in p["meta"].get("ignored") or []:
            if f not in covered:
                problems.append("the file %s was ignored by part %d and no part covers it" % (f, i))
    if expected_files is not None:
        missing, extra = sorted(set(expected_files) - covered), sorted(covered - set(expected_files))
        if missing:
            problems.append("%d expected file(s) are in no part: %s" % (len(missing), ", ".join(missing[:5])))
        if extra:
            problems.append("%d file(s) are not expected: %s" % (len(extra), ", ".join(extra[:5])))
    if expected_tests is not None:
        missing, extra = sorted(set(expected_tests) - set(owner)), sorted(set(owner) - set(expected_tests))
        if missing:
            problems.append("%d expected test(s) are in no part: %s" % (len(missing), ", ".join(missing[:5])))
        if extra:
            problems.append("%d test(s) are not expected: %s" % (len(extra), ", ".join(extra[:5])))
    return problems


def merge_records(parts, infos, label="", expected_files=None, expected_tests=None, expected_source=None):
    """The merged record of ``parts`` (``infos``: one dict per part naming it, e.g. its file and hash); raises ``ValueError`` when they are not a valid merge."""
    problems = merge_problems(parts, expected_files, expected_tests)
    if problems:
        raise ValueError("; ".join(problems))
    meta = dict((f, parts[0]["meta"].get(f)) for f in MERGE_SAME)
    meta.update({"kind": "merge", "subset": False, "label": label, "ignored": [], "hosts": sorted(set(p["meta"].get("host") for p in parts if p["meta"].get("host"))),
                 "parts": [dict(info, label=p["meta"].get("label"), junit=p["meta"].get("junit"), host=p["meta"].get("host"), started=p["meta"].get("started"),
                                tests=len(p["tests"])) for p, info in zip(parts, infos)]})
    if expected_files is not None or expected_tests is not None:
        meta["expected"] = {"source": expected_source, "files": None if expected_files is None else len(expected_files),
                            "tests": None if expected_tests is None else len(expected_tests)}
    tests, files = {}, {}
    for p in parts:
        tests.update(p["tests"])
    for tid, outcome in tests.items():
        c = files.setdefault(tid.split("::", 1)[0], dict.fromkeys(COUNTS, 0))
        c[outcome] += 1
        c["total"] += 1
    parsed = {"files": files, "tests": tests, "failed": sorted(t for t, o in tests.items() if o in ("failed", "errors")),
              "skipped": sorted((s for p in parts for s in p["skipped"]), key=lambda s: s["id"])}
    return build_record(parsed, meta)


def _expected(tree, ids_path):
    """``(files, ids)``: the ``tests/cfbi/test_*.py`` of a source tree and the test ids of a collection file (None where not given)."""
    files = ids = None
    if tree:
        names = os.listdir(os.path.join(tree, "tests", "cfbi"))
        files = sorted("tests/cfbi/%s" % n for n in names if n.startswith("test_") and n.endswith(".py"))
    if ids_path:
        with open(ids_path, encoding="utf-8") as f:
            ids = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return files, ids


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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
    ap.add_argument("--subset", action="store_true", help="the run left files of its suite out (ignored): the record is a part of a merge, not a baseline of the suite")
    ap.add_argument("--merge", nargs="+", metavar="PART", help="merge suite records of separate runs (needs --out)")
    ap.add_argument("--complete", metavar="RECORD", help="check that a record certifies the whole suite: no subset, nothing ignored, the expected files and tests")
    ap.add_argument("--expect-files-from", default=None, metavar="TREE", help="merge / complete: every tests/cfbi/test_*.py of this source tree must be covered")
    ap.add_argument("--expect-tests", default=None, metavar="FILE",
                    help="merge / complete: the tests must be exactly the ids listed in this file (a pytest collection, one id per line)")
    ap.add_argument("--check", nargs=2, metavar=("BASELINE", "CANDIDATE"))
    ap.add_argument("--allow-failures", default=None)
    a = ap.parse_args(argv)
    try:
        if a.merge:
            if not a.out:
                print("--merge needs --out", file=sys.stderr)
                return 2
            files, ids = _expected(a.expect_files_from, a.expect_tests)
            rec = merge_records([_load(p) for p in a.merge], [{"file": os.path.basename(p), "sha256": _sha256(p)} for p in a.merge], a.label, files, ids,
                                a.expect_files_from or a.expect_tests)
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=1, sort_keys=True)
                f.write("\n")
            t = rec["totals"]
            print("merge of %d records: %d files, %d passed, %d failed, %d errors, %d skipped" % (len(a.merge), len(rec["files"]), t["passed"], t["failed"], t["errors"], t["skipped"]))
            return 0
        if a.complete:
            files, ids = _expected(a.expect_files_from, a.expect_tests)
            problems = completeness_problems(_load(a.complete), files, ids)
            for p in problems:
                print("PROBLEM: %s" % p)
            print("VERDICT: %s" % ("COMPLETE" if not problems else "INCOMPLETE"))
            return 0 if not problems else 1
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
                "env": dict(e.split("=", 1) for e in a.env), "ignored": list(a.ignored), "label": a.label, "junit": os.path.basename(a.junit), "kind": "single",
                "subset": bool(a.subset or a.ignored)}
        if a.probe:
            with open(a.probe, encoding="utf-8", errors="replace") as f:
                probe = parse_probe(f.read())
            for k in ("gpu", "sm_count"):
                if meta[k] is None:
                    meta[k] = probe.get(k)
            meta.update({k: probe.get(k) for k in ("python", "torch", "cuda")})
        parsed = parse_junit(a.junit)
        if parsed["duplicates"]:
            print("unusable: %d test id(s) occur more than once in %s, e.g. %s" % (len(parsed["duplicates"]), a.junit, parsed["duplicates"][0]), file=sys.stderr)
            return 2
        meta.update({"host": parsed["session"].get("hostname"), "started": parsed["session"].get("timestamp"), "duration_s": parsed["session"].get("time")})
        rec = build_record(parsed, meta)
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
