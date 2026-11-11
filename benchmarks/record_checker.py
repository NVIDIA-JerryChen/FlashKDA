#!/usr/bin/env python3
"""Checker of the evidence records of the decode work (plain Python, no torch: runs on the login node).

One module for the rules that every later gate reuses. Each rule is a function that returns a list of problems (empty: the record passes) and a subcommand that prints
them (``PROBLEM: ...``) followed by ``VERDICT: PASS|FAIL``; exit status 0 pass, 1 a check fails, 2 unusable input (unreadable file, wrong schema, no repository).

  record_checker.py profile  --record FILE ... [--device NAME ...] [--shapes N ...] [--heads H] [--tolerance FRACTION] [--expect-commit HASH]
  record_checker.py ncu      --record FILE ... [--device NAME ...] [--shapes N ...] [--heads H] [--expect-commit HASH] [--min-samples N]
  record_checker.py suite    --baseline FILE --candidate FILE [--allow-failures FILE]
  record_checker.py baseline  --record FILE [--allow-missing-raw]
  record_checker.py commit   --record-commit HASH --repo DIR [--head REV] [--require-clean]
  record_checker.py markers  --ledger FILE --tree DIR [--repo DIR]
  record_checker.py ledger   --ledger FILE [--repo DIR] [--decision FILE] [--require-all-steps]
  record_checker.py decision --record FILE
  record_checker.py handoff  --record FILE --repo DIR [--ledger FILE]
  record_checker.py final    --record FILE [--document FILE]

What each rule asks (the schemas of the records that later steps write are the ones below):

profile  ``cfbi-rows-profile-v1`` records of ``benchmarks/micro/rows_profile.py``: the profile build, the expected shapes at the expected number of heads, one cell for each of them,
         every row of every call recorded, the phases adding up to the measured row duration within the tolerance (default 5 %), the shares, the kernel total, the groups and the
         per-position blocks derived from the same rows (arithmetic to 1e-6), a clean tree and a commit; ``--device`` demands a valid record for every named device and one commit
         for the whole set.
ncu      ``cfbi-ncu-segments-v1`` records (``benchmarks/micro/ncu_segments.py --json``): the case (row position, cache control, shape), the collection (--set full, boost clock, the warp
         sampling interval), the hashes of the report and of the source page, segments that add up to the totals, and enough stall samples (a launch with fewer than 1000 says nothing
         about a segment); ``--device`` demands every shape at the required position for every device.
suite    two ``cfbi-suite-v1`` records: both valid, of the same suite, device, environment and toolchain, no file with fewer passed tests, no failure that is not allowed.
baseline ``cfbi-baseline-v2`` (the consolidated baseline of one device): its own consistency (the self-check of the assembler passed, every A/A verdict is PASS), every raw record it
         names loaded again and validated with the committed validators, every verdict recomputed with the committed comparator, the suite records validated.
commit   the record commit rule of the final validation records: the commit that a record names differs from HEAD only under ``docs/``. Historical records (profile, baseline, per-step
         A/B) are checked against their own snapshot commit instead, never against HEAD.
markers  the step ledger names for every step the identifiers that only that step adds (or ``none``); the checker reads the names from the ledger and never hard-codes them. It searches
         ``csrc/``, ``flash_kda/``, ``tests/`` and ``benchmarks/`` of the tree, without this script and every ``test_record_checker*`` file and never ``docs/``: the markers of an adopted
         step must be in ``csrc/`` or ``flash_kda/``, those of a step that was not adopted or was skipped nowhere. For a step without symbols the commit decides: an adopted step's commit
         is an ancestor of HEAD and no later revert commit names it, a step that was not adopted or skipped has no commit or its revert commit is an ancestor of HEAD and names it.
ledger   ``cfbi-step-ledger-v1``: one row for each step (``single_launch``, ``trims``, ``e_tile``, ``a_fragments``, ``kd_tile``) with its outcome (``adopted``, ``not_adopted``, ``skipped`` by the
         decision record), commit, markers and the evidence of the phases the step reached (compile, exactness, codegen, timing; the measurements it never reached are null and the row
         carries a reason). An adopted row must carry all of it; the E tile and the A fragments are one adoption unit decided on three arms (T the previous accepted state, E, F) by a
         table (E passes against T and F beats E: both adopted; E passes and F does not beat E: E only; E fails against T and F beats T: both; neither: none) and a ledger that
         contradicts the table is rejected. ``--decision`` cross-checks the skipped steps with the decision record.
decision a decision record has the keys ``row_engine_order`` (the steps of the row engine in their default order, a step may only be left out, never moved), ``value_split_inputs`` and ``budget_table``.
handoff  ``cfbi-handoff-v1``: the value-split and the operand-swap hand-offs (branch of the form ``explore/<run>/<topic>``, commit, both existing in the repository, the probe test on the branch,
         the next free implementation bit: 128 when single-launch decode took 64, else 64) and the known-limit section of the performance document (topics named without plan identifiers).
final    ``cfbi-final-v1``: the final A/B record of every device has a verdict, a median difference and a tolerance for every cell (verdict and difference agree); a regression that was confirmed
         is a known limit only if the same shift also appears with all of the changes reverted (both measurements quoted) and the performance document names the cell.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_decode as bd  # noqa: E402
import bench_records as br  # noqa: E402
import ncu_record as nr  # noqa: E402
import rows_profile_record as rpr  # noqa: E402
import suite_record as sr  # noqa: E402

PLAN_VOCABULARY = re.compile(r"\b(AC-[0-9]+|FUT-[0-9]+|DEC-[0-9]+|Milestone [0-9]+|Stage [0-9]+|dir-0[0-9]|task[0-9]+)\b")


# ---- small helpers ---------------------------------------------------------------------------------------------------------------------------
def num(x):
    """A finite real number (a boolean is not one)."""
    return type(x) in (int, float) and math.isfinite(x)


def count(x, minimum=0):
    return type(x) is int and x >= minimum


def text(x):
    return isinstance(x, str) and bool(x.strip())


def close(a, b, rel=1e-6):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-12)


def hexcommit(x):
    return isinstance(x, str) and bool(br.COMMIT_PATTERN.match(x))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    if not isinstance(rec, dict):
        raise ValueError("%s: a record must be a JSON object" % path)
    rec["_path"] = str(path)
    return rec


# ---- git: the record commit rule, reverts ------------------------------------------------------------------------------------------------------
def git(repo, *args):
    """``(exit status, stdout, stderr)`` of ``git -C repo args``."""
    try:
        p = subprocess.Popen(["git", "-C", str(repo)] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as e:
        return 127, "", "cannot run git: %s" % e
    out, err = p.communicate()
    return p.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


def resolve(repo, rev):
    """The full hash of a commit-ish, or None."""
    rc, out, _ = git(repo, "rev-parse", "--verify", "--quiet", "%s^{commit}" % rev)
    return out.strip() if rc == 0 and out.strip() else None


def is_ancestor(repo, ancestor, descendant):
    return git(repo, "merge-base", "--is-ancestor", ancestor, descendant)[0] == 0


def reverts_naming(repo, commit, head):
    """The revert commits between ``commit`` and ``head`` whose message names ``commit`` (a hash of at least 7 characters that is a prefix of it)."""
    rc, out, _ = git(repo, "log", "--format=%H%x1f%B%x1e", "%s..%s" % (commit, head))
    found = []
    if rc != 0:
        return found
    for entry in out.split("\x1e"):
        if "\x1f" not in entry:
            continue
        sha, body = entry.strip("\n").split("\x1f", 1)
        if not body.lstrip().startswith("Revert"):
            continue
        if any(commit.startswith(tok) for tok in re.findall(r"\b[0-9a-f]{7,40}\b", body)):
            found.append(sha.strip())
    return found


def record_commit_problems(commit, repo, head="HEAD", docs="docs", where="record"):
    """The record commit rule: the commit ``commit`` that a result record names is valid for ``head`` if every path changed between them lies under ``docs/``."""
    if not hexcommit(commit):
        return ["%s: commit %r is not a hexadecimal commit hash of at least 7 characters" % (where, commit)]
    full, head_full = resolve(repo, commit), resolve(repo, head)
    if full is None:
        return ["%s names commit %s, which the repository does not have" % (where, commit)]
    if head_full is None:
        return ["cannot resolve %s in the repository" % head]
    if full != head_full and not is_ancestor(repo, full, head_full):
        return ["%s names commit %s, which is not an ancestor of %s" % (where, commit, head)]
    rc, out, err = git(repo, "diff", "--name-only", "--no-renames", full, head_full)
    if rc != 0:
        return ["git diff %s %s failed: %s" % (commit, head, err.strip())]
    outside = [path for path in out.splitlines() if not (path == docs or path.startswith(docs + "/"))]
    if outside:
        return ["%s names commit %s, which differs from %s in %d path(s) outside %s/: %s" % (where, commit, head, len(outside), docs, ", ".join(outside[:5]))]
    return []


UNTRACKED_OK = "?? .humanize/"      # the one directory whose untracked contents do not make a tree dirty: the evidence records of a run live there, outside the sources


def clean_tree_problems(repo):
    """The tree is clean: ``git status --porcelain`` shows nothing except untracked paths of the evidence directory."""
    rc, out, err = git(repo, "status", "--porcelain")
    if rc != 0:
        return ["git status failed: %s" % err.strip()]
    dirty = [ln for ln in out.splitlines() if not ln.startswith(UNTRACKED_OK)]
    return ["the working tree is not clean: %s" % "; ".join(dirty[:5])] if dirty else []


# ---- profile records -------------------------------------------------------------------------------------------------------------------------
PROFILE_DEVICES = ("B200", "B300")
PROFILE_SHAPES = (1, 16, 128)
PROFILE_HEADS = 12
PROFILE_VARIANT = ["CFBI_ROWS_PROFILE=1"]
ARITHMETIC_TOLERANCE = 1e-6      # what is derived from the same rows (shares, kernel total, groups, positions) agrees to this relative error
BLOCK_NUMBERS = ("prologue", "epilogue", "row_total", "kernel_total", "row_total_p50", "row_total_p95", "row_total_max")


def block_problems(block, where, tolerance=rpr.SUM_TOLERANCE):
    """One block of means (a group or a position): the numbers are there, the phases add up to the measured row duration within ``tolerance``, the shares and the kernel total follow."""
    if not isinstance(block, dict):
        return ["%s: not a block" % where]
    if not count(block.get("rows"), 1):
        return ["%s: a block needs at least one row (rows = %r)" % (where, block.get("rows"))]
    problems = ["%s: %s is missing or not a finite number" % (where, f) for f in BLOCK_NUMBERS if not num(block.get(f))]
    phases, shares = block.get("phases"), block.get("shares")
    if not (isinstance(phases, dict) and sorted(phases) == sorted(rpr.PHASES) and all(num(v) for v in phases.values())):
        problems.append("%s: phases must hold exactly %s as finite numbers" % (where, ", ".join(rpr.PHASES)))
    if not (isinstance(shares, dict) and sorted(shares) == sorted(rpr.PHASES) and all(num(v) for v in shares.values())):
        problems.append("%s: shares must hold exactly %s as finite numbers" % (where, ", ".join(rpr.PHASES)))
    if problems:
        return problems
    if block["prologue"] <= 0:
        problems.append("%s: the prologue is %r cycles: the entry stamp must precede the first row" % (where, block["prologue"]))
    if block["epilogue"] < 0 or any(v < 0 for v in phases.values()):
        problems.append("%s: a phase or the epilogue is negative: the stamps of a row never decrease" % where)
    if block["row_total"] <= 0:
        return problems + ["%s: the row duration is %r cycles" % (where, block["row_total"])]
    total = sum(phases.values())
    if abs(total - block["row_total"]) > tolerance * block["row_total"]:
        problems.append("%s: the phases add up to %.1f cycles, the measured row duration is %.1f (more than %.1f %% apart)" % (where, total, block["row_total"], 100 * tolerance))
    for k in rpr.PHASES:
        if not close(shares[k], phases[k] / block["row_total"], ARITHMETIC_TOLERANCE):
            problems.append("%s: the share of %s is %r, the phase over the row duration is %r" % (where, k, shares[k], phases[k] / block["row_total"]))
    if not close(block["kernel_total"], block["prologue"] + block["row_total"] + block["epilogue"], ARITHMETIC_TOLERANCE):
        problems.append("%s: the kernel total %r is not prologue + row duration + epilogue = %r" % (where, block["kernel_total"], block["prologue"] + block["row_total"] + block["epilogue"]))
    if not block["row_total_p50"] <= block["row_total_p95"] <= block["row_total_max"] or block["row_total"] > block["row_total_max"] * (1 + ARITHMETIC_TOLERANCE):
        problems.append("%s: the median, the 95th percentile and the maximum of the row duration are not ordered around the mean" % where)
    return problems


def combined(blocks):
    """The means over the rows of several blocks (each block weighs by its rows)."""
    n = float(sum(b["rows"] for b in blocks))
    mean = lambda f: sum(b["rows"] * f(b) for b in blocks) / n
    out = {"rows": int(n), "row_total_max": max(b["row_total_max"] for b in blocks), "phases": dict((k, mean(lambda b, k=k: b["phases"][k])) for k in rpr.PHASES)}
    for f in ("prologue", "epilogue", "row_total", "kernel_total"):
        out[f] = mean(lambda b, f=f: b[f])
    return out


def cell_problems(cell, N, H, sm_count=None, tolerance=rpr.SUM_TOLERANCE):
    """One cell (N sequences, H heads) of a profile record."""
    where = "cell %s" % rpr.cell_key(N, H)
    if not isinstance(cell, dict):
        return ["%s: not a cell" % where]
    ctas, positions = N * H, rpr.POSITIONS
    problems = []
    for field, expected in (("key", rpr.cell_key(N, H)), ("N", N), ("H", H), ("ctas", ctas), ("rows_expected", positions * ctas)):
        if cell.get(field) != expected or type(cell.get(field)) is bool:
            problems.append("%s: %s is %r, expected %r" % (where, field, cell.get(field), expected))
    if cell.get("rows_recorded") != positions * ctas or type(cell.get("rows_recorded")) is bool:
        problems.append("%s: %r of %d rows were recorded: every row of every call must be in the profile" % (where, cell.get("rows_recorded"), positions * ctas))
    sms = cell.get("sms")
    if not (isinstance(sms, list) and sms and all(count(s) for s in sms) and len(set(sms)) == len(sms)):
        problems.append("%s: sms must be a list of distinct SM numbers" % where)
    elif sm_count is not None and (max(sms) >= sm_count or len(sms) > sm_count):
        problems.append("%s: SM numbers up to %d on a device with %d SMs" % (where, max(sms), sm_count))
    rows, per_position = cell.get("rows"), cell.get("per_position")
    if not isinstance(rows, dict) or sorted(rows) != sorted(rpr.GROUPS):
        problems.append("%s: rows must hold the groups %s" % (where, ", ".join(rpr.GROUPS)))
    if not (isinstance(per_position, list) and len(per_position) == positions):
        problems.append("%s: per_position must hold %d blocks" % (where, positions))
    if problems:
        return problems
    expected_rows = {"regular": (positions - 1) * ctas, "chunk_end": ctas, "all": positions * ctas}
    for g in rpr.GROUPS:
        problems.extend(block_problems(rows[g], "%s, %s rows" % (where, g), tolerance))
        if isinstance(rows[g], dict) and rows[g].get("rows") != expected_rows[g]:
            problems.append("%s, %s rows: %r rows, expected %d" % (where, g, rows[g].get("rows"), expected_rows[g]))
    for pos, block in enumerate(per_position):
        problems.extend(block_problems(block, "%s, position %d" % (where, pos), tolerance))
        if isinstance(block, dict) and (block.get("pos") != pos or block.get("rows") != ctas):
            problems.append("%s, position %d: pos %r with %r rows, expected %d rows" % (where, pos, block.get("pos"), block.get("rows"), ctas))
    if problems:
        return problems
    for g, subset in (("regular", per_position[:positions - 1]), ("chunk_end", per_position[positions - 1:]), ("all", per_position)):
        derived = combined(subset)
        for f in ("prologue", "epilogue", "row_total", "kernel_total"):
            if not close(rows[g][f], derived[f], ARITHMETIC_TOLERANCE):
                problems.append("%s, %s rows: %s is %r, the means of its positions give %r" % (where, g, f, rows[g][f], derived[f]))
        for k in rpr.PHASES:
            if not close(rows[g]["phases"][k], derived["phases"][k], ARITHMETIC_TOLERANCE):
                problems.append("%s, %s rows: the mean of %s is %r, the means of its positions give %r" % (where, g, k, rows[g]["phases"][k], derived["phases"][k]))
        if rows[g]["row_total_max"] != derived["row_total_max"]:
            problems.append("%s, %s rows: the maximum row duration %r is not the largest of its positions (%r)" % (where, g, rows[g]["row_total_max"], derived["row_total_max"]))
    return problems


def profile_problems(rec, shapes=PROFILE_SHAPES, heads=PROFILE_HEADS, tolerance=rpr.SUM_TOLERANCE, expect_commit=None):
    """Problems of one ``cfbi-rows-profile-v1`` record: schema, identity (a clean tree and a commit, the expected one when given), the profile build, completeness (one cell for every
    expected shape, no other), and the consistency of every cell."""
    where = "profile record %s" % (rec.get("_path", "?") if isinstance(rec, dict) else "?")
    if not isinstance(rec, dict) or rec.get("schema") != rpr.SCHEMA:
        return ["%s: not a %s record" % (where, rpr.SCHEMA)]
    meta, cells = rec.get("meta"), rec.get("cells")
    if not isinstance(meta, dict) or not isinstance(cells, list):
        return ["%s: metadata and cells are required" % where]
    problems = br.identity_problems(meta, where, expect_commit, True)
    if meta.get("variant") != PROFILE_VARIANT:
        problems.append("%s: not the profile build: variant %r, expected %r" % (where, meta.get("variant"), PROFILE_VARIANT))
    if meta.get("heads") != heads or type(meta.get("heads")) is bool:
        problems.append("%s: heads %r, expected %d" % (where, meta.get("heads"), heads))
    if not (isinstance(meta.get("shapes"), list) and sorted(meta["shapes"]) == sorted(shapes)):
        problems.append("%s: shapes %r, expected %s" % (where, meta.get("shapes"), list(shapes)))
    if meta.get("positions") != rpr.POSITIONS or type(meta.get("positions")) is bool:
        problems.append("%s: positions %r, expected %d" % (where, meta.get("positions"), rpr.POSITIONS))
    for field in ("timestamp", "torch", "cuda", "cc"):
        if not text(meta.get(field)):
            problems.append("%s: metadata field %r must be a non-empty string" % (where, field))
    keys = [c.get("key") if isinstance(c, dict) else None for c in cells]
    wanted = [rpr.cell_key(n, heads) for n in shapes]
    for key in wanted:
        if keys.count(key) == 0:
            problems.append("%s: the cell %s is missing" % (where, key))
        elif keys.count(key) > 1:
            problems.append("%s: the cell %s appears %d times" % (where, key, keys.count(key)))
    for key in keys:
        if key not in wanted:
            problems.append("%s: the cell %r is not one of the expected cells" % (where, key))
    sm_count = meta.get("sm_count") if count(meta.get("sm_count"), 1) else None
    for n in shapes:
        cell = next((c for c in cells if isinstance(c, dict) and c.get("key") == rpr.cell_key(n, heads)), None)
        if cell is not None:
            problems.extend("%s: %s" % (where, p) for p in cell_problems(cell, n, heads, sm_count, tolerance))
    return problems


def profile_set_problems(records, devices=PROFILE_DEVICES, shapes=PROFILE_SHAPES, heads=PROFILE_HEADS, tolerance=rpr.SUM_TOLERANCE, expect_commit=None):
    """The profile records of an evidence set: every record valid, every named device covered, no other device, one commit for the whole set."""
    problems = []
    for rec in records:
        problems.extend(profile_problems(rec, shapes, heads, tolerance, expect_commit))
    covered = dict((d, 0) for d in devices)
    for rec in records:
        gpu = ((rec.get("meta") or {}).get("gpu") or "") if isinstance(rec, dict) else ""
        match = [d for d in devices if d.lower() in str(gpu).lower()]
        if len(match) != 1:
            problems.append("profile record %s: GPU %r is none of %s" % (rec.get("_path", "?") if isinstance(rec, dict) else "?", gpu, ", ".join(devices)))
        else:
            covered[match[0]] += 1
    for d in devices:
        if not covered[d]:
            problems.append("no profile record for %s" % d)
    commits = sorted(set(str((r.get("meta") or {}).get("commit")) for r in records if isinstance(r, dict)))
    if len(commits) > 1 and not all(commits[0].startswith(c) or c.startswith(commits[0]) for c in commits[1:]):
        problems.append("the profile records name different commits: %s" % ", ".join(commits))
    return problems


# ---- markers ---------------------------------------------------------------------------------------------------------------------------------
SCOPE_DIRS = ("csrc", "flash_kda", "tests", "benchmarks")
CODE_DIRS = ("csrc", "flash_kda")
TEXT_SUFFIXES = (".py", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".c", ".txt", ".md", ".json", ".yaml", ".yml", ".toml", ".cfg", ".cmake", ".sh")
MARKER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_:]*$")


def scope_files(tree):
    """``[(path relative to tree, absolute path)]`` of the searched files: the source directories, without this script, without every ``test_record_checker*`` file, never ``docs/``."""
    out = []
    for d in SCOPE_DIRS:
        for dirpath, dirnames, filenames in os.walk(os.path.join(str(tree), d)):
            dirnames[:] = sorted(x for x in dirnames if x not in ("__pycache__", ".git"))
            for name in sorted(filenames):
                rel = os.path.relpath(os.path.join(dirpath, name), str(tree)).replace(os.sep, "/")
                if not name.endswith(TEXT_SUFFIXES) or rel == "benchmarks/record_checker.py" or name.startswith("test_record_checker"):
                    continue
                out.append((rel, os.path.join(dirpath, name)))
    return out


def marker_search(markers, tree):
    """``{marker: [relative paths of the files that name it]}`` (the identifier as a whole word)."""
    patterns = dict((m, re.compile(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(m))) for m in markers)
    found = dict((m, []) for m in markers)
    for rel, path in scope_files(tree):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                body = f.read()
        except OSError:
            continue
        for m, pat in patterns.items():
            if pat.search(body):
                found[m].append(rel)
    return found


def commit_marker_problems(step, repo, head="HEAD"):
    """The commit rule for a step without symbols (markers: none)."""
    sid, outcome, commit, revert = step.get("id"), step.get("outcome"), step.get("commit"), step.get("revert_commit")
    if repo is None:
        return ["step %s has no symbols (markers: none): its commit can only be checked against a repository" % sid]
    head_full = resolve(repo, head)
    if head_full is None:
        return ["cannot resolve %s in the repository" % head]
    problems = []
    if outcome == "adopted":
        full = resolve(repo, commit) if hexcommit(commit) else None
        if full is None:
            problems.append("adopted step %s: its commit %r is not a commit of the repository" % (sid, commit))
        elif not (full == head_full or is_ancestor(repo, full, head_full)):
            problems.append("adopted step %s: its commit %s is not an ancestor of %s" % (sid, commit, head))
        elif reverts_naming(repo, full, head_full):
            problems.append("adopted step %s: a later revert commit names %s: %s" % (sid, commit, ", ".join(c[:10] for c in reverts_naming(repo, full, head_full))))
    elif commit is not None:
        full = resolve(repo, commit) if hexcommit(commit) else None
        rfull = resolve(repo, revert) if hexcommit(revert) else None
        if full is None:
            problems.append("step %s (%s): its commit %r is not a commit of the repository" % (sid, outcome, commit))
        elif rfull is None or not (rfull == head_full or is_ancestor(repo, rfull, head_full)):
            problems.append("step %s (%s) keeps its commit %s and has no revert commit that is an ancestor of %s" % (sid, outcome, commit, head))
        elif rfull not in reverts_naming(repo, full, head_full):
            problems.append("step %s (%s): the commit %s does not name %s as the commit it reverts" % (sid, outcome, revert, commit))
    elif revert is not None:
        problems.append("step %s (%s) has a revert commit but no commit" % (sid, outcome))
    return problems


def marker_problems(ledger, tree, repo=None, head="HEAD"):
    """The marker check of a step ledger against a source tree (and the repository for the steps without symbols)."""
    steps = ledger.get("steps") if isinstance(ledger, dict) else None
    if not isinstance(steps, list):
        return ["the ledger has no list of steps"]
    problems = []
    names = []
    for step in steps:
        markers = step.get("markers") if isinstance(step, dict) else None
        if not (isinstance(markers, list) and markers and all(isinstance(m, str) for m in markers)):
            problems.append("step %s: markers must be a non-empty list of identifiers or [\"none\"]" % (step.get("id") if isinstance(step, dict) else "?"))
        elif markers == ["none"]:
            continue
        elif "none" in markers or not all(MARKER_PATTERN.match(m) for m in markers):
            problems.append("step %s: markers %r must be identifiers (or exactly [\"none\"])" % (step.get("id"), markers))
        else:
            names.extend(markers)
    found = marker_search(sorted(set(names)), tree)
    for step in steps:
        if not isinstance(step, dict):
            continue
        sid, outcome, markers = step.get("id"), step.get("outcome"), step.get("markers")
        if not isinstance(markers, list) or markers == ["none"]:
            if markers == ["none"]:
                problems.extend(commit_marker_problems(step, repo, head))
            continue
        for m in markers:
            if m not in found:
                continue
            files = found[m]
            if outcome == "adopted":
                if not [f for f in files if f.split("/", 1)[0] in CODE_DIRS]:
                    problems.append("adopted step %s: the marker %s is not in csrc/ or flash_kda/ (found in: %s)" % (sid, m, ", ".join(files) or "nowhere"))
            elif files:
                problems.append("step %s (%s): the marker %s is still in the tree: %s" % (sid, outcome, m, ", ".join(files[:3])))
    return problems


# ---- suite and baseline records --------------------------------------------------------------------------------------------------------------
BASELINE_SCHEMA = "cfbi-baseline-v2"
VERDICTS = ("PASS", "FAIL", "INVALID")
LOOP_METRICS = ("cpu_eager_us", "cpu_graph_us")                       # host timings of a loop of calls: only the regimes without a helper between the calls have them
LOOP_CELLS = tuple("regime=%s;" % r for r in bd.LOOP_REGIMES)


def metric_cells(bench, metric):
    """The cell filter that a campaign applies to ``metric`` (None: every cell): the host loop metrics of a decode campaign exist for the loop regimes only."""
    return list(LOOP_CELLS) if bench == "decode" and metric in LOOP_METRICS else None


def suite_problems(baseline, candidate, allowed_failures=()):
    """Two ``cfbi-suite-v1`` records: both valid, comparable, and the candidate has at least the passed tests of every file of the baseline and no failure that is not allowed."""
    return sr.check(baseline, candidate, allowed_failures)


def _paths(listing):
    return [p for side in ("A", "B") for p in (listing.get(side) or [])] if isinstance(listing, dict) else []


def campaign_problems(c, name, commit, allow_missing_raw=False):
    """One A/A campaign of a consolidated baseline (the decode matrix or the layer chain): keys and types, the verdicts, the raw records loaded again and validated, and every
    verdict recomputed with the committed comparator."""
    where = "baseline campaign %s" % name
    if not isinstance(c, dict):
        return ["%s: missing" % where]
    problems = []
    metrics, cells = c.get("metrics"), c.get("cells")
    if not (isinstance(metrics, list) and metrics and all(text(m) for m in metrics)):
        problems.append("%s: metrics must be a non-empty list" % where)
        metrics = []
    if not (isinstance(cells, list) and cells and all(text(k) for k in cells)):
        problems.append("%s: cells must be a non-empty list of cell keys" % where)
        cells = []
    if not (isinstance(c.get("tolerance_pct"), (int, float)) and type(c.get("tolerance_pct")) is not bool and c["tolerance_pct"] > 0):
        problems.append("%s: tolerance_pct must be a positive number" % where)
    if not count(c.get("samples_per_side"), 3):
        problems.append("%s: samples_per_side must be an integer of at least 3" % where)
    if c.get("bench") not in ("decode", "chain"):
        problems.append("%s: bench must be decode or chain, got %r" % (where, c.get("bench")))
    if c.get("a") != c.get("b") or not text(c.get("a")):
        problems.append("%s: a baseline is an A/A campaign: both sides must be the same tree (a=%r, b=%r)" % (where, c.get("a"), c.get("b")))
    for field in ("final_verdicts", "first_campaign_verdicts", "noise_limited_cells", "rerun_cells", "cells_needing_rerun", "worst_abs_median_delta_pct"):
        got = c.get(field)
        if not isinstance(got, dict) or any(m not in got for m in metrics):
            problems.append("%s: %s must have an entry for every metric" % (where, field))
    if problems:
        return problems
    for m in metrics:
        if c["final_verdicts"][m] != "PASS":
            problems.append("%s: the final verdict of %s is %r: a baseline needs PASS on every metric" % (where, m, c["final_verdicts"][m]))
        if c["first_campaign_verdicts"][m] not in VERDICTS:
            problems.append("%s: the first-campaign verdict of %s is %r" % (where, m, c["first_campaign_verdicts"][m]))
        for field in ("rerun_cells", "noise_limited_cells"):
            if not isinstance(c[field][m], list) or any(k not in cells for k in c[field][m]):
                problems.append("%s: %s of %s must list cells of the campaign" % (where, field, m))
        if not count(c["cells_needing_rerun"][m]) or (isinstance(c["rerun_cells"][m], list) and c["cells_needing_rerun"][m] > len(c["rerun_cells"][m])):
            problems.append("%s: cells_needing_rerun of %s must be a count that the re-run cells cover" % (where, m))
        if not num(c["worst_abs_median_delta_pct"][m]) or c["worst_abs_median_delta_pct"][m] < 0:
            problems.append("%s: worst_abs_median_delta_pct of %s must be a non-negative number" % (where, m))
    first, rerun = c.get("records_first"), c.get("records_rerun")
    if not (isinstance(first, dict) and isinstance(rerun, dict)):
        return problems + ["%s: records_first and records_rerun must be given as {A: [...], B: [...]}" % where]
    for side in ("A", "B"):
        if len(first.get(side) or []) != c["samples_per_side"]:
            problems.append("%s: records_first holds %d record(s) for %s, samples_per_side is %d" % (where, len(first.get(side) or []), side, c["samples_per_side"]))
    if len(rerun.get("A") or []) != len(rerun.get("B") or []):
        problems.append("%s: records_rerun must hold as many records for A as for B" % where)
    if any(c["cells_needing_rerun"][m] for m in metrics) and not rerun.get("A"):
        problems.append("%s: cells need a re-run and no re-run record is named" % where)
    loaded = {}
    for path in _paths(first) + _paths(rerun):
        try:
            loaded[path] = br.load_record(path)
        except (OSError, ValueError) as e:
            if not allow_missing_raw:
                problems.append("%s: the raw record %s cannot be read again (%s)" % (where, path, e))
    if problems or len(loaded) != len(_paths(first) + _paths(rerun)):
        return problems
    for path, rec in sorted(loaded.items()):
        problems.extend(br.validate_record(rec, "baseline", commit, True))
    for label, listing in (("first campaign", first), ("re-run", rerun)):
        for side in ("A", "B"):
            if listing.get(side):
                problems.extend("%s: %s %s: %s" % (where, label, side, p) for p in br.side_config_problems([loaded[p] for p in listing[side]], "side %s" % side))
    if problems:
        return problems
    base_first, cand_first = [loaded[p] for p in first["A"]], [loaded[p] for p in first["B"]]
    base_rerun, cand_rerun = [loaded[p] for p in rerun.get("A") or []], [loaded[p] for p in rerun.get("B") or []]
    for m in metrics:
        kw = dict(metric=m, tol_pct=c["tolerance_pct"], mode="regress", expect_base_commit=commit, require_clean_cand=True, min_repeats=3, baseline_unstable="widen",
                  cells=metric_cells(c.get("bench"), m))
        final = br.compare(base_first, cand_first, base_rerun=base_rerun or None, cand_rerun=cand_rerun or None, **kw)["verdict"]
        if final != c["final_verdicts"][m]:
            problems.append("%s: the final verdict of %s is %s when recomputed from the raw records, the baseline says %s" % (where, m, final, c["final_verdicts"][m]))
        initial = br.compare(base_first, cand_first, **kw)["verdict"]
        if initial != c["first_campaign_verdicts"][m]:
            problems.append("%s: the first-campaign verdict of %s is %s when recomputed from the raw records, the baseline says %s" % (where, m, initial, c["first_campaign_verdicts"][m]))
    return problems


def baseline_problems(rec, allow_missing_raw=False):
    """Problems of one ``cfbi-baseline-v2`` record (the consolidated baseline of one device)."""
    where = "baseline record %s" % (rec.get("_path", "?") if isinstance(rec, dict) else "?")
    if not isinstance(rec, dict) or rec.get("schema") != BASELINE_SCHEMA:
        return ["%s: not a %s record" % (where, BASELINE_SCHEMA)]
    problems = br.identity_problems(dict(rec, dirty=False), where, None, True, ("commit", "gpu", "sm_count"))
    for field in ("tag", "tree", "torch", "cuda", "created"):
        if not text(rec.get(field)):
            problems.append("%s: field %r must be a non-empty string" % (where, field))
    if not (text(rec.get("tag")) and rec["tag"].lower() in str(rec.get("gpu", "")).lower()):
        problems.append("%s: the device tag %r does not occur in the GPU name %r" % (where, rec.get("tag"), rec.get("gpu")))
    check = rec.get("self_check")
    if not isinstance(check, dict) or check.get("ok") is not True or check.get("failed") != 0 or check.get("problems") != []:
        problems.append("%s: the self-check of the assembler did not pass" % where)
    elif not (count(check.get("checks"), 1) and isinstance(check.get("list"), list) and len(check["list"]) == check["checks"] and all(isinstance(x, dict) and x.get("ok") is True for x in check["list"])):
        problems.append("%s: the self-check lists %r checks of which not every one passed" % (where, check.get("checks")))
    commit = rec.get("commit") if hexcommit(rec.get("commit")) else None
    for name in ("decode_matrix", "layer_chain"):
        problems.extend("%s: %s" % (where, p) for p in campaign_problems(rec.get(name), name, commit, allow_missing_raw))
    suites = rec.get("suites")
    if not isinstance(suites, dict):
        return problems + ["%s: no suites" % where]
    for name in ("S", "gate"):
        entry = suites.get(name)
        if not isinstance(entry, dict) or not text(entry.get("record")):
            problems.append("%s: suite %s has no record" % (where, name))
            continue
        try:
            suite = load_json(entry["record"])
        except (OSError, ValueError) as e:
            if not allow_missing_raw:
                problems.append("%s: the suite %s record cannot be read again (%s)" % (where, name, e))
            continue
        problems.extend("%s: suite %s: %s" % (where, name, p) for p in sr.structure_problems(suite))
        if sr.structure_problems(suite):
            continue
        meta = suite["meta"]
        if meta["suite"] != name or not (str(meta["commit"]).startswith(str(commit)[:7]) if commit else True) or meta["dirty"] is not False:
            problems.append("%s: suite %s: the record is of suite %s, commit %s, dirty %s" % (where, name, meta["suite"], meta["commit"], meta["dirty"]))
        if meta["gpu"] != rec.get("gpu") or meta["sm_count"] != rec.get("sm_count"):
            problems.append("%s: suite %s: the record is of %s with %s SMs, the baseline of %s with %s" % (where, name, meta["gpu"], meta["sm_count"], rec.get("gpu"), rec.get("sm_count")))
        if suite["failed"]:
            problems.append("%s: suite %s: %d failed or errored test(s) in the baseline" % (where, name, len(suite["failed"])))
        if name == "S":
            problems.extend("%s: suite S: %s" % (where, p) for p in sr.completeness_problems(suite))
            if not isinstance((suite["meta"].get("expected")), dict):
                problems.append("%s: suite S: the record does not say which expected files and tests it was merged against" % where)
        if entry.get("totals") != suite["totals"]:
            problems.append("%s: suite %s: the totals of the baseline are not those of its record" % (where, name))
    return problems


# ---- ncu segment records ---------------------------------------------------------------------------------------------------------------------
NCU_SHAPES = (1, 128)
NCU_REQUIRED_CASES = ((49, "all"),)          # (row position, cache control): the launch that every device and shape must have (a regular row of the deepest sub-block, cold caches)
NCU_STALL_SHARE_SUM = 0.01                   # the shares of the segments add up to 100 % within this many percentage points


def ncu_problems(rec, heads=PROFILE_HEADS, expect_commit=None, min_samples=nr.MIN_SAMPLES):
    """Problems of one ``cfbi-ncu-segments-v1`` record: identity, the case and the collection that it describes, the hash of the report and of the page, and a summary whose segments
    add up to its totals and that holds enough stall samples to say anything about a segment."""
    where = "ncu record %s" % (rec.get("_path", "?") if isinstance(rec, dict) else "?")
    if not isinstance(rec, dict) or rec.get("schema") != nr.SCHEMA:
        return ["%s: not a %s record" % (where, nr.SCHEMA)]
    meta, summary = rec.get("meta"), rec.get("summary")
    if not isinstance(meta, dict) or not isinstance(summary, dict):
        return ["%s: metadata and summary are required" % where]
    problems = br.identity_problems(meta, where, expect_commit, True)
    if not text(meta.get("ncu_version")):
        problems.append("%s: ncu_version must be a non-empty string" % where)
    case = meta.get("case") if isinstance(meta.get("case"), dict) else {}
    if case.get("kernel") != "rows_kernel":
        problems.append("%s: case.kernel is %r, expected rows_kernel" % (where, case.get("kernel")))
    if not count(case.get("N"), 1):
        problems.append("%s: case.N must be a positive integer" % where)
    if case.get("H") != heads or type(case.get("H")) is bool:
        problems.append("%s: case.H is %r, expected %d" % (where, case.get("H"), heads))
    if not (count(case.get("position")) and case["position"] < rpr.POSITIONS):
        problems.append("%s: case.position must be a row position 0 .. %d" % (where, rpr.POSITIONS - 1))
    elif case.get("launch_skip") != case["position"]:
        problems.append("%s: the %r-th launch of the kernel is the row at position %r, but the record says launch_skip %r" % (where, case["position"], case["position"], case.get("launch_skip")))
    if case.get("cache_control") not in ("all", "none"):
        problems.append("%s: case.cache_control must be all (ncu flushes the caches) or none (warm)" % where)
    col = meta.get("collection") if isinstance(meta.get("collection"), dict) else {}
    if col.get("set") != "full" or col.get("clock_control") != "boost":
        problems.append("%s: the collection must be --set full at --clock-control boost (got %r, %r)" % (where, col.get("set"), col.get("clock_control")))
    if not (isinstance(col.get("warp_sampling_interval"), str) and col["warp_sampling_interval"].isdigit() and int(col["warp_sampling_interval"]) <= 31) or not text(col.get("command")):
        problems.append("%s: the collection must name its warp sampling interval (0 .. 31) and its command line" % where)
    for key, suffix in (("report", ".ncu-rep"), ("source_csv", ".csv")):
        entry = meta.get(key) if isinstance(meta.get(key), dict) else {}
        if not (text(entry.get("name")) and entry["name"].endswith(suffix) and isinstance(entry.get("sha256"), str) and re.match(r"^[0-9a-f]{64}$", entry["sha256"]) and count(entry.get("bytes"), 1)):
            problems.append("%s: %s needs a %s name, a sha256 and a size in bytes" % (where, key, suffix))
    launch = meta.get("launch") if isinstance(meta.get("launch"), dict) else {}
    if count(case.get("N"), 1) and launch.get("grid_size") != case["N"] * heads:
        problems.append("%s: the launch has %r CTAs, a decode of %r sequences with %d heads has %d" % (where, launch.get("grid_size"), case.get("N"), heads, case["N"] * heads))
    segs = summary.get("segments")
    if not (isinstance(segs, list) and segs and count(summary.get("static"), 1) and num(summary.get("executed")) and num(summary.get("samples"))):
        return problems + ["%s: the summary needs totals and a list of segments" % where]
    bad = False
    for g in segs:
        if not (isinstance(g, dict) and count(g.get("seg")) and count(g.get("static")) and all(num(g.get(f)) and g[f] >= 0 for f in ("executed", "samples", "exec_pct", "samples_pct"))
                and isinstance(g.get("ops"), dict) and all(count(g["ops"].get(o)) for o in nr.OPS) and isinstance(g.get("top_stalls"), list)):
            problems.append("%s: segment %r is malformed" % (where, g.get("seg") if isinstance(g, dict) else g))
            bad = True
            continue
        shares = [x[1] if isinstance(x, list) and len(x) == 2 and text(x[0]) and num(x[1]) else None for x in g["top_stalls"]]
        if None in shares or len(shares) > 3 or shares != sorted(shares, reverse=True) or any(v < 0 or v > 100 + 1e-9 for v in shares) or (g["samples"] > 0 and not shares):
            problems.append("%s: segment %d: top_stalls must be up to three [reason, percent] pairs, largest first (empty only without samples)" % (where, g["seg"]))
    if bad:
        return problems
    if [g["seg"] for g in segs] != list(range(len(segs))):
        problems.append("%s: the segments are not numbered 0 .. %d in order" % (where, len(segs) - 1))
    if sum(g["static"] for g in segs) != summary["static"]:
        problems.append("%s: the segments hold %d static instructions, the summary says %d" % (where, sum(g["static"] for g in segs), summary["static"]))
    for field in ("executed", "samples"):
        if not close(sum(g[field] for g in segs), summary[field], ARITHMETIC_TOLERANCE):
            problems.append("%s: the segments hold %.0f %s, the summary says %.0f" % (where, sum(g[field] for g in segs), field, summary[field]))
    for field, total in (("exec_pct", summary["executed"]), ("samples_pct", summary["samples"])):
        if total > 0 and abs(sum(g[field] for g in segs) - 100.0) > NCU_STALL_SHARE_SUM:
            problems.append("%s: the %s of the segments add up to %.3f, not 100" % (where, field, sum(g[field] for g in segs)))
    if summary["samples"] < min_samples:
        problems.append("%s: %.0f stall samples in the launch, at least %d are needed for per-segment statistics (collect with --warp-sampling-interval 0)" % (where, summary["samples"], min_samples))
    return problems


def ncu_set_problems(records, devices=PROFILE_DEVICES, shapes=NCU_SHAPES, heads=PROFILE_HEADS, required_cases=NCU_REQUIRED_CASES, expect_commit=None, min_samples=nr.MIN_SAMPLES):
    """The ncu records of an evidence set: every record valid, and for every device and every shape a record of each required case (row position, cache control); one commit."""
    problems = []
    for rec in records:
        problems.extend(ncu_problems(rec, heads, expect_commit, min_samples))
    for dev in devices:
        for n in shapes:
            for pos, cache in required_cases:
                have = [r for r in records if isinstance(r, dict) and isinstance(r.get("meta"), dict) and dev.lower() in str(r["meta"].get("gpu", "")).lower()
                        and isinstance(r["meta"].get("case"), dict) and r["meta"]["case"].get("N") == n and r["meta"]["case"].get("position") == pos
                        and r["meta"]["case"].get("cache_control") == cache]
                if not have:
                    problems.append("no ncu record of %s for N=%d at position %d with cache control %s" % (dev, n, pos, cache))
    commits = sorted(set(str((r.get("meta") or {}).get("commit")) for r in records if isinstance(r, dict)))
    if len(commits) > 1 and not all(commits[0].startswith(c) or c.startswith(commits[0]) for c in commits[1:]):
        problems.append("the ncu records name different commits: %s" % ", ".join(commits))
    return problems


# ---- the step ledger, the decision record ----------------------------------------------------------------------------------------------------
LEDGER_SCHEMA = "cfbi-step-ledger-v1"
STEP_IDS = ("single_launch", "trims", "e_tile", "a_fragments", "kd_tile")
ROW_ENGINE_ORDER = ("trims", "e_tile", "a_fragments")        # the default order of the row-engine steps; the decision record may leave a step out, never move one
SKIPPABLE = ROW_ENGINE_ORDER
UNIT = ("e_tile", "a_fragments")                              # adopted or reverted together by the three-arm rule
OUTCOMES = ("adopted", "not_adopted", "skipped")
GATE_PHASES = ("compile", "exactness", "codegen", "timing")   # the order of the step gate protocol; a step that failed ends at one of them
TOOLCHAINS = ("conda", "container")
DEVICES = ("b200", "b300")
RESOURCE_FIGURES = ("reg", "spill_loads", "spill_stores", "stack")
TIMING_RESULTS = ("pass", "fail", "not_run")
CHAIN_RESULTS = ("pass", "fail", "not_applicable", "not_run")


def unit_outcomes(unit):
    """The outcomes ``(e_tile, a_fragments)`` that the three-arm rule prescribes, or ``(None, problem)``.

    ``unit``: ``{"e_vs_t": pass|fail, "f_vs_e": pass|fail|not_run, "f_vs_t": pass|fail|not_run}`` (T: the previous accepted state, E: T plus the E tile, F: E plus the A fragments).
    E passes against T: F is adopted on top of E only if F beats E, otherwise E alone. E fails against T: both are adopted if F beats T, otherwise neither."""
    if not isinstance(unit, dict):
        return None, "the adoption unit is missing"
    e_t, f_e, f_t = unit.get("e_vs_t"), unit.get("f_vs_e"), unit.get("f_vs_t")
    if e_t == "pass":
        if f_e == "pass":
            return ("adopted", "adopted"), None
        if f_e == "fail":
            return ("adopted", "not_adopted"), None
        return None, "E passed against T, so F must have been compared with E (f_vs_e pass or fail), got %r" % (f_e,)
    if e_t == "fail":
        if f_t == "pass":
            return ("adopted", "adopted"), None
        if f_t == "fail":
            return ("not_adopted", "not_adopted"), None
        return None, "E failed against T, so F must have been compared with T (f_vs_t pass or fail), got %r" % (f_t,)
    return None, "e_vs_t must be pass or fail, got %r" % (e_t,)


def _figure_problems(step, where, reached_codegen, reached_timing, both_required):
    problems = []
    for key, fields in (("codegen_gate", None), ("resource_gate", RESOURCE_FIGURES)):
        entry = step.get(key)
        if not reached_codegen:
            if entry not in (None, {}):
                problems.append("%s: %s has figures for a phase the step never reached" % (where, key))
            continue
        if not isinstance(entry, dict) or any(t not in TOOLCHAINS for t in entry):
            problems.append("%s: %s must give figures per toolchain (%s)" % (where, key, ", ".join(TOOLCHAINS)))
            continue
        needed = TOOLCHAINS if both_required else [t for t in TOOLCHAINS if t in entry][:1]
        if not needed:
            problems.append("%s: %s has no figures for any toolchain" % (where, key))
        for t in needed:
            fig = entry.get(t)
            if not isinstance(fig, dict):
                problems.append("%s: %s has no figures for the %s toolchain" % (where, key, t))
            elif fields is None:
                if not (text(fig.get("verdict")) and count(fig.get("compared"), 1) and count(fig.get("differences"))):
                    problems.append("%s: codegen_gate (%s) needs a verdict, the number of kernels compared and of differences" % (where, t))
                elif both_required and step.get("outcome") == "adopted" and fig["verdict"] != "PASS":
                    problems.append("%s: codegen_gate (%s) is %s: an adopted step passes the gate" % (where, t, fig["verdict"]))
            elif any(not count(fig.get(f)) for f in fields):
                problems.append("%s: resource_gate (%s) needs %s as non-negative integers" % (where, t, ", ".join(fields)))
    ab = step.get("ab")
    if not reached_timing:
        if ab not in (None, {}):
            problems.append("%s: ab has medians for a phase the step never reached" % where)
    elif not isinstance(ab, dict) or sorted(ab) != sorted(DEVICES):
        problems.append("%s: ab must give the A/B medians for %s" % (where, ", ".join(DEVICES)))
    else:
        for dev in DEVICES:
            med = ab[dev].get("medians") if isinstance(ab[dev], dict) else None
            if not (isinstance(med, dict) and med and all(isinstance(v, dict) and num(v.get("base")) and num(v.get("cand")) and v["base"] > 0 and v["cand"] > 0 for v in med.values())):
                problems.append("%s: ab (%s) needs the base and candidate median of every cell (positive numbers)" % (where, dev))
            if not isinstance(ab[dev].get("noise_limited"), list):
                problems.append("%s: ab (%s) must list its noise-limited cells (possibly none)" % (where, dev))
    return problems


def step_problems(step, unit_member=False):
    """One row of the ledger. The evidence required depends on the outcome and on the phases of the gate protocol that the step reached."""
    sid = step.get("id") if isinstance(step, dict) else None
    where = "step %s" % sid
    if not isinstance(step, dict) or sid not in STEP_IDS:
        return ["%s: the id must be one of %s" % (where, ", ".join(STEP_IDS))]
    outcome = step.get("outcome")
    if outcome not in OUTCOMES:
        return ["%s: outcome %r is none of %s" % (where, outcome, ", ".join(OUTCOMES))]
    problems = []
    if outcome == "skipped" and sid not in SKIPPABLE:
        problems.append("%s: only a row-engine step can be skipped by the decision record" % where)
    for field in ("commit", "revert_commit"):
        if step.get(field) is not None and not hexcommit(step[field]):
            problems.append("%s: %s must be a hexadecimal commit hash of at least 7 characters or null" % (where, field))
    markers = step.get("markers")
    if not (isinstance(markers, list) and markers and all(isinstance(m, str) for m in markers)):
        problems.append("%s: markers must be a non-empty list of identifiers, or [\"none\"] for a step that adds no symbol" % where)
    gates, timing = step.get("gates") or {}, step.get("timing") or {}
    if outcome == "skipped":
        if step.get("commit") is not None or step.get("revert_commit") is not None:
            problems.append("%s: a skipped step has no commit" % where)
        if not text(step.get("reason")):
            problems.append("%s: a skipped step carries the reason of the decision record" % where)
        for field in ("codegen_gate", "resource_gate", "ab", "timing", "gates"):
            if step.get(field) not in (None, {}):
                problems.append("%s: a skipped step has no %s" % (where, field))
        return problems
    if outcome == "adopted":
        if step.get("commit") is None:
            problems.append("%s: an adopted step names its commit" % where)
        if step.get("revert_commit") is not None:
            problems.append("%s: an adopted step has no revert commit" % where)
        if step.get("failed_at") is not None:
            problems.append("%s: an adopted step failed at nothing (failed_at %r)" % (where, step.get("failed_at")))
        if gates.get("exactness") is not True or gates.get("codegen") is not True:
            problems.append("%s: an adopted step passes the exactness and the code-generation gate (gates %r)" % (where, gates))
        problems.extend(_figure_problems(step, where, True, True, True))
        if not unit_member:
            if timing.get("vs_previous") != "pass" or timing.get("vs_base") != "pass" or timing.get("chain") not in ("pass", "not_applicable"):
                problems.append("%s: an adopted step beats the previous accepted state, does not regress against the base and keeps the layer chain (timing %r)" % (where, timing))
        return problems
    # not adopted: the evidence stops at the phase where it failed
    failed_at = step.get("failed_at")
    if failed_at not in GATE_PHASES:
        return problems + ["%s: a step that was not adopted says where it failed (failed_at is one of %s), got %r" % (where, ", ".join(GATE_PHASES), failed_at)]
    if not text(step.get("reason")):
        problems.append("%s: a step that was not adopted carries a reason" % where)
    if step.get("commit") is not None and step.get("revert_commit") is None:
        problems.append("%s: not adopted but its commit %s is not reverted (no revert_commit)" % (where, step["commit"]))
    reached = GATE_PHASES.index(failed_at)
    expected_gates = {"exactness": (None if reached < 1 else (False if reached == 1 else True)), "codegen": (None if reached < 2 else (False if reached == 2 else True))}
    for g in ("exactness", "codegen"):
        if gates.get(g) is not expected_gates[g]:
            problems.append("%s: failed at %s, so the %s gate is %r, got %r" % (where, failed_at, g, expected_gates[g], gates.get(g)))
    problems.extend(_figure_problems(step, where, reached >= 2, reached == 3, reached == 3))
    if reached < 3:
        if timing not in ({}, None) and any(v not in ("not_run", None) for v in timing.values()):
            problems.append("%s: failed at %s, so no timing was measured (timing %r)" % (where, failed_at, timing))
    else:
        results = [timing.get("vs_previous"), timing.get("vs_base"), timing.get("chain")]
        if timing.get("vs_previous") not in TIMING_RESULTS or timing.get("vs_base") not in TIMING_RESULTS or timing.get("chain") not in CHAIN_RESULTS:
            problems.append("%s: timing needs vs_previous and vs_base (pass, fail, not_run) and chain (pass, fail, not_applicable, not_run), got %r" % (where, timing))
        elif "fail" not in results:
            problems.append("%s: failed at timing, but no timing result is a failure (%r): a step that passes every gate is adopted" % (where, timing))
    return problems


def decision_problems(rec):
    """A decision record: ``row_engine_order`` (a subsequence of the default order: a step may only be left out), ``value_split_inputs`` and ``budget_table`` (non-empty), and a
    reason for every step that was left out."""
    where = "decision record %s" % (rec.get("_path", "?") if isinstance(rec, dict) else "?")
    if not isinstance(rec, dict):
        return ["%s: not a record" % where]
    problems = ["%s: the key %s is missing or empty" % (where, k) for k in ("row_engine_order", "value_split_inputs", "budget_table") if not rec.get(k)]
    order = rec.get("row_engine_order")
    if isinstance(order, list) and order:
        if not all(isinstance(s, str) for s in order) or len(set(order)) != len(order) or any(s not in ROW_ENGINE_ORDER for s in order):
            problems.append("%s: row_engine_order %r must name steps of %s, each once" % (where, order, list(ROW_ENGINE_ORDER)))
        else:
            default = [s for s in ROW_ENGINE_ORDER if s in order]
            if order != default:
                problems.append("%s: row_engine_order %r reorders the steps: the order is %s, a step may only be left out" % (where, order, list(ROW_ENGINE_ORDER)))
            skipped = rec.get("skipped") if isinstance(rec.get("skipped"), dict) else {}
            for s in ROW_ENGINE_ORDER:
                if s not in order and not text(skipped.get(s)):
                    problems.append("%s: the step %s is left out of row_engine_order without a reason (skipped.%s)" % (where, s, s))
    elif "row_engine_order" in rec and not isinstance(order, list):
        problems.append("%s: row_engine_order must be a list" % where)
    return problems


def ledger_problems(ledger, repo=None, decision=None, require_all_steps=False, head="HEAD"):
    """Problems of a ``cfbi-step-ledger-v1`` record: every row (``step_problems``), the three-arm table for the adoption unit, the decision record, and the commits against a repository."""
    where = "ledger %s" % (ledger.get("_path", "?") if isinstance(ledger, dict) else "?")
    if not isinstance(ledger, dict) or ledger.get("schema") != LEDGER_SCHEMA:
        return ["%s: not a %s record" % (where, LEDGER_SCHEMA)]
    steps = ledger.get("steps")
    if not isinstance(steps, list):
        return ["%s: no list of steps" % where]
    ids = [s.get("id") if isinstance(s, dict) else None for s in steps]
    problems = ["%s: the step %s appears %d times" % (where, i, ids.count(i)) for i in sorted(set(ids), key=str) if ids.count(i) > 1]
    if require_all_steps:
        problems.extend("%s: there is no row for the step %s" % (where, s) for s in STEP_IDS if s not in ids)
    by_id = dict((s["id"], s) for s in steps if isinstance(s, dict) and s.get("id") in STEP_IDS)
    unit = ledger.get("unit")
    in_unit = isinstance(unit, dict) and all(u in by_id and by_id[u].get("outcome") != "skipped" for u in UNIT)
    if unit is not None and not in_unit:
        problems.append("%s: the adoption unit names rows that are missing or skipped" % where)
    elif unit is None and require_all_steps and all(u in by_id and by_id[u].get("outcome") != "skipped" for u in UNIT):
        problems.append("%s: the E tile and the A fragments are one adoption unit: the ledger needs its three arms (unit)" % where)
    elif in_unit:
        wanted, why = unit_outcomes(unit)
        if wanted is None:
            problems.append("%s: %s" % (where, why))
        else:
            for u, w in zip(UNIT, wanted):
                if by_id[u].get("outcome") != w:
                    problems.append("%s: the three-arm rule prescribes %s for %s (E vs T %s, F vs E %s, F vs T %s), the ledger says %s" % (
                        where, w, u, unit.get("e_vs_t"), unit.get("f_vs_e"), unit.get("f_vs_t"), by_id[u].get("outcome")))
            e_prev = unit.get("e_vs_t")
            f_prev = unit.get("f_vs_e") if unit.get("e_vs_t") == "pass" else unit.get("f_vs_t")
            for u, prev in zip(UNIT, (e_prev, f_prev)):
                reached_timing = by_id[u].get("outcome") == "adopted" or by_id[u].get("failed_at") == "timing"      # an earlier failure has no timing to compare
                if reached_timing and (by_id[u].get("timing") or {}).get("vs_previous") != prev:
                    problems.append("%s: the row of %s says %r against the previous accepted state, the arms say %r" % (where, u, (by_id[u].get("timing") or {}).get("vs_previous"), prev))
    for s in steps:
        problems.extend("%s: %s" % (where, p) for p in step_problems(s, unit_member=in_unit and isinstance(s, dict) and s.get("id") in UNIT))
    if decision is not None:
        problems.extend(decision_problems(decision))
        order = decision.get("row_engine_order") if isinstance(decision, dict) else None
        if isinstance(order, list):
            for sid in ROW_ENGINE_ORDER:
                row = by_id.get(sid)
                if row is None:
                    continue
                if sid not in order and row.get("outcome") != "skipped":
                    problems.append("%s: the decision record leaves %s out, the ledger says %s" % (where, sid, row.get("outcome")))
                if sid in order and row.get("outcome") == "skipped":
                    problems.append("%s: %s is skipped in the ledger, the decision record keeps it" % (where, sid))
    if repo is not None:
        head_full = resolve(repo, head)
        for s in steps:
            if not isinstance(s, dict) or s.get("outcome") == "skipped":
                continue
            c = s.get("commit")
            if c is not None and hexcommit(c):
                full = resolve(repo, c)
                if full is None:
                    problems.append("%s: step %s names commit %s, which the repository does not have" % (where, s.get("id"), c))
                elif s.get("outcome") == "adopted" and head_full and not (full == head_full or is_ancestor(repo, full, head_full)):
                    problems.append("%s: adopted step %s: commit %s is not an ancestor of %s" % (where, s.get("id"), c, head))
    return problems


# ---- hand-off records ------------------------------------------------------------------------------------------------------------------------
HANDOFF_SCHEMA = "cfbi-handoff-v1"
BRANCH_PATTERN = re.compile(r"^explore/[A-Za-z0-9._-]+/([A-Za-z0-9._-]+)$")
KNOWN_LIMIT_HEADING = "Not done / known limits"
KNOWN_LIMIT_TOPICS = ("programmatic dependent launch", "paired-column loads", "duplicate slots", "arrival residue", "register relief", "plan-emitted sequence order",
                      "chunk-end helpers", "decode-specialised plan kernel", "Hopper calibration", "CPU emulation harness")


def _branch_problems(h, where, topic, repo):
    problems = []
    m = BRANCH_PATTERN.match(h.get("branch") or "")
    if not m or m.group(1) != topic:
        problems.append("%s: the branch %r is not of the form explore/<run>/%s" % (where, h.get("branch"), topic))
    if not hexcommit(h.get("commit")):
        problems.append("%s: the commit %r is not a hexadecimal hash of at least 7 characters" % (where, h.get("commit")))
    if not problems and repo is not None:
        tip, full = resolve(repo, "refs/heads/%s" % h["branch"]), resolve(repo, h["commit"])
        if tip is None:
            problems.append("%s: the repository has no branch %s" % (where, h["branch"]))
        if full is None:
            problems.append("%s: the repository has no commit %s" % (where, h["commit"]))
        elif tip is not None and not (full == tip or is_ancestor(repo, full, tip)):
            problems.append("%s: the commit %s is not on the branch %s" % (where, h["commit"], h["branch"]))
    return problems


def known_limit_section(document_text):
    """The text of the section of the performance document that is headed ``Not done / known limits`` (None when there is none)."""
    lines = document_text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^(#+)\s*(.*)$", line)
        if m and KNOWN_LIMIT_HEADING.lower() in m.group(2).lower():
            level, body = len(m.group(1)), []
            for nxt in lines[i + 1:]:
                n = re.match(r"^(#+)\s", nxt)
                if n and len(n.group(1)) <= level:
                    break
                body.append(nxt)
            return "\n".join(body)
    return None


def handoff_problems(rec, repo=None, single_launch_adopted=None):
    """Problems of a ``cfbi-handoff-v1`` record, each verifiable without doing the work it hands off. ``single_launch_adopted`` (from the ledger) overrides what the record says."""
    where = "handoff record %s" % (rec.get("_path", "?") if isinstance(rec, dict) else "?")
    if not isinstance(rec, dict) or rec.get("schema") != HANDOFF_SCHEMA:
        return ["%s: not a %s record" % (where, HANDOFF_SCHEMA)]
    handoffs = rec.get("handoffs")
    by_id = dict((h.get("id"), h) for h in handoffs if isinstance(h, dict)) if isinstance(handoffs, list) else {}
    problems = ["%s: there is no hand-off %s" % (where, k) for k in ("value_split", "operand_swap", "known_limits") if k not in by_id]
    adopted = rec.get("single_launch_adopted") if single_launch_adopted is None else single_launch_adopted
    if type(adopted) is not bool:
        problems.append("%s: single_launch_adopted must say whether single-launch decode was adopted" % where)
    h = by_id.get("value_split")
    if h is not None:
        w = "%s, value_split" % where
        problems.extend(_branch_problems(h, w, "value-split-decode-ctas", repo))
        if not (isinstance(h.get("evidence"), list) and h["evidence"] and all(text(e) for e in h["evidence"])):
            problems.append("%s: the go / no-go evidence must be named (a non-empty list of records or sections)" % w)
        if adopted is True and not (text(h.get("arrival_count_requirement")) and "slice" in h["arrival_count_requirement"].lower()):
            problems.append("%s: with single-launch decode adopted the requirement that its arrival count be generalised to H times the number of slices must be stated" % w)
    h = by_id.get("operand_swap")
    if h is not None:
        w = "%s, operand_swap" % where
        problems.extend(_branch_problems(h, w, "operand-swapped-mma-tiles", repo))
        if type(adopted) is bool and h.get("next_free_bit") != (128 if adopted else 64):
            problems.append("%s: the next free implementation bit is %r, it is %d when single-launch decode %s" % (w, h.get("next_free_bit"), 128 if adopted else 64, "took bit 64" if adopted else "did not"))
        probe = h.get("probe_recipe")
        if not (isinstance(probe, dict) and text(probe.get("test")) and sorted(probe.get("devices") or []) == ["B200", "B300", "H100"]):
            problems.append("%s: the probe recipe needs the test and the devices B200, B300 and H100" % w)
        elif repo is not None and hexcommit(h.get("commit")) and resolve(repo, h["commit"]) and git(repo, "cat-file", "-e", "%s:%s" % (h["commit"], probe["test"]))[0] != 0:
            problems.append("%s: the probe test %s is not in commit %s" % (w, probe["test"], h["commit"]))
        doc = h.get("bit_document")
        if repo is not None and text(doc):
            try:
                with open(os.path.join(str(repo), doc), encoding="utf-8") as f:
                    body = f.read()
            except OSError:
                body = None
            if body is None:
                problems.append("%s: the document %s that reserves the bit cannot be read" % (w, doc))
            elif not re.search(r"(?i)\bbit\s+{b}\b[^.\n]*\breserved\b|\breserved\b[^.\n]*\bbit\s+{b}\b".format(b=re.escape(str(h.get("next_free_bit")))), body):
                problems.append("%s: the document %s does not document the bit %r as reserved" % (w, doc, h.get("next_free_bit")))
        elif not text(doc):
            problems.append("%s: bit_document must name the document that reserves the bit" % w)
    h = by_id.get("known_limits")
    if h is not None:
        w = "%s, known_limits" % where
        topics = h.get("topics")
        if not (isinstance(topics, list) and topics and all(t in KNOWN_LIMIT_TOPICS for t in topics)):
            problems.append("%s: topics must be a non-empty list of %s" % (w, ", ".join(KNOWN_LIMIT_TOPICS)))
        if not text(h.get("document")):
            problems.append("%s: the document that holds the section must be named" % w)
        elif repo is not None:
            try:
                with open(os.path.join(str(repo), h["document"]), encoding="utf-8") as f:
                    section = known_limit_section(f.read())
            except OSError:
                section = None
            if section is None:
                problems.append("%s: the document %s has no section %r" % (w, h["document"], KNOWN_LIMIT_HEADING))
            else:
                problems.extend("%s: the section does not name the topic %r" % (w, t) for t in (topics if isinstance(topics, list) else []) if t in KNOWN_LIMIT_TOPICS and t.lower() not in section.lower())
                if PLAN_VOCABULARY.search(section):
                    problems.append("%s: the section uses plan identifiers (%s)" % (w, PLAN_VOCABULARY.search(section).group(0)))
    return problems


# ---- the final record ------------------------------------------------------------------------------------------------------------------------
FINAL_SCHEMA = "cfbi-final-v1"


def final_problems(rec, document_text=None):
    """Problems of a ``cfbi-final-v1`` record (the final A/B of every device against the base)."""
    where = "final record %s" % (rec.get("_path", "?") if isinstance(rec, dict) else "?")
    if not isinstance(rec, dict) or rec.get("schema") != FINAL_SCHEMA:
        return ["%s: not a %s record" % (where, FINAL_SCHEMA)]
    problems = br.identity_problems(rec.get("meta") or {}, where, None, True, ("commit", "dirty"))
    ab = rec.get("final_ab")
    if not isinstance(ab, dict) or sorted(ab) != sorted(DEVICES):
        return problems + ["%s: final_ab must give the final A/B for %s" % (where, ", ".join(DEVICES))]
    limits = rec.get("known_limits") if isinstance(rec.get("known_limits"), list) else []
    section = known_limit_section(document_text) if document_text is not None else None
    for dev in DEVICES:
        cells = ab[dev].get("cells") if isinstance(ab[dev], dict) else None
        if not (isinstance(cells, dict) and cells):
            problems.append("%s: final_ab (%s) has no cells" % (where, dev))
            continue
        for key, c in sorted(cells.items()):
            w = "%s: %s, %s" % (where, dev, key)
            if not isinstance(c, dict) or c.get("verdict") not in ("PASS", "FAIL") or not num(c.get("median_delta_pct")) or not (num(c.get("tolerance_pct")) and c["tolerance_pct"] > 0):
                problems.append("%s: a verdict (PASS or FAIL), the median difference and the tolerance are required" % w)
                continue
            limit = c["tolerance_pct"] if not c.get("noise_limited") else c.get("effective_tolerance_pct")
            if c.get("noise_limited") and not (num(limit) and limit >= c["tolerance_pct"]):
                problems.append("%s: a noise-limited cell gives the widened effective_tolerance_pct" % w)
                continue
            slower = c["median_delta_pct"] > limit
            if (c["verdict"] == "FAIL") != slower:
                problems.append("%s: the verdict %s does not agree with the median difference %.3f %% against the tolerance %.3f %%" % (w, c["verdict"], c["median_delta_pct"], limit))
            if c["verdict"] == "FAIL":
                if c.get("confirmed") is not True:
                    problems.append("%s: a regression that was not confirmed (second allocation) is neither accepted nor dismissed" % w)
                    continue
                entry = next((k for k in limits if isinstance(k, dict) and k.get("device") == dev and k.get("cell") == key), None)
                if entry is None:
                    problems.append("%s: a confirmed regression has no known-limit entry" % w)
                    continue
                rev = entry.get("all_changes_reverted")
                if not (isinstance(rev, dict) and rev.get("verdict") == "FAIL" and num(rev.get("median_delta_pct")) and rev["median_delta_pct"] > c["tolerance_pct"]):
                    problems.append("%s: a known limit needs the measurement with all changes reverted, and the same shift in it (not caused by the changes)" % w)
                if document_text is not None and (section is None or key not in section):
                    problems.append("%s: the performance document does not name the cell in its known-limit section" % w)
    return problems


# ---- command line ----------------------------------------------------------------------------------------------------------------------------
def _report(problems):
    for p in problems:
        print("PROBLEM: %s" % p)
    print("VERDICT: %s" % ("PASS" if not problems else "FAIL"))
    return 0 if not problems else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("profile")
    p.add_argument("--record", nargs="+", required=True)
    p.add_argument("--device", nargs="*", default=None)
    p.add_argument("--shapes", nargs="+", type=int, default=list(PROFILE_SHAPES))
    p.add_argument("--heads", type=int, default=PROFILE_HEADS)
    p.add_argument("--tolerance", type=float, default=rpr.SUM_TOLERANCE)
    p.add_argument("--expect-commit", default=None)
    p = sub.add_parser("ncu")
    p.add_argument("--record", nargs="+", required=True)
    p.add_argument("--device", nargs="*", default=None)
    p.add_argument("--shapes", nargs="+", type=int, default=list(NCU_SHAPES))
    p.add_argument("--heads", type=int, default=PROFILE_HEADS)
    p.add_argument("--expect-commit", default=None)
    p.add_argument("--min-samples", type=int, default=nr.MIN_SAMPLES)
    p = sub.add_parser("suite")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--allow-failures", default=None)
    p = sub.add_parser("baseline")
    p.add_argument("--record", required=True)
    p.add_argument("--allow-missing-raw", action="store_true")
    p = sub.add_parser("commit")
    p.add_argument("--record-commit", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--head", default="HEAD")
    p.add_argument("--require-clean", action="store_true")
    p = sub.add_parser("markers")
    p.add_argument("--ledger", required=True)
    p.add_argument("--tree", required=True)
    p.add_argument("--repo", default=None)
    p = sub.add_parser("ledger")
    p.add_argument("--ledger", required=True)
    p.add_argument("--repo", default=None)
    p.add_argument("--decision", default=None)
    p.add_argument("--require-all-steps", action="store_true")
    p = sub.add_parser("decision")
    p.add_argument("--record", required=True)
    p = sub.add_parser("handoff")
    p.add_argument("--record", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--ledger", default=None)
    p = sub.add_parser("final")
    p.add_argument("--record", required=True)
    p.add_argument("--document", default=None)
    a = ap.parse_args(argv)
    if not a.cmd:
        ap.print_usage(sys.stderr)
        return 2
    try:
        if a.cmd == "profile":
            recs = [load_json(f) for f in a.record]
            if a.device is not None:
                return _report(profile_set_problems(recs, tuple(a.device), tuple(a.shapes), a.heads, a.tolerance, a.expect_commit))
            return _report([q for r in recs for q in profile_problems(r, tuple(a.shapes), a.heads, a.tolerance, a.expect_commit)])
        if a.cmd == "ncu":
            recs = [load_json(f) for f in a.record]
            if a.device is not None:
                return _report(ncu_set_problems(recs, tuple(a.device), tuple(a.shapes), a.heads, NCU_REQUIRED_CASES, a.expect_commit, a.min_samples))
            return _report([q for r in recs for q in ncu_problems(r, a.heads, a.expect_commit, a.min_samples)])
        if a.cmd == "suite":
            allowed = []
            if a.allow_failures:
                with open(a.allow_failures, encoding="utf-8") as f:
                    allowed = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
            return _report(suite_problems(load_json(a.baseline), load_json(a.candidate), allowed))
        if a.cmd == "baseline":
            return _report(baseline_problems(load_json(a.record), a.allow_missing_raw))
        if a.cmd == "commit":
            problems = record_commit_problems(a.record_commit, a.repo, a.head)
            return _report(problems + (clean_tree_problems(a.repo) if a.require_clean else []))
        if a.cmd == "markers":
            return _report(marker_problems(load_json(a.ledger), a.tree, a.repo))
        if a.cmd == "ledger":
            return _report(ledger_problems(load_json(a.ledger), a.repo, load_json(a.decision) if a.decision else None, a.require_all_steps))
        if a.cmd == "decision":
            return _report(decision_problems(load_json(a.record)))
        if a.cmd == "handoff":
            adopted = None
            if a.ledger:
                row = next((s for s in load_json(a.ledger).get("steps", []) if isinstance(s, dict) and s.get("id") == "single_launch"), None)
                adopted = None if row is None else row.get("outcome") == "adopted"
            return _report(handoff_problems(load_json(a.record), a.repo, adopted))
        if a.cmd == "final":
            doc = None
            if a.document:
                with open(a.document, encoding="utf-8") as f:
                    doc = f.read()
            return _report(final_problems(load_json(a.record), doc))
    except (OSError, ValueError) as e:
        print("unusable input: %s" % e, file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
