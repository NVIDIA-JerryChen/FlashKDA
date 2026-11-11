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
  record_checker.py decision --record FILE [--evidence MANIFEST --repo DIR [--document FILE] [--head REV]]
  record_checker.py handoff  --record FILE --repo DIR [--ledger FILE]
  record_checker.py final    --record FILE --repo DIR [--head REV] [--document FILE]

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
         carries a reason). The code-generation and resource evidence is not a summary that the row writes about itself: for the conda toolchain and for the production container
         toolchain on each device the row names the base dump and the candidate dump, and the checker runs ``codegen_gate.compare_dumps`` on them and recomputes the verdict, the kernels
         compared, the differences, the instantiations of the row engine and their registers, stack and spill; the dumps must be of the base commit of the ledger and of the commit of the
         row, of the toolchain and the architectures of the comparison, and the ledger's figures must be the dumps'. An adopted row passes every comparison and all of timing (beats the
         previous accepted state, does not regress against the base, keeps the layer chain where the step changes the shared-memory requirement of the row kernel: ``e_tile``,
         ``a_fragments``, ``kd_tile``, the type of the step decides); the E tile and the A fragments are one adoption unit decided on three arms (T the previous accepted state, E, F)
         by a table (E passes against T and F beats E: both adopted; E passes and F does not beat E: E only; E fails against T and F beats T: both; neither: none): the table only changes
         against which state the gain is judged, so every adopted row passes everything itself, except the E tile that F rescued, which the selected state E + F stands in for. A ledger
         that contradicts the table is rejected. ``--decision`` cross-checks the skipped steps with the decision record.
decision a decision record has the keys ``row_engine_order`` (the steps of the row engine in their default order, a step may only be left out, never moved), ``value_split_inputs`` and ``budget_table``.
         With ``--evidence`` (the manifest of the raw evidence) the record is derived again from the evidence and must equal the given one, and the section of the performance document
         (``--document``) must be the generated one (see ``decision_record.py``).
handoff  ``cfbi-handoff-v1``: the value-split and the operand-swap hand-offs (branch of the form ``explore/<run>/<topic>``, commit, both existing in the repository, the probe test on the branch,
         the next free implementation bit: 128 when single-launch decode took 64, else 64) and the known-limit section of the performance document (topics named without plan identifiers).
final    ``cfbi-final-v1``: the final A/B of the final candidate against the base on every device. The record commit rule (the commit differs from HEAD only under ``docs/``) and a clean tree
         are applied against ``--repo`` (required). Per device the consolidated baseline must declare the benchmark's own coverage (the decode matrix and probe cells of
         ``bench_decode.cell_specs``, the layer-chain cells of ``bench_layer_chain.cell_specs``, the five decode and three chain metrics, tolerance 1 %) on the device's GPU. Each campaign
         (the decode matrix and the layer chain) lists its raw records as ``records_first`` and ``records_rerun``, each ``{A: [...], B: [...]}`` (side A the base tree, side B the final
         tree), plus the allocation and the comparison report; the verdicts are RECOMPUTED from the raw records with the committed comparator under the campaign driver's parameters
         (regress, 1 %, three interleaved repeats, widening, a clean candidate) and the report must agree row by row (a claim, never a source). The raw files are of the commits their
         side must be, of the device's GPU and the baseline's, from one host, and hold every cell of the benchmark (a re-run file some of them); the judged cells are the benchmark's.
         A regression (a FAIL row of the recomputation) is a known limit only with a confirmation campaign (the flagged cells and no others, three times the calls of the first
         campaign, never re-run, on another node and allocation, and the cell fails again) and a campaign with every change reverted (recomputed too: the same shift is there, so the
         changes do not cause it), and the performance document (``--document``) must name the cell in its known-limit section.
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
import bench_layer_chain as bl  # noqa: E402
import bench_records as br  # noqa: E402
import codegen_gate as cg  # noqa: E402
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


def compare_kwargs(metric, tolerance, base_commit, cells):
    """The parameters of the campaign driver's comparison, which every recomputation of a verdict from raw records uses (the producer's: the regress gate, three interleaved repeats,
    the widening of noise-limited cells, a clean candidate, the cell filter of the metric)."""
    return dict(metric=metric, tol_pct=tolerance, mode="regress", expect_base_commit=base_commit, require_clean_cand=True, min_repeats=3, baseline_unstable="widen", cells=cells)


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
        kw = compare_kwargs(m, c["tolerance_pct"], commit, metric_cells(c.get("bench"), m))
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
    if not (isinstance(meta.get("ncu_version"), str) and nr.NCU_VERSION_VALUE.match(meta["ncu_version"])):
        problems.append("%s: ncu_version %r is not a Nsight Compute version (YYYY.N.N or YYYY.N.N.N): the version of another tool in the log must not stand for it" % (where, meta.get("ncu_version")))
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
ROW_ENGINE_STEPS = ROW_ENGINE_ORDER + ("kd_tile",)             # the steps whose commits come after the decision record
UNIT = ("e_tile", "a_fragments")                              # adopted or reverted together by the three-arm rule
OUTCOMES = ("adopted", "not_adopted", "skipped")
GATE_PHASES = ("compile", "exactness", "codegen", "timing")   # the order of the step gate protocol; a step that failed ends at one of them
# the code-generation evidence of a row is one comparison of a base dump and a candidate dump for each of these: the conda toolchain of the login node (the three architectures) and the
# production container toolchain on each device (one architecture each)
COMPARISONS = ("conda", "container_b200", "container_b300")
TOOLCHAINS = COMPARISONS
COMPARISON_ARCHS = {"conda": ("sm_100a", "sm_103a", "sm_90a"), "container_b200": ("sm_100a",), "container_b300": ("sm_103a",)}
CHAIN_STEPS = ("e_tile", "a_fragments", "kd_tile")            # the steps that change the shared-memory requirement of the row kernel (the E tile unit, the chunk-end KD tile): the layer chain is judged too
DEVICES = ("b200", "b300")
RESOURCE_FIGURES = ("reg", "spill_loads", "spill_stores", "stack")
MEASURED = ("pass", "fail")                                   # a timing result of a step that reached the timing phase
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


def unit_waivers(unit):
    """What the three-arm rule changes in the checks of the two rows of the adoption unit, and nothing else: the gain is judged against another state than the previous accepted one.
    When E fails against T and F rescues it (both adopted), E is not selected on its own: its own gain against the previous state may be a failure and it need not pass the cumulative
    and chain checks by itself, because the selected state is E plus F and F passes all of them. Every other adopted row (also E when it passed against T) passes everything itself."""
    outcomes, _ = unit_outcomes(unit)
    if outcomes == ("adopted", "adopted") and isinstance(unit, dict) and unit.get("e_vs_t") == "fail":
        return {"e_tile": "rescued"}
    return {}


def toolchain_class(label):
    low = str(label).lower()
    return "container" if "container" in low else ("conda" if "conda" in low else None)


def commit_matches(a, b):
    return isinstance(a, str) and isinstance(b, str) and bool(a) and bool(b) and (a.startswith(b) or b.startswith(a))


def _first_line(exc):
    """The first two lines of an exception's message (the gate's ``Unusable`` names its reason on the second)."""
    return " ".join(x.strip() for x in str(exc).split("\n")[:2])


def _load_dump(path, what):
    try:
        return cg.load_dump(path), None
    except (cg.Unusable, OSError, ValueError) as e:
        return None, "%s %s cannot be used: %s" % (what, path, _first_line(e))


def comparison_problems(row, key, entry, resources, meta, where):
    """One code-generation comparison of a row bound to its dumps, and what the comparison says. Returns ``(problems, report)``; ``report`` is the result of
    ``codegen_gate.compare_dumps`` (None when the dumps cannot be used). The ledger declares a verdict, the kernels compared, the differences, the instantiations of the row engine and the
    resource figures; every one of them is recomputed from the dumps (the base dump is the base of the loop, the candidate dump the commit of the row, both of the toolchain and the
    architectures that ``key`` names) and a declared figure that is not the recomputed one is a problem. The recomputed Gate R (registers, base-relative stack and spill, launch bounds)
    is the comparison's own result, never the ledger's."""
    w = "%s: codegen_gate (%s)" % (where, key)
    if not isinstance(entry, dict):
        return ["%s is not an object" % w], None
    problems = ["%s needs %s (the path of a dump)" % (w, f) for f in ("base_dump", "cand_dump") if not text(entry.get(f))]
    if entry.get("mode") != "change":
        problems.append("%s: the mode of a step's comparison is change (the row engine differs from the base), got %r" % (w, entry.get("mode")))
    if problems:
        return problems, None
    base, e1 = _load_dump(entry["base_dump"], "the base dump")
    cand, e2 = _load_dump(entry["cand_dump"], "the candidate dump")
    prev, e3, allow = None, None, None
    if entry.get("prev_dump"):
        prev, e3 = _load_dump(entry["prev_dump"], "the previous accepted dump")
    if entry.get("allow"):
        try:
            allow = cg.load_allow(entry["allow"])
        except (cg.Unusable, OSError, ValueError) as e:
            e3 = e3 or "the allow-list %s cannot be used: %s" % (entry["allow"], _first_line(e))
    errors = [e for e in (e1, e2, e3) if e]
    if errors:
        return ["%s: %s" % (w, e) for e in errors], None
    cclass = toolchain_class(cand["meta"]["toolchain"].get("label"))
    if cclass != key.split("_")[0]:
        problems.append("%s: the candidate dump was made with the toolchain %r, which is not the %s toolchain" % (w, cand["meta"]["toolchain"].get("label"), key.split("_")[0]))
    if sorted(cand["meta"]["toolchain"].get("archs") or []) != sorted(COMPARISON_ARCHS[key]):
        problems.append("%s: the candidate dump is for %s, the comparison is for %s" % (w, sorted(cand["meta"]["toolchain"].get("archs") or []), sorted(COMPARISON_ARCHS[key])))
    base_commit = (meta or {}).get("base")
    if not hexcommit(base_commit):
        problems.append("%s: the ledger names no base commit (meta.base) that the base dump could be bound to" % w)
    elif not commit_matches(base["meta"]["tree"].get("commit"), base_commit):
        problems.append("%s: the base dump is of commit %s, the ledger's base is %s" % (w, base["meta"]["tree"].get("commit"), base_commit))
    if row.get("commit") is not None and not commit_matches(cand["meta"]["tree"].get("commit"), row["commit"]):
        problems.append("%s: the candidate dump is of commit %s, the row names commit %s" % (w, cand["meta"]["tree"].get("commit"), row["commit"]))
    try:
        report = cg.compare_dumps(base, cand, allow, "change", prev)
    except cg.Unusable as e:
        return problems + ["%s: the dumps cannot be compared: %s" % (w, _first_line(e))], None
    passed = report["exit"] == 0
    if entry.get("verdict") != ("PASS" if passed else "FAIL"):
        problems.append("%s: the ledger says %r, the comparison of the dumps says %s (%d difference(s), %d resource failure(s))" % (
            w, entry.get("verdict"), "PASS" if passed else "FAIL", len(report["differences"]), len(report["gate_r"]["failures"])))
    for field, got in (("compared", report["compared"]), ("differences", len(report["differences"]))):
        if entry.get(field) != got or type(entry.get(field)) is bool:
            problems.append("%s: the ledger says %s = %r, the comparison of the dumps says %r" % (w, field, entry.get(field), got))
    owned = dict((arch, sorted(k for k in cand["kernels"].get(arch, {}) if cg.family(k) in cg.DEFAULT_OWNED)) for arch in COMPARISON_ARCHS[key])
    declared = entry.get("instantiations")
    if not (isinstance(declared, list) and all(isinstance(x, str) for x in declared)) or any(sorted(declared) != found for found in owned.values()):
        problems.append("%s: the ledger names the instantiations %r of the row engine, the dump has %s" % (w, declared, "; ".join("%s %s" % (a, v) for a, v in sorted(owned.items()))))
    figures = resources.get(key) if isinstance(resources, dict) else None
    if not isinstance(figures, dict) or sorted(figures) != sorted(COMPARISON_ARCHS[key]):
        problems.append("%s: resource_gate (%s) must give the resource figures of the row engine for %s" % (where, key, ", ".join(sorted(COMPARISON_ARCHS[key]))))
    else:
        for arch in COMPARISON_ARCHS[key]:
            for inst in owned[arch]:
                got = dict((f, cand["kernels"][arch][inst].get(f)) for f in RESOURCE_FIGURES)
                say = figures[arch].get(inst) if isinstance(figures[arch], dict) else None
                if not isinstance(say, dict) or any(say.get(f) != got[f] or type(say.get(f)) is bool for f in RESOURCE_FIGURES):
                    problems.append("%s: resource_gate (%s, %s, %s) says %r, the dump has %r" % (where, key, arch, inst, say, got))
    return problems, report


def _codegen_problems(step, where, reached_codegen, both_required, meta):
    """The code-generation and resource evidence of a row: none before the phase was reached, a bound comparison for every toolchain of the gate when the step passed it (or went on to
    the timing), at least one when it failed there; and the recomputed result decides: an adopted row passes every comparison, a row that failed at the code generation fails at least one.
    Returns ``(problems, results)`` with the recomputed pass / fail of the comparisons that could be made."""
    entries, resources = step.get("codegen_gate"), step.get("resource_gate")
    if not reached_codegen:
        return ["%s: %s has figures for a phase the step never reached" % (where, key) for key, v in (("codegen_gate", entries), ("resource_gate", resources)) if v not in (None, {})], []
    if not isinstance(entries, dict) or any(k not in COMPARISONS for k in entries):
        return ["%s: codegen_gate must give one comparison of dumps per toolchain (%s)" % (where, ", ".join(COMPARISONS))], []
    if not isinstance(resources, dict) or any(k not in COMPARISONS for k in resources):
        return ["%s: resource_gate must give the resource figures per toolchain (%s)" % (where, ", ".join(COMPARISONS))], []
    needed = list(COMPARISONS) if both_required else [k for k in COMPARISONS if k in entries][:1]
    if not needed:
        return ["%s: codegen_gate has no comparison for any toolchain" % where], []
    problems, results = [], []
    for key in needed:
        if key not in entries:
            problems.append("%s: codegen_gate has no comparison for %s" % (where, key))
            continue
        probs, report = comparison_problems(step, key, entries[key], resources, meta, where)
        problems.extend(probs)
        if report is not None:
            results.append(report["exit"] == 0)
    if both_required and step.get("outcome") == "adopted" and results and not all(results):
        problems.append("%s: the comparison of the dumps fails (code generation or resources): an adopted step passes it on every toolchain" % where)
    return problems, results


def _ab_problems(step, where, reached_timing):
    problems = []
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


def _timing_problems(step, where, adopted, waiver):
    """The timing phase of a row: every result is measured (no not_run once the phase was reached), the layer chain is measured for the steps that change the shared-memory requirement
    and for no other (the type of the step decides, a row cannot call it not applicable), an adopted row passes all three (beats the previous accepted state, no regression against the
    base, keeps the chain) except where the three-arm rule waives them, and a row that failed here has a failing result."""
    timing = step.get("timing")
    if not isinstance(timing, dict):
        return ["%s: a step that reached the timing phase has timing results (vs_previous, vs_base, chain)" % where]
    sid, problems = step.get("id"), []
    chain_expected = "measured" if sid in CHAIN_STEPS else "not_applicable"
    if timing.get("vs_previous") not in MEASURED or timing.get("vs_base") not in MEASURED:
        problems.append("%s: vs_previous and vs_base are measured (pass or fail) once the timing phase was reached, got %r" % (where, timing))
    if chain_expected == "measured" and timing.get("chain") not in MEASURED:
        problems.append("%s: the step changes the shared-memory requirement of the row kernel, so the layer chain is measured (pass or fail), got %r" % (where, timing.get("chain")))
    if chain_expected == "not_applicable" and timing.get("chain") != "not_applicable":
        problems.append("%s: the layer chain is not applicable to this step, got %r" % (where, timing.get("chain")))
    if problems:
        return problems
    results = [timing["vs_previous"], timing["vs_base"], timing["chain"]]
    if adopted:
        needed = {"vs_previous": "beat the previous accepted state", "vs_base": "not regress against the base"}
        if chain_expected == "measured":
            needed["chain"] = "keep the layer chain"
        waived = {"vs_previous", "vs_base", "chain"} if waiver == "rescued" else set()
        if waiver == "rescued" and timing["vs_previous"] != "fail":
            problems.append("%s: waived as the E tile that F rescued, so it says it failed against the previous accepted state, got %r" % (where, timing["vs_previous"]))
        for field, what in sorted(needed.items()):
            if timing[field] != "pass" and field not in waived:
                problems.append("%s: an adopted step must %s (timing %s: %r)" % (where, what, field, timing))
    elif "fail" not in results:
        problems.append("%s: failed at timing, but no timing result is a failure (%r): a step that passes every gate is adopted" % (where, timing))
    return problems


def step_problems(step, waiver=None, meta=None):
    """One row of the ledger. The evidence required depends on the outcome and on the phases of the gate protocol that the step reached; ``waiver`` is what the three-arm rule changes for
    the row (``unit_waivers``), ``meta`` the ledger's metadata (the base commit that the base dumps are bound to)."""
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
    gates = step.get("gates") or {}
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
        probs, _ = _codegen_problems(step, where, True, True, meta)
        problems.extend(probs + _ab_problems(step, where, True) + _timing_problems(step, where, True, waiver))
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
    probs, results = _codegen_problems(step, where, reached >= 2, reached == 3, meta)
    problems.extend(probs + _ab_problems(step, where, reached == 3))
    if failed_at == "codegen" and results and all(results):
        problems.append("%s: failed at the code generation, but every comparison of the dumps passes" % where)
    if reached < 3:
        timing = step.get("timing")
        if timing not in ({}, None) and any(v not in ("not_run", None) for v in (timing.values() if isinstance(timing, dict) else [timing])):
            problems.append("%s: failed at %s, so no timing was measured (timing %r)" % (where, failed_at, timing))
    else:
        problems.extend(_timing_problems(step, where, False, waiver))
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
    waivers = unit_waivers(unit) if in_unit else {}
    meta = ledger.get("meta") if isinstance(ledger.get("meta"), dict) else {}
    for s in steps:
        problems.extend("%s: %s" % (where, p) for p in step_problems(s, waivers.get(s.get("id")) if isinstance(s, dict) else None, meta))
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
        recorded = decision["meta"].get("recorded_on") if isinstance(decision, dict) and isinstance(decision.get("meta"), dict) else None
        recorded_full = resolve(repo, recorded) if hexcommit(recorded) else None
        if hexcommit(recorded) and recorded_full is None:
            problems.append("%s: the decision record was written on commit %s, which the repository does not have" % (where, recorded))
        for s in steps:
            if not isinstance(s, dict) or s.get("outcome") == "skipped":
                continue
            c = s.get("commit")
            if c is not None and hexcommit(c):
                full = resolve(repo, c)
                if full is None:
                    problems.append("%s: step %s names commit %s, which the repository does not have" % (where, s.get("id"), c))
                    continue
                if s.get("outcome") == "adopted" and head_full and not (full == head_full or is_ancestor(repo, full, head_full)):
                    problems.append("%s: adopted step %s: commit %s is not an ancestor of %s" % (where, s.get("id"), c, head))
                if recorded_full and s.get("id") in ROW_ENGINE_STEPS and (full == recorded_full or not is_ancestor(repo, recorded_full, full)):
                    problems.append("%s: step %s: commit %s does not come after the decision record (written on %s): the order of the row-engine steps is recorded before the first of them"
                                    % (where, s.get("id"), c, recorded))
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
CAMPAIGNS = ("decode", "chain")
CAMPAIGN_KEY = {"decode": "decode_matrix", "chain": "layer_chain"}          # where the consolidated baseline of a device lists the cells and metrics of a campaign
TOLERANCE_PCT = 1.0                                                          # the regress gate of Method M
CANONICAL_METRICS = {"decode": tuple(sorted(br.WALL_METRICS + ("total_us",))), "chain": ("cpu_graph_us", "gpu_graph_us", "total_us")}
DEVICE_GPU = {"b200": "B200", "b300": "B300"}                                # what the GPU name of the raw records of a device must contain
CONFIRMATION_SAME_CONFIG = tuple(f for f in br.RERUN_SAME_CONFIG if f != "host")      # a confirmation runs on another node: it shares every other field with the first campaign
ROW_EXACT = ("key", "kind", "metric", "status", "judged_on", "noise_limited", "needs_rerun", "n_base", "n_cand")
ROW_NUMERIC = ("base_median", "cand_median", "median_delta_pct", "delta_pct", "tol_pct_effective", "base_spread_pct", "cand_spread_pct", "cand_judged", "base_reference")
REPORT_SCALARS = ("verdict", "tol_pct", "mode", "unstable_policy")


def canonical_cells(campaign):
    """The cells of a complete campaign, from the benchmark that defines them (never from a record): the decode matrix and probe cells, or the layer-chain shapes."""
    return [s["key"] for s in (bd.cell_specs() if campaign == "decode" else bl.cell_specs())]


def canonical_coverage(campaign):
    """``{metric: [cell keys]}`` that a complete campaign judges: every canonical metric on every cell, the host loop metrics of a decode campaign on the loop regimes only."""
    keys = canonical_cells(campaign)
    out = {}
    for metric in CANONICAL_METRICS[campaign]:
        sub = metric_cells(campaign, metric)
        out[metric] = sorted(k for k in keys if not sub or any(x in k for x in sub))
    return out


META_KIND = {"decode": "decode-wall", "chain": "chain-wall"}                 # the kind of record that the benchmark of a campaign writes
SIDE_NAMES = (("A", "baseline"), ("B", "candidate"))                          # the two sides of an A/B campaign by their role
EFFECTIVE_COUNTS = ("calls_per_trial", "profile_calls")                      # recorded by every cell: the calls of a trial and of its profiled pass (the kernel sum needs both)


def key_fields(key):
    """``{field: text}`` of a cell key such as ``chain;N=1;H=12;Dm=2048;regime=lockstep;``."""
    return dict(part.split("=", 1) for part in key.split(";")[1:] if "=" in part)


def canonical_workloads(campaign):
    """``{cell key: {argument: value}}`` of a campaign, from the benchmark that defines its cells (never from a record): what each cell must measure. A decode cell: N, H, the regime and
    the probe position (None outside the probe regime); a chain cell: N, H, the model width Dm and the regime that its key names (the chain benchmark runs one)."""
    if campaign == "decode":
        return dict((s["key"], {"N": s["N"], "H": s["H"], "regime": s["regime"], "pos": s["pos"]}) for s in bd.cell_specs())
    return dict((s["key"], {"N": s["N"], "H": s["H"], "Dm": s["Dm"], "regime": key_fields(s["key"])["regime"]}) for s in bl.cell_specs())


def _exactly(a, b):
    """Equal and of the same type: ``True`` is not the integer 1 and ``1.0`` is not 1."""
    return a == b and type(a) is type(b)


def campaign_contract_problems(records, campaign, where):
    """What the raw records of a final campaign must be, whatever the comparator is willing to judge (``br.compare`` takes pooled records and records of any workload, on purpose): the
    process-median records of the campaign's benchmark (meta kind, sample statistic ``process_median``), at least ``br.MIN_TRIALS`` trials that each record declares and each cell holds,
    and cells that are of the campaign's kind, measure the workload that their key names (``canonical_workloads``) and record their effective counts as positive integers. The first
    offence of a kind in a record is named, with the number of the others."""
    out = []
    want, kind = canonical_workloads(campaign), META_KIND[campaign]
    for rec in records:
        name = rec.get("_path", "?")
        meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
        stat, trials = br.sample_stat_of(rec), meta.get("trials")
        if stat != br.PROCESS_MEDIAN:
            out.append("%s: the raw record %s has sample_stat %r, not %r: a final campaign is judged on the per-process medians of Method M (pooled records are for the A/A tools)" % (where, name, stat, br.PROCESS_MEDIAN))
        trials_ok = count(trials, br.MIN_TRIALS)
        if not trials_ok:
            out.append("%s: the raw record %s declares %r trials per cell: Method M needs at least %d trials" % (where, name, trials, br.MIN_TRIALS))
        if meta.get("kind") != kind:
            out.append("%s: the raw record %s is not a %s record (meta kind %r)" % (where, name, kind, meta.get("kind")))
        found = {}
        for cell in rec.get("cells", []):
            key, args = cell.get("key"), cell.get("args") or {}
            held = len(cell.get("samples") or [])
            if trials_ok and held != trials:
                found.setdefault("held", []).append("declares %d trials per cell, the cell %r holds %d samples" % (trials, key, held))
            if cell.get("kind") != campaign:
                found.setdefault("kind", []).append("the cell %r is not a %s cell (kind %r)" % (key, campaign, cell.get("kind")))
            if key not in want:
                found.setdefault("key", []).append("the cell %r is not a cell of the %s benchmark" % (key, campaign))
            else:
                wrong = sorted(f for f, v in want[key].items() if not _exactly(args.get(f), v))
                if wrong:
                    found.setdefault("workload", []).append("the cell %r does not measure the workload its key names (%s: record %s, benchmark %s)"
                                                            % (key, ", ".join(wrong), [args.get(f) for f in wrong], [want[key][f] for f in wrong]))
            for field in EFFECTIVE_COUNTS:
                if not count(args.get(field), 1):
                    found.setdefault("count " + field, []).append("the cell %r has %s = %r: an effective count must be a positive integer" % (key, field, args.get(field)))
        for details in found.values():
            more = " (and %d more cell(s) of the record)" % (len(details) - 1) if len(details) > 1 else ""
            out.append("%s: the raw record %s: %s%s" % (where, name, details[0], more))
    return out


def _load_report(path):
    """``(results, problem)`` of a comparison report (``compare_bench.py --json`` writes a list with one result per metric)."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        return None, "cannot be read (%s)" % e
    if not isinstance(data, list) or not data or any(not isinstance(r, dict) or not isinstance(r.get("rows"), list) for r in data):
        return None, "is not a comparison report (a list with one result, with its rows, per metric)"
    return data, None


def _listing(entry, name, where, problems, allow_empty):
    """``{A: [paths], B: [paths]}`` of a record listing of a campaign entry (``allow_empty``: both lists may be empty, a re-run that did not take place)."""
    listing = entry.get(name)
    if listing is None and allow_empty:
        return {"A": [], "B": []}
    if not isinstance(listing, dict) or not all(isinstance(listing.get(side), list) and all(text(x) for x in listing[side]) for side in ("A", "B")):
        problems.append("%s: %s must be given as {A: [paths], B: [paths]}: which side a raw file belongs to is part of the evidence, it is not guessed" % (where, name))
        return None
    if allow_empty and bool(listing["A"]) != bool(listing["B"]):
        problems.append("%s: %s must hold records for both sides or for neither" % (where, name))
        return None
    if not allow_empty and not (listing["A"] and listing["B"]):
        problems.append("%s: %s must name the raw records of both sides" % (where, name))
        return None
    return listing


def _same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return num(a) and num(b) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    return a == b


def _missing_unexpected(want, got):
    """``missing [...], unexpected [...]``: the first three keys of ``want`` that ``got`` lacks and of ``got`` that ``want`` lacks."""
    want, got = set(want), set(got)
    return "missing %s, unexpected %s" % (sorted(want - got)[:3], sorted(got - want)[:3])


def _cell_keys(rec):
    """The sorted keys of the cells of a raw record."""
    return sorted(c.get("key") for c in rec.get("cells", []) if isinstance(c, dict))


def report_problems(report, result, where):
    """Where a comparison report differs from the comparison recomputed from the raw records: the verdict, the parameters and every row (median, delta, spread, effective
    tolerance, status, the campaign it was judged on, noise-limited, needs a re-run)."""
    out = []
    for field in REPORT_SCALARS:
        if not _same(report.get(field), result.get(field)):
            out.append("%s: the report says %s = %r, the raw records give %r" % (where, field, report.get(field), result.get(field)))
    rows = dict((r.get("key"), r) for r in report["rows"] if isinstance(r, dict))
    want = dict((r["key"], r) for r in result["rows"])
    if len(rows) != len(report["rows"]):
        out.append("%s: the report's rows are not unique cells" % where)
    if set(rows) != set(want):
        out.append("%s: the report has %d cell(s), the raw records judge %d: %s" % (where, len(rows), len(want), _missing_unexpected(want, rows)))
    for key in sorted(set(rows) & set(want)):
        bad = [f for f in ROW_EXACT + ROW_NUMERIC if not _same(rows[key].get(f), want[key].get(f))]
        if bad:
            out.append("%s, %s: the report differs from the raw records in %s (%s)" % (where, key, ", ".join(bad), "; ".join("%s: report %r, raw %r" % (f, rows[key].get(f), want[key].get(f)) for f in bad[:3])))
    return out


def baseline_contract_problems(baseline, dev, campaign, where):
    """What the consolidated baseline of a device must say about a campaign: the benchmark, the tolerance, the metrics and the cells of the benchmark's own definition (the final
    campaigns are judged on that coverage, so a baseline that declares less is not an authority for it), and a GPU that is the device."""
    spec = baseline.get(CAMPAIGN_KEY[campaign])
    if not isinstance(spec, dict):
        return ["%s: the baseline has no %s campaign" % (where, CAMPAIGN_KEY[campaign])]
    out = []
    if spec.get("bench") != campaign:
        out.append("%s: the baseline's %s campaign is of the benchmark %r" % (where, CAMPAIGN_KEY[campaign], spec.get("bench")))
    if spec.get("tolerance_pct") != TOLERANCE_PCT:
        out.append("%s: the baseline's tolerance is %r %%, the regress gate is %s %%" % (where, spec.get("tolerance_pct"), TOLERANCE_PCT))
    if sorted(spec.get("metrics") or []) != sorted(CANONICAL_METRICS[campaign]):
        out.append("%s: the baseline's metrics %s are not the campaign's %s" % (where, sorted(spec.get("metrics") or []), sorted(CANONICAL_METRICS[campaign])))
    want = sorted(canonical_cells(campaign))
    if sorted(spec.get("cells") or []) != want:
        out.append("%s: the baseline's cells are not the benchmark's %d cells (%s)" % (where, len(want), _missing_unexpected(want, spec.get("cells") or [])))
    if DEVICE_GPU[dev] not in str(baseline.get("gpu")):
        out.append("%s: the baseline is of the GPU %r, the device is %s" % (where, baseline.get("gpu"), dev))
    return out


def campaign_evidence_problems(entry, where, campaign, dev, baseline, base, a_commit, b_commit, scope=None, allow_rerun=True):
    """One measured campaign of the final record (the final A/B, a confirmation, the run with every change reverted), recomputed from its raw records.

    ``entry``: ``{allocation, report, records_first: {A, B}, records_rerun: {A, B}}``: the raw record files of the first campaign and of the re-run over the unstable cells, side A (the
    base tree) and side B (the candidate tree, or the base tree again when every change is reverted), and the comparison report (``compare_bench.py --json``), which is a claim.
    Every raw record must be a Method M record of the benchmark's own workload first (``campaign_contract_problems``). Every metric is then compared again with ``br.compare`` under the
    parameters of the campaign driver (regress gate, 1 %, three interleaved repeats, widening of noise-limited cells, a clean candidate) and the report must say what the comparison
    gives, row by row. The sides' commits are ``a_commit`` / ``b_commit``, the GPU of every file the device's (and the
    baseline's), the coverage the benchmark's own: ``scope`` None means the complete campaign (every canonical metric on its cells), else ``{metric: [cells]}``, which the campaign
    must judge exactly (a confirmation or a reverted run covers the flagged cells and no others). ``allow_rerun`` False: the campaign is itself the three-times campaign and is never re-run.
    Returns ``(problems, info)``; ``info`` holds the host, the first-campaign records by side and the recomputed rows by ``(metric, cell)``."""
    info = {"host": None, "rows": {}, "first": None}
    if not isinstance(entry, dict):
        return ["%s: the campaign evidence is missing" % where], info
    problems = []
    if "records" in entry and "records_first" not in entry:
        problems.append("%s: the raw records are listed as records_first and records_rerun, each {A: [...], B: [...]} (a flat list does not say which side a file is)" % where)
    if not text(entry.get("allocation")):
        problems.append("%s: the allocation (the node job that the campaign ran on) must be named" % where)
    first = _listing(entry, "records_first", where, problems, False)
    rerun = _listing(entry, "records_rerun", where, problems, True)
    if first is None or rerun is None:
        return problems, info
    if rerun["A"] and not allow_rerun:
        problems.append("%s: a campaign that is itself the three-times campaign is never re-run (the factor would compound): records_rerun must be empty" % where)
    paths = _paths(first) + _paths(rerun)
    loaded = {}
    for path in paths:
        try:
            loaded[path] = br.load_record(path)
        except (OSError, ValueError) as e:
            problems.append("%s: the raw record %s cannot be read again (%s)" % (where, path, e))
    if len(loaded) != len(set(paths)):
        return problems, info
    sides = {"A": a_commit, "B": b_commit}
    hosts = set()
    for side in ("A", "B"):
        for path in first[side] + rerun[side]:
            meta = loaded[path].get("meta") or {}
            hosts.add(meta.get("host"))
            if not commit_matches(str(meta.get("commit")), sides[side]):
                problems.append("%s: the raw record %s of side %s is of commit %s, side %s must be %s" % (where, path, side, meta.get("commit"), side, sides[side][:10]))
            if DEVICE_GPU[dev] not in str(meta.get("gpu")):
                problems.append("%s: the raw record %s is of the GPU %r, the device is %s" % (where, path, meta.get("gpu"), dev))
            if meta.get("gpu") != baseline.get("gpu") or meta.get("sm_count") != baseline.get("sm_count"):
                problems.append("%s: the raw record %s is of %r with %r SMs, the baseline's device is %r with %r SMs" % (where, path, meta.get("gpu"), meta.get("sm_count"), baseline.get("gpu"), baseline.get("sm_count")))
    if len(hosts) == 1:
        info["host"] = next(iter(hosts))
    else:
        problems.append("%s: the raw records come from %d hosts (%s): a campaign runs on one allocation" % (where, len(hosts), ", ".join(sorted(str(h) for h in hosts))))
    if problems:
        return problems, info
    if scope is None:
        wanted = canonical_coverage(campaign)
        every = sorted(canonical_cells(campaign))
        what = "benchmark's definition"
    else:
        wanted = dict((m, sorted(c)) for m, c in scope.items())
        every = sorted(set(k for cells in wanted.values() for k in cells))
        what = "flagged set"
    for side in ("A", "B"):
        for path in first[side]:                                  # a first-campaign file holds every cell of the campaign
            got = _cell_keys(loaded[path])
            if got != every:
                problems.append("%s: the raw record %s holds %d cell(s), the %s holds %d: %s" % (where, path, len(got), what, len(every), _missing_unexpected(every, got)))
        for path in rerun[side]:                                  # a re-run file holds the cells that were re-run: some of them, and none that the campaign does not have
            got = _cell_keys(loaded[path])
            if not got:
                problems.append("%s: the re-run record %s holds no cell: a re-run file holds the cells that were re-run" % (where, path))
            elif set(got) - set(every):
                problems.append("%s: the re-run record %s holds cell(s) that the %s does not have: %s" % (where, path, what, sorted(set(got) - set(every))[:3]))
    if problems:
        return problems, info
    problems.extend(campaign_contract_problems([loaded[p] for p in paths], campaign, where))        # before the comparator: it judges what it is given, the contract says what is to be given
    if problems:
        return problems, info
    a_first, b_first = [loaded[p] for p in first["A"]], [loaded[p] for p in first["B"]]
    a_rerun, b_rerun = [loaded[p] for p in rerun["A"]], [loaded[p] for p in rerun["B"]]
    info["first"] = {"A": a_first, "B": b_first}
    results, why = _load_report(entry.get("report")) if text(entry.get("report")) else (None, "is not named")
    if results is None:
        problems.append("%s: the comparison report %s" % (where, why))
    elif sorted(str(r.get("metric")) for r in results) != sorted(wanted):
        problems.append("%s: the report has the metrics %s, the campaign judges %s" % (where, sorted(str(r.get("metric")) for r in results), sorted(wanted)))
        results = None
    by_metric = dict((r.get("metric"), r) for r in results) if results else {}
    for metric in sorted(wanted):
        w = "%s, %s" % (where, metric)
        cells = metric_cells(campaign, metric) if scope is None else list(wanted[metric])
        res = br.compare(a_first, b_first, base_rerun=a_rerun or None, cand_rerun=b_rerun or None, **compare_kwargs(metric, TOLERANCE_PCT, base, cells))
        if res["verdict"] not in ("PASS", "FAIL"):
            problems.append("%s: the comparison recomputed from the raw records is %s, not a measurement: %s" % (w, res["verdict"], "; ".join(res["problems"][:3])))
            continue
        rows = [r["key"] for r in res["rows"]]
        if sorted(rows) != sorted(wanted[metric]):
            problems.append("%s: the raw records judge %d cell(s), the %s has %d: %s" % (w, len(rows), what, len(wanted[metric]), _missing_unexpected(wanted[metric], rows)))
        for row in res["rows"]:
            info["rows"][(metric, row["key"])] = row
        if metric in by_metric:
            problems.extend(report_problems(by_metric[metric], res, w))
    return problems, info


def scale_problems(first, confirmation, wanted, where):
    """Why a confirmation is not the three-times campaign of the first campaign for the cells it confirms: for every flagged cell and side the calls per trial (and the profiled calls of a
    kernel-sum metric) are recorded as one positive integer in both and the confirmation's are exactly three times the first campaign's (taken from the first campaign, not from a re-run
    of it: the factor never compounds), under the same configuration but for the node, and every cell that it holds has the arguments of the first campaign's (workload, routing, launch
    mapping) but for the scaled counts: it confirms what was measured, not another shape or another route."""
    out = []
    for metric, keys in sorted(wanted.items()):
        for key in keys:
            for side, name in SIDE_NAMES:
                out.extend("%s: %s" % (where, p) for p in br.rerun_problems(first[side], confirmation[side], name, key, metric, same_config=CONFIRMATION_SAME_CONFIG))
    for side, name in SIDE_NAMES:
        out.extend("%s: %s" % (where, p) for p in br.workload_problems(first[side], confirmation[side], "the final A/B %s" % name, "the confirmation %s" % name, br.SCALED_ARGS))
    return out


DEFAULT_ROUTE = {"impl": 0, "env_impl": None}              # what a record of the final A/B says about the routing of its tree: the default one, no environment override


def default_route_problems(first, where):
    """Why a side of the final A/B did not run its tree at the default routing: the final A/B compares the tree that is delivered with the base as both of them behave by default, so
    neither side may pin another routing (the frozen reference, a forced launch structure) or carry an environment override. A baseline pinned on a slower route would hide a regression,
    and everything that a known limit binds to the baseline (the confirmation, the reverted run) would be consistent with it and still not mean anything."""
    out = []
    for side, name in SIDE_NAMES:
        for rec in first[side]:
            meta = rec.get("meta") or {}
            for field, want in DEFAULT_ROUTE.items():
                if not _exactly(meta.get(field), want):
                    out.append("%s: the %s record %s ran with meta %s = %r: the final A/B measures both trees at their default routing (impl 0, no environment override)"
                               % (where, name, rec.get("_path", "?"), field, meta.get(field)))
            cells = [c.get("key") for c in rec.get("cells", []) if not _exactly((c.get("args") or {}).get("impl"), DEFAULT_ROUTE["impl"])]
            if cells:
                out.append("%s: the %s record %s has %d cell(s) that did not run at the default routing, impl 0 (first: %r)" % (where, name, rec.get("_path", "?"), len(cells), cells[0]))
    return out


def reverted_problems(original, reverted, where):
    """Why the run with all changes reverted does not repeat the baseline of the final A/B on its cells. The base commit alone does not say what was executed (the routing is chosen at run
    time and the frozen reference is the same commit), so both arms must be configured as the baseline side of the final A/B (``CONFIRMATION_SAME_CONFIG``: commit, snapshot, routing,
    toolchain, trials, statistic, device; the node is what may differ) and the cells that they hold must have its arguments (workload, routing, launch mapping) but for the scaled counts."""
    out = []
    what = "the run with all changes reverted is not the baseline behaviour of the final A/B"
    for arm in ("A", "B"):
        out.extend("%s: %s: arm %s differs in meta %s (%s vs %s)" % (where, what, arm, f, a, b) for f, a, b in br.config_differences(original["A"], reverted[arm], CONFIRMATION_SAME_CONFIG))
        out.extend("%s: %s: arm %s: %s" % (where, what, arm, p)
                   for p in br.workload_problems(original["A"], reverted[arm], "the baseline of the final A/B", "the reverted arm %s" % arm, br.SCALED_ARGS))
    return out


def final_problems(rec, repo=None, head="HEAD", document_text=None):
    """Problems of a ``cfbi-final-v1`` record (the final A/B of the final candidate against the base on every device).

    The record names the commit that was measured (``meta.commit``) and, per device, the consolidated baseline and the final A/B campaigns (decode matrix and layer chain) and, for every
    regression that the final A/B shows (a FAIL row of the comparison recomputed from the raw records), a known limit: the confirmation campaign on another allocation and the campaign
    with every change reverted. What is checked: the docs-only record-commit rule and a clean tree against ``repo`` / ``head`` (the commit must be one of the repository and differ from
    HEAD only under ``docs/``); per device the baseline's coverage is the benchmark's own and its GPU the device; every campaign is recomputed from its raw records (see
    ``campaign_evidence_problems``: the sides, the commits, the GPU, the coverage, the comparison itself and the report row by row); a regression needs a known-limit entry whose
    confirmation judges exactly the flagged cells with three times the calls of the first campaign on another node and allocation, on the workload and route of the first campaign (see
    ``scale_problems``), and fails again, whose run with all changes reverted repeats the baseline of the final A/B (``reverted_problems``) and shows the same shift (so the changes do
    not cause it), and the performance document must name the cell in its known-limit section."""
    where = "final record %s" % (rec.get("_path", "?") if isinstance(rec, dict) else "?")
    if not isinstance(rec, dict) or rec.get("schema") != FINAL_SCHEMA:
        return ["%s: not a %s record" % (where, FINAL_SCHEMA)]
    meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
    problems = br.identity_problems(meta, where, None, True, ("commit", "dirty"))
    if not hexcommit(meta.get("base")):
        problems.append("%s: meta.base must be the base commit of the loop" % where)
    if repo is None:
        problems.append("%s: a final record is only accepted against a repository and a head (the record commit rule): repo is required" % where)
    elif hexcommit(meta.get("commit")):
        problems.extend(record_commit_problems(meta["commit"], repo, head, where=where))
        problems.extend(clean_tree_problems(repo))
    devices = rec.get("devices")
    if not isinstance(devices, dict) or sorted(devices) != sorted(DEVICES):
        return problems + ["%s: devices must give the final A/B for %s" % (where, ", ".join(DEVICES))]
    base, final = meta.get("base"), meta.get("commit")
    if not (hexcommit(base) and hexcommit(final)):
        return problems
    regressions, campaign_info, baselines = [], {}, {}
    for dev in DEVICES:
        d = devices[dev] if isinstance(devices[dev], dict) else {}
        try:
            baseline = load_json(d["baseline"])
        except (KeyError, OSError, ValueError) as e:
            problems.append("%s: %s: the consolidated baseline %r cannot be read (%s)" % (where, dev, d.get("baseline"), e))
            continue
        baselines[dev] = baseline
        if not commit_matches(str(baseline.get("commit")), base):
            problems.append("%s: %s: the consolidated baseline is of commit %s, the base of the loop is %s" % (where, dev, baseline.get("commit"), base))
        for campaign in CAMPAIGNS:
            cw = "%s: %s %s" % (where, dev, campaign)
            contract = baseline_contract_problems(baseline, dev, campaign, cw)
            problems.extend(contract)
            if contract:
                continue
            entry = d["campaigns"].get(campaign) if isinstance(d.get("campaigns"), dict) else None
            probs, info = campaign_evidence_problems(entry, cw, campaign, dev, baseline, base, base, final)
            problems.extend(probs)
            if info["first"]:
                problems.extend(default_route_problems(info["first"], cw))
            campaign_info[(dev, campaign)] = (entry, info)
            regressions.extend((dev, campaign, metric, key) for (metric, key), row in sorted(info["rows"].items()) if row.get("status") == "FAIL")
    limits = rec.get("known_limits") if isinstance(rec.get("known_limits"), list) else []
    section = known_limit_section(document_text) if document_text is not None else None
    for dev, campaign, metric, key in regressions:
        w = "%s: %s %s %s %s" % (where, dev, campaign, metric, key)
        entry = next((k for k in limits if isinstance(k, dict) and (k.get("device"), k.get("campaign"), k.get("metric"), k.get("cell")) == (dev, campaign, metric, key)), None)
        if entry is None:
            problems.append("%s: a regression that the final A/B shows has no known-limit entry with its confirmation and its run with all changes reverted" % w)
            continue
        original, info = campaign_info[(dev, campaign)]
        flagged = {}
        for (m, k), row in info["rows"].items():
            if row.get("status") == "FAIL":
                flagged.setdefault(m, []).append(k)
        conf_probs, conf = campaign_evidence_problems(entry.get("confirmation"), "%s, confirmation" % w, campaign, dev, baselines[dev], base, base, final, scope=flagged, allow_rerun=False)
        rev_probs, rev = campaign_evidence_problems(entry.get("all_changes_reverted"), "%s, all changes reverted" % w, campaign, dev, baselines[dev], base, base, base, scope=flagged)
        problems.extend(conf_probs + rev_probs)
        if not conf_probs:
            problems.extend(scale_problems(info["first"], conf["first"], flagged, "%s, confirmation" % w))
        if not rev_probs:
            problems.extend(reverted_problems(info["first"], rev["first"], "%s, all changes reverted" % w))
        if conf_probs or rev_probs:
            continue
        if conf["rows"].get((metric, key), {}).get("status") != "FAIL":
            problems.append("%s: the confirmation campaign does not confirm the regression: the cell does not fail there" % w)
        if conf["host"] is not None and conf["host"] == info["host"]:
            problems.append("%s: the confirmation ran on the same node (%s) as the final A/B: a second allocation is required" % (w, conf["host"]))
        if entry["confirmation"].get("allocation") == (original or {}).get("allocation"):
            problems.append("%s: the confirmation names the allocation of the final A/B: a second allocation is required" % w)
        if rev["rows"].get((metric, key), {}).get("status") != "FAIL":
            problems.append("%s: the same shift is not there with all changes reverted: the regression is caused by the changes, a known limit is not allowed (revert the latest step and re-validate)" % w)
        if document_text is None:
            problems.append("%s: a known limit needs the performance document (give it): the cell must be named in its known-limit section" % w)
        elif section is None or key not in section:
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
    p.add_argument("--evidence", default=None)
    p.add_argument("--repo", default=None)
    p.add_argument("--document", default=None)
    p.add_argument("--head", default="HEAD")
    p = sub.add_parser("handoff")
    p.add_argument("--record", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--ledger", default=None)
    p = sub.add_parser("final")
    p.add_argument("--record", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--head", default="HEAD")
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
            if a.evidence:
                import decision_record as dr
                document = None
                if a.document:
                    with open(a.document, encoding="utf-8") as f:
                        document = f.read()
                return _report(dr.record_problems(load_json(a.record), dr.load_evidence(a.evidence), a.repo, document, a.head))
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
            return _report(final_problems(load_json(a.record), a.repo, a.head, doc))
    except (OSError, ValueError) as e:
        print("unusable input: %s" % e, file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
