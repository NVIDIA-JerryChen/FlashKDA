"""Self-tests of the record checker for the records of the optimisation steps (benchmarks/record_checker.py): the step ledger with its outcome-dependent evidence and the
three-arm adoption table, the decision record, the marker check, the hand-off records, the final record and the consolidated baseline.

Plain Python, no GPU and no torch (``pytest --noconftest tests/cfbi/test_record_checker_steps.py``). Every rule has a valid record and mutants that break it; the marker and the
hand-off checks run against synthetic source trees and real git histories built in a temporary directory.
"""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "benchmarks", _ROOT / "tests" / "cfbi"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import record_checker as rc  # noqa: E402
import test_bench_tools as tbt  # noqa: E402
from test_record_checker import commit_file, git, has, mutated  # noqa: E402

C1, C2, C3, C4, C5 = ("%07x%033x" % (i, i) for i in range(1, 6))        # commit-like hashes (resolved against a repository only where a test builds one)
CELL = "decode;N=1;H=12;regime=lockstep;"


# ---- the ledger ------------------------------------------------------------------------------------------------------------------------------
def codegen_figures(verdict="PASS", toolchains=rc.TOOLCHAINS):
    return dict((t, {"verdict": verdict, "compared": 54, "differences": 0}) for t in toolchains)


def resource_figures(toolchains=rc.TOOLCHAINS):
    return dict((t, {"reg": 128, "spill_loads": 4, "spill_stores": 4, "stack": 112}) for t in toolchains)


def ab_figures(devices=rc.DEVICES):
    return dict((d, {"medians": {CELL: {"base": 14.0, "cand": 13.2}}, "noise_limited": []}) for d in devices)


def adopted(sid, commit=C1, markers=("none",), vs_previous="pass", vs_base="pass", chain="not_applicable"):
    return {"id": sid, "outcome": "adopted", "commit": commit, "revert_commit": None, "markers": list(markers), "failed_at": None, "reason": "",
            "gates": {"exactness": True, "codegen": True}, "timing": {"vs_previous": vs_previous, "vs_base": vs_base, "chain": chain},
            "codegen_gate": codegen_figures(), "resource_gate": resource_figures(), "ab": ab_figures()}


def not_adopted(sid, failed_at, commit=None, revert=None, markers=("none",), timing=None):
    reached = rc.GATE_PHASES.index(failed_at)
    step = {"id": sid, "outcome": "not_adopted", "commit": commit, "revert_commit": revert, "markers": list(markers), "failed_at": failed_at, "reason": "the %s phase failed" % failed_at,
            "gates": {"exactness": None if reached < 1 else reached > 1, "codegen": None if reached < 2 else reached > 2}, "timing": None,
            "codegen_gate": None, "resource_gate": None, "ab": None}
    if reached >= 2:
        step["codegen_gate"] = codegen_figures("FAIL" if failed_at == "codegen" else "PASS", rc.TOOLCHAINS if reached == 3 else ("conda",))
        step["resource_gate"] = resource_figures(rc.TOOLCHAINS if reached == 3 else ("conda",))
    if reached == 3:
        step["ab"] = ab_figures()
        step["timing"] = timing or {"vs_previous": "fail", "vs_base": "pass", "chain": "not_applicable"}
    return step


def skipped(sid, markers=("none",)):
    return {"id": sid, "outcome": "skipped", "commit": None, "revert_commit": None, "markers": list(markers), "reason": "the profile shows no gain to expect", "failed_at": None,
            "gates": None, "timing": None, "codegen_gate": None, "resource_gate": None, "ab": None}


UNIT_CASES = [
    ("E passes against T and F beats E", {"e_vs_t": "pass", "f_vs_e": "pass", "f_vs_t": "not_run"}, ("adopted", "adopted")),
    ("E passes against T and F does not beat E", {"e_vs_t": "pass", "f_vs_e": "fail", "f_vs_t": "not_run"}, ("adopted", "not_adopted")),
    ("E fails against T and F beats T", {"e_vs_t": "fail", "f_vs_e": "not_run", "f_vs_t": "pass"}, ("adopted", "adopted")),
    ("E fails against T and F does not beat T", {"e_vs_t": "fail", "f_vs_e": "not_run", "f_vs_t": "fail"}, ("not_adopted", "not_adopted")),
]


def unit_rows(unit, outcomes, e_marker=("e_tile_marker",), f_marker=("a_fragments_marker",)):
    """The rows of the two members of the adoption unit for the given arms and outcomes."""
    prev_e = unit["e_vs_t"]
    prev_f = unit["f_vs_e"] if unit["e_vs_t"] == "pass" else unit["f_vs_t"]
    rows = []
    for sid, outcome, prev, markers, commit in (("e_tile", outcomes[0], prev_e, e_marker, C2), ("a_fragments", outcomes[1], prev_f, f_marker, C3)):
        if outcome == "adopted":
            rows.append(adopted(sid, commit, markers, vs_previous=prev, chain="pass"))
        else:
            rows.append(not_adopted(sid, "timing", commit, C4, markers, {"vs_previous": prev, "vs_base": "pass", "chain": "pass"}))
    return rows


def ledger(unit_case=0, **overrides):
    _, unit, outcomes = UNIT_CASES[unit_case]
    e, f = unit_rows(unit, outcomes)
    steps = {"single_launch": adopted("single_launch", C1, ("single_launch_marker",)), "trims": adopted("trims", C5), "e_tile": e, "a_fragments": f,
             "kd_tile": not_adopted("kd_tile", "exactness")}
    steps.update(overrides)
    return {"schema": rc.LEDGER_SCHEMA, "meta": {"base": C1, "head": C5}, "steps": [steps[s] for s in rc.STEP_IDS], "unit": dict(unit), "_path": "mem"}


def test_a_complete_ledger_passes_for_every_branch_of_the_three_arm_rule():
    for k, (name, unit, outcomes) in enumerate(UNIT_CASES):
        assert rc.ledger_problems(ledger(k), require_all_steps=True) == [], name


@pytest.mark.parametrize("k", range(len(UNIT_CASES)))
def test_a_ledger_that_contradicts_the_three_arm_rule_is_rejected(k):
    name, unit, outcomes = UNIT_CASES[k]
    for index, sid in enumerate(rc.UNIT):
        flipped = list(outcomes)
        flipped[index] = "not_adopted" if outcomes[index] == "adopted" else "adopted"
        e, f = unit_rows(unit, flipped)
        bad = ledger(k, e_tile=e, a_fragments=f)
        assert has(rc.ledger_problems(bad), "the three-arm rule prescribes %s for %s" % (outcomes[index], sid)), (name, sid)


def test_the_two_counterexamples_of_the_adoption_unit_are_rejected():
    """The A-fragment commit kept although it failed against the E tile (E passed), and the E tile kept although neither arm passed."""
    _, unit, _ = UNIT_CASES[1]
    e, f = unit_rows(unit, ("adopted", "adopted"))
    assert has(rc.ledger_problems(ledger(1, e_tile=e, a_fragments=f)), "prescribes not_adopted for a_fragments")
    _, unit, _ = UNIT_CASES[3]
    e, f = unit_rows(unit, ("adopted", "not_adopted"))
    assert has(rc.ledger_problems(ledger(3, e_tile=e, a_fragments=f)), "prescribes not_adopted for e_tile")


def test_the_arms_of_the_unit_must_be_complete_and_agree_with_the_rows():
    base = ledger(0)
    for edit, expected in ((lambda l: l["unit"].update(e_vs_t="maybe"), "e_vs_t must be pass or fail"),
                           (lambda l: l["unit"].update(f_vs_e="not_run"), "F must have been compared with E"),
                           (lambda l: l["unit"].update(e_vs_t="fail", f_vs_t="not_run"), "F must have been compared with T"),
                           (lambda l: l.pop("unit"), None),
                           (lambda l: l.update(unit="x"), "the adoption unit")):
        problems = rc.ledger_problems(mutated(base, edit))
        if expected:
            assert has(problems, expected), expected
    wrong_row = mutated(base, lambda l: l["steps"][3]["timing"].update(vs_previous="fail"))               # the F row says it did not beat E, the arms say it did
    assert has(rc.ledger_problems(wrong_row), "the row of a_fragments says 'fail' against the previous accepted state")
    # without the unit the rows stand alone: an adopted E tile has to beat the previous state by its own row
    alone = mutated(base, lambda l: l.pop("unit"))
    alone["steps"][2]["timing"]["vs_previous"] = "fail"
    assert has(rc.ledger_problems(alone), "beats the previous accepted state")


ADOPTED_MUTANTS = [
    ("no gates", lambda s: s.update(gates=None), "passes the exactness and the code-generation gate"),
    ("a failed exactness gate", lambda s: s["gates"].update(exactness=False), "passes the exactness"),
    ("a failed code-generation gate", lambda s: s["gates"].update(codegen=False), "passes the exactness"),
    ("no code-generation figures", lambda s: s.update(codegen_gate=None), "codegen_gate must give figures per toolchain"),
    ("code-generation figures for one toolchain only", lambda s: s["codegen_gate"].pop("container"), "has no figures for the container toolchain"),
    ("a code-generation gate that did not pass", lambda s: s["codegen_gate"]["conda"].update(verdict="FAIL"), "an adopted step passes the gate"),
    ("code-generation figures without the kernels compared", lambda s: s["codegen_gate"]["conda"].pop("compared"), "the number of kernels compared"),
    ("no resource figures", lambda s: s.update(resource_gate=None), "resource_gate must give figures per toolchain"),
    ("a resource figure that is missing", lambda s: s["resource_gate"]["container"].pop("spill_stores"), "spill_stores"),
    ("a resource figure that is negative", lambda s: s["resource_gate"]["conda"].update(reg=-1), "non-negative integers"),
    ("no A/B medians", lambda s: s.update(ab=None), "ab must give the A/B medians"),
    ("A/B medians for one device only", lambda s: s["ab"].pop("b300"), "ab must give the A/B medians"),
    ("an A/B median that is zero", lambda s: s["ab"]["b200"]["medians"][CELL].update(cand=0), "positive numbers"),
    ("no list of noise-limited cells", lambda s: s["ab"]["b200"].pop("noise_limited"), "noise-limited"),
    ("no commit", lambda s: s.update(commit=None), "names its commit"),
    ("a commit that is not a hash", lambda s: s.update(commit="HEAD"), "hexadecimal"),
    ("a revert commit", lambda s: s.update(revert_commit=C4), "no revert commit"),
    ("a step that says where it failed", lambda s: s.update(failed_at="timing"), "failed at nothing"),
    ("a step that does not beat the previous state", lambda s: s["timing"].update(vs_previous="fail"), "beats the previous accepted state"),
    ("a step that regresses against the base", lambda s: s["timing"].update(vs_base="fail"), "does not regress against the base"),
    ("a step that breaks the layer chain", lambda s: s["timing"].update(chain="fail"), "keeps the layer chain"),
    ("a timing that was not run", lambda s: s["timing"].update(vs_base="not_run"), "does not regress against the base"),
    ("no markers", lambda s: s.update(markers=[]), "markers must be a non-empty list"),
]


@pytest.mark.parametrize("name,edit,expected", ADOPTED_MUTANTS, ids=[m[0] for m in ADOPTED_MUTANTS])
def test_an_adopted_row_with_missing_or_failing_evidence_is_rejected(name, edit, expected):
    good = adopted("trims")
    assert rc.step_problems(good) == []
    assert has(rc.step_problems(mutated(good, edit)), expected), name


@pytest.mark.parametrize("failed_at", rc.GATE_PHASES)
def test_a_step_that_was_not_adopted_carries_the_evidence_of_the_phases_it_reached(failed_at):
    good = not_adopted("kd_tile", failed_at)
    assert rc.step_problems(good) == []
    assert rc.step_problems(not_adopted("kd_tile", failed_at, C2, C3)) == []                       # committed and reverted
    assert has(rc.step_problems(mutated(good, lambda s: s.update(reason=""))), "carries a reason")
    assert has(rc.step_problems(mutated(good, lambda s: s.update(commit=C2))), "is not reverted")
    assert has(rc.step_problems(mutated(good, lambda s: s.update(failed_at="later"))), "says where it failed")
    assert has(rc.step_problems(mutated(good, lambda s: s["gates"].update(exactness="x"))), "gate is")


def test_the_measurements_a_step_never_reached_are_not_run_and_cannot_be_invented():
    """compile / exactness: no figures at all; codegen: figures but no timing; timing: everything and at least one failing result."""
    compiled = not_adopted("kd_tile", "compile")
    assert has(rc.step_problems(mutated(compiled, lambda s: s.update(codegen_gate=codegen_figures()))), "never reached")
    exactness = not_adopted("kd_tile", "exactness")
    assert has(rc.step_problems(mutated(exactness, lambda s: s.update(resource_gate=resource_figures()))), "never reached")
    assert has(rc.step_problems(mutated(exactness, lambda s: s["gates"].update(exactness=True))), "the exactness gate is False")
    assert has(rc.step_problems(mutated(exactness, lambda s: s.update(timing={"vs_previous": "fail", "vs_base": "pass", "chain": "pass"}))), "no timing was measured")
    codegen = not_adopted("kd_tile", "codegen")
    assert has(rc.step_problems(mutated(codegen, lambda s: s.update(ab=ab_figures()))), "never reached")
    assert has(rc.step_problems(mutated(codegen, lambda s: s.update(codegen_gate=None, resource_gate=None))), "must give figures per toolchain")
    assert has(rc.step_problems(mutated(codegen, lambda s: s["gates"].update(codegen=True))), "the codegen gate is False")
    timing = not_adopted("kd_tile", "timing")
    assert has(rc.step_problems(mutated(timing, lambda s: s.update(ab=None))), "ab must give the A/B medians")
    assert has(rc.step_problems(mutated(timing, lambda s: s["codegen_gate"].pop("container"))), "has no figures for the container toolchain")
    assert has(rc.step_problems(mutated(timing, lambda s: s.update(timing={"vs_previous": "pass", "vs_base": "pass", "chain": "pass"}))), "a step that passes every gate is adopted")
    assert has(rc.step_problems(mutated(timing, lambda s: s.update(timing={"vs_previous": "slow", "vs_base": "pass", "chain": "pass"}))), "timing needs vs_previous")
    for results in ({"vs_previous": "fail", "vs_base": "pass", "chain": "pass"}, {"vs_previous": "pass", "vs_base": "fail", "chain": "pass"}, {"vs_previous": "pass", "vs_base": "pass", "chain": "fail"}):
        assert rc.step_problems(mutated(timing, lambda s: s.update(timing=results))) == [], results


def test_a_skipped_step_has_a_reason_and_no_evidence():
    assert rc.step_problems(skipped("trims")) == []
    assert has(rc.step_problems(mutated(skipped("trims"), lambda s: s.update(reason=""))), "carries the reason")
    assert has(rc.step_problems(mutated(skipped("trims"), lambda s: s.update(commit=C1))), "has no commit")
    assert has(rc.step_problems(mutated(skipped("trims"), lambda s: s.update(ab=ab_figures()))), "has no ab")
    assert has(rc.step_problems(skipped("single_launch")), "only a row-engine step can be skipped")
    assert has(rc.step_problems(skipped("kd_tile")), "only a row-engine step can be skipped")


def test_the_ledger_has_one_valid_row_per_step():
    good = ledger(0)
    assert rc.ledger_problems(good) == []
    assert has(rc.ledger_problems(mutated(good, lambda l: l["steps"].pop(0)), require_all_steps=True), "no row for the step single_launch")
    assert rc.ledger_problems(mutated(good, lambda l: l["steps"].pop(0))) == []                      # an intermediate ledger may have fewer rows
    assert has(rc.ledger_problems(mutated(good, lambda l: l["steps"].append(copy.deepcopy(l["steps"][0])))), "appears 2 times")
    assert has(rc.ledger_problems(mutated(good, lambda l: l["steps"][0].update(id="other"))), "the id must be one of")
    assert has(rc.ledger_problems(mutated(good, lambda l: l["steps"][0].update(outcome="maybe"))), "none of adopted")
    assert has(rc.ledger_problems(mutated(good, lambda l: l.update(schema="x"))), "not a cfbi-step-ledger-v1 record")
    assert has(rc.ledger_problems(mutated(good, lambda l: l.update(steps="x"))), "no list of steps")
    assert has(rc.ledger_problems(mutated(good, lambda l: l["steps"][0].update(commit=None))), "names its commit")
    assert has(rc.ledger_problems(mutated(good, lambda l: l["steps"][1].update(gates={"exactness": True, "codegen": False}))), "passes the exactness")


# ---- the decision record ---------------------------------------------------------------------------------------------------------------------
def decision(order=("trims", "e_tile", "a_fragments"), **kw):
    rec = {"row_engine_order": list(order), "value_split_inputs": {"prefix_share": 0.41, "n_h_range": [12, 1536]}, "budget_table": {"b200": [{"tile": "e", "reg": 128}]}, "_path": "mem"}
    rec.update(kw)
    return rec


def test_the_decision_record_has_its_three_keys_and_an_order_that_only_skips():
    assert rc.decision_problems(decision()) == []
    assert rc.decision_problems(decision(order=("trims", "a_fragments"), skipped={"e_tile": "the shared memory of the tile does not fit"})) == []
    for key in ("row_engine_order", "value_split_inputs", "budget_table"):
        assert has(rc.decision_problems(mutated(decision(), lambda d: d.pop(key))), "the key %s is missing" % key)
        assert has(rc.decision_problems(mutated(decision(), lambda d: d.update({key: []}))), "the key %s is missing" % key)
    assert has(rc.decision_problems(decision(order=("e_tile", "trims", "a_fragments"))), "reorders the steps")
    assert has(rc.decision_problems(decision(order=("a_fragments", "e_tile"), skipped={"trims": "x"})), "reorders the steps")
    assert has(rc.decision_problems(decision(order=("trims", "a_fragments"))), "without a reason")
    assert has(rc.decision_problems(decision(order=("trims", "trims", "e_tile"))), "each once")
    assert has(rc.decision_problems(decision(order=("trims", "kd_tile"))), "must name steps of")
    assert has(rc.decision_problems(decision(row_engine_order="trims")), "must be a list")


def test_the_ledger_and_the_decision_record_agree_on_the_steps_that_were_skipped():
    d = decision(order=("trims", "a_fragments"), skipped={"e_tile": "the tile does not fit"})
    _, unit, _ = UNIT_CASES[0]
    good = {"schema": rc.LEDGER_SCHEMA, "meta": {}, "steps": [adopted("single_launch", C1), adopted("trims", C5), skipped("e_tile"), adopted("a_fragments", C3, vs_previous="pass"),
                                                              not_adopted("kd_tile", "exactness")], "_path": "mem"}
    assert rc.ledger_problems(good, decision=d, require_all_steps=True) == []
    assert has(rc.ledger_problems(good, decision=decision()), "e_tile is skipped in the ledger, the decision record keeps it")
    assert has(rc.ledger_problems(good, decision=decision(order=("trims",), skipped={"e_tile": "x", "a_fragments": "y"})), "leaves a_fragments out, the ledger says adopted")
    assert has(rc.ledger_problems(good, decision=decision(order=("trims", "a_fragments", "e_tile"))), "reorders")


# ---- ledger against a repository: commits and reverts ----------------------------------------------------------------------------------------
def build_steps_repo(repo):
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    h = {"repo": repo}
    h["base"] = commit_file(repo, "csrc/k.cu", "base\n", "base")
    h["step"] = commit_file(repo, "csrc/step.cu", "the step\n", "add the step")
    h["other"] = commit_file(repo, "csrc/o.cu", "other\n", "another change")
    h["revert"] = commit_file(repo, "csrc/step.cu", "base again\n", "Revert \"add the step\"\n\nThis reverts commit %s." % h["step"])
    h["unrelated_revert"] = commit_file(repo, "csrc/o.cu", "other again\n", "Revert \"another change\"\n\nThis reverts commit %s." % h["other"])
    return h


@pytest.fixture(scope="module")
def steps_repo(tmp_path_factory):
    """One history shared by the tests that only read it."""
    return build_steps_repo(tmp_path_factory.mktemp("steps") / "repo")


@pytest.fixture()
def steps_repo_rw(tmp_path):
    return build_steps_repo(tmp_path / "repo")


def test_ledger_commits_are_checked_against_the_repository(steps_repo_rw):
    repo, h = steps_repo_rw["repo"], steps_repo_rw
    row = lambda **kw: adopted("trims", h["base"], **kw)
    ok = {"schema": rc.LEDGER_SCHEMA, "meta": {}, "steps": [row()], "_path": "mem"}
    assert rc.ledger_problems(ok, repo=repo) == []
    assert has(rc.ledger_problems(mutated(ok, lambda l: l["steps"][0].update(commit="0" * 40)), repo=repo), "does not have")
    side = steps_repo_rw["repo"]
    git(side, "checkout", "-q", "-b", "elsewhere", h["base"])
    off = commit_file(side, "csrc/e.cu", "not on main\n", "a commit on another line")
    git(side, "checkout", "-q", "main")
    assert has(rc.ledger_problems(mutated(ok, lambda l: l["steps"][0].update(commit=off)), repo=repo), "is not an ancestor")


# ---- markers ---------------------------------------------------------------------------------------------------------------------------------
def make_tree(root, files):
    for rel, body in files.items():
        path = Path(root) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return root


TREE = {"csrc/k.cu": "__global__ void rows_kernel() { stage_e_tile(); }\n", "flash_kda/api.py": "ROUTE = 'single_launch_route'\n", "tests/test_a.py": "def test_x(): marker_in_tests_only()\n",
        "benchmarks/bench.py": "bench_marker_name = 1\n", "docs/perf.md": "docs_only_marker and stage_e_tile and a_fragments_marker\n",
        "benchmarks/record_checker.py": "checker_marker_name\n", "tests/cfbi/test_record_checker_x.py": "checker_test_marker\n", "csrc/__pycache__/junk.py": "junk_marker\n"}


def marker_ledger(*rows):
    return {"schema": rc.LEDGER_SCHEMA, "meta": {}, "steps": list(rows), "_path": "mem"}


def row(sid, outcome, markers, **kw):
    r = {"id": sid, "outcome": outcome, "markers": list(markers), "commit": None, "revert_commit": None}
    r.update(kw)
    return r


def test_the_markers_of_an_adopted_step_must_be_in_the_source_directories(tmp_path):
    tree = make_tree(tmp_path / "t", TREE)
    check = lambda *rows: rc.marker_problems(marker_ledger(*rows), tree)
    assert check(row("e_tile", "adopted", ["stage_e_tile"]), row("single_launch", "adopted", ["single_launch_route"])) == []              # csrc/ and flash_kda/
    assert has(check(row("e_tile", "adopted", ["marker_in_tests_only"])), "is not in csrc/ or flash_kda/")                                   # tests/ is searched but is no source
    assert has(check(row("e_tile", "adopted", ["bench_marker_name"])), "is not in csrc/ or flash_kda/")
    assert has(check(row("e_tile", "adopted", ["docs_only_marker"])), "found in: nowhere")                                                  # docs/ is never searched
    assert has(check(row("e_tile", "adopted", ["stage_e_tile", "absent_marker"])), "absent_marker")
    assert has(check(row("e_tile", "adopted", ["checker_marker_name"])), "found in: nowhere")                                                # the checker script is not searched
    assert has(check(row("e_tile", "adopted", ["checker_test_marker"])), "found in: nowhere")                                                # nor a test_record_checker* file
    assert has(check(row("e_tile", "adopted", ["junk_marker"])), "found in: nowhere")                                                        # nor a __pycache__
    assert has(check(row("e_tile", "adopted", ["stage_e"])), "found in: nowhere")                                                            # the identifier as a whole word


def test_the_markers_of_a_step_that_was_not_adopted_or_skipped_are_absent_everywhere(tmp_path):
    tree = make_tree(tmp_path / "t", TREE)
    check = lambda *rows: rc.marker_problems(marker_ledger(*rows), tree)
    for outcome in ("not_adopted", "skipped"):
        assert check(row("a_fragments", outcome, ["a_fragments_marker"])) == []                                                              # only docs/ names it
        assert has(check(row("a_fragments", outcome, ["stage_e_tile"])), "is still in the tree: csrc/k.cu")
        assert has(check(row("a_fragments", outcome, ["marker_in_tests_only"])), "is still in the tree: tests/test_a.py")                  # absent in tests/ and benchmarks/ too
        assert has(check(row("a_fragments", outcome, ["bench_marker_name"])), "benchmarks/bench.py")
        assert check(row("a_fragments", outcome, ["checker_test_marker"])) == []


def test_marker_lists_are_validated(tmp_path):
    tree = make_tree(tmp_path / "t", TREE)
    check = lambda r: rc.marker_problems(marker_ledger(r), tree)
    assert has(check(row("e_tile", "adopted", [])), "markers must be a non-empty list")
    assert has(check(dict(row("e_tile", "adopted", []), markers="x")), "markers must be a non-empty list")
    assert has(check(row("e_tile", "adopted", ["none", "stage_e_tile"])), "must be identifiers")
    assert has(check(row("e_tile", "adopted", ["has space"])), "must be identifiers")
    assert has(rc.marker_problems({"steps": "x"}, tree), "no list of steps")
    assert has(check(row("e_tile", "adopted", ["none"])), "can only be checked against a repository")


def test_a_step_without_symbols_is_judged_by_its_commit(steps_repo, tmp_path):
    repo, h = steps_repo["repo"], steps_repo
    tree = make_tree(tmp_path / "t", TREE)
    check = lambda r, head=None: rc.marker_problems(marker_ledger(r), tree, repo, head or "HEAD")
    # adopted: the commit is an ancestor of HEAD and no later revert commit names it
    assert check(row("trims", "adopted", ["none"], commit=h["base"])) == []
    assert has(check(row("trims", "adopted", ["none"], commit=h["step"])), "a later revert commit names")                                   # reverted by h['revert']
    assert check(row("trims", "adopted", ["none"], commit=h["step"]), head=h["other"]) == []                                                 # not yet reverted at that head
    assert has(check(row("trims", "adopted", ["none"], commit=h["other"])), "a later revert commit names")                                  # reverted by h['unrelated_revert']
    assert check(row("trims", "adopted", ["none"], commit=h["revert"])) == []                                                                # a revert commit itself is not reverted
    assert has(check(row("trims", "adopted", ["none"], commit="0" * 40)), "is not a commit of the repository")
    assert has(check(row("trims", "adopted", ["none"], commit=None)), "is not a commit of the repository")
    # not adopted or skipped: no commit, or a revert commit that is an ancestor of HEAD and names it
    for outcome in ("not_adopted", "skipped"):
        assert check(row("trims", outcome, ["none"])) == []
        assert check(row("trims", outcome, ["none"], commit=h["step"], revert_commit=h["revert"])) == []
        assert has(check(row("trims", outcome, ["none"], commit=h["step"])), "has no revert commit")
        assert has(check(row("trims", outcome, ["none"], commit=h["step"], revert_commit=h["unrelated_revert"])), "does not name")
        assert has(check(row("trims", outcome, ["none"], commit=h["step"], revert_commit="0" * 40)), "has no revert commit")
        assert has(check(row("trims", outcome, ["none"], revert_commit=h["revert"])), "a revert commit but no commit")
        assert has(check(row("trims", outcome, ["none"], commit=h["step"], revert_commit=h["revert"]), head=h["other"]), "has no revert commit that is an ancestor")


def test_the_marker_command_line(steps_repo, tmp_path, capsys):
    tree = make_tree(tmp_path / "t", TREE)
    ledger_file = tmp_path / "ledger.json"
    ledger_file.write_text(json.dumps(marker_ledger(row("e_tile", "adopted", ["stage_e_tile"]), row("trims", "adopted", ["none"], commit=steps_repo["base"]))))
    assert rc.main(["markers", "--ledger", str(ledger_file), "--tree", str(tree), "--repo", str(steps_repo["repo"])]) == 0
    assert rc.main(["markers", "--ledger", str(ledger_file), "--tree", str(tree)]) == 1
    assert "can only be checked against a repository" in capsys.readouterr().out
    assert rc.main(["markers", "--ledger", str(tmp_path / "missing.json"), "--tree", str(tree)]) == 2
    assert rc.main(["ledger", "--ledger", str(ledger_file)]) == 1


# ---- hand-off records ------------------------------------------------------------------------------------------------------------------------
KNOWN_LIMITS_DOC = """# Performance of the decode path

## Results
Numbers.

## Not done / known limits
Programmatic dependent launch is not used. The paired-column loads of the M tiles are not done. Duplicate slots are not validated and the arrival residue is not
checked. The register relief of the row engine, the plan-emitted sequence order, the chunk-end helpers, the decode-specialised plan kernel, the Hopper calibration of the
single-launch limit and the merged CPU emulation harness are all open.

## Appendix
Something else.
"""
BIT_DOC = "Implementation bits: 64 single-launch decode; bit 128 is reserved for the next implementation. Bit 256 is free.\n"


def build_handoff_repo(repo):
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    h = {"repo": repo}
    h["main"] = commit_file(repo, "docs/perf.md", KNOWN_LIMITS_DOC, "the performance document")
    commit_file(repo, "flash_kda/cfbi/api.py", "# " + BIT_DOC, "document the bits")
    git(repo, "checkout", "-q", "-b", "explore/run1/value-split-decode-ctas", h["main"])
    h["split"] = commit_file(repo, "csrc/split.cu", "split\n", "value split prototype")
    git(repo, "checkout", "-q", "-b", "explore/run1/operand-swapped-mma-tiles", h["main"])
    h["swap"] = commit_file(repo, "tests/cfbi/test_mma_probe_swap.py", "def test_probe(): pass\n", "operand swap probe")
    git(repo, "checkout", "-q", "main")
    return h


@pytest.fixture(scope="module")
def handoff_repo(tmp_path_factory):
    """One repository shared by the tests that only read it."""
    return build_handoff_repo(tmp_path_factory.mktemp("handoff") / "repo")


@pytest.fixture()
def handoff_repo_rw(tmp_path):
    return build_handoff_repo(tmp_path / "repo")


def handoff_record(h, adopted_single=True):
    return {"schema": rc.HANDOFF_SCHEMA, "single_launch_adopted": adopted_single, "_path": "mem", "handoffs": [
        {"id": "value_split", "branch": "explore/run1/value-split-decode-ctas", "commit": h["split"], "evidence": ["decision record: value_split_inputs"],
         "arrival_count_requirement": "generalise the arrival count to H times the number of slices" if adopted_single else ""},
        {"id": "operand_swap", "branch": "explore/run1/operand-swapped-mma-tiles", "commit": h["swap"], "next_free_bit": 128 if adopted_single else 64, "bit_document": "flash_kda/cfbi/api.py",
         "probe_recipe": {"test": "tests/cfbi/test_mma_probe_swap.py", "devices": ["B200", "B300", "H100"]}},
        {"id": "known_limits", "document": "docs/perf.md", "section": rc.KNOWN_LIMIT_HEADING, "topics": list(rc.KNOWN_LIMIT_TOPICS)}]}


def test_a_complete_handoff_record_passes(handoff_repo):
    h = handoff_repo
    assert rc.handoff_problems(handoff_record(h), h["repo"]) == []
    assert rc.handoff_problems(handoff_record(h), None) == []                                          # without a repository the checks that need one are not made
    assert rc.known_limit_section(KNOWN_LIMITS_DOC).strip().startswith("Programmatic dependent launch")
    assert "Appendix" not in rc.known_limit_section(KNOWN_LIMITS_DOC) and rc.known_limit_section("# x\nno section\n") is None


def test_the_next_free_bit_follows_the_single_launch_outcome(handoff_repo_rw):
    h = handoff_repo_rw
    no_single = handoff_record(h, adopted_single=False)
    commit_file(h["repo"], "flash_kda/cfbi/api.py", "# Implementation bits: bit 64 is reserved. Bit 128 is free.\n", "reserve bit 64")
    assert rc.handoff_problems(no_single, h["repo"]) == []                                              # 64 when single-launch decode did not take it
    assert has(rc.handoff_problems(mutated(no_single, lambda r: r["handoffs"][1].update(next_free_bit=128)), h["repo"]), "it is 64 when single-launch decode did not")
    assert has(rc.handoff_problems(handoff_record(h, True), h["repo"]), "does not document the bit 128 as reserved")
    # the ledger decides what the record says when it is given
    assert has(rc.handoff_problems(no_single, h["repo"], single_launch_adopted=True), "it is 128 when single-launch decode took bit 64")


HANDOFF_MUTANTS = [
    ("no value-split hand-off", lambda r: r["handoffs"].pop(0), "there is no hand-off value_split"),
    ("no operand-swap hand-off", lambda r: r["handoffs"].pop(1), "there is no hand-off operand_swap"),
    ("no known-limits hand-off", lambda r: r["handoffs"].pop(2), "there is no hand-off known_limits"),
    ("no list of hand-offs", lambda r: r.update(handoffs=None), "there is no hand-off"),
    ("a branch of another form", lambda r: r["handoffs"][0].update(branch="explore/value-split"), "is not of the form explore/<run>/value-split-decode-ctas"),
    ("the branch of the other hand-off", lambda r: r["handoffs"][0].update(branch="explore/run1/operand-swapped-mma-tiles"), "is not of the form"),
    ("a branch that does not exist", lambda r: r["handoffs"][0].update(branch="explore/run9/value-split-decode-ctas"), "has no branch"),
    ("no commit", lambda r: r["handoffs"][0].pop("commit"), "is not a hexadecimal hash"),
    ("a commit that is not a hash", lambda r: r["handoffs"][1].update(commit="tip"), "is not a hexadecimal hash"),
    ("a commit the repository does not have", lambda r: r["handoffs"][0].update(commit="0" * 40), "has no commit"),
    ("a commit that is not on the branch", lambda r: r["handoffs"][0].update(commit="@SWAP@"), "is not on the branch"),
    ("a commit that is on the branch but is not its tip", lambda r: r["handoffs"][0].update(commit="@MAIN@"), None),
    ("no evidence", lambda r: r["handoffs"][0].update(evidence=[]), "evidence must be named"),
    ("no arrival-count requirement", lambda r: r["handoffs"][0].update(arrival_count_requirement=""), "H times the number of slices"),
    ("an arrival-count requirement without the slices", lambda r: r["handoffs"][0].update(arrival_count_requirement="keep it"), "H times the number of slices"),
    ("the wrong next free bit", lambda r: r["handoffs"][1].update(next_free_bit=64), "the next free implementation bit is 64"),
    ("a probe on two devices only", lambda r: r["handoffs"][1]["probe_recipe"].update(devices=["B200", "B300"]), "probe recipe needs"),
    ("a probe test that is not in the commit", lambda r: r["handoffs"][1]["probe_recipe"].update(test="tests/cfbi/test_other.py"), "is not in commit"),
    ("a bit document that cannot be read", lambda r: r["handoffs"][1].update(bit_document="docs/none.md"), "cannot be read"),
    ("no bit document", lambda r: r["handoffs"][1].pop("bit_document"), "bit_document must name"),
    ("no topics", lambda r: r["handoffs"][2].update(topics=[]), "topics must be a non-empty list"),
    ("a topic that is not a known limit", lambda r: r["handoffs"][2]["topics"].append("a new idea"), "topics must be a non-empty list"),
    ("no document", lambda r: r["handoffs"][2].pop("document"), "must be named"),
    ("a document without the section", lambda r: r["handoffs"][2].update(document="flash_kda/cfbi/api.py"), "has no section"),
    ("a single-launch flag that is not a boolean", lambda r: r.update(single_launch_adopted="yes"), "single_launch_adopted must say"),
]


@pytest.mark.parametrize("name,edit,expected", HANDOFF_MUTANTS, ids=[m[0] for m in HANDOFF_MUTANTS])
def test_a_broken_handoff_record_is_rejected(handoff_repo, name, edit, expected):
    h = handoff_repo
    rec = handoff_record(h)
    bad = mutated(rec, edit)
    bad = json.loads(json.dumps(bad).replace("@MAIN@", h["main"]).replace("@SWAP@", h["swap"]))
    if expected is None:
        assert rc.handoff_problems(bad, h["repo"]) == [], name                       # the commit of the prototype may be anywhere on its branch
    else:
        assert has(rc.handoff_problems(bad, h["repo"]), expected), (name, rc.handoff_problems(bad, h["repo"])[:3])


def test_the_known_limit_section_names_its_topics_without_plan_identifiers(handoff_repo_rw):
    h = handoff_repo_rw
    for topic, text in (("programmatic dependent launch", "Programmatic dependent launch is not used"), ("Hopper calibration", "the Hopper calibration of")):
        commit_file(h["repo"], "docs/perf.md", KNOWN_LIMITS_DOC.replace(text, "something else"), "drop a topic")
        assert has(rc.handoff_problems(handoff_record(h), h["repo"]), "does not name the topic %r" % topic)
        commit_file(h["repo"], "docs/perf.md", KNOWN_LIMITS_DOC, "restore")
    identifier = "FUT" + "-2"                                                         # built here: the tests themselves must not carry plan vocabulary
    commit_file(h["repo"], "docs/perf.md", KNOWN_LIMITS_DOC.replace("are all open.", "are all open (see %s)." % identifier), "plan identifier")
    assert has(rc.handoff_problems(handoff_record(h), h["repo"]), "plan identifiers (%s)" % identifier)


def test_the_handoff_command_line(handoff_repo, tmp_path):
    h = handoff_repo
    good = tmp_path / "handoff.json"
    good.write_text(json.dumps(handoff_record(h)))
    ledger_file = tmp_path / "ledger.json"
    ledger_file.write_text(json.dumps({"schema": rc.LEDGER_SCHEMA, "steps": [adopted("single_launch")]}))
    assert rc.main(["handoff", "--record", str(good), "--repo", str(h["repo"]), "--ledger", str(ledger_file)]) == 0
    ledger_file.write_text(json.dumps({"schema": rc.LEDGER_SCHEMA, "steps": [not_adopted("single_launch", "timing")]}))
    assert rc.main(["handoff", "--record", str(good), "--repo", str(h["repo"]), "--ledger", str(ledger_file)]) == 1          # the ledger says single-launch was not adopted: the bit is 64


# ---- the final record ------------------------------------------------------------------------------------------------------------------------
FINAL_DOC = "# Performance\n\n## Not done / known limits\nThe cell %s is slower than the base on B200 with all changes reverted too.\n\n## Appendix\nx\n" % CELL


def final_record(**cells):
    def entry(delta, **kw):
        e = {"verdict": "PASS", "median_delta_pct": delta, "tolerance_pct": 1.0, "base_median": 14.0, "cand_median": 14.0 * (1 + delta / 100.0), "noise_limited": False, "confirmed": False}
        e.update(kw)
        return e
    ab = dict((d, {"cells": {CELL: entry(-3.2), "decode;N=64;H=12;regime=lockstep;": entry(0.4)}}) for d in rc.DEVICES)
    for key, e in cells.items():
        ab["b200"]["cells"][key] = e
    return {"schema": rc.FINAL_SCHEMA, "meta": {"commit": C5, "dirty": False}, "final_ab": ab, "known_limits": [], "_path": "mem"}


def confirmed_regression(delta=2.5):
    return {"verdict": "FAIL", "median_delta_pct": delta, "tolerance_pct": 1.0, "base_median": 14.0, "cand_median": 14.35, "noise_limited": False, "confirmed": True}


def test_a_final_record_with_a_verdict_for_every_cell_passes():
    assert rc.final_problems(final_record()) == []


FINAL_MUTANTS = [
    ("a dirty tree", lambda r: r["meta"].update(dirty=True), "dirty"),
    ("a commit that is not a hash", lambda r: r["meta"].update(commit="x"), "hexadecimal"),
    ("one device only", lambda r: r["final_ab"].pop("b300"), "final_ab must give the final A/B"),
    ("a device without cells", lambda r: r["final_ab"]["b300"].update(cells={}), "has no cells"),
    ("a cell without a verdict", lambda r: r["final_ab"]["b200"]["cells"][CELL].pop("verdict"), "a verdict (PASS or FAIL)"),
    ("a verdict that is neither", lambda r: r["final_ab"]["b200"]["cells"][CELL].update(verdict="INVALID"), "a verdict (PASS or FAIL)"),
    ("a cell without a median difference", lambda r: r["final_ab"]["b300"]["cells"][CELL].update(median_delta_pct=None), "the median difference"),
    ("a cell without a tolerance", lambda r: r["final_ab"]["b300"]["cells"][CELL].update(tolerance_pct=0), "the tolerance"),
    ("a PASS for a cell that is slower beyond the tolerance", lambda r: r["final_ab"]["b200"]["cells"][CELL].update(median_delta_pct=2.0), "does not agree"),
    ("a FAIL for a cell that is not slower", lambda r: r["final_ab"]["b200"]["cells"][CELL].update(verdict="FAIL", median_delta_pct=0.2, confirmed=True), "does not agree"),
    ("a noise-limited cell without its widened tolerance", lambda r: r["final_ab"]["b200"]["cells"][CELL].update(noise_limited=True), "effective_tolerance_pct"),
]


@pytest.mark.parametrize("name,edit,expected", FINAL_MUTANTS, ids=[m[0] for m in FINAL_MUTANTS])
def test_a_broken_final_record_is_rejected(name, edit, expected):
    assert has(rc.final_problems(mutated(final_record(), edit)), expected), name


def test_a_noise_limited_cell_is_judged_against_its_widened_tolerance():
    rec = final_record()
    rec["final_ab"]["b200"]["cells"][CELL].update(noise_limited=True, effective_tolerance_pct=2.5, median_delta_pct=2.0)
    assert rc.final_problems(rec) == []
    rec["final_ab"]["b200"]["cells"][CELL].update(median_delta_pct=3.0)
    assert has(rc.final_problems(rec), "does not agree")


def test_a_confirmed_final_regression_is_a_known_limit_only_if_it_survives_reverting_everything():
    key = "decode;N=64;H=12;regime=staggered;"
    rec = final_record(**{key: confirmed_regression()})
    assert has(rc.final_problems(rec, FINAL_DOC), "has no known-limit entry")
    limit = {"device": "b200", "cell": key, "final": {"median_delta_pct": 2.5}, "all_changes_reverted": {"verdict": "FAIL", "median_delta_pct": 2.4}}
    with_limit = mutated(rec, lambda r: r["known_limits"].append(limit))
    doc = FINAL_DOC.replace(CELL, key)
    assert rc.final_problems(with_limit, doc) == []
    assert has(rc.final_problems(with_limit, FINAL_DOC), "does not name the cell")                                  # the performance document must name the cell
    assert has(rc.final_problems(with_limit, "# no section at all\n"), "does not name the cell")
    no_shift = mutated(with_limit, lambda r: r["known_limits"][0].update(all_changes_reverted={"verdict": "PASS", "median_delta_pct": 0.1}))
    assert has(rc.final_problems(no_shift, doc), "the same shift")                                                  # reverting removes it: caused by the changes, not a known limit
    missing = mutated(with_limit, lambda r: r["known_limits"][0].pop("all_changes_reverted"))
    assert has(rc.final_problems(missing, doc), "with all changes reverted")
    other_cell = mutated(with_limit, lambda r: r["known_limits"][0].update(cell=CELL))
    assert has(rc.final_problems(other_cell, doc), "has no known-limit entry")
    other_device = mutated(with_limit, lambda r: r["known_limits"][0].update(device="b300"))
    assert has(rc.final_problems(other_device, doc), "has no known-limit entry")
    unconfirmed = final_record(**{key: dict(confirmed_regression(), confirmed=False)})
    assert has(rc.final_problems(unconfirmed, FINAL_DOC), "was not confirmed")


def test_the_final_command_line(tmp_path):
    good, bad, doc = tmp_path / "final.json", tmp_path / "bad.json", tmp_path / "perf.md"
    key = "decode;N=64;H=12;regime=staggered;"
    limit = {"device": "b200", "cell": key, "all_changes_reverted": {"verdict": "FAIL", "median_delta_pct": 2.4}}
    rec = mutated(final_record(**{key: confirmed_regression()}), lambda r: r["known_limits"].append(limit))
    rec.pop("_path")
    good.write_text(json.dumps(rec))
    bad.write_text(json.dumps(mutated(rec, lambda r: r.update(known_limits=[]))))
    doc.write_text(FINAL_DOC.replace(CELL, key))
    assert rc.main(["final", "--record", str(good), "--document", str(doc)]) == 0
    assert rc.main(["final", "--record", str(bad), "--document", str(doc)]) == 1
    assert rc.main(["final", "--record", str(tmp_path / "missing.json")]) == 2
    assert rc.main(["decision", "--record", str(good)]) == 1


# ---- the consolidated baseline ---------------------------------------------------------------------------------------------------------------
def campaign(d, name, bench="decode", cand_scale=1.0, samples=3):
    """An A/A campaign: ``samples`` interleaved records per side written to ``d``; the entry that a consolidated baseline holds for it."""
    base, cand = tbt.interleaved([10.0] * 5, [10.0 * cand_scale] * 5, base_commit=tbt.BASE_COMMIT, cand_commit=tbt.BASE_COMMIT)
    pa = tbt.write_records(str(d), "%s-A-r" % name, base[:samples])
    pb = tbt.write_records(str(d), "%s-B-r" % name, cand[:samples])
    verdict = "PASS" if cand_scale == 1.0 else "FAIL"
    return {"a": "at-2dcb4f8", "b": "at-2dcb4f8", "bench": bench, "cells": ["c"], "metrics": ["k2_stage_us"], "tolerance_pct": 1.0, "samples_per_side": samples,
            "records_first": {"A": pa, "B": pb}, "records_rerun": {"A": [], "B": []}, "first_campaign_verdicts": {"k2_stage_us": verdict}, "final_verdicts": {"k2_stage_us": verdict},
            "noise_limited_cells": {"k2_stage_us": []}, "rerun_cells": {"k2_stage_us": []}, "cells_needing_rerun": {"k2_stage_us": 0}, "worst_abs_median_delta_pct": {"k2_stage_us": 0.0},
            "host": "node1", "stamp": "x", "report": "none", "tools": {}}


def baseline_record(d):
    d = Path(d)
    suite_s, suite_gate = tbt.merged_suite_record(), tbt.suite_record_of(tbt.junit_of(["test_a", "test_b"]), suite="gate", env={"CFBI_BIG": "1"})
    paths = {}
    for name, rec in (("S", suite_s), ("gate", suite_gate)):
        paths[name] = str(d / ("suite-%s.json" % name))
        Path(paths[name]).write_text(json.dumps(rec))
    checks = [{"name": "decode campaign: working tree clean", "ok": True, "detail": ""}, {"name": "chain campaign: working tree clean", "ok": True, "detail": ""}]
    return {"schema": rc.BASELINE_SCHEMA, "tag": "b200", "gpu": "NVIDIA B200", "sm_count": 148, "commit": tbt.BASE_COMMIT, "tree": "at-2dcb4f8", "torch": "2.14", "cuda": "13.4",
            "created": "2026-11-11T05:10:30Z", "round": 1, "self_check": {"checks": 2, "failed": 0, "list": checks, "ok": True, "problems": []},
            "decode_matrix": campaign(d, "decode"), "layer_chain": campaign(d, "chain", "chain"),
            "suites": {"S": {"record": paths["S"], "totals": suite_s["totals"]}, "gate": {"record": paths["gate"], "totals": suite_gate["totals"]},
                       "S_round0_subset": {"note": "a first run without the sanitizer file", "record": str(d / "none.json")}}, "_path": "mem"}


@pytest.fixture()
def baseline(tmp_path):
    return baseline_record(tmp_path), tmp_path


def test_a_consolidated_baseline_passes_when_its_raw_records_reproduce_its_verdicts(baseline):
    rec, _ = baseline
    assert rc.baseline_problems(rec) == []


BASELINE_MUTANTS = [
    ("another schema", lambda r: r.update(schema="cfbi-baseline-v1"), "not a cfbi-baseline-v2 record"),
    ("an empty commit", lambda r: r.update(commit=""), "is not a hexadecimal commit hash"),
    ("a device tag that is not in the GPU name", lambda r: r.update(tag="b300"), "does not occur in the GPU name"),
    ("a self-check that did not pass", lambda r: r["self_check"].update(ok=False), "self-check of the assembler did not pass"),
    ("a self-check with a failed check", lambda r: r["self_check"].update(failed=1), "self-check of the assembler did not pass"),
    ("a self-check list that does not match", lambda r: r["self_check"].update(checks=5), "lists 5 checks"),
    ("a self-check entry that failed", lambda r: r["self_check"]["list"][0].update(ok=False), "not every one passed"),
    ("a final verdict that is not PASS", lambda r: r["decode_matrix"]["final_verdicts"].update(k2_stage_us="INVALID"), "needs PASS on every metric"),
    ("an unknown first verdict", lambda r: r["decode_matrix"]["first_campaign_verdicts"].update(k2_stage_us="MAYBE"), "first-campaign verdict"),
    ("a baseline of two different trees", lambda r: r["decode_matrix"].update(b="at-other"), "A/A campaign"),
    ("another bench", lambda r: r["layer_chain"].update(bench="prefill"), "bench must be decode or chain"),
    ("a metric without verdicts", lambda r: r["decode_matrix"]["final_verdicts"].pop("k2_stage_us"), "must have an entry for every metric"),
    ("a noise-limited cell that is not a cell", lambda r: r["decode_matrix"]["noise_limited_cells"].update(k2_stage_us=["ghost"]), "must list cells of the campaign"),
    ("a re-run count that the re-run cells do not cover", lambda r: r["decode_matrix"]["cells_needing_rerun"].update(k2_stage_us=2), "cells_needing_rerun"),
    ("fewer samples than declared", lambda r: r["decode_matrix"].update(samples_per_side=5), "records_first holds 3 record(s)"),
    ("fewer than three samples", lambda r: r["decode_matrix"].update(samples_per_side=2), "at least 3"),
    ("a tolerance that is not positive", lambda r: r["decode_matrix"].update(tolerance_pct=0), "tolerance_pct"),
    ("a re-run on one side only", lambda r: r["decode_matrix"]["records_rerun"].update(A=["x.json"]), "as many records for A as for B"),
    ("cells that need a re-run without re-run records", lambda r: r["decode_matrix"].update(cells_needing_rerun={"k2_stage_us": 1}, rerun_cells={"k2_stage_us": ["c"]}),
     "no re-run record is named"),
    ("a suite without a record", lambda r: r["suites"]["gate"].pop("record"), "suite gate has no record"),
    ("suite totals that are not the record's", lambda r: r["suites"]["gate"].update(totals={"passed": 1}), "totals of the baseline are not those of its record"),
    ("no suites", lambda r: r.pop("suites"), "no suites"),
]


@pytest.mark.parametrize("name,edit,expected", BASELINE_MUTANTS, ids=[m[0] for m in BASELINE_MUTANTS])
def test_a_broken_baseline_is_rejected(baseline, name, edit, expected):
    rec, _ = baseline
    assert has(rc.baseline_problems(mutated(rec, edit)), expected), (name, rc.baseline_problems(mutated(rec, edit))[:3])


def test_a_raw_record_that_cannot_be_read_again_or_does_not_validate_is_rejected(baseline):
    rec, d = baseline
    first = rec["decode_matrix"]["records_first"]["A"]
    missing = mutated(rec, lambda r: r["decode_matrix"]["records_first"]["A"].__setitem__(0, str(d / "gone.json")))
    assert has(rc.baseline_problems(missing), "cannot be read again")
    assert not has(rc.baseline_problems(missing, allow_missing_raw=True), "cannot be read again")
    for field, value, expected in (("commit", "", "is not a hexadecimal commit hash"), ("dirty", True, "dirty"), ("dirty", "no", "must be a boolean"), ("gpu", "", "non-empty string"),
                                   ("sm_count", True, "positive integer"), ("commit", "f" * 40, "is not the expected")):
        data = json.loads(Path(first[0]).read_text())
        data["meta"][field] = value
        copy_path = d / ("tampered-%s.json" % field)
        copy_path.write_text(json.dumps(data))
        bad = mutated(rec, lambda r: r["decode_matrix"]["records_first"]["A"].__setitem__(0, str(copy_path)))
        assert has(rc.baseline_problems(bad), expected), (field, value)


def test_a_verdict_that_the_raw_records_do_not_reproduce_is_rejected(tmp_path):
    slow = baseline_record(tmp_path)
    slow["decode_matrix"] = campaign(tmp_path, "slow", cand_scale=1.05)                       # B is 5 % slower than A: a gate FAIL, not an A/A baseline
    slow["decode_matrix"].update(final_verdicts={"k2_stage_us": "PASS"}, first_campaign_verdicts={"k2_stage_us": "PASS"})
    problems = rc.baseline_problems(slow)
    assert has(problems, "the final verdict of k2_stage_us is FAIL when recomputed from the raw records, the baseline says PASS")
    assert has(problems, "the first-campaign verdict of k2_stage_us is FAIL when recomputed")


def test_suite_records_of_a_baseline_are_validated_with_the_strengthened_rules(baseline):
    rec, d = baseline
    s_path = Path(rec["suites"]["S"]["record"])
    for name, edit, expected in (
            ("a failed test", lambda s: (s["tests"].update({next(iter(s["tests"])): "failed"}), s["failed"].append(next(iter(s["tests"]))), s["totals"].update(failed=1, passed=s["totals"]["passed"] - 1),
                                         s["files"]["tests/cfbi/test_x.py"].update(failed=1, passed=s["files"]["tests/cfbi/test_x.py"]["passed"] - 1)), "failed or errored"),
            ("another device", lambda s: s["meta"].update(gpu="NVIDIA B300"), "the record is of NVIDIA B300"),
            ("another commit", lambda s: s["meta"].update(commit="f" * 40), "commit"),
            ("a dirty tree", lambda s: s["meta"].update(dirty=True), "dirty"),
            ("a subset", lambda s: s["meta"].update(subset=True, kind="single", parts=None), "subset"),
            ("totals that are not the sum", lambda s: s["totals"].update(passed=999), "totals do not match")):
        data = json.loads(s_path.read_text())
        edit(data)
        path = d / ("S-%s.json" % name.replace(" ", "-"))
        path.write_text(json.dumps(data))
        bad = mutated(rec, lambda r: r["suites"]["S"].update(record=str(path), totals=data["totals"]))
        assert has(rc.baseline_problems(bad), expected), (name, rc.baseline_problems(bad)[:3])
    not_merged = json.loads(s_path.read_text())
    not_merged["meta"].pop("expected")
    path = d / "S-no-expected.json"
    path.write_text(json.dumps(not_merged))
    assert has(rc.baseline_problems(mutated(rec, lambda r: r["suites"]["S"].update(record=str(path)))), "does not say which expected files and tests")


def test_the_baseline_and_suite_command_lines(baseline, tmp_path, capsys):
    rec, d = baseline
    rec.pop("_path")
    good, bad = d / "baseline.json", d / "baseline-bad.json"
    good.write_text(json.dumps(rec))
    bad.write_text(json.dumps(mutated(rec, lambda r: r["self_check"].update(ok=False))))
    assert rc.main(["baseline", "--record", str(good)]) == 0
    assert rc.main(["baseline", "--record", str(bad)]) == 1
    assert rc.main(["baseline", "--record", str(d / "missing.json")]) == 2
    s = rec["suites"]["S"]["record"]
    assert rc.main(["suite", "--baseline", s, "--candidate", s]) == 0
    broken = json.loads(Path(s).read_text())
    broken["meta"]["gpu"] = "NVIDIA B300"
    other = d / "S-other-device.json"
    other.write_text(json.dumps(broken))
    assert rc.main(["suite", "--baseline", s, "--candidate", str(other)]) == 1
    assert "not comparable: gpu" in capsys.readouterr().out
