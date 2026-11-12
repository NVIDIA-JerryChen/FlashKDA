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
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "benchmarks", _ROOT / "tests" / "cfbi"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import codegen_gate as cg  # noqa: E402
import bench_records as br  # noqa: E402
import decision_record as dr  # noqa: E402
import record_checker as rc  # noqa: E402
import test_bench_tools as tbt  # noqa: E402
from test_record_checker import commit_file, git, has, mutated  # noqa: E402

C1, C2, C3, C4, C5 = ("%07x%033x" % (i, i) for i in range(1, 6))        # commit-like hashes (resolved against a repository only where a test builds one)
CELL = "decode;N=1;H=12;regime=lockstep;"


# ---- the ledger ------------------------------------------------------------------------------------------------------------------------------
DUMPS = tempfile.mkdtemp(prefix="rc-ledger-dumps-")             # the dumps that the rows of the synthetic ledgers are bound to
BASE = "b" * 40                                                  # the base commit of the synthetic loop
META = {"base": BASE, "head": C5}
LABELS = {"conda": "Cuda compilation tools, release 13.1 | conda gcc 14.3", "container_b200": "container NGC pytorch 26.08: nvcc 13.4.59", "container_b300": "container NGC pytorch 26.08: nvcc 13.4.59"}
SPILLS = {"sm_100a": (4, 4), "sm_103a": (12, 8), "sm_90a": (428, 172)}      # (stores, loads) of the base row engine
ROWS = "cfbi::rows_kernel"
LB0, LB1 = ROWS + "<Lb0E>", ROWS + "<Lb1E>"                      # the instantiations of a commit with the template, as the dumps spell them
REFERENCE = "cfbi::ref::rows_kernel"
RESERVE = 1024
SHARED = 8192 + RESERVE                                          # static shared memory of the row engine and its reference in the dumps (the figure includes the reserve of the driver)
GPU = {"b200": "NVIDIA B200", "b300": "NVIDIA B300 SXM6 AC"}
LIMITS = {"registers_per_sm": 65536, "shared_per_sm": 233472, "threads_per_sm": 2048, "reserve_per_cta": RESERVE}
SINGLE_ALLOW = os.path.join(DUMPS, "allow-single-launch.json")   # the allow-list of the identity comparison: the plan kernel's instantiation is the base's row engine
with open(SINGLE_ALLOW, "w") as _f:
    json.dump({"schema": cg.ALLOW_SCHEMA, "aliases": {LB0: ROWS}, "may_differ": []}, _f)
PRIMITIVES, LAUNCH = "constexpr int NTHREADS = 256;\n", "constexpr size_t ROWS_DYNAMIC_SMEM = 0;\n"      # the launch of the row engine as the sources define it
_dumps = {}
_occupancy = {}


def dump_file(key, commit, role="cand", rows=None, others=None, extra=None, label=None, templated=False):
    """The path of a synthetic dump for the comparison ``key``: the base of the loop (role base) or a candidate of ``commit``. ``rows`` edits the figures of the row engine of a candidate (of the
    decode instantiation ``<Lb1E>`` when ``templated``), ``others`` those of other kernels (kernel -> figures), ``extra`` adds kernels (kernel -> arguments of ``cg_kernel``). A ``templated``
    candidate has the template: ``<Lb0E>`` is the base's row engine (the same machine code) and ``<Lb1E>`` the single-launch decode, a new kernel."""
    sig = json.dumps([key, commit, role, rows, others, extra, label, templated], sort_keys=True)
    if sig not in _dumps:
        kernels = {}
        for arch in rc.COMPARISON_ARCHS[key]:
            def engine(sass):
                return tbt.cg_kernel(reg=128, stack=112, spill=SPILLS[arch], sass=sass, shared=SHARED)
            if role == "base":
                k = {ROWS: engine("rows")}
            elif templated == "swapped":     # the review's candidate: the plan kernel's instantiation is CHANGED and the decode instantiation carries the base's machine code
                k = {LB0: engine("rows-" + commit[:7]), LB1: engine("rows")}
            elif templated:
                k = {LB0: engine("rows"), LB1: tbt.cg_kernel(reg=128, stack=96, spill=(0, 0), sass="rows-" + commit[:7], shared=SHARED)}
            else:
                k = {ROWS: engine("rows-" + commit[:7])}
            k.update({REFERENCE: tbt.cg_kernel(reg=48, sass="ref-rows", maxreg=None, maxthr=None, shared=SHARED),
                      "cfbi::k2_kernel<Li16E>": tbt.cg_kernel(reg=172, sass="k2"), "cfbi::plan_kernel": tbt.cg_kernel(reg=40, sass="plan")})
            if role != "base":
                k[LB1 if templated else ROWS].update(rows or {})
                for name, fig in (others or {}).items():
                    k[name].update(fig)
                for name, args in (extra or {}).items():
                    k[name] = tbt.cg_kernel(**args)
            kernels[arch] = k
        dump = tbt.cg_dump(kernels, label=label or LABELS[key])
        dump["meta"]["tree"]["commit"] = BASE if role == "base" else commit
        path = os.path.join(DUMPS, "%s-%d.json" % (role, len(_dumps)))
        with open(path, "w") as f:
            json.dump(dump, f)
        _dumps[sig] = path
    return _dumps[sig]


def occupancy_file(dev, commit, dump_path, edit=None):
    """The occupancy-API record of ``dev`` at ``commit`` for the instantiations of a candidate dump (the figures are the dump's, the CTAs per SM the arithmetic's); ``edit`` edits the record."""
    sig = (dev, commit, dump_path, None if edit is None else id(edit))
    if sig not in _occupancy:
        kernels = cg.load_dump(dump_path)["kernels"][dr.DEVICE_ARCH[dev]]
        instances = []
        for key in sorted(k for k in kernels if cg.family(k) == ROWS) + [REFERENCE]:
            k, static = kernels[key], kernels[key]["shared"] - RESERVE
            instances.append({"name": key.replace("<Lb0E>", "<false>").replace("<Lb1E>", "<true>"), "registers": k["reg"], "static_smem": static, "dynamic_smem": 0, "threads": 256,
                              "ctas_per_sm": dr.ctas_per_sm(k["reg"], 256, static, 0, LIMITS)})
        rec = {"capability": [10, 0], "cuda": "13.4", "gpu": GPU[dev], "label": "%s-%s" % (dev, commit[:9]), "sm_count": 148, "torch": "2.14.0",
               "occupancy": {"device": {"max_threads_per_sm": 2048, "regs_per_sm": 65536, "sm_count": 148, "smem_per_sm": 233472, "smem_reserved_per_block": RESERVE}, "instances": instances}}
        if edit is not None:
            edit(rec)
        path = os.path.join(DUMPS, "occupancy-%s-%d.json" % (dev, len(_occupancy)))
        with open(path, "w") as f:
            json.dump(rec, f)
        _occupancy[sig] = path
    return _occupancy[sig]


def comparison(key, commit, instantiations=None, sid="trims", templated=None, allow=None, **kw):
    """The honest ledger entry of the comparison ``key`` for a candidate of ``commit`` (``kw`` edits the candidate dump) and the resource figures of its row engine. The steps of
    ``rc.IDENTITY_STEPS`` are compared in identity mode and their candidate has the template (``templated``, ``<Lb0E>`` renamed onto the base's row engine by the allow-list); another step may
    be templated too. A production-toolchain comparison names the occupancy record of its device."""
    single = sid in rc.IDENTITY_STEPS
    templated = single if templated is None else templated
    instantiations = tuple(instantiations or ((LB0, LB1) if templated else (ROWS,)))
    base, cand = dump_file(key, BASE, "base"), dump_file(key, commit, "cand", templated=templated, **kw)
    bd, cd = cg.load_dump(base), cg.load_dump(cand)
    allow = allow or (SINGLE_ALLOW if templated else None)
    mode = "identity" if single else "change"
    report = cg.compare_dumps(bd, cd, cg.load_allow(allow), mode)
    entry = {"base_dump": base, "cand_dump": cand, "prev_dump": None, "allow": allow, "mode": mode, "instantiations": list(instantiations),
             "verdict": "PASS" if report["exit"] == 0 else "FAIL", "compared": report["compared"], "differences": len(report["differences"])}
    if key.startswith("container_"):
        entry["occupancy"] = occupancy_file(key.split("_", 1)[1], commit, cand)
    figures = dict((arch, dict((i, dict((f, cd["kernels"][arch][i][f]) for f in rc.RESOURCE_FIGURES)) for i in instantiations)) for arch in rc.COMPARISON_ARCHS[key])
    return entry, figures


def gate_evidence(commit, keys=rc.COMPARISONS, **kw):
    codegen, resources = {}, {}
    for key in keys:
        codegen[key], resources[key] = comparison(key, commit, **kw)
    return codegen, resources


def ab_figures(devices=rc.DEVICES):
    return dict((d, {"medians": {CELL: {"base": 14.0, "cand": 13.2}}, "noise_limited": []}) for d in devices)


def chain_of(sid):
    return "pass" if sid in rc.CHAIN_STEPS else "not_applicable"


def adopted(sid, commit=C1, markers=("none",), vs_previous="pass", vs_base="pass", chain=None, **cand_kw):
    codegen, resources = gate_evidence(commit, sid=sid, **cand_kw)
    return {"id": sid, "outcome": "adopted", "commit": commit, "revert_commit": None, "markers": list(markers), "failed_at": None, "reason": "",
            "gates": {"exactness": True, "codegen": True}, "timing": {"vs_previous": vs_previous, "vs_base": vs_base, "chain": chain or chain_of(sid)},
            "codegen_gate": codegen, "resource_gate": resources, "ab": ab_figures()}


def not_adopted(sid, failed_at, commit=None, revert=None, markers=("none",), timing=None):
    reached = rc.GATE_PHASES.index(failed_at)
    step = {"id": sid, "outcome": "not_adopted", "commit": commit, "revert_commit": revert, "markers": list(markers), "failed_at": failed_at, "reason": "the %s phase failed" % failed_at,
            "gates": {"exactness": None if reached < 1 else reached > 1, "codegen": None if reached < 2 else reached > 2}, "timing": None,
            "codegen_gate": None, "resource_gate": None, "ab": None}
    candidate = commit or "e" * 40
    if reached == 2:                                                      # the resource gate failed on the conda toolchain: 129 registers
        step["codegen_gate"], step["resource_gate"] = gate_evidence(candidate, keys=("conda",), sid=sid, rows={"reg": 129})
    if reached == 3:
        step["codegen_gate"], step["resource_gate"] = gate_evidence(candidate, sid=sid)
        step["ab"] = ab_figures()
        step["timing"] = timing or {"vs_previous": "fail", "vs_base": "pass", "chain": chain_of(sid)}
    return step


def skipped(sid, markers=("none",)):
    return {"id": sid, "outcome": "skipped", "commit": None, "revert_commit": None, "markers": list(markers), "reason": "the profile shows no gain to expect", "failed_at": None,
            "gates": None, "timing": None, "codegen_gate": None, "resource_gate": None, "ab": None}


def step_check(step, waiver=None):
    return rc.step_problems(step, waiver, META)


UNIT_CASES = [
    ("E passes against T and F beats E", {"e_vs_t": "pass", "f_vs_e": "pass", "f_vs_t": "not_run"}, ("adopted", "adopted")),
    ("E passes against T and F does not beat E", {"e_vs_t": "pass", "f_vs_e": "fail", "f_vs_t": "not_run"}, ("adopted", "not_adopted")),
    ("E fails against T and F beats T", {"e_vs_t": "fail", "f_vs_e": "not_run", "f_vs_t": "pass"}, ("adopted", "adopted")),
    ("E fails against T and F does not beat T", {"e_vs_t": "fail", "f_vs_e": "not_run", "f_vs_t": "fail"}, ("not_adopted", "not_adopted")),
]
E_ROW, F_ROW = 2, 3                                                      # the positions of the two members of the unit in a ledger


def unit_rows(unit, outcomes, e_marker=("e_tile_marker",), f_marker=("a_fragments_marker",)):
    """The rows of the two members of the adoption unit for the given arms and outcomes."""
    prev_e = unit["e_vs_t"]
    prev_f = unit["f_vs_e"] if unit["e_vs_t"] == "pass" else unit["f_vs_t"]
    rows = []
    for sid, outcome, prev, markers, commit in (("e_tile", outcomes[0], prev_e, e_marker, C2), ("a_fragments", outcomes[1], prev_f, f_marker, C3)):
        if outcome == "adopted":
            rows.append(adopted(sid, commit, markers, vs_previous=prev))
        else:
            rows.append(not_adopted(sid, "timing", commit, C4, markers, {"vs_previous": prev, "vs_base": "pass", "chain": "pass"}))
    return rows


def ledger(unit_case=0, **overrides):
    _, unit, outcomes = UNIT_CASES[unit_case]
    e, f = unit_rows(unit, outcomes)
    steps = {"single_launch": adopted("single_launch", C1, ("single_launch_marker",)), "trims": adopted("trims", C5), "e_tile": e, "a_fragments": f,
             "kd_tile": not_adopted("kd_tile", "exactness")}
    steps.update(overrides)
    return {"schema": rc.LEDGER_SCHEMA, "meta": dict(META), "steps": [steps[s] for s in rc.STEP_IDS], "unit": dict(unit), "_path": "mem"}


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
    wrong_row = mutated(base, lambda l: l["steps"][F_ROW]["timing"].update(vs_previous="fail"))            # the F row says it did not beat E, the arms say it did
    assert has(rc.ledger_problems(wrong_row), "the row of a_fragments says 'fail' against the previous accepted state")
    assert has(rc.ledger_problems(mutated(base, lambda l: l.pop("unit")), require_all_steps=True), "needs its three arms")           # a complete ledger names the arms
    assert not has(rc.ledger_problems(mutated(base, lambda l: l.pop("unit"))), "three arms")                                         # an intermediate one need not yet
    early = ledger(3, e_tile=not_adopted("e_tile", "exactness"))                                          # a member that failed an earlier phase has no timing to agree with the arms
    assert rc.ledger_problems(early, require_all_steps=True) == []
    alone = mutated(base, lambda l: l.pop("unit"))                                                         # without the unit the rows stand alone: E has to beat the previous state itself
    alone["steps"][E_ROW]["timing"]["vs_previous"] = "fail"
    assert has(rc.ledger_problems(alone), "beat the previous accepted state")


# ---- B9: the selected state passes the cumulative and chain gates, the figures are those of the dumps ---------------------------------------------
def timing_edit(row, **fields):
    return lambda l: l["steps"][row]["timing"].update(fields)


@pytest.mark.parametrize("row,fields", [
    (E_ROW, {"vs_base": "fail"}), (E_ROW, {"vs_base": "not_run"}), (E_ROW, {"chain": "fail"}), (E_ROW, {"chain": "not_run"}), (E_ROW, {"chain": "not_applicable"}),
    (F_ROW, {"vs_base": "fail", "chain": "fail"}), (F_ROW, {"vs_base": "fail"}), (F_ROW, {"chain": "fail"}), (F_ROW, {"chain": "not_applicable"}), (F_ROW, {"vs_base": "not_run"})])
def test_both_adopted_members_of_the_unit_pass_the_cumulative_and_the_chain_gates(row, fields):
    """The review's counterexamples on the ledger where E passes against T and F beats E (both adopted): each of the two rows must pass them by itself."""
    good = ledger(0)
    assert rc.ledger_problems(good, require_all_steps=True) == []
    assert rc.ledger_problems(mutated(good, timing_edit(row, **fields)), require_all_steps=True) != [], (row, fields)


def test_when_only_the_e_tile_is_kept_it_is_the_selected_state_and_passes_everything_itself():
    good = ledger(1)
    assert rc.ledger_problems(good, require_all_steps=True) == []
    for fields in ({"vs_base": "fail"}, {"chain": "fail"}, {"vs_base": "not_run"}, {"vs_previous": "fail"}):
        assert rc.ledger_problems(mutated(good, timing_edit(E_ROW, **fields)), require_all_steps=True) != [], fields


def test_the_e_tile_that_the_a_fragments_rescued_is_waived_and_the_selected_a_fragments_are_not():
    """E fails against T and F beats T (both adopted): E need not pass the gain, the cumulative or the chain gate on its own, F (the selected state) must pass all of them."""
    good = ledger(2)
    assert rc.ledger_problems(good, require_all_steps=True) == []
    waived = mutated(good, timing_edit(E_ROW, vs_base="fail", chain="fail"))
    assert rc.ledger_problems(waived, require_all_steps=True) == []
    assert has(rc.ledger_problems(mutated(good, timing_edit(E_ROW, vs_base="not_run")), require_all_steps=True), "measured")                  # waived, still a measurement
    assert has(rc.ledger_problems(mutated(good, timing_edit(E_ROW, vs_previous="pass")), require_all_steps=True), "the arms say 'fail'")      # the row must agree with the arms
    for fields in ({"vs_base": "fail"}, {"chain": "fail"}, {"vs_base": "fail", "chain": "fail"}, {"chain": "not_applicable"}):
        assert rc.ledger_problems(mutated(good, timing_edit(F_ROW, **fields)), require_all_steps=True) != [], fields


@pytest.mark.parametrize("sid", ["e_tile", "a_fragments", "kd_tile"])
def test_the_layer_chain_is_measured_for_the_steps_that_change_the_shared_memory_requirement(sid):
    assert step_check(adopted(sid)) == []
    for chain in ("not_applicable", "not_run", "fail"):
        assert step_check(adopted(sid, chain=chain)) != [], chain
    assert has(step_check(adopted(sid, chain="not_applicable")), "layer chain is measured")
    assert has(step_check(adopted(sid, chain="fail")), "keep the layer chain")


@pytest.mark.parametrize("sid", ["single_launch", "trims"])
def test_the_layer_chain_is_not_applicable_to_the_other_steps(sid):
    assert step_check(adopted(sid)) == []
    assert has(step_check(adopted(sid, chain="pass")), "not applicable to this step")
    assert has(step_check(adopted(sid, chain="fail")), "not applicable to this step")


def edit_entry(key, **fields):
    return lambda s: s["codegen_gate"][key].update(fields)


def edit_figure(key, arch, inst=ROWS, **fields):
    return lambda s: s["resource_gate"][key][arch][inst].update(fields)


def swap_dump(key, field, **dump_kw):
    """The row with the dump of one comparison replaced by another dump (``dump_kw`` of ``dump_file``)."""
    def edit(s):
        s["codegen_gate"][key][field] = dump_file(dump_kw.pop("dump_key", key), dump_kw.pop("commit"), dump_kw.pop("role", "cand"), **dump_kw)
    return edit


CODEGEN_MUTANTS = [
    # the figures that the ledger declares are the ones of the dumps
    ("a resource figure the dump does not have: 129 registers", edit_figure("container_b200", "sm_100a", reg=129), "says"),
    ("a stack figure the dump does not have", edit_figure("conda", "sm_90a", stack=0), "says"),
    ("a spill figure the dump does not have", edit_figure("container_b300", "sm_103a", spill_loads=0), "says"),
    ("a resource figure that is a boolean", edit_figure("conda", "sm_100a", reg=True), "says"),
    ("a resource figure that is missing", edit_figure("conda", "sm_100a", spill_stores=None), "says"),
    ("no resource figures for an architecture", lambda s: s["resource_gate"]["conda"].pop("sm_90a"), "must give the resource figures of the row engine"),
    ("a verdict that the dumps contradict", edit_entry("container_b200", verdict="FAIL"), "the comparison of the dumps says PASS"),
    ("100 differences that the dumps do not have", edit_entry("container_b300", verdict="PASS", differences=100), "differences = 100"),
    ("a number of kernels compared that the dumps do not have", edit_entry("conda", compared=1), "compared = 1"),
    ("an instantiation that the ledger leaves out", edit_entry("conda", instantiations=[]), "names the instantiations"),
    ("an instantiation that the dump does not have", edit_entry("conda", instantiations=[ROWS, "cfbi::rows_kernel<Lb1E>"]), "names the instantiations"),
    ("the mode identity", edit_entry("conda", mode="identity"), "the mode of a step's comparison is change"),
    ("no base dump", lambda s: s["codegen_gate"]["conda"].pop("base_dump"), "needs base_dump"),
    ("a dump that cannot be read", edit_entry("container_b200", cand_dump="/nonexistent/dump.json"), "cannot be used"),
    # the dumps are those of the row's commit, of the base of the loop, of the toolchain and the architectures of the comparison
    ("a candidate dump of another commit", swap_dump("conda", "cand_dump", commit="d" * 40), "the candidate dump is of commit"),
    ("a base dump that is not the base of the loop", swap_dump("conda", "base_dump", commit="d" * 40), "the base dump is of commit"),
    ("a candidate dump of another toolchain", swap_dump("conda", "cand_dump", commit=C1, label=LABELS["container_b200"]), "is not the conda toolchain"),
    ("a candidate dump for another architecture", swap_dump("container_b200", "cand_dump", dump_key="container_b300", commit=C1), "the candidate dump is for"),
]


@pytest.mark.parametrize("name,edit,expected", CODEGEN_MUTANTS, ids=[m[0] for m in CODEGEN_MUTANTS])
def test_the_figures_of_a_row_are_those_of_the_dumps_it_is_bound_to(name, edit, expected):
    good = adopted("kd_tile")
    assert step_check(good) == []
    assert has(step_check(mutated(good, edit)), expected), (name, step_check(mutated(good, edit))[:3])


def test_the_ledger_names_the_base_commit_that_the_base_dumps_are_bound_to():
    assert has(rc.ledger_problems(mutated(ledger(0), lambda l: l["meta"].pop("base"))), "names no base commit")
    assert has(rc.ledger_problems(mutated(ledger(0), lambda l: l["meta"].update(base="c" * 40))), "the base dump is of commit")


@pytest.mark.parametrize("name,cand_kw", [
    ("129 registers", {"rows": {"reg": 129}}),
    ("stack above the base", {"rows": {"stack": 113}}),
    ("spill stores above the base", {"rows": {"spill_stores": 5}}),
    ("spill loads above the base", {"rows": {"spill_loads": 5}}),
    ("a launch bound that differs", {"rows": {"max_threads": 512}}),
    ("another kernel that is no longer identical", {"others": {"cfbi::plan_kernel": {"sass_hash": tbt.cg_hash("plan2"), "sass_hash_strict": tbt.cg_hash("plan2")}}}),
    ("a new instantiation of the row engine with 129 registers", {"templated": True, "rows": {"reg": 129}}),
])
def test_a_resource_or_identity_failure_of_the_dumps_rejects_an_adopted_row_whatever_the_ledger_says(name, cand_kw):
    """The review's 129 registers, and the base-relative stack and spill: the comparison of the dumps decides, an honest ledger that declares the failure is rejected too."""
    honest = adopted("trims", **cand_kw)                         # a second instantiation of the family is declared in the ledger by the honest row
    problems = step_check(honest)
    assert has(problems, "the comparison of the dumps fails"), (name, problems[:3])
    dishonest = mutated(honest, lambda s: [e.update(verdict="PASS", differences=0) for e in s["codegen_gate"].values()])
    assert has(step_check(dishonest), "the comparison of the dumps says FAIL") or has(step_check(dishonest), "differences ="), name


def test_a_dump_without_a_launch_bound_of_the_row_engine_is_unusable_evidence_not_an_equality():
    row = adopted("trims")
    row["codegen_gate"]["conda"]["cand_dump"] = dump_file("conda", C1, "cand", rows={"max_threads": None})
    assert has(step_check(row), "cannot be used") and has(step_check(row), "launch-bound attribute max_threads")


def test_a_second_instantiation_of_the_row_engine_is_bound_and_gated_like_the_first():
    row = adopted("trims", templated=True)
    assert step_check(row) == []
    hidden = mutated(row, lambda s: [e.update(instantiations=[LB0]) for e in s["codegen_gate"].values()])
    assert has(step_check(hidden), "names the instantiations")


# ---- the single launch: the identity comparison, and the occupancy record of every device --------------------------------------------------------
def single(commit=C1, **kw):
    return adopted("single_launch", commit, ("single_launch_marker",), **kw)


def allow_file(aliases=None, may_differ=()):
    path = os.path.join(DUMPS, "allow-%d.json" % len(os.listdir(DUMPS)))
    with open(path, "w") as f:
        json.dump({"schema": cg.ALLOW_SCHEMA, "aliases": {LB0: ROWS} if aliases is None else aliases, "may_differ": list(may_differ)}, f)
    return path


def changed(name):
    return {"sass_hash": tbt.cg_hash(name), "sass_hash_strict": tbt.cg_hash(name)}


def test_the_single_launch_row_is_judged_by_the_identity_comparison_and_the_other_steps_by_the_change_mode():
    row = single()
    assert step_check(row) == []
    for key in rc.COMPARISONS:
        entry = row["codegen_gate"][key]
        assert (entry["mode"], entry["instantiations"], entry["allow"]) == ("identity", [LB0, LB1], SINGLE_ALLOW)
        assert all(set(row["resource_gate"][key][arch]) == {LB0, LB1} for arch in rc.COMPARISON_ARCHS[key])
    assert adopted("trims")["codegen_gate"]["conda"]["mode"] == "change" and step_check(adopted("trims")) == []


SINGLE_MUTANTS = [
    ("the change mode for the single launch", edit_entry("conda", mode="change"), "the mode of the comparison of single_launch is identity"),
    ("an allow-list that lets the row engine differ", edit_entry("conda", allow=allow_file(may_differ=[ROWS])), "allows no kernel to differ"),
    ("an allow-list that renames another kernel", edit_entry("container_b200", allow=allow_file({LB0: ROWS, "cfbi::plan_kernel": "cfbi::old_plan_kernel"})), "the only rename of an identity comparison is"),
    ("an allow-list that cannot be read", edit_entry("conda", allow="/nonexistent/allow.json"), "cannot be used"),
    ("no resource figures for the decode instantiation", lambda s: s["resource_gate"]["conda"]["sm_90a"].pop(LB1), "says None"),
    ("a resource figure of the decode instantiation that the dump does not have", edit_figure("container_b300", "sm_103a", LB1, reg=127), "says"),
    ("the decode instantiation left out of the ledger", edit_entry("conda", instantiations=[LB0]), "names the instantiations"),
]


@pytest.mark.parametrize("name,edit,expected", SINGLE_MUTANTS, ids=[m[0] for m in SINGLE_MUTANTS])
def test_the_comparison_of_the_single_launch_is_bound_to_its_mode_its_allow_list_and_its_dumps(name, edit, expected):
    good = single()
    assert step_check(good) == []
    assert has(step_check(mutated(good, edit)), expected), (name, step_check(mutated(good, edit))[:3])


@pytest.mark.parametrize("name,cand_kw", [
    ("the plan kernel's instantiation <Lb0E> that is not the base's row engine", {"others": {LB0: changed("changed")}}),
    ("the plan kernel's instantiation <Lb0E> with one register less", {"others": {LB0: {"reg": 127}}}),
    ("another kernel that is no longer identical", {"others": {"cfbi::plan_kernel": changed("plan2")}}),
    ("the decode instantiation with 129 registers", {"rows": {"reg": 129}}),
    ("the decode instantiation with a stack above the base", {"rows": {"stack": 113}}),
    ("the decode instantiation with spill stores above the base", {"rows": {"spill_stores": 5}}),
    ("a launch bound of the decode instantiation that differs", {"rows": {"max_threads": 512}}),
])
def test_a_difference_of_the_default_path_or_a_resource_failure_of_the_decode_instantiation_rejects_the_single_launch_row(name, cand_kw):
    honest = single(**cand_kw)
    problems = step_check(honest)
    assert has(problems, "the comparison of the dumps fails"), (name, problems[:3])
    dishonest = mutated(honest, lambda s: [e.update(verdict="PASS", differences=0) for e in s["codegen_gate"].values()])
    assert has(step_check(dishonest), "the comparison of the dumps says FAIL") or has(step_check(dishonest), "differences ="), name


def test_the_same_change_of_the_plan_kernels_instantiation_is_tolerated_in_change_mode_and_fails_the_identity_comparison():
    edit = {"others": {LB0: changed("changed")}}
    assert step_check(adopted("trims", templated=True, **edit)) == []                         # a row-engine step may change the row engine
    assert has(step_check(single(**edit)), "the comparison of the dumps fails")               # the single launch may not


# the review's attack on the identity comparison: an allow-list that is right by family and wrong by instance
SWAP_ALIASES = {LB1: ROWS, LB0: ROWS + "<kept>"}          # the decode instantiation stands for the base's engine, the plan kernel's instantiation is moved to a name that the base does not have


def swapped_row(aliases=None):
    """A single-launch row whose plan-kernel instantiation <Lb0E> has CHANGED machine code and whose decode instantiation <Lb1E> carries the base's, compared with an allow-list of ``aliases``."""
    return adopted("single_launch", C1, ("single_launch_marker",), templated="swapped", allow=allow_file(aliases or SWAP_ALIASES))


def test_the_review_s_alias_swap_passed_every_check_of_the_comparison_except_the_allow_list():
    row = swapped_row()
    entry = row["codegen_gate"]["conda"]
    base, cand = cg.load_dump(entry["base_dump"]), cg.load_dump(entry["cand_dump"])
    report = cg.compare_dumps(base, cand, cg.load_allow(entry["allow"]), "identity")
    assert report["exit"] == 0 and not report["differences"] and "sm_100a cfbi::rows_kernel<Lb0E>" in report["added"], report      # the dump comparison itself is fooled: <Lb0E> counts as a new kernel
    assert cand["kernels"]["sm_100a"][LB0]["sass_hash"] != base["kernels"]["sm_100a"][ROWS]["sass_hash"]                              # although its machine code is not the base's
    assert has(step_check(row), "the only rename of an identity comparison is")                                                         # the allow-list is what stops it


@pytest.mark.parametrize("name,aliases,whole_row", [
    ("the decode instantiation onto the base's engine and the plan instantiation to another name", SWAP_ALIASES, True),
    ("only the decode instantiation renamed", {LB1: ROWS}, True),
    ("the right rename and a second one of the same family", {LB0: ROWS, ROWS + "<Lb2E>": ROWS}, True),
    ("the plan instantiation to a name that the base does not have", {LB0: ROWS + "<kept>"}, True),
    ("an instance of another family to the right name", {"cfbi::k2_kernel<Li16E>": ROWS}, True),
    ("the right instance to the reference kernel", {LB0: "cfbi::ref::rows_kernel"}, False),       # the dump comparison itself refuses two kernels under one name: no whole row to build
])
def test_only_the_plan_kernels_instantiation_may_be_renamed_onto_the_base_s_engine(name, aliases, whole_row):
    problems = rc.identity_allow_problems({"aliases": aliases, "may_differ": []}, "w")
    assert any("the only rename of an identity comparison is" in p for p in problems), (name, problems)
    if whole_row:
        assert has(step_check(swapped_row(aliases)), "the only rename of an identity comparison is"), name


def test_the_exact_rename_or_none_is_a_valid_identity_allow_list_and_the_honest_row_passes():
    assert rc.identity_allow_problems({"aliases": {LB0: ROWS}, "may_differ": []}, "w") == []                    # the allow-list of the real comparisons (records/allow-import1-r6.json)
    assert rc.identity_allow_problems({"aliases": {}, "may_differ": []}, "w") == []
    assert rc.identity_allow_problems({"may_differ": []}, "w") == []
    assert json.load(open(SINGLE_ALLOW))["aliases"] == {"cfbi::rows_kernel<Lb0E>": "cfbi::rows_kernel"}          # the fixture is the real allow-list
    assert step_check(single()) == []
    assert has(rc.identity_allow_problems({"aliases": {LB0: ROWS}, "may_differ": [ROWS]}, "w"), "allows no kernel to differ")


def occupancy_edit(dev, edit, commit=C1):
    """A row edit that gives the comparison of ``dev`` the occupancy record that ``edit`` changed (the record is built from the candidate dump of the comparison)."""
    def apply(s):
        key = "container_" + dev
        s["codegen_gate"][key]["occupancy"] = occupancy_file(dev, commit, s["codegen_gate"][key]["cand_dump"], edit)
    return apply


OCCUPANCY_MUTANTS = [
    ("no occupancy record", edit_entry("container_b200", occupancy=None), "needs occupancy"),
    ("an occupancy record that cannot be read", edit_entry("container_b300", occupancy="/nonexistent/occupancy.json"), "cannot be used"),
    ("the record of the other device", lambda s: s["codegen_gate"]["container_b200"].update(occupancy=s["codegen_gate"]["container_b300"]["occupancy"]), "is not b200"),
    ("a record of another commit", occupancy_edit("b200", lambda r: r.update(label="b200-" + "d" * 9)), "the record is of commit"),
    ("a record whose label does not name the commit", occupancy_edit("b200", lambda r: r.update(label="b200")), "does not name the device and the commit"),
    ("an instantiation of the dump that the record lacks", occupancy_edit("b200", lambda r: r["occupancy"]["instances"].pop(1)), "the record has no instance for it"),
    ("only the decode instantiation in the record", occupancy_edit("b200", lambda r: r["occupancy"]["instances"].pop(0)), "either the plain cfbi::rows_kernel or both of its instantiations"),
    ("a name that the dump does not have", occupancy_edit("b200", lambda r: r["occupancy"]["instances"][1].update(name="cfbi::rows_kernel<3>")), "the record has no instance for it"),
    ("registers that are not the dump's", occupancy_edit("b200", lambda r: r["occupancy"]["instances"][1].update(registers=96)), "the production dump"),
    ("static shared memory that is not the dump's", occupancy_edit("b200", lambda r: r["occupancy"]["instances"][0].update(static_smem=7680)), "static shared memory"),
    ("fewer than two CTAs per SM", occupancy_edit("b300", lambda r: r["occupancy"]["instances"][1].update(ctas_per_sm=1)), "at least 2 CTAs"),
    ("a CTA of other threads", occupancy_edit("b300", lambda r: r["occupancy"]["instances"][0].update(threads=128, ctas_per_sm=4)), "256 are required"),
    ("a CTA count that the arithmetic does not reproduce", occupancy_edit("b300", lambda r: r["occupancy"]["instances"][0].update(ctas_per_sm=3)), "the arithmetic"),
    ("the record of another GPU", occupancy_edit("b300", lambda r: r.update(gpu="NVIDIA H100")), "is not b300"),
]


@pytest.mark.parametrize("name,edit,expected", OCCUPANCY_MUTANTS, ids=[m[0] for m in OCCUPANCY_MUTANTS])
def test_the_occupancy_record_of_a_device_is_bound_to_the_candidate_dump_of_its_comparison(name, edit, expected):
    good = single()
    assert step_check(good) == [] and "occupancy" not in good["codegen_gate"]["conda"]      # the conda comparison has no device
    assert has(step_check(mutated(good, edit)), expected), (name, step_check(mutated(good, edit))[:4])


@pytest.mark.parametrize("sid", rc.STEP_IDS)
def test_every_step_that_reaches_the_codegen_gate_on_the_production_toolchain_carries_the_occupancy_of_both_devices(sid):
    row = adopted(sid)
    assert step_check(row) == []
    for dev in rc.DEVICES:
        assert has(step_check(mutated(row, edit_entry("container_" + dev, occupancy=None))), "container_%s) needs occupancy" % dev), (sid, dev)
    late = not_adopted(sid, "timing", commit=C2, revert=C3)                                  # a row that was measured has built on both devices
    assert step_check(late) == [] and has(step_check(mutated(late, edit_entry("container_b200", occupancy=None))), "needs occupancy")
    early = not_adopted(sid, "codegen")                                                      # a row that failed on the conda toolchain has no device evidence to ask for
    assert step_check(early) == [] and "container_b200" not in early["codegen_gate"]


def build_launch_repo(repo):
    """A history with the launch of the row engine at the second commit and another thread count at the third."""
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    h = {"repo": repo}
    h["bare"] = commit_file(repo, "docs/readme.md", "no sources yet\n", "a commit without the launch")
    commit_file(repo, "csrc/cfbi/cfbi_primitives.cuh", PRIMITIVES, "the sizes of the kernels")
    h["launch"] = commit_file(repo, "csrc/cfbi/cfbi_launch.cu", LAUNCH, "the launch of the row engine")
    h["threads"] = commit_file(repo, "csrc/cfbi/cfbi_primitives.cuh", PRIMITIVES.replace("256", "128"), "another thread count")
    h["dynamic"] = commit_file(repo, "csrc/cfbi/cfbi_launch.cu", LAUNCH.replace("= 0;", "= 4096;"), "a dynamic shared memory")
    return h


def test_the_launch_of_the_sources_at_the_commit_of_the_record_binds_the_threads_and_the_dynamic_shared_memory(tmp_path):
    h = build_launch_repo(tmp_path / "repo")
    repo = h["repo"]
    assert rc.step_problems(single(h["launch"]), None, META, repo) == []
    dynamic = mutated(single(h["launch"]), occupancy_edit("b200", lambda r: r["occupancy"]["instances"][1].update(dynamic_smem=4096), h["launch"]))
    assert has(rc.step_problems(dynamic, None, META, repo), "the launch in the sources")
    assert rc.step_problems(dynamic, None, META) == []                                         # without a repository the launch is not read: the CTAs per SM do not change
    assert has(rc.step_problems(single(h["threads"]), None, META, repo), "the launch in the sources")                       # the sources say 128 threads, the records 256
    assert has(rc.step_problems(single(h["dynamic"]), None, META, repo), "the launch in the sources")                       # the sources say 4096 B, the records 0
    assert has(rc.step_problems(single(h["bare"]), None, META, repo), "the launch cannot be read from the sources")
    assert has(rc.step_problems(single(h["launch"]), None, META, tmp_path / "nowhere"), "the launch cannot be read from the sources")
    ledger_row = {"schema": rc.LEDGER_SCHEMA, "meta": dict(META), "steps": [single(h["threads"])], "_path": "mem"}
    assert has(rc.ledger_problems(ledger_row, repo=repo), "the launch in the sources")         # the ledger hands its repository to the rows


ADOPTED_MUTANTS = [
    ("no gates", lambda s: s.update(gates=None), "passes the exactness and the code-generation gate"),
    ("a failed exactness gate", lambda s: s["gates"].update(exactness=False), "passes the exactness"),
    ("a failed code-generation gate", lambda s: s["gates"].update(codegen=False), "passes the exactness"),
    ("no code-generation evidence", lambda s: s.update(codegen_gate=None), "codegen_gate must give one comparison of dumps per toolchain"),
    ("code-generation evidence for one toolchain only", lambda s: s["codegen_gate"].pop("container_b300"), "codegen_gate has no comparison for container_b300"),
    ("an entry that is not an object", lambda s: s["codegen_gate"].update(conda="PASS"), "is not an object"),
    ("no resource evidence", lambda s: s.update(resource_gate=None), "resource_gate must give the resource figures per toolchain"),
    ("an unknown toolchain", lambda s: s["codegen_gate"].update(unknown={}), "one comparison of dumps per toolchain"),
    ("no A/B medians", lambda s: s.update(ab=None), "ab must give the A/B medians"),
    ("A/B medians for one device only", lambda s: s["ab"].pop("b300"), "ab must give the A/B medians"),
    ("an A/B median that is zero", lambda s: s["ab"]["b200"]["medians"][CELL].update(cand=0), "positive numbers"),
    ("no list of noise-limited cells", lambda s: s["ab"]["b200"].pop("noise_limited"), "noise-limited"),
    ("no commit", lambda s: s.update(commit=None), "names its commit"),
    ("a commit that is not a hash", lambda s: s.update(commit="HEAD"), "hexadecimal"),
    ("a revert commit", lambda s: s.update(revert_commit=C4), "no revert commit"),
    ("a step that says where it failed", lambda s: s.update(failed_at="timing"), "failed at nothing"),
    ("a step that does not beat the previous state", lambda s: s["timing"].update(vs_previous="fail"), "beat the previous accepted state"),
    ("a step that regresses against the base", lambda s: s["timing"].update(vs_base="fail"), "not regress against the base"),
    ("a step that breaks the layer chain", lambda s: s["timing"].update(chain="fail"), "keep the layer chain"),
    ("a timing that was not run", lambda s: s["timing"].update(vs_base="not_run"), "measured"),
    ("no timing", lambda s: s.update(timing=None), "has timing results"),
    ("no markers", lambda s: s.update(markers=[]), "markers must be a non-empty list"),
]


@pytest.mark.parametrize("name,edit,expected", ADOPTED_MUTANTS, ids=[m[0] for m in ADOPTED_MUTANTS])
def test_an_adopted_row_with_missing_or_failing_evidence_is_rejected(name, edit, expected):
    good = adopted("kd_tile")
    assert step_check(good) == []
    assert has(step_check(mutated(good, edit)), expected), (name, step_check(mutated(good, edit))[:3])


@pytest.mark.parametrize("failed_at", rc.GATE_PHASES)
def test_a_step_that_was_not_adopted_carries_the_evidence_of_the_phases_it_reached(failed_at):
    good = not_adopted("kd_tile", failed_at)
    assert step_check(good) == []
    assert step_check(not_adopted("kd_tile", failed_at, C2, C3)) == []                              # committed and reverted
    assert has(step_check(mutated(good, lambda s: s.update(reason=""))), "carries a reason")
    assert has(step_check(mutated(good, lambda s: s.update(commit=C2))), "is not reverted")
    assert has(step_check(mutated(good, lambda s: s.update(failed_at="later"))), "says where it failed")
    assert has(step_check(mutated(good, lambda s: s["gates"].update(exactness="x"))), "gate is")


def test_the_measurements_a_step_never_reached_are_not_run_and_cannot_be_invented():
    """compile / exactness: no evidence at all; codegen: dumps but no timing; timing: everything and at least one failing result."""
    codegen_evidence = not_adopted("kd_tile", "codegen")
    compiled = not_adopted("kd_tile", "compile")
    assert has(step_check(mutated(compiled, lambda s: s.update(codegen_gate=codegen_evidence["codegen_gate"]))), "never reached")
    exactness = not_adopted("kd_tile", "exactness")
    assert has(step_check(mutated(exactness, lambda s: s.update(resource_gate=codegen_evidence["resource_gate"]))), "never reached")
    assert has(step_check(mutated(exactness, lambda s: s["gates"].update(exactness=True))), "the exactness gate is False")
    assert has(step_check(mutated(exactness, lambda s: s.update(timing={"vs_previous": "fail", "vs_base": "pass", "chain": "pass"}))), "no timing was measured")
    assert has(step_check(mutated(codegen_evidence, lambda s: s.update(ab=ab_figures()))), "never reached")
    assert has(step_check(mutated(codegen_evidence, lambda s: s.update(codegen_gate=None, resource_gate=None))), "one comparison of dumps per toolchain")
    assert has(step_check(mutated(codegen_evidence, lambda s: s["gates"].update(codegen=True))), "the codegen gate is False")
    timing = not_adopted("kd_tile", "timing")
    assert has(step_check(mutated(timing, lambda s: s.update(ab=None))), "ab must give the A/B medians")
    assert has(step_check(mutated(timing, lambda s: s["codegen_gate"].pop("container_b200"))), "no comparison for container_b200")
    assert has(step_check(mutated(timing, lambda s: s.update(timing={"vs_previous": "pass", "vs_base": "pass", "chain": "pass"}))), "a step that passes every gate is adopted")
    assert has(step_check(mutated(timing, lambda s: s.update(timing={"vs_previous": "slow", "vs_base": "pass", "chain": "pass"}))), "measured (pass or fail)")
    assert has(step_check(mutated(timing, lambda s: s.update(timing={"vs_previous": "fail", "vs_base": "not_run", "chain": "pass"}))), "measured (pass or fail)")
    for results in ({"vs_previous": "fail", "vs_base": "pass", "chain": "pass"}, {"vs_previous": "pass", "vs_base": "fail", "chain": "pass"}, {"vs_previous": "pass", "vs_base": "pass", "chain": "fail"}):
        assert step_check(mutated(timing, lambda s: s.update(timing=results))) == [], results


def test_a_step_that_failed_at_the_code_generation_fails_a_comparison_of_its_dumps():
    failed = not_adopted("kd_tile", "codegen")
    assert step_check(failed) == []                                                 # 129 registers on the conda toolchain: the dumps say FAIL, the row says FAIL
    passing = not_adopted("kd_tile", "codegen")
    passing["codegen_gate"], passing["resource_gate"] = gate_evidence("e" * 40, keys=("conda",))
    assert has(step_check(passing), "failed at the code generation, but every comparison of the dumps passes")
    lying = mutated(failed, lambda s: s["codegen_gate"]["conda"].update(verdict="PASS"))
    assert has(step_check(lying), "the comparison of the dumps says FAIL")


def test_a_skipped_step_has_a_reason_and_no_evidence():
    assert step_check(skipped("trims")) == []
    assert has(step_check(mutated(skipped("trims"), lambda s: s.update(reason=""))), "carries the reason")
    assert has(step_check(mutated(skipped("trims"), lambda s: s.update(commit=C1))), "has no commit")
    assert has(step_check(mutated(skipped("trims"), lambda s: s.update(ab=ab_figures()))), "has no ab")
    assert has(step_check(skipped("single_launch")), "only a row-engine step can be skipped")
    assert has(step_check(skipped("kd_tile")), "only a row-engine step can be skipped")


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
    good = {"schema": rc.LEDGER_SCHEMA, "meta": dict(META), "steps": [adopted("single_launch", C1), adopted("trims", C5), skipped("e_tile"), adopted("a_fragments", C3, vs_previous="pass"),
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
    commit_file(repo, "csrc/cfbi/cfbi_primitives.cuh", PRIMITIVES, "the sizes of the kernels")                 # the launch of the row engine, which the occupancy records are read against
    commit_file(repo, "csrc/cfbi/cfbi_launch.cu", LAUNCH, "the launch of the row engine")
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
    ok = {"schema": rc.LEDGER_SCHEMA, "meta": dict(META), "steps": [row()], "_path": "mem"}
    assert rc.ledger_problems(ok, repo=repo) == []
    assert has(rc.ledger_problems(mutated(ok, lambda l: l["steps"][0].update(commit="0" * 40)), repo=repo), "does not have")
    side = steps_repo_rw["repo"]
    git(side, "checkout", "-q", "-b", "elsewhere", h["base"])
    off = commit_file(side, "csrc/e.cu", "not on main\n", "a commit on another line")
    git(side, "checkout", "-q", "main")
    assert has(rc.ledger_problems(mutated(ok, lambda l: l["steps"][0].update(commit=off)), repo=repo), "is not an ancestor")


def test_the_decision_record_is_written_before_the_first_row_engine_commit(steps_repo_rw):
    """The order of the steps is recorded before any of them is committed: a step commit must come after the commit that the decision record was written on."""
    repo, h = steps_repo_rw["repo"], steps_repo_rw
    later = commit_file(repo, "csrc/after.cu", "a row-engine step\n", "a step after the record")
    ok = {"schema": rc.LEDGER_SCHEMA, "meta": dict(META), "steps": [adopted("trims", later)], "_path": "mem"}
    dec = dict(decision(), meta={"recorded_on": h["base"]})
    assert rc.ledger_problems(ok, repo=repo, decision=dec) == []
    early = mutated(ok, lambda l: l["steps"][0].update(commit=h["base"]))                                  # the step is the commit that the record was written on
    assert has(rc.ledger_problems(early, repo=repo, decision=dec), "does not come after the decision record")
    after = dict(decision(), meta={"recorded_on": later})                                                    # the record was written after the step
    assert has(rc.ledger_problems(ok, repo=repo, decision=after), "does not come after the decision record")
    assert rc.ledger_problems(ok, repo=repo, decision=decision()) == []                                      # a decision record without the commit asks nothing of the history
    unknown = dict(decision(), meta={"recorded_on": "0" * 40})                                                # a commit that the repository does not have is not skipped silently
    assert has(rc.ledger_problems(ok, repo=repo, decision=unknown), "which the repository does not have")
    other = {"schema": rc.LEDGER_SCHEMA, "meta": dict(META), "steps": [adopted("single_launch", h["base"])], "_path": "mem"}
    assert rc.ledger_problems(other, repo=repo, decision=dec) == []                                          # the single-launch step comes before the record, by the plan's order


def test_the_ledger_and_decision_command_lines(steps_repo, tmp_path, capsys):
    rec = ledger(0)
    rec.pop("_path")
    good, bad, dec = tmp_path / "ledger.json", tmp_path / "ledger-bad.json", tmp_path / "decision.json"
    good.write_text(json.dumps(rec))
    e, f = unit_rows(UNIT_CASES[1][1], ("adopted", "adopted"))                               # F kept although it did not beat E
    bad.write_text(json.dumps(dict(ledger(1, e_tile=e, a_fragments=f), _path=None)))
    dec.write_text(json.dumps(decision()))
    assert rc.main(["ledger", "--ledger", str(good), "--require-all-steps"]) == 0
    assert rc.main(["ledger", "--ledger", str(good), "--decision", str(dec), "--require-all-steps"]) == 0
    assert rc.main(["decision", "--record", str(dec)]) == 0
    assert rc.main(["ledger", "--ledger", str(bad)]) == 1
    assert "the three-arm rule prescribes not_adopted for a_fragments" in capsys.readouterr().out
    assert rc.main(["ledger", "--ledger", str(good), "--repo", str(steps_repo["repo"])]) == 1      # the commits of the synthetic ledger are not in this repository
    assert "does not have" in capsys.readouterr().out
    # the review's counterexamples through the command line: the selected F row regressing against the base, 129 registers in the dump of a row, 100 differences that are none
    cases = (("regressing", lambda l: l["steps"][F_ROW]["timing"].update(vs_base="fail", chain="fail"), "not regress against the base"),
             ("e-chain", lambda l: l["steps"][E_ROW]["timing"].update(chain="not_run"), "layer chain is measured"),
             ("differences", lambda l: l["steps"][E_ROW]["codegen_gate"]["container_b200"].update(differences=100), "differences = 100"),
             ("figure", lambda l: l["steps"][F_ROW]["resource_gate"]["container_b200"]["sm_100a"][ROWS].update(reg=129), "resource_gate (container_b200, sm_100a, cfbi::rows_kernel) says"))
    for name, edit, expected in cases:
        path = tmp_path / ("ledger-%s.json" % name)
        path.write_text(json.dumps(mutated(rec, edit)))
        capsys.readouterr()
        assert rc.main(["ledger", "--ledger", str(path), "--require-all-steps"]) == 1, name
        assert expected in capsys.readouterr().out, name


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
PERF_DOC = ("# Performance\n\n## Results\nNumbers.\n\n## Not done / known limits\nThe cell %s is slower than the base on B200 with every change reverted too (the node, not the code).\n\n"
            "## Appendix\nx\n")
GPU = {"b200": ("NVIDIA B200", 148), "b300": ("NVIDIA B300 SXM6 AC", 148)}
DECODE_KEYS, CHAIN_KEYS = rc.canonical_cells("decode"), rc.canonical_cells("chain")
MEDIANS = {"decode": dict((k, 14.0 + 0.5 * i) for i, k in enumerate(DECODE_KEYS)), "chain": dict((k, 90.0 + 25.0 * i) for i, k in enumerate(CHAIN_KEYS))}
SLOW_CELL = "decode;N=64;H=12;regime=lockstep;"
SLOW_METRIC = "gpu_graph_us"                       # the metric that a regression of SLOW_CELL shows in
LOOP_SUBSTRINGS = tuple("regime=%s;" % r for r in rc.bd.LOOP_REGIMES)
KIND = {"decode": "decode-wall", "chain": "chain-wall"}                       # the kind of record that the benchmark of a campaign writes
SPECS = {"decode": dict((s["key"], s) for s in rc.bd.cell_specs()), "chain": dict((s["key"], s) for s in rc.bl.cell_specs())}
ROUTE = {"decode": {"kernels": ["plan_kernel", "rows_kernel"]}, "chain": {"compute_nodes": {"plan_kernel": 1, "rows_kernel": 1}, "aux_nodes": {}}}      # what a cell launched (base routing)


def workload_args(campaign, key):
    """The cell arguments that name what a cell measures, as the benchmark that writes the cell sets them (a decode cell: N, H, regime, pos; a chain cell: N, H, Dm, regime)."""
    s = SPECS[campaign][key]
    return {"N": s["N"], "H": s["H"], "regime": s["regime"], "pos": s["pos"]} if campaign == "decode" else {"N": s["N"], "H": s["H"], "Dm": s["Dm"], "regime": "lockstep"}


@pytest.fixture(scope="module")
def final_repo(tmp_path_factory):
    """A history: the base, the measured final candidate, a docs-only commit after it, and (on another line) a source change after the measurement."""
    repo = tmp_path_factory.mktemp("final") / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    h = {"repo": repo}
    h["base"] = commit_file(repo, "csrc/k.cu", "base\n", "base")
    h["final"] = commit_file(repo, "csrc/k.cu", "the final candidate\n", "the final candidate")
    h["docs"] = commit_file(repo, "docs/perf.md", PERF_DOC % SLOW_CELL, "the performance document, committed after the records it quotes")
    git(repo, "checkout", "-q", "-b", "later", h["docs"])
    h["source"] = commit_file(repo, "csrc/k.cu", "changed again after the measurement\n", "a source change")
    git(repo, "checkout", "-q", "main")
    return h


def final_cell(campaign, key, median, slow=1.0, scale=1, spread=0.0, idx=0):
    """A cell of a final campaign with the wall metrics of its kind (the host loop metrics only in the loop regimes, a chain cell has the graph ones) and a kernel sum; ``slow`` scales the
    ``gpu_graph_us`` of the cell only; ``scale``: the factor of a three-times campaign on the benchmark's default calls; ``spread``: the relative step of the per-process median from one
    record file of a side to the next (``idx`` is the file's index): the spread that Method M judges is the one between the files, not the one between the trials of a file."""
    median = median * (1.0 + spread * idx)
    loop = campaign == "decode" and any(x in key for x in LOOP_SUBSTRINGS)
    wall = []
    for t in tbt.trials(median, spread):
        w = {"gpu_graph_us": t * slow}
        if campaign == "decode":
            w["gpu_eager_us"] = t + 1.0
            if loop:
                w.update(cpu_graph_us=t + 3.0, cpu_eager_us=t + 4.0)
        else:
            w["cpu_graph_us"] = t + 3.0
        wall.append(w)
    runs = [{tbt.PLAN_K: 2.0, tbt.ROWS_K: 10.0, tbt.FILL_K: 40.0} for _ in wall]
    args = dict(workload_args(campaign, key), impl=0, calls_per_trial=scale * tbt.CALLS, profile_calls=scale * tbt.PROFILE_CALLS, calls_scale=scale, **ROUTE[campaign])
    return br.make_cell(campaign, key, args, runs, drop_helpers=True, wall=wall)


def final_rec(campaign, dev, commit, ts, host, slows=None, scale=1, cells=None, spreads=None, idx=0):
    gpu, sm = GPU[dev]
    keys = cells if cells is not None else (DECODE_KEYS if campaign == "decode" else CHAIN_KEYS)
    meta = {"commit": commit, "dirty": False, "gpu": gpu, "sm_count": sm, "host": host, "timestamp": ts, "sample_stat": br.PROCESS_MEDIAN, "trials": 5, "kind": KIND[campaign], "impl": 0,
            "env_impl": None}
    return {"schema": br.SCHEMA, "meta": meta, "cells": [final_cell(campaign, k, MEDIANS[campaign][k], (slows or {}).get(k, 1.0), scale, (spreads or {}).get(k, 0.0), idx) for k in keys]}


def campaign_files(d, name, campaign, dev, a_commit, b_commit, host, allocation, slows=None, cells=None, scale=1, spreads=None, rerun_cells=None, rerun_slows=None, scope=None):
    """One campaign: three interleaved records per side (``cells``: the cells that they hold, all of them by default), a re-run of ``rerun_cells`` with three times the calls when asked for,
    and the comparison report that ``compare_bench --json`` would write for it (made by the comparator with the driver's parameters); the entry of the final record that names them.
    ``scope``: ``{metric: [cells]}`` for a campaign that judges flagged cells only, else the complete campaign."""
    def make(side, commit, hour, slow, sc, keys, sp, offset):
        return [final_rec(campaign, dev, commit, "2026-11-10T%s:%02d:00Z" % (hour, 2 * i + offset), host, slows=slow, scale=sc, cells=keys, spreads=sp, idx=i) for i in range(3)]

    a1, b1 = make("A", a_commit, "10", None, scale, cells, spreads, 0), make("B", b_commit, "10", slows, scale, cells, spreads, 1)
    a2 = make("A", a_commit, "11", None, 3 * scale, rerun_cells, None, 0) if rerun_cells else []
    b2 = make("B", b_commit, "11", rerun_slows if rerun_slows is not None else slows, 3 * scale, rerun_cells, None, 1) if rerun_cells else []
    files = {"A": tbt.write_records(str(d), name + "-A1", a1), "B": tbt.write_records(str(d), name + "-B1", b1)}
    again = {"A": tbt.write_records(str(d), name + "-A2", a2) if a2 else [], "B": tbt.write_records(str(d), name + "-B2", b2) if b2 else []}
    wanted = rc.canonical_coverage(campaign) if scope is None else scope
    results = [br.compare(a1, b1, base_rerun=a2 or None, cand_rerun=b2 or None, **rc.compare_kwargs(m, rc.TOLERANCE_PCT, a_commit, rc.metric_cells(campaign, m) if scope is None else list(scope[m])))
               for m in sorted(wanted)]
    report = Path(d) / (name + "-report.json")
    report.write_text(json.dumps(results, default=str))
    return {"report": str(report), "records_first": files, "records_rerun": again, "allocation": allocation}


def final_record(d, h, slow=None, known_limits=(), host="node-a"):
    """The final record of both devices (each with a decode and a chain campaign of the measured commit against the base on the benchmark's own cells) and its baselines."""
    d = Path(d)
    devices = {}
    for dev in rc.DEVICES:
        baseline = d / ("baseline-%s.json" % dev)
        gpu, sm = GPU[dev]
        baseline.write_text(json.dumps({"commit": h["base"], "gpu": gpu, "sm_count": sm,
                                        "decode_matrix": {"cells": sorted(DECODE_KEYS), "metrics": list(rc.CANONICAL_METRICS["decode"]), "bench": "decode", "tolerance_pct": 1.0},
                                        "layer_chain": {"cells": sorted(CHAIN_KEYS), "metrics": list(rc.CANONICAL_METRICS["chain"]), "bench": "chain", "tolerance_pct": 1.0}}))
        node = "%s-%s" % (host, dev)
        campaigns = {"decode": campaign_files(d, "%s-decode" % dev, "decode", dev, h["base"], h["final"], node, "sc-%s-1" % dev, slows=(slow or {}).get(dev)),
                     "chain": campaign_files(d, "%s-chain" % dev, "chain", dev, h["base"], h["final"], node, "sc-%s-1" % dev)}
        devices[dev] = {"baseline": str(baseline), "campaigns": campaigns}
    return {"schema": rc.FINAL_SCHEMA, "meta": {"commit": h["final"], "base": h["base"], "dirty": False}, "devices": devices, "known_limits": list(known_limits), "_path": "mem"}


def known_limit(d, h, dev="b200", cell=SLOW_CELL, factor=1.08, confirm_host="node-b", confirm_alloc="sc-b200-2", reverted_factor=1.08, confirm_scale=3, confirm_cells=None, name="limit"):
    """The known-limit entry for a regression of ``cell``: its confirmation (the flagged cell only, with three times the calls, on another node and allocation) and the run with every change
    reverted (the candidate side is the base again)."""
    scope = {SLOW_METRIC: [cell]}
    conf = campaign_files(d, "%s-%s-confirm" % (name, dev), "decode", dev, h["base"], h["final"], confirm_host, confirm_alloc, slows={cell: factor}, cells=confirm_cells or [cell], scale=confirm_scale,
                          scope=scope if confirm_cells is None else None)
    rev = campaign_files(d, "%s-%s-reverted" % (name, dev), "decode", dev, h["base"], h["base"], "node-c", "sc-b200-3", slows={cell: reverted_factor}, cells=[cell], scope=scope)
    return {"device": dev, "campaign": "decode", "metric": SLOW_METRIC, "cell": cell, "confirmation": conf, "all_changes_reverted": rev}


def fcheck(rec, h, head=None, document=PERF_DOC % SLOW_CELL):
    return rc.final_problems(rec, h["repo"], head or h["docs"], document)


def test_a_complete_final_record_passes_against_a_docs_only_history(tmp_path, final_repo):
    rec = final_record(tmp_path, final_repo)
    assert fcheck(rec, final_repo) == []
    assert fcheck(rec, final_repo, head=final_repo["final"]) == []                         # HEAD is the measured commit itself


def test_the_coverage_is_the_benchmarks_own(tmp_path, final_repo):
    """38 decode cells (the matrix in three regimes and the probe positions) and the host loop metrics on the 12 loop cells; two chain cells; five and three metrics."""
    cov = rc.canonical_coverage("decode")
    assert len(DECODE_KEYS) == 38 and sorted(cov) == sorted(br.WALL_METRICS + ("total_us",))
    assert len(cov["gpu_graph_us"]) == 38 and len(cov["cpu_graph_us"]) == 12 and len(cov["cpu_eager_us"]) == 12 and all(any(x in k for x in LOOP_SUBSTRINGS) for k in cov["cpu_graph_us"])
    assert len(CHAIN_KEYS) == 2 and sorted(rc.canonical_coverage("chain")) == ["cpu_graph_us", "gpu_graph_us", "total_us"]
    rec = final_record(tmp_path, final_repo)
    assert len(json.loads(Path(rec["devices"]["b200"]["campaigns"]["decode"]["report"]).read_text())) == 5


def test_the_record_commit_rule_and_the_clean_tree_apply_to_the_final_record(tmp_path, final_repo):
    h = final_repo
    rec = final_record(tmp_path, h)
    assert has(fcheck(mutated(rec, lambda r: r["meta"].update(commit="0000000")), h), "does not have")                   # a commit that does not exist
    assert has(fcheck(rec, h, head=h["source"]), "outside docs/")                                                        # a source change since the measured commit
    assert has(rc.final_problems(rec, None, "HEAD", PERF_DOC % SLOW_CELL), "repo is required")
    assert has(fcheck(mutated(rec, lambda r: r["meta"].update(base="x")), h), "meta.base")
    assert has(fcheck(mutated(rec, lambda r: r["meta"].update(dirty=True)), h), "dirty")
    tracked = Path(h["repo"]) / "csrc" / "k.cu"
    original = tracked.read_text()
    try:
        tracked.write_text("an uncommitted change\n")
        assert has(fcheck(rec, h), "not clean")
    finally:
        tracked.write_text(original)
    assert fcheck(rec, h) == []


# ---- the round-3 review's counterexamples: the final record is recomputed from the raw records ----------------------------------------------------------------
def rewrite_records(entry, side, edit):
    """Rewrite every raw record file of ``side`` (first campaign and re-run) of a campaign entry after ``edit(record)``."""
    for path in entry["records_first"][side] + entry["records_rerun"][side]:
        rec = json.loads(Path(path).read_text())
        edit(rec)
        Path(path).write_text(json.dumps(rec))


def double_samples(rec):
    for cell in rec["cells"]:
        for sm in cell["samples"]:
            for k in list(sm):
                if k.endswith("_us") and isinstance(sm[k], (int, float)):
                    sm[k] *= 2.0


def test_doubled_raw_candidate_samples_with_an_unchanged_report_are_rejected(tmp_path, final_repo):
    """The comparator on the raw records says FAIL (every cell +100 %); the report still says PASS: the report is a claim that is recomputed, not a source."""
    rec = final_record(tmp_path, final_repo)
    assert fcheck(rec, final_repo) == []
    entry = rec["devices"]["b200"]["campaigns"]["decode"]
    rewrite_records(entry, "B", double_samples)
    probs = fcheck(rec, final_repo)
    assert has(probs, "the report differs from the raw records") and has(probs, "has no known-limit entry")


def test_one_raw_record_per_side_while_the_report_claims_three_is_rejected(tmp_path, final_repo):
    rec = final_record(tmp_path, final_repo)
    entry = rec["devices"]["b300"]["campaigns"]["chain"]
    entry["records_first"] = {"A": entry["records_first"]["A"][:1], "B": entry["records_first"]["B"][:1]}
    probs = fcheck(rec, final_repo)
    assert has(probs, "the comparison recomputed from the raw records is INVALID") and has(probs, "at least 3 interleaved repeats")


def test_raw_cells_missing_in_every_file_are_rejected_whatever_the_report_claims(tmp_path, final_repo):
    rec = final_record(tmp_path, final_repo)
    entry = rec["devices"]["b200"]["campaigns"]["decode"]
    dropped = DECODE_KEYS[-3:]
    rewrite_records(entry, "A", lambda r: r.update(cells=[c for c in r["cells"] if c["key"] not in dropped]))
    rewrite_records(entry, "B", lambda r: r.update(cells=[c for c in r["cells"] if c["key"] not in dropped]))
    probs = fcheck(rec, final_repo)
    assert has(probs, "holds 35 cell(s), the benchmark's definition holds 38") and has(probs, "missing %s" % sorted(dropped)[:3])


def row_of(results, key=SLOW_CELL):
    return next(r for r in results[0]["rows"] if r["key"] == key)


def test_a_self_consistent_invented_report_row_is_rejected(tmp_path, final_repo):
    """cand = 2 x base, difference 100 %, effective tolerance 1000 %, status PASS: consistent in itself, not what the raw records give."""
    rec = final_record(tmp_path, final_repo)
    entry = rec["devices"]["b200"]["campaigns"]["decode"]
    edit_report(entry, lambda res: row_of(res).update(cand_median=row_of(res)["base_median"] * 2, median_delta_pct=100.0, delta_pct=100.0, tol_pct_effective=1000.0, status="PASS"))
    assert has(fcheck(rec, final_repo), "the report differs from the raw records in")


def test_raw_records_of_another_gpu_under_a_device_entry_are_rejected(tmp_path, final_repo):
    """The round-3 positive fixture accepted NVIDIA B200 raw records under the b300 entry."""
    rec = final_record(tmp_path, final_repo)
    entry = rec["devices"]["b300"]["campaigns"]["decode"]
    rewrite_records(entry, "A", lambda r: r["meta"].update(gpu="NVIDIA B200"))
    rewrite_records(entry, "B", lambda r: r["meta"].update(gpu="NVIDIA B200"))
    probs = fcheck(rec, final_repo)
    assert has(probs, "is of the GPU 'NVIDIA B200', the device is b300") and has(probs, "the baseline's device is")


def test_a_baseline_that_declares_less_than_the_benchmarks_coverage_is_not_an_authority(tmp_path, final_repo):
    """Three decode cells and one chain cell with a metric that decode does not use were accepted as the baseline's coverage."""
    rec = final_record(tmp_path, final_repo)
    path = Path(rec["devices"]["b200"]["baseline"])
    thin = json.loads(path.read_text())
    thin["decode_matrix"].update(cells=DECODE_KEYS[:3], metrics=["k2_stage_us"])
    thin["layer_chain"].update(cells=CHAIN_KEYS[:1])
    path.write_text(json.dumps(thin))
    probs = fcheck(rec, final_repo)
    assert has(probs, "the baseline's metrics ['k2_stage_us'] are not the campaign's") and has(probs, "the baseline's cells are not the benchmark's 38 cells") and has(probs, "not the benchmark's 2 cells")
    path.write_text(json.dumps(dict(thin, gpu="NVIDIA B300 SXM6 AC")))
    assert has(fcheck(rec, final_repo), "the baseline is of the GPU 'NVIDIA B300 SXM6 AC', the device is b200")


def edit_report(entry, edit):
    """Rewrite a campaign's report file after ``edit(results)``."""
    path = Path(entry["report"])
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize("name,edit,expected", [
    ("a candidate median that contradicts the declared difference", lambda res: row_of(res).update(cand_median=row_of(res)["cand_median"] * 2), "the report differs from the raw records in"),
    ("a difference that is missing", lambda res: row_of(res).pop("median_delta_pct"), "the report differs from the raw records in"),
    ("a status that the raw records do not give", lambda res: row_of(res).update(status="FAIL"), "the report differs from the raw records in"),
    ("a spread that is invented", lambda res: row_of(res).update(base_spread_pct=2.5), "the report differs from the raw records in"),
    ("a cell judged on another campaign", lambda res: row_of(res).update(judged_on="rerun"), "the report differs from the raw records in"),
    ("a cell that is missing", lambda res: res[0]["rows"].pop(0), "the report has"),
    ("a cell that is not one of the benchmark's", lambda res: res[0]["rows"].append(dict(row_of(res), key="decode;N=9;H=9;regime=lockstep;")), "unexpected"),
    ("a cell twice", lambda res: res[0]["rows"].append(dict(row_of(res))), "not unique"),
    ("a report with another metric", lambda res: res[0].update(metric="k1_us"), "the report has the metrics"),
    ("a report without a metric", lambda res: res.pop(), "the report has the metrics"),
    ("a verdict that the raw records do not give", lambda res: res[0].update(verdict="FAIL"), "the report says verdict"),
    ("a comparison of another mode", lambda res: res[0].update(mode="within"), "the report says mode"),
    ("another tolerance", lambda res: res[0].update(tol_pct=5.0), "the report says tol_pct"),
    ("an unstable policy that is not the driver's", lambda res: res[0].update(unstable_policy="invalid"), "the report says unstable_policy"),
    ("a report that is not a list", lambda res: res.append(3), "not a comparison report"),
])
def test_a_final_report_that_contradicts_the_raw_records_is_rejected(tmp_path, final_repo, name, edit, expected):
    rec = final_record(tmp_path, final_repo)
    assert fcheck(rec, final_repo) == []
    edit_report(rec["devices"]["b300"]["campaigns"]["decode"], edit)
    assert has(fcheck(rec, final_repo), expected), (name, fcheck(rec, final_repo)[:3])


def test_a_report_that_is_not_a_report_or_is_gone_is_rejected(tmp_path, final_repo):
    rec = final_record(tmp_path, final_repo)
    Path(rec["devices"]["b200"]["campaigns"]["chain"]["report"]).write_text('{"not": "a list"}')
    assert has(fcheck(rec, final_repo), "is not a comparison report")
    rec["devices"]["b300"]["campaigns"]["chain"]["report"] = str(tmp_path / "gone.json")
    assert has(fcheck(rec, final_repo), "cannot be read")


def test_the_sides_of_the_raw_records_are_given_not_guessed_and_carry_the_commits_they_must(tmp_path, final_repo):
    h = final_repo
    rec = final_record(tmp_path, h)
    entry = rec["devices"]["b200"]["campaigns"]["decode"]
    flat = dict(entry, records=entry["records_first"]["A"] + entry["records_first"]["B"])
    flat.pop("records_first")
    flat.pop("records_rerun")
    assert has(fcheck(mutated(rec, lambda r: r["devices"]["b200"]["campaigns"].update(decode=flat)), h), "listed as records_first and records_rerun")
    swapped = dict(entry, records_first={"A": entry["records_first"]["B"], "B": entry["records_first"]["A"]})
    assert has(fcheck(mutated(rec, lambda r: r["devices"]["b200"]["campaigns"].update(decode=swapped)), h), "side A must be")
    one_side = dict(entry, records_first={"A": entry["records_first"]["A"], "B": []})
    assert has(fcheck(mutated(rec, lambda r: r["devices"]["b200"]["campaigns"].update(decode=one_side)), h), "must name the raw records of both sides")
    gone = dict(entry, records_first={"A": entry["records_first"]["A"], "B": entry["records_first"]["B"] + [str(tmp_path / "gone.json")]})
    assert has(fcheck(mutated(rec, lambda r: r["devices"]["b200"]["campaigns"].update(decode=gone)), h), "cannot be read again")
    older = campaign_files(tmp_path, "older", "decode", "b200", h["base"], "a" * 40, "node-a-b200", "sc-b200-1")                      # the candidate side of an older tree
    assert has(fcheck(mutated(rec, lambda r: r["devices"]["b200"]["campaigns"].update(decode=older)), h), "side B must be")
    other_host = campaign_files(tmp_path, "otherhost", "decode", "b200", h["base"], h["final"], "node-z", "sc-b200-1")
    mixed = dict(entry, records_first={"A": entry["records_first"]["A"], "B": other_host["records_first"]["B"]})
    assert has(fcheck(mutated(rec, lambda r: r["devices"]["b200"]["campaigns"].update(decode=mixed)), h), "hosts")
    rewrite_records(entry, "B", lambda r: r["meta"].update(dirty=True))
    assert has(fcheck(rec, h), "dirty")


def test_a_final_record_needs_both_devices_both_campaigns_and_the_baselines(tmp_path, final_repo):
    h = final_repo
    rec = final_record(tmp_path, h)
    assert has(fcheck(mutated(rec, lambda r: r["devices"].pop("b300")), h), "devices must give the final A/B")
    assert has(fcheck(mutated(rec, lambda r: r["devices"]["b200"]["campaigns"].pop("chain")), h), "the campaign evidence is missing")
    assert has(fcheck(mutated(rec, lambda r: r["devices"]["b200"]["campaigns"]["decode"].pop("allocation")), h), "allocation")
    assert has(fcheck(mutated(rec, lambda r: r["devices"]["b200"].update(baseline=str(tmp_path / "gone.json"))), h), "consolidated baseline")
    wrong = tmp_path / "wrong-base.json"
    wrong.write_text(json.dumps({"commit": "9" * 40, "decode_matrix": {}, "layer_chain": {}}))
    assert has(fcheck(mutated(rec, lambda r: r["devices"]["b200"].update(baseline=str(wrong))), h), "the base of the loop is")


def test_an_unstable_cell_needs_its_rerun_with_three_times_the_calls_and_a_legal_one_passes(tmp_path, final_repo):
    h = final_repo
    cell = DECODE_KEYS[3]
    spreads = {cell: 0.01}                                                    # the per-process medians of a side step by 1 %: a 2 % spread, unstable in the first campaign
    assert fcheck(final_record(tmp_path, h), h) == []
    without = campaign_files(tmp_path, "norerun", "decode", "b200", h["base"], h["final"], "node-a-b200", "sc-b200-1", spreads=spreads)
    rec = mutated(final_record(tmp_path, h), lambda r: r["devices"]["b200"]["campaigns"].update(decode=without))
    assert has(fcheck(rec, h), "the comparison recomputed from the raw records is INVALID")
    legal = campaign_files(tmp_path, "rerun", "decode", "b200", h["base"], h["final"], "node-a-b200", "sc-b200-1", spreads=spreads, rerun_cells=[cell])
    assert fcheck(mutated(rec, lambda r: r["devices"]["b200"]["campaigns"].update(decode=legal)), h) == []
    scale_one = campaign_files(tmp_path, "rerun1", "decode", "b200", h["base"], h["final"], "node-a-b200", "sc-b200-1", spreads=spreads, rerun_cells=[cell])
    rewrite_records(scale_one, "A", lambda r: [c["args"].update(calls_per_trial=tbt.CALLS, profile_calls=tbt.PROFILE_CALLS) for c in r["cells"] if r["meta"]["timestamp"].startswith("2026-11-10T11")])
    rewrite_records(scale_one, "B", lambda r: [c["args"].update(calls_per_trial=tbt.CALLS, profile_calls=tbt.PROFILE_CALLS) for c in r["cells"] if r["meta"]["timestamp"].startswith("2026-11-10T11")])
    assert has(fcheck(mutated(rec, lambda r: r["devices"]["b200"]["campaigns"].update(decode=scale_one)), h), "recomputed from the raw records is INVALID")


def test_a_rerun_file_holds_some_of_the_campaigns_cells_and_none_that_it_does_not_have(tmp_path, final_repo):
    h = final_repo
    cell = DECODE_KEYS[3]
    entry = campaign_files(tmp_path, "rerunfiles", "decode", "b200", h["base"], h["final"], "node-a-b200", "sc-b200-1", spreads={cell: 0.01}, rerun_cells=[cell])
    rec = mutated(final_record(tmp_path, h), lambda r: r["devices"]["b200"]["campaigns"].update(decode=entry))
    assert fcheck(rec, h) == []                                                                   # a legal re-run of one cell: the file holds that cell only
    path = entry["records_rerun"]["A"][0]
    data = json.loads(Path(path).read_text())
    original = Path(path).read_text()
    data["cells"] = []
    Path(path).write_text(json.dumps(data))
    assert has(fcheck(rec, h), "holds no cell")
    data["cells"] = json.loads(original)["cells"] + [dict(json.loads(original)["cells"][0], key="decode;N=9;H=9;regime=lockstep;")]
    Path(path).write_text(json.dumps(data))
    assert has(fcheck(rec, h), "holds cell(s) that the benchmark's definition does not have")


def test_a_regression_of_the_final_ab_needs_a_known_limit_with_its_evidence(tmp_path, final_repo):
    h = final_repo
    slow = {"b200": {SLOW_CELL: 1.08}}
    unconfirmed = final_record(tmp_path, h, slow=slow)
    assert has(fcheck(unconfirmed, h), "has no known-limit entry")
    limit = known_limit(tmp_path, h)
    rec = final_record(tmp_path, h, slow=slow, known_limits=[limit])
    assert fcheck(rec, h) == []                                                                                          # the legal flagged-only confirmation with three times the calls passes
    cases = (
        ("no document", lambda: rc.final_problems(rec, h["repo"], h["docs"], None), "needs the performance document"),
        ("a document that does not name the cell", lambda: fcheck(rec, h, document=PERF_DOC % "decode;N=2;H=2;regime=cold;"), "does not name the cell"),
        ("a document without the section", lambda: fcheck(rec, h, document="# nothing\n"), "does not name the cell"),
        ("a confirmation on the node of the final A/B", lambda: fcheck(mutated(rec, lambda r: r["known_limits"][0].update(
            confirmation=known_limit(tmp_path, h, confirm_host="node-a-b200", name="samenode")["confirmation"])), h), "same node"),
        ("a confirmation under the same allocation", lambda: fcheck(mutated(rec, lambda r: r["known_limits"][0]["confirmation"].update(allocation="sc-b200-1")), h), "second allocation is required"),
        ("a confirmation that does not fail the cell", lambda: fcheck(mutated(rec, lambda r: r["known_limits"][0].update(
            confirmation=known_limit(tmp_path, h, factor=1.0, name="nofail")["confirmation"])), h), "does not confirm"),
        ("a confirmation with the calls of the first campaign", lambda: fcheck(mutated(rec, lambda r: r["known_limits"][0].update(
            confirmation=known_limit(tmp_path, h, confirm_scale=1, name="scale1")["confirmation"])), h), "not 3 times the first campaign's"),
        ("a confirmation with nine times the calls (compounded)", lambda: fcheck(mutated(rec, lambda r: r["known_limits"][0].update(
            confirmation=known_limit(tmp_path, h, confirm_scale=9, name="scale9")["confirmation"])), h), "not 3 times the first campaign's"),
        ("a confirmation of every cell instead of the flagged ones", lambda: fcheck(mutated(rec, lambda r: r["known_limits"][0].update(
            confirmation=campaign_files(tmp_path, "everycell", "decode", "b200", h["base"], h["final"], "node-b", "sc-b200-2", slows={SLOW_CELL: 1.08}, scale=3))), h), "flagged set"),
        ("a confirmation that is itself re-run", lambda: fcheck(mutated(rec, lambda r: r["known_limits"][0].update(
            confirmation=campaign_files(tmp_path, "rerunconf", "decode", "b200", h["base"], h["final"], "node-b", "sc-b200-2", slows={SLOW_CELL: 1.08}, cells=[SLOW_CELL], scale=3,
                                        rerun_cells=[SLOW_CELL], scope={SLOW_METRIC: [SLOW_CELL]}))), h), "never re-run"),
        ("a run with all changes reverted that shows no shift", lambda: fcheck(mutated(rec, lambda r: r["known_limits"][0].update(
            all_changes_reverted=known_limit(tmp_path, h, reverted_factor=1.0, name="noshift")["all_changes_reverted"])), h), "caused by the changes"),
        ("a run with all changes reverted that is of the final tree", lambda: fcheck(mutated(rec, lambda r: r["known_limits"][0].update(
            all_changes_reverted=known_limit(tmp_path, h, name="finaltree")["confirmation"])), h), "side B must be"),
        ("no confirmation", lambda: fcheck(mutated(rec, lambda r: r["known_limits"][0].pop("confirmation")), h), "the campaign evidence is missing"),
        ("no run with all changes reverted", lambda: fcheck(mutated(rec, lambda r: r["known_limits"][0].pop("all_changes_reverted")), h), "the campaign evidence is missing"),
        ("a limit for another cell", lambda: fcheck(mutated(rec, lambda r: r["known_limits"][0].update(cell="decode;N=1;H=12;regime=lockstep;")), h), "has no known-limit entry"),
        ("a limit of another device", lambda: fcheck(mutated(rec, lambda r: r["known_limits"][0].update(device="b300")), h), "has no known-limit entry"),
    )
    for name, run, expected in cases:
        assert has(run(), expected), (name, run()[:3])


def test_a_confirmation_report_that_contradicts_its_raw_records_is_rejected(tmp_path, final_repo):
    """The known limit's campaigns are recomputed from their raw records too: a confirmation whose report says FAIL for the cell while its raw records do not fail it."""
    h = final_repo
    rec = final_record(tmp_path, h, slow={"b200": {SLOW_CELL: 1.08}}, known_limits=[known_limit(tmp_path, h, factor=1.0, name="invented")])
    probs = fcheck(rec, h)
    assert has(probs, "does not confirm")                                                                                 # the raw records of the confirmation do not fail the cell
    edit_report(rec["known_limits"][0]["confirmation"], lambda res: row_of(res).update(status="FAIL", delta_pct=8.0, median_delta_pct=8.0, cand_median=row_of(res)["base_median"] * 1.08))
    assert has(fcheck(rec, h), "the report differs from the raw records")


def test_the_final_command_line(tmp_path, final_repo, capsys):
    h = final_repo
    slow = {"b200": {SLOW_CELL: 1.08}}
    rec = final_record(tmp_path, h, slow=slow, known_limits=[known_limit(tmp_path, h)])
    rec.pop("_path")
    good, bad, doc = tmp_path / "final.json", tmp_path / "bad.json", tmp_path / "perf.md"
    good.write_text(json.dumps(rec))
    bad.write_text(json.dumps(mutated(rec, lambda r: r.update(known_limits=[]))))
    doc.write_text(PERF_DOC % SLOW_CELL)
    args = ["final", "--repo", str(h["repo"]), "--head", h["docs"]]
    assert rc.main(args + ["--record", str(good), "--document", str(doc)]) == 0
    capsys.readouterr()
    assert rc.main(args + ["--record", str(good)]) == 1                                                           # a known limit without the performance document
    assert "needs the performance document" in capsys.readouterr().out
    assert rc.main(args + ["--record", str(bad), "--document", str(doc)]) == 1
    assert rc.main(["final", "--repo", str(h["repo"]), "--head", h["source"], "--record", str(good), "--document", str(doc)]) == 1      # a source change since the measured commit
    assert "outside docs/" in capsys.readouterr().out
    assert rc.main(args + ["--record", str(tmp_path / "missing.json")]) == 2
    assert rc.main(["decision", "--record", str(good)]) == 1


def test_the_final_command_line_rejects_the_doubled_raw_candidate_samples(tmp_path, final_repo, capsys):
    """The round-3 counterexample at the command line: the raw candidate samples doubled, the report untouched: exit 1 and VERDICT: FAIL (it was exit 0 and PASS)."""
    h = final_repo
    rec = final_record(tmp_path, h)
    path = tmp_path / "final.json"
    rec.pop("_path")
    path.write_text(json.dumps(rec))
    args = ["final", "--repo", str(h["repo"]), "--head", h["docs"], "--record", str(path)]
    assert rc.main(args) == 0
    capsys.readouterr()
    rewrite_records(rec["devices"]["b200"]["campaigns"]["decode"], "B", double_samples)
    assert rc.main(args) == 1
    out = capsys.readouterr().out
    assert "VERDICT: FAIL" in out and "the report differs from the raw records" in out


# ---- the round-4 review's counterexamples: a final campaign is made of Method M records of the benchmark's own workload ------------------------------------------
PROBE_CELL = "decode;N=1;H=12;regime=probe;pos=63;"
CHAIN_CELL = CHAIN_KEYS[0]


def one_trial_pooled(rec):
    """The first control of the review: every cell keeps the middle one of its trials and the record says that it is a pooled record of one trial."""
    rec["meta"].update(sample_stat="pooled", trials=1)
    for cell in rec["cells"]:
        cell["samples"] = [cell["samples"][len(cell["samples"]) // 2]]


def keep_samples(n):
    def edit(rec):
        for cell in rec["cells"]:
            cell["samples"] = cell["samples"][:n]
    return edit


def set_meta(**fields):
    return lambda rec: rec["meta"].update(fields)


def set_args(args, keys=None):
    """Set cell arguments of the cells ``keys`` (every cell by default)."""
    def edit(rec):
        for cell in rec["cells"]:
            if keys is None or cell["key"] in keys:
                cell["args"].update(args)
    return edit


def drop_arg(name):
    return lambda rec: [cell["args"].pop(name) for cell in rec["cells"]]


def both(*edits):
    return lambda rec: [e(rec) for e in edits]


NODES_TWO = {"plan_kernel": 1, "rows_kernel": 1}


def with_nodes(mapping, keys=None):
    """The cells of a record as the benchmarks record them since the launch mapping is per trial: ``compute_nodes`` / ``aux_nodes`` in the cell and ``nodes`` in every sample."""
    def edit(rec):
        for cell in rec["cells"]:
            if keys is None or cell["key"] in keys:
                cell["args"].update(compute_nodes=dict(mapping), aux_nodes={}, compute_kernels=sum(mapping.values()))
                for s in cell["samples"]:
                    s["nodes"] = {"compute": dict(mapping), "aux": {}}
    return edit


def trial_nodes(trial, mapping, keys):
    """The launches of one trial of the cells ``keys`` as another mapping (that trial ran another path)."""
    def edit(rec):
        for cell in rec["cells"]:
            if cell["key"] in keys:
                cell["samples"][trial]["nodes"]["compute"] = dict(mapping)
    return edit


FINAL_CAMPAIGN_CASES = [
    # the sampling of Method M: process-median records of at least five trials, declared = held
    ("one-trial pooled records on both sides, the report untouched (first control of the review)", "decode", "AB", one_trial_pooled, "process_median"),
    ("the same in the chain campaign", "chain", "AB", one_trial_pooled, "process_median"),
    ("pooled records of five trials on both sides", "decode", "AB", set_meta(sample_stat="pooled"), "process_median"),
    ("no sample statistic on both sides", "decode", "AB", lambda rec: rec["meta"].pop("sample_stat"), "process_median"),
    ("one trial declared, five held", "decode", "B", set_meta(trials=1), "at least 5 trials"),
    ("three trials declared and held", "decode", "A", both(set_meta(trials=3), keep_samples(3)), "at least 5 trials"),
    ("five trials declared, four held", "decode", "B", keep_samples(4), "declares 5 trials"),
    ("a record of another kind", "decode", "B", set_meta(kind="chain-wall"), "is not a decode-wall record"),
    # the effective counts: positive integers
    ("no calls per trial", "decode", "B", drop_arg("calls_per_trial"), "positive integer"),
    ("no profiled calls", "decode", "A", set_args({"profile_calls": 0}), "positive integer"),
    ("fractional calls per trial", "chain", "B", set_args({"calls_per_trial": 1536.5}), "positive integer"),
    # the workload: the arguments of a cell are the ones that its key names
    ("every cell measures N = 1, H = 1, lock-step, position 0, on both sides (second control of the review)", "decode", "AB",
     set_args({"N": 1, "H": 1, "regime": "lockstep", "pos": 0}), "workload its key names"),
    ("one cell of another N", "decode", "B", set_args({"N": 2}, [SLOW_CELL]), "workload its key names"),
    ("a probe cell at another position", "decode", "AB", set_args({"pos": 0}, [PROBE_CELL]), "workload its key names"),
    ("a lock-step cell run in another regime", "decode", "B", set_args({"regime": "cold"}, [SLOW_CELL]), "workload its key names"),
    ("a chain cell of another model width", "chain", "AB", set_args({"Dm": 4096}, [CHAIN_CELL]), "workload its key names"),
    ("a chain cell of another regime", "chain", "B", set_args({"regime": "cold"}, [CHAIN_CELL]), "workload its key names"),
    ("a cell of another kind", "decode", "B", lambda rec: rec["cells"][3].update(kind="chain"), "is not a decode cell"),
    # the launches of every trial: a cell that records per-trial mappings records them consistently (the decode benchmark did not record them before, the chain's cells did)
    ("a middle trial of a decode cell on another launch path (no plan kernel)", "decode", "B", both(with_nodes(NODES_TWO), trial_nodes(2, {"rows_kernel": 1}, [SLOW_CELL])), "trial 2"),
    ("a trial of a decode cell with a duplicated launch, on both sides", "decode", "AB", both(with_nodes(NODES_TWO), trial_nodes(4, {"plan_kernel": 1, "rows_kernel": 2}, [PROBE_CELL])), "trial 4"),
    ("a trial of a chain cell on another mapping", "chain", "A", both(with_nodes(ROUTE["chain"]["compute_nodes"]), trial_nodes(1, {"rows_kernel": 1}, [CHAIN_CELL])), "trial 1"),
    ("a decode cell whose samples carry mappings and whose arguments do not", "decode", "B", both(with_nodes(NODES_TWO), drop_arg("compute_nodes")), "records no compute_nodes"),
]


@pytest.mark.parametrize("name,campaign,sides,edit,expected", FINAL_CAMPAIGN_CASES, ids=[c[0][:70] for c in FINAL_CAMPAIGN_CASES])
def test_a_final_campaign_is_made_of_method_m_records_of_the_benchmarks_own_workload(tmp_path, final_repo, name, campaign, sides, edit, expected):
    rec = final_record(tmp_path, final_repo)
    assert fcheck(rec, final_repo) == []
    for side in sides:
        rewrite_records(rec["devices"]["b200"]["campaigns"][campaign], side, edit)
    probs = fcheck(rec, final_repo)
    assert has(probs, expected), (name, probs[:3])


def test_a_final_campaign_whose_cells_record_consistent_per_trial_mappings_passes_like_a_legacy_one(tmp_path, final_repo):
    rec = final_record(tmp_path, final_repo)
    assert fcheck(rec, final_repo) == []                                                          # cells without mappings (the records of before): valid
    for side in "AB":
        rewrite_records(rec["devices"]["b200"]["campaigns"]["decode"], side, with_nodes(NODES_TWO))
        rewrite_records(rec["devices"]["b200"]["campaigns"]["chain"], side, with_nodes(ROUTE["chain"]["compute_nodes"]))
    assert fcheck(rec, final_repo) == []                                                          # cells that record them: valid when every trial equals the cell's mapping


def test_the_generic_comparator_keeps_its_pooled_support():
    """The contract above is the final campaigns': ``br.compare`` still judges pooled records of one trial or of five (the A/A tools and older records use them)."""
    for edit in (lambda r: r["meta"].update(sample_stat="pooled"), one_trial_pooled):
        base, cand = tbt.wall_sets([10.0] * 3, [10.0] * 3)
        for r in base + cand:
            edit(r)
        assert br.compare(base, cand, **rc.compare_kwargs("gpu_graph_us", 1.0, tbt.BASE_COMMIT, None))["verdict"] == "PASS"


def final_command_line(tmp_path, h, rec, capsys):
    """Write ``rec`` and run the final command on it; ``(exit status, output)``."""
    path = tmp_path / "final.json"
    rec = dict(rec)
    rec.pop("_path", None)
    path.write_text(json.dumps(rec))
    capsys.readouterr()
    status = rc.main(["final", "--repo", str(h["repo"]), "--head", h["docs"], "--record", str(path), "--document", str(tmp_path / "perf.md")])
    return status, capsys.readouterr().out


def test_the_final_command_line_rejects_one_trial_pooled_records_and_foreign_workloads(tmp_path, final_repo, capsys):
    """The two controls of the review at the command line: exit 1 and VERDICT: FAIL (both were exit 0 and PASS)."""
    (tmp_path / "perf.md").write_text(PERF_DOC % SLOW_CELL)
    for name, edit, expected in (("pooled", one_trial_pooled, "process_median"), ("foreign", set_args({"N": 1, "H": 1, "regime": "lockstep", "pos": 0}), "workload its key names")):
        d = tmp_path / name
        d.mkdir()
        (d / "perf.md").write_text(PERF_DOC % SLOW_CELL)
        rec = final_record(d, final_repo)
        assert final_command_line(d, final_repo, rec, capsys)[0] == 0
        for side in "AB":
            rewrite_records(rec["devices"]["b200"]["campaigns"]["decode"], side, edit)
        status, out = final_command_line(d, final_repo, rec, capsys)
        assert status == 1 and "VERDICT: FAIL" in out and expected in out, (name, out[:400])


# ---- B12: a known limit's confirmation and reverted run measure the original workload on the original baseline's route -------------------------------------------
def limit_record(tmp_path, h):
    return final_record(tmp_path, h, slow={"b200": {SLOW_CELL: 1.08}}, known_limits=[known_limit(tmp_path, h)])


def rewrite_limit(rec, part, sides, edit):
    """Rewrite the raw records of the ``sides`` of the confirmation or of the run with all changes reverted of the first known limit."""
    for side in sides:
        rewrite_records(rec["known_limits"][0][part], side, edit)


def on_route(impl, env_impl):
    """The run was made on routing ``impl``: the meta of the record and the argument of every cell say so."""
    def edit(rec):
        rec["meta"].update(impl=impl, env_impl=env_impl)
        for cell in rec["cells"]:
            cell["args"]["impl"] = impl
    return edit


FOREIGN = {"N": 256, "H": 33, "regime": "cold", "pos": 63}
LIMIT_CASES = [
    ("the confirmation and the reverted run on N = 256, H = 33, cold, position 63 (first control of the review)",
     [("confirmation", "AB", set_args(FOREIGN)), ("all_changes_reverted", "AB", set_args(FOREIGN))], "workload its key names"),
    ("the reverted run with A on impl 0 and B on the frozen reference (impl 3, env_impl 3), both commits the base (second control)",
     [("all_changes_reverted", "B", on_route(3, "3"))], "baseline behaviour of the final A/B"),
    ("both reverted arms on another route than the baseline of the final A/B", [("all_changes_reverted", "AB", on_route(3, "3"))], "baseline behaviour of the final A/B"),
    ("a reverted arm with another launch mapping than the baseline", [("all_changes_reverted", "B", set_args({"kernels": ["rows_kernel"]}))], "baseline behaviour of the final A/B"),
    ("a confirmation of the candidate on another launch mapping than the final A/B", [("confirmation", "B", set_args({"kernels": ["rows_kernel"]}))], "differ in kernels"),
    ("a confirmation of the baseline on another route than the final A/B", [("confirmation", "A", on_route(3, "3"))], "same snapshot / configuration"),
    ("a confirmation of cells of another shape", [("confirmation", "AB", set_args({"H": 33}))], "workload its key names"),
]


@pytest.mark.parametrize("name,edits,expected", LIMIT_CASES, ids=[c[0][:70] for c in LIMIT_CASES])
def test_a_known_limit_measures_the_original_workload_on_the_original_baseline_route(tmp_path, final_repo, name, edits, expected):
    h = final_repo
    rec = limit_record(tmp_path, h)
    assert fcheck(rec, h) == []
    for part, sides, edit in edits:
        rewrite_limit(rec, part, sides, edit)
    probs = fcheck(rec, h)
    assert has(probs, expected), (name, probs[:3])


ROUTE_CASES = [
    ("the baseline pinned on the frozen reference (impl 3, env_impl 3)", "A", on_route(3, "3")),
    ("the candidate forced to a launch structure (impl 64)", "B", on_route(64, None)),
    ("an environment override on the baseline", "A", set_meta(env_impl="3")),
    ("cells of the baseline that ran on another routing than the record says", "A", set_args({"impl": 3}, [SLOW_CELL])),
]


@pytest.mark.parametrize("name,side,edit", ROUTE_CASES, ids=[c[0][:70] for c in ROUTE_CASES])
def test_the_final_ab_measures_both_trees_at_their_default_routing(tmp_path, final_repo, name, side, edit):
    """What the confirmation and the reverted run are bound to must itself be the baseline: a baseline on a slower route would hide a regression and make everything bound to it consistent."""
    rec = final_record(tmp_path, final_repo)
    assert fcheck(rec, final_repo) == []
    rewrite_records(rec["devices"]["b200"]["campaigns"]["decode"], side, edit)
    probs = fcheck(rec, final_repo)
    assert has(probs, "default routing"), (name, probs[:3])


def test_the_final_command_line_rejects_a_reverted_run_on_another_route(tmp_path, final_repo, capsys):
    """The second known-limit control of the review at the command line: the reverted run with A on impl 0 and B on the frozen reference, both commits the base."""
    (tmp_path / "perf.md").write_text(PERF_DOC % SLOW_CELL)
    rec = limit_record(tmp_path, final_repo)
    assert final_command_line(tmp_path, final_repo, rec, capsys)[0] == 0
    rewrite_limit(rec, "all_changes_reverted", "B", on_route(3, "3"))
    status, out = final_command_line(tmp_path, final_repo, rec, capsys)
    assert status == 1 and "VERDICT: FAIL" in out and "baseline behaviour of the final A/B" in out, out[:400]


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
