"""The decision and the ledger row of a row-engine step (``benchmarks/step_record.py``), on synthetic campaigns built the way the final-record tests build theirs (plain Python, no GPU and no
torch: ``PYTHONPATH=. pytest --noconftest tests/cfbi/test_step_record.py``).

The rule under test: a step is adopted only with no confirmed slower cell of the KERNEL SUM against the previous accepted state and the base, and at least one confirmed faster cell against the
previous accepted state on each device (stable, judged on the re-run, faster by the tolerance or more on the re-run and in the first campaign); a comparison that is not a measurement leaves
the outcome undecided; the wall metrics are reported and never decide.
"""
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "benchmarks", _ROOT / "tests" / "cfbi"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bench_records as br  # noqa: E402
import record_checker as rc  # noqa: E402
import step_record as sp  # noqa: E402
import test_bench_order as tbo  # noqa: E402
import test_bench_tools as tbt  # noqa: E402
import test_record_checker_steps as trs  # noqa: E402
from test_record_checker import commit_file, git  # noqa: E402
import suite_record as sr  # noqa: E402

PREV, STEP, BASE = ("%07x%033x" % (i, i) for i in (11, 12, 13))
GAIN_A, GAIN_B = "decode;N=4;H=12;regime=lockstep;", "decode;N=64;H=12;regime=lockstep;"


def kernel_cell(key, median, factor, scale=1, spread=0.0, idx=0, wall_factor=1.0):
    """A decode cell whose kernel sum is the plan kernel plus ``rows_kernel`` scaled by ``factor`` (and whose graph replay is scaled by ``wall_factor``)."""
    median = median * (1.0 + spread * idx)
    loop = any(x in key for x in trs.LOOP_SUBSTRINGS)
    wall = []
    for t in tbt.trials(median, spread):
        w = {"gpu_graph_us": t * wall_factor, "gpu_eager_us": t + 1.0}
        if loop:
            w.update(cpu_graph_us=t + 3.0, cpu_eager_us=t + 4.0)
        wall.append(w)
    runs = [{tbt.PLAN_K: 2.0, tbt.ROWS_K: 10.0 * factor, tbt.FILL_K: 40.0} for _ in wall]
    args = dict(trs.workload_args("decode", key), impl=0, calls_per_trial=scale * tbt.CALLS, profile_calls=scale * tbt.PROFILE_CALLS, calls_scale=scale, **trs.ROUTE["decode"])
    return br.make_cell("decode", key, args, runs, drop_helpers=True, wall=wall)


def record(dev, commit, ts, host, factors, scale=1, cells=None, idx=0, wall=None):
    gpu, sm = trs.GPU[dev]
    meta = {"commit": commit, "dirty": False, "gpu": gpu, "sm_count": sm, "host": host, "timestamp": ts, "sample_stat": br.PROCESS_MEDIAN, "trials": 5, "kind": trs.KIND["decode"], "impl": 0,
            "env_impl": None, "l2_cache_bytes": 1 << 26, "eviction_bytes": (1 << 27) + (1 << 20)}
    keys = cells if cells is not None else trs.DECODE_KEYS
    return {"schema": br.SCHEMA, "meta": meta, "cells": [kernel_cell(k, trs.MEDIANS["decode"][k], (factors or {}).get(k, 1.0), scale, idx=idx, wall_factor=(wall or {}).get(k, 1.0)) for k in keys]}


START_ORDER = "ABBABAAB"                                                                     # the start order of the eight record files of a campaign: two blocks of four, ABBA and BAAB
NODES = {"A": {"plan_kernel": 1, "rows_kernel": 1}, "B": {"rows_kernel": 1}}                   # the launch mapping that every trial of a side records (the content is not judged here)


def campaign(d, name, dev, a_commit, b_commit, factors=None, rerun=None, rerun_factors=None, wall=None, allocation=None, order=None):
    """One campaign as a step runs it: four record files per side in the ABBA start order with the launch mapping of every trial, the cells ``rerun`` re-run with three times the calls in the same
    order, the comparator's report under the same contract, and the entry of the step tool."""
    host = "node-%s-%s" % (dev, name)

    def side(label, hour, commit, side_factors, scale, cells):
        slots = [slot for slot, x in enumerate(order or START_ORDER) if x == label]
        recs = [record(dev, commit, "2026-11-13T%s:%02d:00Z" % (hour, slot), host, side_factors, scale=scale, cells=cells, idx=i, wall=wall if label == "B" else None) for i, slot in enumerate(slots)]
        return tbo.with_nodes(recs, NODES[label])
    a1, b1 = side("A", "10", a_commit, None, 1, None), side("B", "10", b_commit, factors, 1, None)
    a2 = side("A", "11", a_commit, None, 3, rerun) if rerun else []
    b2 = side("B", "11", b_commit, rerun_factors if rerun_factors is not None else factors, 3, rerun) if rerun else []
    files = {"A": tbt.write_records(str(d), name + "-first-a", a1), "B": tbt.write_records(str(d), name + "-first-b", b1)}
    again = {"A": tbt.write_records(str(d), name + "-again-a", a2) if a2 else [], "B": tbt.write_records(str(d), name + "-again-b", b2) if b2 else []}
    kwargs = lambda m: rc.compare_kwargs(m, rc.TOLERANCE_PCT, a_commit, rc.metric_cells("decode", m), sp.AB_ORDER, sp.REQUIRE_NODES)        # noqa: E731
    results = [br.compare(a1, b1, base_rerun=a2 or None, cand_rerun=b2 or None, **kwargs(m)) for m in sorted(rc.canonical_coverage("decode"))]
    report = Path(d) / (name + "-report.json")
    report.write_text(json.dumps(results, default=str))
    entry = {"report": str(report), "records_first": files, "records_rerun": again, "allocation": allocation or "sc-%s-%s" % (dev, name), "a_commit": a_commit, "b_commit": b_commit}
    path = Path(d) / (name + "-entry.json")
    path.write_text(json.dumps(entry))
    return str(path)


def manifest(d, per_device, wall=None):
    """The evidence manifest of a step for the campaigns of ``per_device``: ``{dev: {"previous": kwargs, "base": kwargs}}`` (the keyword arguments of ``campaign``)."""
    Path(d).mkdir(exist_ok=True)
    devices = {}
    for dev in rc.DEVICES:
        gpu, sm = trs.GPU[dev]
        baseline = Path(d) / ("baseline-%s.json" % dev)
        baseline.write_text(json.dumps({"commit": BASE, "gpu": gpu, "sm_count": sm}))
        devices[dev] = {"baseline": str(baseline),
                        "previous": campaign(d, "%s-previous" % dev, dev, PREV, STEP, wall=wall, **per_device[dev]["previous"]),
                        "base": campaign(d, "%s-base" % dev, dev, BASE, STEP, wall=wall, **per_device[dev]["base"])}
    return {"schema": sp.SCHEMA, "step": "trims", "commit": STEP, "previous": PREV, "base": BASE, "markers": ["step_marker_example"], "devices": devices}


GAIN = {"factors": {GAIN_A: 0.9, GAIN_B: 0.95}, "rerun": [GAIN_A, GAIN_B]}                       # the kernel sum of two cells falls by about 8 % and 4 %, and both are re-run
FLAT = {"factors": None}
SLOWER = {"factors": {GAIN_B: 1.08}, "rerun": [GAIN_B]}                                          # one cell about 6 % slower in the kernel sum, re-run and confirmed
EVERYWHERE = {"previous": GAIN, "base": FLAT}
ADOPTABLE = dict((dev, EVERYWHERE) for dev in rc.DEVICES)                                       # a step that both devices adopt
GAIN_AND_SLOWER = {"factors": dict(GAIN["factors"], **SLOWER["factors"]), "rerun": GAIN["rerun"]}   # GAIN_A faster, GAIN_B about 6 % slower, both re-run


def decision_of(tmp_path, per_device, wall=None):
    m = manifest(tmp_path, per_device, wall)
    infos = sp.campaign_infos(m)
    return m, infos, sp.decide(m, infos)


def test_a_confirmed_gain_without_a_regression_adopts_the_step(tmp_path):
    m, infos, d = decision_of(tmp_path, ADOPTABLE)
    assert all(infos[dev][k][0] == [] for dev in rc.DEVICES for k in sp.KINDS), [infos[dev][k][0][:2] for dev in rc.DEVICES for k in sp.KINDS]
    assert d["outcome"] == "adopted", d
    assert d["timing"] == {"vs_previous": "pass", "vs_base": "pass", "chain": "not_applicable"}
    assert all(d["per_device"][dev]["previous"]["faster"] == 2 and d["per_device"][dev]["previous"]["slower"] == 0 for dev in rc.DEVICES)


def test_a_confirmed_slower_cell_of_the_kernel_sum_against_the_previous_state_or_the_base_does_not_adopt(tmp_path):
    for kind, base in (("previous", FLAT), ("base", SLOWER)):                        # b300's campaign against the previous state is GAIN_AND_SLOWER in both
        per = {"b200": EVERYWHERE, "b300": {"previous": GAIN_AND_SLOWER, "base": base}}
        m, infos, d = decision_of(tmp_path / kind, per)
        assert d["outcome"] == "not_adopted", (kind, d["outcome"], d["reasons"][:2])
        assert any("slower than A in the kernel sum" in r and "b300" in r and GAIN_B in r for r in d["reasons"]), d["reasons"]
        assert d["timing"]["vs_previous" if kind == "previous" else "vs_base"] == "fail"


def test_a_device_without_a_confirmed_gain_does_not_adopt_the_step(tmp_path):
    m, infos, d = decision_of(tmp_path, {"b200": EVERYWHERE, "b300": {"previous": FLAT, "base": FLAT}})
    assert d["outcome"] == "not_adopted" and any("b300: no confirmed gain" in r for r in d["reasons"]), d["reasons"]
    assert d["timing"]["vs_previous"] == "fail"


def test_a_gain_that_the_confirmation_did_not_reproduce_is_noise(tmp_path):
    first_only = {"factors": GAIN["factors"], "rerun": GAIN["rerun"], "rerun_factors": {GAIN_A: 1.0, GAIN_B: 1.0}}       # fast in the first campaign, neutral on the re-run
    m, infos, d = decision_of(tmp_path, {"b200": {"previous": first_only, "base": FLAT}, "b300": EVERYWHERE})
    assert d["outcome"] == "not_adopted" and d["per_device"]["b200"]["previous"]["faster"] == 0, d
    late = {"factors": None, "rerun": GAIN["rerun"], "rerun_factors": GAIN["factors"]}                                    # neutral in the first campaign, fast on the re-run: the sign was not reproduced
    m, infos, d = decision_of(tmp_path / "late", {"b200": {"previous": late, "base": FLAT}, "b300": EVERYWHERE})
    assert d["per_device"]["b200"]["previous"]["faster"] == 0


def test_the_wall_metrics_are_reported_and_never_decide(tmp_path):
    wall = {GAIN_A: 1.10}                                                                                                    # the graph replay of one cell gets 10 % slower, the kernel sum does not
    m, infos, d = decision_of(tmp_path, ADOPTABLE, wall=wall)
    assert d["outcome"] == "adopted"
    assert d["per_device"]["b200"]["reported"]["previous"]["gpu_graph_us"]["slower"] >= 1


def test_a_comparison_that_is_not_a_measurement_leaves_the_outcome_undecided(tmp_path):
    m = manifest(tmp_path, ADOPTABLE)
    entry = json.loads(Path(m["devices"]["b300"]["previous"]).read_text())
    entry["records_first"]["B"] = entry["records_first"]["B"][:1]                                                           # one record per side: not three interleaved repeats
    Path(m["devices"]["b300"]["previous"]).write_text(json.dumps(entry))
    d = sp.decide(m, sp.campaign_infos(m))
    assert d["outcome"] == "undecided" and any("b300" in r for r in d["reasons"]), d
    m = manifest(tmp_path / "wrong", ADOPTABLE)
    entry = json.loads(Path(m["devices"]["b200"]["base"]).read_text())
    entry["b_commit"] = PREV                                                                                                   # the entry names another candidate than the step
    Path(m["devices"]["b200"]["base"]).write_text(json.dumps(entry))
    assert sp.decide(m, sp.campaign_infos(m))["outcome"] == "undecided"


def test_the_manifest_names_what_the_decision_needs():
    assert sp.manifest_problems({}) and sp.manifest_problems({"schema": sp.SCHEMA})
    good = {"schema": sp.SCHEMA, "step": "trims", "commit": STEP, "previous": PREV, "base": BASE, "markers": ["x"],
            "devices": dict((d, {"baseline": "b", "previous": "p", "base": "q"}) for d in rc.DEVICES),
            "suites": dict((d, {"gate": {"baseline": "a", "candidate": "b"}, "S": {"baseline": "a", "candidate": "b"}}) for d in rc.DEVICES),
            "codegen": {"allow": "a", "conda": {"base": "b", "cand": "c", "prev": "p"}, "container_b200": {"base": "b", "cand": "c", "prev": "p", "occupancy": "o"},
                        "container_b300": {"base": "b", "cand": "c", "prev": "p", "occupancy": "o"}}}
    assert sp.manifest_problems(good) == []
    for edit, needle in ((lambda m: m.pop("markers"), "markers"), (lambda m: m["devices"].pop("b300"), "devices"), (lambda m: m["suites"]["b200"].pop("S"), "suites.b200.S"),
                         (lambda m: m["codegen"]["container_b300"].pop("occupancy"), "occupancy"), (lambda m: m.update(step="e_tile_x"), "step"), (lambda m: m.update(commit="x"), "commit")):
        bad = json.loads(json.dumps(good))
        edit(bad)
        assert any(needle in p for p in sp.manifest_problems(bad)), needle


def adopted_case(tmp_path, monkeypatch, per_device=None):
    """The manifest, the recomputed campaigns, the decision and the row of a step (adopted unless ``per_device`` says otherwise) whose dumps are those of the record checker's own tests."""
    m = manifest(tmp_path, per_device or ADOPTABLE)
    codegen = {}
    for key in rc.COMPARISONS:
        entry, _ = trs.comparison(key, STEP, sid="trims")
        prev = trs.dump_file(key, PREV, "cand", rows=None)
        codegen[key] = {"base": entry["base_dump"], "cand": entry["cand_dump"], "prev": prev}
        if key.startswith("container_"):
            codegen[key]["occupancy"] = entry["occupancy"]
    m["codegen"] = dict(codegen, allow=trs.allow_file({}))
    m["suites"] = dict((d, {"gate": {"baseline": "x", "candidate": "y"}, "S": {"baseline": "x", "candidate": "y"}}) for d in rc.DEVICES)
    monkeypatch.setattr(sp, "exactness_problems", lambda *args, **kwargs: [])
    infos = sp.campaign_infos(m)
    decision = sp.decide(m, infos)
    return m, infos, decision, sp.build_row(m, decision, infos)


def test_the_section_of_the_document_holds_the_numbers_of_the_recomputed_evidence(tmp_path, monkeypatch):
    m, infos, decision, row = adopted_case(tmp_path, monkeypatch)
    text = sp.render_section(m, decision, row, infos, number=10)
    lines = text.split("\n")
    assert lines[0] == "## 10. Row engine step 1: the schedule trims" and "The step is **adopted**." in lines
    assert any("%s: 2 confirmed faster cell(s) of the kernel sum" % dev in line for dev in rc.DEVICES for line in lines)
    for dev in rc.DEVICES:                                                           # the table: one row per device, side and metric, with the counts that the decision has
        mine = [line for line in lines if line.startswith("| %s | the previous accepted state | total_us |" % dev)]
        assert len(mine) == 1 and mine[0].split("|")[5:7] == [" 0 ", " 2 "], mine
        assert any(line.startswith("| %s | the base | total_us |" % dev) for line in lines)
    assert all(not line.startswith("| b200 | the base | total_us | ") or " 0 | 0 |" in line for line in lines)               # the base campaign is flat: nothing slower, nothing faster
    assert any(line.startswith("| conda | sm_100a | cfbi::rows_kernel |") for line in lines)                                       # the resource table names the row engine of every toolchain
    assert sum(1 for line in lines if line.startswith("| container_b200 |")) == 1 and sum(1 for line in lines if line.startswith("| conda |")) == 3
    assert any(line.startswith("Gate X in change mode") and "conda PASS" in line and "container_b300 PASS" in line for line in lines)
    assert any(line.startswith("- Campaigns:") and " 4 record files per side in the abba start order" in line for line in lines)      # the number of record files of the campaigns, not their minimum
    assert text.endswith("\n") and not text.endswith("\n\n")
    refused = json.loads(json.dumps(decision))
    refused["outcome"], refused["reasons"] = "not_adopted", ["b200, against the base, some cell: B is 3.00 % slower than A in the kernel sum"]
    assert "The step is **not adopted**." in sp.render_section(m, refused, row, infos).split("\n")


def test_an_adopted_row_built_from_the_evidence_passes_the_record_checker(tmp_path, monkeypatch):
    """The row of an adopted step (dumps in change mode with the previous accepted dump, the occupancy records, the A/B medians, the timing results) is accepted by ``step_problems``; the
    same row with a failed resource gate or a regression is not."""
    m, infos, decision, row = adopted_case(tmp_path, monkeypatch)
    assert row["outcome"] == "adopted" and row["gates"] == {"exactness": True, "codegen": True} and row["timing"]["chain"] == "not_applicable"
    meta = {"base": trs.BASE}
    assert rc.step_problems(row, None, meta) == []
    worse = json.loads(json.dumps(row))
    worse["codegen_gate"]["container_b200"]["verdict"] = "FAIL"                                                            # a verdict that the dumps do not give
    assert any("the ledger says" in p for p in rc.step_problems(worse, None, meta))
    regressed = json.loads(json.dumps(row))
    regressed["timing"]["vs_base"] = "fail"                                                                                 # an adopted step does not regress against the base
    assert any("not regress against the base" in p for p in rc.step_problems(regressed, None, meta))


def test_a_suite_on_a_later_commit_proves_the_step_only_with_the_same_kernel_sources(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    base = commit_file(repo, "csrc/k.cu", "kernel v1\n", "kernel")
    step = commit_file(repo, "csrc/k.cu", "kernel v2\n", "the step")
    tests_only = commit_file(repo, "tests/cfbi/test_x.py", "def test_x(): pass\n", "a test")
    tools_too = commit_file(repo, "benchmarks/tool.py", "tool\n", "a tool")
    assert sp.source_problems(repo, step, step) == [] and sp.source_problems(repo, step, tests_only) == [] and sp.source_problems(repo, step, tools_too) == []
    changed = commit_file(repo, "flash_kda/cfbi/_ext.py", "flags\n", "a flag of the build")                 # an input of the compilation changes
    found = sp.source_problems(repo, step, changed)
    assert len(found) == 1 and "kernel sources differ" in found[0] and "flash_kda/cfbi/_ext.py" in found[0], found
    older = sp.source_problems(repo, tests_only, step)                                                      # a suite that ran before the step proves nothing about it
    assert len(older) == 1 and "not a descendant" in older[0], older
    assert "does not have" in sp.source_problems(repo, step, "f" * 40)[0] and "does not have" in sp.source_problems(repo, "e" * 40, step)[0]
    git(repo, "checkout", "-q", "-b", "elsewhere", base)
    other = commit_file(repo, "docs/x.md", "x\n", "a commit on another line")
    git(repo, "checkout", "-q", "main")
    assert "not a descendant" in sp.source_problems(repo, step, other)[0]
    assert sp.source_problems(tmp_path / "nowhere", step, step)                                             # no repository: the step cannot be placed


def suite_pair(directory, suite, gpu, failed=()):
    """The baseline (of the previous accepted state) and the candidate record (of the step, with the failed tests ``failed``) of one suite on ``gpu``, written as files (``cfbi-suite-v1``)."""
    def junit(names, bad=()):
        cases = "".join('<testcase classname="tests.cfbi.test_x" name="%s" time="0.1">%s</testcase>' % (n, '<failure message="boom">t</failure>' if n in bad else "") for n in names)
        return ('<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests"><testsuite name="pytest" errors="0" failures="%d" skipped="0" tests="%d">%s</testsuite></testsuites>'
                % (len(bad), len(names), cases))
    names = ["test_a", "test_b", "test_c"]
    meta = {"suite": suite, "gpu": gpu, "dirty": False, "sm_count": 148, "env": {}, "python": "3.12.3", "torch": "2.14.0", "cuda": "13.4"}
    paths = []
    for name, commit, bad in (("base", PREV, ()), ("cand", STEP, failed)):
        path = Path(directory) / ("suite-%s.json" % name)
        path.write_text(json.dumps(tbt.suite_record_of(junit(names, bad), commit=commit, **meta)))
        paths.append(str(path))
    return paths


def exactness_manifest(directory, failed=()):
    suites = {}
    for dev in rc.DEVICES:
        suites[dev] = {}
        for name in ("gate", "S"):
            sub = Path(directory) / dev / name
            sub.mkdir(parents=True, exist_ok=True)
            suites[dev][name] = dict(zip(("baseline", "candidate"), suite_pair(sub, name, trs.GPU[dev][0], failed)))
    return {"schema": sp.SCHEMA, "step": "trims", "commit": STEP, "previous": PREV, "base": BASE, "markers": ["x"], "suites": suites}


def test_the_exactness_gate_reads_the_suite_records_of_both_devices(tmp_path):
    m = exactness_manifest(tmp_path)
    assert sp.exactness_problems(m) == [], sp.exactness_problems(m)
    found = sp.exactness_problems(exactness_manifest(tmp_path / "failing", failed=("test_b",)))
    assert any("failed or errored test tests/cfbi/test_x.py::test_b" in p and p.startswith("b200, suite gate") for p in found), found[:3]
    assert any(p.startswith("b300, suite S") for p in found)
    other = dict(m, suite_commit=PREV)                        # the suites ran on a commit that is not the one of the records
    assert any("the candidate record is of commit" in p for p in sp.exactness_problems(other, tmp_path / "nowhere"))


def test_a_campaign_is_recomputed_under_the_contract_it_was_run_under(tmp_path):
    """The checker recomputes a campaign from its raw records under the start order and the launch-mapping requirement that the CALLER names; a step runs in ABBA blocks, and the same records are
    not a measurement under the strict alternation of the final A/B (nor under a requirement of mappings that the records do not carry)."""
    m = manifest(tmp_path, ADOPTABLE)
    entry = json.loads(Path(m["devices"]["b200"]["previous"]).read_text())
    baseline = json.loads(Path(m["devices"]["b200"]["baseline"]).read_text())
    args = (entry, "x", "decode", "b200", baseline, PREV, PREV, STEP)
    problems, info = rc.campaign_evidence_problems(*args, order="abba", require_nodes=True)
    assert problems == [] and info["rows"], problems[:2]
    problems, _ = rc.campaign_evidence_problems(*args)                                          # the defaults of the final A/B: strict alternation, no mappings required
    assert problems and all("not interleaved in time" in p for p in problems), problems[:2]
    assert rc.compare_kwargs("total_us", 1.0, PREV, None)["order"] == "alternate" and rc.compare_kwargs("total_us", 1.0, PREV, None)["require_nodes"] is False
    plain = manifest(tmp_path / "plain", ADOPTABLE)
    for path in json.loads(Path(plain["devices"]["b200"]["previous"]).read_text())["records_first"]["A"][:1]:
        rec = json.loads(Path(path).read_text())
        for cell in rec["cells"]:
            for sample in cell["samples"]:
                sample.pop("nodes", None)                                                          # one record whose trials lost their launch mappings
        Path(path).write_text(json.dumps(rec))
    problems, _ = rc.campaign_evidence_problems(json.loads(Path(plain["devices"]["b200"]["previous"]).read_text()), "x", "decode", "b200", baseline, PREV, PREV, STEP, order="abba", require_nodes=True)
    assert problems and any("node mapping" in p or "compute_nodes" in p for p in problems), problems[:2]


def test_a_step_that_a_confirmed_slower_cell_stops_gets_a_not_adopted_row_that_the_record_checker_accepts(tmp_path, monkeypatch):
    """The row of a step that failed at the timing phase: not adopted, failed_at timing, the revert commit, the reason naming the cell, every earlier gate passed, the A/B medians kept; it is
    refused without the revert commit and with a failed_at that contradicts the gates."""
    per = {"b200": EVERYWHERE, "b300": dict(EVERYWHERE, previous=GAIN_AND_SLOWER)}
    m, infos, decision, row = adopted_case(tmp_path, monkeypatch, per)
    assert decision["outcome"] == "not_adopted" and row["outcome"] == "not_adopted" and row["failed_at"] == "timing", decision["reasons"][:2]
    assert GAIN_B in row["reason"] and "b300" in row["reason"] and row["gates"] == {"exactness": True, "codegen": True}
    assert row["timing"]["vs_previous"] == "fail" and row["revert_commit"] is None
    meta = {"base": trs.BASE}
    assert any("not reverted" in p for p in rc.step_problems(row, None, meta)), rc.step_problems(row, None, meta)[:2]
    m["revert_commit"] = "%07x%033x" % (14, 14)
    row = sp.build_row(m, decision, infos)
    assert rc.step_problems(row, None, meta) == [], rc.step_problems(row, None, meta)
    wrong = json.loads(json.dumps(row))
    wrong["failed_at"] = "codegen"                                                                   # the gates say that the code generation passed
    assert any("failed at codegen" in p for p in rc.step_problems(wrong, None, meta)), rc.step_problems(wrong, None, meta)[:2]
