"""The decision and the ledger row of a row-engine step (``benchmarks/step_record.py``), on synthetic campaigns built the way the final-record tests build theirs (plain Python, no GPU and no
torch: ``PYTHONPATH=. pytest --noconftest tests/cfbi/test_step_record.py``).

The rule under test: a step is adopted only with no confirmed slower cell of the KERNEL SUM against the previous accepted state and the base, and at least one confirmed faster cell against the
previous accepted state on each device (stable, judged on the re-run, faster by the tolerance or more on the re-run and in the first campaign); a comparison that is not a measurement leaves
the outcome undecided; the wall metrics are reported and never decide.
"""
import hashlib
import json
import re
import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr

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
            "suites": dict((d, {"gate": {"baseline": "a", "candidate": "b", "collection": "c", "baseline_collection": "d"}, "S": {"baseline": "a", "candidate": "b", "collection": "c", "baseline_collection": "d"}})
                           for d in rc.DEVICES),
            "codegen": {"allow": "a", "conda": {"base": "b", "cand": "c", "prev": "p"}, "container_b200": {"base": "b", "cand": "c", "prev": "p", "occupancy": "o"},
                        "container_b300": {"base": "b", "cand": "c", "prev": "p", "occupancy": "o"}}}
    assert sp.manifest_problems(good) == []
    for edit, needle in ((lambda m: m.pop("markers"), "markers"), (lambda m: m["devices"].pop("b300"), "devices"), (lambda m: m["suites"]["b200"].pop("S"), "suites.b200.S"), (lambda m: m["suites"]["b300"]["gate"].pop("collection"), "suites.b300.gate"),
                         (lambda m: m["suites"]["b300"]["S"].pop("baseline_collection"), "suites.b300.S"), (lambda m: m["suites"]["b200"]["gate"].update(baseline_collection=3), "suites.b200.gate"),
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
    assert any(line.startswith("- Step commit") and "Suite S is every test file of `tests/cfbi`, the four cases of the K2 sanitizer file included" in line for line in lines)        # what the admission proved
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


SANITIZER = "tests/cfbi/test_k2_dyn_sanitizer.py"
GATE_FILE = "tests/cfbi/test_vs_ref.py"
GUARD = "tests/cfbi/test_guard_zones.py"
GUARD_TEST = "test_prefill_varlen_warm_and_decode_stay_inside_their_buffers[1005-16]"       # the one test that is skipped by design on B200 and B300, in the base and in every later commit
GUARD_REASON = "the 1000 + x hook belongs to the old kernels; the device-scheduled K2 rejects it"
RACECHECK = "test_racecheck_where_the_tool_supports_the_kernels"                              # the test whose skip is the documented exception of the sanitizer policy
RACECHECK_REASON = "racecheck does not support these kernels: Unsupported architecture"
BY_DESIGN = {GUARD_TEST: GUARD_REASON}
# the tests of the world of the exactness tests: classname -> test names. The base has four files, the commit of the suites adds the step's own tests.
BASE_TESTS = {"tests.cfbi.test_vs_ref": ["test_g1", "test_g2"], "tests.cfbi.test_x": ["test_a", "test_b"], "tests.cfbi.test_guard_zones": ["test_plain", GUARD_TEST],
              "tests.cfbi.test_k2_dyn_sanitizer": ["test_case[memcheck]", "test_case[racecheck]", RACECHECK]}
STEP_TESTS = {"tests.cfbi.test_new_mechanism": ["test_pin", "test_control"]}


def ids_of(files):
    return ["%s.py::%s" % (cls.replace(".", "/"), name) for cls, names in files.items() for name in names]


def junit_of(files, failed=(), skipped=None):
    """The JUnit XML of a run of ``files`` ({classname: names}): the tests of ``failed`` fail, those of ``skipped`` ({name: reason}) are skipped with that reason."""
    skipped = skipped or {}

    def body(n):
        if n in failed:
            return '<failure message="boom">t</failure>'
        if n in skipped:
            return '<skipped type="pytest.skip" message=%s>skipped</skipped>' % quoteattr(skipped[n])
        return ""
    cases = "".join('<testcase classname="%s" name="%s" time="0.1">%s</testcase>' % (cls, n, body(n)) for cls, names in files.items() for n in names)
    return ('<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests"><testsuite name="pytest" errors="0" failures="%d" skipped="%d" tests="%d">%s</testsuite></testsuites>'
            % (len(failed), sum(1 for v in files.values() for n in v if n in skipped), sum(len(v) for v in files.values()), cases))


def suite_record(commit, suite, gpu, files, failed=(), ignored=(), subset=False, host="node-a", skipped=None, big=None):
    """A ``cfbi-suite-v1`` record of ``files`` ({classname: names}) as the node pipelines build them from a JUnit file (``big``: the CFBI_BIG that the record says it ran with)."""
    if big is None:
        big = "1" if suite == "S" else "0"
    meta = {"dirty": False, "sm_count": 148, "tree": "at-" + commit[:7], "env": {"CFBI_BIG": big}, "python": "3.12.3", "torch": "2.14.0", "cuda": "13.4",
            "host": host, "ignored": list(ignored), "subset": subset, "label": "part", "junit": "part.xml"}
    return tbt.suite_record_of(junit_of(files, failed, skipped), commit=commit, suite=suite, gpu=gpu, **meta)


def merged_suite(commit, gpu, files, expected_files, expected_ids, failed=(), skipped=None):
    """Suite S as a merge of the run without the sanitizer file and the sanitizer cases (the way ``suite_record.py --merge`` makes it), verified against the expected collection."""
    rest = dict((c, n) for c, n in files.items() if "k2_dyn" not in c)
    parts = [suite_record(commit, "S", gpu, rest, failed, ignored=(SANITIZER,), subset=True, skipped=skipped),
             suite_record(commit, "S", gpu, {"tests.cfbi.test_k2_dyn_sanitizer": files["tests.cfbi.test_k2_dyn_sanitizer"]}, failed, subset=True, host="node-b", skipped=skipped)]
    infos = [{"file": "part-%d.json" % i, "sha256": hashlib.sha256(json.dumps(part, sort_keys=True).encode()).hexdigest()} for i, part in enumerate(parts)]
    return sr.merge_records(parts, infos, "merge of two runs", expected_files, expected_ids, "collection")


def collection_file(path, commit, ids, big):
    Path(path).write_text("# pytest --collect-only -q tests/cfbi, CFBI_BIG=%s, snapshot of commit %s, conda env tlang, 2026-11-13T00:00:00Z\n%s\n" % (big, commit, "\n".join(ids)))
    return str(path)


def exactness_world(tmp_path, failed=()):
    """A repository (the base with four test files, the step's kernel change, a later commit that adds the step's own test file: the commit of the suites), the expected collections of the base and
    of that commit and a complete record of Gate G and of Suite S (a merge) on each device, with the records of the base as baselines; returns ``(manifest, repo, facts)``. The base and the
    commit of the suites skip the one test that is skipped by design."""
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    for rel in (GATE_FILE, "tests/cfbi/test_x.py", GUARD, SANITIZER):
        commit_file(repo, rel, "# test file\n", "a test file")
    base = git(repo, "rev-parse", "HEAD")
    step = commit_file(repo, "csrc/k.cu", "the step\n", "the step")
    suite = commit_file(repo, "tests/cfbi/test_new_mechanism.py", "# the step's own tests\n", "the tests of the step")
    world = dict(BASE_TESTS, **STEP_TESTS)
    files = sorted("tests/cfbi/%s.py" % c.rsplit(".", 1)[1] for c in world)

    def gate_of(tests):
        return {"tests.cfbi.test_vs_ref": tests["tests.cfbi.test_vs_ref"]}
    collections = {"S": collection_file(tmp_path / "collected-S.txt", suite, ids_of(world), 1), "gate": collection_file(tmp_path / "collected-gate.txt", suite, ids_of(gate_of(world)), 0)}
    base_collections = {"S": collection_file(tmp_path / "collected-base-S.txt", base, ids_of(BASE_TESTS), 1), "gate": collection_file(tmp_path / "collected-base-gate.txt", base, ids_of(gate_of(BASE_TESTS)), 0)}
    suites = {}
    for dev in rc.DEVICES:
        gpu, sub = trs.GPU[dev][0], tmp_path / dev
        sub.mkdir()
        recs = {"gate": (suite_record(base, "gate", gpu, gate_of(BASE_TESTS)), suite_record(suite, "gate", gpu, gate_of(world), failed)),
                "S": (suite_record(base, "S", gpu, BASE_TESTS, skipped=BY_DESIGN), merged_suite(suite, gpu, world, files, ids_of(world), failed, skipped=BY_DESIGN))}
        suites[dev] = {}
        for name, (baseline, candidate) in recs.items():
            paths = {}
            for role, rec in (("baseline", baseline), ("candidate", candidate)):
                paths[role] = str(sub / ("%s-%s.json" % (name, role)))
                Path(paths[role]).write_text(json.dumps(rec))
            suites[dev][name] = dict(paths, collection=collections[name], baseline_collection=base_collections[name])
    m = {"schema": sp.SCHEMA, "step": "trims", "commit": step, "suite_commit": suite, "previous": base, "base": base, "markers": ["x"], "suites": suites}
    return m, repo, {"base": base, "step": step, "suite": suite, "world": world, "files": files, "collections": collections, "base_collections": base_collections}


def edit_record(m, dev, name, role, edit):
    path = Path(m["suites"][dev][name][role])
    rec = json.loads(path.read_text())
    edit(rec)
    path.write_text(json.dumps(rec))


def recount(rec):
    rec["totals"] = dict((k, sum(c[k] for c in rec["files"].values())) for k in rec["totals"])


def as_one_run(rec):
    """An edited record is the record of one run: the parts and the expected collection of a merge no longer describe it."""
    rec["meta"].pop("parts", None)
    rec["meta"].pop("expected", None)
    rec["meta"]["kind"] = "single"


def skip_test(tid, reason):
    """The edit that turns the outcome of the test ``tid`` into a skip with ``reason``: the counts, the totals and the list of skipped tests stay consistent (a record that a run could have written)."""
    def edit(rec):
        file, was = tid.split("::")[0], rec["tests"][tid]
        rec["tests"][tid] = "skipped"
        rec["files"][file][was] -= 1
        rec["files"][file]["skipped"] += 1
        recount(rec)
        rec["failed"] = [t for t in rec["failed"] if t != tid]
        rec["skipped"] = sorted(rec["skipped"] + [{"id": tid, "reason": reason}], key=lambda s: s["id"])
    return edit


def reason_of(tid, reason):
    """The edit that gives the skip of ``tid`` another reason."""
    def edit(rec):
        for s in rec["skipped"]:
            if s["id"] == tid:
                s["reason"] = reason
    return edit


def set_env(env):
    """The edit that replaces the environment that a record says it ran in (None: the record has none)."""
    def edit(rec):
        if env is None:
            rec["meta"].pop("env", None)
        else:
            rec["meta"]["env"] = env
    return edit


def one_id_per_file(rec):
    """A record cut to the first test of every file (all files are kept, the counts say what is left): structurally valid, and 1 of n ids per file."""
    keep = {}
    for tid in sorted(rec["tests"]):
        keep.setdefault(tid.split("::")[0], tid)
    rec["tests"] = dict((t, rec["tests"][t]) for t in keep.values())
    for f in rec["files"]:
        rec["files"][f] = dict.fromkeys(rec["files"][f], 0)
        rec["files"][f][rec["tests"][keep[f]]] = 1
        rec["files"][f]["total"] = 1
    recount(rec)
    rec["failed"] = [t for t in rec["failed"] if t in rec["tests"]]
    rec["skipped"] = [s for s in rec["skipped"] if s["id"] in rec["tests"]]
    as_one_run(rec)


def test_complete_records_bound_to_the_collection_of_the_suite_commit_are_the_exactness_evidence(tmp_path):
    m, repo, facts = exactness_world(tmp_path)
    assert sp.exactness_problems(m, repo) == [], sp.exactness_problems(m, repo)
    cand = json.loads(Path(m["suites"]["b200"]["S"]["candidate"]).read_text())
    assert cand["meta"]["kind"] == "merge" and cand["meta"]["subset"] is False and sorted(cand["files"]) == facts["files"]            # the legitimate merge of the two runs
    mf, repof, _ = exactness_world(tmp_path / "failing", failed=("test_pin",))                                                          # complete, but a test of the step failed
    found = sp.exactness_problems(mf, repof)
    assert any("failed or errored test tests/cfbi/test_new_mechanism.py::test_pin" in p and p.startswith("b200, suite S") for p in found), found[:3]


def test_a_record_that_is_not_the_whole_suite_at_the_commit_of_the_suites_is_no_exactness_evidence(tmp_path):
    def subset(rec):                                                          # the run without the sanitizer file, as a subset record: what round 12 had
        as_one_run(rec)
        rec["meta"].update(subset=True, ignored=[SANITIZER])
        for tid in [t for t in rec["tests"] if "k2_dyn" in t]:
            rec["tests"].pop(tid)
        rec["files"].pop("tests/cfbi/test_k2_dyn_sanitizer.py")
        recount(rec)
        rec["skipped"] = [s for s in rec["skipped"] if "k2_dyn" not in s["id"]]

    def without_file(file):
        def edit(rec):
            for tid in [t for t in rec["tests"] if t.startswith(file + "::")]:
                rec["tests"].pop(tid)
            rec["files"].pop(file)
            recount(rec)
            as_one_run(rec)
        return edit

    def without_id(tid):
        def edit(rec):
            file = tid.split("::")[0]
            rec["tests"].pop(tid)
            rec["files"][file]["passed"] -= 1
            rec["files"][file]["total"] -= 1
            recount(rec)
            as_one_run(rec)
        return edit
    cases = [("a subset record that ignores the sanitizer file", subset, ["subset record", "ignores " + SANITIZER]),
             ("a record without the whole file of the step's own tests", without_file("tests/cfbi/test_new_mechanism.py"), ["expected file(s) are missing: tests/cfbi/test_new_mechanism.py", "expected test(s) are missing"]),
             ("a record without one control of the step", without_id("tests/cfbi/test_new_mechanism.py::test_control"), ["1 expected test(s) are missing: tests/cfbi/test_new_mechanism.py::test_control"]),
             ("a record without the K2 sanitizer module", without_file(SANITIZER), ["expected file(s) are missing: " + SANITIZER])]
    for i, (label, edit, needles) in enumerate(cases):
        m, repo, _ = exactness_world(tmp_path / ("case%d" % i))
        edit_record(m, "b300", "S", "candidate", edit)
        found = sp.exactness_problems(m, repo)
        assert found and all(p.startswith("b300, suite S") for p in found), (label, found[:3])
        for needle in needles:
            assert any(needle in p for p in found), (label, needle, found[:4])
    m, repo, _ = exactness_world(tmp_path / "baseline")                      # a baseline that leaves files out certifies nothing about them
    edit_record(m, "b200", "S", "baseline", subset)
    assert any("baseline record is not complete" in p and "subset record" in p for p in sp.exactness_problems(m, repo)), sp.exactness_problems(m, repo)[:3]
    m, repo, _ = exactness_world(tmp_path / "other-base")                    # a baseline of another commit than the base
    edit_record(m, "b200", "gate", "baseline", lambda rec: rec["meta"].update(commit="f" * 40))
    assert any("the baseline record is of commit" in p for p in sp.exactness_problems(m, repo)), sp.exactness_problems(m, repo)[:3]


def test_the_expected_collection_is_the_one_of_the_suite_commit_and_of_the_environment_of_the_suite(tmp_path):
    m, repo, facts = exactness_world(tmp_path)
    for label, edit, needle in (("another commit", lambda path, text: text.replace(facts["suite"], "e" * 40), "does not name the commit of the suites"),
                                ("another environment", lambda path, text: text.replace("CFBI_BIG=1", "CFBI_BIG=0"), "CFBI_BIG=1"),
                                ("a file that the commit does not have", lambda path, text: text + "tests/cfbi/test_gone.py::test_z\n", "not a test file of the commit")):
        path = Path(facts["collections"]["S"])
        original = path.read_text()
        path.write_text(edit(path, original))
        found = sp.exactness_problems(m, repo)
        assert any(needle in p and "suite S" in p for p in found), (label, found[:3])
        path.write_text(original)
    Path(facts["collections"]["gate"]).unlink()
    assert any("collection" in p and "cannot be read" in p and "suite gate" in p for p in sp.exactness_problems(m, repo))


def test_a_commit_that_the_repository_does_not_have_is_a_named_problem_not_a_skipped_check(tmp_path):
    m, repo, _ = exactness_world(tmp_path)
    found = sp.exactness_problems(dict(m, base="1" * 40), repo)
    assert any("cannot list tests/cfbi of the commit 1111111111" in p and p.startswith("b200, suite S") for p in found), found[:3]          # the base: its files cannot be read
    found = sp.exactness_problems(dict(m, suite_commit="2" * 40), repo)
    assert any("cannot list tests/cfbi of the commit 2222222222" in p or "does not have the commit of the suites" in p for p in found), found[:3]       # the commit of the suites


def test_gate_g_is_held_to_its_own_file_and_ids_and_the_manifest_names_a_collection_for_each_suite(tmp_path):
    m, repo, _ = exactness_world(tmp_path)
    edit_record(m, "b200", "gate", "candidate", lambda rec: (rec["tests"].update({"tests/cfbi/test_vs_ref.py::test_extra": "passed"}), rec["files"]["tests/cfbi/test_vs_ref.py"].update(passed=3, total=3),
                                                             rec["totals"].update(passed=3, total=3)))
    assert any("suite gate" in p and "1 test(s) are not expected" in p for p in sp.exactness_problems(m, repo)), sp.exactness_problems(m, repo)[:3]
    good = exactness_world(tmp_path / "again")[0]
    del good["suites"]["b300"]["S"]["collection"]
    assert any("suites.b300.S must name" in p and "collection" in p for p in sp.manifest_problems(dict(good, devices=dict((d, {"baseline": "b", "previous": "p", "base": "q"}) for d in rc.DEVICES),
                                                                                                           codegen={"allow": "a", "conda": {"base": "b", "cand": "c", "prev": "p"}, "container_b200": {"base": "b", "cand": "c", "prev": "p", "occupancy": "o"},
                                                                                                                    "container_b300": {"base": "b", "cand": "c", "prev": "p", "occupancy": "o"}})))


NEW = "tests/cfbi/test_new_mechanism.py::"


def test_a_skipped_test_is_no_exactness_evidence_unless_it_is_skipped_by_design(tmp_path):
    """A complete record in which the tests of a step were SKIPPED holds every expected id and fails nothing, and it proves nothing about those tests: only the test that is skipped by design on
    B200 and B300 (named in the tool with the beginning of its reason) may be skipped, in the baseline as in the candidate."""
    m, repo, _ = exactness_world(tmp_path)
    assert sp.exactness_problems(m, repo) == []                                                   # the legal skip: the test that is skipped by design is skipped in both records of Suite S
    both = [skip_test(NEW + "test_pin", "no CUDA device"), skip_test(NEW + "test_control", "no CUDA device")]
    cases = [("every test of the step's own file skipped", "candidate", both, [NEW + "test_pin", NEW + "test_control"]),
             ("one control of the step skipped", "candidate", both[1:], [NEW + "test_control"]),
             ("a test of the base skipped in the candidate", "candidate", [skip_test("tests/cfbi/test_x.py::test_a", "no CUDA device")], ["tests/cfbi/test_x.py::test_a"]),
             ("the test that is skipped by design, for another reason", "candidate", [reason_of(GUARD + "::" + GUARD_TEST, "no CUDA device")], [GUARD + "::" + GUARD_TEST]),
             ("a test of the base skipped in the baseline", "baseline", [skip_test("tests/cfbi/test_x.py::test_b", "no CUDA device")], ["tests/cfbi/test_x.py::test_b"])]
    for i, (label, role, edits, ids) in enumerate(cases):
        mm, rr, _ = exactness_world(tmp_path / ("case%d" % i))
        for edit in edits:
            edit_record(mm, "b200", "S", role, edit)
        found = sp.exactness_problems(mm, rr)
        assert found and all(p.startswith("b200, suite S") for p in found), (label, found[:3])
        for tid in ids:
            assert any("the %s record skips %s" % (role, tid) in p for p in found), (label, tid, found[:3])
    mm, rr, _ = exactness_world(tmp_path / "gate")                                                # Gate G has no by-design skip at all
    edit_record(mm, "b300", "gate", "candidate", skip_test("tests/cfbi/test_vs_ref.py::test_g2", "no CUDA device"))
    assert any(p.startswith("b300, suite gate") and "the candidate record skips tests/cfbi/test_vs_ref.py::test_g2" in p for p in sp.exactness_problems(mm, rr))


def test_a_skipped_racecheck_is_the_documented_exception_only_with_substitute_controls_that_passed(tmp_path):
    """The sanitizer policy: when racecheck says that it cannot analyse the kernels, the hazard class is covered by the step's negative controls, which gate in its place. The skip of the one test
    that does that is allowed when the controls named by the manifest passed in the same record, and never without them."""
    rid, control = SANITIZER + "::" + RACECHECK, NEW + "test_control"
    m, repo, _ = exactness_world(tmp_path)
    for dev in rc.DEVICES:                                                                         # the tool cannot analyse the kernels of the device: the base skipped the test too
        for role in ("baseline", "candidate"):
            edit_record(m, dev, "S", role, skip_test(rid, RACECHECK_REASON))
    found = sp.exactness_problems(m, repo)
    assert found and any("the candidate record skips racecheck" in p and "substitute controls" in p for p in found), found[:3]       # no control named: a skipped racecheck is a skipped test
    for dev in rc.DEVICES:
        m["suites"][dev]["S"]["substitute_controls"] = [control]
    assert sp.exactness_problems(m, repo) == [], sp.exactness_problems(m, repo)[:3]                 # the control passed in the record: the documented exception
    edit_record(m, "b300", "S", "candidate", skip_test(control, "no CUDA device"))
    found = sp.exactness_problems(m, repo)
    assert any(p.startswith("b300, suite S") and "the substitute control %s" % control in p and "did not pass" in p for p in found), found[:3]       # the control itself was skipped
    m["suites"]["b300"]["S"]["substitute_controls"] = [NEW + "test_absent"]
    assert any("the substitute control %s" % (NEW + "test_absent") in p and "absent" in p for p in sp.exactness_problems(m, repo))
    m2, repo2, _ = exactness_world(tmp_path / "other-reason")                                      # a skip with another reason is not the exception, whatever the controls say
    for dev in rc.DEVICES:
        m2["suites"][dev]["S"]["substitute_controls"] = [control]
    edit_record(m2, "b200", "S", "candidate", skip_test(rid, "no CUDA device"))
    assert any("the candidate record skips %s" % rid in p for p in sp.exactness_problems(m2, repo2))


def test_the_tests_that_the_tool_allows_to_skip_are_in_the_sources_with_that_reason():
    """The tables of the skips that the tool allows name real tests and reasons: a renamed test or a reworded reason is found here, not on a device."""
    assert sp.BY_DESIGN_SKIPS and sp.UNSUPPORTED_RACECHECK_SKIPS
    for tid, reason in list(sp.BY_DESIGN_SKIPS.items()) + list(sp.UNSUPPORTED_RACECHECK_SKIPS.items()):
        path, name = tid.split("::", 1)
        text = (_ROOT / path).read_text()
        assert "def %s(" % name.split("[", 1)[0] in text and reason in text, tid


def test_the_environment_of_both_records_is_the_one_of_the_suite_and_the_header_of_a_collection_is_read_as_a_field(tmp_path):
    """The collection says which CFBI_BIG the ids were collected with; the records say which they ran with. Both are bound to the suite (S: 1, Gate G: 0) for the baseline and the candidate alike:
    records that both ran with the other value, a record that says nothing and a header with another value that has the right one as its beginning are all refused."""
    m, repo, _ = exactness_world(tmp_path)
    assert sp.exactness_problems(m, repo) == []
    cases = [("both records of Suite S ran with CFBI_BIG=0", "S", {"baseline": set_env({"CFBI_BIG": "0"}), "candidate": set_env({"CFBI_BIG": "0"})}, ["the baseline record ran with CFBI_BIG='0'", "the candidate record ran with CFBI_BIG='0'"]),
             ("both records of Suite S ran with CFBI_BIG=10", "S", {"baseline": set_env({"CFBI_BIG": "10"}), "candidate": set_env({"CFBI_BIG": "10"})}, ["the candidate record ran with CFBI_BIG='10'"]),
             ("the candidate says nothing about its environment", "S", {"baseline": set_env({"CFBI_BIG": "1"}), "candidate": set_env({})}, ["the candidate record does not say which CFBI_BIG"]),
             ("neither record has an environment", "S", {"baseline": set_env(None), "candidate": set_env(None)}, ["the baseline record does not say which CFBI_BIG", "the candidate record does not say which CFBI_BIG"]),
             ("the value is a number", "S", {"baseline": set_env({"CFBI_BIG": 1}), "candidate": set_env({"CFBI_BIG": 1})}, ["the baseline record ran with CFBI_BIG=1"]),
             ("both records of Gate G ran with CFBI_BIG=1", "gate", {"baseline": set_env({"CFBI_BIG": "1"}), "candidate": set_env({"CFBI_BIG": "1"})}, ["the baseline record ran with CFBI_BIG='1'", "the candidate record ran with CFBI_BIG='1'"])]
    for i, (label, name, edits, needles) in enumerate(cases):
        mm, rr, _ = exactness_world(tmp_path / ("env%d" % i))
        for role, edit in edits.items():
            edit_record(mm, "b200", name, role, edit)
        found = sp.exactness_problems(mm, rr)
        assert found and all(p.startswith("b200, suite %s" % name) for p in found), (label, found[:3])
        for needle in needles:
            assert any(needle in p and "needs CFBI_BIG=%s" % sp.SUITE_BIG[name] in p for p in found), (label, needle, found[:3])
    for which in ("collection", "baseline_collection"):                                          # the header of the expected collections, the suite's and the base's
        for label, edit in (("a header with CFBI_BIG=10", lambda t: t.replace("CFBI_BIG=1,", "CFBI_BIG=10,")), ("a header with two fields", lambda t: t.replace("CFBI_BIG=1,", "CFBI_BIG=1, CFBI_BIG=0,")),
                            ("a header without the field", lambda t: t.replace("CFBI_BIG=1,", "")), ("a field that continues", lambda t: t.replace("CFBI_BIG=1,", "CFBI_BIG=1x,"))):
            path = Path(m["suites"]["b200"]["S"][which])
            original = path.read_text()
            path.write_text(edit(original))
            found = sp.exactness_problems(m, repo)
            assert any("was not made with CFBI_BIG=1" in p and p.startswith("b200, suite S") and path.name in p for p in found), (which, label, found[:3])
            path.write_text(original)
    assert sp.exactness_problems(m, repo) == []


def test_the_baseline_is_complete_against_the_collection_of_the_base_and_the_manifest_binds_that_collection(tmp_path):
    """The baseline is held to the exact ids of the collection of the base, like the candidate to those of the commit of the suites: a record that keeps every file of the base and one test of
    each is no baseline. The manifest names the collection, and its header names the base and the environment of the suite."""
    m, repo, facts = exactness_world(tmp_path)
    assert sp.exactness_problems(m, repo) == []
    mm, rr, _ = exactness_world(tmp_path / "cut")
    edit_record(mm, "b300", "S", "baseline", one_id_per_file)
    found = sp.exactness_problems(mm, rr)
    assert any(p.startswith("b300, suite S") and "the baseline record is not complete" in p and "expected test(s) are missing" in p for p in found), found[:3]
    mm, rr, _ = exactness_world(tmp_path / "old")                                                   # a baseline record of the older format: counts per file, no outcome per test
    edit_record(mm, "b200", "gate", "baseline", lambda rec: rec.pop("tests"))
    assert any(p.startswith("b200, suite gate") and "the baseline record is not complete" in p and "no per-test outcomes" in p for p in sp.exactness_problems(mm, rr))

    def with_an_id_that_the_base_never_had(rec):
        rec["tests"]["tests/cfbi/test_x.py::test_extra"] = "passed"
        rec["files"]["tests/cfbi/test_x.py"].update(passed=3, total=3)
        recount(rec)
    mm, rr, _ = exactness_world(tmp_path / "more")
    edit_record(mm, "b200", "S", "baseline", with_an_id_that_the_base_never_had)
    assert any("the baseline record is not complete" in p and "1 test(s) are not expected" in p for p in sp.exactness_problems(mm, rr))
    spec = m["suites"]["b200"]["S"]
    path = Path(facts["base_collections"]["S"])
    original = path.read_text()
    for label, text, needle in (("a collection of the suites' commit", original.replace(facts["base"], facts["suite"]), "does not name the base"),
                                ("a collection made with CFBI_BIG=0", original.replace("CFBI_BIG=1,", "CFBI_BIG=0,"), "was not made with CFBI_BIG=1"),
                                ("a file that the base does not have", original + "tests/cfbi/test_new_mechanism.py::test_pin\n", "not a test file of the commit")):
        path.write_text(text)
        found = sp.exactness_problems(m, repo)
        assert any(needle in p and p.startswith("b200, suite S") and path.name in p for p in found), (label, found[:3])
        path.write_text(original)
    path.unlink()
    assert any(p.startswith("b200, suite S") and "collected-base-S.txt cannot be read" in p for p in sp.exactness_problems(m, repo))
    path.write_text(original)
    del spec["baseline_collection"]                                                                 # a manifest that names no collection of the base: a named problem, not an exception
    assert any(p.startswith("b200, suite S") and "baseline_collection" in p for p in sp.exactness_problems(m, repo))


def test_the_commit_of_the_suites_stays_bound_to_the_step_by_its_kernel_sources(tmp_path):
    m, repo, facts = exactness_world(tmp_path)
    changed = commit_file(repo, "flash_kda/cfbi/_ext.py", "a flag of the build\n", "a compiled input changes after the step")
    bad = dict(m, suite_commit=changed)
    found = sp.exactness_problems(bad, repo)
    assert any("kernel sources differ from the step's" in p for p in found), found[:3]
    assert any("the candidate record is of commit" in p for p in found)


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


# ---- a step that ended by the budget of its repairs: the row is built from the evidence, the checker recomputes the proof, the section of the document holds only numbers of the records -------------
def budget_case(tmp_path):
    """The manifest of a step (the dumps of the record checker's own tests, the suites of ``exactness_world``), the termination of its repairs and the row that the builder makes of them."""
    m, _, _ = exactness_world(tmp_path / "world")
    m = dict(m, step="e_tile", commit=trs.C2, previous=trs.C1, markers=["tile_marker"], revert_commit=trs.C3)
    codegen = {}
    for key in rc.COMPARISONS:
        entry, _ = trs.comparison(key, m["commit"], sid="e_tile")
        codegen[key] = {"base": entry["base_dump"], "cand": entry["cand_dump"], "prev": trs.dump_file(key, trs.C1, "cand")}
        if key.startswith("container_"):
            codegen[key]["occupancy"] = entry["occupancy"]
    m["codegen"] = dict(codegen, allow=trs.allow_file({}))
    (tmp_path / "budget").mkdir()
    termination = trs.budget_row(tmp_path / "budget")["termination"]
    return m, termination, sp.build_termination_row(m, termination, "the repairs of the failed resource gate exceeded the budget of the gate protocol", exactness=[])


def test_the_row_of_a_step_that_ended_by_the_budget_of_its_repairs_is_built_from_the_evidence_and_passes_the_record_checker(tmp_path):
    m, termination, row = budget_case(tmp_path)
    assert row["failed_at"] == "repair_budget" and row["outcome"] == "not_adopted" and row["gates"] == {"exactness": True, "codegen": True}
    assert row["timing"] == {"vs_previous": "not_run", "vs_base": "not_run", "chain": "not_run"} and row["ab"] is None and row["revert_commit"] == trs.C3
    assert all(e["verdict"] == "PASS" for e in row["codegen_gate"].values())                      # the final form passes: the row does not call it failed
    assert rc.step_problems(row, None, trs.META) == [], rc.step_problems(row, None, trs.META)
    assert row["termination"]["counted"] == 4 and row["termination"]["allowed"] == rc.REPAIR_BUDGET
    failed = sp.build_termination_row(m, termination, "x", exactness=["a suite did not pass"])      # exactness is what the evidence gave
    assert failed["gates"]["exactness"] is False and rc.step_problems(failed, None, trs.META) == []
    no_revert = sp.build_termination_row(dict(m, revert_commit=None), termination, "x", exactness=[])
    assert any("is not reverted" in p for p in rc.step_problems(no_revert, None, trs.META))


def test_the_termination_of_a_row_is_counted_by_the_checker_alone(tmp_path):
    m, termination, row = budget_case(tmp_path)
    rebuilt = sp.repair_termination(termination["first_failed_gate"]["dump"], termination["attempts"])
    assert rebuilt == termination and rebuilt["counted"] == 4                                  # four repairs, one ablation: the ablation is not counted
    fewer = sp.repair_termination(termination["first_failed_gate"]["dump"], termination["attempts"][:3] + termination["attempts"][4:])
    assert fewer["counted"] == 3 and any("not exhausted" in p for p in rc.step_problems(dict(row, termination=fewer), None, trs.META))


def test_the_section_of_a_step_that_ended_by_the_budget_holds_the_numbers_of_the_records_and_no_plan_vocabulary(tmp_path):
    m, termination, row = budget_case(tmp_path)
    text = sp.render_termination_section(m, row, number=11)
    lines = text.split("\n")
    assert lines[0] == "## 11. Row engine step 2: the staged E tile" and "The step is **not adopted**: it ended by the budget of its repairs, not by a measurement." in lines
    assert any("4 distinct repair candidates were dumped after it (1 ablation(s)" in line for line in lines)
    attempts = [line for line in lines if line.startswith("| V")]
    assert len(attempts) == 5 and sum(1 for line in attempts if "| repair |" in line) == 4 and sum(1 for line in attempts if "| ablation |" in line) == 1
    for dev in rc.DEVICES:                                                           # the suites: the numbers of the complete candidate records, one row per device
        gate, whole = (json.loads(Path(m["suites"][dev][name]["candidate"]).read_text()) for name in ("gate", "S"))
        expected = "| %s | %d | %d | %d | %d | %d | %s |" % (dev, gate["totals"]["passed"], len(whole["files"]), whole["totals"]["passed"], whole["totals"]["skipped"],
                                                           whole["totals"]["failed"] + whole["totals"].get("errors", 0), m["suite_commit"][:12])
        assert lines.count(expected) == 1, (expected, [l for l in lines if l.startswith("| %s |" % dev)][:2])
    assert sum(1 for line in lines if line.startswith("| conda |")) == 3 and sum(1 for line in lines if line.startswith("| container_b200 |")) == 1
    assert "| b200 | cfbi::rows_kernel | 128 | 8192 B | 0 B | 2 |" in lines and "| b300 | cfbi::rows_kernel | 128 | 8192 B | 0 B | 2 |" in lines            # the occupancy of every instance
    assert any(line.startswith("Gate X in change mode") and "conda PASS" in line for line in lines) and "Not run: the step ended before the timing phase" in text
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert not rc.PLAN_VOCABULARY.search(text), rc.PLAN_VOCABULARY.search(text).group(0)
    assert not re.search(r"\b(Q|B|M|R)[0-9]{1,2}\b", text)                           # nor the loop's numbers
