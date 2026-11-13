"""The decision, the ledger row, the section of the performance document and the command line of the single-launch decode record (benchmarks/single_launch_record.py).

Plain Python, no GPU and no torch (``pytest --noconftest tests/cfbi/test_single_launch_record.py``). The decision is exercised on stub comparator results (hand-computable figures, see
test_decode_boundary_table.py) and on the REAL comparator: synthetic raw record files in the ABBA order (every cell has a median of 100 microseconds for A, 97 where the single launch is 3 % faster)
are read by the tool, the comparator results are computed again from them with the parameters of the campaign driver, the table is derived, checked, rendered and checked again. Every rule has a
mutant that the check rejects: a table figure that was edited by hand, a record whose launches are not those of the routing specification, a document number, a ledger median.
"""
import copy
import functools
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "benchmarks", _ROOT / "tests" / "cfbi"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bench_decode as bd  # noqa: E402
import bench_records as br  # noqa: E402
import decode_policy as dp  # noqa: E402
import record_checker as rc  # noqa: E402
import single_launch_record as slr  # noqa: E402
import test_decode_boundary_table as tdb  # noqa: E402
from test_decode_boundary_table import CLEAN, CONSTANTS, K_ABOVE, K_BIG, K_COLD, K_SMALL, K_WITHIN, LIMIT, PER_SM, SMS, at, default_values, has, key_of, mutated, row_of, stub_result, stub_row  # noqa: E402

GPU = {"b200": "NVIDIA B200", "b300": "NVIDIA B300 SXM6 AC"}
CC = {"b200": "10.0", "b300": "10.3"}
COMMIT = tdb.CANDIDATE
PLAN = br.run_plan(4, "abba")                              # four record files per side, in the order of the first block ABBA and the second BAAB
OFFSETS = (-1.0, 0.0, 0.0, 1.0)                            # the process medians of the four files of a side are m (1 + e * OFFSET): their median is m, their spread is 2 e
TWO, SINGLE = dp.TWO_LAUNCH, dp.SINGLE_LAUNCH


@pytest.fixture(autouse=True)
def the_tree_of_the_synthetic_campaigns(monkeypatch):
    """The synthetic campaigns of this module are measured at PER_SM CTAs per SM, so the tree that they were measured at says that (the limit of the real tree is another number, and
    test_decode_boundary_table.py checks it against the header and the survey)."""
    monkeypatch.setattr(slr.dp, "header_per_sm", lambda root: PER_SM)
    monkeypatch.setattr(slr.dp, "MAX_CTAS_PER_SM", PER_SM)


# ---- synthetic raw records, the way the benchmark writes them --------------------------------------------------------------------------------------------
def stamp(dev, arm, phase, slot):
    hour = 6 + 4 * list(tdb.DEVICES).index(dev) + 2 * list(slr.ARMS).index(arm) + list(slr.PHASES).index(phase)
    return "2026-11-11T%02d:%02d:00.000000Z" % (hour, 3 * slot)


def raw_meta(dev, arm, side, phase, slot, scale, per_sm, commit):
    return {"commit": commit, "dirty": False, "tree": "snapshot", "gpu": GPU[dev], "sm_count": SMS, "cc": CC[dev], "host": "node-" + dev, "torch": "2.9.0", "cuda": "13.0",
            "timestamp": stamp(dev, arm, phase, slot), "kind": "decode-wall", "sample_stat": br.PROCESS_MEDIAN, "impl": 64 if side == "A" else 0, "env_impl": None, "regimes": list(bd.REGIMES),
            "trials": 5, "calls_per_trial": {r: 1536 * scale for r in bd.REGIMES}, "profile_calls": {r: 64 * scale for r in bd.REGIMES}, "profile_calls_option": 64, "calls_scale": scale,
            "l2_cache_bytes": 1 << 26, "eviction_bytes": (1 << 27) + (1 << 20), "single_force": side == "B" and arm == "forced", "single_per_sm": per_sm, "boundary_sms": SMS,
            "graph_nodes": False, "route_order": arm == "default", "cuda_module_loading": "EAGER", "cells_filter": None, "label": "%s-%s-%s" % (side, arm, phase), "tag": None, "build_dir": None, "src": "tree", "argv": []}


def metrics_of(f):
    return [m for m in slr.GATE_METRICS if m not in slr.LOOP_METRICS or f["regime"] in bd.LOOP_REGIMES]


def raw_cell(key, route, medians, scale, force, side):
    """One decode cell: five trials whose median is ``medians[metric]`` for every metric (the kernel sum is the plan kernel of 2 microseconds and the row engine, or the row engine alone)."""
    f = slr.cell_fields(key)
    compute = dict(bd.CANONICAL_COMPUTE[route])
    runs, walls = [], []
    for t in range(5):
        factor = 1.0 + (t - 2) * 1e-4                                    # the middle trial has the median exactly
        total = medians["total_us"] * factor
        runs.append({"cfbi::plan_kernel": 2.0 * factor, "cfbi::rows_kernel": total - 2.0 * factor} if route == TWO else {"cfbi::rows_kernel": total})
        walls.append({m: v * factor for m, v in medians.items() if m != "total_us"})
    base_calls = 1536 if f["regime"] in bd.LOOP_REGIMES else 256
    loop = f["regime"] != "probe"
    args = {"N": f["N"], "H": f["H"], "regime": f["regime"], "pos": f["pos"], "impl": 64 if side == "A" else 0, "calls_per_trial": base_calls * scale, "profile_calls": 64 * scale,
            "profile_cycles": scale if loop else None, "profile_start_phase": 1 if loop else None, "calls_scale": scale, "kernels": sorted(k.replace("cfbi::", "") for k in runs[0]),
            "compute_nodes": compute, "compute_kernels": sum(compute.values()), "aux_nodes": {}, "expect_compute": compute, "single_force": force, "expect_route": "spec" if side == "B" else None}
    cell = br.make_cell("decode", key, args, runs, drop_helpers=True, wall=walls)
    for sample in cell["samples"]:
        sample["nodes"] = {"compute": dict(compute), "aux": {}}
    return cell


def raw_records(dev, arm, values, per_sm=PER_SM, commit=COMMIT, cells=None):
    """The record files of one arm of one device: ``{"first": {"A": [...], "B": [...]}, "rerun": {...}}``. The figures of every cell and metric come from ``values`` (the stub scenarios of
    test_decode_boundary_table.py): base, change in percent, spread in percent, and whether the cell is judged on the re-run, which the driver gets by re-running the cell."""
    if cells is None:
        cells = slr.expected_cells(per_sm, SMS, {d: slr.cc_number(CC[d]) for d in CC})[dev][arm]
    figures = {key: {m: values(dev, arm, m, key, slr.cell_fields(key)) for m in metrics_of(slr.cell_fields(key))} for key in cells}
    out = {"first": {"A": [], "B": []}, "rerun": {"A": [], "B": []}}
    for phase, keys, scale in (("first", list(cells), 1), ("rerun", [k for k in cells if any(v["judged_on"] == "rerun" for v in figures[k].values())], 3)):
        for slot, (side, index) in enumerate(PLAN if keys else []):
            recs = []
            for key in keys:
                f = slr.cell_fields(key)
                route = TWO if side == "A" else dp.expected_route(0, f["N"], f["H"], SMS, slr.cc_number(CC[dev]), per_sm, arm == "forced")
                medians = {}
                for m, v in figures[key].items():
                    delta = v.get("first_delta", v["delta"]) if phase == "first" else v["delta"]                  # the first campaign may have measured another change than its re-run
                    median = v["base"] if side == "A" else v["base"] * (1.0 + delta / 100.0)
                    medians[m] = median * (1.0 + v["spread"] / 200.0 * OFFSETS[index - 1])
                recs.append(raw_cell(key, route, medians, scale, side == "B" and arm == "forced", side))
            out[phase][side].append({"schema": br.SCHEMA, "meta": raw_meta(dev, arm, side, phase, slot, scale, per_sm, commit), "cells": recs})
    return out


def digest(rec):
    return hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()


def make_ev(values=default_values, per_sm=PER_SM, cells=None, arms=slr.ARMS, devices=tuple(tdb.DEVICES), commit=COMMIT):
    """The evidence of a campaign in memory (what ``load_evidence`` makes of a manifest): the records of every device and arm and the name and digest of each."""
    ev = {"devices": {}, "inputs": {}, "not_run": {}}
    for dev in devices:
        ev["devices"][dev], ev["inputs"][dev] = {}, {}
        for arm in arms:
            recs = raw_records(dev, arm, values, per_sm, commit, None if cells is None else cells[arm])
            ev["devices"][dev][arm] = recs
            ev["inputs"][dev][arm] = {phase: {side: [{"name": "%s-%s-%s-%s%d.json" % (dev, arm, phase, side, i + 1), "sha256": digest(r)} for i, r in enumerate(recs[phase][side])] for side in slr.SIDES}
                                      for phase in slr.PHASES}
    return ev


def write_ev(ev, directory):
    """The record files and the manifest of ``ev`` in ``directory``; returns the path of the manifest."""
    directory = Path(directory)
    manifest = {"schema": slr.EVIDENCE_SCHEMA, "devices": {}}
    for dev, arms in ev["devices"].items():
        manifest["devices"][dev] = {}
        for arm, phases in arms.items():
            manifest["devices"][dev][arm] = {}
            for phase in slr.PHASES:
                if not phases[phase]["A"]:
                    continue
                manifest["devices"][dev][arm][phase] = {}
                for side in slr.SIDES:
                    names = [e["name"] for e in ev["inputs"][dev][arm][phase][side]]
                    for name, rec in zip(names, phases[phase][side]):
                        (directory / name).write_text(json.dumps(dict((k, v) for k, v in rec.items() if k != "_path")))
                    manifest["devices"][dev][arm][phase][side] = names
    path = directory / "evidence.json"
    path.write_text(json.dumps(manifest))
    return path


MINI = {"default": [K_SMALL, key_of(4, 12, "staggered"), key_of(64, 12), K_BIG], "forced": [K_BIG, key_of(256, 12)]}      # a few cells: the decision does not need the whole campaign


def mini_results(values):
    ev = make_ev(values, cells=MINI)
    return ev, slr.recompute(ev)


def all_metrics(delta, judged_on="rerun", spread=0.1):
    """``values`` that give every metric of every cell of both arms the same change."""
    return lambda dev, arm, metric, key, f: {"base": 100.0, "delta": delta, "spread": spread, "judged_on": judged_on}


def limits():
    return {"b200": LIMIT, "b300": LIMIT}


@pytest.fixture(scope="module")
def full():
    """The whole campaign in the default scenario (the single launch wins within the limit, loses from 1536 CTAs on, one cell is noisy in its graph metric): evidence, comparator results, table."""
    values = at(key_of(16, 12), metrics=("gpu_graph_us",), spread=2.0, judged_on="rerun", base=default_values)
    ev = make_ev(values)
    results = slr.recompute(ev)
    return {"ev": ev, "results": results, "table": slr.build_table(ev, results), "values": values}


@pytest.fixture(scope="module")
def results_of(full):
    return full["results"]


# ---- the decision on stub results ---------------------------------------------------------------------------------------------------------------------
def decided(values=default_values, **kw):
    t = tdb.synthetic_table(values, **kw)
    return slr.decide_table(t), t


def test_the_default_scenario_is_adopted_with_the_gains_of_both_devices():
    d, t = decided()
    assert d["outcome"] == "adopted" and d == t["decision"]
    per_sm = {dev: d["per_device"][dev] for dev in slr.GATING}
    assert all(i["limit"] == LIMIT and i["regressions"] == 0 and i["invalid"] == 0 and set(i["verdicts"]) == set(slr.GATE_METRICS) and set(i["verdicts"].values()) == {"PASS"} for i in per_sm.values())
    gain_cells = [(1, 12), (4, 12), (16, 12), (64, 12), (17, 33), (18, 33), (35, 33), (49, 12), (50, 12), (98, 12), (98, 6), (99, 6), (197, 6)]       # the shapes within 1184 CTAs, hand-listed
    assert len(gain_cells) == 13 and all(N * H <= 1184 for N, H in gain_cells)
    assert {g["key"] for g in d["gains"]} == {key_of(N, H, r) for N, H in gain_cells for r in ("lockstep", "staggered")}
    assert per_sm["b200"]["gains"] == per_sm["b300"]["gains"] == 13 * 2 * 2 and len(d["gains"]) == 4 * 13 * 2
    assert {g["metric"] for g in d["gains"]} == {"gpu_graph_us", "gpu_eager_us"} and {g["delta_pct"] for g in d["gains"]} == {-3.0} and {g["base_median"] for g in d["gains"]} == {100.0}
    assert d["reasons"][0] == "no gating metric regresses on b200, b300" and "b300: 52 confirmed gain cell(s)" in d["reasons"][2]


@pytest.mark.parametrize("first,rerun,confirmed,what", [
    (-3.0, -3.0, True, "both campaigns beyond the tolerance"),
    (-1.0, -3.0, True, "the first campaign exactly at the tolerance"),
    (-2.5, -3.0, True, "a first campaign beyond the tolerance whose re-run is faster still (a noisy first campaign with the same sign is a legal one)"),
    (-0.2, -3.0, False, "a neutral first campaign and a fast re-run"),
    (-0.99, -3.0, False, "a first campaign just inside the tolerance"),
    (1.5, -3.0, False, "a slower first campaign: the sign flips"),
    (-3.0, -0.5, False, "a re-run inside the tolerance"),
    (-3.0, 0.5, False, "a re-run that is slower")])
def test_a_gain_needs_the_first_campaign_and_the_confirmation_to_agree_beyond_the_tolerance(first, rerun, confirmed, what):
    """The re-run is a confirmation: a cell is a confirmed gain only when the re-run reproduces what the first campaign found, the same sign beyond the tolerance in both."""
    t = tdb.synthetic_table(at(K_SMALL, metrics=slr.GAIN_METRICS, delta=rerun, first_delta=first, judged_on="rerun"))
    d = slr.decide_table(t)
    found = {(g["device"], g["metric"]) for g in d["gains"] if g["key"] == K_SMALL}
    assert (found == {(dev, m) for dev in slr.GATING for m in slr.GAIN_METRICS}) is confirmed and (confirmed or found == set()), what
    assert d == t["decision"] and d["outcome"] == "adopted"                                          # the other cells still give both devices their gains
    entry = row_of(t, "b200", "default", K_SMALL)["metrics"]["gpu_graph_us"]
    assert (entry["first_delta_pct"], entry["delta_pct"]) == (first, rerun)
    assert all(g["first_delta_pct"] <= -slr.TOL_PCT and g["delta_pct"] <= -slr.TOL_PCT for g in d["gains"])


def test_a_row_without_the_change_of_its_first_campaign_is_no_gain_and_is_no_valid_row():
    t = tdb.synthetic_table(at(K_SMALL, metrics=slr.GAIN_METRICS, delta=-3.0, judged_on="rerun"))
    results = slr.results_of_table(t)
    for dev in slr.GATING:
        for row in results[dev]["default"]["gpu_graph_us"]["rows"]:
            if row["key"] == K_SMALL:
                row["first_delta_pct"] = None
    assert K_SMALL not in {g["key"] for g in slr.decide(results, limits())["gains"] if g["metric"] == "gpu_graph_us"}
    gone = mutated(t, lambda x: row_of(x, "b200", "default", K_SMALL)["metrics"]["gpu_graph_us"].update(first_delta_pct=None))
    assert has(slr.table_problems(gone, PER_SM, constants=CONSTANTS), "the entry is malformed")


def test_a_cell_judged_on_its_first_campaign_has_one_change_in_both_places():
    t = tdb.synthetic_table()
    row = row_of(t, "b200", "default", K_SMALL)["metrics"]["total_us"]
    assert row["judged_on"] == "first" and row["first_delta_pct"] == row["delta_pct"] == -15.0
    edited = mutated(t, lambda x: row_of(x, "b200", "default", K_SMALL)["metrics"]["total_us"].update(first_delta_pct=-2.0))
    assert has(slr.table_problems(edited, PER_SM, constants=CONSTANTS), "a cell that was judged on the first campaign has the change -2.000000 % in its first campaign and -15.000000 % as its figure")


def table_with(**figures):
    """The table that the real comparator gives for a small campaign in which the cell K_SMALL has the figures ``figures`` in its graph metric (judged on the re-run)."""
    ev = make_ev(at(K_SMALL, metrics=("gpu_graph_us",), judged_on="rerun", base=default_values, **figures), cells=MINI)
    return slr.build_table(ev, slr.recompute(ev))


def graph_gains(t):
    return {(g["device"], g["metric"]) for g in t["decision"]["gains"] if g["key"] == K_SMALL and g["metric"] == "gpu_graph_us"}


def test_the_first_campaign_of_a_rerun_cell_comes_from_the_raw_records_and_gates_the_gain():
    """The tool derives the change of the first campaign from the raw first records (the comparison again without the re-run records); a hand edit of that figure is a difference."""
    t = table_with(delta=-3.0, first_delta=-0.2)
    entry = row_of(t, "b200", "default", K_SMALL)["metrics"]["gpu_graph_us"]
    assert (entry["judged_on"], entry["delta_pct"], entry["first_delta_pct"]) == ("rerun", -3.0, -0.2)
    assert graph_gains(t) == set()                                                                  # the first campaign found nothing: the re-run alone confirms nothing
    assert graph_gains(table_with(delta=-3.0, first_delta=-3.0)) == {("b200", "gpu_graph_us"), ("b300", "gpu_graph_us")}
    flipped = table_with(delta=-3.0, first_delta=1.5)
    assert row_of(flipped, "b200", "default", K_SMALL)["metrics"]["gpu_graph_us"]["first_delta_pct"] == 1.5 and graph_gains(flipped) == set()


def test_a_first_campaign_figure_that_was_edited_by_hand_is_a_difference_from_the_derived_table(full):
    edited = mutated(full["table"], lambda x: row_of(x, "b200", "default", K_SMALL)["metrics"]["gpu_graph_us"].update(first_delta_pct=-0.2))
    assert row_of(full["table"], "b200", "default", K_SMALL)["metrics"]["gpu_graph_us"]["first_delta_pct"] == -3.0
    assert has(slr.check_problems(edited, full["ev"], results=full["results"]), "first_delta_pct")


def test_a_candidate_that_is_3_percent_slower_fails_the_gate_and_one_that_is_3_percent_faster_passes_it_on_the_real_comparator():
    _, slow = mini_results(all_metrics(3.0))
    verdicts = {dev: {m: slow[dev]["default"][m]["verdict"] for m in slr.GATE_METRICS} for dev in slr.GATING}
    assert all(v == "FAIL" for per in verdicts.values() for v in per.values())
    d = slr.decide(slow, limits())
    assert d["outcome"] == "not_adopted" and d["gains"] == [] and has(d["reasons"], "b200 gpu_graph_us decode;N=1;H=12;regime=lockstep;: B is 3.00 % slower than A, the tolerance of the cell is 1.00 % (within the limit of 1184 CTAs)")
    assert has(d["reasons"], "b300 cpu_eager_us") and has(d["reasons"], "beyond the limit of 1184 CTAs")                                    # a cell that the limit does not route fails too
    _, fast = mini_results(all_metrics(-3.0))
    assert {fast[dev]["default"][m]["verdict"] for dev in slr.GATING for m in slr.GATE_METRICS} == {"PASS"}
    d = slr.decide(fast, limits())
    assert d["outcome"] == "adopted" and {g["key"] for g in d["gains"]} == {K_SMALL, key_of(4, 12, "staggered"), key_of(64, 12)}                # the cell beyond the limit is no evidence
    _, within = mini_results(all_metrics(0.9))                                                                                              # slower, but within the tolerance: passes, no gain
    assert {within[dev]["default"][m]["verdict"] for dev in slr.GATING for m in slr.GATE_METRICS} == {"PASS"} and slr.decide(within, limits())["outcome"] == "not_adopted"


def test_a_noise_limited_gain_is_not_a_gain():
    """The cell is 3 % faster, but its spread stays above 0.3 % after the re-run: it passes with a widened tolerance and is never the evidence of a gain."""
    d, t = decided(lambda dev, arm, metric, key, f: dict(CLEAN(dev, arm, metric, key, f), **({"spread": 2.0, "judged_on": "rerun"} if metric in slr.GAIN_METRICS and f["regime"] in bd.LOOP_REGIMES else {})))
    row = row_of(t, "b200", "default", K_SMALL)["metrics"]["gpu_graph_us"]
    assert row["noise_limited"] and row["judged_on"] == "rerun" and row["delta_pct"] == -3.0 and row["status"] == "PASS"
    assert d["outcome"] == "not_adopted" and d["gains"] == [] and has(d["reasons"], "b200: no confirmed gain") and has(d["reasons"], "b300: no confirmed gain")
    one = at(K_SMALL, metrics=("gpu_graph_us",), spread=2.0, base=CLEAN)
    other, _ = decided(one)
    assert other["outcome"] == "adopted" and K_SMALL not in {g["key"] for g in other["gains"] if g["metric"] == "gpu_graph_us"} and K_SMALL in {g["key"] for g in other["gains"] if g["metric"] == "gpu_eager_us"}


def test_a_gain_that_is_not_confirmed_on_the_rerun_or_is_not_a_gpu_elapsed_cell_within_the_limit_is_not_a_gain():
    first = lambda dev, arm, metric, key, f: dict(CLEAN(dev, arm, metric, key, f), judged_on="first")                                      # noqa: E731
    assert decided(first)[0]["outcome"] == "not_adopted"                                                                                    # not judged on the re-run
    only_other = lambda dev, arm, metric, key, f: dict(CLEAN(dev, arm, metric, key, f), **({"delta": 0.0} if metric in slr.GAIN_METRICS else {"delta": -9.0}))     # noqa: E731
    d = decided(only_other)[0]
    assert d["outcome"] == "not_adopted" and d["gains"] == []                                                                               # the kernel sum and the host times fall, GPU time does not
    cpu = lambda dev, arm, metric, key, f: dict(CLEAN(dev, arm, metric, key, f), **({"delta": 0.0} if metric not in slr.LOOP_METRICS else {"delta": -9.0, "judged_on": "rerun"}))   # noqa: E731
    assert decided(cpu)[0]["outcome"] == "not_adopted"
    beyond = lambda dev, arm, metric, key, f: dict(CLEAN(dev, arm, metric, key, f), **({"delta": -3.0, "judged_on": "rerun"} if f["N"] * f["H"] > LIMIT and arm == "default" else {"delta": 0.0}))     # noqa: E731
    assert decided(beyond)[0]["outcome"] == "not_adopted" and decided(beyond)[0]["gains"] == []                                            # only cells beyond the limit are faster
    cold = lambda dev, arm, metric, key, f: dict(CLEAN(dev, arm, metric, key, f), **({"delta": -3.0, "judged_on": "rerun"} if f["regime"] == "cold" and arm == "default" and metric == "gpu_graph_us" else {"delta": 0.0}))   # noqa: E731
    d = decided(cold)[0]
    assert d["outcome"] == "adopted" and {slr.cell_fields(g["key"])["regime"] for g in d["gains"]} == {"cold"}                              # a cold cell is evidence like any other stable cell


def test_a_gain_on_one_device_only_is_not_adoption():
    only_b200 = lambda dev, arm, metric, key, f: dict(CLEAN(dev, arm, metric, key, f), **({"delta": 0.0} if dev == "b300" and metric in slr.GAIN_METRICS else {}))     # noqa: E731
    d, t = decided(only_b200)
    assert d["outcome"] == "not_adopted" and d["per_device"]["b200"]["gains"] > 0 and d["per_device"]["b300"]["gains"] == 0
    assert d["reasons"] == ["b300: no confirmed gain: no stable GPU-elapsed cell within the limit of 1184 CTAs, judged on the re-run, is faster than A by 1.00 % or more"]
    assert slr.timing_of(d) == ("pass", "fail")                                                                                             # nothing regresses, nothing is gained on b300


@pytest.mark.parametrize("delta,counts", [(-1.0, True), (-0.9999, False), (-1.0001, True)])
def test_a_gain_needs_the_tolerance_exactly(delta, counts):
    d, t = decided(at([K_SMALL], metrics=("gpu_graph_us", "gpu_eager_us"), delta=delta, judged_on="rerun", base=lambda *a: {"base": 100.0, "delta": 0.0, "spread": 0.1, "judged_on": "first"}))
    assert (d["outcome"] == "adopted") is counts and (K_SMALL in {g["key"] for g in d["gains"]}) is counts


def test_a_regression_of_any_gating_metric_on_either_device_is_not_adopted():
    for metric in slr.GATE_METRICS:
        for dev in slr.GATING:
            d, t = decided(at([K_SMALL], devices=(dev,), arms=("default",), metrics=(metric,), delta=2.0, judged_on="rerun", base=default_values))
            assert d["outcome"] == "not_adopted" and d["per_device"][dev]["verdicts"][metric] == "FAIL" and has(d["reasons"], "%s %s %s: B is 2.00 %% slower" % (dev, metric, K_SMALL)), (dev, metric)
            assert slr.timing_of(d) == ("fail", "fail")
    d, t = decided(at([K_SMALL], arms=("default",), metrics=("gpu_graph_us",), delta=4.0, spread=1.0, judged_on="rerun", base=default_values))      # a noise-limited cell beyond its widened tolerance
    assert row_of(t, "b200", "default", K_SMALL)["metrics"]["gpu_graph_us"]["noise_limited"] and d["outcome"] == "not_adopted"


def test_the_forced_arm_does_not_decide_and_a_fail_there_is_what_the_calibration_is_for():
    d, t = decided()
    assert t["comparisons"]["b200"]["forced"]["gpu_graph_us"]["verdict"] == "FAIL" and d["outcome"] == "adopted"
    assert slr.calibrate(t) == PER_SM


@pytest.mark.parametrize("name,edit,expected", [
    ("an INVALID comparison", lambda r: r["b200"]["default"]["total_us"].update(verdict="INVALID", problems=["cell x needs a re-run"]), "b200 total_us: the comparison is INVALID (cell x needs a re-run)"),
    ("a comparison that is missing", lambda r: r["b300"]["default"].pop("cpu_eager_us"), "b300 cpu_eager_us: there is no comparison"),
    ("a device that is missing", lambda r: r.pop("b300"), "b300 gpu_graph_us: there is no comparison"),
    ("the arm that decides is missing", lambda r: r["b200"].pop("default"), "b200 gpu_graph_us: there is no comparison"),
    ("a comparison in the alternate order", lambda r: r["b200"]["default"]["gpu_eager_us"].update(order="alternate"), "b200 gpu_eager_us: it was made with order 'alternate' instead of 'abba'"),
    ("a comparison without an order", lambda r: r["b200"]["default"]["gpu_eager_us"].pop("order"), "order None instead of 'abba'"),
    ("a comparison at another tolerance", lambda r: r["b300"]["default"]["gpu_graph_us"].update(tol_pct=5.0), "tol_pct 5.0 instead of 1.0"),
    ("a comparison in the within mode", lambda r: r["b300"]["default"]["gpu_graph_us"].update(mode="within"), "mode 'within' instead of 'regress'"),
    ("a comparison without widening", lambda r: r["b300"]["default"]["gpu_graph_us"].update(unstable_policy="worst"), "unstable_policy 'worst'"),
    ("a comparison that did not require the launch mappings", lambda r: r["b300"]["default"]["gpu_graph_us"].update(require_nodes=False), "require_nodes False instead of True"),
    ("a verdict that is none of the three", lambda r: r["b200"]["default"]["total_us"].update(verdict="PASSED"), "the comparison is PASSED"),
])
def test_an_invalid_comparison_or_one_made_under_other_parameters_leaves_the_outcome_undecided(name, edit, expected):
    results = tdb.campaign_results(default_values)
    assert slr.decide(results, limits())["outcome"] == "adopted"
    d = slr.decide(mutated(results, edit), limits())
    assert d["outcome"] == "undecided" and has(d["reasons"], expected), (name, d["reasons"][:2])
    assert d["gains"] == [] or d["outcome"] == "undecided"


def test_invalid_is_undecided_and_never_not_adopted_even_next_to_a_failing_comparison():
    results = tdb.campaign_results(default_values)
    both = mutated(results, lambda r: (r["b200"]["default"]["total_us"].update(verdict="INVALID"), r["b300"]["default"]["gpu_graph_us"].update(
        verdict="FAIL", rows=[stub_row(K_SMALL, delta=3.0, judged_on="rerun")])))
    d = slr.decide(both, limits())
    assert d["outcome"] == "undecided" and has(d["reasons"], "b200 total_us") and not has(d["reasons"], "slower")
    with pytest.raises(ValueError, match="undecided"):
        slr.timing_of(d)


def test_the_limit_of_a_device_is_an_integer_or_the_outcome_is_undecided():
    results = tdb.campaign_results(default_values)
    for bad in (None, 1184.5, True, "1184", -1):
        d = slr.decide(results, {"b200": bad, "b300": LIMIT})
        assert d["outcome"] == "undecided" and has(d["reasons"], "b200: the limit is"), bad
    assert slr.decide(results, {"b200": LIMIT})["outcome"] == "undecided"


def test_the_decision_does_not_depend_on_the_order_of_the_rows_and_the_table_gives_the_same_decision():
    results = tdb.campaign_results(default_values)
    shuffled = copy.deepcopy(results)
    for dev in shuffled.values():
        for arm in dev.values():
            for res in arm.values():
                res["rows"].reverse()
    assert slr.decide(shuffled, limits()) == slr.decide(results, limits())
    t = slr.table_of(results, tdb.facts_for())
    assert slr.decide_table(t) == t["decision"] == slr.decide(results, limits())


def test_a_fail_row_inside_a_passing_comparison_still_counts_and_a_failing_comparison_without_a_row_is_named():
    results = tdb.campaign_results(default_values)
    odd = mutated(results, lambda r: r["b200"]["default"]["gpu_graph_us"].update(verdict="FAIL"))                                       # FAIL with no failing row
    d = slr.decide(odd, limits())
    assert d["outcome"] == "not_adopted" and has(d["reasons"], "b200 gpu_graph_us: the comparison FAILs")


# ---- the ledger row, accepted by the record checker ---------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def steps():
    import test_record_checker_steps as trs
    return trs


def row_table(values=default_values, **kw):
    """A table of the decision, measured at the commit of the evidence that the record checker's own fixtures build for the single-launch row."""
    import test_record_checker_steps as trs
    return tdb.slr.table_of(tdb.campaign_results(values), tdb.facts_for(candidate=trs.C1)) if not kw else tdb.synthetic_table(values, **kw)


def real_row(steps, table, revert=None):
    codegen, resources = steps.gate_evidence(steps.C1, sid="single_launch")
    return slr.ledger_row(table, table["decision"], steps.C1, ["single_launch_marker"], codegen, resources, revert)


def step_check(steps, row):
    return rc.step_problems(row, None, steps.META)


def make_repo(tmp_path):
    """A repository whose commits are: ``first`` (every kind of file), ``tools`` (tests, benchmarks and documents change, the code does not), ``code`` (the extension's source changes) and
    ``wrapper`` (the Python wrapper changes); ``{name: full hash}``."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(["git", "-c", "commit.gpgsign=false", "-C", str(repo)] + list(args), check=True, stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
    git("init", "-q")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    commits = {}
    for name, edits in (("first", {"csrc/k.cu": "1", "flash_kda/api.py": "1", "setup.py": "1", "tests/t.py": "1", "benchmarks/b.py": "1", "docs/d.md": "1"}),
                        ("tools", {"tests/t.py": "2", "benchmarks/b.py": "2", "docs/d.md": "2"}), ("code", {"csrc/k.cu": "2"}), ("wrapper", {"flash_kda/api.py": "2"})):
        for path, text in edits.items():
            (repo / path).parent.mkdir(parents=True, exist_ok=True)
            (repo / path).write_text(text)
        git("add", "-A")
        git("commit", "-q", "-m", name)
        commits[name] = git("rev-parse", "HEAD")
    return str(repo), commits


def test_the_code_of_a_row_is_the_code_that_was_measured_when_the_paths_of_the_code_are_the_same_in_both_commits(tmp_path):
    repo, c = make_repo(tmp_path)
    assert slr.code_problems(None, c["first"], c["first"]) == [] and slr.code_problems(None, c["first"][:10], c["first"]) == []        # the same commit is the same code
    assert slr.code_problems(repo, c["first"], c["tools"]) == [] and slr.code_problems(repo, c["tools"], c["first"]) == []             # tests, benchmarks and documents are not the code
    assert has(slr.code_problems(None, c["first"], c["tools"]), "without a repository")
    for other, path in (("code", "csrc/k.cu"), ("wrapper", "flash_kda/api.py")):
        problems = slr.code_problems(repo, c["first"], c[other])
        assert has(problems, "differ in the code that is measured") and has(problems, path), other
    assert has(slr.code_problems(repo, c["first"], "f" * 40), "git cannot compare")
    assert has(slr.code_problems(repo, None, c["first"]), "not a commit hash") and has(slr.code_problems(repo, "main", c["first"]), "not a commit hash")


def test_the_row_may_name_another_commit_than_the_measured_one_only_when_the_code_is_the_same(steps, tmp_path):
    repo, c = make_repo(tmp_path)
    table = tdb.slr.table_of(tdb.campaign_results(default_values), tdb.facts_for(candidate=c["tools"]))             # measured at the commit that only added tools
    codegen, resources = steps.gate_evidence(steps.C1, sid="single_launch")
    row = slr.ledger_row(table, table["decision"], c["first"], ["m"], codegen, resources, None, repo)                # the row is of the commit whose code the dumps are of
    assert row["commit"] == c["first"] and slr.row_problems(row, table, repo) == []
    with pytest.raises(ValueError, match="without a repository"):
        slr.ledger_row(table, table["decision"], c["first"], ["m"], codegen, resources)
    with pytest.raises(ValueError, match="differ in the code that is measured"):
        slr.ledger_row(table, table["decision"], c["code"], ["m"], codegen, resources, None, repo)
    assert has(slr.row_problems(row, table), "without a repository to compare them in")
    assert has(slr.row_problems(dict(row, commit=c["code"]), table, repo), "differ in the code that is measured")
    assert has(slr.row_problems(dict(row, commit=None), table, repo), "not a commit hash")


def test_the_row_of_an_adopted_step_is_accepted_by_the_record_checker(steps):
    table = row_table()
    row = real_row(steps, table)
    assert table["decision"]["outcome"] == "adopted" and row["outcome"] == "adopted" and row["commit"] == steps.C1 and row["revert_commit"] is None and row["failed_at"] is None
    assert row["timing"] == {"vs_previous": "pass", "vs_base": "pass", "chain": "not_applicable"} and row["gates"] == {"exactness": True, "codegen": True} and row["reason"] == ""
    assert step_check(steps, row) == []
    assert slr.row_problems(row, table) == []
    assert rc.ledger_problems({"schema": rc.LEDGER_SCHEMA, "meta": dict(steps.META), "steps": [row], "_path": "mem"}) == []


def test_the_ab_figures_of_the_row_are_the_medians_of_the_table_keyed_by_metric_and_cell(steps):
    table = row_table(at(key_of(16, 12), metrics=("gpu_graph_us",), spread=2.0, judged_on="rerun", base=default_values))
    row = real_row(steps, table)
    assert sorted(row["ab"]) == list(rc.DEVICES) == list(slr.GATING)
    for dev in rc.DEVICES:
        medians = row["ab"][dev]["medians"]
        default_rows = [r for r in table["rows"] if r["device"] == dev and r["arm"] == "default"]
        assert len(medians) == sum(len(r["metrics"]) for r in default_rows) == 92 * 3 + 48 * 2                                              # three metrics everywhere, two more in the 48 cells of the loop regimes
        assert medians["gpu_graph_us|" + K_SMALL] == {"base": 100.0, "cand": 97.0} and medians["total_us|" + K_SMALL] == {"base": 100.0, "cand": 85.0}
        assert medians["cpu_eager_us|" + K_SMALL] == {"base": 100.0, "cand": 100.0} and "cpu_eager_us|" + K_COLD not in medians              # a cold cell has no host metric
        assert "gpu_graph_us|" + K_BIG in medians and "gpu_eager_us|" + K_ABOVE in medians                                                  # every cell of the default arm, beyond the limit too
        assert all(isinstance(v["base"], float) and v["base"] > 0 and v["cand"] > 0 for v in medians.values())
        assert row["ab"][dev]["noise_limited"] == ["gpu_graph_us|" + key_of(16, 12)]
    assert step_check(steps, row) == []


def test_the_row_follows_the_table_and_an_edited_figure_is_rejected(steps):
    table = row_table()
    row = real_row(steps, table)
    for name, edit, expected in (
            ("a median edited", lambda r: r["ab"]["b200"]["medians"]["gpu_graph_us|" + K_SMALL].update(cand=60.0), "ab.b200.medians.gpu_graph_us|%s.cand is 60.0, the evidence gives 97.0" % K_SMALL),
            ("a base median edited", lambda r: r["ab"]["b300"]["medians"]["total_us|" + K_BIG].update(base=99.0), "ab.b300.medians.total_us|%s.base is 99.0" % K_BIG),
            ("a cell dropped", lambda r: r["ab"]["b300"]["medians"].pop("gpu_eager_us|" + K_ABOVE), "gpu_eager_us|%s is missing" % K_ABOVE),
            ("a cell invented", lambda r: r["ab"]["b200"]["medians"].update({"gpu_eager_us|" + key_of(3, 7): {"base": 1.0, "cand": 1.0}}), "is not in the derived table"),
            ("a noise-limited cell hidden", lambda r: r["ab"]["b200"]["noise_limited"].append("gpu_graph_us|" + K_SMALL), "ab.b200.noise_limited has 1 entries, the derived table 0"),
            ("a device dropped", lambda r: r["ab"].pop("b300"), "ab.b300 is missing"),
            ("the timing edited", lambda r: r["timing"].update(vs_previous="fail"), "the timing of the row is"),
            ("the chain applied", lambda r: r["timing"].update(chain="pass"), "the timing of the row is"),
            ("another outcome", lambda r: r.update(outcome="not_adopted"), "the table's outcome is adopted, the row says 'not_adopted'"),
            ("another commit", lambda r: r.update(commit="f" * 40), "the row is of commit"),
            ("another failed_at", lambda r: r.update(failed_at="timing"), "failed_at is 'timing'"),
            ("another step", lambda r: r.update(id="trims"), "the row is of 'trims'")):
        problems = slr.row_problems(mutated(row, edit), table)
        assert has(problems, expected), (name, problems[:3])
    changed = tdb.slr.table_of(tdb.campaign_results(at(K_SMALL, metrics=("gpu_graph_us",), delta=-5.0, judged_on="rerun", base=default_values)), tdb.facts_for(candidate=steps.C1))
    assert has(slr.row_problems(row, changed), "gpu_graph_us|%s.cand is 97.0, the evidence gives 95.0" % K_SMALL)                       # the figures of the row follow the table


def test_the_row_of_a_step_that_regressed_or_gained_nothing_is_accepted_when_it_names_its_revert(steps):
    worse = row_table(at([K_SMALL], arms=("default",), metrics=("gpu_graph_us",), delta=2.0, judged_on="rerun", base=default_values))
    assert worse["decision"]["outcome"] == "not_adopted"
    row = real_row(steps, worse, revert=steps.C4)
    assert row["outcome"] == "not_adopted" and row["failed_at"] == "timing" and row["revert_commit"] == steps.C4 and row["timing"] == {"vs_previous": "fail", "vs_base": "fail", "chain": "not_applicable"}
    assert has([row["reason"]], "b200 gpu_graph_us") and has([row["reason"]], "slower") and step_check(steps, row) == [] and slr.row_problems(row, worse) == []
    nogain = row_table(lambda dev, arm, metric, key, f: dict(CLEAN(dev, arm, metric, key, f), **({"delta": 0.0} if metric in slr.GAIN_METRICS else {})))
    assert nogain["decision"]["outcome"] == "not_adopted" and nogain["decision"]["gains"] == []
    row = real_row(steps, nogain, revert=steps.C4)
    assert row["timing"] == {"vs_previous": "fail", "vs_base": "pass", "chain": "not_applicable"}                                          # no regression, no gain: one result must fail
    assert has([row["reason"]], "no confirmed gain") and step_check(steps, row) == [] and slr.row_problems(row, nogain) == []
    assert has(step_check(steps, mutated(row, lambda r: r["timing"].update(vs_previous="pass"))), "failed at timing, but no timing result is a failure")      # the checker needs the fail


def test_a_table_that_decides_nothing_has_no_row_and_the_arguments_of_a_row_must_fit(steps):
    undecided = row_table(at(K_SMALL, spread=2.0, judged_on="first", metrics=("gpu_graph_us",), arms=("default",), base=default_values))
    assert undecided["decision"]["outcome"] == "undecided"
    codegen, resources = steps.gate_evidence(steps.C1, sid="single_launch")
    with pytest.raises(ValueError, match="there is no row to write until the evidence is repaired"):
        slr.ledger_row(undecided, undecided["decision"], steps.C1, ["m"], codegen, resources)
    assert has(slr.row_problems({"id": "single_launch"}, undecided), "the table decides nothing")
    table = row_table()
    good = (table, table["decision"], steps.C1, ["single_launch_marker"], codegen, resources)
    for what, args, error in (("another commit", {2: "e" * 40}, "the table was measured at"), ("no markers", {3: []}, "markers"), ("a marker that is no string", {3: [1]}, "markers"),
                              ("no code-generation entry", {4: None}, "supplied by the caller"), ("no resource entry", {5: None}, "supplied by the caller"),
                              ("a revert for an adopted step", {6: steps.C4}, "no revert commit"), ("a decision that is not the table's", {1: dict(table["decision"], outcome="not_adopted")}, "not the one that the table gives")):
        a = list(good) + [None]
        for i, v in args.items():
            a[i] = v
        with pytest.raises(ValueError, match=error):
            slr.ledger_row(*a)
    worse = row_table(at([K_SMALL], arms=("default",), metrics=("gpu_graph_us",), delta=2.0, judged_on="rerun", base=default_values))
    with pytest.raises(ValueError, match="one that was not adopted names it"):
        slr.ledger_row(worse, worse["decision"], steps.C1, ["m"], codegen, resources)


def test_the_timing_results_of_every_outcome():
    adopted, _ = decided()
    assert slr.timing_of(adopted) == ("pass", "pass")
    regress, _ = decided(at([K_SMALL], arms=("default",), metrics=("gpu_eager_us",), delta=2.0, judged_on="rerun", base=default_values))
    assert slr.timing_of(regress) == ("fail", "fail")
    nogain, _ = decided(lambda dev, arm, metric, key, f: dict(CLEAN(dev, arm, metric, key, f), **({"delta": 0.0} if metric in slr.GAIN_METRICS else {})))
    assert slr.timing_of(nogain) == ("pass", "fail")


# ---- the section of the performance document ---------------------------------------------------------------------------------------------------------
DOC = "# Performance\n\n## 7. Something\ntext\n\n## 8. Another\nmore\n\n"
AFTER = "\n## 10. Later\nmore\n"


def document_of(table, number=9):
    return DOC + slr.render_section(table, table["decision"], number) + AFTER


def test_the_section_is_generated_from_the_table_and_regenerates_without_a_diff():
    table = tdb.synthetic_table()
    section = slr.render_section(table, table["decision"], 9)
    assert section == slr.render_section(copy.deepcopy(table), copy.deepcopy(table["decision"]), 9) and section.startswith("## 9. Single-launch decode: calibration and decision\n")
    doc = document_of(table)
    assert slr.document_problems(table, doc) == []
    assert slr.replace_section(doc, section) == doc and slr.replace_section(slr.replace_section(doc, section), section) == doc
    assert slr.section_bounds(doc) is not None and doc.index("## 8.") < doc.index("## 9.") < doc.index("## 10.")
    appended = slr.replace_section("# Performance\n\ntext\n", section)
    assert appended.count(slr.SECTION_TITLE) == 1 and slr.document_problems(table, appended) == []
    renumbered = slr.render_section(table, table["decision"], 12)
    assert renumbered.startswith("## 12. ") and slr.document_problems(table, DOC + renumbered) == []                                      # the number that the document gives its section is kept
    assert slr.replace_section(doc, slr.render_section(table, table["decision"], 9).replace("adopted**", "adopted**")) == doc


def test_the_section_says_what_the_table_says():
    table = tdb.synthetic_table(extra_devices={"h100": tdb.H100})
    table["meta"]["not_run"] = {}
    text = slr.render_section(table, table["decision"], 9)
    assert "Single-launch decode is **adopted**." in text and "Limit of the tree: 8 CTAs per SM on 148 SMs, N * H at most 1184." in text and "Candidate commit 0123456789ab." in text
    assert "- no gating metric regresses on b200, b300" in text and "- b200: 52 confirmed gain cell(s) (GPU-elapsed graph or eager, stable, judged on the re-run, within the limit of 1184 CTAs)" in text
    assert "| b200 | default | gpu_graph_us | PASS | 92 | 0 | 26 | 0 |" in text and "| b200 | forced | gpu_graph_us | FAIL | 33 | 0 | 30 | 0 |" in text and "| b200 | default | cpu_eager_us | PASS | 48 | 0 | 0 | 0 |" in text
    assert "The largest tested N * H below the smallest unsafe one (the largest of all when none is unsafe) is 1188; divided by the 148 SMs and rounded down that is 8 CTAs per SM. The limit of the tree is 8 CTAs per SM, the calibrated value." in text
    assert "| b200 | gpu_graph_us | 1 | 12 | 12 | lockstep | 100.000 | 97.000 | 3.000 | -3.00 |" in text
    assert "| 1 | 12 | 12 | lockstep | - | single-launch | 100.000 | 97.000 | 100.000 | 97.000 | -3.00 r | -3.00 r | -15.00 | +0.00 | +0.00 |" in text
    assert "| 128 | 12 | 1536 | lockstep | - | single-launch | 100.000 | 102.500 | 100.000 | 102.500 | +2.50 F r | +2.50 F r | +0.00 | +0.00 | +0.00 |" in text
    assert "| 64 | 12 | 768 | probe | 31 | single-launch |" in text and "### h100, the control on, the cells that the default rule does not route" in text
    assert slr.DRAFT_ESTIMATE in text and "a direction, not a gate" in text
    unsafe = [u for u in table["calibration"]["unsafe"] if u["device"] == "b200" and u["key"] == K_BIG]
    assert unsafe and "| b200 | forced | 128 | 12 | 1536 | lockstep | gpu_eager_us, gpu_graph_us |" in text
    not_run = tdb.synthetic_table()
    not_run["meta"]["not_run"] = {"h100": "no allocation was granted"}
    assert "- Not run: h100: no allocation was granted." in slr.render_section(not_run, not_run["decision"], 9)
    assert not slr.plan_markers(text) and "\n\n\n" not in text and text.endswith("|\n")


def test_the_section_of_every_outcome():
    for values, outcome in ((default_values, "adopted"), (at([K_SMALL], arms=("default",), metrics=("gpu_graph_us",), delta=2.0, judged_on="rerun", base=default_values), "not adopted"),
                            (at(K_SMALL, spread=2.0, judged_on="first", metrics=("gpu_graph_us",), arms=("default",), base=default_values), "undecided")):
        table = tdb.synthetic_table(values)
        text = slr.render_section(table, table["decision"], 9)
        assert "Single-launch decode is **%s**." % outcome in text, outcome
        assert slr.document_problems(table, DOC + text + AFTER) == []
    regress = tdb.synthetic_table(at([K_SMALL], arms=("default",), metrics=("gpu_graph_us",), delta=2.0, judged_on="rerun", base=default_values))
    assert "- b200 gpu_graph_us %s: B is 2.00 %% slower than A" % K_SMALL in slr.render_section(regress, regress["decision"], 9)


@pytest.mark.parametrize("name,edit,expected", [
    ("a median edited in a table", lambda t: t.replace("| 1 | 12 | 12 | lockstep | - | single-launch | 100.000 | 97.000 |", "| 1 | 12 | 12 | lockstep | - | single-launch | 100.000 | 96.000 |", 1), "not the generated one"),
    ("a change edited", lambda t: t.replace("-3.00 r", "-4.00 r", 1), "not the generated one"),
    ("the limit edited", lambda t: t.replace("8 CTAs per SM on 148 SMs", "16 CTAs per SM on 148 SMs", 1), "not the generated one"),
    ("the outcome edited", lambda t: t.replace("is **adopted**", "is **not adopted**", 1), "not the generated one"),
    ("a sentence added", lambda t: t.replace("### Outcome", "### Outcome\n\nThis was faster by 3 percent.", 1), "not the generated one"),
    ("a verdict edited", lambda t: t.replace("| b200 | forced | gpu_graph_us | FAIL |", "| b200 | forced | gpu_graph_us | PASS |", 1), "not the generated one"),
    ("a row removed", lambda t: t.replace("| 128 | 12 | 1536 | lockstep | - | single-launch |", "| 128 | 12 | 1536 | lockstep | - | two-launch |", 1), "not the generated one"),
    ("the section dropped", lambda t: t[:t.index("## 9.")] + t[t.index("## 10."):], "no section headed"),
])
def test_a_document_that_is_not_the_generated_section_is_rejected(name, edit, expected):
    table = tdb.synthetic_table()
    doc = document_of(table)
    assert slr.document_problems(table, doc) == []
    problems = slr.document_problems(table, edit(doc))
    assert has(problems, expected), (name, problems)


# ---- the evidence: launches against the specification, identity of the records ------------------------------------------------------------------
def test_the_evidence_of_the_campaign_is_accepted_and_its_comparisons_are_those_of_the_gate(full):
    ev, results = full["ev"], full["results"]
    assert slr.evidence_problems(ev) == []
    for dev in slr.GATING:
        for arm in slr.ARMS:
            for metric in slr.GATE_METRICS:
                res = results[dev][arm][metric]
                assert res["order"] == "abba" and res["require_nodes"] is True and res["mode"] == "regress" and res["unstable_policy"] == "widen" and res["tol_pct"] == 1.0, (dev, arm, metric)
                assert res["verdict"] == ("FAIL" if arm == "forced" and metric in slr.GAIN_METRICS else "PASS"), (dev, arm, metric, res["problems"][:2])
    assert {len(results["b200"]["default"][m]["rows"]) for m in ("gpu_graph_us", "gpu_eager_us", "total_us")} == {92} and len(results["b200"]["default"]["cpu_graph_us"]["rows"]) == 48
    noisy = [r for r in results["b200"]["default"]["gpu_graph_us"]["rows"] if r["key"] == key_of(16, 12)][0]
    assert noisy["noise_limited"] and noisy["judged_on"] == "rerun" and abs(noisy["tol_pct_effective"] - 3.0) < 1e-9 and noisy["status"] == "PASS"


def test_the_table_that_the_evidence_gives_satisfies_every_rule_and_has_the_hand_computed_figures(full):
    t = full["table"]
    assert slr.final_problems(t, CONSTANTS) == [] and slr.table_problems(t, PER_SM, constants=CONSTANTS) == []
    assert t["decision"]["outcome"] == "adopted" and slr.calibrate(t) == 8 and t["calibration"]["first_unsafe_nh"] == 1536 and t["calibration"]["limit_nh"] == 1188
    assert t["decision"]["per_device"]["b200"]["gains"] == 51 and t["decision"]["per_device"]["b300"]["gains"] == 51                         # 26 cells, two metrics, minus the noisy graph metric
    assert not [g for g in t["decision"]["gains"] if g["key"] == key_of(16, 12) and g["metric"] == "gpu_graph_us"]
    g = row_of(t, "b200", "default", K_SMALL)["metrics"]["gpu_graph_us"]
    assert (g["base_median"], g["cand_median"], g["delta_pct"], g["judged_on"], g["noise_limited"]) == (100.0, 97.0, -3.0, "rerun", False) and abs(g["base_spread_pct"] - 0.1) < 1e-9
    k = row_of(t, "b300", "forced", K_BIG)["metrics"]["gpu_eager_us"]
    assert (k["base_median"], k["cand_median"], k["status"], k["judged_on"]) == (100.0, 102.5, "FAIL", "rerun")
    assert row_of(t, "b200", "default", K_ABOVE)["route"] == "two-launch" and row_of(t, "b200", "default", K_ABOVE)["metrics"]["total_us"]["cand_median"] == 100.0
    assert t["meta"]["devices"]["b300"] == {"gpu": "NVIDIA B300 SXM6 AC", "sm_count": 148, "cc": "10.3", "gating": True} and t["meta"]["graph_nodes"] is False
    assert sorted(t["inputs"]["b200"]["default"]["rerun"]["A"][0]) == ["name", "sha256"] and len(t["inputs"]["b200"]["forced"]["first"]["B"]) == 4


def test_an_a_cell_that_launches_the_row_engine_alone_is_rejected_by_the_launch_check(full):
    """Route vacuity: side A is the explicit two-launch path. A record whose A cell shows ``rows_kernel`` only (the default single launch ran as A) is a problem, however self-consistent it is."""
    def single_launch_as_a(ev):
        rec = ev["devices"]["b200"]["default"]["first"]["A"][1]
        cell = [c for c in rec["cells"] if c["key"] == K_SMALL][0]
        cell["args"].update(compute_nodes={"rows_kernel": 1}, compute_kernels=1, expect_compute={"rows_kernel": 1}, kernels=["rows_kernel"])
        for s in cell["samples"]:
            s["nodes"]["compute"] = {"rows_kernel": 1}
    ev = mutated(full["ev"], single_launch_as_a)
    problems = slr.evidence_problems(ev)
    assert has(problems, "b200 default first A record") and has(problems, "the recorded compute nodes {'rows_kernel': 1} are not the expected {'plan_kernel': 1, 'rows_kernel': 1}")
    assert has(slr.check_problems(full["table"], ev, results=full["results"]), "not the expected")


def test_a_graph_replay_that_a_record_does_not_describe_is_a_problem_and_never_an_exception(full):
    """With the replay profiled (meta graph_nodes), every trial of every cell has to show the launches of the specification: a sample that is no object, a replay that is no object and one that
    shows another route are problems, named, and no mistake of the record makes the check raise."""
    def damage(ev):
        rec = ev["devices"]["b200"]["default"]["first"]["B"][0]
        rec["meta"]["graph_nodes"] = True
        for cell in rec["cells"]:                                                                          # the replay of every trial shows the launches of the trial's eager pass ...
            for sample in cell["samples"]:
                sample["graph_nodes"] = {"compute": dict(sample["nodes"]["compute"]), "aux": {}}
        cells = {c["key"]: c for c in rec["cells"]}                                                        # ... except in these three
        cells[K_SMALL]["samples"][0] = 5
        cells[K_SMALL]["samples"][1]["graph_nodes"] = ["rows_kernel"]
        cells[K_ABOVE]["samples"][0]["graph_nodes"] = {"compute": {"rows_kernel": 1}}                    # the specification says two launches beyond the limit
    problems = slr.launch_problems(mutated(full["ev"], damage))
    assert has(problems, "b200 default first B record: cell %s, trial 0: the graph replay launched None" % K_SMALL)
    assert has(problems, "cell %s, trial 1: the graph replay launched None" % K_SMALL)
    assert has(problems, "cell %s, trial 0: the graph replay launched {'rows_kernel': 1}, the specification says {'plan_kernel': 1, 'rows_kernel': 1}" % K_ABOVE)
    assert slr.launch_problems(mutated(full["ev"], lambda ev: ev["devices"]["b200"]["default"]["first"]["B"][0]["meta"].update(graph_nodes=False))) == []     # without the option nothing is asked


@pytest.mark.parametrize("name,dev,arm,key,mapping,expected", [
    ("a cell that the limit routes to one launch ran two", "b200", "default", K_SMALL, {"plan_kernel": 1, "rows_kernel": 1}, "the recorded compute nodes {'plan_kernel': 1, 'rows_kernel': 1} are not the expected {'rows_kernel': 1}"),
    ("a cell beyond the limit ran one launch under the default routing", "b300", "default", K_ABOVE, {"rows_kernel": 1}, "the recorded compute nodes {'rows_kernel': 1} are not the expected {'plan_kernel': 1, 'rows_kernel': 1}"),
    ("a forced cell ran two launches", "b200", "forced", K_BIG, {"plan_kernel": 1, "rows_kernel": 1}, "are not the expected {'rows_kernel': 1}"),
    ("a launch on the reference row engine", "b200", "default", K_SMALL, {"plan_kernel": 1, "ref::rows_kernel": 1}, "are not the expected"),
    ("a launch that is twice as many", "b300", "default", K_ABOVE, {"plan_kernel": 2, "rows_kernel": 2}, "are not the expected"),
])
def test_a_flipped_route_of_side_b_is_rejected(full, name, dev, arm, key, mapping, expected):
    def flip(ev):
        rec = ev["devices"][dev][arm]["first"]["B"][2]
        cell = [c for c in rec["cells"] if c["key"] == key][0]
        cell["args"].update(compute_nodes=dict(mapping), expect_compute=dict(mapping))
        for s in cell["samples"]:
            s["nodes"]["compute"] = dict(mapping)
    problems = slr.evidence_problems(mutated(full["ev"], flip))
    assert has(problems, "%s %s first B record" % (dev, arm)) and has(problems, expected), (name, problems[:2])


def test_a_trial_on_the_other_route_and_a_record_without_launch_mappings_are_rejected(full):
    def one_trial(ev):
        cell = [c for c in ev["devices"]["b200"]["default"]["first"]["B"][0]["cells"] if c["key"] == K_SMALL][0]
        cell["samples"][3]["nodes"]["compute"] = {"plan_kernel": 1, "rows_kernel": 1}                 # the cell says one launch; one trial in the middle says two
    assert has(slr.evidence_problems(mutated(full["ev"], one_trial)), "trial 3")

    def legacy(ev):
        for rec in ev["devices"]["b300"]["forced"]["first"]["A"]:
            for cell in rec["cells"]:
                for s in cell["samples"]:
                    s.pop("nodes")
    assert has(slr.evidence_problems(mutated(full["ev"], legacy)), "carries no node mapping")


@pytest.mark.parametrize("name,edit,expected", [
    ("side A measured with the default path", lambda ev: ev["devices"]["b200"]["default"]["first"]["A"][0]["meta"].update(impl=0), "side A measured impl 0"),
    ("side B measured with the explicit two-launch path", lambda ev: ev["devices"]["b300"]["default"]["first"]["B"][1]["meta"].update(impl=64), "side B measured impl 64"),
    ("an environment override of the path", lambda ev: ev["devices"]["b200"]["default"]["first"]["B"][0]["meta"].update(env_impl="64"), "the environment override of the path was set"),
    ("the default arm not in the route order", lambda ev: ev["devices"]["b200"]["default"]["first"]["B"][0]["meta"].update(route_order=False), "the cells were not measured in the route order (meta route_order is False)"),
    ("a default-arm record of before the option", lambda ev: ev["devices"]["b300"]["default"]["first"]["A"][0]["meta"].pop("route_order"), "the cells were not measured in the route order (meta route_order is None)"),
    ("lazy module loading on a gating device", lambda ev: ev["devices"]["b200"]["default"]["first"]["B"][0]["meta"].update(cuda_module_loading=None), "the process ran with CUDA_MODULE_LOADING None, a campaign on a gating device needs EAGER"),
    ("another module loading mode", lambda ev: ev["devices"]["b300"]["forced"]["first"]["A"][0]["meta"].update(cuda_module_loading="LAZY"), "the process ran with CUDA_MODULE_LOADING 'LAZY'"),
    ("a record without the field", lambda ev: ev["devices"]["b300"]["default"]["first"]["A"][0]["meta"].pop("cuda_module_loading"), "the process ran with CUDA_MODULE_LOADING None"),
    ("the control on in the default arm", lambda ev: ev["devices"]["b200"]["default"]["first"]["B"][0]["meta"].update(single_force=True), "the single-launch control was True, side B of the default arm needs False"),
    ("the control off in the forced arm", lambda ev: ev["devices"]["b200"]["forced"]["first"]["B"][0]["meta"].update(single_force=False), "side B of the forced arm needs True"),
    ("the control on for side A", lambda ev: ev["devices"]["b200"]["forced"]["first"]["A"][0]["meta"].update(single_force=True), "side A of the forced arm needs False"),
    ("a record without the control recorded", lambda ev: ev["devices"]["b200"]["default"]["first"]["A"][0]["meta"].pop("single_force"), "the single-launch control was None"),
    ("a cell that does not say the control", lambda ev: ev["devices"]["b200"]["default"]["first"]["A"][0]["cells"][0]["args"].pop("single_force"), "says impl 64, single_force None"),
    ("a cell of another path than its record", lambda ev: ev["devices"]["b200"]["default"]["first"]["B"][0]["cells"][0]["args"].update(impl=64), "the key and the record say otherwise"),
    ("no limit constant", lambda ev: ev["devices"]["b200"]["default"]["first"]["A"][0]["meta"].pop("single_per_sm"), "the limit constant of the measured tree (single_per_sm) is None"),
    ("a dirty tree", lambda ev: ev["devices"]["b200"]["default"]["first"]["A"][0]["meta"].update(dirty=True), "the measured tree is not clean"),
    ("another kind of record", lambda ev: ev["devices"]["b200"]["default"]["first"]["A"][0]["meta"].update(kind="chain-wall"), "the record is of kind 'chain-wall'"),
    ("a record of another GPU", lambda ev: ev["devices"]["b300"]["default"]["first"]["A"][0]["meta"].update(gpu="NVIDIA B200"), "the GPU 'NVIDIA B200' is not a B300"),
    ("a compute capability that is no number", lambda ev: ev["devices"]["b300"]["default"]["first"]["A"][0]["meta"].update(cc="ten"), "the compute capability 'ten' is not of the form 10.0"),
    ("a cell that is not a decode cell", lambda ev: ev["devices"]["b200"]["default"]["first"]["A"][0]["cells"][0].update(key="chain;N=1;"), "is not the key of a decode cell"),
])
def test_the_identity_and_the_path_of_every_record_are_checked(full, name, edit, expected):
    assert has(slr.evidence_problems(mutated(full["ev"], edit)), expected), name


@pytest.mark.parametrize("name,edit,expected", [
    ("two commits", lambda ev: ev["devices"]["b300"]["default"]["first"]["B"][0]["meta"].update(commit="1" * 40), "the records are of more than one commit"),
    ("two limit constants", lambda ev: ev["devices"]["b300"]["default"]["first"]["B"][0]["meta"].update(single_per_sm=16), "more than one limit constant"),
    ("a device whose arms ran on two hosts", lambda ev: [r["meta"].update(host="another-node") for r in ev["devices"]["b200"]["forced"]["first"]["A"] + ev["devices"]["b200"]["forced"]["first"]["B"]],
     "b200: the records of its arms were not measured on one device and host"),
    ("two SM counts", lambda ev: [r["meta"].update(sm_count=132) for arm in ev["devices"]["b300"].values() for ph in arm.values() for side in ph.values() for r in side], "the gating devices have different SM counts"),
    ("cells made for another SM count", lambda ev: [r["meta"].update(boundary_sms=132) for arm in ev["devices"]["b200"].values() for ph in arm.values() for side in ph.values() for r in side],
     "the cells of the records were not all made for the [148] SMs of the gating devices (boundary_sms:"),
])
def test_the_records_of_a_campaign_are_of_one_commit_one_limit_and_one_device_per_name(full, name, edit, expected):
    assert has(slr.evidence_problems(mutated(full["ev"], edit)), expected), name


def test_one_record_file_is_one_measurement_and_both_arms_of_both_devices_are_needed(full):
    def same_file(ev):
        ev["inputs"]["b200"]["default"]["first"]["B"][0]["sha256"] = ev["inputs"]["b200"]["default"]["first"]["A"][0]["sha256"]
    assert has(slr.evidence_problems(mutated(full["ev"], same_file)), "has the content of b200 default first A record b200-default-first-A1.json: one record file is one measurement")
    assert has(slr.evidence_problems(mutated(full["ev"], lambda ev: ev["devices"]["b300"].pop("forced"))), "b300 has no first campaign of the forced arm")
    assert has(slr.evidence_problems(mutated(full["ev"], lambda ev: ev["devices"]["b200"]["default"]["first"].update(B=[]))), "b200 has no first campaign of the default arm")


def every_record(ev):
    return [r for arms in ev["devices"].values() for phases in arms.values() for ph in phases.values() for side in ph.values() for r in side]


def pooled_with_one_trial(ev):
    """Every record becomes a pooled one with a single trial per cell (the middle trial, which has the median exactly: what the cells compare is not changed)."""
    for rec in every_record(ev):
        rec["meta"].update(sample_stat="pooled", trials=1)
        for cell in rec["cells"]:
            cell["samples"] = cell["samples"][2:3]


def keys_over_lockstep_cells(ev):
    """The keys still name the cold, staggered and probe cells; every cell measured the lock-step workload at no position."""
    for rec in every_record(ev):
        for cell in rec["cells"]:
            if cell["args"]["regime"] != "lockstep":
                cell["args"].update(regime="lockstep", pos=None)


def no_eviction_buffer(ev):
    for rec in every_record(ev):
        rec["meta"].update(eviction_bytes=0)


def a_cell_that_is_no_cell_of_the_campaign(ev):
    cell = ev["devices"]["b300"]["forced"]["first"]["B"][1]["cells"][0]
    cell.update(key=key_of(3, 7))
    cell["args"].update(N=3, H=7)


@pytest.mark.parametrize("name,edit,expected", [
    ("pooled records with one trial each: not Method M", pooled_with_one_trial, "has sample_stat 'pooled', not 'process_median'"),
    ("pooled records with one trial each: one trial", pooled_with_one_trial, "declares 1 trials per cell: Method M needs at least 5 trials"),
    ("cold, staggered and probe keys over lock-step cells", keys_over_lockstep_cells, "does not measure the workload its key names (regime"),
    ("no eviction buffer under the cold keys", no_eviction_buffer, "holds cold cells and an eviction buffer of 0 bytes, less than twice the L2 cache"),
    ("a cell that is no cell of the campaign", a_cell_that_is_no_cell_of_the_campaign, "is not a cell of the decode benchmark"),
])
def test_records_that_are_not_method_m_records_of_the_campaigns_cells_are_no_evidence_whatever_the_comparator_makes_of_them(full, name, edit, expected):
    """The comparator takes pooled records and records of any workload. The evidence goes through the contract of the final campaigns first: the whole path (the check of the table against its
    evidence) returns these problems and no table, and the table of the campaign that the unchanged records give is not an adopted one for them."""
    mut = mutated(full["ev"], edit)
    assert has(slr.evidence_problems(mut), expected), name
    assert has(slr.check_problems(full["table"], mut, results=full["results"]), expected), name
    assert slr.evidence_problems(full["ev"]) == [] and slr.check_problems(full["table"], full["ev"], results=full["results"]) == []


def test_the_graph_replays_are_held_to_the_same_route_when_they_were_profiled(full):
    def profiled(ev):
        for dev, arms in ev["devices"].items():
            for arm, phases in arms.items():
                for phase in slr.PHASES:
                    for side in slr.SIDES:
                        for rec in phases[phase][side]:
                            rec["meta"]["graph_nodes"] = True
                            for cell in rec["cells"]:
                                for s in cell["samples"]:
                                    s["graph_nodes"] = {"compute": dict(s["nodes"]["compute"]), "aux": {}}
    ev = mutated(full["ev"], profiled)
    assert slr.evidence_problems(ev) == [] and slr.facts_of(ev)["graph_nodes"] is True and slr.table_of(full["results"], slr.facts_of(ev))["meta"]["graph_nodes"] is True
    wrong = mutated(ev, lambda e: [c["samples"][1]["graph_nodes"].update(compute={"plan_kernel": 1, "rows_kernel": 1}) for c in e["devices"]["b200"]["default"]["first"]["B"][0]["cells"] if c["key"] == K_SMALL])
    assert has(slr.evidence_problems(wrong), "cell %s, trial 1: the graph replay launched {'plan_kernel': 1, 'rows_kernel': 1}, the specification says {'rows_kernel': 1}" % K_SMALL)
    missing = mutated(ev, lambda e: [c["samples"][0].pop("graph_nodes") for c in e["devices"]["b300"]["forced"]["first"]["B"][0]["cells"] if c["key"] == K_BIG])
    assert has(slr.evidence_problems(missing), "the graph replay launched None")


def test_a_comparison_that_the_comparator_finds_invalid_makes_the_outcome_undecided_not_not_adopted():
    """No re-run for a noisy cell: the comparator has no verdict, the decision has none, and the table is not final."""
    values = at(K_BIG, metrics=("gpu_graph_us",), spread=2.0, judged_on="first", arms=("default",), base=default_values)         # a noisy cell that no re-run covers
    ev = make_ev(values, cells=MINI)
    assert slr.evidence_problems(ev) == []
    results = slr.recompute(ev)
    assert results["b200"]["default"]["gpu_graph_us"]["verdict"] == "INVALID" and has(results["b200"]["default"]["gpu_graph_us"]["problems"], "no valid re-run")
    d = slr.decide(results, limits())
    assert d["outcome"] == "undecided" and has(d["reasons"], "b200 gpu_graph_us: the comparison is INVALID") and has(d["reasons"], "no valid re-run")
    wrong_order = slr.recompute(ev, compare=lambda base, cand, **kw: br.compare(base, cand, **dict(kw, order="alternate")))
    assert wrong_order["b200"]["default"]["total_us"]["verdict"] == "INVALID" and slr.decide(wrong_order, limits())["outcome"] == "undecided"       # the ABBA files are not strictly alternating


def test_the_comparator_is_called_with_the_parameters_of_the_campaign_driver():
    calls = []

    def spy(base, cand, **kw):
        calls.append(kw)
        return br.compare(base, cand, **kw)
    ev = make_ev(default_values, cells=MINI)
    slr.recompute(ev, compare=spy)
    assert len(calls) == 2 * 2 * 5 * 2                                                         # every metric of every arm: with the re-run records, and again without them (the first campaign of every cell)
    assert sum(kw["base_rerun"] is None for kw in calls) == sum(kw["base_rerun"] is not None for kw in calls) == 2 * 2 * 5
    for kw in calls:
        assert (kw["tol_pct"], kw["mode"], kw["min_repeats"], kw["baseline_unstable"], kw["require_clean_cand"], kw["order"], kw["require_nodes"], kw["expect_base_commit"]) == (
            1.0, "regress", 3, "widen", True, "abba", True, COMMIT)
        assert (kw["cells"] == ["regime=lockstep;", "regime=staggered;"]) == (kw["metric"] in ("cpu_graph_us", "cpu_eager_us")) and (kw["cells"] is None) == (kw["metric"] not in slr.LOOP_METRICS)
    assert {kw["metric"] for kw in calls} == set(slr.GATE_METRICS) and all(kw["base_rerun"] is None or len(kw["base_rerun"]) == 4 for kw in calls)


# ---- the table is checked by deriving it again ------------------------------------------------------------------------------------------------------
def set_path(path, value):
    def edit(obj):
        node = obj
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return edit


def index_of(table, dev, arm, key):
    return next(i for i, r in enumerate(table["rows"]) if (r["device"], r["arm"], r["key"]) == (dev, arm, key))


def test_the_table_passes_its_own_check_and_a_derivation_difference_is_rejected(full):
    t, ev, results = full["table"], full["ev"], full["results"]
    assert slr.check_problems(t, ev, results=results) == []
    i, j = index_of(t, "b200", "default", K_SMALL), index_of(t, "b300", "forced", K_BIG)
    for name, edit, expected in (
            ("a median edited", set_path(["rows", i, "metrics", "gpu_graph_us", "cand_median"], 96.0), "table.rows[%d].metrics.gpu_graph_us.cand_median is 96.0, the evidence gives 97.0" % i),
            ("a status edited", set_path(["rows", j, "metrics", "gpu_graph_us", "status"], "PASS"), "status is 'PASS', the evidence gives 'FAIL'"),
            ("a spread edited", set_path(["rows", i, "metrics", "gpu_eager_us", "base_spread_pct"], 0.0), "base_spread_pct is 0.0"),
            ("a judged-on label edited", set_path(["rows", i, "metrics", "gpu_graph_us", "judged_on"], "first"), "judged_on is 'first', the evidence gives 'rerun'"),
            ("a verdict edited", set_path(["comparisons", "b200", "forced", "gpu_graph_us", "verdict"], "PASS"), "verdict is 'PASS', the evidence gives 'FAIL'"),
            ("the outcome edited", set_path(["decision", "outcome"], "not_adopted"), "decision.outcome is 'not_adopted', the evidence gives 'adopted'"),
            ("a gain invented", lambda x: x["decision"]["gains"].pop(0), "decision.gains has"),
            ("the calibration edited", set_path(["calibration", "per_sm"], 16), "calibration.per_sm is 16, the evidence gives 8"),
            ("the limit constant edited", set_path(["meta", "per_sm"], 16), "meta.per_sm is 16, the evidence gives 8"),
            ("the candidate edited", set_path(["meta", "candidate"], "1" * 40), "meta.candidate is"),
            ("a digest edited", set_path(["inputs", "b200", "default", "first", "A", 0, "sha256"], "0" * 64), "sha256 is"),
            ("a row dropped", lambda x: x["rows"].pop(i), "table.rows has"),
            ("a key added", lambda x: x.update(extra={"a": 1}), "table.extra is not in the derived table")):
        problems = slr.check_problems(mutated(t, edit), ev, results=results)
        assert problems and has(problems, expected), (name, problems[:2])


def test_a_table_without_its_evidence_is_checked_by_its_own_rules(full):
    t = full["table"]
    assert slr.check_problems(t, constants=CONSTANTS) == []
    i = index_of(t, "b200", "default", K_SMALL)
    assert has(slr.check_problems(mutated(t, set_path(["decision", "outcome"], "not_adopted")), constants=CONSTANTS), "the decision or the calibration of the table is not the one that its figures give")
    assert has(slr.check_problems(mutated(t, set_path(["rows", i, "metrics", "gpu_graph_us", "status"], "FAIL")), constants=CONSTANTS), "the status is FAIL")
    assert has(slr.check_problems(mutated(t, lambda x: x["rows"].pop(i)), constants=CONSTANTS), "no row")
    assert has(slr.check_problems(mutated(t, set_path(["meta", "per_sm"], 16)), constants=CONSTANTS), "calibrates to 8")                  # a table of the limit 16 whose data say 8


def test_a_table_that_carries_a_plan_identifier_is_rejected(full):
    t = mutated(full["table"], lambda x: x["meta"]["devices"]["b200"].update(gpu="NVIDIA B200 " + "task" + "12"))
    assert has(slr.check_problems(t, constants=CONSTANTS), "plan identifier")


def test_the_document_and_the_row_are_checked_against_the_derived_table(full):
    t, ev, results = full["table"], full["ev"], full["results"]
    doc = document_of(t)
    assert slr.check_problems(t, ev, document_text=doc, results=results) == []
    assert has(slr.check_problems(t, ev, document_text=doc.replace("-3.00 r", "-4.00 r", 1), results=results), "not the generated one")
    codegen, resources = {"conda": {}}, {"conda": {}}
    row = slr.ledger_row(t, t["decision"], COMMIT, ["m"], codegen, resources)
    assert slr.check_problems(t, ev, row=row, results=results) == []
    assert has(slr.check_problems(t, ev, row=mutated(row, lambda r: r["ab"]["b200"]["medians"]["gpu_graph_us|" + K_SMALL].update(cand=96.0)), results=results), "cand is 96.0")


# ---- the command line ----------------------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def files(full, tmp_path_factory):
    directory = tmp_path_factory.mktemp("single-launch")
    return {"dir": directory, "manifest": write_ev(full["ev"], directory)}


def test_the_command_lines_build_render_check_and_write_the_row(files, tmp_path, capsys):
    manifest, out, doc = str(files["manifest"]), tmp_path / "table.json", tmp_path / "perf.md"
    doc.write_text("# Performance\n\n## 7. Something\ntext\n")
    assert slr.main(["build", "--evidence", manifest, "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "OUTCOME: adopted" in text and "CALIBRATED CTAs PER SM: 8 (the tree has 8)" in text and "VERDICT: PASS" in text and "REASON: no gating metric regresses on b200, b300" in text
    table = json.loads(out.read_text())
    assert table["decision"]["outcome"] == "adopted" and slr.final_problems(table, CONSTANTS) == []
    assert slr.main(["render", "--table", str(out), "--document", str(doc)]) == 0
    assert doc.read_text().count(slr.SECTION_TITLE) == 1 and doc.read_text().startswith("# Performance\n\n## 7. Something\ntext\n\n## 9. ")
    before = doc.read_text()
    assert slr.main(["render", "--table", str(out), "--document", str(doc)]) == 0 and doc.read_text() == before                  # regenerates without a diff
    capsys.readouterr()
    assert slr.main(["check", "--table", str(out), "--evidence", manifest, "--document", str(doc)]) == 0
    assert "OUTCOME: adopted" in capsys.readouterr().out
    assert slr.main(["check", "--table", str(out)]) == 0                                                                           # the rules of the table alone
    capsys.readouterr()
    doc.write_text(before.replace("| 1 | 12 | 12 | lockstep | - | single-launch | 100.000 | 97.000 |", "| 1 | 12 | 12 | lockstep | - | single-launch | 100.000 | 95.000 |", 1))
    assert slr.main(["check", "--table", str(out), "--evidence", manifest, "--document", str(doc)]) == 1
    assert "not the generated one" in capsys.readouterr().out
    bad = json.loads(out.read_text())
    bad["rows"][0]["metrics"]["gpu_graph_us"]["cand_median"] += 1.0
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    assert slr.main(["check", "--table", str(tmp_path / "bad.json"), "--evidence", manifest]) == 1
    assert "cand_median" in capsys.readouterr().out
    assert slr.main(["check", "--table", str(tmp_path / "missing.json"), "--evidence", manifest]) == 2
    codegen, resources, row_file = tmp_path / "codegen.json", tmp_path / "resources.json", tmp_path / "row.json"
    codegen.write_text(json.dumps({"conda": {}}))
    resources.write_text(json.dumps({"conda": {}}))
    assert slr.main(["row", "--table", str(out), "--markers", "one_marker", "another", "--codegen-gate", str(codegen), "--resource-gate", str(resources), "--out", str(row_file)]) == 0
    row = json.loads(row_file.read_text())
    assert row["id"] == "single_launch" and row["markers"] == ["one_marker", "another"] and row["commit"] == COMMIT and "_path" not in row["codegen_gate"]
    assert slr.main(["check", "--table", str(out), "--evidence", manifest, "--row", str(row_file)]) == 0
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"schema": rc.LEDGER_SCHEMA, "steps": [row]}))
    assert slr.main(["check", "--table", str(out), "--row", str(ledger)]) == 0
    row["ab"]["b200"]["medians"]["gpu_graph_us|" + K_SMALL]["cand"] = 1.0
    row_file.write_text(json.dumps(row))
    capsys.readouterr()
    assert slr.main(["check", "--table", str(out), "--row", str(row_file)]) == 1 and "cand is 1.0" in capsys.readouterr().out
    assert slr.main(["row", "--table", str(out), "--markers", "m", "--codegen-gate", str(codegen), "--resource-gate", str(resources), "--revert-commit", "1" * 40, "--out", str(row_file)]) == 2


def test_the_build_refuses_evidence_whose_launches_are_not_the_specifications(files, full, tmp_path, capsys):
    ev = mutated(full["ev"], lambda e: [s["nodes"].update(compute={"rows_kernel": 1}) for c in e["devices"]["b200"]["default"]["first"]["A"][0]["cells"] if c["key"] == K_SMALL for s in c["samples"]])
    directory = tmp_path / "bad"
    directory.mkdir()
    for dev, arms in ev["devices"].items():
        for arm, phases in arms.items():
            for phase in slr.PHASES:
                for side in slr.SIDES:
                    for entry, rec in zip(ev["inputs"][dev][arm][phase][side], phases[phase][side]):
                        (directory / entry["name"]).write_text(json.dumps(rec))
    manifest = write_ev(ev, directory)
    assert slr.main(["build", "--evidence", str(manifest), "--out", str(tmp_path / "t.json")]) == 1
    out = capsys.readouterr().out
    assert "PROBLEM:" in out and "expected {'plan_kernel': 1, 'rows_kernel': 1}" in out and "VERDICT: FAIL" in out and not (tmp_path / "t.json").exists()


@pytest.mark.parametrize("name,edit,expected", [
    ("no schema", lambda m: m.pop("schema"), "not a cfbi-single-launch-evidence-v1 manifest"),
    ("a missing arm", lambda m: m["devices"]["b300"].pop("forced"), "the manifest has no arm forced of b300"),
    ("a device that does not exist", lambda m: m["devices"].update(b100={"default": {"first": {"A": ["x"], "B": ["y"]}}}), "'b100' is not a device of the campaign"),
    ("no first campaign file", lambda m: m["devices"]["b200"]["default"]["first"].update(A=[]), "b200 default first A must be a non-empty list of file names"),
    ("a re-run of one side", lambda m: m["devices"]["b200"]["default"]["rerun"].update(B=[]), "the re-run needs records for both sides or for neither"),
    ("an unknown entry", lambda m: m.update(note="x"), "unknown entries"),
    ("a reason that is no text", lambda m: m.update(not_run={"h100": 3}), "not_run must map a device to the reason why it was not run"),
])
def test_a_manifest_that_cannot_be_read_is_unusable_input(files, tmp_path, capsys, name, edit, expected):
    manifest = json.loads(files["manifest"].read_text())
    edit(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert has(slr.manifest_problems(manifest), expected), name
    assert slr.main(["build", "--evidence", str(path), "--out", str(tmp_path / "t.json")]) == 2


# ---- the rule against the plan vocabulary applies to the generated text too -----------------------------------------------------------------------------
def test_the_generated_text_and_the_reasons_carry_no_plan_vocabulary(full):
    t = full["table"]
    text = slr.render_section(t, t["decision"], 9) + json.dumps(t)
    assert slr.plan_markers(text) == []
    bad = mutated(t, lambda x: x["decision"]["reasons"].append("see " + "DEC" + "-4"))
    assert slr.plan_markers(json.dumps(bad)) == ["DEC" + "-4"]
