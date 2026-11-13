"""The table of the boundary cells of the single-launch decode and the calibration of its limit (benchmarks/single_launch_record.py): ``calibrate`` and ``table_problems``.

Plain Python, no GPU and no torch (``pytest --noconftest tests/cfbi/test_decode_boundary_table.py``), and no device data: the tables are built by hand from stub comparator results whose figures can be
computed by hand (every cell has a median of 100 microseconds for A, a change of -3 % where the single launch wins, +2.5 % where it loses, a spread of 0.1 %). The rules are those of the table test
of the limit: the rows are exactly the cells of the campaign, the limit of the tree equals the constant of the header and of the specification, no unsafe cell is within the limit, the table calibrates
to the limit and the boundary cells straddle it. Every rule has a table that breaks it and a table that does not, and the doubled limit is rejected by the data alone.
"""
import copy
import functools
import json
import re
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
import single_launch_record as slr  # noqa: E402

CANDIDATE = "0123456789abcdef0123456789abcdef01234567"
PER_SM, SMS = 8, 148
LIMIT = PER_SM * SMS                                                  # 1184 CTAs
UNSAFE_FROM = 1536                                                    # the first grid (N * H) on which the single launch loses in the default scenario
CONSTANTS = {"header": PER_SM, "specification": PER_SM}               # what the tree says about the limit, for the tables that do not read the real tree
DEVICES = {"b200": {"gpu": "NVIDIA B200", "sm_count": SMS, "cc": "10.0", "gating": True}, "b300": {"gpu": "NVIDIA B300 SXM6 AC", "sm_count": SMS, "cc": "10.3", "gating": True}}
H100 = {"gpu": "NVIDIA H100 80GB HBM3", "sm_count": 132, "cc": "9.0", "gating": False}


def has(problems, needle):
    return any(needle in p for p in problems)


def mutated(obj, edit):
    out = copy.deepcopy(obj)
    edit(out)
    return out


def key_of(N, H, regime="lockstep", pos=None):
    return br.cell_key("decode", N=N, H=H, regime=regime, **({"pos": pos} if pos is not None else {}))


K_SMALL = key_of(1, 12)                                               # 12 CTAs
K_WITHIN = key_of(98, 12)                                             # 1176 CTAs: the largest grid of 12 heads within the limit
K_ABOVE = key_of(99, 12)                                              # 1188 CTAs: the next one
K_BIG = key_of(128, 12)                                               # 1536 CTAs
K_COLD = key_of(64, 12, "cold")
K_PROBE = key_of(64, 12, "probe", 31)


# ---- stub comparator results ------------------------------------------------------------------------------------------------------------------------
def stub_row(key, base=100.0, delta=0.0, spread=0.1, judged_on="first", tol=slr.TOL_PCT, status=None, first_delta=None):
    """A row of a comparator result as ``bench_records.compare`` makes it (the same fields, the same rules): A has the median ``base``, B ``base * (1 + delta / 100)``, both have the spread ``spread``
    percent. A spread above 0.3 % judged on the first campaign needs its re-run; judged on the re-run it makes the cell noise-limited and widens the tolerance by the spread. ``first_delta``: the
    change of the median in the first campaign (``first_delta_pct``, which the table tool adds to the rows of the comparator): by default the change of the row itself."""
    cand = base * (1.0 + delta / 100.0)
    noisy = spread > 100 * br.UNSTABLE_SPREAD
    needs, noise_limited = noisy and judged_on == "first", noisy and judged_on == "rerun"
    tol_eff = tol + (spread if noisy else 0.0)
    change = 100.0 * (cand / base - 1.0)
    if status is None:
        status = "NEEDS_RERUN" if needs else ("FAIL" if change > tol_eff else "PASS")
    return {"key": key, "kind": "decode", "metric": "stub", "base_median": base, "base_reference": base, "base_spread_pct": spread, "cand_median": cand, "cand_spread_pct": spread, "cand_judged": cand,
            "delta_pct": change, "median_delta_pct": change, "first_delta_pct": change if first_delta is None else first_delta, "status": status, "note": "", "n_base": 4, "n_cand": 4,
            "noise_limited": noise_limited, "needs_rerun": needs, "tol_pct_effective": tol_eff, "judged_on": judged_on}


def stub_result(rows, metric="gpu_graph_us", verdict=None, problems=None, **over):
    """A comparator result: the verdict is the comparator's (INVALID with a row that needs its re-run, FAIL with a row that fails, otherwise PASS) unless one is given."""
    if verdict is None:
        verdict = "INVALID" if any(r["needs_rerun"] for r in rows) else ("FAIL" if any(r["status"] == "FAIL" for r in rows) else "PASS")
        if verdict == "INVALID" and problems is None:
            problems = ["a cell needs a re-run with 3 times the calls per trial"]
    res = {"verdict": verdict, "problems": list(problems or []), "rows": rows, "metric": metric, "tol_pct": 1.0, "mode": "regress", "unstable_policy": "widen", "order": "abba", "require_nodes": True}
    res.update(over)
    return res


def default_values(dev, arm, metric, key, f, limit=LIMIT, unsafe_from=UNSAFE_FROM):
    """The figures of one cell and metric of the default scenario: the single launch (side B of the forced arm, and of the default arm within the limit) is 3 % faster in the GPU-elapsed metrics of the
    loop regimes and 15 % faster in the kernel sum (the plan kernel is gone), judged on the re-run, until a grid of ``unsafe_from`` CTAs, where it is 2.5 % slower in them (judged on the re-run too).
    Everything else is unchanged."""
    nh = f["N"] * f["H"]
    single = arm == "forced" or nh <= limit
    v = {"base": 100.0, "delta": 0.0, "spread": 0.1, "judged_on": "first"}
    if single and metric in slr.GAIN_METRICS and nh >= unsafe_from:
        v.update(delta=2.5, judged_on="rerun")
    elif single and metric in slr.GAIN_METRICS and f["regime"] in bd.LOOP_REGIMES:
        v.update(delta=-3.0, judged_on="rerun")
    elif single and metric == "total_us" and nh < unsafe_from:
        v.update(delta=-15.0)
    return v


CLEAN = functools.partial(default_values, unsafe_from=10 ** 9)           # the single launch never loses


def at(keys, devices=tuple(DEVICES), arms=slr.ARMS, metrics=("gpu_graph_us",), base=CLEAN, **figures):
    """A ``values`` function that overrides the figures of the cells ``keys`` (on the given devices, arms and metrics) with ``figures`` (``delta``, ``spread``, ``judged_on``, ``base``)."""
    keys = (keys,) if isinstance(keys, str) else tuple(keys)

    def values(dev, arm, metric, key, f):
        v = dict(base(dev, arm, metric, key, f))
        if key in keys and dev in devices and arm in arms and metric in metrics:
            v.update(figures)
        return v
    return values


def facts_for(per_sm=PER_SM, sms=SMS, devices=DEVICES, candidate=CANDIDATE):
    return {"candidate": candidate, "per_sm": per_sm, "sm_count": sms, "devices": copy.deepcopy(dict(devices)), "graph_nodes": False}


def campaign_results(values=default_values, per_sm=PER_SM, sms=SMS, extra_devices=None):
    """``{device: {arm: {metric: result}}}`` of the cells of the campaign for ``per_sm`` on both gating devices (a cell has a host metric in the loop regimes only); ``extra_devices``:
    ``{device: facts}`` of devices that are recorded and do not gate (the forced arm only, like the H100 timing of the campaign)."""
    cc = {d: slr.cc_number(DEVICES[d]["cc"]) for d in DEVICES}
    want = slr.expected_cells(per_sm, sms, cc)
    out = {}
    for dev, arms in [(d, slr.ARMS) for d in DEVICES] + [(d, ("forced",)) for d in (extra_devices or {})]:
        out[dev] = {}
        for arm in arms:
            keys = want[dev][arm] if dev in want else want["b200"]["default"]
            out[dev][arm] = {}
            for metric in slr.GATE_METRICS:
                rows = []
                for key in keys:
                    f = slr.cell_fields(key)
                    if metric in slr.LOOP_METRICS and f["regime"] not in bd.LOOP_REGIMES:
                        continue
                    rows.append(stub_row(key, **values(dev, arm, metric, key, f)))
                out[dev][arm][metric] = stub_result(rows, metric)
    return out


def synthetic_table(values=default_values, per_sm=PER_SM, sms=SMS, extra_devices=None, candidate=CANDIDATE):
    """The table of the stub campaign: the real ``table_of`` on stub results, so that its rows, comparisons, calibration and decision are the tool's own."""
    devices = dict(DEVICES, **(extra_devices or {}))
    return slr.table_of(campaign_results(values, per_sm, sms, extra_devices), facts_for(per_sm, sms, devices, candidate))


def row_of(table, dev, arm, key):
    rows = [r for r in table["rows"] if r["device"] == dev and r["arm"] == arm and r["key"] == key]
    assert len(rows) == 1, (dev, arm, key)
    return rows[0]


def problems_of(table, per_sm=PER_SM, **kw):
    kw.setdefault("constants", {"header": per_sm, "specification": per_sm})
    return slr.table_problems(table, per_sm, **kw)


@pytest.fixture(scope="module")
def good():
    return synthetic_table()


# ---- the cell keys and the cells of the campaign ------------------------------------------------------------------------------------------------------------
def test_nh_is_the_grid_of_the_cell_key():
    assert slr.nh(K_SMALL) == 12 and slr.nh(K_ABOVE) == 1188 and slr.nh(K_PROBE) == 768 and slr.nh(key_of(1, 12, "cold")) == 12
    assert slr.cell_fields(K_PROBE) == {"N": 64, "H": 12, "regime": "probe", "pos": 31}
    for bad in ("decode;N=0;H=12;regime=lockstep;", "decode;N=1;H=12;regime=nowhere;", "decode;N=1;H=12;regime=probe;", "decode;N=1;H=12;regime=lockstep;pos=3;", "decode;H=12;N=1;regime=lockstep;",
                "decode;N=1;H=12;regime=lockstep", "chain;N=1;H=12;regime=lockstep;", "decode;N=1;H=x;regime=lockstep;", "decode;N=+1;H=12;regime=lockstep;"):
        with pytest.raises(ValueError):
            slr.nh(bad)


def test_the_cells_of_the_campaign_are_the_hand_counted_ones():
    """24 shapes in three regimes and the probe: 92 cells; the forced arm holds the 33 cells that the limit of 1184 does not route (the grids from 1188 CTAs on)."""
    shapes = set(bd.boundary_shapes(PER_SM, SMS))
    for H in (6, 12, 33):                                                                 # worked out here, not read from the generator
        for x in (592, 1184, 2368):
            assert (x // H, H) in shapes and (x // H + 1, H) in shapes and (x // H) * H <= x < (x // H + 1) * H
    assert len(shapes) == 18 and shapes >= {(98, 12), (99, 12), (197, 6), (198, 6), (35, 33), (36, 33), (49, 12), (50, 12), (17, 33), (18, 33), (71, 33), (72, 33), (394, 6), (395, 6), (98, 6), (99, 6)}
    want = slr.expected_cells(PER_SM, SMS, {"b200": 100, "b300": 103})
    assert len(want["b200"]["default"]) == len(want["b300"]["default"]) == 92 and len(want["b200"]["forced"]) == len(want["b300"]["forced"]) == 33
    assert sorted({slr.nh(k) for k in want["b200"]["default"]}) == [12, 48, 192, 561, 588, 594, 600, 768, 1155, 1176, 1182, 1188, 1536, 2343, 2364, 2370, 2376, 3072]
    assert sorted({slr.nh(k) for k in want["b200"]["forced"]}) == [1188, 1536, 2343, 2364, 2370, 2376, 3072] and K_ABOVE in want["b200"]["forced"] and K_WITHIN not in want["b200"]["forced"]
    assert len([k for k in want["b200"]["default"] if "regime=probe" in k]) == 20 and K_PROBE in want["b200"]["default"]


def test_the_rows_of_a_table_are_the_hand_computed_figures(good):
    assert len(good["rows"]) == 2 * (92 + 33)
    gain = row_of(good, "b200", "default", K_SMALL)
    assert (gain["N"], gain["H"], gain["nh"], gain["regime"], gain["pos"], gain["route"]) == (1, 12, 12, "lockstep", None, "single-launch")
    g = gain["metrics"]["gpu_graph_us"]
    assert (g["base_median"], g["cand_median"], g["delta_pct"], g["status"], g["noise_limited"], g["judged_on"]) == (100.0, 97.0, -3.0, "PASS", False, "rerun")
    assert gain["metrics"]["total_us"]["delta_pct"] == -15.0 and gain["metrics"]["total_us"]["cand_median"] == 85.0 and set(gain["metrics"]) == set(slr.GATE_METRICS)
    cold = row_of(good, "b300", "default", K_COLD)
    assert set(cold["metrics"]) == {"gpu_graph_us", "gpu_eager_us", "total_us"} and cold["metrics"]["gpu_graph_us"]["delta_pct"] == 0.0          # no host metrics in a cold cell
    big = row_of(good, "b200", "forced", K_BIG)
    assert big["route"] == "single-launch" and big["metrics"]["gpu_graph_us"]["status"] == "FAIL" and big["metrics"]["gpu_graph_us"]["delta_pct"] == 2.5
    assert row_of(good, "b200", "default", K_ABOVE)["route"] == "two-launch" and row_of(good, "b300", "default", K_WITHIN)["route"] == "single-launch"
    assert [r["nh"] for r in good["rows"] if r["device"] == "b200" and r["arm"] == "default"] == sorted(r["nh"] for r in good["rows"] if r["device"] == "b200" and r["arm"] == "default")
    assert all(json.loads(json.dumps(r)) == r for r in good["rows"])                                                                             # JSON, nothing else


def test_the_figures_of_the_table_are_rounded_and_the_rules_read_the_rounded_figures():
    rows = [stub_row(K_SMALL, base=100.0, delta=-1.0000004, judged_on="rerun"), stub_row(K_ABOVE, base=100.0, delta=-0.9999996, judged_on="rerun")]
    res = {"b200": {"default": {"gpu_graph_us": stub_result(rows)}}}
    table_rows = slr.table_rows(res, PER_SM, {"b200": {"sm_count": SMS, "cc": "10.0"}})
    assert [r["metrics"]["gpu_graph_us"]["delta_pct"] for r in table_rows] == [-1.0, -1.0]                                    # rounded to six decimals
    assert slr.figure(-0.0) == 0.0 and str(slr.figure(-0.0)) == "0.0" and slr.figure(2 / 3) == 0.666667


# ---- the right limit passes every rule ----------------------------------------------------------------------------------------------------------------
def test_a_table_of_the_right_limit_passes_every_rule(good):
    assert problems_of(good) == []
    assert slr.calibrate(good) == PER_SM and type(slr.calibrate(good)) is int
    assert slr.calibration_of(good) == good["calibration"] and good["calibration"]["first_unsafe_nh"] == UNSAFE_FROM and good["calibration"]["limit_nh"] == 1188
    assert slr.decide_table(good) == good["decision"] and good["decision"]["outcome"] == "adopted"


# ---- calibrate: the largest tested N * H below the smallest unsafe one, divided by the SM count -----------------------------------------------------------
@pytest.mark.parametrize("unsafe_from,limit_nh,per_sm", [
    (1536, 1188, 8),            # the single launch loses from 1536 CTAs on: the largest tested grid below is 1188, 1188 // 148 = 8
    (3072, 2376, 16),           # only the largest grid loses: 2376 // 148 = 16
    (2343, 1536, 10),           # between the measured values: 1536 // 148 = 10 (an integer that no boundary cell straddles)
    (2364, 2343, 15),
    (1188, 1182, 7),            # the grid right above the limit loses: the largest tested grid below it is 1182, and 1182 // 148 = 7 (not 8: the limit is a floor)
    (10 ** 9, 3072, 20)])       # nothing loses: the largest tested grid is 3072, 3072 // 148 = 20
def test_calibrate_returns_the_limit_of_a_straddling_table(unsafe_from, limit_nh, per_sm):
    t = synthetic_table(functools.partial(default_values, unsafe_from=unsafe_from))
    assert slr.calibration_of(t)["limit_nh"] == limit_nh and slr.calibrate(t) == per_sm and type(slr.calibrate(t)) is int


def test_calibrate_is_zero_when_even_the_smallest_cell_is_unsafe():
    t = synthetic_table(at(K_SMALL, delta=2.5, judged_on="rerun"))
    assert slr.calibration_of(t)["first_unsafe_nh"] == 12 and slr.calibration_of(t)["limit_nh"] == 0 and slr.calibrate(t) == 0


def test_calibrate_takes_the_tested_values_that_it_is_given():
    t = synthetic_table(functools.partial(default_values, unsafe_from=UNSAFE_FROM))
    assert slr.calibrate(t, tested=[12, 768, 1176, 1182, 1536, 3072]) == 7                      # 1182 is the largest of these below 1536: 1182 // 148
    assert slr.calibrate(t, tested=[12, 768]) == 5                                              # 768 // 148
    assert slr.calibrate(t, tested=[1536, 3072]) == 0                                           # nothing below the first unsafe grid
    assert slr.calibrate(t, tested=range(1, 1536)) == 10                                        # 1535 // 148


@pytest.mark.parametrize("what,values,per_sm", [
    ("an unsafe cell of the staggered regime", at(key_of(128, 12, "staggered"), delta=2.5, judged_on="rerun", arms=("forced",)), 8),
    ("an unsafe cold cell", at(key_of(256, 12, "cold"), delta=2.5, judged_on="rerun", arms=("forced",)), 16),         # 3072 CTAs: 2376 // 148
    ("an unsafe probe cell", at(key_of(64, 12, "probe", 63), delta=2.5, judged_on="rerun"), 4),                        # 768 CTAs: the largest tested grid below is 600, 600 // 148
    ("the kernel sum alone", at(K_BIG, metrics=("total_us",), delta=2.5, judged_on="rerun", arms=("forced",)), 8),
    ("the host time of the graph loop alone", at(K_BIG, metrics=("cpu_graph_us",), delta=2.5, judged_on="rerun", arms=("forced",)), 8),
    ("the host time of the eager loop alone", at(K_BIG, metrics=("cpu_eager_us",), delta=2.5, judged_on="rerun", arms=("forced",)), 8),
    ("the eager metric alone", at(K_BIG, metrics=("gpu_eager_us",), delta=2.5, judged_on="rerun", arms=("forced",)), 8),
    ("a cell that fails on the first campaign (no re-run)", at(K_BIG, delta=2.5, arms=("forced",)), 8),
    ("one device only", at(K_BIG, devices=("b300",), delta=2.5, judged_on="rerun", arms=("forced",)), 8)])
def test_any_gating_metric_in_any_regime_on_either_device_makes_a_cell_unsafe(what, values, per_sm):
    """The unsafe cell is a comparator FAIL of any gating metric (the kernel sum and the host times too), in any regime, on either device, on the first campaign or on the re-run."""
    t = synthetic_table(values)
    assert slr.calibrate(t) == per_sm, what
    assert slr.calibration_of(t)["unsafe"] and all(u["metrics"] for u in slr.calibration_of(t)["unsafe"])


def test_a_fail_that_only_a_noise_limited_row_shows_still_counts():
    """A noise-limited cell fails only beyond its widened tolerance, and when it does it is unsafe: its FAIL caps the limit like any other."""
    t = synthetic_table(at(K_BIG, delta=4.0, spread=1.0, judged_on="rerun", arms=("forced",)))
    row = row_of(t, "b200", "forced", K_BIG)["metrics"]["gpu_graph_us"]
    assert row["noise_limited"] and row["status"] == "FAIL" and row["tol_pct_effective"] == 2.0 and row["delta_pct"] == 4.0                # 1 % plus the spread of 1 %
    assert [u["key"] for u in slr.calibration_of(t)["unsafe"] if u["metrics"] == ["gpu_graph_us"]] == [K_BIG, K_BIG] and slr.calibrate(t) == 8
    clean = synthetic_table(CLEAN)
    assert slr.calibrate(clean) == 20
    assert has(problems_of(mutated(t, lambda x: None), per_sm=16, constants=CONSTANTS), "calibrates to 8")                  # and the doubled limit is not what this data allows


def test_a_noise_limited_row_inside_its_widened_tolerance_and_a_row_without_a_verdict_do_not_count():
    inside = synthetic_table(at(K_BIG, delta=1.5, spread=1.0, judged_on="rerun", arms=("forced",)))                       # 1.5 % slower, tolerance 2 %: PASS, noise-limited
    row = row_of(inside, "b200", "forced", K_BIG)["metrics"]["gpu_graph_us"]
    assert row["noise_limited"] and row["status"] == "PASS" and row["delta_pct"] == 1.5 and slr.calibrate(inside) == 20 and slr.calibration_of(inside)["unsafe"] == []
    needs = synthetic_table(at(K_BIG, delta=4.0, spread=1.0, judged_on="first", arms=("forced",)))                         # a spread above 0.3 % without a re-run: no verdict yet
    row = row_of(needs, "b200", "forced", K_BIG)["metrics"]["gpu_graph_us"]
    assert row["needs_rerun"] and row["status"] == "NEEDS_RERUN" and slr.calibrate(needs) == 20 and slr.calibration_of(needs)["unsafe"] == []
    assert needs["comparisons"]["b200"]["forced"]["gpu_graph_us"]["verdict"] == "INVALID"                  # the comparison has no verdict: that is what holds the table back (see below)


def test_a_device_that_does_not_gate_never_caps_the_limit():
    t = synthetic_table(CLEAN, extra_devices={"h100": H100})
    assert any(r["device"] == "h100" for r in t["rows"]) and slr.calibrate(t) == 20
    worse = synthetic_table(at(key_of(1, 12), devices=("h100",), delta=5.0, judged_on="rerun", base=CLEAN), extra_devices={"h100": H100})
    h = [r for r in worse["rows"] if r["device"] == "h100"]
    assert h and any(r["metrics"]["gpu_graph_us"]["status"] == "FAIL" for r in h) and slr.calibrate(worse) == 20 and slr.calibration_of(worse)["unsafe"] == []
    assert not has(problems_of(worse, per_sm=8), "h100")                                                   # its rows are checked for consistency and for nothing else


def test_the_limit_is_an_integer_number_of_ctas_per_sm_and_only_the_integer_is_accepted():
    t = synthetic_table()
    for bad in (8.0, True, "8", 0, -8, None, 8.5):
        assert has(slr.table_problems(t, bad, constants=CONSTANTS), "integer number of CTAs per SM"), bad
    assert problems_of(t, per_sm=8) == []


# ---- the doubled limit, a dropped boundary row, a FAIL within the limit -----------------------------------------------------------------------------------
def test_the_doubled_limit_is_rejected_by_the_table_while_the_right_one_passes(good):
    """No device data: the table of the right limit (the single launch loses from 1536 CTAs on) passes; the same table tested at twice the limit is rejected by every rule that can see it: it was not
    measured at 16 CTAs per SM, it does not hold the cells of 16, FAIL rows lie within 2368 CTAs, and it calibrates to 8."""
    assert problems_of(good, per_sm=PER_SM) == []
    problems = slr.table_problems(good, 2 * PER_SM, constants={"header": 2 * PER_SM, "specification": 2 * PER_SM})
    assert has(problems, "measured at 8 CTAs per SM, the limit under test is 16") and has(problems, "cells of the campaign for 16 CTAs per SM") and has(problems, "within the limit of 2368")
    assert has(problems, "calibrates to 8 CTAs per SM") and has(problems, "not to the 16 under test")


def test_the_doubled_limit_is_rejected_by_the_data_alone_when_the_cells_and_the_constants_are_those_of_the_doubled_limit():
    """A table that was measured at 16 CTAs per SM (its cells are the generator's for 16, the constants say 16) in which the single launch loses from 1536 CTAs on: only the data are left to reject it."""
    t16 = synthetic_table(functools.partial(default_values, limit=2 * LIMIT), per_sm=2 * PER_SM)
    problems = slr.table_problems(t16, 16, constants={"header": 16, "specification": 16})
    assert not has(problems, "cells of the campaign") and not has(problems, "straddles") and not has(problems, "measured at") and not has(problems, "policy header") and not has(problems, "specification")
    assert has(problems, "within the limit of 2368") and has(problems, "the table calibrates to 8 CTAs per SM (the smallest unsafe N * H is 1536, the largest tested N * H below it 1188, 148 SMs), not to the 16")
    cal = slr.calibration_of(t16)
    assert cal["first_unsafe_nh"] == 1536 and cal["limit_nh"] == 1188 and cal["per_sm"] == 8
    assert slr.final_problems(t16, {"header": 16, "specification": 16}) != []                                  # and it is no final table either


def test_a_table_whose_data_allow_a_larger_limit_than_the_one_under_test_is_not_the_table_of_that_limit():
    """Only the largest grid loses (3072 CTAs), so the calibration set gives 16 CTAs per SM. The table has the complete boundary of 8 (nothing missing), and it is rejected for 8: the data set the
    limit, a boundary that was measured for another value does not stand in for them. The table of 16 needs the cells of 16, which this one does not have."""
    t = synthetic_table(functools.partial(default_values, unsafe_from=3072))
    assert slr.calibrate(t) == 16
    at8 = problems_of(t, per_sm=8)
    assert has(at8, "the table calibrates to 16 CTAs per SM (the smallest unsafe N * H is 3072, the largest tested N * H below it 2376, 148 SMs), not to the 8 under test")
    assert not has(at8, "straddles") and not has(at8, "cells of the campaign") and not has(at8, "is within the limit")             # nothing else is wrong with it: the calibration alone rejects it
    at16 = problems_of(t, per_sm=16)
    assert has(at16, "cells of the campaign for 16 CTAs per SM") and has(at16, "the table was measured at 8 CTAs per SM")
    assert slr.final_problems(t, CONSTANTS) != []                                                                                   # and it is no final table of the limit that it was measured at


def test_the_literal_case_of_the_review_a_table_with_no_unsafe_cell_and_a_complete_boundary_of_8_is_rejected_for_8():
    """No cell at or below the horizon is unsafe: the set gives 3072 // 148 = 20 CTAs per SM. The shipped 8 with its complete 8-boundary is rejected (it passed when the limit that a table verifies
    was accepted in place of the calibrated one)."""
    t = synthetic_table(CLEAN)
    assert slr.calibrate(t) == 20 and slr.calibration_of(t)["unsafe"] == [] and slr.calibration_of(t)["first_unsafe_nh"] is None
    problems = problems_of(t, per_sm=8)
    assert has(problems, "the table calibrates to 20 CTAs per SM (no cell at or below the horizon of 3072 CTAs is unsafe: the largest tested N * H is 3072, 148 SMs), not to the 8 under test")
    assert not has(problems, "straddles") and not has(problems, "cells of the campaign") and not has(problems, "is within the limit") and not has(problems, "policy header")
    assert slr.final_problems(t, CONSTANTS) != [] and slr.decide_table(t)["outcome"] == "adopted"                                  # an adopted outcome at a limit that the data do not give


CLEAN_AT_20 = functools.partial(default_values, unsafe_from=10 ** 9, limit=20 * SMS)


def test_the_calibrated_limit_lands_when_the_table_has_the_cells_of_that_limit():
    """The table of 20 CTAs per SM with the cells of 20 (the boundary at 1480, 2960 and 5920 CTAs) and nothing unsafe: the set gives 20, the limit under test is 20, both conditions hold on their
    own and the table passes every rule with the constants that the real tree states."""
    t = synthetic_table(CLEAN_AT_20, per_sm=20)
    assert slr.calibrate(t) == 20 and t["meta"]["per_sm"] == 20 and any(r["nh"] > slr.CALIBRATION_HORIZON_NH for r in t["rows"])
    assert slr.table_problems(t, 20, constants={"header": 20, "specification": 20}) == []
    assert slr.calibration_problems(t, 20) == [] and slr.straddle_problems(t, 20) == []
    assert slr.decide_table(t)["outcome"] == "adopted" and slr.final_problems(t, {"header": 20, "specification": 20}) == []


def test_the_boundary_is_a_condition_of_its_own_and_a_calibrated_limit_does_not_excuse_a_missing_cell():
    t = synthetic_table(CLEAN_AT_20, per_sm=20)
    k = key_of(247, 12)                                                                      # 2964 CTAs: the next grid above the limit at 12 heads
    assert slr.nh(k) == 2964
    broken = mutated(t, lambda x: x["rows"].remove(row_of(x, "b300", "default", k)))
    problems = slr.table_problems(broken, 20, constants={"header": 20, "specification": 20})
    assert slr.calibration_problems(broken, 20) == [] and has(problems, "which straddles 2960 CTAs (the limit times 1/1), has no row in the default arm")


def test_rows_above_the_horizon_are_shown_and_never_calibrate():
    """The twice-limit cells of a campaign for 20 CTAs per SM lie above the horizon: an unsafe one is listed apart and does not lower the limit, and a clean one cannot raise it."""
    k = key_of(494, 12)                                                                      # 5928 CTAs: the next grid above twice the limit at 12 heads, a cell of the forced arm
    t = synthetic_table(at(k, arms=("forced",), delta=2.5, judged_on="rerun", base=CLEAN_AT_20), per_sm=20)
    cal = slr.calibration_of(t)
    assert cal["unsafe"] == [] and [(u["device"], u["nh"]) for u in cal["unsafe_above_horizon"]] == [("b200", 5928), ("b300", 5928)] and cal["per_sm"] == 20 == slr.calibrate(t)
    assert slr.calibration_problems(t, 20) == [] and slr.unsafe_problems(t, 20) == []
    assert has(slr.table_problems(synthetic_table(CLEAN_AT_20, per_sm=20), 40, constants={"header": 40, "specification": 40}), "the table calibrates to 20 CTAs per SM")     # a clean higher cell does not raise it


def test_the_horizon_is_the_largest_grid_of_the_fixed_matrix_and_cuts_the_tested_values_that_calibrate():
    assert slr.CALIBRATION_HORIZON_NH == max(N * H for N, H in bd.MATRIX_D) == 3072
    t = synthetic_table(CLEAN)
    assert slr.calibration_of(t)["horizon_nh"] == 3072 and slr.calibration_of(t)["tested"][-1] == 3072
    assert slr.calibrate(t, tested=[12, 768, 1176, 5000, 9000]) == 7 and slr.calibration_of(t, tested=[12, 5000])["tested"] == [12]        # 1176 // 148; the grids above the horizon are not in the set


def test_the_section_says_whether_the_limit_of_the_tree_is_the_calibrated_value():
    assert ", the calibrated value." in slr.render_section(synthetic_table(), synthetic_table()["decision"])
    clean = synthetic_table(CLEAN)
    text = slr.render_section(clean, clean["decision"])
    assert "that is 20 CTAs per SM. The limit of the tree is 8 CTAs per SM, NOT the calibrated value." in text and "The data allow more than that" not in text
    assert "at or below the horizon of 3072 CTAs" in text


def test_a_fail_within_the_limit_rejects_the_limit():
    t = synthetic_table(at(K_WITHIN, delta=2.5, judged_on="rerun", arms=("default",)))
    problems = problems_of(t)
    assert has(problems, "b200 default %s: FAIL in gpu_graph_us, N * H = 1176 is within the limit of 1184 CTAs" % K_WITHIN) and has(problems, "b300 default %s" % K_WITHIN)
    assert has(problems, "the table calibrates to 7 CTAs per SM") and slr.calibrate(t) == 7
    other = synthetic_table(at(K_SMALL, delta=2.5, judged_on="rerun", metrics=("total_us",), devices=("b300",)))
    assert has(problems_of(other), "b300 default %s: FAIL in total_us, N * H = 12" % K_SMALL) and slr.calibrate(other) == 0


def test_a_fail_exactly_at_the_limit_rejects_it_too():
    """With 6 CTAs per SM the limit is 888 CTAs and the grid of 148 sequences with 6 heads has exactly 888."""
    exact = key_of(148, 6)
    consts = {"header": 6, "specification": 6}
    t6 = synthetic_table(at(exact, delta=2.5, judged_on="rerun", arms=("default",), base=functools.partial(default_values, limit=888, unsafe_from=889)), per_sm=6)
    assert slr.nh(exact) == 6 * SMS == 888
    assert has(slr.table_problems(t6, 6, constants=consts), "N * H = 888 is within the limit of 888 CTAs")
    just_above = synthetic_table(functools.partial(default_values, limit=888, unsafe_from=889), per_sm=6)                 # the first grid above the limit that loses: 891 CTAs, nothing within it
    assert not has(slr.table_problems(just_above, 6, constants=consts), "is within the limit")


def test_a_dropped_boundary_row_is_rejected(good):
    for dev, arm, key, expected in (("b300", "default", K_ABOVE, "no row"), ("b200", "default", key_of(99, 12, "cold"), "no row"), ("b200", "forced", K_ABOVE, "no row"),
                                    ("b300", "default", K_PROBE, "no row"), ("b200", "default", K_SMALL, "no row")):
        t = mutated(good, lambda x: x["rows"].remove(row_of(x, dev, arm, key)))
        problems = problems_of(t)
        assert has(problems, "%s %s: 1 of the %d cells of the campaign for 8 CTAs per SM have no row" % (dev, arm, 92 if arm == "default" else 33)), (dev, arm, key)
        assert has(problems, key), (dev, arm, key)
    t = mutated(good, lambda x: x["rows"].remove(row_of(x, "b300", "default", K_ABOVE)))
    assert has(problems_of(t), "which straddles 1184 CTAs (the limit times 1/1), has no row in the default arm") and has(problems_of(t), "N=99, H=12, lockstep")
    probe = mutated(good, lambda x: x["rows"].remove(row_of(x, "b300", "default", K_PROBE)))
    assert not has(problems_of(probe), "straddles")                                                       # the probe is not a boundary cell: only the cell set rule sees it


def test_the_straddle_rule_does_not_depend_on_the_generator(monkeypatch):
    """The generator and the table agree (both lack the cells N = 99, H = 12), so the cell set rule is satisfied: the arithmetic of the straddle rule is what finds the gap."""
    real = bd.campaign_cell_specs
    monkeypatch.setattr(bd, "campaign_cell_specs", lambda per_sm, sms: [s for s in real(per_sm, sms) if not (s["N"] == 99 and s["H"] == 12)])
    t = synthetic_table()
    problems = problems_of(t)
    assert not has(problems, "cells of the campaign") and has(problems, "N=99, H=12, lockstep") and has(problems, "N=99, H=12, cold") and has(problems, "N=99, H=12, staggered")
    assert len([p for p in problems if "straddles" in p]) == 2 * 3 * 1                                    # two devices, three regimes, the one cell at the limit times 1


def test_extra_and_repeated_rows_and_rows_of_another_campaign_are_rejected(good):
    twice = mutated(good, lambda x: x["rows"].append(copy.deepcopy(row_of(x, "b200", "default", K_SMALL))))
    assert has(problems_of(twice), "have more than one row")
    foreign = mutated(good, lambda x: x["rows"].append(dict(copy.deepcopy(row_of(x, "b200", "default", K_SMALL)), key=key_of(3, 7), N=3, H=7, nh=21)))
    assert has(problems_of(foreign), "1 row(s) are not cells of the campaign")
    cells_of_16 = synthetic_table(per_sm=16)
    assert has(slr.table_problems(cells_of_16, 8, constants=CONSTANTS), "cells of the campaign for 8 CTAs per SM")


def test_the_forced_arm_holds_the_cells_that_the_default_rule_does_not_route():
    """The limit decides which cells the forced arm has: at 16 CTAs per SM the cells up to 2368 CTAs are routed, so the forced arm of that table holds only the larger ones."""
    t16 = synthetic_table(per_sm=16)
    forced = sorted({slr.nh(r["key"]) for r in t16["rows"] if r["device"] == "b200" and r["arm"] == "forced"})
    assert forced and min(forced) > 2368 and 3072 in forced
    assert has(problems_of(synthetic_table(), per_sm=16, constants={"header": 16, "specification": 16}), "cells of the campaign for 16 CTAs per SM")


# ---- the limit of the tree: the header, the specification, the table -----------------------------------------------------------------------------------
def test_the_limit_must_be_the_constant_of_the_header_and_of_the_specification(good):
    assert problems_of(good, constants={"header": 8, "specification": 8}) == []
    assert has(slr.table_problems(good, 8, constants={"header": 16, "specification": 8}), "policy header (DECODE_SINGLE_MAX_CTAS_PER_SM) says 16")
    assert has(slr.table_problems(good, 8, constants={"header": 8, "specification": 9}), "specification (decode_policy.MAX_CTAS_PER_SM) says 9")
    both = slr.table_problems(good, 8, constants={"header": None, "specification": "8"})
    assert has(both, "header") and has(both, "specification")


def test_the_real_tree_says_one_limit_and_the_default_constants_are_read_from_it():
    assert dp.header_per_sm(_ROOT) == dp.MAX_CTAS_PER_SM
    assert slr.constant_problems(dp.MAX_CTAS_PER_SM) == [] and slr.constant_problems(dp.MAX_CTAS_PER_SM + 1) != []
    assert has(slr.constant_problems(dp.MAX_CTAS_PER_SM * 2), "the limit under test is %d CTAs per SM" % (dp.MAX_CTAS_PER_SM * 2))


def test_a_table_of_the_limit_of_the_real_tree_passes_with_the_constants_read_from_the_tree():
    t = synthetic_table(functools.partial(default_values, limit=dp.MAX_CTAS_PER_SM * SMS), per_sm=dp.MAX_CTAS_PER_SM)
    probs = slr.table_problems(t, dp.MAX_CTAS_PER_SM)
    assert not has(probs, "policy header") and not has(probs, "specification")


def test_a_table_that_was_measured_at_another_limit_is_not_the_table_of_this_one(good):
    other = mutated(good, lambda x: x["meta"].update(per_sm=4))
    assert has(problems_of(other), "the table was measured at 4 CTAs per SM, the limit under test is 8")


# ---- the figures of a row agree with each other: a hand edit is a difference ------------------------------------------------------------------------
@pytest.mark.parametrize("name,edit,expected", [
    ("a FAIL turned into a PASS", lambda t: row_of(t, "b200", "forced", K_BIG)["metrics"]["gpu_graph_us"].update(status="PASS"), "the status is PASS for a change of 2.500000 %"),
    ("a PASS turned into a FAIL", lambda t: row_of(t, "b200", "default", K_SMALL)["metrics"]["gpu_graph_us"].update(status="FAIL"), "the status is FAIL for a change of -3.000000 %"),
    ("a change edited", lambda t: row_of(t, "b200", "forced", K_BIG)["metrics"]["gpu_graph_us"].update(delta_pct=0.5), "the change of 0.500000 % is not the change of the medians"),
    ("a median edited", lambda t: row_of(t, "b300", "default", K_SMALL)["metrics"]["gpu_graph_us"].update(cand_median=99.0), "is not the change of the medians"),
    ("a median that is zero", lambda t: row_of(t, "b300", "default", K_SMALL)["metrics"]["gpu_graph_us"].update(base_median=0.0), "the medians must be positive"),
    ("the noise-limited flag set by hand", lambda t: row_of(t, "b200", "default", K_SMALL)["metrics"]["gpu_graph_us"].update(noise_limited=True), "noise_limited is True for a spread of 0.100 %"),
    ("a spread edited so that the cell is noisy", lambda t: row_of(t, "b200", "default", K_SMALL)["metrics"]["gpu_graph_us"].update(base_spread_pct=2.0), "noise_limited is False for a spread of 2.000 % judged on the rerun"),
    ("the tolerance widened by hand", lambda t: row_of(t, "b200", "forced", K_BIG)["metrics"]["gpu_graph_us"].update(tol_pct_effective=5.0), "the tolerance is 5.0000 %"),
    ("a cell that needs its re-run marked as judged", lambda t: row_of(t, "b200", "default", K_SMALL)["metrics"]["gpu_graph_us"].update(needs_rerun=True), "needs_rerun and the status PASS disagree"),
    ("a first campaign judged cell with a large spread", lambda t: row_of(t, "b200", "default", K_COLD)["metrics"]["gpu_graph_us"].update(base_spread_pct=2.0), "was judged on the first campaign without a re-run"),
    ("N edited", lambda t: row_of(t, "b200", "default", K_SMALL).update(N=2), "N is 2, the key says 1"),
    ("N * H edited", lambda t: row_of(t, "b200", "default", K_SMALL).update(nh=11), "nh is 11, the key says 12"),
    ("the regime edited", lambda t: row_of(t, "b200", "default", K_SMALL).update(regime="cold"), "regime is 'cold', the key says 'lockstep'"),
    ("the route edited", lambda t: row_of(t, "b200", "default", K_SMALL).update(route="two-launch"), "the route is two-launch, the specification gives single-launch"),
    ("a metric that the cell has no figure for", lambda t: row_of(t, "b200", "default", K_COLD)["metrics"].update(cpu_graph_us=copy.deepcopy(row_of(t, "b200", "default", K_COLD)["metrics"]["gpu_graph_us"])),
     "are not judged in this cell"),
    ("an entry that is malformed", lambda t: row_of(t, "b200", "default", K_SMALL)["metrics"]["gpu_graph_us"].pop("status"), "the entry is malformed"),
])
def test_a_row_that_disagrees_with_itself_is_rejected(good, name, edit, expected):
    assert problems_of(good) == []
    problems = problems_of(mutated(good, edit))
    assert has(problems, expected), (name, problems[:3])


# ---- the comparisons that a table rests on ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,edit,expected", [
    ("a comparison that is INVALID", lambda t: t["comparisons"]["b300"]["default"]["total_us"].update(verdict="INVALID", problems=["cell x has no valid re-run"]),
     "the comparison of b300 default total_us is INVALID (cell x has no valid re-run): it is no verdict"),
    ("the order of the start", lambda t: t["comparisons"]["b200"]["forced"]["gpu_eager_us"].update(order="alternate"), "was made with order 'alternate', the gate is 'abba'"),
    ("the tolerance of a comparison", lambda t: t["comparisons"]["b200"]["default"]["gpu_graph_us"].update(tol_pct=5.0), "tol_pct 5.0, the gate is 1.0"),
    ("the tolerance of the table", lambda t: t["meta"].update(tol_pct=5.0), "the tolerance of the table is 5.0 %"),
    ("the mode of a comparison", lambda t: t["comparisons"]["b200"]["default"]["gpu_graph_us"].update(mode="within"), "mode 'within', the gate is 'regress'"),
    ("the widening", lambda t: t["comparisons"]["b200"]["default"]["gpu_graph_us"].update(unstable_policy="worst"), "unstable_policy 'worst'"),
    ("the launch mappings not required", lambda t: t["comparisons"]["b200"]["default"]["gpu_graph_us"].update(require_nodes=False), "require_nodes False, the gate is True"),
    ("a comparison that is missing", lambda t: t["comparisons"]["b300"]["forced"].pop("cpu_eager_us"), "the table has no comparison of b300 forced cpu_eager_us"),
    ("a verdict that no row gives", lambda t: t["comparisons"]["b200"]["default"]["gpu_graph_us"].update(verdict="FAIL"), "is FAIL and no row fails"),
    ("a PASS above a failing row", lambda t: t["comparisons"]["b200"]["forced"]["gpu_graph_us"].update(verdict="PASS"), "is PASS and a row is ['FAIL']"),
    ("a count of cells that is not the rows", lambda t: t["comparisons"]["b200"]["default"]["gpu_graph_us"].update(cells=5), "counts 5 cells, the table has 92 rows"),
])
def test_a_table_whose_comparisons_are_not_those_of_the_gate_is_rejected(good, name, edit, expected):
    assert has(problems_of(mutated(good, edit)), expected), name


def test_a_comparison_that_is_invalid_is_no_verdict_and_leaves_the_table_unfinished():
    t = synthetic_table(at(K_SMALL, spread=2.0, judged_on="first", metrics=("gpu_graph_us",), arms=("default",)))                  # a spread of 2 % without a re-run
    assert t["comparisons"]["b200"]["default"]["gpu_graph_us"]["verdict"] == "INVALID"
    assert has(problems_of(t), "the comparison of b200 default gpu_graph_us is INVALID") and not has(problems_of(t), "disagree")
    assert t["decision"]["outcome"] == "undecided" and has(slr.final_problems(t, CONSTANTS), "the outcome is undecided")


def test_a_malformed_table_is_a_problem_and_not_a_crash(good):
    for edit in (lambda t: t.pop("rows"), lambda t: t.update(rows="x"), lambda t: t.update(schema="other"), lambda t: t["meta"].pop("devices"), lambda t: t["meta"].update(candidate="x"),
                 lambda t: t["meta"].update(sm_count=0), lambda t: t["rows"][0].update(key="decode;N=1;"), lambda t: t["rows"][0].update(device="b100"), lambda t: t.pop("comparisons")):
        assert slr.table_problems(mutated(good, edit), 8, constants=CONSTANTS) != []
    assert slr.table_problems("not a table", 8) == ["not a %s table" % slr.SCHEMA] and slr.SCHEMA.endswith("-v2")
    assert slr.check_problems({"schema": slr.SCHEMA, "meta": good["meta"]}) != []


# ---- not every rule ties a table to a limit that a tree ships ---------------------------------------------------------------------------------------------
def test_a_table_of_a_step_that_was_not_adopted_needs_no_shipped_limit():
    t = synthetic_table(at(K_SMALL, delta=2.5, judged_on="rerun"))                                          # unsafe at the smallest cell: calibrates to 0, the outcome is not adopted
    assert t["decision"]["outcome"] == "not_adopted" and slr.calibrate(t) == 0
    assert problems_of(t) != []                                                                              # as a table of the limit 8 it is wrong (c, d) ...
    assert slr.table_problems(t, 8, constants={"header": None, "specification": None}, limit_rules=False) == []        # ... and as the evidence of a step without a limit it is complete
    assert slr.final_problems(t, {"header": None, "specification": None}) == []


def test_a_not_adopted_table_with_a_smaller_limit_that_was_never_tried_is_not_final():
    """An unsafe cell inside the limit and a calibration to a smaller positive limit: the step is not concluded as not adopted before that limit has been verified."""
    t = synthetic_table(at(K_WITHIN, delta=2.5, judged_on="rerun", arms=("default",)))
    assert t["decision"]["outcome"] == "not_adopted" and slr.calibrate(t) == 7
    assert has(slr.final_problems(t, CONSTANTS), "verify the calibrated limit at the commit that holds it before concluding that the single launch is not adopted")
    assert slr.premature_problems(synthetic_table()) == []


# ---- the tracked table (generated by the campaign) -------------------------------------------------------------------------------------------------------
TRACKED_TABLE = _ROOT / "docs" / "cfbi-single-launch-table.json"


@pytest.mark.skipif(not TRACKED_TABLE.exists(), reason="the table is generated by the campaign (benchmarks/single_launch_record.py build) and tracked after it: nothing to read before")
def test_the_tracked_table_satisfies_every_rule_for_the_limit_of_the_tree():
    table = json.loads(TRACKED_TABLE.read_text())
    assert slr.final_problems(table) == []
    if slr.decide_table(table)["outcome"] == "adopted":
        assert table["meta"]["per_sm"] == dp.MAX_CTAS_PER_SM == dp.header_per_sm(_ROOT)
        assert slr.calibrate(table) == dp.MAX_CTAS_PER_SM                                                  # the calibration set of the table gives the limit of the tree
        assert slr.table_problems(table, dp.MAX_CTAS_PER_SM) == []
        doubled = 2 * dp.MAX_CTAS_PER_SM
        assert slr.table_problems(table, doubled, constants={"header": doubled, "specification": doubled}) != []


SURVEY_TABLE = _ROOT / "docs" / "cfbi-single-launch-survey-table.json"


@pytest.mark.skipif(not SURVEY_TABLE.exists(), reason="the survey table is the table of the first campaign, tracked with the limit that it calibrated")
def test_the_survey_table_calibrates_to_the_limit_of_the_tree():
    """The first campaign was measured at a starting limit and timed the single launch above it with the calibration control, up to the largest grid of the matrix. It is the calibration set of
    the limit that the tree ships: no cell of it is unsafe, and the limit that it calibrates to is the constant of the header and of the specification."""
    table = json.loads(SURVEY_TABLE.read_text())
    assert slr.table_problems(table, table["meta"]["per_sm"], limit_rules=False) == []              # its own cells, figures and comparisons are the table of a campaign
    cal = slr.calibration_of(table)
    assert cal["unsafe"] == [] and cal["limit_nh"] == 3072 and cal["tested"][-1] == 3072
    assert slr.calibrate(table) == dp.MAX_CTAS_PER_SM == dp.header_per_sm(_ROOT)


# ---- text of the tracked files ----------------------------------------------------------------------------------------------------------------------------
def test_the_tool_and_its_tests_carry_no_plan_vocabulary_or_marker_of_the_removable_code():
    """The text that the tool and its tests add is checked by the same expressions as the marker scan of the loop (plan identifiers, paths of the loop) and by the identifiers that the absence
    check looks for after a step that was not adopted: these files stay in the tree whatever the outcome."""
    import record_checker as rc
    loop = re.compile("(" + r"\." + "humanize" + r"\b|" + "rl" + "cr" + r"\b|" + "RL" + "CR" + r"\b|" + "ask" + "-codex|" + "goal" + "-tracker|round-[0-9]+-(summary|contract|review)|BL-2026[0-9]{4}-)")
    identifiers = re.compile(r"\b(" + "|".join(("IMPL" + "_DECODE_PLAN", "POS_COUNT" + "_SHIFT", "decode_single" + "_launch", "seq" + "_window", "DE" + "CODE")) + r")\b")
    loop_ids = re.compile(r"\b(Q|B|M|T|D|R)[0-9]{1,2}\b")
    for path in (_ROOT / "benchmarks" / "single_launch_record.py", Path(__file__), _ROOT / "tests" / "cfbi" / "test_single_launch_record.py"):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            for what, rx in (("plan vocabulary", rc.PLAN_VOCABULARY), ("a word of the loop", loop), ("an identifier of the removable code", identifiers), ("a number of the loop", loop_ids)):
                m = rx.search(line)
                assert m is None, "%s:%d: %s (%s): %s" % (path.name, n, what, m.group(0), line.strip()[:120])
