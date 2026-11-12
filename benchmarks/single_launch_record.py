#!/usr/bin/env python3
"""The table, the calibration and the decision of the single-launch decode A/B (plain Python, no torch: runs on the login node).

Side A of the A/B is the candidate tree at the explicit two-launch path (bit 64). Side B is the same tree at its default routing (arm ``default``: every cell of the campaign) or with the hidden
calibration control on (arm ``forced``: the cells that the default rule does not route, so that the single launch is also timed beyond the limit). Everything below is derived from the raw record
files and nothing else: the comparator results are computed again from them with the parameters of the campaign driver (regress gate, tolerance 1 %, three interleaved repeats, widening of
noise-limited cells, a clean candidate, the ABBA start order, the launch mapping of every trial required), the launch mapping of every cell is held against the routing specification (never against
what a record says about itself) and the table is the JSON object that ``build_table`` derives from them. The checker (``check_problems``) reads the evidence again, derives the table again and
rejects every difference, so a figure that was edited by hand is a difference. The section of the performance document is generated from the table (``render_section``) and must equal the generated text.

The rules (the functions below are these rules and nothing else):

  * a cell is unsafe when a gating metric (GPU-elapsed graph replay, GPU-elapsed eager loop, kernel sum, host time of the graph loop, host time of the eager loop) is a comparator FAIL on either
    gating device, noise-limited or not, in either arm; a comparison that is INVALID is no verdict (``calibrate``)
  * the limit is the largest tested N * H below the smallest N * H of an unsafe cell, divided by the SM count and rounded down: an integer number of CTAs per SM (``calibrate``)
  * a table is accepted for a limit when its rows are exactly the cells of the campaign, the limit equals the constant of the policy header and of the specification, no unsafe cell is within the limit,
    the calibration of the table gives the limit and the boundary cells straddle it (``table_problems``)
  * the outcome is "adopted" when no gating metric regresses (every comparison of the default arm is PASS on both gating devices) and a confirmed gain exists on both: a GPU-elapsed graph or eager
    cell within the limit that is stable (not noise-limited), judged on the re-run and faster by the tolerance or more. The kernel sum and the host times gate but are never the evidence of a gain.
    The outcome is "not_adopted" when a comparison FAILs or a device has no confirmed gain. Any INVALID comparison leaves it "undecided": a repair, never "not adopted" (``decide``)

The evidence manifest (a JSON file next to the evidence, paths relative to it) is ``{"schema": "cfbi-single-launch-evidence-v1", "devices": {"b200": {"default": ARM, "forced": ARM}, "b300": {...},
"h100": {"forced": ARM}}, "not_run": {"h100": "why"}}`` where ``ARM`` is ``{"first": {"A": [files], "B": [files]}, "rerun": {"A": [files], "B": [files]}}`` (the re-run may be missing). The gating
devices are b200 and b300 and both need both arms; any other device is recorded and never gates.

  single_launch_record.py build  --evidence MANIFEST --out TABLE
  single_launch_record.py render --table TABLE --document FILE [--number N]
  single_launch_record.py check  --table TABLE [--evidence MANIFEST] [--document FILE] [--row FILE] [--repo DIR]
  single_launch_record.py row    --table TABLE --markers NAME ... --codegen-gate FILE --resource-gate FILE [--revert-commit HASH] [--commit HASH] --out FILE

``check`` without ``--evidence`` applies the rules to the table alone (what the tracked table test does); with it the table is derived again from the raw records as well. Exit status 0 pass,
1 a check fails (an outcome that is undecided is a failure of the record: it is not final), 2 unusable input.
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys

TOOL = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOL)
sys.path.insert(0, TOOL)
sys.path.insert(0, os.path.join(REPO, "tests", "cfbi"))
import bench_decode as bd  # noqa: E402
import bench_records as br  # noqa: E402
import decode_policy as dp  # noqa: E402

SCHEMA = "cfbi-single-launch-table-v1"
EVIDENCE_SCHEMA = "cfbi-single-launch-evidence-v1"
SECTION_TITLE = "Single-launch decode: calibration and decision"
SECTION_NUMBER = 9                                          # the number of the section in the performance document (the next free one)

GATE_METRICS = ("gpu_graph_us", "gpu_eager_us", "total_us", "cpu_graph_us", "cpu_eager_us")      # every one of them gates; the host metrics exist for the loop regimes only
GAIN_METRICS = ("gpu_graph_us", "gpu_eager_us")             # the only evidence of a gain: GPU-elapsed time (the kernel sum falls by construction, the host time is no GPU time)
LOOP_METRICS = ("cpu_graph_us", "cpu_eager_us")
LOOP_CELLS = tuple("regime=%s;" % r for r in bd.LOOP_REGIMES)    # the cell filter of the host metrics, the campaign driver's
AB_ORDER = "abba"                                           # the start order of the record files of an A/B that is evidence of the single launch
TOL_PCT = 1.0                                               # the regress gate
MIN_REPEATS = 3                                             # interleaved record files per side
GATING = ("b200", "b300")                                   # the devices that decide (the same as the ledger's)
KNOWN_DEVICES = GATING + ("h100",)
DEVICE_TAGS = {"b200": "B200", "b300": "B300", "h100": "H100"}      # what the GPU name of a device's records must contain
ARMS = ("default", "forced")
PHASES = ("first", "rerun")
SIDES = ("A", "B")
STRADDLE_HEADS = (6, 12, 33)                                # the heads of the boundary cells: the plan's, written here again so that the generator of the benchmark is not its own witness
STRADDLE_FRACTIONS = ((1, 2), (1, 1), (2, 1))               # the limit divided by two, the limit and twice the limit
STRADDLE_REGIMES = ("lockstep", "staggered", "cold")
SINGLE_LAUNCH, TWO_LAUNCH = dp.SINGLE_LAUNCH, dp.TWO_LAUNCH
OUTCOMES = ("adopted", "not_adopted", "undecided")
STATUSES = ("PASS", "FAIL", "NEEDS_RERUN", "UNSTABLE_BASELINE")
JUDGED = ("first", "rerun")
ROUND = 6                                                   # decimals of every figure of the table: the rules read the rounded figures, so that the table alone gives the decision
ROW_FIGURES = ("base_median", "cand_median", "delta_pct", "base_spread_pct", "cand_spread_pct", "tol_pct_effective")
SLACK = 1e-3                                                # what a rounded figure may differ from the figure derived from other rounded figures (percent)
LEDGER_STEP = "single_launch"
DRAFT_ESTIMATE = "1.5 to 2.2 us per call at N <= 64 and about nothing at N >= 128"


# ---- small helpers -----------------------------------------------------------------------------------------------------------------------------
def load_json(path):
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("%s: a record must be a JSON object" % path)
    return obj


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def figure(x):
    """A figure of the table: a float rounded to ``ROUND`` decimals (negative zero is zero)."""
    return round(float(x), ROUND) + 0.0


def is_int(x):
    return type(x) is int


def is_count(x, minimum=0):
    return type(x) is int and x >= minimum


def is_number(x):
    return type(x) in (int, float) and math.isfinite(x)


def hexcommit(x):
    return isinstance(x, str) and bool(br.COMMIT_PATTERN.match(x))


def commit_matches(a, b):
    return isinstance(a, str) and isinstance(b, str) and bool(a) and bool(b) and (a.startswith(b) or b.startswith(a))


def device_order(dev):
    return (GATING.index(dev) if dev in GATING else len(GATING), dev)


def cc_number(text):
    """``"10.3"`` -> 103: the compute capability as the policy takes it (``decode_policy.cc_number``)."""
    major, minor = text.split(".")
    return dp.cc_number((int(major), int(minor)))


def diff(want, got, path="table", limit=6):
    """The paths at which two JSON values differ (at most ``limit``): ``want`` is the derived value, ``got`` the given one."""
    out = []

    def walk(a, b, p):
        if len(out) >= limit:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b), key=str):
                if k not in a:
                    out.append("%s.%s is not in the derived table" % (p, k))
                elif k not in b:
                    out.append("%s.%s is missing" % (p, k))
                else:
                    walk(a[k], b[k], "%s.%s" % (p, k))
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                out.append("%s has %d entries, the derived table %d" % (p, len(b), len(a)))
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, "%s[%d]" % (p, i))
        elif a != b or type(a) is not type(b):
            out.append("%s is %r, the evidence gives %r" % (p, b, a))
    walk(want, got, path)
    return out[:limit]


# ---- the cells --------------------------------------------------------------------------------------------------------------------------------
def cell_fields(key):
    """``{"N", "H", "regime", "pos"}`` of a decode cell key (``decode;N=1;H=12;regime=lockstep;``, a probe cell also ``pos=0;``); ValueError for a key that is not one, in any other spelling
    of the fields included (the key is what the benchmark writes, ``bench_records.cell_key``)."""
    if not isinstance(key, str) or not key.startswith("decode;") or not key.endswith(";"):
        raise ValueError("%r is not the key of a decode cell" % (key,))
    fields = {}
    for part in key.split(";")[1:-1]:
        name, sep, value = part.partition("=")
        if not sep or name not in ("N", "H", "regime", "pos") or name in fields or (name != "regime" and not re.fullmatch(r"[0-9]+", value)):
            raise ValueError("%r is not the key of a decode cell (the field %r)" % (key, part))
        fields[name] = value
    if "N" not in fields or "H" not in fields or fields.get("regime") not in bd.REGIMES:
        raise ValueError("%r is not the key of a decode cell (N, H and a known regime are required)" % (key,))
    out = {"N": int(fields["N"]), "H": int(fields["H"]), "regime": fields["regime"], "pos": int(fields["pos"]) if "pos" in fields else None}
    if out["N"] < 1 or out["H"] < 1 or (out["pos"] is not None) != (out["regime"] == "probe"):
        raise ValueError("%r is not the key of a decode cell (a probe cell has a position, no other cell has one; N and H are positive)" % (key,))
    spelled = br.cell_key("decode", N=out["N"], H=out["H"], regime=out["regime"], **({"pos": out["pos"]} if out["pos"] is not None else {}))
    if spelled != key:
        raise ValueError("%r is not the canonical spelling %r of a decode cell key" % (key, spelled))
    return out


def nh(cell_key):
    """N * H of a decode cell key: the CTAs of the grid of the call."""
    f = cell_fields(cell_key)
    return f["N"] * f["H"]


def cell_order(key):
    f = cell_fields(key)
    return (f["N"] * f["H"], f["H"], f["N"], bd.REGIMES.index(f["regime"]), -1 if f["pos"] is None else f["pos"])


def route_of(arm, N, H, sms, cc, per_sm):
    """The launches that the specification gives to side B of an arm for a grid of N sequences and H heads (side A is always the two launches): the default routing, or the control on."""
    return dp.expected_route(0, N, H, sms, cc, per_sm, arm == "forced")


def expected_cells(per_sm, sms, cc_by_device):
    """``{device: {arm: [cell keys]}}`` of the gating devices: the default arm holds every cell of the campaign (the matrix, the boundary cells and the probe), the forced arm the cells that the default
    rule does not route (``cc_by_device``: the compute capability of each device as the policy takes it)."""
    specs = bd.campaign_cell_specs(per_sm, sms)
    out = {}
    for dev in GATING:
        out[dev] = {"default": [s["key"] for s in specs],
                    "forced": [s["key"] for s in specs if route_of("default", s["N"], s["H"], sms, cc_by_device[dev], per_sm) == TWO_LAUNCH]}
    return out


# ---- the rows of the table ---------------------------------------------------------------------------------------------------------------------
def entry_of(row):
    """The figures of one cell and metric that the table keeps, from a row of a comparator result."""
    return {"base_median": figure(row["base_median"]), "cand_median": figure(row["cand_median"]), "delta_pct": figure(row["median_delta_pct"]),
            "status": row["status"], "noise_limited": bool(row["noise_limited"]), "needs_rerun": bool(row["needs_rerun"]), "judged_on": row["judged_on"],
            "base_spread_pct": figure(row["base_spread_pct"]), "cand_spread_pct": figure(row["cand_spread_pct"]), "tol_pct_effective": figure(row["tol_pct_effective"])}


def table_rows(results, per_sm, devices):
    """One row per (device, arm, cell), sorted by device, arm and the size of the grid: N, H, N * H, regime, position, the route that the specification gives to side B of the arm on the device
    (its SM count and compute capability), and per metric the medians of A and B, the change of the median in percent, the status, whether the cell is noise-limited or still needs its re-run,
    on which campaign it was judged, both spreads and the tolerance that judged it. ``results``: ``{device: {arm: {metric: comparator result}}}`` (the arms ``default`` and ``forced``),
    ``devices``: ``{device: {"sm_count": n, "cc": "10.0"}}``; a metric that a cell has no figure for (the host metrics of a cell without a loop) is not in the row."""
    rows = []
    for dev in sorted(results, key=device_order):
        for arm in ARMS:
            by_metric = (results[dev] or {}).get(arm)
            if not by_metric:
                continue
            cells = {}
            for metric in GATE_METRICS:
                for r in (by_metric.get(metric) or {}).get("rows", []):
                    cells.setdefault(r["key"], {})[metric] = r
            for key in sorted(cells, key=cell_order):
                f = cell_fields(key)
                rows.append({"device": dev, "arm": arm, "key": key, "N": f["N"], "H": f["H"], "nh": f["N"] * f["H"], "regime": f["regime"], "pos": f["pos"],
                             "route": route_of(arm, f["N"], f["H"], devices[dev]["sm_count"], cc_number(devices[dev]["cc"]), per_sm),
                             "metrics": {m: entry_of(cells[key][m]) for m in GATE_METRICS if m in cells[key]}})
    return rows


def comparisons_of(results):
    """The verdict of every comparison (device, arm, metric) with the parameters it was made under and the counts of its cells."""
    out = {}
    for dev in sorted(results, key=device_order):
        for arm in ARMS:
            for metric in GATE_METRICS:
                res = ((results[dev] or {}).get(arm) or {}).get(metric)
                if res is None:
                    continue
                rows = res.get("rows") or []
                out.setdefault(dev, {}).setdefault(arm, {})[metric] = {
                    "verdict": res["verdict"], "problems": list(res.get("problems") or []), "order": res.get("order"), "tol_pct": res.get("tol_pct"), "mode": res.get("mode"),
                    "unstable_policy": res.get("unstable_policy"), "require_nodes": res.get("require_nodes"), "cells": len(rows), "noise_limited": sum(1 for r in rows if r["noise_limited"]),
                    "rerun": sum(1 for r in rows if r["judged_on"] == "rerun"), "needs_rerun": sum(1 for r in rows if r["needs_rerun"])}
    return out


def results_of_table(table):
    """The comparator results of a table, as far as the decision reads them: ``{device: {arm: {metric: {verdict, problems, order, ..., rows}}}}`` (the inverse of ``comparisons_of`` and
    ``table_rows``, so that the decision can be made from the table alone)."""
    out = {}
    for dev, arms in table["comparisons"].items():
        for arm, metrics in arms.items():
            for metric, summary in metrics.items():
                res = dict((k, summary[k]) for k in ("verdict", "problems", "order", "tol_pct", "mode", "unstable_policy", "require_nodes"))
                res.update(metric=metric, rows=[])
                out.setdefault(dev, {}).setdefault(arm, {})[metric] = res
    for row in table["rows"]:
        for metric, e in row["metrics"].items():
            out[row["device"]][row["arm"]][metric]["rows"].append({
                "key": row["key"], "status": e["status"], "noise_limited": e["noise_limited"], "needs_rerun": e["needs_rerun"], "judged_on": e["judged_on"], "median_delta_pct": e["delta_pct"],
                "delta_pct": e["delta_pct"], "base_median": e["base_median"], "cand_median": e["cand_median"], "tol_pct_effective": e["tol_pct_effective"]})
    return out


# ---- the calibration of the limit --------------------------------------------------------------------------------------------------------------
def unsafe_rows(table):
    """The rows of the gating devices (both arms) in which a gating metric is a comparator FAIL: the cells that are unsafe for the single launch. A noise-limited row counts when it is a FAIL (its
    tolerance was widened already); a row that only needs its re-run, or whose comparison is INVALID, is no FAIL."""
    return [r for r in table["rows"] if r["device"] in GATING and any(isinstance(e, dict) and e.get("status") == "FAIL" for m, e in r["metrics"].items() if m in GATE_METRICS)]


def calibration_of(table, tested=None):
    """What the calibration is made of: the tested N * H (``tested``: given, or those of the rows of the gating devices), the unsafe cells, the smallest N * H among them, the largest tested N * H
    below it (all of them when none is unsafe, 0 when even the smallest is unsafe) and that divided by the SM count, rounded down: the integer limit in CTAs per SM."""
    sms = table["meta"]["sm_count"]
    if tested is None:
        tested = [nh(r["key"]) for r in table["rows"] if r["device"] in GATING]
    tested = sorted(set(tested))
    unsafe = unsafe_rows(table)
    first = min((nh(r["key"]) for r in unsafe), default=None)
    below = [t for t in tested if first is None or t < first]
    limit = max(below, default=0)
    return {"tested": tested, "unsafe": [{"device": r["device"], "arm": r["arm"], "key": r["key"], "nh": nh(r["key"]), "metrics": sorted(m for m, e in r["metrics"].items() if isinstance(e, dict) and e.get("status") == "FAIL")}
                                         for r in sorted(unsafe, key=lambda r: (device_order(r["device"]), r["arm"], cell_order(r["key"])))],
            "first_unsafe_nh": first, "limit_nh": limit, "per_sm": limit // sms}


def calibrate(table, tested=None):
    """The limit in CTAs per SM that the table calibrates: L is the largest tested N * H below the smallest N * H of any unsafe row (any gating device, any arm, H and regime, any gating metric) and
    the limit is L // sm_count, an integer. A table in which even the smallest tested cell is unsafe calibrates to 0."""
    return calibration_of(table, tested)["per_sm"]


# ---- the checks of a table ---------------------------------------------------------------------------------------------------------------------
def structure_problems(table):
    """Why the table cannot be read at all (a missing or malformed part); the rules of ``table_problems`` need these to hold."""
    if not isinstance(table, dict) or table.get("schema") != SCHEMA:
        return ["not a %s table" % SCHEMA]
    problems = []
    meta = table.get("meta")
    if not isinstance(meta, dict):
        return ["the table has no meta"]
    if not hexcommit(meta.get("candidate")):
        problems.append("meta.candidate must be a commit hash")
    for name in ("per_sm", "sm_count"):
        if not is_count(meta.get(name), 1):
            problems.append("meta.%s must be a positive integer" % name)
    if not is_number(meta.get("tol_pct")):
        problems.append("meta.tol_pct must be a number")
    devices = meta.get("devices")
    if not isinstance(devices, dict) or any(d not in devices for d in GATING):
        problems.append("meta.devices must describe the gating devices %s" % ", ".join(GATING))
    else:
        for dev, d in devices.items():
            if not (isinstance(d, dict) and is_count(d.get("sm_count"), 1) and isinstance(d.get("cc"), str) and re.fullmatch(r"[0-9]+\.[0-9]+", d["cc"]) and isinstance(d.get("gpu"), str)):
                problems.append("meta.devices.%s needs a GPU name, a positive SM count and a compute capability of the form 10.0" % dev)
    for part in ("rows", "calibration", "decision", "comparisons", "inputs"):
        if part not in table:
            problems.append("the table has no %s" % part)
    if problems:
        return problems
    if not isinstance(table["rows"], list) or not isinstance(table["comparisons"], dict):
        return ["rows must be a list and comparisons an object"]
    for i, row in enumerate(table["rows"]):
        if not (isinstance(row, dict) and row.get("device") in meta["devices"] and row.get("arm") in ARMS and isinstance(row.get("metrics"), dict) and isinstance(row.get("key"), str)):
            problems.append("row %d is not a row of the table (device, arm, key and metrics are needed)" % i)
            continue
        try:
            cell_fields(row["key"])
        except ValueError as e:
            problems.append("row %d: %s" % (i, e))
    return problems


def row_consistency_problems(table):
    """Every row says what its key says, and every figure of a row is consistent with the others: FAIL exactly when the change of the median exceeds the tolerance that judged the cell, the
    tolerance is the plain one or, for a noise-limited cell, widened by the larger spread, a cell is noise-limited exactly when it was judged on the re-run and its spread is above 0.3 %, and a
    row that still needs its re-run has no verdict. A rounded figure that was edited by hand makes one of these fail."""
    meta = table["meta"]
    tol, out = meta["tol_pct"], []
    limit_pct = 100 * br.UNSTABLE_SPREAD
    for row in table["rows"]:
        where = "%s %s %s" % (row["device"], row["arm"], row["key"])
        f = cell_fields(row["key"])
        for name, want in (("N", f["N"]), ("H", f["H"]), ("nh", f["N"] * f["H"]), ("regime", f["regime"]), ("pos", f["pos"])):
            if row.get(name) != want or type(row.get(name)) is not type(want):
                out.append("%s: %s is %r, the key says %r" % (where, name, row.get(name), want))
        dev = meta["devices"][row["device"]]
        want_route = route_of(row["arm"], f["N"], f["H"], dev["sm_count"], cc_number(dev["cc"]), meta["per_sm"])
        if row.get("route") != want_route:
            out.append("%s: the route is %s, the specification gives %s" % (where, row.get("route"), want_route))
        extra = sorted(m for m in row["metrics"] if m not in GATE_METRICS)
        if extra or ((f["regime"] not in bd.LOOP_REGIMES) and any(m in row["metrics"] for m in LOOP_METRICS)):
            out.append("%s: the metrics %s are not judged in this cell" % (where, extra or [m for m in LOOP_METRICS if m in row["metrics"]]))
        for metric, e in sorted(row["metrics"].items()):
            if metric not in GATE_METRICS:
                continue
            w = "%s: %s" % (where, metric)
            if not (isinstance(e, dict) and all(is_number(e.get(k)) for k in ROW_FIGURES) and e.get("status") in STATUSES and e.get("judged_on") in JUDGED
                    and type(e.get("noise_limited")) is bool and type(e.get("needs_rerun")) is bool):
                out.append("%s: the entry is malformed" % w)
                continue
            if e["base_median"] <= 0 or e["cand_median"] <= 0:
                out.append("%s: the medians must be positive" % w)
                continue
            if abs(e["delta_pct"] - 100.0 * (e["cand_median"] / e["base_median"] - 1.0)) > SLACK:
                out.append("%s: the change of %.6f %% is not the change of the medians %.6f and %.6f" % (w, e["delta_pct"], e["base_median"], e["cand_median"]))
            if e["needs_rerun"] != (e["status"] == "NEEDS_RERUN"):
                out.append("%s: needs_rerun and the status %s disagree" % (w, e["status"]))
            if e["needs_rerun"] and e["judged_on"] != "first":
                out.append("%s: a cell that needs its re-run is judged on the first campaign" % w)
            spread = max(e["base_spread_pct"], e["cand_spread_pct"])
            if e["status"] in ("PASS", "FAIL"):
                if e["judged_on"] == "first" and spread > limit_pct + SLACK:
                    out.append("%s: a cell with a spread of %.3f %% was judged on the first campaign without a re-run" % (w, spread))
                if abs(spread - limit_pct) > SLACK and e["noise_limited"] != (e["judged_on"] == "rerun" and spread > limit_pct):
                    out.append("%s: noise_limited is %s for a spread of %.3f %% judged on the %s" % (w, e["noise_limited"], spread, e["judged_on"]))
                want_tol = tol + (spread if e["noise_limited"] else 0.0)
                if abs(e["tol_pct_effective"] - want_tol) > SLACK:
                    out.append("%s: the tolerance is %.4f %%, a cell with this spread is judged with %.4f %%" % (w, e["tol_pct_effective"], want_tol))
                if abs(e["delta_pct"] - e["tol_pct_effective"]) > SLACK and (e["status"] == "FAIL") != (e["delta_pct"] > e["tol_pct_effective"]):
                    out.append("%s: the status is %s for a change of %.6f %% judged with %.4f %%" % (w, e["status"], e["delta_pct"], e["tol_pct_effective"]))
    return out


def comparison_problems(table):
    """The comparisons that the table rests on: every gating device has every gating metric in both arms, each one made under the parameters of the gate (regress, the tolerance, widening, the ABBA
    order, the launch mappings required) and with a verdict (PASS or FAIL, never INVALID), and the verdict is the one that its rows give."""
    out = []
    comparisons = table["comparisons"]
    if table["meta"]["tol_pct"] != TOL_PCT:
        out.append("the tolerance of the table is %r %%, the gate is %r %%" % (table["meta"]["tol_pct"], TOL_PCT))
    for dev in GATING:
        for arm in ARMS:
            for metric in GATE_METRICS:
                s = ((comparisons.get(dev) or {}).get(arm) or {}).get(metric)
                where = "%s %s %s" % (dev, arm, metric)
                if s is None:
                    out.append("the table has no comparison of %s" % where)
                    continue
                rows = [r["metrics"][metric] for r in table["rows"] if r["device"] == dev and r["arm"] == arm and isinstance(r["metrics"].get(metric), dict)]
                if not (isinstance(s, dict) and s.get("verdict") in ("PASS", "FAIL", "INVALID")):
                    out.append("the comparison of %s has no verdict" % where)
                    continue
                if s["verdict"] == "INVALID":
                    out.append("the comparison of %s is INVALID (%s): it is no verdict" % (where, "; ".join(str(p) for p in (s.get("problems") or ["no reason given"])[:2])))
                for name, want in (("order", AB_ORDER), ("tol_pct", table["meta"]["tol_pct"]), ("mode", "regress"), ("unstable_policy", "widen"), ("require_nodes", True)):
                    if s.get(name) != want:
                        out.append("the comparison of %s was made with %s %r, the gate is %r" % (where, name, s.get(name), want))
                if s["verdict"] == "FAIL" and not any(e.get("status") == "FAIL" for e in rows):
                    out.append("the comparison of %s is FAIL and no row fails" % where)
                if s["verdict"] == "PASS" and any(e.get("status") != "PASS" for e in rows):
                    out.append("the comparison of %s is PASS and a row is %s" % (where, sorted({str(e.get("status")) for e in rows if e.get("status") != "PASS"})))
                if s.get("cells") != len(rows):
                    out.append("the comparison of %s counts %r cells, the table has %d rows" % (where, s.get("cells"), len(rows)))
    return out


def cell_set_problems(table, per_sm):
    """The rows are exactly the cells of the campaign for ``per_sm``, on both gating devices: the default arm every cell (the matrix, the boundary cells of the generator, the probe), the forced
    arm the cells that the default rule does not route; none missing, none twice, none that is not a cell of the campaign."""
    meta, out = table["meta"], []
    cc = {dev: cc_number(meta["devices"][dev]["cc"]) for dev in GATING}
    try:
        want = expected_cells(per_sm, meta["sm_count"], cc)
    except (AttributeError, ValueError) as e:
        return ["the cells of the campaign cannot be generated for %d CTAs per SM: %s" % (per_sm, e)]
    for dev in GATING:
        for arm in ARMS:
            keys = [r["key"] for r in table["rows"] if r["device"] == dev and r["arm"] == arm]
            dup = sorted(set(k for k in keys if keys.count(k) > 1))
            missing, extra = sorted(set(want[dev][arm]) - set(keys), key=cell_order), sorted(set(keys) - set(want[dev][arm]), key=cell_order)
            if dup:
                out.append("%s %s: the cells %s have more than one row" % (dev, arm, dup[:3]))
            if missing:
                out.append("%s %s: %d of the %d cells of the campaign for %d CTAs per SM have no row (%s%s)" % (dev, arm, len(missing), len(want[dev][arm]), per_sm, ", ".join(missing[:3]),
                                                                                                           ", ..." if len(missing) > 3 else ""))
            if extra:
                out.append("%s %s: %d row(s) are not cells of the campaign for %d CTAs per SM (%s%s)" % (dev, arm, len(extra), per_sm, ", ".join(extra[:3]), ", ..." if len(extra) > 3 else ""))
    return out


def straddle_problems(table, per_sm):
    """The boundary cells straddle the limit: for every head count of ``STRADDLE_HEADS`` and for the limit divided by two, the limit and twice the limit (x CTAs), the largest N with N * H <= x and
    the next larger N have a row in every regime of ``STRADDLE_REGIMES`` of the default arm on both gating devices. Worked out here from the arithmetic, not from the generator."""
    sms, out = table["meta"]["sm_count"], []
    have = set((r["device"], r["key"]) for r in table["rows"] if r["arm"] == "default")
    for dev in GATING:
        for H in STRADDLE_HEADS:
            for num, den in STRADDLE_FRACTIONS:
                x = per_sm * sms * num // den if (per_sm * sms * num) % den == 0 else None
                lo = (per_sm * sms * num) // (den * H)                  # the largest N with N * H <= x (x need not be an integer: the floor of the quotient is the same)
                for N in (lo, lo + 1):
                    for regime in STRADDLE_REGIMES:
                        key = br.cell_key("decode", N=N, H=H, regime=regime)
                        if N >= 1 and (dev, key) not in have:
                            out.append("%s: the cell N=%d, H=%d, %s, which straddles %s CTAs (the limit times %d/%d), has no row in the default arm" % (
                                dev, N, H, regime, x if x is not None else "%.1f" % (per_sm * sms * num / den), num, den))
    return out


def constant_problems(per_sm, constants=None):
    """The limit under test equals the constant of the policy header and the one of the specification (``constants``: ``{"header": n, "specification": n}``, by default read from the tree that
    holds this tool)."""
    if constants is None:
        try:
            constants = {"header": dp.header_per_sm(REPO), "specification": dp.MAX_CTAS_PER_SM}
        except (OSError, ValueError) as e:
            return ["the limit constant of the policy header cannot be read: %s" % e]
    out = []
    for name, what in (("header", "the policy header (DECODE_SINGLE_MAX_CTAS_PER_SM)"), ("specification", "the specification (decode_policy.MAX_CTAS_PER_SM)")):
        if constants.get(name) != per_sm:
            out.append("the limit under test is %d CTAs per SM, %s says %r" % (per_sm, what, constants.get(name)))
    return out


def unsafe_problems(table, per_sm):
    """No unsafe row (a gating metric FAILs, in either arm, noise-limited or not) lies within the limit: a cell that the limit routes to the single launch must not lose."""
    limit = per_sm * table["meta"]["sm_count"]
    return ["%s %s %s: FAIL in %s, N * H = %d is within the limit of %d CTAs (%d per SM on %d SMs)" % (
        u["device"], u["arm"], u["key"], ", ".join(u["metrics"]), u["nh"], limit, per_sm, table["meta"]["sm_count"]) for u in calibration_of(table)["unsafe"] if u["nh"] <= limit]


def calibration_problems(table, per_sm):
    cal = calibration_of(table)
    if cal["per_sm"] == per_sm:
        return []
    why = ("no cell is unsafe: the largest tested N * H is %d" % cal["limit_nh"]) if cal["first_unsafe_nh"] is None else (
        "the smallest unsafe N * H is %d, the largest tested N * H below it %d" % (cal["first_unsafe_nh"], cal["limit_nh"]))
    return ["the table calibrates to %d CTAs per SM (%s, %d SMs), not to the %d under test" % (cal["per_sm"], why, table["meta"]["sm_count"], per_sm)]


def table_problems(table, per_sm, constants=None, limit_rules=True):
    """Everything that makes a table not the table of the limit ``per_sm`` (an integer number of CTAs per SM): its structure, the figures of every row against each other, the comparisons that it
    rests on (a verdict each, under the parameters of the gate) and
      (a) the rows are exactly the cells of the campaign for ``per_sm`` (the matrix, the boundary cells, the probe) on both gating devices,
      (b) ``per_sm`` equals the constant of the policy header and of the specification (``constants``) and the limit that the table was measured at,
      (c) no unsafe row (a gating metric FAILs) is within the limit,
      (d) the calibration of the table gives ``per_sm``,
      (e) the boundary cells straddle the limit for every head count at the limit divided by two, the limit and twice the limit.
    ``limit_rules`` False leaves out (b) to (d), the rules that tie the table to a limit that a tree ships (a table of a step that was not adopted ships none)."""
    problems = structure_problems(table)
    if problems:
        return problems
    if not (is_int(per_sm) and per_sm >= 1):
        return ["the limit must be an integer number of CTAs per SM of at least 1, got %r" % (per_sm,)]
    meta = table["meta"]
    problems += ["meta.devices.%s: %d SMs, the table says %d" % (dev, d["sm_count"], meta["sm_count"]) for dev, d in sorted(meta["devices"].items()) if dev in GATING and d["sm_count"] != meta["sm_count"]]
    problems += row_consistency_problems(table)
    problems += comparison_problems(table)
    problems += cell_set_problems(table, per_sm)
    problems += straddle_problems(table, per_sm)
    if limit_rules:
        if meta["per_sm"] != per_sm:
            problems.append("the table was measured at %d CTAs per SM, the limit under test is %d" % (meta["per_sm"], per_sm))
        problems += constant_problems(per_sm, constants)
        problems += unsafe_problems(table, per_sm)
        problems += calibration_problems(table, per_sm)
    return problems


# ---- the decision ------------------------------------------------------------------------------------------------------------------------------
def decide(results, limit_by_device, tol_pct=TOL_PCT):
    """The outcome of the A/B: ``{"outcome": "adopted" | "not_adopted" | "undecided", "reasons": [...], "gains": [...], "per_device": {...}}``.

    Only the comparisons of the default arm decide (``results[device]["default"][metric]``, the comparator results of the gating devices), against the limit of each device
    (``limit_by_device``: N * H, the CTAs that the limit allows). Any comparison that is missing, INVALID, or not made under the parameters of the gate (regress at ``tol_pct``, widening, the ABBA
    order, the launch mappings required) leaves the outcome undecided: that is a repair and never "not adopted". Otherwise a confirmed regression (a comparison of any gating metric that FAILs)
    on either device means not adopted, and so does a device without a confirmed gain. A confirmed gain is a row of a GPU-elapsed metric (``GAIN_METRICS``) at N * H within the limit that is stable
    (not noise-limited, not in need of a re-run), judged on the re-run and faster by ``tol_pct`` or more; the kernel sum and the host times never are one. Both devices need one."""
    invalid, regress, nogain, gains, per_device = [], [], [], [], {}
    for dev in GATING:
        limit = (limit_by_device or {}).get(dev)
        info = {"limit": limit, "verdicts": {}, "regressions": 0, "invalid": 0, "gains": 0}
        per_device[dev] = info
        if not is_count(limit):
            invalid.append("%s: the limit is %r CTAs, an integer is needed" % (dev, limit))
            info["invalid"] += 1
            continue
        for metric in GATE_METRICS:
            res = (((results or {}).get(dev) or {}).get("default") or {}).get(metric)
            if not (isinstance(res, dict) and isinstance(res.get("rows"), list)):
                invalid.append("%s %s: there is no comparison" % (dev, metric))
                info["verdicts"][metric], info["invalid"] = None, info["invalid"] + 1
                continue
            verdict = res.get("verdict")
            info["verdicts"][metric] = verdict
            wrong = ["%s %r instead of %r" % (name, res.get(name), want) for name, want in (("order", AB_ORDER), ("tol_pct", tol_pct), ("mode", "regress"), ("unstable_policy", "widen"),
                                                                                         ("require_nodes", True)) if res.get(name) != want]
            if verdict not in ("PASS", "FAIL") or wrong:
                why = (["the comparison is %s" % verdict] if verdict != "PASS" and verdict != "FAIL" else []) + (["it was made with " + ", ".join(wrong)] if wrong else [])
                problem = (res.get("problems") or [""])[0]
                invalid.append("%s %s: %s%s" % (dev, metric, "; ".join(why), " (%s)" % problem if problem and verdict not in ("PASS", "FAIL") else ""))
                info["invalid"] += 1
                continue
            for r in sorted(res["rows"], key=lambda r: cell_order(r["key"])):
                if r["status"] == "FAIL":
                    info["regressions"] += 1
                    inside = nh(r["key"]) <= limit
                    regress.append("%s %s %s: B is %.2f %% slower than A, the tolerance of the cell is %.2f %% (%s the limit of %d CTAs)" % (
                        dev, metric, r["key"], figure(r["median_delta_pct"]), figure(r.get("tol_pct_effective", tol_pct)), "within" if inside else "beyond", limit))
                elif metric in GAIN_METRICS and r["judged_on"] == "rerun" and not r["noise_limited"] and not r["needs_rerun"] and r["status"] == "PASS" and nh(r["key"]) <= limit \
                        and figure(r["median_delta_pct"]) <= -tol_pct:
                    info["gains"] += 1
                    gains.append({"device": dev, "metric": metric, "key": r["key"], "nh": nh(r["key"]), "base_median": figure(r["base_median"]), "cand_median": figure(r["cand_median"]),
                                  "delta_pct": figure(r["median_delta_pct"])})
            if verdict == "FAIL" and not any(r["status"] == "FAIL" for r in res["rows"]):
                info["regressions"] += 1
                regress.append("%s %s: the comparison FAILs" % (dev, metric))
        if not info["gains"] and not info["invalid"]:
            nogain.append("%s: no confirmed gain: no stable GPU-elapsed cell within the limit of %d CTAs, judged on the re-run, is faster than A by %.2f %% or more" % (dev, limit, tol_pct))
    gains.sort(key=lambda g: (device_order(g["device"]), GATE_METRICS.index(g["metric"]), cell_order(g["key"])))
    if invalid:
        outcome, reasons = "undecided", invalid
    elif regress or nogain:
        outcome, reasons = "not_adopted", regress + nogain
    else:
        outcome = "adopted"
        reasons = ["no gating metric regresses on %s" % ", ".join(GATING)] + ["%s: %d confirmed gain cell(s) (GPU-elapsed graph or eager, stable, judged on the re-run, within the limit of %d CTAs)" % (
            dev, per_device[dev]["gains"], per_device[dev]["limit"]) for dev in GATING]
    return {"outcome": outcome, "reasons": reasons, "gains": gains, "per_device": per_device}


def limits_of(table):
    """N * H of the limit of each gating device: the limit of the table times the SM count of the device."""
    return {dev: table["meta"]["per_sm"] * table["meta"]["devices"][dev]["sm_count"] for dev in GATING}


def decide_table(table):
    """The decision that the table alone gives (the same function as ``decide``, on the results that the table holds)."""
    return decide(results_of_table(table), limits_of(table), table["meta"]["tol_pct"])


def timing_of(decision):
    """``(vs_base, vs_previous)`` for the ledger row of a decided outcome. The previous accepted state of this first step is the base: ``vs_base`` passes when no gating metric regresses, and
    ``vs_previous`` when in addition a confirmed gain exists on both devices, i.e. when the step is adopted (a step that was not adopted needs one of the two to fail, and one without a confirmed
    gain fails here although it regresses nothing)."""
    if decision["outcome"] == "undecided":
        raise ValueError("an undecided outcome has no timing result")
    clean = all(v == "PASS" for info in decision["per_device"].values() for v in info["verdicts"].values())
    return ("pass" if clean else "fail"), ("pass" if decision["outcome"] == "adopted" else "fail")


# ---- the ledger row ----------------------------------------------------------------------------------------------------------------------------
def ab_key(metric, cell_key):
    return "%s|%s" % (metric, cell_key)


def ab_figures(table):
    """The ``ab`` block of the step ledger: for each gating device the medians of A and B of every cell and metric of the default arm (key ``<metric>|<cell>``, one pair of numbers per key) and the
    keys that are noise-limited."""
    out = {}
    for dev in GATING:
        medians, noise = {}, []
        for row in table["rows"]:
            if row["device"] != dev or row["arm"] != "default":
                continue
            for metric in GATE_METRICS:
                e = row["metrics"].get(metric)
                if e is None:
                    continue
                key = ab_key(metric, row["key"])
                medians[key] = {"base": e["base_median"], "cand": e["cand_median"]}
                if e["noise_limited"]:
                    noise.append(key)
        out[dev] = {"medians": medians, "noise_limited": sorted(noise)}
    return out


def reason_text(decision, limit=3):
    reasons = decision["reasons"]
    return "; ".join(reasons[:limit]) + ("; and %d more" % (len(reasons) - limit) if len(reasons) > limit else "")


def ledger_row(table, decision, commit, markers=None, codegen_gate=None, resource_gate=None, revert_commit=None):
    """The ``single_launch`` row of the step ledger for the outcome that the table gives (``decision`` must be that outcome: the one of ``decide_table(table)``). The timing results are those of
    ``timing_of`` (the layer chain does not apply to this step), ``ab`` holds the figures of the table (``ab_figures``). The caller supplies what no table holds: the ``markers`` of the step, the
    ``codegen_gate`` and ``resource_gate`` entries (the comparisons of the dumps), and for a step that was not adopted the ``revert_commit``. ValueError when the outcome is undecided (there is no
    row to write until the evidence is repaired) or the arguments do not fit."""
    want = decide_table(table)
    if decision != want:
        raise ValueError("the decision is not the one that the table gives")
    if want["outcome"] == "undecided":
        raise ValueError("the table decides nothing (%s): there is no row to write until the evidence is repaired" % reason_text(want, 2))
    if not commit_matches(commit, table["meta"]["candidate"]):
        raise ValueError("the row is of commit %s, the table was measured at %s" % (commit, table["meta"]["candidate"]))
    if not (isinstance(markers, (list, tuple)) and markers and all(isinstance(m, str) and m for m in markers)):
        raise ValueError("the markers of the step must be a non-empty list of identifiers")
    if not isinstance(codegen_gate, dict) or not isinstance(resource_gate, dict):
        raise ValueError("the code-generation and resource entries are supplied by the caller (codegen_gate and resource_gate)")
    adopted = want["outcome"] == "adopted"
    if adopted == bool(revert_commit):
        raise ValueError("a step that was adopted has no revert commit and one that was not adopted names it")
    vs_base, vs_previous = timing_of(want)
    return {"id": LEDGER_STEP, "outcome": want["outcome"], "commit": commit, "revert_commit": revert_commit if not adopted else None, "markers": list(markers),
            "failed_at": None if adopted else "timing", "reason": "" if adopted else reason_text(want), "gates": {"exactness": True, "codegen": True},
            "timing": {"vs_previous": vs_previous, "vs_base": vs_base, "chain": "not_applicable"}, "codegen_gate": codegen_gate, "resource_gate": resource_gate, "ab": ab_figures(table)}


def row_problems(row, table):
    """Why a ``single_launch`` ledger row is not the row of the table: its outcome, commit, timing results and ``ab`` figures (every median and the noise-limited keys) must be what the table gives,
    and the table must decide something."""
    if not isinstance(row, dict):
        return ["the row is not an object"]
    where = "step %s" % LEDGER_STEP
    out = []
    if row.get("id") != LEDGER_STEP:
        out.append("%s: the row is of %r" % (where, row.get("id")))
    want = decide_table(table)
    if want["outcome"] == "undecided":
        return out + ["%s: the table decides nothing (%s)" % (where, reason_text(want, 2))]
    if row.get("outcome") != want["outcome"]:
        out.append("%s: the table's outcome is %s, the row says %r" % (where, want["outcome"], row.get("outcome")))
    if not commit_matches(row.get("commit"), table["meta"]["candidate"]):
        out.append("%s: the row is of commit %r, the table was measured at %s" % (where, row.get("commit"), table["meta"]["candidate"]))
    vs_base, vs_previous = timing_of(want)
    if row.get("timing") != {"vs_previous": vs_previous, "vs_base": vs_base, "chain": "not_applicable"}:
        out.append("%s: the timing of the row is %r, the table gives vs_previous %s, vs_base %s, chain not_applicable" % (where, row.get("timing"), vs_previous, vs_base))
    if row.get("failed_at") != (None if want["outcome"] == "adopted" else "timing"):
        out.append("%s: failed_at is %r" % (where, row.get("failed_at")))
    out.extend("%s: %s" % (where, d) for d in diff(ab_figures(table), row.get("ab"), "ab"))
    return out


# ---- the evidence -------------------------------------------------------------------------------------------------------------------------------
def manifest_problems(manifest):
    """Why the manifest cannot be read (a ValueError of ``load_evidence``)."""
    if not isinstance(manifest, dict) or manifest.get("schema") != EVIDENCE_SCHEMA:
        return ["not a %s manifest" % EVIDENCE_SCHEMA]
    out = []
    devices = manifest.get("devices")
    if not isinstance(devices, dict):
        return ["the manifest needs a devices object"]

    def files(listing, what, required):
        if not isinstance(listing, list) or (required and not listing) or not all(isinstance(p, str) and p for p in listing):
            out.append("%s must be a %slist of file names" % (what, "non-empty " if required else ""))

    for dev in GATING:
        for arm in ARMS:
            if arm not in (devices.get(dev) or {}):
                out.append("the manifest has no arm %s of %s" % (arm, dev))
    for dev, arms in sorted(devices.items()):
        if dev not in KNOWN_DEVICES or not isinstance(arms, dict):
            out.append("%r is not a device of the campaign (%s)" % (dev, ", ".join(KNOWN_DEVICES)))
            continue
        for arm, phases in sorted(arms.items()):
            if arm not in ARMS or not isinstance(phases, dict) or any(k not in PHASES for k in phases):
                out.append("%s %r: an arm is %s with the campaigns %s" % (dev, arm, " or ".join(ARMS), " and ".join(PHASES)))
                continue
            for phase in PHASES:
                sides = phases.get(phase) or {}
                if not isinstance(sides, dict) or any(s not in SIDES for s in sides):
                    out.append("%s %s %s: the sides are %s" % (dev, arm, phase, " and ".join(SIDES)))
                    continue
                for side in SIDES:
                    files(sides.get(side, []), "%s %s %s %s" % (dev, arm, phase, side), phase == "first")
                if phase == "rerun" and bool(sides.get("A")) != bool(sides.get("B")):
                    out.append("%s %s: the re-run needs records for both sides or for neither" % (dev, arm))
    not_run = manifest.get("not_run", {})
    if not isinstance(not_run, dict) or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in not_run.items()):
        out.append("not_run must map a device to the reason why it was not run")
    unknown = sorted(k for k in manifest if k not in ("schema", "devices", "not_run"))
    if unknown:
        out.append("the manifest has the unknown entries %s" % unknown)
    return out


def load_evidence(manifest_path):
    """The evidence that a manifest names: ``ev["devices"][device][arm][phase][side]`` is the list of loaded records, ``ev["inputs"]`` the name and digest of every file in the same shape.
    ValueError for a manifest that cannot be read, OSError for a record that cannot be opened."""
    manifest = load_json(manifest_path)
    problems = manifest_problems(manifest)
    if problems:
        raise ValueError("%s: %s" % (manifest_path, "; ".join(problems)))
    root = os.path.dirname(os.path.abspath(manifest_path))

    def path(p):
        return p if os.path.isabs(p) else os.path.join(root, p)

    ev = {"devices": {}, "inputs": {}, "not_run": dict(manifest.get("not_run") or {})}
    for dev, arms in sorted(manifest["devices"].items()):
        ev["devices"][dev], ev["inputs"][dev] = {}, {}
        for arm, phases in sorted(arms.items()):
            ev["devices"][dev][arm], ev["inputs"][dev][arm] = {}, {}
            for phase in PHASES:
                ev["devices"][dev][arm][phase], ev["inputs"][dev][arm][phase] = {}, {}
                for side in SIDES:
                    names = (phases.get(phase) or {}).get(side) or []
                    ev["devices"][dev][arm][phase][side] = [br.load_record(path(p)) for p in names]
                    ev["inputs"][dev][arm][phase][side] = [{"name": os.path.basename(p), "sha256": sha256_file(path(p))} for p in names]
    return ev


def all_records(ev):
    """``[(device, arm, phase, side, record, input entry)]`` of every record of the evidence, in manifest order."""
    out = []
    for dev in sorted(ev["devices"], key=device_order):
        for arm in ARMS:
            for phase in PHASES:
                for side in SIDES:
                    recs = (ev["devices"][dev].get(arm) or {}).get(phase, {}).get(side, [])
                    out.extend((dev, arm, phase, side, rec, entry) for rec, entry in zip(recs, ev["inputs"][dev][arm][phase][side]))
    return out


def meta_problems(rec, where, dev, arm, side):
    """What a record must say about the tree and the arm that it measured: one clean commit, side A at the explicit two-launch path (bit 64) and side B at the default path, the control off
    except for the forced arm of side B, no environment override of the path, a limit constant, the device, and that every cell is a decode cell that carries the same path and control."""
    meta, out = rec.get("meta"), []
    if not isinstance(meta, dict):
        return ["%s: no meta" % where]
    if not hexcommit(meta.get("commit")):
        out.append("%s: the commit %r is not a commit hash" % (where, meta.get("commit")))
    if meta.get("dirty") is not False:
        out.append("%s: the measured tree is not clean (dirty %r)" % (where, meta.get("dirty")))
    if meta.get("kind") != "decode-wall":
        out.append("%s: the record is of kind %r, a decode campaign writes decode-wall" % (where, meta.get("kind")))
    want_impl = 64 if side == "A" else 0
    if meta.get("impl") != want_impl or type(meta.get("impl")) is not int:
        out.append("%s: side %s measured impl %r, the explicit two-launch path (64) is side A and the default path (0) side B" % (where, side, meta.get("impl")))
    if meta.get("env_impl") is not None:
        out.append("%s: the environment override of the path was set (%r)" % (where, meta.get("env_impl")))
    want_force = side == "B" and arm == "forced"
    if meta.get("single_force") is not want_force:
        out.append("%s: the single-launch control was %r, side %s of the %s arm needs %r" % (where, meta.get("single_force"), side, arm, want_force))
    if not is_count(meta.get("single_per_sm"), 1):
        out.append("%s: the limit constant of the measured tree (single_per_sm) is %r, a positive integer is needed" % (where, meta.get("single_per_sm")))
    if not is_count(meta.get("sm_count"), 1):
        out.append("%s: sm_count is %r" % (where, meta.get("sm_count")))
    if not (isinstance(meta.get("cc"), str) and re.fullmatch(r"[0-9]+\.[0-9]+", meta["cc"])):
        out.append("%s: the compute capability %r is not of the form 10.0" % (where, meta.get("cc")))
    if not (isinstance(meta.get("gpu"), str) and DEVICE_TAGS[dev] in meta["gpu"].upper()):
        out.append("%s: the GPU %r is not a %s" % (where, meta.get("gpu"), DEVICE_TAGS[dev]))
    for cell in rec.get("cells") or []:
        key = cell.get("key") if isinstance(cell, dict) else None
        args = cell.get("args") if isinstance(cell, dict) and isinstance(cell.get("args"), dict) else {}
        try:
            f = cell_fields(key)
        except ValueError as e:
            out.append("%s: %s" % (where, e))
            continue
        if args.get("impl") != meta.get("impl") or args.get("single_force") is not meta.get("single_force") or (args.get("N"), args.get("H")) != (f["N"], f["H"]):
            out.append("%s: cell %s says impl %r, single_force %r, N %r, H %r: the key and the record say otherwise" % (where, key, args.get("impl"), args.get("single_force"), args.get("N"), args.get("H")))
    return out[:8]


def identity_problems(ev):
    """One commit, one limit constant and one SM count for the whole campaign; one GPU, host and compute capability for all the records of a device (all arms of a device run on one allocation)."""
    out, commits, per_sms, sms, boundary = [], set(), set(), {}, set()
    by_device = {}
    for dev, arm, phase, side, rec, entry in all_records(ev):
        meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
        commits.add(meta.get("commit"))
        per_sms.add(meta.get("single_per_sm"))
        boundary.add(meta.get("boundary_sms"))
        by_device.setdefault(dev, set()).add((meta.get("gpu"), meta.get("host"), meta.get("sm_count"), meta.get("cc")))
        sms.setdefault(dev, set()).add(meta.get("sm_count"))
    if len(commits) > 1:
        out.append("the records are of more than one commit: %s" % sorted(str(c) for c in commits))
    if len(per_sms) > 1:
        out.append("the records were measured with more than one limit constant (single_per_sm): %s" % sorted(str(c) for c in per_sms))
    for dev, ids in sorted(by_device.items()):
        if len(ids) > 1:
            out.append("%s: the records of its arms were not measured on one device and host with one compute capability (%s)" % (dev, sorted(str(i) for i in ids)))
    gating_sms = set(next(iter(v)) for d, v in sms.items() if d in GATING and len(v) == 1)
    if len(gating_sms) > 1:
        out.append("the gating devices have different SM counts (%s): the cells of the campaign and the limit are for one" % sorted(gating_sms))
    elif len(boundary) != 1 or boundary != gating_sms:
        out.append("the cells of the records were not all made for the %s SMs of the gating devices (boundary_sms: %s)" % (sorted(gating_sms), sorted(str(b) for b in boundary)))
    return out


def facts_of(ev):
    """The identity of the campaign as the records state it (``identity_problems`` has to hold): the candidate commit, the limit constant, the SM count of the gating devices and per device the GPU
    name, the SM count and the compute capability."""
    first = [rec for dev, arm, phase, side, rec, entry in all_records(ev)]
    devices = {}
    for dev, arm, phase, side, rec, entry in all_records(ev):
        m = rec["meta"]
        devices.setdefault(dev, {"gpu": m["gpu"], "sm_count": m["sm_count"], "cc": m["cc"], "gating": dev in GATING})
    return {"candidate": first[0]["meta"]["commit"], "per_sm": first[0]["meta"]["single_per_sm"], "sm_count": devices[GATING[0]]["sm_count"], "devices": devices,
            "graph_nodes": all(rec["meta"].get("graph_nodes") is True for rec in first)}


def launch_problems(ev):
    """Every cell of every record launched the kernels that the routing specification names for it: side A the two launches whatever the cell, side B the route of the specification for the cell
    on the device at the limit of the tree that was measured (the default routing, or with the control on). The specification is the independent witness: a record that says what it launched is
    compared with it (``bench_records.route_problems``), so an A cell that launched the single launch only, or a B cell on the other route, is a problem."""
    out = []
    for dev, arm, phase, side, rec, entry in all_records(ev):
        meta = rec["meta"]
        if not (is_count(meta.get("single_per_sm"), 1) and is_count(meta.get("sm_count"), 1) and isinstance(meta.get("cc"), str) and re.fullmatch(r"[0-9]+\.[0-9]+", meta["cc"])):
            continue                                                    # named by meta_problems
        want = {}
        for cell in rec.get("cells") or []:
            try:
                f = cell_fields(cell.get("key"))
            except (ValueError, AttributeError):
                continue                                                # named by meta_problems
            route = TWO_LAUNCH if side == "A" else route_of(arm, f["N"], f["H"], meta["sm_count"], cc_number(meta["cc"]), meta["single_per_sm"])
            want[cell["key"]] = dict(bd.CANONICAL_COMPUTE[route])
        out.extend(br.route_problems([rec], want, "%s %s %s %s" % (dev, arm, phase, side)))
        if meta.get("graph_nodes") is True:                              # the replay of the graph is what the graph metric times: when its launches were profiled they are held to the same route
            bad = ["cell %s, trial %d: the graph replay launched %s, the specification says %s" % (cell["key"], i, (s.get("graph_nodes") or {}).get("compute"), want[cell["key"]])
                   for cell in rec.get("cells") or [] if cell.get("key") in want for i, s in enumerate(cell.get("samples") or [])
                   if not isinstance(s, dict) or (s.get("graph_nodes") or {}).get("compute") != want[cell["key"]]]
            out.extend("%s %s %s %s record: %s" % (dev, arm, phase, side, b) for b in bad[:3])
    return out


def evidence_problems(ev):
    """Why the evidence cannot be the basis of a table (the table is not written, a check fails): a record file that is named twice, the identity and the path of every record (``meta_problems``), the
    identity of the campaign (``identity_problems``) and the launches of every cell against the routing specification (``launch_problems``). What the comparator finds (the start order, the counts of
    a re-run, a missing cell sample) is not listed here: it makes a comparison INVALID, and the outcome undecided."""
    out, seen = [], {}
    for dev, arm, phase, side, rec, entry in all_records(ev):
        where = "%s %s %s %s record %s" % (dev, arm, phase, side, entry["name"])
        if entry["sha256"] in seen:
            out.append("%s has the content of %s: one record file is one measurement" % (where, seen[entry["sha256"]]))
        seen[entry["sha256"]] = where
        out.extend(meta_problems(rec, where, dev, arm, side))
    for dev in GATING:
        for arm in ARMS:
            if not (ev["devices"].get(dev, {}).get(arm, {}).get("first", {}).get("A") and ev["devices"][dev][arm]["first"].get("B")):
                out.append("%s has no first campaign of the %s arm" % (dev, arm))
    if out:
        return out
    out = identity_problems(ev)
    if out:
        return out
    return launch_problems(ev)


def candidate_commit(ev):
    return facts_of(ev)["candidate"]


def comparison_kwargs(metric, commit, rerun):
    """The parameters of the campaign driver's comparison: the regress gate at the tolerance, three interleaved repeats, the widening of noise-limited cells, a clean candidate, the host metrics
    judged on the loop regimes only, the ABBA order and the launch mappings required. The record lists of the first campaign are the arguments, the re-run the keywords given here."""
    return dict(metric=metric, tol_pct=TOL_PCT, mode="regress", expect_base_commit=commit, require_clean_cand=True, min_repeats=MIN_REPEATS, baseline_unstable="widen",
                cells=list(LOOP_CELLS) if metric in LOOP_METRICS else None, base_rerun=rerun["A"] or None, cand_rerun=rerun["B"] or None, order=AB_ORDER, require_nodes=True)


def recompute(ev, compare=None):
    """The comparator results of every device, arm and gating metric, computed again from the raw records (``compare``: the comparison function, ``bench_records.compare``)."""
    compare = compare or br.compare
    commit = candidate_commit(ev)
    results = {}
    for dev in sorted(ev["devices"], key=device_order):
        results[dev] = {}
        for arm in ARMS:
            phases = ev["devices"][dev].get(arm)
            if not phases:
                continue
            results[dev][arm] = {}
            for metric in GATE_METRICS:
                try:
                    results[dev][arm][metric] = compare(phases["first"]["A"], phases["first"]["B"], **comparison_kwargs(metric, commit, phases["rerun"]))
                except (KeyError, TypeError, ValueError, IndexError) as e:        # an unforeseen record shape is unusable input, never a verdict
                    raise ValueError("cannot compare the %s %s records of %s: %s: %s" % (arm, metric, dev, type(e).__name__, e))
    return results


# ---- the table ----------------------------------------------------------------------------------------------------------------------------------
def table_of(results, facts, inputs=None, not_run=None):
    """The table of comparator ``results`` (``{device: {arm: {metric: result}}}``) of a campaign whose identity is ``facts`` (``candidate``: the commit, ``per_sm``: the limit constant of the
    measured tree, ``sm_count``: the SM count of the gating devices, ``devices``: per device ``gpu``, ``sm_count``, ``cc`` and whether it gates, ``graph_nodes``): the comparisons, one row per
    (device, arm, cell), the calibration and the decision. ``inputs``: the digests of the raw files (``load_evidence``), ``not_run``: the devices that were not run and why."""
    per_sm, sms = facts["per_sm"], facts["sm_count"]
    table = {"schema": SCHEMA,
             "meta": {"candidate": facts["candidate"], "per_sm": per_sm, "sm_count": sms, "tol_pct": TOL_PCT, "mode": "regress", "unstable_policy": "widen", "order": AB_ORDER, "min_repeats": MIN_REPEATS,
                      "gate_metrics": list(GATE_METRICS), "gain_metrics": list(GAIN_METRICS), "loop_metrics": list(LOOP_METRICS), "graph_nodes": bool(facts.get("graph_nodes")),
                      "devices": dict((dev, dict(d)) for dev, d in sorted(facts["devices"].items(), key=lambda kv: device_order(kv[0]))), "not_run": dict(sorted((not_run or {}).items()))},
             "inputs": inputs or {},
             "comparisons": comparisons_of(results),
             "rows": table_rows(results, per_sm, facts["devices"])}
    table["calibration"] = calibration_of(table)
    table["decision"] = decide(results, limits_of(table), TOL_PCT)
    table = json.loads(json.dumps(table))
    if decide_table(table) != table["decision"]:
        raise ValueError("the decision that the comparator results give is not the one that the table gives: the figures of the table are rounded differently from the rules")
    return table


def build_table(ev, results):
    """The table that the evidence gives (``evidence_problems`` has to be empty): ``table_of`` the results, with the identity of the campaign as its records state it and the digests of the raw files."""
    return table_of(results, facts_of(ev), ev["inputs"], ev["not_run"])


def premature_problems(table):
    """A table that concludes "not adopted" while an unsafe cell lies within the limit and the data calibrate to a smaller, positive limit has not concluded anything: the limit that the table
    calibrates to is to be verified at the commit that holds it (its own campaign), and that campaign decides."""
    if decide_table(table)["outcome"] != "not_adopted":
        return []
    cal, limit = calibration_of(table), table["meta"]["per_sm"] * table["meta"]["sm_count"]
    if cal["per_sm"] >= 1 and cal["first_unsafe_nh"] is not None and cal["first_unsafe_nh"] <= limit:
        return ["an unsafe cell (N * H = %d) lies within the limit of %d CTAs and the data calibrate to %d CTAs per SM, not to the %d of the tree: verify the calibrated limit at the commit that "
                "holds it before concluding that the single launch is not adopted" % (cal["first_unsafe_nh"], limit, cal["per_sm"], table["meta"]["per_sm"])]
    return []


def final_problems(table, constants=None):
    """Why a table is not the final record of the A/B: the rules of ``table_problems`` for the limit that it was measured at (all of them when it is adopted, the rules that do not tie it to a
    shipped limit otherwise), an outcome that is undecided, and a "not adopted" that a smaller limit has not been tried for (``premature_problems``)."""
    problems = structure_problems(table)
    if problems:
        return problems
    decision = decide_table(table)
    problems = table_problems(table, table["meta"]["per_sm"], constants, limit_rules=decision["outcome"] == "adopted")
    if decision["outcome"] == "undecided":
        problems.append("the outcome is undecided, the evidence has to be repaired first (%s)" % reason_text(decision, 2))
    return problems + premature_problems(table)


def plan_markers(text):
    """The plan identifiers and loop paths in a text that goes into a tracked document (the checker's own expression)."""
    import record_checker as rc                                           # imported here: the rest of this module does not need the checker
    found = rc.PLAN_VOCABULARY.search(text)
    loop = rc.UNTRACKED_OK.split()[-1].rstrip("/")
    return [found.group(0)] if found else ([loop] if loop in text else [])


def check_problems(table, ev=None, document_text=None, row=None, constants=None, results=None):
    """Everything that is wrong with a table: the evidence it rests on (``evidence_problems``), a difference from the table that the evidence gives (the figures, the calibration, the
    decision), ``final_problems``, a plan identifier in it, a section of the performance document that is not the generated one and a ledger row that is not the row of the table. Without
    ``ev`` the rules are applied to the table alone: its decision must be the one that its figures give. ``results``: the comparator results of ``ev`` when they are at hand already."""
    if not isinstance(table, dict):
        return ["the table is not an object"]
    given = json.loads(json.dumps(table))
    problems = structure_problems(given)
    if problems:
        return problems
    want = given
    try:
        if ev is not None:
            problems = evidence_problems(ev)
            if problems:
                return problems
            want = build_table(ev, results if results is not None else recompute(ev))
            problems = diff(want, given)
            if problems:
                return problems                                               # everything below reads the derived table: a table that differs from it is a problem already
        elif decide_table(given) != given["decision"] or calibration_of(given) != given["calibration"]:
            problems.append("the decision or the calibration of the table is not the one that its figures give")
        problems += final_problems(want, constants)
        markers = plan_markers(json.dumps(want, sort_keys=True))
        if markers:
            problems.append("the table carries a plan identifier or a path of the loop (%s)" % markers[0])
        if document_text is not None:
            problems += document_problems(want, document_text)
        if row is not None:
            problems += row_problems(row, want)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as e:     # a table whose parts do not fit each other is a problem of the table, not a crash
        problems.append("the table is malformed: %s: %s" % (type(e).__name__, e))
    return problems


# ---- the section of the performance document ---------------------------------------------------------------------------------------------------
def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return "%d" % v
    if isinstance(v, float):
        return "%.3f" % v
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x) for x in v) if v else "none"
    return str(v)


def _table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + ["| " + " | ".join(_fmt(c) for c in r) + " |" for r in rows]


def _change(e):
    """The change of the median of one cell and metric in percent with its flags: F the cell FAILs, n it is noise-limited, r it was judged on the re-run, ? it still needs its re-run."""
    if e is None:
        return None
    flags = "".join(" " + f for f, on in (("F", e["status"] == "FAIL"), ("n", e["noise_limited"]), ("r", e["judged_on"] == "rerun"), ("?", e["needs_rerun"])) if on)
    return "%+.2f%s" % (e["delta_pct"], flags)


def _cell_rows(rows):
    out = []
    for r in rows:
        m = r["metrics"]
        out.append([r["N"], r["H"], r["nh"], r["regime"], r["pos"], r["route"],
                    m["gpu_graph_us"]["base_median"] if "gpu_graph_us" in m else None, m["gpu_graph_us"]["cand_median"] if "gpu_graph_us" in m else None,
                    m["gpu_eager_us"]["base_median"] if "gpu_eager_us" in m else None, m["gpu_eager_us"]["cand_median"] if "gpu_eager_us" in m else None] + [_change(m.get(x)) for x in GATE_METRICS])
    return out


def render_section(table, decision, number=SECTION_NUMBER):
    """The text of the section of the performance document: a pure function of the table and its decision (the document must equal it, so that every number of the document is a number of the table)."""
    meta, rows, cal = table["meta"], table["rows"], table["calibration"]
    sms, per_sm = meta["sm_count"], meta["per_sm"]
    out = ["## %d. %s" % (number, SECTION_TITLE), "",
           "Generated from the table of the single-launch decode A/B by `benchmarks/single_launch_record.py`; every number below is a number of the table, and the table is derived from the raw record files "
           "(it is checked by deriving it again).", "",
           "### Outcome", "", "Single-launch decode is **%s**." % decision["outcome"].replace("_", " "), ""]
    out += ["- " + s for s in decision["reasons"]] + [""]
    out += ["### What was measured", "",
            "- Candidate commit %s. Side A is the candidate tree at the explicit two-launch path (bit 64), side B the same tree at its default routing on every cell of the campaign (arm default) and with the "
            "calibration control on for the cells that the default rule does not route (arm forced), so that the single launch is also timed beyond the limit." % meta["candidate"][:12],
            "- Limit of the tree: %d CTAs per SM on %d SMs, N * H at most %d. Compute capability of the devices: %s." % (
                per_sm, sms, per_sm * sms, ", ".join("%s %s" % (d, meta["devices"][d]["cc"]) for d in sorted(meta["devices"], key=device_order) if meta["devices"][d].get("gating", d in GATING))),
            "- Regress gate: a cell fails when B is slower than A by more than %.2f %% of the median (%s mode), at least %d interleaved record files per side in the %s start order (blocks of four that are "
            "ABBA or BAAB), every trial's launch mapping asserted against the routing specification." % (meta["tol_pct"], meta["mode"], meta["min_repeats"], meta["order"]),
            "- A cell whose spread is above 0.3 % is re-run once with three times the calls per trial; one that stays above it is noise-limited and judged with the tolerance widened by its spread. "
            "A cell that FAILs in a gating metric on either gating device, noise-limited or not, is unsafe.",
            "- Gating metrics: %s. The evidence of a gain is a stable cell judged on the re-run in %s within the limit, on both devices; the kernel sum and the host times gate and are never that evidence." % (
                ", ".join(meta["gate_metrics"]), " or ".join(meta["gain_metrics"])), ""]
    for dev in sorted(meta["devices"], key=device_order):
        if not meta["devices"][dev].get("gating", dev in GATING):
            out += ["- %s (%s, %d SMs, compute capability %s) is recorded and does not gate." % (dev, meta["devices"][dev]["gpu"], meta["devices"][dev]["sm_count"], meta["devices"][dev]["cc"])]
    out += ["- Not run: %s: %s." % (dev, why) for dev, why in sorted(meta["not_run"].items())]
    out += [""]
    out += ["### Comparisons", ""]
    out += _table(["device", "arm", "metric", "verdict", "cells", "noise-limited", "judged on the re-run", "still need a re-run"],
                  [[dev, arm, metric, s["verdict"], s["cells"], s["noise_limited"], s["rerun"], s["needs_rerun"]]
                   for dev in sorted(table["comparisons"], key=device_order) for arm in ARMS for metric in GATE_METRICS for s in [table["comparisons"][dev].get(arm, {}).get(metric)] if s]) + [""]
    out += ["### Calibration of the limit", "",
            "Tested N * H: %s." % _fmt(cal["tested"]),
            "Unsafe cells: %s." % ("none" if not cal["unsafe"] else "%d on the two devices together (the smallest N * H is %d)" % (len(cal["unsafe"]), cal["first_unsafe_nh"])),
            "The largest tested N * H below the smallest unsafe one is %d; divided by the %d SMs and rounded down that is %d CTAs per SM. The tree's limit is %d CTAs per SM." % (
                cal["limit_nh"], sms, cal["per_sm"], per_sm), ""]
    if cal["unsafe"]:
        out += _table(["device", "arm", "N", "H", "N * H", "regime", "metrics that FAIL"], [[u["device"], u["arm"], cell_fields(u["key"])["N"], cell_fields(u["key"])["H"], u["nh"],
                                                                                          cell_fields(u["key"])["regime"], u["metrics"]] for u in cal["unsafe"]]) + [""]
    out += ["### Confirmed gains", "",
            "GPU-elapsed graph or eager cells of the default arm within the limit that are stable, judged on the re-run and faster by the tolerance or more (A minus B in microseconds per call). The draft "
            "estimated %s: a direction, not a gate." % DRAFT_ESTIMATE, ""]
    out += _table(["device", "metric", "N", "H", "N * H", "regime", "A us", "B us", "A - B us", "change %"],
                  [[g["device"], g["metric"], cell_fields(g["key"])["N"], cell_fields(g["key"])["H"], g["nh"], cell_fields(g["key"])["regime"], g["base_median"], g["cand_median"],
                    figure(g["base_median"] - g["cand_median"]), "%+.2f" % g["delta_pct"]] for g in decision["gains"]]) + [""]
    for dev in sorted(meta["devices"], key=device_order):
        for arm, what in (("default", "the default routing"), ("forced", "the control on, the cells that the default rule does not route")):
            sel = [r for r in rows if r["device"] == dev and r["arm"] == arm]
            if not sel:
                continue
            out += ["### %s, %s" % (dev, what), "",
                    "Medians of A and B in microseconds per call for the two GPU-elapsed metrics and the change of the median in percent for every gating metric; F: the cell FAILs, n: noise-limited, r: judged "
                    "on the re-run, ?: still needs its re-run. Route: the launches that the specification gives to B.", ""]
            out += _table(["N", "H", "N * H", "regime", "pos", "route", "graph A", "graph B", "eager A", "eager B"] + ["%s %%" % m.replace("_us", "").replace("_", " ") for m in GATE_METRICS],
                          _cell_rows(sel)) + [""]
    return "\n".join(out).rstrip("\n") + "\n"


def section_bounds(document_text):
    """``(first line, last line + 1)`` of the section headed ``SECTION_TITLE`` (None when there is none): up to the next heading of the same or a higher level."""
    lines = document_text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^(#+)\s*(.*)$", line)
        if m and SECTION_TITLE.lower() in m.group(2).lower():
            for j in range(i + 1, len(lines)):
                n = re.match(r"^(#+)\s", lines[j])
                if n and len(n.group(1)) <= len(m.group(1)):
                    return i, j
            return i, len(lines)
    return None


def replace_section(document_text, section):
    """The document with the section replaced by ``section`` (appended when the document has none)."""
    b = section_bounds(document_text)
    lines = document_text.splitlines()
    new = section.rstrip("\n").splitlines()
    if b is None:
        return "\n".join(lines + [""] + new) + "\n"
    return "\n".join(lines[:b[0]] + new + [""] + lines[b[1]:]).rstrip("\n") + "\n"


def document_problems(table, document_text):
    b = section_bounds(document_text)
    if b is None:
        return ["the performance document has no section headed %r" % SECTION_TITLE]
    lines = document_text.splitlines()
    m = re.match(r"^#+\s*(\d+)\.", lines[b[0]])
    number = int(m.group(1)) if m else SECTION_NUMBER
    have = "\n".join(lines[b[0]:b[1]]).rstrip("\n") + "\n"
    want = render_section(table, table["decision"], number)
    if have == want:
        return []
    h, w = have.splitlines(), want.splitlines()
    first = next((i for i, (x, y) in enumerate(zip(h, w)) if x != y), min(len(h), len(w)))
    return ["the section of the performance document is not the generated one (line %d of the section: %r, generated %r)" % (first + 1, h[first] if first < len(h) else None, w[first] if first < len(w) else None)]


# ---- command line ------------------------------------------------------------------------------------------------------------------------------
def _report(problems):
    for p in problems:
        print("PROBLEM: %s" % p)
    print("VERDICT: %s" % ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


REASON_LINES = 20                                           # reasons that ``build`` prints (the table holds them all)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")


def _ledger_row_of(obj):
    """The ``single_launch`` row of a file that holds a row, or a ledger with steps."""
    if isinstance(obj.get("steps"), list):
        rows = [s for s in obj["steps"] if isinstance(s, dict) and s.get("id") == LEDGER_STEP]
        if len(rows) != 1:
            raise ValueError("the ledger has %d rows of %s, one is needed" % (len(rows), LEDGER_STEP))
        return rows[0]
    return obj


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("build")
    p.add_argument("--evidence", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("render")
    p.add_argument("--table", required=True)
    p.add_argument("--document", required=True)
    p.add_argument("--number", type=int, default=SECTION_NUMBER)
    p = sub.add_parser("check")
    p.add_argument("--table", required=True)
    p.add_argument("--evidence", default=None)
    p.add_argument("--document", default=None)
    p.add_argument("--row", default=None)
    p.add_argument("--repo", default=REPO, help="the source tree whose policy header holds the limit constant (default: the tree of this tool)")
    p = sub.add_parser("row")
    p.add_argument("--table", required=True)
    p.add_argument("--markers", nargs="+", required=True)
    p.add_argument("--codegen-gate", required=True)
    p.add_argument("--resource-gate", required=True)
    p.add_argument("--revert-commit", default=None)
    p.add_argument("--commit", default=None)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "build":
            ev = load_evidence(a.evidence)
            problems = evidence_problems(ev)
            if problems:
                return _report(problems)
            table = build_table(ev, recompute(ev))
            _write_json(a.out, table)
            print("wrote %s (%d bytes)" % (a.out, os.path.getsize(a.out)))
            cal = table["calibration"]
            print("OUTCOME: %s" % table["decision"]["outcome"])
            print("CALIBRATED CTAs PER SM: %d (the tree has %d)" % (cal["per_sm"], table["meta"]["per_sm"]))
            for reason in table["decision"]["reasons"][:REASON_LINES]:
                print("REASON: %s" % reason)
            return _report(final_problems(table))
        if a.cmd == "render":
            table = load_json(a.table)
            doc = _read(a.document)
            with open(a.document, "w", encoding="utf-8") as f:
                f.write(replace_section(doc, render_section(table, table["decision"], a.number)))
            return 0
        if a.cmd == "check":
            table = load_json(a.table)
            ev = load_evidence(a.evidence) if a.evidence else None
            doc = _read(a.document) if a.document else None
            row = _ledger_row_of(load_json(a.row)) if a.row else None
            constants = {"header": dp.header_per_sm(a.repo), "specification": dp.MAX_CTAS_PER_SM}
            problems = check_problems(table, ev, doc, row, constants)
            print("OUTCOME: %s" % (table.get("decision") or {}).get("outcome"))
            return _report(problems)
        if a.cmd == "row":
            table = load_json(a.table)
            row = ledger_row(table, table["decision"], a.commit or table["meta"]["candidate"], a.markers, load_json(a.codegen_gate), load_json(a.resource_gate), a.revert_commit)
            _write_json(a.out, row)
            print("wrote %s: outcome %s" % (a.out, row["outcome"]))
            return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as e:
        print("unusable input: %s" % e, file=sys.stderr)
        return 2
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
