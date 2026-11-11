"""Record of the per-phase clock profile of the optimised row engine (``cfbi-rows-profile-v1``; plain Python, no torch).

``benchmarks/micro/rows_profile.py`` runs the profile build (``-DCFBI_ROWS_PROFILE=1``) and turns the stamp buffer of the device into one cell per shape; this module
holds the schema and the aggregation, so that the record checker and the tests read the same definitions without a GPU.

Stamps (cycles of ``clock64``, relative to the stamp taken at kernel entry): per row ``ts[0..7]`` are the stamps ``ROWS_TS(0..7)`` of ``cfbi_rows.cuh``, ``ts[8]`` is the
kernel exit (kept in the slot of the last row of a CTA) and ``ts[9]`` is the SM id + 1 (0: the slot holds no row). The seven row phases are the differences of neighbouring
stamps; the prologue is ``ts[0]`` (kernel entry to the first row), the epilogue ``ts[8] - ts[7]`` and the measured row duration ``ts[7] - ts[0]``.

Two limits of what a stamp says. The exit stamp is taken by thread 0 after its own last global store (the ``pos`` store of the fused finish): it does not wait for the stores of the
other threads or of the memory system to retire, so the kernel total is the time of the last instruction of thread 0, not the time at which the grid is complete. And ``ts[0]`` is the
time from kernel entry, so it is the prologue only for the first row of a CTA: the records are made for decode, where every CTA runs one row per call (a CTA with several rows would
carry the earlier rows in the ``ts[0]`` of the later ones).

Every mean of a group is taken over the same rows, so the phases add up to the row duration (a median of totals is not a sum of medians).
Groups: ``regular`` = the rows at positions 0 .. 62, ``chunk_end`` = the rows at position 63 (the chunk-end row), ``all`` = both. A cell is one shape (N sequences, H heads).
"""
import json

SCHEMA = "cfbi-rows-profile-v1"
PHASES = ("prep", "akk_aqk", "m_tiles", "ut_w_chain", "p1_p2", "u_o", "chunk_end_sync")   # the seven phases between ROWS_TS(0) .. ROWS_TS(7)
GROUPS = ("regular", "chunk_end", "all")
STAMPS = 10
POSITIONS = 64
CHUNK_END_POS = POSITIONS - 1
SUM_TOLERANCE = 0.05   # the phases must add up to the measured row duration within 5 %


def cell_key(N, H):
    """``rows-profile;N=16;H=12;``: every field ends with a delimiter (``N=1;`` does not match ``N=16;``)."""
    return "rows-profile;N=%d;H=%d;" % (N, H)


def row_measures(r):
    """The measures of one recorded row ``r`` (a list of ``STAMPS`` stamps), or None for a slot that holds no row."""
    if len(r) != STAMPS or not r[9]:
        return None
    return {"prologue": r[0], "phases": [r[k + 1] - r[k] for k in range(7)], "epilogue": r[8] - r[7], "row_total": r[7] - r[0], "kernel_total": r[8], "sm": r[9] - 1}


def _mean_block(rows):
    """Means of a list of ``row_measures`` over the same rows, with the phase shares of the row duration."""
    n = len(rows)
    if n == 0:
        return {"rows": 0}
    mean = lambda f: sum(f(x) for x in rows) / float(n)
    out = {"rows": n, "prologue": mean(lambda x: x["prologue"]), "phases": dict((name, mean(lambda x, k=k: x["phases"][k])) for k, name in enumerate(PHASES)),
           "epilogue": mean(lambda x: x["epilogue"]), "row_total": mean(lambda x: x["row_total"]), "kernel_total": mean(lambda x: x["kernel_total"])}
    out["shares"] = dict((name, out["phases"][name] / out["row_total"] if out["row_total"] else None) for name in PHASES)
    totals = sorted(x["row_total"] for x in rows)
    out["row_total_p50"], out["row_total_p95"], out["row_total_max"] = totals[n // 2], totals[min(n - 1, int(0.95 * n))], totals[-1]
    return out


def cell_from_slots(slots, N, H):
    """The cell of one shape from ``slots[position][cta]`` = the ``STAMPS`` stamps of that row (``cta = n * H + h``); ``slots`` has ``POSITIONS`` entries, each at least
    ``N * H`` CTAs long. A slot without a row (``ts[9] == 0``) is not counted: ``rows_recorded`` against ``rows_expected`` says whether the profile is complete."""
    ctas = N * H
    if len(slots) != POSITIONS or any(len(p) < ctas for p in slots):
        raise ValueError("the stamp buffer must hold %d positions of at least %d CTAs" % (POSITIONS, ctas))
    per_position, groups = [], {"regular": [], "chunk_end": [], "all": []}
    for pos in range(POSITIONS):
        rows = [m for m in (row_measures(r) for r in slots[pos][:ctas]) if m is not None]
        groups["chunk_end" if pos == CHUNK_END_POS else "regular"].extend(rows)
        groups["all"].extend(rows)
        block = _mean_block(rows)
        block["pos"] = pos
        per_position.append(block)
    cell = {"key": cell_key(N, H), "N": N, "H": H, "ctas": ctas, "rows_expected": POSITIONS * ctas, "rows_recorded": len(groups["all"]),
            "sms": sorted(set(m["sm"] for m in groups["all"])), "rows": dict((g, _mean_block(groups[g])) for g in GROUPS), "per_position": per_position}
    return cell


def build_record(meta, cells):
    return {"schema": SCHEMA, "meta": meta, "cells": cells}


def write_record(path, meta, cells):
    with open(path, "w") as f:
        json.dump(build_record(meta, cells), f, indent=1, sort_keys=True)
        f.write("\n")


def load_record(path):
    with open(path) as f:
        rec = json.load(f)
    if not isinstance(rec, dict):
        raise ValueError("%s: a profile record must be a JSON object" % path)
    rec["_path"] = str(path)
    return rec
