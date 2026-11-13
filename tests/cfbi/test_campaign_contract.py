"""The Method M contract of the raw records of a final campaign (``record_checker.campaign_contract_problems``) for the cells of the caller, and the eviction buffer of the cold regime.

Plain Python, no GPU and no torch (``pytest --noconftest tests/cfbi/test_campaign_contract.py``). The records are the producer-like synthetic ones of the single-launch table tests (the fields,
the samples and the cells that the decode benchmark writes). The comparator takes pooled records and records of any workload, on purpose, so what a campaign has to be is checked here
before it judges anything: process-median samples, at least five trials that the record declares and every cell holds, the cells that the caller's campaign defines, each measuring the
workload its key names, positive effective counts, and, for a record with cold cells, an eviction buffer of at least twice the L2 cache.
"""
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "benchmarks", _ROOT / "tests" / "cfbi"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bench_decode as bd  # noqa: E402
import record_checker as rc  # noqa: E402
import test_single_launch_record as tsl  # noqa: E402
from test_decode_boundary_table import K_BIG, K_COLD, K_SMALL, K_WITHIN, PER_SM, SMS, has, key_of, mutated  # noqa: E402

CELLS = {"default": [K_SMALL, K_WITHIN, K_COLD, key_of(1, 12, "probe", 63)], "forced": [K_BIG]}      # K_WITHIN (98 sequences, 12 heads) is a boundary cell: not a cell of the default matrix
L2 = 1 << 26                                                                                          # what the synthetic records declare as the L2 cache; they evict twice that and a megabyte


@pytest.fixture(scope="module")
def evidence():
    return tsl.make_ev(cells=CELLS, devices=("b200",))


def a_record(ev, arm="default"):
    return ev["devices"]["b200"][arm]["first"]["A"][0]


CAMPAIGN = rc.spec_workloads(bd.campaign_cell_specs(PER_SM, SMS))


def problems(rec, **kw):
    return rc.campaign_contract_problems([rec], "decode", "b200 default first A", workloads=kw.get("workloads", CAMPAIGN))


# ---- the cells are those of the caller -------------------------------------------------------------------------------------------------------------------
def test_a_record_of_the_campaign_passes_with_the_cells_of_the_campaign_and_not_with_the_default_matrix(evidence):
    rec = a_record(evidence)
    assert problems(rec) == []
    default = rc.campaign_contract_problems([rec], "decode", "b200 default first A")                   # no workloads: the cells of the benchmark's own matrix, as before
    assert has(default, "the cell %r is not a cell of the decode benchmark" % K_WITHIN) and len(default) == 1
    assert set(rc.spec_workloads(bd.cell_specs())) == set(rc.canonical_workloads("decode")) and K_WITHIN not in rc.canonical_workloads("decode")
    assert set(CAMPAIGN) > set(rc.canonical_workloads("decode"))                                       # the campaign's cells are the matrix and the boundary cells


def test_the_workloads_of_a_cell_are_those_that_its_key_names():
    spec = rc.spec_workloads([{"key": K_COLD, "N": 64, "H": 12, "regime": "cold", "pos": None}, {"key": key_of(1, 12, "probe", 63), "N": 1, "H": 12, "regime": "probe", "pos": 63}])
    assert spec == {K_COLD: {"N": 64, "H": 12, "regime": "cold", "pos": None}, key_of(1, 12, "probe", 63): {"N": 1, "H": 12, "regime": "probe", "pos": 63}}


@pytest.mark.parametrize("name,edit,expected", [
    ("a cell that is no cell of the campaign", lambda r: r["cells"][0].update(key=key_of(3, 7)), "the cell %r is not a cell of the decode benchmark" % key_of(3, 7)),
    ("a cold key over a lock-step cell", lambda r: r["cells"][2]["args"].update(regime="lockstep"), "does not measure the workload its key names (regime"),
    ("a probe key at another position", lambda r: r["cells"][3]["args"].update(pos=0), "does not measure the workload its key names (pos"),
    ("a probe key without a position", lambda r: r["cells"][3]["args"].update(pos=None), "does not measure the workload its key names (pos"),
    ("another number of sequences", lambda r: r["cells"][1]["args"].update(N=99), "does not measure the workload its key names (N"),
    ("a float for the heads", lambda r: r["cells"][1]["args"].update(H=12.0), "does not measure the workload its key names (H"),
    ("a cell that is not a decode cell", lambda r: r["cells"][0].update(kind="chain"), "is not a decode cell (kind 'chain')"),
    ("no effective calls", lambda r: r["cells"][0]["args"].pop("calls_per_trial"), "an effective count must be a positive integer"),
    ("zero profiled calls", lambda r: r["cells"][1]["args"].update(profile_calls=0), "has profile_calls = 0"),
])
def test_a_cell_that_is_not_the_workload_of_its_key_is_rejected(evidence, name, edit, expected):
    assert has(problems(mutated(a_record(evidence), edit)), expected), name


# ---- Method M ------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,edit,expected", [
    ("pooled samples", lambda r: r["meta"].update(sample_stat="pooled"), "has sample_stat 'pooled', not 'process_median'"),
    ("no sample statistic", lambda r: r["meta"].pop("sample_stat"), "has sample_stat"),
    ("one trial", lambda r: r["meta"].update(trials=1), "declares 1 trials per cell: Method M needs at least 5 trials"),
    ("four trials", lambda r: r["meta"].update(trials=4), "declares 4 trials per cell"),
    ("trials that are no integer", lambda r: r["meta"].update(trials=5.0), "declares 5.0 trials per cell"),
    ("a cell with fewer samples than the record declares", lambda r: r["cells"][1].update(samples=r["cells"][1]["samples"][:3]), "declares 5 trials per cell, the cell %r holds 3 samples" % K_WITHIN),
    ("another kind of record", lambda r: r["meta"].update(kind="chain-wall"), "is not a decode-wall record (meta kind 'chain-wall')"),
])
def test_a_record_that_is_not_a_method_m_record_is_rejected(evidence, name, edit, expected):
    assert has(problems(mutated(a_record(evidence), edit)), expected), name


# ---- the eviction buffer of the cold regime -------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,meta,expected", [
    ("no eviction at all", dict(eviction_bytes=0), "an eviction buffer of 0 bytes, less than twice the L2 cache (%d bytes)" % L2),
    ("a buffer of the size of the L2", dict(eviction_bytes=L2), "an eviction buffer of %d bytes, less than twice the L2 cache" % L2),
    ("one byte short of twice the L2", dict(eviction_bytes=2 * L2 - 1), "less than twice the L2 cache"),
    ("no eviction field", dict(eviction_bytes=None), "an eviction buffer of None bytes"),
    ("an eviction size that is a string", dict(eviction_bytes=str(2 * L2)), "an eviction buffer of '%d' bytes" % (2 * L2)),
    ("an eviction size that is a float", dict(eviction_bytes=float(2 * L2)), "an eviction buffer of %r bytes" % float(2 * L2)),
    ("an eviction size that is a boolean", dict(eviction_bytes=True), "an eviction buffer of True bytes"),
    ("no L2 size", dict(l2_cache_bytes=None), "declares an L2 cache of None bytes"),
    ("an L2 size of zero", dict(l2_cache_bytes=0), "declares an L2 cache of 0 bytes"),
    ("an L2 size that is a boolean", dict(l2_cache_bytes=True), "declares an L2 cache of True bytes"),
    ("a negative L2 size", dict(l2_cache_bytes=-L2), "declares an L2 cache of %d bytes" % -L2),
])
def test_a_record_with_cold_cells_needs_an_eviction_buffer_of_twice_the_l2_cache(evidence, name, meta, expected):
    rec = mutated(a_record(evidence), lambda r: r["meta"].update(meta))
    assert any(c["key"] == K_COLD for c in rec["cells"])
    assert has(problems(rec), expected), name


def test_exactly_twice_the_l2_cache_is_enough_and_a_record_without_cold_cells_needs_no_buffer(evidence):
    assert problems(mutated(a_record(evidence), lambda r: r["meta"].update(eviction_bytes=2 * L2))) == []
    without_cold = mutated(a_record(evidence), lambda r: r.update(cells=[c for c in r["cells"] if c["key"] != K_COLD]))
    assert not any("cold" in c["key"] for c in without_cold["cells"])
    assert problems(mutated(without_cold, lambda r: r["meta"].update(eviction_bytes=0, l2_cache_bytes=None))) == []        # no cold cell: no cache was evicted, none has to be


def test_a_cell_that_names_the_cold_regime_in_one_place_only_still_needs_the_buffer(evidence):
    """The record says cold in its key or in its arguments: either is enough for the rule, so a record does not escape it by lying in the other place."""
    by_key = mutated(a_record(evidence), lambda r: (r["cells"][2]["args"].update(regime="lockstep"), r["meta"].update(eviction_bytes=0)))
    by_args = mutated(a_record(evidence), lambda r: (r["cells"][0].update(key=K_WITHIN), r["cells"][0]["args"].update(regime="cold"), r["cells"][2]["args"].update(regime="lockstep"),
                                                     r["meta"].update(eviction_bytes=0)))
    assert has(problems(by_key), "an eviction buffer of 0 bytes") and has(problems(by_args), "an eviction buffer of 0 bytes")


def test_the_eviction_rule_is_for_decode_records_only():
    """A chain record has no cold regime (its cells name one regime each and the layer-chain benchmark evicts nothing): the rule is not applied to it."""
    rec = {"meta": {"kind": "chain-wall", "sample_stat": "process_median", "trials": 5, "eviction_bytes": 0, "l2_cache_bytes": None},
           "cells": [{"key": "chain;N=1;H=12;Dm=2048;regime=cold;", "kind": "chain", "args": {"N": 1, "H": 12, "Dm": 2048, "regime": "cold", "calls_per_trial": 8, "profile_calls": 8}, "samples": [{}] * 5}]}
    got = rc.campaign_contract_problems([rec], "chain", "chain")
    assert not has(got, "eviction") and not has(got, "L2")
