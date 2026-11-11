"""Decode families of the row engine on grids beyond 2.5 resident waves, bit for bit against the frozen reference (``_impl=3``).

The tests that could pass before only reached grids that fit in one or two waves of resident CTAs, one kind of state and one kind of call. These families reach the conditions where a change
of the row engine could break and a small test would not notice (a flag, a spin or a schedule that works while every CTA is resident; a pool slot that is permuted, unused or padding; an open
chunk at every position, the chunk end included; rows that follow a chunk end inside one launch):

* the grid is sized from the device: ``helpers.decode_sequences`` = SMs times the CTAs per SM that the occupancy API reports (the hidden binding), so the *live* CTAs (not the padding or
  empty entries, which exit at once) fill at least 2.5 waves; the numbers are asserted, not assumed;
* mode 1 with a fixed length (one row = decode, five rows = a multi-row launch that runs over the chunk end) and with ``cu_seqlens`` (lengths 1 .. 8, empty sequences);
* a compact state and a pool with permuted, non-contiguous slots, padding entries (slot -1) and slots that no sequence uses; a warm state whose open chunks sit at every kind of position
  (made by the reference kernels, so that both arms start from the oracle's own state), carried through three successive launches (positions advance and wrap);
* after every launch the optimised arm is checked on its own (positions stay in [0, 63] and advance by the rows that ran, a slot that no sequence used is untouched bit for bit, padding and
  empty rows keep their sentinel, rows that ran are finite) and against the reference (bf16 and fp32 output, E, the valid rows of the cache, pos: identical bits).

Controls that must fail, so that the comparison can fail: one flipped bit in each compared object is reported (and a flipped bit in a stale cache row, which is not part of the state, is
not); a launch from a stale state differs; the build with the sabotaged fma (``CFBI_NEG_CONTROL``) differs from the reference on these workloads; a variant that does not build fails its
test instead of skipping it.
"""
import pytest
import torch

import rows_workload as rw
from helpers import WAVES, resident_ctas, skip_or_fail
from rows_workload import Family

FAMILIES = [
    Family("decode_compact", fixed=1, pool=False, fills="mixed"),
    Family("decode_pool_permuted", fixed=1, pool=True, fills="mixed"),
    Family("decode_pool_padding", fixed=1, pool=True, pad=0.3, fills="mixed"),
    Family("decode_open_chunks", fixed=1, pool=True, pad=0.2, fills="open"),
    Family("multirow_compact", fixed=5, pool=False, fills="crossing"),
    Family("multirow_pool_padding", fixed=5, pool=True, pad=0.3, fills="crossing"),
    Family("varlen_compact", lens=[1, 2, 3, 5, 8], pool=False, empty=0.25, fills="crossing"),
    Family("varlen_pool", lens=[1, 2, 3, 5, 8], pool=True, pad=0.25, empty=0.25, fills="mixed"),
]
BY_NAME = {f.name: f for f in FAMILIES}
HEADS = [12, 3]


@pytest.fixture(autouse=True)
def _optimised_kernels_run_here(ext):
    """Below sm_90 ``_impl=0`` resolves to the reference kernels: the comparison would be a build with itself."""
    if ext.resolve_impl(rw.OPTIMISED, -1) == ext.resolve_impl(rw.REFERENCE, -1):
        skip_or_fail("the optimised kernels do not run on this device (_impl=0 resolves to the reference kernels)")


def expectations(fam, cov, ext):
    """The conditions that the family must have reached: the problems found (an empty list = every condition reached)."""
    probs = []
    cap = resident_ctas(ext)
    if cov["live_ctas"] < WAVES * cap:
        probs.append("live grid %d CTAs is below %.1f waves of %d" % (cov["live_ctas"], WAVES, cap))
    if cov["grid_ctas"] < WAVES * cap:
        probs.append("grid %d CTAs is below %.1f waves" % (cov["grid_ctas"], WAVES))
    if cov["chunk_end_rows"] == 0:
        probs.append("no row finished a chunk (position 63)")
    if fam.pool and not cov["permuted"]:
        probs.append("the slots were not permuted")
    if fam.pool and cov["pool_unused"] < rw.POOL_SPARE:
        probs.append("the pool had only %d unused slots" % cov["pool_unused"])
    if fam.pad and cov["padding"] == 0:
        probs.append("no padding entry")
    if fam.empty and cov["empty"] == 0:
        probs.append("no empty sequence")
    if fam.fixed and fam.fixed > 1 and cov["rows_after_chunk_end"] == 0:
        probs.append("no launch ran on past a chunk end")
    if fam.fixed is None and cov["rows_after_chunk_end"] == 0:
        probs.append("no launch ran on past a chunk end")
    if fam.fills == "open" and cov["open_chunk_sequences"] < cov["live_sequences"] * 0.9:
        probs.append("only %d of %d sequences continued an open chunk" % (cov["open_chunk_sequences"], cov["live_sequences"]))
    if (fam.fixed or 0) <= 1 and fam.fixed is not None and cov["max_rows"] != 1:
        probs.append("a decode family ran more than one row of a sequence")
    return probs


@pytest.mark.parametrize("H", HEADS)
@pytest.mark.parametrize("fam", FAMILIES, ids=repr)
def test_the_family_is_bitwise_the_reference_beyond_2_5_waves(ext, fam, H):
    results = rw.run_family(ext, fam, H, seed=11)
    cov = rw.coverage(results)
    assert cov["steps"] == fam.steps
    assert expectations(fam, cov, ext) == [], cov


# ---- controls: the comparison can fail ------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def two_steps(ext):
    """Two successive launches of a pool family with padding and empty sequences, both arms equal (asserted by ``run_family``)."""
    results = rw.run_family(ext, BY_NAME["varlen_pool"], 12, seed=21)
    assert len(results) >= 2
    return results


@pytest.mark.parametrize("what,field", [("out", "out"), ("out32", "out32"), ("E", "E"), ("row", "rows"), ("pos", "pos")])
def test_a_single_flipped_bit_is_reported_in_the_object_that_holds_it(two_steps, what, field):
    step, opt, ref, warm = two_steps[-1]
    assert all(v == 0 for v in rw.mismatch(opt, ref).values())
    m = rw.mismatch(rw.corrupt(opt, what), ref)
    assert m == {k: (1 if k == field else 0) for k in m}, (what, m)


def test_a_flipped_bit_in_a_stale_cache_row_is_not_part_of_the_state(two_steps):
    step, opt, ref, warm = two_steps[-1]
    assert all(v == 0 for v in rw.mismatch(rw.corrupt(opt, "stale_row"), ref).values())


def test_a_launch_from_a_stale_state_differs(ext, two_steps):
    (step1, opt1, ref1, warm1), (step2, opt2, ref2, warm2) = two_steps[0], two_steps[1]
    stale = rw.run_arm(ext, rw.OPTIMISED, warm1, step2)       # the state before the previous launch: the oracle ran from the state after it
    m = rw.mismatch(stale, ref2)
    assert sum(m.values()) > 0, m


def test_the_sabotaged_row_engine_build_differs_from_the_reference_on_these_workloads(ext):
    bad = rw.build_variant("negctl", ("CFBI_NEG_CONTROL=1",))
    fam = BY_NAME["decode_pool_padding"]
    results = rw.run_family(ext, fam, 12, seed=31)
    step, opt, ref, warm = results[0]
    sab = rw.run_arm(bad, rw.OPTIMISED, warm, step)
    m = rw.mismatch(sab, ref)
    assert m["out32"] > 0 or m["out"] > 0 or m["E"] > 0 or m["rows"] > 0, "the sabotaged build matches the reference: the workload cannot see a changed row engine (%s)" % m


def test_a_variant_that_does_not_build_fails_instead_of_skipping():
    with pytest.raises(pytest.fail.Exception) as e:
        rw.build_variant("rowsbroken", ("CFBI_ROWS_TEST_COMPILE_ERROR=1",))
    assert "does not build" in str(e.value) and "deliberate compile error" in str(e.value), str(e.value)[-500:]
