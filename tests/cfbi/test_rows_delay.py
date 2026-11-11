"""The delay variant of the row engine (``-DCFBI_ROWS_DELAY_NS``, a separate JIT build of the extension, see ``rows_workload.build_variant``): bits that do not depend on timing.

A build with the delay hook lets a chosen warp of chosen CTAs sleep (``__nanosleep``) at the delay sites of the row engine: the kernel entry before the first loads (site 0), before each
of the barriers of a row (sites 1 .. 7, barrier 7 ends the row) and right after barrier 6, inside the chunk-end phase (site 8). The host chooses sites, warp, CTAs and length at run time
(``rows_test_delay_config``) and reads back how many warps slept (``rows_test_delay_hits``), so one build covers every configuration.

* The default build has no hook: its setters raise (a test of a variant that ran on the wrong build would otherwise pass without having delayed anything).
* Without a configuration the variant sleeps nowhere and its bits are the default build's.
* For 2000 ns and 20000 ns, at every site and at all of them, with all warps, with one warp, and with one warp of every third CTA, five repetitions each on a multi-row mode-1 launch beyond
  2.5 resident waves (pool, permuted slots, padding entries, chunk ends inside the launch): every bit of both outputs, E, the valid rows and pos equals the default build's and the
  reference's. The delay is *live*: the hit counter read back equals the number of warps that must have slept at that site (rows that pass it times the warps and CTAs chosen), and the
  launch takes longer by a fraction of the serial sleeping time of its slowest CTA, which a delay that did nothing would not.
* The variant without the barrier that ends a row (``CFBI_ROWS_TEST_NO_END_BARRIER``) is a racy control: a warp that sleeps in the chunk-end phase lets the others start the next row and
  rewrite ``S.g2`` under it, and the result differs from the reference in at least one of five repetitions; the same delay on the variant that keeps the barrier changes nothing.
"""
import random
import statistics

import pytest

import rows_workload as rw
from helpers import skip_or_fail
from rows_workload import Family

DELAY = ("CFBI_ROWS_DELAY_NS=2000",)
RACY = DELAY + ("CFBI_ROWS_TEST_NO_END_BARRIER=1",)
H = 12
REPS = 5
NS = (2000, 20000)
SITES = {"entry": 0, "barrier1": 1, "barrier2": 2, "barrier3": 3, "barrier4": 4, "barrier5": 5, "barrier6": 6, "end_of_row": 7, "chunk_end_phase": 8}
ALL_SITES = sum(1 << s for s in SITES.values())
TARGETS = {"all_warps": (-1, 0, 0), "warp0": (0, 0, 0), "warp7_of_every_third_cta": (7, 3, 1)}    # warp, cta_mod, cta_rem
MIXED = Family("delay_multirow", fixed=5, pool=True, pad=0.3, fills="crossing", steps=1)
CROSSING = Family("racy_multirow", fixed=5, pool=True, pad=0.0, fills="sixty", steps=1)
WARPS = 8


@pytest.fixture(autouse=True)
def _optimised_kernels_run_here(ext):
    if ext.resolve_impl(rw.OPTIMISED, -1) == ext.resolve_impl(rw.REFERENCE, -1):
        skip_or_fail("the optimised kernels do not run on this device (_impl=0 resolves to the reference kernels)")


def make_workload(ext, fam, seed):
    """``(step, warm, reference outcome, default-build outcome)`` of one launch of the family, both arms equal (asserted)."""
    rng = random.Random(seed)
    par = rw.make_params(H, seed)
    step, S = rw.draw_step(ext, fam, H, par, rng, seed * 10)
    warm = rw.warm_pool(ext, S, H, rw.draw_fills(rng, S, fam.fills), par, seed)
    ref = rw.run_arm(ext, rw.REFERENCE, warm, step)
    opt = rw.run_arm(ext, rw.OPTIMISED, warm, step)
    assert all(v == 0 for v in rw.mismatch(opt, ref).values()), "the default build differs from the reference on the workload of the delay tests"
    return step, warm, ref, opt


@pytest.fixture(scope="module")
def workload(ext):
    return make_workload(ext, MIXED, 77)


@pytest.fixture(scope="module")
def crossing(ext):
    return make_workload(ext, CROSSING, 78)


@pytest.fixture(scope="module")
def dly(ext):
    return rw.build_variant("rowsdelay", DELAY)


@pytest.fixture(scope="module")
def racy(ext):
    return rw.build_variant("rowsracy", RACY)


def rows_through(step, warm, n, site):
    """How many rows of sequence n pass delay site ``site`` (the kernel entry is passed once by every CTA of the grid, the padding and empty ones included)."""
    if site == 0:
        return 1
    if not step.live(n):
        return 0
    pos0 = int(warm.pos[step.slot(n)])
    positions = [(pos0 + r) % rw.BT for r in range(step.lens[n])]
    if site in (1, 2, 4, 5, 7):
        return len(positions)
    if site == 3:
        return sum(1 for p in positions if p >= 16)        # the cross-sub-block barrier runs for rows beyond the first 16 positions
    if site in (6, 8):
        return sum(1 for p in positions if p == rw.BT - 1)  # the chunk-end phase
    raise ValueError(site)


def expected_hits(step, warm, mask, target):
    """The warps that must have slept: per CTA (sequence n, head h; cta = n * H + h) in the chosen set, the rows that pass each chosen site times the warps chosen."""
    warp, cta_mod, cta_rem = target
    per_cta_warps = WARPS if warp < 0 else 1
    total = 0
    for n in range(step.N):
        for h in range(step.H):
            if cta_mod and (n * step.H + h) % cta_mod != cta_rem:
                continue
            total += per_cta_warps * sum(rows_through(step, warm, n, s) for s in range(9) if mask >> s & 1)
    return total


def serial_sleeps(step, warm, mask):
    """The most sleeps that one CTA must go through one after the other: its rows that pass the chosen sites (a lower bound of the launch time is this times the length of a sleep)."""
    return max(sum(rows_through(step, warm, n, s) for s in range(9) if mask >> s & 1) for n in range(step.N))


# ---- the hook is inert where it was not compiled ----------------------------------------------------------------------------------------------------------------
def test_the_default_build_has_no_delay_hook(ext):
    with pytest.raises(RuntimeError, match="not compiled"):
        ext.rows_test_delay_config(2000, -1, 0, 0, 1)
    with pytest.raises(RuntimeError, match="not compiled"):
        ext.rows_test_delay_hits()
    with pytest.raises(RuntimeError, match="not compiled"):
        ext.rows_test_pos_offset(1)


def test_the_delay_variant_without_a_configuration_sleeps_nowhere_and_changes_no_bit(dly, workload):
    step, warm, ref, opt = workload
    dly.rows_test_delay_config(0, -1, 0, 0, 0)
    out = rw.run_arm(dly, rw.OPTIMISED, warm, step)
    assert dly.rows_test_delay_hits() == 0
    assert all(v == 0 for v in rw.mismatch(out, ref).values())


# ---- the bits do not depend on timing; the delay is live -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("ns", NS)
@pytest.mark.parametrize("site", list(SITES) + ["all_sites"])
def test_a_delay_at_a_site_changes_no_bit_and_the_hits_are_read_back(dly, workload, ns, site):
    step, warm, ref, opt = workload
    mask = ALL_SITES if site == "all_sites" else 1 << SITES[site]
    for tname, target in TARGETS.items():
        want = expected_hits(step, warm, mask, target)
        assert want > 0, (site, tname, "the configuration cannot hit anything on this workload: the test would be vacuous")
        for rep in range(REPS):
            dly.rows_test_delay_config(ns, target[0], target[1], target[2], mask)
            out = rw.run_arm(dly, rw.OPTIMISED, warm, step)
            m = rw.mismatch(out, ref)
            assert all(v == 0 for v in m.values()), "delay %d ns at %s, %s, repetition %d: %s" % (ns, site, tname, rep, m)
            assert dly.rows_test_delay_hits() == want, "delay %d ns at %s, %s: %d warps slept, %d expected" % (ns, site, tname, dly.rows_test_delay_hits(), want)


def test_the_delay_takes_time(dly, workload):
    """Liveness by timing: with a 20 us sleep at every site the launch takes longer than without by a fraction of the serial sleeping time of its slowest CTA."""
    step, warm, ref, opt = workload
    ns = 20000
    dly.rows_test_delay_config(ns, -1, 0, 0, 0)
    quiet = statistics.median(rw.time_step(dly, rw.OPTIMISED, warm, step, REPS))
    dly.rows_test_delay_config(ns, -1, 0, 0, ALL_SITES)
    slow = statistics.median(rw.time_step(dly, rw.OPTIMISED, warm, step, REPS))
    floor_ms = 0.3 * serial_sleeps(step, warm, ALL_SITES) * ns * 1e-6      # __nanosleep is approximate (it may return early): a third of the nominal serial time
    assert slow >= quiet + floor_ms, "launch time %.3f ms with delays, %.3f ms without; the slowest CTA sleeps %d times for %d ns" % (slow, quiet, serial_sleeps(step, warm, ALL_SITES), ns)


# ---- the barrier that ends a row is needed: a racy control ---------------------------------------------------------------------------------------------------------
def test_the_variant_without_the_end_of_row_barrier_differs_under_a_delay_in_the_chunk_end_phase(racy, crossing):
    step, warm, ref, opt = crossing
    assert expected_hits(step, warm, 1 << SITES["chunk_end_phase"], TARGETS["warp0"]) > 0
    diffs = []
    for rep in range(REPS):
        racy.rows_test_delay_config(20000, 7, 0, 0, 1 << SITES["chunk_end_phase"])
        out = rw.run_arm(racy, rw.OPTIMISED, warm, step)
        assert racy.rows_test_delay_hits() == expected_hits(step, warm, 1 << SITES["chunk_end_phase"], (7, 0, 0)), "the delay did not run in the racy variant"
        diffs.append(sum(rw.mismatch(out, ref).values()))
    assert any(d > 0 for d in diffs), "the variant without the end-of-row barrier matched the reference in %d of %d repetitions: the delay tests cannot see a missing barrier (%s)" % (REPS, REPS, diffs)


def test_the_same_delay_changes_nothing_where_the_barrier_is_kept(dly, crossing):
    step, warm, ref, opt = crossing
    for rep in range(REPS):
        dly.rows_test_delay_config(20000, 7, 0, 0, 1 << SITES["chunk_end_phase"])
        out = rw.run_arm(dly, rw.OPTIMISED, warm, step)
        assert dly.rows_test_delay_hits() > 0
        assert all(v == 0 for v in rw.mismatch(out, ref).values()), "repetition %d" % rep
