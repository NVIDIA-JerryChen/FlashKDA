"""The delay variant of the row engine (``-DCFBI_ROWS_DELAY_NS``, a separate JIT build of the extension, see ``rows_workload.build_variant``): bits that do not depend on timing.

A build with the delay hook lets a chosen warp of chosen CTAs sleep (``__nanosleep``) at the delay sites of the row engine: the kernel entry before the first loads (site 0), before each
of the barriers of a row (sites 1 .. 7, barrier 7 ends the row) and right after barrier 6, inside the chunk-end phase (site 8); the staged A fragments add site 13 (before barrier 13, which publishes
the AB tile) and site 14 (in the window of that barrier, in front of the stores of the AB producers: only the warps 0 .. 3, the threads below HEAD_K, pass it). The host chooses sites, warp, CTAs and length at run time
(``rows_test_delay_config``) and reads back how many warps slept (``rows_test_delay_hits``), so one build covers every configuration.

* The default build has no hook: its setters raise (a test of a variant that ran on the wrong build would otherwise pass without having delayed anything).
* Without a configuration the variant sleeps nowhere and its bits are the default build's.
* For 2000 ns and 20000 ns, at every site and at all of them, with all warps, with one warp, and with one warp of every third CTA, five repetitions each on a multi-row mode-1 launch and on
  the single-row decode launch, both beyond 2.5 resident waves (pool, permuted slots, padding entries, chunk ends) of the row-engine instantiation that the path launches, and on BOTH paths of the
  decode, each named and asserted (the single launch: one ``rows_kernel<true>``, ``_impl=0`` with the hidden control on; the two launches: the plan kernel and ``rows_kernel<false>``, bit 64): every
  bit of both outputs, E, the valid rows and pos equals the default build's and the reference's. The delay is *live*: the hit counter read back equals the number of warps that must have slept at that site (rows that pass it times the warps and CTAs chosen), and the
  launch takes longer by a fraction of the serial sleeping time of its slowest CTA, which a delay that did nothing would not.
* The variant without the barrier that ends a row (``CFBI_ROWS_TEST_NO_END_BARRIER``) is a racy control: a warp that sleeps in the chunk-end phase lets the others start the next row and
  rewrite ``S.g2`` under it, and the result differs from the reference in at least one of five repetitions; the same delay on the variant that keeps the barrier changes nothing.

A delay that never ends the sleeping warp's wait would hang the test, and a test cannot interrupt a wait on the device from inside its own process: every group of cases that launches
kernels (the sites of one delay length, the liveness / timing / racy cases) therefore runs in a child process under ``GROUP_TIMEOUT_S`` (``rows_bounded``): a hang fails the test and kills
the child's process group. Every case keeps its id, its delay lengths, its five repetitions and its assertions (the group functions return the outcome of each case; the tests assert the
one they are named after); the group size, duration and limit are recorded.
"""
import random
import statistics

import pytest

import rows_bounded as rb
import rows_workload as rw
from helpers import skip_or_fail
from rows_workload import Family

PATHS = {"single": rw.SINGLE, "two_launch": rw.TWO_LAUNCH}
DELAY = ("CFBI_ROWS_DELAY_NS=2000",)
RACY = DELAY + ("CFBI_ROWS_TEST_NO_END_BARRIER=1",)
H = 12
REPS = 5
NS = (2000, 20000)
SITES = {"entry": 0, "barrier1": 1, "barrier2": 2, "barrier3": 3, "barrier4": 4, "barrier5": 5, "barrier6": 6, "end_of_row": 7, "chunk_end_phase": 8, "barrier13": 13, "ab_window": 14}
ALL_SITES = sum(1 << s for s in SITES.values())
TARGETS = {"all_warps": (-1, 0, 0), "warp0": (0, 0, 0), "warp7_of_every_third_cta": (7, 3, 1)}    # warp, cta_mod, cta_rem
SITE_WARPS = {14: 4}      # the sites that only the first warps pass: site 14 is inside ``if (tid < HEAD_K)`` (the AB producers), the other sites are passed by all WARPS warps
# a configuration that names a warp that never passes the site would be vacuous: the warp that stands in for warp 7 there is the last one that does
SITE_TARGETS = {site: dict(TARGETS, warp7_of_every_third_cta=(min(7, w - 1), 3, 1)) for site, w in SITE_WARPS.items()}
MIXED = Family("delay_multirow", fixed=5, pool=True, pad=0.3, fills="crossing", steps=1)
DECODE = Family("delay_decode", fixed=1, pool=True, pad=0.3, fills="mixed", steps=1)         # one row per sequence: the production decode launch
CROSSING = Family("racy_multirow", fixed=5, pool=True, pad=0.0, fills="sixty", steps=1)
WARPS = 8
GROUP_TIMEOUT_S = 1200    # per group: the sites of one delay length take about 25 s on a B200 per path (a first run adds the JIT builds of the variants); a hang ends here


@pytest.fixture(autouse=True)
def _optimised_kernels_run_here(ext):
    if ext.resolve_impl(rw.OPTIMISED, -1) == ext.resolve_impl(rw.REFERENCE, -1):
        skip_or_fail("the optimised kernels do not run on this device (_impl=0 resolves to the reference kernels)")


def on_path(mod, path):
    """The single-launch control of ``mod`` as the path needs it (on for the single launch; bit 64 asks for the two launches whatever it says)."""
    mod.decode_single_force(path is rw.SINGLE)


def make_workload(ext, fam, seed, path):
    """``(step, warm, reference outcome, default-build outcome)`` of one launch of the family on ``path``, both arms equal (asserted), the grid sized from the instantiation that the path launches."""
    rng = random.Random(seed)
    par = rw.make_params(H, seed)
    on_path(ext, path)
    step, S = rw.draw_step(ext, fam, H, par, rng, seed * 10, instance=path.instance)
    warm = rw.warm_pool(ext, S, H, rw.draw_fills(rng, S, fam.fills), par, seed)
    ref = rw.run_arm(ext, rw.ORACLE, warm, step)
    opt = rw.run_arm(ext, path, warm, step)
    assert all(v == 0 for v in rw.mismatch(opt, ref).values()), "the default build differs from the reference on the workload of the delay tests"
    return step, warm, ref, opt


def rows_through(step, warm, n, site):
    """How many rows of sequence n pass delay site ``site`` (the kernel entry is passed once by every CTA of the grid, the padding and empty ones included)."""
    if site == 0:
        return 1
    if not step.live(n):
        return 0
    pos0 = int(warm.pos[step.slot(n)])
    positions = [(pos0 + r) % rw.BT for r in range(step.lens[n])]
    if site in (1, 2, 4, 5, 7, 13, 14):
        return len(positions)
    if site == 3:
        return sum(1 for p in positions if p >= 16)        # the cross-sub-block barrier runs for rows beyond the first 16 positions
    if site in (6, 8):
        return sum(1 for p in positions if p == rw.BT - 1)  # the chunk-end phase
    raise ValueError(site)


def warps_at(site, warp):
    """How many warps of a CTA that pass ``site`` sleep when the configuration names ``warp`` (-1: every warp): the chosen warp counts only if the site is passed by it."""
    passing = SITE_WARPS.get(site, WARPS)
    return passing if warp < 0 else int(warp < passing)


def expected_hits(step, warm, mask, target):
    """The warps that must have slept: per CTA (sequence n, head h; cta = n * H + h) in the chosen set, the rows that pass each chosen site times the warps chosen (that pass it)."""
    warp, cta_mod, cta_rem = target
    total = 0
    for n in range(step.N):
        for h in range(step.H):
            if cta_mod and (n * step.H + h) % cta_mod != cta_rem:
                continue
            total += sum(warps_at(s, warp) * rows_through(step, warm, n, s) for s in SITES.values() if mask >> s & 1)
    return total


def serial_sleeps(step, warm, mask):
    """The most sleeps that one CTA must go through one after the other: its rows that pass the chosen sites (a lower bound of the launch time is this times the length of a sleep)."""
    return max(sum(rows_through(step, warm, n, s) for s in SITES.values() if mask >> s & 1) for n in range(step.N))


# ---- the hook is inert where it was not compiled ----------------------------------------------------------------------------------------------------------------
def test_the_default_build_has_no_delay_hook(ext):
    with pytest.raises(RuntimeError, match="not compiled"):
        ext.rows_test_delay_config(2000, -1, 0, 0, 1)
    with pytest.raises(RuntimeError, match="not compiled"):
        ext.rows_test_delay_hits()
    with pytest.raises(RuntimeError, match="not compiled"):
        ext.rows_test_pos_offset(1)


# ---- the cases (they run in the child process of their group) -------------------------------------------------------------------------------------------
def no_configuration_case(dly, workload, path):
    step, warm, ref, opt = workload
    on_path(dly, path)
    dly.rows_test_delay_config(0, -1, 0, 0, 0)
    out = rw.run_arm(dly, path, warm, step)
    assert dly.rows_test_delay_hits() == 0
    assert all(v == 0 for v in rw.mismatch(out, ref).values())


def site_case(dly, shaped, ns, site, path):
    """The bits do not depend on timing; the delay is live (its hits are read back)."""
    step, warm, ref, opt = shaped
    on_path(dly, path)
    mask = ALL_SITES if site == "all_sites" else 1 << SITES[site]
    for tname, target in SITE_TARGETS.get(SITES.get(site), TARGETS).items():
        want = expected_hits(step, warm, mask, target)
        assert want > 0, (site, tname, "the configuration cannot hit anything on this workload: the test would be vacuous")
        for rep in range(REPS):
            dly.rows_test_delay_config(ns, target[0], target[1], target[2], mask)
            out = rw.run_arm(dly, path, warm, step)
            m = rw.mismatch(out, ref)
            assert all(v == 0 for v in m.values()), "delay %d ns at %s, %s, repetition %d: %s" % (ns, site, tname, rep, m)
            assert dly.rows_test_delay_hits() == want, "delay %d ns at %s, %s: %d warps slept, %d expected" % (ns, site, tname, dly.rows_test_delay_hits(), want)


def timing_case(dly, workload, path):
    """Liveness by timing: with a 20 us sleep at every site the launch takes longer than without by a fraction of the serial sleeping time of its slowest CTA."""
    step, warm, ref, opt = workload
    on_path(dly, path)
    ns = 20000
    dly.rows_test_delay_config(ns, -1, 0, 0, 0)
    quiet = statistics.median(rw.time_step(dly, path.impl, warm, step, REPS))
    dly.rows_test_delay_config(ns, -1, 0, 0, ALL_SITES)
    slow = statistics.median(rw.time_step(dly, path.impl, warm, step, REPS))
    floor_ms = 0.3 * serial_sleeps(step, warm, ALL_SITES) * ns * 1e-6      # __nanosleep is approximate (it may return early): a third of the nominal serial time
    assert slow >= quiet + floor_ms, "launch time %.3f ms with delays, %.3f ms without; the slowest CTA sleeps %d times for %d ns" % (slow, quiet, serial_sleeps(step, warm, ALL_SITES), ns)


def racy_case(racy, crossing, path):
    step, warm, ref, opt = crossing
    on_path(racy, path)
    assert expected_hits(step, warm, 1 << SITES["chunk_end_phase"], TARGETS["warp0"]) > 0
    diffs = []
    for rep in range(REPS):
        racy.rows_test_delay_config(20000, 7, 0, 0, 1 << SITES["chunk_end_phase"])
        out = rw.run_arm(racy, path, warm, step)
        assert racy.rows_test_delay_hits() == expected_hits(step, warm, 1 << SITES["chunk_end_phase"], (7, 0, 0)), "the delay did not run in the racy variant"
        diffs.append(sum(rw.mismatch(out, ref).values()))
    assert any(d > 0 for d in diffs), "the variant without the end-of-row barrier matched the reference in %d of %d repetitions: the delay tests cannot see a missing barrier (%s)" % (REPS, REPS, diffs)


def barrier_kept_case(dly, crossing, path):
    step, warm, ref, opt = crossing
    on_path(dly, path)
    for rep in range(REPS):
        dly.rows_test_delay_config(20000, 7, 0, 0, 1 << SITES["chunk_end_phase"])
        out = rw.run_arm(dly, path, warm, step)
        assert dly.rows_test_delay_hits() > 0
        assert all(v == 0 for v in rw.mismatch(out, ref).values()), "repetition %d" % rep


# ---- the groups (one child process each) ------------------------------------------------------------------------------------------------------------------
def group_sites(ext, ns):
    """Every delay site (and all of them) of the delay length ``ns``, on the five-row launch and on the single-row decode launch, on both paths of the decode."""
    dly = rw.build_variant("rowsdelay", DELAY)
    cases = {}
    for pname, path in PATHS.items():
        workloads = {"multirow": make_workload(ext, MIXED, 77, path), "decode": make_workload(ext, DECODE, 79, path)}
        for wname, wl in workloads.items():
            for site in list(SITES) + ["all_sites"]:
                cases["%s/%s/%s" % (pname, wname, site)] = rb.capture(site_case, dly, wl, ns, site, path)
    return {"cases": cases, "reps": REPS}


def group_misc(ext):
    """The cases that do not vary with the delay length (on both paths): no configuration, liveness by timing, the racy control and the barrier-kept control."""
    dly = rw.build_variant("rowsdelay", DELAY)
    racy = rw.build_variant("rowsracy", RACY)
    cases = {}
    for pname, path in PATHS.items():
        workload, crossing = make_workload(ext, MIXED, 77, path), make_workload(ext, CROSSING, 78, path)
        cases.update({"%s/no_configuration" % pname: rb.capture(no_configuration_case, dly, workload, path), "%s/timing" % pname: rb.capture(timing_case, dly, workload, path),
                      "%s/racy" % pname: rb.capture(racy_case, racy, crossing, path), "%s/barrier_kept" % pname: rb.capture(barrier_kept_case, dly, crossing, path)})
    return {"cases": cases, "reps": REPS}


@pytest.fixture(scope="module")
def site_groups(ext):
    """The result of the site group of a delay length, run once in its own bounded child process."""
    cache = {}

    def get(ns):
        if ns not in cache:
            cache[ns] = rb.run_group("test_rows_delay", "group_sites", {"ns": ns}, GROUP_TIMEOUT_S)
        return cache[ns]

    return get


@pytest.fixture(scope="module")
def misc_group(ext):
    return rb.run_group("test_rows_delay", "group_misc", {}, GROUP_TIMEOUT_S)


# ---- the tests ----------------------------------------------------------------------------------------------------------------------------------------------
def test_the_default_build_has_no_delay_hook(ext):
    with pytest.raises(RuntimeError, match="not compiled"):
        ext.rows_test_delay_config(2000, -1, 0, 0, 1)
    with pytest.raises(RuntimeError, match="not compiled"):
        ext.rows_test_delay_hits()
    with pytest.raises(RuntimeError, match="not compiled"):
        ext.rows_test_pos_offset(1)


@pytest.mark.parametrize("path_name", list(PATHS))
def test_the_delay_variant_without_a_configuration_sleeps_nowhere_and_changes_no_bit(misc_group, path_name):
    assert misc_group["cases"]["%s/no_configuration" % path_name] is None, misc_group["cases"]["%s/no_configuration" % path_name]


@pytest.mark.parametrize("path_name", list(PATHS))
@pytest.mark.parametrize("workload_name", ["multirow", "decode"])
@pytest.mark.parametrize("ns", NS)
@pytest.mark.parametrize("site", list(SITES) + ["all_sites"])
def test_a_delay_at_a_site_changes_no_bit_and_the_hits_are_read_back(site_groups, workload_name, ns, site, path_name):
    outcome = site_groups(ns)["cases"]["%s/%s/%s" % (path_name, workload_name, site)]
    assert outcome is None, outcome


@pytest.mark.parametrize("path_name", list(PATHS))
def test_the_delay_takes_time(misc_group, path_name):
    assert misc_group["cases"]["%s/timing" % path_name] is None, misc_group["cases"]["%s/timing" % path_name]


# ---- the barrier that ends a row is needed: a racy control ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("path_name", list(PATHS))
def test_the_variant_without_the_end_of_row_barrier_differs_under_a_delay_in_the_chunk_end_phase(misc_group, path_name):
    assert misc_group["cases"]["%s/racy" % path_name] is None, misc_group["cases"]["%s/racy" % path_name]


@pytest.mark.parametrize("path_name", list(PATHS))
def test_the_same_delay_changes_nothing_where_the_barrier_is_kept(misc_group, path_name):
    assert misc_group["cases"]["%s/barrier_kept" % path_name] is None, misc_group["cases"]["%s/barrier_kept" % path_name]
