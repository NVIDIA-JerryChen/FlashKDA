"""The arrival protocol of the single-launch decode under delays, and the harness shown to be able to fail (``rows_entry`` in csrc/cfbi/cfbi_rows.cuh; the variants are separate JIT builds,
``rows_workload.build_variant``; the default build contains none of their macros).

The single-launch decode has no plan kernel and no pos snapshot: the H head CTAs of a sequence count their arrivals in the high bits of the pos word, take the fill of the open chunk from what
their own atomic returned, and the last one to arrive stores the plain new fill. A wrong protocol is only visible when a CTA STARTS (or reads) after another CTA of its sequence has stored: on a
grid that fits one wave all CTAs read the word within a microsecond of each other and the plain decode tests cannot tell. The delay hook of the row engine (sites 9 .. 12, thread 0 of every CTA that
reaches the arrival) makes the late CTAs on demand, at run time, in one build (``rows_test_delay_config``):

* the four placements of the plan: BEFORE the arrival (site 9, the late starters), after the arrival of the FIRST CTA of its sequence (site 10), after the arrival of the LAST one and before its store
  (site 11), after the arrival of EVERY CTA (site 12); also all four together. First and last are told by the count that the atomic returned, never by the head index;
* 2000 ns and 20000 ns, five repetitions of each, on grids with at least 2.5 resident waves of live CTAs of ``rows_kernel<true>`` of every build involved (pool with permuted slots and padding
  entries, one-token decode, five-row calls across chunks ends, ``cu_seqlens`` with empty sequences, and H = 1) and on a CO-RESIDENT grid (a few sequences, every CTA on the device at once), the CTAs
  delayed all, the even ones, the odd ones and every third one: every bit of both outputs, E, the valid rows and pos equals the frozen reference kernels' (``_impl=3``), and every delay is LIVE,
  shown by its exact hit count read back (every live CTA at sites 9 and 12, one CTA per live sequence at sites 10 and 11: the plan of the launch says how many). Why both shapes: the grid is
  ``(N, H)`` with the sequence on x, so the H heads of one sequence are N CTAs apart in launch order and, beyond two waves, arrive WAVES apart (tens of microseconds and more): the late starters of
  a beyond-waves grid come for free and are what the first two wrong protocols below need, but a late READ after the arrival (the third) only meets the elected store if the siblings arrive together,
  which is what a co-resident grid gives;
* the wrong protocols, each built together with the hook: every CTA stores from the fill it read (no counting), the FIRST arrival stores, the fill is read from the word AFTER the arrival, and no
  elected store: under the delay that exposes it (the late starters for the first two, the late first arriver on a co-resident grid for the third) the same bit comparison must catch each, and the
  missing store is caught by the at-rest check; otherwise the harness proves nothing;
* every group of cases runs in a child process under an explicit limit (``rows_bounded``): a protocol that never completes fails the test and kills the child's process group instead of waiting for
  the end of the allocation. Every case keeps its id, its delay lengths, its repetitions and its assertions; the group size, duration and limit are recorded.

The ordering rules that no comparison of results can see (a plain store, the atomic behind a barrier, an extra read) are pinned as text in ``test_decode_protocol_source.py``.
"""
import random

import pytest

import helpers
import live_grid as lg
import rows_bounded as rb
import rows_workload as rw
from helpers import skip_or_fail
from rows_workload import Family

DELAY = ("CFBI_ROWS_DELAY_NS=2000",)           # the hook (its value is only the default length; the host sets the length at run time)
WRONG = {
    "every_cta": DELAY + ("CFBI_DECODE_TEST_EVERY_CTA_WRITES",),
    "first_arriver": DELAY + ("CFBI_DECODE_TEST_FIRST_ARRIVER",),
    "snapshot_after_arrival": DELAY + ("CFBI_DECODE_TEST_SNAPSHOT_AFTER_ARRIVAL",),
    "no_elected_store": DELAY + ("CFBI_DECODE_TEST_NO_ELECTED_STORE",),
}
REPS = 5
NS = (2000, 20000)
SITES = {name: (site,) for name, site in lg.PLACEMENTS.items()}
SITES["all_four"] = tuple(sorted(lg.PLACEMENTS.values()))
TARGETS = {"all_ctas": (0, 0), "even_ctas": (2, 0), "odd_ctas": (2, 1), "every_third_cta": (3, 1)}      # (cta_mod, cta_rem) with cta = sequence * H + head
SELECTIVE = ("before", "all")                      # the placements that may select CTAs by index; the first and the last arrival are not told by the index
DECODE = Family("decv_decode", fixed=1, pool=True, pad=0.3, fills="mixed", steps=1)                    # one row per sequence: the production decode launch
MULTIROW = Family("decv_multirow", fixed=5, pool=True, pad=0.2, fills="crossing", steps=1)             # five rows per sequence across chunk ends
VARLEN = Family("decv_varlen", lens=[1, 2, 7, 33, 64, 65], pool=True, pad=0.2, empty=0.2, fills="mixed", steps=1)    # cu_seqlens, empty sequences and padding entries
CORESIDENT = "coresident"          # not a family: CORESIDENT_SEQUENCES live sequences and a padding entry, every CTA on the device at once (siblings of a sequence arrive together)
CORESIDENT_SEQUENCES = 4
WORKLOADS = {"decode": (DECODE, 12, 91), "multirow": (MULTIROW, 12, 92), "varlen": (VARLEN, 12, 93), "decode_one_head": (DECODE, 1, 94), "coresident": (CORESIDENT, 12, 95)}      # (family, heads, seed)
# the delay that exposes each wrong protocol: (workload, placement, target, ns); a protocol is caught when the bit comparison (or the at-rest check) fails in some repetition
EXPOSURE = {
    "every_cta": [("decode", "before", "odd_ctas", 20000), ("multirow", "before", "odd_ctas", 20000), ("coresident", "before", "odd_ctas", 20000)],
    "first_arriver": [("decode", "before", "odd_ctas", 20000), ("multirow", "before", "odd_ctas", 20000), ("coresident", "before", "odd_ctas", 20000)],
    "snapshot_after_arrival": [("coresident", "first", "all_ctas", 20000)],
    "no_elected_store": [("decode", "all", "all_ctas", 2000), ("multirow", "all", "all_ctas", 2000)],
}
GROUP_TIMEOUT_S = 2400        # per group: a matrix group takes a few minutes on a B200 (a first run adds the JIT build of the variant), the controls group builds four variants; a hang ends here


def configs():
    """The ``(placement, target name)`` cells of the matrix of one workload and one delay length."""
    out = []
    for placement in SITES:
        for tname in (TARGETS if placement in SELECTIVE else ("all_ctas",)):
            out.append((placement, tname))
    return out


@pytest.fixture(autouse=True)
def _optimised_kernels_run_here(ext):
    if ext.resolve_impl(rw.OPTIMISED, -1) == ext.resolve_impl(rw.REFERENCE, -1):
        skip_or_fail("the optimised kernels do not run on this device (_impl=0 resolves to the reference kernels)")


_BUILT = {}


def decode_variant(name, defines):
    """A variant of the extension (its own build directory, a build that fails fails the test) with the single launch forced: a variant is a module of its own, with its own switch."""
    if name not in _BUILT:
        mod = rw.build_variant(name, defines)
        mod.decode_single_force(True)
        _BUILT[name] = mod
    return _BUILT[name]


def make_workload(ext, variant, family, H, seed):
    """``(step, warm, oracle outcome)``: a launch of the family whose LIVE CTAs (padding entries and empty sequences exit at once and do not count) are at least 2.5 resident waves of
    ``rows_kernel<true>`` of the default build and of the variant, asserted before any launch (the co-resident shape instead asserts that all of them fit on the device at once); its warm state; and the
    outcome of the frozen reference kernels (``_impl=3``) on it."""
    rng = random.Random(seed * 1000 + H)
    par = rw.make_params(H, seed)
    caps = {"default build": helpers.resident_ctas(ext, rw.SINGLE.instance), "variant": helpers.resident_ctas(variant, rw.SINGLE.instance)}
    if family == CORESIDENT:
        N = CORESIDENT_SEQUENCES + 1
        S = N + rw.POOL_SPARE
        slots = rng.sample(range(S), N)
        slots[2] = -1                                          # a padding entry among the live sequences, permuted slots
        step, fills = rw.Step(par, H, [1] * N, slots, True, seed * 10), "mixed"
        assert all(step.live_ctas() <= cap for cap in caps.values()), "the co-resident grid (%d live CTAs) does not fit on the device at once (%s)" % (step.live_ctas(), caps)
    else:
        step, S = rw.draw_step(variant, family, H, par, rng, seed * 10, instance=rw.SINGLE.instance)
        fills = family.fills
        problems = rw.beyond_waves_problems([step.live_ctas()], caps)
        assert not problems, "; ".join(problems)
    warm = rw.warm_pool(ext, S, H, rw.draw_fills(rng, S, fills), par, seed)
    return step, warm, rw.run_arm(ext, rw.ORACLE, warm, step)


def configure(mod, ns, placement, target):
    mod_, rem = TARGETS[target]
    mod.rows_test_delay_config(ns, -1, mod_, rem, sum(1 << s for s in SITES[placement]))


# ---- the cases (they run in the child process of their group) -------------------------------------------------------------------------------------------
def correct_case(variant, workload, ns, placement, target):
    """The correct protocol: bits equal the oracle's in every repetition, and the delay ran exactly as often as the plan of the launch says."""
    step, warm, ref = workload
    mod_, rem = TARGETS[target]
    want = lg.arrival_hits(step.lens, step.slots, step.H, SITES[placement], mod_, rem)
    assert want > 0, "%s / %s: the configuration cannot hit anything on this workload, the case would be vacuous" % (placement, target)
    for rep in range(REPS):
        configure(variant, ns, placement, target)
        out = rw.run_arm(variant, rw.SINGLE, warm, step)
        m = rw.mismatch(out, ref)
        assert all(v == 0 for v in m.values()), "delay %d ns at %s (%s), repetition %d: %s" % (ns, placement, target, rep, m)
        got = variant.rows_test_delay_hits()
        assert got == want, "delay %d ns at %s (%s): %d delays ran, the plan of the launch says %d" % (ns, placement, target, got, want)


def no_configuration_case(variant, workload):
    step, warm, ref = workload
    variant.rows_test_delay_config(0, -1, 0, 0, 0)
    out = rw.run_arm(variant, rw.SINGLE, warm, step)
    assert variant.rows_test_delay_hits() == 0
    assert all(v == 0 for v in rw.mismatch(out, ref).values())


def exposure_case(mod, workload, ns, placement, target):
    """A wrong protocol under a delay: it must differ from the oracle in some repetition (a position that leaves 0 .. 63, the arrival bits that stayed in the word, counts as differing)."""
    step, warm, ref = workload
    cap = helpers.resident_ctas(mod, rw.SINGLE.instance)
    if step.live_ctas() > cap:                              # not the co-resident shape (all of it fits at once): beyond waves
        floor = rw.beyond_waves_problems([step.live_ctas()], {"this variant": cap})
        assert not floor, "; ".join(floor)
    seen = []
    for rep in range(REPS):
        configure(mod, ns, placement, target)
        out = rw.run_arm(mod, rw.SINGLE, warm, step)
        try:
            m = rw.mismatch(out, ref)
        except AssertionError:                            # the at-rest invariant of the comparison: a position left [0, 63], the arrival bits stayed in the word
            return None
        if any(m.values()):
            return None
        seen.append(m)
    raise AssertionError("the wrong protocol matched the oracle in all %d repetitions under a %d ns delay at %s (%s): the harness cannot see it (%s)" % (REPS, ns, placement, target, seen[-1]))


# ---- the groups (one child process each) --------------------------------------------------------------------------------------------------------------
def group_matrix(ext, ns):
    """Every cell (workload x placement x delayed CTAs) of the delay length ``ns``, five repetitions each, on the variant with the hook."""
    variant = decode_variant("decdelay", DELAY)
    cases = {}
    for wname, (family, H, seed) in WORKLOADS.items():
        workload = make_workload(ext, variant, family, H, seed)
        cases["%s/no_configuration" % wname] = rb.capture(no_configuration_case, variant, workload)
        for placement, target in configs():
            cases["%s/%s/%s" % (wname, placement, target)] = rb.capture(correct_case, variant, workload, ns, placement, target)
    return {"cases": cases, "reps": REPS}


def group_controls(ext):
    """The wrong protocols (a build each, with the hook): every one is exposed by the delay that the plan names for it."""
    cases = {}
    workloads = {}
    for proto, defines in WRONG.items():
        mod = decode_variant("decwrong_" + proto, defines)
        for wname, placement, target, ns in EXPOSURE[proto]:
            if wname not in workloads:
                family, H, seed = WORKLOADS[wname]
                workloads[wname] = make_workload(ext, mod, family, H, seed)
            cases["%s/%s" % (proto, wname)] = rb.capture(exposure_case, mod, workloads[wname], ns, placement, target)
    return {"cases": cases, "reps": REPS}


@pytest.fixture(scope="module")
def matrix_groups(ext):
    """The result of the matrix group of a delay length, run once in its own bounded child process."""
    cache = {}

    def get(ns):
        if ns not in cache:
            cache[ns] = rb.run_group("test_decode_single_launch_variants", "group_matrix", {"ns": ns}, GROUP_TIMEOUT_S)
        return cache[ns]

    return get


@pytest.fixture(scope="module")
def controls_group(ext):
    return rb.run_group("test_decode_single_launch_variants", "group_controls", {}, GROUP_TIMEOUT_S)


# ---- the tests ----------------------------------------------------------------------------------------------------------------------------------------
def test_the_default_build_has_no_delay_hook_and_no_offset_control(ext):
    with pytest.raises(RuntimeError, match="not compiled"):
        ext.rows_test_delay_config(2000, -1, 0, 0, 1 << lg.SITE_BEFORE)
    with pytest.raises(RuntimeError, match="not compiled"):
        ext.rows_test_delay_hits()
    with pytest.raises(RuntimeError, match="not compiled"):
        ext.rows_test_pos_offset(1, 2)


@pytest.mark.parametrize("workload", list(WORKLOADS))
def test_the_variant_without_a_configuration_sleeps_nowhere_and_changes_no_bit(matrix_groups, workload):
    outcome = matrix_groups(NS[0])["cases"]["%s/no_configuration" % workload]
    assert outcome is None, outcome


@pytest.mark.parametrize("ns", NS)
@pytest.mark.parametrize("workload", list(WORKLOADS))
@pytest.mark.parametrize("placement,target", configs(), ids=["%s-%s" % c for c in configs()])
def test_a_delay_at_the_arrival_changes_no_bit_and_every_delay_is_read_back(matrix_groups, workload, placement, target, ns):
    outcome = matrix_groups(ns)["cases"]["%s/%s/%s" % (workload, placement, target)]
    assert outcome is None, outcome


@pytest.mark.parametrize("protocol", list(WRONG))
def test_a_wrong_protocol_is_caught_by_the_comparison(controls_group, protocol):
    for wname, placement, target, ns in EXPOSURE[protocol]:
        outcome = controls_group["cases"]["%s/%s" % (protocol, wname)]
        assert outcome is None, "%s: %s" % (protocol, outcome)
