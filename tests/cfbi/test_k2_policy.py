"""Model-level tests of the K2 schedule policy (host reference ``k2_policy.py``): plain Python, no GPU.

The policy decides the CTA order and the value-tile width of every (sequence, head) pair of the dynamic-schedule K2 from the chunk counts. These tests
check the reference itself against independent computations (brute-force list scheduling, fully expanded item lists), the bounds it uses, the
agreement with the closed form of the fixed-length launcher for equal lengths, and the property the policy is built for: on the deterministic corpus
no length list gets a slower exact list-scheduling makespan than the old rule (the gate of the full device test, ``test_k2_sched.py``, compares the
device schedule with this reference). The default-routing rule (``route_default``) is checked against its documented thresholds and against the B200
measurements it was derived from. The negative variants show that the checks can fail.
"""
import itertools
import multiprocessing
import os
import random

import pytest

import k2_policy as kp
from varlen_corpus import CORPUS_V

SMS = kp.CORPUS_SMS
BIG = os.environ.get("CFBI_BIG", "0") == "1"


def chunks_of(lens):
    return [kp.chunk_count(n) for n in lens]


def old_items_of(c, H, sms, call_fixed=False):
    items, _ = kp._natural_items(c, H, *kp.old_rule(len(c), H, sms, call_fixed))
    return items


def ratio_to_old(c, H, sms, items, call_fixed=False):
    return kp.list_makespan(items, c, sms, len(c)) / kp.list_makespan(old_items_of(c, H, sms, call_fixed), c, sms, len(c))


# ---- the corpus and the constants ------------------------------------------------------------------------------------------------------------------
def test_corpus_p_is_deterministic_and_covers_the_named_distributions():
    a, b = kp.corpus_p(), kp.corpus_p()
    assert a == b and len(a) >= 1000
    assert len({e["name"] for e in a}) == len(a)
    assert {e["kind"] for e in a} == {"equal", "uniform", "lognormal0.5", "lognormal1.0", "lognormal1.5", "onelong4", "onelong8", "onelong32", "half", "twolong"}
    assert {e["N"] for e in a} == set(kp.CORPUS_N) == {1, 2, 3, 4, 6, 8, 13, 16, 19, 32, 64}
    assert set(kp.CORPUS_H) == {1, 3, 6, 12, 32} and set(kp.CORPUS_SMS) == {132, 144, 148}
    assert all(len(e["lengths"]) == e["N"] and all(n > 0 for n in e["lengths"]) for e in a)
    assert kp.corpus_p(1) != a, "the seed is used"


def test_constants_and_capacity():
    t = kp.T_CHUNK
    assert (t[8], t[16], t[32]) == (34, 47, 78)
    assert abs(t[16] / t[8] - 4.7 / 3.4) < 0.01 and abs(t[32] / t[8] - 7.8 / 3.4) < 0.01
    assert {w: kp.HEAD_V // w for w in (8, 16, 32)} == {8: 16, 16: 8, 32: 4}
    assert all(kp.schedule_capacity(H) >= 512 for H in range(1, 34))
    assert (kp.schedule_capacity(1), kp.schedule_capacity(6), kp.schedule_capacity(33), kp.schedule_capacity(34)) == (1024, 1024, 512, 496)
    assert kp.schedule_capacity(0) == 0


def test_chunk_count_follows_the_plan_kernel():
    assert [kp.chunk_count(n) for n in (0, 1, 63, 64, 65, 128, 129)] == [0, 1, 1, 1, 2, 2, 3]
    assert kp.chunk_count(100, pos0=10) == 1            # 54 head rows finish the open chunk, 46 rows remain
    assert kp.chunk_count(30, pos0=10) == 0             # all rows are head rows
    assert kp.chunk_count(64 + 54, pos0=10) == 1 and kp.chunk_count(64 + 54 + 1, pos0=10) == 2


# ---- the closed form of the fixed-length launcher is the exact makespan for equal lengths -------------------------------------------------------------
def _simulate_equal(nh, x32, bvb, sms):
    durations = [kp.T_CHUNK[32]] * (4 * x32) + [kp.T_CHUNK[bvb]] * ((nh - x32) * (kp.HEAD_V // bvb))
    free = [0] * sms
    for d in durations:
        free[free.index(min(free))] += d
    return max(free)


@pytest.mark.parametrize("sms", SMS)
def test_grid_cost_equals_list_scheduling_of_equal_items(sms):
    rng = random.Random(sms)
    cases = [(nh, x32, bvb) for nh in (1, 6, 24, 37, 48, 96, 150) for bvb in (8, 16) for x32 in (0, 1, nh // 2, nh)]
    cases += [(rng.randint(1, 120), rng.randint(0, 120), rng.choice((8, 16))) for _ in range(40)]
    for nh, x32, bvb in cases:
        x32 = min(x32, nh)
        assert kp.grid_cost(nh, x32, bvb, sms) == _simulate_equal(nh, x32, bvb, sms), (nh, x32, bvb, sms)


# ---- the statistics and the bounds the policy uses ---------------------------------------------------------------------------------------------------
def _brute_stats(pairs, a, b, x, sms):
    durs = sorted([pairs[i][0] * kp.T_CHUNK[a] for i in range(x) for _ in range(kp.HEAD_V // a)] +
                  [pairs[j][0] * kp.T_CHUNK[b] for j in range(x, len(pairs)) for _ in range(kp.HEAD_V // b)], reverse=True)
    return sum(durs), durs[0], (durs[sms - 1] if len(durs) >= sms else 0), (durs[sms] if len(durs) > sms else 0)


def test_candidate_statistics_equal_the_brute_force_expansion():
    rng = random.Random(11)
    for _ in range(120):
        N = rng.choice((1, 2, 3, 5, 8, 13))
        H = rng.choice((1, 3, 6))
        c = [rng.choice((0, 1, 2, 5, 16, 40, 128)) if rng.random() < 0.8 else rng.randint(1, 300) for _ in range(N)]
        pairs = kp.active_pairs_sorted(c, H)
        if not pairs:
            continue
        sms = rng.choice((4, 17, 132, 148))
        for a, b in itertools.product((8, 16, 32), repeat=2):
            for x in {0, len(pairs) // 3, len(pairs)}:
                assert kp.candidate_stats(pairs, a, b, x, sms) == _brute_stats(pairs, a, b, x, sms), (c, H, a, b, x, sms)


def test_bounds_bracket_the_exact_makespan():
    rng = random.Random(12)
    for _ in range(80):
        N = rng.choice((2, 3, 6, 8, 13))
        H = rng.choice((1, 3, 6))
        c = [rng.randint(1, 130) for _ in range(N)]
        sms = rng.choice((17, 40, 132, 148))
        pairs = kp.active_pairs_sorted(c, H)
        for a, b in ((16, 16), (8, 32), (32, 16), (8, 8)):
            x = len(pairs) // 3
            items, _ = kp._sorted_items(c, pairs, [a if i < x else b for i in range(len(pairs))])
            exact = kp.list_makespan(items, c, sms, N)
            st = kp.candidate_stats(pairs, a, b, x, sms)
            assert kp.bound_pig(sms, *st) <= exact <= kp.bound_ub(sms, *st), (c, H, a, b, sms)
        natural = old_items_of(c, H, sms)
        w_old = 16 if N * H * 16 > sms else 8
        st = kp.candidate_stats(pairs, w_old, w_old, 0, sms)
        assert kp.list_makespan(natural, c, sms, N) <= kp.bound_graham(sms, *st)


# ---- agreement with the fixed-length closed form, invariants -------------------------------------------------------------------------------------
@pytest.mark.parametrize("call_fixed", [False, True])
def test_equal_length_lists_use_the_widths_of_the_fixed_length_launcher(call_fixed):
    equal = [e for e in kp.corpus_p() if e["kind"] == "equal"]
    assert len(equal) == len(kp.CORPUS_N)
    for e in equal:
        c = chunks_of(e["lengths"])
        for H, sms in itertools.product(kp.CORPUS_H, SMS):
            s = kp.build_schedule(c, H, sms, call_fixed=call_fixed)
            x32, bvb = kp.choose_k2_tiles(len(c) * H, sms)
            counts = s.width_counts()
            assert counts[32] == x32 and counts[bvb] == len(c) * H - x32, (e["name"], H, sms, counts, x32, bvb)
            if call_fixed:   # the old schedule of a fixed-length call IS that closed form
                assert s.kind == "old" and s.items == old_items_of(c, H, sms, True)


def test_schedules_of_the_corpus_satisfy_the_invariants():
    lists = [e for e in kp.corpus_p() if e["kind"] not in ("uniform", "lognormal0.5", "lognormal1.0", "lognormal1.5")]
    lists += [e for e in kp.corpus_p() if e["name"].endswith("/0") or e["name"].endswith("/7")]
    kinds = set()
    for e in lists:
        c = chunks_of(e["lengths"])
        for H, sms in itertools.product((1, 3, 6), SMS):
            s = kp.build_schedule(c, H, sms)
            kinds.add(s.kind)
            assert kp.check_invariants(s.items, c, H, s.kind) == [], (e["name"], H, sms)
    assert kinds == {"old", "sorted"}


def test_zero_chunk_sequences_and_forced_widths():
    c = [0, 7, 0, 130, 1]
    for H, sms in ((3, 148), (6, 132)):
        for bv in (8, 16, 32):
            s = kp.build_schedule(c, H, sms, forced_bv=bv)
            assert kp.check_invariants(s.items, c, H, s.kind) == [] and set(s.widths.values()) == {bv}
            assert len(s.items) == 3 * H * (kp.HEAD_V // bv)
    assert kp.build_schedule([0, 0], 3, 148).items == []


# ---- fixed-length calls with unequal active chunk counts: the schedule is the old launcher's, exactly --------------------------------------------------
def launcher_layout(c, H, sms):
    """The CTAs of the old fixed-length launcher in grid order, written out from its structure (not from ``_natural_items``): of the N * H pair positions
    p = h * N + n the first x32 are 32-column tiles and the others bvb-column tiles, (x32, bvb) the closed form ``choose_k2_tiles(N * H)`` of ALL positions;
    the CTAs of a pair without a chunk (a skipped slot, or a sequence whose rows all fit the open chunk) return at once and are absent."""
    N = len(c)
    x32, bvb = kp.choose_k2_tiles(N * H, sms)
    out = []
    for p in range(N * H):
        if c[p % N] > 0:
            w = 32 if p < x32 else bvb
            out += [(p, s, w) for s in range(kp.HEAD_V // w)]
    return out


def fixed_call_chunks(rng, N, L, skip=0.25):
    """Chunk counts of a fixed-length call of N sequences of L tokens: every sequence starts at an open-chunk fill (pos) of its own, skipped slots (slot id -1)
    have no chunk. The counts take two values {m, m + 1} (and 0): the lists a fixed-length call can produce."""
    return [0 if rng.random() < skip else kp.chunk_count(L, rng.choice((0, 1, 17, 33, 63))) for _ in range(N)]


def test_fixed_length_call_with_unequal_chunk_counts_is_the_old_schedule_reproduction_of_the_review():
    c = [2] + [1] * 7           # eight sequences of equal length; one of them starts in another open-chunk fill: chunk counts 2, 1, ..., 1
    assert kp.old_rule(8, 6, 148, True) == (37, 16)
    s = kp.build_schedule(c, 6, 148, call_fixed=True)
    assert s.kind == "old" and s.choice is None
    assert s.items == launcher_layout(c, 6, 148) and len(s.items) == 37 * 4 + 11 * 8 == 236
    assert s.width_counts() == {8: 0, 16: 11, 32: 37}
    assert kp.check_invariants(s.items, c, 6, s.kind) == []
    assert kp.list_makespan(s.items, c, 148, 8) == kp.list_makespan(old_items_of(c, 6, 148, True), c, 148, 8)
    assert kp.build_schedule(c, 6, 148, call_fixed=False).items != s.items          # the schedule of a variable-length call is another one


@pytest.mark.parametrize("sms", SMS)
def test_fixed_length_calls_with_warm_positions_and_skipped_slots_get_the_old_schedule(sms):
    rng = random.Random(20261108 + sms)
    unequal = equal = mixed = 0
    for _ in range(500):
        N = rng.choice((2, 3, 5, 8, 13, 16, 19, 32, 64))
        H = rng.choice(kp.CORPUS_H)
        L = rng.choice((1, 40, 63, 64, 65, 100, 129, 424, 4136, 8232))
        c = fixed_call_chunks(rng, N, L)
        if not any(c):
            continue
        s = kp.build_schedule(c, H, sms, call_fixed=True)
        assert kp.check_invariants(s.items, c, H, s.kind) == [], (N, H, L, c)
        ratio = ratio_to_old(c, H, sms, s.items, call_fixed=True)
        assert ratio <= 1.0 + 1e-12, (N, H, L, c, ratio)                           # never slower than the old schedule of the call type
        if len({v for v in c if v}) > 1:
            unequal += 1
            x32, _ = kp.choose_k2_tiles(N * H, sms)
            mixed += 0 < x32 < N * H
            assert s.kind == "old" and s.choice is None and s.items == launcher_layout(c, H, sms), (N, H, L, sms, c)
            assert ratio == 1.0
            assert s.width_counts()[32] == sum(1 for p in range(x32) if c[p % N] > 0)
        else:
            equal += 1
    assert unequal >= 150 and equal >= 50 and mixed >= 30, (unequal, equal, mixed)


def test_a_homogeneous_reference_for_fixed_length_calls_is_rejected(monkeypatch):
    """Negative control: the variable-length rule (8 or 16 columns for every pair) used as the old schedule of a fixed-length call is not the launcher's layout."""
    c = [2] + [1] * 7
    want = launcher_layout(c, 6, 148)
    assert kp.build_schedule(c, 6, 148, call_fixed=True).items == want
    monkeypatch.setattr(kp, "old_rule", lambda N, H, sms, call_fixed: (0, 8 if N * H * 16 <= sms else 16))
    variant = kp.build_schedule(c, 6, 148, call_fixed=True)
    assert variant.items != want and variant.width_counts() == {8: 0, 16: 48, 32: 0}
    monkeypatch.undo()
    # ... and so is a layout that counts the 32-column positions over the active pairs only (instead of over all N * H positions of the launcher)
    c2 = [0, 3, 3, 2, 0, 3, 3, 3]
    x32, bvb = kp.choose_k2_tiles(8 * 6, 148)
    wrong = []
    for i, p in enumerate(q for q in range(48) if c2[q % 8] > 0):
        w = 32 if i < x32 else bvb
        wrong += [(p, s_, w) for s_ in range(kp.HEAD_V // w)]
    assert kp.build_schedule(c2, 6, 148, call_fixed=True).items == launcher_layout(c2, 6, 148)
    assert wrong != launcher_layout(c2, 6, 148)


# ---- the policy against the old rule, scored by the exact list-scheduling makespan --------------------------------------------------------------
@pytest.mark.parametrize("sms", SMS)
def test_corpus_v_is_never_slower_than_the_old_rule(sms):
    gains = {}
    for name, e in CORPUS_V.items():
        c = chunks_of(e.lens)
        s = kp.build_schedule(c, e.H, sms)
        r = ratio_to_old(c, e.H, sms, s.items)
        assert r <= 1.0 + 1e-12, (name, sms, r)
        gains[name] = r
    # the entries the schedule is meant for gain a lot (the model of the planning round: skewed batches -20 .. -35 %, equal lengths -11 %)
    assert gains["V2"] < 0.8 and gains["V2r"] < 0.8 and gains["V6"] < 0.75 and gains["V7b"] < 0.8 and gains["V1"] < 0.92 and gains["V3"] < 0.9


def _one_case(arg):
    name, lens, H, sms = arg
    c = chunks_of(lens)
    s = kp.build_schedule(c, H, sms)
    problems = kp.check_invariants(s.items, c, H, s.kind)
    return name, H, sms, ratio_to_old(c, H, sms, s.items), len(problems)


def _corpus_cases(max_pairs):
    return [(e["name"], e["lengths"], H, sms) for e in kp.corpus_p() for H in kp.CORPUS_H for sms in SMS if e["N"] * H <= max_pairs]


def test_corpus_p_is_not_slower_than_the_old_rule_by_more_than_one_percent():
    cases = _corpus_cases(10 ** 9 if BIG else 192)   # the default run keeps the cases of up to 192 pairs; CFBI_BIG=1 runs all 16830
    assert len(cases) >= (16000 if BIG else 5000)
    with multiprocessing.Pool(min(32, os.cpu_count() or 1)) as pool:
        res = pool.map(_one_case, cases, chunksize=16)
    assert sum(r[4] for r in res) == 0
    ratios = [r[3] for r in res]
    worst = max(res, key=lambda r: r[3])
    assert max(ratios) <= 1.01, worst
    assert sum(ratios) / len(ratios) < 0.88, "the policy must actually gain on the corpus (mean ratio of the K2 makespan %.3f)" % (sum(ratios) / len(ratios))


# ---- negative variants: the checks can fail ------------------------------------------------------------------------------------------------------------
def test_a_policy_with_32_columns_everywhere_is_rejected_by_the_no_regression_check():
    c = chunks_of([8192] + [1024] * 7)
    for sms in SMS:
        pairs = kp.active_pairs_sorted(c, 6)
        items, _ = kp._sorted_items(c, pairs, [32] * len(pairs))
        assert ratio_to_old(c, 6, sms, items) > 1.2          # the wide tiles lengthen the critical path of the long sequence
        good = kp.build_schedule(c, 6, sms)
        assert ratio_to_old(c, 6, sms, good.items) < 1.0


def test_an_equal_length_policy_disagrees_with_the_reference_on_skewed_lists():
    skewed = chunks_of([8192] + [1024] * 7)
    for sms in SMS:
        x32, bvb = kp.choose_k2_tiles(len(skewed) * 6, sms)     # the variant: the closed form regardless of the lengths
        ref = kp.build_schedule(skewed, 6, sms).width_counts()
        assert (ref[32], ref[bvb]) != (x32, len(skewed) * 6 - x32), "the reference must see the lengths"
    equal = chunks_of([8192] * 8)
    x32, bvb = kp.choose_k2_tiles(48, 148)
    assert kp.build_schedule(equal, 6, 148).width_counts()[32] == x32     # and on equal lists the two agree


def test_a_variant_without_the_old_schedule_fails_the_reproduction_check(monkeypatch):
    c = chunks_of([8192] * 8)
    want = old_items_of(c, 6, 148, True)
    assert kp.old_rule(8, 6, 148, True) == (37, 16)                                # the closed form of 48 equal pairs is a mixed grid
    s = kp.build_schedule(c, 6, 148, call_fixed=True)
    assert s.kind == "old" and s.items == want                                     # the reference reproduces the old schedule of a fixed-length call
    monkeypatch.setattr(kp, "choose_k2_tiles", lambda nh, sms: (nh, 16))         # a candidate set without it: always 32 columns
    assert kp.build_schedule(c, 6, 148, call_fixed=True).items != want
    monkeypatch.undo()
    c2 = chunks_of([4096] * 4)
    v = kp.build_schedule(c2, 3, 148, call_fixed=False)                           # variable-length call, 12 pairs: the old rule's 16 columns is the best homogeneous width
    assert v.kind == "old" and v.items == old_items_of(c2, 3, 148, False)


def test_invariant_checker_flags_corrupted_schedules():
    c = chunks_of([300, 1500, 64, 700])
    s = kp.build_schedule(c, 3, 148)
    assert kp.check_invariants(s.items, c, 3, s.kind) == []
    duplicated = s.items + [s.items[5]]
    assert any("duplicate" in p for p in kp.check_invariants(duplicated, c, 3, s.kind))
    dropped = s.items[:7] + s.items[8:]
    assert any("missing" in p for p in kp.check_invariants(dropped, c, 3, s.kind))
    swapped = list(s.items)
    swapped[0], swapped[-1] = swapped[-1], swapped[0]
    assert any("total order" in p for p in kp.check_invariants(swapped, c, 3, s.kind))
    two_widths = [(p, s_, 16 if (p, s_) == (s.items[0][0], 0) else w) for p, s_, w in s.items]
    assert any("two widths" in p or "bad slice" in p or "missing" in p for p in kp.check_invariants(two_widths, c, 3, s.kind))
    zero = chunks_of([0, 300])
    z = kp.build_schedule(zero, 2, 148)
    assert kp.check_invariants(z.items, zero, 2, z.kind) == []
    assert any("inactive" in p for p in kp.check_invariants(z.items + [(0, 0, 8)], zero, 2, z.kind))


# ---- the default routing rule ----------------------------------------------------------------------------------------------------------------------
# K2-stage change (percent, schedule builder included) of impl 16 against the old K2 kernels, measured on a B200 (148 SMs; interleaved A/B, repeats of profiled calls,
# medians): the evidence behind ``route_default``. The K2 kernels are identical in the trees of all the campaigns; a list that was measured more than once carries the value of the
# latest campaign (the campaign of the final routing rule, then those of its two earlier versions; the lists the first rule routed come from its default-routing campaign). A list is written "COUNT*LENGTH + ... @H" and appears here as (sequences, heads, tokens, change). The corpus V entries
# and the explicit lists around every condition of the rule fixed its parameters; the "held-out" lists are 40 random lists (equal, skewed and random lengths, 2 - 48 sequences,
# 2 - 24 heads) that were measured after the first model-based parameters were fixed; they decided no parameter, but the later candidates were evaluated on them too.
B200_ROUTING_MEASUREMENTS = {
    "held-out 16": (2, 2, 2496, -0.6), "1*512@6": (1, 6, 512, -0.7), "1*2048@6": (1, 6, 2048, -0.2), "V7a": (1, 6, 8192, -0.2),
    "held-out 34": (2, 3, 9216, -0.4), "1*2048@10": (1, 10, 2048, 10.8), "1*4096@10": (1, 10, 4096, 8.2), "1*8192@10": (1, 10, 8192, 5.2),
    "2*4096@5": (2, 5, 8192, 7.9), "1*16384@10": (1, 10, 16384, 3.3), "1*32768@10": (1, 10, 32768, 2.4), "1*8192@11": (1, 11, 8192, 5.3),
    "2*128@6": (2, 6, 256, 63.7), "2*1024@6": (2, 6, 2048, 18.7), "2*2048@6": (2, 6, 4096, 11.4), "1*8192@12": (1, 12, 8192, 5.1),
    "2*4096@6": (2, 6, 8192, 8.2), "V7b": (2, 6, 9216, -25.7), "1*16384@12": (1, 12, 16384, 3.3), "2*8192@6": (2, 6, 16384, 5.1),
    "1*8192@16": (1, 16, 8192, 5.0), "3*1536@6": (3, 6, 4608, 13.4), "1*8192@20": (1, 20, 8192, -15.5), "held-out 36": (10, 2, 29696, -27.1),
    "4*64@6": (4, 6, 256, 24.2), "4*512@6": (4, 6, 2048, -3.1), "held-out 4": (3, 8, 2304, -6.9), "2048+3*512@6": (4, 6, 3584, -24.0),
    "4*1024@6": (4, 6, 4096, -9.3), "held-out 10": (12, 2, 6144, -2.9), "2*4096@12": (2, 12, 8192, -13.9), "4*2048@6": (4, 6, 8192, -11.3),
    "4*4096@6": (4, 6, 16384, -14.2), "held-out 27": (10, 3, 2560, 3.5), "held-out 6": (10, 3, 2752, -4.6), "held-out 24": (10, 3, 21696, -22.6),
    "held-out 21": (16, 2, 17920, -25.7), "6*1024@6": (6, 6, 6144, -8.1), "3*6000@12": (3, 12, 18000, -14.6), "held-out 25": (12, 3, 31744, -37.0),
    "held-out 22": (20, 2, 5120, 5.9), "held-out 40": (20, 2, 37760, -16.2), "4*512@12": (4, 12, 2048, 1.8), "8*512@6": (8, 6, 4096, 2.2),
    "8*520@6": (8, 6, 4160, 1.5), "8*640@6": (8, 6, 5120, 1.6), "4096+7*256@6": (8, 6, 5888, -31.6), "8*768@6": (8, 6, 6144, 0.8),
    "4*2048@12": (4, 12, 8192, -5.6), "8*1024@6": (8, 6, 8192, -1.3), "held-out 30": (8, 6, 10304, -9.2), "V2": (8, 6, 15360, -24.0),
    "V2r": (8, 6, 15360, -24.3), "8*2048@6": (8, 6, 16384, -5.6), "held-out 11": (6, 8, 29696, -6.4), "8*4096@6": (8, 6, 32768, -8.4),
    "V5": (8, 6, 33536, -9.4), "V3": (8, 6, 36864, -15.1), "V4": (8, 6, 38592, -16.3), "V6": (8, 6, 39936, -32.2), "V1": (8, 6, 65536, -9.9),
    "held-out 7": (10, 5, 7680, 1.0), "1000+7*300+2048@6": (9, 6, 5148, -24.7), "held-out 39": (5, 12, 13824, -25.1), "held-out 9": (20, 3, 15360, -40.5),
    "held-out 8": (20, 3, 20480, -10.8), "held-out 15": (20, 3, 20992, -35.8), "8*384@8": (8, 8, 3072, -7.3), "4*4096@16": (4, 16, 16384, -15.2),
    "8*2048@8": (8, 8, 16384, -13.6), "16*2048@4": (16, 4, 32768, -13.5), "32*1024@2": (32, 2, 32768, -10.6), "64*512@1": (64, 1, 32768, -6.7),
    "held-out 23": (16, 4, 37184, -5.7), "12*512@6": (12, 6, 6144, -6.9), "held-out 5": (12, 6, 21760, -17.9), "held-out 28": (16, 5, 10496, -14.2),
    "2*8192@40": (2, 40, 16384, -12.9), "16*64@6": (16, 6, 1024, 6.1), "1024+15*64@6": (16, 6, 1984, -17.1), "16*256@6": (16, 6, 4096, -6.1),
    "8*512@12": (8, 12, 4096, -9.1), "held-out 3": (8, 12, 4416, -22.8), "16*320@6": (16, 6, 5120, -6.6), "held-out 35": (8, 12, 5568, -13.0),
    "2048+15*256@6": (16, 6, 5888, -22.5), "held-out 17": (8, 12, 6144, -11.3), "held-out 20": (6, 16, 6848, -9.9), "16*512@6": (16, 6, 8192, -9.1),
    "held-out 1": (12, 8, 10752, -28.1), "16*1024@6": (16, 6, 16384, -12.5), "held-out 19": (32, 3, 32768, -43.2), "held-out 14": (32, 3, 47616, -37.9),
    "16*4096@6": (16, 6, 65536, -16.0), "held-out 31": (20, 5, 5120, -6.3), "held-out 26": (5, 24, 14336, -27.1), "4*128@32": (4, 32, 512, 5.6),
    "held-out 29": (6, 24, 12288, -31.0), "held-out 32": (48, 3, 73728, -14.3), "24*256@8": (24, 8, 6144, -4.5), "32*256@6": (32, 6, 8192, -4.3),
    "32*257@6": (32, 6, 8224, -3.0), "4096+31*256@6": (32, 6, 12032, -25.2), "32*384@6": (32, 6, 12288, -7.1), "held-out 12": (8, 24, 42944, -18.3),
    "held-out 37": (48, 4, 62976, -34.0), "held-out 38": (48, 5, 15168, -13.4), "held-out 2": (48, 6, 94912, -18.7), "100*333@3": (100, 3, 33300, -7.3),
    "32*128@12": (32, 12, 4096, 3.1), "64*64@6": (64, 6, 4096, 8.8), "2048+63*128@6": (64, 6, 10112, -8.9), "64*256@6": (64, 6, 16384, -3.6),
    "64*384@6": (64, 6, 24576, -6.8), "held-out 13": (48, 8, 34944, -24.8), "64*1024@6": (64, 6, 65536, -12.9), "V7c": (64, 6, 282944, -18.1),
    "held-out 33": (20, 24, 8640, -19.3), "16*64@32": (16, 32, 1024, 5.1), "16*512@32": (16, 32, 8192, -10.7), "V7d": (16, 32, 56768, -19.9),
    "held-out 18": (32, 16, 139264, -27.4), "128*64@6": (128, 6, 8192, 5.2), "8192+127*64@6": (128, 6, 16320, -22.1), "128*256@6": (128, 6, 32768, -5.8),
    "512*256@2": (512, 2, 131072, -4.1), "1024*256@1": (1024, 1, 262144, 1.0), "256*64@6": (256, 6, 16384, 4.6), "512*64@6": (512, 6, 32768, 4.6),
}
ROUTING_MIN_GAIN_PCT = 2.0      # every measured list that the rule routes gained at least this much (the weakest: 3.0 %)


def test_route_default_follows_the_documented_conditions():
    sms = 148
    assert kp.route_default(16, 6, 16 * 256, sms) and not kp.route_default(16, 6, 16 * 255, sms)            # tokens per sequence: at least 256
    assert not kp.route_default(16, 6, 16 * 256, sms, cc_major=9) and not kp.route_default(16, 6, 16 * 256, sms, cc_major=12)   # only Blackwell (compute capability 10)
    assert not kp.route_default(1, 6, 8192, sms)                                  # 96 items at 8 columns fit one wave of 148 SMs: nothing to order or choose
    for N, H in ((1, 10), (1, 18), (2, 6), (3, 6)):                               # between one and two waves at 8 columns the closed form picks the old grid: nothing to gain
        assert kp.choose_k2_tiles(N * H, sms) == kp.old_rule(N, H, sms, False)
        assert not kp.route_default(N, H, 8192 * N, sms), (N, H)
    assert kp.route_default(1, 20, 8192, sms) and kp.route_default(4, 6, 4 * 1024, sms)   # two waves and more: the closed form picks 32-column tiles
    # 48 pairs: the closed form saves about 11 % of the K2 stage on equal lengths, the builder costs 7 us and the table-driven kernel a few %: only long sequences repay
    assert not kp.route_default(8, 6, 8 * 1536, sms) and kp.route_default(8, 6, 8 * 2048, sms)
    assert not kp.route_default(4, 12, 4 * 1024, sms) and kp.route_default(4, 12, 4 * 2048, sms)
    assert kp.route_default(512, 1, 512 * 1024, sms) and not kp.route_default(513, 1, 513 * 1024, sms) and not kp.route_default(1024, 1, 1024 * 4096, sms)   # at most 512 sequences
    assert kp.route_default(512, 33, 512 * 4096, sms) and not kp.route_default(513, 33, 513 * 4096, sms)            # ... and within the capacity of the builder (512 sequences at 33 heads)
    assert kp.schedule_capacity(1) == 1024 and kp.schedule_capacity(33) == 512
    assert not kp.route_default(0, 6, 1000, sms) and not kp.route_default(4, 0, 1000, sms)
    for N, H in ((4, 6), (8, 6), (16, 6), (1, 20), (4, 12), (32, 6), (64, 1)):    # monotone in the number of tokens (more work never switches the routing off)
        r = [kp.route_default(N, H, 64 * N * c, sms) for c in range(1, 400)]
        assert all(not a or b for a, b in zip(r, r[1:])), (N, H)
    # the estimate saturates (and the C++ arithmetic stays inside 64 bits): a call of the largest sequences the plan allows is routed like the saturated one
    assert kp.route_default(8, 6, 8 * (1 << 28), sms) == kp.route_default(8, 6, 8 * (1 << 20), sms)


def test_route_default_has_no_measured_loser_inside_its_region_and_keeps_the_big_wins():
    for name, (N, H, T, delta) in B200_ROUTING_MEASUREMENTS.items():
        if kp.route_default(N, H, T, 148):
            assert delta <= -ROUTING_MIN_GAIN_PCT, f"{name}: routed but measured {delta:+.1f} % on the K2 stage"
    # ... and the region is not empty: it keeps every corpus-V list but the one-wave list and the two skewed sequences between one and two waves (equal-length lists of that
    # size lose 3 - 13 %, and the host cannot tell them from skewed ones)
    for name in ("V1", "V2", "V2r", "V3", "V4", "V5", "V6", "V7c", "V7d"):
        N, H, T, delta = B200_ROUTING_MEASUREMENTS[name]
        assert kp.route_default(N, H, T, 148), name
    for name in ("V7a", "V7b"):
        assert not kp.route_default(*B200_ROUTING_MEASUREMENTS[name][:3], 148), name
    routed = [n for n, (N, H, T, d) in B200_ROUTING_MEASUREMENTS.items() if kp.route_default(N, H, T, 148)]
    outside_losers = [n for n, (N, H, T, d) in B200_ROUTING_MEASUREMENTS.items() if d > 1.0 and not kp.route_default(N, H, T, 148)]
    assert len(routed) >= 67 and len(outside_losers) >= 23, (len(routed), len(outside_losers))   # the measurements contain what the rule routes and what it keeps


def test_a_looser_routing_rule_is_rejected_by_the_measurements(monkeypatch):
    """Negative controls: without the allowance for the table-driven kernel the 48-pair lists of 512 - 1024 tokens per sequence are routed although they lose or gain
    little; without the tokens-per-sequence condition the lists of one-chunk sequences that lost 5 - 9 % are routed; without the limit on the number of sequences the list of
    1024 sequences, whose builder takes 68 us, is routed and loses."""
    for name, value in (("ROUTE_OVERHEAD_PCT", 0), ("ROUTE_MIN_TOKENS_PER_SEQ", 0), ("ROUTE_MAX_SEQUENCES", 1024)):
        monkeypatch.setattr(kp, name, value)
        with pytest.raises(AssertionError, match="routed but measured"):
            test_route_default_has_no_measured_loser_inside_its_region_and_keeps_the_big_wins()
        monkeypatch.undo()
