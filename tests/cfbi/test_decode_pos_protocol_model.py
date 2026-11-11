"""Model-level test of the arrival protocol that advances ``pos`` in the single-launch decode kernel (``rows_entry`` in csrc/cfbi/cfbi_rows.cuh): plain Python, no GPU.

Decode used to be two launches: a plan kernel that SNAPSHOTS ``pos`` (every head CTA of a sequence must start from the same fill of the open chunk, so none of them may advance
the word before all of them have read it) and the row engine. The single-launch kernel has no snapshot. Each of the H head CTAs of a participating sequence starts with
``old = atomicAdd(&pos[slot], 1 << 8)``: the low bits of the result are the fill (the arrival counter above them is masked off), the high bits the number of CTAs that arrived
before; the CTA that sees H - 1 is the last of the H, every other CTA has already read the word, so it stores the plain new value ``(pos0 + len) % 64``.

The model executes the two memory operations of every CTA (the arrival, then the store step of the elected CTA) in EVERY interleaving that is possible for a small H
(exhaustively up to H = 5) and in random interleavings for larger H and for several sequences at once, and checks the two properties the kernel relies on: every CTA reads
the fill ``pos0`` that the word had at the start of the call, and the word ends as the plain ``(pos0 + len) % 64``. The model proves the logic of the protocol, not the
hardware (the CUDA memory model argument is in the comment of ``rows_entry``: all accesses to the word are atomics or relaxed stores); what runs on the device is checked by
``test_decode_single_launch.py`` and, with delayed CTAs and with the wrong protocols below built into the kernel, by ``test_decode_single_launch_variants.py``.

Negative controls (each must be caught by the same check, or the check is vacuous): no counting with every CTA storing (the plan-less version of the old scheme: the late starters
read an advanced fill), the FIRST arrival storing, the counter bits not masked off by the readers, and nobody storing.
"""
import itertools
import random

import pytest

BT = 64
SHIFT = 8
MASK = BT - 1


def new_pos(pos0, length):
    return (pos0 + length) % BT


def interleavings(counts):
    """Every distinct order of the multiset in which CTA c appears ``counts[c]`` times."""
    counts = list(counts)
    total = sum(counts)
    out = []

    def rec(prefix):
        if len(prefix) == total:
            yield tuple(prefix)
            return
        for c in range(len(counts)):
            if counts[c] > 0:
                counts[c] -= 1
                prefix.append(c)
                for s in rec(prefix):
                    yield s
                prefix.pop()
                counts[c] += 1

    return rec(out)


def elected(variant, old, H):
    """Does the CTA whose arrival returned ``old`` store the new value?"""
    if variant in ("protocol", "no_mask"):
        return (old >> SHIFT) == H - 1
    if variant == "first_arrival":
        return (old >> SHIFT) == 0
    if variant == "every_cta":
        return True
    if variant == "nobody":
        return False
    raise ValueError(variant)


def run_words(seqs, schedule, variant="protocol"):
    """seqs: list of (slot, H, pos0, length); schedule: tuple of (sequence, head) pairs, each twice (arrival, then the store step). Returns
    (words by slot, fill seen by every CTA, number of stores per sequence)."""
    words = {}
    for slot, _, pos0, _ in seqs:
        words[slot] = pos0 if slot not in words else words[slot]
    old, seen, started, stores = {}, {}, set(), [0] * len(seqs)
    for (s, c) in schedule:
        slot, H, pos0, length = seqs[s]
        if (s, c) not in started:
            started.add((s, c))
            old[(s, c)] = words[slot]
            if variant != "every_cta":                       # every_cta: a plain read, no counting
                words[slot] = words[slot] + (1 << SHIFT)     # the atomic add (the result is the word before the add)
            seen[(s, c)] = old[(s, c)] if variant == "no_mask" else (old[(s, c)] & MASK)
        elif elected(variant, old[(s, c)], H):
            stores[s] += 1
            words[slot] = new_pos(seen[(s, c)], length)      # the store: the plain new value (the counter is gone with it)
    return words, seen, stores


def check_one(H, pos0, length, schedule, variant):
    """The two properties of the protocol for one sequence; returns a description of the first violation or None."""
    words, seen, stores = run_words([(0, H, pos0, length)], [(0, c) for c in schedule], variant)
    for c in range(H):
        if seen[(0, c)] != pos0:
            return "CTA %d started from fill %d, not %d" % (c, seen[(0, c)], pos0)
    if stores[0] != 1:
        return "%d CTAs stored the new fill" % stores[0]
    if words[0] != new_pos(pos0, length):
        return "the word ends as %d (expected the plain %d)" % (words[0], new_pos(pos0, length))
    return None


def violations(H, pos0, length, variant, schedules):
    return [(s, v) for s in schedules for v in [check_one(H, pos0, length, s, variant)] if v is not None]


# ---- the protocol is correct in every interleaving ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("H", [1, 2, 3, 4])
def test_every_interleaving_of_a_small_sequence_is_correct(H):
    scheds = list(interleavings([2] * H))
    assert len(scheds) == {1: 1, 2: 6, 3: 90, 4: 2520}[H]
    for pos0 in (0, 1, 17, 62, 63):
        for length in (1, 2, 3, 64, 65, 200):
            bad = violations(H, pos0, length, "protocol", scheds)
            assert not bad, (H, pos0, length, bad[0])


def test_every_interleaving_of_five_heads_is_correct():
    scheds = list(interleavings([2] * 5))
    assert len(scheds) == 113400
    for pos0, length in ((0, 1), (63, 1), (40, 70)):
        assert not violations(5, pos0, length, "protocol", scheds), (pos0, length)


@pytest.mark.parametrize("H", [6, 12, 33, 64, 257])
def test_random_interleavings_of_many_heads_are_correct(H):
    rng = random.Random(100 + H)
    base = [c for c in range(H) for _ in range(2)]
    scheds = []
    for _ in range(400):
        s = list(base)
        rng.shuffle(s)
        scheds.append(tuple(s))
    for pos0, length in ((0, 1), (63, 1), (31, 3), (5, 130)):
        assert not violations(H, pos0, length, "protocol", scheds), (H, pos0, length)
    # the extreme orders: heads in launch order, heads in reverse, one head arriving long after all others
    for order in (list(range(H)), list(range(H))[::-1]):
        scheds = [tuple(c for c in order for _ in range(2)), tuple(order + order)]
        assert not violations(H, 7, 1, "protocol", scheds)


def test_arrivals_of_different_sequences_interleave_freely():
    """Several sequences (distinct slots) with different H, fills and lengths in random interleavings: each word ends as its own plain new fill."""
    rng = random.Random(7)
    seqs = [(3, 3, 0, 1), (0, 5, 63, 1), (9, 1, 40, 70), (4, 4, 17, 65)]
    base = [(s, c) for s, (_, H, _, _) in enumerate(seqs) for c in range(H) for _ in range(2)]
    for _ in range(2000):
        sched = list(base)
        rng.shuffle(sched)
        words, seen, stores = run_words(seqs, tuple(sched))
        for s, (slot, H, pos0, length) in enumerate(seqs):
            assert stores[s] == 1 and words[slot] == new_pos(pos0, length), (s, sched)
            assert all(seen[(s, c)] == pos0 for c in range(H))


def test_the_elected_cta_is_the_last_one_to_arrive():
    """Whichever CTA arrives last stores, whatever its head index: the arrival order is all that counts."""
    H = 6
    for last in range(H):
        order = [c for c in range(H) if c != last] + [last]
        words, seen, stores = run_words([(0, H, 21, 1)], tuple((0, c) for c in order + order))
        assert stores == [1] and words[0] == 22


def test_the_new_fill_is_the_expression_of_the_two_launch_decode():
    """The value stored is (pos0 + len) % 64 with the same integers as the plan kernel's SeqPlan (pos0 masked, len from the call)."""
    for pos0 in range(BT):
        for length in (1, 2, 5, 63, 64, 65, 127, 128, 129, 1000):
            assert new_pos(pos0, length) == (pos0 + length) % 64 and 0 <= new_pos(pos0, length) < BT


# ---- the checks can fail: wrong protocols are caught ------------------------------------------------------------------------------------------------------
WRONG = ["every_cta", "first_arrival", "no_mask", "nobody"]


@pytest.mark.parametrize("variant", WRONG)
def test_a_wrong_protocol_is_caught_in_small_sequences(variant):
    for H in (2, 3, 4):
        scheds = list(interleavings([2] * H))
        bad = violations(H, 12, 1, variant, scheds)
        assert bad, "the check missed the wrong protocol %r with H = %d" % (variant, H)


def test_the_naive_scheme_fails_exactly_when_a_cta_starts_after_a_store():
    """The plan-less version of the old scheme (every CTA reads the plain word and stores the new value) is right only if no CTA starts after the first store: that is
    what the late starters of a multi-wave grid do, and what a delayed CTA does in the test variants."""
    H = 3
    for sched in interleavings([2] * H):
        first_store_step = None
        started_after_store = False
        seen_started = set()
        for i, c in enumerate(sched):
            if c not in seen_started:
                seen_started.add(c)
                if first_store_step is not None:
                    started_after_store = True
            elif first_store_step is None:
                first_store_step = i
        bad = check_one(H, 12, 1, sched, "every_cta")
        if started_after_store:
            assert bad is not None, sched
        else:
            # nobody started after a store: every CTA read the original fill; the word ends as the new fill (every CTA stores the same value) but H stores happened
            assert bad == "%d CTAs stored the new fill" % H, (sched, bad)


def test_duplicate_slots_are_outside_the_contract():
    """Distinct ``slot_ids`` stay an unchecked precondition (a check would need a device synchronisation): two sequences on ONE word make the count reach 2 H before
    every CTA has arrived, so a CTA can start from an advanced fill. The model shows the precondition is needed."""
    seqs = [(5, 2, 10, 1), (5, 2, 10, 1)]
    base = [(s, c) for s in range(2) for c in range(2) for _ in range(2)]
    rng = random.Random(3)
    broken = 0
    for _ in range(500):
        sched = list(base)
        rng.shuffle(sched)
        words, seen, stores = run_words(seqs, tuple(sched))
        if any(seen[k] != 10 for k in seen):
            broken += 1
    assert broken > 0


def test_every_cta_of_a_sequence_takes_the_same_decision_to_participate():
    """A CTA arrives only if the sequence has tokens and a slot (len > 0, slot >= 0); both are properties of the sequence, so all H heads arrive or none does and the
    counter can only ever reach H (never stop short of it)."""
    def participates(length, slot, head):          # the kernel's early exit: it does not look at the head index
        return length > 0 and slot >= 0

    for length in (0, 1, 7):
        for slot in (-1, 0, 5):
            assert len({participates(length, slot, h) for h in range(33)}) == 1
