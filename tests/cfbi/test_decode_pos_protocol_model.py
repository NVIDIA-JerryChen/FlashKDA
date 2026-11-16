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

What the model is bound to, and what it is not. Its constants (``BT``, ``POS_COUNT_SHIFT``, ``POS_MASK``) and its election (the condition of the elected store) are READ from the CUDA sources
(``protocol_source``), so a change of either in the kernel changes the model, and the pins of ``test_decode_protocol_source.py`` fail on the same edit. The interleaving checks use Python
integers; the 32-bit word is modelled explicitly only where it matters (the capacity boundary below). The model says nothing about the CUDA memory model (the argument is in the comment of
``rows_entry``: every access to the word is an atomic or a relaxed store, the elected store needs no ordering against E and rows), nor about the hardware, and the preconditions that the
tests below show to be necessary (distinct slots, no arrival residue) are documented in ``flash_kda/cfbi/api.py``, not validated on the device.
"""
import random
import re

import pytest

import protocol_source as ps

SOURCES = ps.load()
CONSTANTS = ps.constants(SOURCES)
BT, SHIFT, MASK = CONSTANTS["BT"], CONSTANTS["POS_COUNT_SHIFT"], CONSTANTS["POS_MASK"]
PROTOCOL_ELECTS = ps.election(SOURCES)       # f(count, H): the CTA whose arrival found ``count`` CTAs before it stores (the condition of the elected store in rows_entry)


def new_pos(pos0, length):
    return (pos0 + length) % BT


def interleavings(counts):
    """Every distinct order of the multiset in which CTA c appears ``counts[c]`` times."""
    counts = list(counts)
    total = sum(counts)

    def rec(prefix):
        if len(prefix) == total:
            yield tuple(prefix)
            return
        for c in range(len(counts)):
            if counts[c] > 0:
                counts[c] -= 1
                prefix.append(c)
                yield from rec(prefix)
                prefix.pop()
                counts[c] += 1

    return rec([])


def elected(variant, old, H, elect=None):
    """Does the CTA whose arrival returned ``old`` store the new value? (``elect``: another election than the one that the sources pin)"""
    if variant in ("protocol", "no_mask"):
        return (elect or PROTOCOL_ELECTS)(old >> SHIFT, H)
    if variant == "first_arrival":
        return (old >> SHIFT) == 0
    if variant == "every_cta":
        return True
    if variant == "nobody":
        return False
    raise ValueError(variant)


def run_words(seqs, schedule, variant="protocol", elect=None):
    """seqs: list of (slot, H, pos0, length[, residue]); ``residue`` = arrival bits that an interrupted earlier call left in the word (default 0: a word at rest); schedule: tuple of
    (sequence, head) pairs, each twice (arrival, then the store step). Returns (words by slot, fill seen by every CTA, number of stores per sequence)."""
    words = {}
    for slot, _, pos0, _, *residue in seqs:
        if slot not in words:
            words[slot] = pos0 + ((residue[0] if residue else 0) << SHIFT)
    old, seen, stores = {}, {}, [0] * len(seqs)
    for s, c in schedule:
        slot, H, _, length, *_ = seqs[s]
        if (s, c) not in old:                                # the first step of a CTA: its arrival
            old[(s, c)] = words[slot]
            if variant != "every_cta":                       # every_cta: a plain read, no counting
                words[slot] = words[slot] + (1 << SHIFT)     # the atomic add (the result is the word before the add)
            seen[(s, c)] = old[(s, c)] if variant == "no_mask" else (old[(s, c)] & MASK)
        elif elected(variant, old[(s, c)], H, elect):
            stores[s] += 1
            words[slot] = new_pos(seen[(s, c)], length)      # the store: the plain new value (the counter is gone with it)
    return words, seen, stores


def check_one(H, pos0, length, schedule, variant, elect=None, residue=0):
    """The two properties of the protocol for one sequence; returns a description of the first violation or None."""
    words, seen, stores = run_words([(0, H, pos0, length, residue)], [(0, c) for c in schedule], variant, elect)
    for c in range(H):
        if seen[(0, c)] != pos0:
            return "CTA %d started from fill %d, not %d" % (c, seen[(0, c)], pos0)
    if stores[0] != 1:
        return "%d CTAs stored the new fill" % stores[0]
    if words[0] != new_pos(pos0, length):
        return "the word ends as %d (expected the plain %d)" % (words[0], new_pos(pos0, length))
    return None


def violations(H, pos0, length, variant, schedules, elect=None, residue=0):
    return [(s, v) for s in schedules for v in [check_one(H, pos0, length, s, variant, elect, residue)] if v is not None]


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
        started, stored, started_after_store = set(), False, False
        for c in sched:
            if c in started:
                stored = True                     # the second step of a CTA: in this scheme every CTA stores
            else:
                started.add(c)
                if stored:
                    started_after_store = True
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
        _, seen, _ = run_words(seqs, tuple(sched))
        if any(v != 10 for v in seen.values()):
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


# ---- the model is the kernel's: constants and election come from the sources ------------------------------------------------------------------------------------
def test_the_model_constants_are_the_ones_of_the_headers():
    assert (BT, SHIFT, MASK) == (64, 8, 63) and CONSTANTS == ps.constants(ps.load())
    assert MASK < (1 << SHIFT), "the fill must fit below the arrival counter"


def test_the_model_election_is_the_pinned_expression():
    for H in (1, 2, 3, 5, 64):
        assert [c for c in range(H + 3) if PROTOCOL_ELECTS(c, H)] == [H - 1], H


def edited_sources(tmp_path, old, new):
    import test_decode_protocol_source as pins

    return pins.copy_with(tmp_path, ps.load(), [(pins.R, old, new)])


@pytest.mark.parametrize("name,new", [("p.H", "== p.H) st_relaxed_gpu("), ("0", "== 0) st_relaxed_gpu("), ("p.H - 2", "== p.H - 2) st_relaxed_gpu(")])
def test_a_source_with_another_election_makes_the_model_fail(tmp_path, name, new):
    """The model follows the edit: the protocol that the edited source would run is wrong, and the interleavings find it."""
    elect = ps.election(edited_sources(tmp_path, "== p.H - 1) st_relaxed_gpu(", new))
    for H in (2, 3, 4):
        assert violations(H, 12, 1, "protocol", list(interleavings([2] * H)), elect=elect), (name, H)


# ---- the preconditions are necessary: residue of an interrupted call, the capacity of the counter ----------------------------------------------------------------------
def test_arrival_residue_is_outside_the_contract():
    """An interrupted call can leave arrival bits in the word (they are not healed). With residue r the election fires after H - r arrivals (premature: the CTAs that arrive after the
    store see a plain fill again and count anew) and with r >= H nobody is ever elected. Neither is a protocol that works; hence the precondition: pos at rest without residue."""
    H = 3
    scheds = list(interleavings([2] * H))
    for r in (1, 2):
        assert violations(H, 12, 1, "protocol", scheds, residue=r), "residue %d did not break the protocol in any interleaving" % r
    words, seen, stores = run_words([(0, H, 12, 1, 3)], tuple((0, c) for c in [0, 1, 2, 0, 1, 2]))
    assert stores == [0] and words[0] == 12 + ((3 + H) << SHIFT), "residue >= H: nobody stores, the counter just grows"


def i32(x):
    return ((x + 2 ** 31) % 2 ** 32) - 2 ** 31


def last_arrival_is_elected_in_a_32_bit_word(H, fill=63):
    """The arrival of the last CTA of H in a 32-bit signed word: does it see H - 1 above the shift? (The word before that arrival is the fill plus H - 1 additions of 2^SHIFT, reduced to
    32 bits: a sum of adds that wrap equals the wrapped sum.)"""
    old = i32(fill + (H - 1) * (1 << SHIFT))
    return (old >> SHIFT) == H - 1


def test_the_counter_fits_the_word_for_every_head_count_the_binding_admits():
    assert ps.counter_limit_problems(SOURCES) == []
    limit = int(re.search(r"H\s*<\s*(\d+)\s*,\s*\"problem too large", ps.strip_comments(SOURCES["cfbi_binding.cpp"])).group(1))
    assert last_arrival_is_elected_in_a_32_bit_word(limit - 1) and last_arrival_is_elected_in_a_32_bit_word(1)


def test_the_counter_capacity_is_two_to_the_23_heads():
    """The 23 bits above the shift hold the count of CTAs that arrived before: the last CTA of 2^23 heads still sees 2^23 - 1, one head more makes the word negative and nobody is
    elected. The binding admits fewer than 2^16 heads, so the boundary is far away; it is pinned (R10) so that raising the limit or the shift cannot cross it unnoticed."""
    assert last_arrival_is_elected_in_a_32_bit_word(2 ** (31 - SHIFT))
    assert not last_arrival_is_elected_in_a_32_bit_word(2 ** (31 - SHIFT) + 1)
