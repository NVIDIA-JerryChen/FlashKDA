"""The schedule builder of the dynamic-schedule K2 (impl bit 16): ``k2_sched_kernel`` against the host reference ``k2_policy.py``.

The builder reads the plan of a call and writes the table the K2 kernel runs: one packed int per item (pair, value slice, width) in the total order of the
chosen schedule. ``probe_k2_schedule`` (test only) runs the plan kernel and the builder on given inputs, with a test-only SM-count override, and the tests
read the table back. Checked: the device equals the reference for every list of the corpus (kind, widths, item order), the structural invariants for hand
built and seeded random lists (N up to 512, edge lengths, warm calls, padding entries), determinism, the capacity, and that a corrupted table is flagged.
"""
import itertools
import multiprocessing
import os
import random

import numpy as np
import pytest
import torch

import k2_policy as kp
from varlen_corpus import CORPUS_V

BIG = os.environ.get("CFBI_BIG", "0") == "1"
HDR = 16
WCODE = {8: 0, 16: 1, 32: 2}


@pytest.fixture(scope="module")
def dyn(ext):
    if not ext.resolve_impl(16, -1) & 16:
        pytest.skip("the device cannot run the device-scheduled K2 (impl bit 16 is dropped here)")
    return ext


class Sched(object):
    """A schedule read back from the plan buffer: ``kind``, ``items`` [(pair, slice, width)] (decoded on demand), ``header``, ``pad`` (table entries
    beyond the count, which must hold the sentinel)."""

    def __init__(self, plan, N, H):
        base = (N + 1) * 8
        self.header = plan[base:base + HDR].tolist()
        self.count = self.header[0]
        self.kind = "old" if self.header[1] == 0 else "sorted"
        raw = plan[base + HDR: base + HDR + 16 * N * H]
        self.raw_items = raw[:self.count]
        self.pad = raw[self.count:]

    @property
    def items(self):
        return [(it & ((1 << 25) - 1), (it >> 25) & 15, 8 << ((it >> 29) & 3)) for it in self.raw_items.tolist()]


_PLANS = {}


def probe(ext, lens=None, H=1, N=None, fixed_len=0, pos=None, slot_ids=None, sms=0, k2_bv=0):
    """Run the plan kernel + schedule builder on a variable-length call (``lens``) or a fixed-length one (``N`` sequences of ``fixed_len`` tokens)."""
    cu = None
    if lens is not None:
        N = len(lens)
        cu = torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int64, device="cuda")
    size = ext.workspace_sizes(1, N, H)[1]
    plan = _PLANS.get(size)
    if plan is None:
        plan = _PLANS[size] = torch.empty(size, dtype=torch.int32, device="cuda")
    plan.fill_(-7)    # a builder that leaves part of its region unwritten shows up
    ext.probe_k2_schedule(plan, H, N, fixed_len, cu, pos, slot_ids, sms, k2_bv)
    return Sched(plan.cpu(), N, H)


def reference(lens, H, sms, call_fixed=False, k2_bv=0, pos0s=None, slots=None):
    c = [kp.chunk_count(n, 0 if pos0s is None else pos0s[i]) if (slots is None or slots[i] >= 0) else 0 for i, n in enumerate(lens)]
    return c, kp.build_schedule(c, H, sms, call_fixed=call_fixed, forced_bv=k2_bv)


def as_ints(items):
    return np.array([p | (s << 25) | (WCODE[w] << 29) for p, s, w in items], dtype=np.int64)


def assert_matches(dev, c, ref, H, what=""):
    items = dev.items
    assert dev.kind == ref.kind, (what, dev.kind, ref.kind)
    assert items == ref.items, (what, "first difference at %s" % next((i for i, (a, b) in enumerate(zip(items, ref.items)) if a != b), min(len(items), len(ref.items))))
    assert (dev.pad == -1).all(), (what, "table entries beyond the count must hold the sentinel")
    assert dev.count == len(ref.items) and dev.header[2] == len({p for p, _, _ in ref.items}) and dev.header[6] == sum(1 for x in c if x > 0), (what, dev.header)
    assert all(v == 0 for v in dev.header[7:]), (what, "unused header ints are zero")
    assert kp.check_invariants(items, c, H, dev.kind) == [], what


# ---- the device against the reference on the corpus -------------------------------------------------------------------------------------------------
def _ref_case(arg):
    name, lens, H, sms = arg
    _, s = reference(lens, H, sms)
    return as_ints(s.items), s.kind


def test_device_schedule_equals_the_reference_on_corpus_p(dyn):
    lists = kp.corpus_p()
    cases = [(e["name"], e["lengths"], H, sms) for e in lists for H in kp.CORPUS_H for sms in kp.CORPUS_SMS if BIG or e["N"] * H <= 768]
    assert len(cases) >= (16000 if BIG else 14000)
    with multiprocessing.Pool(min(32, os.cpu_count() or 1)) as pool:
        refs = pool.map(_ref_case, cases, chunksize=16)
    for (name, lens, H, sms), (ref_items, ref_kind) in zip(cases, refs):
        dev = probe(dyn, lens, H, sms=sms)
        assert dev.kind == ref_kind and np.array_equal(dev.raw_items.numpy().astype(np.int64), ref_items), (name, H, sms)
        assert (dev.pad == -1).all()


def test_named_corpus_v_entries_match_the_reference_for_every_width_setting(dyn):
    for name, e in CORPUS_V.items():
        for sms, bv in itertools.product((0, 132, 144, 148), (0, 8, 16, 32)):
            c, ref = reference(e.lens, e.H, sms or torch.cuda.get_device_properties(0).multi_processor_count, k2_bv=bv)
            assert_matches(probe(dyn, e.lens, e.H, sms=sms, k2_bv=bv), c, ref, e.H, (name, sms, bv))


def test_probe_with_the_device_sm_count_equals_the_override_of_that_count(dyn):
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    lens = CORPUS_V["V4"].lens
    a, b = probe(dyn, lens, 6, sms=0), probe(dyn, lens, 6, sms=sms)
    assert a.items == b.items and a.kind == b.kind


def test_forced_width_keeps_the_device_order_and_every_pair_one_width(dyn):
    lens = CORPUS_V["V4"].lens
    for bv in (8, 16, 32):
        d = probe(dyn, lens, 6, k2_bv=bv)
        assert {w for _, _, w in d.items} == {bv} and d.kind == "sorted" and len(d.items) == 48 * (128 // bv)
        keys = [(-kp.chunk_count(lens[p % 8]) * kp.T_CHUNK[bv], p, s) for p, s, _ in d.items]
        assert keys == sorted(keys)


# ---- invariants: hand-built and seeded random lists, edge lengths, warm calls, padding entries ----------------------------------------------------
EDGE = [0, 1, 15, 16, 17, 63, 64, 65, 129]


def test_edge_lengths_tail_only_warm_calls_and_padding_entries(dyn):
    H = 3
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    lens = EDGE + [64 * 5 + 3, 200, 1]
    N = len(lens)
    # cold call
    c, ref = reference(lens, H, sms)
    d = probe(dyn, lens, H)
    assert_matches(d, c, ref, H, "cold")
    assert not any(c[p % N] == 0 for p, _, _ in d.items), "no entry may exist for a zero-chunk sequence"
    # tail-only call: every sequence shorter than a chunk
    short = [1, 5, 63, 17, 9, 33, 63, 2]
    c, ref = reference(short, 6, sms)
    assert_matches(probe(dyn, short, 6), c, ref, 6, "tail only")
    # warm call: sequences start inside an open chunk (pos != 0); a state slot per sequence, slots distinct but not contiguous, two padding entries
    slots = [7, 3, -1, 12, 0, 9, -1, 5, 20, 11, 2, 8]
    pos = torch.zeros(21, dtype=torch.int32)
    pos0 = [0, 5, 30, 63, 1, 0, 40, 10, 17, 0, 64 - 1, 2]
    for i, sl in enumerate(slots):
        if sl >= 0:
            pos[sl] = pos0[i]
    d = probe(dyn, lens, H, pos=pos.cuda(), slot_ids=torch.tensor(slots, dtype=torch.int32, device="cuda"))
    c, ref = reference(lens, H, sms, pos0s=pos0, slots=slots)
    assert_matches(d, c, ref, H, "warm + padding")
    assert not any(slots[p % N] < 0 for p, _, _ in d.items), "padding entries (slot -1) get no work"
    # all sequences skipped: an empty table
    d = probe(dyn, [10, 20], 2, slot_ids=torch.tensor([-1, -1], dtype=torch.int32, device="cuda"), pos=torch.zeros(2, dtype=torch.int32, device="cuda"))
    assert d.count == 0 and d.items == [] and (d.pad == -1).all()


def test_fixed_length_calls_use_the_closed_form(dyn):
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    for N, L, H in ((8, 8192, 6), (13, 512, 3), (19, 100, 6), (5, 64, 32), (3, 1, 1)):
        d = probe(dyn, None, H, N=N, fixed_len=L)
        c, ref = reference([L] * N, H, sms, call_fixed=True)
        assert_matches(d, c, ref, H, (N, L, H))
        assert d.kind == "old"      # no padding entries: the schedule of a fixed-length call is the old one


# ---- fixed-length calls with warm positions, skipped slots and non-contiguous slot ids ---------------------------------------------------------------
def fixed_probe(dyn, N, L, H, pos0, slots=None, sms=0, k2_bv=0):
    """Probe a fixed-length call (N sequences of L tokens, no cu_seqlens) whose sequences start at the open-chunk fills ``pos0``, read from the position array of a
    state pool through the slots ``slots`` (-1: padding entry, no work; None: sequence i uses slot i); returns the device schedule, the chunk counts and the reference
    schedule."""
    dev_sms = sms or torch.cuda.get_device_properties(0).multi_processor_count
    ids = list(range(N)) if slots is None else slots
    pos = torch.full((max(ids) + 4,), 11, dtype=torch.int32)    # a slot no sequence uses holds a nonzero fill that must not be read
    for sl, p0 in zip(ids, pos0):
        if sl >= 0:
            pos[sl] = p0
    slot_t = None if slots is None else torch.tensor(slots, dtype=torch.int32, device="cuda")
    d = probe(dyn, None, H, N=N, fixed_len=L, pos=pos.cuda(), slot_ids=slot_t, sms=sms, k2_bv=k2_bv)
    c, ref = reference([L] * N, H, dev_sms, call_fixed=True, k2_bv=k2_bv, pos0s=pos0, slots=slots)
    return d, c, ref


def assert_old_layout_header(d, c, N, H, sms):
    """The header of the old schedule of a fixed-length call: CHOICE_A / CHOICE_B / CHOICE_X = 32 columns (0 when no pair has them) / the width of the other pairs / the
    number of active pairs among the first x32 of the N * H pair positions."""
    x32, bvb = kp.old_rule(N, H, sms, True)
    ax = sum(1 for p in range(x32) if c[p % N] > 0)
    assert d.header[3:6] == [32 if ax else 0, bvb, ax], (d.header, x32, bvb, ax)


# open-chunk fills that end in two different chunk counts: L % 64 == 40, so the fills 1, 5, 12, 17 leave q chunks and 0, 33, 40, 63 leave q + 1
FIXED_SHAPES = [(8, 4136, 6), (8, 8232, 6), (13, 1064, 3), (19, 744, 6), (5, 296, 32), (16, 2600, 12), (3, 360, 1), (10, 424, 1), (6, 424, 6), (4, 616, 12),
                (64, 296, 6), (100, 232, 3), (257, 168, 3), (512, 104, 2)]
FILLS = (1, 33, 17, 0, 63, 12, 40, 5)


def test_fixed_length_calls_with_warm_positions_build_the_exact_old_schedule(dyn):
    """Unequal active chunk counts (warm sequences, some skipped, slots not contiguous): the schedule of the old fixed-length launcher over all N * H pair positions."""
    rng = random.Random(20261108)
    layouts = set()
    for N, L, H in FIXED_SHAPES:
        for sms in (0, 132, 144, 148):
            dev_sms = sms or torch.cuda.get_device_properties(0).multi_processor_count
            x32, bvb = kp.old_rule(N, H, dev_sms, True)
            if x32 == 0:
                layouts.add("x32=0/bvb=%d" % bvb)
            elif x32 == N * H:
                layouts.add("all 32")
            else:
                layouts.add("mixed")
            # (a) every sequence active, slot i for sequence i: the fills alternate over the two chunk counts
            pos0 = [FILLS[i % 8] for i in range(N)]
            d, c, ref = fixed_probe(dyn, N, L, H, pos0, None, sms)
            assert len({v for v in c if v}) == 2 and all(c), (N, L, H, c[:8])
            assert_matches(d, c, ref, H, ("warm", N, L, H, sms))
            assert d.kind == "old"
            assert_old_layout_header(d, c, N, H, dev_sms)
            # (b) skipped slots (-1) and slots that are neither contiguous nor in order
            if N >= 3:
                slots = rng.sample(range(2 * N + 3), N)
                for i in rng.sample(range(N), max(1, N // 4)):
                    slots[i] = -1
                pos0 = [rng.choice(FILLS) for _ in range(N)]
                act = [i for i in range(N) if slots[i] >= 0]
                pos0[act[0]], pos0[act[-1]] = 1, 33                     # two chunk counts among the active sequences
                d, c, ref = fixed_probe(dyn, N, L, H, pos0, slots, sms)
                assert len({v for v in c if v}) == 2 and not all(c), (N, L, H, c[:8])
                assert_matches(d, c, ref, H, ("warm + skipped", N, L, H, sms))
                assert d.kind == "old"
                assert_old_layout_header(d, c, N, H, dev_sms)
                assert not any(slots[p % N] < 0 for p, _, _ in d.items), "padding entries (slot -1) get no work"
    assert layouts == {"x32=0/bvb=8", "x32=0/bvb=16", "mixed", "all 32"}, layouts        # every kind of old layout was exercised on the device


def test_fixed_length_calls_with_equal_warm_positions_and_skipped_slots(dyn):
    """Equal active chunk counts with skipped slots: the closed form of the active pairs (the old schedule when it coincides with the launcher's layout)."""
    rng = random.Random(20261109)
    kinds = set()
    for N, L, H in FIXED_SHAPES + [(1, 4136, 10), (2, 4136, 20)]:
        for sms in (0, 132, 148):
            for fill in (17, 33):                                          # one chunk count (q for 17, q + 1 for 33) for all
                slots = rng.sample(range(2 * N + 3), N)
                for i in rng.sample(range(N), N // 3):
                    slots[i] = -1
                d, c, ref = fixed_probe(dyn, N, L, H, [fill] * N, slots, sms)
                assert len({v for v in c if v}) == 1, c
                assert_matches(d, c, ref, H, ("equal warm + skipped", N, L, H, sms, fill))
                kinds.add(d.kind)
    assert kinds == {"old", "sorted"}, kinds


def test_forced_widths_on_fixed_length_calls_with_unequal_positions(dyn):
    N, L, H = 8, 4136, 6
    pos0 = [FILLS[i % 8] for i in range(N)]
    for bv in (8, 16, 32):
        d, c, ref = fixed_probe(dyn, N, L, H, pos0, [5, 1, -1, 9, 3, 14, 0, -1], k2_bv=bv)
        assert {w for _, _, w in d.items} == {bv} and d.kind == "sorted"
        assert_matches(d, c, ref, H, ("forced", bv))


def _random_case(rng):
    N = rng.choice([1, 2, 3, 5, 8, 13, 16, 19, 32, 64, 100, 257, 512])
    H = rng.choice([1, 3, 6, 33])
    N = min(N, kp.schedule_capacity(H))
    kind = rng.choice(["uniform", "lognormal", "edge", "onelong", "mixedzero"])
    if kind == "uniform":
        lens = [rng.randint(1, 140) * 64 + rng.choice([0, 0, 5, 63]) for _ in range(N)]
    elif kind == "lognormal":
        lens = [max(1, min(128, int(rng.lognormvariate(4.2, 1.0)))) * 64 for _ in range(N)]
    elif kind == "edge":
        lens = [rng.choice(EDGE) for _ in range(N)]
    elif kind == "onelong":
        lens = [rng.choice([512, 4096, 32768])] + [rng.choice([64, 128, 1024]) for _ in range(N - 1)]
    else:
        lens = [rng.choice([0, 0, 1, 64, 700, 4000]) for _ in range(N)]
    return lens, H


def test_seeded_random_lists_satisfy_the_invariants_and_match_the_reference(dyn):
    rng = random.Random(20261104)
    sms_all = (132, 144, 148)
    n = 0
    for _ in range(200):
        lens, H = _random_case(rng)
        sms = rng.choice(sms_all)
        bv = rng.choice([0, 0, 0, 8, 16, 32])
        c, ref = reference(lens, H, sms, k2_bv=bv)
        assert_matches(probe(dyn, lens, H, sms=sms, k2_bv=bv), c, ref, H, (len(lens), H, sms, bv))
        n += 1
    assert n == 200


def test_two_builds_of_the_same_input_are_identical(dyn):
    lens = [1536, 2368, 5312, 2432, 8192, 5632, 8192, 4928, 64, 0, 700]
    a, b = probe(dyn, lens, 6), probe(dyn, lens, 6)
    assert a.items == b.items and a.header == b.header


# ---- capacity, sizes, negative checks --------------------------------------------------------------------------------------------------------------
def test_capacity_and_workspace_sizes(dyn):
    assert all(dyn.sched_capacity(H) >= 512 for H in range(1, 34))
    for H in (1, 6, 33, 34, 64, 1000):
        assert dyn.sched_capacity(H) == kp.schedule_capacity(H)
    for N, H in ((1, 1), (8, 6), (512, 33), (1024, 6)):
        assert dyn.workspace_sizes(8192, N, H)[1] == (N + 1) * 8 + 16 + 16 * N * H
    cap = dyn.sched_capacity(33)
    assert dyn.workspace_sizes(8192, cap + 1, 33)[1] == (cap + 2) * 8      # beyond the capacity: the plan entries only, the old kernels run
    with pytest.raises(RuntimeError):
        dyn.probe_k2_schedule(torch.empty(1024, dtype=torch.int32, device="cuda"), 33, cap + 1, 64, None, None, None, 0, 0)


def test_corrupted_device_schedule_is_flagged(dyn):
    lens = CORPUS_V["V3"].lens
    c, ref = reference(lens, 6, 148)
    d = probe(dyn, lens, 6, sms=148)
    assert kp.check_invariants(d.items, c, 6, d.kind) == []
    assert any("duplicate" in p for p in kp.check_invariants(d.items + [d.items[3]], c, 6, d.kind))
    assert any("missing" in p for p in kp.check_invariants(d.items[:9] + d.items[10:], c, 6, d.kind))
    assert any("total order" in p for p in kp.check_invariants(d.items[::-1], c, 6, d.kind))
