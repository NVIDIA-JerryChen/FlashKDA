"""State-pool indirection (`slot_ids`): sequence n of a call uses pool slot slot_ids[n] (-1 = padding entry).

Gate: running through the pool is bit-identical to gathering the slots into a compact state, running without
slot_ids and scattering back; untouched slots stay untouched; padding entries do nothing; the indirection is
resolved on the device (CUDA graph replays with a changed slot_ids buffer); both kernel implementations.
"""
import random

import pytest
import torch

from flash_kda.cfbi import alloc_state, decode, prefill, state_equal, state_mismatch

from helpers import make_case, n_diff


def _cu(lens):
    cu = [0]
    for L in lens:
        cu.append(cu[-1] + L)
    return cu


def _slice_call(case, cu, done, take):
    pieces = [{k: case[k][:, cu[n] + done[n]:cu[n] + done[n] + take[n]] for k in ("q", "k", "v", "g", "beta")}
              for n in range(len(take))]
    sub = {k: torch.cat([p[k] for p in pieces], dim=1).contiguous() for k in ("q", "k", "v", "g", "beta")}
    return sub, torch.tensor(_cu(take), dtype=torch.int64, device="cuda")


def _run(case, sub, cu_call, state, slot_ids, impl):
    out = torch.zeros_like(sub["q"])
    out32 = torch.zeros(sub["q"].shape, dtype=torch.float32, device="cuda")
    prefill(sub["q"], sub["k"], sub["v"], sub["g"], sub["beta"], case["scale"], out, case["A_log"], case["dt_bias"],
            case["lower_bound"], state, cu_call, out_fp32=out32, slot_ids=slot_ids, _impl=impl)
    return out, out32


@pytest.mark.parametrize("impl", [0, 3])
@pytest.mark.parametrize("seed", [0, 1])
def test_pool_equals_gather_scatter(ext, impl, seed):
    H, S = 3, 11
    rng = random.Random(40 + seed)
    N = 5
    first = [rng.randint(1, 100) for _ in range(N)]
    second = [rng.choice([0, 1, 33, 64, 90]) for _ in range(N)]
    lens = [f + s_ + rng.randint(0, 40) for f, s_ in zip(first, second)]   # every call slices inside its sequence
    case = make_case(sum(lens), H, "authors", seed=60 + seed)
    cu = _cu(lens)
    slots = rng.sample(range(S), N)
    ids = torch.tensor(slots, dtype=torch.int32, device="cuda")

    pool = alloc_state(S, H)
    # warm the chosen slots through the pool, also warm one slot that is NOT used by the second call
    bystander = [s for s in range(S) if s not in slots][0]
    sub0, cu0 = _slice_call(case, cu, [0] * N, first)
    _run(case, sub0, cu0, pool, ids, impl)
    compact = alloc_state(N, H)
    _run(case, sub0, cu0, compact, None, impl)
    assert state_equal(pool.select(ids.long()), compact), "first call through the pool differs from the compact state"
    assert int(pool.pos[bystander]) == 0 and not bool(pool.E[bystander].any()), "untouched slot was written"

    # second call: through the pool vs gather -> run -> scatter
    sub1, cu1 = _slice_call(case, cu, first, second)
    before = pool.clone()
    o_pool, f_pool = _run(case, sub1, cu1, pool, ids, impl)
    gathered = before.select(ids.long())
    o_ref, f_ref = _run(case, sub1, cu1, gathered, None, impl)
    assert n_diff(o_pool, o_ref) == 0 and n_diff(f_pool, f_ref) == 0
    assert state_equal(pool.select(ids.long()), gathered), state_mismatch(pool.select(ids.long()), gathered)
    for s in range(S):
        if s not in slots:
            assert state_equal(pool.select(torch.tensor([s], device="cuda")), before.select(torch.tensor([s], device="cuda")))


@pytest.mark.parametrize("impl", [0, 3])
def test_negative_slot_is_skipped(ext, impl):
    H, S = 2, 6
    lens = [150, 40, 90, 64]
    case = make_case(sum(lens), H, "authors", seed=70)
    cu = _cu(lens)
    sub, cu_call = _slice_call(case, cu, [0] * 4, lens)
    ids = torch.tensor([2, -1, 5, -1], dtype=torch.int32, device="cuda")
    pool = alloc_state(S, H)
    out = torch.full_like(sub["q"], 1.0)   # sentinel in the rows of the skipped sequences
    out32 = torch.full(sub["q"].shape, 7.0, dtype=torch.float32, device="cuda")
    prefill(sub["q"], sub["k"], sub["v"], sub["g"], sub["beta"], case["scale"], out, case["A_log"], case["dt_bias"],
            case["lower_bound"], pool, cu_call, out_fp32=out32, slot_ids=ids, _impl=impl)
    for n in (1, 3):
        assert bool((out[:, cu[n]:cu[n + 1]] == 1.0).all()) and bool((out32[:, cu[n]:cu[n + 1]] == 7.0).all()), \
            f"padding entry {n} wrote outputs"
    for s in (0, 1, 3, 4):
        assert int(pool.pos[s]) == 0 and not bool(pool.E[s].any()) and not bool(pool.rows[s].any()), f"slot {s} was touched"
    # the real sequences are unaffected by their padded neighbours
    for n, s in ((0, 2), (2, 5)):
        solo = alloc_state(1, H)
        sq = {k: case[k][:, cu[n]:cu[n + 1]].contiguous() for k in ("q", "k", "v", "g", "beta")}
        so = torch.zeros_like(sq["q"])
        so32 = torch.zeros(sq["q"].shape, dtype=torch.float32, device="cuda")
        prefill(sq["q"], sq["k"], sq["v"], sq["g"], sq["beta"], case["scale"], so, case["A_log"], case["dt_bias"],
                case["lower_bound"], solo, out_fp32=so32, _impl=impl)
        assert n_diff(out[:, cu[n]:cu[n + 1]], so) == 0 and n_diff(out32[:, cu[n]:cu[n + 1]], so32) == 0
        assert state_equal(pool.select(torch.tensor([s], device="cuda")), solo)


def test_decode_pool_cuda_graph_changing_slots(ext):
    """Fixed-shape decode batch (N = 5 entries, one padding), slot_ids rewritten between graph replays."""
    H, S, steps = 2, 8, 50
    N = 5
    lens = [steps] * 4
    case = make_case(sum(lens), H, "authors", seed=80)
    cu = _cu(lens)
    # reference: every sequence alone, whole
    refs = []
    for n in range(4):
        st = alloc_state(1, H)
        sq = {k: case[k][:, cu[n]:cu[n + 1]].contiguous() for k in ("q", "k", "v", "g", "beta")}
        o = torch.zeros_like(sq["q"])
        o32 = torch.zeros(sq["q"].shape, dtype=torch.float32, device="cuda")
        prefill(sq["q"], sq["k"], sq["v"], sq["g"], sq["beta"], case["scale"], o, case["A_log"], case["dt_bias"],
                case["lower_bound"], st, out_fp32=o32)
        refs.append((o, o32, st))
    pool = alloc_state(S, H)
    bufs = {k: torch.zeros(N, 1, *case[k].shape[2:], dtype=case[k].dtype, device="cuda") for k in ("q", "k", "v", "g", "beta")}
    o = torch.zeros_like(bufs["q"])
    o32 = torch.zeros(bufs["q"].shape, dtype=torch.float32, device="cuda")
    ids = torch.full((N,), -1, dtype=torch.int32, device="cuda")

    def launch():
        decode(bufs["q"], bufs["k"], bufs["v"], bufs["g"], bufs["beta"], case["scale"], o, case["A_log"], case["dt_bias"],
               case["lower_bound"], pool, out_fp32=o32, slot_ids=ids)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):   # warm-up on a scratch pool (attribute setup, allocations)
        saved = pool
        pool = alloc_state(S, H)
        launch()
        pool = saved
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        launch()
    rng = random.Random(5)
    slots = rng.sample(range(S), 4)     # sequence n -> pool slot (fixed for the sequence's lifetime)
    for t in range(steps):
        # entry order and padding change every step; the slot of a sequence never does
        order = list(range(4)) + [None]
        rng.shuffle(order)
        for i, n in enumerate(order):
            if n is None:
                ids[i] = -1
                for k in bufs:
                    bufs[k][i].zero_()
            else:
                ids[i] = slots[n]
                for k in bufs:
                    bufs[k][i, 0] = case[k][0, cu[n] + t]
        o.fill_(0), o32.fill_(0)
        g.replay()
        for i, n in enumerate(order):
            if n is not None:
                assert torch.equal(o[i, 0].view(torch.int16), refs[n][0][0, t].view(torch.int16)), f"step {t} seq {n}"
                assert torch.equal(o32[i, 0].view(torch.int32), refs[n][1][0, t].view(torch.int32)), f"step {t} seq {n}"
    for n in range(4):
        assert state_equal(pool.select(torch.tensor([slots[n]], device="cuda")), refs[n][2]), f"final state of sequence {n}"


def test_bad_slot_arguments(ext):
    H = 2
    case = make_case(64, H, "authors", seed=90)
    out = torch.zeros_like(case["q"])
    pool = alloc_state(4, H)
    with pytest.raises(RuntimeError, match="slot_ids"):   # wrong length
        prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], 0.1, out, case["A_log"], case["dt_bias"], -5.0,
                pool, slot_ids=torch.tensor([0, 1], dtype=torch.int32, device="cuda"))
    with pytest.raises(RuntimeError, match="slot_ids"):   # wrong dtype
        prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], 0.1, out, case["A_log"], case["dt_bias"], -5.0,
                pool, slot_ids=torch.tensor([0], dtype=torch.int64, device="cuda"))
    with pytest.raises(RuntimeError, match="must be"):    # a pool needs slot_ids; without them E must be [N, ...]
        prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], 0.1, out, case["A_log"], case["dt_bias"], -5.0, pool)
