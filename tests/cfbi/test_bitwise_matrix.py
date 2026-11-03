"""M4 gate: the bitwise matrix. Zero tolerance: every comparison is on bits (bf16 outputs, the fp32 values before
the bf16 rounding, and the state: E, the valid open-chunk rows, pos).

Axes: whole prefill vs T=1 x n decode, random cuts (incl. non-multiples of 64), a continuous-batching scheduler
simulation (idle slots, mixed decode/prefill rows in one call), head sharding (train TP != infer TP), a large
batch vs every sequence alone, CUDA-graph replay, run-to-run determinism, and a NEGATIVE CONTROL kernel (one
fma replaced by mul+sub) that this harness must catch. CFBI_BIG=1 runs the 32K-token / N=512 variants.
"""
import os
import random

import pytest
import torch

from flash_kda.cfbi import CfbiState, alloc_state, decode, forward_train, prefill, state_equal, state_mismatch
from flash_kda.cfbi._ext import load_variant

from helpers import REGIMES, make_case, n_diff, split_tokens

BIG = os.environ.get("CFBI_BIG", "0") == "1"
PIECE_LENGTHS = [1, 2, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 200]


def _out_buffers(like):
    return torch.zeros_like(like), torch.zeros(like.shape, dtype=torch.float32, device=like.device)


def call(case, state, cu=None, mode=0, mod=None):
    out, out32 = _out_buffers(case["q"])
    if mod is None:
        prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], out, case["A_log"],
                case["dt_bias"], case["lower_bound"], state, cu, out_fp32=out32, _mode=mode)
    else:   # a differently built extension (negative control)
        mod.forward(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], out, case["A_log"],
                    case["dt_bias"], case["lower_bound"], state.E, state.rows, state.pos, cu, mode, out32)
    return out, out32


def assert_same(a, b, what):
    """a, b = (out, out32, state)."""
    assert n_diff(a[0], b[0]) == 0, f"{what}: bf16 outputs differ in {n_diff(a[0], b[0])} of {a[0].numel()} elements"
    assert n_diff(a[1], b[1]) == 0, f"{what}: fp32 outputs differ in {n_diff(a[1], b[1])} of {a[1].numel()} elements"
    assert state_equal(a[2], b[2]), f"{what}: state differs {state_mismatch(a[2], b[2])}"


def whole(case, H):
    st = alloc_state(1, H)
    o, o32 = call(case, st)
    return o, o32, st


def random_pieces(T, rng):
    pieces, rem = [], T
    while rem > 0:
        L = min(rem, rng.choice(PIECE_LENGTHS + [rng.randint(1, 150)]))
        pieces.append(L)
        rem -= L
    return pieces


def run_pieces(case, pieces, H):
    st = alloc_state(1, H)
    out, out32 = _out_buffers(case["q"])
    lo = 0
    for L in pieces:
        o, o32 = call(split_tokens(case, lo, lo + L), st)
        out[:, lo:lo + L], out32[:, lo:lo + L] = o, o32
        lo += L
    return out, out32, st


# ---------------------------------------------------------------------------------------------------------
# whole call vs T=1 decode calls vs random cuts
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("regime", REGIMES)
def test_whole_vs_decode_vs_cuts(ext, regime):
    H, T = 2, 333
    case = make_case(T, H, regime, seed=11)
    ref = whole(case, H)

    st = alloc_state(1, H)   # T=1 x n through the decode wrapper
    out, out32 = _out_buffers(case["q"])
    for t in range(T):
        decode(case["q"][:, t:t + 1], case["k"][:, t:t + 1], case["v"][:, t:t + 1], case["g"][:, t:t + 1],
               case["beta"][:, t:t + 1], case["scale"], out[:, t:t + 1], case["A_log"], case["dt_bias"],
               case["lower_bound"], st, out_fp32=out32[:, t:t + 1])
    assert_same(ref, (out, out32, st), "decode x n")

    for seed in range(6):
        pieces = random_pieces(T, random.Random(seed))
        assert_same(ref, run_pieces(case, pieces, H), f"cuts {pieces}")


def test_every_single_cut_point(ext):
    """Two calls cut at every position of two chunk boundaries' neighbourhood."""
    H, T = 1, 200
    case = make_case(T, H, "authors", seed=12)
    ref = whole(case, H)
    for cut in list(range(1, 70)) + [127, 128, 129, 191, 192, 193, 199]:
        assert_same(ref, run_pieces(case, [cut, T - cut], H), f"cut at {cut}")


# ---------------------------------------------------------------------------------------------------------
# multiple sequences: varlen call vs solo, N large, scheduler simulation, head sharding
# ---------------------------------------------------------------------------------------------------------
def make_multi(lens, H, regime="authors", seed=0):
    case = make_case(sum(lens), H, regime, seed=seed)
    cu = [0]
    for L in lens:
        cu.append(cu[-1] + L)
    return case, cu


def cu_tensor(cu):
    return torch.tensor(cu, dtype=torch.int64, device="cuda")


def solo_runs(case, cu, H):
    return [whole(split_tokens(case, cu[n], cu[n + 1]), H) for n in range(len(cu) - 1)]


def check_batch_vs_solo(case, cu, H, label):
    N = len(cu) - 1
    st = alloc_state(N, H)
    out, out32 = call(case, st, cu_tensor(cu))
    for n, (so, so32, sst) in enumerate(solo_runs(case, cu, H)):
        lo, hi = cu[n], cu[n + 1]
        sel = st.select([n])
        assert_same((out[:, lo:hi], out32[:, lo:hi], sel), (so, so32, sst), f"{label}: sequence {n} (len {hi - lo})")


@pytest.mark.parametrize("regime", ["authors", "heavy_tail", "long_memory"])
def test_batch_vs_solo(ext, regime):
    rng = random.Random(5)
    lens = [rng.randint(1, 260) for _ in range(12)] + [64, 63, 65, 1, 128]
    case, cu = make_multi(lens, 2, regime, seed=21)
    check_batch_vs_solo(case, cu, 2, "batch vs solo")


def test_large_batch_vs_solo(ext):
    N = 512 if BIG else 64
    rng = random.Random(6)
    lens = [rng.choice([1, 1, 1, 2, 7, 33, 64, 65, 130, rng.randint(1, 300)]) for _ in range(N)]
    case, cu = make_multi(lens, 2 if not BIG else 4, "authors", seed=22)
    check_batch_vs_solo(case, cu, 2 if not BIG else 4, f"N={N} vs solo")


def test_batch_order_and_company_do_not_matter(ext):
    """The same sequence next to different neighbours / at a different index: same bits."""
    H = 2
    rng = random.Random(7)
    lens = [rng.randint(1, 200) for _ in range(8)]
    case, cu = make_multi(lens, H, "authors", seed=23)
    N = len(lens)
    st = alloc_state(N, H)
    out, out32 = call(case, st, cu_tensor(cu))
    perm = list(range(N))
    random.Random(8).shuffle(perm)
    # same sequences, packed in permuted order
    parts = [split_tokens(case, cu[n], cu[n + 1]) for n in perm]
    case_p = dict(case)
    for key in ("q", "k", "v", "g", "beta"):
        case_p[key] = torch.cat([p[key] for p in parts], dim=1).contiguous()
    cu_p = [0]
    for n in perm:
        cu_p.append(cu_p[-1] + lens[n])
    st_p = alloc_state(N, H)
    out_p, out32_p = call(case_p, st_p, cu_tensor(cu_p))
    for i, n in enumerate(perm):
        a = (out[:, cu[n]:cu[n + 1]], out32[:, cu[n]:cu[n + 1]], st.select([n]))
        b = (out_p[:, cu_p[i]:cu_p[i + 1]], out32_p[:, cu_p[i]:cu_p[i + 1]], st_p.select([i]))
        assert_same(a, b, f"sequence {n} at index {i}")


def scheduler_sim(case, cu, H, rng):
    """Continuous batching: every step each sequence gets 0 (idle), 1 (decode) or a random prefill chunk."""
    N = len(cu) - 1
    lens = [cu[n + 1] - cu[n] for n in range(N)]
    done = [0] * N
    st = alloc_state(N, H)
    out, out32 = _out_buffers(case["q"])
    steps = 0
    while any(done[n] < lens[n] for n in range(N)):
        steps += 1
        take = []
        for n in range(N):
            r = lens[n] - done[n]
            c = rng.choices([0, 1, rng.randint(2, 130)], weights=[3, 4, 3])[0]
            take.append(min(c, r))
        if sum(take) == 0:
            continue
        active = [n for n in range(N) if take[n] > 0]
        pure_decode = all(take[n] == 1 for n in active) and rng.random() < 0.6
        if pure_decode:   # compact the active slots, run the T=1 decode wrapper, scatter back
            ids = torch.tensor(active, device="cuda")
            sub = st.select(ids)
            idx = [cu[n] + done[n] for n in active]
            parts = {key: torch.cat([case[key][:, i:i + 1] for i in idx], dim=0).contiguous()
                     for key in ("q", "k", "v", "g", "beta")}
            o, o32 = torch.zeros_like(parts["q"]), torch.zeros(parts["q"].shape, dtype=torch.float32, device="cuda")
            decode(parts["q"], parts["k"], parts["v"], parts["g"], parts["beta"], case["scale"], o, case["A_log"],
                   case["dt_bias"], case["lower_bound"], sub, out_fp32=o32)
            st.E[ids], st.rows[ids], st.pos[ids] = sub.E, sub.rows, sub.pos
            for j, n in enumerate(active):
                out[:, idx[j]], out32[:, idx[j]] = o[j], o32[j]
        else:   # one varlen prefill call over all slots (idle slots have length 0)
            pieces = [split_tokens(case, cu[n] + done[n], cu[n] + done[n] + take[n]) for n in active]
            sub_case = dict(case)
            for key in ("q", "k", "v", "g", "beta"):
                sub_case[key] = torch.cat([p[key] for p in pieces], dim=1).contiguous()
            cu_call = [0]
            for n in range(N):
                cu_call.append(cu_call[-1] + take[n])
            o, o32 = call(sub_case, st, cu_tensor(cu_call))
            p = 0
            for n in active:
                lo = cu[n] + done[n]
                out[:, lo:lo + take[n]], out32[:, lo:lo + take[n]] = o[:, p:p + take[n]], o32[:, p:p + take[n]]
                p += take[n]
        for n in range(N):
            done[n] += take[n]
    return out, out32, st, steps


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_scheduler_simulation(ext, seed):
    H, N = 2, 7
    rng = random.Random(100 + seed)
    lens = [rng.randint(5, 420) for _ in range(N)]
    case, cu = make_multi(lens, H, "authors", seed=30 + seed)
    st = alloc_state(N, H)
    ref_out, ref_out32 = call(case, st, cu_tensor(cu))
    out, out32, st_sim, steps = scheduler_sim(case, cu, H, rng)
    assert_same((ref_out, ref_out32, st), (out, out32, st_sim), f"scheduler sim ({steps} steps)")


def test_head_sharding(ext):
    """Train TP != infer TP: any head grouping gives the same bits per head."""
    H, T = 6, 150
    case = make_case(T, H, "authors", seed=40)
    ref = whole(case, H)
    for groups in ([[0, 1, 2], [3, 4, 5]], [[0], [1], [2], [3], [4], [5]], [[0, 5], [1, 2, 3, 4]]):
        out, out32 = _out_buffers(case["q"])
        sts = []
        for hs in groups:
            sub = dict(case)
            for key in ("q", "k", "v", "g"):
                sub[key] = case[key][:, :, hs].contiguous()
            sub["beta"], sub["A_log"], sub["dt_bias"] = case["beta"][:, :, hs].contiguous(), case["A_log"][hs], case["dt_bias"][hs]
            st = alloc_state(1, len(hs))
            o, o32 = call(sub, st)
            out[:, :, hs], out32[:, :, hs] = o, o32
            sts.append(st)
        assert n_diff(out, ref[0]) == 0 and n_diff(out32, ref[1]) == 0, f"outputs differ for head groups {groups}"
        for hs, st in zip(groups, sts):
            sel = ref[2].select([0], heads=hs)
            assert state_equal(st, sel), f"state differs for heads {hs}: {state_mismatch(st, sel)}"


# ---------------------------------------------------------------------------------------------------------
# CUDA graphs, determinism, training forward
# ---------------------------------------------------------------------------------------------------------
def test_cuda_graph_replay(ext):
    H, N = 2, 3
    lens = [70, 5, 130]
    case, cu = make_multi(lens, H, "authors", seed=50)
    cu_t = cu_tensor(cu)
    init = alloc_state(N, H)
    # a non-trivial initial state: first a call with a head start for every sequence
    warm, _ = make_multi([21, 40, 64], H, "authors", seed=51)
    call(warm, init, cu_tensor([0, 21, 61, 125]))

    eager_state = init.clone()
    eager = call(case, eager_state, cu_t)

    g_state = init.clone()
    out, out32 = _out_buffers(case["q"])

    def launch():
        prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], out, case["A_log"],
                case["dt_bias"], case["lower_bound"], g_state, cu_t, out_fp32=out32)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):   # warm-up (allocations, attribute setup) off the capture
        w = init.clone()
        g_state_backup = g_state
        g_state = w
        launch()
        g_state = g_state_backup
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for _ in range(3):   # replay from the same initial state each time
        g_state.E.copy_(init.E), g_state.rows.copy_(init.rows), g_state.pos.copy_(init.pos)
        out.zero_(), out32.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert_same((eager[0], eager[1], eager_state), (out, out32, g_state), "graph replay")


def test_decode_graph_replay(ext):
    H, N, steps = 2, 4, 40
    case, cu = make_multi([steps] * N, H, "authors", seed=52)
    # eager reference: whole sequences in one call
    st_ref = alloc_state(N, H)
    ref_out, ref_out32 = call(case, st_ref, cu_tensor(cu))
    # graph: static [N,1,H,128] buffers, replay once per step
    st = alloc_state(N, H)
    sq = {key: torch.zeros(N, 1, *case[key].shape[2:], dtype=case[key].dtype, device="cuda") for key in ("q", "k", "v", "g", "beta")}
    o = torch.zeros_like(sq["q"])
    o32 = torch.zeros(sq["q"].shape, dtype=torch.float32, device="cuda")

    def launch():
        decode(sq["q"], sq["k"], sq["v"], sq["g"], sq["beta"], case["scale"], o, case["A_log"], case["dt_bias"],
               case["lower_bound"], st, out_fp32=o32)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        scratch = st.clone()
        saved, st = st, scratch
        launch()
        st = saved
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    st.E.zero_(), st.rows.zero_(), st.pos.zero_()
    for t in range(steps):
        for key in sq:
            sq[key].copy_(torch.stack([case[key][0, cu[n] + t] for n in range(N)], dim=0).unsqueeze(1))
        graph.replay()
        for n in range(N):
            assert torch.equal(o[n, 0].view(torch.int16), ref_out[0, cu[n] + t].view(torch.int16)), f"step {t} seq {n}"
            assert torch.equal(o32[n, 0].view(torch.int32), ref_out32[0, cu[n] + t].view(torch.int32)), f"step {t} seq {n}"
    assert state_equal(st, st_ref)


def test_run_to_run_determinism(ext):
    H, T = 2, 400
    case = make_case(T, H, "heavy_tail", seed=60)
    ref = whole(case, H)
    for _ in range(5):
        assert_same(ref, whole(case, H), "repeat")


def test_cross_process_digest(ext):
    """Two independent processes produce the same bits for the same workload."""
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parent / "digest_run.py"
    digests = []
    for _ in range(2):
        r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=True)
        digests.append([ln for ln in r.stdout.splitlines() if ln.startswith("DIGEST")][-1])
    assert digests[0] == digests[1], digests


def test_training_forward_equals_prefill_varlen(ext):
    H = 2
    lens = [100, 7, 64, 129]
    case, cu = make_multi(lens, H, "authors", seed=61)
    st = alloc_state(len(lens), H)
    o, o32 = call(case, st, cu_tensor(cu))
    out32 = torch.zeros(case["q"].shape, dtype=torch.float32, device="cuda")
    ot = forward_train(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], case["A_log"],
                       case["dt_bias"], case["lower_bound"], cu_tensor(cu), out_fp32=out32)
    assert n_diff(o, ot) == 0 and n_diff(o32, out32) == 0


# ---------------------------------------------------------------------------------------------------------
# long runs
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not BIG, reason="set CFBI_BIG=1 for the 32K-token runs")
@pytest.mark.parametrize("regime", ["authors", "heavy_tail", "long_memory"])
def test_32k_whole_vs_cuts_vs_decode(ext, regime):
    H, T = 4, 32768 + 37
    case = make_case(T, H, regime, seed=70)
    ref = whole(case, H)
    for seed in range(3):
        pieces = random_pieces(T, random.Random(seed))
        assert_same(ref, run_pieces(case, pieces, H), "32K random cuts")
    st = alloc_state(1, H)
    out, out32 = _out_buffers(case["q"])
    for t in range(T):
        decode(case["q"][:, t:t + 1], case["k"][:, t:t + 1], case["v"][:, t:t + 1], case["g"][:, t:t + 1],
               case["beta"][:, t:t + 1], case["scale"], out[:, t:t + 1], case["A_log"], case["dt_bias"],
               case["lower_bound"], st, out_fp32=out32[:, t:t + 1])
    assert_same(ref, (out, out32, st), "32K decode x n")


# ---------------------------------------------------------------------------------------------------------
# negative control
# ---------------------------------------------------------------------------------------------------------
def test_negative_control_is_caught(ext):
    """A build that replaces one fma of the row engine by mul+sub is a different program: the harness must see it."""
    bad = load_variant("negctl", ("CFBI_NEG_CONTROL=1",))
    H, T = 2, 200
    case = make_case(T, H, "authors", seed=80)
    sa, sb = alloc_state(1, H), alloc_state(1, H)
    oa, fa = call(case, sa, mode=0, mod=bad)   # K1/K2: untouched
    ob, fb = call(case, sb, mode=1, mod=bad)   # row engine: sabotaged
    assert n_diff(fa, fb) > 0, "negative control produced identical fp32 outputs: the harness is vacuous"
    print(f"\n[negative control] fp32 outputs differing: {n_diff(fa, fb)}/{fa.numel()}, bf16: {n_diff(oa, ob)}/{oa.numel()}, "
          f"state: {state_mismatch(sa, sb)}")
