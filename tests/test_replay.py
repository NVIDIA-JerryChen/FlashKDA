"""Bitwise tests of flash_kda.replay against the unmodified flash_kda.fwd.

The reference everywhere is ONE flash_kda.fwd call over the whole sequence (same inputs, same initial state). Every
comparison is on raw bits (int16 views of bf16), so -0/+0 and NaN payloads count as differences.
"""
import math
import random

import pytest
import torch

import flash_kda
from flash_kda import replay as R

D = 128
C = R.CHUNK
dev = "cuda"


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------
def make_params(H, regime="default", seed=0):
    gen = torch.Generator(device=dev).manual_seed(10_000 + seed)
    A_log = torch.rand(H, generator=gen, device=dev, dtype=torch.float32)
    dt_bias = torch.rand(H, D, generator=gen, device=dev, dtype=torch.float32)
    lower_bound = -5.0
    if regime == "slow_decay":
        dt_bias = dt_bias - 8.0
    elif regime == "fast_decay":
        dt_bias = dt_bias + 8.0
    elif regime == "lb_small":
        lower_bound = -0.1
    elif regime == "lb_zero":
        lower_bound = 0.0
    elif regime == "bias_extreme":
        dt_bias = torch.where(torch.rand(H, D, generator=gen, device=dev) < 0.5, -8.0, 8.0).float()
    return dict(scale=1.0 / math.sqrt(D), A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound)


def make_tokens(T, H, regime="default", seed=0):
    """Raw per-token inputs, packed [T, H, D] (beta [T, H]); the kernels apply l2norm / gate / sigmoid themselves."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, generator=gen, device=dev, dtype=torch.float32)  # noqa: E731
    q, k, v, g, beta = rnd(T, H, D), rnd(T, H, D), rnd(T, H, D), rnd(T, H, D), rnd(T, H)
    if regime == "big_v":
        v = v * 30.0
    elif regime == "collinear_k":
        k = rnd(1, H, D) + 0.02 * k
    elif regime == "beta_extreme":
        beta = beta * 12.0
    elif regime == "gate_extreme":
        g = g * 8.0
    bf = lambda t: t.to(torch.bfloat16).contiguous()  # noqa: E731
    return dict(q=bf(q), k=bf(k), v=bf(v), g=bf(g), beta=bf(beta))


def cat_tokens(parts):
    return {n: torch.cat([p[n] for p in parts], dim=0).contiguous() for n in ("q", "k", "v", "g", "beta")}


def slice_tokens(x, a, b):
    return {n: t[a:b].contiguous() for n, t in x.items()}


def fwd_full(x, p, initial_state=None, cu_seqlens=None, final_dtype=torch.bfloat16):
    """One flash_kda.fwd call. x: packed tokens. Returns (out [T, H, D], final_state)."""
    T, H = x["q"].shape[0], x["q"].shape[1]
    out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device=dev)
    N = 1 if cu_seqlens is None else cu_seqlens.numel() - 1
    fin = torch.empty(N, H, D, D, dtype=final_dtype, device=dev)
    init = initial_state
    if init is not None and init.dtype != final_dtype:
        init = init.to(final_dtype)
    flash_kda.fwd(x["q"][None], x["k"][None], x["v"][None], x["g"][None], x["beta"][None], p["scale"], out,
                  p["A_log"], p["dt_bias"], p["lower_bound"], initial_state=init, final_state=fin,
                  cu_seqlens=cu_seqlens)
    return out[0], fin


def prefill(x, p, st, cu_seqlens=None, slot_ids=None, path=None):
    T, H = x["q"].shape[0], x["q"].shape[1]
    out = torch.full((1, T, H, D), float("nan"), dtype=torch.bfloat16, device=dev)
    R.prefill(x["q"][None], x["k"][None], x["v"][None], x["g"][None], x["beta"][None], p["scale"], out,
              p["A_log"], p["dt_bias"], p["lower_bound"], st, cu_seqlens=cu_seqlens, slot_ids=slot_ids, _path=path)
    return out[0]


def decode(x, p, st, slot_ids=None):
    """x holds N tokens (one per sequence) packed [N, H, D]."""
    out = torch.full_like(x["q"], float("nan"))
    R.decode(x["q"], x["k"], x["v"], x["g"], x["beta"], p["scale"], out, p["A_log"], p["dt_bias"],
             p["lower_bound"], st, slot_ids=slot_ids)
    return out


def bits(t):
    return t.contiguous().view(torch.int16) if t.dtype in (torch.bfloat16, torch.float16) else t.contiguous().view(torch.int32)


def assert_bitwise(a, b, what):
    assert a.shape == b.shape, f"{what}: shape {tuple(a.shape)} vs {tuple(b.shape)}"
    diff = bits(a) != bits(b)
    n = int(diff.sum())
    if n:
        first = tuple(int(i) for i in diff.nonzero()[0])
        raise AssertionError(f"{what}: {n} of {diff.numel()} elements differ; first at {first}: "
                             f"{a[first[:-1]].flatten()[first[-1]].item()} vs {b[first[:-1]].flatten()[first[-1]].item()}")


def rand_state(N, H, seed, scale=0.05):
    gen = torch.Generator(device=dev).manual_seed(777 + seed)
    return (torch.randn(N, H, D, D, generator=gen, device=dev) * scale)


# ---------------------------------------------------------------------------------------------------------------------
# 1. the premises, on the unmodified flash_kda.fwd
# ---------------------------------------------------------------------------------------------------------------------
def test_fwd_rows_depend_only_on_earlier_rows():
    """Row t of fwd over [0, T) == row t of fwd over [0, t+1): later tokens (same chunk or not) never change it."""
    H, T = 4, 80
    p, x = make_params(H), make_tokens(T, H, seed=1)
    full, _ = fwd_full(x, p)
    for t in [0, 1, 2, 7, 14, 15, 16, 17, 30, 31, 32, 47, 63, 64, 65, 79]:
        pre, _ = fwd_full(slice_tokens(x, 0, t + 1), p)
        assert_bitwise(pre[t], full[t], f"row {t}")


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_fwd_split_at_chunk_boundaries(state_dtype):
    """fwd over [0, 32) then [32, 80) with the state handed over == one call (outputs and final state)."""
    H, T = 3, 80
    p, x = make_params(H), make_tokens(T, H, seed=2)
    h0 = rand_state(1, H, 2).to(state_dtype)
    full, fin = fwd_full(x, p, initial_state=h0, final_dtype=state_dtype)
    for cut in (16, 32, 48, 64):
        a, fa = fwd_full(slice_tokens(x, 0, cut), p, initial_state=h0, final_dtype=state_dtype)
        b, fb = fwd_full(slice_tokens(x, cut, T), p, initial_state=fa, final_dtype=state_dtype)
        assert_bitwise(torch.cat([a, b]), full, f"outputs, cut {cut}")
        assert_bitwise(fb, fin, f"final state, cut {cut}")


def test_fwd_open_chunk_oracle():
    """The replay idea expressed with fwd alone: fwd over the open chunk's rows from the chunk-start state."""
    H, T = 2, 70
    p, x = make_params(H), make_tokens(T, H, seed=3)
    full, _ = fwd_full(x, p)
    for t in range(T):
        c0 = (t // C) * C
        _, S = fwd_full(slice_tokens(x, 0, c0), p) if c0 else (None, torch.zeros(1, H, D, D, dtype=torch.bfloat16, device=dev))
        o, _ = fwd_full(slice_tokens(x, c0, t + 1), p, initial_state=S)
        assert_bitwise(o[-1], full[t], f"row {t}")


def test_fwd_zero_length_sequences():
    """flash_kda.fwd with empty sequences in cu_seqlens (prefill() relies on it): outputs of the others unchanged,
    final state of an empty sequence == its initial state."""
    H = 2
    p = make_params(H)
    lens = [0, 37, 0, 0, 16, 5, 0]
    xs = [make_tokens(n, H, seed=40 + i) for i, n in enumerate(lens)]
    x = cat_tokens([t for t in xs if t["q"].shape[0] > 0])
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int64, device=dev)
    h0 = rand_state(len(lens), H, 4).to(torch.bfloat16)
    out, fin = fwd_full(x, p, initial_state=h0, cu_seqlens=cu)
    off = 0
    for i, n in enumerate(lens):
        if n == 0:
            assert_bitwise(fin[i], h0[i], f"empty sequence {i} final state")
            continue
        o, f = fwd_full(xs[i], p, initial_state=h0[i:i + 1])
        assert_bitwise(out[off:off + n], o, f"sequence {i} outputs")
        assert_bitwise(fin[i:i + 1], f, f"sequence {i} final state")
        off += n


# ---------------------------------------------------------------------------------------------------------------------
# 2. decode / prefill vs one fwd call
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("H", [1, 4])
@pytest.mark.parametrize("regime", ["default", "slow_decay", "fast_decay", "lb_small", "lb_zero", "bias_extreme"])
def test_decode_from_empty(H, regime):
    T = 70
    p, x = make_params(H, regime), make_tokens(T, H, seed=5)
    full, fin = fwd_full(x, p)
    st = R.alloc_state(1, H)
    outs = [decode(slice_tokens(x, t, t + 1), p, st) for t in range(T)]
    assert_bitwise(torch.cat(outs), full, "decode outputs")
    assert_bitwise(R.final_state(st, p["A_log"], p["dt_bias"], p["lower_bound"]), fin, "final state")
    assert bool((st.pos[0] == T % C).all()), st.pos


@pytest.mark.parametrize("regime", ["big_v", "collinear_k", "beta_extreme", "gate_extreme"])
def test_decode_data_regimes(regime):
    H, T = 4, 50
    p, x = make_params(H), make_tokens(T, H, regime=regime, seed=6)
    full, _ = fwd_full(x, p)
    st = R.alloc_state(1, H)
    outs = [decode(slice_tokens(x, t, t + 1), p, st) for t in range(T)]
    assert_bitwise(torch.cat(outs), full, "decode outputs")


@pytest.mark.parametrize("prompt", [1, 5, 15, 16, 17, 31, 32, 33, 100, 255, 1000])
@pytest.mark.parametrize("init", [None, "bf16", "fp32"])
def test_prefill_then_decode(prompt, init):
    H, n_dec = 4, 35
    p, x = make_params(H), make_tokens(prompt + n_dec, H, seed=prompt)
    h0 = None
    if init is not None:
        h0 = rand_state(1, H, prompt).to(torch.bfloat16 if init == "bf16" else torch.float32)
    full, fin = fwd_full(x, p, initial_state=h0, final_dtype=torch.float32 if init == "fp32" else torch.bfloat16)
    st = R.alloc_state(1, H, initial_state=h0)
    o1 = prefill(slice_tokens(x, 0, prompt), p, st)
    o2 = [decode(slice_tokens(x, t, t + 1), p, st) for t in range(prompt, prompt + n_dec)]
    assert_bitwise(torch.cat([o1] + o2), full, "outputs")
    fs = R.final_state(st, p["A_log"], p["dt_bias"], p["lower_bound"],
                       dtype=torch.float32 if init == "fp32" else torch.bfloat16)
    assert_bitwise(fs, fin, "final state")


@pytest.mark.parametrize("step", [2, 3, 7, 15, 16, 17, 33])
def test_multi_token_decode(step):
    """Several tokens per decode call (speculative decoding style), crossing chunk boundaries."""
    H, T = 3, 120
    p, x = make_params(H), make_tokens(T, H, seed=7)
    full, _ = fwd_full(x, p)
    st = R.alloc_state(1, H)
    outs, t = [], 0
    while t < T:
        n = min(step, T - t)
        xt = slice_tokens(x, t, t + n)
        o = torch.full_like(xt["q"], float("nan"))
        R.decode(xt["q"][None], xt["k"][None], xt["v"][None], xt["g"][None], xt["beta"][None], p["scale"], o[None],
                 p["A_log"], p["dt_bias"], p["lower_bound"], st)
        outs.append(o)
        t += n
    assert_bitwise(torch.cat(outs), full, "outputs")


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_chunked_prefill_random_cuts(seed):
    """A sequence cut at random offsets into prefill calls (both the replay and the fwd path get exercised)."""
    H, T = 4, 700
    p, x = make_params(H), make_tokens(T, H, seed=8 + seed)
    full, fin = fwd_full(x, p)
    rng = random.Random(seed)
    cuts = sorted(rng.sample(range(1, T), 7))
    st, outs, a = R.alloc_state(1, H), [], 0
    for b in cuts + [T]:
        outs.append(prefill(slice_tokens(x, a, b), p, st))
        a = b
    assert_bitwise(torch.cat(outs), full, f"outputs, cuts {cuts}")
    assert_bitwise(R.final_state(st, p["A_log"], p["dt_bias"], p["lower_bound"]), fin, "final state")


@pytest.mark.parametrize("path", ["replay", "fwd"])
def test_prefill_paths(path):
    """Force each prefill path on a call it would not pick by default."""
    H = 2
    p = make_params(H)
    for T, cut in [(600, 0), (600, 37), (40, 3), (17, 1), (16, 15)]:
        x = make_tokens(T, H, seed=T + cut)
        full, _ = fwd_full(x, p)
        st = R.alloc_state(1, H)
        o0 = prefill(slice_tokens(x, 0, cut), p, st) if cut else x["q"][:0]
        o1 = prefill(slice_tokens(x, cut, T), p, st, path=path)
        assert_bitwise(torch.cat([o0, o1]), full, f"T={T} cut={cut} path={path}")


def test_varlen_batch():
    """Batched varlen prefill (sequences at different chunk rows) + decode steps; per sequence == its own fwd call."""
    H = 4
    p = make_params(H)
    lens = [1, 15, 16, 17, 100, 333, 64, 2]
    n_dec = 21
    N = len(lens)
    xs = [make_tokens(n + n_dec, H, seed=90 + i) for i, n in enumerate(lens)]
    h0 = rand_state(N, H, 9).to(torch.bfloat16)
    refs = [fwd_full(xs[i], p, initial_state=h0[i:i + 1]) for i in range(N)]
    st = R.alloc_state(N, H, initial_state=h0)
    xp = cat_tokens([slice_tokens(xs[i], 0, n) for i, n in enumerate(lens)])
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int64, device=dev)
    op = prefill(xp, p, st, cu_seqlens=cu)
    od = [decode(cat_tokens([slice_tokens(xs[i], n + s, n + s + 1) for i, n in enumerate(lens)]), p, st)
          for s in range(n_dec)]
    fs = R.final_state(st, p["A_log"], p["dt_bias"], p["lower_bound"])
    for i, n in enumerate(lens):
        got = torch.cat([op[int(cu[i]):int(cu[i + 1])]] + [o[i:i + 1] for o in od])
        assert_bitwise(got, refs[i][0], f"sequence {i} (len {n})")
        assert_bitwise(fs[i:i + 1], refs[i][1], f"sequence {i} final state")


def test_varlen_continuation_mixed():
    """Second varlen prefill call where sequences continue open chunks (heads), skip bodies, or only add tails."""
    H = 2
    p = make_params(H)
    first = [5, 16, 30, 3, 1]
    second = [40, 3, 2, 13, 50]
    N = len(first)
    xs = [make_tokens(a + b, H, seed=120 + i) for i, (a, b) in enumerate(zip(first, second))]
    refs = [fwd_full(xs[i], p) for i in range(N)]
    st = R.alloc_state(N, H)
    outs = []
    for part, offs in ((first, [0] * N), (second, first)):
        xp = cat_tokens([slice_tokens(xs[i], offs[i], offs[i] + n) for i, n in enumerate(part)])
        cu = torch.tensor([0] + torch.tensor(part).cumsum(0).tolist(), dtype=torch.int64, device=dev)
        o = prefill(xp, p, st, cu_seqlens=cu, path="fwd")
        outs.append([o[int(cu[i]):int(cu[i + 1])] for i in range(N)])
    for i in range(N):
        assert_bitwise(torch.cat([outs[0][i], outs[1][i]]), refs[i][0], f"sequence {i}")


def test_batched_mode():
    """[B, T, H, D] prefill (no cu_seqlens), then decode."""
    H, B, T, n_dec = 3, 5, 45, 9
    p = make_params(H)
    xs = [make_tokens(T + n_dec, H, seed=150 + b) for b in range(B)]
    refs = [fwd_full(xs[b], p) for b in range(B)]
    st = R.alloc_state(B, H)
    xb = {n: torch.stack([xs[b][n][:T] for b in range(B)]) for n in ("q", "k", "v", "g", "beta")}
    out = torch.full_like(xb["q"], float("nan"))
    R.prefill(xb["q"], xb["k"], xb["v"], xb["g"], xb["beta"], p["scale"], out, p["A_log"], p["dt_bias"],
              p["lower_bound"], st)
    od = [decode(cat_tokens([slice_tokens(xs[b], T + s, T + s + 1) for b in range(B)]), p, st) for s in range(n_dec)]
    for b in range(B):
        got = torch.cat([out[b]] + [o[b:b + 1] for o in od])
        assert_bitwise(got, refs[b][0], f"batch {b}")


def test_slot_ids_and_padding():
    """State pool with slot indirection; decode entries with slot -1 are skipped and leave out untouched."""
    H, P = 2, 8
    p = make_params(H)
    slots = [5, 2, 7, 0]
    lens = [20, 3, 47, 16]
    n_dec = 18
    xs = [make_tokens(n + n_dec, H, seed=170 + i) for i, n in enumerate(lens)]
    refs = [fwd_full(xs[i], p) for i in range(len(lens))]
    st = R.alloc_state(P, H)
    xp = cat_tokens([slice_tokens(xs[i], 0, n) for i, n in enumerate(lens)])
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int64, device=dev)
    sid = torch.tensor(slots, dtype=torch.int32, device=dev)
    op = prefill(xp, p, st, cu_seqlens=cu, slot_ids=sid)
    od = []
    for s in range(n_dec):
        # entries: the 4 sequences in a shuffled order plus two padding entries
        order = [2, -1, 0, 3, -1, 1]
        xt = []
        for e in order:
            i = e if e >= 0 else 0
            xt.append(slice_tokens(xs[i], lens[i] + s, lens[i] + s + 1))
        sid_d = torch.tensor([slots[e] if e >= 0 else -1 for e in order], dtype=torch.int32, device=dev)
        o = decode(cat_tokens(xt), p, st, slot_ids=sid_d)
        for j, e in enumerate(order):
            if e < 0:
                assert bool(torch.isnan(o[j].float()).all()), "padding entry wrote its out row"
        od.append({e: o[j:j + 1] for j, e in enumerate(order) if e >= 0})
    for i in range(len(lens)):
        got = torch.cat([op[int(cu[i]):int(cu[i + 1])]] + [d[i] for d in od])
        assert_bitwise(got, refs[i][0], f"sequence {i} (slot {slots[i]})")


def test_matches_torch_ref():
    """Three-way: torch_ref (the bit-exact PyTorch twin of fwd) == fwd == replay decode."""
    from torch_ref import torch_ref
    H, T = 2, 50
    p, x = make_params(H), make_tokens(T, H, seed=11)
    h0 = rand_state(1, H, 11).to(torch.bfloat16)
    out_ref = torch.zeros(1, T, H, D, dtype=torch.bfloat16, device=dev)
    fin_ref = torch.zeros(1, H, D, D, dtype=torch.bfloat16, device=dev)
    torch_ref(x["q"][None], x["k"][None], x["v"][None], x["g"][None], x["beta"][None], p["scale"], out_ref,
              A_log=p["A_log"], dt_bias=p["dt_bias"], lower_bound=p["lower_bound"],
              initial_state=h0.clone(), final_state=fin_ref)
    full, fin = fwd_full(x, p, initial_state=h0)
    st = R.alloc_state(1, H, initial_state=h0)
    outs = [decode(slice_tokens(x, t, t + 1), p, st) for t in range(T)]
    got, got_fin = torch.cat(outs), R.final_state(st, p["A_log"], p["dt_bias"], p["lower_bound"])
    assert_bitwise(got, full, "replay vs fwd")
    assert_bitwise(got_fin, fin, "replay final state vs fwd")
    assert_bitwise(got, out_ref[0], "replay vs torch_ref")
    assert_bitwise(got_fin, fin_ref, "replay final state vs torch_ref")
    # torch_ref vs fwd is established by tests/test_fwd.py with torch.equal; checked here on raw bits too
    assert_bitwise(out_ref[0], full, "torch_ref vs fwd")


def test_cuda_graph_decode():
    """A captured decode step replayed for many tokens gives the eager bits."""
    H, N, T0, n_dec = 4, 6, 37, 40
    p = make_params(H)
    xs = [make_tokens(T0 + n_dec, H, seed=200 + i) for i in range(N)]
    refs = [fwd_full(xs[i], p)[0] for i in range(N)]
    st = R.alloc_state(N, H)
    prefill(cat_tokens([slice_tokens(xs[i], 0, T0) for i in range(N)]), p, st,
            cu_seqlens=torch.arange(N + 1, device=dev, dtype=torch.int64) * T0)
    buf = cat_tokens([slice_tokens(xs[i], T0, T0 + 1) for i in range(N)])
    out = torch.empty_like(buf["q"])
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):   # warm-up outside the graph on a throw-away copy of the state
        tmp = st.clone()
        R.decode(buf["q"], buf["k"], buf["v"], buf["g"], buf["beta"], p["scale"], out, p["A_log"], p["dt_bias"],
                 p["lower_bound"], tmp)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        R.decode(buf["q"], buf["k"], buf["v"], buf["g"], buf["beta"], p["scale"], out, p["A_log"], p["dt_bias"],
                 p["lower_bound"], st)
    for s_ in range(n_dec):
        nxt = cat_tokens([slice_tokens(xs[i], T0 + s_, T0 + s_ + 1) for i in range(N)])
        for n in buf:
            buf[n].copy_(nxt[n])
        graph.replay()
        for i in range(N):
            assert_bitwise(out[i], refs[i][T0 + s_], f"graph step {s_} sequence {i}")


def test_deterministic():
    H = 4
    p, x = make_params(H), make_tokens(300, H, seed=12)
    res = []
    for _ in range(2):
        st = R.alloc_state(1, H)
        o = [prefill(slice_tokens(x, 0, 133), p, st)]
        o += [decode(slice_tokens(x, t, t + 1), p, st) for t in range(133, 170)]
        res.append((torch.cat(o), st.clone()))
    assert_bitwise(res[0][0], res[1][0], "outputs")
    for a, b in zip(res[0][1]._tensors(), res[1][1]._tensors()):
        assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)), "state bits differ"


@pytest.mark.parametrize("H", [32, 96])
def test_many_heads(H):
    T0, n_dec = 517, 20
    p, x = make_params(H), make_tokens(T0 + n_dec, H, seed=13)
    full, fin = fwd_full(x, p)
    st = R.alloc_state(1, H)
    o = [prefill(slice_tokens(x, 0, T0), p, st)]
    o += [decode(slice_tokens(x, t, t + 1), p, st) for t in range(T0, T0 + n_dec)]
    assert_bitwise(torch.cat(o), full, "outputs")
    assert_bitwise(R.final_state(st, p["A_log"], p["dt_bias"], p["lower_bound"]), fin, "final state")


def test_large_random_sweep():
    """Many sequences / heads in one varlen fwd vs the replay path (catches rare-bit contraction differences)."""
    H = 16
    p = make_params(H, "bias_extreme")
    gen = random.Random(14)
    lens = [gen.randint(1, 400) for _ in range(24)]
    x = make_tokens(sum(lens), H, regime="gate_extreme", seed=14)
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int64, device=dev)
    h0 = rand_state(len(lens), H, 14).to(torch.bfloat16)
    full, fin = fwd_full(x, p, initial_state=h0, cu_seqlens=cu)
    st = R.alloc_state(len(lens), H, initial_state=h0)
    o = prefill(x, p, st, cu_seqlens=cu, path="replay")
    assert_bitwise(o, full, "replay path outputs")
    assert_bitwise(R.final_state(st, p["A_log"], p["dt_bias"], p["lower_bound"]), fin, "final state")


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("pooled", [False, True])
def test_prefill_host_plan(batched, pooled):
    """prefill(context_lens_cpu=...) plans the split on the host; rounds cover the plain-fwd case (fresh, lengths
    multiple of 16), tails, heads and bodies; same bits as one fwd call per sequence."""
    H, N = 3, 4
    p = make_params(H)
    rounds = [[32, 32, 32, 32], [21, 21, 21, 21], [40, 40, 40, 40]] if batched else \
             [[32, 48, 16, 64], [5, 40, 16, 3], [27, 11, 50, 13]]
    total = [sum(r[i] for r in rounds) for i in range(N)]
    xs = [make_tokens(total[i], H, seed=300 + i) for i in range(N)]
    refs = [fwd_full(xs[i], p)[0] for i in range(N)]
    slots = [4, 1, 5, 0] if pooled else list(range(N))
    st = R.alloc_state(6 if pooled else N, H)
    sid = torch.tensor(slots, dtype=torch.int32, device=dev) if pooled else None
    ctx = [0] * N
    outs = [[] for _ in range(N)]
    for lens in rounds:
        parts = [slice_tokens(xs[i], ctx[i], ctx[i] + lens[i]) for i in range(N)]
        if batched:
            xb = {n: torch.stack([parts[i][n] for i in range(N)]) for n in ("q", "k", "v", "g", "beta")}
            o = torch.full_like(xb["q"], float("nan"))
            R.prefill(xb["q"], xb["k"], xb["v"], xb["g"], xb["beta"], p["scale"], o, p["A_log"], p["dt_bias"],
                      p["lower_bound"], st, slot_ids=sid, context_lens_cpu=ctx)
            for i in range(N):
                outs[i].append(o[i])
        else:
            xp = cat_tokens(parts)
            cu_h = [0] + torch.tensor(lens).cumsum(0).tolist()
            cu = torch.tensor(cu_h, dtype=torch.int64, device=dev)
            o = torch.full((1,) + tuple(xp["q"].shape), float("nan"), dtype=torch.bfloat16, device=dev)
            R.prefill(xp["q"][None], xp["k"][None], xp["v"][None], xp["g"][None], xp["beta"][None], p["scale"], o,
                      p["A_log"], p["dt_bias"], p["lower_bound"], st, cu_seqlens=cu, slot_ids=sid,
                      context_lens_cpu=ctx, cu_seqlens_cpu=cu_h)
            for i in range(N):
                outs[i].append(o[0, cu_h[i]:cu_h[i + 1]])
        ctx = [c + n for c, n in zip(ctx, lens)]
    for i in range(N):
        assert_bitwise(torch.cat(outs[i]), refs[i], f"sequence {i}")
    assert bool((st.pos[torch.tensor(slots)] == torch.tensor([c % C for c in ctx], device=dev, dtype=torch.int32)[:, None]).all())


# ---------------------------------------------------------------------------------------------------------------------
# coverage extras (review findings)
# ---------------------------------------------------------------------------------------------------------------------
def test_multi_token_decode_batch_slots_padding():
    """Several sequences, T tokens per decode call, slot pool, padding entries, sequences at different chunk rows."""
    H, P, T = 2, 7, 5
    p = make_params(H)
    lens = [3, 16, 29, 40]
    slots = [6, 0, 3, 2]
    n_calls = 9
    xs = [make_tokens(L + T * n_calls, H, seed=500 + i) for i, L in enumerate(lens)]
    refs = [fwd_full(xs[i], p)[0] for i in range(len(lens))]
    st = R.alloc_state(P, H)
    xp = cat_tokens([slice_tokens(xs[i], 0, L) for i, L in enumerate(lens)])
    cu_h = [0] + torch.tensor(lens).cumsum(0).tolist()
    op = prefill(xp, p, st, cu_seqlens=torch.tensor(cu_h, dtype=torch.int64, device=dev),
                 slot_ids=torch.tensor(slots, dtype=torch.int32, device=dev))
    outs = [[op[cu_h[i]:cu_h[i + 1]]] for i in range(len(lens))]
    order = [3, -1, 1, 0, 2]
    for c in range(n_calls):
        parts = [slice_tokens(xs[e if e >= 0 else 0], lens[e if e >= 0 else 0] + c * T,
                              lens[e if e >= 0 else 0] + (c + 1) * T) for e in order]
        xb = {n: torch.stack([pt[n] for pt in parts]) for n in ("q", "k", "v", "g", "beta")}
        o = torch.full_like(xb["q"], float("nan"))
        sid = torch.tensor([slots[e] if e >= 0 else -1 for e in order], dtype=torch.int32, device=dev)
        R.decode(xb["q"], xb["k"], xb["v"], xb["g"], xb["beta"], p["scale"], o, p["A_log"], p["dt_bias"],
                 p["lower_bound"], st, slot_ids=sid)
        for j, e in enumerate(order):
            if e >= 0:
                outs[e].append(o[j])
            else:
                assert bool(torch.isnan(o[j].float()).all()), "padding entry wrote its out rows"
    for i in range(len(lens)):
        assert_bitwise(torch.cat(outs[i]), refs[i], f"sequence {i}")


def test_fwd_path_head_only_and_zero_length():
    """fwd-path prefill where some sequences only partly fill their open chunk, some get 0 tokens, cu_seqlens[0] != 0."""
    H = 2
    p = make_params(H)
    first = [7, 30, 9, 21, 4]
    second = [5, 40, 0, 2, 60]          # seq 0: head only (7+5 < 16); seq 2: no tokens; seq 3: head only (5+2 < 16)
    N = len(first)
    xs = [make_tokens(a + b, H, seed=600 + i) for i, (a, b) in enumerate(zip(first, second))]
    refs = [fwd_full(xs[i], p) for i in range(N)]
    st = R.alloc_state(N, H)
    outs = [[] for _ in range(N)]
    for lens, offs, lead in ((first, [0] * N, 0), (second, first, 3)):
        parts = [slice_tokens(xs[i], offs[i], offs[i] + n) for i, n in enumerate(lens)]
        pad = make_tokens(lead, H, seed=699) if lead else None     # `lead` junk tokens before cu_seqlens[0]
        xp = cat_tokens(([pad] if pad else []) + parts)
        cu_h = [lead] + (lead + torch.tensor(lens).cumsum(0)).tolist()
        o = prefill(xp, p, st, cu_seqlens=torch.tensor(cu_h, dtype=torch.int64, device=dev), path="fwd")
        for i in range(N):
            outs[i].append(o[cu_h[i]:cu_h[i + 1]])
        if lead:
            assert bool(torch.isnan(o[:lead].float()).all()), "rows before cu_seqlens[0] were written"
    fs = R.final_state(st, p["A_log"], p["dt_bias"], p["lower_bound"])
    for i in range(N):
        assert_bitwise(torch.cat(outs[i]), refs[i][0], f"sequence {i}")
        assert_bitwise(fs[i:i + 1], refs[i][1], f"sequence {i} final state")


def test_final_state_slots_and_boundary():
    """final_state with slot_ids, including a slot sitting exactly at a chunk boundary (pos 0)."""
    H, P = 2, 5
    p = make_params(H)
    lens = [32, 21, 16, 7]              # pos after prefill: 0, 5, 0, 7
    slots = [4, 0, 2, 1]
    xs = [make_tokens(L, H, seed=700 + i) for i, L in enumerate(lens)]
    st = R.alloc_state(P, H)
    xp = cat_tokens(xs)
    cu_h = [0] + torch.tensor(lens).cumsum(0).tolist()
    sid = torch.tensor(slots, dtype=torch.int32, device=dev)
    prefill(xp, p, st, cu_seqlens=torch.tensor(cu_h, dtype=torch.int64, device=dev), slot_ids=sid)
    fs = R.final_state(st, p["A_log"], p["dt_bias"], p["lower_bound"], slot_ids=sid, dtype=torch.float32)
    for i in range(len(lens)):
        _, ref = fwd_full(xs[i], p, final_dtype=torch.float32)
        assert_bitwise(fs[i:i + 1], ref, f"sequence {i} (slot {slots[i]}) final state")


def test_single_head_fwd_path():
    """H = 1 through the fwd path (fwd's beta transpose does not copy there) at an odd token offset."""
    H = 1
    p = make_params(H)
    x = make_tokens(300, H, seed=800)
    full, _ = fwd_full(x, p)
    st = R.alloc_state(1, H)
    o0 = prefill(slice_tokens(x, 0, 3), p, st)
    o1 = prefill(slice_tokens(x, 3, 300), p, st, path="fwd")
    assert_bitwise(torch.cat([o0, o1]), full, "H=1")


def test_cuda_graph_serving_config():
    """Captured decode with a slot pool and padding entries, replayed with new inputs and a new slot mapping."""
    H, P, N, T0, n_dec = 3, 6, 4, 20, 30
    p = make_params(H)
    slots = [5, 1, 3, 0]
    xs = [make_tokens(T0 + n_dec, H, seed=900 + i) for i in range(N)]
    refs = [fwd_full(xs[i], p)[0] for i in range(N)]
    st = R.alloc_state(P, H)
    prefill(cat_tokens([slice_tokens(xs[i], 0, T0) for i in range(N)]), p, st,
            cu_seqlens=torch.arange(N + 1, device=dev, dtype=torch.int64) * T0,
            slot_ids=torch.tensor(slots, dtype=torch.int32, device=dev))
    E = N + 2                                    # graph batch: the 4 sequences + 2 padding entries
    buf = cat_tokens([slice_tokens(xs[0], 0, 1)] * E)
    sid = torch.full((E,), -1, dtype=torch.int32, device=dev)
    out = torch.empty_like(buf["q"])
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        tmp = st.clone()
        R.decode(buf["q"], buf["k"], buf["v"], buf["g"], buf["beta"], p["scale"], out, p["A_log"], p["dt_bias"],
                 p["lower_bound"], tmp, slot_ids=sid)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        R.decode(buf["q"], buf["k"], buf["v"], buf["g"], buf["beta"], p["scale"], out, p["A_log"], p["dt_bias"],
                 p["lower_bound"], st, slot_ids=sid)
    for step in range(n_dec):
        perm = [(step + j) % E for j in range(E)]          # rotate which graph entry serves which sequence
        new_sid = [-1] * E
        for i in range(N):
            e = perm[i]
            new_sid[e] = slots[i]
            nxt = slice_tokens(xs[i], T0 + step, T0 + step + 1)
            for n in buf:
                buf[n][e:e + 1].copy_(nxt[n])
        sid.copy_(torch.tensor(new_sid, dtype=torch.int32, device=dev))
        graph.replay()
        for i in range(N):
            assert_bitwise(out[perm[i]], refs[i][T0 + step], f"graph step {step} sequence {i}")
