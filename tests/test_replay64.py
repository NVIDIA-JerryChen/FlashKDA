"""Bitwise tests of flash_kda.replay64 (chunk-64 open-chunk replay) against its specification, replay64.fwd64.

The reference everywhere is ONE fwd64 call over the whole sequence (same inputs, same initial state). Every comparison is
on raw bits (int16 views of bf16, int32 views of fp32), so -0/+0 and NaN payloads count as differences. Output buffers
start as NaN and every fwd64 reference is checked finite, so no comparison can pass on two untouched buffers.

  1. premises, on fwd64 alone          2. one-token decode               3. prompt + decode
  4. multi-token decode, slot pools    5. prefill cuts / paths / varlen / host plan    6. final_state
  7. head counts, CUDA graphs, determinism    8. accuracy vs the fp64 recurrence of tests/c64_helpers.py
"""
import inspect
import math
import os
import random
import sys

import pytest
import torch

from flash_kda import replay64 as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c64_helpers as Hh  # noqa: E402

D = 128
C = R.CHUNK
dev = "cuda"
NAN = float("nan")
TOK = ("q", "k", "v", "g", "beta")
STATE_FIELDS = ("S", "rows_k", "rows_v", "rows_g", "rows_beta", "pos")


# ---------------------------------------------------------------------------------------------------------------------
# helpers (make_params / make_tokens / cat_tokens / slice_tokens / bits / assert_bitwise / rand_state: tests/test_replay.py)
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
    return {n: torch.cat([p[n] for p in parts], dim=0).contiguous() for n in TOK}


def stack_tokens(parts):
    """Equal-length per-entry tokens -> [N, T, H, D] (beta [N, T, H])."""
    return {n: torch.stack([p[n] for p in parts]).contiguous() for n in TOK}


def slice_tokens(x, a, b):
    return {n: t[a:b].contiguous() for n, t in x.items()}


def make_cu(lens, lead=0):
    """cu_seqlens as a host list and an int64 device tensor; the first sequence starts at row `lead`."""
    cu = [lead]
    for n in lens:
        cu.append(cu[-1] + n)
    return cu, torch.tensor(cu, dtype=torch.int64, device=dev)


def slot_tensor(slots):
    return torch.tensor(slots, dtype=torch.int32, device=dev)


def rand_state(N, H, seed, scale=0.05):
    gen = torch.Generator(device=dev).manual_seed(777 + seed)
    return torch.randn(N, H, D, D, generator=gen, device=dev) * scale


def init_state(kind, N, H, seed):
    """None, or a random [N, H, 128, 128] initial state in bf16 ("bf16") / fp32 ("fp32")."""
    if kind is None:
        return None
    return rand_state(N, H, seed).to(torch.bfloat16 if kind == "bf16" else torch.float32)


def from_case(case):
    """tests/c64_helpers.make_case -> (params, packed tokens) in this file's layout."""
    return {n: case[n] for n in ("scale", "A_log", "dt_bias", "lower_bound")}, {n: case[n][0].contiguous() for n in TOK}


def fwd64_call(x4, p, initial_state=None, cu_seqlens=None, k2_cfg=None):
    """ONE fwd64 call on [B, T, H, D] inputs. Returns (out [B, T, H, D], final_state bf16 [N, H, D, D]); both buffers
    start as NaN and must come back finite (a bitwise comparison against them is then never vacuous)."""
    B, H = x4["q"].shape[0], x4["q"].shape[2]
    N = B if cu_seqlens is None else cu_seqlens.numel() - 1
    out = torch.full_like(x4["q"], NAN)
    fin = torch.full((N, H, D, D), NAN, dtype=torch.bfloat16, device=dev)
    R.fwd64(x4["q"], x4["k"], x4["v"], x4["g"], x4["beta"], p["scale"], out, p["A_log"], p["dt_bias"],
            p["lower_bound"], initial_state=initial_state, final_state=fin, cu_seqlens=cu_seqlens, _k2_cfg=k2_cfg)
    assert bool(torch.isfinite(out.float()).all()), "fwd64 reference outputs are not all finite"
    assert bool(torch.isfinite(fin.float()).all()), "fwd64 reference final state is not all finite"
    return out, fin


def fwd64_full(x, p, initial_state=None, cu_seqlens=None, k2_cfg=None):
    """One fwd64 call on packed tokens [T, H, D] (B = 1). Returns (out [T, H, D], final_state bf16 [N, H, D, D])."""
    out, fin = fwd64_call({n: t[None] for n, t in x.items()}, p, initial_state, cu_seqlens, k2_cfg)
    return out[0], fin


def prefill(x, p, st, cu_seqlens=None, slot_ids=None, path=None, k2_cfg=None):
    """replay64.prefill on packed tokens (B = 1), `out` starting as NaN. Returns out [T, H, D]."""
    T, H = x["q"].shape[0], x["q"].shape[1]
    out = torch.full((1, T, H, D), NAN, dtype=torch.bfloat16, device=dev)
    R.prefill(x["q"][None], x["k"][None], x["v"][None], x["g"][None], x["beta"][None], p["scale"], out,
              p["A_log"], p["dt_bias"], p["lower_bound"], st, cu_seqlens=cu_seqlens, slot_ids=slot_ids,
              _path=path, _k2_cfg=k2_cfg)
    return out[0]


def decode(x, p, st, slot_ids=None):
    """replay64.decode, `out` starting as NaN. x: one token per entry ([N, H, D], beta [N, H]) or T tokens per entry
    ([N, T, H, D], beta [N, T, H])."""
    out = torch.full_like(x["q"], NAN)
    R.decode(x["q"], x["k"], x["v"], x["g"], x["beta"], p["scale"], out, p["A_log"], p["dt_bias"],
             p["lower_bound"], st, slot_ids=slot_ids)
    return out


def fstate(st, p, **kw):
    return R.final_state(st, p["A_log"], p["dt_bias"], p["lower_bound"], **kw)


def bits(t):
    return t.contiguous().view(torch.int16) if t.dtype in (torch.bfloat16, torch.float16) else t.contiguous().view(torch.int32)


def n_diff(a, b):
    return int((bits(a) != bits(b)).sum())


def assert_bitwise(a, b, what):
    assert a.shape == b.shape, f"{what}: shape {tuple(a.shape)} vs {tuple(b.shape)}"
    assert a.dtype == b.dtype, f"{what}: dtype {a.dtype} vs {b.dtype}"
    diff = bits(a) != bits(b)
    n = int(diff.sum())
    if n:
        first = tuple(int(i) for i in diff.nonzero()[0])
        raise AssertionError(f"{what}: {n} of {diff.numel()} elements differ; first at {first}: "
                             f"{a[first[:-1]].flatten()[first[-1]].item()} vs {b[first[:-1]].flatten()[first[-1]].item()}")


def assert_untouched(t, what):
    """`t` was NaN-filled before the call: it must still hold exactly those bits."""
    assert_bitwise(t, torch.full_like(t, NAN), what)


def _bytes(t):
    return t.contiguous().view(torch.uint8)


def assert_state_identical(a, b, what):
    """Every state tensor bit-identical (stale rows >= pos included)."""
    for name, x, y in zip(STATE_FIELDS, a._tensors(), b._tensors()):
        assert torch.equal(_bytes(x), _bytes(y)), f"{what}: state.{name} differs"


def snapshot_slots(st, slots):
    return {s: [t[s].clone() for t in st._tensors()] for s in slots}


def assert_slots_unchanged(st, snap, what):
    for s, saved in snap.items():
        for name, a, b in zip(STATE_FIELDS, saved, (t[s] for t in st._tensors())):
            assert torch.equal(_bytes(a), _bytes(b)), f"{what}: slot {s} state.{name} changed"


def assert_state_equivalent(a, b, what):
    """Same continuation point: S, pos and the meaningful rows (< pos) of the open chunk; stale rows may differ."""
    assert torch.equal(a.pos, b.pos), f"{what}: pos {a.pos.tolist()} vs {b.pos.tolist()}"
    assert_bitwise(a.S, b.S, f"{what}: S")
    for s in range(a.P):
        r = int(a.pos[s, 0])
        for name in ("rows_k", "rows_v", "rows_g", "rows_beta"):
            assert_bitwise(getattr(a, name)[s, :, :r], getattr(b, name)[s, :, :r], f"{what}: slot {s} {name}[:{r}]")


def assert_pos(st, slots, ctx, what):
    """pos of every slot (all heads) == its context length % 64."""
    got = st.pos[torch.tensor(slots, dtype=torch.int64, device=dev)]
    exp = torch.tensor([c % C for c in ctx], dtype=torch.int32, device=dev)[:, None].expand_as(got)
    assert torch.equal(got, exp), f"{what}: pos {got.tolist()}, expected {[c % C for c in ctx]} on every head"


def spy_launches(monkeypatch):
    """Record prefill's launch structure in a list: ("host",) when the host plan runs, ("replay",) per replay-kernel
    launch, ("fwd", [segment lengths]) per fwd64 launch (white-box: wraps the module's private launchers)."""
    log = []
    real_host, real_replay, real_fwd = R._prefill_host, R._launch_replay, R._launch_fwd
    fwd_sig = inspect.signature(real_fwd)

    def host(*a, **kw):
        log.append(("host",))
        return real_host(*a, **kw)

    def replay(*a, **kw):
        log.append(("replay",))
        return real_replay(*a, **kw)

    def fwd(*a, **kw):
        log.append(("fwd", [int(n) for n in fwd_sig.bind(*a, **kw).arguments["seg_len"].tolist()]))
        return real_fwd(*a, **kw)

    monkeypatch.setattr(R, "_prefill_host", host)
    monkeypatch.setattr(R, "_launch_replay", replay)
    monkeypatch.setattr(R, "_launch_fwd", fwd)
    return log


def host_plan_launches(lens, ctx):
    """What the host plan launches for one call: the replay kernel for the heads (rows that complete an open chunk),
    fwd64 over the non-empty 64-aligned bodies only, the replay kernel for the tails; empty phases are not launched."""
    head = [min(L, (C - c % C) % C) for L, c in zip(lens, ctx)]
    body = [(L - h) // C * C for L, h in zip(lens, head)]
    tail = [L - h - b for L, h, b in zip(lens, head, body)]
    return [("host",)] + [("replay",)] * any(head) + [("fwd", [b for b in body if b])] * any(body) + \
        [("replay",)] * any(tail)


def capture_decode(buf, out, p, st, slot_ids=None):
    """CUDA graph of one decode call on fixed buffers (warm-up outside the graph on a throw-away copy of the state)."""
    args = (buf["q"], buf["k"], buf["v"], buf["g"], buf["beta"], p["scale"], out, p["A_log"], p["dt_bias"],
            p["lower_bound"])
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        tmp = st.clone()
        R.decode(*args, tmp, slot_ids=slot_ids)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        R.decode(*args, st, slot_ids=slot_ids)
    return graph


# ---------------------------------------------------------------------------------------------------------------------
# 1. the premises, on fwd64 alone
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("init", [None, "bf16"])
def test_fwd64_rows_depend_only_on_earlier_rows(init):
    """Row t of fwd64 over [0, T) == row t of fwd64 over [0, t+1): later tokens (same chunk or not) never change it."""
    H, T = 3, 200
    p, x = make_params(H), make_tokens(T, H, seed=1)
    h0 = init_state(init, 1, H, 1)
    full, _ = fwd64_full(x, p, initial_state=h0)
    for t in [0, 1, 15, 16, 17, 47, 48, 62, 63, 64, 65, 66, 79, 126, 127, 128, 129, 130, 191, 192, 199]:
        pre, _ = fwd64_full(slice_tokens(x, 0, t + 1), p, initial_state=h0)
        assert_bitwise(pre[t], full[t], f"row {t}")


def test_fwd64_open_chunk_oracle():
    """The replay idea with fwd64 alone: row t == the last row of fwd64 over the open chunk's rows [c0, t+1) started from
    the bf16 final state of fwd64 over [0, c0) (c0 = t rounded down to a multiple of 64)."""
    H, T = 2, 140
    p, x = make_params(H), make_tokens(T, H, seed=3)
    full, _ = fwd64_full(x, p)
    S = {0: torch.zeros(1, H, D, D, dtype=torch.bfloat16, device=dev)}
    for c0 in range(C, T, C):
        S[c0] = fwd64_full(slice_tokens(x, 0, c0), p)[1]
    for t in range(T):
        c0 = (t // C) * C
        o, _ = fwd64_full(slice_tokens(x, c0, t + 1), p, initial_state=S[c0])
        assert_bitwise(o[-1], full[t], f"row {t}")


@pytest.mark.parametrize("init", [None, "bf16", "fp32"])
def test_fwd64_split_at_chunk_boundaries(init):
    """fwd64 over [0, cut) then [cut, T) with the bf16 state handed over == one call (outputs and final state) for cuts
    on the 64-grid; an fp32 initial state == its bf16 rounding. Control: a cut off the grid changes bits."""
    H, T = 3, 260
    p, x = make_params(H), make_tokens(T, H, seed=2)
    h0 = init_state(init, 1, H, 2)
    full, fin = fwd64_full(x, p, initial_state=h0)
    for cut in (64, 128, 192, 256):
        a, fa = fwd64_full(slice_tokens(x, 0, cut), p, initial_state=h0)
        b, fb = fwd64_full(slice_tokens(x, cut, T), p, initial_state=fa)
        assert_bitwise(torch.cat([a, b]), full, f"outputs, cut {cut}")
        assert_bitwise(fb, fin, f"final state, cut {cut}")
    if init == "fp32":
        o2, f2 = fwd64_full(x, p, initial_state=h0.to(torch.bfloat16))
        assert_bitwise(o2, full, "fp32 initial state vs its bf16 rounding: outputs")
        assert_bitwise(f2, fin, "fp32 initial state vs its bf16 rounding: final state")
    a, fa = fwd64_full(slice_tokens(x, 0, 37), p, initial_state=h0)
    b, _ = fwd64_full(slice_tokens(x, 37, T), p, initial_state=fa)
    assert n_diff(torch.cat([a, b]), full) > 0, "a cut at row 37 reproduced one call: the comparison is not sensitive"


def test_fwd64_zero_length_sequences():
    """Empty sequences in cu_seqlens: every other sequence gets the bits of its own call, and with ONE buffer as
    initial_state and final_state (how prefill() calls the kernel on state.S) an empty sequence keeps its state."""
    H = 2
    p = make_params(H)
    lens = [0, 37, 0, 0, 64, 5, 130, 0]
    xs = [make_tokens(n, H, seed=40 + i) for i, n in enumerate(lens)]
    x = cat_tokens([t for t, n in zip(xs, lens) if n])
    _, cu = make_cu(lens)
    h0 = rand_state(len(lens), H, 4).to(torch.bfloat16)
    st = h0.clone()
    out = torch.full((1,) + tuple(x["q"].shape), NAN, dtype=torch.bfloat16, device=dev)
    R.fwd64(x["q"][None], x["k"][None], x["v"][None], x["g"][None], x["beta"][None], p["scale"], out, p["A_log"],
            p["dt_bias"], p["lower_bound"], initial_state=st, final_state=st, cu_seqlens=cu)
    off = 0
    for i, n in enumerate(lens):
        if n == 0:
            assert_bitwise(st[i], h0[i], f"empty sequence {i}: state changed")
            continue
        o, f = fwd64_full(xs[i], p, initial_state=h0[i:i + 1])
        assert_bitwise(out[0, off:off + n], o, f"sequence {i} outputs")
        assert_bitwise(st[i:i + 1], f, f"sequence {i} final state")
        off += n


def test_fwd64_zero_length_final_state_separate_buffer():
    """flash_kda.fwd's contract (test_replay.py::test_fwd_zero_length_sequences), which fwd64 adopts ("arguments as
    flash_kda.fwd"): with separate buffers, the final state of an empty sequence is its initial state."""
    H = 2
    p = make_params(H)
    lens = [0, 37, 0]
    x = make_tokens(37, H, seed=41)
    _, cu = make_cu(lens)
    h0 = rand_state(len(lens), H, 5).to(torch.bfloat16)
    out = torch.full((1, 37, H, D), NAN, dtype=torch.bfloat16, device=dev)
    fin = torch.full((len(lens), H, D, D), NAN, dtype=torch.bfloat16, device=dev)
    R.fwd64(x["q"][None], x["k"][None], x["v"][None], x["g"][None], x["beta"][None], p["scale"], out, p["A_log"],
            p["dt_bias"], p["lower_bound"], initial_state=h0, final_state=fin, cu_seqlens=cu)
    for i in (0, 2):
        assert_bitwise(fin[i], h0[i], f"empty sequence {i}: final_state is not its initial_state")


@pytest.mark.parametrize("layout", ["single", "batched", "varlen"])
def test_fwd64_k2_cfgs_identical(layout):
    """The four K2 CTA shapes (_k2_cfg 0..3) only move output elements between warps / CTAs: identical outputs and final
    states. Batched ([B, T, H, D]) and varlen sequences also equal their own single-sequence calls."""
    H = 3
    p = make_params(H, "bias_extreme")
    lens = {"single": [1000], "batched": [300, 300, 300], "varlen": [100, 64, 300, 236, 1, 63, 65, 128]}[layout]
    xs = [make_tokens(n, H, seed=60 + i) for i, n in enumerate(lens)]
    h0 = rand_state(len(lens), H, 6).to(torch.bfloat16)
    if layout == "varlen":
        x4, cu = {n: t[None] for n, t in cat_tokens(xs).items()}, make_cu(lens)[1]
    else:
        x4, cu = stack_tokens(xs), None
    res = [fwd64_call(x4, p, initial_state=h0, cu_seqlens=cu, k2_cfg=c) for c in range(4)]
    for c in range(1, 4):
        assert_bitwise(res[c][0], res[0][0], f"outputs, k2_cfg {c} vs 0")
        assert_bitwise(res[c][1], res[0][1], f"final state, k2_cfg {c} vs 0")
    if len(lens) > 1:
        out, off = res[0][0].reshape(-1, H, D), 0
        for i, n in enumerate(lens):
            o, f = fwd64_full(xs[i], p, initial_state=h0[i:i + 1])
            assert_bitwise(out[off:off + n], o, f"sequence {i} outputs vs its own call")
            assert_bitwise(res[0][1][i:i + 1], f, f"sequence {i} final state vs its own call")
            off += n


# ---------------------------------------------------------------------------------------------------------------------
# 2. one-token decode vs one fwd64 call
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("H", [1, 4])
@pytest.mark.parametrize("regime", ["default", "slow_decay", "fast_decay", "lb_small", "lb_zero", "bias_extreme"])
def test_decode_from_empty(H, regime):
    """130 one-token decode calls from an empty state (two chunk closes, pos ends at 2); final_state also checked after
    1, 63, 64, 65 and 128 tokens (pos 1, 63, 0, 1, 0)."""
    T = 130
    p, x = make_params(H, regime), make_tokens(T, H, seed=5)
    full, fin = fwd64_full(x, p)
    st = R.alloc_state(1, H)
    outs = []
    for t in range(T):
        outs.append(decode(slice_tokens(x, t, t + 1), p, st))
        if t + 1 in (1, 63, 64, 65, 128):
            assert_bitwise(fstate(st, p), fwd64_full(slice_tokens(x, 0, t + 1), p)[1], f"final state after {t + 1} tokens")
    assert_bitwise(torch.cat(outs), full, "decode outputs")
    assert_bitwise(fstate(st, p), fin, "final state")
    assert_pos(st, [0], [T], "after decode")


@pytest.mark.parametrize("regime", ["big_v", "collinear_k", "beta_extreme", "gate_extreme"])
def test_decode_data_regimes(regime):
    H, T = 4, 130
    p, x = make_params(H), make_tokens(T, H, regime=regime, seed=6)
    full, fin = fwd64_full(x, p)
    st = R.alloc_state(1, H)
    outs = [decode(slice_tokens(x, t, t + 1), p, st) for t in range(T)]
    assert_bitwise(torch.cat(outs), full, "decode outputs")
    assert_bitwise(fstate(st, p), fin, "final state")


@pytest.mark.parametrize("regime", Hh.REGIMES)
def test_c64_regimes_prefill_decode(regime):
    """The eight data regimes of tests/c64_helpers.py (normalised / heavy-tailed / eps-sized / one-hot q and k, long
    memory, no decay, saturated gates): a 37-token prompt then 103 one-token decode steps == one fwd64 call."""
    T, H, L = 140, 2, 37
    p, x = from_case(Hh.make_case(T, H, regime, seed=21))
    full, fin = fwd64_full(x, p)
    st = R.alloc_state(1, H)
    o = [prefill(slice_tokens(x, 0, L), p, st)] + [decode(slice_tokens(x, t, t + 1), p, st) for t in range(L, T)]
    assert_bitwise(torch.cat(o), full, "outputs")
    assert_bitwise(fstate(st, p), fin, "final state")


# ---------------------------------------------------------------------------------------------------------------------
# 3. prompt (default prefill path) + 70 one-token decode steps
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("prompt", [1, 5, 63, 64, 65, 127, 128, 129, 300, 1000])
@pytest.mark.parametrize("init", [None, "bf16", "fp32"])
def test_prefill_then_decode(prompt, init):
    H, n_dec = 4, 70
    p, x = make_params(H), make_tokens(prompt + n_dec, H, seed=prompt)
    h0 = init_state(init, 1, H, prompt)
    full, fin = fwd64_full(x, p, initial_state=h0)
    st = R.alloc_state(1, H, initial_state=h0)
    o1 = prefill(slice_tokens(x, 0, prompt), p, st)
    assert_bitwise(fstate(st, p), fwd64_full(slice_tokens(x, 0, prompt), p, initial_state=h0)[1],
                   "final state after the prompt")
    o2 = [decode(slice_tokens(x, t, t + 1), p, st) for t in range(prompt, prompt + n_dec)]
    assert_bitwise(torch.cat([o1] + o2), full, "outputs")
    assert_bitwise(fstate(st, p), fin, "final state after the decode steps")
    assert_bitwise(fstate(st, p, dtype=torch.float32), fin.float(), "final state as fp32")
    assert_pos(st, [0], [prompt + n_dec], "after decode")


# ---------------------------------------------------------------------------------------------------------------------
# 4. several tokens per decode call; batches with slot pools and padding entries
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("lead", [0, 61])
@pytest.mark.parametrize("step", [2, 3, 7, 16, 31, 63, 64, 65, 70, "mixed"])
def test_multi_token_decode(step, lead):
    """T tokens per decode call ([1, T, H, D]) after a `lead`-token prompt: calls that stay inside a chunk, close one,
    close one and open the next, and (T = 70 from row 61) close two; "mixed" varies T between calls (1..70)."""
    H, T_total = 3, 260
    p, x = make_params(H), make_tokens(T_total, H, seed=7)
    full, fin = fwd64_full(x, p)
    st = R.alloc_state(1, H)
    outs = [prefill(slice_tokens(x, 0, lead), p, st)] if lead else []
    sched = [5, 70, 1, 64, 2, 63, 33, 3, 65, 17] if step == "mixed" else [step]
    t, i = lead, 0
    while t < T_total:
        n = min(sched[i % len(sched)], T_total - t)
        xt = slice_tokens(x, t, t + n)
        outs.append(decode({k: v[None] for k, v in xt.items()}, p, st)[0])
        t, i = t + n, i + 1
    assert_bitwise(torch.cat(outs), full, "outputs")
    assert_bitwise(fstate(st, p), fin, "final state")
    assert_pos(st, [0], [T_total], "after decode")


@pytest.mark.parametrize("T", [1, 5, 9, 64, 70])
def test_decode_batch_slots_padding(T):
    """Batched decode, T tokens per entry (T = 1: the [N, H, D] layout): four sequences at different chunk rows on an
    8-slot pool, two -1 padding entries. Padding rows of out stay untouched; slots that no entry names (never used, or
    holding a parked 21-token sequence) keep every state bit."""
    H, P = 2, 8
    p = make_params(H)
    lens = [3, 64, 100, 130]                  # chunk rows after the prompt: 3, 0, 36, 2
    slots = [6, 0, 3, 2]
    park_slot, park_len = 4, 21
    idle = [1, 4, 5, 7]
    order = [3, -1, 1, 0, -1, 2]              # decode entries: sequence index, -1 = padding
    n_calls = -(-140 // T)
    xs = [make_tokens(L + T * n_calls, H, seed=500 + i) for i, L in enumerate(lens)]
    xpark = make_tokens(park_len, H, seed=599)
    h0 = rand_state(P, H, 50).to(torch.bfloat16)
    refs = [fwd64_full(xs[i], p, initial_state=h0[s:s + 1]) for i, s in enumerate(slots)]
    st = R.alloc_state(P, H, initial_state=h0)
    cu_h, cu = make_cu(lens + [park_len])
    op = prefill(cat_tokens([slice_tokens(xs[i], 0, L) for i, L in enumerate(lens)] + [xpark]), p, st,
                 cu_seqlens=cu, slot_ids=slot_tensor(slots + [park_slot]))
    snap = snapshot_slots(st, idle)
    sid = slot_tensor([slots[e] if e >= 0 else -1 for e in order])
    outs = [[op[cu_h[i]:cu_h[i + 1]]] for i in range(len(lens))]
    for c in range(n_calls):
        parts = [slice_tokens(xs[max(e, 0)], lens[max(e, 0)] + c * T, lens[max(e, 0)] + (c + 1) * T) for e in order]
        o = decode(cat_tokens(parts) if T == 1 else stack_tokens(parts), p, st, slot_ids=sid)
        for j, e in enumerate(order):
            if e >= 0:
                outs[e].append(o[j].reshape(T, H, D))
            else:
                assert_untouched(o[j], f"call {c}: padding entry {j} wrote its out rows")
        assert_slots_unchanged(st, snap, f"call {c}")
    fs = fstate(st, p, slot_ids=slot_tensor(slots + [park_slot]))
    for i in range(len(lens)):
        assert_bitwise(torch.cat(outs[i]), refs[i][0], f"sequence {i} (slot {slots[i]})")
        assert_bitwise(fs[i:i + 1], refs[i][1], f"sequence {i} final state")
    assert_bitwise(fs[-1:], fwd64_full(xpark, p, initial_state=h0[park_slot:park_slot + 1])[1], "parked slot final state")


# ---------------------------------------------------------------------------------------------------------------------
# 5. prefill: cuts, forced paths, varlen batches, continuation
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", [None, "replay", "fwd"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_chunked_prefill_random_cuts(seed, path):
    """A 700-token sequence cut at 7 random offsets (seed 3 adds cuts on the 64-grid) into prefill calls on the
    default path or a forced one."""
    H, T = 4, 700
    p, x = make_params(H), make_tokens(T, H, seed=8 + seed)
    full, fin = fwd64_full(x, p)
    rng = random.Random(seed)
    cuts = set(rng.sample(range(1, T), 7))
    if seed == 3:
        cuts |= {64, 128, 320, 640}
    cuts = sorted(cuts)
    st, outs, a = R.alloc_state(1, H), [], 0
    for b in cuts + [T]:
        outs.append(prefill(slice_tokens(x, a, b), p, st, path=path))
        a = b
    assert_bitwise(torch.cat(outs), full, f"outputs, cuts {cuts}")
    assert_bitwise(fstate(st, p), fin, "final state")
    assert_pos(st, [0], [T], "after the last call")


@pytest.mark.parametrize("T,cut", [(600, 0), (600, 37), (200, 63), (200, 64), (65, 1), (64, 63), (130, 64), (300, 100),
                                   (129, 0), (64, 0), (127, 1)])
def test_prefill_paths(T, cut):
    """Each prefill path forced on the same call (after a `cut`-token first call on the default path): same outputs and
    final state as one fwd64 call, and the same continuation state (S, pos, the open rows)."""
    H = 2
    p = make_params(H)
    x = make_tokens(T, H, seed=T + cut)
    full, fin = fwd64_full(x, p)
    states = {}
    for path in ("replay", "fwd"):
        st = R.alloc_state(1, H)
        o0 = prefill(slice_tokens(x, 0, cut), p, st) if cut else x["q"][:0]
        o1 = prefill(slice_tokens(x, cut, T), p, st, path=path)
        assert_bitwise(torch.cat([o0, o1]), full, f"path {path}: outputs")
        assert_bitwise(fstate(st, p), fin, f"path {path}: final state")
        states[path] = st
    assert_state_equivalent(states["replay"], states["fwd"], "replay path vs fwd path state")


@pytest.mark.parametrize("k2_cfg", [0, 1, 2, 3])
def test_prefill_split_k2_cfgs(k2_cfg):
    """The prefill split with each K2 CTA shape forced (two varlen rounds: heads, 64-aligned bodies, tails, an empty
    sequence) == one fwd64 call per sequence on the default shape."""
    H = 3
    p = make_params(H)
    rounds = [[5, 0, 64, 61], [700, 64, 129, 300]]
    N = len(rounds[0])
    xs = [make_tokens(sum(r[i] for r in rounds), H, seed=80 + i) for i in range(N)]
    refs = [fwd64_full(xs[i], p) for i in range(N)]
    st = R.alloc_state(N, H)
    ctx, outs = [0] * N, [[] for _ in range(N)]
    for lens in rounds:
        cu_h, cu = make_cu(lens)
        o = prefill(cat_tokens([slice_tokens(xs[i], ctx[i], ctx[i] + n) for i, n in enumerate(lens)]), p, st,
                    cu_seqlens=cu, path="fwd", k2_cfg=k2_cfg)
        for i in range(N):
            outs[i].append(o[cu_h[i]:cu_h[i + 1]])
        ctx = [c + n for c, n in zip(ctx, lens)]
    fs = fstate(st, p)
    for i in range(N):
        assert_bitwise(torch.cat(outs[i]), refs[i][0], f"sequence {i}")
        assert_bitwise(fs[i:i + 1], refs[i][1], f"sequence {i} final state")


@pytest.mark.parametrize("path", ["fwd", "replay"])
def test_varlen_batch(path):
    """Batched varlen prefill (sequences end at different chunk rows, bf16 initial states) + 70 lockstep decode steps;
    per sequence == its own fwd64 call."""
    H = 4
    p = make_params(H)
    lens = [1, 63, 64, 65, 100, 333, 128, 2]
    n_dec, N = 70, len(lens)
    xs = [make_tokens(n + n_dec, H, seed=90 + i) for i, n in enumerate(lens)]
    h0 = rand_state(N, H, 9).to(torch.bfloat16)
    refs = [fwd64_full(xs[i], p, initial_state=h0[i:i + 1]) for i in range(N)]
    st = R.alloc_state(N, H, initial_state=h0)
    cu_h, cu = make_cu(lens)
    op = prefill(cat_tokens([slice_tokens(xs[i], 0, n) for i, n in enumerate(lens)]), p, st, cu_seqlens=cu, path=path)
    od = [decode(cat_tokens([slice_tokens(xs[i], n + s, n + s + 1) for i, n in enumerate(lens)]), p, st)
          for s in range(n_dec)]
    fs = fstate(st, p)
    for i, n in enumerate(lens):
        got = torch.cat([op[cu_h[i]:cu_h[i + 1]]] + [o[i:i + 1] for o in od])
        assert_bitwise(got, refs[i][0], f"sequence {i} (len {n})")
        assert_bitwise(fs[i:i + 1], refs[i][1], f"sequence {i} final state")


@pytest.mark.parametrize("path", [None, "fwd", "replay"])
def test_varlen_continuation_mixed(path):
    """Three varlen prefill rounds on one state: sequences continue open chunks (heads), complete one exactly, carry
    64-aligned bodies, leave tails, or get 0 new tokens; outputs, pos and final states as one fwd64 call each."""
    H = 2
    p = make_params(H)
    rounds = [[5, 64, 30, 3, 1, 100],
              [140, 3, 34, 61, 64, 200],   # split: heads 59/0/34/61/63/28, bodies 64/0/0/0/0/128, tails 17/3/0/0/1/44
              [0, 70, 1, 64, 129, 0]]      # rows before this round: 17/3/0/0/1/44
    N = len(rounds[0])
    xs = [make_tokens(sum(r[i] for r in rounds), H, seed=120 + i) for i in range(N)]
    refs = [fwd64_full(xs[i], p) for i in range(N)]
    st = R.alloc_state(N, H)
    ctx, outs = [0] * N, [[] for _ in range(N)]
    for lens in rounds:
        cu_h, cu = make_cu(lens)
        o = prefill(cat_tokens([slice_tokens(xs[i], ctx[i], ctx[i] + n) for i, n in enumerate(lens)]), p, st,
                    cu_seqlens=cu, path=path)
        for i in range(N):
            outs[i].append(o[cu_h[i]:cu_h[i + 1]])
        ctx = [c + n for c, n in zip(ctx, lens)]
        assert_pos(st, list(range(N)), ctx, f"after round {lens}")
    fs = fstate(st, p)
    for i in range(N):
        assert_bitwise(torch.cat(outs[i]), refs[i][0], f"sequence {i}")
        assert_bitwise(fs[i:i + 1], refs[i][1], f"sequence {i} final state")


@pytest.mark.parametrize("path", ["fwd", "replay"])
def test_prefill_lead_offset_and_zero_length(path):
    """Second prefill round with cu_seqlens[0] = 3 (rows before it stay untouched), a sequence with 0 new tokens,
    sequences that only extend their open chunk, and one with head 60 + body 64 + tail 6."""
    H = 2
    p = make_params(H)
    first = [7, 70, 9, 21, 4]
    second = [5, 40, 0, 2, 130]
    N = len(first)
    xs = [make_tokens(a + b, H, seed=600 + i) for i, (a, b) in enumerate(zip(first, second))]
    refs = [fwd64_full(xs[i], p) for i in range(N)]
    st = R.alloc_state(N, H)
    outs = [[] for _ in range(N)]
    for lens, offs, lead in ((first, [0] * N, 0), (second, first, 3)):
        parts = [slice_tokens(xs[i], offs[i], offs[i] + n) for i, n in enumerate(lens)]
        pad = [make_tokens(lead, H, seed=699)] if lead else []          # junk rows before cu_seqlens[0]
        cu_h, cu = make_cu(lens, lead=lead)
        o = prefill(cat_tokens(pad + parts), p, st, cu_seqlens=cu, path=path)
        for i in range(N):
            outs[i].append(o[cu_h[i]:cu_h[i + 1]])
        if lead:
            assert_untouched(o[:lead], "rows before cu_seqlens[0] were written")
    fs = fstate(st, p)
    for i in range(N):
        assert_bitwise(torch.cat(outs[i]), refs[i][0], f"sequence {i}")
        assert_bitwise(fs[i:i + 1], refs[i][1], f"sequence {i} final state")


@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("T", [64, 100])
def test_batched_mode(T, pooled):
    """[B, T, H, D] prefill without cu_seqlens (T = 64: replay path, closes the chunk; T = 100: the split), optionally on
    a slot pool, then 30 one-token decode steps."""
    H, B, n_dec = 3, 5, 30
    p = make_params(H)
    xs = [make_tokens(T + n_dec, H, seed=150 + b) for b in range(B)]
    refs = [fwd64_full(xs[b], p) for b in range(B)]
    slots = [6, 2, 0, 5, 3] if pooled else list(range(B))
    sid = slot_tensor(slots) if pooled else None
    st = R.alloc_state(7 if pooled else B, H)
    xb = stack_tokens([slice_tokens(xs[b], 0, T) for b in range(B)])
    out = torch.full_like(xb["q"], NAN)
    R.prefill(xb["q"], xb["k"], xb["v"], xb["g"], xb["beta"], p["scale"], out, p["A_log"], p["dt_bias"],
              p["lower_bound"], st, slot_ids=sid)
    od = [decode(cat_tokens([slice_tokens(xs[b], T + s, T + s + 1) for b in range(B)]), p, st, slot_ids=sid)
          for s in range(n_dec)]
    fs = fstate(st, p, slot_ids=slot_tensor(slots))
    for b in range(B):
        assert_bitwise(torch.cat([out[b]] + [o[b:b + 1] for o in od]), refs[b][0], f"batch entry {b}")
        assert_bitwise(fs[b:b + 1], refs[b][1], f"batch entry {b} final state")


@pytest.mark.parametrize("path", ["replay", "fwd"])
def test_large_random_sweep(path):
    """24 sequences of random length (1..400), 16 heads, extreme gate biases and gate inputs, bf16 initial states: one
    varlen prefill on each forced path == one varlen fwd64 call."""
    H = 16
    p = make_params(H, "bias_extreme")
    gen = random.Random(14)
    lens = [gen.randint(1, 400) for _ in range(24)]
    x = make_tokens(sum(lens), H, regime="gate_extreme", seed=14)
    _, cu = make_cu(lens)
    h0 = rand_state(len(lens), H, 14).to(torch.bfloat16)
    full, fin = fwd64_full(x, p, initial_state=h0, cu_seqlens=cu)
    st = R.alloc_state(len(lens), H, initial_state=h0)
    o = prefill(x, p, st, cu_seqlens=cu, path=path)
    assert_bitwise(o, full, "outputs")
    assert_bitwise(fstate(st, p), fin, "final state")


@pytest.mark.parametrize("path", [None, "fwd"])
@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_prefill_host_plan(batched, pooled, path, monkeypatch):
    """prefill(context_lens_cpu=..., cu_seqlens_cpu=...) plans the split on the host. Rounds: fresh prompts whose lengths
    are multiples of 64 (bodies only), body + tail, head + body, heads + tails with no body anywhere (no fwd64 launch),
    tails only, heads only, (varlen) sequences with 0 new tokens. Every round: the host plan runs exactly when the split
    is taken (default: > 64 tokens per sequence on average; path "fwd" forces it) and launches only its non-empty
    phases, fwd64 on the non-empty bodies; outputs == the device plan (the same calls without context_lens_cpu on a twin
    state), equivalent states. At the end: outputs and final states == one fwd64 call per sequence; idle pool slots
    untouched. pooled also passes the host arrays as CPU tensors instead of lists."""
    H, N = 3, 4
    p = make_params(H)
    rounds = ([[L] * N for L in (128, 85, 107, 122, 65, 69, 3, 61)] if batched else
              [[128, 64, 192, 64], [5, 140, 70, 0], [59, 3, 200, 130], [0, 61, 1, 64], [100, 0, 0, 1]])
    slots = [5, 1, 3, 0] if pooled else list(range(N))
    P = 6 if pooled else N
    sid = slot_tensor(slots) if pooled else None
    xs = [make_tokens(sum(r[i] for r in rounds), H, seed=300 + i) for i in range(N)]
    h0 = rand_state(P, H, 30).to(torch.bfloat16)
    refs = [fwd64_full(xs[i], p, initial_state=h0[s:s + 1]) for i, s in enumerate(slots)]
    st_host, st_dev = R.alloc_state(P, H, initial_state=h0), R.alloc_state(P, H, initial_state=h0)
    idle = snapshot_slots(st_host, [s for s in range(P) if s not in slots])
    log = spy_launches(monkeypatch)
    ctx, outs = [0] * N, [[] for _ in range(N)]
    for lens in rounds:
        parts = [slice_tokens(xs[i], ctx[i], ctx[i] + n) for i, n in enumerate(lens)]
        cu_h, cu = make_cu(lens)
        res = {}
        for plan, st in (("host", st_host), ("device", st_dev)):
            kw = dict(slot_ids=sid, _path=path)
            if plan == "host":
                kw["context_lens_cpu"] = torch.tensor(ctx) if pooled else list(ctx)
            if batched:
                xb = stack_tokens(parts)
                o = torch.full_like(xb["q"], NAN)
                log.clear()
                R.prefill(xb["q"], xb["k"], xb["v"], xb["g"], xb["beta"], p["scale"], o, p["A_log"], p["dt_bias"],
                          p["lower_bound"], st, **kw)
                res[plan] = [o[i] for i in range(N)]
            else:
                if plan == "host":
                    kw["cu_seqlens_cpu"] = torch.tensor(cu_h) if pooled else cu_h
                xp = cat_tokens(parts)
                o = torch.full((1,) + tuple(xp["q"].shape), NAN, dtype=torch.bfloat16, device=dev)
                log.clear()
                R.prefill(xp["q"][None], xp["k"][None], xp["v"][None], xp["g"][None], xp["beta"][None], p["scale"], o,
                          p["A_log"], p["dt_bias"], p["lower_bound"], st, cu_seqlens=cu, **kw)
                res[plan] = [o[0, cu_h[i]:cu_h[i + 1]] for i in range(N)]
            if plan == "host":
                split = path == "fwd" or sum(lens) > C * N
                expected = host_plan_launches(lens, ctx) if split else [("replay",)]
                assert log == expected, f"round {lens} at rows {[c % C for c in ctx]}: launches {log}, expected {expected}"
        for i in range(N):
            assert_bitwise(res["host"][i], res["device"][i], f"round {lens}: sequence {i}, host plan vs device plan")
            outs[i].append(res["host"][i])
        ctx = [c + n for c, n in zip(ctx, lens)]
        assert_pos(st_host, slots, ctx, f"host plan, after round {lens}")
        assert_state_equivalent(st_host, st_dev, f"after round {lens}: host plan vs device plan state")
    fs = fstate(st_host, p, slot_ids=slot_tensor(slots))
    for i in range(N):
        assert_bitwise(torch.cat(outs[i]), refs[i][0], f"sequence {i} (slot {slots[i]})")
        assert_bitwise(fs[i:i + 1], refs[i][1], f"sequence {i} final state")
    assert_slots_unchanged(st_host, idle, "host plan: idle slots")


# ---------------------------------------------------------------------------------------------------------------------
# 6. final_state
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", [None, "fwd"])
def test_final_state_slots_and_boundary(path):
    """final_state with slot_ids (permuted, incl. a slot never written) for slots at a chunk boundary (pos 0) and
    mid-chunk (pos 7, 21, 63), in bf16 and fp32 and without slot_ids; it never changes any state bit, and decoding
    continues from the state afterwards (2 steps, the pos-63 slot closes its chunk)."""
    H, P = 2, 6
    p = make_params(H)
    lens = [128, 21, 64, 7, 63]               # pos after the prefill: 0, 21, 0, 7, 63
    slots = [4, 0, 2, 1, 5]                   # slot 3 is never written
    n_dec = 2
    xs = [make_tokens(L + n_dec, H, seed=700 + i) for i, L in enumerate(lens)]
    h0 = rand_state(P, H, 70).to(torch.bfloat16)
    full = [fwd64_full(xs[i], p, initial_state=h0[s:s + 1]) for i, s in enumerate(slots)]
    ref = {s: h0[s:s + 1] for s in range(P)}  # expected final state per slot after the prefill
    for i, s in enumerate(slots):
        ref[s] = fwd64_full(slice_tokens(xs[i], 0, lens[i]), p, initial_state=h0[s:s + 1])[1]
    st = R.alloc_state(P, H, initial_state=h0)
    cu_h, cu = make_cu(lens)
    sid = slot_tensor(slots)
    op = prefill(cat_tokens([slice_tokens(xs[i], 0, L) for i, L in enumerate(lens)]), p, st, cu_seqlens=cu,
                 slot_ids=sid, path=path)
    assert_pos(st, slots, lens, "after the prefill")
    before = st.clone()
    query = [2, 4, 3, 1, 0, 5]
    fs = fstate(st, p, slot_ids=slot_tensor(query))
    fs32 = fstate(st, p, slot_ids=slot_tensor(query), dtype=torch.float32)
    fs_all = fstate(st, p)
    assert_state_identical(st, before, "final_state changed the state")
    assert fs.dtype == torch.bfloat16 and fs32.dtype == torch.float32
    for j, s in enumerate(query):
        assert_bitwise(fs[j:j + 1], ref[s], f"slot {s} (pos {int(st.pos[s, 0])}) final state")
        assert_bitwise(fs32[j:j + 1], ref[s].float(), f"slot {s} final state as fp32")
        assert_bitwise(fs_all[s:s + 1], ref[s], f"slot {s} final state without slot_ids")
    od = [decode(cat_tokens([slice_tokens(xs[i], L + k, L + k + 1) for i, L in enumerate(lens)]), p, st, slot_ids=sid)
          for k in range(n_dec)]
    fs = fstate(st, p, slot_ids=sid)
    for i, L in enumerate(lens):
        got = torch.cat([op[cu_h[i]:cu_h[i + 1]]] + [o[i:i + 1] for o in od])
        assert_bitwise(got, full[i][0], f"sequence {i} (slot {slots[i]}): outputs")
        assert_bitwise(fs[i:i + 1], full[i][1], f"sequence {i} (slot {slots[i]}): final state after decode")


# ---------------------------------------------------------------------------------------------------------------------
# 7. head counts, CUDA graphs, determinism
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("H", [1, 32, 96])
def test_head_counts(H):
    """H = 1 / 32 / 96 (auto K2 shapes 3 / 2 / 1): a 3-token prompt, a 187-token prefill continuing it at row 3 through
    the split (head 61, body 64, tail 62), then 12 decode steps across the 192 boundary."""
    T0, n_dec = 190, 12
    p, x = make_params(H), make_tokens(T0 + n_dec, H, seed=13)
    full, fin = fwd64_full(x, p)
    st = R.alloc_state(1, H)
    o = [prefill(slice_tokens(x, 0, 3), p, st), prefill(slice_tokens(x, 3, T0), p, st, path="fwd")]
    o += [decode(slice_tokens(x, t, t + 1), p, st) for t in range(T0, T0 + n_dec)]
    assert_bitwise(torch.cat(o), full, "outputs")
    assert_bitwise(fstate(st, p), fin, "final state")


def test_cuda_graph_decode():
    """A captured one-token decode step (6 sequences at different chunk rows, no slot_ids) replayed for 90 tokens: every
    step == an eager twin, every state bit == the twin's at the end, and == one fwd64 call per sequence."""
    H = 4
    p = make_params(H)
    lens = [50, 1, 64, 100, 127, 7]
    N, n_dec = len(lens), 90
    xs = [make_tokens(L + n_dec, H, seed=200 + i) for i, L in enumerate(lens)]
    refs = [fwd64_full(xs[i], p) for i in range(N)]
    st = R.alloc_state(N, H)
    prefill(cat_tokens([slice_tokens(xs[i], 0, L) for i, L in enumerate(lens)]), p, st, cu_seqlens=make_cu(lens)[1])
    st_eager = st.clone()
    buf = cat_tokens([slice_tokens(xs[i], L, L + 1) for i, L in enumerate(lens)])
    out = torch.empty_like(buf["q"])
    graph = capture_decode(buf, out, p, st)
    got = []
    for s_ in range(n_dec):
        nxt = cat_tokens([slice_tokens(xs[i], L + s_, L + s_ + 1) for i, L in enumerate(lens)])
        for n in TOK:
            buf[n].copy_(nxt[n])
        graph.replay()
        got.append(out.clone())
        assert_bitwise(got[-1], decode(buf, p, st_eager), f"step {s_}: graph vs eager outputs")
    assert_state_identical(st, st_eager, "graph vs eager state")
    got = torch.stack(got)                    # [n_dec, N, H, D]
    fs = fstate(st, p)
    for i, L in enumerate(lens):
        assert_bitwise(got[:, i], refs[i][0][L:], f"sequence {i}")
        assert_bitwise(fs[i:i + 1], refs[i][1], f"sequence {i} final state")


@pytest.mark.parametrize("T", [1, 3])
def test_cuda_graph_serving_config(T):
    """Captured decode (T tokens per entry) on a 7-slot pool with 2 padding entries, replayed with new inputs and a slot
    mapping that rotates every step: == an eager twin (outputs incl. the untouched padding rows, every state bit) and
    == one fwd64 call per sequence."""
    H, P = 3, 7
    p = make_params(H)
    lens = [20, 63, 64, 100]
    slots = [5, 1, 3, 0]
    N, E = len(lens), len(lens) + 2
    n_steps = 140 // T
    xs = [make_tokens(L + T * n_steps, H, seed=900 + i) for i, L in enumerate(lens)]
    refs = [fwd64_full(xs[i], p) for i in range(N)]
    st = R.alloc_state(P, H)
    prefill(cat_tokens([slice_tokens(xs[i], 0, L) for i, L in enumerate(lens)]), p, st, cu_seqlens=make_cu(lens)[1],
            slot_ids=slot_tensor(slots))
    st_eager = st.clone()
    one = slice_tokens(xs[0], 0, T)
    buf = cat_tokens([one] * E) if T == 1 else stack_tokens([one] * E)
    sid = torch.full((E,), -1, dtype=torch.int32, device=dev)
    out = torch.empty_like(buf["q"])
    graph = capture_decode(buf, out, p, st, slot_ids=sid)
    outs = [[] for _ in range(N)]
    for step in range(n_steps):
        perm = [(step + j) % E for j in range(E)]          # graph entry perm[i] serves sequence i (i < N)
        new_sid = [-1] * E
        for i in range(N):
            e = perm[i]
            new_sid[e] = slots[i]
            nxt = slice_tokens(xs[i], lens[i] + step * T, lens[i] + (step + 1) * T)
            for n in TOK:
                buf[n][e].copy_(nxt[n].view_as(buf[n][e]))
        sid.copy_(slot_tensor(new_sid))
        out.fill_(NAN)
        graph.replay()
        assert_bitwise(out, decode(buf, p, st_eager, slot_ids=sid), f"step {step}: graph vs eager outputs")
        for e in perm[N:]:
            assert_untouched(out[e], f"step {step}: padding entry {e} wrote its out rows")
        for i in range(N):
            outs[i].append(out[perm[i]].reshape(T, H, D).clone())
    assert_state_identical(st, st_eager, "graph vs eager state")
    fs = fstate(st, p, slot_ids=slot_tensor(slots))
    for i, L in enumerate(lens):
        assert_bitwise(torch.cat(outs[i]), refs[i][0][L:], f"sequence {i} (slot {slots[i]})")
        assert_bitwise(fs[i:i + 1], refs[i][1], f"sequence {i} final state")


def test_deterministic():
    """Two identical runs (prefill + decode + a replay-path prefill, final_state, fwd64) give identical bits, every state
    bit included."""
    H = 4
    p, x = make_params(H), make_tokens(300, H, seed=12)
    res = []
    for _ in range(2):
        st = R.alloc_state(1, H)
        o = [prefill(slice_tokens(x, 0, 133), p, st)]
        o += [decode(slice_tokens(x, t, t + 1), p, st) for t in range(133, 200)]
        o.append(prefill(slice_tokens(x, 200, 300), p, st, path="replay"))
        res.append((torch.cat(o), st.clone(), fstate(st, p)) + fwd64_full(x, p))
    (o_a, st_a, fs_a, f_a, fin_a), (o_b, st_b, fs_b, f_b, fin_b) = res
    assert_bitwise(o_a, o_b, "outputs")
    assert_state_identical(st_a, st_b, "state")
    assert_bitwise(fs_a, fs_b, "final_state")
    assert_bitwise(f_a, f_b, "fwd64 outputs")
    assert_bitwise(fin_a, fin_b, "fwd64 final state")
    assert_bitwise(o_a, f_a, "outputs vs fwd64")
    assert_bitwise(fs_a, fin_a, "final_state vs fwd64")


# ---------------------------------------------------------------------------------------------------------------------
# 8. accuracy of the chunk-64 program (the bits everything above reproduces) against the fp64 recurrence
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("regime", ["authors", "heavy_tail", "long_memory"])
def test_accuracy_vs_fp64(regime):
    """prefill + decode outputs (== fwd64) are finite and their relative rms error vs tests/c64_helpers.exact_recurrence
    is below 6x the bf16 rounding floor of the exact outputs (bring-up on B300: 2.9-3.8x at T = 1024)."""
    T, H, n_dec = 320, 2, 40
    case = Hh.make_case(T, H, regime, seed=1)
    p, x = from_case(case)
    full, _ = fwd64_full(x, p)
    st = R.alloc_state(1, H)
    o = [prefill(slice_tokens(x, 0, T - n_dec), p, st)]
    o += [decode(slice_tokens(x, t, t + 1), p, st) for t in range(T - n_dec, T)]
    got = torch.cat(o)
    assert_bitwise(got, full, "prefill + decode vs fwd64")
    assert bool(torch.isfinite(got.float()).all()), "non-finite outputs"
    ref = Hh.exact_recurrence(case)[0]
    floor = Hh.rel_rms(ref.to(torch.bfloat16), ref)
    ratio = Hh.rel_rms(got, ref) / floor
    assert ratio < 6.0, f"{regime}: rel. rms error = {ratio:.2f}x the bf16 floor"
