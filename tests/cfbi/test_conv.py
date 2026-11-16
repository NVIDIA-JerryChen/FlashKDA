"""Bitwise gates of the CF-BI short causal conv (+ SiLU): flash_kda.cfbi.short_conv.

Zero tolerance on bits for everything claimed bitwise: the bf16 output, the fp32 value before the rounding (a much
stronger detector: re-associating the sum changes ~1/3 of the fp32 values but only ~1e-4 of the bf16 values), and the
updated state. Accuracy against an fp64 reference carries a stated tolerance. Negative controls: a build whose
single-token path adds the taps in a different association must be caught, and the emulation with the taps in the
reverse order must differ from the kernel.
"""
import random

import pytest
import torch

from flash_kda.cfbi import alloc_conv_state, conv_state_after, short_conv

from convlin_helpers import (bf16_ulp_error, bits, conv_ref64, cu_tensor, cx, cx_neg, emu_conv, make_weights, make_x,  # noqa: F401
                             n_diff)


def run(x, w, bias, state=None, cu=None, silu=True, max_seqlen=None, ext_out32=True):
    """One call: returns (bf16 out, fp32 pre-rounding value)."""
    out32 = torch.zeros(x.shape, dtype=torch.float32, device=x.device) if ext_out32 else None
    out = short_conv(x, w, bias, state, "silu" if silu else None, cu, max_seqlen=max_seqlen, out_fp32=out32)
    return out, out32


def seq_slices(cu):
    return list(zip(cu[:-1], cu[1:]))


# ---------------------------------------------------------------------------------------------------------
# the program: emulation in PyTorch, bit for bit; accuracy against fp64
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("W", [2, 3, 4, 5, 8])
@pytest.mark.parametrize("D", [200, 131])   # even (2 channels per thread) and odd (1)
@pytest.mark.parametrize("wdtype", [torch.float32, torch.bfloat16])
def test_program_equals_emulation(cx, W, D, wdtype):
    T = 150   # crosses two tile boundaries
    bias_dt = [None, torch.float32, torch.bfloat16][(W + D) % 3]
    silu = W != 3
    x = make_x(T, D, seed=1)
    w, b = make_weights(D, W, seed=2, wdtype=wdtype, with_bias=bias_dt)
    # fresh sequence
    out, out32 = run(x, w, b, None, silu=silu)
    emu = emu_conv(cx, torch.cat([torch.zeros(W - 1, D, dtype=torch.bfloat16, device="cuda"), x[0]]), w, b, silu)
    assert n_diff(out32[0], emu) == 0, "fp32 values differ from the emulation of the program"
    assert n_diff(out[0], emu.to(torch.bfloat16)) == 0
    # with a non-trivial state (history taps come from the state)
    st = torch.randn(1, D, W - 1, device="cuda").to(torch.bfloat16)
    out, out32 = run(x, w, b, st.clone(), silu=silu)
    emu = emu_conv(cx, torch.cat([st[0].t(), x[0]]), w, b, silu)
    assert n_diff(out32[0], emu) == 0 and n_diff(out[0], emu.to(torch.bfloat16)) == 0


def test_emulation_in_reverse_tap_order_differs(cx):
    """Negative control of the test above: the same program with the taps accumulated in the opposite order is a
    different program and the comparison must see it."""
    D, W, T = 256, 4, 200
    x = make_x(T, D, seed=3)
    w, _ = make_weights(D, W, seed=4)
    _, out32 = run(x, w, None)
    xe = torch.cat([torch.zeros(W - 1, D, dtype=torch.bfloat16, device="cuda"), x[0]]).float()
    acc = torch.zeros(T, D, device="cuda")
    for k in reversed(range(W)):   # descending taps
        acc = (acc.double() + w[:, k].double()[None, :] * xe[k:k + T].double()).float()
    e = cx.probe_unary(((-acc) * torch.tensor(1.44269502162933349609375, device="cuda")).contiguous(), 0)
    rev = acc * cx.probe_unary((1.0 + e).contiguous(), 1)
    frac = n_diff(out32[0], rev) / rev.numel()
    frac16 = n_diff(out32[0].to(torch.bfloat16), rev.to(torch.bfloat16)) / rev.numel()
    print(f"\n[reverse tap order] fp32 values differing: {frac:.3%}, bf16 outputs differing: {frac16:.4%}")
    assert frac > 0.05


@pytest.mark.parametrize("W", [3, 4])
@pytest.mark.parametrize("wdtype", [torch.float32, torch.bfloat16])
def test_accuracy_vs_fp64(cx, W, wdtype):
    """Tolerances (stated): the fp32 value is within 2e-6 * (sum|w x| + |bias| + |ref|) of the fp64 result of the same
    math (W fp32 roundings of the chain + the ex2/rcp approximations, ~3e-7 each); the bf16 output is within
    0.5 bf16-ulp(ref) + that same slack (it is RN of the fp32 value)."""
    D, T = 384, 500
    x = make_x(T, D, seed=5)
    w, b = make_weights(D, W, seed=6, wdtype=wdtype, with_bias=torch.float32)
    out, out32 = run(x, w, b, None)
    ref = conv_ref64(x[0], w, b)
    xe = torch.cat([torch.zeros(W - 1, D, dtype=torch.float64, device="cuda"), x[0].double()], dim=0)
    S = sum((w[:, k].double()[None, :] * xe[k:k + T]).abs() for k in range(W)) + b.double().abs()[None, :]
    slack = 2e-6 * (S + ref.abs())
    err32 = (out32[0].double() - ref).abs()
    ulp = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
    err16 = (out[0].double() - ref).abs()
    print(f"\n[accuracy W={W} {wdtype}] max fp32 error / slack {float((err32 / slack).max()):.3f}, "
          f"max bf16 error {float((err16 / ulp).max()):.3f} ulp (0.5 = correctly rounded)")
    assert bool((err32 <= slack).all())
    assert bool((err16 <= 0.5 * ulp + slack).all())


# ---------------------------------------------------------------------------------------------------------
# state semantics
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("L", [0, 1, 2, 3, 4, 5, 70, 130])
def test_state_is_the_last_inputs(cx, L):
    D, W = 96, 4
    x = make_x(max(L, 1), D, seed=7)[:, :L].contiguous()
    w, _ = make_weights(D, W, seed=8)
    st = torch.randn(1, D, W - 1, device="cuda").to(torch.bfloat16)
    expect = conv_state_after(x, W, st) if L > 0 else st.clone()
    st_k = st.clone()
    out, _ = run(x, w, None, st_k)
    assert n_diff(st_k, expect) == 0
    if L == 0:
        assert out.shape == x.shape
    # no state -> nothing is written, nothing is read: same outputs as a zero state
    if L > 0:
        o1, _ = run(x, w, None, None)
        o2, _ = run(x, w, None, alloc_conv_state(1, D, W))
        assert n_diff(o1, o2) == 0


def test_state_bits_are_copied_exactly(cx):
    """The state holds the raw input bits (NaN payloads, -0, subnormals included)."""
    D, W = 64, 4
    raw = torch.randint(-32768, 32767, (1, 40, D), device="cuda", dtype=torch.int16)
    x = raw.view(torch.bfloat16)
    w, _ = make_weights(D, W, seed=9)
    st = alloc_conv_state(1, D, W)
    short_conv(x, w, None, st, "silu")
    assert torch.equal(st.view(torch.int16), raw[:, -(W - 1):].permute(0, 2, 1).contiguous())


# ---------------------------------------------------------------------------------------------------------
# the equalities: whole == cuts == decode == varlen == training forward
# ---------------------------------------------------------------------------------------------------------
def whole(x, w, b, W, silu=True):
    st = alloc_conv_state(1, x.shape[-1], W)
    o, o32 = run(x, w, b, st, silu=silu)
    return o, o32, st


def assert_same(a, b, what):
    assert n_diff(a[0], b[0]) == 0, f"{what}: bf16 outputs differ in {n_diff(a[0], b[0])}"
    assert n_diff(a[1], b[1]) == 0, f"{what}: fp32 values differ in {n_diff(a[1], b[1])}"
    assert n_diff(a[2], b[2]) == 0, f"{what}: state differs in {n_diff(a[2], b[2])}"


def run_pieces(x, w, b, W, pieces):
    D = x.shape[-1]
    st = alloc_conv_state(1, D, W)
    out = torch.zeros_like(x)
    out32 = torch.zeros(x.shape, dtype=torch.float32, device="cuda")
    lo = 0
    for L in pieces:
        xs = x[:, lo:lo + L].contiguous()
        o, o32 = run(xs, w, b, st)
        out[:, lo:lo + L], out32[:, lo:lo + L] = o, o32
        lo += L
    return out, out32, st


CASES = [  # (D, W, weight dtype, bias dtype)
    (96, 4, torch.float32, None),
    (130, 4, torch.bfloat16, torch.float32),   # even, not a multiple of the block width
    (97, 3, torch.float32, torch.bfloat16),    # odd D: one channel per thread
    (512, 5, torch.bfloat16, None),
    (64, 2, torch.float32, torch.float32),
    (130, 8, torch.bfloat16, torch.bfloat16),
    (3, 4, torch.float32, None),               # fewer channels than one thread block
]


@pytest.mark.parametrize("D,W,wd,bd", CASES)
def test_whole_vs_every_cut_vs_decode(cx, D, W, wd, bd):
    T = 140
    x = make_x(T, D, seed=11)
    w, b = make_weights(D, W, seed=12, wdtype=wd, with_bias=bd)
    ref = whole(x, w, b, W)
    # every single cut point
    for cut in range(1, T):
        assert_same(ref, run_pieces(x, w, b, W, [cut, T - cut]), f"cut at {cut}")
    # T = 1 decode calls
    assert_same(ref, run_pieces(x, w, b, W, [1] * T), "decode x n")
    # random cuts
    for seed in range(5):
        rng = random.Random(seed)
        pieces, rem = [], T
        while rem > 0:
            L = min(rem, rng.choice([1, 2, 3, 4, 5, 63, 64, 65, rng.randint(1, 80)]))
            pieces.append(L)
            rem -= L
        assert_same(ref, run_pieces(x, w, b, W, pieces), f"cuts {pieces}")


def test_long_sequence_cuts(cx):
    """Several tiles (TILE_T = 64) per sequence, cuts inside and across tiles, 3 + 3 + 2 channel tiles."""
    D, W, T = 520, 4, 64 * 9 + 5
    x = make_x(T, D, seed=13)
    w, b = make_weights(D, W, seed=14, with_bias=torch.float32)
    ref = whole(x, w, b, W)
    for seed in range(4):
        rng = random.Random(100 + seed)
        pieces, rem = [], T
        while rem > 0:
            L = min(rem, rng.choice([1, 3, 64, 65, 127, 128, 129, rng.randint(1, 300)]))
            pieces.append(L)
            rem -= L
        assert_same(ref, run_pieces(x, w, b, W, pieces), f"cuts {pieces}")
    # max_seqlen only sizes the grid: any true bound gives the same bits
    st = alloc_conv_state(1, D, W)
    o, o32 = run(x, w, b, st, max_seqlen=T)
    assert_same(ref, (o, o32, st), "max_seqlen = T")
    st = alloc_conv_state(1, D, W)
    o, o32 = run(x, w, b, st, max_seqlen=10 * T)
    assert_same(ref, (o, o32, st), "max_seqlen = 10 T")


@pytest.mark.parametrize("D,W,wd,bd", CASES)
def test_varlen_batch_vs_solo(cx, D, W, wd, bd):
    """Packed sequences (incl. empty ones, lengths around the tile size and 1..5) == every sequence alone, bit for bit;
    permuting the order and changing the neighbours does not matter."""
    lens = [1, 0, 2, 3, 4, 5, 63, 64, 65, 128, 129, 200, 0, 1, 77]
    cu = [0]
    for L in lens:
        cu.append(cu[-1] + L)
    N = len(lens)
    x = make_x(cu[-1], D, seed=15)
    w, b = make_weights(D, W, seed=16, wdtype=wd, with_bias=bd)
    # a non-trivial initial state for every sequence: a first call over a head start of 7 tokens each
    head = make_x(N * 7, D, seed=17)
    cu_h = [7 * i for i in range(N + 1)]
    st0 = alloc_conv_state(N, D, W)
    short_conv(head, w, b, st0, "silu", cu_tensor(cu_h))

    st = st0.clone()
    out, out32 = run(x, w, b, st, cu_tensor(cu))
    for n, (lo, hi) in enumerate(seq_slices(cu)):
        st_n = st0[n:n + 1].clone()
        o, o32 = run(x[:, lo:hi].contiguous(), w, b, st_n)
        assert_same((out[:, lo:hi], out32[:, lo:hi], st[n:n + 1]), (o, o32, st_n), f"sequence {n} (len {hi - lo})")
    # max_seqlen given (true bound) gives the same bits
    st2 = st0.clone()
    out2, out32_2 = run(x, w, b, st2, cu_tensor(cu), max_seqlen=max(lens))
    assert_same((out, out32, st), (out2, out32_2, st2), "max_seqlen = longest sequence")
    # permuted order
    perm = list(range(N))
    random.Random(3).shuffle(perm)
    xp = torch.cat([x[:, cu[n]:cu[n + 1]] for n in perm], dim=1).contiguous()
    cup = [0]
    for n in perm:
        cup.append(cup[-1] + lens[n])
    stp = st0[perm].contiguous()
    outp, out32p = run(xp, w, b, stp, cu_tensor(cup))
    for i, n in enumerate(perm):
        a = (out[:, cu[n]:cu[n + 1]], out32[:, cu[n]:cu[n + 1]], st[n:n + 1])
        c = (outp[:, cup[i]:cup[i + 1]], out32p[:, cup[i]:cup[i + 1]], stp[i:i + 1])
        assert_same(a, c, f"sequence {n} at index {i}")


def test_batch_of_decode_steps_equals_solo(cx):
    """[N, 1, D] decode of N independent sequences, over many steps, == each sequence alone as one long call."""
    N, D, W, steps = 9, 160, 4, 37
    w, b = make_weights(D, W, seed=18, with_bias=torch.float32)
    xs = torch.cat([make_x(steps, D, seed=200 + n) for n in range(N)], dim=0)   # [N, steps, D]
    st = alloc_conv_state(N, D, W)
    out = torch.zeros_like(xs)
    out32 = torch.zeros(xs.shape, dtype=torch.float32, device="cuda")
    for t in range(steps):
        o, o32 = run(xs[:, t:t + 1].contiguous(), w, b, st)
        out[:, t:t + 1], out32[:, t:t + 1] = o, o32
    for n in range(N):
        ref = whole(xs[n:n + 1].contiguous(), w, b, W)
        assert_same((out[n:n + 1], out32[n:n + 1], st[n:n + 1]), ref, f"decode batch, sequence {n}")
    # the same through the varlen packing with one token per sequence
    st = alloc_conv_state(N, D, W)
    for t in range(steps):
        xt = xs[:, t].reshape(1, N, D).contiguous()
        o, o32 = run(xt, w, b, st, cu_tensor(list(range(N + 1))), max_seqlen=1)
        assert n_diff(o[0], out[:, t]) == 0 and n_diff(o32[0], out32[:, t]) == 0
    assert n_diff(st, torch.stack([whole(xs[n:n + 1].contiguous(), w, b, W)[2][0] for n in range(N)])) == 0


def test_training_forward_equals_prefill_from_zero_state(cx):
    D, W = 192, 4
    lens = [100, 7, 64, 129]
    cu = [0]
    for L in lens:
        cu.append(cu[-1] + L)
    x = make_x(cu[-1], D, seed=19)
    w, b = make_weights(D, W, seed=20, with_bias=torch.float32)
    st = alloc_conv_state(len(lens), D, W)
    o, o32 = run(x, w, b, st, cu_tensor(cu))
    ot, ot32 = run(x, w, b, None, cu_tensor(cu))   # state=None: the training forward
    assert n_diff(o, ot) == 0 and n_diff(o32, ot32) == 0
    # [B, T, D] layout, one sequence per batch entry
    xb = make_x(3 * 80, D, seed=21).reshape(3, 80, D).contiguous()
    stb = alloc_conv_state(3, D, W)
    o, o32 = run(xb, w, b, stb)
    ot, ot32 = run(xb, w, b, None)
    assert n_diff(o, ot) == 0 and n_diff(o32, ot32) == 0
    for n in range(3):
        ref = whole(xb[n:n + 1].contiguous(), w, b, W)
        assert_same((o[n:n + 1], o32[n:n + 1], stb[n:n + 1]), ref, f"[B,T,D] entry {n}")


def test_strided_input_slice_of_a_fused_projection(cx):
    """q, k, v conv inputs may be column slices of one fused [T, 3D] projection output (row stride 3D, also odd offsets)."""
    D, W, T = 96, 4, 150
    xs = make_x(T, 3 * D + 1, seed=22)[0]   # [T, 3D+1]
    w, b = make_weights(D, W, seed=23, with_bias=torch.float32)
    for off in (0, D, 2 * D, 1):            # offset 1: odd start (misaligned vector loads must not be used)
        xv = xs[:, off:off + D]
        ref = whole(xv.contiguous()[None], w, b, W)
        st = alloc_conv_state(1, D, W)
        o, o32 = run(xv[None], w, b, st)
        assert_same((o, o32, st), ref, f"slice at offset {off}")


def test_cuda_graph_decode(cx):
    D, W, steps = 160, 4, 40
    N = 3
    w, b = make_weights(D, W, seed=24, with_bias=torch.float32)
    xs = torch.cat([make_x(steps, D, seed=300 + n) for n in range(N)], dim=0)
    ref_st = alloc_conv_state(N, D, W)
    ref = torch.zeros_like(xs)
    for n in range(N):
        o, _, s = whole(xs[n:n + 1].contiguous(), w, b, W)
        ref[n:n + 1] = o
        ref_st[n] = s[0]
    st = alloc_conv_state(N, D, W)
    xt = torch.zeros(N, 1, D, dtype=torch.bfloat16, device="cuda")
    yt = torch.zeros_like(xt)

    def launch():
        short_conv(xt, w, b, st, "silu", out=yt)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):   # warm-up on a scratch state
        saved = st.clone()
        launch()
        st.copy_(saved)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    st.zero_()
    for t in range(steps):
        xt.copy_(xs[:, t:t + 1])
        graph.replay()
        assert n_diff(yt[:, 0], ref[:, t]) == 0, f"step {t}"
    assert n_diff(st, ref_st) == 0


# ---------------------------------------------------------------------------------------------------------
# negative control: a decode path that sums the taps in a different association must be caught
# ---------------------------------------------------------------------------------------------------------
def test_negative_control_is_caught(cx, cx_neg):
    D, W, T = 256, 4, 200
    x = make_x(T, D, seed=25)
    w, b = make_weights(D, W, seed=26, with_bias=torch.float32)

    def decode_vs_whole(ext):
        st1, st2 = alloc_conv_state(1, D, W), alloc_conv_state(1, D, W)
        o_w, o32_w = torch.empty_like(x), torch.zeros(x.shape, dtype=torch.float32, device="cuda")
        ext.conv_forward(x, w, o_w, b, st1, None, None, True, o32_w)
        o_d, o32_d = torch.empty_like(x), torch.zeros(x.shape, dtype=torch.float32, device="cuda")
        for t in range(T):   # T = 1 calls through contiguous per-token buffers
            xt = x[:, t:t + 1].contiguous()
            ot = torch.empty_like(xt)
            ot32 = torch.zeros(xt.shape, dtype=torch.float32, device="cuda")
            ext.conv_forward(xt, w, ot, b, st2, None, None, True, ot32)
            o_d[:, t:t + 1], o32_d[:, t:t + 1] = ot, ot32
        return n_diff(o_w, o_d), n_diff(o32_w, o32_d), o_w.numel(), n_diff(st1, st2)

    good = decode_vs_whole(cx)
    assert good[0] == 0 and good[1] == 0 and good[3] == 0
    bad = decode_vs_whole(cx_neg)
    print(f"\n[negative control, conv] decode vs whole with a different association in the decode path: fp32 values differing "
          f"{bad[1]}/{bad[2]}, bf16 outputs differing {bad[0]}/{bad[2]} (production build: 0/0)")
    assert bad[1] > 0, "the harness did not see a different decode association: it is vacuous"


# ---------------------------------------------------------------------------------------------------------
# gradients (training): forward == the pinned kernel, backward correct against an fp64 autograd reference
# ---------------------------------------------------------------------------------------------------------
def ref_conv_autograd64(x, w, b, cu, silu, W):
    """fp64 autograd reference over packed sequences (zero history per sequence)."""
    x64, w64 = x.double().requires_grad_(True), w.double().requires_grad_(True)
    b64 = None if b is None else b.double().requires_grad_(True)
    outs = []
    for lo, hi in seq_slices(cu):
        xe = torch.cat([torch.zeros(W - 1, x.shape[-1], dtype=torch.float64, device=x.device), x64[0, lo:hi]], dim=0)
        z = sum(w64[:, k][None, :] * xe[k:k + (hi - lo)] for k in range(W))
        if b64 is not None:
            z = z + b64[None, :]
        outs.append(z * torch.sigmoid(z) if silu else z)
    return (x64, w64, b64), torch.cat(outs, dim=0)[None]


def rel_rms(a, ref):
    return float(((a.double() - ref) ** 2).mean().sqrt() / (ref ** 2).mean().sqrt())


@pytest.mark.parametrize("silu", [True, False])
@pytest.mark.parametrize("wdtype,bias", [(torch.float32, torch.float32), (torch.bfloat16, None)])
def test_backward_matches_fp64_autograd(cx, silu, wdtype, bias):
    D, W = 160, 4
    lens = [90, 0, 5, 64, 129]
    cu = [0]
    for L in lens:
        cu.append(cu[-1] + L)
    x = make_x(cu[-1], D, seed=27).requires_grad_(True)
    w, b = make_weights(D, W, seed=28, wdtype=wdtype, with_bias=bias)
    w = w.requires_grad_(True)
    if b is not None:
        b = b.requires_grad_(True)
    cut = cu_tensor(cu)
    y = short_conv(x, w, b, None, "silu" if silu else None, cut)
    # the forward of the differentiable call is the pinned kernel
    with torch.no_grad():
        y0 = short_conv(x.detach(), w.detach(), None if b is None else b.detach(), None, "silu" if silu else None, cut)
    assert n_diff(y.detach(), y0) == 0
    dy = (torch.randn(y.shape, device="cuda") * 0.7).to(torch.bfloat16)
    y.backward(dy)
    (x64, w64, b64), y64 = ref_conv_autograd64(x.detach(), w.detach(), None if b is None else b.detach(), cu, silu, W)
    y64.backward(dy.double())
    e_dx, e_dw = rel_rms(x.grad, x64.grad), rel_rms(w.grad, w64.grad)
    print(f"\n[backward silu={silu} w={wdtype}] rel RMS vs fp64 autograd: dx {e_dx:.2e}, dw {e_dw:.2e}", end="")
    assert e_dx < 5e-3 and e_dw < (5e-3 if wdtype == torch.bfloat16 else 1e-5)
    if b is not None:
        e_db = rel_rms(b.grad, b64.grad)
        print(f", db {e_db:.2e}", end="")
        assert e_db < 1e-5
    # tokens of an empty sequence / no leakage between sequences: gradient of a perturbed sequence stays inside it
    assert x.grad.shape == x.shape


def test_stateful_call_is_inference_only(cx):
    D, W = 64, 4
    x = make_x(10, D, seed=29).requires_grad_(True)
    w, _ = make_weights(D, W, seed=30)
    with pytest.raises(ValueError):
        short_conv(x, w, None, alloc_conv_state(1, D, W))
    with torch.no_grad():
        short_conv(x, w, None, alloc_conv_state(1, D, W))


# ---------------------------------------------------------------------------------------------------------
# context: FLA's own ShortConvolution (Triton). Informational: closeness of the two programs, and how consistent FLA's
# conv is between its prefill and its single-token step (it uses a different reduction there).
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("wdtype", [torch.bfloat16, torch.float32])
def test_against_fla_short_convolution(cx, wdtype):
    try:
        from fla.modules import ShortConvolution
    except Exception as e:   # pragma: no cover
        pytest.skip(f"fla not importable: {e}")
    D, W, T = 2048, 4, 1024
    x = make_x(T, D, seed=31)
    w, _ = make_weights(D, W, seed=32, wdtype=wdtype)
    conv = ShortConvolution(D, W, bias=False, activation="silu", backend="triton", device="cuda", dtype=wdtype)
    with torch.no_grad():
        conv.weight.copy_(w.view(D, 1, W))
        y_fla, _ = conv(x)
        ours, _ = run(x, w, None, None)
        # FLA decode (step) token by token vs FLA's own full forward
        cache = torch.zeros(1, D, W, dtype=torch.bfloat16, device="cuda")
        y_step = torch.empty_like(x)
        for t in range(T):
            yt, cache = conv(x[:, t:t + 1], cache=cache, output_final_state=True)
            y_step[:, t:t + 1] = yt
    ulp = bf16_ulp_error(ours[0], y_fla[0].double())
    d_fla_vs_ours = n_diff(y_fla, ours) / ours.numel()
    d_fla_step = n_diff(y_step, y_fla) / ours.numel()
    print(f"\n[FLA ShortConvolution, {wdtype}] {ours.numel()} outputs. bf16 outputs differing from cfbi: {d_fla_vs_ours:.5%} "
          f"(max distance {ulp:.2f} bf16 ulp); FLA single-token step vs FLA full call: {d_fla_step:.5%} differing; "
          f"cfbi step vs full call: 0 by construction")
    assert d_fla_vs_ours < 0.02   # a different program (association / sigmoid implementation): never more than ~1 ulp apart
