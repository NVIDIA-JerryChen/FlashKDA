"""Randomised (seeded, deterministic) sweeps over shapes, widths, strides, cuts and batch compositions for the conv and the
linear: the same bit-exact equalities as the targeted tests, on configurations nobody chose by hand.

conv:   a packed varlen batch with random per-sequence lengths (incl. 0, 1, around the 64-token tile), random D (1 .. 700,
        odd and even), W (2 .. 8), bias / weight dtypes, a random row stride of x, and a random non-trivial initial state, served
        (a) in one call, (b) every sequence alone cut at random points, must agree on out / fp32 / state; also against the
        PyTorch emulation of the contract.
linear: random M, N, K (K % 16 == 0 up to 1056) against the program evaluated through the probe, and a random subset / order
        of rows against the full result.
"""
import random

import pytest
import torch

from flash_kda.cfbi import alloc_conv_state, bi_linear, short_conv

from convlin_helpers import cu_tensor, cx, emu_conv, make_weights, make_x, n_diff  # noqa: F401
from test_linear import lin32, program_via_probe_fp32, randbf


@pytest.mark.parametrize("seed", range(24))
def test_conv_random_batches(cx, seed):
    rng = random.Random(1000 + seed)
    D = rng.choice([1, 2, 3, 5, 31, 64, 97, 128, 129, 255, 256, 257, 300, 512, rng.randint(1, 700)])
    W = rng.randint(2, 8)
    N = rng.randint(1, 9)
    lens = [rng.choice([0, 1, 1, 2, 3, W - 1, W, 7, 63, 64, 65, 100, 128, 129, rng.randint(0, 220)]) for _ in range(N)]
    if sum(lens) == 0:
        lens[0] = 5
    cu = [0]
    for L in lens:
        cu.append(cu[-1] + L)
    wdt = rng.choice([torch.float32, torch.bfloat16])
    bdt = rng.choice([None, torch.float32, torch.bfloat16])
    silu = rng.random() < 0.8
    pad = rng.choice([0, 0, 1, 3, 8])   # row stride of x = D + pad (a slice of a wider tensor)
    wide = make_x(cu[-1], D + pad, seed=seed + 7)
    x = wide[..., :D] if pad else wide
    w, b = make_weights(D, W, seed=seed + 8, wdtype=wdt, with_bias=bdt)
    st0 = torch.randn(N, D, W - 1, device="cuda").to(torch.bfloat16)

    def run(xx, st, cut=None):
        o32 = torch.zeros(xx.shape, dtype=torch.float32, device="cuda")
        o = short_conv(xx, w, b, st, "silu" if silu else None, cut, out_fp32=o32)
        return o, o32

    st = st0.clone()
    out, out32 = run(x, st, cu_tensor(cu))
    for n in range(N):
        lo, hi = cu[n], cu[n + 1]
        stn = st0[n:n + 1].clone()
        pieces, rem = [], hi - lo
        while rem > 0:
            L = min(rem, rng.choice([1, 2, 3, 5, 63, 64, 65, rng.randint(1, 90)]))
            pieces.append(L)
            rem -= L
        o_n = torch.zeros(1, hi - lo, D, dtype=torch.bfloat16, device="cuda")
        o32_n = torch.zeros(1, hi - lo, D, dtype=torch.float32, device="cuda")
        p = lo
        for L in pieces:
            piece = x[:, p:p + L] if pad == 0 else x[:, p:p + L].contiguous()
            o, o32 = run(piece, stn)
            o_n[:, p - lo:p - lo + L], o32_n[:, p - lo:p - lo + L] = o, o32
            p += L
        assert n_diff(out[:, lo:hi], o_n) == 0, f"seed {seed}: bf16 outputs of sequence {n} (len {hi - lo}) differ"
        assert n_diff(out32[:, lo:hi], o32_n) == 0, f"seed {seed}: fp32 values of sequence {n} differ"
        assert n_diff(st[n:n + 1], stn) == 0, f"seed {seed}: state of sequence {n} differs"
        if hi > lo:   # and the contract itself
            xe = torch.cat([st0[n].t(), x[0, lo:hi]], dim=0)
            emu = emu_conv(cx, xe, w, b, silu)
            assert n_diff(out32[0, lo:hi], emu) == 0, f"seed {seed}: sequence {n} differs from the emulation of the contract"


@pytest.mark.parametrize("seed", range(20))
def test_linear_random_shapes(cx, seed):
    rng = random.Random(2000 + seed)
    K = 16 * rng.randint(1, 66)
    N = rng.choice([1, 2, 7, 8, 9, 16, 33, 64, 130, rng.randint(1, 300)])
    M = rng.choice([1, 2, 5, 16, 17, 63, 64, 65, 127, 128, 129, rng.randint(1, 300)])
    bdt = rng.choice([None, torch.float32, torch.bfloat16])
    x, w = randbf((M, K), seed, rng.choice([0.5, 1.0, 4.0])), randbf((N, K), seed + 50, rng.choice([0.02, 0.1]))
    b = None if bdt is None else randbf((N,), seed + 99).to(bdt)
    out, o32 = lin32(x, w, b)
    assert n_diff(o32, program_via_probe_fp32(cx, x, w, b)) == 0, f"seed {seed} (M={M} N={N} K={K}): fp32 values != the program"
    # a random subset of the rows, in random order, in a different batch size
    rows = torch.tensor(rng.sample(range(M), rng.randint(1, M)), device="cuda")
    sub, sub32 = lin32(x[rows], w, b)
    assert n_diff(sub, out[rows]) == 0 and n_diff(sub32, o32[rows]) == 0, f"seed {seed}: rows {rows.tolist()[:8]}.. differ"
    for cfg in range(5):
        assert n_diff(bi_linear(x, w, b, _cfg=cfg), out) == 0, f"seed {seed}: config {cfg}"
