"""Gates of the batch-invariant bf16 linear layer: flash_kda.cfbi.bi_linear.

Zero tolerance on bits for everything claimed bitwise (row m independent of M / of the other rows / of the tile
configuration; the kernel equals the documented program evaluated through the probe), accuracy against an fp64 reference with
a stated tolerance. Negative controls: a build whose small-M configuration walks K backwards must be caught by the
invariance test, and cuBLAS must show the effect we are guarding against (it is not batch invariant).
"""
import pytest
import torch
import torch.nn.functional as F

from flash_kda.cfbi import bi_linear
from flash_kda.cfbi.linear import pick_config

from convlin_helpers import bits, cx, cx_neg, n_diff  # noqa: F401


def randbf(shape, seed, scale=1.0):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return (torch.randn(*shape, device="cuda", generator=g) * scale).to(torch.bfloat16)


def lin32(x, w, b=None, cfg=-1):
    """(bf16 result, fp32 value before the rounding) of bi_linear."""
    o32 = torch.zeros(x.shape[0], w.shape[0], dtype=torch.float32, device="cuda")
    return bi_linear(x, w, b, out_fp32=o32, _cfg=cfg), o32


def same(a, b):
    """Both the bf16 results and the fp32 values equal, bit for bit."""
    return n_diff(a[0], b[0]) == 0 and n_diff(a[1], b[1]) == 0


def program_via_probe_fp32(cx, x, w, bias):
    """The documented program evaluated tile by tile with the probe (one warp, scalar fragment loads, no smem /
    ldmatrix / pipeline): acc = chain of m16n8k16 MMAs over K ascending; (+ bias). fp32, before the bf16 rounding."""
    M, K = x.shape
    N = w.shape[0]
    MT, NT = (M + 15) // 16, (N + 7) // 8
    xp = torch.zeros(MT * 16, K, dtype=torch.bfloat16, device=x.device)
    xp[:M] = x
    wp = torch.zeros(NT * 8, K, dtype=torch.bfloat16, device=x.device)
    wp[:N] = w
    A = xp.view(MT, 16, K)[:, None].expand(MT, NT, 16, K).reshape(-1, 16, K).contiguous()
    B = wp.view(NT, 8, K).transpose(1, 2)[None].expand(MT, NT, K, 8).reshape(-1, K, 8).contiguous()
    D = cx.probe_mma_bf16(A, B).view(MT, NT, 16, 8).permute(0, 2, 1, 3).reshape(MT * 16, NT * 8)[:M, :N]
    if bias is not None:
        D = D + bias.float()[None, :]
    return D.contiguous()


def program_via_probe(cx, x, w, bias):
    """Same, rounded to bf16 (round-to-nearest-even)."""
    return program_via_probe_fp32(cx, x, w, bias).to(torch.bfloat16)


# ---------------------------------------------------------------------------------------------------------
# the program
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("M,N,K", [(37, 21, 256), (100, 72, 1040), (1, 8, 16), (16, 1, 48), (130, 33, 192), (260, 200, 128)])
@pytest.mark.parametrize("bias_dt", [None, torch.float32, torch.bfloat16])
def test_kernel_equals_the_documented_program(cx, M, N, K, bias_dt):
    x, w = randbf((M, K), 1), randbf((N, K), 2, 0.1)
    b = None if bias_dt is None else randbf((N,), 3).to(bias_dt)
    ref = program_via_probe(cx, x, w, b)
    assert n_diff(bi_linear(x, w, b), ref) == 0, "auto config"
    for cfg in range(5):
        assert n_diff(bi_linear(x, w, b, _cfg=cfg), ref) == 0, f"config {cfg}"
    # the fp32 values before the rounding as well: the probe's accumulators (+ bias) are the kernel's accumulators
    acc = program_via_probe_fp32(cx, x, w, b)
    assert n_diff(lin32(x, w, b)[1], acc) == 0


@pytest.mark.parametrize("bias_dt", [None, torch.float32])
def test_accuracy_vs_fp64(cx, bias_dt):
    """Tolerance (stated): |out - ref| <= 0.5 bf16-ulp(ref) + 2e-6 * (sum|x w| + |bias| + |ref|) (an fp32-accurate dot
    product: the tensor-core accumulation truncates at ~2^-24 relative to the running sum)."""
    M, N, K = 200, 160, 2048
    x, w = randbf((M, K), 4), randbf((N, K), 5, 0.05)
    b = None if bias_dt is None else randbf((N,), 6).to(bias_dt)
    out = bi_linear(x, w, b)
    ref = x.double() @ w.double().t()
    S = x.double().abs() @ w.double().abs().t()
    if b is not None:
        ref = ref + b.double()[None, :]
        S = S + b.double().abs()[None, :]
    slack = 2e-6 * (S + ref.abs())
    ulp = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
    err = (out.double() - ref).abs()
    print(f"\n[accuracy bias={bias_dt}] max |out - ref| / (0.5 ulp + slack) = {float((err / (0.5 * ulp + slack)).max()):.3f} (<= 1 passes)")
    assert bool((err <= 0.5 * ulp + slack).all())


# ---------------------------------------------------------------------------------------------------------
# batch invariance: row m of the result does not depend on M, on the other rows, or on the tile configuration
# ---------------------------------------------------------------------------------------------------------
MS = [1, 2, 3, 5, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 255, 256, 257, 511, 1000, 2048]


@pytest.mark.parametrize("K,N", [(4096, 4096), (2048, 1024), (96, 33), (1040, 130), (128, 16)])
def test_row_is_independent_of_M(cx, K, N):
    Mmax = 2048
    X, W = randbf((Mmax, K), 7, 0.5), randbf((N, K), 8, 0.05)
    b = randbf((N,), 9)
    full = lin32(X, W, b)
    for M in MS:
        assert same(lin32(X[:M], W, b), (full[0][:M], full[1][:M])), f"first {M} rows"
    for M, off in [(1, 777), (1, 1), (7, 100), (16, 31), (64, 500), (130, 900), (300, 1700)]:   # views at row offsets
        assert same(lin32(X[off:off + M], W, b), (full[0][off:off + M], full[1][off:off + M])), f"{M} rows at offset {off}"
    perm = torch.randperm(Mmax, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
    assert same(lin32(X[perm], W, b), (full[0][perm], full[1][perm])), "permuted rows"
    # every single row alone
    for r in (0, 1, 15, 16, 127, 128, 1023, 2047):
        assert same(lin32(X[r:r + 1], W, b), (full[0][r:r + 1], full[1][r:r + 1])), f"row {r} alone"


def test_every_M_up_to_300(cx):
    K, N = 1040, 130
    X, W = randbf((300, K), 10, 0.5), randbf((N, K), 11, 0.05)
    full = lin32(X, W)
    for M in range(1, 301):
        assert same(lin32(X[:M], W), (full[0][:M], full[1][:M])), f"M={M}"


@pytest.mark.parametrize("K,N", [(4096, 4096), (1040, 130)])
def test_all_tile_configurations_agree(cx, K, N):
    """The five tile configurations (block / warp tiles, pipeline depths) give the same bits for the same rows."""
    M = 300
    X, W = randbf((M, K), 12, 0.5), randbf((N, K), 13, 0.05)
    b = randbf((N,), 14)
    outs = [lin32(X, W, b, c) for c in range(5)]
    for c in range(1, 5):
        assert same(outs[0], outs[c]), f"config {c} != config 0"
    cfgs = {M_: pick_config(M_, N) for M_ in (1, 16, 17, 64, 65, 128, 129, 1000, 8192)}
    print(f"\n[config choice K={K} N={N}] {cfgs}")


def test_neighbour_rows_do_not_matter(cx):
    """The same row next to zeros / random / huge / NaN / Inf rows: same bits (a NaN row only poisons its own outputs)."""
    K, N, M = 1024, 96, 64
    X, W = randbf((M, K), 15), randbf((N, K), 16, 0.05)
    ref, ref32 = lin32(X, W)
    r = 23
    for fill in ("zero", "random", "huge", "nan", "inf"):
        Y = X.clone()
        mask = torch.ones(M, dtype=torch.bool, device="cuda")
        mask[r] = False
        if fill == "zero":
            Y[mask] = 0
        elif fill == "random":
            Y[mask] = randbf((M - 1, K), 17)
        elif fill == "huge":
            Y[mask] = (randbf((M - 1, K), 18).float() * 2.0 ** 60).to(torch.bfloat16)
        elif fill == "nan":
            Y[mask] = float("nan")
        else:
            Y[mask] = float("inf")
        y16, y32 = lin32(Y, W)
        assert n_diff(y16[r:r + 1], ref[r:r + 1]) == 0 and n_diff(y32[r:r + 1], ref32[r:r + 1]) == 0, f"neighbours = {fill}"


def test_columns_do_not_depend_on_N(cx):
    """Column n of the result does not depend on which other output columns are computed (N = 1 .. full)."""
    K, M = 512, 70
    X, W = randbf((M, K), 19), randbf((200, K), 20, 0.05)
    full = bi_linear(X, W)
    for N in (1, 2, 7, 8, 9, 31, 32, 33, 64, 127, 128, 129, 200):
        assert n_diff(bi_linear(X, W[:N].contiguous()), full[:, :N]) == 0, f"N={N}"
    W2 = W[100:140].contiguous()
    assert n_diff(bi_linear(X, W2), full[:, 100:140]) == 0, "a slice of the output columns"


def test_leading_dims_strided_and_misaligned_inputs(cx):
    K, N = 256, 40
    X, W = randbf((3, 50, K), 21), randbf((N, K), 22, 0.1)
    flat = bi_linear(X.reshape(-1, K), W)
    out3 = bi_linear(X, W)
    assert out3.shape == (3, 50, N) and n_diff(out3.reshape(-1, N), flat) == 0
    wide = randbf((150, K + 16), 23)
    Xs = wide[:, 4:4 + K]   # row stride K + 16, offset 4 elements (8 bytes): misaligned base -> copied
    assert n_diff(bi_linear(Xs, W), bi_linear(Xs.contiguous(), W)) == 0
    Xs2 = wide[:, :K]       # aligned, row stride K + 16: used in place
    assert n_diff(bi_linear(Xs2, W), bi_linear(Xs2.contiguous(), W)) == 0
    out = torch.empty(150, N, dtype=torch.bfloat16, device="cuda")
    bi_linear(Xs2, W, out=out)
    assert n_diff(out, bi_linear(Xs2.contiguous(), W)) == 0
    with pytest.raises(ValueError):
        bi_linear(randbf((4, 24), 24), randbf((8, 24), 25))   # K % 16 != 0 is rejected by the extension
    with pytest.raises(TypeError):
        bi_linear(X.float(), W)


# ---------------------------------------------------------------------------------------------------------
# controls: the invariance test must catch an M-dependent K order; cuBLAS shows the effect we guard against
# ---------------------------------------------------------------------------------------------------------
def test_negative_control_is_caught(cx, cx_neg):
    """Build where the small-M (M <= 64) configuration walks K backwards (an M-dependent program): the same check as above fails."""
    K, N, M = 1024, 256, 512
    X, W = randbf((M, K), 26, 0.5), randbf((N, K), 27, 0.05)

    def lin(ext, x, cfg=-1):
        out = torch.empty(x.shape[0], N, dtype=torch.bfloat16, device="cuda")
        o32 = torch.zeros(x.shape[0], N, dtype=torch.float32, device="cuda")
        ext.linear_forward(x.contiguous(), W, out, None, cfg, o32)
        return out, o32

    ms = (1, 16, 64, 100, 128)
    full, full_neg = lin(cx, X), lin(cx_neg, X)
    good = [(n_diff(lin(cx, X[:m])[0], full[0][:m]), n_diff(lin(cx, X[:m])[1], full[1][:m])) for m in ms]
    assert good == [(0, 0)] * 5
    bad = [(n_diff(lin(cx_neg, X[:m])[0], full_neg[0][:m]), n_diff(lin(cx_neg, X[:m])[1], full_neg[1][:m])) for m in ms]
    print(f"\n[negative control, linear] (bf16, fp32) outputs differing, M rows alone vs inside M=512, M = {ms}: {bad} "
          f"of {[m * N for m in ms]} (production build: all zero)")
    assert bad[0][1] > 0 and bad[1][1] > 0 and bad[2][1] > 0, "the harness did not see an M-dependent K order in the fp32 values: it is vacuous"
    assert all(b[1] == 0 for b in bad[3:])   # M = 100, 128 run configuration 1: untouched
    assert n_diff(lin(cx_neg, X, 4)[1], lin(cx, X, 4)[1]) == 0   # the large-M configuration is untouched


@pytest.mark.parametrize("op", ["matmul", "linear"])
def test_cublas_is_not_batch_invariant(cx, op):
    """Control: row r of x @ W^T computed with M = 1, M = 64 and M = 8192 through cuBLAS. Prints the fraction of differing
    bf16 outputs (rows 0..31 of each); bi_linear is exactly invariant on the same data."""
    rows = []
    any_diff = False
    for K, N in [(2048, 2048), (3072, 3072), (4096, 4096), (2048, 4096), (4096, 2048)]:
        X, W = randbf((8192, K), 30 + K, 0.5), randbf((N, K), 31 + N, 0.05)

        def f(x, W=W):
            return torch.matmul(x, W.t()) if op == "matmul" else F.linear(x, W)

        R = 32
        m1 = torch.cat([f(X[r:r + 1]) for r in range(R)])
        m64 = f(X[:64])[:R]
        m8192 = f(X)[:R]
        d = (n_diff(m1, m64), n_diff(m1, m8192), n_diff(m64, m8192))
        tot = R * N
        bi_full = bi_linear(X, W)[:R]
        bi = [n_diff(torch.cat([bi_linear(X[r:r + 1], W) for r in range(R)]), bi_full),
              n_diff(bi_linear(X[:64], W)[:R], bi_full)]
        rows.append((K, N, [x / tot for x in d], bi))
        any_diff |= any(d)
        assert bi == [0, 0]
    print(f"\n[cuBLAS control, torch.{op}] fraction of differing bf16 outputs over rows 0..31:")
    print(f"  {'K':>5} {'N':>5}  {'M=1 vs 64':>10} {'M=1 vs 8192':>12} {'M=64 vs 8192':>13}   bi_linear differing")
    for K, N, fr, bi in rows:
        print(f"  {K:>5} {N:>5}  {fr[0]:>10.4%} {fr[1]:>12.4%} {fr[2]:>13.4%}   {bi}")
    assert any_diff, "cuBLAS happened to be batch invariant on every shape tried: the control shows nothing"


# ---------------------------------------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------------------------------------
def test_autograd_forward_is_the_kernel_and_backward_is_correct(cx):
    M, N, K = 96, 80, 192
    x = randbf((M, K), 40).requires_grad_(True)
    w = randbf((N, K), 41, 0.1).requires_grad_(True)
    b = randbf((N,), 42).to(torch.float32).requires_grad_(True)
    y = bi_linear(x, w, b)
    with torch.no_grad():
        y0 = bi_linear(x.detach(), w.detach(), b.detach())
    assert n_diff(y.detach(), y0) == 0
    dy = randbf((M, N), 43)
    y.backward(dy)
    x64, w64, b64 = (t.detach().double().requires_grad_(True) for t in (x, w, b))
    (x64 @ w64.t() + b64).backward(dy.double())

    def rel_rms(a, ref):
        return float(((a.double() - ref) ** 2).mean().sqrt() / (ref ** 2).mean().sqrt())

    e = (rel_rms(x.grad, x64.grad), rel_rms(w.grad, w64.grad), rel_rms(b.grad, b64.grad))
    print(f"\n[backward] rel RMS vs fp64 autograd: dx {e[0]:.2e} dw {e[1]:.2e} db {e[2]:.2e}")
    assert e[0] < 5e-3 and e[1] < 5e-3 and e[2] < 1e-5
    assert x.grad.dtype == torch.bfloat16 and b.grad.dtype == torch.float32
