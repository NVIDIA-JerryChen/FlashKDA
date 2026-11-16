"""M1 gate: the hardware premise behind CF-BI.

CF-BI lets decode put ONE valid row in a 16x8 MMA tile while prefill fills all 16 rows. That is only
bitwise-safe if an output element of the three-term TF32 product depends on nothing but its own
(row vector, column vector) pair and the fixed K order. These tests check exactly that on the device
they run on, with a negative control (reversed K order must give different bits) so they are not vacuous.

If a test here fails: STOP. The row-in-a-tile design is not valid on this hardware (the fallback is a
CUDA-core fp32-FMA design, which is a different and slower program).
"""
import pytest
import torch

KS = [8, 16, 64, 128]
REGIMES = ["normal", "wide_exponents", "cancelling", "tiny_times_huge", "sparse"]
N = 64


def bits(x):
    return x.contiguous().view(torch.int32)


def make_pair(regime, n, K, gen):
    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=gen)

    def pow2(*shape, lo, hi):
        e = torch.randint(lo, hi + 1, shape, device="cuda", generator=gen).float()
        return torch.exp2(e)

    a, b = randn(n, K), randn(n, K)
    if regime == "normal":
        pass
    elif regime == "wide_exponents":
        a, b = a * pow2(n, K, lo=-30, hi=30), b * pow2(n, K, lo=-30, hi=30)
    elif regime == "cancelling":
        # large terms of both signs that nearly cancel, plus small terms that decide the result
        a = a.abs() * torch.where(torch.arange(K, device="cuda") % 2 == 0, 1.0, -1.0) * 1e3
        b = b.abs() + 1e-3 * randn(n, K)
    elif regime == "tiny_times_huge":
        # the exponent extremes of ks (<=1) x kjr (<=2^108) in a sub-block
        a, b = a * 2.0 ** -100, b * 2.0 ** 100
    elif regime == "sparse":
        a = a * (torch.rand(n, K, device="cuda", generator=gen) < 0.1)
        b = b * (torch.rand(n, K, device="cuda", generator=gen) < 0.5)
    else:
        raise ValueError(regime)
    return a.contiguous(), b.contiguous()


def filler(fill, shape, gen):
    if fill == "zero":
        return torch.zeros(*shape, device="cuda")
    x = torch.randn(*shape, device="cuda", generator=gen)
    if fill == "random":
        return x
    if fill == "wide":
        return x * torch.exp2(torch.randint(-60, 61, shape, device="cuda", generator=gen).float())
    if fill == "nan":
        return torch.full(shape, float("nan"), device="cuda")
    if fill == "inf":
        return torch.full(shape, float("inf"), device="cuda")
    raise ValueError(fill)


def run_one(ext, a, b, i, j, fill, gen, reverse_k=False):
    n, K = a.shape
    A = filler(fill, (n, 16, K), gen)
    B = filler(fill, (n, K, 8), gen)
    A[:, i, :] = a
    B[:, :, j] = b
    return ext.probe_dot3(A, B, reverse_k)[:, i, j]


@pytest.mark.parametrize("K", KS)
@pytest.mark.parametrize("regime", REGIMES)
def test_output_depends_only_on_its_own_row_and_column(ext, gen, K, regime):
    """Same (a, b) pair in every row slot x column slot x neighbour-data pattern => identical bits."""
    a, b = make_pair(regime, N, K, gen)
    ref = run_one(ext, a, b, 0, 0, "zero", gen)
    assert torch.isfinite(ref).all(), "reference tile must be finite for this regime"
    for i in range(16):
        for j in range(8):
            for fill in ("zero", "random", "wide"):
                d = run_one(ext, a, b, i, j, fill, gen)
                assert torch.equal(bits(d), bits(ref)), (
                    f"MMA output changed with placement/neighbours: K={K} regime={regime} "
                    f"row slot={i} col slot={j} neighbours={fill}"
                )


@pytest.mark.parametrize("fill", ["nan", "inf"])
def test_non_finite_neighbours_do_not_leak(ext, gen, fill):
    """Informational for CF-BI (invalid slots are zero-filled), but it shows the tile is element-wise."""
    a, b = make_pair("normal", N, 128, gen)
    ref = run_one(ext, a, b, 0, 0, "zero", gen)
    for i, j in [(0, 0), (5, 3), (15, 7)]:
        d = run_one(ext, a, b, i, j, fill, gen)
        assert torch.equal(bits(d), bits(ref)), f"non-finite neighbours ({fill}) changed row {i} col {j}"


def test_same_bits_in_every_block_and_every_run(ext, gen):
    a, b = make_pair("normal", 1, 128, gen)
    A = a.repeat(256, 1).reshape(256, 1, 128).expand(256, 16, 128).contiguous()
    B = b.repeat(256, 1).reshape(256, 128, 1).expand(256, 128, 8).contiguous()
    D1 = ext.probe_dot3(A, B)
    D2 = ext.probe_dot3(A, B)
    assert torch.equal(bits(D1), bits(D2))
    assert torch.equal(bits(D1), bits(D1[:1].expand_as(D1).contiguous()))


@pytest.mark.parametrize("regime", ["normal", "wide_exponents", "cancelling"])
def test_negative_control_k_order_matters(ext, gen, regime):
    """Reversed K-step order is a different program: it must change bits in a visible fraction of tiles."""
    a, b = make_pair(regime, 256, 128, gen)
    fwd = run_one(ext, a, b, 3, 2, "zero", gen)
    rev = run_one(ext, a, b, 3, 2, "zero", gen, reverse_k=True)
    frac = (bits(fwd) != bits(rev)).float().mean().item()
    assert frac > 0.05, f"harness is not sensitive to the K order (only {frac:.3%} of tiles differ)"


def test_dot3_accuracy_vs_fp64(ext, gen):
    """The three-term product is much closer to fp32 than a single TF32 product would be."""
    a, b = make_pair("normal", 256, 128, gen)
    d = run_one(ext, a, b, 0, 0, "zero", gen).double()
    exact = (a.double() * b.double()).sum(-1)
    scale = (a.double().abs() * b.double().abs()).sum(-1)
    rel = ((d - exact).abs() / scale).max().item()
    print(f"\n[dot3 accuracy] max |err| / sum|a*b| = {rel:.3e}  (single-term TF32 would be ~1e-3)")
    assert rel < 2e-6
