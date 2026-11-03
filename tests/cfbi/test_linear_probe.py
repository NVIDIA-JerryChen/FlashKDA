"""Hardware premise behind the batch-invariant linear layer (the bf16 analogue of test_mma_probe.py).

``bi_linear`` puts ONE row of x (decode) or up to 128 rows (prefill) into the 16 row slots of ``mma.sync.m16n8k16``
(bf16 x bf16 + fp32). The result is batch invariant only if an output element of the instruction chain depends on nothing but
its own (row vector, column vector) pair, the order of the K steps and its accumulator - not on the slot it sits in
and not on what the other rows / columns of the tile hold. These tests check exactly that on the device they run on, with a
negative control (reversed K order must give different bits) so they are not vacuous.

If a test here fails: STOP. A tile-slot dependent tensor-core result makes a batch-invariant mma.sync GEMM impossible
(the fallback would be a CUDA-core fp32 FMA kernel, a different program).
"""
import pytest
import torch

from convlin_helpers import bits, cx  # noqa: F401

KS = [16, 32, 64, 128, 256, 1024]
REGIMES = ["normal", "wide_exponents", "cancelling", "tiny_times_huge", "sparse"]
N = 64


@pytest.fixture()
def gen():
    g = torch.Generator(device="cuda")
    g.manual_seed(20261103)
    return g


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
        a = a.abs() * torch.where(torch.arange(K, device="cuda") % 2 == 0, 1.0, -1.0) * 1e3
        b = b.abs() + 1e-3 * randn(n, K)
    elif regime == "tiny_times_huge":
        a, b = a * 2.0 ** -60, b * 2.0 ** 60
    elif regime == "sparse":
        a = a * (torch.rand(n, K, device="cuda", generator=gen) < 0.1)
        b = b * (torch.rand(n, K, device="cuda", generator=gen) < 0.5)
    else:
        raise ValueError(regime)
    return a.to(torch.bfloat16).contiguous(), b.to(torch.bfloat16).contiguous()


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
    """Tile with a in row slot i and b in column slot j, every other row / column filled with `fill`."""
    n, K = a.shape
    A = filler(fill, (n, 16, K), gen).to(torch.bfloat16)
    B = filler(fill, (n, K, 8), gen).to(torch.bfloat16)
    A[:, i, :] = a
    B[:, :, j] = b
    return ext.probe_mma_bf16(A.contiguous(), B.contiguous(), reverse_k)[:, i, j]


@pytest.mark.parametrize("K", KS)
@pytest.mark.parametrize("regime", REGIMES)
def test_output_depends_only_on_its_own_row_and_column(cx, gen, K, regime):
    """Same (a, b) pair in every row slot x column slot x neighbour-data pattern => identical bits."""
    a, b = make_pair(regime, N, K, gen)
    ref = run_one(cx, a, b, 0, 0, "zero", gen)
    assert torch.isfinite(ref).all(), "reference tile must be finite for this regime"
    for i in range(16):
        for j in range(8):
            for fill in ("zero", "random", "wide"):
                d = run_one(cx, a, b, i, j, fill, gen)
                assert torch.equal(bits(d), bits(ref)), (
                    f"MMA output changed with placement/neighbours: K={K} regime={regime} "
                    f"row slot={i} col slot={j} neighbours={fill}"
                )


@pytest.mark.parametrize("fill", ["nan", "inf"])
def test_non_finite_neighbours_do_not_leak(cx, gen, fill):
    """Informational for bi_linear (invalid slots are zero-filled), but it shows the tile is element-wise."""
    a, b = make_pair("normal", N, 128, gen)
    ref = run_one(cx, a, b, 0, 0, "zero", gen)
    for i, j in [(0, 0), (5, 3), (15, 7)]:
        d = run_one(cx, a, b, i, j, fill, gen)
        assert torch.equal(bits(d), bits(ref)), f"non-finite neighbours ({fill}) changed row {i} col {j}"


def test_same_bits_in_every_block_and_every_run(cx, gen):
    a, b = make_pair("normal", 1, 256, gen)
    A = a.repeat(256, 1).reshape(256, 1, 256).expand(256, 16, 256).contiguous()
    B = b.repeat(256, 1).reshape(256, 256, 1).expand(256, 256, 8).contiguous()
    D1 = cx.probe_mma_bf16(A, B)
    D2 = cx.probe_mma_bf16(A, B)
    assert torch.equal(bits(D1), bits(D2))
    assert torch.equal(bits(D1), bits(D1[:1].expand_as(D1).contiguous()))


@pytest.mark.parametrize("regime", ["normal", "wide_exponents", "cancelling"])
def test_negative_control_k_order_matters(cx, gen, regime):
    """Reversed K-step order is a different program: it must change bits in a visible fraction of tiles."""
    a, b = make_pair(regime, 256, 1024, gen)
    fwd = run_one(cx, a, b, 3, 2, "zero", gen)
    rev = run_one(cx, a, b, 3, 2, "zero", gen, reverse_k=True)
    frac = (bits(fwd) != bits(rev)).float().mean().item()
    print(f"\n[K-order control {regime}] reversed K order changes {frac:.1%} of the fp32 outputs")
    assert frac > 0.05, f"harness is not sensitive to the K order (only {frac:.3%} of tiles differ)"


def test_chain_accuracy_vs_fp64(cx, gen):
    """The k16 chain is an fp32-accurate dot product (tensor-core products are exact, the accumulation truncates)."""
    a, b = make_pair("normal", 256, 1024, gen)
    d = run_one(cx, a, b, 0, 0, "zero", gen).double()
    exact = (a.double() * b.double()).sum(-1)
    scale = (a.double().abs() * b.double().abs()).sum(-1)
    rel = ((d - exact).abs() / scale).max().item()
    print(f"\n[mma chain accuracy] max |err| / sum|a*b| = {rel:.3e} (K=1024)")
    assert rel < 1e-5
