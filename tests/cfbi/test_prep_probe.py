"""M1: per-token prep (l2norm / gate / beta) is bit-exact against a PyTorch emulation.

The emulation does every IEEE operation in PyTorch fp32 (FMA via an fp64 detour, as master's
tests/torch_ref.py does) and takes the three approximate instructions (ex2 / tanh / rsqrt) from the
extension's own elementwise probe, so what is being verified is the *structure* of the prep program
(operation order, FMA vs mul+add, reduction tree), not the hardware approximations themselves.
"""
import math

import pytest
import torch

LOG2E = 1.4426950408889634


def f32(v):
    return torch.tensor(v, dtype=torch.float32, device="cuda")


def fma32(c, a, b):
    """c + a*b with one rounding (exact product + sum in fp64, then one rounding to fp32)."""
    return (c.double() + a.double() * b.double()).float()


def emu_l2_inv(ext, x):
    xf = x.float().reshape(-1, 16, 8)
    p = torch.zeros(xf.shape[0], 16, device="cuda")
    for i in range(8):
        p = fma32(p, xf[..., i], xf[..., i])
    idx = torch.arange(16, device="cuda")
    for off in (8, 4, 2, 1):
        p = p + p[:, idx ^ off]
    s = (p[:, 0] + f32(1e-6)).contiguous()
    return ext.probe_transc(s, 2)


def emu_prep(ext, q, k, g, beta, A_log, dt_bias, scale, lower_bound):
    inv_k, inv_q = emu_l2_inv(ext, k), emu_l2_inv(ext, q)
    khat = k.float() * inv_k[:, None]
    qhat = (q.float() * inv_q[:, None]) * f32(scale)
    def sigmoid(x):   # rcp(1 + ex2(-x * log2 e))
        e = ext.probe_transc(((-x) * f32(LOG2E)).contiguous(), 0)
        return ext.probe_transc((f32(1.0) + e).contiguous(), 3)

    ale = ext.probe_transc((A_log * f32(LOG2E)).contiguous(), 0)
    a2 = f32(lower_bound * LOG2E) * sigmoid(ale * (g.float() + dt_bias))
    return khat, qhat, a2, sigmoid(beta.float())


def make_inputs(regime, rows, gen):
    def randn(*s):
        return torch.randn(*s, device="cuda", generator=gen)

    q, k = randn(rows, 128), randn(rows, 128)
    g, beta = randn(rows, 128), randn(rows)
    if regime == "unit":
        pass
    elif regime == "heavy_tail":
        q, k = q * torch.exp(2 * randn(rows, 128)), k * torch.exp(2 * randn(rows, 128))
        g = g * 4
    elif regime == "eps_dominated":
        q, k = q * 1e-4, k * 1e-5
    elif regime == "large":
        q, k = q * 1e3, k * 1e4
        beta = beta * 6
    return [x.to(torch.bfloat16).contiguous() for x in (q, k, g, beta)]


@pytest.mark.parametrize("regime", ["unit", "heavy_tail", "eps_dominated", "large"])
@pytest.mark.parametrize("lower_bound", [-5.0, -2.5, -0.5])
def test_prep_matches_emulation_bitwise(ext, gen, regime, lower_bound):
    rows = 512
    q, k, g, beta = make_inputs(regime, rows, gen)
    A_log = torch.rand(1, device="cuda", generator=gen)
    dt_bias = torch.rand(128, device="cuda", generator=gen)
    scale = 1.0 / math.sqrt(128)
    got = ext.probe_prep(q, k, g, beta, A_log, dt_bias, scale, lower_bound)
    want = emu_prep(ext, q, k, g, beta, A_log, dt_bias, scale, lower_bound)
    for name, a, b in zip(("khat", "qhat", "a2", "beta"), got, want):
        assert a.shape == b.shape, name
        mism = (a.view(torch.int32) != b.view(torch.int32))
        assert not mism.any(), f"{name}: {int(mism.sum())} of {mism.numel()} elements differ (regime={regime})"


def test_gate_range(ext, gen):
    """a2 is non-positive and >= lower_bound*log2(e): the exponent-range assumption of the reference rows."""
    rows = 1024
    q, k, g, beta = make_inputs("heavy_tail", rows, gen)
    A_log = torch.rand(1, device="cuda", generator=gen) * 3
    dt_bias = torch.rand(128, device="cuda", generator=gen)
    for lb in (-5.0, -1.0):
        _, _, a2, b = ext.probe_prep(q, k, g, beta, A_log, dt_bias, 0.1, lb)
        assert (a2 <= 0).all() and (a2 >= lb * LOG2E - 1e-6).all()
        assert (b >= 0).all() and (b <= 1).all()
