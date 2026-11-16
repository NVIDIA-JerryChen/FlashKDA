"""Shared helpers and fixtures of the conv / linear / layer-chain tests (extension ``flash_kda_cfbi_conv_C``).

Fixtures (import them into a test module to use them): ``cx`` = the conv/linear extension, ``cx_neg`` = a build with the
two negative-control kernels (CFBI_CONV_NEG_CONTROL, CFBI_LIN_NEG_CONTROL) that the harness must catch.
"""
import math

import pytest
import torch

LOG2E_F = 1.44269502162933349609375


@pytest.fixture(scope="session")
def cx():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from flash_kda.cfbi import load_conv_ext

    return load_conv_ext()


@pytest.fixture(scope="session")
def cx_neg():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from flash_kda.cfbi._ext import load_conv_variant

    return load_conv_variant("negctl", ("CFBI_CONV_NEG_CONTROL=1", "CFBI_LIN_NEG_CONTROL=1"))


def bits(x):
    """Raw bits of a tensor as integers (bf16 -> int16, fp32 -> int32)."""
    x = x.contiguous()
    if x.dtype == torch.bfloat16:
        return x.view(torch.int16)
    if x.dtype == torch.float32:
        return x.view(torch.int32)
    raise TypeError(x.dtype)


def n_diff(a, b):
    assert a.shape == b.shape, (a.shape, b.shape)
    return int((bits(a) != bits(b)).sum())


def fma32(c, a, b):
    """c + a*b with one rounding (exact product and sum in fp64, then one rounding to fp32)."""
    return (c.double() + a.double() * b.double()).float()


def cu_tensor(cu, device="cuda"):
    return torch.tensor(cu, dtype=torch.int64, device=device)


def make_x(T_total, D, seed, device="cuda", scale=1.0):
    """bf16 conv input [1, T_total, D] with a few large / tiny values (random exponents) mixed in."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn(1, T_total, D, device=device, generator=g) * scale
    mix = torch.rand(1, T_total, D, device=device, generator=g)
    x = torch.where(mix < 0.05, x * 16.0, x)
    x = torch.where(mix > 0.97, x * 0.0625, x)
    return x.to(torch.bfloat16).contiguous()


def make_weights(D, W, seed, wdtype=torch.float32, with_bias=None, device="cuda"):
    g = torch.Generator(device=device)
    g.manual_seed(seed + 1000)
    w = (torch.randn(D, W, device=device, generator=g) * 0.6).to(wdtype)
    b = None
    if with_bias is not None:
        b = (torch.randn(D, device=device, generator=g) * 0.3).to(with_bias)
    return w, b


# ---- an emulation of the conv program in plain PyTorch (ex2 / rcp taken from the extension's own probe) ----
def emu_conv(ext, x_ext, w, bias, silu=True):
    """x_ext: bf16 [L + W - 1, D] = (W-1 history inputs ++ L inputs). Returns the fp32 value before the bf16 rounding
    [L, D]: acc = +0; acc = fma(w_k, x_k, acc) ascending; (+ bias); SiLU as z * rcp(1 + ex2(-z * log2e))."""
    W = w.shape[1]
    L = x_ext.shape[0] - (W - 1)
    xf = x_ext.float()
    wf = w.float()
    acc = torch.zeros(L, x_ext.shape[1], device=x_ext.device)
    for k in range(W):
        acc = fma32(acc, wf[:, k][None, :], xf[k:k + L])
    if bias is not None:
        acc = acc + bias.float()[None, :]
    if not silu:
        return acc
    log2e = torch.tensor(LOG2E_F, dtype=torch.float32, device=x_ext.device)
    e = ext.probe_unary(((-acc) * log2e).contiguous(), 0)
    r = ext.probe_unary((1.0 + e).contiguous(), 1)
    return acc * r


def conv_ref64(x, w, bias, state=None, silu=True):
    """fp64 reference of one sequence: x bf16 [L, D] -> fp64 [L, D] (history from state [D, W-1] or zeros)."""
    L, D = x.shape
    W = w.shape[1]
    hist = torch.zeros(W - 1, D, dtype=torch.float64, device=x.device) if state is None else state.t().double()
    xe = torch.cat([hist, x.double()], dim=0)
    z = torch.zeros(L, D, dtype=torch.float64, device=x.device)
    for k in range(W):
        z = z + w[:, k].double()[None, :] * xe[k:k + L]
    if bias is not None:
        z = z + bias.double()[None, :]
    return z * torch.sigmoid(z) if silu else z


def bf16_ulp_error(out, ref64):
    """max over elements of |out - ref| / (2^-8 |ref|) in units of one bf16 ulp-ish (0.5 = exact rounding)."""
    ref = ref64
    err = (out.double() - ref).abs()
    # spacing of bf16 values around ref: 2^(floor(log2|ref|) - 7)
    ulp = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
    return float((err / ulp).max())
