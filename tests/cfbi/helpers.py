"""Shared helpers of the CF-BI tests."""
import math

import torch
import torch.nn.functional as F

D = 128
REGIMES = ["authors", "unit", "heavy_tail", "long_memory", "no_decay", "eps_dominated", "saturated", "one_hot"]


def make_case(T_total, H, regime="authors", seed=0, device="cuda"):
    """Random inputs for ``T_total`` tokens as one ``[1, T_total, H, 128]`` stream (B = 1).

    The regimes stress different parts of the arithmetic: unit/authors-style data, heavy tails, almost no
    decay (long memory), exactly no decay (lower_bound = 0), eps-dominated norms, saturated gates, one-hot keys.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    def randn(*s):
        return torch.randn(*s, device=device, generator=gen)

    q, k, v, g = (randn(1, T_total, H, D) for _ in range(4))
    beta = randn(1, T_total, H)
    A_log = torch.rand(H, device=device, generator=gen)
    dt_bias = torch.rand(H, D, device=device, generator=gen)
    lower_bound = -5.0
    if regime == "authors":
        q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
    elif regime == "unit":
        pass
    elif regime == "heavy_tail":
        q, k, v = q * torch.exp(2 * randn(*q.shape)), k * torch.exp(2 * randn(*k.shape)), v * torch.exp(randn(*v.shape))
        g = g * 4
    elif regime == "long_memory":
        g = -12.0 + 2 * g
        A_log = torch.full_like(A_log, 0.5)
    elif regime == "no_decay":
        lower_bound = 0.0
    elif regime == "eps_dominated":
        q, k = q * 1e-5, k * 1e-5
    elif regime == "saturated":
        g, beta = 20.0 + g, 10.0 + beta
    elif regime == "one_hot":
        idx = torch.randint(0, D, (1, T_total, H), device=device, generator=gen)
        k = F.one_hot(idx, D).float()
        idx = torch.randint(0, D, (1, T_total, H), device=device, generator=gen)
        q = F.one_hot(idx, D).float()
        beta = beta + 3.0
    else:
        raise ValueError(regime)
    bf = torch.bfloat16
    return dict(q=q.to(bf).contiguous(), k=k.to(bf).contiguous(), v=v.to(bf).contiguous(), g=g.to(bf).contiguous(),
                beta=beta.to(bf).contiguous(), A_log=A_log, dt_bias=dt_bias, scale=1.0 / math.sqrt(D),
                lower_bound=lower_bound)


def split_tokens(case, lo, hi):
    """Slice of the token axis as a new case dict (views made contiguous)."""
    out = dict(case)
    for key in ("q", "k", "v", "g", "beta"):
        out[key] = case[key][:, lo:hi].contiguous()
    return out


def bits(x):
    return x.contiguous().view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32)


def n_diff(a, b):
    return int((bits(a) != bits(b)).sum())


def rel_rms(x, ref):
    x, ref = x.double(), ref.double()
    return float(((x - ref) ** 2).mean().sqrt() / (ref ** 2).mean().sqrt())


def require_impl(ext, impl):
    """Skip a scenario that asks for the device-scheduled K2 (impl bit 16) on a device that drops the bit: the comparison would run another K2."""
    if impl & 16 and not ext.resolve_impl(impl, -1) & 16:
        import pytest
        pytest.skip("the device-scheduled K2 does not run on this device (impl bit 16 is dropped)")
