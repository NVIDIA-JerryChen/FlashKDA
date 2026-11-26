"""Shared helpers of the chunk-64 tests: synthetic data regimes, the fp64 exact recurrence (accuracy oracle), bit views.

``make_case`` / ``exact_recurrence`` follow the regimes and the oracle used for the accuracy comparison of the CF-BI
kernels (the KDA recurrence S_t = S_{t-1} diag(2^a2_t); S_t += beta_t (v_t - S_t khat_t) khat_t^T; o_t = S_t qhat_t).
"""
import math

import torch
import torch.nn.functional as F

D = 128
LOG2E = 1.4426950408889634
REGIMES = ["authors", "unit", "heavy_tail", "long_memory", "no_decay", "eps_dominated", "saturated", "one_hot"]


def make_case(T_total, H, regime="authors", seed=0, device="cuda"):
    """Random inputs for ``T_total`` tokens as one ``[1, T_total, H, 128]`` stream (B = 1)."""
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


def exact_recurrence(case):
    """fp64 reference output ``[1, T, H, 128]`` of the plain per-token recurrence (the accuracy oracle)."""
    q, k, v, g, beta = (case[n][0] for n in ("q", "k", "v", "g", "beta"))
    A_log, dt_bias, scale, lb = case["A_log"], case["dt_bias"], case["scale"], case["lower_bound"]
    T, H = q.shape[0], q.shape[1]
    qd, kd = q.double(), k.double()
    khat = kd / torch.sqrt((kd * kd).sum(-1, keepdim=True) + 1e-6)
    qhat = qd / torch.sqrt((qd * qd).sum(-1, keepdim=True) + 1e-6) * float(scale)
    ale = torch.exp2(A_log.double() * LOG2E)[None, :, None]
    sig = torch.sigmoid(ale * (g.double() + dt_bias.double()[None]))
    alpha = torch.exp2((float(lb) * LOG2E) * sig)
    bt = torch.sigmoid(beta.double())
    vd = v.double()
    S = torch.zeros(H, D, D, dtype=torch.float64, device=q.device)
    out = torch.zeros(T, H, D, dtype=torch.float64, device=q.device)
    for t in range(T):
        S = S * alpha[t][:, None, :]
        kS = torch.einsum("hk,hvk->hv", khat[t], S)
        S = S + bt[t][:, None, None] * (vd[t] - kS)[:, :, None] * khat[t][:, None, :]
        out[t] = torch.einsum("hk,hvk->hv", qhat[t], S)
    return out.reshape(1, T, H, D)


def bits(x):
    return x.contiguous().view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32)


def n_diff(a, b):
    return int((bits(a) != bits(b)).sum())


def rel_rms(x, ref):
    x, ref = x.double(), ref.double()
    return float(((x - ref) ** 2).mean().sqrt() / (ref ** 2).mean().sqrt())
