"""References for the CF-BI program.

* ``exact_recurrence``: the KDA math as a plain per-token recurrence in fp64 (the accuracy oracle).
* ``program_emulation``: the contract of docs/20261102-cfbi-numerical-contract.md executed row by row in
  PyTorch fp32 (the "decode form" of the program; prefill is the same program scheduled differently). The
  tensor-core product is modelled as an idealised mma (exact products, one rounding per accumulate), so it
  matches the GPU kernels to ~1e-6 relative, not bit for bit.
"""
import math
from typing import Optional

import torch

LOG2E = 1.4426950408889634
_BIG = float("inf")


def _flatten(x, n_trailing):
    return x.reshape(-1, *x.shape[-n_trailing:])


def _seq_bounds(T_total: int, B: int, T: int, cu_seqlens):
    if cu_seqlens is not None:
        cu = [int(x) for x in cu_seqlens.tolist()]
    else:
        cu = [i * T for i in range(B + 1)]
    assert cu[-1] == T_total
    return cu


# ------------------------------------------------------------------------------------------------------
# fp64 exact recurrence
# ------------------------------------------------------------------------------------------------------
def prep_exact(q, k, g, beta, A_log, dt_bias, scale, lower_bound):
    """fp64 khat, qhat, log2-decay a2 and beta for flattened ``[T, H, 128]`` inputs (exact math, same eps)."""
    qd, kd = q.double(), k.double()
    inv_k = 1.0 / torch.sqrt((kd * kd).sum(-1, keepdim=True) + 1e-6)
    inv_q = 1.0 / torch.sqrt((qd * qd).sum(-1, keepdim=True) + 1e-6)
    khat, qhat = kd * inv_k, qd * inv_q * float(scale)
    ale = torch.exp2(A_log.double() * LOG2E)[None, :, None]
    sig = torch.sigmoid(ale * (g.double() + dt_bias.double()[None]))
    a2 = (float(lower_bound) * LOG2E) * sig
    return khat, qhat, a2, torch.sigmoid(beta.double())


def exact_recurrence(q, k, v, g, beta, A_log, dt_bias, scale, lower_bound, cu_seqlens=None, state0=None):
    """fp64 reference: out ``[B*T, H, 128]`` (fp64) and final states ``[N, H, V, K]`` (fp64).

    S_t = diag(2^a2_t) S_{t-1};  S_t += beta_t (v_t - S_t khat_t) khat_t^T;  o_t = S_t qhat_t   (state is [V, K]).
    """
    B, T = q.shape[0], q.shape[1]
    qf, kf, vf, gf = (_flatten(x, 2) for x in (q, k, v, g))
    bf = _flatten(beta, 1)
    T_total, H = qf.shape[0], qf.shape[1]
    khat, qhat, a2, bt = prep_exact(qf, kf, gf, bf, A_log, dt_bias, scale, lower_bound)
    alpha = torch.exp2(a2)
    vd = vf.double()
    cu = _seq_bounds(T_total, B, T, cu_seqlens)
    N = len(cu) - 1
    out = torch.zeros(T_total, H, 128, dtype=torch.float64, device=q.device)
    final = torch.zeros(N, H, 128, 128, dtype=torch.float64, device=q.device)
    for n in range(N):
        S = (state0[n].double().clone() if state0 is not None
             else torch.zeros(H, 128, 128, dtype=torch.float64, device=q.device))
        for t in range(cu[n], cu[n + 1]):
            S = S * alpha[t][:, None, :]
            kS = torch.einsum("hk,hvk->hv", khat[t], S)
            S = S + bt[t][:, None, None] * (vd[t] - kS)[:, :, None] * khat[t][:, None, :]
            out[t] = torch.einsum("hk,hvk->hv", qhat[t], S)
        final[n] = S
    return out, final


# ------------------------------------------------------------------------------------------------------
# the CF-BI program, row by row, fp32
# ------------------------------------------------------------------------------------------------------
def _f32(x):
    return torch.as_tensor(x, dtype=torch.float32)


def _fma(a, b, c):
    return (a.double() * b.double() + c.double()).float()


def _tf32_split(x):
    xi = x.contiguous().view(torch.int32)
    hi = (xi & -8192).view(torch.float32)   # 0xFFFFE000
    lo = ((x - hi).contiguous().view(torch.int32) & -8192).view(torch.float32)
    return hi, lo


def _dot3(A, B):
    """Idealised dot3: A [..., M, K], B [..., N, K] -> [..., M, N]; K-steps of 8 ascending, BIG/SMALL split."""
    Ah, Al = _tf32_split(A)
    Bh, Bl = _tf32_split(B)
    shape = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2]) + (A.shape[-2], B.shape[-2])
    big = torch.full(shape, -0.0, dtype=torch.float32)
    small = torch.full(shape, -0.0, dtype=torch.float32)

    def mm(x, y, acc):   # exact products + accumulator in fp64, one rounding to fp32
        return (acc.double() + torch.einsum("...mk,...nk->...mn", x.double(), y.double())).float()

    for k0 in range(0, A.shape[-1], 8):
        sl = slice(k0, k0 + 8)
        small = mm(Al[..., sl], Bh[..., sl], small)
        small = mm(Ah[..., sl], Bl[..., sl], small)
        big = mm(Ah[..., sl], Bh[..., sl], big)
    return big + small


def _prep_f32(q, k, g, beta, A_log, dt_bias, scale, lower_bound):
    """q,k,g [H,128] bf16, beta [H] bf16 -> khat, qhat, a2 [H,128], beta [H] (fp32, true transcendentals)."""
    def l2inv(x):
        xf = x.float().reshape(x.shape[0], 16, 8)
        p = torch.zeros(x.shape[0], 16)
        for i in range(8):
            p = _fma(xf[..., i], xf[..., i], p)
        idx = torch.arange(16)
        for off in (8, 4, 2, 1):
            p = p + p[:, idx ^ off]
        return torch.rsqrt(p[:, 0] + _f32(1e-6))

    khat = k.float() * l2inv(k)[:, None]
    qhat = (q.float() * l2inv(q)[:, None]) * _f32(scale)
    ale = torch.exp2(A_log * _f32(LOG2E))[:, None]
    sig = lambda x: _fma(torch.tanh(x * _f32(0.5)), _f32(0.5), _f32(0.5))
    a2 = _f32(lower_bound * LOG2E) * sig(ale * (g.float() + dt_bias))
    return khat, qhat, a2, sig(beta.float())


class ProgramEmulation:
    """Row-by-row evaluation of the contract for ``H`` heads of one sequence (CPU, fp32)."""

    def __init__(self, H):
        self.H = H
        self.E = torch.zeros(H, 128, 128)
        self.rows = []   # dicts of fp32 [H,128] vectors: khat, g2, ut, w, u

    def step(self, q, k, v, g, beta, A_log, dt_bias, scale, lower_bound):
        H, pos = self.H, len(self.rows)
        khat, qhat, a2, bt = _prep_f32(q, k, g, beta, A_log, dt_bias, scale, lower_bound)
        g2 = a2 if pos == 0 else self.rows[-1]["g2"] + a2
        s, b = pos // 16, 16 * (pos // 16)
        gref = g2 if pos == b else self.rows[b]["g2"]
        ex2 = lambda x: torch.where(torch.exp2(x) < torch.finfo(torch.float32).tiny, torch.zeros_like(x), torch.exp2(x))
        ks, qs = khat * ex2(g2 - gref), qhat * ex2(g2 - gref)
        allk = torch.stack([r["khat"] for r in self.rows] + [khat], dim=1)       # [H, pos+1, 128]
        allg = torch.stack([r["g2"] for r in self.rows] + [g2], dim=1)
        kjr = allk * ex2(gref[:, None, :] - allg)                                  # reference row of this t
        akk = bt[:, None] * _dot3(ks[:, None, :], kjr)[:, 0, :]                    # [H, pos+1] (last = diagonal)
        aqk = _dot3(qs[:, None, :], kjr)[:, 0, :]
        gamma = ex2(g2)
        X = {}
        for name, rhs in (("ut", bt[:, None] * v.float()), ("w", bt[:, None] * (gamma * khat))):
            hist = torch.stack([r[name] for r in self.rows] + [rhs], dim=1)        # last entry is a placeholder
            x = rhs if s == 0 else rhs - _dot3(akk[:, None, :b], hist[:, :b, :].transpose(1, 2))[:, 0, :]
            for j in range(b, pos):
                x = _fma(-akk[:, j, None], hist[:, j, :], x)
            X[name] = x
        gq = gamma * qhat
        p1 = _dot3(X["w"][:, None, :], self.E)[:, 0, :]                            # [H, V]
        p2 = _dot3(gq[:, None, :], self.E)[:, 0, :]
        u = X["ut"] - p1
        uh = torch.stack([r["u"] for r in self.rows] + [u], dim=1)
        o = p2 if s == 0 else p2 + _dot3(aqk[:, None, :b], uh[:, :b, :].transpose(1, 2))[:, 0, :]
        for j in range(b, pos + 1):
            o = _fma(aqk[:, j, None], uh[:, j, :], o)
        self.rows.append({"khat": khat, "g2": g2, "ut": X["ut"], "w": X["w"], "u": u})
        if pos == 63:   # chunk end
            allk = torch.stack([r["khat"] for r in self.rows], dim=1)
            allg = torch.stack([r["g2"] for r in self.rows], dim=1)
            kd = allk * ex2(g2[:, None, :] - allg)                                 # [H, 64, K]
            uu = torch.stack([r["u"] for r in self.rows], dim=1)                   # [H, 64, V]
            p3 = _dot3(kd.transpose(1, 2), uu.transpose(1, 2))                     # [H, K, V]
            self.E = self.E * ex2(g2)[:, None, :] + p3.transpose(1, 2)
            self.rows = []
        return o


def program_emulation(q, k, v, g, beta, A_log, dt_bias, scale, lower_bound):
    """Run the contract on one sequence (inputs ``[T, H, 128]`` / beta ``[T, H]``, CPU); returns fp32 O [T,H,128]."""
    q, k, v, g, beta = (x.cpu() for x in (q, k, v, g, beta))
    A_log, dt_bias = A_log.cpu(), dt_bias.cpu()
    eng = ProgramEmulation(q.shape[1])
    return torch.stack([eng.step(q[t], k[t], v[t], g[t], beta[t], A_log, dt_bias, scale, lower_bound)
                        for t in range(q.shape[0])], dim=0)
