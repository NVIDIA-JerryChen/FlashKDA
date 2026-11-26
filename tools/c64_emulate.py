"""CPU emulation of the bf16 chunk programs (an accuracy model, not bit-exact): chunk 16 as in flash_kda.fwd ("orig") and
the chunk-64 program of flash_kda.replay64 ("sub": 16-row sub-blocks with reference rows, docs/20261125-replay-c64.md).
Everything is fp32 with the bf16 round points of the kernels; a tensor-core product is an fp32 matmul of bf16-valued
operands (exact products, fp32 accumulate, in a different order than the GPU), sub / add round once from fp32.

usage (CPU only, a few minutes):  python tools/c64_emulate.py [T] [H] [regime,regime,...]
prints the relative RMS error against the fp64 recurrence in units of the bf16 rounding floor for chunk 16 and chunk 64."""
import math
import torch

bf = torch.bfloat16
LOG2E = 1.4426950408889634


def r(x):
    return x.to(bf).float()


def ex2(x):
    return torch.exp2(x)


def solve_unit_lower(X):  # X [H, n, n] unit lower -> inverse (fp32)
    n = X.shape[-1]
    eye = torch.eye(n, dtype=X.dtype).expand_as(X)
    return torch.linalg.solve_triangular(X, eye, upper=False, unitriangular=True)


def inv16(L):
    """(I+L)^-1 for a 16x16 strictly lower fp32 L: 8x8 fp32 inverses of the diagonal blocks + bf16 merge (as utils.cuh)."""
    H = L.shape[0]
    I8 = torch.eye(8)
    Ai = solve_unit_lower(I8 + L[:, :8, :8])
    Bi = solve_unit_lower(I8 + L[:, 8:, 8:])
    C = L[:, 8:, :8]
    P = torch.zeros(H, 16, 16)
    P[:, :8, :8] = r(Ai)
    P[:, 8:, 8:] = r(Bi)
    M = torch.zeros(H, 16, 16)
    M[:, 8:, :8] = r(C)
    dc = P @ M
    o = r(-dc) @ P
    return P + r(o)


def inv_blocks(Lfull, nsb):
    """(I+L)^-1 of the 16*nsb square strictly lower L: 16x16 diagonal inverses + bf16 block forward substitution."""
    H = Lfull.shape[0]
    blk = lambda M, i, j: M[:, 16 * i:16 * i + 16, 16 * j:16 * j + 16]  # noqa: E731
    INV = torch.zeros_like(Lfull)
    for b in range(nsb):
        INV[:, 16 * b:16 * b + 16, 16 * b:16 * b + 16] = inv16(blk(Lfull, b, b))
    for d in range(1, nsb):
        for j in range(nsb - d):
            i = j + d
            Y = sum(r(blk(Lfull, i, m)) @ blk(INV, m, j) for m in range(j, i))
            INV[:, 16 * i:16 * i + 16, 16 * j:16 * j + 16] = r(-(blk(INV, i, i) @ r(Y)))
    return INV


class Chunker:
    def __init__(self, chunk, variant, A_log, dt_bias, scale, lower_bound):
        assert chunk % 16 == 0 and variant in ("orig", "sub")
        self.C, self.variant = chunk, variant
        self.nsb = chunk // 16
        self.ale = torch.exp(A_log.float())[:, None, None]      # [H,1,1]
        self.dt = dt_bias.float()[:, None, :]                   # [H,1,D]
        self.scale = scale
        self.gate_scale = float(lower_bound * LOG2E)

    def step(self, S, q, k, v, g, beta, nvalid):
        """S [H,V,K] (bf16-valued fp32), rows [H,C,D] bf16-valued fp32 (zero padded), beta [H,C] -> (out [H,C,V], S')."""
        C, nsb = self.C, self.nsb
        H = S.shape[0]
        qh = r(q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6))
        kh = r(k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6))
        a2 = self.gate_scale * torch.sigmoid(self.ale * (g + self.dt))
        a2[:, nvalid:] = 0
        G = torch.cumsum(a2, dim=1)                              # [H,C,D]
        Gend = G[:, -1:, :]                                      # [H,1,D] (rows past nvalid add 0)
        gt = ex2(Gend[:, 0, :])                                  # [H,D]
        sc = r(torch.tensor(self.scale))
        sb = torch.sigmoid(beta)                                 # fp32 sigmoid (K1)
        sbb = r(sb)                                              # bf16 sigmoid (K2)

        if self.variant == "orig":
            assert C == 16
            Ex = r(ex2(G)); Ix = r(ex2(-G))
            qd = r(r(qh * Ex) * sc); kd = r(kh * Ex); ki = r(kh * Ix)
            kr = r(r(kh * Ix) * r(gt)[:, None, :])
            Kd, Qd, Kr = kd, qd, kr
            Lm = (kd @ ki.transpose(1, 2)) * sb[:, :, None]
            Lm = torch.tril(Lm, -1)
            Mq = torch.tril(r(qd @ ki.transpose(1, 2)))
            INV = inv16(Lm)
        else:
            R = []  # reference cumsum before sub-block b
            for b in range(nsb):
                R.append(torch.zeros(H, 1, G.shape[-1]) if b == 0 else G[:, 16 * b - 1:16 * b, :])
            Lm = torch.zeros(H, C, C); Mq = torch.zeros(H, C, C)
            for i in range(nsb):
                rows_i = slice(16 * i, 16 * i + 16)
                Gl = G[:, rows_i, :] - R[i]
                kd_loc = r(kh[:, rows_i] * r(ex2(Gl)))
                qd_loc = r(r(qh[:, rows_i] * r(ex2(Gl))) * sc)
                for j in range(i + 1):
                    rows_j = slice(16 * j, 16 * j + 16)
                    w = r(kh[:, rows_j] * r(ex2(R[i] - G[:, rows_j, :])))
                    Lm[:, rows_i, rows_j] = (kd_loc @ w.transpose(1, 2)) * sb[:, rows_i, None]
                    Mq[:, rows_i, rows_j] = r(qd_loc @ w.transpose(1, 2))
            # masks: strictly lower for L overall, lower incl. diagonal for Mqk
            Lm = torch.tril(Lm, -1)
            Mq = torch.tril(Mq)
            Kd = r(kh * r(ex2(G)))
            Qd = r(r(qh * r(ex2(G))) * sc)
            Kr = r(kh * r(ex2(Gend - G)))
            INV = inv_blocks(Lm, nsb)

        P = r(Kd @ S.transpose(1, 2))                            # [H,C,V]
        O0 = r(Qd @ S.transpose(1, 2))
        Wp = r(r(v - P) * sbb[:, :, None])
        U = r(INV @ Wp)
        out = r(O0 + r(Mq @ U))
        Snew = r(S * gt[:, None, :] + U.transpose(1, 2) @ Kr)
        return out, Snew


def run(q, k, v, g, beta, A_log, dt_bias, scale, lower_bound, chunk, variant):
    """q,k,v,g bf16 [1,T,H,D], beta [1,T,H] -> out bf16 [1,T,H,D] (one sequence from the zero state)."""
    _, T, H, D = q.shape
    ch = Chunker(chunk, variant, A_log, dt_bias, scale, lower_bound)
    f = lambda x: x[0].float().permute(1, 0, 2).contiguous()  # [T,H,D] -> [H,T,D]  # noqa: E731
    qf, kf, vf, gf = f(q), f(k), f(v), f(g)
    bt = beta[0].float().permute(1, 0).contiguous()           # [H,T]
    S = torch.zeros(H, D, D)
    outs = torch.zeros(H, T, D)
    for t0 in range(0, T, chunk):
        n = min(chunk, T - t0)
        pad = lambda x: torch.nn.functional.pad(x[:, t0:t0 + n], (0, 0, 0, chunk - n)) if x.dim() == 3 else torch.nn.functional.pad(x[:, t0:t0 + n], (0, chunk - n))  # noqa: E731
        o, S = ch.step(S, pad(qf), pad(kf), pad(vf), pad(gf), pad(bt), n)
        outs[:, t0:t0 + n] = o[:, :n]
    return outs.permute(1, 0, 2).unsqueeze(0).to(bf)


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
    import c64_helpers as Hh
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    regimes = sys.argv[3].split(",") if len(sys.argv) > 3 else Hh.REGIMES
    for reg in regimes:
        case = Hh.make_case(T, H, reg, seed=1, device="cpu")
        ref = Hh.exact_recurrence(case)
        floor = Hh.rel_rms(ref.to(torch.bfloat16), ref)
        row = [f"{reg:14s} lb={case['lower_bound']:<5g}"]
        for name, chunk, variant in (("chunk16", 16, "orig"), ("chunk64", 64, "sub")):
            out = run(case["q"], case["k"], case["v"], case["g"], case["beta"], case["A_log"], case["dt_bias"],
                      case["scale"], case["lower_bound"], chunk, variant)
            row.append(f"{name} {Hh.rel_rms(out, ref) / floor:5.2f}x floor")
        print("  ".join(row), flush=True)
