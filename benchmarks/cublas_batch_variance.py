"""Control: is cuBLAS (torch.matmul / F.linear, bf16) batch invariant on this GPU? bi_linear is, by construction.

(a) The same row of x @ W^T computed with M = 1, M = 64 and M = 8192 (rows 0..R-1 of each), K = N = 2048 .. 4096.
(b) A wide scan: rows 0..R-1 computed with M in {1, 2, 4, ..., 8192} against the M = 8192 result.
Prints the fraction of differing bf16 outputs (bit comparison). usage: python benchmarks/cublas_batch_variance.py [--rows 32]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from flash_kda.cfbi import bi_linear


def bits(x):
    return x.contiguous().view(torch.int16)


def frac(a, b):
    return float((bits(a) != bits(b)).float().mean())


def rows_with_m(fn, X, W, M, R):
    """Rows 0..R-1 of fn(X[:M], W); for M = 1 every row is its own call."""
    if M == 1:
        return torch.cat([fn(X[r:r + 1], W) for r in range(R)])
    return fn(X[:M], W)[:R]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=32)
    a = ap.parse_args()
    R = a.rows
    mm = lambda x, w: torch.matmul(x, w.t())
    lin = lambda x, w: F.linear(x, w)
    print(f"device: {torch.cuda.get_device_name()}  torch {torch.__version__}  (rows 0..{R - 1} of each call compared)\n")

    print("(a) fraction of differing bf16 outputs: same rows computed with M = 1 / 64 / 8192")
    print(f"{'K=N':>6} | {'matmul 1v64':>12} {'1v8192':>8} {'64v8192':>9} | {'F.linear 1v64':>14} {'1v8192':>8} {'64v8192':>9} | {'bi_linear any':>13}")
    for KN in (2048, 2560, 3072, 3584, 4096):
        g = torch.Generator(device="cuda").manual_seed(KN)
        X = (torch.randn(8192, KN, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
        W = (torch.randn(KN, KN, device="cuda", generator=g) * 0.05).to(torch.bfloat16)
        cells = []
        for fn in (mm, lin):
            y1, y64, y8k = (rows_with_m(fn, X, W, M, R) for M in (1, 64, 8192))
            cells += [frac(y1, y64), frac(y1, y8k), frac(y64, y8k)]
        b1, b64, b8k = (rows_with_m(lambda x, w: bi_linear(x, w), X, W, M, R) for M in (1, 64, 8192))
        bi = max(frac(b1, b64), frac(b1, b8k), frac(b64, b8k))
        print(f"{KN:>6} | {cells[0]:>12.4%} {cells[1]:>8.4%} {cells[2]:>9.4%} | {cells[3]:>14.4%} {cells[4]:>8.4%} {cells[5]:>9.4%} | {bi:>13.4%}", flush=True)

    Ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    print(f"\n(b) fraction of differing bf16 outputs of rows 0..{R - 1}: computed with M rows vs with M = 8192 (torch.matmul)")
    print(f"{'K':>5} {'N':>5} | " + " ".join(f"{M:>7d}" for M in Ms))
    worst = 0.0
    for K, N in [(2048, 2048), (3072, 3072), (4096, 4096), (4096, 2048), (2048, 4096), (4096, 11008), (11008, 4096), (4096, 128), (4096, 16), (128, 2048)]:
        g = torch.Generator(device="cuda").manual_seed(K * 7 + N)
        X = (torch.randn(8192, K, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
        W = (torch.randn(N, K, device="cuda", generator=g) * 0.05).to(torch.bfloat16)
        ref = mm(X, W)[:R]
        cells = []
        for M in Ms:
            got = rows_with_m(mm, X, W, M, R)
            f = frac(got, ref[:got.shape[0]])
            worst = max(worst, f)
            cells.append(f"{f:>7.3%}")
        print(f"{K:>5} {N:>5} | " + " ".join(cells), flush=True)
        # bi_linear on the same data
        bref = bi_linear(X, W)[:R]
        for M in Ms:
            got = rows_with_m(lambda x, w: bi_linear(x, w), X, W, M, R)
            assert frac(got, bref[:got.shape[0]]) == 0.0, "bi_linear is not invariant?!"
    print(f"\nworst cuBLAS fraction in (b): {worst:.3%}; bi_linear: exactly 0 for every K, N, M above")


if __name__ == "__main__":
    main()
