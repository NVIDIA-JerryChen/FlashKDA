"""Accuracy of bi_linear vs cuBLAS against an fp64 reference (the same inputs), and how often their bf16 outputs differ.
usage: python benchmarks/micro/linear_accuracy.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from flash_kda.cfbi import bi_linear


def main():
    print(f"{'M':>5} {'K':>6} {'N':>6} | {'rel RMS bi':>11} {'rel RMS cuBLAS':>15} {'bf16 floor':>11} | {'bi != cuBLAS':>13} | max |err|/(0.5ulp+2e-6 S): bi, cuBLAS")
    for M, K, N in [(256, 4096, 4096), (64, 4096, 2048), (128, 11008, 4096), (1, 4096, 16)]:
        g = torch.Generator(device="cuda").manual_seed(K + N)
        x = (torch.randn(M, K, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
        w = (torch.randn(N, K, device="cuda", generator=g) * 0.05).to(torch.bfloat16)
        ref = x.double() @ w.double().t()
        S = x.double().abs() @ w.double().abs().t()
        bi, cb = bi_linear(x, w), torch.matmul(x, w.t())
        rms = lambda a: float(((a.double() - ref) ** 2).mean().sqrt() / (ref ** 2).mean().sqrt())
        floor = rms(ref.to(torch.bfloat16))   # error of rounding the exact result to bf16
        ulp = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
        bound = lambda a: float(((a.double() - ref).abs() / (0.5 * ulp + 2e-6 * S)).max())
        diff = float((bi.view(torch.int16) != cb.view(torch.int16)).float().mean())
        print(f"{M:>5} {K:>6} {N:>6} | {rms(bi):>11.3e} {rms(cb):>15.3e} {floor:>11.3e} | {diff:>13.4%} | {bound(bi):.3f}, {bound(cb):.3f}")


if __name__ == "__main__":
    main()
