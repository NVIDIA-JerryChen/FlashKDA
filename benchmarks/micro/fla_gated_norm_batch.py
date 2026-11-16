"""Informational: is FLA's FusedRMSNormGated (the op after the KDA core) batch invariant on this GPU? Rows computed with M = 1
vs M = T rows (bf16, head dim 128, sigmoid gate, eps 1e-5), bit comparison. Observation only, nothing here pins the program.
usage: python benchmarks/micro/fla_gated_norm_batch.py   (needs flash-linear-attention on PYTHONPATH)
"""
import torch

from fla.modules import FusedRMSNormGated


def main():
    torch.manual_seed(0)
    H, D = 16, 128
    norm = FusedRMSNormGated(D, activation="sigmoid", eps=1e-5).cuda().to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_((torch.rand(D, device="cuda") + 0.5).to(torch.bfloat16))
    T = 4096
    o = torch.randn(1, T, H, D, device="cuda").to(torch.bfloat16)
    g = torch.randn(1, T, H, D, device="cuda").to(torch.bfloat16)
    with torch.no_grad():
        full = norm(o, g)
        print(f"{'M rows':>8} | rows 0..{min(64, T) - 1} differing from the M = {T} call")
        for M in (1, 2, 3, 8, 17, 64, 129, 1000, T):
            R = min(M, 64)
            if M == 1:
                sub = torch.cat([norm(o[:, r:r + 1], g[:, r:r + 1]) for r in range(64)], dim=1)
            else:
                sub = norm(o[:, :M], g[:, :M])[:, :R]
            d = (sub.view(torch.int16) != full[:, :sub.shape[1]].view(torch.int16)).sum().item()
            print(f"{M:>8} | {d} of {sub.numel()}")


if __name__ == "__main__":
    main()
