"""Smoke test of the UMMA K2: impl 8 vs impl 0 on one small sequence (prints mismatch counts)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests" / "cfbi"))
import torch
from flash_kda.cfbi import alloc_state, prefill, state_equal, state_mismatch
from helpers import make_case, n_diff

def run(case, T, H, impl):
    st = alloc_state(1, H)
    out = torch.zeros_like(case["q"]); o32 = torch.zeros(case["q"].shape, dtype=torch.float32, device="cuda")
    cu = torch.tensor([0, T], dtype=torch.int64, device="cuda")
    prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], out, case["A_log"], case["dt_bias"],
            case["lower_bound"], st, cu, out_fp32=o32, _impl=impl)
    torch.cuda.synchronize()
    return out, o32, st

T = int(sys.argv[1]) if len(sys.argv) > 1 else 64
H = int(sys.argv[2]) if len(sys.argv) > 2 else 1
case = make_case(T, H, "authors", seed=3)
a = run(case, T, H, 0)
b = run(case, T, H, 8)
print(f"T={T} H={H}: bf16 diff {n_diff(a[0], b[0])}/{a[0].numel()}  fp32 diff {n_diff(a[1], b[1])}/{a[1].numel()}  state {state_mismatch(a[2], b[2])}")
d = (a[1] != b[1])
if d.any():
    idx = d.nonzero()[:6]
    for i in idx:
        t_, h_, c_ = int(i[1]), int(i[2]), int(i[3])
        print(f"  token {t_} head {h_} col {c_}: ref {a[1][0,t_,h_,c_].item():.9g} umma {b[1][0,t_,h_,c_].item():.9g}")
    rows = d.any(dim=3).any(dim=2)[0]
    print("  tokens with any diff:", rows.nonzero().flatten().tolist()[:40])
    cols = d.any(dim=1).any(dim=1)[0]
    print("  cols with any diff:", cols.nonzero().flatten().tolist()[:40])
