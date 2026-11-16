"""One CF-BI forward call for Nsight Compute. usage: python benchmarks/micro/ncu_run.py <prefill B L H | decode N H> [impl]"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests" / "cfbi"))
import torch
from flash_kda.cfbi import alloc_state, decode, prefill
from helpers import make_case
kind = sys.argv[1]
if kind == "prefill":
    B, L, H = (int(x) for x in sys.argv[2:5]); impl = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    case = make_case(B * L, H, "authors", seed=0)
    q, k, v, g = (case[x].reshape(B, L, H, 128).contiguous() for x in ("q", "k", "v", "g"))
    beta = case["beta"].reshape(B, L, H).contiguous()
    out = torch.zeros_like(q); st = alloc_state(B, H)
    for _ in range(2):
        st.pos.zero_()
        prefill(q, k, v, g, beta, case["scale"], out, case["A_log"], case["dt_bias"], -5.0, st, _impl=impl)
else:
    N, H = (int(x) for x in sys.argv[2:4]); impl = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    case = make_case(N * 80, H, "authors", seed=0)
    st = alloc_state(N, H)
    for s in range(80):
        d = {x: torch.stack([case[x][0, n * 80 + s] for n in range(N)], 0).unsqueeze(1).contiguous() for x in ("q", "k", "v", "g", "beta")}
        out = torch.zeros_like(d["q"])
        decode(d["q"], d["k"], d["v"], d["g"], d["beta"], case["scale"], out, case["A_log"], case["dt_bias"], -5.0, st, _impl=impl)
torch.cuda.synchronize()
