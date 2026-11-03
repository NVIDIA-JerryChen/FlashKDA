"""One CF-BI prefill call for Nsight Compute. usage: python benchmarks/micro/umma_k2_ncu_run.py B L H [impl]"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests" / "cfbi"))
import torch
from flash_kda.cfbi import alloc_state, prefill
from helpers import make_case
B, L, H = (int(x) for x in sys.argv[1:4]); impl = int(sys.argv[4]) if len(sys.argv) > 4 else 4
case = make_case(B * L, H, "authors", seed=0)
q, k, v, g = (case[x].reshape(B, L, H, 128).contiguous() for x in ("q", "k", "v", "g"))
beta = case["beta"].reshape(B, L, H).contiguous()
out = torch.zeros_like(q); st = alloc_state(B, H)
for _ in range(2):
    st.pos.zero_()
    prefill(q, k, v, g, beta, case["scale"], out, case["A_log"], case["dt_bias"], -5.0, st, _impl=impl)
torch.cuda.synchronize()
