"""Per-phase clock64 profile of K2 (built with -DCFBI_K2_PROFILE). usage: python benchmarks/micro/k2_profile.py [B L H]"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests" / "cfbi"))
import torch
from flash_kda.cfbi import alloc_state
from flash_kda.cfbi._ext import load_variant
from helpers import make_case
B, L, H = (int(x) for x in sys.argv[1:4]) if len(sys.argv) >= 4 else (1, 8192, 6)
mod = load_variant("k2prof", ("CFBI_K2_PROFILE=1",))
case = make_case(B * L, H, "authors", seed=0)
q, k, v, g = (case[x].reshape(B, L, H, 128).contiguous() for x in ("q", "k", "v", "g"))
beta = case["beta"].reshape(B, L, H).contiguous()
out = torch.zeros_like(q); st = alloc_state(B, H)
mod.forward(q, k, v, g, beta, case["scale"], out, case["A_log"], case["dt_bias"], -5.0, st.E, st.rows, st.pos, None, 0, None, 0)
torch.cuda.synchronize()
