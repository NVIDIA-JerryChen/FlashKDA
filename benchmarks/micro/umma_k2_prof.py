"""Timeline of one UMMA K2 chunk (CTA (0,0,0)): build the -DCFBI_UMMA_PROFILE variant and run impl 8.
usage: python benchmarks/micro/umma_k2_prof.py [B] [L] [H]   (prints 'PROF w<warp> <event> <cycles since scan start>')"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests" / "cfbi"))
import torch
from flash_kda.cfbi._ext import load_variant
from flash_kda.cfbi import alloc_state
from helpers import make_case

B = int(sys.argv[1]) if len(sys.argv) > 1 else 1
L = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
H = int(sys.argv[3]) if len(sys.argv) > 3 else 1
defs = ("CFBI_UMMA_PROFILE=1",) + tuple(sys.argv[4:])
mod = load_variant("umma_prof", defs)
case = make_case(B * L, H, "authors", seed=0)
q, k, v, g = (case[x].reshape(B, L, H, 128).contiguous() for x in ("q", "k", "v", "g"))
beta = case["beta"].reshape(B, L, H).contiguous()
out = torch.zeros_like(q)
st = alloc_state(B, H)
for rep in range(1):
    mod.forward(q, k, v, g, beta, float(case["scale"]), out, case["A_log"], case["dt_bias"], -5.0, st.E, st.rows, st.pos, None, 0, None, 4, None)
    torch.cuda.synchronize()
