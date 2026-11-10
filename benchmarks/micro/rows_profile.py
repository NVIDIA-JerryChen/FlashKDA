"""Per-phase clock64 profile of the row engine (built with -DCFBI_ROWS_PROFILE). usage: python benchmarks/micro/rows_profile.py N H steps"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests" / "cfbi"))
import torch
from flash_kda.cfbi import alloc_state
from flash_kda.cfbi._ext import load_variant
from helpers import make_case
N, H, steps = (int(x) for x in sys.argv[1:4])
mod = load_variant("rowsprof", ("CFBI_ROWS_PROFILE=1",))
case = make_case(N * steps, H, "authors", seed=0)
st = alloc_state(N, H)
for s in range(steps):
    d = {x: torch.stack([case[x][0, n * steps + s] for n in range(N)], 0).unsqueeze(1).contiguous() for x in ("q", "k", "v", "g", "beta")}
    out = torch.zeros_like(d["q"])
    mod.forward(d["q"], d["k"], d["v"], d["g"], d["beta"], case["scale"], out, case["A_log"], case["dt_bias"], -5.0, st.E, st.rows, st.pos, None, 1, None, 0)
torch.cuda.synchronize()
