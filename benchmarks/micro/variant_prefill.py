"""Time prefill (K1/K2) of a differently flagged build. usage: python variant_prefill.py NAME DEFINE[,DEFINE...] B,L,H ..."""
import sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests" / "cfbi"))
import torch
from torch.profiler import ProfilerActivity, profile
from flash_kda.cfbi import alloc_state
from flash_kda.cfbi._ext import load_variant
from helpers import make_case
name, defs = sys.argv[1], tuple(d for d in sys.argv[2].split(",") if d)
mod = load_variant(name, defs)
for shp in sys.argv[3:]:
    B, L, H = (int(x) for x in shp.split(","))
    case = make_case(B * L, H, "authors", seed=0)
    q, k, v, g = (case[x].reshape(B, L, H, 128).contiguous() for x in ("q", "k", "v", "g"))
    beta = case["beta"].reshape(B, L, H).contiguous()
    out = torch.zeros_like(q); st = alloc_state(B, H)
    def f():
        mod.forward(q, k, v, g, beta, case["scale"], out, case["A_log"], case["dt_bias"], -5.0, st.E, st.rows, st.pos, None, 0, None, 0)
    for _ in range(2): f()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3): f()
        torch.cuda.synchronize()
    acc = defaultdict(float)
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            acc[ev.name.split("(")[0].split("<")[0]] += getattr(ev, "device_time", None) or getattr(ev, "cuda_time", 0.0)
    print(f"{name:10s} {','.join(defs):18s} prefill B={B} L={L} H={H}: " + "  ".join(f"{k.replace('cfbi::','')}={v/3:9.1f}" for k, v in sorted(acc.items(), key=lambda kv: -kv[1])[:2]), flush=True)
