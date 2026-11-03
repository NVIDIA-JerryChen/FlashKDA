"""Time decode (rows kernel) of a differently flagged build. usage: python variant_bench.py NAME DEFINE[,DEFINE...] [N,H ...]"""
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
shapes = [tuple(int(x) for x in a.split(",")) for a in sys.argv[3:]] or [(1, 12), (16, 12), (64, 12), (128, 12)]
mod = load_variant(name, defs)
for N, H in shapes:
    steps = 64
    case = make_case(N * steps, H, "authors", seed=0)
    st = alloc_state(N, H)
    per = [{x: torch.stack([case[x][0, n * steps + s] for n in range(N)], 0).unsqueeze(1).contiguous() for x in ("q", "k", "v", "g", "beta")} for s in range(steps)]
    out = torch.zeros_like(per[0]["q"])
    cnt = [0]
    def f():
        d = per[cnt[0] % steps]; cnt[0] += 1
        mod.forward(d["q"], d["k"], d["v"], d["g"], d["beta"], case["scale"], out, case["A_log"], case["dt_bias"], -5.0, st.E, st.rows, st.pos, None, 1, None, 0)
    for _ in range(3): f()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(64): f()
        torch.cuda.synchronize()
    acc = defaultdict(float)
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            acc[ev.name.split("(")[0].split("<")[0]] += getattr(ev, "device_time", None) or getattr(ev, "cuda_time", 0.0)
    print(f"{name:12s} {','.join(defs):24s} decode N={N:4d} H={H}: " + "  ".join(f"{k.replace('cfbi::','')}={v/64:7.1f}" for k, v in sorted(acc.items(), key=lambda kv: -kv[1])[:2]), flush=True)
