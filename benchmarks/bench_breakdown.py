"""Per-kernel GPU time of the CF-BI pipeline (plan / row engine / K1 / K2) for one implementation.

usage: python benchmarks/bench_breakdown.py [--impl 0|3|4] [--prefill B L H ...] [--decode N H ...] [--warm POS]
  --impl     0 = optimised kernels (default), 3 = the frozen v1 reference kernels,
             4 = optimised row engine + K1 with the tcgen05 (UMMA) K2 (kernel k2u_kernel instead of k2_kernel)
  --prefill  one or more "B,L,H" shapes (default: the CF-BI report's H6/L8K shapes)
  --decode   one or more "N,H" shapes (decode averaged over 64 consecutive steps = every chunk phase)
  --warm     chunk phase (pos) the sequences start the prefill call at (exercises the row engine's head rows)
Times are the sum of kernel durations from the CUDA profiler (no launch overhead), microseconds per call.
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "cfbi"))

import torch
from torch.profiler import ProfilerActivity, profile

from flash_kda.cfbi import alloc_state, decode, forward_train, prefill
from helpers import make_case

LB = -5.0


def kernel_times(fn, iters=5, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    acc = defaultdict(float)
    cnt = defaultdict(int)
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            name = ev.name.split("(")[0].split("<")[0]
            acc[name] += getattr(ev, "device_time", None) or getattr(ev, "cuda_time", 0.0)
            cnt[name] += 1
    return {k: v / iters for k, v in acc.items()}, {k: c / iters for k, c in cnt.items()}


def short(name):
    for key in ("plan_kernel", "finalize_pos_kernel", "rows_kernel", "k1_kernel", "k2_kernel", "k1_", "k2_", "rows_"):
        if key in name:
            return ("ref::" if "ref" in name else "") + name.split("cfbi")[-1].strip(":") if False else name
    return name


def report(label, times, counts):
    total = sum(times.values())
    parts = "  ".join(f"{k.replace('cfbi::', '')}={v:9.1f}" + (f"(x{counts[k]:.0f})" if counts[k] > 1.01 else "")
                      for k, v in sorted(times.items(), key=lambda kv: -kv[1]))
    print(f"{label:34s} total {total:10.1f} us   {parts}", flush=True)


def bench_prefill(B, L, H, impl, warm_pos):
    case = make_case(B * L, H, "authors", seed=0)
    q = case["q"].reshape(B, L, H, 128).contiguous()
    k = case["k"].reshape(B, L, H, 128).contiguous()
    v = case["v"].reshape(B, L, H, 128).contiguous()
    g = case["g"].reshape(B, L, H, 128).contiguous()
    beta = case["beta"].reshape(B, L, H).contiguous()
    out = torch.zeros_like(q)
    st = alloc_state(B, H)
    if warm_pos:
        st.pos.fill_(warm_pos)

    def f():
        if warm_pos:
            st.pos.fill_(warm_pos)
        prefill(q, k, v, g, beta, case["scale"], out, case["A_log"], case["dt_bias"], LB, st, _impl=impl)

    t, c = kernel_times(f, iters=3)
    report(f"prefill B={B} L={L} H={H} pos0={warm_pos}", t, c)
    del q, k, v, g, beta, out, st
    torch.cuda.empty_cache()


def bench_train(B, L, H, impl):
    case = make_case(B * L, H, "authors", seed=0)
    q = case["q"].reshape(B, L, H, 128).contiguous()
    k = case["k"].reshape(B, L, H, 128).contiguous()
    v = case["v"].reshape(B, L, H, 128).contiguous()
    g = case["g"].reshape(B, L, H, 128).contiguous()
    beta = case["beta"].reshape(B, L, H).contiguous()
    t, c = kernel_times(lambda: forward_train(q, k, v, g, beta, case["scale"], case["A_log"], case["dt_bias"], LB, _impl=impl), iters=3)
    report(f"train-fwd B={B} L={L} H={H}", t, c)


def bench_decode(N, H, impl):
    steps = 64
    case = make_case(N * steps, H, "authors", seed=0)
    st = alloc_state(N, H)
    # per-step contiguous [N,1,H,*] inputs; sequences advance one token per call (all 64 chunk phases are visited)
    per_step = []
    for s in range(steps):
        d = {key: torch.stack([case[key][0, n * steps + s] for n in range(N)], dim=0).unsqueeze(1).contiguous()
             for key in ("q", "k", "v", "g", "beta")}
        per_step.append(d)
    out = torch.zeros_like(per_step[0]["q"])
    i = [0]

    def f():
        d = per_step[i[0] % steps]
        i[0] += 1
        decode(d["q"], d["k"], d["v"], d["g"], d["beta"], case["scale"], out, case["A_log"], case["dt_bias"], LB, st, _impl=impl)

    t, c = kernel_times(f, iters=64, warm=2)
    report(f"decode N={N} H={H} (64-step avg)", t, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", type=int, default=0)
    ap.add_argument("--prefill", nargs="*", default=None)
    ap.add_argument("--decode", nargs="*", default=None)
    ap.add_argument("--train", nargs="*", default=None)
    ap.add_argument("--warm", type=int, default=0)
    a = ap.parse_args()
    print(f"device {torch.cuda.get_device_name()}  impl={a.impl}")
    pre = a.prefill if a.prefill is not None else (["1,8192,6", "8,8192,6", "64,8192,6", "1,32768,32"] if a.decode is None and a.train is None else [])
    dec = a.decode if a.decode is not None else (["1,12", "16,12", "64,12", "128,12"] if a.prefill is None and a.train is None else [])
    for s in pre:
        B, L, H = (int(x) for x in s.split(","))
        bench_prefill(B, L, H, a.impl, a.warm)
    for s in a.train or []:
        B, L, H = (int(x) for x in s.split(","))
        bench_train(B, L, H, a.impl)
    for s in dec:
        N, H = (int(x) for x in s.split(","))
        bench_decode(N, H, a.impl)


if __name__ == "__main__":
    main()
