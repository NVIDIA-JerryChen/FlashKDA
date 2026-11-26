"""Kernel-level time split of fwd64 / replay64 (torch profiler, GPU time per kernel)."""
import os, sys
import torch
from torch.profiler import ProfilerActivity, profile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import flash_kda
from flash_kda import replay64 as R64
dev = "cuda"; D = 128


def tokens(T, H):
    r = lambda *s: torch.randn(*s, device=dev).to(torch.bfloat16)
    return r(T, H, D), r(T, H, D), r(T, H, D), r(T, H, D), r(T, H)


def prof(fn, label, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    tot = {}
    for e in p.key_averages():
        if e.device_time_total > 0 and e.key not in ("cudaDeviceSynchronize",):
            tot[e.key] = e.device_time_total / iters
    clean = lambda k: k.replace('(anonymous namespace)::', '').replace('c64::', '').split('(')[0][:28]
    print(f"[{label}]", "  ".join(f"{clean(k)}={v:8.1f}us" for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:4]), flush=True)


H = 12
for name, lens in (("8x8192", [8192] * 8), ("64x512", [512] * 64), ("1x32768", [32768])):
    T = sum(lens); N = len(lens)
    q, k, v, g, beta = tokens(T, H)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int64, device=dev)
    out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device=dev)
    A_log = torch.rand(H, device=dev); dt = torch.rand(H, D, device=dev)
    st = torch.zeros(N, H, D, D, dtype=torch.bfloat16, device=dev)
    args = (q[None], k[None], v[None], g[None], beta[None], D ** -0.5, out, A_log, dt, -5.0)
    for cfg in (0, 1, 2):
        prof(lambda: R64.fwd64(*args, initial_state=None, final_state=st, cu_seqlens=cu, _k2_cfg=cfg), f"fwd64 cfg{cfg} {name}")
    prof(lambda: flash_kda.fwd(*args, initial_state=None, final_state=st, cu_seqlens=cu), f"fwd   {name}")
for N in (1, 64, 128):
    A_log = torch.rand(H, device=dev); dt = torch.rand(H, D, device=dev)
    s64 = R64.alloc_state(N, H)
    q, k, v, g, b = tokens(N, H)
    out = torch.empty(N, H, D, dtype=torch.bfloat16, device=dev)
    # position 40 of the open chunk (3 sub-blocks)
    s64.pos.fill_(40)
    prof(lambda: (s64.pos.fill_(40), R64.decode(q, k, v, g, b, D ** -0.5, out, A_log, dt, -5.0, s64)), f"decode64 N={N} pos=40")
    s64.pos.fill_(63)
    prof(lambda: (s64.pos.fill_(63), R64.decode(q, k, v, g, b, D ** -0.5, out, A_log, dt, -5.0, s64)), f"decode64 N={N} pos=63 (closes)")
