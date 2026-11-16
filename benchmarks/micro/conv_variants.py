"""Launch-geometry variants of the conv kernel (thread block size x load unroll): builds each variant as its own extension,
checks that it gives the same bits as the production build and times prefill / decode. Performance experiment only.

usage: python benchmarks/micro/conv_variants.py "128,8" "128,16" "256,8" ...
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from flash_kda.cfbi import load_conv_ext
from flash_kda.cfbi._ext import load_conv_variant


def graph_us(fn, n=16, replays=20):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(replays):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0 / n)
    return sorted(ts)[len(ts) // 2]


def main():
    D, W = 4096, 4
    w = (torch.randn(D, W, device="cuda") * 0.5).to(torch.bfloat16)
    xp = (torch.randn(1, 8192, D, device="cuda")).to(torch.bfloat16)
    xd = (torch.randn(128, 1, D, device="cuda")).to(torch.bfloat16)
    ref = load_conv_ext()
    yp_ref = torch.empty_like(xp)
    ref.conv_forward(xp, w, yp_ref, None, None, None, None, True, None)
    variants = [("production", load_conv_ext())]
    for spec in sys.argv[1:]:
        th, un = (int(v) for v in spec.split(","))
        variants.append((f"threads={th} unroll={un}", load_conv_variant(f"cv_{th}_{un}", (f"CFBI_CONV_THREADS={th}", f"CFBI_CONV_UNROLL={un}"))))
    print(f"{'variant':>24} | prefill 8192x4096 (us) | decode N=128 (us) | N=16 | same bits")
    for name, ext in variants:
        yp = torch.empty_like(xp)
        st = torch.zeros(1, D, W - 1, dtype=torch.bfloat16, device="cuda")
        t_p = graph_us(lambda: ext.conv_forward(xp, w, yp, None, st, None, None, True, None), n=4)
        yd = torch.empty_like(xd)
        std = torch.zeros(128, D, W - 1, dtype=torch.bfloat16, device="cuda")
        t_d = graph_us(lambda: ext.conv_forward(xd, w, yd, None, std, None, None, True, None))
        xd16, yd16 = xd[:16].contiguous(), torch.empty(16, 1, D, dtype=torch.bfloat16, device="cuda")
        std16 = torch.zeros(16, D, W - 1, dtype=torch.bfloat16, device="cuda")
        t_16 = graph_us(lambda: ext.conv_forward(xd16, w, yd16, None, std16, None, None, True, None))
        yv = torch.empty_like(xp)
        ext.conv_forward(xp, w, yv, None, None, None, None, True, None)
        same = bool((yv.view(torch.int16) == yp_ref.view(torch.int16)).all())
        print(f"{name:>24} | {t_p:>22.1f} | {t_d:>17.1f} | {t_16:>4.1f} | {same}", flush=True)


if __name__ == "__main__":
    main()
