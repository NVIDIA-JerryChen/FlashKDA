"""Speed of the batch-invariant linear and the pinned short conv (correctness-first v1 kernels) against the baselines.

  linear  bi_linear (mma.sync m16n8k16, fixed K order, no split-K) vs cuBLAS (torch.matmul). Decode shapes M = 1, 16, 64, 128
          with K = N = 4096 and prefill M = 8192. Decode timings rotate through enough distinct weight matrices to exceed
          the L2 cache (the weights stream from HBM, as in real decode); "hot" repeats one matrix (weights stay in L2).
  conv    short_conv (W = 4, SiLU, bf16) vs FLA's ShortConvolution (Triton) and a PyTorch F.conv1d + silu baseline. Decode:
          N sequences x 1 token with state; prefill: B = 1, T = 8192 (varlen with 8 sequences also).

Timing: CUDA-graph replays of a pass over all the buffer copies, time per call. usage: python benchmarks/bench_conv_linear.py [--quick]
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from flash_kda.cfbi import alloc_conv_state, bi_linear, short_conv  # noqa: E402
from flash_kda.cfbi.linear import pick_config  # noqa: E402


def graph_us_per_call(fns, replays=20, warm=3):
    """Median time per call (us) of a CUDA graph that runs every callable in `fns` once."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warm):
            for f in fns:
                f()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(replays):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0 / len(fns))
    return sorted(ts)[len(ts) // 2]


def randbf(*shape, scale=1.0):
    return (torch.randn(*shape, device="cuda") * scale).to(torch.bfloat16)


def bench_linear(quick):
    K = N = 4096
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"\n== linear, K = N = {K}, bf16, B300-class GPU with {sms} SMs: time per call (us)")
    print(f"{'M':>6} {'cfg':>4} | {'bi_linear cold':>14} {'cuBLAS cold':>12} {'ratio':>6} | {'bi hot':>8} {'cuBLAS hot':>10} | "
          f"{'bi GB/s':>8} {'bi TFLOP/s':>10} {'cuBLAS TFLOP/s':>14}")
    Ms = [1, 16, 64, 128] if not quick else [1, 64]
    Ms += [8192] if not quick else []
    for M in Ms:
        wbytes = N * K * 2
        copies = 1 if M >= 1024 else max(2, int(512e6 // wbytes) + 1)
        Ws = [randbf(N, K, scale=0.05) for _ in range(copies)]
        x = randbf(M, K)
        out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        bi_cold = graph_us_per_call([lambda W=W: bi_linear(x, W, out=out) for W in Ws])
        cb_cold = graph_us_per_call([lambda W=W: torch.matmul(x, W.t()) for W in Ws])
        bi_hot = graph_us_per_call([lambda: bi_linear(x, Ws[0], out=out)] * 8)
        cb_hot = graph_us_per_call([lambda: torch.matmul(x, Ws[0].t())] * 8)
        flops = 2.0 * M * N * K
        print(f"{M:>6} {pick_config(M, N):>4} | {bi_cold:>14.1f} {cb_cold:>12.1f} {bi_cold / cb_cold:>6.2f} | {bi_hot:>8.1f} {cb_hot:>10.1f} | "
              f"{wbytes / bi_cold / 1e3:>8.0f} {flops / bi_cold / 1e6:>10.1f} {flops / cb_cold / 1e6:>14.1f}", flush=True)
        del Ws, x, out
        torch.cuda.empty_cache()
    # the small projections of the KDA layer
    print("\n   other projection shapes of a KDA layer (hidden 4096, 16 heads), M = 1 / 128 / 8192, us per call (hot)")
    for (K2, N2, name) in [(4096, 2048, "q/k/v_proj"), (4096, 128, "f_proj[0]"), (128, 2048, "f_proj[1]"), (4096, 16, "b_proj"), (2048, 4096, "o_proj")]:
        W = randbf(N2, K2, scale=0.05)
        row = []
        for M in (1, 128, 8192):
            x = randbf(M, K2)
            out = torch.empty(M, N2, dtype=torch.bfloat16, device="cuda")
            bi = graph_us_per_call([lambda: bi_linear(x, W, out=out)] * 8)
            cb = graph_us_per_call([lambda: torch.matmul(x, W.t())] * 8)
            row.append(f"M={M}: bi {bi:7.1f} / cuBLAS {cb:7.1f}")
        print(f"   {name:>11} K={K2:<5} N={N2:<5} " + " | ".join(row), flush=True)


def bench_conv(quick):
    D, W_ = 4096, 4
    print(f"\n== short conv, D = {D}, W = {W_}, SiLU, bf16: time per call (us)")
    w = (torch.randn(D, W_, device="cuda") * 0.5).to(torch.bfloat16)
    try:
        from fla.modules import ShortConvolution

        fla_conv = ShortConvolution(D, W_, bias=False, activation="silu", backend="triton", device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            fla_conv.weight.copy_(w.view(D, 1, W_))
    except Exception as e:   # pragma: no cover
        fla_conv = None
        print(f"   (FLA ShortConvolution not available: {type(e).__name__})")
    print(f"{'case':>26} | {'short_conv':>10} {'FLA triton':>11} {'torch conv1d':>13} | {'cfbi GB/s':>9}")
    for N in ([1, 16, 64, 128] if not quick else [1, 64]):
        x = randbf(N, 1, D)
        st = alloc_conv_state(N, D, W_)
        out = torch.empty_like(x)
        t_cfbi = graph_us_per_call([lambda: short_conv(x, w, None, st, "silu", out=out)] * 16)
        t_fla = None
        if fla_conv is not None:
            cache = torch.zeros(N, D, W_, dtype=torch.bfloat16, device="cuda")

            def f():
                with torch.no_grad():
                    fla_conv(x, cache=cache, output_final_state=True)

            t_fla = graph_us_per_call([f] * 16)
        bytes_ = N * D * 2 * 2 + N * D * 3 * 2 * 2 + D * W_ * 2
        print(f"{'decode N=' + str(N):>26} | {t_cfbi:>10.1f} {'n/a' if t_fla is None else f'{t_fla:.1f}':>11} {'-':>13} | {bytes_ / t_cfbi / 1e3:>9.0f}", flush=True)
    T = 8192
    for name, B, cu in [("prefill B=1 T=8192", 1, None), ("prefill B=8 T=1024", 8, None), ("varlen 8 x 1024", 1, "cu")]:
        if cu is not None:
            lens = [1024] * 8
            cu_t = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int64, device="cuda")
            x = randbf(1, sum(lens), D)
            n_seq = 8
        else:
            x = randbf(B, T // B, D)
            cu_t, n_seq = None, B
        st = alloc_conv_state(n_seq, D, W_)
        out = torch.empty_like(x)
        t_cfbi = graph_us_per_call([lambda: short_conv(x, w, None, st, "silu", cu_t, out=out)] * 4)
        t_tr = graph_us_per_call([lambda: short_conv(x, w, None, None, "silu", cu_t, out=out)] * 4)   # training forward (no state)
        t_fla = None
        if fla_conv is not None:
            def f():
                with torch.no_grad():
                    fla_conv(x, cu_seqlens=cu_t)

            try:
                t_fla = graph_us_per_call([f] * 4)
            except Exception:
                t_fla = None
        t_tc = None
        if cu is None:
            def tc():
                xt = x.transpose(1, 2)
                y = F.conv1d(F.pad(xt, (W_ - 1, 0)), w.view(D, 1, W_), groups=D)
                return F.silu(y).transpose(1, 2)

            t_tc = graph_us_per_call([tc] * 4)
        bytes_ = x.numel() * 2 * 2
        print(f"{name:>26} | {t_cfbi:>10.1f} {'n/a' if t_fla is None else f'{t_fla:.1f}':>11} {'n/a' if t_tc is None else f'{t_tc:.1f}':>13} | "
              f"{bytes_ / t_cfbi / 1e3:>9.0f}   (stateless training forward: {t_tr:.1f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["linear", "conv"], default=None)
    a = ap.parse_args()
    print(f"device: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    if a.only != "conv":
        bench_linear(a.quick)
    if a.only != "linear":
        bench_conv(a.quick)


if __name__ == "__main__":
    main()
