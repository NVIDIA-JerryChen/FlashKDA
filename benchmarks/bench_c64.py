"""Quick timing of the chunk-64 kernels against the chunk-16 ones (CUDA events, GPU time of the complete calls).

usage: python benchmarks/bench_c64.py [prefill|decode|all]
prefill: flash_kda.fwd (chunk 16) vs flash_kda.replay64.fwd64 (chunk 64) vs replay64.prefill (host-free device plan)
decode : flash_kda.replay (chunk 16) vs flash_kda.replay64 one token per sequence, lockstep (all rows at the same position)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import c64_helpers as Hh  # noqa: E402
import flash_kda  # noqa: E402
from flash_kda import replay as R16, replay64 as R64  # noqa: E402

dev = "cuda"
D = 128
CFGS = [int(c) for c in os.environ.get("C64_CFGS", "0,1,2,3").split(",")]
AUTO = os.environ.get("C64_CFGS") is None


def timeit(fn, iters=30, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1000.0   # us


def params(H, seed=0):
    gen = torch.Generator(device=dev).manual_seed(seed)
    return (torch.rand(H, generator=gen, device=dev), torch.rand(H, D, generator=gen, device=dev), -5.0)


def tokens(T, H):
    r = lambda *s: torch.randn(*s, device=dev).to(torch.bfloat16)  # noqa: E731
    return r(T, H, D), r(T, H, D), r(T, H, D), r(T, H, D), r(T, H)


def prefill_case(name, lens, H):
    A_log, dt_bias, lb = params(H)
    T = sum(lens)
    q, k, v, g, beta = tokens(T, H)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int64, device=dev)
    N = len(lens)
    scale = D ** -0.5
    out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device=dev)
    st = torch.zeros(N, H, D, D, dtype=torch.bfloat16, device=dev)
    args = (q[None], k[None], v[None], g[None], beta[None], scale, out, A_log, dt_bias, lb)
    t_orig = timeit(lambda: flash_kda.fwd(*args, initial_state=None, final_state=st, cu_seqlens=cu))
    ts = {c: timeit(lambda c=c: R64.fwd64(*args, initial_state=None, final_state=st, cu_seqlens=cu, _k2_cfg=c)) for c in CFGS}
    s64 = R64.alloc_state(N, H)
    t_auto = timeit(lambda: R64.fwd64(*args, initial_state=None, final_state=st, cu_seqlens=cu))
    t_pf = timeit(lambda: R64.prefill(*args, s64, cu_seqlens=cu))
    print(f"[prefill] {name:24s} H{H}  fwd(C16) {t_orig:9.1f} us   fwd64 " + "  ".join(f"cfg{c}: {t:8.1f} ({t / t_orig:4.2f}x)" for c, t in ts.items()) +
          f"   auto: fwd64 {t_auto:8.1f} ({t_auto / t_orig:4.2f}x)  replay64.prefill {t_pf:9.1f} us ({t_pf / t_orig:5.2f}x)", flush=True)


def decode_case(N, H, steps=64):
    A_log, dt_bias, lb = params(H)
    scale = D ** -0.5
    xs = [tokens(N, H) for _ in range(4)]
    out = torch.empty(N, H, D, dtype=torch.bfloat16, device=dev)

    s16 = R16.alloc_state(N, H)
    s64 = R64.alloc_state(N, H)

    def step16(i=[0]):
        q, k, v, g, b = xs[i[0] % 4]
        i[0] += 1
        R16.decode(q, k, v, g, b, scale, out, A_log, dt_bias, lb, s16)

    def step64(i=[0]):
        q, k, v, g, b = xs[i[0] % 4]
        i[0] += 1
        R64.decode(q, k, v, g, b, scale, out, A_log, dt_bias, lb, s64)

    t16 = timeit(step16, iters=steps, warm=2)
    t64 = timeit(step64, iters=steps, warm=2)
    print(f"[decode ] N={N:4d} H{H}  replay(C16) {t16:8.1f} us/step   replay64 {t64:8.1f} us/step  ({t64 / t16:5.2f}x)   "
          f"(64 steps from an empty state: positions 0..63)", flush=True)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("prefill", "all"):
        for H in (12,):
            prefill_case("1x4096", [4096], H)
            prefill_case("8x8192", [8192] * 8, H)
            prefill_case("16x2048", [2048] * 16, H)
            prefill_case("64x512", [512] * 64, H)
            prefill_case("1x32768", [32768], H)
            prefill_case("1x196608", [196608], H)
    if what in ("decode", "all"):
        for N in (1, 16, 64, 128):
            decode_case(N, 12)


if __name__ == "__main__":
    main()
