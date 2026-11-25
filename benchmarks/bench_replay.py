"""GPU-time benchmark of flash_kda.replay (bitwise-aligned decode / prefill) next to the plain kernels.

Decode (one token per sequence per step; N sequences, H heads; each number = GPU time per step from CUDA-graph
replays, averaged over 160 consecutive steps so every chunk row 0..15 and the chunk-closing step are included):
  replay.decode       this branch: bitwise equal to one flash_kda.fwd call over the whole sequence
  fwd T=1 (not bitwise)  flash_kda.fwd called with one token per sequence and a bf16 state: the original kernel used
                      naively for decode (closes a one-row chunk every step, so it does NOT match the prefill)
  FLA recurrent fp32  fla fused_recurrent_kda with an fp32 state (bitwise invariant across splits, but a different
                      arithmetic program from flash_kda.fwd), if fla is importable

Prefill (fresh state): flash_kda.fwd (one call) vs replay.prefill (same fwd call on the 16-aligned bodies + the replay
kernel for the tails + bookkeeping). Lengths with a tail (L % 16 != 0) show the tail cost.

    PYTHONPATH=. python benchmarks/bench_replay.py [--quick] [--json out.json]
"""
import argparse
import json
import math

import torch

import flash_kda
from flash_kda import replay as R

D = 128
dev = "cuda"


def graph_time_us(fn, iters, warmup=3):
    """Capture fn once in a CUDA graph and return the mean GPU time per replay in microseconds."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / iters


def params(H, seed=0):
    gen = torch.Generator(device=dev).manual_seed(seed)
    return dict(scale=1.0 / math.sqrt(D), A_log=torch.rand(H, generator=gen, device=dev),
                dt_bias=torch.rand(H, D, generator=gen, device=dev), lower_bound=-5.0)


def tokens(T, H, seed=0):
    gen = torch.Generator(device=dev).manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen, device=dev).to(torch.bfloat16)  # noqa: E731
    return dict(q=r(T, H, D), k=r(T, H, D), v=r(T, H, D), g=r(T, H, D), beta=r(T, H))


def bench_decode(N, H, steps=160):
    p, x = params(H), tokens(N, H, seed=N * 131 + H)
    res = {"N": N, "H": H}
    # replay.decode: state with sequences spread over all chunk rows
    st = R.alloc_state(N, H)
    st.pos.copy_((torch.arange(N, device=dev, dtype=torch.int32) % R.CHUNK)[:, None].expand(N, H))
    st.S.normal_().mul_(0.05)
    for t in (st.rows_k, st.rows_v, st.rows_g):
        t.normal_()
    st.rows_beta.normal_()
    out = torch.empty_like(x["q"])
    res["replay_us"] = graph_time_us(lambda: R.decode(x["q"], x["k"], x["v"], x["g"], x["beta"], p["scale"], out,
                                                      p["A_log"], p["dt_bias"], p["lower_bound"], st), steps)
    # flash_kda.fwd with one token per sequence (reads S, writes S2; not bitwise with the prefill)
    S = torch.randn(N, H, D, D, device=dev).mul_(0.05).to(torch.bfloat16)
    S2 = torch.empty_like(S)
    o4 = torch.empty(N, 1, H, D, dtype=torch.bfloat16, device=dev)
    xq = {n: t.unsqueeze(1) for n, t in x.items()}
    res["fwd_T1_us"] = graph_time_us(lambda: flash_kda.fwd(xq["q"], xq["k"], xq["v"], xq["g"], xq["beta"],
                                                           p["scale"], o4, p["A_log"], p["dt_bias"],
                                                           p["lower_bound"], initial_state=S, final_state=S2), steps)
    try:
        from fla.ops.kda import fused_recurrent_kda
        h = torch.randn(N, H, D, D, device=dev).mul_(0.05)
        qf, kf, vf = (t.unsqueeze(0).reshape(1, N, H, D) for t in (x["q"], x["k"], x["v"]))
        gf = (-torch.rand(1, N, H, D, device=dev)).float()
        bf = torch.sigmoid(x["beta"].float()).reshape(1, N, H).to(torch.bfloat16)
        cu = torch.arange(N + 1, device=dev, dtype=torch.int64)

        def fla_step():
            # one token per sequence, fp32 state; g / beta are activated outside the timed call (the kernel gets
            # them ready-made), so this column slightly favours FLA
            fused_recurrent_kda(q=qf, k=kf, v=vf, g=gf, beta=bf, scale=p["scale"], initial_state=h,
                                output_final_state=True, inplace_final_state=True, cu_seqlens=cu,
                                use_qk_l2norm_in_kernel=True)
        res["fla_fp32_us"] = graph_time_us(fla_step, steps)
    except Exception as e:  # fla missing or API mismatch: report it, keep the rest
        res["fla_fp32_us"] = None
        res["fla_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return res


def bench_prefill(lens, H, iters=20):
    p = params(H)
    T = sum(lens)
    x = tokens(T, H, seed=T + H)
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int64, device=dev)
    N = len(lens)
    out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device=dev)
    fin = torch.empty(N, H, D, D, dtype=torch.bfloat16, device=dev)
    x4 = {n: t[None] for n, t in x.items()}
    res = {"lens": lens if len(lens) <= 8 else f"{len(lens)} seqs, {T} tokens", "H": H}
    res["fwd_us"] = graph_time_us(lambda: flash_kda.fwd(x4["q"], x4["k"], x4["v"], x4["g"], x4["beta"], p["scale"],
                                                        out, p["A_log"], p["dt_bias"], p["lower_bound"],
                                                        final_state=fin, cu_seqlens=cu), iters)
    st = R.alloc_state(N, H)

    def rp():
        # pos = 0: every replay does a fresh prompt's work (no head); S carries over between replays, but no kernel
        # branches on state values, so the time is that of a fresh-state prefill (the zero_ is inside the timing)
        st.pos.zero_()
        R.prefill(x4["q"], x4["k"], x4["v"], x4["g"], x4["beta"], p["scale"], out, p["A_log"], p["dt_bias"],
                  p["lower_bound"], st, cu_seqlens=cu)
    res["replay_prefill_us"] = graph_time_us(rp, iters)
    res["ratio"] = res["replay_prefill_us"] / res["fwd_us"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"device: {torch.cuda.get_device_name()}  replay smem/CTA: {flash_kda.replay._C.replay_smem_bytes()} B")
    dec_cases = [(1, 12), (8, 12), (64, 12)] if args.quick else \
        [(1, 12), (8, 12), (16, 12), (32, 12), (64, 12), (128, 12), (1, 16), (16, 16), (64, 16), (128, 16)]
    pre_cases = [([8192], 6), ([8199], 6)] if args.quick else [
        ([8192], 6), ([8199], 6), ([8192] * 8, 6), ([8199] * 8, 6), ([512] * 64, 12), ([519] * 64, 12),
        ([1300, 547, 2048, 963, 271, 3063], 12), ([32768], 32), ([196608], 12)]
    results = {"decode": [], "prefill": []}
    print("\nDecode, GPU us per step (N sequences x H heads, 1 token each)")
    print(f"{'N':>4} {'H':>3} | {'replay':>8} {'fwd T=1*':>9} {'FLA fp32':>9}")
    for N, H in dec_cases:
        r = bench_decode(N, H)
        results["decode"].append(r)
        fla = f"{r['fla_fp32_us']:9.2f}" if r["fla_fp32_us"] is not None else f"{'-':>9}"
        print(f"{N:>4} {H:>3} | {r['replay_us']:8.2f} {r['fwd_T1_us']:9.2f} {fla}")
        if r.get("fla_error"):
            print("      FLA:", r["fla_error"])
    print("  * fwd T=1 is not bitwise aligned with the prefill (it closes a one-row chunk every step)")
    print("\nPrefill from an empty state, GPU us (replay.prefill = fwd on 16-aligned bodies + replay of tails)")
    print(f"{'lens':>40} {'H':>3} | {'fwd':>9} {'replay':>9} {'ratio':>6}")
    for lens, H in pre_cases:
        r = bench_prefill(lens, H)
        results["prefill"].append(r)
        print(f"{str(r['lens']):>40} {H:>3} | {r['fwd_us']:9.1f} {r['replay_prefill_us']:9.1f} {r['ratio']:6.3f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
