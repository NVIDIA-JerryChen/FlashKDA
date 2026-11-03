"""Speed of the CF-BI kernels (correctness-first v1: mma.sync TF32, global fp32 workspace) against the baselines.

  prefill / train fwd   cfbi.prefill (state in/out) and cfbi.forward_train, vs flash_kda.fwd (if flash_kda_C is
                        built) and FLA chunk_kda (no_grad forward). Shapes from the CF-BI report for orientation
                        only (H6 / L8K, batch 1..64).
  decode                cfbi.decode (CUDA graph, mean over >= 128 steps = all chunk positions) vs FLA
                        fused_recurrent_kda with an fp32 state (eager), N sequences x 1 token, H12.
  train fwd+bwd         cfbi.chunk_kda vs FLA chunk_kda with autograd.

usage: python benchmarks/bench_cfbi.py [--quick] [--json out.json]
Median of CUDA-event timings after warm-up (Triton autotune happens in the warm-up).
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("FLA_FLASH_KDA", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "cfbi"))

import torch

import flash_kda
from flash_kda.cfbi import alloc_state, chunk_kda, decode, forward_train, prefill
from helpers import make_case

LB = -5.0


def median_ms(fn, warm=3, reps=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return sorted(ts)[len(ts) // 2]


def graph_ms_per_call(fn, iters=128, warm=3):
    """Mean GPU time of one call, from CUDA-graph replays (no Python/launch overhead)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warm):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def tolerant(fn):
    try:
        return fn()
    except Exception as ex:  # a baseline that cannot run must not kill the table
        return f"n/a ({type(ex).__name__})"


def fmt(x):
    return f"{x:9.3f}" if isinstance(x, float) else f"{str(x):>9s}"


def bench_prefill(B, L, H, results):
    case = make_case(B * L, H, "authors", seed=0)
    for key in ("q", "k", "v", "g"):
        case[key] = case[key].reshape(B, L, H, 128).contiguous()
    case["beta"] = case["beta"].reshape(B, L, H).contiguous()
    q, k, v, g, beta = case["q"], case["k"], case["v"], case["g"], case["beta"]
    A_log, dt_bias, scale = case["A_log"], case["dt_bias"], case["scale"]
    out = torch.zeros_like(q)
    st = alloc_state(B, H)
    row = {"B": B, "L": L, "H": H}
    row["cfbi_prefill"] = median_ms(lambda: prefill(q, k, v, g, beta, scale, out, A_log, dt_bias, LB, st))
    row["cfbi_train_fwd"] = median_ms(lambda: forward_train(q, k, v, g, beta, scale, A_log, dt_bias, LB, out=out))
    if flash_kda._fwd_raw is not None:
        row["flashkda_fwd"] = tolerant(lambda: median_ms(lambda: flash_kda.fwd(q, k, v, g, beta, scale, out, A_log=A_log, dt_bias=dt_bias, lower_bound=LB)))
    else:
        row["flashkda_fwd"] = "n/a"
    try:
        from fla.ops.kda import chunk_kda as fla_chunk

        def run_fla():
            with torch.no_grad():
                fla_chunk(q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias.reshape(-1), scale=scale,
                          use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                          safe_gate=True, lower_bound=LB)

        row["fla_chunk"] = tolerant(lambda: median_ms(run_fla))
    except ImportError:
        row["fla_chunk"] = "n/a"
    results["prefill"].append(row)
    print(f"prefill B={B:3d} L={L} H={H}: " + "  ".join(f"{k}={fmt(v)}" for k, v in row.items() if k not in ("B", "L", "H")) + "  (ms)", flush=True)
    del case, q, k, v, g, beta, out, st
    torch.cuda.empty_cache()


def bench_decode(N, H, results):
    case = make_case(N, H, "authors", seed=0)
    q = case["q"].reshape(N, 1, H, 128).contiguous()
    k = case["k"].reshape(N, 1, H, 128).contiguous()
    v = case["v"].reshape(N, 1, H, 128).contiguous()
    g = case["g"].reshape(N, 1, H, 128).contiguous()
    beta = case["beta"].reshape(N, 1, H).contiguous()
    A_log, dt_bias, scale = case["A_log"], case["dt_bias"], case["scale"]
    out = torch.zeros_like(q)
    st = alloc_state(N, H)
    row = {"N": N, "H": H}
    row["cfbi_decode_graph"] = graph_ms_per_call(lambda: decode(q, k, v, g, beta, scale, out, A_log, dt_bias, LB, st), iters=256) * 1000
    row["cfbi_decode_eager"] = median_ms(lambda: decode(q, k, v, g, beta, scale, out, A_log, dt_bias, LB, st), reps=50) * 1000
    try:
        from fla.ops.kda import fused_recurrent_kda

        S0 = torch.zeros(N, H, 128, 128, dtype=torch.float32, device="cuda")

        def run_rec():
            with torch.no_grad():
                fused_recurrent_kda(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias.reshape(-1), scale=scale,
                                    initial_state=S0, output_final_state=True, use_qk_l2norm_in_kernel=True,
                                    use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, lower_bound=LB,
                                    state_v_first=True)

        row["fla_recurrent_fp32_eager"] = tolerant(lambda: median_ms(run_rec, reps=50) * 1000)
    except ImportError:
        row["fla_recurrent_fp32_eager"] = "n/a"
    results["decode"].append(row)
    print(f"decode N={N:4d} H={H}: " + "  ".join(f"{k}={fmt(v)}" for k, v in row.items() if k not in ("N", "H")) + "  (us per call)", flush=True)


def bench_train(B, L, H, results):
    case = make_case(B * L, H, "authors", seed=0)
    ins = {key: case[key].reshape(B, L, H, 128).contiguous() for key in ("q", "k", "v", "g")}
    ins["beta"] = case["beta"].reshape(B, L, H).contiguous()
    do = torch.randn(B, L, H, 128, device="cuda").to(torch.bfloat16)
    A_log, dt_bias, scale = case["A_log"], case["dt_bias"], case["scale"]

    def run_cfbi():
        xs = {k_: x.clone().requires_grad_() for k_, x in ins.items()}
        chunk_kda(xs["q"], xs["k"], xs["v"], xs["g"], xs["beta"], scale, A_log.clone().requires_grad_(),
                  dt_bias.clone().requires_grad_(), LB).backward(do)

    row = {"B": B, "L": L, "H": H, "cfbi_fwd_bwd": median_ms(run_cfbi, reps=5)}
    try:
        from fla.ops.kda import chunk_kda as fla_chunk

        def run_fla():
            xs = {k_: x.clone().requires_grad_() for k_, x in ins.items()}
            a, d = A_log.clone().requires_grad_(), dt_bias.reshape(-1).clone().requires_grad_()
            o, _ = fla_chunk(xs["q"], xs["k"], xs["v"], xs["g"], xs["beta"], A_log=a, dt_bias=d, scale=scale,
                             use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                             safe_gate=True, lower_bound=LB)
            o.backward(do)

        row["fla_fwd_bwd"] = tolerant(lambda: median_ms(run_fla, reps=5))
    except ImportError:
        row["fla_fwd_bwd"] = "n/a"
    results["train"].append(row)
    print(f"train  B={B:3d} L={L} H={H}: cfbi_fwd_bwd={fmt(row['cfbi_fwd_bwd'])} fla_fwd_bwd={fmt(row['fla_fwd_bwd'])} (ms)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    dev = torch.cuda.get_device_name()
    print(f"device: {dev}   flash_kda_C: {'built' if flash_kda._fwd_raw is not None else 'not built'}")
    results = {"device": dev, "prefill": [], "decode": [], "train": []}
    prefill_cases = [(1, 8192, 6), (8, 8192, 6)] if args.quick else [(1, 8192, 6), (8, 8192, 6), (64, 8192, 6), (1, 4096, 32), (1, 32768, 32)]
    for B, L, H in prefill_cases:
        bench_prefill(B, L, H, results)
    for N in ([16, 64] if args.quick else [1, 16, 64, 128]):
        bench_decode(N, 12, results)
    for B, L, H in ([(1, 8192, 6)] if args.quick else [(1, 8192, 6), (8, 4096, 6)]):
        bench_train(B, L, H, results)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
