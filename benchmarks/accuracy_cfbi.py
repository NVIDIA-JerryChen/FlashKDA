"""Accuracy of the CF-BI program against the exact math, on 8 data regimes.

Reference: the per-token KDA recurrence in fp64 (``flash_kda.cfbi.reference.exact_recurrence``).
Columns (relative RMS error against the exact output; "flip" = share of bf16 outputs that differ from the
bf16-rounded exact output):
  cfbi fp32   the fp32 output before the final bf16 rounding (what the program computes)
  cfbi bf16   the bf16 output;  floor = the error of rounding the exact output itself to bf16
  fla         FLA chunk_kda forward (Triton, bf16 intermediates), same raw inputs
  flashkda    flash_kda.fwd (CUTLASS chunk-16 kernel, bf16 state), if flash_kda_C is built

usage: python benchmarks/accuracy_cfbi.py [--T 2048] [--H 4]
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("FLA_FLASH_KDA", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "cfbi"))

import torch

import flash_kda
from flash_kda.cfbi import alloc_state, prefill
from flash_kda.cfbi.reference import exact_recurrence
from helpers import REGIMES, make_case, rel_rms


def flip_rate(x_bf16, exact):
    return float((x_bf16 != exact.to(torch.bfloat16)).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--H", type=int, default=4)
    args = ap.parse_args()
    try:
        from fla.ops.kda import chunk_kda as fla_chunk_kda
    except ImportError:
        fla_chunk_kda = None
    have_fk = flash_kda._fwd_raw is not None
    T, H = args.T, args.H
    print(f"T={T} H={H} K=V=128   (relative RMS vs fp64 exact; flip = bf16 outputs != bf16(exact))")
    hdr = f"{'regime':14s} | {'cfbi fp32':>10s} | {'cfbi bf16':>10s} {'floor':>9s} {'flip':>7s} | {'fla':>9s} {'flip':>7s} | {'flashkda':>9s} {'flip':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for regime in REGIMES:
        case = make_case(T, H, regime, seed=1)
        ref, _ = exact_recurrence(case["q"], case["k"], case["v"], case["g"], case["beta"], case["A_log"],
                                  case["dt_bias"], case["scale"], case["lower_bound"])
        ref = ref.reshape(case["q"].shape)
        st = alloc_state(1, H)
        out = torch.zeros_like(case["q"])
        out32 = torch.zeros(case["q"].shape, dtype=torch.float32, device="cuda")
        prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], out, case["A_log"],
                case["dt_bias"], case["lower_bound"], st, out_fp32=out32)
        row = f"{regime:14s} | {rel_rms(out32, ref):10.2e} | {rel_rms(out, ref):10.2e} {rel_rms(ref.to(torch.bfloat16), ref):9.2e} {flip_rate(out, ref):7.3%} |"
        if fla_chunk_kda is not None and case["lower_bound"] < 0:
            o, _ = fla_chunk_kda(case["q"], case["k"], case["v"], case["g"], case["beta"], A_log=case["A_log"],
                                 dt_bias=case["dt_bias"].reshape(-1), scale=case["scale"],
                                 use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                                 use_beta_sigmoid_in_kernel=True, safe_gate=True, lower_bound=case["lower_bound"])
            row += f" {rel_rms(o, ref):9.2e} {flip_rate(o, ref):7.3%} |"
        else:
            row += f" {'n/a':>9s} {'':>7s} |"
        if have_fk:
            o = torch.zeros_like(case["q"])
            flash_kda.fwd(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], o,
                          A_log=case["A_log"], dt_bias=case["dt_bias"], lower_bound=case["lower_bound"])
            row += f" {rel_rms(o, ref):9.2e} {flip_rate(o, ref):7.3%}"
        else:
            row += f" {'n/a':>9s}"
        print(row, flush=True)


if __name__ == "__main__":
    main()
