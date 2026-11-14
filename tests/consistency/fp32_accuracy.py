"""Independent FP64 diagnostics for FP32-state forward; no inferred error gate."""

import argparse
import json
import torch
from .fp32_math_reference import run as reference
from .test_fp32 import case, prefill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument(
        "--gate", choices=["normal", "open", "saturated"], default="normal"
    )
    args = parser.parse_args()
    if args.length < 1 or args.heads < 1:
        parser.error("length and heads must be positive")
    x = case([args.length], h=args.heads)
    if args.gate != "normal":
        x["g"].fill_(-20 if args.gate == "open" else 20)
        x["A_log"].zero_()
        x["dt_bias"].zero_()
    out, state, _ = prefill(x)
    ref = reference(
        x["q"],
        x["k"],
        x["v"],
        x["g"],
        x["beta"],
        x["A_log"],
        x["dt_bias"],
        x["state_in"],
        x["cu_seqlens"],
        full_precision=True,
    )
    result = {}
    for name, value in [("output", out), ("state_out", state)]:
        want = ref[name]
        assert torch.isfinite(value).all() and torch.isfinite(want).all()
        diff = value.double() - want
        result[name] = dict(
            nrmse=float(
                diff.square().mean().sqrt()
                / want.square().mean().sqrt().clamp_min(1e-300)
            ),
            max_normalized=float(diff.abs().max() / want.abs().max().clamp_min(1e-300)),
        )
    print(
        json.dumps(
            dict(length=args.length, heads=args.heads, gate=args.gate, metrics=result),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
