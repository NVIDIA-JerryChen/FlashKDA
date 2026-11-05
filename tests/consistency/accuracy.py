"""Diagnostic FP64 comparison; never substitutes for raw-bit acceptance."""

import argparse
import json
import torch
from .helpers import make_case, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decode", action="store_true", help="evaluate dense single-token decode"
    )
    parser.add_argument(
        "--prefill", action="store_true", help="evaluate fused recurrent prefill"
    )
    args = parser.parse_args()
    case = make_case([129], 12)
    if args.prefill:
        from flash_kda import prepare_consistent_prefill

        out = torch.empty_like(case["q"])
        state = torch.empty_like(case["state_in"])
        xs = {
            n: case[n]
            for n in ("q", "k", "v", "g", "beta", "A_log", "dt_bias", "state_in")
        }
        prepare_consistent_prefill(**xs, out=out, state_out=state).run()
    elif args.decode:
        from flash_kda import prepare_consistent_decode

        state = case["state_in"].clone()
        xs = {n: case[n][:, :1].clone() for n in ("q", "k", "v", "g", "beta")}
        token_out = torch.empty_like(xs["q"])
        plan = prepare_consistent_decode(
            **xs,
            A_log=case["A_log"],
            dt_bias=case["dt_bias"],
            state_in=state,
            state_out=state,
            out=token_out,
        )
        out = torch.empty_like(case["q"])
        for token in range(129):
            for name, tensor in xs.items():
                tensor.copy_(case[name][:, token : token + 1])
            plan.run()
            out[:, token : token + 1].copy_(token_out)
    else:
        out, state = run(case)
    assert torch.isfinite(out).all() and torch.isfinite(state).all()
    s = case["state_in"].double()[0]
    expected = []
    for t in range(129):
        q, k, v, g = (case[n][0, t].double() for n in ("q", "k", "v", "g"))
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * (128**-0.5)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        gate = torch.exp(
            -5
            * torch.sigmoid(
                case["A_log"].double()[:, None].exp() * (g + case["dt_bias"].double())
            )
        )
        s = s * gate[:, None, :]
        pred = (s * k[:, None, :]).sum(-1)
        delta = (v - pred) * case["beta"][0, t].double().sigmoid()[:, None]
        s = s + delta[:, :, None] * k[:, None, :]
        expected.append((s * q[:, None, :]).sum(-1))
        s = s.bfloat16().double()
    reference = torch.stack(expected)[None]

    def error(a, b):
        d = a.double() - b
        return dict(max_abs=d.abs().max().item(), rms=d.square().mean().sqrt().item())

    print(
        json.dumps(
            dict(
                implementation="fused prefill"
                if args.prefill
                else "dense decode"
                if args.decode
                else "generic recurrent",
                reference="FP64 arithmetic with per-token BF16 state rounding",
                output=error(out, reference),
                state=error(state, s[None]),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
