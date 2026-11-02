"""Diagnostic FP64 comparison; never substitutes for raw-bit acceptance."""
import json
import torch
from .helpers import make_case, run


def main():
    case = make_case([129], 12)
    out, state = run(case)
    s = case["state_in"].double()[0]
    expected = []
    for t in range(129):
        q, k, v, g = (case[n][0, t].double() for n in ("q", "k", "v", "g"))
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * (128 ** -0.5)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        gate = torch.exp(-5 * torch.sigmoid(case["A_log"].double()[:, None].exp()
                                         * (g + case["dt_bias"].double())))
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
    print(json.dumps(dict(reference="FP64 arithmetic with per-token BF16 state rounding",
                          output=error(out, reference), state=error(state, s[None])), indent=2))


if __name__ == "__main__":
    main()
