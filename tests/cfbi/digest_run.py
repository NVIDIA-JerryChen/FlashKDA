"""Prints a SHA-256 digest of a fixed CF-BI workload (outputs, fp32 outputs, state, training forward).

Run in two separate processes (tests/cfbi/test_bitwise_matrix.py::test_cross_process_digest does): the digests
must be identical (no dependence on process state, allocator addresses, autotuning, stream timing).

Digests are comparable on one GPU model only: the inputs come from ``torch.randn`` on the GPU, which is not
bit-identical across GPU models (measured: the L40S and the H100, the A10 and the A100 produce different q/k/v/g from the
same seed; A100 and L40S happen to agree). To compare architectures use arch_dump.py / arch_compare.py, which feed every
machine the same CPU-generated inputs.
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from flash_kda.cfbi import alloc_state, forward_train, prefill
from helpers import make_case


def main():
    H, lens = 3, [150, 1, 64, 129, 33]
    case = make_case(sum(lens), H, "heavy_tail", seed=99)
    cu = [0]
    for L in lens:
        cu.append(cu[-1] + L)
    N = len(lens)
    st = alloc_state(N, H)
    h = hashlib.sha256()
    # three calls with different cuts: a schedule that exercises row engine, K1/K2 and tails
    done = [0] * N
    for cut in (0.37, 0.71, 1.0):
        take = [int(L * cut) - d for L, d in zip(lens, done)]
        pieces = []
        for n in range(N):
            lo = cu[n] + done[n]
            pieces.append({k: case[k][:, lo:lo + take[n]] for k in ("q", "k", "v", "g", "beta")})
        sub = {k: torch.cat([p[k] for p in pieces], dim=1).contiguous() for k in ("q", "k", "v", "g", "beta")}
        cu_call = [0]
        for t in take:
            cu_call.append(cu_call[-1] + t)
        out = torch.zeros_like(sub["q"])
        out32 = torch.zeros(sub["q"].shape, dtype=torch.float32, device="cuda")
        prefill(sub["q"], sub["k"], sub["v"], sub["g"], sub["beta"], case["scale"], out, case["A_log"],
                case["dt_bias"], case["lower_bound"], st, torch.tensor(cu_call, dtype=torch.int64, device="cuda"),
                out_fp32=out32)
        h.update(out.view(torch.int16).cpu().numpy().tobytes())
        h.update(out32.view(torch.int32).cpu().numpy().tobytes())
        done = [d + t for d, t in zip(done, take)]
    valid = (torch.arange(64, device="cuda")[None, :] < st.pos[:, None])[:, None, None, :, None]
    h.update(st.E.view(torch.int32).cpu().numpy().tobytes())
    h.update((st.rows * valid).view(torch.int32).cpu().numpy().tobytes())
    h.update(st.pos.cpu().numpy().tobytes())
    ot = forward_train(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], case["A_log"],
                       case["dt_bias"], case["lower_bound"], torch.tensor(cu, dtype=torch.int64, device="cuda"))
    h.update(ot.view(torch.int16).cpu().numpy().tobytes())
    print("DIGEST", h.hexdigest())


if __name__ == "__main__":
    main()
