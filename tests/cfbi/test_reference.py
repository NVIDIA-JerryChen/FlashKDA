"""The CPU emulation of the contract (reference.program_emulation) is the same program as the GPU kernels.

Not bitwise (the emulation models the tensor-core mma as exact products with one rounding per accumulate and
uses true transcendentals), but it must agree with the kernels at the ~1e-6 level, and its own distance to the
fp64 recurrence tells how much of the kernels' error is inherent to the contract (three-term TF32) and how
much comes from the hardware approximations.
"""
import torch

from flash_kda.cfbi import alloc_state, prefill
from flash_kda.cfbi.reference import exact_recurrence, program_emulation

from helpers import make_case, rel_rms


def test_emulation_matches_kernel_and_exact(ext):
    torch.set_num_threads(4)
    H, T = 2, 130   # two full chunks + 2 open rows: exercises sub-blocks, chunk end, the open chunk
    case = make_case(T, H, "authors", seed=9)
    st = alloc_state(1, H)
    out = torch.zeros_like(case["q"])
    out32 = torch.zeros(case["q"].shape, dtype=torch.float32, device="cuda")
    prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], out, case["A_log"],
            case["dt_bias"], case["lower_bound"], st, out_fp32=out32)
    ref, _ = exact_recurrence(case["q"], case["k"], case["v"], case["g"], case["beta"], case["A_log"],
                              case["dt_bias"], case["scale"], case["lower_bound"])
    emu = program_emulation(case["q"][0], case["k"][0], case["v"][0], case["g"][0], case["beta"][0], case["A_log"],
                            case["dt_bias"], case["scale"], case["lower_bound"])
    kern, ref = out32[0].double().cpu(), ref.cpu()
    e_emu, e_kern, e_between = rel_rms(emu, ref), rel_rms(kern, ref), rel_rms(kern, emu)
    print(f"\n[contract] emulation vs exact {e_emu:.2e} | kernel vs exact {e_kern:.2e} | kernel vs emulation {e_between:.2e}")
    assert e_emu < 2e-6 and e_kern < 3e-6 and e_between < 3e-6
