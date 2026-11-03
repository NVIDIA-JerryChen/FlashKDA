"""M3/M4 gates: the CF-BI kernels compute the contract, and prefill == decode bit for bit.

* accuracy against the fp64 exact recurrence (guards against a "consistently wrong" program);
* the row engine (decode, one row at a time) vs K1/K2 (chunk-parallel prefill): identical output bits
  (bf16 and the fp32 values before rounding) and identical state (E, valid cache rows, pos).
"""
import pytest
import torch

from flash_kda.cfbi import alloc_state, decode, forward_train, prefill, state_equal, state_mismatch
from flash_kda.cfbi.reference import exact_recurrence

from helpers import REGIMES, make_case, n_diff, rel_rms


def run(case, state, mode=0, cu=None, B=None):
    q = case["q"]
    out = torch.zeros_like(q)
    out32 = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    prefill(q, case["k"], case["v"], case["g"], case["beta"], case["scale"], out, case["A_log"], case["dt_bias"],
            case["lower_bound"], state, cu, out_fp32=out32, _mode=mode)
    return out, out32


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("T", [64, 200])
def test_accuracy_vs_fp64_exact(ext, regime, T):
    H = 2
    case = make_case(T, H, regime, seed=1)
    state = alloc_state(1, H)
    out, out32 = run(case, state)
    ref, final = exact_recurrence(case["q"], case["k"], case["v"], case["g"], case["beta"], case["A_log"],
                                  case["dt_bias"], case["scale"], case["lower_bound"])
    err = rel_rms(out32.reshape(T, H, 128), ref)
    print(f"\n[accuracy] regime={regime:14s} T={T:4d}  rel RMS of fp32 O vs fp64 exact = {err:.3e}")
    assert err < 8e-6, f"fp32 output too far from the exact recurrence: {err:.3e}"


@pytest.mark.parametrize("regime", ["authors", "heavy_tail", "long_memory", "saturated"])
@pytest.mark.parametrize("T", [1, 15, 16, 17, 63, 64, 65, 127, 128, 200, 300])
def test_row_engine_equals_chunk_kernels(ext, regime, T):
    """The central claim: one call evaluated as chunks (K1/K2) == the same rows appended one by one."""
    H = 3
    case = make_case(T, H, regime, seed=2)
    sa, sb = alloc_state(1, H), alloc_state(1, H)
    oa, fa = run(case, sa, mode=0)   # K1 + K2 (+ tail rows into the cache)
    ob, fb = run(case, sb, mode=1)   # row engine for every row
    assert n_diff(oa, ob) == 0, f"bf16 outputs differ: {n_diff(oa, ob)} of {oa.numel()}"
    assert n_diff(fa, fb) == 0, f"fp32 outputs differ: {n_diff(fa, fb)} of {fa.numel()}"
    assert state_equal(sa, sb), f"state differs: {state_mismatch(sa, sb)}"
    assert int(sa.pos[0]) == T % 64


@pytest.mark.parametrize("T", [70, 130])
def test_decode_calls_equal_one_prefill(ext, T):
    H = 2
    case = make_case(T, H, "authors", seed=3)
    sa, sb = alloc_state(1, H), alloc_state(1, H)
    oa, fa = run(case, sa)
    ob = torch.zeros_like(case["q"])
    fb = torch.zeros_like(fa)
    for t in range(T):
        decode(case["q"][:, t:t + 1], case["k"][:, t:t + 1], case["v"][:, t:t + 1], case["g"][:, t:t + 1],
               case["beta"][:, t:t + 1], case["scale"], ob[:, t:t + 1], case["A_log"], case["dt_bias"],
               case["lower_bound"], sb, out_fp32=fb[:, t:t + 1])
    assert n_diff(oa, ob) == 0 and n_diff(fa, fb) == 0, "decode x T differs from one prefill call"
    assert state_equal(sa, sb), state_mismatch(sa, sb)


def test_training_forward_equals_prefill(ext):
    H, T = 2, 200
    case = make_case(T, H, "authors", seed=4)
    st = alloc_state(1, H)
    oa, fa = run(case, st)
    out32 = torch.zeros(case["q"].shape, dtype=torch.float32, device="cuda")
    ot = forward_train(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], case["A_log"],
                       case["dt_bias"], case["lower_bound"], out_fp32=out32)
    assert n_diff(oa, ot) == 0 and n_diff(fa, out32) == 0
