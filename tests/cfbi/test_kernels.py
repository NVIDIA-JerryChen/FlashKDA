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
    assert err < 3e-6, f"fp32 output too far from the exact recurrence: {err:.3e}"


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


def test_arch_guard(monkeypatch):
    """Premise validated on compute capability >= 10 only: older devices need an explicit opt-in."""
    from flash_kda.cfbi import api

    monkeypatch.setattr(api, "_ARCH_OK", {})
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    monkeypatch.delenv("FLASH_KDA_CFBI_ALLOW_UNVALIDATED_ARCH", raising=False)
    with pytest.raises(RuntimeError, match="validated"):
        api._guard(torch.device("cuda", 0))
    monkeypatch.setenv("FLASH_KDA_CFBI_ALLOW_UNVALIDATED_ARCH", "1")
    monkeypatch.setattr(api, "_ARCH_OK", {})
    api._guard(torch.device("cuda", 0))


def test_empty_call_leaves_state_alone(ext):
    H = 2
    case = make_case(70, H, "authors", seed=5)
    st = alloc_state(2, H)
    run(case, st, cu=torch.tensor([0, 70, 70], dtype=torch.int64, device="cuda"))   # sequence 1 has no tokens
    before = st.clone()
    empty = {k: case[k][:, :0].contiguous() for k in ("q", "k", "v", "g", "beta")}
    empty.update({k: case[k] for k in ("A_log", "dt_bias", "scale", "lower_bound")})
    run(empty, st, cu=torch.tensor([0, 0, 0], dtype=torch.int64, device="cuda"))
    assert state_equal(st, before)
    assert state_equal(st.select([0]), before.select([0])) and int(st.pos[1]) == 0


def test_bad_arguments_are_rejected(ext):
    case = make_case(64, 2, "authors", seed=6)
    st = alloc_state(1, 2)
    out = torch.zeros_like(case["q"])
    bad = dict(case)
    bad["q"] = case["q"].float()
    with pytest.raises(RuntimeError, match="bfloat16"):
        prefill(bad["q"], case["k"], case["v"], case["g"], case["beta"], 0.1, out, case["A_log"], case["dt_bias"], -5.0, st)
    with pytest.raises(RuntimeError, match="lower_bound"):
        prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], 0.1, out, case["A_log"], case["dt_bias"], 1.0, st)
    with pytest.raises(RuntimeError, match="must be"):   # state tensors for the wrong number of sequences
        prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], 0.1, out, case["A_log"], case["dt_bias"], -5.0, alloc_state(3, 2))


def test_misaligned_inputs_are_rejected(ext):
    """16-byte vector accesses: a bf16 view that starts 2 bytes off a 16-byte boundary must not run."""
    H, T = 2, 64
    case = make_case(T, H, "authors", seed=7)
    buf = torch.zeros(T * H * 128 + 8, dtype=torch.bfloat16, device="cuda")
    q = buf[1:1 + T * H * 128].view(1, T, H, 128)
    q.copy_(case["q"])
    out = torch.zeros_like(case["q"])
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        prefill(q, case["k"], case["v"], case["g"], case["beta"], 0.1, out, case["A_log"], case["dt_bias"], -5.0,
                alloc_state(1, H))


@pytest.mark.parametrize("H", [33, 96])
def test_many_heads(ext, H):
    """Large head counts (grid.y / grid.z, per-head addressing): K1/K2 == row engine, batch == solo, head halves == whole."""
    lens = [150, 70]
    case = make_case(sum(lens), H, "authors", seed=8)
    cu = torch.tensor([0, 150, 220], dtype=torch.int64, device="cuda")
    sa, sb = alloc_state(2, H), alloc_state(2, H)
    oa, fa = run(case, sa, mode=0, cu=cu)
    ob, fb = run(case, sb, mode=1, cu=cu)
    assert n_diff(oa, ob) == 0 and n_diff(fa, fb) == 0 and state_equal(sa, sb), "row engine != chunk kernels at large H"
    half = H // 2
    for lo, hi in ((0, half), (half, H)):
        sub = dict(case)
        for key in ("q", "k", "v", "g"):
            sub[key] = case[key][:, :, lo:hi].contiguous()
        sub["beta"], sub["A_log"], sub["dt_bias"] = case["beta"][:, :, lo:hi].contiguous(), case["A_log"][lo:hi], case["dt_bias"][lo:hi]
        st = alloc_state(2, hi - lo)
        o, o32 = run(sub, st, cu=cu)
        assert n_diff(o32, fa[:, :, lo:hi]) == 0 and n_diff(o, oa[:, :, lo:hi]) == 0, f"heads {lo}:{hi} differ from the whole call"
        assert state_equal(st, sa.select([0, 1], heads=slice(lo, hi)))
