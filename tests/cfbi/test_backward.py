"""M5/M6 gates: the differentiable ``flash_kda.cfbi.chunk_kda``.

* forward: bitwise equal to ``forward_train`` (the autograd shell adds nothing) and numerically the same
  function as FLA's ``chunk_kda`` (gate/beta/l2norm forms agree);
* backward: gradients (FLA's chunk backward on a recomputed forward) against an fp64 autograd of the exact
  per-token recurrence, with FLA's own test ratios (dq/dk/dv 0.008, dg/dbeta 0.02).
"""
import os

os.environ.setdefault("FLA_FLASH_KDA", "0")   # keep FLA on its own Triton kernels

import pytest
import torch

pytest.importorskip("fla")

from flash_kda.cfbi import chunk_kda, forward_train
from flash_kda.cfbi.reference import exact_recurrence

from helpers import make_case, n_diff, rel_rms

RATIOS = {"q": 0.008, "k": 0.008, "v": 0.008, "g": 0.02, "beta": 0.02, "A_log": 0.02, "dt_bias": 0.02}
NAMES = list(RATIOS)


def make_do(shape, seed):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    return torch.randn(shape, device="cuda", generator=gen).to(torch.bfloat16)


def reshape_case(case, B):
    """[1, B*T, H, 128] -> [B, T, H, 128] (B sequences of equal length)."""
    out = dict(case)
    for key in ("q", "k", "v", "g", "beta"):
        x = case[key]
        out[key] = x.reshape(B, x.shape[1] // B, *x.shape[2:]).contiguous()
    return out


def our_grads(case, do, cu=None):
    ins = {k: case[k].clone().requires_grad_() for k in ("q", "k", "v", "g", "beta")}
    A, dtb = case["A_log"].clone().requires_grad_(), case["dt_bias"].clone().requires_grad_()
    out = chunk_kda(ins["q"], ins["k"], ins["v"], ins["g"], ins["beta"], case["scale"], A, dtb, case["lower_bound"], cu)
    out.backward(do)
    return out, {**{k: v.grad for k, v in ins.items()}, "A_log": A.grad, "dt_bias": dtb.grad}


def oracle_grads(case, do, cu=None):
    ins = {k: case[k].double().requires_grad_() for k in ("q", "k", "v", "g", "beta")}
    A, dtb = case["A_log"].double().requires_grad_(), case["dt_bias"].double().requires_grad_()
    out, _ = exact_recurrence(ins["q"], ins["k"], ins["v"], ins["g"], ins["beta"], A, dtb, case["scale"],
                              case["lower_bound"], cu)
    (out.reshape(case["q"].shape) * do.double()).sum().backward()
    return {**{k: v.grad for k, v in ins.items()}, "A_log": A.grad, "dt_bias": dtb.grad}


def check_grads(case, do, cu=None, label=""):
    out, tri = our_grads(case, do, cu)
    ref = oracle_grads(case, do, cu)
    worst = {}
    for name in NAMES:
        assert tri[name] is not None and torch.isfinite(tri[name].float()).all(), f"{label} d{name} not finite"
        worst[name] = rel_rms(tri[name].float(), ref[name])
    print(f"\n[grads {label}] " + "  ".join(f"d{n}={worst[n]:.2e}" for n in NAMES))
    for name in NAMES:
        assert worst[name] <= RATIOS[name], f"{label}: d{name} rel RMS {worst[name]:.3e} > {RATIOS[name]}"
    return out


@pytest.mark.parametrize("lower_bound", [-5.0, -2.5])
@pytest.mark.parametrize("regime", ["authors", "unit"])
def test_gradients_vs_fp64_oracle(ext, regime, lower_bound):
    H, T = 2, 192
    case = make_case(T, H, regime, seed=3)
    case["lower_bound"] = lower_bound
    check_grads(case, make_do(case["q"].shape, 1), label=f"{regime} lb={lower_bound}")


def test_gradients_varlen(ext):
    H, lens = 2, [100, 37, 64, 70]
    case = make_case(sum(lens), H, "authors", seed=4)
    cu = torch.tensor([0, 100, 137, 201, 271], dtype=torch.int64, device="cuda")
    check_grads(case, make_do(case["q"].shape, 2), cu, label="varlen")


def test_gradients_batched(ext):
    case = reshape_case(make_case(2 * 130, 2, "authors", seed=5), 2)
    check_grads(case, make_do(case["q"].shape, 3), label="B=2")


def test_forward_is_forward_train_bitwise(ext):
    H, T = 2, 200
    case = make_case(T, H, "authors", seed=6)
    ref = forward_train(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], case["A_log"],
                        case["dt_bias"], case["lower_bound"])
    ins = {k: case[k].clone().requires_grad_() for k in ("q", "k", "v")}
    out = chunk_kda(ins["q"], ins["k"], ins["v"], case["g"], case["beta"], case["scale"], case["A_log"],
                    case["dt_bias"], case["lower_bound"])
    assert out.requires_grad and n_diff(out, ref) == 0
    with torch.no_grad():
        out2 = chunk_kda(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], case["A_log"],
                         case["dt_bias"], case["lower_bound"])
    assert n_diff(out2, ref) == 0


def test_forward_matches_fla_semantics(ext):
    """Same function as FLA's chunk_kda (different rounding): relative RMS at the bf16 level."""
    from fla.ops.kda import chunk_kda as fla_chunk_kda

    H, T = 2, 256
    case = make_case(T, H, "authors", seed=7)
    ours = forward_train(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], case["A_log"],
                         case["dt_bias"], case["lower_bound"])
    theirs, _ = fla_chunk_kda(case["q"], case["k"], case["v"], case["g"], case["beta"], A_log=case["A_log"],
                              dt_bias=case["dt_bias"].reshape(-1), scale=case["scale"], use_qk_l2norm_in_kernel=True,
                              use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, safe_gate=True,
                              lower_bound=case["lower_bound"])
    ref, _ = exact_recurrence(case["q"], case["k"], case["v"], case["g"], case["beta"], case["A_log"],
                              case["dt_bias"], case["scale"], case["lower_bound"])
    ref = ref.reshape(case["q"].shape)
    e_ours, e_fla, e_between = rel_rms(ours, ref), rel_rms(theirs, ref), rel_rms(ours, theirs)
    print(f"\n[forward vs exact] cfbi={e_ours:.2e}  fla={e_fla:.2e}  cfbi-vs-fla={e_between:.2e}")
    assert e_between < 2e-2 and e_ours < 4e-3


def test_zero_lower_bound_rejected_only_when_grad_needed(ext):
    case = make_case(64, 1, "no_decay", seed=8)
    assert case["lower_bound"] == 0.0
    with torch.no_grad():
        chunk_kda(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], case["A_log"],
                  case["dt_bias"], case["lower_bound"])
    q = case["q"].clone().requires_grad_()
    with pytest.raises(ValueError, match="lower_bound"):
        chunk_kda(q, case["k"], case["v"], case["g"], case["beta"], case["scale"], case["A_log"], case["dt_bias"],
                  case["lower_bound"])
