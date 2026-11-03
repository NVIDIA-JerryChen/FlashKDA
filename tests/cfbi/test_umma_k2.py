"""The tcgen05 / UMMA K2 (``_impl=4``) against the mma.sync K2 (``_impl=0``) and the frozen reference (``_impl=3``).

Zero tolerance: bf16 outputs, fp32 outputs (before the bf16 rounding), E, the valid open-chunk rows (the tail chunk's U
rows are written by K2) and pos are compared bit for bit after every call. ``_impl=4`` = optimised row engine and K1 with
the UMMA K2, so everything that differs is the K2 program: dot3 on tcgen05 (K = 8 per instruction, TMEM accumulators),
the per-chunk epilogues and the operand pipeline.

Scenarios: single long sequence, cold varlen batch with empty sequences, warm-state call chains (partial tail chunks,
head rows through the row engine), the training forward (state-free), head counts, several data regimes, and a negative
control (a deliberately wrong UMMA program must be flagged).
"""
import random

import pytest
import torch

from flash_kda.cfbi import alloc_state, forward_train, state_equal, state_mismatch

from helpers import REGIMES, make_case, n_diff
from test_vs_ref import REF, assert_same_results, random_takes, run_chain

UMMA = 4


@pytest.mark.parametrize("regime", ["authors", "heavy_tail", "long_memory"])
@pytest.mark.parametrize("H", [1, 6])
def test_single_long_sequence(ext, regime, H):
    T = 64 * 12 + 37   # 12 full chunks + a 37-row tail
    case = make_case(T, H, regime, seed=3)
    takes = [[T]]
    a = run_chain(case, [T], takes, H, UMMA)
    assert_same_results(a, run_chain(case, [T], takes, H, 0), f"umma-vs-opt/{regime}/H{H}")
    assert_same_results(a, run_chain(case, [T], takes, H, REF), f"umma-vs-ref/{regime}/H{H}")


@pytest.mark.parametrize("regime", REGIMES)
def test_cold_varlen_with_empty_sequences(ext, regime):
    H = 3
    rng = random.Random(1)
    lens = [rng.choice([1, 15, 16, 17, 63, 64, 65, 128, 130, 200, 257, rng.randint(1, 300)]) for _ in range(9)] + [0, 64, 0, 1]
    case = make_case(sum(lens), H, regime, seed=5)
    takes = [lens]
    a = run_chain(case, lens, takes, H, UMMA)
    assert_same_results(a, run_chain(case, lens, takes, H, 0), f"cold/{regime}")
    assert_same_results(a, run_chain(case, lens, takes, H, REF), f"cold-ref/{regime}")


@pytest.mark.parametrize("regime", ["authors", "heavy_tail", "long_memory"])
@pytest.mark.parametrize("seed", [0, 1])
def test_warm_call_chain(ext, regime, seed):
    """Calls that start inside a chunk (row engine head rows), end inside a chunk (K2 tail rows -> U row cache)."""
    H = 2
    rng = random.Random(10 + seed)
    lens = [rng.randint(60, 420) for _ in range(6)] + [64, 1, 130]
    case = make_case(sum(lens), H, regime, seed=20 + seed)
    takes = random_takes(lens, 4, rng)
    a = run_chain(case, lens, takes, H, UMMA)
    assert_same_results(a, run_chain(case, lens, takes, H, 0), f"warm/{regime}/{seed}")
    assert_same_results(a, run_chain(case, lens, takes, H, REF), f"warm-ref/{regime}/{seed}")


@pytest.mark.parametrize("H", [1, 6, 33])
def test_head_counts(ext, H):
    lens = [150, 70, 64, 3, 0, 129]
    case = make_case(sum(lens), H, "authors", seed=60 + H)
    takes = random_takes(lens, 3, random.Random(4))
    a = run_chain(case, lens, takes, H, UMMA)
    assert_same_results(a, run_chain(case, lens, takes, H, 0), f"H={H}")


@pytest.mark.parametrize("T", [1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 191, 192, 193, 256, 257, 1000])
def test_single_sequence_lengths(ext, T):
    H = 2
    case = make_case(T, H, "heavy_tail", seed=70)
    takes = [[T]]
    assert_same_results(run_chain(case, [T], takes, H, UMMA), run_chain(case, [T], takes, H, REF), f"T={T}")


def _train(case, cu, impl):
    o32 = torch.zeros(case["q"].shape, dtype=torch.float32, device="cuda")
    o = forward_train(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], case["A_log"],
                      case["dt_bias"], case["lower_bound"], cu, out_fp32=o32, _impl=impl)
    return o, o32


@pytest.mark.parametrize("regime", ["authors", "heavy_tail"])
def test_training_forward(ext, regime):
    """State-free schedule: E starts at zero and is never written back."""
    H = 3
    lens = [100, 7, 64, 129, 200, 0, 70]
    case = make_case(sum(lens), H, regime, seed=50)
    cu = torch.tensor([0] + list(__import__("itertools").accumulate(lens)), dtype=torch.int64, device="cuda")
    a, a32 = _train(case, cu, UMMA)
    for impl in (0, REF):
        b, b32 = _train(case, cu, impl)
        assert n_diff(a, b) == 0 and n_diff(a32, b32) == 0, f"training forward differs from impl {impl}"


def test_negative_control_wrong_umma_program(ext):
    """The comparison must be able to fail: a UMMA K2 that drops the lo*hi term (compiled in with
    CFBI_UMMA_NEG_NOLO) has to differ from the correct kernels in a visible fraction of the outputs."""
    from flash_kda.cfbi._ext import load_variant

    try:
        bad = load_variant("umma_nolo", ("CFBI_UMMA_NEG_NOLO=1",))
    except Exception as e:   # pragma: no cover - the variant needs a JIT build
        pytest.skip(f"cannot build the negative-control variant: {e}")
    H, T = 2, 64 * 6 + 5
    case = make_case(T, H, "authors", seed=3)
    cu = torch.tensor([0, T], dtype=torch.int64, device="cuda")

    def run(mod, impl):
        o32 = torch.zeros(case["q"].shape, dtype=torch.float32, device="cuda")
        o = torch.zeros_like(case["q"])
        mod.forward(case["q"], case["k"], case["v"], case["g"], case["beta"], float(case["scale"]), o, case["A_log"],
                    case["dt_bias"], float(case["lower_bound"]), None, None, None, cu, 0, o32, impl, None)
        return o, o32

    good = run(ext, UMMA)
    wrong = run(bad, UMMA)
    frac = n_diff(good[1], wrong[1]) / good[1].numel()
    # dropping the lo*hi cross term only changes the last bits of the fp32 results (8% of the outputs here), but any visible
    # fraction proves the bitwise comparison is sensitive to the UMMA program
    assert frac > 0.01, f"the harness is not sensitive to a wrong UMMA program (only {frac:.2%} of fp32 outputs differ)"
