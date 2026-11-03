"""Optimised kernels vs the frozen v1 reference (csrc/cfbi/ref): identical bits, zero tolerance.

The reference kernels are the simplest implementation of the numerical contract. Every optimisation of the
row engine / K1 / K2 must leave every bit unchanged, so each scenario below is run twice (``_impl=0``: the
optimised kernels, ``_impl=3``: the reference kernels for both the row engine and K1/K2) and compared:
bf16 outputs, fp32 outputs (before the bf16 rounding), E, the valid open-chunk rows and pos, after every call.

Scenarios: cold varlen prefill, warm-state call chains (head rows through the row engine, K1/K2 tails),
decode chains from every chunk phase, mode 1 (row engine for everything), the training forward.
"""
import random

import pytest
import torch

from flash_kda.cfbi import alloc_state, decode, forward_train, prefill, state_equal, state_mismatch

from helpers import REGIMES, make_case, n_diff

REF = 3


@pytest.fixture(autouse=True)
def _optimised_kernels_run_here(ext):
    """Where the optimised kernels cannot run (sm < 90: ``_impl=0`` resolves to the reference kernels, see
    ``resolve_impl``) both sides of every comparison below would be the same code: skip instead of passing vacuously."""
    if ext.resolve_impl(0, -1) == ext.resolve_impl(REF, -1):
        pytest.skip("the optimised kernels do not run on this device (_impl=0 resolves to the reference kernels)")


def _cu(lens):
    cu = [0]
    for L in lens:
        cu.append(cu[-1] + L)
    return cu


def _call_inputs(case, cu, done, take):
    """Concatenate, for every sequence n, the next ``take[n]`` tokens (from ``done[n]``) into one varlen call."""
    pieces = [{k: case[k][:, cu[n] + done[n]:cu[n] + done[n] + take[n]] for k in ("q", "k", "v", "g", "beta")}
              for n in range(len(take))]
    sub = {k: torch.cat([p[k] for p in pieces], dim=1).contiguous() for k in ("q", "k", "v", "g", "beta")}
    return sub, torch.tensor(_cu(take), dtype=torch.int64, device="cuda")


def run_chain(case, lens, takes, H, impl, mode=0):
    """``takes``: per call, per sequence number of tokens. Returns [(out, out32, state_snapshot)] per call."""
    N = len(lens)
    cu = _cu(lens)
    st = alloc_state(N, H)
    done = [0] * N
    res = []
    for take in takes:
        sub, cu_call = _call_inputs(case, cu, done, take)
        out = torch.zeros_like(sub["q"])
        out32 = torch.zeros(sub["q"].shape, dtype=torch.float32, device="cuda")
        prefill(sub["q"], sub["k"], sub["v"], sub["g"], sub["beta"], case["scale"], out, case["A_log"], case["dt_bias"],
                case["lower_bound"], st, cu_call, out_fp32=out32, _mode=mode, _impl=impl)
        res.append((out, out32, st.clone()))
        done = [d + t for d, t in zip(done, take)]
    return res


def assert_same_results(a, b, what):
    assert len(a) == len(b)
    for i, ((oa, fa, sa), (ob, fb, sb)) in enumerate(zip(a, b)):
        assert n_diff(oa, ob) == 0, f"{what} call {i}: bf16 outputs differ in {n_diff(oa, ob)} of {oa.numel()}"
        assert n_diff(fa, fb) == 0, f"{what} call {i}: fp32 outputs differ in {n_diff(fa, fb)} of {fa.numel()}"
        assert state_equal(sa, sb), f"{what} call {i}: state differs {state_mismatch(sa, sb)}"


def random_takes(lens, ncalls, rng, zero_prob=0.2):
    """Split every sequence's length into ``ncalls`` pieces (some empty)."""
    takes = [[0] * len(lens) for _ in range(ncalls)]
    for n, L in enumerate(lens):
        rem = L
        for c in range(ncalls):
            if c == ncalls - 1:
                t = rem
            elif rng.random() < zero_prob:
                t = 0
            else:
                t = min(rem, rng.choice([1, 2, 5, 15, 16, 17, 31, 47, 63, 64, 65, 100, 128, 129, rng.randint(1, 200)]))
            takes[c][n] = t
            rem -= t
    return takes


@pytest.mark.parametrize("regime", REGIMES)
def test_cold_varlen_prefill(ext, regime):
    H = 3
    rng = random.Random(1)
    lens = [rng.choice([1, 15, 16, 17, 63, 64, 65, 128, 130, 200, 257, rng.randint(1, 300)]) for _ in range(9)] + [0, 64, 1]
    case = make_case(sum(lens), H, regime, seed=5)
    takes = [lens]
    assert_same_results(run_chain(case, lens, takes, H, 0), run_chain(case, lens, takes, H, REF), f"cold/{regime}")


@pytest.mark.parametrize("regime", ["authors", "heavy_tail", "long_memory", "saturated", "one_hot"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_warm_call_chain(ext, regime, seed):
    H = 2
    rng = random.Random(10 + seed)
    lens = [rng.randint(60, 420) for _ in range(6)] + [64, 1, 130]
    case = make_case(sum(lens), H, regime, seed=20 + seed)
    takes = random_takes(lens, 4, rng)
    assert_same_results(run_chain(case, lens, takes, H, 0), run_chain(case, lens, takes, H, REF), f"warm/{regime}/{seed}")


@pytest.mark.parametrize("phase", [0, 1, 15, 16, 17, 31, 32, 33, 47, 48, 62, 63])
@pytest.mark.parametrize("N", [1, 5])
def test_decode_from_every_phase(ext, phase, N):
    """Prefill to a given chunk phase, then 70 single-token decode calls (crosses a chunk end)."""
    H, steps = 2, 70
    lens = [64 * 2 + phase + steps] * N
    case = make_case(sum(lens), H, "authors", seed=30 + phase)
    cu = _cu(lens)
    first = [64 * 2 + phase] * N
    results = {}
    for impl in (0, REF):
        st = alloc_state(N, H)
        sub, cu_call = _call_inputs(case, cu, [0] * N, first)
        out = torch.zeros_like(sub["q"])
        prefill(sub["q"], sub["k"], sub["v"], sub["g"], sub["beta"], case["scale"], out, case["A_log"], case["dt_bias"],
                case["lower_bound"], st, cu_call, _impl=impl)
        outs = []
        for t in range(steps):
            idx = [cu[n] + first[n] + t for n in range(N)]
            q1 = {k: torch.cat([case[k][:, i:i + 1] for i in idx], dim=0).contiguous() for k in ("q", "k", "v", "g", "beta")}
            o = torch.zeros_like(q1["q"])
            o32 = torch.zeros(q1["q"].shape, dtype=torch.float32, device="cuda")
            decode(q1["q"], q1["k"], q1["v"], q1["g"], q1["beta"], case["scale"], o, case["A_log"], case["dt_bias"],
                   case["lower_bound"], st, out_fp32=o32, _impl=impl)
            outs.append((o, o32, st.clone() if t % 7 == 0 or t >= steps - 2 else None))
        results[impl] = outs
    for t, ((oa, fa, sa), (ob, fb, sb)) in enumerate(zip(results[0], results[REF])):
        assert n_diff(oa, ob) == 0 and n_diff(fa, fb) == 0, f"decode step {t}: outputs differ"
        if sa is not None:
            assert state_equal(sa, sb), f"decode step {t}: state differs {state_mismatch(sa, sb)}"


@pytest.mark.parametrize("regime", ["authors", "heavy_tail", "long_memory"])
def test_mode1_row_engine_everything(ext, regime):
    H = 2
    lens = [150, 1, 64, 129, 33]
    case = make_case(sum(lens), H, regime, seed=40)
    takes = random_takes(lens, 3, random.Random(3))
    assert_same_results(run_chain(case, lens, takes, H, 0, mode=1), run_chain(case, lens, takes, H, REF, mode=1),
                        f"mode1/{regime}")


def test_training_forward(ext):
    H = 3
    lens = [100, 7, 64, 129, 200]
    case = make_case(sum(lens), H, "authors", seed=50)
    cu = torch.tensor(_cu(lens), dtype=torch.int64, device="cuda")
    res = []
    for impl in (0, REF):
        o32 = torch.zeros(case["q"].shape, dtype=torch.float32, device="cuda")
        o = forward_train(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], case["A_log"],
                          case["dt_bias"], case["lower_bound"], cu, out_fp32=o32, _impl=impl)
        res.append((o, o32))
    assert n_diff(res[0][0], res[1][0]) == 0 and n_diff(res[0][1], res[1][1]) == 0


@pytest.mark.parametrize("H", [1, 6, 33])
def test_head_counts(ext, H):
    lens = [150, 70, 64, 3]
    case = make_case(sum(lens), H, "authors", seed=60 + H)
    takes = random_takes(lens, 3, random.Random(4))
    assert_same_results(run_chain(case, lens, takes, H, 0), run_chain(case, lens, takes, H, REF), f"H={H}")


@pytest.mark.parametrize("T", [1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 1000])
def test_single_sequence_lengths(ext, T):
    H = 2
    case = make_case(T, H, "heavy_tail", seed=70)
    takes = [[T]]
    assert_same_results(run_chain(case, [T], takes, H, 0), run_chain(case, [T], takes, H, REF), f"T={T}")
