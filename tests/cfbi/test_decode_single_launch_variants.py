"""Compile-time test variants of the single-launch decode kernel (``rows_entry`` in csrc/cfbi/cfbi_rows.cuh): the late-start race is exercised on demand, and the harness
is shown to be able to fail.

The single-launch decode has no plan kernel and no pos snapshot: the H head CTAs of a sequence count their arrivals in the high bits of the pos word and the last one to
arrive stores the plain new fill. A wrong protocol (every CTA stores the new fill from the fill it read, the first arrival stores instead of the last) is only visible when a
CTA STARTS after another CTA of its sequence has stored: on a grid that fits one wave of the SMs all CTAs read the word within a microsecond of each other and the plain
decode tests cannot tell. These tests build the extension with test macros (``load_variant``; the default build contains none of them):

* ``CFBI_DECODE_TEST_DELAY_NS=d``: every CTA lingers for 0 .. 3 x d before it arrives (the late starters of a multi-wave grid, forced) - the correct protocol must leave
  every bit identical to the frozen reference kernels (each case runs several times),
* ``CFBI_DECODE_TEST_EVERY_CTA_WRITES`` (no counting: every CTA reads the word and stores the new fill at the end, the plan-less version of the old scheme without its
  snapshot) and ``CFBI_DECODE_TEST_FIRST_ARRIVER`` (the first arrival stores the new fill, not the last), each together with the delays: NEGATIVE CONTROLS that the same bit
  comparison must catch (otherwise the harness proves nothing),
* the delay hook is live: a delayed build makes the call slower by far more than its noise.
"""
import re
from pathlib import Path

import pytest
import torch

from flash_kda.cfbi import state_mismatch

from helpers import n_diff
from test_decode_single_launch import kernel_counts, many_wave_n, prefilled, require_single_launch, step_inputs

REF = 3
H, STEPS, REPEATS = 12, 5, 3
DELAY_NS = 50000


@pytest.fixture(scope="module", autouse=True)
def single_launch_forced(ext):
    """The hidden calibration control on for every test of the module (and off again afterwards): the default build runs the single launch whatever the device's routing and the size of the
    grid, and every variant build is forced the same way when it is built (``build_variant``)."""
    previous = ext.decode_single_force(True)
    try:
        yield
    finally:
        ext.decode_single_force(previous)


@pytest.fixture(scope="module")
def workload(ext):
    """More than two waves of CTAs, every sequence at its own chunk phase (the chain crosses chunk ends), the state every run starts from."""
    N = many_wave_n(H)
    require_single_launch(ext, N, H)
    first = [125 + (n % 6) for n in range(N)]
    case, cu, st = prefilled(H, N, N, first, STEPS, None, seed=31)
    return {"N": N, "first": first, "case": case, "cu": cu, "st": st}


_BUILT = {}


def build_variant(name, defines):
    from flash_kda.cfbi._ext import load_variant

    if name not in _BUILT:
        try:
            _BUILT[name] = load_variant(name, tuple(defines))
        except Exception as e:   # pragma: no cover - the variants need a JIT build
            pytest.skip(f"cannot build the variant {name}: {e}")
        _BUILT[name].decode_single_force(True)   # a variant is a module of its own, with its own switch: its decode calls are the single launch whatever the device and the grid
    return _BUILT[name]


def decode_args(w, t):
    """The tensors of ``mod.forward`` for decode step t (mode 1, one token per sequence, no pool): returns ((q, k, v, g, beta), out, out32)."""
    case = w["case"]
    ins = step_inputs(case, w["cu"], w["first"], [t] * w["N"])
    q4, k4, v4, g4 = (ins[x].unsqueeze(1) for x in ("q", "k", "v", "g"))
    out = torch.full_like(q4, 3.0)
    o32 = torch.full(q4.shape, 3.0, dtype=torch.float32, device="cuda")
    return (q4, k4, v4, g4, ins["beta"].unsqueeze(1)), out, o32


def forward(mod, w, st, t, impl):
    args, out, o32 = decode_args(w, t)
    case = w["case"]
    mod.forward(*args, float(case["scale"]), out, case["A_log"], case["dt_bias"], float(case["lower_bound"]), st.E, st.rows, st.pos, None, 1, o32, impl, None, None, None, 3, 0)
    return out, o32


def run(mod, impl, w):
    """STEPS decode calls through ``mod.forward`` from a copy of the prefilled state; returns ([(bf16 out, fp32 out)] per step, final state)."""
    st = w["st"].clone()
    res = [forward(mod, w, st, t, impl) for t in range(STEPS)]
    torch.cuda.synchronize()
    return res, st


def differences(a, b):
    """Differing elements of the outputs of every step and of the final state (E, valid rows, pos) of two runs."""
    (ra, sa), (rb, sb) = a, b
    d = {"out": sum(n_diff(x[0], y[0]) for x, y in zip(ra, rb)), "out32": sum(n_diff(x[1], y[1]) for x, y in zip(ra, rb))}
    d.update(state_mismatch(sa, sb))
    return d


def total(d):
    return sum(d.values())


def require_one_launch(mod, w):
    """A comparison is only meaningful if the kernel under test is what ran."""
    st = w["st"].clone()
    cnt = kernel_counts(lambda: forward(mod, w, st, 0, 0))
    assert dict(cnt) == {"rows_kernel<true>": 1}, dict(cnt)


# ---- the default build has no test macro ----------------------------------------------------------------------------------------------------------------
def test_the_default_build_contains_no_decode_test_macro():
    from flash_kda.cfbi import _ext

    assert not any("TEST" in f for f in _ext.NVCC_FLAGS)
    root = Path(__file__).resolve().parents[2] / "csrc" / "cfbi"
    for f in root.rglob("*"):
        if f.suffix in (".cu", ".cuh", ".h", ".cpp"):
            for ln, line in enumerate(f.read_text().splitlines(), 1):
                assert not re.match(r"\s*#\s*define\s+CFBI_DECODE_TEST", line), f"{f.name}:{ln}: a test macro must only come from a -D flag"


def test_the_unmodified_build_equals_the_reference_on_the_variant_workload(ext, workload):
    require_one_launch(ext, workload)
    ref = run(ext, REF, workload)
    assert total(differences(ref, run(ext, 0, workload))) == 0


# ---- stress: delayed CTAs, the correct protocol stays identical --------------------------------------------------------------------------------------
STRESS = [
    ("dec_d2000", ("CFBI_DECODE_TEST_DELAY_NS=2000",)),
    ("dec_d50000", (f"CFBI_DECODE_TEST_DELAY_NS={DELAY_NS}",)),
]


@pytest.mark.parametrize("name,defines", STRESS, ids=[s[0] for s in STRESS])
def test_delayed_ctas_leave_every_bit_identical(ext, workload, name, defines):
    mod = build_variant(name, defines)
    require_one_launch(mod, workload)
    ref = run(ext, REF, workload)
    for rep in range(REPEATS):
        d = differences(ref, run(mod, 0, workload))
        assert total(d) == 0, (name, rep, d)


def test_the_delay_hook_is_live(ext, workload):
    """The delays are real: a call of the delayed build takes tens of microseconds longer than the same call of the default build (the CTAs of the first wave sleep up to
    3 x 50 us, the ones behind them wait for their SM)."""
    mod = build_variant("dec_d50000", (f"CFBI_DECODE_TEST_DELAY_NS={DELAY_NS}",))

    def per_call_us(m, reps=10):
        st = workload["st"].clone()
        args, out, o32 = decode_args(workload, 0)
        case = workload["case"]

        def f():
            m.forward(*args, float(case["scale"]), out, case["A_log"], case["dt_bias"], float(case["lower_bound"]), st.E, st.rows, st.pos, None, 1, o32, 0, None, None, None, 3, 0)
        f()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            f()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) * 1000.0 / reps

    t_default, t_delayed = per_call_us(ext), per_call_us(mod)
    assert t_delayed > t_default + 40.0, (t_default, t_delayed)


# ---- negative controls: wrong protocols must be caught -----------------------------------------------------------------------------------------------
CONTROLS = [
    ("dec_naive_d50000", (f"CFBI_DECODE_TEST_DELAY_NS={DELAY_NS}", "CFBI_DECODE_TEST_EVERY_CTA_WRITES")),   # no counting: every CTA stores from the fill it read
    ("dec_first_d50000", (f"CFBI_DECODE_TEST_DELAY_NS={DELAY_NS}", "CFBI_DECODE_TEST_FIRST_ARRIVER")),      # the first arrival stores, not the last
]


@pytest.mark.parametrize("name,defines", CONTROLS, ids=[c[0] for c in CONTROLS])
def test_a_wrong_protocol_is_caught_by_the_comparison(ext, workload, name, defines):
    ref = run(ext, REF, workload)
    assert total(differences(ref, run(ext, 0, workload))) == 0, "the correct build must match, or the comparison proves nothing"
    mod = build_variant(name, defines)
    require_one_launch(mod, workload)
    bad = differences(ref, run(mod, 0, workload))
    assert total(bad) > 0, f"the harness missed the wrong protocol {name}"
    assert bad["out"] > 0 or bad["out32"] > 0, bad     # CTAs that started from an advanced fill computed other rows
