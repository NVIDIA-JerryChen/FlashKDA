"""Device tests of the row-engine clock profile (``-DCFBI_ROWS_PROFILE=1``, a separate JIT build of the extension, see ``load_variant``).

* The profile build changes no bit of the default build: the decode workload of ``test_vs_ref.py::test_decode_from_every_phase`` (prefill to a chunk phase, then 70 single-token decode
  calls, every phase, 1 and 5 sequences) and a 64-call decode of 16 sequences with 12 heads run on both builds and every output (bf16 and fp32), E, valid row and ``pos`` is compared after every
  call. A profile build that perturbs one output bit (``CFBI_ROWS_PROFILE_PERTURB=1``) must fail the same comparison, so the comparison can fail.
* The default build carries no profile buffer (the accessors return an empty tensor); the profile build keeps one row per (position, CTA), at the slot of the row, with ordered stamps: the
  entry stamp precedes the first row stamp (the prologue takes time), the eight row stamps never decrease and the exit stamp follows the last of them.

The build of a variant that fails is an error here, not a skip: a skipped comparison proves nothing.
"""
import pytest
import torch

from flash_kda.cfbi import alloc_state, state_mismatch

from helpers import make_case, n_diff, skip_or_fail
from test_vs_ref import _call_inputs, _cu

PROFILE = ("CFBI_ROWS_PROFILE=1",)
PERTURBED = ("CFBI_ROWS_PROFILE=1", "CFBI_ROWS_PROFILE_PERTURB=1")
STEPS = 70
POSITIONS, STAMPS = 64, 10
CALL_KEYS = ("q", "k", "v", "g", "beta")


@pytest.fixture(autouse=True)
def _optimised_kernels_run_here(ext):
    """Below sm_90 ``_impl=0`` resolves to the reference kernels, which carry no profile: skip (fail on B200 / B300) instead of comparing a build with itself."""
    if ext.resolve_impl(0, -1) == ext.resolve_impl(3, -1):
        skip_or_fail("the optimised kernels do not run on this device (_impl=0 resolves to the reference kernels)")


@pytest.fixture(scope="module")
def prof(ext):
    from flash_kda.cfbi._ext import load_variant

    return load_variant("rowsprof", PROFILE)


@pytest.fixture(scope="module")
def perturbed(ext):
    from flash_kda.cfbi._ext import load_variant

    return load_variant("rowsprof_perturb", PERTURBED)


def forward(mod, case, q, k, v, g, beta, out, st, cu, mode, out32):
    mod.forward(q, k, v, g, beta, float(case["scale"]), out, case["A_log"], case["dt_bias"], float(case["lower_bound"]), st.E, st.rows, st.pos, cu, mode, out32, 0, None)


def run_phase_workload(mod, phase, N, H=2):
    """The decode workload of ``test_decode_from_every_phase`` through ``mod``: ``[(bf16 out, fp32 out, state snapshot)]`` per decode call (the prefill goes through ``mod`` as well)."""
    lens = [64 * 2 + phase + STEPS] * N
    case = make_case(sum(lens), H, "authors", seed=30 + phase)
    cu = _cu(lens)
    first = [64 * 2 + phase] * N
    st = alloc_state(N, H)
    sub, cu_call = _call_inputs(case, cu, [0] * N, first)
    forward(mod, case, sub["q"], sub["k"], sub["v"], sub["g"], sub["beta"], torch.zeros_like(sub["q"]), st, cu_call, 0, None)
    res = []
    for t in range(STEPS):
        idx = [cu[n] + first[n] + t for n in range(N)]
        d = {k: torch.cat([case[k][:, i:i + 1] for i in idx], dim=0).contiguous() for k in CALL_KEYS}
        o = torch.zeros_like(d["q"])
        o32 = torch.zeros(d["q"].shape, dtype=torch.float32, device="cuda")
        forward(mod, case, d["q"], d["k"], d["v"], d["g"], d["beta"], o, st, None, 1, o32)
        res.append((o, o32, st.clone()))
    return res


def run_shape_workload(mod, N, H, steps=POSITIONS):
    """``steps`` decode calls of ``N`` sequences with ``H`` heads from a fresh state (every sequence at the same position, advancing by one per call), as ``benchmarks/micro/rows_profile.py`` runs them."""
    case = make_case(N * steps, H, "authors", seed=0)
    st = alloc_state(N, H)
    res = []
    for s in range(steps):
        d = {k: torch.stack([case[k][0, n * steps + s] for n in range(N)], 0).unsqueeze(1).contiguous() for k in CALL_KEYS}
        o = torch.zeros_like(d["q"])
        o32 = torch.zeros(d["q"].shape, dtype=torch.float32, device="cuda")
        forward(mod, case, d["q"], d["k"], d["v"], d["g"], d["beta"], o, st, None, 1, o32)
        res.append((o, o32, st.clone()))
    return res


def differences(a, b):
    """Numbers of differing elements over all calls of two runs: bf16 outputs, fp32 outputs, E, valid rows, pos."""
    assert len(a) == len(b)
    tot = {"out": 0, "out32": 0, "E": 0, "rows": 0, "pos": 0}
    for (oa, fa, sa), (ob, fb, sb) in zip(a, b):
        tot["out"] += int(n_diff(oa, ob))
        tot["out32"] += int(n_diff(fa, fb))
        sm = state_mismatch(sa, sb)
        for k in ("E", "rows", "pos"):
            tot[k] += int(sm[k])
    return tot


# ---- the default build carries no profile; the profile build keeps one --------------------------------------------------------------------------------
def test_the_default_build_has_no_profile_buffer(ext):
    assert ext.rows_profile_read().numel() == 0
    ext.rows_profile_clear()         # inert, must not raise


def test_the_profile_build_has_a_buffer_that_a_clear_empties(prof):
    prof.rows_profile_clear()
    buf = prof.rows_profile_read()
    assert tuple(buf.shape) == (POSITIONS, 4096, STAMPS) and buf.dtype == torch.int32 and int(buf.abs().sum()) == 0


# ---- the profile build changes no bit ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("N", [1, 5])
@pytest.mark.parametrize("phase", [0, 1, 15, 16, 17, 31, 32, 33, 47, 48, 62, 63])
def test_the_profile_build_is_bit_identical_on_the_decode_workload(ext, prof, phase, N):
    base = run_phase_workload(ext, phase, N)
    prof.rows_profile_clear()
    got = run_phase_workload(prof, phase, N)
    assert differences(base, got) == {"out": 0, "out32": 0, "E": 0, "rows": 0, "pos": 0}
    assert int((prof.rows_profile_read()[:, :N * 2, 9] != 0).sum()) > 0, "the profile build recorded no row: the comparison above would be vacuous"


def test_the_profile_build_is_bit_identical_on_the_profile_shapes(ext, prof):
    base = run_shape_workload(ext, 16, 12)
    got = run_shape_workload(prof, 16, 12)
    assert differences(base, got) == {"out": 0, "out32": 0, "E": 0, "rows": 0, "pos": 0}


def test_a_perturbed_profile_build_is_caught_by_the_comparison(ext, prof, perturbed):
    """The perturbation flips the lowest bit of one bf16 output element per call: the comparison that passes for the profile build must report exactly those flips."""
    base = run_phase_workload(ext, 33, 5)
    assert differences(base, run_phase_workload(prof, 33, 5)) == {"out": 0, "out32": 0, "E": 0, "rows": 0, "pos": 0}
    bad = differences(base, run_phase_workload(perturbed, 33, 5))
    assert bad["out"] == STEPS, bad
    assert bad["out32"] == 0 and bad["E"] == 0 and bad["rows"] == 0 and bad["pos"] == 0, bad


# ---- what the profile build records ----------------------------------------------------------------------------------------------------------------
def test_one_decode_call_records_one_row_per_cta_at_its_position(prof):
    N, H = 3, 2
    case = make_case(N, H, "authors", seed=1)
    st = alloc_state(N, H)
    d = {k: case[k][0].unsqueeze(1).contiguous() for k in CALL_KEYS}      # N tokens as N sequences of one token
    o = torch.zeros_like(d["q"])
    prof.rows_profile_clear()
    forward(prof, case, d["q"], d["k"], d["v"], d["g"], d["beta"], o, st, None, 1, None)
    torch.cuda.synchronize()
    smid = prof.rows_profile_read()[:, :, 9]
    assert int((smid[0, :N * H] != 0).sum()) == N * H, "every CTA records its row at position 0"
    assert int((smid[1:] != 0).sum()) == 0 and int((smid[0, N * H:] != 0).sum()) == 0, "nothing is recorded at another position or CTA"


def test_the_stamps_of_every_row_are_ordered_and_the_entry_stamp_precedes_the_first_row(prof):
    N, H = 16, 12
    prof.rows_profile_clear()
    run_shape_workload(prof, N, H)
    buf = prof.rows_profile_read()
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    ctas = N * H
    rows = buf[:, :ctas, :].reshape(-1, STAMPS)
    assert rows.shape[0] == POSITIONS * ctas and int((rows[:, 9] != 0).sum()) == POSITIONS * ctas, "every row of the 64 calls is recorded exactly once"
    assert int(buf[:, ctas:, :].abs().sum()) == 0, "no slot beyond the CTAs of the grid is written"
    ts = rows[:, :9].to(torch.int64)
    assert bool((ts[:, 0] > 0).all()), "the entry stamp precedes the first row stamp: the prologue takes time"
    assert bool((ts[:, 1:] >= ts[:, :-1]).all()), "the stamps of a row never decrease, the exit stamp follows the last row stamp"
    assert bool((ts[:, 8] > ts[:, 7]).all()), "the exit stamp follows the last row stamp"
    sm = rows[:, 9].to(torch.int64) - 1
    assert bool(((sm >= 0) & (sm < sm_count)).all())
