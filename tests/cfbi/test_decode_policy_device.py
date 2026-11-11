"""The single-launch decode policy on a device (needs the extension and a GPU; on B200 and B300 nothing here may be skipped).

* the binding's pure policy (any SM count and capability, no device) equals the specification on the whole shape matrix, and the controls of the host-compiled test are differences here too;
* the wrapper that reads the device equals the specification with this device's SM count and capability, with and without an explicit device;
* the hidden calibration control: while it is on the wrapper answers the eligible part of the rule and nothing else (the pure policy is not affected), and it reports its previous setting;
* the plan buffer: a call that is one launch (the control on) allocates no plan buffer and launches no plan kernel; a call that needs the plan kernel (bit 64, or a grid beyond the limit)
  with a plan buffer that is too small (empty or one int short) is an error, with the right size it runs;
* bit 64 changes no launch of a prefill call (with ``cu_seqlens``, where the default routing may choose the device-scheduled K2, and with a fixed length), and no prefill call launches
  ``rows_kernel<true>`` even with the control on;
* ``resolve_impl`` precedence: bit 64 is a request of its own (no bit clears it, it clears no other bit), and neither it nor the control brings the reference row engine to the single launch.
"""
import pytest
import torch

import decode_policy as dp
from helpers import make_case
from flash_kda.cfbi import alloc_state
from test_decode_policy import CONTROLS
from test_decode_single_launch import _cc, _sms, kernel_counts
from test_k2_dyn import fixed_fn, launched, varlen_fn

REF_ROWS, REF, ROWS_ONLY, DECODE_PLAN = 1, 3, 4, 64


@pytest.fixture()
def unforced(ext):
    """The control off for the test (and restored afterwards): the default policy of the device."""
    previous = ext.decode_single_force(False)
    try:
        yield
    finally:
        ext.decode_single_force(previous)


# ---- the pure policy through the binding -------------------------------------------------------------------------------------------------------------------
def test_the_binding_policy_equals_the_specification_on_the_matrix(ext):
    rows = dp.shape_matrix()
    got = [ext.decode_single_launch_policy(*r) for r in rows]
    assert [(r, g) for r, g in zip(rows, got) if g != dp.policy(*r)] == []
    assert 0 < sum(got) < len(got)


@pytest.mark.parametrize("name,knobs", CONTROLS, ids=[c[0] for c in CONTROLS])
def test_a_specification_with_another_rule_differs_from_the_binding_on_the_matrix(ext, name, knobs):
    rows = dp.shape_matrix()
    assert [r for r in rows if ext.decode_single_launch_policy(*r) != dp.policy(*r, **knobs)], name


def test_the_pure_policy_takes_every_input_from_its_arguments(ext):
    for r in [(1, 0, 98, 12, 148, 100), (1, 0, 99, 12, 148, 100), (1, 0, 4, 12, 132, 90), (1, 0, 4, 12, 148, 103)]:
        assert ext.decode_single_launch_policy(*r) == dp.policy(*r)
    for bad in [(1 << 40, 0, 4, 12, 148, 100), (1, 1 << 40, 4, 12, 148, 100), (1, 0, 4, 12, 1 << 40, 100), (1, 0, 4, 12, 148, 1 << 40)]:
        with pytest.raises((RuntimeError, ValueError), match="32-bit"):
            ext.decode_single_launch_policy(*bad)


# ---- the wrapper and the control ---------------------------------------------------------------------------------------------------------------------------
def device_rows():
    return sorted({(m, i, N, H) for m, i, N, H, s, c in dp.shape_matrix()})


def test_the_wrapper_reads_this_devices_sm_count_and_capability(ext, unforced):
    sms, cc, device = _sms(), _cc(), torch.cuda.current_device()
    rows = device_rows()
    got = [ext.decode_single_launch(*r) for r in rows]
    assert [(r, g) for r, g in zip(rows, got) if g != dp.policy(*r, sms, cc)] == []
    assert got == [ext.decode_single_launch(*r, device) for r in rows]
    assert 0 < sum(got) or cc // 10 != dp.ROUTING_CC_MAJOR                  # on a device of the routing set some rows are single launches


def test_the_control_applies_the_eligible_part_of_the_rule_and_nothing_else(ext, unforced):
    rows = device_rows()
    assert ext.decode_single_force(True) is False                            # the previous setting: off
    try:
        assert ext.decode_single_force(True) is True
        forced = [ext.decode_single_launch(*r) for r in rows]
        assert [(r, g) for r, g in zip(rows, forced) if g != dp.eligible(*r)] == []
        assert any(forced) and not all(forced)
        assert any(g and not dp.policy(*r, _sms(), _cc()) for r, g in zip(rows, forced))     # the control forces calls that the policy of this device keeps on the plan kernel ...
        assert not any(g for r, g in zip(rows, forced) if r[1] & (1 | 64) or r[0] != 1)      # ... but never mode 0, the reference row engine or bit 64
        pure = [ext.decode_single_launch_policy(*r, 148, 100) for r in rows]
        assert pure == [dp.policy(*r, 148, 100) for r in rows]               # the pure function does not know the control
    finally:
        assert ext.decode_single_force(False) is True
    assert ext.decode_single_launch(1, 0, 10 ** 6, 12) is False             # off again: a grid far beyond the limit keeps the plan kernel


# ---- the plan buffer --------------------------------------------------------------------------------------------------------------------------------------
def decode_call(ext, N, H, impl, plan=None):
    """One decode call through the extension with the plan buffer ``plan`` (None: the binding's own)."""
    case = make_case(N, H, "authors", seed=2)
    st = alloc_state(N, H)
    q, k, v, g = (case[x][0].unsqueeze(1) for x in ("q", "k", "v", "g"))
    beta = case["beta"][0].unsqueeze(1)
    out = torch.zeros_like(q)

    def call():
        ext.forward(q, k, v, g, beta, float(case["scale"]), out, case["A_log"], case["dt_bias"], float(case["lower_bound"]), E=st.E, rows=st.rows, pos=st.pos, mode=1, impl=impl, plan=plan)
    return call


def allocations(fn):
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    fn()
    torch.cuda.synchronize()
    return torch.cuda.memory_stats()["allocation.all.allocated"] - before


def test_a_single_launch_allocates_no_plan_buffer(ext, forced):
    N, H = 4, 6
    assert ext.decode_single_launch(1, ext.resolve_impl(0, -1), N, H)
    one, two = decode_call(ext, N, H, 0), decode_call(ext, N, H, DECODE_PLAN)
    one()
    two()                                                                    # warm: attribute set-up and the first-call checks are behind us
    assert dict(kernel_counts(one, warm=False)) == {"rows_kernel<true>": 1}
    assert dict(kernel_counts(two, warm=False)) == {"plan_kernel": 1, "rows_kernel<false>": 1}
    assert (allocations(two), allocations(one)) == (1, 0), "the plan buffer is the one allocation that the single launch saves"


@pytest.mark.parametrize("why", ["bit 64", "a grid beyond the limit"])
def test_a_call_that_needs_the_plan_kernel_rejects_a_missing_or_small_plan_buffer(ext, forced, why):
    if why == "bit 64":
        N, H, impl = 4, 6, DECODE_PLAN
    else:
        ext.decode_single_force(False)                                       # the control would make the grid beyond the limit a single launch
        N, H, impl = dp.MAX_CTAS_PER_SM * _sms() // 12 + 1, 12, 0
    assert not ext.decode_single_launch(1, ext.resolve_impl(impl, -1), N, H)
    need = (N + 1) * 8                                                       # what a decode call needs: (N + 1) plan entries of 8 ints (the schedule of K2 is for prefill)
    for size in (0, need - 1):
        with pytest.raises((RuntimeError, ValueError), match="plan buffer"):
            decode_call(ext, N, H, impl, plan=torch.empty(size, dtype=torch.int32, device="cuda"))()
    decode_call(ext, N, H, impl, plan=torch.empty(need, dtype=torch.int32, device="cuda"))()           # the right size runs
    decode_call(ext, N, H, impl)()                                                                       # none given: the binding allocates one


# ---- prefill is not affected by bit 64 ---------------------------------------------------------------------------------------------------------------------
PREFILLS = [("varlen routed", lambda impl: varlen_fn([256] * 16, 6, impl)), ("varlen with empty and short sequences", lambda impl: varlen_fn([150, 1, 64, 129, 33, 0, 70], 3, impl)),
            ("fixed length", lambda impl: fixed_fn(4, 192, 6, impl))]


@pytest.mark.parametrize("name,make", PREFILLS, ids=[p[0] for p in PREFILLS])
def test_prefill_launches_the_same_kernels_with_and_without_bit_64(ext, name, make):
    plain, with_bit = launched(make(0)), launched(make(DECODE_PLAN))
    assert plain == with_bit, (dict(plain), dict(with_bit))
    assert plain["plan_kernel"] >= 1 and sum(v for k, v in plain.items() if k.endswith("rows_kernel")) >= 1          # prefill, not an empty comparison
    if name == "varlen routed":
        routed = ext.k2_route(16, 6, 16 * 256, -1)
        assert (plain["k2_dyn_kernel"] >= 1) == routed, (routed, dict(plain))        # the default routing is live where the rule says so, and bit 64 does not switch it off


@pytest.mark.parametrize("name,make", PREFILLS, ids=[p[0] for p in PREFILLS])
def test_no_prefill_call_launches_the_single_launch_kernel_even_when_the_control_is_on(ext, forced, name, make):
    for impl in (0, DECODE_PLAN):
        cnt = kernel_counts(make(impl))
        assert cnt["rows_kernel<true>"] == 0 and cnt["rows_kernel<false>"] + cnt["ref::rows_kernel"] >= 1, (impl, dict(cnt))


# ---- the precedence of the bits ---------------------------------------------------------------------------------------------------------------------------
def test_resolve_impl_keeps_bit_64_a_request_of_its_own(ext):
    for v in range(dp.IMPL_MAX + 1):
        r = ext.resolve_impl(v, -1)
        assert r & DECODE_PLAN == v & DECODE_PLAN, v                                   # no bit clears it (and the device cannot add it)
        assert r & ~DECODE_PLAN == ext.resolve_impl(v & ~DECODE_PLAN, -1) & ~DECODE_PLAN, v      # it clears no other bit
        assert ext.resolve_impl(r, -1) == r, v                                          # idempotent


@pytest.mark.parametrize("impl", [REF_ROWS, REF_ROWS | DECODE_PLAN, REF, REF | DECODE_PLAN])
def test_the_reference_row_engine_keeps_its_plan_kernel_whatever_else_is_requested(ext, forced, impl):
    assert dict(kernel_counts(decode_call(ext, 4, 6, impl))) == {"plan_kernel": 1, "ref::rows_kernel": 1}
    assert not ext.decode_single_launch(1, ext.resolve_impl(impl, -1), 4, 6)           # not even with the control on (it is the fixture here)
