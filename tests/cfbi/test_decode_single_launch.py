"""Single-launch decode (``rows_kernel<true>``, ``rows_entry`` in csrc/cfbi/cfbi_rows.cuh): decode on the optimised row engine is ONE kernel launch, bit-identical to
the two-launch decode (plan kernel + row engine) and to the frozen reference.

What changed is only where a sequence's token window, slot and open-chunk fill come from and who advances ``pos``: no plan kernel runs; every head CTA derives its entry
itself (``seq_window``, the plan kernel's own function) and the H head CTAs of a sequence count their arrivals in the high bits of the ``pos`` word, the last one to arrive
stores the plain new fill. No floating-point operation moved, so every scenario compares bits with zero tolerance:

* the launch sets (a profiler check: one ``rows_kernel<true>`` and nothing else where the policy says so or the hidden calibration control is on; the plan kernel plus
  ``rows_kernel<false>`` for ``_impl=64``; the frozen reference keeps its plan kernel; mode 0 is untouched) and the policy on the grid size and the device
  (``decode_single_launch``, specified in ``decode_policy.py``),
* decode chains from every chunk phase on grids of MORE than two waves of CTAs (the late starters that the arrival count exists for), with a state pool, padding entries
  (``slot_ids == -1``) and chunk ends inside the chain, against the reference kernels and against the two-launch optimised decode,
* mode 1 with several tokens per sequence (fixed length and ``cu_seqlens``, empty sequences, padding entries),
* a CUDA graph replayed with rewritten ``slot_ids``.
The late-start race itself is stressed in ``test_decode_single_launch_variants.py`` (delayed CTAs, wrong protocols that must be caught); the logic of the protocol is checked
exhaustively in ``test_decode_pos_protocol_model.py``.
"""
import collections
import random

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from flash_kda.cfbi import alloc_state, decode, prefill, state_equal, state_mismatch

import decode_policy as dp
from helpers import make_case, n_diff, skip_or_fail
from test_vs_ref import _call_inputs, _cu

REF_ROWS, REF, ROWS_ONLY, DECODE_PLAN = 1, 3, 4, 64


def _sms():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def _cc():
    return dp.cc_number(torch.cuda.get_device_capability())


def spec_single_launch(mode, impl, N, H, sms):
    """The policy of the device that runs the test (``decode_policy.py``: the one specification of ``decode_single_launch_policy``), ``impl`` as resolve_impl returns it."""
    return dp.policy(mode, impl, N, H, sms, _cc())


def many_wave_n(H):
    """A sequence count whose grid (N x H CTAs) needs more than two waves of the two resident CTAs per SM, and still passes the gate of the policy."""
    sms = _sms()
    N = (6 * sms) // H + 1
    assert 4 * sms < N * H <= dp.MAX_CTAS_PER_SM * sms
    return N


def require_single_launch(ext, N, H):
    """The call that follows is ONE launch: by the policy of the device or because the control is on (``forced``). A device that cannot run it this way is a skip, except on B200 and
    B300, where it is a failure."""
    impl = ext.resolve_impl(0, -1)
    if not ext.decode_single_launch(1, impl, N, H):
        skip_or_fail("this device does not run decode as one launch here (impl 0 resolves to %d, grid %d x %d)" % (impl, N, H))


# ---- observing the launches --------------------------------------------------------------------------------------------------------------------------
def kernel_counts(fn, warm=True):
    """The CF-BI kernels one call of ``fn`` launches: name (argument list removed, template arguments kept: ``rows_kernel<true>``) -> launches. ``fn`` runs once before
    (JIT build, attribute set-up, the hardware check of the first call) unless ``warm`` is false."""
    if warm:
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    cnt = collections.Counter()
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA and "cfbi::" in ev.name:
            name = ev.name.split("(")[0].strip()
            if name.startswith("void "):
                name = name[5:]
            cnt[name.replace("cfbi::", "", 1)] += 1
    return cnt


def decode_fn(N, H, impl, seed=1):
    """A call that decodes one token per sequence from a zero state of N slots."""
    case = make_case(N, H, "authors", seed=seed)
    st = alloc_state(N, H)
    q, k, v, g, beta = (case[x][0] for x in ("q", "k", "v", "g", "beta"))
    out = torch.zeros_like(q)

    def fn():
        decode(q, k, v, g, beta, case["scale"], out, case["A_log"], case["dt_bias"], case["lower_bound"], st, _impl=impl)
    return fn


# ---- which kernels launch ----------------------------------------------------------------------------------------------------------------------------
def expected_decode_launches(ext, impl, N, H):
    """The launch set of a decode call of N sequences and H heads that the specification predicts for this device (the control off): one ``rows_kernel<true>`` and nothing else, or
    the plan kernel and the row engine behind it (the frozen reference where ``impl`` resolves to it)."""
    eff = ext.resolve_impl(impl, -1)
    if spec_single_launch(1, eff, N, H, _sms()):
        return {"rows_kernel<true>": 1}
    return {"plan_kernel": 1, ("ref::rows_kernel" if eff & REF_ROWS else "rows_kernel<false>"): 1}


@pytest.mark.parametrize("N, H", [(1, 12), (4, 6), (16, 3), (40, 12)])
def test_the_default_launch_set_is_what_the_specification_predicts_for_this_device(ext, N, H):
    assert dict(kernel_counts(decode_fn(N, H, 0))) == expected_decode_launches(ext, 0, N, H)    # no plan kernel, no finalize kernel where the policy says one launch


@pytest.mark.parametrize("N, H", [(1, 12), (4, 6), (16, 3), (40, 12)])
def test_the_control_makes_decode_one_launch(ext, forced, N, H):
    require_single_launch(ext, N, H)
    assert dict(kernel_counts(decode_fn(N, H, 0))) == {"rows_kernel<true>": 1}


def test_the_control_forces_a_grid_beyond_the_limit_and_nothing_else(ext, forced):
    """Above the limit the policy keeps the plan kernel; the control takes the limit and the routing set away, and nothing else: the request for two launches (bit 64) and the reference
    row engine keep theirs."""
    sms = _sms()
    N, H = dp.MAX_CTAS_PER_SM * sms // 12 + 1, 12
    require_single_launch(ext, N, H)
    assert dict(kernel_counts(decode_fn(N, H, 0))) == {"rows_kernel<true>": 1}
    for impl in (DECODE_PLAN, REF_ROWS, REF):
        assert dict(kernel_counts(decode_fn(4, 6, impl))) == expected_decode_launches(ext, impl, 4, 6) != {"rows_kernel<true>": 1}


def test_the_other_requests_keep_the_plan_kernel(ext):
    N, H = 4, 6
    for impl in (DECODE_PLAN, REF_ROWS, REF, REF | DECODE_PLAN, ROWS_ONLY | DECODE_PLAN):
        eff = ext.resolve_impl(impl, -1)
        rows = "ref::rows_kernel" if eff & REF_ROWS else "rows_kernel<false>"
        assert dict(kernel_counts(decode_fn(N, H, impl))) == {"plan_kernel": 1, rows: 1}, (impl, eff)


def test_the_gate_keeps_the_plan_kernel_for_grids_beyond_it(ext):
    H, sms = 12, _sms()
    N = (dp.MAX_CTAS_PER_SM * sms) // H + 1               # one sequence more than the gate takes
    assert not spec_single_launch(1, ext.resolve_impl(0, -1), N, H, sms)
    assert dict(kernel_counts(decode_fn(N, H, 0))) == expected_decode_launches(ext, 0, N, H) == {"plan_kernel": 1, "rows_kernel<false>": 1}
    N = (dp.MAX_CTAS_PER_SM * sms) // H                   # the largest grid the gate takes (N * H <= 8 * SMs), on a device of the routing set
    assert spec_single_launch(1, ext.resolve_impl(0, -1), N, H, sms) == (_cc() // 10 == dp.ROUTING_CC_MAJOR)
    assert dict(kernel_counts(decode_fn(N, H, 0))) == expected_decode_launches(ext, 0, N, H)


def test_prefill_keeps_its_launches(ext):
    """Mode 0 is untouched: plan kernel first, then the row engine behind it (rows_kernel<false>), K1 and K2 as before; never the single-launch kernel."""
    lens = [150, 1, 64, 129, 33, 0, 70]
    H = 3
    case = make_case(sum(lens), H, "authors", seed=3)
    cu = torch.tensor(_cu(lens), dtype=torch.int64, device="cuda")
    st = alloc_state(len(lens), H)
    out = torch.zeros_like(case["q"])

    def fn():
        prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], out, case["A_log"], case["dt_bias"], case["lower_bound"], st, cu)
    cnt = kernel_counts(fn)
    assert cnt["plan_kernel"] == 1 and cnt["rows_kernel<true>"] == 0, dict(cnt)
    assert cnt["rows_kernel<false>"] + cnt["ref::rows_kernel"] == 1, dict(cnt)


def test_the_gate_matches_the_specification(ext):
    sms = _sms()
    shapes = [(N, H) for N in (1, 2, 7, 64, 65, 255, 1024) for H in (1, 3, 12, 33, 64)]
    for H in (1, 6, 12, 32):                                   # the boundary N * H == 8 * SMs and its neighbours
        for dn in (-1, 0, 1):
            shapes.append((max(1, (dp.MAX_CTAS_PER_SM * sms) // H + dn), H))
    shapes.append((dp.MAX_CTAS_PER_SM * sms, 1))
    shapes.append((dp.MAX_CTAS_PER_SM * sms + 1, 1))
    for mode in (0, 1):
        for impl in (0, 1, 2, 3, 4, 5, 8, 16, 32, 64, 65, 66, 67, 68, 127):
            eff = ext.resolve_impl(impl, -1)
            for N, H in shapes:
                assert ext.decode_single_launch(mode, eff, N, H) == spec_single_launch(mode, eff, N, H, sms), (mode, impl, eff, N, H)
    assert not ext.decode_single_launch(1, 0, 0, 12) and not ext.decode_single_launch(1, 0, 4, 0), "no sequences or no heads: nothing to launch"


def test_the_decode_plan_bit_is_wired(ext):
    assert ext.resolve_impl(DECODE_PLAN, -1) & DECODE_PLAN, "the bit stays (it is a request of its own)"
    for v in range(128):
        r = ext.resolve_impl(v, -1)
        assert ext.resolve_impl(r, -1) == r, v
    with pytest.raises((RuntimeError, ValueError)):
        ext.resolve_impl(128, -1)
    fn = decode_fn(2, 2, 127)                                    # every bit: only bits 4 and 5 exclude each other, and that is rejected by the forward call
    with pytest.raises((RuntimeError, ValueError)):
        fn()
    decode_fn(2, 2, 64 | 15)()                                   # the new bit with the others is accepted


# ---- bit identity ------------------------------------------------------------------------------------------------------------------------------------
def step_inputs(case, cu, first, done):
    """Token ``first[n] + done[n]`` of every sequence n as [N, H, 128] decode inputs (the sequences start at ``cu[n]`` of the flattened case)."""
    idx = torch.tensor([cu[n] + first[n] + done[n] for n in range(len(first))], dtype=torch.int64, device="cuda")
    return {k: case[k][0].index_select(0, idx).contiguous() for k in ("q", "k", "v", "g", "beta")}


def decode_with(case, ins, st, ids, impl):
    """One decode call on ``st`` (updated in place); the outputs are pre-filled with a sentinel, so rows a call does not write are visible."""
    out = torch.full_like(ins["q"], 3.0)
    o32 = torch.full(ins["q"].shape, 3.0, dtype=torch.float32, device="cuda")
    decode(ins["q"], ins["k"], ins["v"], ins["g"], ins["beta"], case["scale"], out, case["A_log"], case["dt_bias"], case["lower_bound"], st, out_fp32=o32, slot_ids=ids, _impl=impl)
    return out, o32


def prefilled(H, N, S, first, steps, ids, seed):
    """A case of ``first[n] + steps`` tokens per sequence, and a state pool of S slots in which sequence n has run its first ``first[n]`` tokens (through the reference)."""
    lens = [f + steps for f in first]
    case = make_case(sum(lens), H, "authors", seed=seed)
    cu = _cu(lens)
    sub, cu_call = _call_inputs(case, cu, [0] * N, first)
    pool = alloc_state(S, H)
    prefill(sub["q"], sub["k"], sub["v"], sub["g"], sub["beta"], case["scale"], torch.zeros_like(sub["q"]), case["A_log"], case["dt_bias"], case["lower_bound"], pool, cu_call,
            slot_ids=ids, _impl=REF)
    return case, cu, pool


def plain_pos(st):
    """No arrival counter is left in the pos words: every entry is a fill 0 .. 63."""
    return bool(((st.pos >= 0) & (st.pos < 64)).all())


@pytest.mark.parametrize("other", [REF, DECODE_PLAN], ids=["reference", "two_launch"])
@pytest.mark.parametrize("pool", [False, True], ids=["compact", "pool_and_padding"])
def test_decode_chain_beyond_two_waves_is_bit_identical(ext, forced, other, pool):
    """Prefill every sequence to a different chunk phase (61, 62, 63, 0, 1, 2 mod 64: the chain crosses chunk ends), then decode with the default implementation and with
    ``other`` from copies of the same state: identical bf16 and fp32 outputs, E, valid row-cache planes and pos after every step, on a grid of more than two waves."""
    H, steps = 12, 5
    N = many_wave_n(H)
    require_single_launch(ext, N, H)
    first = [125 + (n % 6) for n in range(N)]
    rng = random.Random(11)
    S = N + 9 if pool else N
    ids = torch.tensor(rng.sample(range(S), N), dtype=torch.int32, device="cuda") if pool else None
    case, cu, st = prefilled(H, N, S, first, steps, ids, seed=21)
    st_a, st_b = st.clone(), st.clone()
    skip = [i for i in range(N) if pool and i % 7 == 3]          # padding entries of the decode batch: no tokens, no state access
    ids_call = None
    if pool:
        ids_call = ids.clone()
        ids_call[skip] = -1
    for t in range(steps):
        ins = step_inputs(case, cu, first, [t] * N)
        if t == 0:                                               # the comparison is only meaningful if the single-launch kernel is what ran (the hardware check of the first call is behind us)
            box = []
            cnt = kernel_counts(lambda: box.append(decode_with(case, ins, st_a, ids_call, 0)), warm=False)
            assert dict(cnt) == {"rows_kernel<true>": 1}, dict(cnt)
            oa, fa = box[-1]
        else:
            oa, fa = decode_with(case, ins, st_a, ids_call, 0)
        ob, fb = decode_with(case, ins, st_b, ids_call, other)
        assert n_diff(oa, ob) == 0 and n_diff(fa, fb) == 0, f"step {t}: outputs differ"
        assert state_equal(st_a, st_b), f"step {t}: state differs {state_mismatch(st_a, st_b)}"
        assert plain_pos(st_a), f"step {t}: an arrival counter is left in pos"
        if skip:
            assert bool((oa[skip] == 3.0).all()) and bool((fa[skip] == 3.0).all()), f"step {t}: a padding entry wrote its outputs"
    # the chain really crossed chunk ends: the fills after `steps` tokens are not the fills before
    assert not torch.equal(st_a.pos, st.pos)


@pytest.mark.parametrize("N, H", [(1, 1), (1, 12), (3, 7), (24, 12), (75, 12), (5, 33)])
def test_one_token_from_a_zero_state_matches_the_reference_on_every_grid_shape(ext, forced, N, H):
    require_single_launch(ext, N, H)
    ref = None
    for impl in (REF, DECODE_PLAN, 0):
        case = make_case(N, H, "heavy_tail", seed=5)
        st = alloc_state(N, H)
        ins = {k: case[k][0].contiguous() for k in ("q", "k", "v", "g", "beta")}
        out, o32 = decode_with(case, ins, st, None, impl)
        if ref is None:
            ref = (out, o32, st)
        else:
            assert n_diff(out, ref[0]) == 0 and n_diff(o32, ref[1]) == 0, f"impl {impl}: outputs differ"
            assert state_equal(st, ref[2]), f"impl {impl}: state differs {state_mismatch(st, ref[2])}"
            assert plain_pos(st)


# ---- mode 1 with several tokens per sequence ---------------------------------------------------------------------------------------------------------
def _mode1_call(case, sub, cu_call, st, ids, impl):
    out = torch.full_like(sub["q"], 3.0)
    o32 = torch.full(sub["q"].shape, 3.0, dtype=torch.float32, device="cuda")
    prefill(sub["q"], sub["k"], sub["v"], sub["g"], sub["beta"], case["scale"], out, case["A_log"], case["dt_bias"], case["lower_bound"], st, cu_call, out_fp32=o32,
            slot_ids=ids, _mode=1, _impl=impl)
    return out, o32


@pytest.mark.parametrize("impl", [0, ROWS_ONLY, DECODE_PLAN | ROWS_ONLY], ids=["mode1", "rows_only", "rows_only_plan"])
def test_fixed_length_mode1_call_matches_the_reference(ext, forced, impl):
    """No ``cu_seqlens``: every batch entry is one sequence of T tokens (``fixed_len`` = T), run through the row engine from a warm state; 60 + 7 tokens cross a chunk end."""
    N, H, T0, T1 = 40, 3, 60, 7
    case = make_case(N * (T0 + T1), H, "authors", seed=8)
    q, k, v, g, beta = (case[x].reshape(N, T0 + T1, *case[x].shape[2:]) for x in ("q", "k", "v", "g", "beta"))
    warm = {"q": q[:, :T0], "k": k[:, :T0], "v": v[:, :T0], "g": g[:, :T0], "beta": beta[:, :T0]}
    call = {"q": q[:, T0:], "k": k[:, T0:], "v": v[:, T0:], "g": g[:, T0:], "beta": beta[:, T0:]}
    warm = {x: t.contiguous() for x, t in warm.items()}
    call = {x: t.contiguous() for x, t in call.items()}
    st = alloc_state(N, H)
    prefill(warm["q"], warm["k"], warm["v"], warm["g"], warm["beta"], case["scale"], torch.zeros_like(warm["q"]), case["A_log"], case["dt_bias"], case["lower_bound"], st,
            _impl=REF)
    st_a, st_b = st.clone(), st.clone()
    oa, fa = _mode1_call(case, call, None, st_a, None, impl)
    ob, fb = _mode1_call(case, call, None, st_b, None, REF)
    assert n_diff(oa, ob) == 0 and n_diff(fa, fb) == 0 and state_equal(st_a, st_b), state_mismatch(st_a, st_b)
    assert plain_pos(st_a)


@pytest.mark.parametrize("impl", [0, DECODE_PLAN], ids=["mode1", "two_launch"])
def test_varlen_mode1_call_with_empty_sequences_and_padding_entries(ext, forced, impl):
    """``cu_seqlens`` with empty sequences and entries whose slot is -1 (they own tokens that are skipped), over a state pool warmed to different fills."""
    H, S = 2, 9
    lens = [5, 0, 70, 1, 0, 130, 64, 33]
    N = len(lens)
    ids_warm = torch.tensor([3, 6, 0, 7, 1, 5, 8, 2], dtype=torch.int32, device="cuda")
    ids_call = torch.tensor([3, 6, 0, -1, 1, 5, -1, 2], dtype=torch.int32, device="cuda")
    warm = [(n * 13) % 70 + 1 for n in range(N)]                # tokens every sequence has run before the call (fills 1 .. 70)
    total = [w + L for w, L in zip(warm, lens)]
    case = make_case(sum(total), H, "authors", seed=9)
    cu = _cu(total)
    sub0, cu0 = _call_inputs(case, cu, [0] * N, warm)
    pool = alloc_state(S, H)
    prefill(sub0["q"], sub0["k"], sub0["v"], sub0["g"], sub0["beta"], case["scale"], torch.zeros_like(sub0["q"]), case["A_log"], case["dt_bias"], case["lower_bound"], pool, cu0,
            slot_ids=ids_warm, _impl=REF)
    sub1, cu1 = _call_inputs(case, cu, warm, lens)               # the call: the next ``lens[n]`` tokens of every sequence
    st_a, st_b = pool.clone(), pool.clone()
    oa, fa = _mode1_call(case, sub1, cu1, st_a, ids_call, impl)
    ob, fb = _mode1_call(case, sub1, cu1, st_b, ids_call, REF)
    assert n_diff(oa, ob) == 0 and n_diff(fa, fb) == 0, "outputs differ"
    assert state_equal(st_a, st_b), state_mismatch(st_a, st_b)
    assert plain_pos(st_a)
    cu_l = _cu(lens)
    for n in (3, 6):                                           # the padding entries wrote nothing
        assert bool((oa[:, cu_l[n]:cu_l[n + 1]] == 3.0).all())
    assert torch.equal(st_a.pos[7], pool.pos[7]) and torch.equal(st_a.pos[8], pool.pos[8]), "the slots of the padding entries were touched"


# ---- CUDA graph --------------------------------------------------------------------------------------------------------------------------------------
def test_cuda_graph_replay_with_rewritten_slot_ids_beyond_two_waves(ext, forced):
    """A fixed-shape decode batch of more than two waves of CTAs captured once, replayed with the slot assignment rewritten every step (entries change slots, some are
    padding); every replay equals the reference decode on a copy of the pool, bit for bit, and no arrival counter is left behind. No plan buffer exists in the graph."""
    H, steps = 12, 4
    N = many_wave_n(H)
    require_single_launch(ext, N, H)
    S = N + 9
    first = [125 + (n % 6) for n in range(N)]
    rng = random.Random(17)
    slots = rng.sample(range(S), N)                            # sequence n -> pool slot (fixed for its lifetime)
    ids0 = torch.tensor(slots, dtype=torch.int32, device="cuda")
    case, cu, pool = prefilled(H, N, S, first, steps, ids0, seed=23)
    ref_pool = pool.clone()
    bufs = {k: torch.zeros(N, *case[k].shape[2:], dtype=case[k].dtype, device="cuda") for k in ("q", "k", "v", "g", "beta")}
    out = torch.zeros_like(bufs["q"])
    o32 = torch.zeros(bufs["q"].shape, dtype=torch.float32, device="cuda")
    ids = torch.full((N,), -1, dtype=torch.int32, device="cuda")

    def launch():
        decode(bufs["q"], bufs["k"], bufs["v"], bufs["g"], bufs["beta"], case["scale"], out, case["A_log"], case["dt_bias"], case["lower_bound"], pool, out_fp32=o32, slot_ids=ids)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):                               # warm-up on a scratch pool (attribute set-up, the hardware check of the first call)
        saved = pool
        pool = alloc_state(S, H)
        launch()
        pool = saved
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    saved = pool
    pool = alloc_state(S, H)
    cnt = kernel_counts(launch)                                 # on a scratch pool (every entry is padding): one kernel, so the graph holds one kernel node and no plan buffer
    pool = saved
    assert dict(cnt) == {"rows_kernel<true>": 1}, dict(cnt)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    done = [0] * N
    for t in range(steps):
        order = list(range(N))
        rng.shuffle(order)                                      # entry i serves sequence order[i]
        pad = set(rng.sample(range(N), N // 8))                 # entries that are padding this step
        for i, n in enumerate(order):
            if i in pad:
                ids[i] = -1
                for k in bufs:
                    bufs[k][i].zero_()
            else:
                ids[i] = slots[n]
                idx = cu[n] + first[n] + done[n]
                for k in bufs:
                    bufs[k][i] = case[k][0, idx]
        out.fill_(3.0)
        o32.fill_(3.0)
        graph.replay()
        ins = {k: bufs[k].clone() for k in bufs}
        ids_ref = ids.clone()
        ro, rf = decode_with(case, ins, ref_pool, ids_ref, REF)
        torch.cuda.synchronize()
        assert n_diff(out, ro) == 0 and n_diff(o32, rf) == 0, f"replay {t}: outputs differ from the reference"
        assert state_equal(pool, ref_pool), f"replay {t}: state differs {state_mismatch(pool, ref_pool)}"
        assert plain_pos(pool), f"replay {t}: an arrival counter is left in pos"
        for i, n in enumerate(order):
            if i not in pad:
                done[n] += 1
