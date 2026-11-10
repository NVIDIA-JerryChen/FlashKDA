"""Indexed per-token checkpoints and exact accepted-prefix continuation."""

import pytest
import torch
from flash_kda import prepare_consistent_prefill, consistent_prefill_fwd
from .test_prefill import case, oracle
from .test_decode import bits
from .flashinfer_reference import step

NAMES = ("q", "k", "v", "g", "beta")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def fixture(b=3, h=12, length=8):
    torch.manual_seed(20261105 + h + b + length)
    c = case(b, h, length)
    for n in NAMES:
        c[n].normal_()
    c["A_log"].uniform_(-1, 1)
    c["dt_bias"].normal_()
    return c


def pool_for(c, extra=7):
    b, length, h, _ = c["q"].shape
    p = b * length + extra
    pool = torch.full(
        (p, h, 128, 128), -7.0, device=c["q"].device, dtype=torch.bfloat16
    )
    slots = (
        torch.randperm(p, device=c["q"].device, dtype=torch.int32)[: b * length]
        .reshape(b, length)
        .contiguous()
    )
    return pool, slots


def reference(c, **kw):
    state = c["state_in"].clone()
    outputs, states = [], []
    for t in range(c["q"].shape[1]):
        xs = [c[n][:, t : t + 1].clone() for n in NAMES]
        o = torch.empty_like(xs[0])
        step(*xs, state, c["A_log"], c["dt_bias"], o, **kw)
        outputs.append(o)
        states.append(state.clone())
    return torch.cat(outputs, 1), torch.stack(states, 1)


@pytest.mark.parametrize("h", [1, 3, 6, 12, 16, 32])
@pytest.mark.parametrize("mode", ["direct", "staged", "auto"])
@pytest.mark.parametrize("final", [False, True])
def test_every_checkpoint(h, mode, final):
    c = fixture(1 if h < 12 else 7, h)
    expected, states = reference(c)
    initial = c["state_in"].clone()
    pool, slots = pool_for(c)
    before = pool.clone()
    indices = slots.clone()
    c["state_out"] = c["state_out"] if final else None
    plan = prepare_consistent_prefill(
        **c,
        mode=mode,
        checkpoint_states=pool,
        checkpoint_indices=slots,
        workspace_limit_bytes=c["q"].shape[0] * h * 1568 * 3
    )
    o, s = plan.run()
    bits(o, expected)
    bits(pool[slots.long()], states)
    before[slots.long()] = states
    bits(pool, before)
    bits(c["state_in"], initial)
    bits(slots, indices)
    assert plan.checkpoint_states is pool and plan.checkpoint_indices is slots
    if final:
        bits(s, states[:, -1])
    else:
        assert s is None


@pytest.mark.parametrize("length", [1, 2, 7, 8, 9, 16, 17, 65])
@pytest.mark.parametrize("mode", ["direct", "staged", "auto"])
def test_lengths_and_math(length, mode):
    c = fixture(2, 12, length)
    c["g"][..., ::2] = 1000
    c["g"][..., 1::2] = -1000
    c["beta"][..., ::2] = 1000
    eo, es = reference(c, scale=0.2, eps=1e-4, lower_bound=-2.5)
    pool, slots = pool_for(c)
    consistent_prefill_fwd(
        **c,
        checkpoint_states=pool,
        checkpoint_indices=slots,
        mode=mode,
        scale=0.2,
        eps=1e-4,
        lower_bound=-2.5,
        workspace_limit_bytes=2 * 12 * 1568 * 7
    )
    bits(c["out"], eo)
    bits(pool[slots.long()], es)
    bits(c["state_out"], es[:, -1])


@pytest.mark.parametrize("h", [12, 16, 32])
def test_all_accept_counts_and_reorder(h):
    c = fixture(9, h)
    eo, es = reference(c)
    pool, slots = pool_for(c)
    prepare_consistent_prefill(
        **dict(c, state_out=None), checkpoint_states=pool, checkpoint_indices=slots
    ).run()
    bits(c["out"], eo)
    bits(pool[slots.long()], es)
    accepted = torch.arange(9, device="cuda")
    rows = torch.arange(9, device="cuda")
    last = (accepted - 1).clamp_min(0)
    selected = torch.where(
        (accepted > 0)[:, None, None, None],
        pool[slots[rows, last].long()],
        c["state_in"],
    )
    expected = torch.where(
        (accepted > 0)[:, None, None, None], es[rows, last], c["state_in"]
    )
    order = torch.tensor([8, 0, 4, 2, 6, 1, 7, 3, 5], device="cuda")
    follow = fixture(9, h, 5)
    follow["state_in"] = expected[order].clone()
    yo, ys = reference(follow)
    follow["state_in"] = selected[order].clone()
    prepare_consistent_prefill(**follow).run()
    bits(follow["out"], yo)
    bits(follow["state_out"], ys[:, -1])
    # The rejected suffix did not replace any request's accepted state.
    bits(c["state_in"], fixture(9, h)["state_in"])


@pytest.mark.parametrize("mode", ["direct", "staged", "auto"])
def test_graph_dynamic_slots_and_cache(mode):
    c = fixture(3, 12)
    pool, slots = pool_for(c, 17)
    plan = prepare_consistent_prefill(
        **dict(c, state_out=None),
        mode=mode,
        checkpoint_states=pool,
        checkpoint_indices=slots,
        workspace_limit_bytes=3 * 12 * 1568 * 3
    )
    plan.run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run()
    for _ in range(3):
        for n in (*NAMES, "state_in", "A_log", "dt_bias"):
            c[n].normal_(std=0.2)
        slots.copy_(
            torch.randperm(pool.shape[0], device="cuda", dtype=torch.int32)[
                :24
            ].reshape(3, 8)
        )
        pool.fill_(-7)
        before = pool.clone()
        eo, es = reference(c)
        graph.replay()
        bits(c["out"], eo)
        before[slots.long()] = es
        bits(pool, before)
    from flash_kda._consistent_prefill_kernel import compiled_prefill_checkpoints
    from flash_kda._consistent_prefill_staged import (
        compiled_scan_checkpoints,
        compiled_coeff_kernel,
    )

    caches = [
        compiled_prefill_checkpoints,
        compiled_scan_checkpoints,
        compiled_coeff_kernel,
    ]
    before = [f.cache_info().misses for f in caches]
    c2 = fixture(4, 12, 17)
    p2, i2 = pool_for(c2, 31)
    prepare_consistent_prefill(
        **dict(c2, state_out=None),
        mode=mode,
        checkpoint_states=p2,
        checkpoint_indices=i2
    ).run()
    assert before == [f.cache_info().misses for f in caches]


@pytest.mark.parametrize("h", [12, 16, 32])
def test_two_gpu_head_shards(h):
    if torch.cuda.device_count() < 2:
        pytest.skip("two GPUs required")
    c = fixture(2, h)
    pool, slots = pool_for(c)
    prepare_consistent_prefill(
        **dict(c, state_out=None), checkpoint_states=pool, checkpoint_indices=slots
    ).run()
    torch.cuda.synchronize(0)
    for lo, hi in [(0, h // 2), (h // 2, h)]:
        with torch.cuda.device(1):
            d = {
                n: (
                    x[lo:hi]
                    if n in ("A_log", "dt_bias")
                    else x[:, lo:hi] if n.startswith("state_") else x[:, :, lo:hi]
                )
                .to("cuda:1")
                .contiguous()
                for n, x in c.items()
            }
            local_pool = torch.empty(
                (pool.shape[0], hi - lo, 128, 128),
                device="cuda:1",
                dtype=torch.bfloat16,
            )
            local_slots = slots.to("cuda:1")
            prepare_consistent_prefill(
                **dict(d, state_out=None),
                checkpoint_states=local_pool,
                checkpoint_indices=local_slots
            ).run()
            bits(d["out"].to("cuda:0"), c["out"][:, :, lo:hi])
            bits(
                local_pool[local_slots.long()].to("cuda:0"),
                pool[slots.long()][:, :, lo:hi],
            )


@pytest.mark.parametrize("h", [12, 16, 32])
@pytest.mark.parametrize("length", [4096, 32768])
def test_long_prefill_then_checkpoints(h, length):
    c = fixture(1, h, length)
    eo, es = oracle(c)
    prepare_consistent_prefill(**c).run()
    bits(c["out"], eo)
    bits(c["state_out"], es)
    follow = fixture(1, h, 8)
    follow["state_in"] = c["state_out"].clone()
    yo, ys = reference(follow)
    pool, slots = pool_for(follow)
    prepare_consistent_prefill(
        **dict(follow, state_out=None), checkpoint_states=pool, checkpoint_indices=slots
    ).run()
    bits(follow["out"], yo)
    bits(pool[slots.long()], ys)


@pytest.mark.parametrize("b,length", [(0, 8), (2, 0), (0, 0)])
@pytest.mark.parametrize("final", [False, True])
def test_empty(b, length, final):
    c = fixture(b, 12, length)
    pool, slots = pool_for(c, 0)
    o, s = prepare_consistent_prefill(
        **dict(c, state_out=c["state_out"] if final else None),
        checkpoint_states=pool,
        checkpoint_indices=slots
    ).run()
    assert o.numel() == 0
    if final and b:
        bits(s, c["state_in"])


def test_checkpoint_errors():
    c = fixture()
    pool, slots = pool_for(c)
    changes = [
        dict(checkpoint_states=pool),
        dict(checkpoint_indices=slots),
        dict(checkpoint_states=pool.float(), checkpoint_indices=slots),
        dict(checkpoint_states=pool, checkpoint_indices=slots.long()),
        dict(checkpoint_states=pool, checkpoint_indices=slots[:, :7].clone()),
        dict(checkpoint_states=pool[:2], checkpoint_indices=slots),
        dict(checkpoint_states=pool[:, 0].clone(), checkpoint_indices=slots),
    ]
    for change in changes:
        with pytest.raises(ValueError):
            prepare_consistent_prefill(**c, **change)
    with pytest.raises(ValueError):
        prepare_consistent_prefill(**dict(c, state_out=None))
    with pytest.raises(ValueError):
        prepare_consistent_prefill(
            **dict(c, state_out=c["state_in"]),
            checkpoint_states=pool,
            checkpoint_indices=slots
        )
    # Input/output aliases through a shared state-pool allocation are forbidden.
    with pytest.raises(ValueError):
        prepare_consistent_prefill(
            **dict(c, state_in=pool[:3]),
            checkpoint_states=pool,
            checkpoint_indices=slots
        )


@pytest.mark.parametrize("cut", [0, 1, 4, 7, 8, 9, 16, 17])
def test_open_gate_segment_checkpoints(cut):
    c = fixture(2, 12, 17)
    c["g"].fill_(-1000)
    eo, es = reference(c)
    state = c["state_in"].clone()
    for lo, hi in [(0, cut), (cut, 17)]:
        d = {
            n: (x[:, lo:hi].clone() if n in (*NAMES, "out") else x.clone())
            for n, x in c.items()
        }
        d["state_in"] = state.clone()
        pool, slots = pool_for(d)
        prepare_consistent_prefill(
            **d, checkpoint_states=pool, checkpoint_indices=slots
        ).run()
        bits(d["out"], eo[:, lo:hi])
        bits(pool[slots.long()], es[:, lo:hi])
        state = d["state_out"]
    bits(state, es[:, -1])
