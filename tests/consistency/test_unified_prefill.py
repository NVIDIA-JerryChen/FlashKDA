"""Unified prefill must preserve the existing recurrent continuation contract."""

import pytest
import torch

from flash_kda import (
    consistent_fwd,
    prepare_consistent_prefill,
    prepare_unified_prefill,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def inputs(b, length, h):
    xs = [
        torch.randn(b, length, h, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(4)
    ]
    xs.append(torch.randn(b, length, h, device="cuda", dtype=torch.bfloat16))
    a = torch.zeros(h, device="cuda")
    bias = torch.randn(h, 128, device="cuda")
    state = (torch.randn(b, h, 128, 128, device="cuda") * 0.1).bfloat16()
    return xs, a, bias, state


def equal_bits(a, b):
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


@torch.no_grad()
def test_dense_segmentation_and_runtime_compile_cache():
    from flash_kda._prefill_window.shared_stage_scan import compiled_scan_kernel
    from flash_kda._prefill_window.aligned_induction_coeff import compiled_coeff_kernel

    torch.manual_seed(351)
    expected_misses = None
    for b, h, length in [
        (1, 3, 257),
        (3, 12, 129),
        (4, 12, 129),
        (2, 16, 8),
        (1, 32, 17),
    ]:
        xs, a, bias, initial = inputs(b, length, h)
        out, state = torch.empty_like(xs[0]), torch.empty_like(initial)
        plan = prepare_unified_prefill(*xs, a, bias, initial, out, state)
        plan.run()
        misses = (
            compiled_scan_kernel.cache_info().misses,
            compiled_coeff_kernel.cache_info().misses,
        )
        if expected_misses is None:
            expected_misses = misses
        else:
            assert misses == expected_misses
        expected, expected_state = torch.empty_like(out), torch.empty_like(state)
        prepare_consistent_prefill(
            *xs, a, bias, initial, expected, expected_state, mode="direct"
        ).run()
        equal_bits(out, expected)
        equal_bits(state, expected_state)
        current = initial.clone()
        outputs = []
        start = 0
        for end in sorted(set([0, 1, 7, min(128, length), length])):
            step = [
                x[:, start:end].clone(memory_format=torch.contiguous_format) for x in xs
            ]
            o = torch.empty_like(step[0])
            p = prepare_unified_prefill(*step, a, bias, current, o, current)
            p.run()
            outputs.append(o)
            start = end
        equal_bits(torch.cat(outputs, 1), out)
        equal_bits(current, state)
        assert (
            compiled_scan_kernel.cache_info().misses,
            compiled_coeff_kernel.cache_info().misses,
        ) == misses
        assert plan.workspace_bytes <= 256 * 1024 * 1024


@torch.no_grad()
def test_graph_varlen_dynamic_metadata_and_capacity_tail():
    xs, a, bias, _ = inputs(1, 512, 3)
    initial = torch.randn(4, 3, 128, 128, device="cuda").bfloat16() * 0.1
    cu = torch.tensor([0, 0, 1, 128, 512], device="cuda", dtype=torch.int32)
    out = torch.full_like(xs[0], float("nan"))
    state = torch.empty_like(initial)
    plan = prepare_unified_prefill(
        *xs,
        a,
        bias,
        initial,
        out,
        state,
        cu_seqlens=cu,
        workspace_limit_bytes=1024 * 1024
    )
    plan.run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run()
    cu.copy_(torch.tensor([0, 17, 17, 31, 257], device="cuda", dtype=torch.int32))
    out.fill_(float("nan"))
    graph.replay()
    reference = torch.empty_like(out)
    reference_state = torch.empty_like(state)
    consistent_fwd(*xs, a, bias, cu.long(), initial, reference, reference_state)
    equal_bits(out[:, :257], reference[:, :257])
    equal_bits(state, reference_state)
    assert torch.isnan(out[:, 257:]).all()
    assert plan.workspace_bytes <= 1024 * 1024


@torch.no_grad()
def test_checkpoint_prefix_bridge():
    xs, a, bias, initial = inputs(2, 8, 3)
    indices = torch.randperm(16, device="cuda", dtype=torch.int32).view(2, 8)
    cp = torch.empty(16, 3, 128, 128, device="cuda", dtype=torch.bfloat16)
    reference = torch.empty_like(xs[0])
    prepare_consistent_prefill(
        *xs,
        a,
        bias,
        initial,
        reference,
        None,
        checkpoint_states=cp,
        checkpoint_indices=indices
    ).run()
    for accepted in [1, 4, 7, 8]:
        step = [
            x[:, :accepted].clone(memory_format=torch.contiguous_format) for x in xs
        ]
        out, state = torch.empty_like(step[0]), torch.empty_like(initial)
        prepare_unified_prefill(*step, a, bias, initial, out, state).run()
        equal_bits(out, reference[:, :accepted])
        equal_bits(state, cp[indices[:, accepted - 1].long()])


@torch.no_grad()
def test_empty_batch_and_invalid_descriptors():
    xs, a, bias, state = inputs(0, 0, 3)
    out, final = torch.empty_like(xs[0]), torch.empty_like(state)
    prepare_unified_prefill(*xs, a, bias, state, out, final).run()
    empty_state = state
    xs, a, bias, _ = inputs(1, 16, 3)
    untouched = torch.full_like(xs[0], float("nan"))
    prepare_unified_prefill(
        *xs,
        a,
        bias,
        empty_state,
        untouched,
        torch.empty_like(empty_state),
        cu_seqlens=torch.zeros(1, device="cuda", dtype=torch.int32)
    ).run()
    assert torch.isnan(untouched).all()
    xs, a, bias, state = inputs(1, 16, 3)
    out, final = torch.empty_like(xs[0]), torch.empty_like(state)
    with pytest.raises(ValueError, match="workspace"):
        prepare_unified_prefill(
            *xs, a, bias, state, out, final, workspace_limit_bytes=0
        )
    with pytest.raises(ValueError, match="dtype"):
        prepare_unified_prefill(
            *xs,
            a,
            bias,
            state,
            out,
            final,
            cu_seqlens=torch.tensor([0, 16], device="cuda", dtype=torch.int64)
        )
    with pytest.raises(ValueError, match="overlapping"):
        prepare_unified_prefill(*xs, a, bias, state, xs[0], final)
    # Reject overflow in the device-side ceiling division without allocating data.
    huge = torch.empty(1, 2**31 - 127, 3, 128, device="meta", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Int32"):
        prepare_unified_prefill(huge, *xs[1:], a, bias, state, out, final)
