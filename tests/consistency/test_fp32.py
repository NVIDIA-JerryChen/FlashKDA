"""FP32-state public API, continuation, checkpoints and graph regressions."""

import itertools

import pytest
import torch

from flash_kda import prepare_fp32_decode, prepare_fp32_prefill
from .helpers import make_case, select

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires SM100/SM103 CUDA"
)


def bits(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def case(lengths, h=3, padding=0):
    x = make_case(lengths, h=h, padding=padding)
    x["state_in"] = x["state_in"].float()
    x["cu_seqlens"] = x["cu_seqlens"].int()
    return x


def prefill(x, **kwargs):
    out, state = torch.full_like(x["q"], 31), torch.empty_like(x["state_in"])
    p = prepare_fp32_prefill(**x, out=out, state_out=state, **kwargs)
    p.run()
    return out, state, p


def segment(x, lengths, starts, ends, state=None):
    y, ls = select(x, lengths, list(zip(starts, ends)), state=state)
    y["cu_seqlens"] = y["cu_seqlens"].int()
    return y, ls


@pytest.mark.parametrize("h", [1, 3, 6, 12, 16, 32])
def test_arbitrary_splits(h):
    lengths = [0, 1, 7, 8, 16, 17, 127, 128, 129, 257]
    x = case(lengths, h)
    expected, terminal, _ = prefill(x)
    state = x["state_in"].clone()
    pieces = [[] for _ in lengths]
    offsets = [0] * len(lengths)
    for step in itertools.cycle([1, 8, 31, 128]):
        ends = [min(l, s + step) for l, s in zip(lengths, offsets)]
        y, ls = segment(x, lengths, offsets, ends, state)
        out, state, _ = prefill(y)
        cursor = 0
        for i, length in enumerate(ls):
            pieces[i].append(out[:, cursor : cursor + length].clone())
            cursor += length
        offsets = ends
        if offsets == lengths:
            break
    bits(torch.cat([torch.cat(p, 1) for p in pieces], 1), expected)
    bits(state, terminal)


@pytest.mark.parametrize("b,h", [(1, 1), (3, 3), (5, 6), (64, 12), (64, 16), (64, 32)])
def test_q8_all_checkpoints_and_acceptance(b, h):
    x = case([8] * b, h)
    dense = {k: x[k].view(b, 8, *x[k].shape[2:]) for k in ("q", "k", "v", "g", "beta")}
    dense.update({k: x[k] for k in ("A_log", "dt_bias", "state_in")})
    out = torch.empty_like(dense["q"])
    checkpoints = torch.empty(b, 8, h, 128, 128, device="cuda", dtype=torch.float32)
    plan = prepare_fp32_decode(**dense, out=out, checkpoint_states=checkpoints)
    got, final = plan.run()
    expected, terminal, _ = prefill(x)
    bits(got.view_as(expected), expected)
    bits(final, terminal)
    state = x["state_in"]
    for t in range(8):
        token = {
            k: dense[k][:, t : t + 1].clone(memory_format=torch.contiguous_format)
            for k in ("q", "k", "v", "g", "beta")
        }
        token.update(A_log=x["A_log"], dt_bias=x["dt_bias"], state_in=state)
        output, next_state = torch.empty_like(token["q"]), torch.empty_like(state)
        prepare_fp32_decode(**token, out=output, state_out=next_state).run()
        bits(output[:, 0], out[:, t])
        bits(next_state, checkpoints[:, t])
        state = next_state
    # Each request accepts a different prefix, including zero and all eight.
    accepted = [i % 9 for i in range(b)]
    indices = torch.tensor(accepted, device="cuda")
    selected = torch.where(
        (indices == 0)[:, None, None, None],
        x["state_in"],
        checkpoints[torch.arange(b, device="cuda"), (indices - 1).clamp_min(0)],
    ).contiguous()
    prefix, _ = segment(x, [8] * b, [0] * b, accepted)
    _, expected_state, _ = prefill(prefix)
    bits(selected, expected_state)
    suffix, _ = segment(x, [8] * b, accepted, [8] * b, selected)
    _, continued, _ = prefill(suffix)
    bits(continued, terminal)


@pytest.mark.parametrize("lengths", [[], [0], [0, 0, 0], [0, 1, 8, 129]])
def test_empty_padding_and_inplace(lengths):
    x = case(lengths, padding=13)
    out, terminal, _ = prefill(x)
    assert (out[:, sum(lengths) :] == 31).all()
    y = dict(x, state_in=x["state_in"].clone())
    oo = torch.full_like(out, 31)
    prepare_fp32_prefill(**y, out=oo, state_out=y["state_in"]).run()
    bits(oo, out)
    bits(y["state_in"], terminal)


def test_dynamic_graph_and_workspace():
    lengths = [0, 17, 129, 258]
    x = case(lengths, h=12)
    _, _, plan = prefill(x, workspace_limit_bytes=8 * 1024 * 1024)
    assert plan.workspace_bytes <= 8 * 1024 * 1024
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run()
    for ls in [[101, 101, 101, 101], [404, 0, 0, 0], [0, 0, 0, 0]]:
        y = case(ls, h=12, padding=404 - sum(ls))
        for key in x:
            x[key].copy_(y[key])
        graph.replay()
        expected, state, _ = prefill(y)
        bits(plan.out[:, : sum(ls)], expected[:, : sum(ls)])
        bits(plan.state_out, state)


def test_api_errors_and_empty_decode():
    x = case([8], h=3)
    for patch in [
        dict(state_in=x["state_in"].bfloat16()),
        dict(cu_seqlens=x["cu_seqlens"].long()),
    ]:
        with pytest.raises(ValueError):
            prefill(dict(x, **patch))
    with pytest.raises(ValueError):
        prefill(x, workspace_limit_bytes=0)
    with pytest.raises(ValueError):
        prefill(x, eps=0)
    with pytest.raises(ValueError):
        prepare_fp32_prefill(**x, out=x["q"], state_out=torch.empty_like(x["state_in"]))
    for q in [1, 8]:
        d = {
            k: torch.empty(
                (0, q, 3, 128) if k != "beta" else (0, q, 3),
                device="cuda",
                dtype=torch.bfloat16,
            )
            for k in ("q", "k", "v", "g", "beta")
        }
        d.update(A_log=x["A_log"], dt_bias=x["dt_bias"], state_in=x["state_in"][:0])
        kw = (
            dict(state_out=torch.empty_like(d["state_in"]))
            if q == 1
            else dict(
                checkpoint_states=torch.empty(
                    0, 8, 3, 128, 128, device="cuda", dtype=torch.float32
                )
            )
        )
        plan = prepare_fp32_decode(**d, out=torch.empty_like(d["q"]), **kw)
        a, b = plan.run()
        assert a.numel() == b.numel() == 0 and plan.workspace_bytes == 0


@pytest.mark.parametrize(
    "scale,eps,lower_bound", [(0.125, 0.01, -1.0), (0.0, 1e-6, 0.0)]
)
def test_scalars_q8_graph_and_cache(scale, eps, lower_bound):
    from flash_kda._fp32.decode_q8 import compiled_q8_kernel

    kwargs = dict(scale=scale, eps=eps, lower_bound=lower_bound)
    for b, h in [(3, 3), (5, 7)]:
        x = case([8] * b, h)
        dense = {
            k: x[k].view(b, 8, *x[k].shape[2:]) for k in ("q", "k", "v", "g", "beta")
        }
        dense.update({k: x[k] for k in ("A_log", "dt_bias", "state_in")})
        checkpoints = torch.empty(b, 8, h, 128, 128, device="cuda", dtype=torch.float32)
        out = torch.empty_like(dense["q"])
        plan = prepare_fp32_decode(
            **dense, out=out, checkpoint_states=checkpoints, **kwargs
        )
        plan.run()
        before = compiled_q8_kernel.cache_info().misses
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            plan.run()
        x["g"].fill_(-20)
        graph.replay()
        expected, state, _ = prefill(x, **kwargs)
        bits(out.view_as(expected), expected)
        bits(checkpoints[:, -1], state)
        order = list(reversed(range(b)))
        y = {
            k: dense[k][order].contiguous()
            for k in ("q", "k", "v", "g", "beta", "state_in")
        }
        y.update(A_log=x["A_log"], dt_bias=x["dt_bias"])
        oo, cc = torch.empty_like(out), torch.empty_like(checkpoints)
        prepare_fp32_decode(**y, out=oo, checkpoint_states=cc, **kwargs).run()
        bits(oo, out[order])
        bits(cc, checkpoints[order])
        assert compiled_q8_kernel.cache_info().misses == before
