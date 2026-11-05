"""Dense decode contract, continuation, cache and graph checks."""

import os
import pytest
import torch
from flash_kda import consistent_decode_fwd, prepare_consistent_decode, consistent_fwd
from .flashinfer_reference import step

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def case(b=4, h=12, device="cuda"):
    def rand(shape, dtype=torch.bfloat16):
        return (torch.randn(shape, device=device) * 0.2).to(dtype)

    xs = [rand((b, 1, h, 128)) for _ in range(4)]
    return dict(
        zip(("q", "k", "v", "g"), xs),
        beta=rand((b, 1, h)),
        A_log=rand((h,), torch.float32),
        dt_bias=rand((h, 128), torch.float32),
        state_in=rand((b, h, 128, 128)),
        out=torch.empty_like(xs[0]),
        state_out=torch.empty((b, h, 128, 128), device=device, dtype=torch.bfloat16),
    )


def bits(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
    assert torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def reference(c, state, out, **kwargs):
    step(
        *(c[n] for n in ("q", "k", "v", "g", "beta")),
        state,
        c["A_log"],
        c["dt_bias"],
        out,
        **kwargs,
    )


@pytest.mark.parametrize("h", [1, 3, 6, 12, 16, 32])
@pytest.mark.parametrize("b", [0, 1, 7, 16, 64, 128])
def test_dense_reference_and_cache(h, b):
    c = case(b, h)
    before = c["state_in"].clone()
    plan = prepare_consistent_decode(**c)
    expected = c["state_in"].clone()
    eo = torch.empty_like(c["out"])
    if b:
        reference(c, expected, eo)
    else:
        eo = c["out"].clone()
    plan.run()
    bits(c["out"], eo)
    bits(c["state_out"], expected)
    bits(c["state_in"], before)
    consistent_decode_fwd(**c)
    bits(c["out"], eo)
    bits(c["state_out"], expected)
    if b:
        from flash_kda._consistent_decode_kernel import compiled_decode_kernel

        misses = compiled_decode_kernel.cache_info().misses
        prepare_consistent_decode(**case(b + 1, h)).run()
        assert compiled_decode_kernel.cache_info().misses == misses


@pytest.mark.parametrize("kind", ["normal", "zero_qk", "saturated"])
@pytest.mark.parametrize("lower_bound", [-5.0, -2.5, 0.0])
def test_inplace_multistep(kind, lower_bound):
    c = case()
    c["state_out"] = c["state_in"]
    expected = c["state_in"].clone()
    eo = torch.empty_like(c["out"])
    plan = prepare_consistent_decode(**c, lower_bound=lower_bound)
    for _ in range(17):
        for n in ("q", "k", "v", "g", "beta"):
            c[n].normal_()
        if kind == "zero_qk":
            c["q"].zero_()
            c["k"].zero_()
        if kind == "saturated":
            c["g"][..., ::2] = 1000
            c["g"][..., 1::2] = -1000
            c["beta"].fill_(1000)
        plan.run()
        reference(c, expected, eo, lower_bound=lower_bound)
        bits(c["out"], eo)
        bits(c["state_out"], expected)


def test_reorder_and_head_shards():
    c = case(7, 12)
    consistent_decode_fwd(**c)
    order = torch.tensor([6, 0, 4, 2, 1, 5, 3], device="cuda")
    reordered = {
        n: (x[order].clone() if n not in ("A_log", "dt_bias") else x)
        for n, x in c.items()
    }
    consistent_decode_fwd(**reordered)
    bits(reordered["out"], c["out"][order])
    bits(reordered["state_out"], c["state_out"][order])
    for start, end in [(0, 1), (1, 4), (4, 6), (6, 12)]:
        part = {}
        for n, x in c.items():
            part[n] = (
                x[start:end]
                if n in ("A_log", "dt_bias")
                else x[:, start:end]
                if n.startswith("state_")
                else x[:, :, start:end]
            ).clone()
        consistent_decode_fwd(**part)
        bits(part["out"], c["out"][:, :, start:end])
        bits(part["state_out"], c["state_out"][:, start:end])


def test_graph_changes_contents_and_one_node(tmp_path):
    c = case()
    plan = prepare_consistent_decode(**c)
    plan.run()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        plan.run()
    graph.instantiate()
    graph.debug_dump(str(tmp_path / "decode.dot"))
    import re

    dot = (tmp_path / "decode.dot").read_text()
    assert len(re.findall(r'label="\{\s*KERNEL\b', dot)) == 1
    assert "MEMCPY" not in dot
    for _ in range(4):
        for n in ("q", "k", "v", "g", "beta", "A_log", "dt_bias", "state_in"):
            c[n].normal_(std=0.2)
        state = c["state_in"].clone()
        out = torch.empty_like(c["out"])
        reference(c, state, out)
        graph.replay()
        bits(c["out"], out)
        bits(c["state_out"], state)


@pytest.mark.parametrize("cut", [0, 1, 15, 16, 17, 32, 33])
def test_prefill_decode_continuation(cut):
    from .helpers import make_case, run

    length = 33
    c = make_case([length], 12)
    whole, final = run(c)
    state = c["state_in"].clone()
    if cut:
        xs = {n: c[n][:, :cut].clone() for n in ("q", "k", "v", "g", "beta")}
        cu = torch.tensor([0, cut], device="cuda", dtype=torch.int64)
        consistent_fwd(
            **xs,
            A_log=c["A_log"],
            dt_bias=c["dt_bias"],
            cu_seqlens=cu,
            state_in=state,
            out=torch.empty_like(xs["q"]),
            state_out=state,
        )
    for t in range(cut, length):
        xs = {n: c[n][:, t : t + 1].clone() for n in ("q", "k", "v", "g", "beta")}
        out = torch.empty_like(xs["q"])
        consistent_decode_fwd(
            **xs,
            A_log=c["A_log"],
            dt_bias=c["dt_bias"],
            state_in=state,
            out=out,
            state_out=state,
        )
        bits(out, whole[:, t : t + 1])
    bits(state, final)


@pytest.mark.parametrize("length", [4096, 32768])
def test_long_prefill_decode(length):
    if os.environ.get("FLASH_KDA_LONG_TESTS") != "1":
        pytest.skip("enable long tests")
    from .helpers import make_case, run

    c = make_case([length], 12)
    whole, final = run(c, lower_bound=0.0)
    x = case(1, 12)
    x["A_log"].copy_(c["A_log"])
    x["dt_bias"].copy_(c["dt_bias"])
    x["state_in"].copy_(c["state_in"])
    x["state_out"] = x["state_in"]
    plan = prepare_consistent_decode(**x, lower_bound=0.0)
    for t in range(length):
        for n in ("q", "k", "v", "g", "beta"):
            x[n].copy_(c[n][:, t : t + 1])
        plan.run()
        bits(x["out"], whole[:, t : t + 1])
    bits(x["state_out"], final)


def test_invalid_contract():
    c = case()
    bad = case()
    bad["q"] = bad["q"].float()
    with pytest.raises(ValueError, match="dtype"):
        prepare_consistent_decode(**bad)
    shifted = torch.empty(c["q"].numel() + 1, device="cuda", dtype=torch.bfloat16)[
        1:
    ].view_as(c["q"])
    with pytest.raises(ValueError, match="aligned"):
        prepare_consistent_decode(**dict(c, q=shifted))
    with pytest.raises(ValueError, match="overlapping"):
        prepare_consistent_decode(**dict(c, out=c["q"]))
    with pytest.raises(ValueError, match="finite"):
        prepare_consistent_decode(**c, scale=float("nan"))
    with pytest.raises(ValueError, match="positive"):
        prepare_consistent_decode(**c, eps=0.0)
    with pytest.raises(ValueError):
        prepare_consistent_decode(**c, lower_bound=-6.0)
    c["q"].requires_grad_()
    with pytest.raises(RuntimeError, match="no_grad"):
        prepare_consistent_decode(**c)
    with torch.no_grad():
        p = prepare_consistent_decode(**c)
        p.run()
    with pytest.raises(RuntimeError, match="no_grad"):
        p.run()


@pytest.mark.parametrize("h", [12, 16, 32])
def test_two_gpu_whole_head_shards(h):
    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices required")
    with torch.cuda.device(0):
        c = case(4, h, "cuda:0")
        prepare_consistent_decode(**c).run()
    for device, (start, end) in enumerate([(0, h // 2), (h // 2, h)]):
        with torch.cuda.device(device):
            part = {}
            for n, x in c.items():
                view = (
                    x[start:end]
                    if n in ("A_log", "dt_bias")
                    else x[:, start:end]
                    if n.startswith("state_")
                    else x[:, :, start:end]
                )
                part[n] = view.to(device=f"cuda:{device}").clone().contiguous()
            prepare_consistent_decode(**part).run()
            bits(part["out"].to("cuda:0"), c["out"][:, :, start:end])
            bits(part["state_out"].to("cuda:0"), c["state_out"][:, start:end])


def test_two_gpu_prefill_state_handoff():
    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices required")
    from .helpers import make_case, run

    with torch.cuda.device(0):
        c = make_case([33], 12)
        whole, final = run(c)
        prefix = {n: c[n][:, :17].clone() for n in ("q", "k", "v", "g", "beta")}
        state = c["state_in"].clone()
        consistent_fwd(
            **prefix,
            A_log=c["A_log"],
            dt_bias=c["dt_bias"],
            state_in=state,
            state_out=state,
            out=torch.empty_like(prefix["q"]),
            cu_seqlens=torch.tensor([0, 17], device="cuda:0", dtype=torch.int64),
        )
    with torch.cuda.device(1):
        state = state.to("cuda:1")
        x = case(1, 12, "cuda:1")
        x["state_in"] = state
        x["state_out"] = state
        x["A_log"].copy_(c["A_log"])
        x["dt_bias"].copy_(c["dt_bias"])
        plan = prepare_consistent_decode(**x)
        for token in range(17, 33):
            for n in ("q", "k", "v", "g", "beta"):
                x[n].copy_(c[n][:, token : token + 1])
            plan.run()
            bits(x["out"].to("cuda:0"), whole[:, token : token + 1])
        bits(state.to("cuda:0"), final)
