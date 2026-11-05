"""Fused recurrent prefill: exact continuation, workspace and graph contracts."""

import os
import re

import pytest
import torch
from flash_kda import (
    prepare_consistent_prefill,
    prepare_consistent_decode,
    consistent_fwd,
)
from .test_decode import bits
from .flashinfer_reference import step

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def case(b=2, h=12, length=33, device="cuda"):
    def rand(shape, dtype=torch.bfloat16):
        return (torch.randn(shape, device=device) * 0.2).to(dtype)

    xs = [rand((b, length, h, 128)) for _ in range(4)]
    return dict(
        zip(("q", "k", "v", "g"), xs),
        beta=rand((b, length, h)),
        A_log=rand((h,), torch.float32),
        dt_bias=rand((h, 128), torch.float32),
        state_in=rand((b, h, 128, 128)),
        state_out=rand((b, h, 128, 128)),
        out=torch.empty_like(xs[0]),
    )


def oracle(c, **kw):
    b, l, h, _ = c["q"].shape
    if b == 0:
        return torch.empty_like(c["out"]), c["state_in"].clone()
    xs = {
        n: c[n].reshape(1, b * l, *c[n].shape[2:]) for n in ("q", "k", "v", "g", "beta")
    }
    out = torch.empty_like(c["out"])
    state = torch.empty_like(c["state_in"])
    consistent_fwd(
        **xs,
        A_log=c["A_log"],
        dt_bias=c["dt_bias"],
        cu_seqlens=torch.arange(b + 1, device=c["q"].device, dtype=torch.int64) * l,
        state_in=c["state_in"],
        out=out.reshape_as(xs["q"]),
        state_out=state,
        **kw,
    )
    return out, state


@pytest.mark.parametrize("h", [1, 3, 6, 12, 16, 32])
@pytest.mark.parametrize(
    "b,length", [(0, 17), (1, 0), (1, 1), (1, 15), (2, 16), (3, 17), (7, 65)]
)
@pytest.mark.parametrize("mode", ["direct", "staged"])
def test_reference(b, h, length, mode):
    c = case(b, h, length)
    initial = c["state_in"].clone()
    expected = oracle(c)
    plan = prepare_consistent_prefill(**c, mode=mode)
    plan.run()
    bits(c["out"], expected[0])
    bits(c["state_out"], expected[1])
    bits(c["state_in"], initial)


@pytest.mark.parametrize("lower_bound", [-5.0, -2.5, 0.0])
@pytest.mark.parametrize("kind", ["normal", "zero_qk", "saturated"])
def test_pinned_fi_steps(kind, lower_bound):
    c = case(2, 12, 35)
    if kind == "zero_qk":
        c["q"].zero_()
        c["k"].zero_()
    if kind == "saturated":
        c["g"][..., ::2] = 1000
        c["g"][..., 1::2] = -1000
        c["beta"][..., ::2] = 1000
        c["beta"][..., 1::2] = -1000
    state = c["state_in"].clone()
    for t in range(35):
        xs = [c[n][:, t : t + 1].clone() for n in ("q", "k", "v", "g", "beta")]
        o = torch.empty_like(xs[0])
        step(*xs, state, c["A_log"], c["dt_bias"], o, lower_bound=lower_bound)
        # Every prefix checks the full live state, including pre-processing boundaries.
        short = {
            n: (
                x[:, : t + 1].clone() if n in ("q", "k", "v", "g", "beta", "out") else x
            )
            for n, x in c.items()
        }
        prepare_consistent_prefill(
            **short,
            mode="staged",
            lower_bound=lower_bound,
            workspace_limit_bytes=2 * 12 * 392 * 4 * 7,
        ).run()
        bits(short["out"][:, -1:], o)
        bits(short["state_out"], state)


@pytest.mark.parametrize("cut", [0, 1, 15, 16, 17, 32, 33])
def test_segments_and_decode(cut):
    c = case()
    expected = oracle(c)
    state = c["state_in"].clone()
    for lo, hi, mode in [(0, cut, "direct"), (cut, 33, "staged")]:
        part = {
            n: (x[:, lo:hi].clone() if n in ("q", "k", "v", "g", "beta", "out") else x)
            for n, x in c.items()
        }
        part.update(state_in=state, state_out=state)
        prepare_consistent_prefill(**part, mode=mode).run()
        bits(part["out"], expected[0][:, lo:hi])
    bits(state, expected[1])
    one = case(2, 12, 1)
    one.update(state_in=state, state_out=state)
    ref = state.clone()
    eo = torch.empty_like(one["out"])
    step(
        *(one[n] for n in ("q", "k", "v", "g", "beta")),
        ref,
        one["A_log"],
        one["dt_bias"],
        eo,
    )
    prepare_consistent_decode(**one).run()
    bits(one["out"], eo)
    bits(state, ref)


def test_reorder_shards_and_cache():
    c = case(7, 12, 65)
    prepare_consistent_prefill(**c).run()
    order = torch.tensor([6, 0, 4, 2, 1, 5, 3], device="cuda")
    part = {
        n: (x[order].clone() if n not in ("A_log", "dt_bias") else x)
        for n, x in c.items()
    }
    prepare_consistent_prefill(**part).run()
    bits(part["out"], c["out"][order])
    bits(part["state_out"], c["state_out"][order])
    for lo, hi in [(0, 1), (1, 4), (4, 6), (6, 12)]:
        part = {
            n: (
                x[lo:hi]
                if n in ("A_log", "dt_bias")
                else x[:, lo:hi]
                if n.startswith("state_")
                else x[:, :, lo:hi]
            ).clone()
            for n, x in c.items()
        }
        prepare_consistent_prefill(**part).run()
        bits(part["out"], c["out"][:, :, lo:hi])
        bits(part["state_out"], c["state_out"][:, lo:hi])
    from flash_kda._consistent_prefill_staged import (
        compiled_coeff_kernel,
        compiled_scan_kernel,
    )

    prepare_consistent_prefill(**case(3, 12, 33), mode="staged")
    before = (
        compiled_coeff_kernel.cache_info().misses,
        compiled_scan_kernel.cache_info().misses,
    )
    prepare_consistent_prefill(**case(4, 12, 67), mode="staged")
    assert before == (
        compiled_coeff_kernel.cache_info().misses,
        compiled_scan_kernel.cache_info().misses,
    )


@pytest.mark.parametrize("mode", ["direct", "staged"])
def test_graph_and_window(mode, tmp_path):
    c = case(2, 12, 33)
    window = 7
    plan = prepare_consistent_prefill(
        **c, mode=mode, workspace_limit_bytes=2 * 12 * 392 * 4 * window
    )
    plan.run()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        plan.run()
    graph.instantiate()
    graph.debug_dump(str(tmp_path / "prefill.dot"))
    dot = (tmp_path / "prefill.dot").read_text()
    assert len(re.findall(r'label="\{\s*KERNEL\b', dot)) == (
        1 if mode == "direct" else 10
    )
    assert "MEMCPY" not in dot
    for _ in range(3):
        for n in ("q", "k", "v", "g", "beta", "A_log", "dt_bias", "state_in"):
            c[n].normal_(std=0.2)
        expected = oracle(c)
        graph.replay()
        bits(c["out"], expected[0])
        bits(c["state_out"], expected[1])


@pytest.mark.parametrize("length", [4096, 32768])
@pytest.mark.parametrize("h", [12, 16, 32])
def test_long(length, h):
    if os.environ.get("FLASH_KDA_LONG_TESTS") != "1":
        pytest.skip("enable long tests")
    c = case(1, h, length)
    expected = oracle(c, lower_bound=0.0)
    for mode in ("direct", "staged"):
        prepare_consistent_prefill(
            **c, mode=mode, lower_bound=0.0, workspace_limit_bytes=32 * 1024 * 1024
        ).run()
        bits(c["out"], expected[0])
        bits(c["state_out"], expected[1])


@pytest.mark.parametrize("h", [12, 16, 32])
def test_multigpu_shards_and_handoff(h):
    if torch.cuda.device_count() < 2:
        pytest.skip("two GPUs required")
    c = case(2, h, 65, device="cuda:0")
    prepare_consistent_prefill(**c).run()
    mid = h // 2
    with torch.cuda.device(1):
        for lo, hi in [(0, mid), (mid, h)]:
            part = {
                n: (
                    x[lo:hi]
                    if n in ("A_log", "dt_bias")
                    else x[:, lo:hi]
                    if n.startswith("state_")
                    else x[:, :, lo:hi]
                )
                .to("cuda:1")
                .contiguous()
                for n, x in c.items()
            }
            prepare_consistent_prefill(**part).run()
            bits(part["out"].to("cuda:0"), c["out"][:, :, lo:hi])
            bits(part["state_out"].to("cuda:0"), c["state_out"][:, lo:hi])
    prefix = {
        n: (
            x[:, :17].clone() if n in ("q", "k", "v", "g", "beta", "out") else x.clone()
        )
        for n, x in c.items()
    }
    prepare_consistent_prefill(**prefix).run()
    torch.cuda.synchronize(0)
    with torch.cuda.device(1):
        rest = {
            n: (x[:, 17:] if n in ("q", "k", "v", "g", "beta", "out") else x)
            .to("cuda:1")
            .contiguous()
            for n, x in c.items()
        }
        rest["state_in"] = prefix["state_out"].to("cuda:1")
        prepare_consistent_prefill(**rest).run()
        bits(rest["out"].to("cuda:0"), c["out"][:, 17:])
        bits(rest["state_out"].to("cuda:0"), c["state_out"])


def test_errors_and_grad():
    c = case()
    for change in [
        dict(q=c["q"].float()),
        dict(out=c["q"]),
        dict(q=c["q"].transpose(0, 1)),
    ]:
        with pytest.raises(ValueError):
            prepare_consistent_prefill(**dict(c, **change))
    with pytest.raises(ValueError):
        prepare_consistent_prefill(**c, mode="bad")
    with pytest.raises(ValueError):
        prepare_consistent_prefill(**c, mode="staged", workspace_limit_bytes=1)
    c["q"].requires_grad_()
    with pytest.raises(RuntimeError):
        prepare_consistent_prefill(**c)
    with torch.no_grad():
        plan = prepare_consistent_prefill(**c)
        plan.run()
    with pytest.raises(RuntimeError):
        plan.run()


@pytest.mark.parametrize("scale,eps", [(128**-0.5, 1e-6), (0.2, 1e-4)])
def test_bf16_native_random_contractions(scale, eps):
    # BF16-native random values and A_log=0 expose contraction differences that
    # a small FP32-generated fixture does not exercise reliably.
    torch.manual_seed(20261105)
    b, length, h = 8, 8192, 6

    def rand(shape, dtype=torch.bfloat16):
        return torch.randn(shape, device="cuda", dtype=dtype)

    xs = [*[rand((b, length, h, 128)) for _ in range(4)], rand((b, length, h))]
    c = dict(
        zip(("q", "k", "v", "g", "beta"), xs),
        A_log=torch.zeros(h, device="cuda"),
        dt_bias=rand((h, 128), torch.float32),
        state_in=(rand((b, h, 128, 128), torch.float32) * 0.1).bfloat16(),
        out=torch.empty_like(xs[0]),
        state_out=rand((b, h, 128, 128)),
    )
    expected = oracle(c, scale=scale, eps=eps)
    for mode in ("direct", "staged"):
        prepare_consistent_prefill(**c, mode=mode, scale=scale, eps=eps).run()
        bits(c["out"], expected[0])
        bits(c["state_out"], expected[1])
