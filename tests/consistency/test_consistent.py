import os
import pytest
import torch
from flash_kda import consistent_fwd
from .helpers import make_case, run, select, assert_bits

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("h", [12, 16, 32])
@pytest.mark.parametrize("length", [0, 1, 2, 15, 16, 17, 127, 128, 129])
def test_reference_every_step(h, length):
    from .flashinfer_reference import step
    case = make_case([length], h)
    whole, final = run(case)
    state = case["state_in"].clone()
    for t in range(length):
        segment, _ = select(case, [length], [(t, t + 1)], state=state.clone())
        actual, next_state = run(segment)
        golden = torch.empty_like(segment["q"])
        step(*(segment[n] for n in ("q", "k", "v", "g", "beta")), state,
             case["A_log"], case["dt_bias"], golden)
        assert_bits(actual, golden, label="FlashInfer output", lengths=[1], token=t)
        assert_bits(next_state, state, label="FlashInfer state", token=t)
        assert_bits(whole[:, t:t+1], golden, label="whole/step output", lengths=[1], token=t)
    assert_bits(final, state, label="whole/step final", token=length)


@pytest.mark.parametrize("h", [12, 16, 32])
@pytest.mark.parametrize("batch", [1, 4, 7, 8, 9, 16, 32])
def test_batch_and_packing(h, batch):
    lengths = ([0, 1, 2, 15, 16, 17, 129] * 5)[:batch]
    case = make_case(lengths, h, padding=7)
    whole, final = run(case)
    offset = 0
    for request, length in enumerate(lengths):
        solo, _ = select(case, lengths, [(0, x) for x in lengths], [request])
        out, state = run(solo)
        assert_bits(out, whole[:, offset:offset+length], label="batch/solo output", lengths=[length])
        assert_bits(state, final[request:request+1], label="batch/solo state")
        offset += length
    assert torch.all(whole[:, offset:] == 31), "padding was modified"
    order = list(reversed(range(batch)))
    reordered, lens = select(case, lengths, [(0, x) for x in lengths], order)
    out, state = run(reordered)
    expected = torch.cat(list(whole[:, :offset].split(lengths, dim=1))[::-1], dim=1)
    assert_bits(out, expected, label="permutation output", lengths=lens)
    assert_bits(state, final[order], label="permutation state")


@pytest.mark.parametrize("lower_bound", [-5., -2.5, 0.])
@pytest.mark.parametrize("kind", ["zero_qk", "saturated_gate", "normal"])
def test_extremes_and_inplace(lower_bound, kind):
    case = make_case([17, 0, 2], 12)
    if kind == "zero_qk":
        case["q"].zero_(); case["k"].zero_()
    elif kind == "saturated_gate":
        case["g"][..., ::2] = 1000
        case["g"][..., 1::2] = -1000
        case["beta"][:, ::2] = 1000
        case["beta"][:, 1::2] = -1000
    expected, final = run(case, lower_bound=lower_bound)
    out = torch.empty_like(expected)
    state = case["state_in"].clone()
    consistent_fwd(**dict(case, state_in=state), out=out, state_out=state, lower_bound=lower_bound)
    assert_bits(out, expected, label="inplace output", lengths=[17, 0, 2])
    assert_bits(state, final, label="inplace state")
    # Extreme/scalar cases must also agree with the independent upstream path.
    from .flashinfer_reference import step
    offset = 0
    for request, length in enumerate([17, 0, 2]):
        golden_state = case["state_in"][request:request+1].clone()
        for t in range(length):
            segment, _ = select(case, [17, 0, 2],
                                [(t, t+1) if r == request else (0, 0) for r in range(3)],
                                [request], state=golden_state.clone())
            actual, next_state = run(segment, lower_bound=lower_bound)
            golden_out = torch.empty_like(actual)
            step(*(segment[n] for n in ("q", "k", "v", "g", "beta")), golden_state,
                 case["A_log"], case["dt_bias"], golden_out, lower_bound=lower_bound)
            assert_bits(actual, golden_out, label="extreme reference output", lengths=[1], token=t)
            assert_bits(next_state, golden_state, label="extreme reference state", token=t)
            assert_bits(expected[:, offset+t:offset+t+1], golden_out,
                        label="extreme whole output", lengths=[1], token=t)
        assert_bits(final[request:request+1], golden_state, label="extreme final", token=length)
        offset += length



@pytest.mark.parametrize("h", [12, 16, 32])
@pytest.mark.parametrize("length", [4096, 32768])
def test_long_random_segments(h, length):
    if os.getenv("FLASH_KDA_LONG_TESTS") != "1":
        pytest.skip("set FLASH_KDA_LONG_TESTS=1 for long-sequence acceptance")
    import random
    case = make_case([length], h)
    whole, final = run(case, lower_bound=0.)
    rng = random.Random(51)
    cuts = sorted(set([0, length] + [rng.randrange(length + 1) for _ in range(23)]))
    state = case["state_in"].clone()
    for begin, end in zip(cuts, cuts[1:]):
        segment, _ = select(case, [length], [(begin, end)], state=state)
        out, state = run(segment, lower_bound=0.)
        assert_bits(out, whole[:, begin:end], label="long segment", lengths=[end-begin])
    assert_bits(state, final, label="long final", token=length)


def test_graph_dynamic_metadata_and_compile_cache():
    from flash_kda._consistent_kernel import compiled_kernel
    lengths = [17, 0, 2]
    case = make_case(lengths, padding=128)
    expected, final = run(case)
    out, state = torch.empty_like(expected), torch.empty_like(final)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        consistent_fwd(**case, out=out, state_out=state)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        consistent_fwd(**case, out=out, state_out=state)
    before = compiled_kernel.cache_info().misses
    for lens in ([17, 0, 2], [1, 2, 127], [0, 0, 0], [129, 0, 0]):
        offsets = [0]
        for length in lens:
            offsets.append(offsets[-1] + length)
        case["cu_seqlens"].copy_(torch.tensor(offsets, device="cuda"))
        expected, final = run(case)
        graph.replay()
        assert_bits(out[:, :sum(lens)], expected[:, :sum(lens)], label="graph output", lengths=lens)
        assert_bits(state, final, label="graph state")
    for lens in ([1], [0]*8, [2]*9, [1]*32):
        run(make_case(lens))
    assert compiled_kernel.cache_info().misses == before


def test_repeated_zero_capacity_and_unaligned():
    case = make_case([0, 0], padding=0)
    out, state = run(case)
    assert_bits(state, case["state_in"], label="empty sequences")
    for lens in ([], [1, 2]):
        case = make_case(lens)
        # Contiguous views need not have 16-byte aligned pointers.
        for name in ("q", "k", "v", "g", "state_in"):
            x = case[name]
            buf = torch.empty(x.numel()+1, device="cuda", dtype=x.dtype)
            shifted = buf[1:].view_as(x); shifted.copy_(x); case[name] = shifted
        a, sa = run(case)
        b, sb = run(case)
        assert_bits(a, b, label="repeat")
        assert_bits(sa, sb, label="repeat state")


def test_contract_errors():
    case = make_case([1])
    out, state = torch.empty_like(case["q"]), torch.empty_like(case["state_in"])
    with pytest.raises(ValueError, match="overlapping"):
        consistent_fwd(**case, out=case["q"], state_out=state)
    with pytest.raises(ValueError, match="lower_bound"):
        consistent_fwd(**case, out=out, state_out=state, lower_bound=1)
    case["q"].requires_grad_(True)
    with pytest.raises(RuntimeError, match="backward"):
        consistent_fwd(**case, out=out, state_out=state)
    with torch.no_grad():
        consistent_fwd(**case, out=out, state_out=state)


@pytest.mark.parametrize("length", [4096, 32768])
def test_long_reference_every_step(length):
    if os.getenv("FLASH_KDA_LONG_TESTS") != "1":
        pytest.skip("set FLASH_KDA_LONG_TESTS=1 for long reference acceptance")
    # Risk-based coverage: long independent oracle at H=12, other H values
    # covered by short oracle tests and full-length segmentation tests.
    test_reference_every_step(12, length)


def test_partial_state_overlap_is_rejected():
    case = make_case([1])
    count = case["state_in"].numel()
    buffer = torch.empty(count+1, device="cuda", dtype=torch.bfloat16)
    case["state_in"] = buffer[:-1].view_as(case["state_in"])
    destination = buffer[1:].view_as(case["state_in"])
    with pytest.raises(ValueError, match="overlapping"):
        consistent_fwd(**case, out=torch.empty_like(case["q"]), state_out=destination)


@pytest.mark.parametrize("offsets", [[1, 1], [0, -1], [0, 2]])
def test_invalid_device_metadata(offsets):
    # A device assertion poisons its context: isolate each negative case.
    import subprocess
    import sys
    code = ("from tests.consistency.helpers import *; "
            "c=make_case([1]); "
            f"c['cu_seqlens']=torch.tensor({offsets!r},device='cuda',dtype=torch.int64); "
            "run(c); torch.cuda.synchronize()")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=60,
                            env=dict(os.environ, CUDA_LAUNCH_BLOCKING="1"))
    assert result.returncode != 0
    assert "invalid cu_seqlens" in result.stderr or "device-side assert" in result.stderr


def test_empty_preserves_all_finite_state_bits():
    case = make_case([0])
    bits = torch.arange(case["state_in"].numel(), device="cuda", dtype=torch.int32) % 65536
    bits = torch.where((bits & 0x7f80) == 0x7f80, 0, bits).to(torch.int16)
    case["state_in"] = bits.view(torch.bfloat16).view_as(case["state_in"])
    _, state = run(case)
    assert_bits(state, case["state_in"], label="empty finite bit patterns", token=0)
