import pytest
import torch
from flash_kda.continuation import Continuation, rebase_continuations
from .helpers import make_case, run, select, assert_bits

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("cut", list(range(18)))
def test_pd_every_cut_rebatch(cut):
    lengths = [17, 2, 0, 16]
    case = make_case(lengths, 12)
    whole, final = run(case)
    cuts = [min(cut, length) for length in lengths]
    prefix, lens = select(case, lengths, [(0, t) for t in cuts])
    pre, state = run(prefix)
    ids = ("a", "b", "empty", "d")
    continuation = Continuation(ids, tuple(cuts), state)
    order = [3, 0, 2, 1]
    rebased = rebase_continuations([continuation], [ids[r] for r in order], torch.empty_like(state))
    assert rebased.positions == tuple(cuts[r] for r in order)
    suffix, suffix_lens = select(case, lengths, list(zip(cuts, lengths)), order, state=rebased.state)
    out, result = run(suffix)
    expected, _ = select(dict(case, q=whole), lengths, list(zip(cuts, lengths)), order)
    assert_bits(out, expected["q"], label="PD rebatch output", lengths=suffix_lens)
    assert_bits(result, final[order], label="PD rebatch state")
    expected_pre, _ = select(dict(case, q=whole), lengths, [(0, t) for t in cuts])
    assert_bits(pre, expected_pre["q"], label="PD prefix", lengths=lens)


def test_dynamic_add_remove_and_contract_guard():
    case = make_case([2, 15, 1])
    _, state = run(case)
    old = Continuation(("a", "b", "c"), (2, 15, 1), state)
    new = Continuation(("new",), (0,), torch.zeros_like(state[:1]))
    result = rebase_continuations([old, new], ["c", "new", "a"], torch.empty_like(state))
    assert result.positions == (1, 0, 2)
    assert_bits(result.state, torch.cat([state[2:], new.state, state[:1]]), label="dynamic batch")
    decode = make_case([1, 1, 1], seed=788)
    decode["state_in"] = result.state
    out, final = run(decode)
    for r in range(3):
        solo, _ = select(decode, [1]*3, [(0, 1)]*3, [r])
        o, s = run(solo)
        assert_bits(o, out[:, r:r+1], label="dynamic decode", lengths=[1])
        assert_bits(s, final[r:r+1], label="dynamic decode state")
    bad = Continuation(("new",), (0,), new.state, lower_bound=0.)
    with pytest.raises(ValueError, match="incompatible"):
        rebase_continuations([old, bad], ["a"], torch.empty_like(state[:1]))


def test_pd_scalar_bits_and_invalid_scalars():
    state = make_case([0])["state_in"]
    positive = Continuation(("a",), (0,), state, scale=0.0)
    negative = Continuation(("b",), (0,), state.clone(), scale=-0.0)
    with pytest.raises(ValueError, match="incompatible"):
        rebase_continuations([positive, negative], ["a"], torch.empty_like(state))
    with pytest.raises(ValueError, match="positive in FP32"):
        Continuation(("a",), (0,), state, eps=1e-100)
    with pytest.raises(ValueError, match="finite FP32"):
        Continuation(("a",), (0,), state, scale=1e100)
