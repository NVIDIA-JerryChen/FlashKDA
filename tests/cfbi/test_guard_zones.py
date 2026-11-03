"""Memory safety at the edges: every tensor of a call sits inside a larger buffer whose margins are filled with a
sentinel (NaN bit pattern); after the call the margins must be untouched (no write past either end of an input,
an output or the state) and the inputs unchanged. Catches the classic partial-chunk / last-row / empty-sequence overruns
that comparing results alone cannot see when the overrun lands outside the compared tensor."""
import math
import random

import pytest
import torch

from flash_kda.cfbi import CfbiState, decode, forward_train, prefill, prefill_k1, prefill_k2

from helpers import make_case

PAD = 4096   # elements of margin on each side (a multiple of 8 -> every view stays 16-byte aligned)
FILL = {torch.bfloat16: 0x7FC1, torch.float32: 0x7FC00001, torch.int32: 0x7FFFFFF1, torch.int64: 0x7FFFFFFFFFFFFFF1}
INT = {torch.bfloat16: torch.int16, torch.float32: torch.int32, torch.int32: torch.int32, torch.int64: torch.int64}


class Guarded:
    def __init__(self, data):
        """``data``: a tensor whose content is copied into a guarded buffer."""
        self.n = data.numel()
        self.dtype = data.dtype
        bits = torch.full((self.n + 2 * PAD,), FILL[data.dtype], dtype=INT[data.dtype], device="cuda")
        self.buf = bits.view(data.dtype) if data.dtype not in (torch.int32, torch.int64) else bits
        self.t = self.buf[PAD:PAD + self.n].view(data.shape)
        self.t.copy_(data)
        self.ref = data.clone()

    def margins_intact(self):
        ib = self.buf.view(INT[self.dtype]) if self.dtype not in (torch.int32, torch.int64) else self.buf
        want = FILL[self.dtype]
        return bool((ib[:PAD] == want).all()) and bool((ib[PAD + self.n:] == want).all())

    def content_unchanged(self):
        return bool(torch.equal(self.t.contiguous().view(INT[self.dtype]), self.ref.contiguous().view(INT[self.dtype])))


def _zeros_guarded(shape, dtype):
    return Guarded(torch.zeros(shape, dtype=dtype, device="cuda"))


def _state(N, H):
    E, rows, pos = _zeros_guarded((N, H, 128, 128), torch.float32), _zeros_guarded((N, H, 5, 64, 128), torch.float32), _zeros_guarded((N,), torch.int32)
    return CfbiState(E.t, rows.t, pos.t), (E, rows, pos)


def _inputs(case):
    gs = {k: Guarded(case[k]) for k in ("q", "k", "v", "g", "beta")}
    return gs


def _check(guards, what):
    for name, gd in guards.items():
        assert gd.margins_intact(), f"{what}: write outside the buffer of '{name}'"


@pytest.mark.parametrize("impl", [0, 3])
@pytest.mark.parametrize("bv", [8, 16])
def test_prefill_varlen_warm_and_decode_stay_inside_their_buffers(ext, impl, bv):
    H = 3
    rng = random.Random(1)
    lens = [1, 63, 64, 65, 130, 0, 7]
    tot = sum(lens)
    case = make_case(tot, H, "authors", seed=2)
    cu_list = [0]
    for L in lens:
        cu_list.append(cu_list[-1] + L)
    state, sg = _state(len(lens), H)

    # a first call over part of every sequence, then a second one that continues (head rows, tails)
    first = [min(L, rng.randint(1, 70)) for L in lens]
    sub = lambda lo, hi: {k: torch.cat([case[k][:, cu_list[n] + lo[n]:cu_list[n] + hi[n]] for n in range(len(lens))], dim=1).contiguous() for k in ("q", "k", "v", "g", "beta")}
    for lo, hi in ((([0] * len(lens)), first), (first, lens)):
        s = sub(lo, hi)
        takes = [h - l for l, h in zip(lo, hi)]
        c = [0]
        for t in takes:
            c.append(c[-1] + t)
        gi = {k: Guarded(v) for k, v in s.items()}
        gcu = Guarded(torch.tensor(c, dtype=torch.int64, device="cuda"))
        go = _zeros_guarded(s["q"].shape, torch.bfloat16)
        go32 = _zeros_guarded(s["q"].shape, torch.float32)
        prefill(gi["q"].t, gi["k"].t, gi["v"].t, gi["g"].t, gi["beta"].t, case["scale"], go.t, case["A_log"], case["dt_bias"],
                case["lower_bound"], state, gcu.t, out_fp32=go32.t, _impl=impl, _k2_bv=bv)
        torch.cuda.synchronize()
        _check({**{f"in_{k}": v for k, v in gi.items()}, "cu": gcu, "out": go, "out32": go32, "E": sg[0], "rows": sg[1], "pos": sg[2]}, "prefill")
        for k, v in gi.items():
            assert v.content_unchanged(), f"prefill modified its input '{k}'"

    # single-token decode steps through a pool with padding entries
    pool, pg = _state(6, H)
    slots = Guarded(torch.tensor([4, -1, 0], dtype=torch.int32, device="cuda"))
    bufs = {k: Guarded(case[k][:, :3].reshape(3, 1, *case[k].shape[2:]).contiguous()) for k in ("q", "k", "v", "g", "beta")}
    o1, o132 = _zeros_guarded(bufs["q"].t.shape, torch.bfloat16), _zeros_guarded(bufs["q"].t.shape, torch.float32)
    for _ in range(3):
        decode(bufs["q"].t, bufs["k"].t, bufs["v"].t, bufs["g"].t, bufs["beta"].t, case["scale"], o1.t, case["A_log"], case["dt_bias"],
               case["lower_bound"], pool, out_fp32=o132.t, slot_ids=slots.t, _impl=impl)
    torch.cuda.synchronize()
    _check({**{f"dec_{k}": v for k, v in bufs.items()}, "slots": slots, "out": o1, "out32": o132, "E": pg[0], "rows": pg[1], "pos": pg[2]}, "decode")
    for k, v in bufs.items():
        assert v.content_unchanged(), f"decode modified its input '{k}'"


@pytest.mark.parametrize("bv", [8, 16])
def test_training_forward_and_split_phases_stay_inside_their_buffers(ext, bv):
    H = 2
    case = make_case(64 * 3 + 21, H, "authors", seed=3)
    gi = _inputs(case)
    out, out32 = _zeros_guarded(case["q"].shape, torch.bfloat16), _zeros_guarded(case["q"].shape, torch.float32)
    forward_train(gi["q"].t, gi["k"].t, gi["v"].t, gi["g"].t, gi["beta"].t, case["scale"], case["A_log"], case["dt_bias"],
                  case["lower_bound"], out=out.t, out_fp32=out32.t, _k2_bv=bv)
    torch.cuda.synchronize()
    _check({**gi, "out": out, "out32": out32}, "forward_train")

    state, sg = _state(1, H)
    out2 = _zeros_guarded(case["q"].shape, torch.bfloat16)
    h1 = prefill_k1(gi["q"].t, gi["k"].t, gi["v"].t, gi["g"].t, gi["beta"].t, case["scale"], case["A_log"], case["dt_bias"], case["lower_bound"], state)
    prefill_k2(h1, out2.t, state)
    torch.cuda.synchronize()
    _check({**gi, "out": out2, "E": sg[0], "rows": sg[1], "pos": sg[2]}, "prefill_k1/k2")
    assert torch.equal(out.t.view(torch.int16), out2.t.view(torch.int16))
