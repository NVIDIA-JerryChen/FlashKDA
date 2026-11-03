"""CF-BI short causal depthwise convolution (+ SiLU) with a pinned floating-point program.

The layer around the KDA core that turns ``Linear(x)`` into q / k / v (FLA's ``ShortConvolution`` with
``activation="silu"``, kernel size 4). For one channel ``d`` and sequence position ``t``::

    acc = +0.0f
    for k in 0 .. W-1 (ascending):  acc = fmaf(w[d][k], float(x[t-(W-1)+k][d]), acc)     # w[W-1] multiplies x[t]
    z   = acc (+ bias[d])                                                                # one fadd_rn
    y   = z * rcp(1 + ex2(-z * log2e))                                                   # SiLU, rcp/ex2 = one PTX instr each
    out = bf16_rn(y)

and the state is the last ``W-1`` *inputs* (bf16, copied bit for bit). Positions before the start of a sequence read
the state (zeros for a fresh sequence, which is also what the training forward uses). Because the program is one
function of ``W`` input values, the bits of ``out`` and of the updated state do not depend on how a sequence is cut into
calls, on being batched with other sequences, or on the launch geometry: one prefill call == any cuts == T=1 decode
calls == a varlen batch == the training forward (empty state). See docs/20261103-cfbi-conv-linear.md.

``short_conv`` is differentiable when no state is passed (training: zero history); its backward is plain fp32 PyTorch
and is not bit-pinned.
"""
from typing import Optional

import torch

from ._ext import load_conv_ext

__all__ = ["short_conv", "alloc_conv_state", "conv_state_after"]

_ACTIVATIONS = (None, "silu", "swish")


def alloc_conv_state(N: int, D: int, W: int = 4, device="cuda") -> torch.Tensor:
    """Zero conv state for ``N`` fresh sequences: bf16 ``[N, D, W-1]``; entry ``m`` of channel ``d`` is the input of
    sequence position ``m - (W-1)`` (so the last entry is the most recent input)."""
    return torch.zeros(N, D, W - 1, dtype=torch.bfloat16, device=device)


def _check_args(x, weight, bias, state, activation, cu_seqlens):
    if activation not in _ACTIVATIONS:
        raise ValueError(f"activation must be one of {_ACTIVATIONS}, got {activation!r}")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"x must be bf16 (the output of a bf16 Linear), got {x.dtype}")
    if x.dim() not in (2, 3):
        raise ValueError("x must be [B, T, D] or [T, D]")
    if weight.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("weight must be fp32 or bf16")
    if bias is not None and bias.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("bias must be fp32 or bf16")
    if cu_seqlens is not None and (x.dim() == 3 and x.shape[0] != 1):
        raise ValueError("with cu_seqlens x must be [1, T_total, D]")


def _short_conv_fwd(x, weight, bias, state, silu, cu_seqlens, max_seqlen, out, out_fp32):
    ext = load_conv_ext()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    if out is None:
        out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    ext.conv_forward(x, weight, out, bias, state, cu_seqlens, max_seqlen, silu, out_fp32)
    return out


def _seq_ids(R: int, T: int, B: int, cu_seqlens: Optional[torch.Tensor], device):
    """Sequence index of every row of the flattened [R, D] token axis."""
    r = torch.arange(R, device=device)
    if cu_seqlens is None:
        return r // max(T, 1)
    return torch.searchsorted(cu_seqlens[1:].contiguous(), r, right=True)


def _conv_backward(dy, x, weight, bias, silu, cu_seqlens):
    """Gradients of out = act(causal_conv(x) + bias) in plain fp32 PyTorch (zero history; sequences do not leak into
    each other). Not bit-pinned: only has to be correct. Returns (dx, dw, db) in the dtypes of x / weight / bias."""
    D = x.shape[-1]
    W = weight.shape[-1]
    B, T = (x.shape[0], x.shape[1]) if x.dim() == 3 else (1, x.shape[0])
    xf = x.reshape(-1, D).float()
    dyf = dy.reshape(-1, D).float()
    R = xf.shape[0]
    w = weight.reshape(D, W).float()
    seq = _seq_ids(R, T, B, cu_seqlens, x.device)

    def shift_fwd(a, s):   # row r -> a[r - s] if r - s is in the same sequence, else 0
        if s == 0:
            return a
        out = torch.zeros_like(a)
        if s < R:
            ok = (seq[s:] == seq[:R - s])[:, None]
            out[s:] = torch.where(ok, a[:R - s], torch.zeros((), dtype=a.dtype, device=a.device))
        return out

    def shift_bwd(a, s):   # row r -> a[r + s] if r + s is in the same sequence, else 0
        if s == 0:
            return a
        out = torch.zeros_like(a)
        if s < R:
            ok = (seq[s:] == seq[:R - s])[:, None]
            out[:R - s] = torch.where(ok, a[s:], torch.zeros((), dtype=a.dtype, device=a.device))
        return out

    xs = [shift_fwd(xf, W - 1 - k) for k in range(W)]   # xs[k][r] = x[r - (W-1-k)]: the input tap k multiplies
    z = torch.zeros_like(xf)
    for k in range(W):
        z = z + w[:, k] * xs[k]
    if bias is not None:
        z = z + bias.float()
    if silu:
        sg = torch.sigmoid(z)
        dz = dyf * (sg * (1.0 + z * (1.0 - sg)))
    else:
        dz = dyf
    dx = torch.zeros_like(xf)
    dw = torch.empty_like(w)
    for k in range(W):
        dw[:, k] = (dz * xs[k]).sum(0)
        dx = dx + w[:, k] * shift_bwd(dz, W - 1 - k)
    db = dz.sum(0) if bias is not None else None
    return (dx.reshape(x.shape).to(x.dtype), dw.reshape(weight.shape).to(weight.dtype),
            None if db is None else db.to(bias.dtype))


class _ShortConvFn(torch.autograd.Function):
    """Training entry point: forward = the pinned kernel with an empty state, backward = plain fp32 PyTorch."""

    @staticmethod
    def forward(ctx, x, weight, bias, cu_seqlens, silu, max_seqlen):
        out = _short_conv_fwd(x, weight, bias, None, silu, cu_seqlens, max_seqlen, None, None)
        ctx.save_for_backward(x, weight, bias, cu_seqlens)
        ctx.silu = silu
        return out

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias, cu_seqlens = ctx.saved_tensors
        dx, dw, db = _conv_backward(dy, x, weight, bias, ctx.silu, cu_seqlens)
        return (dx if ctx.needs_input_grad[0] else None, dw if ctx.needs_input_grad[1] else None,
                db if (bias is not None and ctx.needs_input_grad[2]) else None, None, None, None)


def short_conv(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
               state: Optional[torch.Tensor] = None, activation: Optional[str] = "silu",
               cu_seqlens: Optional[torch.Tensor] = None, *, max_seqlen: Optional[int] = None,
               out: Optional[torch.Tensor] = None, out_fp32: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Causal depthwise convolution over the token axis + SiLU, bf16 in / bf16 out, fixed fp32 program (see module doc).

    Args:
        x: bf16 ``[B, T, D]`` (every batch entry is one sequence) or ``[1, T_total, D]`` with ``cu_seqlens``
            (packed sequences); ``[T, D]`` is one sequence. Channels contiguous; tokens may sit at a constant row stride
            (a slice of a fused QKV projection works).
        weight: ``[D, W]`` or ``[D, 1, W]`` (``nn.Conv1d`` layout), fp32 or bf16, ``2 <= W <= 8`` (KDA: 4). Tap ``W-1``
            multiplies the current token. The program works on ``float(weight)``.
        bias: ``[D]`` fp32 / bf16 or None (FLA's KDA layer has none).
        state: None (training / stateless: zero history, nothing written) or bf16 ``[N, D, W-1]`` (``alloc_conv_state``),
            ``N = B`` or ``len(cu_seqlens) - 1``; updated IN PLACE to the last ``W-1`` inputs of (old state ++ x). A
            sequence of length 0 leaves its state untouched.
        activation: ``"silu"`` (``"swish"``) or None.
        cu_seqlens: int64 ``[N+1]`` for packed varlen input (requires ``B == 1``).
        max_seqlen: optional upper bound of the sequence lengths (a host int). The grid is sized from it (default: ``T``
            or, with ``cu_seqlens``, ``T_total``); a small bound avoids launching idle blocks, an incorrect (too small)
            one silently drops tokens, so only pass a true bound.
        out: optional bf16 output buffer (shape of x, contiguous, must not alias x).
        out_fp32: optional fp32 buffer (shape of x) that receives the value before the bf16 rounding (tests: comparing
            bf16 outputs only is a weak detector of arithmetic differences).

    Returns bf16, shape of ``x``. Differentiable when ``state is None`` (``torch.autograd.Function``; backward is plain
    fp32 PyTorch, not bit-pinned); a stateful call is inference-only and raises if grad is required.
    """
    _check_args(x, weight, bias, state, activation, cu_seqlens)
    silu = activation is not None
    needs_grad = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x, weight, bias))
    if needs_grad:
        if state is not None or out is not None or out_fp32 is not None:
            raise ValueError("a differentiable call must be stateless (state=None) and allocate its own output; "
                             "stateful / in-place calls are inference-only: wrap them in torch.no_grad()")
        return _ShortConvFn.apply(x, weight, bias, cu_seqlens, silu, max_seqlen)
    return _short_conv_fwd(x, weight, bias, state, silu, cu_seqlens, max_seqlen, out, out_fp32)


def conv_state_after(x: torch.Tensor, W: int, state: Optional[torch.Tensor] = None,
                     cu_seqlens: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Reference (plain PyTorch) of the state update: the last ``W-1`` inputs of (state ++ x) per sequence, returned as a
    new bf16 ``[N, D, W-1]`` tensor (``x`` is the bf16 conv input before the call). For tests and documentation."""
    D = x.shape[-1]
    if cu_seqlens is None:
        xs = x if x.dim() == 3 else x[None]
        bounds = [(n * xs.shape[1], (n + 1) * xs.shape[1]) for n in range(xs.shape[0])]
        flat = xs.reshape(-1, D)
    else:
        flat = x.reshape(-1, D)
        cu = cu_seqlens.tolist()
        bounds = list(zip(cu[:-1], cu[1:]))
    N = len(bounds)
    out = torch.zeros(N, D, W - 1, dtype=torch.bfloat16, device=x.device)
    for n, (lo, hi) in enumerate(bounds):
        old = state[n] if state is not None else torch.zeros(D, W - 1, dtype=torch.bfloat16, device=x.device)
        ext = torch.cat([old.t(), flat[lo:hi]], dim=0)   # [W-1 + len, D]
        out[n] = ext[ext.shape[0] - (W - 1):].t()
    return out
