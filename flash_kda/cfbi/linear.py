"""CF-BI batch-invariant bf16 linear layer.

``bi_linear(x, W, bias)`` computes ``bf16_rn(x @ W^T (+ bias))`` where row ``m`` of the result is bitwise independent of
``M`` and of the other rows: every output element is the same fixed program::

    acc = +0.0f
    for s = 0 .. K/16 - 1 (ascending):  acc = mma.sync.m16n8k16.bf16(x[m][16s:16s+16], W[n][16s:16s+16], acc)
    out = bf16_rn(acc (+ float(bias[n])))                                    # one fadd_rn when there is a bias

One accumulator chain over K ascending, no split-K, no second accumulator. The tile configuration the kernel picks (by
``M`` and ``N``, for speed) only decides which lane holds an element; that a tensor-core result does not depend on its slot
in the tile or on the other rows / columns is the hardware premise, probed by ``tests/cfbi/test_linear_probe.py``.
cuBLAS (``torch.matmul`` / ``F.linear``) does NOT have this property: it switches kernels / split-K with the shape (the
control in ``tests/cfbi/test_linear.py`` measures it).

Restrictions: bf16 ``x`` and ``W``, ``K % 16 == 0``. Differentiable (backward = plain torch matmuls, not bit-pinned).
See docs/20261103-cfbi-conv-linear.md.
"""
from typing import Optional

import torch

from ._ext import load_conv_ext

__all__ = ["bi_linear", "pick_config"]


def pick_config(M: int, N: int) -> int:
    """Tile configuration (0..4) the kernel chooses for an ``[M, K] x [N, K]^T`` product."""
    return int(load_conv_ext().linear_pick_cfg(M, N))


def _aligned2d(t: torch.Tensor) -> torch.Tensor:
    if t.stride(-1) != 1 or t.stride(0) % 8 != 0 or t.data_ptr() % 16 != 0:
        t = t.clone(memory_format=torch.contiguous_format)
    return t


def _bi_linear_fwd(x, weight, bias, out, cfg, out_fp32=None):
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError(f"bi_linear needs bf16 x and weight, got {x.dtype} and {weight.dtype}")
    if weight.dim() != 2 or x.shape[-1] != weight.shape[1]:
        raise ValueError(f"weight must be [N, K] with K == x.shape[-1] ({x.shape[-1]}), got {tuple(weight.shape)}")
    if bias is not None and bias.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("bias must be fp32 or bf16")
    lead = x.shape[:-1]
    K, N = x.shape[-1], weight.shape[0]
    if K % 16 != 0 or K == 0:
        raise ValueError(f"K must be a positive multiple of 16 (the program is a chain of k16 MMAs), got {K}")
    x2 = _aligned2d(x.reshape(-1, K))
    w2 = _aligned2d(weight)
    M = x2.shape[0]
    if out is None:
        out2 = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    else:
        out2 = out.view(M, N)
    load_conv_ext().linear_forward(x2, w2, out2, None if bias is None else bias.contiguous(), cfg,
                                   None if out_fp32 is None else out_fp32.view(M, N))
    return out2.view(*lead, N)


class _BiLinearFn(torch.autograd.Function):
    """Forward = the batch-invariant kernel; backward = standard torch matmuls (not bit-pinned)."""

    @staticmethod
    def forward(ctx, x, weight, bias):
        out = _bi_linear_fwd(x, weight, bias, None, -1)
        ctx.save_for_backward(x, weight)
        ctx.has_bias, ctx.bias_dtype = bias is not None, None if bias is None else bias.dtype
        return out

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        K, N = weight.shape[1], weight.shape[0]
        dy2 = dy.reshape(-1, N)
        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            dx = (dy2 @ weight).reshape(x.shape)
        if ctx.needs_input_grad[1]:
            dw = dy2.t() @ x.reshape(-1, K)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            db = dy2.sum(0, dtype=torch.float32).to(ctx.bias_dtype)
        return dx, dw, db


def bi_linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, *,
              out: Optional[torch.Tensor] = None, out_fp32: Optional[torch.Tensor] = None, _cfg: int = -1) -> torch.Tensor:
    """Batch-invariant ``F.linear`` for bf16: ``x`` ``[..., K]``, ``weight`` ``[N, K]``, ``bias`` ``[N]`` (fp32 / bf16 / None)
    -> bf16 ``[..., N]``. Row ``m`` of the result does not depend on how many other rows there are or on what they hold.

    ``_cfg`` (tests / benchmarks) forces one of the five tile configurations (every one gives the same bits); ``out`` is an
    optional contiguous bf16 ``[..., N]`` buffer, ``out_fp32`` an optional contiguous fp32 ``[..., N]`` buffer that receives
    the value before the bf16 rounding (tests: a much stronger detector of arithmetic differences than the bf16 result);
    both inference only.
    """
    needs_grad = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x, weight, bias))
    if needs_grad:
        if out is not None or out_fp32 is not None or _cfg != -1:
            raise ValueError("out= / out_fp32= / _cfg are inference-only")
        return _BiLinearFn.apply(x, weight, bias)
    return _bi_linear_fwd(x, weight, bias, out, _cfg, out_fp32)
