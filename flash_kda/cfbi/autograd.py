"""Training entry point: forward = the CF-BI program (bitwise equal to prefill/decode), backward = FLA's chunk
backward on a recomputed forward.

Constraints (documented, enforced):
  * K = V = 128, bf16 q/k/v/g/beta, fp32 A_log/dt_bias;
  * ``lower_bound`` in ``[-5, 0)`` when gradients are requested (FLA's ``safe_gate`` path; 0 is rejected);
  * the initial state is the empty state (a constant: no ``dh0``), no context parallelism;
  * the backward is NOT bit-pinned: it recomputes the same math with FLA's Triton chunk kernels, so it costs one
    extra forward-sized pass per layer, and its rounding differs from the forward. Only the forward is
    bitwise-aligned with prefill/decode.
"""
from typing import Optional

import torch

from ._bwd._fla import fla_chunk_kda
from .api import forward_train


class _ChunkKDACFBI(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, A_log, dt_bias, scale, lower_bound, cu_seqlens):
        out = forward_train(q, k, v, g, beta, scale, A_log, dt_bias, lower_bound, cu_seqlens)
        ctx.save_for_backward(q, k, v, g, beta, A_log, dt_bias)
        ctx.scale, ctx.lower_bound, ctx.cu_seqlens = scale, lower_bound, cu_seqlens
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, g, beta, A_log, dt_bias = ctx.saved_tensors
        need = ctx.needs_input_grad[:7]
        chunk_kda = fla_chunk_kda()
        with torch.enable_grad():
            # FLA takes dt_bias flattened to [H*K]; the gradient is reshaped back below
            tensors = (q, k, v, g, beta, A_log, dt_bias.reshape(-1))
            ins = [x.detach().requires_grad_(n) for x, n in zip(tensors, need)]
            o, _ = chunk_kda(
                ins[0], ins[1], ins[2], ins[3], ins[4], A_log=ins[5], dt_bias=ins[6], scale=ctx.scale,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                safe_gate=True, lower_bound=ctx.lower_bound, cu_seqlens=ctx.cu_seqlens, disable_recompute=True,
            )
            diff = [x for x in ins if x.requires_grad]
            grads = iter(torch.autograd.grad(o, diff, do.to(o.dtype), allow_unused=True))
        res = [next(grads) if x.requires_grad else None for x in ins]
        if res[6] is not None:
            res[6] = res[6].reshape(dt_bias.shape)
        return (*res, None, None, None)


def chunk_kda(q, k, v, g, beta, scale, A_log, dt_bias, lower_bound, cu_seqlens: Optional[torch.Tensor] = None):
    """Differentiable KDA over ``[B, T, H, 128]`` inputs (``cu_seqlens`` with B == 1 for varlen).

    Same inputs as ``flash_kda.fwd`` (``g`` raw gate input, ``beta`` raw logits); returns the bf16 output. The
    forward is exactly ``flash_kda.cfbi.forward_train``, i.e. the same bits as ``prefill`` / ``decode`` of the
    same sequence.
    """
    wants_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (q, k, v, g, beta, A_log, dt_bias)
    )
    if wants_grad and not (-5.0 <= lower_bound < 0.0):
        raise ValueError(
            f"gradients need lower_bound in [-5, 0) (FLA's safe_gate path rejects 0), got {lower_bound}; "
            "the forward alone accepts [-5, 0]"
        )
    return _ChunkKDACFBI.apply(q, k, v, g, beta, A_log, dt_bias, float(scale), float(lower_bound), cu_seqlens)
