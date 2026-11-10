"""Unified dense/varlen prefill for the recurrent BF16 contract."""

from dataclasses import dataclass, field

import torch

from .consistent import _contract_scalars, _range


@dataclass(frozen=True)
class UnifiedPrefillPlan:
    """Own workspace and bind caller buffers; prepare outside CUDA Graph capture."""

    _plan: object = field(repr=False)
    workspace_bytes: int
    out: torch.Tensor = field(repr=False)
    state_out: torch.Tensor = field(repr=False)
    cu_seqlens: torch.Tensor = field(repr=False)
    _requires_grad: bool = field(repr=False)

    def run(self):
        """Enqueue on this device's current stream without allocation or D2H."""
        if self._requires_grad and torch.is_grad_enabled():
            raise RuntimeError("unified prefill supports no_grad forward only")
        with torch.cuda.device(self.out.device):
            self._plan.run()
        return self.out, self.state_out


def prepare_unified_prefill(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    state_in,
    out,
    state_out,
    *,
    cu_seqlens=None,
    scale=128**-0.5,
    eps=1e-6,
    lower_bound=-5.0,
    workspace_limit_bytes=256 * 1024 * 1024,
):
    """Bind BF16 D128 prefill using one compiled scan for all B/H/lengths.

    Dense inputs are [B,L,H,128]; packed inputs are [1,capacity,H,128] with
    device Int32 cu_seqlens [N+1]. State is BF16 [N,H,V,K] and is required.
    Valid metadata starts at zero, is nondecreasing, and ends at or below
    capacity. Its values remain device-resident; callers must maintain these
    preconditions, including during CUDA Graph replay. See docs/unified-prefill.md.
    """
    tensors = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        state_in=state_in,
        out=out,
        state_out=state_out,
    )
    if cu_seqlens is not None:
        tensors["cu_seqlens"] = cu_seqlens
    if not all(isinstance(t, torch.Tensor) for t in tensors.values()):
        raise TypeError("all inputs and output buffers must be torch.Tensor")
    if q.ndim != 4 or q.shape[2] < 1 or q.shape[3] != 128:
        raise ValueError("q must have shape [B,L,H,128], H > 0")
    b, length, h, _ = q.shape
    total = b * length
    owns_cu = cu_seqlens is None
    if cu_seqlens is None:
        n = b
    else:
        if b != 1 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 1:
            raise ValueError("packed input requires B=1 and cu_seqlens [N+1]")
        n = cu_seqlens.numel() - 1
    if total + 127 >= 2**31 or n * h * 8 >= 2**31 or (total + 127) // 128 + n >= 2**31:
        raise ValueError("token capacity, schedule capacity and grid must fit Int32")
    shapes = dict(
        q=q.shape,
        k=q.shape,
        v=q.shape,
        g=q.shape,
        out=q.shape,
        beta=(b, length, h),
        A_log=(h,),
        dt_bias=(h, 128),
        state_in=(n, h, 128, 128),
        state_out=(n, h, 128, 128),
        cu_seqlens=(n + 1,),
    )
    requires_grad = False
    for name, tensor in tensors.items():
        dtype = (
            torch.int32
            if name == "cu_seqlens"
            else (torch.float32 if name in ("A_log", "dt_bias") else torch.bfloat16)
        )
        if tensor.device.type != "cuda" or tensor.device != q.device:
            raise ValueError(f"{name} must be on the same CUDA device as q")
        if tensor.dtype != dtype or tuple(tensor.shape) != tuple(shapes[name]):
            raise ValueError(f"{name} must have dtype {dtype} and shape {shapes[name]}")
        if not tensor.is_contiguous() or (tensor.numel() and tensor.data_ptr() % 32):
            raise ValueError(f"{name} must be contiguous and 32-byte aligned")
        requires_grad |= tensor.requires_grad
    if requires_grad and torch.is_grad_enabled():
        raise RuntimeError("unified prefill supports no_grad forward only")
    _contract_scalars(scale, eps, lower_bound)
    if not isinstance(workspace_limit_bytes, int) or workspace_limit_bytes < 0:
        raise ValueError("workspace_limit_bytes must be a nonnegative integer")
    ranges = [(name, *_range(t)) for name, t in tensors.items()]
    for i, (name, lo, hi) in enumerate(ranges):
        for other, ulo, uhi in ranges[i + 1 :]:
            if max(lo, ulo) < min(hi, uhi):
                if {name, other} == {"state_in", "state_out"} and (lo, hi) == (
                    ulo,
                    uhi,
                ):
                    continue
                raise ValueError(f"overlapping tensors are forbidden: {name}, {other}")
    cu_bytes = (n + 1) * 4 if owns_cu else 0
    if workspace_limit_bytes < cu_bytes:
        raise ValueError("workspace cannot hold dense sequence metadata")
    with torch.cuda.device(q.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "prepare_unified_prefill must run outside CUDA Graph capture"
            )
        if cu_seqlens is None:
            cu_seqlens = (
                torch.arange(n + 1, device=q.device, dtype=torch.int32) * length
            )
        from ._prefill_window.plan import prepare

        packed = [x.view(1, total, *x.shape[2:]) for x in (q, k, v, g, beta)]
        plan = prepare(
            *packed,
            A_log,
            dt_bias,
            state_in,
            cu_seqlens,
            out.view(1, total, h, 128),
            state_out,
            workspace_limit_bytes - cu_bytes,
            scale=scale,
            eps=eps,
            lower_bound=lower_bound,
        )
    workspace_bytes = plan.workspace_bytes + cu_bytes
    return UnifiedPrefillPlan(
        plan, workspace_bytes, out, state_out, cu_seqlens, requires_grad
    )
