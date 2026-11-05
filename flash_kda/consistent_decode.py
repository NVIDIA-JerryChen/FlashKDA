"""Dense single-token decode under the existing recurrent numerical contract."""

import torch
from dataclasses import dataclass, field

from .consistent import _contract_scalars, _range


@dataclass(frozen=True)
class ConsistentDecodePlan:
    """Buffer-bound launch prepared by prepare_consistent_decode.

    Keep tensor addresses, metadata and scalar parameters unchanged. Tensor
    contents may change on the caller's stream. Recreate the plan to rebind
    buffers or shapes; compatible plans reuse the compiled kernel.
    """

    _kernel: object = field(repr=False)
    _arguments: tuple = field(repr=False)
    _requires_grad: bool = field(repr=False)
    out: torch.Tensor = field(repr=False)
    state_out: torch.Tensor = field(repr=False)

    def run(self):
        """Enqueue one token on the current stream; no allocation or copies."""
        if self._requires_grad and torch.is_grad_enabled():
            raise RuntimeError("consistent decode supports no_grad forward only")
        if self._kernel is not None:
            self._kernel(*self._arguments)
        return self.out, self.state_out


def prepare_consistent_decode(
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
    scale=128**-0.5,
    eps=1e-6,
    lower_bound=-5.0,
):
    """Validate and bind one-token buffers, then compile outside Graph capture.

    BF16 q/k/v/g/out: [B,1,H,128]; beta: [B,1,H]; state: [B,H,128,128].
    FP32 A_log: [H]; dt_bias: [H,128]. All tensors must be contiguous CUDA
    tensors on one device, with 32-byte aligned data pointers. State input and
    output may be identical; all other overlap is forbidden. Buffers belong to
    the caller. Empty batches are no-ops. Prepare outside CUDA Graph capture.
    See docs/consistent-decode.md for the complete contract.
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
    if not all(isinstance(t, torch.Tensor) for t in tensors.values()):
        raise TypeError("all inputs and output buffers must be torch.Tensor")
    if q.ndim != 4 or q.shape[1] != 1 or q.shape[2] < 1 or q.shape[3] != 128:
        raise ValueError("q must have shape [B,1,H,128], H > 0")
    b, _, h, _ = q.shape
    if b * h * 8 >= 2**31:
        raise ValueError("decode requires B * H * 8 < 2**31")
    shapes = dict(
        q=q.shape,
        k=q.shape,
        v=q.shape,
        g=q.shape,
        out=q.shape,
        beta=(b, 1, h),
        A_log=(h,),
        dt_bias=(h, 128),
        state_in=(b, h, 128, 128),
        state_out=(b, h, 128, 128),
    )
    for name, t in tensors.items():
        dtype = torch.float32 if name in ("A_log", "dt_bias") else torch.bfloat16
        if t.device.type != "cuda" or t.device != q.device:
            raise ValueError(f"{name} must be on the same CUDA device as q")
        if t.dtype != dtype or tuple(t.shape) != tuple(shapes[name]):
            raise ValueError(f"{name} must have dtype {dtype} and shape {shapes[name]}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if t.numel() and t.data_ptr() % 32:
            raise ValueError(f"{name} data pointer must be 32-byte aligned")
        if torch.is_grad_enabled() and t.requires_grad:
            raise RuntimeError("consistent_decode_fwd supports no_grad forward only")
    _contract_scalars(scale, eps, lower_bound)
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
    if b == 0:
        return ConsistentDecodePlan(
            None, (), any(t.requires_grad for t in tensors.values()), out, state_out
        )
    from ._consistent_decode_kernel import compiled_decode_kernel

    with torch.cuda.device(q.device):
        kernel = compiled_decode_kernel(
            q.device.index, torch.cuda.get_device_capability(q.device)
        )
        arguments = (
            q,
            k,
            v,
            g,
            beta,
            state_out,
            state_in,
            out,
            A_log,
            dt_bias.view(-1),
            float(scale),
            float(eps),
            float(lower_bound),
        )
    return ConsistentDecodePlan(
        kernel,
        arguments,
        any(t.requires_grad for t in tensors.values()),
        out,
        state_out,
    )


def consistent_decode_fwd(
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
    scale=128**-0.5,
    eps=1e-6,
    lower_bound=-5.0,
):
    """Validate and launch one dense token; use a prepared plan for hot loops."""
    return prepare_consistent_decode(
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
        scale=scale,
        eps=eps,
        lower_bound=lower_bound,
    ).run()
