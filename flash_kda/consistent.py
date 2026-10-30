"""Fixed-path recurrent KDA forward. CuTe is imported only at first use."""
import math
import struct
import torch

# Change this identifier if any arithmetic or rounding boundary changes.
NUMERICAL_CONTRACT = "flashkda.recurrent.bf16.d128.row16.dual.fi-c3af0c13328f.v1"


def _contract_scalars(scale, eps, lower_bound):
    bits = []
    for name, value in (("scale", scale), ("eps", eps), ("lower_bound", lower_bound)):
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite host scalar")
        try:
            encoded = struct.pack("!f", value)
        except (OverflowError, struct.error) as error:
            raise ValueError(f"{name} must be representable as finite FP32") from error
        bits.append(encoded)
    if struct.unpack("!f", bits[1])[0] <= 0 or not -5 <= lower_bound <= 0:
        raise ValueError("eps must be positive in FP32 and lower_bound must lie in [-5, 0]")
    return tuple(bits)


def _range(t):
    start = t.data_ptr()
    return start, start + t.numel() * t.element_size()


def _validate(q, k, v, g, beta, A_log, dt_bias, cu_seqlens,
              state_in, out, state_out, scale, eps, lower_bound):
    tensors = dict(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log,
                   dt_bias=dt_bias, cu_seqlens=cu_seqlens, state_in=state_in,
                   out=out, state_out=state_out)
    if not all(isinstance(x, torch.Tensor) for x in tensors.values()):
        raise TypeError("all inputs and output buffers must be torch.Tensor")
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128 or q.shape[2] < 1:
        raise ValueError("q must have shape [1, capacity, H, 128], H > 0")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 1:
        raise ValueError("cu_seqlens must have shape [N+1]")
    n, h = cu_seqlens.numel() - 1, q.shape[2]
    shapes = dict(q=q.shape, k=q.shape, v=q.shape, g=q.shape, out=q.shape,
                  beta=q.shape[:3], A_log=(h,), dt_bias=(h, 128),
                  cu_seqlens=(n + 1,), state_in=(n, h, 128, 128),
                  state_out=(n, h, 128, 128))
    for name, t in tensors.items():
        dtype = (torch.int64 if name == "cu_seqlens" else torch.float32
                 if name in ("A_log", "dt_bias") else torch.bfloat16)
        if t.device.type != "cuda" or t.device != q.device:
            raise ValueError(f"{name} must be on the same CUDA device as q")
        if t.dtype != dtype or tuple(t.shape) != tuple(shapes[name]):
            raise ValueError(f"{name} must have dtype {dtype} and shape {shapes[name]}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if torch.is_grad_enabled() and t.requires_grad:
            raise RuntimeError("consistent_fwd supports no_grad forward only; backward is unavailable")
    _contract_scalars(scale, eps, lower_bound)
    items = list(tensors.items())
    for i, (name, t) in enumerate(items):
        lo, hi = _range(t)
        for other, u in items[i + 1:]:
            ulo, uhi = _range(u)
            if max(lo, ulo) < min(hi, uhi):
                if {name, other} == {"state_in", "state_out"} and (lo, hi) == (ulo, uhi):
                    continue
                raise ValueError(f"overlapping tensors are forbidden: {name}, {other}")
    # Device-side checks avoid a D2H synchronization, also during graph replay.
    valid = ((cu_seqlens[0] == 0) & (cu_seqlens[-1] <= q.shape[1])
             & torch.all(cu_seqlens[1:] >= cu_seqlens[:-1]))
    torch._assert_async(valid, "invalid cu_seqlens: require 0 <= offsets <= capacity in order")
    return n


def consistent_fwd(q, k, v, g, beta, A_log, dt_bias, cu_seqlens, state_in,
                   out, state_out, *, scale=128 ** -0.5, eps=1e-6, lower_bound=-5.0):
    """Write packed recurrent outputs and BF16 continuation state.

    All buffers are caller-owned, contiguous and on the same CUDA device.
    q/k/v/g/out: BF16 [1,T,H,128]; beta: BF16 [1,T,H].
    A_log: FP32 [H]; dt_bias: FP32 [H,128]; cu_seqlens: int64 [N+1].
    state_in/state_out: BF16 [N,H,V=128,K=128]. g/beta are raw logits.
    T is capacity; only tokens before cu_seqlens[-1] are written. Empty
    sequences preserve state. Exact state aliasing is the only allowed overlap.
    Returns the supplied (out, state_out). See docs/consistency.md for lifetime,
    graph capture, numerical and distributed contracts.
    """
    n = _validate(q, k, v, g, beta, A_log, dt_bias, cu_seqlens, state_in,
                  out, state_out, scale, eps, lower_bound)
    if n == 0:
        return out, state_out
    try:
        from ._consistent_kernel import compiled_kernel
    except ImportError as error:
        raise ImportError("consistent_fwd requires the 'consistency' extra; see docs/consistency.md") from error
    with torch.cuda.device(q.device):
        kernel = compiled_kernel(q.device.index, torch.cuda.get_device_capability(q.device))
        kernel(q, k, v, g, beta, state_out, state_in, out, A_log,
               dt_bias.view(-1), cu_seqlens, float(scale), float(eps), float(lower_bound))
    return out, state_out
