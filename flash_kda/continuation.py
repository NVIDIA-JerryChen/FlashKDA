"""Explicit CP state handoff and request-identified PD continuation buffers."""
from dataclasses import dataclass
import torch
from .consistent import consistent_fwd, NUMERICAL_CONTRACT, _contract_scalars


@dataclass(frozen=True)
class Continuation:
    """Caller-owned state plus host metadata. Storage is borrowed, not cloned.

    The producer must finish on the consumer stream before reuse/transport.
    Contract scalars and software/GPU identity must match at the receiver.
    """
    request_ids: tuple
    positions: tuple
    state: torch.Tensor
    scale: float = 128 ** -0.5
    eps: float = 1e-6
    lower_bound: float = -5.0
    contract: str = NUMERICAL_CONTRACT

    def __post_init__(self):
        _contract_scalars(self.scale, self.eps, self.lower_bound)
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("request IDs must be unique")
        if len(self.positions) != len(self.request_ids) or any(
                not isinstance(p, int) or p < 0 for p in self.positions):
            raise ValueError("positions must be nonnegative host integers, one per request")
        s = self.state
        if (s.ndim != 4 or s.shape[0] != len(self.request_ids) or
                s.shape[2:] != (128, 128) or s.dtype != torch.bfloat16 or
                not s.is_cuda or not s.is_contiguous()):
            raise ValueError("state must be contiguous CUDA BF16 [N,H,128,128]")
        if self.contract != NUMERICAL_CONTRACT:
            raise ValueError("unsupported numerical contract")


def rebase_continuations(parts, request_ids, state_out):
    """Gather selected request states into a new batch without changing bits.

    All sources must share a numerical contract, H, device and software stack.
    The destination must not overlap source storage. New requests must be supplied
    as separate zero-position continuations with caller-initialized states.
    """
    parts = tuple(parts)
    if not parts:
        raise ValueError("at least one source continuation is required")
    first = parts[0]
    key = (first.contract, _contract_scalars(first.scale, first.eps, first.lower_bound))
    lookup = {}
    for part in parts:
        if (part.contract, _contract_scalars(part.scale, part.eps, part.lower_bound)) != key:
            raise ValueError("incompatible numerical contracts/scalars")
        if part.state.device != state_out.device or part.state.shape[1:] != state_out.shape[1:]:
            raise ValueError("state device/head count mismatch")
        if torch._C._overlaps(part.state, state_out):
            raise ValueError("PD destination must not overlap source storage")
        for i, rid in enumerate(part.request_ids):
            if rid in lookup:
                raise ValueError("duplicate source request ID")
            lookup[rid] = (part, i)
    request_ids = tuple(request_ids)
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("duplicate destination request ID")
    if any(rid not in lookup for rid in request_ids):
        raise ValueError("unknown destination request ID")
    positions = tuple(lookup[rid][0].positions[lookup[rid][1]] for rid in request_ids)
    result = Continuation(request_ids, positions, state_out, first.scale, first.eps,
                          first.lower_bound, first.contract)
    for dst, rid in enumerate(request_ids):
        part, src = lookup[rid]
        state_out[dst].copy_(part.state[src])
    return result


def cp_consistent_fwd(q, k, v, g, beta, A_log, dt_bias, cu_seqlens,
                      state_in, out, state_out, *, group=None,
                      scale=128 ** -0.5, eps=1e-6, lower_bound=-5.0):
    """Rank-ordered contiguous time partitions with exact BF16 handoff.

    All ranks call with identical request ordering and state shapes, including
    requests with empty local intervals. On rank > 0, state_in is a writable
    receive buffer. Rank 0 supplies the initial state. Only the last group rank
    holds the final sequence state. Local outputs stay on their producing ranks.
    Uses the current stream, NCCL, and group rank order; no graph capture here.
    """
    import torch.distributed as dist
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("CP transport must run outside CUDA Graph capture")
    if not dist.is_initialized() or dist.get_backend(group) != "nccl":
        raise RuntimeError("CP requires an initialized NCCL process group")
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    def global_rank(r):
        return r if group is None else dist.get_global_rank(group, r)
    if rank:
        dist.recv(state_in, src=global_rank(rank - 1), group=group)
    result = consistent_fwd(q, k, v, g, beta, A_log, dt_bias, cu_seqlens,
                            state_in, out, state_out, scale=scale, eps=eps,
                            lower_bound=lower_bound)
    if rank + 1 < size:
        dist.send(state_out, dst=global_rank(rank + 1), group=group)
    return result
