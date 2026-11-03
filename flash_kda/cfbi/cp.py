"""Context parallelism for the CF-BI prefill / training forward, bit-identical to the single-rank result.

A long sequence is cut at multiples of 64 tokens from its start and one shard goes to each rank. K1 (all the
E-independent work: coefficients, triangular solves, decay products) needs only the shard's own tokens, so every rank
runs it at the same time. K2 is the recurrence over chunks and needs the entry state of its shard, which is the final
``E`` (fp32, 64 KiB per head) of the previous shard: ranks run K2 one after the other and pass ``E`` along. Because the
program is anchored at the sequence start and ``E`` is handed over unchanged, the outputs and the final state are the
same bits as one ``prefill`` call on one device. (Associative-scan style CP would re-associate the recurrence and
change the bits; this one trades the serial K2 chain for exactness.)
"""
from typing import Optional

import torch
import torch.distributed as dist

from .api import prefill_k1, prefill_k2
from .state import CfbiState


def cp_prefill(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, state: CfbiState, *, group=None,
               rank: Optional[int] = None, world_size: Optional[int] = None, transport: str = "device",
               out_fp32: Optional[torch.Tensor] = None):
    """One rank's part of a context-parallel prefill of ONE sequence (``q`` etc. ``[1, T_r, H, 128]``).

    ``T_r`` must be a multiple of 64 on every rank but the last. ``state`` is this rank's state object (``N == 1``): the
    first rank's state is the sequence's initial state (``pos`` must be 0); on the other ranks only ``rows``/``pos``
    scratch matter, ``E`` is received. After the call ``state`` of the LAST rank is the sequence's final state; the other
    ranks' ``E`` holds their shard's final ``E`` (what they sent on). ``transport="device"`` sends the CUDA tensor
    (NCCL), ``"cpu"`` goes through host memory (gloo; for tests)."""
    rank = dist.get_rank(group) if rank is None else rank
    world = dist.get_world_size(group) if world_size is None else world_size
    h1 = prefill_k1(q, k, v, g, beta, scale, A_log, dt_bias, lower_bound, state)   # in parallel on all ranks
    if rank > 0:
        src = dist.get_global_rank(group, rank - 1) if group is not None else rank - 1
        if transport == "cpu":
            buf = torch.empty(state.E.shape, dtype=torch.float32)
            dist.recv(buf, src=src, group=group)
            state.E.copy_(buf)
        else:
            dist.recv(state.E, src=src, group=group)
    prefill_k2(h1, out, state, out_fp32=out_fp32)
    if rank < world - 1:
        dst = dist.get_global_rank(group, rank + 1) if group is not None else rank + 1
        if transport == "cpu":
            dist.send(state.E.cpu(), dst=dst, group=group)
        else:
            dist.send(state.E, dst=dst, group=group)
