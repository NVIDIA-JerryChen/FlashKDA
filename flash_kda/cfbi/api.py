"""Python entry points of the CF-BI kernels (prefill, decode, training forward)."""
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ._ext import load_ext
from .state import CfbiState

_ARCH_OK = {}


def _guard(device: torch.device):
    """The bitwise premise (MMA output independent of other rows' data) was validated on B300 only."""
    key = torch.device(device).index if torch.device(device).index is not None else torch.cuda.current_device()
    ok = _ARCH_OK.get(key)
    if ok is None:
        major, minor = torch.cuda.get_device_capability(key)
        ok = major >= 10 or os.environ.get("FLASH_KDA_CFBI_ALLOW_UNVALIDATED_ARCH", "0") == "1"
        if not ok:
            raise RuntimeError(
                f"flash_kda.cfbi was validated on compute capability >= 10.0 (B300) only, this device is {major}.{minor}. "
                "Set FLASH_KDA_CFBI_ALLOW_UNVALIDATED_ARCH=1 to run anyway (the bitwise guarantee is unverified there)."
            )
        _ARCH_OK[key] = ok


def prefill(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, state: CfbiState,
            cu_seqlens: Optional[torch.Tensor] = None, *, out_fp32: Optional[torch.Tensor] = None,
            slot_ids: Optional[torch.Tensor] = None, _mode: int = 0, _impl: int = 0, _k2_bv: int = 0):
    """Append ``T`` tokens per sequence to the sequences' CF-BI state and write the outputs in place.

    Argument names/shapes follow ``flash_kda.fwd``: q, k, v, g bf16 ``[B, T, H, 128]``, beta bf16 raw logits
    ``[B, T, H]``, A_log fp32 ``[H]``, dt_bias fp32 ``[H, 128]``, ``lower_bound`` in ``[-5, 0]``; ``out`` bf16
    ``[B, T, H, 128]``. Without ``cu_seqlens`` every batch entry is one sequence (state entry ``b``); with
    ``cu_seqlens`` (int64 ``[N+1]``, B == 1) sequence ``n`` owns tokens ``cu_seqlens[n]:cu_seqlens[n+1]``.
    ``state`` is updated in place. The bits of ``out`` and ``state`` do not depend on how the sequence is cut
    into calls, nor on the other sequences of the batch.

    ``slot_ids`` (int32 ``[N]``, optional) turns ``state`` into a POOL: sequence ``n`` of the call reads and writes
    pool slot ``slot_ids[n]`` (``alloc_state(S, H)`` with any ``S``), so a serving engine can keep one state pool and
    pass the active sequences' slots without gathering/scattering 224 KiB per head. ``slot_ids[n] == -1`` skips the
    entry (no tokens processed, no state access, ``out`` rows untouched), which keeps fixed-shape CUDA-graph decode
    batches possible. Slots of one call must be distinct. The indirection is resolved on the device, so the contents of
    ``slot_ids`` may change between graph replays.
    """
    _guard(q.device)
    load_ext().forward(q, k, v, g, beta, float(scale), out, A_log, dt_bias, float(lower_bound),
                       state.E, state.rows, state.pos, cu_seqlens, _mode, out_fp32, _impl, slot_ids, k2_bv=_k2_bv)


def decode(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, state: CfbiState, *,
           out_fp32: Optional[torch.Tensor] = None, slot_ids: Optional[torch.Tensor] = None, _impl: int = 0):
    """One token per sequence (``q`` bf16 ``[N, 1, H, 128]`` or ``[N, H, 128]``); same arithmetic as ``prefill``.
    ``slot_ids``: see ``prefill`` (pool slot per sequence, ``-1`` = padding entry)."""
    if q.dim() == 3:
        q, k, v, g, out = (x.unsqueeze(1) for x in (q, k, v, g, out))
        beta = beta.unsqueeze(1)
        if out_fp32 is not None:
            out_fp32 = out_fp32.unsqueeze(1)
    _guard(q.device)
    load_ext().forward(q, k, v, g, beta, float(scale), out, A_log, dt_bias, float(lower_bound),
                       state.E, state.rows, state.pos, None, 1, out_fp32, _impl, slot_ids)


def forward_train(q, k, v, g, beta, scale, A_log, dt_bias, lower_bound,
                  cu_seqlens: Optional[torch.Tensor] = None, *, out: Optional[torch.Tensor] = None,
                  out_fp32: Optional[torch.Tensor] = None, _impl: int = 0, _k2_bv: int = 0) -> torch.Tensor:
    """Training-forward schedule of the same program, from the empty state; nothing is persisted."""
    _guard(q.device)
    if out is None:
        out = torch.empty_like(q)
    load_ext().forward(q, k, v, g, beta, float(scale), out, A_log, dt_bias, float(lower_bound),
                       None, None, None, cu_seqlens, 0, out_fp32, _impl, k2_bv=_k2_bv)
    return out


@dataclass
class PrefillPhase1:
    """What ``prefill_k1`` hands to ``prefill_k2``: the K1 workspace (E-independent chunk coefficients, 139 KiB per
    chunk and head), the call plan and the inputs of the call (kept alive until K2 has run)."""

    ws: torch.Tensor
    plan: torch.Tensor
    inputs: Tuple
    cu_seqlens: Optional[torch.Tensor]
    slot_ids: Optional[torch.Tensor]
    impl: int


def prefill_k1(q, k, v, g, beta, scale, A_log, dt_bias, lower_bound, state: CfbiState,
               cu_seqlens: Optional[torch.Tensor] = None, *, slot_ids: Optional[torch.Tensor] = None,
               _impl: int = 0) -> PrefillPhase1:
    """First half of ``prefill`` for context parallelism: the plan and K1, everything that does NOT need the entry state
    ``E`` (more than half of the prefill work), so every context-parallel rank can run it at the same time.

    Requirements: every sequence of the call starts on a 64-row grid line (``state.pos == 0``: shards of a long sequence
    are cut at multiples of 64 tokens from the sequence start, only the last shard may end inside a chunk); nothing of
    ``E`` is read or written here, a trailing partial chunk's rows go into ``state.rows``. Follow with
    ``prefill_k2`` once the entry ``E`` (the previous shard's final ``E``) is in ``state.E``. The bits of the combination
    are identical to one ``prefill`` call over the concatenated shards."""
    _guard(q.device)
    ext = load_ext()
    B, T, H = q.shape[0], q.shape[1], q.shape[2]
    N = (cu_seqlens.numel() - 1) if cu_seqlens is not None else B
    ws_n, plan_n = ext.workspace_sizes(B * T, N, H)
    ws = torch.empty(ws_n, dtype=torch.float32, device=q.device)
    plan = torch.empty(plan_n, dtype=torch.int32, device=q.device)
    ext.forward(q, k, v, g, beta, float(scale), q, A_log, dt_bias, float(lower_bound), state.E, state.rows, state.pos,
                cu_seqlens, 0, None, _impl, slot_ids, ws, plan, 1)   # `out` (= q) is not written by K1
    return PrefillPhase1(ws, plan, (q, k, v, g, beta, scale, A_log, dt_bias, lower_bound), cu_seqlens, slot_ids, _impl)


def prefill_k2(h1: PrefillPhase1, out, state: CfbiState, *, out_fp32: Optional[torch.Tensor] = None):
    """Second half of ``prefill`` (see ``prefill_k1``): the serial scan over the chunks from ``state.E``; writes ``out``,
    updates ``state`` (E, row cache, pos) in place."""
    q, k, v, g, beta, scale, A_log, dt_bias, lower_bound = h1.inputs
    _guard(q.device)
    load_ext().forward(q, k, v, g, beta, float(scale), out, A_log, dt_bias, float(lower_bound), state.E, state.rows,
                       state.pos, h1.cu_seqlens, 0, out_fp32, h1.impl, h1.slot_ids, h1.ws, h1.plan, 2)
