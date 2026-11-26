"""Chunk-64 bf16 KDA: prefill (``fwd64``) and a bit-for-bit aligned decode (open-chunk replay).

``fwd64`` is a bf16 program in the style of ``flash_kda.fwd`` (bf16 intermediates, bf16 tensor-core products with fp32
accumulate, bf16 state rounded once per chunk end) with a 64-token chunk: sub-blocks of 16 rows with reference rows keep
every exponential in range, so the lower-bound restriction of the chunk-16 kernel does not tighten. ``prefill`` /
``decode`` / ``final_state`` below give, for every token, the bits of that token's row in ONE ``fwd64`` call over the
sequence so far (same inputs, same initial state), however the sequence is cut into calls (prompt, chunked-prefill pieces
at any offset, one or several decode tokens per step, any batch composition, slot pools).

How. ``fwd64`` walks a sequence in chunks of 64 tokens anchored at its first token and keeps the state in bf16, updated
only at chunk ends; inside a chunk, row r depends only on rows 0..r and on the chunk-start state. So the 64-aligned middle
of a prefill call runs through ``fwd64`` and every token of a chunk that is not complete yet (prompt tail, decode tokens,
the head of a call that continues an open chunk) runs through the replay kernel, which re-executes the chunk arithmetic
(the same device functions as ``fwd64``) on the open chunk from the chunk-start state and emits only the new rows.

State per (sequence, head) (``ReplayState64``): ``S`` bf16 ``[V, K]`` at the last 64-token boundary, the raw k / v / g /
beta of the open chunk's rows, ``pos`` = rows in the open chunk (one copy per head, so a decode step is one launch).

Preconditions (not checked on the device): distinct non-negative ``slot_ids`` within a call (``decode`` also accepts -1 =
padding entry); ``out`` does not overlap q / k / v / g / beta; one stream owns the state while a call runs.
"""
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

import flash_kda_C64 as _C

CHUNK = 64
HEAD_DIM = 128
_K2_CFG_ENV = os.environ.get("FLASH_KDA_C64_K2_CFG")   # force a K2 CTA shape (experiments): see launch_fwd64


def _auto_k2_cfg(n_seg: int, H: int) -> int:
    """K2 has one serial CTA chain per (sequence, head); splitting the 128 value columns over more CTAs (same bits) pays
    while the GPU has idle SMs: <= 16 chains -> 4 CTAs of 2 warps, <= 64 -> 2 CTAs of 4 warps, else 1 CTA of 8 warps."""
    nsh = n_seg * H
    return 3 if nsh <= 16 else (2 if nsh <= 64 else 1)

__all__ = ["CHUNK", "ReplayState64", "alloc_state", "fwd64", "prefill", "decode", "final_state"]


@dataclass
class ReplayState64:
    """Per (slot, head) state; ``P`` slots, ``H`` heads, K = V = 128.

    S:         bf16  ``[P, H, 128, 128]``  state at the start of the open chunk (``[V, K]``)
    rows_k:    bf16  ``[P, H, 64, 128]``   raw k of the open chunk's rows (only rows < pos are meaningful)
    rows_v:    bf16  ``[P, H, 64, 128]``   raw v
    rows_g:    bf16  ``[P, H, 64, 128]``   raw g (gate input before activation)
    rows_beta: bf16  ``[P, H, 64]``        raw beta logits
    pos:       int32 ``[P, H]``            rows in the open chunk (0..63; equal for all heads of a slot)
    """

    S: torch.Tensor
    rows_k: torch.Tensor
    rows_v: torch.Tensor
    rows_g: torch.Tensor
    rows_beta: torch.Tensor
    pos: torch.Tensor

    @property
    def P(self) -> int:
        return self.S.shape[0]

    @property
    def H(self) -> int:
        return self.S.shape[1]

    def clone(self) -> "ReplayState64":
        return ReplayState64(*(t.clone() for t in self._tensors()))

    def select(self, idx) -> "ReplayState64":
        return ReplayState64(*(t[idx].clone() for t in self._tensors()))

    def _tensors(self):
        return (self.S, self.rows_k, self.rows_v, self.rows_g, self.rows_beta, self.pos)


def alloc_state(P: int, H: int, device="cuda", initial_state: Optional[torch.Tensor] = None) -> ReplayState64:
    """``P`` empty slots. ``initial_state`` (bf16 or fp32 ``[P, H, 128, 128]``) seeds ``S`` (fp32 is rounded to
    nearest-even bf16, once, as ``fwd64`` does on entry)."""
    S = torch.zeros(P, H, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device=device)
    if initial_state is not None:
        assert initial_state.shape == S.shape, f"initial_state must be {tuple(S.shape)}"
        assert initial_state.dtype in (torch.bfloat16, torch.float32)
        S.copy_(initial_state)
    rows = lambda: torch.zeros(P, H, CHUNK, HEAD_DIM, dtype=torch.bfloat16, device=device)  # noqa: E731
    return ReplayState64(
        S=S, rows_k=rows(), rows_v=rows(), rows_g=rows(),
        rows_beta=torch.zeros(P, H, CHUNK, dtype=torch.bfloat16, device=device),
        pos=torch.zeros(P, H, dtype=torch.int32, device=device),
    )


# ---------------------------------------------------------------------------------------------------------------------
def _flat(q, k, v, g, beta, out):
    """[B, T, H, D] -> packed [B*T, H, D] views (``out`` must be writable in place)."""
    B, T, H, D = q.shape
    assert D == HEAD_DIM, "only K = V = 128 is supported"
    for t in (q, k, v, g, out):
        assert t.shape == q.shape and t.dtype == torch.bfloat16 and t.is_contiguous()
    assert beta.shape == (B, T, H) and beta.dtype == torch.bfloat16 and beta.is_contiguous()
    n = B * T
    return (q.view(n, H, D), k.view(n, H, D), v.view(n, H, D), g.view(n, H, D), beta.view(n, H), out.view(n, H, D))


def _entries(q, cu_seqlens) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """(N, first token row, token count) per sequence of the call; no device synchronisation."""
    B, T = q.shape[0], q.shape[1]
    if cu_seqlens is None:
        starts = torch.arange(B, device=q.device, dtype=torch.int64) * T
        lens = torch.full((B,), T, device=q.device, dtype=torch.int64)
        return B, starts, lens
    assert B == 1, "B must be 1 when cu_seqlens is provided"
    assert cu_seqlens.dtype == torch.int64 and cu_seqlens.dim() == 1
    return cu_seqlens.numel() - 1, cu_seqlens[:-1].contiguous(), (cu_seqlens[1:] - cu_seqlens[:-1]).contiguous()


def _slots(slot_ids, N):
    if slot_ids is None:
        return None
    assert slot_ids.dim() == 1 and slot_ids.numel() == N
    return slot_ids.to(torch.int32).contiguous()


def _launch_replay(x, scale, A_log, dt_bias, lower_bound, state, n_entries, out, *, tok_start=None, tok_count=None,
                   tokens_per_entry=0, out_row=None, slot_ids=None, final_out=None):
    q3, k3, v3, g3, beta2 = x
    _C.replay64(q3, k3, v3, g3, beta2, float(scale), out, A_log, dt_bias, float(lower_bound),
                state.S, state.rows_k, state.rows_v, state.rows_g, state.rows_beta, state.pos,
                int(n_entries), int(tokens_per_entry),
                tok_start=tok_start, tok_count=None if tok_count is None else tok_count.to(torch.int32),
                out_row=out_row, slot_ids=slot_ids, final_out=final_out)


def _launch_fwd(x, out3, scale, A_log, dt_bias, lower_bound, seg_tok0, seg_len, seg_slot, total_tiles, st_in, st_out,
                k2_cfg=None):
    N = seg_len.numel()
    H = x[0].shape[1]
    ws = torch.empty(_C.fwd64_ws_size(int(total_tiles), H, N), dtype=torch.uint8, device=x[0].device)
    _C.fwd64(x[0], x[1], x[2], x[3], x[4], float(scale), out3, A_log, dt_bias, float(lower_bound),
             seg_tok0.contiguous(), seg_len.to(torch.int32).contiguous(), seg_slot.to(torch.int32).contiguous(),
             int(total_tiles), ws, state_in=st_in, state_out=st_out,
             k2_cfg=(int(_K2_CFG_ENV) if _K2_CFG_ENV is not None else _auto_k2_cfg(N, H)) if k2_cfg is None else int(k2_cfg))


# ---------------------------------------------------------------------------------------------------------------------
def fwd64(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, initial_state=None, final_state=None,
          cu_seqlens=None, *, _k2_cfg=None):
    """One chunk-64 prefill call (the specification of this module). Arguments as ``flash_kda.fwd``; the states are
    bf16 (an fp32 ``initial_state`` is rounded to bf16 once; ``final_state`` must be bf16)."""
    x6 = _flat(q, k, v, g, beta, out)
    x, out3 = x6[:5], x6[5]
    N, starts, lens = _entries(q, cu_seqlens)
    T_total = x[0].shape[0]
    if cu_seqlens is None:
        total_tiles = N * ((q.shape[1] + CHUNK - 1) // CHUNK)
    else:
        total_tiles = (T_total + CHUNK - 1) // CHUNK + N
    st_in = None
    if initial_state is not None:
        st_in = initial_state if initial_state.dtype == torch.bfloat16 else initial_state.to(torch.bfloat16)
        st_in = st_in.contiguous()
    if final_state is not None:
        assert final_state.dtype == torch.bfloat16, "fwd64 keeps a bf16 state"
    seg_slot = torch.arange(N, device=q.device, dtype=torch.int32)
    _launch_fwd(x, out3, scale, A_log, dt_bias, lower_bound, starts, lens, seg_slot, total_tiles, st_in, final_state,
                k2_cfg=_k2_cfg)


# ---------------------------------------------------------------------------------------------------------------------
def prefill(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, state: ReplayState64,
            cu_seqlens: Optional[torch.Tensor] = None, slot_ids: Optional[torch.Tensor] = None, *,
            context_lens_cpu=None, cu_seqlens_cpu=None, _path: Optional[str] = None, _k2_cfg=None):
    """Append ``T`` tokens per sequence to ``state`` and write their outputs into ``out`` (in place).

    Arguments follow ``flash_kda.fwd``; sequence ``n`` continues state slot ``slot_ids[n]`` (default ``n``), every
    ``slot_ids`` entry must be a distinct slot >= 0. The 64-aligned middle of every sequence goes through ``fwd64``; the
    rows that complete an open chunk and the trailing rows that leave a chunk open go through the replay kernel.
    Host plan: with ``context_lens_cpu`` (tokens each sequence already holds, host ints; their ``% 64`` must equal the
    device ``pos``, not checked) and, with ``cu_seqlens``, ``cu_seqlens_cpu``, the split is planned on the host: only
    non-empty phases are launched and the ``fwd64`` segment table holds only non-empty bodies. Without them the plan
    is built on the device (no synchronisation, CUDA-graph capturable). Both give the same bits.

    ``_path`` (tests): ``"replay"`` runs everything through the replay kernel, ``"fwd"`` forces the split; by default the
    split is used when the call carries more than 64 tokens per sequence on average.
    """
    x6 = _flat(q, k, v, g, beta, out)
    x, out3 = x6[:5], x6[5]
    T_total, H = x[0].shape[0], x[0].shape[1]
    N, starts, lens = _entries(q, cu_seqlens)
    slots = _slots(slot_ids, N)
    if slots is None and N > state.P:
        raise ValueError(f"flash_kda.replay64.prefill: {N} sequences but the state has {state.P} slots (pass slot_ids)")
    path = _path or ("fwd" if T_total > CHUNK * N else "replay")
    assert path in ("fwd", "replay")
    if path == "replay":
        _launch_replay(x, scale, A_log, dt_bias, lower_bound, state, N, out3, tok_start=starts, tok_count=lens,
                       slot_ids=slots)
        return
    for t in (x[0], x[1], x[2], x[3], out3):
        if t.data_ptr() % 16:
            raise ValueError("flash_kda.replay64.prefill: q/k/v/g/out must be 16-byte aligned")

    if context_lens_cpu is not None:
        B, T = q.shape[0], q.shape[1]
        if cu_seqlens is None:
            starts_h, lens_h = [b * T for b in range(B)], [T] * B
        else:
            assert cu_seqlens_cpu is not None, "context_lens_cpu with cu_seqlens needs cu_seqlens_cpu"
            cu_h = [int(c) for c in cu_seqlens_cpu]
            starts_h, lens_h = cu_h[:-1], [b - a for a, b in zip(cu_h[:-1], cu_h[1:])]
        ctx_h = [int(c) for c in context_lens_cpu]
        assert len(ctx_h) == N and len(lens_h) == N
        _prefill_host(x, out3, scale, A_log, dt_bias, lower_bound, state, slots, starts_h, lens_h, ctx_h, _k2_cfg)
        return

    dev = q.device
    slot_idx = slots.to(torch.int64) if slots is not None else torch.arange(N, device=dev)
    r0 = state.pos[slot_idx, 0].to(torch.int64)
    head = torch.minimum(lens, torch.remainder(CHUNK - r0, CHUNK))   # rows that complete the open chunk
    body = torch.div(lens - head, CHUNK, rounding_mode="floor") * CHUNK
    tail = lens - head - body                                         # rows that stay in a new open chunk

    # A. heads (outputs go straight into `out`)
    _launch_replay(x, scale, A_log, dt_bias, lower_bound, state, N, out3, tok_start=starts, tok_count=head,
                   slot_ids=slots)
    # B. bodies: one fwd64 call over a segment table (every body is a multiple of 64 rows from a chunk boundary)
    total_tiles = max(T_total // CHUNK, 1)
    _launch_fwd(x, out3, scale, A_log, dt_bias, lower_bound, starts + head, body, slot_idx.to(torch.int32), total_tiles,
                state.S, state.S, k2_cfg=_k2_cfg)
    # C. tails (pos is 0 here whenever tail > 0)
    _launch_replay(x, scale, A_log, dt_bias, lower_bound, state, N, out3, tok_start=starts + head + body,
                   tok_count=tail, slot_ids=slots)


def _host_to_device(dev, *lists):
    """Several host int lists -> int64 device views, through ONE pinned, non-blocking copy (no host-device sync)."""
    flat = [int(v) for lst in lists for v in lst]
    t = torch.tensor(flat, dtype=torch.int64).pin_memory().to(dev, non_blocking=True)
    views, o = [], 0
    for lst in lists:
        views.append(t[o:o + len(lst)])
        o += len(lst)
    return views


def _prefill_host(x, out3, scale, A_log, dt_bias, lower_bound, state, slots, starts_h, lens_h, ctx_h, k2_cfg):
    """prefill() with the split planned on the host (see its docstring); same bits as the device plan."""
    dev = x[0].device
    N = len(lens_h)
    r0 = [c % CHUNK for c in ctx_h]
    head = [min(L, (CHUNK - r) % CHUNK) for L, r in zip(lens_h, r0)]
    body = [(L - h) // CHUNK * CHUNK for L, h in zip(lens_h, head)]
    tail = [L - h - b for L, h, b in zip(lens_h, head, body)]
    bidx = [n for n in range(N) if body[n] > 0]
    d_starts, d_head, d_tstart, d_tail, d_bstart, d_blen, d_bidx = _host_to_device(
        dev, starts_h, head, [s + h + b for s, h, b in zip(starts_h, head, body)], tail,
        [starts_h[n] + head[n] for n in bidx], [body[n] for n in bidx], bidx)
    if any(head):   # A. rows that complete an open chunk
        _launch_replay(x, scale, A_log, dt_bias, lower_bound, state, N, out3, tok_start=d_starts, tok_count=d_head,
                       slot_ids=slots)
    if bidx:        # B. the 64-aligned bodies: one fwd64 launch over the non-empty ones
        seg_slot = slots[d_bidx] if slots is not None else d_bidx
        _launch_fwd(x, out3, scale, A_log, dt_bias, lower_bound, d_bstart, d_blen, seg_slot, sum(body) // CHUNK,
                    state.S, state.S, k2_cfg=k2_cfg)
    if any(tail):   # C. trailing rows -> replay (they become the open chunk)
        _launch_replay(x, scale, A_log, dt_bias, lower_bound, state, N, out3, tok_start=d_tstart, tok_count=d_tail,
                       slot_ids=slots)


def decode(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, state: ReplayState64,
           slot_ids: Optional[torch.Tensor] = None):
    """Decode step: q, k, v, g, out bf16 ``[N, H, 128]`` (one token per sequence) or ``[N, T, H, 128]`` (T tokens per
    sequence), beta ``[N, H]`` / ``[N, T, H]``. Same bits as ``prefill`` / one ``fwd64`` call.
    ``slot_ids[n] == -1`` skips entry ``n`` (no state access, its ``out`` rows untouched)."""
    if q.dim() == 3:
        q, k, v, g, out = (t.unsqueeze(1) for t in (q, k, v, g, out))
        beta = beta.unsqueeze(1)
    x6 = _flat(q, k, v, g, beta, out)
    N, T = q.shape[0], q.shape[1]
    _launch_replay(x6[:5], scale, A_log, dt_bias, lower_bound, state, N, x6[5], tokens_per_entry=T,
                   slot_ids=_slots(slot_ids, N))


def final_state(state: ReplayState64, A_log, dt_bias, lower_bound, *, dtype=torch.bfloat16,
                slot_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``fwd64``'s ``final_state`` for the tokens appended so far: ``S`` with the open chunk's rows closed as a short last
    chunk (computed by the replay kernel). ``state`` is not changed."""
    dev = state.S.device
    idx = slot_ids.to(torch.int32).contiguous() if slot_ids is not None else torch.arange(state.P, device=dev, dtype=torch.int32)
    n, H = idx.numel(), state.H
    res = torch.empty(n, H, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    z = torch.zeros(1, H, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    zb = torch.zeros(1, H, dtype=torch.bfloat16, device=dev)
    _launch_replay((z, z, z, z, zb), 1.0, A_log, dt_bias, lower_bound, state, n, z, slot_ids=idx, final_out=res)
    return res.to(dtype) if dtype != torch.bfloat16 else res
