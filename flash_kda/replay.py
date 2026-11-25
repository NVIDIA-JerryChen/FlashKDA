"""Prefill + decode that are bit-for-bit equal to ONE ``flash_kda.fwd`` call over the whole sequence.

``flash_kda.fwd`` (the chunk-16 prefill kernel, used unmodified) is the specification: for every token, the output of
``prefill`` / ``decode`` below is the bits of that token's row in one ``flash_kda.fwd`` call over the sequence so far
(same inputs, same initial state), however the sequence is cut into calls (prompt, chunked-prefill pieces at any
offset, one or several decode tokens per step, any batch composition).

How. ``flash_kda.fwd`` walks a sequence in chunks of 16 tokens anchored at its first token, keeps the state in bf16 and
updates it only at chunk ends; inside a chunk, row r depends only on rows 0..r and on the chunk-start state. So:

* the 16-aligned middle of a prefill call runs through ``flash_kda.fwd`` itself;
* tokens of a chunk that is not complete yet (the prompt tail, decode tokens, the head of a call that continues an open
  chunk) run through the replay kernel (``csrc/replay``), which re-executes ``flash_kda.fwd``'s chunk arithmetic on the
  open chunk from the chunk-start state and emits only the new rows.

State per (sequence, head) (``ReplayState``): ``S`` bf16 ``[V, K]`` = the state at the last 16-token boundary (the layout
of ``flash_kda.fwd`` states), the raw k / v / g / beta of the open chunk's rows, ``pos`` = rows in the open chunk (kept
per head, so that a decode step is a single kernel launch that advances it).
An fp32 initial state is rounded to bf16 once, as ``flash_kda.fwd`` does on entry. ``final_state`` returns what
``flash_kda.fwd``'s ``final_state`` would be (the open chunk closed as a short last chunk) without changing the state.

Preconditions (not checked on the device): distinct non-negative ``slot_ids`` within a call (``decode`` also accepts -1
= padding entry; a slot >= P traps); ``pos`` in 0..15 (a corrupted ``pos`` traps); ``out`` does not overlap q / k / v /
g / beta (the tail rows are read after the bodies' outputs are written); one stream owns the state while a call runs.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

import flash_kda_C as _C

CHUNK = 16
HEAD_DIM = 128

__all__ = ["CHUNK", "ReplayState", "alloc_state", "prefill", "decode", "final_state"]


@dataclass
class ReplayState:
    """Per (slot, head) replay state; ``P`` slots, ``H`` heads, K = V = 128.

    S:         bf16  ``[P, H, 128, 128]``  state at the start of the open chunk (``[V, K]``, like flash_kda.fwd states)
    rows_k:    bf16  ``[P, H, 16, 128]``   raw k of the open chunk's rows (only rows < pos are meaningful)
    rows_v:    bf16  ``[P, H, 16, 128]``   raw v
    rows_g:    bf16  ``[P, H, 16, 128]``   raw g (gate input before activation)
    rows_beta: bf16  ``[P, H, 16]``        raw beta logits
    pos:       int32 ``[P, H]``            rows in the open chunk (0..15; equal for all heads of a slot); the chunk
                                           grid starts at the sequence start
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

    def clone(self) -> "ReplayState":
        return ReplayState(*(t.clone() for t in self._tensors()))

    def select(self, idx) -> "ReplayState":
        """A copy restricted to some slots."""
        return ReplayState(*(t[idx].clone() for t in self._tensors()))

    def _tensors(self):
        return (self.S, self.rows_k, self.rows_v, self.rows_g, self.rows_beta, self.pos)


def alloc_state(P: int, H: int, device="cuda", initial_state: Optional[torch.Tensor] = None) -> ReplayState:
    """``P`` empty slots (zero state, empty open chunk). ``initial_state`` (bf16 or fp32 ``[P, H, 128, 128]``, the
    ``flash_kda.fwd`` layout) seeds ``S``; fp32 is rounded to nearest-even bf16, which is what flash_kda.fwd does."""
    S = torch.zeros(P, H, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device=device)
    if initial_state is not None:
        assert initial_state.shape == S.shape, f"initial_state must be {tuple(S.shape)}"
        assert initial_state.dtype in (torch.bfloat16, torch.float32)
        S.copy_(initial_state)
    rows = lambda: torch.zeros(P, H, CHUNK, HEAD_DIM, dtype=torch.bfloat16, device=device)  # noqa: E731
    return ReplayState(
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


def _launch(x, scale, A_log, dt_bias, lower_bound, state, n_entries, out, *, tok_start=None, tok_count=None,
            tokens_per_entry=0, out_row=None, slot_ids=None):
    """One launch of the replay kernel; it also advances ``state.pos`` of the slots it touches."""
    q3, k3, v3, g3, beta2 = x
    _C.replay(q3, k3, v3, g3, beta2, float(scale), out, A_log, dt_bias, float(lower_bound),
              state.S, state.rows_k, state.rows_v, state.rows_g, state.rows_beta, state.pos,
              int(n_entries), int(tokens_per_entry),
              tok_start=tok_start, tok_count=None if tok_count is None else tok_count.to(torch.int32),
              out_row=out_row, slot_ids=slot_ids)


def _slots(slot_ids, N):
    if slot_ids is None:
        return None
    assert slot_ids.dim() == 1 and slot_ids.numel() == N
    return slot_ids.to(torch.int32).contiguous()


# ---------------------------------------------------------------------------------------------------------------------
def prefill(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, state: ReplayState,
            cu_seqlens: Optional[torch.Tensor] = None, slot_ids: Optional[torch.Tensor] = None, *,
            context_lens_cpu=None, cu_seqlens_cpu=None, _path: Optional[str] = None):
    """Append ``T`` tokens per sequence to ``state`` and write their outputs into ``out`` (in place).

    Arguments follow ``flash_kda.fwd``: q, k, v, g bf16 ``[B, T, H, 128]``, beta bf16 logits ``[B, T, H]``, A_log fp32
    ``[H]``, dt_bias fp32 ``[H, 128]``, ``lower_bound``, ``out`` bf16 ``[B, T, H, 128]``; without ``cu_seqlens`` batch
    entry ``b`` is a sequence, with ``cu_seqlens`` (int64 ``[N+1]``, B == 1) sequence ``n`` owns tokens
    ``cu_seqlens[n]:cu_seqlens[n+1]``. Sequence ``n`` continues state slot ``slot_ids[n]`` (default ``n``); every
    ``slot_ids`` entry must be a distinct slot >= 0 here (padding entries are a ``decode`` feature).

    The 16-aligned middle of every sequence goes through ``flash_kda.fwd`` (unmodified); the rows that complete an open
    chunk and the trailing rows that leave a chunk open go through the replay kernel.

    Host fast path: pass ``context_lens_cpu`` (tokens each sequence already holds, host ints; their ``% 16`` must equal
    the device ``pos`` of the slot, which is not checked) and, with ``cu_seqlens``, its host copy ``cu_seqlens_cpu``.
    The split is then planned on the host: no device-side index arithmetic, only non-empty segments become
    pseudo-sequences of the ``flash_kda.fwd`` call, and when no sequence has a head or a tail (fresh prompts whose
    lengths are multiples of 16, no ``slot_ids``) the call IS one plain ``flash_kda.fwd`` call on the state pool (with
    ``slot_ids`` the bodies' states are gathered into / scattered from a scratch buffer). Without them, the plan
    is built on the device (no synchronisation, CUDA-graph capturable). Both give the same bits.

    ``_path`` (tests): ``"replay"`` runs everything through the replay kernel, ``"fwd"`` forces the split; by default
    the split is used when the call carries more than 16 tokens per sequence on average.
    """
    x6 = _flat(q, k, v, g, beta, out)
    x, out3 = x6[:5], x6[5]
    T_total, H = x[0].shape[0], x[0].shape[1]
    N, starts, lens = _entries(q, cu_seqlens)
    slots = _slots(slot_ids, N)
    if slots is None and N > state.P:
        raise ValueError(f"flash_kda.replay.prefill: {N} sequences but the state has {state.P} slots (pass slot_ids)")
    path = _path or ("fwd" if T_total > CHUNK * N else "replay")
    assert path in ("fwd", "replay")
    if path == "fwd":
        _check_fwd_alignment(x, out3)   # before any phase touches the state
    if context_lens_cpu is not None and path == "fwd":
        B, T = q.shape[0], q.shape[1]
        if cu_seqlens is None:
            starts_h, lens_h = [b * T for b in range(B)], [T] * B
        else:
            assert cu_seqlens_cpu is not None, "context_lens_cpu with cu_seqlens needs cu_seqlens_cpu"
            cu_h = [int(c) for c in cu_seqlens_cpu]
            starts_h, lens_h = cu_h[:-1], [b - a for a, b in zip(cu_h[:-1], cu_h[1:])]
        ctx_h = [int(c) for c in context_lens_cpu]
        assert len(ctx_h) == N and len(lens_h) == N
        _prefill_host(q, k, v, g, beta, out, x, out3, scale, A_log, dt_bias, lower_bound, state, slots,
                      cu_seqlens, starts_h, lens_h, ctx_h)
        return
    if path == "replay":
        _launch(x, scale, A_log, dt_bias, lower_bound, state, N, out3, tok_start=starts, tok_count=lens, slot_ids=slots)
        return

    dev = q.device
    slot_idx = slots.to(torch.int64) if slots is not None else torch.arange(N, device=dev)
    r0 = state.pos[slot_idx, 0].to(torch.int64)
    head = torch.minimum(lens, torch.remainder(CHUNK - r0, CHUNK))   # rows that complete the open chunk (0 if none)
    body = torch.div(lens - head, CHUNK, rounding_mode="floor") * CHUNK
    tail = lens - head - body                                         # rows that stay in a new open chunk

    # A. heads: replay into a side buffer (flash_kda.fwd below writes those rows of `out` too)
    head_row = torch.arange(N, device=dev, dtype=torch.int64) * CHUNK
    head_out = torch.empty(N * CHUNK, H, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    _launch(x, scale, A_log, dt_bias, lower_bound, state, N, head_out, tok_start=starts, tok_count=head,
            out_row=head_row, slot_ids=slots)

    # B. bodies: ONE unmodified flash_kda.fwd call over the whole packed input, cut into 2N+1 pseudo-sequences
    #    [gap_0 | body_0 | gap_1 | body_1 | ... | body_{N-1} | gap_N], gap_0 = head_0, gap_n = tail_{n-1} + head_n,
    #    gap_N = tail_{N-1}. Gaps are throw-away (their rows are rewritten in C / D, their states dropped; an empty gap
    #    is a zero-length sequence). Body n is pseudo-sequence 2n+1; its 16-aligned grid starts at a chunk boundary.
    body_start = starts + head
    body_end = body_start + body
    cu2 = torch.empty(2 * N + 2, dtype=torch.int64, device=dev)
    cu2[:1] = starts[:1]
    cu2[1:2 * N:2] = body_start
    cu2[2:2 * N + 1:2] = body_end
    cu2[2 * N + 1:] = starts[-1:] + lens[-1:]
    # One buffer is both initial_state and final_state: each K2 CTA loads its own slot first and stores it last. Gap
    # slots are never initialised (they only feed throw-away rows of their own pseudo-sequence).
    st2 = torch.empty(2 * N + 1, H, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    st2[1::2].copy_(state.S[slot_idx] if slots is not None else state.S[:N])
    _fwd_bodies(x, out3, scale, A_log, dt_bias, lower_bound, st2, cu2)
    if slots is not None:
        state.S[slot_idx] = st2[1::2]      # an empty body is a zero-length sequence: final state == initial state
    else:
        state.S[:N].copy_(st2[1::2])

    # C. tails: replay (pos is 0 here whenever tail > 0)
    _launch(x, scale, A_log, dt_bias, lower_bound, state, N, out3, tok_start=starts + head + body, tok_count=tail,
            slot_ids=slots)

    # D. head rows into place
    _C.replay_copy_rows(out3, head_out, starts, head_row, head.to(torch.int32), CHUNK)


def _check_fwd_alignment(x, out3):
    """flash_kda.fwd hands every base pointer to TMA (16-byte alignment). q/k/v/g/out rows are 256 B apart, so only an
    unaligned base can fail (beta is handled in _fwd_bodies)."""
    for t, name in ((x[0], "q"), (x[1], "k"), (x[2], "v"), (x[3], "g"), (out3, "out")):
        if t.data_ptr() % 16:
            raise ValueError(f"flash_kda.replay.prefill: {name} must be 16-byte aligned for flash_kda.fwd")


def _fwd_bodies(x, out3, scale, A_log, dt_bias, lower_bound, initial_final, cu):
    """The unmodified flash_kda.fwd over the packed call with pseudo-sequence boundaries ``cu``."""
    # beta is transposed inside fwd by a .contiguous() that does not copy when T_total == 1 or H == 1, so a beta view at
    # an odd token offset would reach TMA as is: copy it (T_total * H * 2 bytes).
    beta_f = x[4] if x[4].data_ptr() % 16 == 0 else x[4].clone()
    from . import fwd as _fwd
    _fwd(x[0][None], x[1][None], x[2][None], x[3][None], beta_f[None], scale, out3[None], A_log, dt_bias, lower_bound,
         initial_state=initial_final, final_state=initial_final, cu_seqlens=cu)


def _host_to_device(dev, *lists):
    """Several host int lists -> int64 device views, through ONE pinned, non-blocking copy (no host-device sync)."""
    flat = [int(v) for lst in lists for v in lst]
    t = torch.tensor(flat, dtype=torch.int64).pin_memory().to(dev, non_blocking=True)
    views, o = [], 0
    for lst in lists:
        views.append(t[o:o + len(lst)])
        o += len(lst)
    return views


def _prefill_host(q, k, v, g, beta, out, x, out3, scale, A_log, dt_bias, lower_bound, state, slots, cu_seqlens,
                  starts_h, lens_h, ctx_h):
    """prefill() with the split planned on the host (see its docstring); same bits as the device plan."""
    dev = x[0].device
    N, H = len(lens_h), x[0].shape[1]
    r0 = [c % CHUNK for c in ctx_h]
    head = [min(L, (CHUNK - r) % CHUNK) for L, r in zip(lens_h, r0)]
    body = [(L - h) // CHUNK * CHUNK for L, h in zip(lens_h, head)]
    tail = [L - h - b for L, h, b in zip(lens_h, head, body)]

    # the pseudo-sequences of the flash_kda.fwd call: only non-empty segments, throw-away ones merged
    cuts, body_pseudo, body_seq = [starts_h[0]], [], []
    if any(body):
        cur, pending = starts_h[0], False
        for n in range(N):
            for seg, real in ((head[n], False), (body[n], True), (tail[n], False)):
                if seg == 0:
                    continue
                if real:
                    if pending:
                        cuts.append(cur)
                        pending = False
                    body_pseudo.append(len(cuts) - 1)
                    body_seq.append(n)
                    cur += seg
                    cuts.append(cur)
                else:
                    cur += seg
                    pending = True
        if pending:
            cuts.append(cur)
    plain = slots is None and not any(head) and not any(tail) and body_pseudo == list(range(N))
    if plain:
        # every sequence is exactly one 16-aligned body from a chunk boundary: this IS one plain flash_kda.fwd call,
        # on the state pool in place, with the caller's own cu_seqlens (nothing to copy to the device)
        pool = state.S[:N]
        if cu_seqlens is None:
            from . import fwd as _fwd
            beta_b = beta if beta.data_ptr() % 16 == 0 else beta.clone()   # see _fwd_bodies (alignment checked above)
            _fwd(q, k, v, g, beta_b, scale, out, A_log, dt_bias, lower_bound, initial_state=pool, final_state=pool)
        else:
            _fwd_bodies(x, out3, scale, A_log, dt_bias, lower_bound, pool, cu_seqlens)
        return

    d_starts, d_head, d_tail_start, d_tail, d_cuts, d_bp, d_bseq, d_hrow = _host_to_device(
        dev, starts_h, head, [s + h + b for s, h, b in zip(starts_h, head, body)], tail, cuts, body_pseudo,
        body_seq, [n * CHUNK for n in range(N)])

    if any(head):   # A. rows that complete an open chunk -> replay, into a side buffer
        head_out = torch.empty(N * CHUNK, H, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        _launch(x, scale, A_log, dt_bias, lower_bound, state, N, head_out, tok_start=d_starts, tok_count=d_head,
                out_row=d_hrow, slot_ids=slots)

    if body_pseudo:   # B. bodies -> one flash_kda.fwd call over the non-empty pseudo-sequences
        bslot = slots.to(torch.int64)[d_bseq] if slots is not None else d_bseq
        st = torch.empty(len(cuts) - 1, H, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        st.index_copy_(0, d_bp, state.S.index_select(0, bslot))
        _fwd_bodies(x, out3, scale, A_log, dt_bias, lower_bound, st, d_cuts)
        state.S.index_copy_(0, bslot, st.index_select(0, d_bp))

    if any(tail):   # C. trailing rows -> replay (they become the open chunk)
        _launch(x, scale, A_log, dt_bias, lower_bound, state, N, out3, tok_start=d_tail_start, tok_count=d_tail,
                slot_ids=slots)

    if any(head):   # D. head rows into place
        _C.replay_copy_rows(out3, head_out, d_starts, d_hrow, d_head.to(torch.int32), CHUNK)


def decode(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, state: ReplayState,
           slot_ids: Optional[torch.Tensor] = None):
    """Decode step: q, k, v, g, out bf16 ``[N, H, 128]`` (one token per sequence) or ``[N, T, H, 128]`` (T tokens per
    sequence, e.g. speculative tokens), beta ``[N, H]`` / ``[N, T, H]``. Same bits as ``prefill`` / one fwd call.
    ``slot_ids[n] == -1`` skips entry ``n`` (no state access, its ``out`` rows untouched)."""
    if q.dim() == 3:
        q, k, v, g, out = (t.unsqueeze(1) for t in (q, k, v, g, out))
        beta = beta.unsqueeze(1)
    x6 = _flat(q, k, v, g, beta, out)
    N, T = q.shape[0], q.shape[1]
    # one kernel launch: entry n owns tokens n*T .. n*T+T-1 and advances its own pos
    _launch(x6[:5], scale, A_log, dt_bias, lower_bound, state, N, x6[5], tokens_per_entry=T,
            slot_ids=_slots(slot_ids, N))


def final_state(state: ReplayState, A_log, dt_bias, lower_bound, *, dtype=torch.bfloat16,
                slot_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``flash_kda.fwd``'s ``final_state`` for the tokens appended so far: ``S`` with the open chunk's rows closed as a
    short last chunk (computed by flash_kda.fwd itself). ``state`` is not changed; serving continues from it (a closed
    short chunk is NOT a valid point to continue from, which is why the state keeps the open rows)."""
    idx = slot_ids.to(torch.int64) if slot_ids is not None else torch.arange(state.P, device=state.S.device)
    n, H = idx.numel(), state.H
    dev = state.S.device
    rows = lambda t: t[idx].permute(0, 2, 1, 3).reshape(1, n * CHUNK, H, HEAD_DIM).contiguous()  # noqa: E731
    k, v, g = rows(state.rows_k), rows(state.rows_v), rows(state.rows_g)
    beta = state.rows_beta[idx].permute(0, 2, 1).reshape(1, n * CHUNK, H).contiguous()
    q = torch.zeros_like(k)
    pos = state.pos[idx, 0].to(torch.int64)
    seg = torch.stack([pos, CHUNK - pos], dim=1).reshape(-1)           # real rows | throw-away rows
    cu = torch.zeros(2 * n + 1, dtype=torch.int64, device=dev)
    cu[1:] = torch.cumsum(seg, dim=0)
    init = torch.zeros(2 * n, H, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    init[0::2] = state.S[idx]   # (k, v, g, beta above are fresh copies, so their pointers are aligned for TMA)
    fin = torch.empty_like(init)
    from . import fwd as _fwd
    _fwd(q, k, v, g, beta, 1.0, torch.empty_like(q), A_log, dt_bias, lower_bound,
         initial_state=init, final_state=fin, cu_seqlens=cu)
    res = fin[0::2].contiguous()
    return res.to(dtype) if dtype != torch.bfloat16 else res
