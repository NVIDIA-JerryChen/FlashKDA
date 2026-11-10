"""Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Adapted from FlashInfer c3af0c13328fd23b256c53b9b4f94d9e160d118c.

Register-resident varlen recurrence with shared FP32 coefficients.

The producer stages coefficients and values. Two/four-row consumer engines
preserve the same FP32 operations and token-local BF16 state boundary.
"""

import functools
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.utils import SmemAllocator
from . import packed_math as prims
from .state_bits import (
    _known_aligned,
)

from .schedule import SEGMENT, segment_range
from .halving_consumer import run_halving_consumer
from .halving2_consumer import run_halving2_consumer
from .consumer import (
    PIPE_STAGES,
    PAIRS,
    CONSUMER_WARPS,
    STAGE_FLOATS,
    VALUE_FLOAT_OFFSET,
)

COEFF_BYTES = 388 * 4  # q, k, gates, beta (+3 pad) of one coefficient row


@cute.jit
def _produce(
    staged,
    full_bars,
    empty_bars,
    lane,
    count,
    coeff_row0,
    coeff_stride,
    value_address0,
    head_stride,
    ranks,
    rank_offset,
    first_segment,
    first_slot,
    head,
    HV,
    memspace: cutlass.Constexpr,
    value_rows,
    ring_shift,
):
    """Producer warp (one elected lane): stage tokens 0..count, two per pair,
    with TMA bulk copies whose bytes complete the pair's full barrier.

    Token `count` is a placeholder (no copies) so the consumers' look-ahead
    after their last token never waits on data that does not exist.
    """
    value_bytes = cutlass.Int32(value_rows * 2)
    value_address = value_address0
    if lane >= 0:
        # The whole warp runs the loop (converged); one elected lane issues.
        # Outer loop over the window's 128-token segments: the segment's
        # workspace slot (one rank lookup) is resolved once per segment, so
        # the per-pair body carries no predicated lookup code. SEGMENT is
        # even, so pairs never straddle segments.
        for segment in cutlass.range((count + SEGMENT - 1) // SEGMENT):
            slot = ranks[rank_offset + first_segment + segment] - first_slot
            coeff_address = coeff_row0 + (cutlass.Int64(slot) * SEGMENT * HV + head) * (
                392 * 4
            )
            segment_tokens = cute.min(count - segment * SEGMENT, SEGMENT)
            for local_pair in cutlass.range((segment_tokens + 1) // 2):
                k = segment * (SEGMENT // 2) + local_pair
                # The ring holds 1 << ring_shift pairs for this CTA's engine.
                pair = k & ((1 << ring_shift) - 1)
                cute.arch.mbarrier_wait(empty_bars + pair, ((k >> ring_shift) & 1) ^ 1)
                valid = cute.min(count - 2 * k, 2)
                prims.elected_arrive_expect_tx(
                    full_bars + pair, valid * (COEFF_BYTES + value_bytes)
                )
                for i in cutlass.range_constexpr(2):
                    if 2 * k + i < count:
                        destination = staged.iterator + (2 * pair + i) * STAGE_FLOATS
                        prims.elected_bulk_copy_g2s(
                            destination,
                            coeff_address,
                            cutlass.Int32(COEFF_BYTES),
                            full_bars + pair,
                        )
                        prims.elected_bulk_copy_g2s(
                            destination + VALUE_FLOAT_OFFSET,
                            value_address,
                            value_bytes,
                            full_bars + pair,
                        )
                    coeff_address = coeff_address + coeff_stride
                    value_address = value_address + head_stride
        if (count & 1) == 0:
            # Token `count` opens a pair of its own: a placeholder (no bytes).
            k = count // 2
            pair = k & ((1 << ring_shift) - 1)
            cute.arch.mbarrier_wait(empty_bars + pair, ((k >> ring_shift) & 1) ^ 1)
            prims.elected_arrive_expect_tx(full_bars + pair, cutlass.Int32(0))


@cute.kernel
def _scan_kernel(
    gV: cute.Tensor,
    coeff: cute.Tensor,
    gH: cute.Tensor,
    gInitialStateSource: cute.Tensor,
    gO: cute.Tensor,
    cu_seqlens: cute.Tensor,
    prefix: cute.Tensor,
    ranks: cute.Tensor,
    order: cute.Tensor,
    window_start: cutlass.Int32,
    window_count: cutlass.Int32,
    runtime_rows: cutlass.Int32,
    long_pairs: cutlass.Int32,
    radix4: cutlass.Int32,
    checkpoints: cute.Tensor,
    checkpoint_indices: cute.Tensor,
    CHECKPOINTS: cutlass.Constexpr[bool],
    WRITE_FINAL: cutlass.Constexpr[bool],
    RADIX4: cutlass.Constexpr[bool],
    TRIGGER: cutlass.Constexpr[bool],
    LEAN: cutlass.Constexpr[bool],
):
    """Four consumer warps own 16 or 32 V rows; warp 4 stages coefficients.

    A uniform runtime branch chooses the row engine inside this one kernel.
    Both engines use the same arithmetic and BF16 continuation-state bits.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp = tidx // 32
    lane = tidx % 32
    HV = cutlass.Int32(gV.shape[2])
    # The first long_pairs request-head pairs (requests longest-first from
    # the schedule kernel's order, heads in order within a request) use the
    # two-row engine (eight CTAs per pair); the rest use runtime_rows.
    # Placement only: both engines run the identical arithmetic.
    long_blocks = long_pairs * 8
    is_long = bidx < long_blocks
    rows = runtime_rows
    if is_long:
        rows = cutlass.Int32(2)
    local_bidx = bidx if is_long else bidx - long_blocks
    pair_base = cutlass.Int32(0) if is_long else long_pairs
    row_shift = 3 if rows == 2 else 2
    row_block = local_bidx & ((1 << row_shift) - 1)
    slot_bh = local_bidx >> row_shift
    skip = cutlass.Int32(0)
    if (radix4 & 16) != 0:
        # Spare layout (long_pairs == HV sizes the grid): the first 4 HV blocks
        # are spares for the longest request, then the usual four-row blocks of
        # every request in longest-first order. When the longest request is
        # strictly longer than the runner-up (device metadata), its four-row
        # blocks and the spares become eight two-row blocks per head (rows 0-63
        # and 64-127, all in the first placement round); otherwise the spares
        # exit at once. Placement and engine only: same arithmetic.
        dominant = cutlass.Int32(0)
        # Longest-first records: lengths of the leading two requests.
        if order[2] > order[6]:
            dominant = cutlass.Int32(1)
        rows = runtime_rows
        pair_base = cutlass.Int32(0)
        spares = HV * 4
        row_block = (bidx - spares) & 3
        slot_bh = (bidx - spares) >> 2
        if bidx < spares:
            slot_bh = bidx >> 2
            row_block = 4 + (bidx & 3)
            if dominant == 0:
                skip = cutlass.Int32(1)
        if slot_bh < HV and dominant == 1:
            rows = cutlass.Int32(2)
    value_rows = CONSUMER_WARPS * 2 * rows
    pair_index = pair_base + slot_bh
    value_head_idx = pair_index % HV
    # One 16-byte record (index, first token, length, first rank) per request
    # in longest-first order, written by the schedule kernel.
    record = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Int32)
    cute.autovec_copy(
        cute.make_tensor(
            _known_aligned(order.iterator + (pair_index // HV) * 4, 16),
            cute.make_layout((4,), stride=(1,)),
        ),
        record,
    )
    batch_idx = record[0]
    bh = batch_idx * HV + value_head_idx
    cta_v_offset = row_block * value_rows
    seq_start = record[1]
    length = record[2]
    rank_offset = record[3]
    segments = (length + SEGMENT - 1) // SEGMENT
    # Ranks are a permutation of [0, total segments). When this window covers
    # all of them (the usual single window) the searches below are known to
    # return [0, segments); skip their dependent global loads.
    first_segment = cutlass.Int32(0)
    end_segment = segments
    if window_start != 0 or window_count < prefix[cu_seqlens.shape[0] - 1]:
        first_segment, end_segment = segment_range(
            ranks, rank_offset, segments, window_start, window_start + window_count
        )
    start = first_segment * SEGMENT
    count = cute.max(cute.min(length, end_segment * SEGMENT) - start, 0)
    # PDL secondary of this window's coefficient kernel: everything above
    # reads only schedule outputs (complete before that kernel triggered).
    # Wait here for the coefficients and (through that kernel's ordered
    # completion) the previous window's continuation state.
    cute.arch.griddepcontrol_wait()
    # The next window's coefficient kernel (PDL, other workspace region)
    # may start once every CTA of this scan has passed its wait, i.e.
    # after the previous scan, which read that region, has completed.
    cute.arch.griddepcontrol_launch_dependents()
    if skip == 0 and length == 0 and window_start == 0:
        for i in cutlass.range(tidx, value_rows * 128, (CONSUMER_WARPS + 1) * 32):
            offset = cutlass.Int64(bh) * 16384 + cta_v_offset * 128 + i
            target = cute.make_tensor(gH.iterator + offset, cute.make_layout((1,)))
            source = cute.make_tensor(
                gInitialStateSource.iterator + offset, cute.make_layout((1,))
            )
            target[0] = source[0]
    if skip == 0 and count > 0:
        memspace = gV.iterator.memspace
        value_base = (cutlass.Int64(seq_start) * HV + value_head_idx) * 128
        coeff_base = cutlass.Int64(0)
        coeff_row0 = cutlass.Int64((coeff.iterator + coeff_base).toint())
        value_row0 = cutlass.Int64((gV.iterator + value_base).toint())
        output_row0 = cutlass.Int64((gO.iterator + value_base).toint())
        coeff_stride = cutlass.Int64(HV) * (392 * 4)
        head_stride = cutlass.Int64(HV) * (128 * 2)

        allocator = SmemAllocator()
        staged = allocator.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((PIPE_STAGES, STAGE_FLOATS), stride=(STAGE_FLOATS, 1)),
            byte_alignment=128,
        )
        staged_values = cute.make_tensor(
            cute.recast_ptr(staged.iterator, dtype=cutlass.BFloat16),
            cute.make_layout(
                (PIPE_STAGES, 2 * STAGE_FLOATS), stride=(2 * STAGE_FLOATS, 1)
            ),
        )
        full_bars = allocator.allocate_array(
            cutlass.Int64, num_elems=PAIRS, byte_alignment=8
        )
        empty_bars = allocator.allocate_array(
            cutlass.Int64, num_elems=PAIRS, byte_alignment=8
        )
        # Every engine reads one slot past its last token (the look-ahead
        # after token count - 1): slot count of the 32-slot ring, or of the
        # sixteen-slot ring of two-row halving CTAs. Zero both so that read is
        # defined; every other slot is staged by TMA before it is read.
        zeros = cute.make_rmem_tensor(
            cute.make_layout((4,), stride=(1,)), cutlass.Float32
        )
        zeros.fill(0.0)
        for chunk in cutlass.range_constexpr(
            (2 * STAGE_FLOATS + 4 * (CONSUMER_WARPS + 1) * 32 - 1)
            // (4 * (CONSUMER_WARPS + 1) * 32)
        ):
            index = chunk * (CONSUMER_WARPS + 1) * 32 + tidx
            # (The two slots coincide when bit 4 of count is clear: once.)
            limit = 2 * STAGE_FLOATS if (count & 16) != 0 else STAGE_FLOATS
            if index * 4 < limit:
                zero_slot = count & (PIPE_STAGES - 1)
                if index * 4 >= STAGE_FLOATS:
                    zero_slot = count & 15
                cute.autovec_copy(
                    zeros,
                    cute.make_tensor(
                        _known_aligned(
                            staged.iterator
                            + zero_slot * STAGE_FLOATS
                            + (index * 4) % STAGE_FLOATS,
                            16,
                        ),
                        cute.make_layout((4,), stride=(1,)),
                    ),
                )
        # The zero-filled slots are later overwritten by the async (TMA) proxy.
        cute.arch.fence_view_async_shared()
        if tidx == 0:
            for s in cutlass.range_constexpr(PAIRS):
                cute.arch.mbarrier_init(full_bars + s, 1)
                cute.arch.mbarrier_init(empty_bars + s, CONSUMER_WARPS * 32)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        if warp == CONSUMER_WARPS:
            # Two-row halving CTAs use a sixteen-slot ring (static slot offsets
            # in their sixteen-token loop); every other engine the whole ring.
            ring_shift = cutlass.Int32(PAIRS.bit_length() - 1)
            if rows == 2 and (radix4 & 64) != 0:
                ring_shift = cutlass.Int32(3)
            _produce(
                staged,
                full_bars,
                empty_bars,
                lane,
                count,
                coeff_row0,
                coeff_stride,
                value_row0 + start * head_stride + cutlass.Int64(cta_v_offset * 2),
                head_stride,
                ranks,
                rank_offset,
                first_segment,
                window_start,
                value_head_idx,
                HV,
                memspace,
                value_rows,
                ring_shift,
            )
        else:
            if rows == 2:
                # Dispatch the two runtime consumer configurations (every
                # two-row CTA has bit 64, every four-row CTA the halving
                # engine), so the two hot loops sit next to each other in a
                # kernel less than half the size. Same engines, same bits.
                run_halving2_consumer(
                    lane,
                    warp,
                    cta_v_offset,
                    bh,
                    start,
                    count,
                    gInitialStateSource,
                    gH,
                    staged,
                    staged_values,
                    full_bars,
                    empty_bars,
                    output_row0,
                    head_stride,
                    memspace,
                )
            else:
                run_halving_consumer(
                    lane,
                    warp,
                    cta_v_offset,
                    bh,
                    start,
                    count,
                    gInitialStateSource,
                    gH,
                    staged,
                    staged_values,
                    full_bars,
                    empty_bars,
                    output_row0,
                    head_stride,
                    memspace,
                )


@cute.jit
def _launch_scan(
    v: cute.Tensor,
    coeff: cute.Tensor,
    state_out: cute.Tensor,
    state_in: cute.Tensor,
    out: cute.Tensor,
    cu_seqlens: cute.Tensor,
    prefix: cute.Tensor,
    ranks: cute.Tensor,
    order: cute.Tensor,
    start: cutlass.Int32,
    count: cutlass.Int32,
    runtime_rows: cutlass.Int32,
    long_pairs: cutlass.Int32,
    radix4: cutlass.Int32,
    RADIX4: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
    TRIGGER: cutlass.Constexpr[bool],
    LEAN: cutlass.Constexpr[bool],
):
    # Three resident CTAs per SM (128 registers per thread): the latency-bound
    # engines use the registers to overlap consecutive tokens; the 4096 registers left
    # still holds one coefficient block of the next window (PDL overlap).
    _scan_kernel(
        v,
        coeff,
        state_out,
        state_in,
        out,
        cu_seqlens,
        prefix,
        ranks,
        order,
        start,
        count,
        runtime_rows,
        long_pairs,
        radix4,
        state_out,
        coeff,
        False,
        True,
        RADIX4,
        TRIGGER,
        LEAN,
    ).launch(
        grid=[
            long_pairs * 8
            + (state_out.shape[0] * v.shape[2] - long_pairs)
            * (128 // (CONSUMER_WARPS * 2 * runtime_rows)),
            1,
            1,
        ],
        block=[32 * (CONSUMER_WARPS + 1), 1, 1],
        stream=stream,
        min_blocks_per_mp=3,
        use_pdl=True,
    )


def _fake(shape, dtype=cute.BFloat16):
    return cute.runtime.make_fake_compact_tensor(
        dtype, shape, assumed_align=32, stride_order=tuple(reversed(range(len(shape))))
    )


@functools.cache
def compiled_scan_kernel(device_index, capability):
    radix4, trigger, lean = False, True, True
    b, length, h, n, m, r, o = (cute.sym_int() for _ in range(7))
    x = _fake((1, length, h, 128))
    state = _fake((b, h, 128, 128))
    return cute.compile(
        _launch_scan,
        x,
        _fake((n,), cute.Float32),
        state,
        state,
        x,
        _fake((m,), cute.Int32),
        _fake((m,), cute.Int32),
        _fake((r,), cute.Int32),
        _fake((o,), cute.Int32),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(4),
        cutlass.Int32(0),
        cutlass.Int32(0),
        radix4,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        trigger,
        lean,
        options="--enable-tvm-ffi",
    )
