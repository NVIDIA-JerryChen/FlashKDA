"""Four-row recursive-halving consumer.

Each token gates the state, forms the prediction and delta, updates the state,
and projects the output in FP32. The state is then rounded to BF16.
Lane permutations preserve the fixed reduction pairings.
"""

import cutlass
import cutlass.cute as cute
from . import packed_math as prims
from .state_bits import _expand_state_bits, _round_state, _known_aligned, _pack_state
from .consumer import (
    wait_test,
    finish_wait,
    _gmem_ptr,
    _load_natural8,
    PIPE_STAGES,
    VALUE_FLOAT_OFFSET,
    FULL_MASK,
    STEADY_TOKENS,
)

ROWS = 4


@cute.jit
def _halving_levels84(partials):
    """Levels 8 and 4 of the tree for row x (sum over K lanes k, k^4, k^8, k^12)."""
    # Levels 8 and 4 in one radix-4 round: (slot 0 + slot 2 of l^8) +
    # (slot 1 of l^4 + slot 3 of l^12) pairs the same partials as the two
    # halving rounds (all four hold row x), with one shuffle round less.
    received8 = cute.arch.shuffle_sync_bfly(partials[2], offset=16, mask=FULL_MASK)
    received4 = cute.arch.shuffle_sync_bfly(partials[1], offset=8, mask=FULL_MASK)
    received12 = cute.arch.shuffle_sync_bfly(partials[3], offset=24, mask=FULL_MASK)
    # Both inner sums of the radix-4 round in one packed add (same pairings).
    low, high = prims.add_packed_f32x2(
        (partials[0], received4), (received8, received12)
    )
    return low + high


@cute.jit
def _halving_tree(partials):
    """partials[s]: dual-accumulator sum of slot s (row s ^ x) for this K lane.
    Returns the 8/4/2/1-tree sum of row x."""
    total = _halving_levels84(partials)
    near = cute.arch.shuffle_sync_bfly(total, offset=4, mask=FULL_MASK)
    far = cute.arch.shuffle_sync_bfly(total, offset=2, mask=FULL_MASK)
    both = cute.arch.shuffle_sync_bfly(total, offset=6, mask=FULL_MASK)
    low, high = prims.add_packed_f32x2((total, far), (near, both))
    return low + high


@cute.jit
def _halving_token(
    t,
    parity: cutlass.Constexpr[int],
    staged,
    staged_values,
    full_bars,
    empty_bars,
    k_lane,
    value_index,
    h,
    gates,
    keys,
    q_reg,
    output_address,
    memspace: cutlass.Constexpr,
    held,
    head_stride,
    PAIRED: cutlass.Constexpr[bool],
    slot,
    next_slot,
    next_phase,
):
    """Advance this lane's four slots by local token t (t % 2 == parity).
    On entry gates/keys[parity] hold token t.

    PAIRED: tokens come in (even, odd) pairs. The even token keeps its output
    after tree levels 8 and 4 in `held`; the odd token finishes levels 2 and
    1 for both at once (lanes with K-lane bit 1 clear keep the even token,
    the others the odd one), one shuffle per level instead of three per
    token. Every sum pairs the same K lanes in the same order as
    _halving_tree; lane k_lane ends with row x of token t - 1 + bit1."""
    other = 1 - parity
    zero = cutlass.Float32(0.0)
    if cutlass.const_expr(parity == 1):
        # Probe the next pair's full barrier now; the spin (rarely taken)
        # comes after the state update, so the probe latency is hidden.
        ready = cute.arch.mbarrier_test_wait(full_bars + (next_slot >> 1), next_phase)
    beta = staged[slot, 384]
    v_own = staged_values[slot, 2 * VALUE_FLOAT_OFFSET + value_index]
    # Gate the BF16 state (slots paired with a broadcast coefficient).
    for i in [2, 3, 0, 1, 4, 5, 6, 7]:
        for jp in cutlass.range_constexpr(ROWS // 2):
            h[2 * jp, i], h[2 * jp + 1, i] = prims.mul_packed_f32x2(
                (h[2 * jp, i], h[2 * jp + 1, i]), (gates[parity, i], gates[parity, i])
            )
    # Prediction: even chain term 2 rounds first, then 0, 4, 6 fused; odd chain 3, 1, 5, 7.
    accum = cute.make_rmem_tensor(
        cute.make_layout((ROWS, 2), stride=(2, 1)), cutlass.Float32
    )
    for c in cutlass.range_constexpr(2):
        for jp in cutlass.range_constexpr(ROWS // 2):
            accum[2 * jp, c], accum[2 * jp + 1, c] = prims.mul_packed_f32x2(
                (h[2 * jp, 2 + c], h[2 * jp + 1, 2 + c]),
                (keys[parity, 2 + c], keys[parity, 2 + c]),
            )
    for i in [0, 4, 6]:
        for c in cutlass.range_constexpr(2):
            for jp in cutlass.range_constexpr(ROWS // 2):
                accum[2 * jp, c], accum[2 * jp + 1, c] = prims.fma_packed_f32x2(
                    (h[2 * jp, i + c], h[2 * jp + 1, i + c]),
                    (keys[parity, i + c], keys[parity, i + c]),
                    (accum[2 * jp, c], accum[2 * jp + 1, c]),
                )
    partials = cute.make_rmem_tensor(
        cute.make_layout((ROWS,), stride=(1,)), cutlass.Float32
    )
    for jp in cutlass.range_constexpr(ROWS // 2):
        partials[2 * jp], partials[2 * jp + 1] = prims.add_packed_f32x2(
            (accum[2 * jp, 0], accum[2 * jp + 1, 0]),
            (accum[2 * jp, 1], accum[2 * jp + 1, 1]),
        )
    prediction = _halving_tree(partials)
    delta_own = (v_own.to(cutlass.Float32) - prediction) * beta
    deltas = cute.make_rmem_tensor(
        cute.make_layout((ROWS,), stride=(1,)), cutlass.Float32
    )
    deltas[0] = delta_own
    for s in cutlass.range_constexpr(1, ROWS):
        deltas[s] = cute.arch.shuffle_sync_bfly(
            delta_own, offset=s << 3, mask=FULL_MASK
        )
    _load_natural8(staged, slot, 0, k_lane, q_reg, 0)
    for i in cutlass.range_constexpr(8):
        for jp in cutlass.range_constexpr(ROWS // 2):
            h[2 * jp, i], h[2 * jp + 1, i] = prims.fma_packed_f32x2(
                (keys[parity, i], keys[parity, i]),
                (deltas[2 * jp], deltas[2 * jp + 1]),
                (h[2 * jp, i], h[2 * jp + 1, i]),
            )
    # Look ahead: the next pair's first slot needs its full barrier.
    if cutlass.const_expr(parity == 1):
        finish_wait(full_bars + (next_slot >> 1), next_phase, ready)
    _load_natural8(staged, next_slot, 2, k_lane, gates, other)
    _load_natural8(staged, next_slot, 1, k_lane, keys, other)
    # Output dot on the updated FP32 state: even chain 0 then 2, 4, 6; odd 1, 3, 5, 7.
    for c in cutlass.range_constexpr(2):
        for jp in cutlass.range_constexpr(ROWS // 2):
            accum[2 * jp, c], accum[2 * jp + 1, c] = prims.mul_packed_f32x2(
                (h[2 * jp, c], h[2 * jp + 1, c]), (q_reg[0, c], q_reg[0, c])
            )
    for p in cutlass.range_constexpr(1, 4):
        for c in cutlass.range_constexpr(2):
            for jp in cutlass.range_constexpr(ROWS // 2):
                accum[2 * jp, c], accum[2 * jp + 1, c] = prims.fma_packed_f32x2(
                    (h[2 * jp, 2 * p + c], h[2 * jp + 1, 2 * p + c]),
                    (q_reg[0, 2 * p + c], q_reg[0, 2 * p + c]),
                    (accum[2 * jp, c], accum[2 * jp + 1, c]),
                )
    for jp in cutlass.range_constexpr(ROWS // 2):
        partials[2 * jp], partials[2 * jp + 1] = prims.add_packed_f32x2(
            (accum[2 * jp, 0], accum[2 * jp + 1, 0]),
            (accum[2 * jp, 1], accum[2 * jp + 1, 1]),
        )
    # Release the pair only after the output dot consumed the q values of
    # its last token, so no shared read can still be in flight when the
    # producer's TMA refills the slot.
    if cutlass.const_expr(parity == 1):
        cute.arch.mbarrier_arrive(empty_bars + (slot >> 1))
    if cutlass.const_expr(not PAIRED):
        output = _halving_tree(partials)
        if (k_lane & 3) == 0:
            # output_address is the lane's paired-store address (one token
            # back on lanes with K-lane bit 1 clear, see run_halving_consumer).
            target = cute.make_tensor(
                _gmem_ptr(output_address + head_stride, cutlass.BFloat16, memspace, 2),
                cute.make_layout((1,)),
            )
            target[0] = output.to(cutlass.BFloat16)
    elif cutlass.const_expr(parity == 0):
        held[0] = _halving_levels84(partials)
    else:
        current = _halving_levels84(partials)
        odd = (k_lane & 2) != 0
        keep = current if odd else held[0]
        send = held[0] if odd else current
        level2 = keep + cute.arch.shuffle_sync_bfly(send, offset=4, mask=FULL_MASK)
        output = level2 + cute.arch.shuffle_sync_bfly(level2, offset=2, mask=FULL_MASK)
        if (k_lane & 1) == 0:
            target = cute.make_tensor(
                _gmem_ptr(output_address, cutlass.BFloat16, memspace, 2),
                cute.make_layout((1,)),
            )
            target[0] = output.to(cutlass.BFloat16)
    # Every token ends with the state rounded to BF16 (kept in FP32 encoding).
    for s in cutlass.range_constexpr(ROWS):
        for i in cutlass.range_constexpr(8):
            h[s, i] = _round_state(h[s, i], zero)


@cute.jit
def run_halving_consumer(
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
    memspace: cutlass.Constexpr,
):
    k_lane = lane >> 1
    v_lane = lane & 1
    x = ((k_lane >> 3) & 1) * 2 + ((k_lane >> 2) & 1)
    block_row0 = cta_v_offset + warp * (2 * ROWS) + v_lane * ROWS
    value_index = warp * (2 * ROWS) + v_lane * ROWS + x
    state_offset = cutlass.Int64(bh) * 128 * 128
    state_layout = cute.make_layout((128, 128), stride=(128, 1))
    h_read = cute.make_tensor(
        _known_aligned(gInitialStateSource.iterator + state_offset, 32), state_layout
    )
    if start > 0:
        h_read = cute.make_tensor(
            _known_aligned(gH.iterator + state_offset, 32), state_layout
        )
    h = cute.make_rmem_tensor(
        cute.make_layout((ROWS, 8), stride=(8, 1)), cutlass.Float32
    )
    h_bf16 = cute.make_rmem_tensor(
        cute.make_layout((8,), stride=(1,)), cutlass.BFloat16
    )
    for s in cutlass.range_constexpr(ROWS):
        h_tile = cute.local_tile(h_read, (1, 8), (block_row0 + (s ^ x), k_lane))
        cute.autovec_copy(h_tile, h_bf16)
        bits = cute.recast_tensor(h_bf16, cutlass.Uint32)
        for p in cutlass.range_constexpr(4):
            h[s, 2 * p], h[s, 2 * p + 1] = _expand_state_bits(bits[p])
    pair8 = cute.make_layout((2, 8), stride=(8, 1))
    gates = cute.make_rmem_tensor(pair8, cutlass.Float32)
    keys = cute.make_rmem_tensor(pair8, cutlass.Float32)
    q_reg = cute.make_rmem_tensor(
        cute.make_layout((1, 8), stride=(8, 1)), cutlass.Float32
    )
    held = cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), cutlass.Float32)
    held[0] = cutlass.Float32(0.0)
    # Paired output trees store token t - 1 on lanes with K-lane bit 1 clear
    # and token t on the others: fold that one-token offset into the lane's
    # address once, so the loop carries no per-token address select.
    output_address = (
        output_row0
        + start * head_stride
        + cutlass.Int64((block_row0 + x) * 2)
        - cutlass.Int64(((k_lane >> 1) & 1) ^ 1) * head_stride
    )
    wait_test(full_bars + 0, 0)
    _load_natural8(staged, 0, 2, k_lane, gates, 0)
    _load_natural8(staged, 0, 1, k_lane, keys, 0)
    groups = count // STEADY_TOKENS
    for group_index in cutlass.range(groups):
        # Eight tokens per group fill one part of the ring: slots base + i,
        # so every shared offset in the body is static.
        base = (group_index % (PIPE_STAGES // STEADY_TOKENS)) * STEADY_TOKENS
        phase = (group_index // (PIPE_STAGES // STEADY_TOKENS)) & 1
        for i in cutlass.range_constexpr(STEADY_TOKENS):
            t = STEADY_TOKENS * group_index + i
            if cutlass.const_expr(i + 1 < STEADY_TOKENS):
                next_slot, next_phase = base + i + 1, phase
            else:
                next_slot = (base + STEADY_TOKENS) & (PIPE_STAGES - 1)
                next_phase = ((group_index + 1) // (PIPE_STAGES // STEADY_TOKENS)) & 1
            _halving_token(
                t,
                i % 2,
                staged,
                staged_values,
                full_bars,
                empty_bars,
                k_lane,
                value_index,
                h,
                gates,
                keys,
                q_reg,
                output_address,
                memspace,
                held,
                head_stride,
                True,
                base + i,
                next_slot,
                next_phase,
            )
            output_address = output_address + head_stride
    tail_start = STEADY_TOKENS * groups
    tail_count = count - tail_start
    for pair in cutlass.range(tail_count // 2):
        for i in cutlass.range_constexpr(2):
            t = tail_start + 2 * pair + i
            _halving_token(
                t,
                i,
                staged,
                staged_values,
                full_bars,
                empty_bars,
                k_lane,
                value_index,
                h,
                gates,
                keys,
                q_reg,
                output_address,
                memspace,
                held,
                head_stride,
                True,
                t & (PIPE_STAGES - 1),
                (t + 1) & (PIPE_STAGES - 1),
                ((t + 1) // PIPE_STAGES) & 1,
            )
            output_address = output_address + head_stride
    if (tail_count & 1) == 1:
        t = count - 1
        _halving_token(
            t,
            0,
            staged,
            staged_values,
            full_bars,
            empty_bars,
            k_lane,
            value_index,
            h,
            gates,
            keys,
            q_reg,
            output_address,
            memspace,
            held,
            head_stride,
            False,
            t & (PIPE_STAGES - 1),
            (t + 1) & (PIPE_STAGES - 1),
            ((t + 1) // PIPE_STAGES) & 1,
        )
    h_out = cute.make_tensor(
        _known_aligned(gH.iterator + state_offset, 32), state_layout
    )
    words = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Uint32)
    for s in cutlass.range_constexpr(ROWS):
        for p in cutlass.range_constexpr(4):
            words[p] = _pack_state(h[s, 2 * p], h[s, 2 * p + 1])
        tile = cute.make_tensor(
            _known_aligned(
                h_out.iterator + (block_row0 + (s ^ x)) * 128 + k_lane * 8, 16
            ),
            cute.make_layout((8,), stride=(1,)),
        )
        cute.autovec_copy(cute.recast_tensor(words, cutlass.BFloat16), tile)
