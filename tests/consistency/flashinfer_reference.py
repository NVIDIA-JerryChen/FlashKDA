"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Unmodified arithmetic and one-warp kernel extracted from FlashInfer
c3af0c13328fd23b256c53b9b4f94d9e160d118c for independent golden testing.
Only unrelated imports/dispatchers were removed. Never import production helpers.
"""

import functools
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import math as mlir_math
import tvm_ffi  # noqa: F401
DOT_REDUCTION_DUAL_ACCUM = 1

@cute.jit
def compute_gate_value(
    g_val,
    k_idx,
    A_log_val,
    gDtBias,
    h_K_offset,
    lower_bound_val,
    USE_GATE_IN_KERNEL: cutlass.Constexpr[int],
    HAS_DT_BIAS: cutlass.Constexpr[int],
    USE_LOWER_BOUND: cutlass.Constexpr[int],
):
    """Compute one exp(gate) value in registers.

    When USE_GATE_IN_KERNEL=0: g is pre-computed log-space gate, just exp it.
    When USE_GATE_IN_KERNEL=1, USE_LOWER_BOUND=0:
        g_log = -exp(A_log) * softplus(g + dt_bias); return exp(g_log)
    When USE_GATE_IN_KERNEL=1, USE_LOWER_BOUND=1:
        g_log = lower_bound * sigmoid(exp(A_log) * (g + dt_bias)); return exp(g_log)

    Args:
        g_val: raw or log-space gate scalar
        k_idx: K-dimension index for the optional dt_bias load
        A_log_val: exp(A_log) for this head (Float32), precomputed outside
        gDtBias: dt_bias tensor [H*K] Float32 (or dummy if HAS_DT_BIAS=0)
        h_K_offset: query_head_idx * K offset into dt_bias
        lower_bound_val: lower bound float (negative, e.g. -5.0)
        USE_GATE_IN_KERNEL: 0 = pre-computed gate, 1 = in-kernel gate
        HAS_DT_BIAS: 0 = no dt_bias, 1 = add dt_bias
        USE_LOWER_BOUND: 0 = softplus formula, 1 = lower_bound * sigmoid formula

    The optional dt_bias load remains at the token-local gate computation site.
    """
    if USE_GATE_IN_KERNEL == 1:
        if HAS_DT_BIAS == 1:
            g_val = g_val + gDtBias[h_K_offset + k_idx].to(cutlass.Float32)
        if USE_LOWER_BOUND == 1:
            LOG2_E = cutlass.Float32(1.4426950408889634)
            neg_A_log2e = -A_log_val * LOG2_E
            lb_log2e = lower_bound_val * LOG2_E
            one = cutlass.Float32(1.0)
            fm = mlir_arith.FastMathFlags.fast
            one_ir = one.ir_value()
            neg_A_ir = neg_A_log2e.ir_value()
            lb_ir = lb_log2e.ir_value()
            ag = mlir_arith.mulf(neg_A_ir, g_val.ir_value(), fastmath=fm)
            exp_neg = mlir_math.exp2(ag, fastmath=fm)
            denom = mlir_arith.addf(one_ir, exp_neg, fastmath=fm)
            sig = mlir_arith.divf(one_ir, denom, fastmath=fm)
            ls = mlir_arith.mulf(lb_ir, sig, fastmath=fm)
            g_val = mlir_math.exp2(ls, fastmath=fm)
        else:
            exp_g = cute.exp(g_val, fastmath=True)
            one = cutlass.Float32(1.0)
            log2_v = cute.log2(one + exp_g, fastmath=True)
            g_val = cute.exp2(-A_log_val * log2_v, fastmath=True)
    else:
        g_val = cute.exp(g_val, fastmath=True)
    return g_val


# ==============================================================================
# REGISTER-TILE KERNEL
# ==============================================================================


@cute.jit
def _dot8_row(lhs, row, rhs, reduction_schedule: cutlass.Constexpr[int]):
    """Evaluate an eight-element dot product with the selected dependency graph."""
    if cutlass.const_expr(reduction_schedule == DOT_REDUCTION_DUAL_ACCUM):
        even = lhs[row, 0] * rhs[0]
        odd = lhs[row, 1] * rhs[1]
        for pair in cutlass.range_constexpr(1, 4):
            i = 2 * pair
            even = lhs[row, i] * rhs[i] + even
            odd = lhs[row, i + 1] * rhs[i + 1] + odd
        return even + odd
    return (
        (lhs[row, 0] * rhs[0] + lhs[row, 1] * rhs[1])
        + (lhs[row, 2] * rhs[2] + lhs[row, 3] * rhs[3])
    ) + (
        (lhs[row, 4] * rhs[4] + lhs[row, 5] * rhs[5])
        + (lhs[row, 6] * rhs[6] + lhs[row, 7] * rhs[7])
    )


@cute.jit
def _reduce_k_group(value, HEAD_DIM: cutlass.Constexpr[int]):
    """Reduce one partial dot product across the lanes that partition K."""
    if cutlass.const_expr(HEAD_DIM == 64):
        for offset in [4, 2, 1]:
            value = value + cute.arch.shuffle_sync_bfly(
                value, offset=offset, mask=0xFFFFFFFF
            )
    else:
        for offset in [8, 4, 2, 1]:
            value = value + cute.arch.shuffle_sync_bfly(
                value, offset=offset, mask=0xFFFFFFFF
            )
    return value


@cute.kernel
def recurrent_kda_decode_kernel(
    gQ: cute.Tensor,
    gK: cute.Tensor,
    gV: cute.Tensor,
    gG: cute.Tensor,
    gBeta: cute.Tensor,
    gH: cute.Tensor,
    gInitialStateSource: cute.Tensor,
    gInitialStateIndices: cute.Tensor,
    gO: cute.Tensor,
    gALog: cute.Tensor,
    gDtBias: cute.Tensor,
    gCuSeqlens: cute.Tensor,
    gSsmStateIndices: cute.Tensor,
    gNumAcceptedTokens: cute.Tensor,
    scale: cutlass.Float32,
    eps: cutlass.Float32,
    lower_bound: cutlass.Float32,
    HEAD_DIM: cutlass.Constexpr[int],
    USE_QK_L2NORM: cutlass.Constexpr[int],
    USE_GATE_IN_KERNEL: cutlass.Constexpr[int],
    HAS_DT_BIAS: cutlass.Constexpr[int],
    USE_LOWER_BOUND: cutlass.Constexpr[int],
    USE_CU_SEQLENS: cutlass.Constexpr[int],
    NUM_TOKENS: cutlass.Constexpr[int],
    TILE_ROWS: cutlass.Constexpr[int],
    DOT_REDUCTION_SCHEDULE: cutlass.Constexpr[int],
    HAS_INITIAL_STATE_SOURCE: cutlass.Constexpr[int],
    BETA_IS_LOGIT: cutlass.Constexpr[int],
    ZERO_PADDED_OUTPUT: cutlass.Constexpr[int],
    HAS_NUM_ACCEPTED_TOKENS: cutlass.Constexpr[int],
):
    """One warp owns a [TILE_ROWS V, HEAD_DIM K] tile in registers.

    Lanes partition K into contiguous eight-element vectors; the remaining lane
    groups partition V rows. State I/O uses 128-bit autovectorized copies.
    TILE_ROWS is 8, 16, or 32.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    HV = cutlass.Int32(gV.shape[2])
    H = cutlass.Int32(gQ.shape[2])
    NUM_V_TILES = HEAD_DIM // TILE_ROWS
    v_tile_idx = bidx % NUM_V_TILES
    bh = bidx // NUM_V_TILES
    value_head_idx = bh % HV
    batch_idx = bh // HV
    query_head_idx = value_head_idx // (HV // H)
    v_offset = v_tile_idx * TILE_ROWS

    K_LANES = HEAD_DIM // 8
    V_LANES = 32 // K_LANES
    VALUES_PER_THREAD = HEAD_DIM // 32
    k_lane = tidx % K_LANES
    v_lane = tidx // K_LANES

    token_base_offset = cutlass.Int32(0)
    seq_len = cutlass.Int32(1)
    if USE_CU_SEQLENS == 1:
        token_base_offset = gCuSeqlens[batch_idx].to(cutlass.Int32)
        seq_len = gCuSeqlens[batch_idx + 1].to(cutlass.Int32) - token_base_offset
    if ZERO_PADDED_OUTPUT == 1 and tidx < TILE_ROWS:
        padded_o_head = gO[(0, batch_idx, value_head_idx, None)]
        padded_o_head[v_offset + tidx] = cutlass.BFloat16(0.0)

    init_seq_idx = batch_idx
    if USE_CU_SEQLENS == 1:
        init_raw_slot = gSsmStateIndices[batch_idx * NUM_TOKENS].to(cutlass.Int32)
        if NUM_TOKENS > 1 and HAS_NUM_ACCEPTED_TOKENS == 1:
            nat_raw = gNumAcceptedTokens[batch_idx].to(cutlass.Int32)
            nat_offset = nat_raw - 1
            nat_offset = cutlass.Int32(0) if nat_offset < 0 else nat_offset
            nat_offset = (
                cutlass.Int32(NUM_TOKENS - 1)
                if nat_offset >= NUM_TOKENS
                else nat_offset
            )
            init_raw_slot = gSsmStateIndices[batch_idx * NUM_TOKENS + nat_offset].to(
                cutlass.Int32
            )
        init_seq_idx = cutlass.Int32(0) if init_raw_slot < 0 else init_raw_slot

    h_read = gH[(init_seq_idx, value_head_idx, None, None)]
    if HAS_INITIAL_STATE_SOURCE == 1:
        source_raw_slot = gInitialStateIndices[batch_idx].to(cutlass.Int32)
        source_seq_idx = cutlass.Int32(0) if source_raw_slot < 0 else source_raw_slot
        h_read = gInitialStateSource[(source_seq_idx, value_head_idx, None, None)]

    h_layout = cute.make_layout((TILE_ROWS // V_LANES, 8), stride=(8, 1))
    vec8_layout = cute.make_layout((8,), stride=(1,))
    src_layout = cute.make_layout((VALUES_PER_THREAD,), stride=(1,))
    h_reg = cute.make_rmem_tensor(h_layout, cutlass.Float32)
    h_bf16 = cute.make_rmem_tensor(vec8_layout, cutlass.BFloat16)
    q_src = cute.make_rmem_tensor(src_layout, cutlass.Float32)
    k_src = cute.make_rmem_tensor(src_layout, cutlass.Float32)
    gate_src = cute.make_rmem_tensor(src_layout, cutlass.Float32)
    q_bf16 = cute.make_rmem_tensor(src_layout, cutlass.BFloat16)
    k_bf16 = cute.make_rmem_tensor(src_layout, cutlass.BFloat16)
    gate_bf16 = cute.make_rmem_tensor(src_layout, cutlass.BFloat16)
    q_reg = cute.make_rmem_tensor(vec8_layout, cutlass.Float32)
    k_reg = cute.make_rmem_tensor(vec8_layout, cutlass.Float32)
    gate_reg = cute.make_rmem_tensor(vec8_layout, cutlass.Float32)

    for j in cutlass.range_constexpr(TILE_ROWS // V_LANES):
        v_idx = v_offset + v_lane + V_LANES * j
        h_tile = cute.local_tile(h_read, (1, 8), (v_idx, k_lane))
        cute.autovec_copy(h_tile, h_bf16)
        for i in cutlass.range_constexpr(8):
            h_reg[j, i] = h_bf16[i].to(cutlass.Float32)

    A_log_val = cutlass.Float32(0.0)
    h_K_offset = query_head_idx * HEAD_DIM
    if USE_GATE_IN_KERNEL == 1:
        A_log_val = cute.exp(gALog[query_head_idx].to(cutlass.Float32), fastmath=True)

    seq_idx = batch_idx
    is_active = seq_len > 0
    raw_slot = cutlass.Int32(0)
    token_offset = token_base_offset
    beta = cutlass.Float32(0.0)
    v_loaded = cutlass.Float32(0.0)
    q_head = gQ[(0, 0, query_head_idx, None)]
    k_head = gK[(0, 0, query_head_idx, None)]
    gate_head = gG[(0, 0, value_head_idx, None)]
    v_head = gV[(0, 0, value_head_idx, None)]
    o_head = gO[(0, 0, value_head_idx, None)]
    h_out = gH[(batch_idx, value_head_idx, None, None)]

    for token_t in cutlass.range(NUM_TOKENS):
        token_offset = token_base_offset + token_t
        if USE_CU_SEQLENS == 1:
            raw_slot = gSsmStateIndices[batch_idx * NUM_TOKENS + token_t].to(
                cutlass.Int32
            )
            has_token = token_t < seq_len
            is_active = raw_slot >= 0 and has_token
            token_offset = token_offset if has_token else cutlass.Int32(0)
            seq_idx = cutlass.Int32(0) if raw_slot < 0 else raw_slot
        else:
            seq_idx = batch_idx
            is_active = seq_len > 0
        if USE_CU_SEQLENS == 1:
            q_head = gQ[(0, token_offset, query_head_idx, None)]
            k_head = gK[(0, token_offset, query_head_idx, None)]
            gate_head = gG[(0, token_offset, value_head_idx, None)]
            v_head = gV[(0, token_offset, value_head_idx, None)]
            o_head = gO[(0, token_offset, value_head_idx, None)]
            beta = gBeta[(0, token_offset, value_head_idx)].to(cutlass.Float32)
        else:
            q_head = gQ[(batch_idx, 0, query_head_idx, None)]
            k_head = gK[(batch_idx, 0, query_head_idx, None)]
            gate_head = gG[(batch_idx, 0, value_head_idx, None)]
            v_head = gV[(batch_idx, 0, value_head_idx, None)]
            o_head = gO[(batch_idx, 0, value_head_idx, None)]
            beta = gBeta[(batch_idx, 0, value_head_idx)].to(cutlass.Float32)

        if BETA_IS_LOGIT == 1:
            beta = cutlass.Float32(1.0) / (
                cutlass.Float32(1.0) + cute.exp(-beta, fastmath=True)
            )

        q_tile = cute.local_tile(q_head, (VALUES_PER_THREAD,), (tidx,))
        k_tile = cute.local_tile(k_head, (VALUES_PER_THREAD,), (tidx,))
        gate_tile = cute.local_tile(gate_head, (VALUES_PER_THREAD,), (tidx,))
        cute.autovec_copy(q_tile, q_bf16)
        cute.autovec_copy(k_tile, k_bf16)
        cute.autovec_copy(gate_tile, gate_bf16)
        v_loaded = cutlass.Float32(0.0)
        # D128 row-16 hides V latency behind gate/norm work. Other measured
        # specializations are faster with the load left next to its consumer.
        if cutlass.const_expr(HEAD_DIM == 128 and TILE_ROWS == 16):
            if tidx < TILE_ROWS:
                v_loaded = v_head[v_offset + tidx].to(cutlass.Float32)

        for i in cutlass.range_constexpr(VALUES_PER_THREAD):
            k_idx = tidx * VALUES_PER_THREAD + i
            q_src[i] = q_bf16[i].to(cutlass.Float32)
            k_src[i] = k_bf16[i].to(cutlass.Float32)
            gate_src[i] = compute_gate_value(
                gate_bf16[i].to(cutlass.Float32),
                k_idx,
                A_log_val,
                gDtBias,
                h_K_offset,
                lower_bound,
                USE_GATE_IN_KERNEL,
                HAS_DT_BIAS,
                USE_LOWER_BOUND,
            )

        q_sum_sq = cutlass.Float32(0.0)
        k_sum_sq = cutlass.Float32(0.0)
        if HEAD_DIM == 64:
            q_sum_sq = q_src[0] * q_src[0] + q_src[1] * q_src[1]
            k_sum_sq = k_src[0] * k_src[0] + k_src[1] * k_src[1]
        elif cutlass.const_expr(DOT_REDUCTION_SCHEDULE == DOT_REDUCTION_DUAL_ACCUM):
            q_sum_even = q_src[0] * q_src[0]
            q_sum_odd = q_src[1] * q_src[1]
            q_sum_even = q_src[2] * q_src[2] + q_sum_even
            q_sum_odd = q_src[3] * q_src[3] + q_sum_odd
            q_sum_sq = q_sum_even + q_sum_odd
            k_sum_even = k_src[0] * k_src[0]
            k_sum_odd = k_src[1] * k_src[1]
            k_sum_even = k_src[2] * k_src[2] + k_sum_even
            k_sum_odd = k_src[3] * k_src[3] + k_sum_odd
            k_sum_sq = k_sum_even + k_sum_odd
        else:
            q_sum_sq = (q_src[0] * q_src[0] + q_src[1] * q_src[1]) + (
                q_src[2] * q_src[2] + q_src[3] * q_src[3]
            )
            k_sum_sq = (k_src[0] * k_src[0] + k_src[1] * k_src[1]) + (
                k_src[2] * k_src[2] + k_src[3] * k_src[3]
            )
        for offset in [16, 8, 4, 2, 1]:
            q_sum_sq = q_sum_sq + cute.arch.shuffle_sync_bfly(
                q_sum_sq, offset=offset, mask=0xFFFFFFFF
            )
            k_sum_sq = k_sum_sq + cute.arch.shuffle_sync_bfly(
                k_sum_sq, offset=offset, mask=0xFFFFFFFF
            )

        q_scale_factor = scale
        k_scale_factor = cutlass.Float32(1.0)
        if USE_QK_L2NORM == 1:
            q_scale_factor = cute.rsqrt(q_sum_sq + eps, fastmath=True) * scale
            k_scale_factor = cute.rsqrt(k_sum_sq + eps, fastmath=True)
        for i in cutlass.range_constexpr(8):
            source_lane = V_LANES * k_lane + i // VALUES_PER_THREAD
            source_value = i % VALUES_PER_THREAD
            q_reg[i] = (
                cute.arch.shuffle_sync(
                    q_src[source_value], offset=source_lane, mask=0xFFFFFFFF
                )
                * q_scale_factor
            )
            k_reg[i] = (
                cute.arch.shuffle_sync(
                    k_src[source_value], offset=source_lane, mask=0xFFFFFFFF
                )
                * k_scale_factor
            )
            gate_reg[i] = cute.arch.shuffle_sync(
                gate_src[source_value], offset=source_lane, mask=0xFFFFFFFF
            )

        if cutlass.const_expr(HEAD_DIM != 128 or TILE_ROWS != 16):
            if tidx < TILE_ROWS:
                v_loaded = v_head[v_offset + tidx].to(cutlass.Float32)
        for j in cutlass.range_constexpr(TILE_ROWS // V_LANES):
            for i in cutlass.range_constexpr(8):
                h_reg[j, i] = h_reg[j, i] * gate_reg[i]
            pred = _reduce_k_group(
                _dot8_row(h_reg, j, k_reg, DOT_REDUCTION_SCHEDULE), HEAD_DIM
            )

            v_idx = v_offset + v_lane + V_LANES * j
            v_val = cute.arch.shuffle_sync(
                v_loaded, offset=v_lane + V_LANES * j, mask=0xFFFFFFFF
            )
            delta = (v_val - pred) * beta
            for i in cutlass.range_constexpr(8):
                h_reg[j, i] = k_reg[i] * delta + h_reg[j, i]
            out = _reduce_k_group(
                _dot8_row(h_reg, j, q_reg, DOT_REDUCTION_SCHEDULE), HEAD_DIM
            )

            if is_active:
                if k_lane == j:
                    o_head[v_idx] = out.to(cutlass.BFloat16)

        if is_active:
            h_out = gH[(seq_idx, value_head_idx, None, None)]
            for j in cutlass.range_constexpr(TILE_ROWS // V_LANES):
                v_idx = v_offset + v_lane + V_LANES * j
                for i in cutlass.range_constexpr(8):
                    h_bf16[i] = h_reg[j, i].to(cutlass.BFloat16)
                h_tile = cute.local_tile(h_out, (1, 8), (v_idx, k_lane))
                cute.autovec_copy(h_bf16, h_tile)


@cute.jit
def recurrent_kda_launch(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mG: cute.Tensor,
    mBeta: cute.Tensor,
    mH: cute.Tensor,
    mInitialStateSource: cute.Tensor,
    mInitialStateIndices: cute.Tensor,
    mO: cute.Tensor,
    mALog: cute.Tensor,
    mDtBias: cute.Tensor,
    mCuSeqlens: cute.Tensor,
    mSsmStateIndices: cute.Tensor,
    mNumAcceptedTokens: cute.Tensor,
    scale: cutlass.Float32,
    eps: cutlass.Float32,
    lower_bound: cutlass.Float32,
    stream: cuda.CUstream,
    HEAD_DIM: cutlass.Constexpr[int],
    USE_QK_L2NORM: cutlass.Constexpr[int],
    USE_GATE_IN_KERNEL: cutlass.Constexpr[int],
    HAS_DT_BIAS: cutlass.Constexpr[int],
    USE_LOWER_BOUND: cutlass.Constexpr[int],
    USE_CU_SEQLENS: cutlass.Constexpr[int],
    NUM_TOKENS: cutlass.Constexpr[int],
    TILE_ROWS: cutlass.Constexpr[int],
    DOT_REDUCTION_SCHEDULE: cutlass.Constexpr[int],
    HAS_INITIAL_STATE_SOURCE: cutlass.Constexpr[int],
    BETA_IS_LOGIT: cutlass.Constexpr[int],
    ZERO_PADDED_OUTPUT: cutlass.Constexpr[int],
    HAS_NUM_ACCEPTED_TOKENS: cutlass.Constexpr[int],
):
    batch_size = mQ.shape[0]
    if USE_CU_SEQLENS == 1:
        batch_size = mCuSeqlens.shape[0] - 1
    HV = mV.shape[2]
    recurrent_kda_decode_kernel(
        mQ,
        mK,
        mV,
        mG,
        mBeta,
        mH,
        mInitialStateSource,
        mInitialStateIndices,
        mO,
        mALog,
        mDtBias,
        mCuSeqlens,
        mSsmStateIndices,
        mNumAcceptedTokens,
        scale,
        eps,
        lower_bound,
        HEAD_DIM,
        USE_QK_L2NORM,
        USE_GATE_IN_KERNEL,
        HAS_DT_BIAS,
        USE_LOWER_BOUND,
        USE_CU_SEQLENS,
        NUM_TOKENS,
        TILE_ROWS,
        DOT_REDUCTION_SCHEDULE,
        HAS_INITIAL_STATE_SOURCE,
        BETA_IS_LOGIT,
        ZERO_PADDED_OUTPUT,
        HAS_NUM_ACCEPTED_TOKENS,
    ).launch(
        grid=[batch_size * HV * (HEAD_DIM // TILE_ROWS), 1, 1],
        block=[32, 1, 1],
        stream=stream,
    )



@functools.cache
def compiled_reference(device_index, capability):
    # Always the upstream one-warp T=1 path; no production helper or dispatcher.
    b, h = cute.sym_int(), cute.sym_int()
    def fake(shape, dtype=cute.BFloat16):
        return cute.runtime.make_fake_compact_tensor(
            dtype, shape, assumed_align=32,
            stride_order=tuple(reversed(range(len(shape)))))
    x = fake((b, 1, h, 128))
    state = fake((b, h, 128, 128))
    idx = fake((cute.sym_int(),), cute.Int32)
    return cute.compile(
        recurrent_kda_launch, x, x, x, x, fake((b, 1, h)), state, state,
        idx, x, fake((h,), cute.Float32), fake((h * 128,), cute.Float32),
        idx, idx, idx, cutlass.Float32(0.0), cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        128, 1, 1, 1, 1, 0, 1, 16, DOT_REDUCTION_DUAL_ACCUM, 0, 1, 0, 0,
        options="--enable-tvm-ffi --generate-line-info")


def step(q, k, v, g, beta, state, a_log, dt_bias, out, *,
         scale=128 ** -0.5, eps=1e-6, lower_bound=-5.0):
    import torch
    dummy = torch.empty(1, device=q.device, dtype=torch.int32)
    compiled_reference(q.device.index, torch.cuda.get_device_capability(q.device))(
        q, k, v, g, beta, state, state, dummy, out, a_log, dt_bias.view(-1),
        dummy, dummy, dummy, scale, eps, lower_bound)
