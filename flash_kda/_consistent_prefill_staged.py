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

Adapted from FlashInfer c3af0c13328fd23b256c53b9b4f94d9e160d118c.
Changes: fuse dense multi-token recurrence with register-resident state,
optional parallel coefficient preparation, and bounded workspace windows.
Arithmetic, normalization, gate, reduction order and rounding are unchanged.
"""

import functools
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from ._consistent_decode_kernel import compute_gate_value, _reduce_k_group


from ._consistent_prefill_math import _prediction_dot8, _output_dot8


@cute.kernel
def _coeff_kernel(
    gQ: cute.Tensor,
    gK: cute.Tensor,
    gG: cute.Tensor,
    gBeta: cute.Tensor,
    gALog: cute.Tensor,
    gDtBias: cute.Tensor,
    coeff: cute.Tensor,
    scale: cutlass.Float32,
    eps: cutlass.Float32,
    lower_bound: cutlass.Float32,
    start: cutlass.Int32,
    count: cutlass.Int32,
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    tidx = thread % 32
    row = block * 4 + thread // 32
    HV = cutlass.Int32(gQ.shape[2])
    length = cutlass.Int32(gQ.shape[1])
    VALUES_PER_THREAD = 4
    head_layout = cute.make_layout((128,), stride=(1,))
    src_layout = cute.make_layout((VALUES_PER_THREAD,), stride=(1,))
    q_src = cute.make_rmem_tensor(src_layout, cutlass.Float32)
    k_src = cute.make_rmem_tensor(src_layout, cutlass.Float32)
    gate_src = cute.make_rmem_tensor(src_layout, cutlass.Float32)
    q_bf16 = cute.make_rmem_tensor(src_layout, cutlass.BFloat16)
    k_bf16 = cute.make_rmem_tensor(src_layout, cutlass.BFloat16)
    gate_bf16 = cute.make_rmem_tensor(src_layout, cutlass.BFloat16)
    if row < gQ.shape[0] * count * HV:
        batch_idx = row // (count * HV)
        token = row // HV % count + start
        query_head_idx = row % HV
        token_offset = (
            (cutlass.Int64(batch_idx) * length + token) * HV + query_head_idx
        ) * 128
        h_K_offset = query_head_idx * 128
        A_log_val = cute.exp(gALog[query_head_idx].to(cutlass.Float32), fastmath=True)
        q_head = cute.make_tensor((gQ.iterator + token_offset).align(32), head_layout)
        k_head = cute.make_tensor((gK.iterator + token_offset).align(32), head_layout)
        gate_head = cute.make_tensor(
            (gG.iterator + token_offset).align(32), head_layout
        )
        beta_head = cute.make_tensor(
            gBeta.iterator + token_offset // 128, cute.make_layout((1,))
        )
        beta = beta_head[0].to(cutlass.Float32)
        beta = cutlass.Float32(1.0) / (
            cutlass.Float32(1.0) + cute.exp(-beta, fastmath=True)
        )
        q_tile = cute.local_tile(q_head, (VALUES_PER_THREAD,), (tidx,))
        k_tile = cute.local_tile(k_head, (VALUES_PER_THREAD,), (tidx,))
        gate_tile = cute.local_tile(gate_head, (VALUES_PER_THREAD,), (tidx,))
        cute.autovec_copy(q_tile, q_bf16)
        cute.autovec_copy(k_tile, k_bf16)
        cute.autovec_copy(gate_tile, gate_bf16)
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
                1,
                1,
                1,
            )
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
        for offset in [16, 8, 4, 2, 1]:
            q_sum_sq = q_sum_sq + cute.arch.shuffle_sync_bfly(
                q_sum_sq, offset=offset, mask=4294967295
            )
            k_sum_sq = k_sum_sq + cute.arch.shuffle_sync_bfly(
                k_sum_sq, offset=offset, mask=4294967295
            )
        q_scale_factor = cute.rsqrt(q_sum_sq + eps, fastmath=True) * scale
        k_scale_factor = cute.rsqrt(k_sum_sq + eps, fastmath=True)
        base = cutlass.Int64(row) * 392
        for i in cutlass.range_constexpr(4):
            index = tidx * 4 + i
            coeff[base + index] = q_src[i] * q_scale_factor
            coeff[base + 128 + index] = k_src[i] * k_scale_factor
            coeff[base + 256 + index] = gate_src[i]
        if tidx == 0:
            coeff[base + 384] = beta


@cute.kernel
def _scan_kernel(
    gV: cute.Tensor,
    coeff: cute.Tensor,
    gH: cute.Tensor,
    gInitialStateSource: cute.Tensor,
    gO: cute.Tensor,
    start: cutlass.Int32,
    count: cutlass.Int32,
    V_TILE: cutlass.Constexpr[int],
    checkpoints: cute.Tensor,
    checkpoint_indices: cute.Tensor,
    CHECKPOINTS: cutlass.Constexpr[bool],
    WRITE_FINAL: cutlass.Constexpr[bool],
):
    """One warp owns a [V_TILE V, 128 K] tile in registers.

    Lanes partition K into eight-element vectors and partition V in two groups.
    State I/O uses 128-bit autovectorized copies. All requests are active.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    HV = cutlass.Int32(gV.shape[2])
    NUM_V_TILES = 128 // V_TILE
    v_tile_idx = bidx % NUM_V_TILES
    bh = bidx // NUM_V_TILES
    value_head_idx = bh % HV
    v_offset = v_tile_idx * V_TILE
    K_LANES = 128 // 8
    V_LANES = 32 // K_LANES
    k_lane = tidx % K_LANES
    v_lane = tidx // K_LANES
    # Dense contiguous buffers share one linear batch-head index. Use 64-bit
    # offsets so a large state allocation cannot overflow an Int32 product.
    batch_idx = bh // HV
    length = cutlass.Int32(gV.shape[1])
    state_offset = cutlass.Int64(bh) * 128 * 128
    state_layout = cute.make_layout((128, 128), stride=(128, 1))
    h_read = cute.make_tensor(
        (gInitialStateSource.iterator + state_offset).align(32), state_layout
    )
    if cutlass.const_expr(CHECKPOINTS):
        if start > 0:
            slot = checkpoint_indices[batch_idx, start - 1]
            checkpoint_offset = (cutlass.Int64(slot) * HV + value_head_idx) * 128 * 128
            h_read = cute.make_tensor(
                (checkpoints.iterator + checkpoint_offset).align(32), state_layout
            )
    h_layout = cute.make_layout((V_TILE // V_LANES, 8), stride=(8, 1))
    vec8_layout = cute.make_layout((8,), stride=(1,))
    h_reg = cute.make_rmem_tensor(h_layout, cutlass.Float32)
    h_bf16 = cute.make_rmem_tensor(vec8_layout, cutlass.BFloat16)
    q_reg = cute.make_rmem_tensor(vec8_layout, cutlass.Float32)
    k_reg = cute.make_rmem_tensor(vec8_layout, cutlass.Float32)
    gate_reg = cute.make_rmem_tensor(vec8_layout, cutlass.Float32)
    for j in cutlass.range_constexpr(V_TILE // V_LANES):
        v_idx = v_offset + v_lane + V_LANES * j
        h_tile = cute.local_tile(h_read, (1, 8), (v_idx, k_lane))
        cute.autovec_copy(h_tile, h_bf16)
        for i in cutlass.range_constexpr(8):
            h_reg[j, i] = h_bf16[i].to(cutlass.Float32)
    value_base = (cutlass.Int64(batch_idx) * length * HV + value_head_idx) * 128
    value_layout = cute.make_layout((length, 128), stride=(cutlass.Int64(HV) * 128, 1))
    values = cute.make_tensor((gV.iterator + value_base).align(32), value_layout)
    outputs = cute.make_tensor((gO.iterator + value_base).align(32), value_layout)
    coeff_base = (cutlass.Int64(batch_idx) * count * HV + value_head_idx) * 392
    coeffs = cute.make_tensor(
        (coeff.iterator + coeff_base).align(32),
        cute.make_layout((count, 392), stride=(cutlass.Int64(HV) * 392, 1)),
    )
    for local_token in cutlass.range(count):
        token = start + local_token
        c = coeffs[local_token, None]
        q_tile = cute.local_tile(c, (8,), (k_lane,))
        k_tile = cute.local_tile(c, (8,), (16 + k_lane,))
        gate_tile = cute.local_tile(c, (8,), (32 + k_lane,))
        cute.autovec_copy(q_tile, q_reg)
        cute.autovec_copy(k_tile, k_reg)
        cute.autovec_copy(gate_tile, gate_reg)
        beta = c[384]
        v_head = values[token, None]
        o_head = outputs[token, None]
        v_loaded = cutlass.Float32(0.0)
        if tidx < V_TILE:
            v_loaded = v_head[v_offset + tidx].to(cutlass.Float32)
        out_for_lane = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(V_TILE // V_LANES):
            for i in cutlass.range_constexpr(8):
                h_reg[j, i] = h_reg[j, i] * gate_reg[i]
            pred = _reduce_k_group(_prediction_dot8(h_reg, j, k_reg), 128)
            v_val = cute.arch.shuffle_sync(
                v_loaded, offset=v_lane + V_LANES * j, mask=4294967295
            )
            delta = (v_val - pred) * beta
            for i in cutlass.range_constexpr(8):
                h_reg[j, i] = k_reg[i] * delta + h_reg[j, i]
            out = _reduce_k_group(_output_dot8(h_reg, j, q_reg), 128)
            out_for_lane = out if k_lane == j else out_for_lane
        # Each active lane stores its selected row once, preserving FP32 output
        # calculation and the final BF16 conversion without eight divergent stores.
        if k_lane < V_TILE // V_LANES:
            o_head[v_offset + v_lane + V_LANES * k_lane] = out_for_lane.to(
                cutlass.BFloat16
            )
        for j in cutlass.range_constexpr(V_TILE // V_LANES):
            for i in cutlass.range_constexpr(8):
                h_reg[j, i] = h_reg[j, i].to(cutlass.BFloat16).to(cutlass.Float32)
        if cutlass.const_expr(CHECKPOINTS):
            slot = checkpoint_indices[batch_idx, token]
            checkpoint_offset = (cutlass.Int64(slot) * HV + value_head_idx) * 128 * 128
            checkpoint = cute.make_tensor(
                (checkpoints.iterator + checkpoint_offset).align(32), state_layout
            )
            for j in cutlass.range_constexpr(V_TILE // V_LANES):
                v_idx = v_offset + v_lane + V_LANES * j
                for i in cutlass.range_constexpr(8):
                    h_bf16[i] = h_reg[j, i].to(cutlass.BFloat16)
                h_tile = cute.local_tile(checkpoint, (1, 8), (v_idx, k_lane))
                cute.autovec_copy(h_bf16, h_tile)
    if cutlass.const_expr(WRITE_FINAL):
        if not CHECKPOINTS or start + count == length:
            h_out = cute.make_tensor(
                (gH.iterator + state_offset).align(32), state_layout
            )
            for j in cutlass.range_constexpr(V_TILE // V_LANES):
                v_idx = v_offset + v_lane + V_LANES * j
                for i in cutlass.range_constexpr(8):
                    h_bf16[i] = h_reg[j, i].to(cutlass.BFloat16)
                h_tile = cute.local_tile(h_out, (1, 8), (v_idx, k_lane))
                cute.autovec_copy(h_bf16, h_tile)


@cute.jit
def _launch_coeff(
    q: cute.Tensor,
    k: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    a: cute.Tensor,
    bias: cute.Tensor,
    coeff: cute.Tensor,
    scale: cutlass.Float32,
    eps: cutlass.Float32,
    lower_bound: cutlass.Float32,
    start: cutlass.Int32,
    count: cutlass.Int32,
    stream: cuda.CUstream,
):
    _coeff_kernel(
        q, k, g, beta, a, bias, coeff, scale, eps, lower_bound, start, count
    ).launch(
        grid=[cute.ceil_div(q.shape[0] * count * q.shape[2], 4), 1, 1],
        block=[128, 1, 1],
        stream=stream,
    )


@cute.jit
def _launch_scan(
    v: cute.Tensor,
    coeff: cute.Tensor,
    state_out: cute.Tensor,
    state_in: cute.Tensor,
    out: cute.Tensor,
    start: cutlass.Int32,
    count: cutlass.Int32,
    V_TILE: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    _scan_kernel(
        v,
        coeff,
        state_out,
        state_in,
        out,
        start,
        count,
        V_TILE,
        state_out,
        coeff,
        False,
        True,
    ).launch(
        grid=[v.shape[0] * v.shape[2] * (128 // V_TILE), 1, 1],
        block=[32, 1, 1],
        stream=stream,
    )


def _fake(shape, dtype=cute.BFloat16):
    return cute.runtime.make_fake_compact_tensor(
        dtype, shape, assumed_align=32, stride_order=tuple(reversed(range(len(shape))))
    )


@functools.cache
def compiled_coeff_kernel(device_index, capability):
    b, length, h, n = (cute.sym_int() for _ in range(4))
    x = _fake((b, length, h, 128))
    return cute.compile(
        _launch_coeff,
        x,
        x,
        x,
        _fake((b, length, h)),
        _fake((h,), cute.Float32),
        _fake((h * 128,), cute.Float32),
        _fake((n,), cute.Float32),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --generate-line-info",
    )


@functools.cache
def compiled_scan_kernel(device_index, capability, v_tile):
    b, length, h, n = (cute.sym_int() for _ in range(4))
    x = _fake((b, length, h, 128))
    state = _fake((b, h, 128, 128))
    return cute.compile(
        _launch_scan,
        x,
        _fake((n,), cute.Float32),
        state,
        state,
        x,
        cutlass.Int32(0),
        cutlass.Int32(0),
        v_tile,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --generate-line-info",
    )


@cute.jit
def _launch_scan_checkpoints(
    v: cute.Tensor,
    coeff: cute.Tensor,
    state_out: cute.Tensor,
    state_in: cute.Tensor,
    out: cute.Tensor,
    checkpoints: cute.Tensor,
    checkpoint_indices: cute.Tensor,
    start: cutlass.Int32,
    count: cutlass.Int32,
    V_TILE: cutlass.Constexpr[int],
    WRITE_FINAL: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    _scan_kernel(
        v,
        coeff,
        state_out,
        state_in,
        out,
        start,
        count,
        V_TILE,
        checkpoints,
        checkpoint_indices,
        True,
        WRITE_FINAL,
    ).launch(
        grid=[v.shape[0] * v.shape[2] * (128 // V_TILE), 1, 1],
        block=[32, 1, 1],
        stream=stream,
    )


@functools.cache
def compiled_scan_checkpoints(device_index, capability, v_tile, write_final):
    b, length, h, n = (cute.sym_int() for _ in range(4))
    x = _fake((b, length, h, 128))
    state = _fake((b, h, 128, 128))
    return cute.compile(
        _launch_scan_checkpoints,
        x,
        _fake((n,), cute.Float32),
        state,
        state,
        x,
        _fake((cute.sym_int(), h, 128, 128)),
        _fake((b, length), cute.Int32),
        cutlass.Int32(0),
        cutlass.Int32(0),
        v_tile,
        write_final,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --generate-line-info",
    )
