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
def _prefill_kernel(
    gQ: cute.Tensor,
    gK: cute.Tensor,
    gV: cute.Tensor,
    gG: cute.Tensor,
    gBeta: cute.Tensor,
    gH: cute.Tensor,
    gInitialStateSource: cute.Tensor,
    gO: cute.Tensor,
    gALog: cute.Tensor,
    gDtBias: cute.Tensor,
    scale: cutlass.Float32,
    eps: cutlass.Float32,
    lower_bound: cutlass.Float32,
    V_TILE: cutlass.Constexpr[int],
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
    query_head_idx = value_head_idx
    v_offset = v_tile_idx * V_TILE
    K_LANES = 128 // 8
    V_LANES = 32 // K_LANES
    VALUES_PER_THREAD = 128 // 32
    k_lane = tidx % K_LANES
    v_lane = tidx // K_LANES
    # Dense contiguous buffers share one linear batch-head index. Use 64-bit
    # offsets so a large state allocation cannot overflow an Int32 product.
    batch_idx = bh // HV
    length = cutlass.Int32(gQ.shape[1])
    state_offset = cutlass.Int64(bh) * 128 * 128
    state_layout = cute.make_layout((128, 128), stride=(128, 1))
    head_layout = cute.make_layout((128,), stride=(1,))
    h_read = cute.make_tensor(
        (gInitialStateSource.iterator + state_offset).align(32), state_layout
    )
    h_layout = cute.make_layout((V_TILE // V_LANES, 8), stride=(8, 1))
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
    for j in cutlass.range_constexpr(V_TILE // V_LANES):
        v_idx = v_offset + v_lane + V_LANES * j
        h_tile = cute.local_tile(h_read, (1, 8), (v_idx, k_lane))
        cute.autovec_copy(h_tile, h_bf16)
        for i in cutlass.range_constexpr(8):
            h_reg[j, i] = h_bf16[i].to(cutlass.Float32)
    h_K_offset = query_head_idx * 128
    A_log_val = cute.exp(gALog[query_head_idx].to(cutlass.Float32), fastmath=True)
    for token in cutlass.range(length):
        token_offset = (
            (cutlass.Int64(batch_idx) * length + token) * HV + value_head_idx
        ) * 128
        q_head = cute.make_tensor((gQ.iterator + token_offset).align(32), head_layout)
        k_head = cute.make_tensor((gK.iterator + token_offset).align(32), head_layout)
        gate_head = cute.make_tensor(
            (gG.iterator + token_offset).align(32), head_layout
        )
        v_head = cute.make_tensor((gV.iterator + token_offset).align(32), head_layout)
        o_head = cute.make_tensor((gO.iterator + token_offset).align(32), head_layout)
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
        v_loaded = cutlass.Float32(0.0)
        if tidx < V_TILE:
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
        for i in cutlass.range_constexpr(8):
            source_lane = V_LANES * k_lane + i // VALUES_PER_THREAD
            source_value = i % VALUES_PER_THREAD
            q_reg[i] = (
                cute.arch.shuffle_sync(
                    q_src[source_value], offset=source_lane, mask=4294967295
                )
                * q_scale_factor
            )
            k_reg[i] = (
                cute.arch.shuffle_sync(
                    k_src[source_value], offset=source_lane, mask=4294967295
                )
                * k_scale_factor
            )
            gate_reg[i] = cute.arch.shuffle_sync(
                gate_src[source_value], offset=source_lane, mask=4294967295
            )
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
    h_out = cute.make_tensor((gH.iterator + state_offset).align(32), state_layout)
    for j in cutlass.range_constexpr(V_TILE // V_LANES):
        v_idx = v_offset + v_lane + V_LANES * j
        for i in cutlass.range_constexpr(8):
            h_bf16[i] = h_reg[j, i].to(cutlass.BFloat16)
        h_tile = cute.local_tile(h_out, (1, 8), (v_idx, k_lane))
        cute.autovec_copy(h_bf16, h_tile)


@cute.jit
def _launch(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    state_out: cute.Tensor,
    state_in: cute.Tensor,
    out: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    scale: cutlass.Float32,
    eps: cutlass.Float32,
    lower_bound: cutlass.Float32,
    V_TILE: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    _prefill_kernel(
        q,
        k,
        v,
        g,
        beta,
        state_out,
        state_in,
        out,
        a_log,
        dt_bias,
        scale,
        eps,
        lower_bound,
        V_TILE,
    ).launch(
        grid=[q.shape[0] * q.shape[2] * (128 // V_TILE), 1, 1],
        block=[32, 1, 1],
        stream=stream,
    )


@functools.cache
def compiled_prefill_kernel(device_index, capability, v_tile=4):
    import torch

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare consistent_prefill_fwd before CUDA Graph capture")
    b, length, h = (cute.sym_int(), cute.sym_int(), cute.sym_int())

    def fake(shape, dtype=cute.BFloat16):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            assumed_align=32,
            stride_order=tuple(reversed(range(len(shape)))),
        )

    x = fake((b, length, h, 128))
    state = fake((b, h, 128, 128))
    return cute.compile(
        _launch,
        x,
        x,
        x,
        x,
        fake((b, length, h)),
        state,
        state,
        x,
        fake((h,), cute.Float32),
        fake((h * 128,), cute.Float32),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        v_tile,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --generate-line-info",
    )
