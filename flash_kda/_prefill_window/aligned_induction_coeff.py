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
from .._consistent_decode_kernel import compute_gate_value
from . import packed_math as prims


from .schedule import SEGMENT


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
    cu_seqlens: cute.Tensor,
    prefix: cute.Tensor,
    tasks: cute.Tensor,
    start: cutlass.Int32,
    count: cutlass.Int32,
    wait_first: cutlass.Int32,
    NATURAL: cutlass.Constexpr[bool],
):
    # Launched with PDL. The first window waits for its predecessor (the
    # schedule kernel); later windows follow the previous window's scan,
    # which triggers after its own wait, and write the other workspace
    # region, so they overlap that scan. The next scan (PDL secondary of this
    # kernel) waits for this kernel's completion before reading anything it
    # or the previous scan wrote.
    thread, _, _ = cute.arch.thread_idx()
    block, head_block, _ = cute.arch.block_idx()
    # Block (0, 0) always waits for the primary (the previous window's scan,
    # at its end below), so this kernel cannot complete before it; the next scan, whose
    # griddepcontrol.wait covers this kernel, therefore also reads after the previous
    # scan finished writing its continuation state. The other blocks wait
    # only when their region may still be read.
    # No explicit trigger: this window's scan (PDL secondary) launches as
    # the last block exits, so its CTAs are placed on SMs the previous scan
    # has left (an early trigger crowded unequal-length layouts).
    if wait_first != 0:
        cute.arch.griddepcontrol_wait()
    tidx = thread % 32
    HV = cutlass.Int32(gQ.shape[2])
    # Grid (window tokens, head groups of four): one warp per token-head, no
    # runtime divisions. row is the token-head's workspace row.
    query_head_idx = head_block * 4 + thread // 32
    task_idx = block // SEGMENT + start
    token_in_segment = block % SEGMENT
    row = block * HV + query_head_idx
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
    if query_head_idx < HV and task_idx < prefix[cu_seqlens.shape[0] - 1]:
        if token_in_segment < tasks[task_idx, 2]:
            batch_idx = cutlass.Int32(0)
            token = (
                cu_seqlens[tasks[task_idx, 0]] + tasks[task_idx, 1] + token_in_segment
            )
            token_offset = (
                (cutlass.Int64(batch_idx) * length + token) * HV + query_head_idx
            ) * 128
            h_K_offset = query_head_idx * 128
            A_log_val = cute.exp(
                gALog[query_head_idx].to(cutlass.Float32), fastmath=True
            )
            q_head = cute.make_tensor(
                _known_aligned(gQ.iterator + token_offset, 32), head_layout
            )
            k_head = cute.make_tensor(
                _known_aligned(gK.iterator + token_offset, 32), head_layout
            )
            gate_head = cute.make_tensor(
                _known_aligned(gG.iterator + token_offset, 32), head_layout
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
            # This lane's four dt_bias values in one 16-byte load, added with
            # the same FP32 add the gate helper applies (then no load inside).
            dt_bias = cute.make_rmem_tensor(src_layout, cutlass.Float32)
            cute.autovec_copy(
                cute.make_tensor(
                    _known_aligned(
                        gDtBias.iterator + h_K_offset + tidx * VALUES_PER_THREAD, 16
                    ),
                    src_layout,
                ),
                dt_bias,
            )
            for i in cutlass.range_constexpr(VALUES_PER_THREAD):
                k_idx = tidx * VALUES_PER_THREAD + i
                q_src[i] = q_bf16[i].to(cutlass.Float32)
                k_src[i] = k_bf16[i].to(cutlass.Float32)
                gate_src[i] = compute_gate_value(
                    gate_bf16[i].to(cutlass.Float32) + dt_bias[i],
                    k_idx,
                    A_log_val,
                    gDtBias,
                    h_K_offset,
                    lower_bound,
                    1,
                    0,
                    1,
                )
            # Same FP32 operations as the scalar form, in packed f32x2 pairs:
            # even/odd squares (product, then fused term), butterfly sums of
            # q and k side by side, and the normalizing products.
            q_sum_even, q_sum_odd = prims.mul_packed_f32x2(
                (q_src[0], q_src[1]), (q_src[0], q_src[1])
            )
            q_sum_even, q_sum_odd = prims.fma_packed_f32x2(
                (q_src[2], q_src[3]), (q_src[2], q_src[3]), (q_sum_even, q_sum_odd)
            )
            k_sum_even, k_sum_odd = prims.mul_packed_f32x2(
                (k_src[0], k_src[1]), (k_src[0], k_src[1])
            )
            k_sum_even, k_sum_odd = prims.fma_packed_f32x2(
                (k_src[2], k_src[3]), (k_src[2], k_src[3]), (k_sum_even, k_sum_odd)
            )
            q_sum_sq, k_sum_sq = prims.add_packed_f32x2(
                (q_sum_even, k_sum_even), (q_sum_odd, k_sum_odd)
            )
            for offset in [16, 8, 4, 2, 1]:
                q_sum_sq, k_sum_sq = prims.add_packed_f32x2(
                    (q_sum_sq, k_sum_sq),
                    (
                        cute.arch.shuffle_sync_bfly(
                            q_sum_sq, offset=offset, mask=4294967295
                        ),
                        cute.arch.shuffle_sync_bfly(
                            k_sum_sq, offset=offset, mask=4294967295
                        ),
                    ),
                )
            q_shifted, k_shifted = prims.add_packed_f32x2(
                (q_sum_sq, k_sum_sq), (eps, eps)
            )
            q_scale_factor = cute.rsqrt(q_shifted, fastmath=True) * scale
            k_scale_factor = cute.rsqrt(k_shifted, fastmath=True)
            base = cutlass.Int64(row) * 392
            q_norm = cute.make_rmem_tensor(src_layout, cutlass.Float32)
            k_norm = cute.make_rmem_tensor(src_layout, cutlass.Float32)
            for i in cutlass.range_constexpr(2):
                q_norm[2 * i], q_norm[2 * i + 1] = prims.mul_packed_f32x2(
                    (q_src[2 * i], q_src[2 * i + 1]), (q_scale_factor, q_scale_factor)
                )
                k_norm[2 * i], k_norm[2 * i + 1] = prims.mul_packed_f32x2(
                    (k_src[2 * i], k_src[2 * i + 1]), (k_scale_factor, k_scale_factor)
                )
            # Parity-split layout: position p * 64 + j * 4 + m holds K = 8 j + 2 m + p,
            # so each consumer lane reads the even or odd members of one K group
            # as one float4. This lane's K = 4 tidx + i go to two float2 spans.
            if cutlass.const_expr(NATURAL):
                # Swizzled natural layout: the float4 holding K = 8 v + 4 h +
                # (0..3) sits in slot 2 v + (h ^ bit 2 of v), i.e. this lane's
                # K = 4 tidx + i go to slot tidx ^ bit 3 of tidx in one 16-byte
                # store. Sixteen K groups read side by side (one float4 each)
                # hit every bank pair once per eight groups.
                slot4 = (tidx ^ ((tidx >> 3) & 1)) * 4
                for component, values in ((0, q_norm), (1, k_norm), (2, gate_src)):
                    cute.autovec_copy(
                        values,
                        cute.make_tensor(
                            _known_aligned(
                                coeff.iterator + base + component * 128 + slot4, 16
                            ),
                            src_layout,
                        ),
                    )
            else:
                lane_base = (tidx // 2) * 4 + (tidx % 2) * 2
                pair_layout = cute.make_layout((2,), stride=(1,))
                for component, values in ((0, q_norm), (1, k_norm), (2, gate_src)):
                    for parity in cutlass.range_constexpr(2):
                        pair = cute.make_rmem_tensor(pair_layout, cutlass.Float32)
                        pair[0] = values[parity]
                        pair[1] = values[parity + 2]
                        dst = cute.make_tensor(
                            _known_aligned(
                                coeff.iterator
                                + base
                                + component * 128
                                + parity * 64
                                + lane_base,
                                8,
                            ),
                            pair_layout,
                        )
                        cute.autovec_copy(pair, dst)
            if tidx == 0:
                # beta plus three zero pad floats: the scan copies this 16-byte
                # chunk whole (TMA bulk copy), so the padding must be defined.
                tail = cute.make_rmem_tensor(src_layout, cutlass.Float32)
                tail.fill(0.0)
                tail[0] = beta
                cute.autovec_copy(
                    tail,
                    cute.make_tensor(
                        _known_aligned(coeff.iterator + base + 384, 16), src_layout
                    ),
                )
    if wait_first == 0 and block == 0 and head_block == 0:
        # Block (0, 0) writes its rows (the region the previous scan does not
        # read) first and only then waits for that scan, so the kernel still
        # cannot complete before it, without a post-scan work tail.
        cute.arch.griddepcontrol_wait()


@cute.jit
def _known_aligned(pointer, alignment: cutlass.Constexpr[int]):
    return cute.make_ptr(
        pointer.dtype, pointer.toint(), pointer.memspace, assumed_align=alignment
    )


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
    cu_seqlens: cute.Tensor,
    prefix: cute.Tensor,
    tasks: cute.Tensor,
    start: cutlass.Int32,
    count: cutlass.Int32,
    wait_first: cutlass.Int32,
    stream: cuda.CUstream,
    PDL: cutlass.Constexpr[bool],
    NATURAL: cutlass.Constexpr[bool],
):
    _coeff_kernel(
        q,
        k,
        g,
        beta,
        a,
        bias,
        coeff,
        scale,
        eps,
        lower_bound,
        cu_seqlens,
        prefix,
        tasks,
        start,
        count,
        wait_first,
        NATURAL,
    ).launch(
        grid=[count * SEGMENT, cute.ceil_div(q.shape[2], 4), 1],
        block=[128, 1, 1],
        stream=stream,
        use_pdl=PDL,
    )


def _fake(shape, dtype=cute.BFloat16):
    return cute.runtime.make_fake_compact_tensor(
        dtype, shape, assumed_align=32, stride_order=tuple(reversed(range(len(shape))))
    )


@functools.cache
def compiled_coeff_kernel(device_index, capability):
    pdl, natural = True, True
    b, length, h, n, m, r = (cute.sym_int() for _ in range(6))
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
        _fake((m,), cute.Int32),
        _fake((m,), cute.Int32),
        _fake((r, 3), cute.Int32),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        pdl,
        natural,
        options="--enable-tvm-ffi",
    )
