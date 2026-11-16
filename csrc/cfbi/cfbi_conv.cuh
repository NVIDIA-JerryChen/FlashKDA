// CF-BI conv: short causal depthwise convolution (+ SiLU) with a pinned floating-point program.
//
// For one channel d and sequence position t (history positions p < 0 come from the conv state, zero for a fresh
// sequence / the training forward):
//     acc = +0.0f
//     for k = 0 .. W-1 (ascending):  acc = fmaf(w[d][k], float(x[t - (W-1) + k][d]), acc)      (one rounding per tap)
//     z   = acc (+ bias[d] with one fadd_rn, if there is a bias)
//     y   = z * sigmoid_act(z)         sigmoid_act(z) = rcp.approx.ftz(1 + ex2.approx.ftz(-z * log2e))   (SiLU)
//     out = bf16_rn(y)
// The new state is the last W-1 *inputs* of (old state ++ x), copied bit for bit.
//
// Every output is the same function of the same W input values however the sequence is cut into calls, batched,
// tiled or vectorised: there is ONE device function (conv_acc / conv_finish) and no other arithmetic on the path.
// See docs/20261103-cfbi-conv-linear.md for the contract.
//
// Work decomposition (no effect on bits): a block is (channel tile, tile of up to CONV_TILE_T tokens of ONE sequence,
// sequence). A tile never crosses a sequence boundary. A thread owns VEC adjacent channels and walks the tokens of
// its tile with a register window of the last W-1 inputs. Only the first tile (j == 0) of a sequence reads the old
// state and writes the new state, in the same thread and in that order, so the in-place update is race free (all
// other tiles start at an offset >= CONV_TILE_T >= W-1 and only read x).
#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cfbi/cfbi_prep.cuh"
#include "cfbi/cfbi_primitives.cuh"

namespace cfbi {
namespace conv {

// Launch geometry (performance only; no effect on bits). The macros let benchmarks/micro/conv_variants.py build variants.
#ifndef CFBI_CONV_THREADS
#define CFBI_CONV_THREADS 128
#endif
#ifndef CFBI_CONV_UNROLL
#define CFBI_CONV_UNROLL 4
#endif
constexpr int THREADS = CFBI_CONV_THREADS;
constexpr int TILE_T = 64;   // tokens per block; must be >= the largest supported W - 1
constexpr int MAX_W = 8;
constexpr int UNROLL = CFBI_CONV_UNROLL;    // tokens whose loads are issued together
static_assert(TILE_T >= MAX_W - 1, "a non-first tile must never need the state");

struct Args {
    const __nv_bfloat16* x;     // token (n, t) at x + (bos + t) * x_stride, channels contiguous
    __nv_bfloat16* y;           // same indexing with y_stride
    float* y_f32;               // optional: the fp32 value before the bf16 rounding (tests), or null
    const void* weight;         // [D][W], fp32 or bf16
    const void* bias;           // [D], fp32 or bf16, or null
    __nv_bfloat16* state;       // [N][D][W-1] bf16, updated in place, or null (zero history, nothing written)
    const int64_t* cu_seqlens;  // [N+1] or null
    int64_t x_stride, y_stride;
    int64_t T;                  // tokens per sequence when cu_seqlens == null
    int D, N, tiles;            // tiles: upper bound of ceil(len / TILE_T) over the sequences (>= 1)
    int silu, w_bf16, bias_bf16;
};

// ---- the program -------------------------------------------------------------------------------------------
// ascending-tap fmaf chain, one rounding per tap, starting from +0.0f
template <int W>
__device__ __forceinline__ float conv_acc(const float (&w)[W], const float (&xin)[W]) {
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < W; ++k) acc = __fmaf_rn(w[k], xin[k], acc);
    return acc;
}

// Negative control (tests only, -DCFBI_CONV_NEG_CONTROL=1): the same taps accumulated in two interleaved chains
// that are added at the end, i.e. a different association. Used by the variant build for single-token tiles only,
// which models "decode uses a different reduction than prefill".
template <int W>
__device__ __forceinline__ float conv_acc_two_chains(const float (&w)[W], const float (&xin)[W]) {
    float e = 0.0f, o = 0.0f;
#pragma unroll
    for (int k = 0; k < W; k += 2) e = __fmaf_rn(w[k], xin[k], e);
#pragma unroll
    for (int k = 1; k < W; k += 2) o = __fmaf_rn(w[k], xin[k], o);
    return __fadd_rn(e, o);
}

// bias (optional), activation (optional): one rounding each
__device__ __forceinline__ float conv_finish(float acc, bool has_bias, float bias, bool silu) {
    float z = has_bias ? __fadd_rn(acc, bias) : acc;
    return silu ? __fmul_rn(z, sigmoid_act(z)) : z;
}

__device__ __forceinline__ float ld_param(const void* p, bool bf16, int64_t i) {
    return bf16 ? bf16_to_f32(static_cast<const __nv_bfloat16*>(p)[i]) : static_cast<const float*>(p)[i];
}
__device__ __forceinline__ uint16_t ld_raw16(const __nv_bfloat16* p) { return *reinterpret_cast<const uint16_t*>(p); }
__device__ __forceinline__ void st_raw16(__nv_bfloat16* p, uint16_t v) { *reinterpret_cast<uint16_t*>(p) = v; }
// raw bf16 bits -> fp32 (exact)
__device__ __forceinline__ float raw_to_f32(uint16_t r) { return __uint_as_float((uint32_t)r << 16); }

template <int W, int VEC>
__global__ void __launch_bounds__(THREADS) conv_kernel(const Args a) {
    constexpr int CH = THREADS * VEC;   // channels per block
    const int dtiles = (a.D + CH - 1) / CH;
    int64_t b = blockIdx.x;
    const int dt = (int)(b % dtiles);
    b /= dtiles;
    const int j = (int)(b % a.tiles);
    const int n = (int)(b / a.tiles);

    int64_t bos, eos;
    if (a.cu_seqlens != nullptr) {
        bos = a.cu_seqlens[n];
        eos = a.cu_seqlens[n + 1];
    } else {
        bos = (int64_t)n * a.T;
        eos = bos + a.T;
    }
    const int64_t len = eos - bos;
    const int64_t t0 = (int64_t)j * TILE_T;
    // max_seqlen (the grid bound) smaller than a real sequence would silently drop its tail: fail loudly instead
    if (j == 0 && len > (int64_t)a.tiles * TILE_T) __trap();
    if (t0 >= len) return;   // also len == 0: nothing happens, the state stays as it is
    const int64_t t1 = min(t0 + (int64_t)TILE_T, len);
    const int d0 = (dt * THREADS + (int)threadIdx.x) * VEC;
    if (d0 >= a.D) return;

    const bool has_bias = a.bias != nullptr;
    const bool silu = a.silu != 0;
    float w[VEC][W], bias[VEC];
#pragma unroll
    for (int c = 0; c < VEC; ++c) {
#pragma unroll
        for (int k = 0; k < W; ++k) w[c][k] = ld_param(a.weight, a.w_bf16 != 0, (int64_t)(d0 + c) * W + k);
        bias[c] = has_bias ? ld_param(a.bias, a.bias_bf16 != 0, d0 + c) : 0.0f;
    }

    const __nv_bfloat16* xs = a.x + bos * a.x_stride + d0;   // xs[p * x_stride + c]: input of position p
    __nv_bfloat16* ys = a.y + bos * a.y_stride + d0;
    float* y32 = a.y_f32 != nullptr ? a.y_f32 + bos * a.y_stride + d0 : nullptr;
    // state entry m (0 .. W-2) of channel d0 + c holds the input of sequence position m - (W-1)
    const __nv_bfloat16* st = a.state != nullptr ? a.state + ((int64_t)n * a.D + d0) * (W - 1) : nullptr;

    // register window = raw bf16 bits of the inputs of positions t-(W-1) .. t-1 (the state of the program at token t)
    uint16_t win[VEC][W - 1];
#pragma unroll
    for (int m = 0; m < W - 1; ++m) {
        const int64_t p = t0 - (W - 1) + m;
#pragma unroll
        for (int c = 0; c < VEC; ++c) {
            if (p >= 0) win[c][m] = ld_raw16(xs + p * a.x_stride + c);
            else win[c][m] = st != nullptr ? ld_raw16(st + (int64_t)c * (W - 1) + (W - 1) + p) : (uint16_t)0;
        }
    }

    const bool single = (t1 - t0) == 1;
    (void)single;
    for (int64_t tb = t0; tb < t1; tb += UNROLL) {
        uint32_t xr[UNROLL];   // the VEC raw inputs of each token (VEC == 2: one 32-bit load)
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            xr[u] = 0;
            if (tb + u < t1) {
                const __nv_bfloat16* px = xs + (tb + u) * a.x_stride;
                if constexpr (VEC == 2) xr[u] = *reinterpret_cast<const uint32_t*>(px);
                else xr[u] = ld_raw16(px);
            }
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            if (tb + u < t1) {
                uint32_t yr = 0;
#pragma unroll
                for (int c = 0; c < VEC; ++c) {
                    const uint16_t cur = (VEC == 2) ? (uint16_t)(c == 0 ? (xr[u] & 0xFFFFu) : (xr[u] >> 16)) : (uint16_t)xr[u];
                    float xin[W];
#pragma unroll
                    for (int m = 0; m < W - 1; ++m) xin[m] = raw_to_f32(win[c][m]);
                    xin[W - 1] = raw_to_f32(cur);
#ifdef CFBI_CONV_NEG_CONTROL
                    const float acc = single ? conv_acc_two_chains<W>(w[c], xin) : conv_acc<W>(w[c], xin);
#else
                    const float acc = conv_acc<W>(w[c], xin);
#endif
                    const float yv = conv_finish(acc, has_bias, bias[c], silu);
                    yr |= (uint32_t)__bfloat16_as_ushort(f32_to_bf16_rn(yv)) << (16 * c);
                    if (y32 != nullptr) y32[(tb + u) * a.y_stride + c] = yv;
#pragma unroll
                    for (int m = 0; m < W - 2; ++m) win[c][m] = win[c][m + 1];
                    win[c][W - 2] = cur;
                }
                __nv_bfloat16* py = ys + (tb + u) * a.y_stride;
                if constexpr (VEC == 2) *reinterpret_cast<uint32_t*>(py) = yr;
                else *reinterpret_cast<uint16_t*>(py) = (uint16_t)yr;
            }
        }
    }

    // new state (first tile only). If this tile ends the sequence the register window IS the new state (the last W-1
    // inputs of old state ++ x); otherwise the sequence is longer than a tile and the last W-1 inputs are plain x rows.
    // All reads of the old state of this thread's channels happened before the loop; only this thread writes them.
    if (j == 0 && a.state != nullptr) {
        uint16_t nv[VEC][W - 1];
#pragma unroll
        for (int m = 0; m < W - 1; ++m) {
#pragma unroll
            for (int c = 0; c < VEC; ++c) {
                if (t1 == len) nv[c][m] = win[c][m];
                else nv[c][m] = ld_raw16(xs + (len - (W - 1) + m) * a.x_stride + c);   // position >= TILE_T - (W-1) > 0
            }
        }
        __nv_bfloat16* stw = a.state + ((int64_t)n * a.D + d0) * (W - 1);
#pragma unroll
        for (int c = 0; c < VEC; ++c) {
#pragma unroll
            for (int m = 0; m < W - 1; ++m) st_raw16(stw + (int64_t)c * (W - 1) + m, nv[c][m]);
        }
    }
}

}  // namespace conv
}  // namespace cfbi
