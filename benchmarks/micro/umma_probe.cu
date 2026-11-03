// umma_probe.cu - does tcgen05.mma (UMMA) kind::tf32 reproduce the mma.sync "dot3" bit for bit?
//
// dot3 (docs/20261102-cfbi-numerical-contract.md section 3): x_hi = bits(x)&0xFFFFE000, x_lo = bits(x - x_hi)&0xFFFFE000
// (same for y); per K-step of 8 (ascending): SMALL = mma(lo_a, hi_b, SMALL); SMALL = mma(hi_a, lo_b, SMALL);
// BIG = mma(hi_a, hi_b, BIG); result = fadd_rn(BIG, SMALL); BIG = SMALL = -0.0 initially.
//   reference path : mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32, one warp per 16x8 tile
//   candidate path : tcgen05.mma.cta_group::1.kind::tf32 (K=8 per instruction), BIG and SMALL are two TMEM
//                    accumulators, tcgen05.ld + fadd_rn in registers.
//
// Stages (argv; default: layout main neg tile time):
//   layout  descriptor / TMEM-layout validation (exact small-integer data) and M=64 lane map discovery
//   main    mismatch counts, regime x K x (M,N), for BOTH accumulator-init variants
//   neg     negative controls (reversed K order, wrong splits)
//   tile    tile-shape / position independence of one output element
//   bisect  (only if main shows mismatches) 1 K-step / few-product reproducers
//   time    cycles per tcgen05.mma, per dot3 K-step, tcgen05.ld cost
// build: nvcc -O3 -arch=sm_103a -o umma_probe umma_probe.cu && ./umma_probe
//
// Where to look when reusing the PTX (function names, see also the report): umma_tf32() [tcgen05.mma],
// umma_commit(), make_smem_desc() / make_idesc() [descriptors], tmem_alloc/dealloc, tmem_ld<>/tmem_st<>,
// smem_off() [no-swizzle K-major canonical layout], umma_kernel<M,N> [whole dot3 pipeline].
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); fflush(stdout); exit(1);} } while (0)

constexpr int SMALL_COL = 128;   // TMEM column of the SMALL accumulator (BIG lives at column 0); N <= 128
constexpr int TMEM_COLS = 256;   // columns allocated by the correctness kernel (power of two >= 32)

// ================================================================================================
// PTX wrappers
// ================================================================================================
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ float bits_f(uint32_t u) { return __uint_as_float(u); }

__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
// warp-wide (all 32 lanes of one warp); the TMEM base address (lane 0, first column) is written to *slot (smem)
__device__ __forceinline__ void tmem_alloc(uint32_t* slot, int ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(slot)), "r"(ncols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}

// One tcgen05.mma, D(tmem) (+)= A(smem desc) x B(smem desc); accumulate==0 => D = A x B (old D ignored).
// Must be executed by ONE thread of the CTA. (the {0,0,0,0} operand is the disable-output-lane mask)
__device__ __forceinline__ void umma_tf32(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t accumulate) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t}\n"
        :
        : "r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accumulate), "r"(0), "r"(0), "r"(0), "r"(0)
        : "memory");
}
// completion of all earlier tcgen05 async ops of this thread => mbarrier arrive (same thread that issued the MMAs)
__device__ __forceinline__ void umma_commit(uint32_t bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ uint32_t mbar_try_wait(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok;
}
__device__ __forceinline__ void fence_proxy_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

// UMMA shared-memory matrix descriptor (64 bit), SWIZZLE_NONE: start>>4 [0,14), LBO>>4 [16,30), SBO>>4 [32,46),
// version=1 [46,48), base_offset=0, lbo_mode=0, layout_type=0 [61,64)  (cute/arch/mma_sm100_desc.hpp SmemDescriptor)
__device__ __forceinline__ uint64_t make_smem_desc(uint32_t saddr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
    uint64_t d = (uint64_t)((saddr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo_bytes >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    return d;
}
// Instruction descriptor (32 bit) for kind::tf32: c_format=F32(1)@4, a_format=TF32(2)@7, b_format=TF32(2)@10,
// a/b negate=0, a/b major=K(0)@15/16, N>>3 @17, M>>4 @24 (cute InstrDescriptor)
__host__ __device__ constexpr uint32_t make_idesc(int M, int N) {
    return (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}

//@@GEN_LD@@

// ================================================================================================
// pinned primitives (copied from csrc/cfbi/cfbi_primitives.cuh)
// ================================================================================================
__device__ __forceinline__ void mma_tf32_m16n8k8(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void split_tf32(float x, uint32_t& hi, uint32_t& lo) {
    hi = __float_as_uint(x) & 0xFFFFE000u;
    lo = __float_as_uint(__fsub_rn(x, __uint_as_float(hi))) & 0xFFFFE000u;
}
// split variants: 0 = contract (truncate), 1 = WRONG (hi rounded to nearest), 2 = WRONG (no lo term)
__device__ __forceinline__ void split_mode_fn(float x, int mode, uint32_t& hi, uint32_t& lo) {
    if (mode == 1) {
        hi = (__float_as_uint(x) + 0x1000u) & 0xFFFFE000u;
        lo = __float_as_uint(__fsub_rn(x, __uint_as_float(hi))) & 0xFFFFE000u;
    } else {
        split_tf32(x, hi, lo);
        if (mode == 2) lo = 0u;
    }
}

// ================================================================================================
// reference: dot3 with mma.sync, one warp per 16x8 output tile.  A [T][M][K] row-major, Bt [T][N][K] (K contiguous)
// ================================================================================================
__global__ void ref_kernel(const float* __restrict__ A, const float* __restrict__ Bt, float* __restrict__ out,
                           float* __restrict__ big_o, float* __restrict__ small_o, int M, int N, int K, int reverse_k,
                           int swap_small) {
    const int t = blockIdx.z, m0 = blockIdx.y * 16, n0 = blockIdx.x * 8;
    const int lane = threadIdx.x, g = lane >> 2, tq = lane & 3;
    const float* a = A + (size_t)t * M * K;
    const float* b = Bt + (size_t)t * N * K;
    float big[4], small[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { big[i] = -0.0f; small[i] = -0.0f; }
    const int nks = K / 8;
#pragma unroll 1
    for (int s = 0; s < nks; ++s) {
        const int k0 = 8 * (reverse_k ? (nks - 1 - s) : s);
        const float af[4] = {a[(m0 + g) * K + k0 + tq], a[(m0 + g + 8) * K + k0 + tq], a[(m0 + g) * K + k0 + tq + 4], a[(m0 + g + 8) * K + k0 + tq + 4]};
        const float bf[2] = {b[(n0 + g) * K + k0 + tq], b[(n0 + g) * K + k0 + tq + 4]};
        uint32_t ah[4], al[4], bh[2], bl[2];
#pragma unroll
        for (int i = 0; i < 4; ++i) split_tf32(af[i], ah[i], al[i]);
#pragma unroll
        for (int i = 0; i < 2; ++i) split_tf32(bf[i], bh[i], bl[i]);
        if (!swap_small) { mma_tf32_m16n8k8(small, al, bh); mma_tf32_m16n8k8(small, ah, bl); }
        else             { mma_tf32_m16n8k8(small, ah, bl); mma_tf32_m16n8k8(small, al, bh); }
        mma_tf32_m16n8k8(big, ah, bh);
    }
    float* o = out + (size_t)t * M * N; float* ob = big_o + (size_t)t * M * N; float* os = small_o + (size_t)t * M * N;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = m0 + g + 8 * (i >> 1), c = n0 + 2 * tq + (i & 1);
        o[r * N + c] = __fadd_rn(big[i], small[i]);
        ob[r * N + c] = big[i];
        os[r * N + c] = small[i];
    }
}

// ================================================================================================
// candidate: the whole dot3 on tcgen05 (single CTA per tile, 128 threads = 4 warps; thread 0 issues the UMMAs)
// ================================================================================================
__device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x;
}
// neighbour filler for the placement test: 0 = zeros, 1 = random (~N(0, .6)), 2 = random with exponent in [-60, 60]
__device__ __forceinline__ float fill_val(int kind, uint32_t seed, uint32_t idx) {
    if (kind == 0) return 0.0f;
    const uint32_t h = hash32(seed * 0x9E3779B9u + idx), h2 = hash32(h ^ 0xABCDEF01u);
    const float u = (float)((h & 0xFFFFu) + (h >> 16) + (h2 & 0xFFFFu) + (h2 >> 16)) * (1.0f / 65536.0f) - 2.0f;
    if (kind == 1) return u;
    const int e = (int)(hash32(h2 ^ 0x1234567u) % 121u) - 60;
    return ldexpf(u, e);
}

struct KParams {
    int K, reverse_k, split_mode, init_mode, layout_mode, input_mode;
    // init_mode: 0 = TMEM pre-set to -0.0 (tcgen05.st), accumulate=1 on every UMMA
    //            1 = no pre-set, accumulate=0 on the first K-step (D = A x B)
    //            2 = like 1 but TMEM pre-set to a NaN sentinel (layout discovery)
    // layout_mode: 0 = core matrices adjacent along K (LBO=128B, SBO=KC/4*128B); 1 = adjacent along M/N (LBO=R/8*128B, SBO=128B)
    // input_mode: 0 = tiles from global memory (A,Bt); 1 = placement (pair_a/pair_b + fillers)
    int swap_small;   // 1 = issue the two SMALL UMMAs in the opposite order (negative control)
    const float* A; const float* Bt;
    float* out; float* big_o; float* small_o;   // [T][128][N]: raw TMEM images, TMEM lane-major
    int* err;
    const float* pair_a; const float* pair_b;   // placement mode: [K] each
    const int* place;                           // [T][4] = row, col, fill kind, seed
    const int* lane_of_row;                     // [M]  (TMEM lane that holds D row r)
    float* res;                                 // [T][3] = out, big, small of the placed element
};

// byte offset of element (row r, k-position kk in 0..KC-1) inside one plane of R rows
__device__ __forceinline__ uint32_t smem_off(int layout_mode, int R, int KC, int r, int kk) {
    const uint32_t core = (uint32_t)((r & 7) * 16 + (kk & 3) * 4);
    if (layout_mode == 0) return (uint32_t)(r >> 3) * (uint32_t)(KC / 4 * 128) + (uint32_t)(kk >> 2) * 128u + core;
    return (uint32_t)(kk >> 2) * (uint32_t)(R / 8 * 128) + (uint32_t)(r >> 3) * 128u + core;
}

template <int M, int N>
__global__ void __launch_bounds__(128, 1) umma_kernel(KParams p) {
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    __shared__ uint32_t tmem_slot;
    __shared__ __align__(8) uint64_t mbar;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, cta = blockIdx.x;
    const int K = p.K, nks = K / 8;
    const int KC = K < 64 ? K : 64, nch = K / KC, spc = KC / 8;
    uint32_t* a_hi = reinterpret_cast<uint32_t*>(smem_raw);
    uint32_t* a_lo = a_hi + M * KC;
    uint32_t* b_hi = a_lo + M * KC;
    uint32_t* b_lo = b_hi + N * KC;

    if (tid == 0) mbar_init(smem_u32(&mbar), 1);
    if (warp == 0) tmem_alloc(&tmem_slot, TMEM_COLS);
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const uint32_t tmem = tmem_slot;
    const uint32_t lane_base = (uint32_t)((warp & 3) * 32) << 16;
    const uint32_t big_t = tmem, small_t = tmem + SMALL_COL;

    if (p.init_mode == 0 || p.init_mode == 2) {
        const uint32_t pat = p.init_mode == 0 ? 0x80000000u : 0x7FC0BEEFu;
        uint32_t r8[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) r8[i] = pat;
        for (int c = 0; c < N; c += 8) {
            tmem_st<8>(big_t + lane_base + c, r8);
            tmem_st<8>(small_t + lane_base + c, r8);
        }
        tmem_wait_st();
    }

    const uint32_t idesc = make_idesc(M, N);
    const int lbo_a = p.layout_mode == 0 ? 128 : M / 8 * 128, sbo_a = p.layout_mode == 0 ? KC / 4 * 128 : 128;
    const int lbo_b = p.layout_mode == 0 ? 128 : N / 8 * 128, sbo_b = p.layout_mode == 0 ? KC / 4 * 128 : 128;
    const uint32_t step_a = p.layout_mode == 0 ? 256u : (uint32_t)(2 * lbo_a);
    const uint32_t step_b = p.layout_mode == 0 ? 256u : (uint32_t)(2 * lbo_b);
    uint32_t parity = 0;

    for (int c = 0; c < nch; ++c) {
        // ---- fill the four planes of this chunk: smem position (row, kk) <- element k = 8*kstep + kk%8
        for (int which = 0; which < 2; ++which) {
            const int R = which == 0 ? M : N;
            uint32_t* hi = which == 0 ? a_hi : b_hi;
            uint32_t* lo = which == 0 ? a_lo : b_lo;
            for (int idx = tid; idx < R * KC; idx += 128) {
                const int r = idx / KC, kk = idx % KC;
                const int s = c * spc + kk / 8;
                const int k = 8 * (p.reverse_k ? (nks - 1 - s) : s) + (kk & 7);
                float x;
                if (p.input_mode == 0) {
                    x = which == 0 ? p.A[((size_t)cta * M + r) * K + k] : p.Bt[((size_t)cta * N + r) * K + k];
                } else {
                    const int* pl = p.place + cta * 4;
                    if (r == pl[which]) x = which == 0 ? p.pair_a[k] : p.pair_b[k];
                    else x = fill_val(pl[2], (uint32_t)pl[3] * 2u + (uint32_t)which, (uint32_t)(r * 4096 + k));
                }
                uint32_t h, l;
                split_mode_fn(x, p.split_mode, h, l);
                const uint32_t off = smem_off(p.layout_mode, R, KC, r, kk) >> 2;
                hi[off] = h;
                lo[off] = l;
            }
        }
        fence_proxy_async_smem();
        tc_fence_before();
        __syncthreads();
        tc_fence_after();
        if (tid == 0) {
            for (int sl = 0; sl < spc; ++sl) {
                const int s = c * spc + sl;
                const uint32_t acc0 = p.init_mode == 0 ? 1u : (s > 0 ? 1u : 0u);
                const uint64_t dah = make_smem_desc(smem_u32(a_hi) + sl * step_a, lbo_a, sbo_a);
                const uint64_t dal = make_smem_desc(smem_u32(a_lo) + sl * step_a, lbo_a, sbo_a);
                const uint64_t dbh = make_smem_desc(smem_u32(b_hi) + sl * step_b, lbo_b, sbo_b);
                const uint64_t dbl = make_smem_desc(smem_u32(b_lo) + sl * step_b, lbo_b, sbo_b);
                if (!p.swap_small) { umma_tf32(small_t, dal, dbh, idesc, acc0); umma_tf32(small_t, dah, dbl, idesc, 1u); }
                else               { umma_tf32(small_t, dah, dbl, idesc, acc0); umma_tf32(small_t, dal, dbh, idesc, 1u); }
                umma_tf32(big_t, dah, dbh, idesc, acc0);
            }
            umma_commit(smem_u32(&mbar));
        }
        uint32_t spins = 0;
        while (!mbar_try_wait(smem_u32(&mbar), parity)) {
            if (++spins > (1u << 21)) { atomicAdd(p.err, 1); break; }
        }
        parity ^= 1u;
        tc_fence_after();
        __syncthreads();   // everybody is done with this chunk's smem before it is refilled
    }

    // ---- read the accumulators: warp w owns TMEM lanes 32*(w%4) .. +31; thread = lane = one D row (of the raw image)
    const int row = (warp & 3) * 32 + lane;
    int target_lane = -1, target_col = -1;
    if (p.input_mode == 1) { target_lane = p.lane_of_row[p.place[cta * 4 + 0]]; target_col = p.place[cta * 4 + 1]; }
    for (int c = 0; c < N; c += 8) {
        uint32_t b8[8], s8[8];
        tmem_ld<8>(big_t + lane_base + c, b8);
        tmem_ld<8>(small_t + lane_base + c, s8);
        tmem_wait_ld();
        if (p.input_mode == 0) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const size_t o = ((size_t)cta * 128 + row) * N + c + i;
                p.big_o[o] = bits_f(b8[i]);
                p.small_o[o] = bits_f(s8[i]);
                p.out[o] = __fadd_rn(bits_f(b8[i]), bits_f(s8[i]));
            }
        } else if (row == target_lane && target_col >= c && target_col < c + 8) {
            const int i = target_col - c;
            p.res[cta * 3 + 0] = __fadd_rn(bits_f(b8[i]), bits_f(s8[i]));
            p.res[cta * 3 + 1] = bits_f(b8[i]);
            p.res[cta * 3 + 2] = bits_f(s8[i]);
        }
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    if (warp == 0) tmem_dealloc(tmem, TMEM_COLS);
}

// ================================================================================================
// timing kernels (M = 128, K = 8 per UMMA)
// ================================================================================================
// MODE 0: back-to-back UMMAs into ONE accumulator; MODE 1: alternating between TWO accumulators;
// MODE 2: the dot3 K-step pattern (SMALL += lo*hi, SMALL += hi*lo, BIG += hi*hi), i.e. 3 UMMAs per iteration.
template <int N, int MODE>
__global__ void __launch_bounds__(128, 1) time_umma_kernel(long long* cyc, int iters_a, int iters_b, int* err) {
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    __shared__ uint32_t tmem_slot;
    __shared__ __align__(8) uint64_t mbar;
    const int tid = threadIdx.x, warp = tid >> 5;
    uint32_t* a_hi = reinterpret_cast<uint32_t*>(smem_raw);
    uint32_t* a_lo = a_hi + 128 * 8;
    uint32_t* b_hi = a_lo + 128 * 8;
    uint32_t* b_lo = b_hi + N * 8;
    for (int i = tid; i < 128 * 8; i += 128) { a_hi[i] = 0x3F800000u + (hash32(i) & 0x7FE000u); a_lo[i] = 0x38800000u + (hash32(i + 77) & 0x7FE000u); }
    for (int i = tid; i < N * 8; i += 128) { b_hi[i] = 0x3F800000u + (hash32(i + 5) & 0x7FE000u); b_lo[i] = 0x38800000u + (hash32(i + 99) & 0x7FE000u); }
    if (tid == 0) mbar_init(smem_u32(&mbar), 1);
    if (warp == 0) tmem_alloc(&tmem_slot, 512);
    fence_proxy_async_smem();
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const uint32_t tmem = tmem_slot;
    const uint32_t idesc = make_idesc(128, N);
    const uint64_t dah = make_smem_desc(smem_u32(a_hi), 128, 256), dal = make_smem_desc(smem_u32(a_lo), 128, 256);
    const uint64_t dbh = make_smem_desc(smem_u32(b_hi), 128, 256), dbl = make_smem_desc(smem_u32(b_lo), 128, 256);
    uint32_t parity = 0;
    long long t_issue[3] = {0, 0, 0}, t_done[3] = {0, 0, 0};
    for (int rep = 0; rep < 3; ++rep) {          // rep 0: warm-up with iters_a, rep 1: iters_a, rep 2: iters_b
        const int iters = rep == 2 ? iters_b : iters_a;
        long long t0 = 0, t1 = 0;
        if (tid == 0) {
            t0 = clock64();
            for (int i = 0; i < iters; ++i) {
                if (MODE == 0) umma_tf32(tmem, dah, dbh, idesc, 1u);
                else if (MODE == 1) umma_tf32(tmem + ((i & 1) ? 256 : 0), dah, dbh, idesc, 1u);
                else { umma_tf32(tmem + 256, dal, dbh, idesc, 1u); umma_tf32(tmem + 256, dah, dbl, idesc, 1u); umma_tf32(tmem, dah, dbh, idesc, 1u); }
            }
            t1 = clock64();
            umma_commit(smem_u32(&mbar));
        }
        uint32_t spins = 0;
        while (!mbar_try_wait(smem_u32(&mbar), parity)) { if (++spins > (1u << 22)) { atomicAdd(err, 1); break; } }
        parity ^= 1u;
        if (tid == 0) { const long long t2 = clock64(); t_issue[rep] = t1 - t0; t_done[rep] = t2 - t0; }
    }
    if (tid == 0) { cyc[0] = t_issue[1]; cyc[1] = t_done[1]; cyc[2] = t_issue[2]; cyc[3] = t_done[2]; }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    if (warp == 0) tmem_dealloc(tmem, 512);
}

// cost of tcgen05.ld.32x32b.xX + wait::ld, `nwarps` warps (each reads its own 32-lane quadrant), `reps` loads per warp
template <int X>
__global__ void __launch_bounds__(128, 1) time_ld_kernel(long long* cyc, int reps, int nwarps, int* sink) {
    __shared__ uint32_t tmem_slot;
    const int tid = threadIdx.x, warp = tid >> 5;
    if (warp == 0) tmem_alloc(&tmem_slot, 512);
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const uint32_t tmem = tmem_slot;
    const uint32_t lane_base = (uint32_t)((warp & 3) * 32) << 16;
    uint32_t acc = 0;
    long long t0 = 0, t1 = 0;
    if (warp < nwarps) {
        for (int rep = 0; rep < 2; ++rep) {   // 0: warm-up
            if (rep == 1) t0 = clock64();
            for (int i = 0; i < (rep == 0 ? 8 : reps); ++i) {
                uint32_t r[X];
                tmem_ld<X>(tmem + lane_base, r);
                tmem_wait_ld();
#pragma unroll
                for (int j = 0; j < X; ++j) acc ^= r[j];
            }
        }
        t1 = clock64();
        if ((tid & 31) == 0) cyc[warp] = t1 - t0;
    }
    if (acc == 0x12345678u) atomicAdd(sink, 1);
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    if (warp == 0) tmem_dealloc(tmem, 512);
}

// ================================================================================================
// host side
// ================================================================================================
enum { R_NORMAL, R_WIDE, R_CANCEL, R_SMALLINT, R_ONES, R_USAGE_ROW, R_USAGE_DECAY, R_TINYHUGE, R_DENORM, R_ZEROS, NREG };
static const char* regime_name[NREG] = {"normal", "wide_exp", "cancel", "small_int", "ones_mant", "usage_row", "usage_decay", "tiny_huge", "denorm_lo", "zeros"};

static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

struct HRng {
    std::mt19937_64 g;
    std::normal_distribution<double> nd{0.0, 1.0};
    std::uniform_real_distribution<double> ud{0.0, 1.0};
    explicit HRng(uint64_t seed) : g(seed) {}
    double normal() { return nd(g); }
    double u01() { return ud(g); }
    int irange(int lo, int hi) { return lo + (int)(g() % (uint64_t)(hi - lo + 1)); }
};

// A [M][K], Bt [N][K]
static void gen_regime(int regime, int M, int N, int K, HRng& rng, float* A, float* Bt) {
    auto fill = [&](float* X, int R, auto fn) { for (int r = 0; r < R; ++r) for (int k = 0; k < K; ++k) X[(size_t)r * K + k] = fn(r, k); };
    switch (regime) {
    case R_NORMAL:
        fill(A, M, [&](int, int) { return (float)rng.normal(); });
        fill(Bt, N, [&](int, int) { return (float)rng.normal(); });
        break;
    case R_WIDE:
        fill(A, M, [&](int, int) { return (float)std::ldexp(rng.normal(), rng.irange(-20, 20)); });
        fill(Bt, N, [&](int, int) { return (float)std::ldexp(rng.normal(), rng.irange(-20, 20)); });
        break;
    case R_CANCEL: {   // zero-sum rows of A against near-constant columns of B => every output is a heavy cancellation
        std::vector<double> v(K);
        for (int r = 0; r < M; ++r) {
            double mean = 0;
            for (int k = 0; k < K; ++k) { v[k] = std::ldexp(rng.normal(), rng.irange(-3, 3)); mean += v[k]; }
            mean /= K;
            for (int k = 0; k < K; ++k) A[(size_t)r * K + k] = (float)(v[k] - mean);
        }
        for (int c = 0; c < N; ++c) {
            const double cj = rng.normal();
            for (int k = 0; k < K; ++k) Bt[(size_t)c * K + k] = (float)(cj * (1.0 + 1e-5 * rng.normal()));
        }
        break;
    }
    case R_SMALLINT:
        fill(A, M, [&](int, int) { return (float)rng.irange(-8, 8); });
        fill(Bt, N, [&](int, int) { return (float)rng.irange(-8, 8); });
        break;
    case R_ONES: {   // all 23 mantissa bits set, random sign, exponent in [-3, 3]
        auto f = [&](int, int) { return u2f(((uint32_t)rng.irange(0, 1) << 31) | ((uint32_t)(127 + rng.irange(-3, 3)) << 23) | 0x7FFFFFu); };
        fill(A, M, f);
        fill(Bt, N, f);
        break;
    }
    case R_USAGE_ROW: {   // A rows = exp(-s) * gaussian (decayed queries), B plain
        for (int r = 0; r < M; ++r) { const double sc = std::exp(-3.0 * rng.u01()); for (int k = 0; k < K; ++k) A[(size_t)r * K + k] = (float)(sc * rng.normal()); }
        fill(Bt, N, [&](int, int) { return (float)rng.normal(); });
        break;
    }
    case R_USAGE_DECAY: {   // ks = khat * 2^(-d_k*(i%16)), kjr = khat * 2^(+d_k*(j%16)) with per-channel log2-decay d_k in [0, 7.2]
        std::vector<double> d(K);
        for (int k = 0; k < K; ++k) { const double u = rng.u01(); d[k] = 7.2 * u * u * u; }
        for (int r = 0; r < M; ++r) for (int k = 0; k < K; ++k) A[(size_t)r * K + k] = (float)(rng.normal() / std::sqrt(128.0) * std::exp2(-d[k] * (r % 16)));
        for (int c = 0; c < N; ++c) for (int k = 0; k < K; ++k) Bt[(size_t)c * K + k] = (float)(rng.normal() / std::sqrt(128.0) * std::exp2(d[k] * (c % 16)));
        break;
    }
    case R_TINYHUGE:
        fill(A, M, [&](int, int) { return (float)std::ldexp(rng.normal(), -100); });
        fill(Bt, N, [&](int, int) { return (float)std::ldexp(rng.normal(), 100); });
        break;
    case R_DENORM:
        fill(A, M, [&](int, int) { return (float)std::ldexp(rng.normal(), -120); });
        fill(Bt, N, [&](int, int) { return (float)std::ldexp(rng.normal(), 120); });
        break;
    case R_ZEROS:
        fill(A, M, [&](int, int) { return rng.u01() < 0.1 ? (float)rng.normal() : (rng.irange(0, 1) ? -0.0f : 0.0f); });
        fill(Bt, N, [&](int, int) { return rng.u01() < 0.5 ? (float)rng.normal() : (rng.irange(0, 1) ? -0.0f : 0.0f); });
        break;
    }
}

struct Tiles {
    int T = 0, M = 0, N = 0, K = 0;
    std::vector<float> A, Bt;   // [T][M][K], [T][N][K]
};
static Tiles make_tiles(int regime, int T, int M, int N, int K, uint64_t seed) {
    Tiles t; t.T = T; t.M = M; t.N = N; t.K = K;
    t.A.resize((size_t)T * M * K); t.Bt.resize((size_t)T * N * K);
    HRng rng(seed * 1000003ULL + (uint64_t)regime * 7919ULL + (uint64_t)K * 131ULL + (uint64_t)M * 17ULL + (uint64_t)N);
    for (int i = 0; i < T; ++i) gen_regime(regime, M, N, K, rng, t.A.data() + (size_t)i * M * K, t.Bt.data() + (size_t)i * N * K);
    return t;
}

struct Out { std::vector<float> out, big, small; };   // [T][M][N]

template <class T> struct DevBuf {
    T* p = nullptr; size_t n = 0;
    void alloc(size_t cnt) { if (n < cnt) { if (p) cudaFree(p); CK(cudaMalloc(&p, cnt * sizeof(T))); n = cnt; } }
    void up(const std::vector<T>& v) { alloc(v.size()); CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    void down(std::vector<T>& v, size_t cnt) { v.resize(cnt); CK(cudaMemcpy(v.data(), p, cnt * sizeof(T), cudaMemcpyDeviceToHost)); }
};

static int g_lane_of_row64[64];     // TMEM lane holding D row r for M=64 (discovered by the layout stage; identity until then)
static int g_layout_mode = 0;       // set by the layout stage
static int g_err_host_total = 0;

static Out run_ref(const Tiles& t, int reverse_k = 0, int swap_small = 0) {
    DevBuf<float> dA, dB, dO, dBig, dSm;
    dA.up(t.A); dB.up(t.Bt);
    const size_t n = (size_t)t.T * t.M * t.N;
    dO.alloc(n); dBig.alloc(n); dSm.alloc(n);
    dim3 grid(t.N / 8, t.M / 16, t.T);
    ref_kernel<<<grid, 32>>>(dA.p, dB.p, dO.p, dBig.p, dSm.p, t.M, t.N, t.K, reverse_k, swap_small);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    Out o; dO.down(o.out, n); dBig.down(o.big, n); dSm.down(o.small, n);
    cudaFree(dA.p); cudaFree(dB.p); cudaFree(dO.p); cudaFree(dBig.p); cudaFree(dSm.p);
    return o;
}

template <int M, int N>
static void launch_umma_t(const KParams& p, int ncta, int K) {
    const int KC = K < 64 ? K : 64;
    const size_t smem = (size_t)(2 * M + 2 * N) * KC * 4;
    CK(cudaFuncSetAttribute(umma_kernel<M, N>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    umma_kernel<M, N><<<ncta, 128, smem>>>(p);
}
static void launch_umma(int M, int N, const KParams& p, int ncta) {
    const int K = p.K;
    if (M == 64) {
        if (N == 8) launch_umma_t<64, 8>(p, ncta, K); else if (N == 16) launch_umma_t<64, 16>(p, ncta, K);
        else if (N == 32) launch_umma_t<64, 32>(p, ncta, K); else if (N == 64) launch_umma_t<64, 64>(p, ncta, K);
        else if (N == 128) launch_umma_t<64, 128>(p, ncta, K); else { printf("bad N\n"); exit(1); }
    } else {
        if (N == 8) launch_umma_t<128, 8>(p, ncta, K); else if (N == 16) launch_umma_t<128, 16>(p, ncta, K);
        else if (N == 32) launch_umma_t<128, 32>(p, ncta, K); else if (N == 64) launch_umma_t<128, 64>(p, ncta, K);
        else if (N == 128) launch_umma_t<128, 128>(p, ncta, K); else { printf("bad N\n"); exit(1); }
    }
}

struct UmmaOpts { int reverse_k = 0, split_mode = 0, init_mode = 0, layout_mode = -1, swap_small = 0; };

// returns the raw TMEM images [T][128][N] (lane-major) in raw; err count in *nerr
static void run_umma_raw(const Tiles& t, const UmmaOpts& o, Out& raw, int* nerr) {
    DevBuf<float> dA, dB, dO, dBig, dSm; DevBuf<int> dErr;
    dA.up(t.A); dB.up(t.Bt);
    const size_t n = (size_t)t.T * 128 * t.N;
    dO.alloc(n); dBig.alloc(n); dSm.alloc(n); dErr.alloc(1);
    CK(cudaMemset(dErr.p, 0, 4));
    KParams p; memset(&p, 0, sizeof(p));
    p.K = t.K; p.reverse_k = o.reverse_k; p.split_mode = o.split_mode; p.init_mode = o.init_mode;
    p.layout_mode = o.layout_mode >= 0 ? o.layout_mode : g_layout_mode; p.input_mode = 0; p.swap_small = o.swap_small;
    p.A = dA.p; p.Bt = dB.p; p.out = dO.p; p.big_o = dBig.p; p.small_o = dSm.p; p.err = dErr.p;
    launch_umma(t.M, t.N, p, t.T);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    dO.down(raw.out, n); dBig.down(raw.big, n); dSm.down(raw.small, n);
    int e = 0; CK(cudaMemcpy(&e, dErr.p, 4, cudaMemcpyDeviceToHost));
    if (nerr) *nerr = e;
    cudaFree(dA.p); cudaFree(dB.p); cudaFree(dO.p); cudaFree(dBig.p); cudaFree(dSm.p); cudaFree(dErr.p);
}
// map TMEM lanes to D rows
static Out map_rows(const Out& raw, int T, int M, int N) {
    Out o; const size_t n = (size_t)T * M * N;
    o.out.resize(n); o.big.resize(n); o.small.resize(n);
    for (int ti = 0; ti < T; ++ti) for (int r = 0; r < M; ++r) {
        const int lane = M == 128 ? r : g_lane_of_row64[r];
        for (int c = 0; c < N; ++c) {
            const size_t src = ((size_t)ti * 128 + lane) * N + c, dst = ((size_t)ti * M + r) * N + c;
            o.out[dst] = raw.out[src]; o.big[dst] = raw.big[src]; o.small[dst] = raw.small[src];
        }
    }
    return o;
}
static Out run_umma(const Tiles& t, const UmmaOpts& o, int* nerr = nullptr) {
    Out raw; run_umma_raw(t, o, raw, nerr);
    return map_rows(raw, t.T, t.M, t.N);
}

struct Mis { int tile, row, col; uint32_t rb, ub, rbig, ubig, rsm, usm; };
struct Cmp {
    size_t n = 0, mo = 0, mb = 0, ms = 0;
    double max_rel = 0, max_ulp = 0;
    std::vector<Mis> first;
};
static Cmp compare(const Out& ref, const Out& um, int T, int M, int N) {
    Cmp c; c.n = (size_t)T * M * N;
    for (size_t i = 0; i < c.n; ++i) {
        const uint32_t rb = f2u(ref.out[i]), ub = f2u(um.out[i]);
        const bool dout = rb != ub, dbig = f2u(ref.big[i]) != f2u(um.big[i]), dsm = f2u(ref.small[i]) != f2u(um.small[i]);
        c.mo += dout; c.mb += dbig; c.ms += dsm;
        if (dout) {
            const double r = ref.out[i], u = um.out[i];
            const double rel = r != 0.0 ? std::fabs(u - r) / std::fabs(r) : (u != 0.0 ? INFINITY : 0.0);
            if (rel > c.max_rel || std::isnan(rel)) c.max_rel = rel;
            if ((rb >> 31) == (ub >> 31)) c.max_ulp = std::max(c.max_ulp, (double)std::abs((long long)(rb & 0x7FFFFFFFu) - (long long)(ub & 0x7FFFFFFFu)));
            else c.max_ulp = std::max(c.max_ulp, 1e30);
        }
        if ((dout || dbig || dsm) && c.first.size() < 6) {
            const int row = (int)((i / N) % M), col = (int)(i % N), tile = (int)(i / ((size_t)M * N));
            c.first.push_back({tile, row, col, rb, ub, f2u(ref.big[i]), f2u(um.big[i]), f2u(ref.small[i]), f2u(um.small[i])});
        }
    }
    return c;
}
static void print_first(const Cmp& c, int maxn = 4) {
    for (size_t i = 0; i < c.first.size() && (int)i < maxn; ++i) {
        const Mis& m = c.first[i];
        printf("      [tile %d r%d c%d] out ref %08x (%.9g) umma %08x (%.9g) | big ref %08x umma %08x | small ref %08x umma %08x\n", m.tile, m.row, m.col,
               m.rb, u2f(m.rb), m.ub, u2f(m.ub), m.rbig, m.ubig, m.rsm, m.usm);
    }
}

static const int MS[2] = {64, 128};
static const int NS[5] = {8, 16, 32, 64, 128};
static bool g_try_m128n8 = false;
static bool cfg_supported(int M, int N) { return !(M == 128 && N == 8) || g_try_m128n8; }

// -------------------------------------------------------------------------------------------------
// stage: layout - descriptors / TMEM layout (exact integer data), M=64 lane map
// -------------------------------------------------------------------------------------------------
static bool stage_layout() {
    printf("== layout: descriptor + TMEM lane/column map (exact data D[r][n] = (r+1) + 1024*(n+1)) ==\n");
    bool all_ok = true;
    for (int M : MS) {
        const int N = 16, K = 8;
        Tiles t; t.T = 1; t.M = M; t.N = N; t.K = K; t.A.assign((size_t)M * K, 0.f); t.Bt.assign((size_t)N * K, 0.f);
        for (int r = 0; r < M; ++r) { t.A[(size_t)r * K + 0] = (float)(r + 1); t.A[(size_t)r * K + 1] = 1.f; }
        for (int n = 0; n < N; ++n) { t.Bt[(size_t)n * K + 0] = 1.f; t.Bt[(size_t)n * K + 1] = 1024.f * (n + 1); }
        bool ok_mode[2] = {false, false};
        for (int lm = 0; lm < 2; ++lm) {
            UmmaOpts o; o.init_mode = 2; o.layout_mode = lm;
            Out raw; int nerr = 0;
            run_umma_raw(t, o, raw, &nerr);
            // decode lane/col -> (r, n)
            int nfound = 0, bad_cols = 0;
            std::vector<int> row_lane(M, -1);
            for (int lane = 0; lane < 128; ++lane) {
                bool lane_ok = true; int lrow = -1;
                for (int c = 0; c < N; ++c) {
                    const float v = raw.big[(size_t)lane * N + c];
                    if (!(v >= 1.f && v < 1e7f)) { lane_ok = false; break; }
                    const int iv = (int)v; const int rr = (iv & 1023) - 1, nn = (iv >> 10) - 1;
                    if (nn != c) { lane_ok = false; bad_cols++; break; }
                    if (lrow < 0) lrow = rr; else if (lrow != rr) { lane_ok = false; break; }
                    if (rr < 0 || rr >= M) { lane_ok = false; break; }
                }
                if (lane_ok && lrow >= 0) { row_lane[lrow] = lane; nfound++; }
            }
            bool complete = nfound == M;
            for (int r = 0; r < M; ++r) if (row_lane[r] < 0) complete = false;
            printf("  M=%3d layout_mode=%d: nerr=%d, %d lanes hold valid D rows, bad_col_lanes=%d, all rows found=%d, SMALL(lane0,col0)=%g\n", M, lm, nerr, nfound,
                   bad_cols, (int)complete, raw.small[0]);
            ok_mode[lm] = complete;
            if (complete && (lm == 0 || !ok_mode[0])) {
                if (M == 64) for (int r = 0; r < 64; ++r) g_lane_of_row64[r] = row_lane[r];
                if (lm == 0 || !ok_mode[0]) g_layout_mode = lm;
                // print the map as ranges
                std::string s; int r0 = 0;
                for (int r = 1; r <= M; ++r) {
                    if (r == M || row_lane[r] != row_lane[r - 1] + 1) { char b[96]; snprintf(b, sizeof b, "%srows %d-%d->lanes %d-%d", s.empty() ? "" : "; ", r0, r - 1, row_lane[r0], row_lane[r - 1]); s += b; r0 = r; }
                }
                printf("      map: %s\n", s.c_str());
            }
        }
        all_ok = all_ok && (ok_mode[0] || ok_mode[1]);
    }
    printf("  selected layout_mode=%d\n", g_layout_mode);
    return all_ok;
}

// -------------------------------------------------------------------------------------------------
// stage: main - mismatch counts
// -------------------------------------------------------------------------------------------------
static size_t g_total_mis_out = 0;
static void stage_main(int init_mode) {
    printf("== main: UMMA dot3 vs mma.sync dot3, init_mode=%d (%s) ==\n", init_mode, init_mode == 0 ? "TMEM pre-set to -0.0, accumulate always" : "accumulate=0 on first K-step");
    printf("   cell = mismatching elements of the final result (out) / of BIG / of SMALL, out of T*M*N elements (T=8 tiles)\n");
    const int Ks[4] = {8, 16, 64, 128};
    size_t tot_mis = 0, tot_n = 0;
    struct Rec { int regime, K, M, N; Cmp c; };
    std::vector<Rec> bad;
    for (int rg = 0; rg < NREG; ++rg) {
        for (int K : Ks) {
            printf("%-12s K=%-3d", regime_name[rg], K);
            for (int M : MS) {
                printf(" | M=%-3d", M);
                for (int N : NS) {
                    if (!cfg_supported(M, N)) { printf(" N%-3d:   n/a    ", N); continue; }
                    Tiles t = make_tiles(rg, 8, M, N, K, 1);
                    Out ref = run_ref(t);
                    int nerr = 0;
                    Out um = run_umma(t, [&] { UmmaOpts o; o.init_mode = init_mode; return o; }(), &nerr);
                    Cmp c = compare(ref, um, t.T, M, N);
                    if (nerr) printf(" N%-3d:TIMEOUT", N);
                    else printf(" N%-3d:%zu/%zu/%zu", N, c.mo, c.mb, c.ms);
                    tot_mis += c.mo; tot_n += c.n;
                    if (c.mo || c.mb || c.ms) bad.push_back({rg, K, M, N, c});
                }
            }
            printf("\n");
        }
    }
    printf("   TOTAL: %zu mismatching final results out of %zu elements (%zu configurations with any mismatch)\n", tot_mis, tot_n, bad.size());
    g_total_mis_out += tot_mis;
    int shown = 0;
    for (auto& b : bad) {
        if (shown++ >= 6) break;
        printf("   first mismatches %s K=%d M=%d N=%d: out/big/small mismatches %zu/%zu/%zu of %zu, max rel diff of out %.3g, max ulp %.3g\n", regime_name[b.regime], b.K, b.M, b.N,
               b.c.mo, b.c.mb, b.c.ms, b.c.n, b.c.max_rel, b.c.max_ulp);
        print_first(b.c, 3);
    }
}

int main(int argc, char** argv) {
    std::vector<std::string> stages;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--m128n8")) g_try_m128n8 = true; else stages.push_back(argv[i]);
    }
    if (stages.empty()) stages = {"layout", "main", "neg", "tile", "time"};
    auto want = [&](const char* s) { return std::find(stages.begin(), stages.end(), std::string(s)) != stages.end(); };
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int clk = 0; CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0));
    printf("device: %s cc=%d.%d SMs=%d clock=%d kHz\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount, clk);
    for (int r = 0; r < 64; ++r) g_lane_of_row64[r] = r;
    if (want("layout")) {
        const bool ok = stage_layout();
        if (!ok) { printf("layout stage FAILED: descriptor/layout not validated; later stages are unreliable\n"); }
    }
    if (want("main")) { stage_main(0); stage_main(1); }
    return 0;
}
