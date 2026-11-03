// umma_probe.cu - does tcgen05.mma (UMMA) kind::tf32 reproduce the mma.sync "dot3" bit for bit?
//
// dot3 (docs/20261102-cfbi-numerical-contract.md section 3): x_hi = bits(x)&0xFFFFE000, x_lo = bits(x - x_hi)&0xFFFFE000
// (same for y); per K-step of 8 (ascending): SMALL = mma(lo_a, hi_b, SMALL); SMALL = mma(hi_a, lo_b, SMALL);
// BIG = mma(hi_a, hi_b, BIG); result = fadd_rn(BIG, SMALL); BIG = SMALL = -0.0 initially.
//   reference path : mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32, one warp per 16x8 tile
//   candidate path : tcgen05.mma.cta_group::1.kind::tf32 (K=8 per instruction), BIG and SMALL are two TMEM
//                    accumulators, tcgen05.ld + fadd_rn in registers.
//
// Stages (argv; default: layout main neg tile deep char time; every stage first runs the layout discovery quietly):
//   layout  descriptor / TMEM-layout validation (exact small-integer data) and the M=64 TMEM lane map
//   main    mismatch counts, regime x K x (M,N), for BOTH accumulator-init variants (TMEM pre-set to -0.0 / accumulate=0 first)
//   neg     negative controls (reversed K order, wrong splits, swapped SMALL order, fp32 FMA-chain emulation)
//   tile    one (row, column) operand pair placed at many rows/columns/tile shapes with zero/random/wide neighbours
//   deep    ~4*10^9 device-generated random output elements, device-side bit comparison of out/BIG/SMALL
//   char    what one K=8 step computes: exact sum rounded RN / RZ vs the hardware result (shows the test is sensitive to the internal arithmetic)
//   m128n8  (run alone) M=128,N=8: not a PTX-documented-valid UMMA shape (M=128 needs N>=16)
//   time    UMMA issue/completion cycles, dot3 round trip latency, tcgen05.ld cost, concurrency of UMMA streams
// build:  nvcc -O3 -std=c++17 -arch=sm_103a -o umma_probe umma_probe.cu && ./umma_probe
//         (-arch=sm_103a also emits a plain compute_103 PTX pass without tcgen05; the kernel bodies are compiled
//          out there via HAVE_TCGEN05. -gencode arch=compute_103a,code=sm_103a works as well.)
//
// Reusable pieces (function names): tcgen05.mma = umma_tf32(), completion = umma_commit() + mbar_init()/mbar_try_wait(),
// descriptors = make_smem_desc() (smem matrix descriptor) / make_idesc() (instruction descriptor), TMEM =
// tmem_alloc()/tmem_dealloc()/tmem_ld<X>()/tmem_st<X>()/tmem_wait_ld(), fences = tc_fence_before()/tc_fence_after()/
// fence_proxy_async_smem(), no-swizzle K-major canonical layout = smem_off() (+ lbo/sbo in umma_kernel), the whole
// fill-smem -> 3 UMMAs per K-step -> commit -> wait -> tcgen05.ld -> fadd pipeline = umma_kernel<M,N>.
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

// tcgen05 exists only in the arch-specific target (compute_103a / sm_103a). `nvcc -arch=sm_103a` also emits a plain
// compute_103 PTX pass; the kernel bodies are compiled out there so that this file builds with either spelling
// (-arch=sm_103a or -gencode arch=compute_103a,code=sm_103a).
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM101_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL) || defined(__CUDA_ARCH_FEAT_SM110_ALL)
#define HAVE_TCGEN05 1
#endif

[[maybe_unused]] constexpr int SMALL_COL = 128;   // TMEM column of the SMALL accumulator (BIG lives at column 0); N <= 128
[[maybe_unused]] constexpr int TMEM_COLS = 256;   // columns allocated by the correctness kernel (power of two >= 32)

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
// same with A in tensor memory (a_tmem: lane 0, first K column of the 128 x 8 A slice; fp32/tf32 words, one per column)
__device__ __forceinline__ void umma_tf32_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t bdesc, uint32_t idesc, uint32_t accumulate) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, {%5, %6, %7, %8}, p;\n\t}\n"
        :
        : "r"(d_tmem), "r"(a_tmem), "l"(bdesc), "r"(idesc), "r"(accumulate), "r"(0), "r"(0), "r"(0), "r"(0)
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

// tcgen05.ld / tcgen05.st .32x32b.xX: warp w may only touch TMEM lanes 32*(w%4)..+31; thread = lane, X = #32-bit columns
template <int X> __device__ __forceinline__ void tmem_ld(uint32_t taddr, uint32_t (&r)[X]);
template <int X> __device__ __forceinline__ void tmem_st(uint32_t taddr, const uint32_t (&r)[X]);
template <> __device__ __forceinline__ void tmem_ld<8>(uint32_t taddr, uint32_t (&r)[8]) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                 : "r"(taddr));
}

template <> __device__ __forceinline__ void tmem_ld<16>(uint32_t taddr, uint32_t (&r)[16]) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
                 : "r"(taddr));
}

template <> __device__ __forceinline__ void tmem_ld<32>(uint32_t taddr, uint32_t (&r)[32]) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
                 : "r"(taddr));
}

template <> __device__ __forceinline__ void tmem_ld<64>(uint32_t taddr, uint32_t (&r)[64]) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]), "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63])
                 : "r"(taddr));
}

template <> __device__ __forceinline__ void tmem_ld<128>(uint32_t taddr, uint32_t (&r)[128]) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x128.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, [%128];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]), "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63]), "=r"(r[64]), "=r"(r[65]), "=r"(r[66]), "=r"(r[67]), "=r"(r[68]), "=r"(r[69]), "=r"(r[70]), "=r"(r[71]), "=r"(r[72]), "=r"(r[73]), "=r"(r[74]), "=r"(r[75]), "=r"(r[76]), "=r"(r[77]), "=r"(r[78]), "=r"(r[79]), "=r"(r[80]), "=r"(r[81]), "=r"(r[82]), "=r"(r[83]), "=r"(r[84]), "=r"(r[85]), "=r"(r[86]), "=r"(r[87]), "=r"(r[88]), "=r"(r[89]), "=r"(r[90]), "=r"(r[91]), "=r"(r[92]), "=r"(r[93]), "=r"(r[94]), "=r"(r[95]), "=r"(r[96]), "=r"(r[97]), "=r"(r[98]), "=r"(r[99]), "=r"(r[100]), "=r"(r[101]), "=r"(r[102]), "=r"(r[103]), "=r"(r[104]), "=r"(r[105]), "=r"(r[106]), "=r"(r[107]), "=r"(r[108]), "=r"(r[109]), "=r"(r[110]), "=r"(r[111]), "=r"(r[112]), "=r"(r[113]), "=r"(r[114]), "=r"(r[115]), "=r"(r[116]), "=r"(r[117]), "=r"(r[118]), "=r"(r[119]), "=r"(r[120]), "=r"(r[121]), "=r"(r[122]), "=r"(r[123]), "=r"(r[124]), "=r"(r[125]), "=r"(r[126]), "=r"(r[127])
                 : "r"(taddr));
}

template <> __device__ __forceinline__ void tmem_st<8>(uint32_t taddr, const uint32_t (&r)[8]) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};\n"
                 :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]));
}

template <> __device__ __forceinline__ void tmem_st<32>(uint32_t taddr, const uint32_t (&r)[32]) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};\n"
                 :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31]));
}


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
    if (layout_mode == 2) return (uint32_t)(r >> 3) * (uint32_t)(KC / 4 * 144) + (uint32_t)(kk >> 2) * 144u + core;   // padded pitch
    return (uint32_t)(kk >> 2) * (uint32_t)(R / 8 * 128) + (uint32_t)(r >> 3) * 128u + core;
}

template <int M, int N>
__global__ void __launch_bounds__(128, 1) umma_kernel(KParams p) {
#ifdef HAVE_TCGEN05
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    __shared__ uint32_t tmem_slot;
    __shared__ __align__(8) uint64_t mbar;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, cta = blockIdx.x;
    const int K = p.K, nks = K / 8;
    const int KC = K < 64 ? K : 64, nch = K / KC, spc = KC / 8;
    const int padw = p.layout_mode == 2 ? 9 : 8;   // plane words = R*KC*padw/8
    uint32_t* a_hi = reinterpret_cast<uint32_t*>(smem_raw);
    uint32_t* a_lo = a_hi + M * KC * padw / 8;
    uint32_t* b_hi = a_lo + M * KC * padw / 8;
    uint32_t* b_lo = b_hi + N * KC * padw / 8;

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
    const int lbo_a = p.layout_mode == 0 ? 128 : (p.layout_mode == 2 ? 144 : M / 8 * 128), sbo_a = p.layout_mode == 0 ? KC / 4 * 128 : (p.layout_mode == 2 ? KC / 4 * 144 : 128);
    const int lbo_b = p.layout_mode == 0 ? 128 : (p.layout_mode == 2 ? 144 : N / 8 * 128), sbo_b = p.layout_mode == 0 ? KC / 4 * 128 : (p.layout_mode == 2 ? KC / 4 * 144 : 128);
    const uint32_t step_a = p.layout_mode == 0 ? 256u : (p.layout_mode == 2 ? 288u : (uint32_t)(2 * lbo_a));
    const uint32_t step_b = p.layout_mode == 0 ? 256u : (p.layout_mode == 2 ? 288u : (uint32_t)(2 * lbo_b));
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
#endif
}

// ================================================================================================
// timing kernels (M = 128, K = 8 per UMMA)
// ================================================================================================
// MODE 0: back-to-back UMMAs into ONE accumulator; MODE 1: alternating between TWO accumulators;
// MODE 2: the dot3 K-step pattern (SMALL += lo*hi, SMALL += hi*lo, BIG += hi*hi), i.e. 3 UMMAs per iteration.
template <int M, int N, int MODE, int TCOLS>
__global__ void __launch_bounds__(128, 1) time_umma_kernel(long long* cyc, int iters_a, int iters_b, int* err, unsigned* gbar) {
#ifdef HAVE_TCGEN05
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    __shared__ uint32_t tmem_slot;
    __shared__ __align__(8) uint64_t mbar;
    const int tid = threadIdx.x, warp = tid >> 5;
    uint32_t* a_hi = reinterpret_cast<uint32_t*>(smem_raw);
    uint32_t* a_lo = a_hi + M * 8;
    uint32_t* b_hi = a_lo + M * 8;
    uint32_t* b_lo = b_hi + N * 8;
    for (int i = tid; i < M * 8; i += 128) { a_hi[i] = 0x3F800000u + (hash32(i) & 0x7FE000u); a_lo[i] = 0x38800000u + (hash32(i + 77) & 0x7FE000u); }
    for (int i = tid; i < N * 8; i += 128) { b_hi[i] = 0x3F800000u + (hash32(i + 5) & 0x7FE000u); b_lo[i] = 0x38800000u + (hash32(i + 99) & 0x7FE000u); }
    if (tid == 0) mbar_init(smem_u32(&mbar), 1);
    if (warp == 0) tmem_alloc(&tmem_slot, TCOLS);
    fence_proxy_async_smem();
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const uint32_t tmem = tmem_slot;
    const uint32_t idesc = make_idesc(M, N);
    const uint64_t dah = make_smem_desc(smem_u32(a_hi), 128, 256), dal = make_smem_desc(smem_u32(a_lo), 128, 256);
    const uint64_t dbh = make_smem_desc(smem_u32(b_hi), 128, 256), dbl = make_smem_desc(smem_u32(b_lo), 128, 256);
    uint32_t parity = 0;
    long long t_issue[3] = {0, 0, 0}, t_done[3] = {0, 0, 0}, g0 = 0, g1 = 0;
    constexpr uint32_t ALT = TCOLS / 2;
    for (int rep = 0; rep < 3; ++rep) {          // rep 0: warm-up with iters_a, rep 1: iters_a, rep 2: iters_b
        const int iters = rep == 2 ? iters_b : iters_a;
        long long t0 = 0, t1 = 0;
        if (gbar != nullptr) {   // grid-wide start barrier (all CTAs are co-resident: <= 2 per SM) so that the timed windows really overlap
            __syncthreads();
            if (tid == 0) {
                atomicAdd(gbar, 1u);
                const unsigned target = gridDim.x * (unsigned)(rep + 1);
                unsigned spins = 0;
                while (*(volatile unsigned*)gbar < target) { if (++spins > (1u << 24)) { atomicAdd(err, 1); break; } }
            }
            __syncthreads();
        }
        if (tid == 0) {
            if (rep == 1) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
            t0 = clock64();
            for (int i = 0; i < iters; ++i) {
                if (MODE == 0) umma_tf32(tmem, dah, dbh, idesc, 1u);
                else if (MODE == 1) umma_tf32(tmem + ((i & 1) ? ALT : 0), dah, dbh, idesc, 1u);
                else { umma_tf32(tmem + ALT, dal, dbh, idesc, 1u); umma_tf32(tmem + ALT, dah, dbl, idesc, 1u); umma_tf32(tmem, dah, dbh, idesc, 1u); }
            }
            t1 = clock64();
            umma_commit(smem_u32(&mbar));
        }
        uint32_t spins = 0;
        while (!mbar_try_wait(smem_u32(&mbar), parity)) { if (++spins > (1u << 22)) { atomicAdd(err, 1); break; } }
        parity ^= 1u;
        if (tid == 0) { const long long t2 = clock64(); t_issue[rep] = t1 - t0; t_done[rep] = t2 - t0; if (rep == 1) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1)); }
    }
    if (tid == 0) {
        long long* c = cyc + (size_t)blockIdx.x * 8;
        unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        c[0] = t_issue[1]; c[1] = t_done[1]; c[2] = t_issue[2]; c[3] = t_done[2]; c[4] = smid; c[5] = g0; c[6] = g1;
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    if (warp == 0) tmem_dealloc(tmem, TCOLS);
#endif
}

// layout study: dot3 K-step pattern (3 UMMAs per iteration, 2 accumulators) with runtime LBO/SBO of A and B and a
// runtime step (bytes) between consecutive k-steps; planes are 40 KB each (data irrelevant)
template <int M, int N>
__global__ void __launch_bounds__(128, 1) time_layout_kernel(long long* cyc, int iters_a, int iters_b, int lbo_a, int sbo_a, int lbo_b, int sbo_b, int kstep_b, int* err) {
#ifdef HAVE_TCGEN05
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    __shared__ uint32_t tmem_slot;
    __shared__ __align__(8) uint64_t mbar;
    const int tid = threadIdx.x, warp = tid >> 5;
    uint32_t* base = reinterpret_cast<uint32_t*>(smem_raw);   // 4 planes x 40 KB: a_hi, a_lo, b_hi, b_lo
    for (int i = tid; i < 4 * 10240; i += 128) base[i] = 0x3F800000u + (hash32(i) & 0x7FE000u);
    if (tid == 0) mbar_init(smem_u32(&mbar), 1);
    if (warp == 0) tmem_alloc(&tmem_slot, 512);
    fence_proxy_async_smem();
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const uint32_t tmem = tmem_slot;
    const uint32_t idesc = make_idesc(M, N);
    const uint32_t a_hi = smem_u32(smem_raw), a_lo = a_hi + 40960, b_hi = a_lo + 40960, b_lo = b_hi + 40960;
    uint32_t parity = 0;
    long long t_done[3] = {0, 0, 0};
    for (int rep = 0; rep < 3; ++rep) {
        const int iters = rep == 2 ? iters_b : iters_a;
        long long t0 = 0;
        if (tid == 0) {
            t0 = clock64();
            for (int i = 0; i < iters; ++i) {
                const uint32_t kb = (uint32_t)((i & 7) * kstep_b);   // walk over 8 k-steps like the kernels do
                const uint64_t dah = make_smem_desc(a_hi, lbo_a, sbo_a), dal = make_smem_desc(a_lo, lbo_a, sbo_a);
                const uint64_t dbh = make_smem_desc(b_hi + kb, lbo_b, sbo_b), dbl = make_smem_desc(b_lo + kb, lbo_b, sbo_b);
                umma_tf32(tmem + 256, dal, dbh, idesc, 1u);
                umma_tf32(tmem + 256, dah, dbl, idesc, 1u);
                umma_tf32(tmem, dah, dbh, idesc, 1u);
            }
            umma_commit(smem_u32(&mbar));
        }
        uint32_t spins = 0;
        while (!mbar_try_wait(smem_u32(&mbar), parity)) { if (++spins > (1u << 22)) { atomicAdd(err, 1); break; } }
        parity ^= 1u;
        if (tid == 0) t_done[rep] = clock64() - t0;
    }
    if (tid == 0) { cyc[0] = t_done[1]; cyc[1] = t_done[2]; }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    if (warp == 0) tmem_dealloc(tmem, 512);
#endif
}
static void time_layout_study() {
    printf("== layout study: cycles per dot3 K-step (3 UMMAs, M=128 N=64) for descriptor strides (bytes) ==\n");
    struct C { const char* name; int la, sa, lb, sb, kb; };
    const C cs[] = {
        {"A 128/256, B 128/256 (canonical)", 128, 256, 128, 256, 256},
        {"A 160/320, B 128/256", 160, 320, 128, 256, 256},
        {"A 128/256, B 144/4608 (E planes)", 128, 256, 144, 4608, 288},
        {"A 160/320, B 144/4608 (kernel P1/P2)", 160, 320, 144, 4608, 288},
        {"A 160/320, B 144/2304 (kernel P3)", 160, 320, 144, 2304, 288},
        {"A 144/288, B 144/288", 144, 288, 144, 288, 288},
    };
    long long* d_cyc; int* d_err; CK(cudaMalloc(&d_cyc, 64)); CK(cudaMalloc(&d_err, 4)); CK(cudaMemset(d_err, 0, 4));
    const size_t smem = 4 * 40960;
    CK(cudaFuncSetAttribute(time_layout_kernel<128, 64>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    for (const C& c : cs) {
        time_layout_kernel<128, 64><<<1, 128, smem>>>(d_cyc, 256, 1024, c.la, c.sa, c.lb, c.sb, c.kb, d_err);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        long long h[2]; CK(cudaMemcpy(h, d_cyc, 16, cudaMemcpyDeviceToHost));
        printf("   %-40s : %7.1f cyc per K-step (%5.1f per UMMA)\n", c.name, (double)(h[1] - h[0]) / (1024 - 256), (double)(h[1] - h[0]) / (1024 - 256) / 3);
    }
    cudaFree(d_cyc); cudaFree(d_err);
}

// ------------------------------------------------------------------------------------------------
// Operand-swapped dot3 (the transposed orientation): the tensor-core A operand holds the Bt rows (the "y" operand of the
// contract's dot3(x = A rows, y = Bt rows)), the B operand the A rows; D'[j][i] must equal the reference out[i][j].
//   UMMA order of one K-step:  SMALL += (A'hi, B'lo) [= lo_x * hi_y];  SMALL += (A'lo, B'hi) [= hi_x * lo_y];  BIG += (A'hi, B'hi)
// TS = 1: the A' planes live in TENSOR MEMORY (written with tcgen05.st, 128 K-columns per plane) and the UMMA reads them from
// there; only B' goes through shared memory. Output: raw TMEM images [T][128 lanes = Bt row j][NB = A row i].
// ------------------------------------------------------------------------------------------------
template <int MA, int NB, int TS>
__global__ void __launch_bounds__(128, 1) umma_swap_kernel(KParams p) {
#ifdef HAVE_TCGEN05
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    __shared__ uint32_t tmem_slot;
    __shared__ __align__(8) uint64_t mbar;
    const int tid = threadIdx.x, warp = tid >> 5, cta = blockIdx.x;
    const int K = p.K, nks = K / 8;
    const int KC = K < 64 ? K : 64, nch = K / KC, spc = KC / 8;
    uint32_t* b_hi = reinterpret_cast<uint32_t*>(smem_raw);
    uint32_t* b_lo = b_hi + NB * KC * 9 / 8;
    uint32_t* a_hi = b_lo + NB * KC * 9 / 8;
    uint32_t* a_lo = a_hi + MA * KC * 9 / 8;   // only when !TS
    if (tid == 0) mbar_init(smem_u32(&mbar), 1);
    if (warp == 0) tmem_alloc(&tmem_slot, 512);
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const uint32_t tmem = tmem_slot;
    const uint32_t lane_off = (uint32_t)((warp & 3) * 32) << 16;
    constexpr uint32_t A_LO_COL = 128, BIG_COL = 256, SMALL_COL2 = 384;
    if (TS) {   // A' (= Bt rows, 128 of them) -> TMEM: thread = lane = row; hi plane columns 0.., lo plane columns 128..
        for (int c = 0; c < K / 8; ++c) {
            uint32_t hv[8], lv[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) split_mode_fn(p.Bt[((size_t)cta * MA + tid) * K + 8 * c + e], 0, hv[e], lv[e]);
            tmem_st<8>(tmem + lane_off + 8 * c, hv);
            tmem_st<8>(tmem + lane_off + A_LO_COL + 8 * c, lv);
        }
        tmem_wait_st();
        tc_fence_before();
        __syncthreads();
        tc_fence_after();
    }
    const uint32_t idesc = make_idesc(MA, NB);
    uint32_t parity = 0;
    for (int c = 0; c < nch; ++c) {
        for (int which = 0; which < 2; ++which) {   // 0: B' = A rows (NB rows), 1: A' = Bt rows (MA rows), smem only if !TS
            if (which == 1 && TS) continue;
            const int R = which == 0 ? NB : MA;
            uint32_t* hi = which == 0 ? b_hi : a_hi;
            uint32_t* lo = which == 0 ? b_lo : a_lo;
            for (int idx = tid; idx < R * KC; idx += 128) {
                const int r = idx / KC, kk = idx % KC;
                const int k = c * KC + kk;
                const float x = which == 0 ? p.A[((size_t)cta * NB + r) * K + k] : p.Bt[((size_t)cta * MA + r) * K + k];
                uint32_t h, l;
                split_mode_fn(x, 0, h, l);
                const uint32_t off = smem_off(2, R, KC, r, kk) >> 2;
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
                const int ks = c * spc + sl;
                const uint32_t acc0 = ks > 0 ? 1u : 0u;
                const uint64_t bh = make_smem_desc(smem_u32(b_hi) + sl * 288, 144, KC / 4 * 144), bl = make_smem_desc(smem_u32(b_lo) + sl * 288, 144, KC / 4 * 144);
                if (TS) {
                    umma_tf32_ts(tmem + SMALL_COL2, tmem + 8 * ks, bl, idesc, acc0);
                    umma_tf32_ts(tmem + SMALL_COL2, tmem + A_LO_COL + 8 * ks, bh, idesc, 1u);
                    umma_tf32_ts(tmem + BIG_COL, tmem + 8 * ks, bh, idesc, acc0);
                } else {
                    const uint64_t ah = make_smem_desc(smem_u32(a_hi) + sl * 288, 144, KC / 4 * 144), al = make_smem_desc(smem_u32(a_lo) + sl * 288, 144, KC / 4 * 144);
                    umma_tf32(tmem + SMALL_COL2, ah, bl, idesc, acc0);
                    umma_tf32(tmem + SMALL_COL2, al, bh, idesc, 1u);
                    umma_tf32(tmem + BIG_COL, ah, bh, idesc, acc0);
                }
            }
            umma_commit(smem_u32(&mbar));
        }
        uint32_t spins = 0;
        while (!mbar_try_wait(smem_u32(&mbar), parity)) { if (++spins > (1u << 21)) { atomicAdd(p.err, 1); break; } }
        parity ^= 1u;
        tc_fence_after();
        __syncthreads();
    }
    const int row = (warp & 3) * 32 + (tid & 31);
    for (int c = 0; c < NB; c += 8) {
        uint32_t b8[8], s8[8];
        tmem_ld<8>(tmem + BIG_COL + lane_off + c, b8);
        tmem_ld<8>(tmem + SMALL_COL2 + lane_off + c, s8);
        tmem_wait_ld();
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const size_t o = ((size_t)cta * 128 + row) * NB + c + i;
            p.big_o[o] = bits_f(b8[i]);
            p.small_o[o] = bits_f(s8[i]);
            p.out[o] = __fadd_rn(bits_f(b8[i]), bits_f(s8[i]));
        }
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    if (warp == 0) tmem_dealloc(tmem, 512);
#endif
}

// TS timing: back-to-back UMMAs with A from TMEM (dot3 K-step pattern: 3 UMMAs per iteration, 2 accumulators)
template <int NB>
__global__ void __launch_bounds__(128, 1) time_ts_kernel(long long* cyc, int iters_a, int iters_b, int ss, int* err) {
#ifdef HAVE_TCGEN05
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    __shared__ uint32_t tmem_slot;
    __shared__ __align__(8) uint64_t mbar;
    const int tid = threadIdx.x, warp = tid >> 5;
    uint32_t* base = reinterpret_cast<uint32_t*>(smem_raw);   // b_hi, b_lo (and, for ss, a_hi, a_lo): 40 KB each
    for (int i = tid; i < 4 * 10240; i += 128) base[i] = 0x3F800000u + (hash32(i) & 0x7FE000u);
    if (tid == 0) mbar_init(smem_u32(&mbar), 1);
    if (warp == 0) tmem_alloc(&tmem_slot, 512);
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const uint32_t tmem = tmem_slot, lane_off = (uint32_t)((warp & 3) * 32) << 16;
    {   // A' = 1.0-ish in TMEM (hi plane columns 0..63, lo plane 128..191)
        uint32_t v8[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) v8[i] = 0x3F800000u + (hash32(tid * 8 + i) & 0x7FE000u);
        for (int c = 0; c < 8; ++c) { tmem_st<8>(tmem + lane_off + 8 * c, v8); tmem_st<8>(tmem + lane_off + 128 + 8 * c, v8); }
        tmem_wait_st();
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const uint32_t idesc = make_idesc(128, NB);
    const uint32_t b_hi = smem_u32(smem_raw), b_lo = b_hi + 40960, a_hi = b_lo + 40960, a_lo = a_hi + 40960;
    uint32_t parity = 0;
    long long t_done[3] = {0, 0, 0};
    for (int rep = 0; rep < 3; ++rep) {
        const int iters = rep == 2 ? iters_b : iters_a;
        long long t0 = 0;
        if (tid == 0) {
            t0 = clock64();
            for (int i = 0; i < iters; ++i) {
                const uint32_t ks = (uint32_t)(i & 7);
                const uint64_t bh = make_smem_desc(b_hi + ks * 288, 144, 2304), bl = make_smem_desc(b_lo + ks * 288, 144, 2304);
                if (ss) {
                    const uint64_t ah = make_smem_desc(a_hi + ks * 288, 144, 2304), al = make_smem_desc(a_lo + ks * 288, 144, 2304);
                    umma_tf32(tmem + 384, ah, bl, idesc, 1u);
                    umma_tf32(tmem + 384, al, bh, idesc, 1u);
                    umma_tf32(tmem + 256, ah, bh, idesc, 1u);
                } else {
                    umma_tf32_ts(tmem + 384, tmem + 8 * ks, bl, idesc, 1u);
                    umma_tf32_ts(tmem + 384, tmem + 128 + 8 * ks, bh, idesc, 1u);
                    umma_tf32_ts(tmem + 256, tmem + 8 * ks, bh, idesc, 1u);
                }
            }
            umma_commit(smem_u32(&mbar));
        }
        uint32_t spins = 0;
        while (!mbar_try_wait(smem_u32(&mbar), parity)) { if (++spins > (1u << 22)) { atomicAdd(err, 1); break; } }
        parity ^= 1u;
        if (tid == 0) t_done[rep] = clock64() - t0;
    }
    if (tid == 0) { cyc[0] = t_done[1]; cyc[1] = t_done[2]; }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    if (warp == 0) tmem_dealloc(tmem, 512);
#endif
}
// issue-loop study: 3 UMMAs per iteration (dot3 K-step, M=128 N=NB, SS canonical layout) with a tcgen05.commit every `commit_every`
// iterations (0 = only at the end) and a try_wait on an already-completed mbarrier every `wait_every` iterations (0 = never)
template <int NB>
__global__ void __launch_bounds__(128, 1) time_issue_kernel(long long* cyc, int iters_a, int iters_b, int commit_every, int wait_every, int* err) {
#ifdef HAVE_TCGEN05
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    __shared__ uint32_t tmem_slot;
    __shared__ __align__(8) uint64_t mbar, mdone, mflag;
    const int tid = threadIdx.x, warp = tid >> 5;
    uint32_t* base = reinterpret_cast<uint32_t*>(smem_raw);
    for (int i = tid; i < 4 * 10240; i += 128) base[i] = 0x3F800000u + (hash32(i) & 0x7FE000u);
    if (tid == 0) { mbar_init(smem_u32(&mbar), 1); mbar_init(smem_u32(&mdone), 1); mbar_init(smem_u32(&mflag), 1);
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(&mflag)) : "memory"); }   // mflag: phase 0 complete
    if (warp == 0) tmem_alloc(&tmem_slot, 512);
    fence_proxy_async_smem();
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const uint32_t tmem = tmem_slot;
    const uint32_t idesc = make_idesc(128, NB);
    const uint32_t a_hi = smem_u32(smem_raw), a_lo = a_hi + 40960, b_hi = a_lo + 40960, b_lo = b_hi + 40960;
    uint32_t parity = 0, mpar = 0;
    long long t_done[3] = {0, 0, 0};
    for (int rep = 0; rep < 3; ++rep) {
        const int iters = rep == 2 ? iters_b : iters_a;
        long long t0 = 0;
        if (tid == 0) {
            t0 = clock64();
            for (int i = 0; i < iters; ++i) {
                const uint32_t kb = (uint32_t)((i & 7) * 256);
                if (wait_every && (i % wait_every) == 0) { uint32_t spins = 0; while (!mbar_try_wait(smem_u32(&mflag), 0)) { if (++spins > (1u << 20)) break; } }
                const uint64_t dah = make_smem_desc(a_hi, 128, 256), dal = make_smem_desc(a_lo, 128, 256);
                const uint64_t dbh = make_smem_desc(b_hi + kb, 128, 256), dbl = make_smem_desc(b_lo + kb, 128, 256);
                umma_tf32(tmem + 256, dal, dbh, idesc, 1u);
                umma_tf32(tmem + 256, dah, dbl, idesc, 1u);
                umma_tf32(tmem, dah, dbh, idesc, 1u);
                if (commit_every && (i % commit_every) == commit_every - 1) umma_commit(smem_u32(&mdone));
            }
            umma_commit(smem_u32(&mbar));
        }
        uint32_t spins = 0;
        while (!mbar_try_wait(smem_u32(&mbar), parity)) { if (++spins > (1u << 22)) { atomicAdd(err, 1); break; } }
        parity ^= 1u;
        if (tid == 0) t_done[rep] = clock64() - t0;
    }
    if (tid == 0) { cyc[0] = t_done[1]; cyc[1] = t_done[2]; }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    if (warp == 0) tmem_dealloc(tmem, 512);
#endif
}
// two independent issuing threads (warp 0 and warp 1, own mbarrier, own accumulators): is the per-UMMA floor per issuer?
// MODE 0: each issues plain UMMAs into its own accumulator; MODE 2: each issues the 3-UMMA dot3 K-step on its own BIG/SMALL pair
template <int M, int N, int MODE>
__global__ void __launch_bounds__(128, 1) time_umma2_kernel(long long* cyc, int iters_a, int iters_b, int* err) {
#ifdef HAVE_TCGEN05
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    __shared__ uint32_t tmem_slot;
    __shared__ __align__(8) uint64_t mbar[2];
    const int tid = threadIdx.x, warp = tid >> 5;
    uint32_t* a_hi = reinterpret_cast<uint32_t*>(smem_raw);
    uint32_t* a_lo = a_hi + M * 8;
    uint32_t* b_hi = a_lo + M * 8;
    uint32_t* b_lo = b_hi + N * 8;
    for (int i = tid; i < M * 8; i += 128) { a_hi[i] = 0x3F800000u + (hash32(i) & 0x7FE000u); a_lo[i] = 0x38800000u + (hash32(i + 77) & 0x7FE000u); }
    for (int i = tid; i < N * 8; i += 128) { b_hi[i] = 0x3F800000u + (hash32(i + 5) & 0x7FE000u); b_lo[i] = 0x38800000u + (hash32(i + 99) & 0x7FE000u); }
    if (tid == 0) { mbar_init(smem_u32(&mbar[0]), 1); mbar_init(smem_u32(&mbar[1]), 1); }
    if (warp == 0) tmem_alloc(&tmem_slot, 512);
    fence_proxy_async_smem();
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const uint32_t tmem = tmem_slot;
    const uint32_t idesc = make_idesc(M, N);
    const uint64_t dah = make_smem_desc(smem_u32(a_hi), 128, 256), dal = make_smem_desc(smem_u32(a_lo), 128, 256);
    const uint64_t dbh = make_smem_desc(smem_u32(b_hi), 128, 256), dbl = make_smem_desc(smem_u32(b_lo), 128, 256);
    const int iss = tid == 0 ? 0 : (tid == 32 ? 1 : -1);
    uint32_t parity = 0;
    long long t_done[3] = {0, 0, 0};
    for (int rep = 0; rep < 3; ++rep) {
        const int iters = rep == 2 ? iters_b : iters_a;
        __syncthreads();
        if (iss >= 0) {
            const uint32_t d_big = tmem + (uint32_t)iss * 256, d_small = d_big + 128;
            const long long t0 = clock64();
            for (int i = 0; i < iters; ++i) {
                if (MODE == 0) umma_tf32(d_big, dah, dbh, idesc, 1u);
                else { umma_tf32(d_small, dal, dbh, idesc, 1u); umma_tf32(d_small, dah, dbl, idesc, 1u); umma_tf32(d_big, dah, dbh, idesc, 1u); }
            }
            umma_commit(smem_u32(&mbar[iss]));
            uint32_t spins = 0;
            while (!mbar_try_wait(smem_u32(&mbar[iss]), parity)) { if (++spins > (1u << 22)) { atomicAdd(err, 1); break; } }
            t_done[rep] = clock64() - t0;
        }
        parity ^= 1u;
    }
    if (iss >= 0) { long long* c = cyc + (size_t)iss * 8; c[0] = t_done[1]; c[1] = t_done[2]; }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    if (warp == 0) tmem_dealloc(tmem, 512);
#endif
}

// cost of tcgen05.ld.32x32b.xX + wait::ld, `nwarps` warps (each reads its own 32-lane quadrant); NOUT loads are issued
// back-to-back (different column ranges) before one wait; the consumer touches only 2 of the X registers
template <int X, int NOUT>
__global__ void __launch_bounds__(128, 1) time_ld_kernel(long long* cyc, int reps, int nwarps, int* sink) {
#ifdef HAVE_TCGEN05
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
                uint32_t r[NOUT][X];
#pragma unroll
                for (int q = 0; q < NOUT; ++q) tmem_ld<X>(tmem + lane_base + q * X, r[q]);
                tmem_wait_ld();
#pragma unroll
                for (int q = 0; q < NOUT; ++q) acc ^= r[q][0] ^ r[q][X - 1];
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
#endif
}

// ================================================================================================
// host side
// ================================================================================================
enum { R_NORMAL, R_WIDE, R_CANCEL, R_SMALLINT, R_ONES, R_USAGE_ROW, R_USAGE_DECAY, R_TINYHUGE, R_DENORM, R_ZEROS, R_WIDE40, R_SPARSE2, R_EXACT0, R_SUBNORMAL, R_OVERFLOW, NREG };
constexpr int NREG_CONTRACT = R_SUBNORMAL;   // regimes >= R_SUBNORMAL are outside the contract range (informational)
static const char* regime_name[NREG] = {"normal", "wide_exp", "cancel", "small_int", "ones_mant", "usage_row", "usage_decay", "tiny_huge", "denorm_lo", "zeros", "wide_exp40", "sparse2", "exact_zero", "subnormal", "overflow"};

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
    case R_WIDE40:      // per-operand exponents in [-40, 40]: product exponents spread over +-80 (beyond any alignment window)
        fill(A, M, [&](int, int) { return (float)std::ldexp(rng.normal(), rng.irange(-40, 40)); });
        fill(Bt, N, [&](int, int) { return (float)std::ldexp(rng.normal(), rng.irange(-40, 40)); });
        break;
    case R_SPARSE2:     // only k%8 in {0,1} non-zero: every K-step of 8 adds exactly TWO products of random relative exponent (alignment sweep)
        fill(A, M, [&](int, int k) { return (k % 8) < 2 ? (float)std::ldexp(rng.normal(), rng.irange(-40, 40)) : 0.0f; });
        fill(Bt, N, [&](int, int k) { return (k % 8) < 2 ? (float)std::ldexp(rng.normal(), rng.irange(-40, 40)) : 0.0f; });
        break;
    case R_EXACT0: {    // A row = (v, -v), B column constant: the exact result is 0 (checks truncation + sign of zero)
        for (int r = 0; r < M; ++r) for (int k = 0; k < K / 2; ++k) { const float v = (float)std::ldexp(rng.normal(), rng.irange(-3, 3)); A[(size_t)r * K + k] = v; A[(size_t)r * K + k + K / 2] = -v; }
        for (int c = 0; c < N; ++c) { const float cj = (float)rng.normal(); for (int k = 0; k < K; ++k) Bt[(size_t)c * K + k] = cj; }
        break;
    }
    case R_SUBNORMAL:   // products ~2^-130 (fp32 subnormal results / accumulators)
        fill(A, M, [&](int, int) { return (float)std::ldexp(rng.normal(), -70); });
        fill(Bt, N, [&](int, int) { return (float)std::ldexp(rng.normal(), -60); });
        break;
    case R_OVERFLOW:    // products ~2^140: overflow to +-inf, inf-inf = NaN
        fill(A, M, [&](int, int) { return (float)std::ldexp(rng.normal(), 70); });
        fill(Bt, N, [&](int, int) { return (float)std::ldexp(rng.normal(), 70); });
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

static Out run_ref(const Tiles& t, int reverse_k = 0, int swap_small = 0) {
    DevBuf<float> dA, dB, dO, dBig, dSm;
    dA.up(t.A); dB.up(t.Bt);
    const size_t n = (size_t)t.T * t.M * t.N;
    dO.alloc(n); dBig.alloc(n); dSm.alloc(n);
    CK(cudaMemset(dO.p, 0xFF, n * 4)); CK(cudaMemset(dBig.p, 0xFF, n * 4)); CK(cudaMemset(dSm.p, 0xFF, n * 4));   // poison: NaN
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
    const size_t smem = (size_t)(2 * M + 2 * N) * KC * 4 * 9 / 8;   // (padded layouts need 1/8 more)
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
    CK(cudaMemset(dO.p, 0xFF, n * 4)); CK(cudaMemset(dBig.p, 0xFF, n * 4)); CK(cudaMemset(dSm.p, 0xFF, n * 4));   // poison: NaN, a kernel that did not write is caught
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
static bool g_quiet = false;
#define LPRINTF(...) do { if (!g_quiet) printf(__VA_ARGS__); } while (0)
static bool stage_layout() {
    LPRINTF("== layout: descriptor + TMEM lane/column map (exact data D[r][n] = (r+1) + 1024*(n+1)) ==\n");
    bool all_ok = true;
    for (int M : MS) {
        const int N = 16, K = 8;
        Tiles t; t.T = 1; t.M = M; t.N = N; t.K = K; t.A.assign((size_t)M * K, 0.f); t.Bt.assign((size_t)N * K, 0.f);
        for (int r = 0; r < M; ++r) { t.A[(size_t)r * K + 0] = (float)(r + 1); t.A[(size_t)r * K + 1] = 1.f; }
        for (int n = 0; n < N; ++n) { t.Bt[(size_t)n * K + 0] = 1.f; t.Bt[(size_t)n * K + 1] = 1024.f * (n + 1); }
        bool ok_mode[3] = {false, false, false};
        for (int lm = 0; lm < 3; ++lm) {
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
            LPRINTF("  M=%3d layout_mode=%d: nerr=%d, %d lanes hold valid D rows, bad_col_lanes=%d, all rows found=%d, SMALL(lane0,col0)=%g\n", M, lm, nerr, nfound,
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
                LPRINTF("      map: %s\n", s.c_str());
            }
        }
        all_ok = all_ok && (ok_mode[0] || ok_mode[1]);
        if (getenv("UMMA_FORCE_LAYOUT")) g_layout_mode = atoi(getenv("UMMA_FORCE_LAYOUT"));
    }
    LPRINTF("  selected layout_mode=%d\n", g_layout_mode);
    return all_ok;
}

// -------------------------------------------------------------------------------------------------
// stage: main - mismatch counts
// -------------------------------------------------------------------------------------------------
static size_t g_total_mis_out = 0;
static void stage_main(int init_mode) {
    printf("== main: UMMA dot3 vs mma.sync dot3, init_mode=%d (%s) ==\n", init_mode, init_mode == 0 ? "TMEM pre-set to -0.0, accumulate always" : "accumulate=0 on first K-step");
    printf("   cell = mismatching elements of the final result (out) / of BIG / of SMALL, T=8 tiles of M*N elements per cell\n");
    const int Ks[4] = {8, 16, 64, 128};
    size_t tot_mis = 0, tot_n = 0, tot_info_mis = 0, tot_info_n = 0;
    struct Rec { int regime, K, M, N; Cmp c; };
    std::vector<Rec> bad;
    for (int rg = 0; rg < NREG; ++rg) {
        if (rg == NREG_CONTRACT) printf("   -- regimes below are OUTSIDE the contract's value range (subnormal products, overflow/NaN): informational --\n");
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
                    if (rg < NREG_CONTRACT) { tot_mis += c.mo; tot_n += c.n; } else { tot_info_mis += c.mo; tot_info_n += c.n; }
                    if (c.mo || c.mb || c.ms) bad.push_back({rg, K, M, N, c});
                }
            }
            printf("\n");
        }
    }
    printf("   TOTAL (in-contract regimes): %zu mismatching final results out of %zu elements\n", tot_mis, tot_n);
    printf("   TOTAL (informational regimes): %zu mismatching final results out of %zu elements\n", tot_info_mis, tot_info_n);
    g_total_mis_out += tot_mis;
    int shown = 0;
    for (auto& b : bad) {
        if (shown++ >= 6) break;
        printf("   first mismatches %s K=%d M=%d N=%d: out/big/small mismatches %zu/%zu/%zu of %zu, max rel diff of out %.3g, max ulp %.3g\n", regime_name[b.regime], b.K, b.M, b.N,
               b.c.mo, b.c.mb, b.c.ms, b.c.n, b.c.max_rel, b.c.max_ulp);
        print_first(b.c, 3);
    }
}

// -------------------------------------------------------------------------------------------------
// CUDA-core emulation (negative control): sequential fp32 FMA chain per output element, same hi/lo split
// -------------------------------------------------------------------------------------------------
__global__ void fma_emul_kernel(const float* __restrict__ A, const float* __restrict__ Bt, float* __restrict__ out, int M, int N, int K) {
    const int t = blockIdx.y, e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= M * N) return;
    const int r = e / N, c = e % N;
    const float* a = A + ((size_t)t * M + r) * K;
    const float* b = Bt + ((size_t)t * N + c) * K;
    float big = -0.0f, small = -0.0f;
    for (int k = 0; k < K; ++k) {
        uint32_t ah, al, bh, bl;
        split_tf32(a[k], ah, al);
        split_tf32(b[k], bh, bl);
        small = __fmaf_rn(bits_f(al), bits_f(bh), small);
        small = __fmaf_rn(bits_f(ah), bits_f(bl), small);
        big = __fmaf_rn(bits_f(ah), bits_f(bh), big);
    }
    out[(size_t)t * M * N + e] = __fadd_rn(big, small);
}
static std::vector<float> run_fma_emul(const Tiles& t) {
    DevBuf<float> dA, dB, dO; dA.up(t.A); dB.up(t.Bt);
    const size_t n = (size_t)t.T * t.M * t.N; dO.alloc(n);
    fma_emul_kernel<<<dim3((t.M * t.N + 127) / 128, t.T), 128>>>(dA.p, dB.p, dO.p, t.M, t.N, t.K);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> o; dO.down(o, n);
    cudaFree(dA.p); cudaFree(dB.p); cudaFree(dO.p);
    return o;
}
static double frac_diff(const std::vector<float>& a, const std::vector<float>& b) {
    size_t d = 0;
    for (size_t i = 0; i < a.size(); ++i) d += f2u(a[i]) != f2u(b[i]);
    return (double)d / (double)a.size();
}

// -------------------------------------------------------------------------------------------------
// stage: neg - negative controls: the comparison must be able to fail
// -------------------------------------------------------------------------------------------------
static void stage_neg() {
    printf("== neg: negative controls; fraction of output elements whose bits differ (T=8 tiles per cell) ==\n");
    printf("   A: mma.sync fwd-K vs mma.sync rev-K   B: UMMA fwd-K vs UMMA rev-K       C: UMMA rev-K vs mma.sync fwd-K\n");
    printf("   D: UMMA rev-K vs mma.sync rev-K (must be 0)   E: UMMA hi=round-nearest split vs mma.sync   F: UMMA no-lo-term vs mma.sync\n");
    printf("   G: UMMA SMALL order swapped vs mma.sync   H: fp32 FMA-chain emulation vs mma.sync   I: UMMA fwd-K vs mma.sync fwd-K (the verdict, must be 0)\n");
    printf("%-12s %-4s %-8s |    A       B       C       D       E       F       G       H       I\n", "regime", "K", "M,N");
    const int shapes[2][2] = {{128, 64}, {64, 32}};
    for (int rg = 0; rg < NREG_CONTRACT; ++rg) {
        for (int K : {16, 128}) {
            for (auto& sh : shapes) {
                const int M = sh[0], N = sh[1];
                Tiles t = make_tiles(rg, 8, M, N, K, 2);
                Out rf = run_ref(t), rr = run_ref(t, 1);
                Out uf = run_umma(t, UmmaOpts{}), ur = run_umma(t, [] { UmmaOpts o; o.reverse_k = 1; return o; }());
                Out us1 = run_umma(t, [] { UmmaOpts o; o.split_mode = 1; return o; }());
                Out us2 = run_umma(t, [] { UmmaOpts o; o.split_mode = 2; return o; }());
                Out usw = run_umma(t, [] { UmmaOpts o; o.swap_small = 1; return o; }());
                std::vector<float> fe = run_fma_emul(t);
                printf("%-12s %-4d %3d,%-4d | %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f\n", regime_name[rg], K, M, N, frac_diff(rf.out, rr.out), frac_diff(uf.out, ur.out),
                       frac_diff(ur.out, rf.out), frac_diff(ur.out, rr.out), frac_diff(us1.out, rf.out), frac_diff(us2.out, rf.out), frac_diff(usw.out, rf.out),
                       frac_diff(fe, rf.out), frac_diff(uf.out, rf.out));
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// stage: tile - the same (row, column) operand pair placed at different rows/columns/tile shapes/neighbour data
// -------------------------------------------------------------------------------------------------
static DevBuf<float> g_pa, g_pb, g_res;
static DevBuf<int> g_place, g_lane, g_perr;
static std::vector<float> run_place(int M, int N, int K, const std::vector<float>& pa, const std::vector<float>& pb, const std::vector<int>& place) {
    const int ncta = (int)(place.size() / 4);
    g_pa.up(pa); g_pb.up(pb); g_place.up(place); g_res.alloc((size_t)ncta * 3); g_perr.alloc(1); g_lane.alloc(128);
    CK(cudaMemset(g_perr.p, 0, 4)); CK(cudaMemset(g_res.p, 0xFF, (size_t)ncta * 3 * 4));
    std::vector<int> lane(128);
    for (int r = 0; r < 128; ++r) lane[r] = M == 128 ? r : (r < 64 ? g_lane_of_row64[r] : 0);
    CK(cudaMemcpy(g_lane.p, lane.data(), 128 * 4, cudaMemcpyHostToDevice));
    KParams p; memset(&p, 0, sizeof(p));
    p.K = K; p.input_mode = 1; p.layout_mode = g_layout_mode; p.init_mode = 0; p.pair_a = g_pa.p; p.pair_b = g_pb.p; p.place = g_place.p;
    p.lane_of_row = g_lane.p; p.res = g_res.p; p.err = g_perr.p;
    launch_umma(M, N, p, ncta);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    int e = 0; CK(cudaMemcpy(&e, g_perr.p, 4, cudaMemcpyDeviceToHost));
    if (e) printf("  [tile] TIMEOUT in %d CTAs\n", e);
    std::vector<float> res; g_res.down(res, (size_t)ncta * 3);
    return res;
}
static void stage_tile() {
    printf("== tile: one (row, column) pair placed at many rows/columns/tile shapes with zero/random/wide-exponent neighbours ==\n");
    const int Ks[3] = {8, 64, 128};
    const int regs[4] = {R_NORMAL, R_WIDE, R_USAGE_DECAY, R_CANCEL};
    const int ntrial = 12;
    struct Shape { int M, N; };
    std::vector<Shape> shapes;
    for (int M : MS) for (int N : NS) if (cfg_supported(M, N)) shapes.push_back({M, N});
    struct Cnt { size_t n = 0, first_o = 0, first_b = 0, first_s = 0, ref_o = 0, ref_b = 0, ref_s = 0; };
    std::vector<Cnt> cnt(shapes.size());
    size_t ref_self_bad = 0, pairs = 0, nonzero_pairs = 0;
    HRng rng(12345);
    for (int rg : regs) for (int K : Ks) for (int trial = 0; trial < ntrial; ++trial) {
        // the pair: a row and a column of a 16x16 tile of this regime
        std::vector<float> A16((size_t)16 * K), B16((size_t)16 * K);
        gen_regime(rg, 16, 16, K, rng, A16.data(), B16.data());
        const int pr = rng.irange(0, 15), pc = rng.irange(0, 15);
        std::vector<float> pa(A16.begin() + (size_t)pr * K, A16.begin() + (size_t)(pr + 1) * K), pb(B16.begin() + (size_t)pc * K, B16.begin() + (size_t)(pc + 1) * K);
        // mma.sync reference: pair at (0,0)/zeros, (5,3)/normal neighbours, (15,7)/wide neighbours of a 16x8 tile
        Tiles rt; rt.T = 3; rt.M = 16; rt.N = 8; rt.K = K; rt.A.assign((size_t)3 * 16 * K, 0.f); rt.Bt.assign((size_t)3 * 8 * K, 0.f);
        const int ri[3] = {0, 5, 15}, rj[3] = {0, 3, 7};
        for (int q = 0; q < 3; ++q) {
            for (int r = 0; r < 16; ++r) for (int k = 0; k < K; ++k) rt.A[((size_t)q * 16 + r) * K + k] = q == 0 ? 0.f : (float)std::ldexp(rng.normal(), q == 2 ? rng.irange(-30, 30) : 0);
            for (int c = 0; c < 8; ++c) for (int k = 0; k < K; ++k) rt.Bt[((size_t)q * 8 + c) * K + k] = q == 0 ? 0.f : (float)std::ldexp(rng.normal(), q == 2 ? rng.irange(-30, 30) : 0);
            for (int k = 0; k < K; ++k) { rt.A[((size_t)q * 16 + ri[q]) * K + k] = pa[k]; rt.Bt[((size_t)q * 8 + rj[q]) * K + k] = pb[k]; }
        }
        Out rr = run_ref(rt);
        float ro[3], rb[3], rs[3];
        for (int q = 0; q < 3; ++q) { const size_t i = ((size_t)q * 16 + ri[q]) * 8 + rj[q]; ro[q] = rr.out[i]; rb[q] = rr.big[i]; rs[q] = rr.small[i]; }
        for (int q = 1; q < 3; ++q) ref_self_bad += (f2u(ro[q]) != f2u(ro[0])) || (f2u(rb[q]) != f2u(rb[0])) || (f2u(rs[q]) != f2u(rs[0]));
        pairs++; nonzero_pairs += ro[0] != 0.f;
        float fo = 0, fb = 0, fs = 0; bool have_first = false;
        for (size_t si = 0; si < shapes.size(); ++si) {
            const int M = shapes[si].M, N = shapes[si].N;
            std::vector<int> rows, cols;
            for (int r : {0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 95, 96, 127, M - 1}) if (r < M && std::find(rows.begin(), rows.end(), r) == rows.end()) rows.push_back(r);
            for (int c : {0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, N - 1}) if (c < N && std::find(cols.begin(), cols.end(), c) == cols.end()) cols.push_back(c);
            std::vector<int> place;
            for (int fill = 0; fill < 3; ++fill) for (int r : rows) for (int c : cols) { place.push_back(r); place.push_back(c); place.push_back(fill); place.push_back(rng.irange(1, 1 << 30)); }
            std::vector<float> res = run_place(M, N, K, pa, pb, place);
            if (!have_first) { fo = res[0]; fb = res[1]; fs = res[2]; have_first = true; }
            for (size_t q = 0; q < place.size() / 4; ++q) {
                cnt[si].n++;
                cnt[si].first_o += f2u(res[q * 3]) != f2u(fo); cnt[si].first_b += f2u(res[q * 3 + 1]) != f2u(fb); cnt[si].first_s += f2u(res[q * 3 + 2]) != f2u(fs);
                cnt[si].ref_o += f2u(res[q * 3]) != f2u(ro[0]); cnt[si].ref_b += f2u(res[q * 3 + 1]) != f2u(rb[0]); cnt[si].ref_s += f2u(res[q * 3 + 2]) != f2u(rs[0]);
            }
        }
    }
    printf("   %zu random operand pairs (regimes normal/wide_exp/usage_decay/cancel, K in {8,64,128}); %zu have a non-zero result;\n", pairs, nonzero_pairs);
    printf("   mma.sync self-check (pair at 3 different slots/neighbours of a 16x8 tile): %zu inconsistent (must be 0)\n", ref_self_bad);
    printf("   per tile shape: placements tested; mismatches (out/big/small) vs the UMMA first placement | vs the mma.sync result\n");
    size_t tot = 0, bad_first = 0, bad_ref = 0;
    for (size_t si = 0; si < shapes.size(); ++si) {
        const Cnt& c = cnt[si];
        printf("   M=%-3d N=%-3d: %7zu placements | vs first: %zu/%zu/%zu | vs mma.sync: %zu/%zu/%zu\n", shapes[si].M, shapes[si].N, c.n, c.first_o, c.first_b, c.first_s, c.ref_o, c.ref_b, c.ref_s);
        tot += c.n; bad_first += c.first_o + c.first_b + c.first_s; bad_ref += c.ref_o + c.ref_b + c.ref_s;
    }
    printf("   TOTAL: %zu placements, %zu mismatches vs first placement, %zu vs mma.sync\n", tot, bad_first, bad_ref);
}

// -------------------------------------------------------------------------------------------------
// stage: deep - ~10^9 device-generated random output elements, device-side comparison (bounds rare mismatches)
// -------------------------------------------------------------------------------------------------
__device__ __forceinline__ float dgauss(uint32_t seed, uint32_t idx) {
    const uint32_t h1 = hash32(seed * 0x9E3779B9u + idx * 2u + 1u), h2 = hash32(h1 ^ (0x85EBCA6Bu + idx));
    const float u1 = (float)((h1 >> 8) + 1u) * (1.0f / 16777217.0f), u2 = (float)(h2 >> 8) * (1.0f / 16777216.0f);
    return sqrtf(-2.0f * logf(u1)) * cospif(2.0f * u2);
}
__global__ void gen_kernel(float* __restrict__ A, float* __restrict__ Bt, int regime, int M, int N, int K, uint32_t seed) {
    const int t = blockIdx.y, which = blockIdx.z;
    const int R = which ? N : M;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= R * K) return;
    const int r = e / K, k = e % K;
    const uint32_t sd0 = seed * 7919u + (uint32_t)t * 104729u, sd = sd0 + (uint32_t)which * 31u;
    float x = 0.f;
    switch (regime) {
    case R_NORMAL: x = dgauss(sd, e); break;
    case R_WIDE: x = ldexpf(dgauss(sd, e), (int)(hash32(sd ^ ((uint32_t)e * 0x27d4eb2fu)) % 41u) - 20); break;
    case R_CANCEL:
        if (which == 0) { const int kk = k < K / 2 ? k : k - K / 2; const float v = dgauss(sd, (uint32_t)(r * K + kk)); x = k < K / 2 ? v : -v; }
        else { const float cj = dgauss(sd ^ 0xABCDu, (uint32_t)r); x = cj * (1.0f + 1e-5f * dgauss(sd, e)); }
        break;
    case R_USAGE_DECAY: {
        const float u = (float)(hash32(sd0 * 3u + (uint32_t)k) >> 8) * (1.0f / 16777216.0f), d = 7.2f * u * u * u;
        x = dgauss(sd, e) * 0.0883883476f * exp2f(which == 0 ? -d * (float)(r % 16) : d * (float)(r % 16));
        break;
    }
    case R_WIDE40: x = ldexpf(dgauss(sd, e), (int)(hash32(sd ^ ((uint32_t)e * 0x27d4eb2fu)) % 81u) - 40); break;
    case R_SPARSE2: x = (k % 8) < 2 ? ldexpf(dgauss(sd, e), (int)(hash32(sd ^ ((uint32_t)e * 0x27d4eb2fu)) % 81u) - 40) : 0.0f; break;
    case R_EXACT0:
        if (which == 0) { const int kk = k < K / 2 ? k : k - K / 2; const float v = ldexpf(dgauss(sd, (uint32_t)(r * K + kk)), (int)(hash32(sd + (uint32_t)(r * K + kk)) % 7u) - 3); x = k < K / 2 ? v : -v; }
        else x = dgauss(sd ^ 0xABCDu, (uint32_t)r);
        break;
    case R_ONES: {
        const uint32_t h = hash32(sd ^ ((uint32_t)e * 0x9E3779B9u));
        x = bits_f(((h & 1u) << 31) | ((uint32_t)(127 + (int)((h >> 1) % 7u) - 3) << 23) | 0x7FFFFFu);
        break;
    }
    }
    (which ? Bt : A)[(size_t)t * R * K + e] = x;
}
// ref [T][M][N] vs raw UMMA image [T][128][N]
__global__ void cmp_kernel(const float* ro, const float* rb, const float* rs, const float* uo, const float* ub, const float* us, const int* lane_of_row,
                           int T, int M, int N, unsigned long long* cnt) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * M * N) return;
    const int t = (int)(i / ((size_t)M * N)), rem = (int)(i % ((size_t)M * N)), r = rem / N, c = rem % N;
    const int lane = M == 128 ? r : lane_of_row[r];
    const size_t u = ((size_t)t * 128 + lane) * N + c;
    if (__float_as_uint(ro[i]) != __float_as_uint(uo[u])) atomicAdd(&cnt[0], 1ull);
    if (__float_as_uint(rb[i]) != __float_as_uint(ub[u])) atomicAdd(&cnt[1], 1ull);
    if (__float_as_uint(rs[i]) != __float_as_uint(us[u])) atomicAdd(&cnt[2], 1ull);
}
static void stage_deep(int launches) {
    printf("== deep: device-generated random tiles (256 tiles per launch x %d launches per cell), device-side bit comparison of out/big/small ==\n", launches);
    const int T = 256;
    const size_t maxA = (size_t)T * 128 * 128;
    DevBuf<float> dA, dB, rO, rB, rS, uO, uB, uS; DevBuf<unsigned long long> dc; DevBuf<int> dl, derr;
    dA.alloc(maxA); dB.alloc(maxA); rO.alloc(maxA); rB.alloc(maxA); rS.alloc(maxA); uO.alloc(maxA); uB.alloc(maxA); uS.alloc(maxA); dc.alloc(4); dl.alloc(128); derr.alloc(1);
    std::vector<int> lane(128); for (int r = 0; r < 128; ++r) lane[r] = r < 64 ? g_lane_of_row64[r] : r;
    CK(cudaMemcpy(dl.p, lane.data(), 128 * 4, cudaMemcpyHostToDevice));
    const int shapes[4][2] = {{128, 128}, {128, 32}, {64, 128}, {64, 16}};
    const int regs[8] = {R_NORMAL, R_WIDE, R_CANCEL, R_USAGE_DECAY, R_ONES, R_WIDE40, R_SPARSE2, R_EXACT0};
    unsigned long long grand_n = 0, grand_mis = 0;
    auto cell = [&](int rg, int M, int N, int K, int nl, int reverse_k, int split_mode, unsigned long long* h, int* e) {
        unsigned long long z[4] = {0, 0, 0, 0}; CK(cudaMemcpy(dc.p, z, 32, cudaMemcpyHostToDevice));
        CK(cudaMemset(derr.p, 0, 4));
        for (int L = 0; L < nl; ++L) {
            const int R = std::max(M, N);
            gen_kernel<<<dim3((R * K + 255) / 256, T, 2), 256>>>(dA.p, dB.p, rg, M, N, K, 1000u + (uint32_t)L);
            ref_kernel<<<dim3(N / 8, M / 16, T), 32>>>(dA.p, dB.p, rO.p, rB.p, rS.p, M, N, K, 0, 0);
            CK(cudaMemset(uO.p, 0xFF, (size_t)T * 128 * N * 4)); CK(cudaMemset(uB.p, 0xFF, (size_t)T * 128 * N * 4)); CK(cudaMemset(uS.p, 0xFF, (size_t)T * 128 * N * 4));
            KParams p; memset(&p, 0, sizeof(p));
            p.K = K; p.layout_mode = g_layout_mode; p.init_mode = 0; p.reverse_k = reverse_k; p.split_mode = split_mode;
            p.A = dA.p; p.Bt = dB.p; p.out = uO.p; p.big_o = uB.p; p.small_o = uS.p; p.err = derr.p;
            launch_umma(M, N, p, T);
            cmp_kernel<<<(unsigned)(((size_t)T * M * N + 255) / 256), 256>>>(rO.p, rB.p, rS.p, uO.p, uB.p, uS.p, dl.p, T, M, N, dc.p);
            CK(cudaGetLastError());
        }
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h, dc.p, 32, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(e, derr.p, 4, cudaMemcpyDeviceToHost));
        return (unsigned long long)T * M * N * nl;
    };
    {   // self-test of the harness: perturbed UMMA programs must be flagged as mismatching
        unsigned long long h[4]; int e;
        unsigned long long n = cell(R_NORMAL, 128, 128, 128, 1, 1, 0, h, &e);
        printf("   harness self-test (normal, K=128, 128x128, %llu elements): UMMA reversed-K  -> mismatches out/big/small = %llu/%llu/%llu\n", n, h[0], h[1], h[2]);
        n = cell(R_NORMAL, 128, 128, 128, 1, 0, 1, h, &e);
        printf("                                                                                  UMMA wrong split -> mismatches out/big/small = %llu/%llu/%llu\n", h[0], h[1], h[2]);
    }
    printf("%-12s %-4s | %s\n", "regime", "K", "per shape: mismatches out/big/small of elements");
    for (int rg : regs) for (int K : {8, 64, 128}) {
        printf("%-12s %-4d |", regime_name[rg], K);
        for (auto& sh : shapes) {
            const int M = sh[0], N = sh[1];
            unsigned long long h[4]; int e = 0;
            const unsigned long long n = cell(rg, M, N, K, launches, 0, 0, h, &e);
            printf(" %3dx%-3d:%llu/%llu/%llu%s", M, N, h[0], h[1], h[2], e ? "(TIMEOUT)" : "");
            grand_n += n; grand_mis += h[0] + h[1] + h[2];
        }
        printf("\n");
    }
    printf("   TOTAL: %llu output elements compared (each element = out, BIG, SMALL), %llu mismatching bit patterns\n", grand_n, grand_mis);
    for (auto* b : {&dA, &dB, &rO, &rB, &rS, &uO, &uB, &uS}) cudaFree(b->p);
}

// -------------------------------------------------------------------------------------------------
// stage: char - what arithmetic does ONE K=8 step implement? (zero accumulator; BIG = sum of 8 exact hi*hi products)
// compares the mma.sync result and the UMMA result with the exact sum (long double, only where the exponent spread of
// the products is small enough for the sum to be exact) rounded to nearest-even (RN) and toward zero (RZ)
// -------------------------------------------------------------------------------------------------
static void stage_char() {
    printf("== char: one K=8 step, accumulator -0.0: BIG vs exact sum of the 8 hi*hi products rounded RN / RZ (elements with product-exponent spread <= 36 bits) ==\n");
    printf("   %-11s %9s | mma.sync: ==RN %8s ==RZ %8s neither | UMMA: ==RN %8s ==RZ %8s neither | UMMA==mma.sync\n", "regime", "elements", "", "", "", "");
    for (int rg : {R_NORMAL, R_WIDE, R_SPARSE2, R_ONES, R_USAGE_DECAY, R_CANCEL}) {
        Tiles t = make_tiles(rg, 16, 64, 64, 8, 5);
        Out ref = run_ref(t);
        Out um = run_umma(t, UmmaOpts{});
        size_t n = 0, ref_rn = 0, ref_rz = 0, um_rn = 0, um_rz = 0, same = 0;
        for (int ti = 0; ti < t.T; ++ti) for (int r = 0; r < t.M; ++r) for (int c = 0; c < t.N; ++c) {
            long double ex = 0; int emin = 1 << 20, emax = -(1 << 20);
            for (int k = 0; k < 8; ++k) {
                const float ah = u2f(f2u(t.A[((size_t)ti * t.M + r) * 8 + k]) & 0xFFFFE000u), bh = u2f(f2u(t.Bt[((size_t)ti * t.N + c) * 8 + k]) & 0xFFFFE000u);
                const long double pr = (long double)ah * (long double)bh;
                if (pr != 0.0L) { int e; frexpl(pr, &e); emin = std::min(emin, e); emax = std::max(emax, e); }
                ex += pr;
            }
            if (emax - emin > 36 || ex == 0.0L) continue;
            const size_t i = ((size_t)ti * t.M + r) * t.N + c;
            const float rn = (float)ex;
            float rz = rn; if (std::fabs((long double)rn) > std::fabs(ex)) rz = std::nextafterf(rn, 0.0f);
            ++n;
            ref_rn += f2u(ref.big[i]) == f2u(rn); ref_rz += f2u(ref.big[i]) == f2u(rz);
            um_rn += f2u(um.big[i]) == f2u(rn); um_rz += f2u(um.big[i]) == f2u(rz);
            same += f2u(um.big[i]) == f2u(ref.big[i]);
        }
        printf("   %-11s %9zu | %12.4f %10.4f %7.4f | %12.4f %10.4f %7.4f | %.6f\n", regime_name[rg], n, (double)ref_rn / n, (double)ref_rz / n, 1.0 - std::max((double)ref_rn, (double)ref_rz) / n,
               (double)um_rn / n, (double)um_rz / n, 1.0 - std::max((double)um_rn, (double)um_rz) / n, (double)same / n);
    }
    printf("   (fractions of elements; RN and RZ coincide whenever the exact sum is representable in fp32, so ==RN and ==RZ overlap there)\n");
}

// -------------------------------------------------------------------------------------------------
// stage: time
// -------------------------------------------------------------------------------------------------
template <int M, int N, int MODE, int TCOLS = 512>
static void time_umma_one(int ncta, double& issue, double& done, int* distinct_sms = nullptr, double* overlap = nullptr) {
    const int iters_a = 256, iters_b = 1024;
    long long* d_cyc; int* d_err; unsigned* d_gbar;
    CK(cudaMalloc(&d_cyc, (size_t)ncta * 8 * 8)); CK(cudaMalloc(&d_err, 4)); CK(cudaMemset(d_err, 0, 4)); CK(cudaMalloc(&d_gbar, 4)); CK(cudaMemset(d_gbar, 0, 4));
    const size_t smem = (size_t)(2 * M * 8 + 2 * N * 8) * 4;
    CK(cudaFuncSetAttribute(time_umma_kernel<M, N, MODE, TCOLS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    time_umma_kernel<M, N, MODE, TCOLS><<<ncta, 128, smem>>>(d_cyc, iters_a, iters_b, d_err, ncta > 1 ? d_gbar : nullptr);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<long long> h((size_t)ncta * 8); CK(cudaMemcpy(h.data(), d_cyc, h.size() * 8, cudaMemcpyDeviceToHost));
    int e = 0; CK(cudaMemcpy(&e, d_err, 4, cudaMemcpyDeviceToHost));
    std::vector<double> di, dd;
    std::vector<long long> sms; long long g0min = INT64_MAX, g1max = 0, g0max = 0, g1min = INT64_MAX;
    for (int c = 0; c < ncta; ++c) {
        di.push_back((double)(h[c * 8 + 2] - h[c * 8 + 0]) / (iters_b - iters_a)); dd.push_back((double)(h[c * 8 + 3] - h[c * 8 + 1]) / (iters_b - iters_a));
        sms.push_back(h[c * 8 + 4]);
        g0min = std::min(g0min, h[c * 8 + 5]); g0max = std::max(g0max, h[c * 8 + 5]); g1max = std::max(g1max, h[c * 8 + 6]); g1min = std::min(g1min, h[c * 8 + 6]);
    }
    std::sort(di.begin(), di.end()); std::sort(dd.begin(), dd.end());
    issue = di[di.size() / 2]; done = dd[dd.size() / 2];
    if (distinct_sms) { std::sort(sms.begin(), sms.end()); *distinct_sms = (int)(std::unique(sms.begin(), sms.end()) - sms.begin()); }
    if (overlap) *overlap = g0max < g1min ? 1.0 : 0.0;   // 1 if every CTA's measured window started before any CTA's window ended
    if (getenv("UMMA_DBG")) printf(   // UMMA_DBG=1: print how well the timed windows of the CTAs overlapped
        "   [dbg] ncta=%d window starts spread %lld ns, ends spread %lld ns, earliest end - latest start = %lld ns\n", ncta, g0max - g0min, g1max - g1min, g1min - g0max);
    if (e) printf("  [time] TIMEOUT (grid barrier: CTAs not co-resident?)\n");
    cudaFree(d_cyc); cudaFree(d_err); cudaFree(d_gbar);
}
template <int M, int N> static void time_umma_row(int ncta) {
    double i0, d0, i1, d1, i2, d2; double ov0 = 1, ov1 = 1, ov2 = 1;
    time_umma_one<M, N, 0>(ncta, i0, d0, nullptr, &ov0); time_umma_one<M, N, 1>(ncta, i1, d1, nullptr, &ov1); time_umma_one<M, N, 2>(ncta, i2, d2, nullptr, &ov2);
    if (ncta > 1 && !(ov0 && ov1 && ov2)) printf("   [warning: timed windows of the %d CTAs did not all overlap]\n", ncta);
    const double macs = (double)M * N * 8;   // MACs of one M x N x K=8 UMMA
    printf("   M=%-3d N=%-3d | 1 acc: issue %6.1f done %6.1f cyc/UMMA (%6.0f MAC/clk/SM) | 2 alt acc: issue %6.1f done %6.1f cyc/UMMA (%6.0f MAC/clk/SM) | dot3 k-step (3 UMMAs): %6.1f cyc (%5.1f cyc/UMMA, %6.0f MAC/clk/SM)\n",
           M, N, i0, d0, macs / d0, i1, d1, macs / d1, d2, d2 / 3, 3 * macs / d2);
}
// (d) is the ~45-cycle per-UMMA floor per issuer / per CTA, or per SM?
template <int M, int N> static void time_floor_row() {
    const double macs = (double)M * N * 8;
    // two CTAs per SM (256 TMEM columns each => both fit), each with one issuing thread
    double i2c, d2c; int nsm = 0; double ov = 0;
    time_umma_one<M, N, 0, 256>(296, i2c, d2c, &nsm, &ov);
    // one CTA, two issuing threads (warps 0 and 1)
    long long* d_cyc; int* d_err; CK(cudaMalloc(&d_cyc, 2 * 8 * 8)); CK(cudaMalloc(&d_err, 4)); CK(cudaMemset(d_err, 0, 4));
    const size_t smem = (size_t)(2 * M * 8 + 2 * N * 8) * 4;
    double r[2];
    for (int mode : {0, 2}) {
        if (mode == 0) { CK(cudaFuncSetAttribute(time_umma2_kernel<M, N, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); time_umma2_kernel<M, N, 0><<<1, 128, smem>>>(d_cyc, 256, 1024, d_err); }
        else { CK(cudaFuncSetAttribute(time_umma2_kernel<M, N, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); time_umma2_kernel<M, N, 2><<<1, 128, smem>>>(d_cyc, 256, 1024, d_err); }
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        long long h[16]; CK(cudaMemcpy(h, d_cyc, 128, cudaMemcpyDeviceToHost));
        r[mode / 2] = std::max((double)(h[1] - h[0]), (double)(h[9] - h[8])) / (1024 - 256);   // slower issuer, cycles per iteration
    }
    int e = 0; CK(cudaMemcpy(&e, d_err, 4, cudaMemcpyDeviceToHost));
    printf("   M=%-3d N=%-3d | 2 CTAs/SM (296 CTAs on %d SMs, windows overlap=%d): %6.1f cyc/UMMA per CTA (%6.0f MAC/clk/SM aggregate) | 1 CTA, 2 issuing warps: %6.1f cyc/UMMA per issuer (%6.0f MAC/clk/SM aggregate), dot3 k-step per issuer %6.1f cyc (%6.0f MAC/clk/SM aggregate)%s\n",
           M, N, nsm, (int)ov, d2c, 2 * macs / d2c, r[0], 2 * macs / r[0], r[1], 2 * 3 * macs / r[1], e ? " TIMEOUT" : "");
    cudaFree(d_cyc); cudaFree(d_err);
}

// round trip of a dot3 chain: issue 3*nks UMMAs (M=128), commit, wait, tcgen05.ld of BIG and SMALL (N cols), fadd
template <int N>
__global__ void __launch_bounds__(128, 1) time_roundtrip_kernel(long long* cyc, int nks, int* sink) {
#ifdef HAVE_TCGEN05
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
    if (warp == 0) tmem_alloc(&tmem_slot, 256);
    fence_proxy_async_smem();
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const uint32_t tmem = tmem_slot, lane_base = (uint32_t)((warp & 3) * 32) << 16;
    const uint32_t idesc = make_idesc(128, N);
    const uint64_t dah = make_smem_desc(smem_u32(a_hi), 128, 256), dal = make_smem_desc(smem_u32(a_lo), 128, 256);
    const uint64_t dbh = make_smem_desc(smem_u32(b_hi), 128, 256), dbl = make_smem_desc(smem_u32(b_lo), 128, 256);
    uint32_t parity = 0, acc = 0;
    for (int rep = 0; rep < 4; ++rep) {
        __syncthreads();
        const long long t0 = clock64();
        if (tid == 0) {
            for (int s = 0; s < nks; ++s) {
                umma_tf32(tmem + SMALL_COL, dal, dbh, idesc, s > 0 ? 1u : 0u);
                umma_tf32(tmem + SMALL_COL, dah, dbl, idesc, 1u);
                umma_tf32(tmem, dah, dbh, idesc, s > 0 ? 1u : 0u);
            }
            umma_commit(smem_u32(&mbar));
        }
        uint32_t spins = 0;
        while (!mbar_try_wait(smem_u32(&mbar), parity)) { if (++spins > (1u << 22)) break; }
        parity ^= 1u;
        const long long t_mma = clock64();
        tc_fence_after();
        constexpr int CH = N < 64 ? N : 64;   // load BIG and SMALL in chunks of <= 64 columns (register budget)
#pragma unroll
        for (int c0 = 0; c0 < N; c0 += CH) {
            uint32_t rb[CH], rs[CH];
            tmem_ld<CH>(tmem + lane_base + c0, rb);
            tmem_ld<CH>(tmem + SMALL_COL + lane_base + c0, rs);
            tmem_wait_ld();
            acc ^= rb[0] ^ rs[CH - 1];
        }
        if (acc == 0x12345678u) atomicAdd(sink, 1);   // data-dependent branch: the clock read below cannot be hoisted above the loads
        const long long t_ld = clock64();
        if (tid == 0 && rep == 3) { cyc[0] = t_mma - t0; cyc[1] = t_ld - t0; }
    }
    if (acc == 0x12345678u) atomicAdd(sink, 1);
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    if (warp == 0) tmem_dealloc(tmem, 256);
#endif
}
template <int N> static void time_roundtrip_row() {
    long long* d_cyc; int* d_sink; CK(cudaMalloc(&d_cyc, 64)); CK(cudaMalloc(&d_sink, 4)); CK(cudaMemset(d_sink, 0, 4));
    const size_t smem = (size_t)(2 * 128 * 8 + 2 * N * 8) * 4;
    CK(cudaFuncSetAttribute(time_roundtrip_kernel<N>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    printf("   N=%-3d |", N);
    for (int nks : {1, 4, 16}) {
        time_roundtrip_kernel<N><<<1, 128, smem>>>(d_cyc, nks, d_sink);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        long long h[3]; CK(cudaMemcpy(h, d_cyc, 16, cudaMemcpyDeviceToHost));
        printf(" K=%-3d: mma-done %5lld, ld-done %5lld |", nks * 8, h[0], h[1]);
    }
    printf("\n");
    cudaFree(d_cyc); cudaFree(d_sink);
}
template <int X, int NOUT> static void time_ld_one(double* res) {
    long long* d_cyc; int* d_sink; CK(cudaMalloc(&d_cyc, 64)); CK(cudaMalloc(&d_sink, 4)); CK(cudaMemset(d_sink, 0, 4));
    const int reps = 200;
    int k = 0;
    for (int nw : {1, 4}) {
        time_ld_kernel<X, NOUT><<<1, 128>>>(d_cyc, reps, nw, d_sink);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        long long h[4]; CK(cudaMemcpy(h, d_cyc, 32, cudaMemcpyDeviceToHost));
        double mx = 0; for (int w = 0; w < nw; ++w) mx = std::max(mx, (double)h[w]);
        res[k++] = mx / reps / NOUT;   // cycles per tcgen05.ld
    }
    cudaFree(d_cyc); cudaFree(d_sink);
}
template <int X> static void time_ld_row() {
    double a[2], b[2];
    time_ld_one<X, 1>(a); time_ld_one<X, 2>(b);
    printf("   X=%-3d (128 x %-3d fp32 tile = 4 warps x 32 lanes) | 1 ld then wait: %6.1f cyc/ld (1 warp), %6.1f cyc/ld (4 warps) | 2 lds then 1 wait: %6.1f cyc/ld (1 warp), %6.1f (4 warps) | %.2f cyc/column\n",
           X, X, a[0], a[1], b[0], b[1], b[1] / X);
}
static void stage_time(int clk_khz) {
    printf("== time: UMMA K=8 per instruction, 1 CTA per SM; clock64 cycles; mma.sync peak = 0.49 m16n8k8/clk/SM = 502 MAC/clk/SM ==\n");
    printf("   (a) UMMA issue / completion cost, 1 CTA, thread 0 issues back-to-back; 'done' includes waiting for tcgen05.commit\n");
    time_umma_row<128, 16>(1); time_umma_row<128, 32>(1); time_umma_row<128, 64>(1); time_umma_row<128, 128>(1);
    time_umma_row<64, 8>(1); time_umma_row<64, 16>(1); time_umma_row<64, 32>(1); time_umma_row<64, 64>(1); time_umma_row<64, 128>(1);
    printf("   (a') same with all 148 SMs busy at once (one CTA each; median CTA)\n");
    time_umma_row<128, 16>(148); time_umma_row<128, 64>(148); time_umma_row<128, 128>(148);
    printf("   (d) is the per-UMMA floor per issuer, per CTA, or per SM? (concurrent independent UMMA streams)\n");
    time_floor_row<128, 16>(); time_floor_row<128, 32>(); time_floor_row<128, 64>(); time_floor_row<128, 128>(); time_floor_row<64, 16>();
    printf("   (b) round trip of a dot3 chain (3 UMMAs per K-step, M=128): cycles from first issue to: mma complete (mbarrier) / tcgen05.ld of BIG+SMALL (all N columns, 4 warps) complete\n");
    time_roundtrip_row<16>(); time_roundtrip_row<64>(); time_roundtrip_row<128>();
    printf("   (c) tcgen05.ld.32x32b.xX + wait::ld (each warp reads its own 32-lane quadrant; consumer touches 2 registers only)\n");
    time_ld_row<8>(); time_ld_row<16>(); time_ld_row<32>(); time_ld_row<64>(); time_ld_row<128>();
    (void)clk_khz;
}

template <int NB> static void time_ts_row() {
    long long* d_cyc; int* d_err; CK(cudaMalloc(&d_cyc, 64)); CK(cudaMalloc(&d_err, 4)); CK(cudaMemset(d_err, 0, 4));
    CK(cudaFuncSetAttribute(time_ts_kernel<NB>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 40960));
    double r[2];
    for (int ss = 0; ss < 2; ++ss) {
        time_ts_kernel<NB><<<1, 128, 4 * 40960>>>(d_cyc, 256, 1024, ss, d_err);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        long long h[2]; CK(cudaMemcpy(h, d_cyc, 16, cudaMemcpyDeviceToHost));
        r[ss] = (double)(h[1] - h[0]) / (1024 - 256);
    }
    printf("   M=128 N=%-3d | A from TMEM (TS): %6.1f cyc per dot3 K-step (%5.1f per UMMA, %5.0f MAC/clk) | A from smem (SS): %6.1f cyc per K-step (%5.1f per UMMA)\n", NB, r[0], r[0] / 3,
           3.0 * 128 * NB * 8 / r[0], r[1], r[1] / 3);
    cudaFree(d_cyc); cudaFree(d_err);
}

template <int MA, int NB, int TS>
static void launch_swap_t(const KParams& p, int ncta, int K) {
    const int KC = K < 64 ? K : 64;
    const size_t smem = (size_t)(2 * NB + (TS ? 0 : 2 * MA)) * KC * 4 * 9 / 8 + 1024;
    CK(cudaFuncSetAttribute(umma_swap_kernel<MA, NB, TS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    umma_swap_kernel<MA, NB, TS><<<ncta, 128, smem>>>(p);
}
// returns D' mapped back to the reference orientation [T][Mref = NB][Nref = MA]
static Out run_swap(const Tiles& t, int ts) {
    const int MA = t.N, NB = t.M;
    DevBuf<float> dA, dB, dO, dBig, dSm; DevBuf<int> dErr;
    dA.up(t.A); dB.up(t.Bt);
    const size_t n = (size_t)t.T * 128 * NB;
    dO.alloc(n); dBig.alloc(n); dSm.alloc(n); dErr.alloc(1);
    CK(cudaMemset(dErr.p, 0, 4));
    CK(cudaMemset(dO.p, 0xFF, n * 4)); CK(cudaMemset(dBig.p, 0xFF, n * 4)); CK(cudaMemset(dSm.p, 0xFF, n * 4));
    KParams p; memset(&p, 0, sizeof(p));
    p.K = t.K; p.A = dA.p; p.Bt = dB.p; p.out = dO.p; p.big_o = dBig.p; p.small_o = dSm.p; p.err = dErr.p;
    if (MA == 128 && NB == 64) { if (ts) launch_swap_t<128, 64, 1>(p, t.T, t.K); else launch_swap_t<128, 64, 0>(p, t.T, t.K); }
    else if (MA == 128 && NB == 128) { if (ts) launch_swap_t<128, 128, 1>(p, t.T, t.K); else launch_swap_t<128, 128, 0>(p, t.T, t.K); }
    else if (MA == 64 && NB == 64) launch_swap_t<64, 64, 0>(p, t.T, t.K);
    else if (MA == 64 && NB == 128) launch_swap_t<64, 128, 0>(p, t.T, t.K);
    else { printf("unsupported swap shape\n"); exit(1); }
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    Out raw; dO.down(raw.out, n); dBig.down(raw.big, n); dSm.down(raw.small, n);
    int e = 0; CK(cudaMemcpy(&e, dErr.p, 4, cudaMemcpyDeviceToHost));
    if (e) printf("  [swap] TIMEOUT\n");
    Out o; const size_t m = (size_t)t.T * NB * MA;
    o.out.resize(m); o.big.resize(m); o.small.resize(m);
    for (int ti = 0; ti < t.T; ++ti) for (int i = 0; i < NB; ++i) for (int j = 0; j < MA; ++j) {
        const int lane = MA == 128 ? j : g_lane_of_row64[j];
        const size_t src = ((size_t)ti * 128 + lane) * NB + i, dst = ((size_t)ti * NB + i) * MA + j;
        o.out[dst] = raw.out[src]; o.big[dst] = raw.big[src]; o.small[dst] = raw.small[src];
    }
    cudaFree(dA.p); cudaFree(dB.p); cudaFree(dO.p); cudaFree(dBig.p); cudaFree(dSm.p); cudaFree(dErr.p);
    return o;
}
static void stage_swap() {
    printf("== swap: operand-swapped UMMA (A' = Bt rows, B' = A rows), D' transposed vs the mma.sync dot3 (reference orientation) ==\n");
    printf("   cell = mismatching elements out/BIG/SMALL (T=8 tiles); TS = A' in tensor memory (tcgen05.st + UMMA [a_tmem])\n");
    const int regs[] = {R_NORMAL, R_WIDE, R_CANCEL, R_USAGE_DECAY, R_ONES, R_EXACT0, R_SPARSE2, R_WIDE40, R_ZEROS, R_SMALLINT};
    struct Cfg { int MA, NB, ts; };
    const Cfg cfgs[] = {{128, 64, 0}, {128, 128, 0}, {64, 64, 0}, {64, 128, 0}, {128, 64, 1}, {128, 128, 1}};
    size_t tot = 0, totn = 0;
    for (int rg : regs) for (int K : {8, 16, 64, 128}) {
        printf("%-12s K=%-3d", regime_name[rg], K);
        for (const Cfg& c : cfgs) {
            Tiles t = make_tiles(rg, 8, c.NB, c.MA, K, 7);   // M = NB rows of A, N = MA rows of Bt
            Out ref = run_ref(t);
            Out sw = run_swap(t, c.ts);
            Cmp cm = compare(ref, sw, t.T, t.M, t.N);
            printf(" | MA%-3d NB%-3d %s %zu/%zu/%zu", c.MA, c.NB, c.ts ? "TS" : "SS", cm.mo, cm.mb, cm.ms);
            tot += cm.mo + cm.mb + cm.ms; totn += cm.n;
            if ((cm.mo || cm.mb || cm.ms) && tot < 100000) print_first(cm, 2);
        }
        printf("\n");
    }
    printf("   TOTAL swapped-operand mismatches (out+BIG+SMALL): %zu over %zu elements per accumulator\n", tot, totn);
}

static void time_issue_study() {
    printf("== issue-loop study: cycles per dot3 K-step (3 UMMAs, M=128, N=64 / 128) ==\n");
    long long* d_cyc; int* d_err; CK(cudaMalloc(&d_cyc, 64)); CK(cudaMalloc(&d_err, 4)); CK(cudaMemset(d_err, 0, 4));
    CK(cudaFuncSetAttribute(time_issue_kernel<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 40960));
    CK(cudaFuncSetAttribute(time_issue_kernel<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 40960));
    struct C { int commit, wait; };
    const C cs[] = {{0, 0}, {1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 1}, {0, 2}, {1, 1}, {4, 4}};
    for (int nb : {64, 128}) {
        for (const C& c : cs) {
            if (nb == 64) time_issue_kernel<64><<<1, 128, 4 * 40960>>>(d_cyc, 256, 1024, c.commit, c.wait, d_err);
            else time_issue_kernel<128><<<1, 128, 4 * 40960>>>(d_cyc, 256, 1024, c.commit, c.wait, d_err);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            long long h[2]; CK(cudaMemcpy(h, d_cyc, 16, cudaMemcpyDeviceToHost));
            printf("   N=%-3d commit every %d iterations, try_wait(done barrier) every %d: %7.1f cyc per K-step\n", nb, c.commit, c.wait, (double)(h[1] - h[0]) / (1024 - 256));
        }
    }
    cudaFree(d_cyc); cudaFree(d_err);
}
int main(int argc, char** argv) {
    std::vector<std::string> stages;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--m128n8")) g_try_m128n8 = true; else stages.push_back(argv[i]);
    }
    if (stages.empty()) stages = {"layout", "main", "neg", "tile", "deep", "char", "time"};
    auto want = [&](const char* s) { return std::find(stages.begin(), stages.end(), std::string(s)) != stages.end(); };
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int clk = 0; CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0));
    printf("device: %s cc=%d.%d SMs=%d clock=%d kHz\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount, clk);
    for (int r = 0; r < 64; ++r) g_lane_of_row64[r] = r;
    {   // the layout stage always runs (it discovers the M=64 lane map every other stage needs); quiet unless requested
        g_quiet = !want("layout");
        const bool ok = stage_layout();
        g_quiet = false;
        if (!ok) { printf("layout stage FAILED: descriptor/layout not validated; later stages are unreliable\n"); }
    }
    if (want("main")) { stage_main(0); stage_main(1); }
    if (want("neg")) stage_neg();
    if (want("tile")) stage_tile();
    if (want("char")) stage_char();
    if (want("layoutstudy")) time_layout_study();
    if (want("issuestudy")) time_issue_study();
    if (want("swap")) stage_swap();
    if (want("ts")) { printf("== ts timing ==\n"); time_ts_row<64>(); time_ts_row<128>(); }
    if (want("time")) stage_time(clk);
    if (want("deep")) stage_deep(24);
    if (want("m128n8")) {   // M=128 with N=8 is not a valid UMMA shape per the PTX tables (M=128: N=16..256 step 16); may fault -> run alone
        printf("== m128n8: attempting M=128 N=8 (not a documented-valid shape) ==\n");
        g_try_m128n8 = true;
        Tiles t = make_tiles(R_NORMAL, 8, 128, 8, 64, 1);
        Out ref = run_ref(t); int nerr = 0; Out um = run_umma(t, UmmaOpts{}, &nerr);
        Cmp c = compare(ref, um, t.T, 128, 8);
        printf("  M=128 N=8 K=64 normal: nerr=%d, mismatches out/big/small %zu/%zu/%zu of %zu\n", nerr, c.mo, c.mb, c.ms, c.n);
        print_first(c, 3);
    }
    return 0;
}
