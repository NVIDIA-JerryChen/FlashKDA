// Chunk-64 KDA prefill (fwd64): K1 = token-parallel operands (c64_k1.cuh), K2 = serial recurrence (c64_k2.cuh).
#include "c64_params.h"
#include "c64_k1.cuh"
#include "c64_k2.cuh"

namespace c64 {

namespace {

constexpr float kLog2e = 1.4426950408889634f;

// tile_prefix[n] = sum_{m<n} ceil(seg_len[m] / 64)
__global__ void build_tile_prefix(const int* __restrict__ seg_len, int N, int* __restrict__ tile_prefix) {
    if (threadIdx.x == 0) {
        int acc = 0;
        tile_prefix[0] = 0;
        for (int i = 0; i < N; ++i) {
            const int len = seg_len[i];
            acc += len > 0 ? (len + CHUNK - 1) / CHUNK : 0;
            tile_prefix[i + 1] = acc;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// K1: one CTA per (tile, head)
// ---------------------------------------------------------------------------------------------------------------
constexpr int kK1Threads = 256;
__global__ void __launch_bounds__(kK1Threads, 2) k1_kernel(const FwdParams p) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    K1Smem& s = *reinterpret_cast<K1Smem*>(smem_raw);
    const int tile_g = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    long long* tp = (p.prof != nullptr && tile_g == 7 && h == 0) ? p.prof : nullptr;
    if (tp && tid == 0) tp[8] = clock64();
    if (tile_g >= p.tile_prefix[p.N]) return;
    int lo = 0, hi = p.N;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) >> 1;
        if (p.tile_prefix[mid] <= tile_g) lo = mid; else hi = mid;
    }
    const int n = lo;
    const int local_t = tile_g - p.tile_prefix[n];
    const int row0 = local_t * CHUNK;
    const int valid = min(CHUNK, p.seg_len[n] - row0);
    const int64_t tok0 = p.seg_tok0[n] + row0;
    const int H = p.H;

    for (int id = tid; id < 3 * 1024; id += kK1Threads) {   // k, q, g: 3 x 1024 pieces of 16 bytes
        const int arr = id >> 10, idx = id & 1023;
        const int row = idx >> 4, c8 = (idx & 15) * 8;
        uint16_t* dst = (arr == 0 ? s.k : (arr == 1 ? s.q : s.gr)) + row * RS + c8;
        if (row < valid) {
            const uint16_t* src = (arr == 0 ? p.k : (arr == 1 ? p.q : p.g)) + ((tok0 + row) * H + h) * D + c8;
            cp_async16(dst, src);
        } else {
            st_smem_u4(dst, make_uint4(0u, 0u, 0u, 0u));
        }
    }
    if (tid < CHUNK) s.braw[tid] = (tid < valid) ? p.beta[(tok0 + tid) * H + h] : uint16_t(0);
    if (tid < D) s.dt[tid] = p.dt_bias[h * D + tid];
    cp_async_commit();
    GateParams gp;
    gp.a_log_exp = ex2_ftz(fmul(p.A_log[h], kLog2e));
    gp.gate_scale = p.gate_scale;
    gp.scale_b = rbf(p.scale);
    cp_async_wait<0>();
    __syncthreads();

    const int64_t ws_idx = (int64_t)h * p.total_tiles + tile_g;
    auto sink = [&](int row, int c8, uint4 kd, uint4 qd, uint4 kr) {
        const int64_t o = (ws_idx * CHUNK + row) * D + c8;
        *reinterpret_cast<uint4*>(p.ws_kd + o) = kd;
        *reinterpret_cast<uint4*>(p.ws_qd + o) = qd;
        *reinterpret_cast<uint4*>(p.ws_kr + o) = kr;
    };
    k1_core<kK1Threads>(s, gp, valid, true, sink, tp);
    __syncthreads();
    for (int idx = tid; idx < CHUNK * CHUNK / 8; idx += kK1Threads) {   // INV, Mqk: [64][64] bf16, 512 pieces each
        const int row = idx >> 3, c8 = (idx & 7) * 8;
        const int64_t o = (ws_idx * CHUNK + row) * CHUNK + c8;
        *reinterpret_cast<uint4*>(p.ws_inv + o) = ld_smem_u4(&s.inv[row * MS + c8]);
        *reinterpret_cast<uint4*>(p.ws_mq + o) = ld_smem_u4(&s.mq[row * MS + c8]);
    }
    if (tid < D / 4) *reinterpret_cast<uint4*>(p.ws_gt + ws_idx * D + 4 * tid) = *reinterpret_cast<const uint4*>(&s.gt[4 * tid]);
    if (tp && tid == 0) tp[6] = clock64();
}

// ---------------------------------------------------------------------------------------------------------------
// K2: one CTA per (segment, head, value slice). The recurrence is independent per value column (S[v][:], the v column of
// v / P / W' / U / out), so the 128 columns may be split over NSPLIT CTAs; every output element is computed by the
// same instructions whatever the split (NWARP, NB, NSPLIT only change which warp owns it), so the bits do not change.
// Operands of the next tile are fetched with cp.async during the step.
// ---------------------------------------------------------------------------------------------------------------
template <int VS>
struct K2StageBuf {
    uint16_t kd[CHUNK * RS];
    uint16_t qd[CHUNK * RS];
    uint16_t kr[CHUNK * RS];
    uint16_t v[CHUNK * (VS + 8)];
    uint16_t inv[CHUNK * MS];
    uint16_t mq[CHUNK * MS];
    float gt[D];
    uint16_t braw[CHUNK];
};
template <int VS>
struct K2Smem {
    uint16_t S[VS * RS];
    K2StageBuf<VS> st[2];
};

template <int NWARP, int NB, int NSPLIT>
__global__ void __launch_bounds__(32 * NWARP, 1) k2_kernel(const FwdParams p) {
    constexpr int T2 = 32 * NWARP;
    constexpr int VS = D / NSPLIT;                       // value columns of this CTA
    static_assert(NWARP * NB * 16 == VS, "the warps of a CTA cover its value slice");
    constexpr int LDV = VS + 8;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    K2Smem<VS>& sm = *reinterpret_cast<K2Smem<VS>*>(smem_raw);
    const int n = blockIdx.x, h = blockIdx.y, zs = blockIdx.z, tid = threadIdx.x;
    const int v0 = zs * VS;                              // first value column of this CTA
    const int len = p.seg_len[n];
    const int slot = p.seg_slot[n];
    const int H = p.H;
    if (len <= 0) {
        // empty segment: its final state is its initial state (flash_kda.fwd's contract); nothing to do in place
        if (p.st_out != nullptr && p.st_out != p.st_in) {
            uint16_t* gd = p.st_out + ((int64_t)slot * H + h) * (D * D) + (int64_t)v0 * D;
            const uint16_t* gs = p.st_in ? p.st_in + ((int64_t)slot * H + h) * (D * D) + (int64_t)v0 * D : nullptr;
            for (int idx = tid; idx < VS * D / 8; idx += T2)
                *reinterpret_cast<uint4*>(gd + idx * 8) = gs ? *reinterpret_cast<const uint4*>(gs + idx * 8) : make_uint4(0u, 0u, 0u, 0u);
        }
        return;
    }
    const int64_t tok0 = p.seg_tok0[n];
    const int ntile = (len + CHUNK - 1) / CHUNK;
    const int tile_base = p.tile_prefix[n];

    // state slice -> smem
    {
        const uint16_t* gs = p.st_in ? p.st_in + ((int64_t)slot * H + h) * (D * D) + (int64_t)v0 * D : nullptr;
        for (int idx = tid; idx < VS * D / 8; idx += T2) {
            const int row = idx >> 4, c8 = (idx & 15) * 8;
            if (gs) cp_async16(&sm.S[row * RS + c8], gs + row * D + c8);
            else st_smem_u4(&sm.S[row * RS + c8], make_uint4(0u, 0u, 0u, 0u));
        }
    }
    auto issue = [&](int t, K2StageBuf<VS>& sb) {
        const int64_t ws_idx = (int64_t)h * p.total_tiles + tile_base + t;
        const int valid = min(CHUNK, len - t * CHUNK);
        const int64_t trow0 = tok0 + (int64_t)t * CHUNK;
        #pragma unroll
        for (int i = 0; i < 3 * 1024 / T2; ++i) {         // Kd, Qd, Kr: 3 x 1024 pieces
            const int id = tid + T2 * i;
            const int arr = id >> 10, idx = id & 1023;
            const int row = idx >> 4, c8 = (idx & 15) * 8;
            uint16_t* dst = (arr == 0 ? sb.kd : (arr == 1 ? sb.qd : sb.kr)) + row * RS + c8;
            const uint16_t* src = (arr == 0 ? p.ws_kd : (arr == 1 ? p.ws_qd : p.ws_kr)) + (ws_idx * CHUNK + row) * D + c8;
            cp_async16(dst, src);
        }
        #pragma unroll
        for (int i = 0; i < CHUNK * (VS / 8) / T2; ++i) { // v: 64 rows x VS/8 pieces
            const int id = tid + T2 * i;
            const int row = id / (VS / 8), c8 = (id % (VS / 8)) * 8;
            uint16_t* dst = sb.v + row * LDV + c8;
            if (row < valid) cp_async16(dst, p.v + ((trow0 + row) * H + h) * D + v0 + c8);
            else st_smem_u4(dst, make_uint4(0u, 0u, 0u, 0u));
        }
        #pragma unroll
        for (int i = 0; i < 1024 / T2; ++i) {             // INV, Mqk: 2 x 512 pieces
            const int id = tid + T2 * i;
            const int arr = id >> 9, idx = id & 511;
            const int row = idx >> 3, c8 = (idx & 7) * 8;
            const uint16_t* src = (arr == 0 ? p.ws_inv : p.ws_mq) + (ws_idx * CHUNK + row) * CHUNK + c8;
            cp_async16((arr == 0 ? sb.inv : sb.mq) + row * MS + c8, src);
        }
        if (tid < D / 4) cp_async16(&sb.gt[4 * tid], p.ws_gt + ws_idx * D + 4 * tid);
    };
    auto load_beta = [&](int t) -> uint16_t {
        const int valid = min(CHUNK, len - t * CHUNK);
        return (tid < CHUNK && tid < valid) ? p.beta[(tok0 + (int64_t)t * CHUNK + tid) * H + h] : uint16_t(0);
    };

    issue(0, sm.st[0]);
    cp_async_commit();
    if (tid < CHUNK) sm.st[0].braw[tid] = load_beta(0);

    long long* tp2 = (p.prof != nullptr && n == 0 && h == 0 && zs == 0) ? p.prof + 16 : nullptr;
    long long t_wait = 0, t_comp = 0;
    for (int t = 0; t < ntile; ++t) {
        long long c0 = clock64();
        cp_async_wait<0>();
        bar_sync(1, T2);                  // tile t has landed everywhere; every warp has finished step t-1
        long long c1 = clock64();
        t_wait += c1 - c0;
        uint16_t bnext = 0;
        if (t + 1 < ntile) {
            issue(t + 1, sm.st[(t + 1) & 1]);
            cp_async_commit();
            bnext = load_beta(t + 1);
        }
        const K2StageBuf<VS>& sb = sm.st[t & 1];
        K2Stage st;
        st.kd = sb.kd; st.qd = sb.qd; st.kr = sb.kr; st.v = sb.v; st.inv = sb.inv; st.mq = sb.mq; st.gt = sb.gt;
        st.braw = sb.braw; st.ldv = LDV;
        const int valid = min(CHUNK, len - t * CHUNK);
        const int64_t orow0 = tok0 + (int64_t)t * CHUNK;
        auto emit = [&](int row, int vc, uint32_t pk) {
            *reinterpret_cast<uint32_t*>(p.out + ((orow0 + row) * H + h) * D + v0 + vc) = pk;
        };
        k2_step<NWARP, NB>(st, sm.S, valid, true, emit);
        if (t + 1 < ntile && tid < CHUNK) sm.st[(t + 1) & 1].braw[tid] = bnext;
        t_comp += clock64() - c1;
    }
    if (tp2 && tid == 0) { tp2[0] = t_wait; tp2[1] = t_comp; tp2[2] = ntile; }

    if (p.st_out) {
        bar_sync(1, T2);
        uint16_t* gd = p.st_out + ((int64_t)slot * H + h) * (D * D) + (int64_t)v0 * D;
        for (int idx = tid; idx < VS * D / 8; idx += T2) {
            const int row = idx >> 4, c8 = (idx & 15) * 8;
            *reinterpret_cast<uint4*>(gd + row * D + c8) = ld_smem_u4(&sm.S[row * RS + c8]);
        }
    }
}

size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

}  // namespace

// workspace layout: Kd | Qd | Kr (bf16 [slots][64][128]) | gt (fp32 [slots][128]) | INV | Mqk (bf16 [slots][64][64]) | tile prefix
size_t fwd64_workspace_bytes(int total_tiles, int H, int N) {
    const size_t slots = (size_t)total_tiles * H;
    size_t b = 3 * slots * CHUNK * D * 2 + slots * D * 4 + 2 * slots * CHUNK * CHUNK * 2;
    b = align_up(b, 128);
    return b + align_up(((size_t)N + 1) * 4, 128);
}

void fwd64_bind_workspace(FwdParams& p, void* wsv) {
    char* ws = reinterpret_cast<char*>(wsv);
    const size_t slots = (size_t)p.total_tiles * p.H;
    p.ws_kd = reinterpret_cast<uint16_t*>(ws);
    p.ws_qd = reinterpret_cast<uint16_t*>(ws + slots * CHUNK * D * 2);
    p.ws_kr = reinterpret_cast<uint16_t*>(ws + 2 * slots * CHUNK * D * 2);
    p.ws_gt = reinterpret_cast<float*>(ws + 3 * slots * CHUNK * D * 2);
    p.ws_inv = reinterpret_cast<uint16_t*>(ws + 3 * slots * CHUNK * D * 2 + slots * D * 4);
    p.ws_mq = reinterpret_cast<uint16_t*>(ws + 3 * slots * CHUNK * D * 2 + slots * D * 4 + slots * CHUNK * CHUNK * 2);
    size_t b = 3 * slots * CHUNK * D * 2 + slots * D * 4 + 2 * slots * CHUNK * CHUNK * 2;
    p.tile_prefix = reinterpret_cast<int*>(ws + align_up(b, 128));
}

int fwd64_k1_smem_bytes() { return int(sizeof(K1Smem)); }
int fwd64_k2_smem_bytes(int cfg) {
    switch (cfg) {
        case 0: return int(sizeof(K2Smem<128>));
        case 1: return int(sizeof(K2Smem<128>));
        case 2: return int(sizeof(K2Smem<64>));
        default: return int(sizeof(K2Smem<32>));
    }
}

namespace {
template <int NWARP, int NB, int NSPLIT>
void launch_k2(FwdParams const& p, cudaStream_t stream) {
    constexpr int VS = D / NSPLIT;
    const int smem = int(sizeof(K2Smem<VS>));
    cudaFuncSetAttribute(k2_kernel<NWARP, NB, NSPLIT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    k2_kernel<NWARP, NB, NSPLIT><<<dim3(p.N, p.H, NSPLIT), 32 * NWARP, smem, stream>>>(p);
}
}  // namespace

// k2_cfg: 0 = 4 warps x 32 columns, 1 = 8 warps x 16, 2 = 4 warps x 16 on 2 CTAs, 3 = 2 warps x 16 on 4 CTAs
void launch_fwd64(FwdParams const& p, int k2_cfg, cudaStream_t stream) {
    if (p.N <= 0 || p.H <= 0 || p.total_tiles <= 0) return;
    build_tile_prefix<<<1, 32, 0, stream>>>(p.seg_len, p.N, p.tile_prefix);
    cudaFuncSetAttribute(k1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(K1Smem)));
    k1_kernel<<<dim3(p.total_tiles, p.H), kK1Threads, sizeof(K1Smem), stream>>>(p);
    switch (k2_cfg) {
        case 1: launch_k2<8, 1, 1>(p, stream); break;
        case 2: launch_k2<4, 1, 2>(p, stream); break;
        case 3: launch_k2<2, 1, 4>(p, stream); break;
        default: launch_k2<4, 2, 1>(p, stream); break;
    }
}

}  // namespace c64
