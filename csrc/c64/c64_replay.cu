// Open-chunk replay for the chunk-64 program: decode (and the head / tail rows of prefill calls) bitwise equal to one fwd64
// call over the whole sequence.
//
// Why this works (same argument as csrc/replay for flash_kda.fwd, now for 64-row chunks): inside a chunk, output row r
// reads only rows 0..r of k, v, g, beta, row r of q and the bf16 state S at the chunk start (the gate cumsum is a
// sequential fold, L / Mqk / (I+L)^-1 are lower triangular); S changes only when the chunk closes. A decode step for the
// token at chunk row r therefore runs the chunk arithmetic (k1_core + k2_step, the very functions the prefill kernels
// run) on rows 0..r of the open chunk from S and emits row r. When the chunk reaches 64 rows, phase 6 of K2 runs and S
// moves to the next chunk boundary.
//
// One CTA per (entry, head), 256 threads (K1's thread mapping, K2 on all 8 warps). Persistent state per (slot, head):
// S bf16 [128][128], the raw k, v, g rows (bf16 [64][128]) and beta (bf16 [64]) of the open chunk, pos.
#include "c64_params.h"
#include "c64_k1.cuh"
#include "c64_k2.cuh"

namespace c64 {

namespace {

constexpr float kLog2e = 1.4426950408889634f;
constexpr int kThreads = 256;

struct ReplaySmem {
    K1Smem k1;
    uint16_t v[CHUNK * RS];
    uint16_t S[D * RS];
};

__global__ void __launch_bounds__(kThreads, 1) replay_kernel(const ReplayParams p) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    ReplaySmem& sm = *reinterpret_cast<ReplaySmem*>(smem_raw);
    K1Smem& s = sm.k1;
    const int tid = threadIdx.x;
    const int entry = blockIdx.x, h = blockIdx.y, H = p.H;
    long long* tp = (p.prof != nullptr && entry == 0 && h == 0) ? p.prof : nullptr;
    if (tp && tid == 0) tp[8] = clock64();
    const int slot = p.slot_ids ? p.slot_ids[entry] : entry;
    if (slot < 0) return;
    if (slot >= p.num_slots) __trap();
    const bool peek = p.final_out != nullptr;
    const int count = peek ? 0 : (p.tok_count ? p.tok_count[entry] : p.tokens_per_entry);
    if (!peek && count <= 0) return;
    const int64_t tok0 = peek ? 0 : (p.tok_start ? p.tok_start[entry] : int64_t(entry) * p.tokens_per_entry);
    const int64_t out0 = p.out_row ? p.out_row[entry] : tok0;
    const int64_t sh = int64_t(slot) * H + h;
    int r0 = p.pos[sh];
    if (r0 < 0 || r0 >= CHUNK) __trap();

    uint16_t* g_state = p.state + sh * (D * D);
    uint16_t* c_k = p.rows_k + sh * (CHUNK * D);
    uint16_t* c_v = p.rows_v + sh * (CHUNK * D);
    uint16_t* c_g = p.rows_g + sh * (CHUNK * D);
    uint16_t* c_beta = p.rows_beta + sh * CHUNK;

    if (peek && r0 == 0) {           // nothing open: the closed state is S itself
        uint16_t* dst = p.final_out + (int64_t(entry) * H + h) * (D * D);
        for (int e = tid * 8; e < D * D; e += kThreads * 8)
            *reinterpret_cast<uint4*>(dst + e) = *reinterpret_cast<const uint4*>(g_state + e);
        return;
    }


    GateParams gp;
    gp.a_log_exp = ex2_ftz(fmul(p.A_log[h], kLog2e));
    gp.gate_scale = p.gate_scale;
    gp.scale_b = rbf(p.scale);

    int consumed = 0;
    bool state_dirty = false;
    // peek: one pass over the open rows 0..r0-1 as a short chunk that is closed
    while (peek ? (consumed == 0) : (consumed < count)) {
        const int take = peek ? 0 : min(CHUNK - r0, count - consumed);
        const int r_end = peek ? r0 : r0 + take;           // rows 0 .. r_end-1 of the open chunk are valid
        const bool close_chunk = peek ? true : (r_end == CHUNK);

        // ---- stage the chunk tile: rows < r0 from the row cache (q = 0), rows r0 .. r_end-1 from the new tokens, the
        //      remaining rows of the valid sub-blocks zero (asynchronous copies; rows past them are never read)
        const int nrows = ((r_end + 15) >> 4) * 16;
        for (int vec = tid; vec < nrows * 16; vec += kThreads) {
            const int row = vec >> 4, c8 = (vec & 15) * 8;
            const uint4 z = make_uint4(0u, 0u, 0u, 0u);
            if (row < r0) {
                cp_async16(&s.k[row * RS + c8], c_k + row * D + c8);
                cp_async16(&sm.v[row * RS + c8], c_v + row * D + c8);
                cp_async16(&s.gr[row * RS + c8], c_g + row * D + c8);
                st_smem_u4(&s.q[row * RS + c8], z);
            } else if (row < r_end) {
                const int64_t off = ((tok0 + consumed + (row - r0)) * H + h) * D + c8;
                cp_async16(&s.q[row * RS + c8], p.q + off);
                cp_async16(&s.k[row * RS + c8], p.k + off);
                cp_async16(&sm.v[row * RS + c8], p.v + off);
                cp_async16(&s.gr[row * RS + c8], p.g + off);
            } else {
                st_smem_u4(&s.q[row * RS + c8], z);
                st_smem_u4(&s.k[row * RS + c8], z);
                st_smem_u4(&sm.v[row * RS + c8], z);
                st_smem_u4(&s.gr[row * RS + c8], z);
            }
        }
        cp_async_commit();
        if (consumed == 0) {     // state -> smem: its own (newest) group, waited for only before K2
            for (int e = tid * 8; e < D * D; e += kThreads * 8) cp_async16(&sm.S[(e / D) * RS + (e % D)], g_state + e);
            cp_async_commit();
        }
        if (tid < CHUNK) {
            uint16_t b = 0;
            if (tid < r0) {
                b = c_beta[tid];
            } else if (tid < r_end) {
                b = p.beta[(tok0 + consumed + (tid - r0)) * H + h];
                if (!close_chunk) c_beta[tid] = b;
            }
            s.braw[tid] = b;
        }
        if (tid < D) s.dt[tid] = p.dt_bias[h * D + tid];
        if (consumed == 0) cp_async_wait<1>(); else cp_async_wait<0>();   // the tile (S may still be in flight)
        __syncthreads();
        if (!close_chunk) {      // the chunk stays open after this call: keep the new raw rows
            for (int vec = r0 * 16 + tid; vec < r_end * 16; vec += kThreads) {
                const int row = vec >> 4, c8 = (vec & 15) * 8;
                *reinterpret_cast<uint4*>(c_k + row * D + c8) = ld_smem_u4(&s.k[row * RS + c8]);
                *reinterpret_cast<uint4*>(c_v + row * D + c8) = ld_smem_u4(&sm.v[row * RS + c8]);
                *reinterpret_cast<uint4*>(c_g + row * D + c8) = ld_smem_u4(&s.gr[row * RS + c8]);
            }
            __syncthreads();     // k1_core's l2norm overwrites s.k (and its gate phase reads s.gr) in place
        }

        // ---- K1 arithmetic, the sink keeps Kd / Qd / Kr in K1's own buffers (in place)
        auto sink = [&](int row, int c8, uint4 kd, uint4 qd, uint4 kr) {
            st_smem_u4(&s.k[row * RS + c8], kd);
            st_smem_u4(&s.q[row * RS + c8], qd);
            if (close_chunk) st_smem_u4(&s.kr[row * RS + c8], kr);
        };
        if (tp && tid == 0) tp[9] = clock64();
        k1_core<kThreads>(s, gp, r_end, close_chunk, sink, tp);
        cp_async_wait<0>();
        __syncthreads();
        if (tp && tid == 0) tp[10] = clock64();

        // ---- K2 arithmetic on all 8 warps
        K2Stage st;
        st.kd = s.k; st.qd = s.q; st.kr = s.kr; st.v = sm.v; st.inv = s.inv; st.mq = s.mq; st.gt = s.gt; st.braw = s.braw; st.ldv = RS;
        auto emit = [&](int row, int vc, uint32_t pk) {
            if (row >= r0) {
                const int64_t orow = out0 + consumed + (row - r0);
                *reinterpret_cast<uint32_t*>(p.out + (orow * H + h) * D + vc) = pk;
            }
        };
        k2_step<8, 1>(st, sm.S, r_end, close_chunk, emit);
        __syncthreads();
        if (tp && tid == 0) tp[11] = clock64();

        state_dirty = state_dirty || close_chunk;
        consumed += (peek ? 1 : take);
        r0 = peek ? r0 : (close_chunk ? 0 : r_end);
        __syncthreads();
    }

    if (peek) {
        uint16_t* dst = p.final_out + (int64_t(entry) * H + h) * (D * D);
        for (int e = tid * 8; e < D * D; e += kThreads * 8)
            *reinterpret_cast<uint4*>(dst + e) = ld_smem_u4(&sm.S[(e / D) * RS + (e % D)]);
        return;
    }
    if (tid == 0) p.pos[sh] = r0;
    if (tp && tid == 0) tp[12] = clock64();
    if (state_dirty) {
        for (int e = tid * 8; e < D * D; e += kThreads * 8)
            *reinterpret_cast<uint4*>(g_state + e) = ld_smem_u4(&sm.S[(e / D) * RS + (e % D)]);
    }
}

}  // namespace

int replay64_smem_bytes() { return int(sizeof(ReplaySmem)); }

void launch_replay64(ReplayParams const& p, int num_entries, cudaStream_t stream) {
    if (num_entries <= 0 || p.H <= 0) return;
    cudaFuncSetAttribute(replay_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(ReplaySmem)));
    replay_kernel<<<dim3(num_entries, p.H), kThreads, sizeof(ReplaySmem), stream>>>(p);
}

}  // namespace c64
