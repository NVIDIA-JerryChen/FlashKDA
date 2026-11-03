# CF-BI KDA: bitwise train/infer-consistent kernels (v1, correctness-first)

`flash_kda.cfbi` (extension `flash_kda_cfbi_C`, sources in `csrc/cfbi/`) implements the "CF-BI" idea: ONE fixed-order,
chunk-form floating-point program that prefill, the training forward and decode all evaluate. The program itself
is specified in [20261102-cfbi-numerical-contract.md](20261102-cfbi-numerical-contract.md); this document describes
the kernels, how the pieces fit, what was verified and what it costs. Status: v1, validated on B300 (`sm_103a`)
only, performance engineering (tcgen05/UMMA, TMEM, bulk-copy pipelines, decode double-CTA residency) is explicitly
not part of it.

## 1. What you get

For one sequence, the **bits of the output and of the continuation state are the same** whether it is processed as

- one prefill call, any number of prefill calls cut anywhere (also at non-multiples of 64), T=1 decode calls, or any mix,
- alone or inside any batch (neighbours, order, idle slots), with any head grouping (TP sharding), or with CUDA graphs,
- the training forward (`forward_train` / `chunk_kda`) instead of prefill/decode.

```python
from flash_kda.cfbi import alloc_state, prefill, decode, forward_train, chunk_kda

state = alloc_state(N, H)                         # CfbiState(E fp32 [N,H,128,128], rows fp32 [N,H,5,64,128], pos int32 [N])
prefill(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, state, cu_seqlens)   # [B,T,H,128] bf16 in/out, in place
decode(q1, k1, v1, g1, beta1, scale, out1, A_log, dt_bias, lower_bound, state)          # [N,1,H,128] or [N,H,128]
y = forward_train(q, k, v, g, beta, scale, A_log, dt_bias, lower_bound, cu_seqlens)     # empty state, nothing persisted
y = chunk_kda(q, k, v, g, beta, scale, A_log, dt_bias, lower_bound, cu_seqlens)         # same forward, autograd backward
```

Argument names, dtypes and shapes follow `flash_kda.fwd` (`q,k,v,g` bf16 `[B,T,H,128]`, `beta` bf16 raw logits,
`A_log` fp32 `[H]`, `dt_bias` fp32 `[H,128]`, `lower_bound` in `[-5,0]`, `cu_seqlens` int64 with `B == 1`; without
`cu_seqlens` every batch entry is one sequence). K = V = 128. `FLASH_KDA_CFBI_ALLOW_UNVALIDATED_ARCH=1` is needed on
devices below compute capability 10.0.

## 2. The idea in one page

A KDA sequence is cut into chunks of 64 rows **anchored at the sequence start**. Inside a chunk the entry state `E`
is frozen; every row `t` is computed from `E`, the earlier rows of the same chunk (the "open-chunk row cache") and its
own inputs, with a fixed operation order (sub-blocks of 16 rows, reference rows for the exponentials, three-term TF32
tensor-core products over K in ascending steps of 8, sequential FMA chains inside a sub-block). The 64th row triggers a
fixed state update `E -> E'`.

```
rows:    0 .......... 63 | 64 .......... 127 | 128 ... 
chunks:  [   chunk 0    ] [    chunk 1     ] [ chunk 2 (open, pos = 20) ]
prefill: K1 (all chunks in parallel: E-independent part)  ->  K2 (serial over chunks: E-dependent part)
decode:  row engine: append ONE row to the open chunk (E and the 20 cached rows -> row 20), pos = 21
```

A call that arrives with `pos = 20` and 100 new rows is split on the grid line: the row engine appends the 44 rows that
finish the open chunk (and performs the chunk-end update), K1/K2 process the remaining 56 rows as a partial chunk whose
rows go into the cache (`pos = 56`). Because every row is the same arithmetic in every schedule, the cut does not matter.

Why the bits agree: (1) all per-token numbers come from one device-function set (`cfbi_prep.cuh`); (2) every
tensor-core product is the same `dot3<>` template with one tile geometry (row of a sub-block in A-row slot `t%16`,
column in slot `n%8`, invalid slots zero, K ascending), K1/K2 fill up to 16 row slots, the row engine fills one;
(3) the tensor-core output element must not depend on the other rows/columns of the tile - the **hardware premise**,
checked by `tests/cfbi/test_mma_probe.py` (it holds on B300, see section 5); (4) all other arithmetic is explicit
IEEE intrinsics, the extension is built without `--use_fast_math` and with `-fmad=false`.

## 3. Kernels and data flow

| kernel | grid x block | smem / regs | work |
|---|---|---|---|
| `plan_kernel` | 1 x 1024 | 4 KiB / 64 | per sequence: `tok0, len, pos0`, head rows `h = 0 if pos0 == 0 else min(len, 64 - pos0)`, `nfull = (len-h)/64`, `tail = (len-h)%64`, exclusive prefix sum of the chunk counts (CUB). No host sync |
| `rows_kernel` (row engine = decode) | (N, H) x 256 | 7 KiB / 48 | loops over the head rows of one (sequence, head): prep, Akk/Aqk row vs cached rows, UT/W row, `P1/P2 = dot3(W/GQ, E^T)`, `U`, `O`, append to the cache; at row 63 the chunk-end update of `E` |
| `k1_kernel` | (max chunks, H) x 256 | 218 KiB / 72 | E-independent part of a chunk: prep, cumsum, Akk/Aqk (`dot3` over K=128), block forward substitution for UT/W, then the workspace (WG, UT, Aqk, KD, GEND); a trailing partial chunk also stores khat, G2, UT, W of its rows in the cache |
| `k2_kernel` | (8, N, H) x 256 | 26 KiB / 54 | serial scan of a sequence's chunks with a `[16 x 128]` E slice in smem: P1/P2, `U = UT - P1`, `O` (cross-sub-block `dot3` + FMA chain, bf16 store), chunk-end update; a partial chunk stores its `U` rows in the cache |
| `finalize_pos_kernel` | N | - | `pos = (pos0 + len) % 64` (`pos` is per sequence but read by every head's CTA, so it is only advanced at the very end) |

- **Workspace** (K1 -> K2, fp32, per chunk and head, 144.5 KiB): `WG=[W;GQ]` 64 KiB, `UT` 32, `Aqk` 16, `KD` 32,
  `GEND` 0.5. One call allocates `(ceil(T/64) + N) * H * 144.5 KiB`; split very long calls yourself (any cut is bit-exact).
- **State** per (sequence, head): `E` 64 KiB + row cache 160 KiB (five fp32 `[64][128]` planes: `khat, G2, UT, W, U`).
- **Mixed batches**: one call can contain decode rows and long prefills; sequences with `pos0 > 0` start with the row
  engine, the others go straight to K1/K2. Zero-length sequences are allowed. All grids are upper bounds with early
  exit and all lookups happen on the device, so the whole call is CUDA-graph safe.
- **Training forward** is the same pipeline from the empty state (`E` slices start at zero, no cache or state writes).
- `mode=1` of the extension entry point sends every row through the row engine; the tests use it as a differential
  test of K1/K2 (one call as chunks == the same rows appended one by one).
- Deviations from the CF-BI report that do not change bits: `mma.sync` everywhere, global fp32 workspace, no warp
  specialisation, whole-head CTA for decode. One deliberate change of the numerics, see the contract doc: the
  sigmoid is `rcp(1 + ex2(-x log2 e))` (master's `tanh.approx` form has a 1.1e-5 max relative error on B300).

## 4. Accuracy

Reference: the per-token recurrence in fp64 (`reference.exact_recurrence`), T = 2048, H = 4 (`benchmarks/accuracy_cfbi.py`).
"flip" = share of bf16 outputs that differ from the bf16-rounded exact output; the bf16 floor is the error of that rounding.

| regime | cfbi fp32 (pre-cast) | cfbi bf16 / floor | flip | FLA chunk_kda | flip | flash_kda.fwd | flip |
|---|---|---|---|---|---|---|---|
| authors | 9.2e-7 | 1.66e-3 / 1.66e-3 | 0.10% | 3.35e-3 | 47.8% | 5.20e-3 | 69.8% |
| unit | 8.9e-7 | 1.66e-3 / 1.66e-3 | 0.09% | 4.10e-3 | 61.5% | 5.70e-3 | 72.1% |
| heavy tail | 1.5e-6 | 1.62e-3 / 1.62e-3 | 0.11% | 3.65e-3 | 57.0% | 5.58e-3 | 68.6% |
| long memory | 1.1e-6 | 1.66e-3 / 1.66e-3 | 0.09% | 5.28e-3 | 69.3% | 7.79e-3 | 78.0% |
| no decay (lb = 0) | 1.0e-6 | 1.66e-3 / 1.66e-3 | 0.09% | n/a | n/a | 7.80e-3 | 78.0% |
| eps-dominated norms | 8.9e-7 | 1.66e-3 / 1.66e-3 | 0.10% | 3.74e-3 | 58.7% | 5.53e-3 | 71.3% |
| saturated gates | 6.7e-7 | 1.66e-3 / 1.66e-3 | 0.04% | 3.40e-3 | 56.7% | 4.91e-3 | 68.1% |
| one-hot keys | 4.5e-7 | 1.67e-3 / 1.67e-3 | 0.09% | 2.42e-3 | 7.7% | 4.66e-3 | 10.9% |

The program's bf16 output sits on the bf16 floor. The contract executed with exact transcendentals and an idealised
mma has 6.7e-7 (`tests/cfbi/test_reference.py`); the kernels add ~2e-7 on top (ex2/rcp/rsqrt approximations, real
tensor-core accumulation). Before switching the sigmoid from `tanh.approx` to ex2+rcp the kernels were at 2.4e-6 and
the flip rate at 0.28% (0.74% for long memory). FLA and flash_kda are different programs (bf16 intermediates / bf16
state), their larger distance to the exact math is not an implementation defect.
Training-time gradients (FLA backward on a recomputed forward) are at ~3e-3 relative RMS against an fp64 autograd of
the exact recurrence (the bf16 floor), FLA's own test ratios are 0.008 (dq/dk/dv) and 0.02 (dg/dbeta).

## 5. Verification (all on a B300, aws-pdx, NGC pytorch 26.08)

Everything below is zero-tolerance on bits unless a tolerance is named. "Bits" = bf16 outputs, the fp32 outputs before
the bf16 rounding, `E`, the valid open-chunk rows and `pos`. Comparing only the bf16 outputs is a *weak* detector: the
negative control below changes 479 of 51200 fp32 outputs and 101 `E` entries (plus 30 cache rows) but not one bf16 output.

| gate | test | result |
|---|---|---|
| hardware premise | `test_mma_probe.py`: same (row, column) vectors in every of 16 row slots x 8 column slots x {zero, random, wide-exponent, NaN, Inf} neighbours, K = 8..128, 5 data regimes | identical bits everywhere; reversed K order differs in >5% of tiles (the test is sensitive); `dot3` max error 3e-7 of `sum|a*b|` |
| prep | `test_prep_probe.py`: l2norm / gate / beta against a PyTorch emulation (the PTX approximations taken from the extension) | bit-exact, 13 cases |
| math | `test_kernels.py`: fp32 output vs the fp64 recurrence, 8 regimes, T = 64/200 | 0.56e-6 .. 1.5e-6 rel RMS |
| row engine == chunk kernels | `mode=0` vs `mode=1`, T = 1, 15, 16, 17, 63, 64, 65, 127, 128, 200, 300, 4 regimes | identical |
| decode / cuts | whole call vs T=1 x n decode vs random cuts vs *every* single cut point, 8 regimes | identical |
| batches | varlen vs each sequence alone (N=64; N=512 with `CFBI_BIG=1`), permuted order, continuous-batching simulation with idle slots and mixed decode/prefill rows, head sharding 6 -> {3+3, 1x6, 2+4} | identical |
| runtime | CUDA-graph replay (prefill with warm state, decode over 40 steps), repeat runs, two separate processes (SHA-256 of everything) | identical |
| long runs (`CFBI_BIG=1`) | 32768+37 tokens: whole vs random cuts vs 32805 decode calls, 3 regimes | identical |
| builds | the same workload through the JIT-built extension and the `setup.py` build (`FLASH_KDA_BUILD_EXTS=cfbi`, different nvcc command line), and the 118 core tests against the installed module | identical SHA-256 digest, tests pass |
| training forward | `forward_train` == `prefill` (also varlen, also through `chunk_kda`) | identical |
| negative control | extension built with one row-engine `fma` replaced by `mul`+`sub` | caught (fp32 outputs, `E` and cache rows differ; bf16 outputs do not) |
| backward | `test_backward.py`: gradients vs fp64 autograd, lower_bound -5 / -2.5, varlen, B = 2 | ~3e-3 rel RMS (FLA ratios 0.008/0.02) |
| regression | master's `tests/test_fwd.py` with `flash_kda_C` built from this branch | "Assert results: Success" |

## 6. Speed of v1 (B300, GPU busy time = sum of kernel durations, ms; `benchmarks/bench_cfbi.py`)

| shape | cfbi prefill | flash_kda.fwd | FLA chunk_kda | note |
|---|---|---|---|---|
| B=1, L=8192, H=6 | 3.70 | 0.74 | 0.39 | K2 86% (serial scan, 48 CTAs) |
| B=8, L=8192, H=6 | 7.42 | 0.88 | 1.18 | |
| B=64, L=8192, H=6 | 55.7 | 2.85 | 8.06 | K1 50% / K2 50% |
| B=1, L=4096, H=32 | 2.89 | 0.42 | 0.45 | |
| B=1, L=32768, H=32 | 23.1 | 3.30 | 3.27 | |

`forward_train` is within 0.5% of `prefill`. Decode, H = 12, one token per sequence (us per call, CUDA graph / GPU busy):

| N | cfbi decode | FLA fused_recurrent (fp32 state) | CF-BI report (Decode B, GB300) |
|---|---|---|---|
| 1 | 34.9 / 33.3 | 2.6 (GPU busy) | |
| 16 | 40.2 / 39.0 | 5.4 | |
| 64 | 106.5 / 105.5 | 16.7 | 52.5 |
| 128 | 173.4 / 172.4 | 33.2 | 91.6 |

Training step, B=1, L=8192, H=6: cfbi forward + (FLA forward recompute + FLA backward) 4.7 ms GPU busy vs FLA
forward + backward 1.35 ms; B=8, L=4096: 5.9 vs 2.5 ms.

Notes: eager FLA timings at these shapes are CPU-launch bound (82-101 us for the decode call even though its kernels
need 3-33 us), which is why GPU busy time is the comparison used here. For orientation, an earlier measurement of the
per-token-bf16-state `consistent_fwd` (branch `feature/kda-consistency`, same node type) gave 7.90 ms (1x4096) and
62.4 ms (1x32768) at H=32 against 2.89 / 23.1 ms here, i.e. 2.7x faster, and FLA's fp32-state recurrent kernel 3.83 / 30.1 ms.
The CF-BI report's own (optimised) kernels are far faster than v1 (e.g. H6 / L8K / batch 1: 0.54 ms on GB300), see 7.

## 7. Limits, and what v1 does not do

- **Performance**: mma.sync TF32 only, no tcgen05/UMMA/TMEM, no bulk-copy pipelines, no warp specialisation. K2 is a
  latency-bound serial scan (one CTA per (V-slice, sequence, head), ~25 us per chunk) which dominates small batches; K1
  runs one CTA per SM (218 KiB smem) with dependent phases; the row engine takes ~100 us at N=64 / H=12 (the report's
  Decode A/B: 67 / 52 us) because each row walks ~10 dependent phases with barriers. These are engineering, not
  numerics, items: the per-element arithmetic is pinned by the contract.
- **Backward** is not bit-pinned and depends on `flash-linear-attention` (lazy import in `_bwd/_fla.py`): it recomputes the
  forward with FLA's Triton kernels under autograd (one extra forward-sized pass per layer). Vendoring FLA's chunk backward
  was the plan, but the symbol-level import closure measured ~10k lines in 28 files (CP code, `gla/chunk.py`,
  `fla.utils`/`fla.backends`), so the plan's fallback is used. Constraints: K=V=128, `lower_bound` in `[-5, 0)` for
  gradients, empty initial state (no `dh0`), no context parallelism.
- **Scope**: validated on B300 only (the guard needs compute capability >= 10 unless overridden); no context parallelism,
  no conv/projection/gated-norm alignment (only the KDA core is covered); no `slot_ids` state-pool indirection (gather the
  active sequences' state yourself, as the scheduler test does); per-call workspace as in section 3; token count < 2^31,
  `N`, `H` < 65536.
- **Unmeasured**: the effect on model quality, and the gate/beta distributions of real checkpoints (all data here is
  synthetic).
- **Bit equality is relative to one GPU architecture and one compiler build**: nothing here promises equality across
  architectures, nvcc versions or other KDA implementations.

## 8. Reproduce

```bash
FLASH_KDA_BUILD_EXTS=cfbi FLASH_KDA_CUDA_ARCHS=103a pip install -v --no-build-isolation .   # or let the tests JIT-build it
python -m pytest tests/cfbi -q                    # probes, accuracy, bitwise matrix, backward (needs flash-linear-attention)
CFBI_BIG=1 python -m pytest tests/cfbi -q         # + 32K-token runs and N=512
python benchmarks/accuracy_cfbi.py                # section 4
python benchmarks/bench_cfbi.py                   # section 6 (needs flash_kda_C and FLA for the baselines)
```
