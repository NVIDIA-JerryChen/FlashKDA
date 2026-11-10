# CF-BI KDA: bitwise train/infer-consistent kernels (v1, correctness-first)

`flash_kda.cfbi` (extension `flash_kda_cfbi_C`, sources in `csrc/cfbi/`) implements the "CF-BI" idea: ONE fixed-order,
chunk-form floating-point program that prefill, the training forward and decode all evaluate. The program itself
is specified in [20261102-cfbi-numerical-contract.md](20261102-cfbi-numerical-contract.md); this document describes
the kernels, how the pieces fit, what was verified and what it costs. Status: v1 was the correctness-first
implementation (validated on B300, `sm_103a`); **the kernels were optimised afterwards without changing a bit of their
results, see [20261103-cfbi-perf.md](20261103-cfbi-perf.md)** (sections 3 and 6-7 below describe the v1 structure and
numbers; the v1 kernels live on as the bitwise reference in `csrc/cfbi/ref/`). Added since: `slot_ids` state pools,
context-parallel prefill (`prefill_k1`/`prefill_k2`/`cp_prefill`). Section 7a lists what was measured on other
architectures (H100, A100, A10, L40S, ...; read it before relying on them: Ampere/Ada run only the row engine, and
Hopper's bits differ from Ampere's).

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
`cu_seqlens` every batch entry is one sequence). K = V = 128. The first call on a device runs
`flash_kda.cfbi.verify_hardware()` (about 0.3 s, cached per device and process): an empirical check of the tensor-core
premise of section 2 on that very device; the entry points raise if it fails. `FLASH_KDA_CFBI_SKIP_VERIFY=1` skips the
check, `FLASH_KDA_CFBI_ALLOW_UNVALIDATED_ARCH=1` runs it but only warns on failure. (It must not be the first cfbi call
inside a CUDA graph capture: call it once beforehand.)

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
| 16 | 40.2 / 39.0 | 5.4 | 18.2 |
| 64 | 106.5 / 105.5 | 16.7 | 52.5 |
| 128 | 173.4 / 172.4 | 33.2 | 91.6 |

Training step, B=1, L=8192, H=6: cfbi forward + (FLA forward recompute + FLA backward) 4.7 ms GPU busy vs FLA
forward + backward 1.35 ms; B=8, L=4096: 5.9 vs 2.5 ms.

Notes: eager FLA timings at these shapes are CPU-launch bound (82-101 us for the decode call even though its kernels
need 3-33 us), which is why GPU busy time is the comparison used here. For orientation, an earlier measurement of the
per-token-bf16-state `consistent_fwd` (branch `feature/kda-consistency`, same node type) gave 7.90 ms (1x4096) and
62.4 ms (1x32768) at H=32 against 2.89 / 23.1 ms here, i.e. 2.7x faster, and FLA's fp32-state recurrent kernel 3.83 / 30.1 ms.
The CF-BI report's own (optimised) kernels are far faster than v1 (e.g. H6 / L8K / batch 1: 0.54 ms on GB300), see 7.

### Is this the CF-BI report's setup? No - read the numbers accordingly

| | CF-BI report | measurements in this document |
|---|---|---|
| GPU | GB300 (some historical records on GB200); "148-SM historical device" | one NVIDIA B300 SXM6 AC (compute capability 10.3) on aws-pdx, driver 580.126.09 (the NGC 26.08 image asks for >= 615.65 and prints an error banner; torch/CUDA still work). SM count, clocks, power cap and HBM bandwidth were not recorded, clocks were not locked |
| parallelism | end to end: 8 GPUs on 2 nodes, TP8 + EP8; micro-benchmarks: one GPU with the per-GPU head count after TP (H6 / H12) | one GPU only; H is set directly (6, 12, 32). No TP/EP/collectives: the head-sharding test only checks that per-head bits do not depend on how heads are grouped |
| workload | end-to-end serving of a full model (8K in / 1K out, concurrency 1-64, prefix cache, 64 sequences, 8192-token step budget) plus attention-core micro-benchmarks | no model, no serving stack: random tensors ("authors" regime), KDA core only (no conv, projections, gated norm, communication) |
| kernels | optimised: tcgen05 UMMA, TMEM, bulk copy + mbarrier pipelines, warp specialisation, Decode A/B | v1: mma.sync TF32, global fp32 workspace, whole-head CTA, no pipelining |
| numerics | the CF-BI program; elementwise approximations, l2norm order and gate form are not published | same program family (chunk 64, D 128, fp32 E, 16-row sub-blocks, reference rows, three-term TF32, same state and workspace sizes) with the unpublished details fixed by this repo's contract: not bit-identical to the report's implementation |
| shapes | prefill H6 / L8192 / batch 1, 8, 64; decode N 16..128 / H12 | the same shapes (plus H32 and N=1) |
| timing | "core latency" with an unspecified protocol (decode: average chunk position, plus a separate cold-state probe) | prefill: median of 10 event-timed calls and profiler GPU-busy time; decode: back-to-back CUDA-graph replays over consecutive steps (all 64 chunk positions averaged, cache state not controlled) |

Consequence: compare cfbi with `flash_kda.fwd` / FLA measured in the same run (same GPU, same harness); the report's
absolute numbers are context only. Nothing here measures the end-to-end throughput loss the report quantifies, and with
v1 kernels that are roughly 4-7x slower in prefill (3.70 / 7.42 / 55.7 ms here against the report's GB300 records of
0.536 / 1.98 / 12.7 ms for batch 1 / 8 / 64) and ~2x in decode (N=64: 105 us against 52.5 us) than the report's optimised
ones (different hardware, so indicative only) that loss cannot be extrapolated from these numbers.

## 7. Limits, and what v1 does not do

- **Performance** (this is the v1 state; the v2 kernels, with bulk-copy + mbarrier pipelines, rewritten K1 / K2 / row engine,
  a tcgen05 K2 experiment and the measured numbers, are in `docs/20261103-cfbi-perf.md`): mma.sync TF32 only, no
  tcgen05/UMMA/TMEM, no bulk-copy pipelines, no warp specialisation. K2 is a
  latency-bound serial scan (one CTA per (V-slice, sequence, head), ~25 us per chunk) which dominates small batches; K1
  runs one CTA per SM (218 KiB smem) with dependent phases; the row engine takes ~100 us at N=64 / H=12 (the report's
  Decode A/B: 67 / 52 us) because each row walks ~10 dependent phases with barriers. These are engineering, not
  numerics, items: the per-element arithmetic is pinned by the contract.
- **Backward** is not bit-pinned and depends on `flash-linear-attention` (lazy import in `_bwd/_fla.py`): it recomputes the
  forward with FLA's Triton kernels under autograd (one extra forward-sized pass per layer). Vendoring FLA's chunk backward
  was the plan, but the symbol-level import closure measured ~10k lines in 28 files (CP code, `gla/chunk.py`,
  `fla.utils`/`fla.backends`), so the plan's fallback is used. Constraints: K=V=128, `lower_bound` in `[-5, 0)` for
  gradients, empty initial state (no `dh0`), no context parallelism.
- **Scope**: validated on B300, and on the other architectures of section 7a to the extent stated there (the guard is
  now the empirical `verify_hardware()`, not a capability test). Context parallelism (`prefill_k1` / `prefill_k2` /
  `cp_prefill`) and the `slot_ids` state-pool indirection exist now (`docs/20261103-cfbi-perf.md`). The layers around the core
  are aligned in a separate extension: the pinned short causal conv + SiLU and the batch-invariant bf16 projection
  (`flash_kda.cfbi.short_conv` / `bi_linear`, `docs/20261103-cfbi-conv-linear.md`; validated on B300 only); the gated RMSNorm
  and communication are not covered. Per-call workspace as in section 3; token count < 2^31, `N`, `H` < 65536.
- **Unmeasured**: the effect on model quality, and the gate/beta distributions of real checkpoints (all data here is
  synthetic).
- **Bit equality is relative to one GPU architecture and one compiler build**: nothing here promises equality across
  architectures, nvcc versions or other KDA implementations. Measured, not assumed (section 7a): Hopper and
  Ampere/Ada give different bits for the same inputs (the tensor-core product differs, the `ex2/rcp/rsqrt.approx`
  instructions do not).

## 7a. Other GPU architectures (measured 2026-11-03)

v1 was written and validated on one B300. This section records what was measured on other GPUs afterwards and what had to
change for it. Setup: cluster `sc`, one GPU per run, the same NGC `pytorch:26.08-py3` container on every machine (nvcc 13.4,
torch 2.14, driver 615.71), JIT build of the extension for the device's own arch (`sm_90a`, `sm_100a`, ...). Raw logs and
the node scripts are in `/home/scratch.jennym_gpu/consist-kda/arch-test/` (outside the repo); `tests/cfbi/arch_dump.py` /
`arch_compare.py` are in it. The per-GPU `.pt` dumps were deleted after the comparison (the SHA-256 that every run printed is
kept in those logs; `arch_dump.py` regenerates a dump). The B300 ran on aws-pdx (driver 580.126.09, same image), all others on `sc`.

**Build.** Before this change the extension did not compile for sm_80: ptxas rejects `cp.async.bulk`,
`mbarrier.arrive.expect_tx`, `mbarrier.try_wait`, `fence.proxy.async` and the cluster-scope mbarrier fence below sm_90 (all of
them live in `cfbi_async.cuh` and are used only by the optimised K2). They now compile to a trap for `__CUDA_ARCH__ < 900`
and the launcher never selects such a kernel there. One nvcc run builds all of sm_80 / 86 / 89 / 90a / 100a / 103a / 120a
(`FLASH_KDA_CFBI_ARCH="8.0;8.6;8.9;9.0a;10.0a;10.3a;12.0a"`, 51 s on a 32-core node; `FLASH_KDA_CUDA_ARCHS=all` of `setup.py`
now includes 80/86/89 for this extension; flags unchanged: `-fmad=false`, no fast math). The arithmetic is untouched: `cuobjdump -sass` of all
11 kernels (probes, plan, row engines, K1, K2, reference and optimised) is instruction-for-instruction identical to the base
commit 3199764 for sm_90a, sm_100a and sm_103a (40,591 / 35,279 / 35,279 SASS lines); the H100 digest is the same before and
after (7078397d...); sm_90 and sm_90a builds give the same bits. One binary
built once for `8.0;8.6;8.9;9.0a;10.0a;10.3a` (a `;`-separated `FLASH_KDA_CFBI_ARCH`, or the `setup.py` build
`FLASH_KDA_BUILD_EXTS=cfbi FLASH_KDA_CUDA_ARCHS=all`: 23 MB, SASS for 80/86/89/90a/100a/103a/120a) loads on the H100, A100, A10,
L40S and B200 (the `setup.py` module: H100, A100, B200, B300), passes `test_arch` + `test_mma_probe` there (41 tests) and reproduces
each device's own-build dump bit for bit; the launcher picks the kernels per device at run time.

**Which kernels run where** (`resolve_impl` in `csrc/cfbi/cfbi_launch.cu`, reported by `verify_hardware()["kernels"]`; an
explicit `_impl` is honoured where possible):

| device class | shared memory opt-in per block | what runs |
|---|---|---|
| sm_90a, sm_100a, sm_103a | 227 KiB | optimised K1/K2/row engine (`_impl=0`), reference kernels (`_impl=3`), row engine only (`_impl=7`) |
| sm_80 | 163 KiB | reference **row engine for every row** (no K1/K2) |
| sm_86, sm_89 (and sm_120, not run) | 99 KiB | same |

On sm_100a / sm_103a the default implementation also chooses between two K2 launches for a variable-length prefill (`cu_seqlens`): the
old K2 kernels, or one driven by a schedule that a builder kernel makes on the device from the sequence lengths (`_impl` bit 4). Calls
that are predicted to gain get the second one (`k2_dyn_default_route` in `csrc/cfbi/cfbi_launch.cu`, rule and measurements in section 7 of the
performance document); both compute the same bits, and `_impl=32` keeps the old kernels where the default would route.

K1 needs 204 KiB (optimised) / 218 KiB (reference) of dynamic shared memory per CTA, so neither fits an Ampere/Ada SM: prefill
and the training forward there run `mode 1` (every row through the row engine, `_impl` bit 2), which evaluates the same program
row by row (bit-identical to K1/K2 on every device where both can run, see `test_vs_ref`, `test_row_engine_equals_chunk_kernels`,
`test_arch.py::test_every_impl_value_is_accepted_and_same_bits`). The stateless training forward allocates a zeroed temporary
state, N x H x 224 KiB. Cost, B=1 / T=4096 / H=6 prefill (event time; B300 chunk kernels: 3.7 ms at L=8192): H100 optimised kernels 0.34 ms, reference chunk kernels 1.90 ms,
row engine only 115 ms (60x / 340x slower); B200 0.30 / 1.63 / 111 ms; B300 0.30 / 1.59 / 108 ms; row engine only on Ampere/Ada: A100 155 ms, A10 144 ms,
L40S 106 ms (26-38 us per token, all heads in parallel). Decode, N=16 / H=12, one token: 34 us optimised and 49 us reference on
the H100, 61 / 124 / 41 us on A100 / A10 / L40S (the row engine is the decode kernel anyway). This is a correctness fallback for
prefill and the training forward, not a performance path.

**Premise and schedule independence.** "Validated" below means: the premise probe and the schedule matrix pass on that device with
their negative controls intact (`test_negative_control_k_order_matters` and the battery's own control: reversed K order differs in 91-96 % of the tiles;
`test_negative_control_is_caught`: a sabotaged fma is caught). Counts are passed tests per file (default `FLASH_KDA_CFBI_IMPL=0`);
every row except B300 also ran the BIG variant (`CFBI_BIG=1`: 32K tokens whole / random cuts / 32805 decode calls, N=512; 28 matrix tests).

| GPU | sm | kernels | probe (`test_mma_probe` 27, `test_prep_probe` 13) | `verify_hardware()` | `test_kernels` 68 + `test_bitwise_matrix` 25 (+3 BIG) + `test_arch` 14 | `test_vs_ref` 71 |
|---|---|---|---|---|---|---|
| B300 SXM6 AC (aws-pdx, driver 580; also 147 tests on 2026-11-02) | 103a | optimised / reference / rows-only | 27 + 13 pass | ok, 0 / 12288, 0.26 s | pass (default = optimised; dumps with `_impl` 3 and 7 bit-identical to it) | 71 pass |
| B200 | 100a | optimised / reference / rows-only | 27 + 13 pass | ok, 0 / 12288, 0.27 s | pass (default = optimised; dumps with `_impl` 3 and 7 bit-identical to it) | 71 pass |
| H100 80GB HBM3 | 90a | optimised / reference / rows-only (all three run the full matrix) | 27 + 13 pass | ok, 0 / 12288, 0.18 s | pass x 3 implementations | 71 pass |
| A100 80GB PCIe | 80 | reference row engine | 27 + 13 pass | ok, 0 / 12288, 0.26 s | pass (row engine only) | 71 skipped: the optimised kernels cannot run |
| A10 | 86 | reference row engine | 27 + 13 pass | ok, 0 / 12288, 0.21 s | pass (row engine only) | 71 skipped |
| L40S | 89 | reference row engine | 27 + 13 pass | ok, 0 / 12288, 0.19 s | pass (row engine only) | 71 skipped |

`verify_hardware()`: 12288 placements (4 data regimes x K in {24, 128} x 16 row slots x 8 column slots x 3 neighbour patterns x 4
pairs), 0 mismatches on every device, `dot3` within 2.3-2.7e-7 of `sum|a*b|`, about 0.2-0.3 s. The battery's negative control
(reversed K) differs in 91-96 % of the tiles; the premise itself is checked on every device that was run, a fake probe that
violates it makes `_guard` raise (`test_arch.py`).

**Cross-architecture bits** (`arch_dump.py` + `arch_compare.py`; the same inputs on every machine - they are generated once on
the CPU, because `torch.randn` on the GPU is *not* bit-identical across GPU models: the L40S and the H100 (and the A10 and the
A100) produce different q/k/v/g from the same seed, so `digest_run.py` digests are comparable within one GPU model only; 3 data regimes, 6 sequences x
4 heads, three prefill calls with cuts and the training forward, plus raw probes of `dot3`, `ex2`, `rcp`, `rsqrt`):

- sm_80 (A100), sm_86 (A10), sm_89 (L40S): **bit-identical on all 31 dumped tensors** (SHA-256 over all results fba39291...).
- sm_90a (H100) vs sm_80: `ex2.approx.ftz`, `rcp.approx.ftz`, `rsqrt.approx.ftz` are bit-identical (2^21 strided samples each);
  the three-term `dot3` is not: 79 % of the products differ (max 2.6e-7 of `sum|a*b|`, i.e. a few fp32 ulp: the tensor-core
  accumulation is not the same hardware behaviour). Propagated through the program, out of 1,039,872 outputs: **184 bf16 outputs
  differ (0.018 %, at most 9 bf16 ulp), 880,424 fp32 outputs (84.7 %)**, 361,441 of 1,179,648 state entries `E` (30.6 %), 181,368 of 2,949,120
  cache entries; the training forward the same (it equals prefill on every device).
- H100: optimised == reference == row-engine-only, bit-identical on all dumped tensors; sm_90 (no `a`) == sm_90a.
- Blackwell: sm_100a (B200) and sm_103a (B300) vs sm_90a (H100): **bit-identical on all 31 dumped tensors** (SHA-256 332f1fef...,
  for `_impl` 0, 3 and 7 alike). Hopper and datacenter Blackwell agree.
- `digest_run.py` (GPU-generated inputs): 7078397d... on B300 (2026-11-02 and again today with this branch), H100 and B200 (all `_impl`),
  which can only happen if inputs and outputs of that workload are bit-identical on the three; 0b8ccdce... on A100 and L40S and
  af13549c... on A10 (different inputs, so these do not compare architectures).

So the contract's "one architecture + one build" is the right unit: Ampere/Ada agree among themselves, Hopper and Blackwell agree bit for bit (H100 == B200 == B300),
and the two families differ in the last bits of every product. A model trained on one family and served on the other gets the same
program but not the same bits (0.02 % of the bf16 outputs differ in this workload).

**Not validated**
- K1/K2 (the chunk kernels) on sm_80 / 86 / 89 - they cannot run there (shared memory), so "row engine == chunk kernels" and the
  training-forward-vs-prefill equality are only tested on devices where K1/K2 run; on Ampere/Ada both sides are the row engine. What
  carries the claim over is the premise probe (every row slot and neighbour pattern gives the same bits) - an inference, not a test of K1/K2.
- The optimised kernels on sm < 90 (compiled as traps, never selected); sm_120 (compiles, no device); other drivers / CUDA
  versions / compiler builds (one image everywhere); Turing and older (no TF32 mma: `verify_hardware()` rejects the device
  without building anything); multi-GPU hosts with mixed architectures (one device at a time was used); the backward pass
  anywhere but on the A100 (`test_backward.py`: 9 passed there after 10 min of Triton compile; row-engine-only forward, FLA's
  backward).
- Performance of anything but the fallback above (no tuning was done for Hopper or Ada: the optimised kernels are B300-tuned).

## 8. Reproduce

```bash
FLASH_KDA_BUILD_EXTS=cfbi FLASH_KDA_CUDA_ARCHS=103a pip install -v --no-build-isolation .   # or let the tests JIT-build it
                                                  # (one wheel for several archs: FLASH_KDA_CUDA_ARCHS=80,86,89,90a,100a,103a or =all)
python -m pytest tests/cfbi -q                    # probes, accuracy, bitwise matrix, backward (needs flash-linear-attention)
CFBI_BIG=1 python -m pytest tests/cfbi -q         # + 32K-token runs and N=512
python benchmarks/accuracy_cfbi.py                # section 4
python benchmarks/bench_cfbi.py                   # section 6 (needs flash_kda_C and FLA for the baselines)
python benchmarks/bench_cfbi_arch.py              # what the Ampere/Ada row-engine fallback costs on this device
FLASH_KDA_CFBI_IMPL=3 python -m pytest tests/cfbi -q   # the whole suite on the reference kernels (7: row engine only)
python tests/cfbi/arch_dump.py OUT.pt && python tests/cfbi/arch_compare.py A.pt B.pt   # cross-architecture bit comparison (7a)
```
