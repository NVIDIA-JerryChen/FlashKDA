# Bitwise-aligned decode for the unmodified FlashKDA prefill (open-chunk replay)

*2026-11-24, branch `feature/kda-replay-decode` (from master d7e3b3a)*

## 1. What this adds

`flash_kda.replay.prefill` / `decode` / `final_state`: a serving path whose outputs are **bit-for-bit equal to one
`flash_kda.fwd` call over the whole sequence**, however the sequence is split into calls (prompt, chunked-prefill
pieces at any offset, one or several decode tokens per step, any batch composition, slot pools).

The prefill kernel is not changed: `csrc/smxx/*`, `csrc/fwd.h` and the `fwd()` entry point are byte-identical to
master, and the 16-aligned part of every prefill call still runs through `flash_kda.fwd`. What is new is a decode
kernel (`csrc/replay/`) that replays `flash_kda.fwd`'s own chunk arithmetic for the rows of a chunk that is still open.
This is the CF-BI idea ("decode = one row of the chunk program") applied to the existing chunk-16 kernel instead of a
new numerical contract.

## 2. Why the decode can match the prefill

`flash_kda.fwd` processes a sequence in chunks of 16 tokens, anchored at the sequence's first token:

```
tokens      0 ........ 15 | 16 ........ 31 | 32 .. 36
chunk            c0       |       c1       |  c2 (open)
state       S0 -------> S16 -------------> S32          S = bf16, rounded once per chunk end (K2 phase 6)
```

Inside a chunk, the output of row r reads only

* rows 0..r of k, v, g, beta and row r of q (the gate cumsum is a sequential fold per column; L, Mqk and (I+L)^-1 are
  lower triangular, their upper parts are exact +0 assignments, so later rows only ever meet a +0 coefficient), and
* the bf16 state at the chunk start (S32 above). The chunk-end quantities (`g_total`, `k_restored`) feed only the state
  update, which runs after the outputs are written.

Hence row 34 of one `fwd` call over 0..36 (or over 0..1000) equals row 2 of a `fwd` call over tokens 32..34 that
starts from S32. A decode step for the token at chunk row r therefore re-runs the chunk arithmetic on rows 0..r of the
open chunk (rows r+1..15 zero, exactly like a short last chunk) from S32 and emits row r. When the chunk reaches 16
rows, the K2 state update runs as in the prefill and S48 becomes the new chunk-start state.

The earlier objection "only changing decode is not enough" had three parts; this is how each is handled:

| condition | how it is met here |
|---|---|
| prefill must not close the prompt's partial last chunk (its state is rounded early and the tail rows are lost) | the state keeps S at the last 16-token boundary plus the raw rows of the open chunk; `final_state()` gives the closed state on demand without advancing |
| decode must reproduce every rounding of a kernel without a spec | the replay kernel is generated from the prefill sources: K1's arithmetic (`fwd_kernel1.cuh:265-511`) and K2's MMA block (`fwd_kernel2.cuh:463-732`) are copied byte for byte by `tools/gen_replay_kernel.py`, built into the same extension with the same flags; `tests/test_replay_codegen.py` fails if the copy is stale |
| training forward (FLA chunk-64 Triton) differs from the inference prefill | **not covered** by this branch: train/infer alignment additionally needs the training forward to run `flash_kda.fwd` (backward through FLA's recompute). Decode == prefill == `fwd` is what this branch guarantees |

## 3. State

Per (slot, head), K = V = 128:

| tensor | dtype / shape | meaning |
|---|---|---|
| `S` | bf16 `[P, H, 128, 128]` | state at the last 16-token boundary, `[V, K]` like `flash_kda.fwd` states |
| `rows_k`, `rows_v`, `rows_g` | bf16 `[P, H, 16, 128]` | raw k, v, g (before l2norm / gate activation) of the open chunk's rows |
| `rows_beta` | bf16 `[P, H, 16]` | raw beta logits |
| `pos` | int32 `[P, H]` | rows in the open chunk (0..15), one copy per head (equal across heads) so that every CTA advances its own and a decode step is a single launch |

44 KiB per (slot, head) (state 32 KiB + rows 12 KiB + beta + pos). q of earlier rows is not kept: only their own
outputs read it. An fp32 initial state is rounded to bf16 once on entry, as `flash_kda.fwd` does.

## 4. Kernels and calls

**Replay kernel** (`csrc/replay/replay_kernel.cuh`, generated): one CTA per (entry, head), 256 threads (K1's thread
mapping), warps 0-3 run K2's MMA block, 123 registers (2 CTAs/SM), no spills. Per call it starts loading S
asynchronously (`cp.async`, overlapping the first segment's K1 work), then for each segment of new tokens that fits
into the open chunk: stages the tile (rows < pos from the row cache, new rows from the inputs, rows beyond zero), runs the
K1 arithmetic with `actual_len` = rows so far, runs K2 phases 1-5, writes the new rows' outputs, and runs phase 6
(state update) only if the chunk is complete. S is written back only if a chunk closed; the new raw rows are appended
to the row cache only if the chunk stays open. Edits to the copied code (all tagged `[replay]`): `actual_len`,
the beta staging offset, the single-stage `k_restored` view, and the `if (close_chunk)` around phase 6.

**`decode`**: exactly one kernel launch per step (one or several tokens per sequence); the kernel advances `pos`.

**`prefill`** (when the call has more than 16 tokens per sequence on average):

1. heads: tokens that complete an open chunk -> replay kernel (outputs into a side buffer);
2. bodies: the 16-aligned middle -> **one unmodified `flash_kda.fwd` call** over the packed input, cut into
   pseudo-sequences `[gap_0 | body_0 | gap_1 | body_1 | ... | gap_N]` (a gap = the previous tail + the next head,
   throw-away; one state buffer serves as `initial_state` and `final_state`);
3. tails: the trailing < 16 tokens -> replay kernel (they become the open chunk);
4. the head outputs are copied into place.

Two planners give the same bits. The device plan (default) needs no host synchronisation and can be captured in a
CUDA graph, but every gap is a pseudo-sequence even when empty and the bookkeeping is a dozen small kernels. The host
plan (`context_lens_cpu=`, plus `cu_seqlens_cpu=` with `cu_seqlens`; a serving engine knows both on the host) only
creates the non-empty segments and skips phases without work; fresh prompts whose lengths are multiples of 16 become
literally one plain `flash_kda.fwd` call on the state pool (without `slot_ids`; with them, the bodies' states go
through a gather / scatter). Short calls (<= 16 tokens per sequence on average) go
through the replay kernel only.

Why the plan matters (measured, not a design choice of this branch): adding pseudo-sequences to a `flash_kda.fwd`
call slows its K2 down even when they are empty, e.g. 8x8192 H12 1068 -> 1340 us with 9 extra empty sequences, because
the extra CTAs make the block scheduler place two long-running K2 CTAs on one SM
(`benchmarks/probe_fwd_pseudo_sequences.py`, numbers in section 6).

**`final_state`**: runs `flash_kda.fwd` on the open rows from S (a short last chunk) and returns its `final_state`,
i.e. exactly what one `fwd` call over the whole sequence returns.

## 5. Verification

`tests/test_replay.py` (GPU, 86 cases; every comparison on raw bits, so -0/+0 count):

* premises on the unmodified kernel: row t of `fwd` over 0..T == row t of `fwd` over 0..t; split at 16-multiples ==
  one call (bf16 and fp32 state API); the open-chunk oracle with `fwd` alone; zero-length sequences in `cu_seqlens`;
* decode from an empty state, 6 gate regimes x H in {1, 4}, plus big-v / near-collinear keys / extreme beta / extreme
  gate inputs; prefill(L) + 35 decode steps for L in {1, 5, 15, 16, 17, 31, 32, 33, 100, 255, 1000} x initial state
  {none, bf16, fp32} incl. `final_state`; multi-token decode steps (2..33 tokens), also batched with a slot pool and
  padding entries; random chunked-prefill cuts; both prefill paths forced; both prefill planners (device / host,
  batched / varlen, with and without slot pools); varlen batches with sequences at different chunk rows; continuation
  calls mixing heads / bodies / tails, heads that do not fill the chunk, sequences with 0 new tokens, `cu_seqlens[0]
  != 0`; `final_state` with slot pools and at a chunk boundary; H = 1 / 32 / 96; slot pools with -1 padding entries;
  captured CUDA graphs (also with a rotating slot mapping); run-to-run determinism; a 24-sequence varlen sweep through
  the replay path; agreement with `tests/torch_ref.py` (the bit-exact PyTorch twin of `fwd`).

`tests/test_replay_codegen.py` (CPU, 2 cases): the committed kernel equals what the generator produces from the current
prefill sources, and the two hand-copied values (`a_log_exp`, `gate_scale`) still match the prefill's own lines.

Prefill machine code: `cuobjdump -sass` of the 17 prefill kernels (`_flash_kda_fwd_prepare`,
`_flash_kda_fwd_recurrence`, `_flash_kda_build_tile_prefix`) in this branch's `flash_kda_C` is byte-identical to the
same kernels built from master's sources with the same container (md5 of the address-stripped dump d54b1a75...).

## 6. Results (one B300, 2026-11-24)

Setup: NVIDIA B300 SXM6 AC, driver 580.126.09, NGC pytorch 26.08 (nvcc 13.4, torch 2.14), sm_103a build. All 88 tests
pass (`tests/test_replay.py`: 86, `tests/test_replay_codegen.py`: 2; master's `tests/test_fwd.py::test_fwd` and
`::test_fwd_varlen` pass on the same build). Timings: the cross-implementation harness of this investigation
(`bench_vs_rec.py`, identical inputs for every implementation, complete API calls replayed from a CUDA graph; decode =
64 consecutive steps in one graph, per-step time; "staggered" = sequences spread over the chunk rows).

What is compared (configuration first; the ratios below only isolate one variable where the contracts match):

| name | entry | chunk | MMA / intermediates | state | accuracy (rel. RMS vs fp64, synthetic, 11-24 unified run) | bitwise decode == prefill |
|---|---|---|---|---|---|---|
| fwd | `flash_kda.fwd` (master, unchanged) | 16 | bf16 HMMA, bf16 intermediates | bf16 | 2.8-4.7x the bf16 floor | no decode entry |
| replay | `flash_kda.replay` (this branch) | 16 | identical to fwd (same bits) | bf16 (+ open-chunk rows) | identical to fwd | yes |
| C64 | `flash_kda.cfbi` (round2-decode HEAD ea86834) | 64 | 3xTF32, fp32 intermediates | fp32 (+ row cache) | at the bf16 floor | yes |

**Prefill**, us (22 cases; `group` as in the harness). replay/fwd is like-for-like (same kernel, same numerics; the
difference is the wrapper). C64 is a different contract (chunk 64, fp32): its column mixes chunk size, precision and
implementation.

| case | fwd | replay, host plan | replay, device plan | C64 |
|---|---|---|---|---|
| 1x4096 H12 | 405.5 | 410.4 | 455.2 | 288.0 |
| 8x8192 H12 | 1060.9 | 1073.8 | 1403.8 | 2972.7 |
| 16x2048 H12 | 416.1 | 428.4 | 505.2 | 1508.7 |
| 1long+7short H12 | 845.0 | 847.1 | 938.9 | 822.6 |
| 4long+4short H12 | 942.4 | 946.8 | 1275.2 | 1664.8 |
| mixed33 H12 | 1901.9 | 1833.0 | 1892.6 | 1972.9 |
| 64x512 H12 | 360.9 | 379.3 | 493.8 | 1603.4 |
| smoke3 (128, 1023, 4096) H12 | 417.9 | 451.9 | 484.3 | 645.5 |
| train24x8192 H12 | 1896.0 | 1928.6 | 2051.6 | 8409.1 |
| train6x32768 H12 | 3905.4 | 3935.1 | 3994.5 | 8269.0 |
| train1x196608 H12 | 19057.1 | 19031.9 | 19084.7 | 11409.5 |
| geomean / fwd, all 22 (H12 + H16) | 1 | 1.018 | 1.165 | 1.845 |

The host plan is one plain `fwd` call whenever no sequence has a head or a tail (here every case except smoke3, whose
1023-token request leaves a 15-row tail: 1.08x). The device plan pays for empty pseudo-sequences and a dozen small
kernels. `benchmarks/probe_fwd_pseudo_sequences.py` isolates the first effect on `fwd` alone (same tokens, only the
`cu_seqlens` layout changes): 8x8192 H12 1068 us plain, 1340 us with 9 empty pseudo-sequences, 1343 us with 16;
64x512 378 / 399 / 423 us; 16x2048 425 / 439 / 444 us.

**Decode**, us per step (one token per sequence; N sequences x H heads):

| N, H | replay | replay staggered | C64 | C64 staggered |
|---|---|---|---|---|
| 1, 12 | 5.64 | - | 10.25 | - |
| 16, 12 | 7.56 | 9.11 | 15.97 | 23.80 |
| 64, 12 | 21.02 | 22.44 | 49.69 | 60.33 |
| 128, 12 | 38.83 | 39.86 | 90.80 | 100.33 |
| 1, 16 | 5.77 | - | 10.32 | - |
| 16, 16 | 7.65 | 9.50 | 16.36 | 26.03 |
| 64, 16 | 26.87 | 28.48 | 64.65 | 73.59 |
| 128, 16 | 49.75 | 51.95 | 114.97 | 125.25 |

Speculative-style calls (8 tokens per request plus a full-state snapshot after every token, us per call), N = 16 / 64 /
128: H12 replay 149 / 314 / 559, C64 256 / 863 / 1599; H16 replay 156 / 394 / 721, C64 309 / 1115 / 2076. The
replay decode is a different contract from C64 (bf16 state of 32 KiB + 12 KiB of rows per head vs fp32 state of
64 KiB + 160 KiB of row cache, bf16 vs 3xTF32 arithmetic), so these columns compare the two designs, not two
implementations of one design.

Other references from `benchmarks/bench_replay.py` (one decode step per CUDA-graph replay, so every number includes a
graph launch and is higher than in the table above), us per step, H12 N = 1 / 8 / 16 / 32 / 64 / 128: replay
7.5 / 8.9 / 11.1 / 16.5 / 23.1 / 40.7; `flash_kda.fwd` called with one token per sequence (the original kernel used
naively for decode, not aligned: it closes a one-row chunk every step) 12.3 / 14.5 / 18.5 / 26.6 / 38.9 / 69.6; FLA
`fused_recurrent_kda` with an fp32 state (a different arithmetic program; g / beta activated outside the timed call)
6.2 / 8.2 / 10.3 / 16.4 / 30.2 / 51.2.

How the decode got there (same kernel arithmetic throughout, bits unchanged): the first version (separate
PyTorch kernels for `pos`, synchronous state load, 166 registers = 1 CTA/SM) took 13.6 / 18.4 / 39.2 / 67.6 us at
H12 N = 1 / 16 / 64 / 128; one launch per step, the `cp.async` state load and 123 registers (2 CTAs/SM) brought it to
5.6 / 7.6 / 21.0 / 38.8 us.

Raw data: `xcmp-bench/results-replay/` of this investigation (vsrec_replay_v2.json decode, vsrec_replay_v3.json
prefill, vsrec_cfbi.json, vsrec_base.json, bench_replay_v2.json).

## 7. Scope and limits

* **Same build, same GPU architecture.** The reference is the compiled `flash_kda.fwd`; the replay kernel is in the
  same extension with the same flags. Different archs (sm_90a vs sm_100a/103a) are different binaries whose MUFU/HMMA
  results are not specified to agree; compiler decisions (e.g. the FMA contraction of the state update, the
  non-contraction of the gate cumsum across the `actual_len` branch) are per toolchain. Re-run `tests/test_replay.py`
  after any toolchain/CUTLASS change.
* **Finite inputs.** `flash_kda.fwd` is only causal for finite values: a NaN/Inf in a later row of the same chunk
  reaches earlier rows through `+0 * NaN` inside the HMMAs, which a decode that has not seen that row cannot
  reproduce. Keep `lower_bound >= -5.5` (`|lb| * 16 * log2(e) < 128`); beyond that `ex2(-g_cumsum)` overflows and
  `flash_kda.fwd` itself produces NaN.
* **Accuracy is the original kernel's** (bf16 intermediates and bf16 state, chunk 16): the alignment does not change
  numerics, it makes decode reproduce them.
* **Upstream inputs.** Bitwise outputs also need bit-identical q/k/v/g/beta between the two paths (projections,
  short conv): outside this kernel.
* **Unchecked preconditions** (checking them would need a device synchronisation): distinct `slot_ids` within a call
  (a slot >= P and a corrupted `pos` trap), `out` not overlapping q / k / v / g / beta, one stream per state, and for
  the host plan `context_lens_cpu % 16 == pos`.
* Pre-existing in `flash_kda.fwd` (not changed here): the final-state TMA store is not followed by `tma_store_wait`
  before the CTA exits; `beta_linear` is int32 (`H * T_total < 2^31`).
* **Prefill planner.** The device plan (no host information, CUDA-graph capturable) costs 1.17x of `fwd` (geomean);
  the host plan (`context_lens_cpu` / `cu_seqlens_cpu`, which a serving engine has) is within 2% of `fwd` and is
  exactly one `fwd` call when no sequence has a head or a tail. A tail costs a short pseudo-sequence in the `fwd` call
  (which can slow `fwd`'s K2 through CTA placement) plus one replay launch. The host plan's `context_lens_cpu % 16`
  must equal the device `pos` (not checked).
* **Decode cost.** Every step re-runs the open chunk's K1 work and K2 phases 1-5 for 16 rows (the MMA instruction
  count is the same as for one row) and reads the 32 KiB bf16 state. Caching per-row K1 products (cumsum, k_inv,
  L rows, U rows) would shorten a step further and is bitwise-equivalent; not done.
* **Not covered:** training forward / backward alignment (needs the training forward to run `flash_kda.fwd`, backward
  through FLA's recompute), context parallelism, other GPU architectures than sm_103a (B300) in this run.
