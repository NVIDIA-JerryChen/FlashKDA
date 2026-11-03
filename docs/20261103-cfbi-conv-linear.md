# CF-BI conv and linear: the layers around the KDA core, bitwise train/infer consistent (v1)

Status: implemented in `csrc/cfbi/cfbi_conv.cuh` and `csrc/cfbi/cfbi_linear.cuh` (launchers `cfbi_convlin.cu`, binding
`cfbi_convlin_binding.cpp`, extension `flash_kda_cfbi_conv_C`), Python `flash_kda.cfbi.short_conv` and
`flash_kda.cfbi.bi_linear`. Validated on one B300 (`sm_103a`) only, like the core. This document is the *specification* of the
two floating-point programs (sections 2 and 3), and records what was verified (section 6) and what it costs (section 5).
The KDA core itself is specified in [20261102-cfbi-numerical-contract.md](20261102-cfbi-numerical-contract.md).

## 0. The problem in plain words

The CF-BI core makes the KDA recurrence give the same bits whether it runs as one prefill, as token-by-token decode or as the
training forward. A layer is more than the recurrence, though. In FLA's `KimiDeltaAttention` the core's inputs are produced by

```
x --Linear--> q_pre --depthwise causal conv (W = 4) + SiLU--> q        (same for k, v)
x --Linear--Linear--> g          x --Linear--> beta
```

If these produce different bf16 values in training and in serving, the core receives different inputs and its guarantee is
worth nothing for the layer. Two separate mechanisms break the bits:

* **Conv.** Floating-point addition is not associative. A prefill kernel that adds the four taps as a chain
  `((w0 x0 + w1 x1) + w2 x2) + w3 x3` and a single-token kernel that adds them in another association (two interleaved chains
  `(w0 x0 + w2 x2) + (w1 x1 + w3 x3)`, or the reverse order) differ in the last fp32 bit for a third to two fifths of the outputs
  (measured here: 30.8% and 41.7%). After the bf16 rounding only about one output in 10^4 .. 10^5 changes, which is why such
  differences go unnoticed, but a changed q / k / v value enters the core's state and never leaves.
* **Linear.** cuBLAS chooses its kernel (tile shape, split-K, stream-K) from the *shape*. Row `m` of `x @ W^T` is a sum over K
  whose order depends on how many other rows are in the call: a decode step (M = 1) and the whole prompt (M = T) can add the
  same numbers in a different order (section 7 measures it: up to 0.8% of the bf16 outputs differ).

The cure is the same as in the core: **one fixed floating-point program per output element**, so that the schedule (how many
tokens, how many rows, how the work is tiled) cannot change a single bit.

```
conv, one channel, W = 4, token t:   acc = fma(w3, x[t], fma(w2, x[t-1], fma(w1, x[t-2], fma(w0, x[t-3], 0))))   <- always this chain
                                     y   = bf16( acc * rcp(1 + ex2(-acc * log2 e)) )
history x[t-3 .. t-1] comes from the same call, or from the 3-input state of the previous call: the values are the same bf16.

linear, one output element (K=256): acc = mma16(x[m][240:256], w[n][240:256], ... mma16(x[m][0:16], w[n][0:16], 0))   <- K ascending, one chain
                                     out = bf16(acc)           (M = 1 or M = 8192: which lane computes it is irrelevant)
```

## 1. API

```python
from flash_kda.cfbi import short_conv, alloc_conv_state, bi_linear

# projections: row m of the result does not depend on M or on the other rows
q_pre = bi_linear(x, Wq)                                  # x [..., K] bf16, Wq [N, K] bf16, K % 16 == 0 -> bf16 [..., N]

# conv: training (state=None: zero history) or serving (bf16 state [N, D, W-1] updated in place)
state = alloc_conv_state(N, D, W=4)                       # zeros for fresh sequences
q = short_conv(q_pre.view(B, T, D), conv_weight, None, state, "silu")                  # prefill (any cut) and decode (T = 1)
q = short_conv(q_pre.view(1, Ttot, D), conv_weight, None, state, "silu", cu_seqlens)   # packed varlen batch
q = short_conv(q_pre.view(B, T, D), conv_weight, None, None, "silu")                   # training forward, differentiable
```

* `x`: bf16 `[B, T, D]` (each batch entry is a sequence) or `[1, Ttot, D]` + `cu_seqlens` (int64 `[N+1]`); `[T, D]` is one
  sequence. Channels contiguous; tokens may sit at a constant row stride (a column slice of a fused QKV GEMM works).
* `weight`: `[D, W]` or `[D, 1, W]` (the `nn.Conv1d` layout), fp32 **or** bf16, `2 <= W <= 8`; tap `W-1` multiplies the current
  token. The program works on `float(weight)`: a bf16 weight is just an fp32 weight with 8 mantissa bits (so every product
  `w * x` of two bf16 values is exact in fp32). `bias`: `[D]` fp32 or bf16 or None (KDA has none). `activation`: `"silu"`
  (`"swish"`) or `None`.
* `state`: see section 2. A call with `state=None` writes no state and reads zero history (the training forward).
  Stateful calls are inference-only and raise if autograd would record them (`torch.no_grad()`).
* `max_seqlen` (optional host int): upper bound of the sequence lengths; the grid is sized from it (default `T`, or `Ttot` for
  packed input). Pass the true bound in packed decode (`max_seqlen=1`) to avoid launching idle blocks; a sequence longer than the bound makes
  the kernel trap (a loud device-side error instead of silently dropped tokens; it poisons the CUDA context).
* `out_fp32=` (tests): receives the fp32 value before the bf16 rounding. Comparing only the bf16 outputs is a *weak* detector of
  arithmetic differences (a sabotaged association changes 15744 of 51200 fp32 values and 0 bf16 values in the control of
  section 6), so all equality tests compare both.
* Differentiable: `short_conv` with `state=None` and `bi_linear` are `torch.autograd.Function`s whose forward is the pinned
  kernel and whose backward is plain PyTorch (fp32 elementwise for the conv, `dy @ W` / `dy^T @ x` for the linear). The backward
  is NOT bit-pinned; it only has to be correct (section 6: gradients against an fp64 autograd reference).

## 2. Contract: the short causal conv (`cfbi_conv.cuh`)

For one sequence, one channel `d`, position `t = 0 .. len-1`. `x_p` is the bf16 input at sequence position `p`; for `p < 0` it is
the entry of the conv state (zero for a fresh sequence and for the training forward). `w_k = float(weight[d][k])`,
`k = 0 .. W-1`. Every `*`, `+` below is its own rounding; nothing is contracted unless written `fma`.

```
acc_0     = +0.0f
acc_{k+1} = fma(w_k, float(x_{t-(W-1)+k}), acc_k)                k = 0 .. W-1, ascending      (__fmaf_rn: one rounding per tap)
z         = acc_W                                                 (no bias)
z         = fadd_rn(acc_W, float(bias[d]))                        (bias)
y         = z                                                     (activation = None)
y         = fmul_rn(z, rcp(fadd_rn(1.0f, ex2(fmul_rn(-z, LOG2E_F)))))     (SiLU)
out[t][d] = bf16_rn(y)                                            (round to nearest even)
```

`ex2` = `ex2.approx.ftz.f32`, `rcp` = `rcp.approx.ftz.f32` (one PTX instruction each, the same functions as the core's `sigmoid_act`
in `cfbi_prep.cuh`; `LOG2E_F` = fp32(log2 e)). The ftz behaviour is intended: for `z < -87.3` the result is `-0.0` instead of a
denormal (an absolute error below 1.1e-36). Measured accuracy of the program against the fp64 result of the same math: the
fp32 value within `2e-6 * (sum|w x| + |bias| + |ref|)`, the bf16 output within 0.5 bf16-ulp plus that slack (`test_accuracy_vs_fp64`).

**State.** `state[n][d][m]`, `m = 0 .. W-2`, is the bf16 input of sequence position `m - (W-1)` of the next call, i.e. the **last
`W-1` inputs of (old state ++ x)**, copied bit for bit (NaN payloads and `-0` included; checked with random 16-bit patterns). It holds
inputs, not outputs, so it does not depend on the arithmetic at all. A call updates it in place; a sequence of length 0 leaves it
untouched; a sequence shorter than `W-1` shifts the old entries (`conv_state_after` in `flash_kda/cfbi/conv.py` is the reference).
FLA's cache is `[N, D, W]` and also holds the current token: `fla_cache[:, :, 1:]` is this state.

**What this makes bitwise identical** (output bits, the fp32 value before the rounding, and the state): one call over the whole
sequence = any cut points (incl. positions that are no multiple of anything, every single cut was tested) = `T = 1` decode
calls = a varlen batch with other sequences in any order and with empty sequences = the training forward (`state=None`).
The reason is structural, not tested-in: the output at `t` is one function (`conv_acc` + `conv_finish`) of `W` bf16 values and `W`
weights; where the `W-1` history values come from (this call's input or the state) cannot matter, and there is no other
arithmetic on the path.

**Kernel** (no effect on bits). A block is (tile of up to 64 tokens of ONE sequence, channel tile, sequence); a thread owns 2
adjacent channels (1 for odd `D` or misaligned rows) and walks its tokens with a register window of the last `W-1` raw inputs. A
tile never crosses a sequence boundary. Only the first tile of a sequence reads the old state and writes the new state - in the
same thread, in that order - and every other tile starts at an offset `>= 64 >= W-1` and only reads `x`, so the in-place update
has no inter-block race (150 repeated launches of a stateful varlen call with sequences of up to 41 tiles are bit-identical).

## 3. Contract: the batch-invariant linear (`cfbi_linear.cuh`)

For `x [M, K]` bf16, `W [N, K]` bf16, optional `bias [N]` (fp32 or bf16), `K % 16 == 0`:

```
acc[m][n]  = +0.0f
for s = 0 .. K/16 - 1 (ascending):
    acc[m][n] = mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(x[m][16s .. 16s+15], W[n][16s .. 16s+15], acc[m][n])
out[m][n]  = bf16_rn(acc[m][n])                                   (no bias)
out[m][n]  = bf16_rn(fadd_rn(acc[m][n], float(bias[n])))          (bias)
```

One accumulator chain over K ascending, one instruction shape, no split-K, no second accumulator, no K padding (a last block
that is shorter than the kernel's k-tile skips the missing k16 steps instead of executing them on zeros). The tile configuration
(five of them: block `16x32`, `64x32`, `128x32`, `128x64`, `128x128`, k-tile 64 or 128, 3-4 cp.async pipeline stages, 4-8 warps) is chosen from
`M` and `N` for speed only: it decides which warp, lane and register holds an output element, never what is added in which
order. Row `m` of the result is therefore independent of `M`, of every other row, of its position in the batch, of `N` and of the
configuration - *provided the hardware premise of section 4 holds*, which the tests probe on the device they run on.

Restrictions: bf16 `x` and `W`; `K % 16 == 0` (the program is a chain of k16 MMAs; every hidden size in practice); `x` and `W` 16-byte
aligned with row strides that are multiples of 8 elements (otherwise a contiguous copy is made); `M` and `N` arbitrary (rows and columns
beyond `M` / `N` are zero-filled and never stored). The same program is run for the fused projection `[3D, K]` and for the
three `[D, K]` projections: column `n` does not depend on which other columns are computed (tested).

## 4. The hardware premise for bf16 `mma.sync` (probe)

`bi_linear` puts one row of `x` (decode) or up to 128 rows (prefill) into the 16 row slots of the MMA. That is bit-safe only if
an output element of the instruction chain depends on nothing but its own (row vector, column vector) pair, the K order and its
accumulator. `tests/cfbi/test_linear_probe.py` checks it with `probe_mma_bf16_kernel` (the *same* instruction and the same
ascending k16 loop, fragments loaded with scalar loads): the same (a, b) pair in each of the 16 row slots x 8 column slots x
{zero, random, wide-exponent (2^-60 .. 2^60)} neighbours, K = 16 .. 1024, five data regimes (normal, wide exponents, cancelling,
tiny x huge, sparse) - identical bits everywhere; NaN / Inf neighbours do not leak. Negative control: the reversed K order changes
95-97% of the fp32 outputs (so the test is sensitive). The chain's error against fp64 is 2.1e-7 of `sum|a b|` at K = 1024.
**Result on B300 (sm_103a): the premise holds.** On any other GPU, run this test before trusting `bi_linear`
(`bi_linear` refuses compute capability < 10 unless `FLASH_KDA_CFBI_ALLOW_UNVALIDATED_ARCH=1`, as the core does).

## 5. Measured speed (v1, correctness first)

One B300 (148 SMs, clocks not locked, shared node), CUDA-graph replays, `benchmarks/bench_conv_linear.py`. These are v1 kernels: the
linear uses `mma.sync` (the pre-Blackwell tensor-core interface), not tcgen05 / UMMA; the conv is bandwidth bound.

**Linear, K = N = 4096, bf16** (time per call in us; "cold" rotates through enough distinct weight matrices to exceed the L2 so the
weights stream from HBM as in real decode, "hot" repeats one matrix):

| M | tile config | bi_linear cold | cuBLAS cold | ratio | bi_linear hot | cuBLAS hot |
|---|---|---|---|---|---|---|
| 1 | 0 (16x32) | 12.1 | 9.1 | 1.3x | 11.4 | 6.5 |
| 16 | 0 | 12.2 | 11.4 | 1.07x | 11.7 | 11.4 |
| 64 | 0 | 20.3 | 9.4 | 2.2x | 20.4 | 6.3 |
| 128 | 1 (64x32) | 29.6 | 9.5 | 3.1x | 29.8 | 6.7 |
| 8192 | 4 (128x128) | 731.7 (376 TFLOP/s) | 154.1 (1784 TFLOP/s) | 4.75x | 726.8 | 159.2 |

Other projections of a KDA layer (hidden 4096, 16 heads, hot, us, `bi_linear / cuBLAS`): q/k/v `4096->2048` M=1 15.9 / 7.9, M=128 35.3 / 8.4,
M=8192 394.5 / 85.9; `f_proj[0]` `4096->128` 11.3 / 4.3, 25.2 / 5.0, 41.9 / 8.9; `f_proj[1]` `128->2048` 2.2 / 2.2, 3.2 / 2.2, 24.9 / 8.3; `b_proj`
`4096->16` 11.4 / 3.8, 27.1 / 4.2, 31.7 / 6.8; `o_proj` `2048->4096` 6.8 / 4.3, 16.5 / 4.6, 374.9 / 80.0.

Why it is slower than cuBLAS (reasoned, not profiled): the sum over K may not be split (no split-K), so the parallelism of a decode GEMM is the
number of N tiles (128 CTAs at N = 4096, or 16-row CTAs re-reading the weight tile from L2 for M <= 64, which is what the shape-to-config table
uses because it measured fastest); at M = 8192 `mma.sync` peaks well below tcgen05 (this kernel reaches 376 TFLOP/s, cuBLAS 1784). `M <= 16` is
memory bound and within 1.1-1.3x of cuBLAS. A throwaway experiment with four more configurations (32x32 and 64x32 tiles with 2 warps, a 128x32 tile with
a 32x32 warp tile, pipelines of 6 and 8 stages, k-tiles of 256; not kept, `benchmarks/micro/linear_configs.py` times the five that are) gave the same
bits and no speed-up beyond the 16-row configuration for M <= 64. Making it faster means a tcgen05 / UMMA kernel with the same chain per element, which has
to be probed for bf16 (the core's tf32 UMMA probe, `benchmarks/micro/umma_probe.cu`, found tcgen05 bit-identical to `mma.sync` for the three-term tf32 product).

**Conv, D = 4096, W = 4, SiLU, bf16** (us per call; FLA = its Triton `ShortConvolution`, torch = `F.conv1d` + `F.silu`):

| case | short_conv | FLA Triton | torch |
|---|---|---|---|
| decode N = 1 / 16 / 64 / 128 (1 token, state in place) | 2.2 / 2.9 / 3.8 / 6.0 | 1.5 / 1.7 / 3.3 / 5.4 | |
| prefill B = 1, T = 8192 (state in / out) | 45.6 | 42.3 | 468 |
| prefill B = 8, T = 1024 | 45.7 | 41.8 | 468 |
| varlen 8 x 1024 | 54.5 | 45.7 | |

`short_conv` is within 9% of FLA's kernel in prefill (2.9 TB/s effective) and within 15% at N = 64 .. 128 decode; the single-sequence decode
call is latency bound (2.2 vs 1.5 us). The training forward (`state=None`) costs the same as the prefill. The launch geometry was tuned a
little (`benchmarks/micro/conv_variants.py`: 128 threads, 4 tokens' loads in flight; larger unrolls are slower).

## 6. Verification (B300, aws-pdx, NGC pytorch 26.08; zero tolerance on bits unless a tolerance is named)

"Bits" = the bf16 output AND the fp32 value before the rounding (a much stronger detector, see section 1) AND the state.
Comparing only bf16 outputs is weak: both controls below change tens of thousands of fp32 values and 0 / 2 bf16 outputs.

| gate | test | result |
|---|---|---|
| conv program = the contract | `test_program_equals_emulation`: the kernel against a PyTorch emulation of section 2 (fma via exact fp64, `ex2` / `rcp` from the extension's own probe), W = 2, 3, 4, 5, 8, D = 200 (2 channels/thread) and 131 (1), fp32 and bf16 weights, bias fp32 / bf16 / none, SiLU / none, fresh and random state | bit-exact, 20 cases (fp32 values and bf16) |
| the test is sensitive to the tap order | the same emulation with the taps added in the reverse order | differs in 41.7% of the fp32 values (0.0039% of the bf16 outputs) |
| conv accuracy | `test_accuracy_vs_fp64`, W = 3, 4, fp32 / bf16 weights | fp32 value within `2e-6 (sum|wx| + |b| + |ref|)` (measured 0.06 of that slack), bf16 within 0.5 ulp + slack |
| state | `test_state_is_the_last_inputs` (L = 0, 1, 2, 3, 4, 5, 70, 130, random old state), `test_state_bits_are_copied_exactly` (random 16-bit patterns: NaN payloads, -0) | identical to the reference slicing, bit exact |
| whole = every cut = decode | `test_whole_vs_every_cut_vs_decode`: 7 (D, W, weight dtype, bias) cases incl. odd D, D = 3, W = 2 / 8, T = 140: **every one of the 139 single cut points**, `T = 1` x 140 calls, 5 random cut sets (pieces of 1-5, 63-65, ...) | identical bits (out, fp32, state) |
| long sequences | `test_long_sequence_cuts`: 9 tiles + 5 tokens, D = 520, cut pieces of 1 .. 300, `max_seqlen` = T and 10 T | identical |
| varlen = solo | `test_varlen_batch_vs_solo`: lengths 1, 0, 2, 3, 4, 5, 63, 64, 65, 128, 129, 200, 0, 1, 77, non-trivial initial state per sequence; permuted order; `max_seqlen` given | identical per sequence, 7 cases |
| decode batch | `test_batch_of_decode_steps_equals_solo`: 9 sequences x 37 steps as `[9, 1, D]`, and as packed 1-token sequences, each against its sequence as one call | identical |
| training forward | `test_training_forward_equals_prefill_from_zero_state` (`state=None`, `[B, T, D]` and varlen), strided slices of a fused `[T, 3D+1]` projection (also an odd offset), CUDA-graph replay over 40 decode steps | identical |
| conv negative control | build with `-DCFBI_CONV_NEG_CONTROL=1`: single-token tiles add the taps as two interleaved chains | **caught**: decode vs whole differ in 15744 / 51200 fp32 values (0 / 51200 bf16 outputs); production build 0 |
| conv gradients | `test_backward_matches_fp64_autograd`: SiLU on / off, fp32 / bf16 weights, bias, 5 packed sequences incl. an empty one | rel. RMS vs fp64 autograd: dx 1.7e-3 (bf16 rounding of dx), dw 2e-7 (fp32 weights) / 1.6e-3 (bf16), db 1e-7; forward of the autograd call = kernel bits |
| bf16 mma premise | `test_linear_probe.py` (section 4) | holds; reversed K differs in 95-97% |
| linear = the contract | `test_kernel_equals_the_documented_program`: `bi_linear` (auto and each of the 5 configs, with and without bias, K up to 1040 incl. a K tail) against the same program run tile by tile through the probe (scalar fragment loads, no smem / ldmatrix) | bit-exact, 18 shapes x 6 configs, fp32 values too |
| linear accuracy | `test_accuracy_vs_fp64`, K = 2048 | within 0.5 ulp + `2e-6 (sum|xw| + |b| + |ref|)` |
| row independent of M | `test_row_is_independent_of_M`: 5 (K, N) incl. 4096 x 4096, M = 1 .. 2048 (24 values), views at row offsets, permuted rows, single rows; `test_every_M_up_to_300` (M = 1..300, K = 1040) | identical bf16 and fp32 values |
| configs agree, neighbours, N | `test_all_tile_configurations_agree` (5 configs), `test_neighbour_rows_do_not_matter` (zero / random / 2^60 / NaN / Inf rows next to the probe row), `test_columns_do_not_depend_on_N` | identical |
| linear negative control | build with `-DCFBI_LIN_NEG_CONTROL=1`: the small-M (M <= 64) configuration walks K backwards | **caught**: rows computed with M = 1 / 16 vs inside M = 512 differ in 248 / 256 and 3968 / 4096 fp32 values (0 and 2 bf16 outputs); production 0 |
| chain: layer front end | `test_layer_chain.py`: whole-sequence training-style call vs prefill(110) + 40 decode calls, hidden 1024 / 4 heads and 4096 / 16 heads, and 1000 tokens (prefill 905 + 95 decode) at 2048 / 8 heads; a varlen batch of 4 sequences + lock-step decode with compaction; one decode step of the whole chain (6 projections, 3 convs, core decode) captured in a CUDA graph and replayed over 40 tokens | 0 differing elements in 18 tensors (projection, conv, core outputs, bf16 and fp32) and in the final conv / core states |
| chain negative controls | sabotaged conv decode association, sabotaged small-M linear, cuBLAS projections (serving side only) | **all caught** (conv: 31169 / 31525 / 31270 fp32 values of q / k / v; linear: 81405 fp32 values of `q_pre` ...; cuBLAS: 3044 bf16 values of `g`, 7 of `beta`, 1481 bf16 core outputs) |
| autograd forward | training step with weights requiring grad (`bi_linear`, `short_conv`, `chunk_kda` as autograd functions) | forward = serving bits; backward runs end to end (needs `flash-linear-attention`) |
| random sweeps | `test_convlin_fuzz.py` (seeded): 24 random packed batches (D 1 .. 700 odd and even, W 2 .. 8, random bias / weight dtypes, random row stride, random initial state, lengths incl. 0 / 1 / 63 / 64 / 65) = every sequence alone cut at random points = the PyTorch emulation of section 2; 20 random `(M, N, K)` linears = the probe program, any row subset in any order, all 5 configs | identical (out, fp32, state) |
| memory / races | `compute-sanitizer` memcheck (with `PYTORCH_NO_CUDA_MEMORY_CACHING=1` so that over-reads of a tensor are detectable) over the conv and linear tests, racecheck and synccheck over the linear tests (all configs) | 0 errors, 0 hazards |
| runtime | 150 repeated launches of a stateful varlen call (sequences of up to 41 tiles), 50 repeated linear calls, a side CUDA stream, a too-small `max_seqlen` traps (subprocess), two processes (SHA-256), JIT build vs `setup.py` build (`FLASH_KDA_BUILD_EXTS=cfbi_conv`, `-gencode arch=compute_103a`) | identical; the two builds give the same digest `b093a85f...` |


## 7. Control: cuBLAS is not batch invariant

`benchmarks/cublas_batch_variance.py`, `torch.matmul` / `F.linear` (bf16, cuBLAS from the NGC 26.08 container), the same rows computed with different `M`;
fraction of differing bf16 outputs (a bit comparison; rows 0..31 of each call). `bi_linear` is exactly 0 in every cell.

(a) the requested comparison, K = N: rows computed with M = 1, M = 64 and M = 8192: **0.0000%** differ for K = N = 2048, 2560, 3072, 3584, 4096. On this B300
cuBLAS happens to use the same K order for those three batch sizes. It does not for other M or other shapes:

(b) rows 0..31 computed with M rows vs with M = 8192 (the `M = 1` column computes each row in its own call):

| K | N | M=1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 2048 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3072 | 3072 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4096 | 4096 | 0 | 0.244% | 0.275% | 0.250% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4096 | 2048 | 0 | 0.122% | 0.085% | 0.159% | 0.186% | 0.180% | 0.180% | 0 | 0 | 0 | 0 | 0 | 0 |
| 2048 | 4096 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4096 | 11008 | 0.224% | 0.173% | 0.198% | 0.211% | 0.221% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 11008 | 4096 | 0.465% | 0.342% | 0.397% | 0.439% | 0.455% | 0.465% | 0.465% | 0.465% | 0 | 0 | 0 | 0 | 0 |
| 4096 | 128 | 0.195% | 0.391% | 0.195% | 0.195% | 0.293% | 0.195% | 0.195% | 0.220% | 0.195% | 0 | 0.195% | 0 | 0 |
| 4096 | 16 | 0.195% | 0 | 0 | 0.781% | 0.391% | 0.195% | 0.195% | 0.195% | 0.195% | 0.195% | 0.195% | 0.195% | 0 |
| 128 | 2048 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

So cuBLAS is batch invariant on this GPU for some shapes (every K = 2048 shape tried) and not for others: for **K >= 4096 and M <= 128 - the decode
regime - it differs from its own M = 8192 result in 0.1-0.8% of the bf16 outputs for many M** (M = 2, 4, 8 at 4096 x 4096; M = 2 .. 64 at 4096 x 2048;
M <= 16 at 4096 x 11008; M <= 128 at 11008 x 4096; nearly every M for N = 16 and 128), because it adds the K terms in a different order (at the fp32 level nearly
all values differ). That is the size of the error that would make a decode step disagree with the prefill row, and the reason the projection of a
CF-BI layer cannot go through cuBLAS. In the layer chain of section 6 (hidden 4096) cuBLAS projections change 3044 bf16 values of `g` (the first `f_proj` linear is
`4096 -> 128`), 7 of `beta` and, through the core's state, 1481 bf16 core outputs of a 150-token sequence.

Accuracy is the same as cuBLAS's (`benchmarks/micro/linear_accuracy.py`): rel. RMS error against fp64 1.65e-3 for both at (256, 4096, 4096), (64, 4096, 2048),
(128, 11008, 4096), i.e. the bf16 rounding floor; where cuBLAS does not split K (e.g. M = 256, K = N = 4096) its bf16 outputs are bit-identical to `bi_linear`'s (0 of
1M differ), elsewhere 0.18% (M = 64, 4096 -> 2048) to 0.50% (K = 11008) differ because it adds the K terms in another order.

For orientation, FLA's own conv (`test_against_fla_short_convolution`): on 2,097,152 outputs (D = 2048, T = 1024, no bias) FLA's Triton full-call kernel gives exactly
the same bf16 bits as `short_conv` (0 differ, bf16 and fp32 weights); its single-token `step` kernel differs from its own full-call kernel in 1 output (bf16 weights)
and in 85 outputs (fp32 weights), i.e. 5e-7 and 4e-5 of the outputs. Rare, nonzero, and not guaranteed by anything but the details of that compiler output.

## 8. Composition with the core

The chain `bi_linear` -> `short_conv` -> `prefill` / `decode` / `forward_train` is gated by `tests/cfbi/test_layer_chain.py`: the
same layer evaluated as one training-style call over the whole sequence and as prefill(110) + 40 single-token decode calls (also
a packed varlen batch of four sequences with lock-step decode and compaction of finished slots, and decode steps replayed from a CUDA
graph) gives bit-identical projection
outputs, conv outputs and core outputs (bf16 and fp32) and final conv / core states, for hidden 1024 / 4 heads and hidden 4096 / 16
heads. With the training step's autograd functions (weights requiring grad) the forward bits are the serving bits.

Recipe for a model integration:

* replace `nn.Linear(x)` of the q/k/v/f/b/g projections by `bi_linear(x, layer.weight, layer.bias)` (the gate's second Linear has a bias;
  bf16 or fp32 bias both work). Fusing q/k/v into one `[3D, K]` GEMM is bit-identical to three GEMMs; `short_conv` accepts the column
  slices.
* replace `ShortConvolution` by `short_conv(x, conv.weight, conv.bias, state, "silu", cu_seqlens)`. Keep one `[N, D, W-1]` state per
  conv (q, k, v). In a scheduler, `state[slot_ids]` (gather) / scatter works exactly like the core's state in the scheduler test.
* The gated RMSNorm after the core and `o_proj`: `o_proj` is a Linear (`bi_linear`); the gated norm is NOT covered (see 9).

## 9. Limits and what is not verified

* **Architecture**: only B300 (`sm_103a`). The conv uses CUDA-core fp32 and two PTX approximations (`ex2`, `rcp`); results of those
  are hardware-defined, so bits can differ between GPU generations (not within one). The linear additionally needs the premise of
  section 4 on the device.
* **Compiler build**: bits are relative to one compiler build; the JIT build and the `setup.py` build (different nvcc command line)
  give the same digest on the fixed workload of `tests/cfbi/digest_convlin.py`. `-fmad=false -ftz=false` and explicit intrinsics are
  part of the contract (no `--use_fast_math`).
* **Not covered**: the gated RMSNorm (`FusedRMSNormGated`), context parallelism, tensor-parallel collectives, fp16 / fp32 inputs, `K % 16 != 0`.
  Splitting a GEMM across TP ranks along K changes the program (it is a split-K by definition); splitting along N (column parallel)
  does not.
* **Speed**: mma.sync only (no tcgen05 / TMA pipelines): `bi_linear` is 1.1-1.3x cuBLAS for `M <= 16` and 2.2-4.8x for larger `M` at
  4096 x 4096 (section 5). A faster batch-invariant kernel is an engineering item (UMMA would have to be probed for bf16, see section 5); the arithmetic per element is pinned by the contract.
* **Backward**: plain PyTorch fp32 for the conv (it materialises a few `[tokens, D]` fp32 temporaries) and cuBLAS matmuls for the linear;
  not bit-pinned, gradients correct to the bf16 floor.
* **`max_seqlen`** must be a true bound (a too-small one traps, see section 1).

## 10. Reproduce

```bash
python -m pytest tests/cfbi/test_conv.py tests/cfbi/test_linear_probe.py tests/cfbi/test_linear.py \
                 tests/cfbi/test_layer_chain.py tests/cfbi/test_convlin_runtime.py tests/cfbi/test_convlin_fuzz.py -q   # JIT-builds flash_kda_cfbi_conv_C (+ flash_kda_cfbi_C for the chain)
python benchmarks/bench_conv_linear.py            # section 5
python benchmarks/cublas_batch_variance.py        # section 7
FLASH_KDA_BUILD_EXTS=cfbi_conv FLASH_KDA_CUDA_ARCHS=103a python setup.py build_ext --build-lib /tmp/convbuild   # only this extension
```
