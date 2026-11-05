# Dense decode validation and performance

[简体中文](consistent-decode-validation.md) · [API contract](consistent-decode.en.md)

Measured on B300 on 2026-11-05: `prepare_consistent_decode(...).run()` versus the single-token
kernel extracted from FlashInfer `c3af0c13328fd23b256c53b9b4f94d9e160d118c`.
The new Graph latency differs by **−7.58% to +1.37%** across this matrix. Performance acceptance
thresholds remain undecided; these measurements do not establish SOL.

## Environment and measurement scope

- Two B300 GPUs connected through NVLink on one node. Correctness uses both GPUs; paired timing uses one GPU and one process.
- Python 3.12.3, PyTorch 2.13.0a0+9186a08b2c.nv26.07, CUDA 13.3, driver 615.71.10, CuTe DSL 4.6.2.
- BF16 inputs, outputs and in-place continuation state; K=V=128, nonzero initial state, `lower_bound=-5`.
- Each cycle has 64 independent T=1 calls with 64 distinct token inputs, without cross-token fusion.
- Both Graphs contain exactly 64 kernels and zero memcpy nodes. All outputs and complete states are checked bitwise at every step before timing.
- Nine alternating paired samples; each sample uses 25 Graph cycles or five eager cycles. Tables show medians.
  Fixed-address warm-cache protocol, no cold-L2 claim, no manual clock lock. Initial state is reset equally before each sample, outside timing.
- The complete prepared `plan.run()` hot path is timed. Compilation, data generation, allocation and binding are outside timing for both sides.
  FlashInfer is an extracted kernel, not its package dispatcher. Per-call validation in the convenience API is excluded from prepared-plan measurements.

## Results

Times are µs/step for generating one token for the entire batch, not divided by B/H.
Graph uses GPU events; eager wall time includes launches and final synchronization.
Change is `new Graph / FlashInfer Graph − 1`; negative values mean the new path is faster.

| H | B | FI Graph | New Graph | Change | FI eager wall | New eager wall |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 1 | 2.242 | 2.273 | +1.37% | 4.224 | 3.865 |
| 6 | 64 | 6.110 | 5.986 | -2.02% | 6.548 | 6.464 |
| 12 | 16 | 3.648 | 3.587 | -1.67% | 4.259 | 4.052 |
| 12 | 32 | 6.106 | 5.893 | -3.48% | 6.561 | 6.406 |
| 12 | 64 | 9.935 | 9.476 | -4.62% | 10.424 | 9.959 |
| 12 | 128 | 17.600 | 16.267 | -7.58% | 18.132 | 16.764 |
| 16 | 1 | 2.339 | 2.342 | +0.14% | 4.297 | 3.879 |
| 16 | 64 | 12.627 | 11.855 | -6.11% | 13.101 | 12.344 |
| 32 | 1 | 2.402 | 2.401 | -0.04% | 4.246 | 3.891 |
| 32 | 64 | 23.496 | 21.964 | -6.52% | 24.149 | 22.546 |

[Raw samples and source hashes](../benchmarks/data/consistent-decode-b300.json) include min/max, medians and every sample.
Small differences are not cross-platform guarantees. Reproduction commands are in the API documentation.

## Correctness and accuracy

`FLASH_KDA_LONG_TESTS=1 python -m pytest -q tests/consistency`: **154 passed**.
New decode tests cover H=12/16/32 and smaller local head counts, empty batches, nonzero initial states,
in-place and separate states, saturated gates, request permutations, arbitrary whole-head shards,
prefill/decode boundaries, 4K/32K sequences, dynamic Graph buffer contents and compile-cache reuse.
Two-GPU checks cover whole-head sharding for H=12/16/32 and continuation after transferring BF16 state
between GPUs. Framework communication implementations are outside this validation.
Valid outputs and complete BF16 continuation states are compared bitwise; NaN/Inf fail. The public example ran successfully.

Independent FP64 diagnostics use B=1, H=12 and 129 tokens, retaining BF16 state rounding after each token:

| Data | Maximum absolute error | RMS |
|---|---:|---:|
| Output | 2.7156554e-5 | 1.3608709e-6 |
| Final state | 6.1035156e-5 | 1.9618364e-7 |

The new and generic recurrent paths produce identical diagnostic results. These errors include normal rounding;
they are diagnostic, not substitutes for bitwise acceptance.

This version retains the original BF16 recurrent numerical contract. The prefill continuation checks in this record use `consistent_fwd`.
See the [prefill documentation](consistent-prefill.en.md) for separate fused-prefill validation and performance.
This record makes no claims about CF-BI chunk implementation, full-model logits or multi-GPU communication throughput.
