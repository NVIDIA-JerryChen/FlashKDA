# Recurrent consistency tests

See [API and commands](../../docs/consistency.en.md) / [中文](../../docs/consistency.md).

`flashinfer_reference.py` contains the unmodified gate/reduction helpers, one-warp
kernel and launch function from
[FlashInfer c3af0c13328fd23b256c53b9b4f94d9e160d118c](https://github.com/flashinfer-ai/flashinfer/blob/c3af0c13328fd23b256c53b9b4f94d9e160d118c/flashinfer/kda_kernels/recurrent_kda.py).
Unrelated imports/dispatch code were removed; a host adapter fixes D=128, rows=16,
dual accumulators, T=1, raw gates/beta and in-kernel normalization. The reference
never imports production arithmetic. Its Apache-2.0 license is in
[licenses](../../licenses/FlashInfer-Apache-2.0.txt), with the applicable
[upstream notice](../../licenses/FlashInfer-NOTICE.txt).

Coverage includes:

- H=12/16/32; lengths 0/1/2/15/16/17/127/128/129: every single-token output and
  state against the pinned reference, plus whole-sequence versus token execution.
- Batches 1/4/7/8/9/16/32: independent requests, permutation, mixed lengths,
  padding, and dynamic request addition/removal.
- All split points of a 17-token sequence, empty/unequal requests, PD regrouping,
  separate/in-place states, nonzero initial state, zero Q/K and saturated gates.
- Repeated calls, unaligned contiguous inputs, no-grad/error contracts, eager
  versus CUDA Graph, changing active metadata and compile-cache reuse.
- Opt-in 4K/32K random segmentation for each H, with lower_bound=0 to exercise
  long state retention. Set `FLASH_KDA_LONG_TESTS=1` for acceptance.
- Opt-in 4K/32K independent single-token reference at H=12, comparing every
  output and continuation state. The matrix is risk-based, not a Cartesian
  product of all lengths, batches, H values and distributed configurations.
- Distributed rank-ordered CP=1/2/4/8 with empty/unequal partitions, intermediate
  boundary states, and CP prefill followed by regrouped decode. The default matrix requires eight ranks; `--cp-sizes` selects an explicit
  diagnostic subset, which does not constitute full acceptance.

Comparisons reject nonfinite values and compare BF16 raw int16 views, preserving
signed-zero distinctions. Failures report the first request/head/token/V or
request/head/token/V/K coordinate. FP64 diagnostics use independent arithmetic
with per-step BF16 state rounding and report max absolute/RMS errors separately.

The CP test also exercises receive → captured local graph replay → send, with
communication outside capture. For two nodes with four allocated GPUs each, run
`python -m tests.consistency.launch --nodes 2 --node-rank R --local-world-size 4
--store /shared/unique-new.store` on each node, replacing R with 0 or 1. The same
initially absent shared path must support file locking. Use a new path each run;
this launcher does not probe and release a TCP port before use.

Dense single-token prepared decode tests are in `test_decode.py`; see the
[decode contract](../../docs/consistent-decode.en.md).

融合 dense prefill 检查：`python -m pytest -q tests/consistency/test_prefill.py`。
覆盖 direct/staged、每步参考状态、工作区边界、PD、两 GPU head 分片和 Graph 内容更新。
独立数学诊断：`python -m tests.consistency.accuracy --prefill`。

Unified dense/varlen prefill: `python -m pytest -q tests/consistency/test_unified_prefill.py`.
This covers split execution, dynamic device metadata during Graph replay, capacity
tails, compile-cache reuse, empty inputs and checkpoint-prefix continuation.
Per-token checkpoint pool tests are in `test_checkpoints.py`.
The older `accuracy` helper measures a BF16-state numerical-contract reference;
it is not a full-precision state-recurrence oracle.
