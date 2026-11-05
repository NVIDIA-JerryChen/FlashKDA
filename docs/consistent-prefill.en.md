# Fused recurrent prefill

[中文](consistent-prefill.md) · [decode](consistent-decode.en.md) · [general variable-length API](consistency.en.md)

`prepare_consistent_prefill(...).run()` processes a contiguous dense sequence under the
same recurrent numerical contract as `prepare_consistent_decode`: each token computes
its output from the updated FP32 state, then rounds that state to BF16 before the next
token. This is not the CF-BI chunk contract and does not promise bitwise equality with
original `fwd`. Only no_grad forward is supported; backward, MTP and full-model logits
guarantees are outside scope.

## Installation and use

Follow the [decode installation instructions](consistent-decode.en.md). The validated
environment is B300, CUDA 13.3, PyTorch 2.13.0a0, CuTe DSL 4.6.2 and apache-tvm-ffi 0.1.12.
Other GPUs or software combinations require numerical revalidation.

```bash
python examples/consistent_prefill.py
```

```python
plan = flash_kda.prepare_consistent_prefill(
    q, k, v, g, beta, A_log, dt_bias, state_in, out, state_out,
    scale=128**-0.5, eps=1e-6, lower_bound=-5.0,
    mode="auto", workspace_limit_bytes=64 * 1024 * 1024,
)
with torch.no_grad():
    plan.run()
```

`consistent_prefill_fwd` accepts the same arguments and prepares and executes one call.
For repeated calls and CUDA Graphs, use a prepared plan to avoid repeated workspace
allocation. `run()` and the convenience function return `(out, state_out)`; prepare
returns the plan.

## Tensor contract

| Argument | Shape | Dtype |
|---|---|---|
| q/k/v/g/out | `[B,L,H,128]` | BF16 |
| beta | `[B,L,H]` | BF16 |
| A_log | `[H]` | FP32 |
| dt_bias | `[H,128]` | FP32 |
| state_in/state_out | `[B,H,128,128]`, last axes V,K | BF16 |

All tensors must be contiguous on the same CUDA device, with nonempty data pointers
aligned to 32 bytes. Require `H>0`, `L<2^31` and `B*H*64<2^31`. The operator performs
Q/K L2 normalization; g and beta are raw activations. Gate, sigmoid and scalar
validation/defaults follow the [recurrent contract](consistency.en.md).

`B=0` is a no-op. `L=0` has empty output and copies the initial state to the final state.
State input/output may alias exactly; all other overlaps, including partial state
aliasing, are rejected. Separate initial state is read-only. This API handles dense,
equal-length sequences; use the general variable-length API for different request
lengths or device-resident dynamic length metadata.

## Scheduling, workspace and lifetime

- `mode="direct"`: one fused kernel processes the entire sequence, without coefficient workspace.
- `mode="staged"`: parallel coefficient preparation followed by state scanning. If the full sequence exceeds the workspace limit, windows reuse the same workspace without changing per-token rounding or outputs.
- `mode="auto"`: one token reuses dedicated decode; short sequences use direct; longer sequences use staged. Scheduling does not change the numerical contract.

The plan owns its workspace. `plan.workspace_bytes` reports actual allocated bytes,
never exceeding `workspace_limit_bytes`. Staged execution needs at least `B*H*1568`
bytes; otherwise prepare raises ValueError. Empty batches and direct execution require
no workspace. Run performs no allocation or layout conversion. Timing must include
coefficient preparation and every scanning window.

Prepare outside Graph capture. Tensor addresses, shapes, strides, dtypes, devices,
requires_grad flags and scalar parameters are fixed for a plan. Contents may change;
rebinding requires another prepare. Compatible plans reuse compiled code; exact batch
and sequence length are not compilation keys. Select the bound GPU's current device
and stream, and synchronize across streams explicitly. Do not concurrently reuse or
modify a plan's workspace, outputs or state. Keep the plan and bound storage alive
throughout Graph use. Segmentation, PD continuation and whole-head sharding transfer
the complete BF16 continuation state.

Invalid arguments raise TypeError/ValueError; violating no_grad or preparing during
capture raises RuntimeError. NaN/Inf are outside valid inputs and acceptance results;
run does not add an input-data scan.

## Validation and benchmarking

```bash
FLASH_KDA_LONG_TESTS=1 python -m pytest -q tests/consistency
python -m tests.consistency.accuracy --prefill
python benchmarks/bench_consistent_prefill.py --output results-prefill
```

Install the original FlashKDA extension for benchmarking. The harness compares fused,
single-kernel direct, single-token replay, generic recurrent and original FlashKDA.
Complete outputs, final states, required conversions and helpers are included, using
same-GPU, fixed-address warm-cache, alternating paired samples and Graph/eager timing.
Single-token replay is not used as the fused prefill performance estimate. Acceptable
performance regression thresholds remain undecided.

## Measured results

2026-11-05, same B300 and software stack above, driver 615.71.10. Full-API Graph medians from nine alternating paired samples:

| H | B | L | auto (ms) | direct (ms) | FlashKDA (ms) |
|---|---|---|---|---|---|
| 6 | 1 | 8192 | 4.888 | 6.276 | 0.768 |
| 6 | 8 | 8192 | 5.687 | 10.596 | 0.909 |
| 6 | 64 | 8192 | 22.382 | 38.885 | 2.870 |
| 12 | 1 | 4096 | 2.506 | 3.329 | 0.401 |
| 12 | 1 | 32768 | 20.216 | 33.392 | 3.155 |

[Raw samples and validation](../benchmarks/data/consistent-prefill-b300.json) include eager, single-token replay, generic recurrent and decode comparisons. Auto is 1.60–3.04× faster than the previous generic implementation and 2.43–4.25× faster than token replay, while remaining 6.24–7.80× slower than original FlashKDA. Direct uses one kernel per sequence call. Under the default workspace cap, auto uses 4/20/148/4/20 kernels for the five cases above.

All 269 tests passed, including whole-head sharding and state handoff across two GPUs. Independent FP64 diagnostics: output max_abs 2.7157e-5 and RMS 1.3609e-6; final state max_abs 6.1035e-5 and RMS 1.9618e-7. These are observed errors, not new acceptance thresholds. Decode code is unchanged; a fresh paired measurement showed approximately −7.54% to +0.06% latency change relative to pinned FlashInfer. No performance-acceptance or SOL claim is made.
