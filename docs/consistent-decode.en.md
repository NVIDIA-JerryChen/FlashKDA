# Dense single-token consistent decode

[简体中文](consistent-decode.md)

`prepare_consistent_decode` implements the same numerical contract as `consistent_fwd`
for exactly one valid token per request. It is for decode, not variable-length prefill,
padding, MTP, or backward. It follows the fixed FlashInfer `c3af0c13328f` D128 numerical
path without requiring the FlashInfer package.

## Installation and validation

Follow the [consistent forward installation instructions](consistency.en.md) for
`.[consistency,test]`; the original FlashKDA C++ extension is not required.
Dependencies include CuTe DSL 4.6.2, TVM FFI 0.1.x and cuda-python. Successful installation
on another version does not establish support. Validated with two NVLink-connected B300 GPUs,
Python 3.12.3, PyTorch 2.13.0a0+9186a08b2c.nv26.07, CUDA 13.3, driver 615.71.10, and
CuTe DSL 4.6.2. The full regression suite passed 154 tests; see the
[validation and performance record](consistent-decode-validation.en.md).

```bash
FLASH_KDA_SKIP_CUDA_BUILD=1 python -m pip install --no-build-isolation '.[consistency,test]'
python examples/consistent_decode.py
```

## API and data contract

```python
plan = flash_kda.prepare_consistent_decode(
    q, k, v, g, beta, A_log, dt_bias, state_in, out, state_out,
    scale=128 ** -0.5, eps=1e-6, lower_bound=-5.0,
)
out, state_out = plan.run()
```

| Argument | dtype | shape |
|---|---|---|
| `q/k/v/g/out` | BF16 | `[B,1,H,128]` |
| `beta` | BF16 | `[B,1,H]` |
| `A_log` | FP32 | `[H]` |
| `dt_bias` | FP32 | `[H,128]` |
| `state_in/state_out` | BF16 | `[B,H,V=128,K=128]` |

All tensors must be contiguous on one CUDA device with 32-byte aligned nonempty data
pointers. Ordinary `torch.empty` allocations satisfy this requirement; contiguous views
may not, and preparation rejects misaligned pointers. Require `B>=0`, `H>0` and
`B*H*8<2**31`. An empty batch launches no kernel and returns the supplied output buffers.
`g/beta` are raw logits. Inputs must be finite. Scalars must be finite FP32-representable
values, with `eps>0` and `-5<=lower_bound<=0`. Q/K normalization is performed in the kernel.
Output uses the updated FP32 state; continuation state is then rounded to BF16. Continuing
from `consistent_fwd` requires the same scalar parameters and head identities on the same
GPU/software stack.

Only exact aliasing of `state_in` and `state_out` is allowed; all other overlap is rejected.
With separate state buffers the input state is unchanged. Autograd is unsupported; tensors
requiring gradients may only be used under `torch.no_grad()` or `torch.inference_mode()`.
Preparation rejects invalid shapes, dtypes, layout, alignment, overlap and scalar parameters
without reading device data.

## Lifetime, streams and CUDA Graph

- Prepare outside capture. The plan validates, compiles and retains tensor references.
  Constructing `ConsistentDecodePlan` directly is unsupported.
- `plan.run()` uses the current stream. The caller must select the matching CUDA device
  and handle synchronization across streams.
- Do not change bound tensor storage, shape, stride, dtype, device, `requires_grad` or
  scalar parameters during the plan lifetime. Contents may change on the same stream.
  Prepare another plan when rebinding addresses, shapes or scalars.
- Plans with different B/H reuse compilation for the same device/architecture. Request
  contents, pointers and streams are not compilation keys.
- The caller owns every buffer; no additional workspace is required. With separate state
  buffers, bind the updated state for the next step. In-place state supports repeated
  `run()` calls. Concurrent calls must not write the same output/state.
- Capture `plan.run()` with fixed addresses/shapes. Contents may change between replays;
  changing shapes requires recapture.
- `consistent_decode_fwd` accepts the same arguments and prepares then runs. Prefer a
  prepared plan in hot loops to avoid repeated host validation.

## Prefill, request reorder and whole-head sharding

Use [`prepare_consistent_prefill`](consistent-prefill.en.md) for dense equal-length prefill
and `consistent_fwd` for variable-length prefill. The latter's packed `[1,T,H,128]` layout differs
from the dense decode layout; both share its BF16 state layout and numerical contract. State needs
no numerical conversion. The caller must reorder tokens and states together to preserve
request identity. Whole-head sharding must slice all head-dependent tensors and provide
aligned contiguous buffers. Splitting K/V dimensions is unsupported. There are no
`cu_seqlens`, slot gathers or inactive masks: remove empty requests from the dense batch.
Framework communication and integration are outside this API.

## Tests and benchmark

```bash
python -m pytest -q tests/consistency/test_decode.py
FLASH_KDA_LONG_TESTS=1 python -m pytest -q tests/consistency
python -m tests.consistency.accuracy --decode
python -m benchmarks.bench_consistent_decode --output /tmp/decode-benchmark
```

The output directory must not exist. The benchmark compares 64 separate T=1 calls against
the pinned extracted FlashInfer kernel and records graph nodes, per-step raw-bit checks,
raw samples and software versions. Graph and eager timings are separate; preparation,
compilation and validation are outside the hot path. This is a fixed-address warmed-cache
protocol, not cold-L2. The reference is not the complete installed FlashInfer dispatcher.
No acceptable-regression threshold has been set; single-GPU operator results do not establish
full-model or multi-GPU communication performance.
