# FP32-state recurrent forward

[简体中文](fp32-state.md)

`prepare_fp32_prefill` and `prepare_fp32_decode` implement one no-grad forward
contract with BF16 inputs/outputs and FP32 arithmetic, state and checkpoints.
They support packed varlen prefill, training-shaped forward workloads, Q1 decode
and Q8 speculative decode. Backward, specialized decode for other Q, automatic
acceptance selection, framework communication and full-model logits guarantees
are outside this API.

## Installation and platform

The target is SM100/SM103 Blackwell. Validation uses B300, Python 3.12,
PyTorch 2.12.0+cu132, CUDA 13.2 and CuTe DSL 4.8.1; other stacks need validation.
Install a matching PyTorch first, then from this checkout:

```bash
pip install 'nvidia-cutlass-dsl==4.8.1' 'nvidia-cutlass-dsl-libs-base==4.8.1' \
  'nvidia-cutlass-dsl-libs-core==4.8.1' 'nvidia-cutlass-dsl-libs-cu13==4.8.1' \
  'apache-tvm-ffi==0.1.10' 'cuda-python>=12.8'
FLASH_KDA_SKIP_CUDA_BUILD=1 pip install --no-build-isolation '.[fp32,test]'
python -m examples.fp32_state
```

These APIs work without the original CUDA extension. See the root README to
install that extension separately.

## API and data

Both prepare functions begin with `q,k,v,g,beta,A_log,dt_bias,state_in,out`.
Defaults: `scale=128**-0.5`, `eps=1e-6`, `lower_bound=-5.0`.
All tensors must be contiguous, on the same CUDA device, with nonempty pointers
aligned to 32 bytes. D is 128 and H must be positive. Q/K/V/g/out and beta are
BF16; A_log, dt_bias, state and checkpoints are FP32. Both g and beta are
pre-activation values; sigmoid is applied to beta internally.

| Parameter | Prefill | Decode |
|---|---|---|
| q/k/v/g/out | dense `[B,L,H,128]` or packed `[1,T,H,128]` | `[B,Q,H,128]`, Q=1 or 8 |
| beta | matching first three dimensions | `[B,Q,H]` |
| A_log / dt_bias | `[H]` / `[H,128]` | same |
| state_in | `[N,H,128,128]`, last axes V,K | `[B,H,128,128]` |
| state_out | required, same shape as state_in | required for Q1; must be None for Q8 |
| checkpoint_states | unsupported | None for Q1; required `[B,8,H,128,128]` for Q8 |
| cu_seqlens | Int32 `[N+1]` for packed; omit for dense, N=B | unused; each request has fixed Q |

Prefill has one entry even for L=1; it does not dispatch to decode by length.
At most 1023 requests are supported. Token capacity, schedule capacity and grid
must satisfy the API's Int32 bounds. Empty batches and requests are valid.
Packed metadata must start at zero, be nondecreasing and end at or below T.
Valid tokens are packed contiguously and output padding remains unwritten.
Callers enforce metadata validity; neither prepare nor run reads device lengths
back to the host. The complete FP32 state represents KV history.

`.run()` returns `(out, state_out)`. For Q8 the terminal state is the usually
noncontiguous view `checkpoint_states[:, -1]`. Copy it into a contiguous state
buffer, or call `.contiguous()`, before passing it to another invocation. For a
request accepting a tokens, use state_in when a=0, otherwise checkpoint a−1.
The caller performs this selection and includes it and required copies in any
end-to-end timing that needs them.

## Numerical contract

Q/K normalization, gate, beta, projections and state updates use FP32. Each step
decays state, computes prediction and delta, updates FP32 state, then computes
BF16 output from the updated state. State is never rounded to BF16 between
tokens or calls; Q8 stores the full FP32 state after every token. Elementary
functions include approximate exp2/reciprocal and are not correctly rounded
mathematical functions.

On the same GPU model, stack and scalar configuration, valid outputs and full
continuation states must be bitwise identical across full/split calls, Q1/Q8,
request reorderings, batch compositions and complete-head sharding. Matching
BF16 recurrent, FlashInfer or FlashKDA bit patterns is not required. Switching
BF16/FP32 contracts does not preserve an existing trajectory; explicitly select
one contract throughout each request. Independent FP64 error diagnostics remain
necessary even when internal bitwise comparisons pass.

## Ownership, replay and errors

Prepare validates types, dtypes, shapes, devices, contiguity, alignment, overlap
and scalars, then compiles and binds fixed-capacity buffers. Prepare must run
outside CUDA Graph capture. Gradient-requiring tensors with autograd enabled
raise RuntimeError; invalid structure/scalars raise TypeError or ValueError.
Scalars must be finite FP32-representable values, eps positive in FP32, and
lower_bound in `[-5,0]`.

Prefill's `workspace_limit_bytes` defaults to 256 MiB including owned metadata
and workspace, excluding caller inputs/outputs. Insufficient capacity raises an
error. Decode needs no additional workspace. Only exact state_in/state_out
aliasing in prefill/Q1 is allowed; all other overlap is forbidden. Q8 checkpoints
must be independent of state_in, preserving the acceptance-zero state.

Run allocates no device memory, compiles nothing, performs no D2H and enqueues on
the current stream of the tensors' device. A plan binds addresses, capacities
and scalars. Contents and valid device metadata can change in place; new shapes,
addresses or scalars require a new plan. Keep the plan and buffers alive through
completion. Synchronize before cross-stream reuse or updates; never concurrently
reuse a plan, its workspace or outputs. These rules also apply to Graph replay.
In-place terminal state updates make replay advance state instead of recomputing
the same prefix.

## Validation and timing

```bash
python -m pytest -q tests/consistency/test_fp32.py
python -m benchmarks.bench_fp32 --mode prefill --lengths 0,128,1024,4096 --heads 16
python -m benchmarks.bench_fp32 --mode decode --batch 64 --query-length 1 --heads 16
python -m benchmarks.bench_fp32 --mode decode --batch 64 --query-length 8 --heads 16
python -m tests.consistency.fp32_accuracy --length 32768 --gate open
```

Timing includes all prepared API GPU work and all eight Q8 checkpoint writes.
Preparation, compilation, prefix preparation, acceptance selection and host time
are excluded. Repeated fixed-address CUDA Graph timing is hot-address timing,
not a cold-L2 claim. Custom lengths support reproduction and diagnostics; formal
comparisons require identical explicit length manifests, entry states, output
obligations and paired measurements on the same node. Do not replace varlen
workloads with equal-length inputs.
