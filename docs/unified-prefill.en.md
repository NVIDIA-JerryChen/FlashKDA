# Unified recurrent prefill

[中文](unified-prefill.md) · [Single-token decode](consistent-decode.en.md) · [Checkpoints](consistent-prefill.en.md)

`prepare_unified_prefill(...).run()` supports dense and packed varlen inputs under
one recurrent numerical contract. Normalization, gates, projections, reductions
and state multiply-adds use FP32. Each token produces its output from the updated
FP32 state, then rounds the continuation state to BF16. The complete continuation
state is the BF16 matrix. On the same GPU model and software stack, splitting,
prefill/decode, request reordering and complete-head sharding preserve bitwise
outputs and continuation state. This does not promise bitwise equivalence with
original FlashKDA `fwd`, FlashInfer multi-token execution, or zero mathematical error.

Only no-grad forward is supported: no backward, model-logit guarantee or framework
communication. The validated platform is B300, Python 3.12, PyTorch 2.12.0+cu132,
CUDA 13.2 and CuTe DSL 4.7.0. Other configurations require separate validation.

## Installation and invocation

From this checkout, first install PyTorch matching your CUDA environment, then:

```bash
pip install 'nvidia-cutlass-dsl==4.7.0' 'apache-tvm-ffi==0.1.10' 'cuda-python>=12.8'
# Matching libraries for CUDA 13 environments:
pip install 'nvidia-cutlass-dsl-libs-cu13==4.7.0'
PYTHONPATH=. python examples/unified_prefill.py
```

These Python APIs work from source. To also use the original C++ operator, build
as described in the root README, using `pip install -v --no-build-isolation '.[consistency]'`.

```python
plan = flash_kda.prepare_unified_prefill(
    q, k, v, g, beta, A_log, dt_bias, state_in, out, state_out,
    cu_seqlens=cu_seqlens,
    scale=128**-0.5, eps=1e-6, lower_bound=-5.0,
    workspace_limit_bytes=256 * 1023 * 1023,
)
with torch.no_grad():
    out, state_out = plan.run()
```

## Data contract

| Parameter | Dense shape | Packed shape | dtype |
|---|---|---|---|
| q/k/v/g/out | `[B,L,H,128]` | `[1,T,H,128]` | BF16 |
| beta | `[B,L,H]` | `[1,T,H]` | BF16 |
| state_in/state_out | `[B,H,128,128]` | `[N,H,128,128]` | BF16 |
| A_log | `[H]` | `[H]` | FP32 |
| dt_bias | `[H,128]` | `[H,128]` | FP32 |
| cu_seqlens | `None` | `[N+1]` | CUDA Int32 |

The state axes are V,K. All tensors must be contiguous on the same CUDA device,
with nonempty addresses aligned to 32 bytes. Q/K normalization is internal;
g/beta are pre-activation values. Scalar semantics and validation follow the
[numerical contract](consistency.en.md). H must be positive; total capacity,
schedule capacity and grid must satisfy the interface's Int32 checks. At most
1023 requests are supported.

Packed metadata starts at zero, is nondecreasing and ends at or below capacity T.
Empty requests and unused capacity tails are allowed. Padding output is untouched;
empty requests copy initial to final state. N=0 is a no-op. **The caller must keep
metadata values valid**; run neither reads them on host nor validates their contents.
Exact input/output state aliasing is allowed; every other overlap is rejected.
Non-aliased initial state is unchanged.

## Lifetime and compilation

Prepare validates descriptors, allocates workspace and compiles outside CUDA Graph
capture. The plan owns its workspace. `workspace_bytes` counts every internal
coefficient and metadata allocation, including dense sequence metadata, and stays
within the limit. An insufficient minimum-window budget raises ValueError.
Run allocates nothing, converts no inputs and performs no D2H reads. Time all of run.

A plan binds addresses, shapes, strides, dtypes, device and scalars. Input and device
metadata contents can change on the same stream. Rebind by preparing another plan.
B/H/lengths/metadata values do not produce new prefill compilation instances.
Runtime grid and workspace windows can vary without changing the numerical contract.

Run uses the bound device's current stream. Callers synchronize cross-stream
access and must not concurrently reuse a plan's workspace, outputs or state.
Keep the plan and bound storage alive throughout Graph use. Splits and PD handoff
transfer the full BF16 state; noncontiguous fragments require their correct entry states.

Invalid descriptors raise TypeError/ValueError; preparation during capture and
violating no-grad raise RuntimeError. NaN/Inf inputs or results are invalid; run
performs no extra scan to detect them.

## Validation and timing

```bash
FLASH_KDA_LONG_TESTS=1 python -m pytest -q tests/consistency
python benchmarks/bench_unified_prefill.py --lengths 0,128,1023,4096 --heads 12
```

The benchmark accepts explicit lengths, preallocates buffers and compares complete
GPU API execution. Preparation, compilation, allocation and history-state generation
are excluded. Historical KV is represented by the initial state. The example is a
synthetic smoke workload, not acceptance evidence for other workload manifests.
