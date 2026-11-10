# Recurrent KDA consistency API

[简体中文](consistency.md) · [Home](../README.md)

`flash_kda.consistent_fwd` uses one recurrent numerical path for training forward,
prefill and decode. It supports no-grad forward only and rejects requests requiring
autograd. Existing `flash_kda.fwd` keeps its behavior. Bitwise guarantees require
identical GPU models, compiler/software versions, parameters and input bits.
Operator tests do not establish full-model consistency. Backward, CPR, Ulysses,
parallel prefix trees, convolution state and full-model integration are excluded.

## Installation and validation scope

Python >= 3.10, CUDA-enabled PyTorch and a CUDA toolkit are required. The initial
target is B300/GB300; see the [validation record](consistency-validation.md) for
actual coverage. Other architectures/stacks require fresh acceptance testing.

```bash
python -m pip install 'nvidia-cutlass-dsl==4.7.0' 'apache-tvm-ffi==0.1.10' 'cuda-python>=12.8' pytest
FLASH_KDA_SKIP_CUDA_BUILD=1 python -m pip install --no-build-isolation '.[consistency,test]'
python examples/consistent_fwd.py
```

Skipping the CUDA extension installs the Python/CuTe path only; `fwd` requires
building the C++ extension using the root installation instructions. Install the
`consistency` extra as well when using both interfaces. CuTe imports/compiles on
first use. Production does not depend on FlashInfer or a local reference checkout.

See [unified prefill](unified-prefill.en.md) for the current software stack and installation.

## API and data contract

```python
from flash_kda import consistent_fwd
out, state_out = consistent_fwd(
    q, k, v, g, beta, A_log, dt_bias, cu_seqlens, state_in, out, state_out,
    scale=128 ** -0.5, eps=1e-6, lower_bound=-5.0,
)
```

| Argument | dtype | Shape / meaning |
|---|---|---|
| q/k/v/g | BF16 | `[1,T,H,128]`; g is preactivation |
| beta | BF16 | `[1,T,H]`; raw logits |
| A_log | FP32 | `[H]` |
| dt_bias | FP32 | `[H,128]` |
| cu_seqlens | CUDA int64 | `[N+1]`; starts at zero, nondecreasing, last value <= T |
| state_in/state_out | BF16 | `[N,H,128,128]`, ordered `[N,H,V,K]` |
| out | BF16 | `[1,T,H,128]` |

All tensors must be contiguous on the same CUDA device; no extra pointer alignment
is required. H > 0, with equal Q/K/V head counts. T is capacity: padding after the
last offset is untouched. Empty sequences preserve state; N=0 and T=0 are supported.
Outputs/states are caller-owned. Exact state_in/state_out aliasing is permitted;
all other overlaps are rejected. Scalars must be finite host numbers representable in FP32, eps > 0 after FP32 conversion and
lower_bound in [-5,0]. Valid inputs and initial states must be finite; acceptance
rejects NaN/Inf.

Shape, dtype, device, contiguity, overlap and scalar errors raise synchronously.
Device-side asynchronous metadata checks avoid D2H synchronization, including in
graph replay. Invalid offsets cause a CUDA device assertion and may invalidate the
CUDA context. Callers must provide valid metadata. With grad enabled, any floating
tensor requiring grad causes RuntimeError.

## Numerical semantics

Each step converts inputs to FP32 and normalizes Q/K with a fixed reduction order
and denominator `sqrt(sum(x²)+eps)`; Q also receives scale. Beta uses sigmoid.
The mathematical gate is
`exp(lower_bound * sigmoid(exp(A_log) * (g + dt_bias)))`.
The state is gated across K, then updated by the delta rule. Output is computed
from the updated, unrounded FP32 state and converted to BF16. State is then rounded
to BF16; the next step restores FP32 from that rounded value. Mathematical formulas
do not replace the prescribed floating-point operation order. Results need not
match the existing chunk `fwd`.

`flash_kda.NUMERICAL_CONTRACT` identifies this path. Handoffs must also match scale,
eps, lower_bound, H, GPU model and software versions. Batch, lengths, metadata
contents, request ordering, addresses and streams are excluded from the compile key.

## Lifetime and CUDA Graph

Calls enqueue asynchronously on the current PyTorch CUDA stream. All buffers must
remain alive until completion. Establish explicit event/wait_stream dependencies
and follow PyTorch allocator lifetime rules across streams. In-place state advances
the recurrence; reset the initial state when repeating identical experiments.

Import, compile and warm up outside capture. Capture local computation with stable
addresses and tensor shapes. Within capacity, update cu_seqlens to represent empty
request slots and the active token count. Finish metadata/input updates before
replay; never modify buffers while in use. Shape changes require recapture but do
not cause a new compute specialization for N/T. CP communication stays outside capture.

## CP and PD

`flash_kda.continuation.cp_consistent_fwd` takes the same arguments plus `group=None`.
Split each request into contiguous time intervals in group-rank order, including
unequal or empty spans. All ranks use identical request order and state shapes.
Rank 0 supplies initial state; state_in on later ranks is a writable receive buffer.
NCCL transfers BF16 data without numerical merging. Each rank computes and sends
state_out to its successor. Only the last rank holds final state; outputs remain
on their producing ranks. Callers ensure correct partitions and collective
participation. This design has no sequence speedup target.

`Continuation(request_ids, positions, state, ...)` borrows BF16 state and stores
host positions, request identities and numerical parameters.
`rebase_continuations(parts, request_ids, state_out)` checks compatibility and
regroups states by identity without changing bits. Scalar compatibility compares the
FP32 bit patterns of scale, eps and lower_bound, including the distinction between
`+0` and `-0`. Supply new requests separately
at position zero with caller-initialized states; omit deleted IDs. Destination
storage must not overlap any source. Network transport is not included: the PD
caller transfers metadata/BF16 state and establishes stream dependencies. The core
needs no historical K/V; complete layers require additional convolution/etc. state.

## Acceptance

```bash
python -m pytest -q tests/consistency
FLASH_KDA_LONG_TESTS=1 python -m pytest -q tests/consistency
python -m tests.consistency.accuracy
torchrun --standalone --nproc-per-node=8 -m tests.consistency.distributed
```

The distributed command requires eight matching GPUs and NCCL; standalone
rendezvous owns its port. [Test notes](../tests/consistency/README.md) describe
reference provenance and first-difference diagnostics. FP64 errors are diagnostic;
raw-bit comparison of valid elements is the acceptance gate, never allclose.
