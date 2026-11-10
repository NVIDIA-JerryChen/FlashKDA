# Dense prefill and checkpoint forward

[中文](consistent-prefill.md) · [Unified varlen prefill](unified-prefill.en.md) · [Q=1 decode](consistent-decode.en.md)

`prepare_consistent_prefill(...).run()` accepts dense `[B,L,H,128]` inputs and can
write a BF16 checkpoint after every token. It follows the [unified recurrent
contract](unified-prefill.en.md): FP32 computation, per-token BF16 state rounding,
and bitwise agreement across its prefill/Q=1/multi-token paths. Checkpoints allow
continuation from an accepted prefix; the caller selects acceptance and integrates
with the framework. Only no-grad forward is supported, without backward or model guarantees.

Installation, validated platform, scalar semantics and common lifetime rules are
in the [unified prefill guide](unified-prefill.en.md).

```python
plan = flash_kda.prepare_consistent_prefill(
    q, k, v, g, beta, A_log, dt_bias, state_in, out, state_out,
    scale=128**-0.5, eps=1e-6, lower_bound=-5.0,
    mode="auto", workspace_limit_bytes=64 * 1024 * 1024,
    checkpoint_states=None, checkpoint_indices=None,
)
with torch.no_grad():
    out, state_out = plan.run()
```

`consistent_prefill_fwd` takes the same arguments and includes preparation plus
one run. Repeated calls and Graph capture should use a prepared plan. Runnable
example: `python examples/consistent_prefill.py`.

## Data and checkpoints

| Parameter | Shape | dtype |
|---|---|---|
| q/k/v/g/out | `[B,L,H,128]` | BF16 |
| beta | `[B,L,H]` | BF16 |
| A_log / dt_bias | `[H]` / `[H,128]` | FP32 |
| state_in/state_out | `[B,H,128,128]`, V,K order | BF16 |
| checkpoint_states | `[P,H,128,128]` | BF16 |
| checkpoint_indices | `[B,L]` | CUDA Int32 |

All tensors share a CUDA device, are contiguous, and have 32-byte aligned nonempty
addresses. H>0, L<2^31 and B*H*64<2^31. Supply both checkpoint arguments or neither.
When enabled, B*L <= P < 2^31; every effective index must be unique and in [0,P).
**Callers guarantee index contents are valid**; run performs no host reads.
Unused slots are untouched. A written slot holds the full BF16 state after that token.

After accepting a tokens, use initial state for a=0, otherwise
`checkpoint_states[checkpoint_indices[request,a-1]]`, then execute remaining tokens.
With checkpoints, state_out may be None; run returns `(out, None)` while writing all
valid checkpoints. All tensor overlap is forbidden. Preserve initial state for
zero-token acceptance. Without checkpoints, exact input/output state aliasing is
allowed; other overlap is rejected.

B=0 is a no-op. L=0 copies initial to final state when supplied and writes no
checkpoints. Non-aliased initial state is unchanged. Use `prepare_unified_prefill`
for unequal request lengths; this API uses the same Q=L for every request.

## Modes, capacity and migration

Without checkpoints, default `mode="auto"` uses unified prefill, with no L-threshold
instance selection. Explicit `"direct"` / `"staged"` remain compatibility modes;
callers must fix the selection rather than dispatch by per-request length.
Direct needs no coefficient workspace; staged uses bounded windows and needs at
least B*H*1568 bytes. Auto follows unified-prefill minimum capacity requirements;
B/H/lengths do not enter its compile key. Checkpoint execution may specialize on
Q and a finite scheduling configuration, but not exact B within a tier or index contents.

The plan owns workspace; `workspace_bytes` reports actual internal capacity within
the limit. Prepare outside capture; run allocates nothing. Callers choose the bound
device/current stream, synchronize cross-stream access and avoid concurrent plan
use. Keep plans and bound storage alive for Graph use. Error types and NaN/Inf
rules follow unified prefill. Use `prepare_consistent_decode` for latency-sensitive Q=1.

```bash
python -m pytest -q tests/consistency/test_prefill.py tests/consistency/test_checkpoints.py tests/consistency/test_unified_prefill.py
```

[Historical performance data](../benchmarks/data/consistent-prefill-b300.json)
describes the old implementation, not the current default path.
