# FlashKDA

FlashKDA: Flash Kimi Delta Attention — high-performance KDA kernels built on CUTLASS

## News

- **2026-04-22** — Deep-Dive Blog: the design decisions behind FlashKDA v1, read it [here](docs/20260420-flashkda-v1-deep-dive.md).

## Requirements
- SM90 and above
- CUDA 12.9 and above
- PyTorch 2.4 and above

## Installation
```bash
git clone https://github.com/MoonshotAI/FlashKDA.git flash-kda
cd flash-kda
git submodule update --init --recursive
pip install -v --no-build-isolation .
```

By default, the build detects the current CUDA device and compiles for that architecture. For wheel or CI builds, compile all supported architectures explicitly:

```bash
FLASH_KDA_CUDA_ARCHS=all pip install -v --no-build-isolation .
```

Supported values are `auto` (default), `all`, or a comma-separated arch list such as `90a,100a`.

## Using FlashKDA as an FLA backend

Once installed, FlashKDA is auto-dispatched from `flash-linear-attention`'s `chunk_kda`. See [fla-org/flash-linear-attention#852](https://github.com/fla-org/flash-linear-attention/pull/852) for integration details.

**Requirements**

1. Install `flash-linear-attention >= 0.5.0`:
   ```bash
   pip install -U flash-linear-attention
   ```
2. Call `chunk_kda` under `torch.inference_mode()` 
   ```python
   import torch
   from fla.ops.kda import chunk_kda

   with torch.inference_mode():
       out, final_state = chunk_kda(
           q=q, k=k, v=v, g=g, beta=beta,
           scale=scale,
           initial_state=h0,
           output_final_state=True,
           use_gate_in_kernel=True,
           use_qk_l2norm_in_kernel=True,
           use_beta_sigmoid_in_kernel=True,
           safe_gate=True,
           A_log=A_log, dt_bias=dt_bias,
           lower_bound=lower_bound,
           transpose_state_layout=True,
           cu_seqlens=cu_seqlens,
       )
   ```

**Opt out:** set `FLA_FLASH_KDA=0` to fall back to the Triton path.

**Debug dispatch:** add `logging.basicConfig(level=logging.INFO)` to see `[FLA Backend] kda.chunk_kda -> flashkda` on hit, or `... rejected: <reason>` on miss.

## Performance

See [BENCHMARK_H20.md](BENCHMARK_H20.md).

## Tests

```bash
bash tests/test.sh
```

- `tests/test_fwd.py` — correctness tests (exact match against the torch reference; compared with `flash-linear-attention`)


## Kernel API

### `flash_kda.fwd`

```python
flash_kda.fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
              initial_state=None, final_state=None, cu_seqlens=None)
```

**Parameters:**

| Parameter | Dtype | Shape | Description |
|---|---|---|---|
| `q` | bf16 | `[B, T, H, K]` | Query |
| `k` | bf16 | `[B, T, H, K]` | Key |
| `v` | bf16 | `[B, T, H, V]` | Value |
| `g` | bf16 | `[B, T, H, K]` | Gate before activation |
| `beta` | bf16 | `[B, T, H]` | Beta logits (pre-activation; sigmoid applied internally) |
| `scale` | float | scalar | scaling factor |
| `out` | bf16 | `[B, T, H, V]` | Output tensor |
| `A_log` | fp32 | `[H]` | Log-gate parameter |
| `dt_bias` | fp32 | `[H, K]` | Gate bias |
| `lower_bound` | float | scalar | Gate lower bound (range from -5.0 to 0) |
| `initial_state` | bf16/fp32/None | `[B, H, V, K]` or `[N, H, V, K]` | (optional) Initial recurrent state |
| `final_state` | bf16/fp32/None | `[B, H, V, K]` or `[N, H, V, K]` | (optional, output) Final recurrent state |
| `cu_seqlens` | int64 | `[N+1]` | (optional) Cumulative sequence lengths for variable-length batching |

- Currently requires `K = V = 128`.
- `initial_state` / `final_state` accept `None` (stateless), bf16, or fp32 tensors. When both are provided, their dtypes must match.
- When `cu_seqlens` is provided, `B` must be 1, `T` is the total length across all sequences, and `initial_state` / `final_state` have shape `[N, H, V, K]`.
- When `cu_seqlens` is `None`, each batch element is treated as an independent sequence, and the state shape is `[B, H, V, K]`.

## Bitwise train/infer-consistent kernels (`flash_kda.cfbi`)

`flash_kda.cfbi` evaluates ONE fixed-order, chunk-form floating-point program ("CF-BI": chunk 64 anchored at the
sequence start, fp32 state, three-term TF32 tensor-core products). Prefill and the training forward evaluate many
rows of it at once, decode evaluates one more row: the output and the continuation state have the **same bits**
however the sequence is cut into calls, whatever else is in the batch, and whether it ran as prefill, decode or
the training forward. Optimised kernels (bulk-copy/mbarrier pipelines, fragment-native workspace, packed decode tiles;
see [perf doc](docs/20261103-cfbi-perf.md)) that are checked bit for bit against the frozen v1 reference kernels
(`_impl=3`). Validated on B300 (`sm_103a`) and, with the caveats of
[docs/20261102-cfbi-design.md](docs/20261102-cfbi-design.md) section 7a, on B200, H100, A100, A10 and L40S (Ampere/Ada run the row
engine only, and their bits differ from Hopper/Blackwell: bit equality holds within one architecture family). The first call on a
device runs `flash_kda.cfbi.verify_hardware()`, an empirical check of the tensor-core premise.

```python
from flash_kda.cfbi import alloc_state, prefill, decode, chunk_kda

state = alloc_state(N, H)                       # E fp32 [N,H,128,128], open-chunk row cache, pos
prefill(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, state, cu_seqlens)   # any cut, in place
decode(q1, k1, v1, g1, beta1, scale, out1, A_log, dt_bias, lower_bound, state)          # one token per sequence
y = chunk_kda(q, k, v, g, beta, scale, A_log, dt_bias, lower_bound)                      # training: autograd

# one state POOL for a serving engine: sequence n of the call uses slot slot_ids[n] (-1 = padding entry)
pool = alloc_state(num_slots, H)
decode(q1, k1, v1, g1, beta1, scale, out1, A_log, dt_bias, lower_bound, pool, slot_ids=slots)   # int32 [N]

# context parallelism: K1 (no E needed) on every rank at once, K2 chained through E; same bits as one prefill call
from flash_kda.cfbi import cp_prefill       # shards cut at multiples of 64 tokens from the sequence start
```

Same argument names/shapes as `flash_kda.fwd`; extra constraints and the backward (FLA's chunk backward on a
recomputed forward, needs `flash-linear-attention`; `lower_bound` in `[-5, 0)` for gradients) are in the
[design doc](docs/20261102-cfbi-design.md); the floating-point program is specified in the
[numerical contract](docs/20261102-cfbi-numerical-contract.md).

```bash
FLASH_KDA_BUILD_EXTS=cfbi pip install -v --no-build-isolation .   # only the cfbi extension (no CUTLASS needed)
python -m pytest tests/cfbi -q                                    # JIT-builds the extension if it is not installed
CFBI_BIG=1 python -m pytest tests/cfbi -q                         # + 32K-token and N=512 runs
python benchmarks/accuracy_cfbi.py && python benchmarks/bench_cfbi.py
```

## Development

To set up IntelliSense (clangd) for the CUDA/C++ sources, run:

```bash
bash setup_clangd.sh
```

This generates a `.clangd` file with the correct repository paths and installs the global clangd `config.yaml` to `~/.config/clangd/`.

## Citation

```bibtex
@misc{flashkda2026,
      title={FlashKDA: Flash Kimi Delta Attention},
      author={Yutian Chen, Zhiyuan Li, Yucheng Wang, Ming Wei},
      year={2026},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/MoonshotAI/FlashKDA}},
}
```
