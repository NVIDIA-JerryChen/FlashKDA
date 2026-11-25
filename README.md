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
- `tests/test_replay.py` — `flash_kda.replay` prefill/decode bit-for-bit against one `flash_kda.fwd` call over the whole sequence


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

### `flash_kda.replay`: decode that is bitwise equal to the prefill

`flash_kda.fwd` stays the prefill kernel (unchanged). `flash_kda.replay` adds a per-sequence state and a decode kernel
that replays `flash_kda.fwd`'s own chunk arithmetic for the rows of the chunk that is still open, so every output is
bit-for-bit the row of ONE `flash_kda.fwd` call over the whole sequence, however the sequence is cut into prefill /
decode calls. Design and limits: [docs/20261124-replay-decode.md](docs/20261124-replay-decode.md).

```python
from flash_kda import replay

state = replay.alloc_state(num_slots, H)               # or initial_state=[N, H, V, K] bf16/fp32
replay.prefill(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, state, cu_seqlens=cu)  # [1, T, H, 128]
replay.decode(q1, k1, v1, g1, beta1, scale, out1, A_log, dt_bias, lower_bound, state)          # [N, H, 128]
final = replay.final_state(state, A_log, dt_bias, lower_bound)   # == flash_kda.fwd's final_state; state unchanged
```

`slot_ids` (int32 `[N]`) maps the call's sequences to state slots (`-1` = padding entry in `decode`). The 16-aligned
part of a prefill runs through `flash_kda.fwd`; prompt tails and decode tokens run through the replay kernel.

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
