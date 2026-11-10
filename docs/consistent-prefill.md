# Dense prefill 与 checkpoint forward

[English](consistent-prefill.en.md) · [统一 varlen prefill](unified-prefill.md) · [Q=1 decode](consistent-decode.md)

`prepare_consistent_prefill(...).run()` 支持 dense `[B,L,H,128]` 输入，可选输出每个 token
之后的 BF16 checkpoint。计算和状态舍入遵循[统一 recurrent 契约](unified-prefill.md)：
FP32 计算、每 token BF16 状态舍入，与自身 prefill/Q=1/多 token 路径逐位一致。
checkpoint 可用于选择接受前缀后续算；调用方负责接受数选择与框架集成。
仅 no_grad forward，不包含 backward 或完整模型保证。

安装、当前平台、标量语义及通用生命周期见[统一 prefill](unified-prefill.md)。

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

`consistent_prefill_fwd` 接受相同参数，包含 prepare 和一次 run；重复调用和 Graph 使用
prepared plan。最小可运行示例：`python examples/consistent_prefill.py`。

## 数据与 checkpoint

| 参数 | shape | dtype |
|---|---|---|
| q/k/v/g/out | `[B,L,H,128]` | BF16 |
| beta | `[B,L,H]` | BF16 |
| A_log / dt_bias | `[H]` / `[H,128]` | FP32 |
| state_in/state_out | `[B,H,128,128]`，V、K 顺序 | BF16 |
| checkpoint_states | `[P,H,128,128]` | BF16 |
| checkpoint_indices | `[B,L]` | CUDA Int32 |

所有 tensor 同 GPU、连续，非空地址按 32 字节对齐；`H>0`、`L<2^31`、`B*H*64<2^31`。
两个 checkpoint 参数必须同时提供或同时省略。启用时 `B*L <= P < 2^31`，每个有效
index 必须位于 `[0,P)` 且彼此唯一。**调用方保证 index 内容有效**，run 不作 host 读取。
未使用 pool slot 不写入；写入 slot 是对应 token 完成后的完整 BF16 状态。

接受 a 个 token 后，a=0 使用原始 state_in；a>0 使用
`checkpoint_states[checkpoint_indices[request,a-1]]`，再执行剩余 token。
启用 checkpoint 时 state_out 可以为 None；run 仍返回 `(out, None)`，不省略任何有效
checkpoint。所有 tensor 禁止重叠，必须保留初态以支持接受零个 token。
无 checkpoint 时 state_in/state_out 允许完全原地重合，其他重叠拒绝。

B=0 无操作；L=0 时有终态则复制初态，checkpoint 不写入。非原地初态保持不变。
不同请求长度使用 `prepare_unified_prefill`；本接口的 L 为所有请求共同的 Q 长度。

## 模式、容量与迁移

无 checkpoint 的默认 `mode="auto"` 使用统一 prefill，不再按 L 阈值选择实例。
`mode="direct"` / `"staged"` 保留为显式兼容模式；选择由调用方固定，不能按每请求长度分派。
direct 无系数 workspace；staged 使用有界窗口且至少需要 `B*H*1568` 字节。
默认 auto 的最小容量要求见统一 prefill；B/H/长度不进入其 compile key。
启用 checkpoint 时 Q 和有限调度配置可决定编译实例；档内精确 B 和 index 内容不进入 key。

plan 持有 workspace，`workspace_bytes` 报告实际内部容量，不超过上限。
prepare 必须在 capture 外，run 不分配。调用方选择绑定 device/current stream，负责跨 stream
同步，禁止同一 plan 并发执行；Graph 生命周期内保持 plan 及绑定存储有效。
错误类型及 NaN/Inf 约束同统一 prefill。Q=1 延迟敏感调用使用 `prepare_consistent_decode`。

```bash
python -m pytest -q tests/consistency/test_prefill.py tests/consistency/test_checkpoints.py tests/consistency/test_unified_prefill.py
```

[早期性能数据](../benchmarks/data/consistent-prefill-b300.json) 对应旧实现，不能代表当前默认路径。
