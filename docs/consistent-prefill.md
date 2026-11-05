# 融合 recurrent prefill

[English](consistent-prefill.en.md) · [decode](consistent-decode.md) · [通用变长接口](consistency.md)

`prepare_consistent_prefill(...).run()` 处理连续布局的整段输入，与
`prepare_consistent_decode` 使用相同的 recurrent 数值契约。每个 token 用更新后的
FP32 状态计算输出，然后将状态舍入成 BF16，供下一步使用。它不采用 CF-BI chunk 契约，
也不保证与原 `fwd` 的输出逐位相同。仅支持 no_grad forward，不支持 backward、MTP
或完整模型 logits 保证。

## 安装与调用

依赖与安装步骤见 [decode 安装说明](consistent-decode.md)。当前验证环境为
B300、CUDA 13.3、PyTorch 2.13.0a0、CuTe DSL 4.6.2、apache-tvm-ffi 0.1.12；
其他 GPU/软件组合需要重新验证数值契约。可运行示例：

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

`consistent_prefill_fwd` 接受相同参数，执行准备和一次调用。重复调用及 CUDA Graph
应使用 prepared plan，避免重复分配工作区。`run()` 和便利接口返回 `(out, state_out)`；prepare 返回 plan。

## 数据契约

| 参数 | shape | dtype |
|---|---|---|
| q/k/v/g/out | `[B,L,H,128]` | BF16 |
| beta | `[B,L,H]` | BF16 |
| A_log | `[H]` | FP32 |
| dt_bias | `[H,128]` | FP32 |
| state_in/state_out | `[B,H,128,128]`，后两维 V、K | BF16 |

所有 tensor 必须在同一 CUDA device、连续布局，非空 data pointer 按 32 字节对齐。
`H>0`、`L<2^31`、`B*H*64<2^31`。q/k 在算子内作 L2 归一化；g、beta 为原始
激活输入，gate 与 sigmoid 语义见 [recurrent 契约](consistency.md)。scale、eps、
lower_bound 沿用通用契约的检查与默认值。

`B=0` 为无操作；`L=0` 输出为空并将初态复制到终态。state_in/state_out 可完全
原地重合；其他重叠（含部分状态重叠）均拒绝。非原地初态不被修改。
本接口处理 dense 等长序列；不同请求长度和设备动态长度 metadata 使用通用变长接口。

## 调度、工作区与生命周期

- `mode="direct"`：一个融合 kernel 处理整段序列，不分配系数工作区。
- `mode="staged"`：先并行生成系数，再扫描序列状态。工作区不足以容纳全段时分窗口复用；窗口边界不改变逐 token 舍入和输出。
- `mode="auto"`：单 token 复用专用 decode，短段使用 direct，长段使用 staged。调度选择不改变数值契约。

工作区由 plan 持有，`plan.workspace_bytes` 是实际分配字节数，不超过
`workspace_limit_bytes`。staged 至少需要 `B*H*1568` 字节；不足时抛出 ValueError。
空请求和 direct 不需要工作区。run 不分配、不作输入布局转换，计时必须包含
系数生成和全部扫描窗口。

在 Graph capture 外 prepare。plan 固定 tensor 的地址、shape、stride、dtype、device、
requires_grad 和 scalar 参数；内容可以更新，改变绑定需重新 prepare。同一调度配置
复用编译产物，序列长度与精确 batch 不进入编译 key。
调用方选择绑定 GPU 的 current device/current stream，并负责跨 stream 同步；
同一个 plan 的工作区、输出和状态不得被并发调用或修改。Graph 生命周期内须保持 plan
及绑定存储有效。切段、PD 续算或完整 head 分片时，传递完整 BF16 状态。

参数错误抛出 TypeError/ValueError；违反 no_grad 或在 capture 内 prepare 抛出 RuntimeError。
NaN/Inf 不属于有效输入或验收结果，run 不额外扫描数据。

## 检查与 benchmark

```bash
FLASH_KDA_LONG_TESTS=1 python -m pytest -q tests/consistency
python -m tests.consistency.accuracy --prefill
python benchmarks/bench_consistent_prefill.py --output results-prefill
```

benchmark 要求已安装原 FlashKDA 扩展，同时测融合接口、单 kernel direct、逐 token
重放、通用 recurrent 和原 FlashKDA。包括完整输出和终态、所需转换与 helper；
使用同 GPU、固定地址 warm-cache、交替配对样本，并报告 Graph 与 eager。
不将逐 token 重放结果当作融合 prefill 的性能。性能容许退化比例尚未确定。

## 已测结果

2026-11-05，同一 B300、上述软件栈、driver 615.71.10，全 API Graph 的9组交替样本中位数：

| H | B | L | auto (ms) | direct (ms) | FlashKDA (ms) |
|---|---|---|---|---|---|
| 6 | 1 | 8192 | 4.888 | 6.276 | 0.768 |
| 6 | 8 | 8192 | 5.687 | 10.596 | 0.909 |
| 6 | 64 | 8192 | 22.382 | 38.885 | 2.870 |
| 12 | 1 | 4096 | 2.506 | 3.329 | 0.401 |
| 12 | 1 | 32768 | 20.216 | 33.392 | 3.155 |

[原始样本与验证记录](../benchmarks/data/consistent-prefill-b300.json) 包含 eager、逐 token 重放、通用 recurrent 和 decode 对照。auto 相比旧 generic 提速 1.60–3.04 倍，相比逐 token 重放提速 2.43–4.25 倍；仍比原 FlashKDA 慢 6.24–7.80 倍。direct 每段一次 kernel；auto 在默认工作区上限下，上述五个 case 分别为4/20/148/4/20个 kernel。

最终269项测试通过，涵盖两GPU的完整head分片和状态续传。独立 FP64 诊断：输出 max_abs 2.7157e-5、RMS 1.3609e-6，终态 max_abs 6.1035e-5、RMS 1.9618e-7；这是误差记录，不是新增验收阈值。decode代码未改，重新配对测得相对固定 FlashInfer 延迟变化约 −7.54% 至 +0.06%。尚未宣称性能达标或 SOL。
