# 统一 recurrent prefill

[English](unified-prefill.en.md) · [单 token decode](consistent-decode.md) · [checkpoint](consistent-prefill.md)

`prepare_unified_prefill(...).run()` 支持 dense 和 packed varlen 输入，使用同一 recurrent
数值契约。归一化、gate、投影、归约和状态乘加使用 FP32；每个 token 用更新后的 FP32
状态生成输出，再把续算状态舍入到 BF16。完整续算状态只有 BF16 状态矩阵。
同 GPU 型号、同软件栈下，任意切段、prefill/decode、请求重排和完整 head 分片保持
有效输出及完整续算状态逐位一致。这不保证与原 FlashKDA 的 `fwd` 或 FlashInfer
多 token 路径逐位相同，也不保证独立实数数学结果零误差。

范围为 no_grad forward；不含 backward、完整模型 logits 或框架通信。
当前验证平台为 B300，Python 3.12、PyTorch 2.12.0+cu132、CUDA 13.2、CuTe DSL 4.7.0。
其他硬件和软件组合需要独立验证。

## 安装和调用

在本仓库 checkout 中，先安装与 CUDA 环境匹配的 PyTorch，再安装：

```bash
pip install 'nvidia-cutlass-dsl==4.7.0' 'apache-tvm-ffi==0.1.10' 'cuda-python>=12.8'
# CUDA 13 environments also require the matching DSL CUDA 13 libraries.
pip install 'nvidia-cutlass-dsl-libs-cu13==4.7.0'
PYTHONPATH=. python examples/unified_prefill.py
```

上述 Python 接口可直接从源码使用；同时需要原 FlashKDA C++ 算子时，按根 README
编译安装，使用 `pip install -v --no-build-isolation '.[consistency]'` 安装相应依赖。

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

## 数据契约

| 参数 | dense shape | packed shape | dtype |
|---|---|---|---|
| q/k/v/g/out | `[B,L,H,128]` | `[1,T,H,128]` | BF16 |
| beta | `[B,L,H]` | `[1,T,H]` | BF16 |
| state_in/state_out | `[B,H,128,128]` | `[N,H,128,128]` | BF16 |
| A_log | `[H]` | `[H]` | FP32 |
| dt_bias | `[H,128]` | `[H,128]` | FP32 |
| cu_seqlens | `None` | `[N+1]` | CUDA Int32 |

状态末两维为 V、K。所有 tensor 在同一 CUDA device、连续布局，非空地址对齐 32 字节。
q/k 在内部作 L2 归一化；g/beta 是激活前输入。scale、eps、lower_bound 的语义与约束见
[数值契约](consistency.md)。`H>0`；总容量、调度容量和 grid 必须满足接口的 Int32
边界检查。最多 1023 个请求。

Packed metadata 必须从零开始、非递减、末项不大于容量 T；允许空请求及未使用的容量尾部。
有效 token 之外的 out 不写入；空请求的终态等于初态。`N=0` 为无操作。
metadata 的**内容由调用方保证有效**，接口不会 D2H 读取或在每次 run 检查。
state_in/state_out 允许完全重合，其他重叠均拒绝；非原地初态不会修改。

## 生命周期和编译缓存

prepare 校验描述符、分配 workspace 并编译，必须在 CUDA Graph capture 外调用。
plan 持有 workspace；`workspace_bytes` 包括全部内部系数和 metadata，不超过上限。
上限不足以容纳最小窗口时抛出 ValueError。dense 模式还包括内部 cu_seqlens。
run 不分配内存、不转换输入、不读取设备数据到 host，计时须包含完整 run。

plan 绑定地址、shape、stride、dtype、device 和标量；输入及设备 metadata 内容可以在
同一 stream 更新。改变绑定需重新 prepare。B/H/长度/metadata 内容不触发新的
prefill 编译实例，运行时 grid 和 workspace 窗口可以变化，数值契约不变。

run 在绑定 device 的 current stream 提交；跨 stream 依赖由调用方同步。同一 plan
的 workspace、输出和状态不得并发使用。Graph 生命周期内须保留 plan 及所有绑定存储。
分段和 PD 续算传递完整 BF16 状态；非连续片段应分别提供正确入口状态。

参数错误为 TypeError/ValueError；capture 内 prepare 或不满足 no_grad 为 RuntimeError。
NaN/Inf 不属于有效输入或验收结果，run 不额外扫描内容。

## 检查和计时

```bash
FLASH_KDA_LONG_TESTS=1 python -m pytest -q tests/consistency
python benchmarks/bench_unified_prefill.py --lengths 0,128,1023,4096 --heads 12
```

benchmark 使用显式长度清单、预分配输出与状态、完整 run 的配对 GPU 计时。
准备、编译、分配及历史前缀状态生成在计时外；KV 历史由输入状态表示。
默认例子是合成 smoke workload，不代表其他已确认 workload 的性能验收。
