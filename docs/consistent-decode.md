# Dense 单 token 一致性 decode

[English](consistent-decode.en.md)

`prepare_consistent_decode` 提供与 `consistent_fwd` 相同数值契约的专用单 token 路径。
每个请求恰好有一个有效 token；适合 decode，不用于变长 prefill、padding、MTP 或 backward。
它采用固定 FlashInfer `c3af0c13328f` 的 D128 数值路径，不依赖 FlashInfer 安装包。

## 安装与验证平台

按[一致性接口安装说明](consistency.md)安装 `.[consistency,test]`，不需要编译原 FlashKDA C++ 扩展。
当前依赖和已测软件组合见[统一 prefill 安装说明](unified-prefill.md)。
[早期验证与性能记录](consistent-decode-validation.md) 保留其原始软件版本，不代表当前版本的计时。

```bash
FLASH_KDA_SKIP_CUDA_BUILD=1 python -m pip install --no-build-isolation '.[consistency,test]'
python examples/consistent_decode.py
```

## API 与数据契约

```python
plan = flash_kda.prepare_consistent_decode(
    q, k, v, g, beta, A_log, dt_bias, state_in, out, state_out,
    scale=128 ** -0.5, eps=1e-6, lower_bound=-5.0,
)
out, state_out = plan.run()
```

| 参数 | dtype | shape |
|---|---|---|
| `q/k/v/g/out` | BF16 | `[B,1,H,128]` |
| `beta` | BF16 | `[B,1,H]` |
| `A_log` | FP32 | `[H]` |
| `dt_bias` | FP32 | `[H,128]` |
| `state_in/state_out` | BF16 | `[B,H,V=128,K=128]` |

全部 tensor 必须为同一 CUDA device 上的连续 tensor，非空 data pointer 必须按 32 bytes 对齐。
普通 `torch.empty` 分配满足；连续切片不一定满足，不对齐时准备接口报 `ValueError`。
`B>=0`、`H>0`、`B*H*8<2**31`。`B=0` 不发射 kernel，返回原输出缓冲区。
`g/beta` 为未激活 logits；输入须为有限值。标量必须可表示为有限 FP32，`eps>0`、`-5<=lower_bound<=0`。
Q/K 在 kernel 内归一化。输出使用更新后的 FP32 状态；续算状态随后舍入为 BF16。
同一软件/GPU 环境中，与 `consistent_fwd` 续算时须保持相同标量参数及 head 对应关系。

只允许 `state_in` 与 `state_out` 完全原地重合，其余 tensor 不得重叠。分离状态时，输入状态不被修改。
函数无 autograd 支持；输入需要梯度时，仅允许在 `torch.no_grad()`/`torch.inference_mode()` 下运行。
shape、dtype、布局、对齐、重叠、标量不合法时在准备阶段报错，不读取设备数据进行验证。

## 生命周期、stream 与 CUDA Graph

- 在 capture 外准备 plan；它验证参数、编译并保留 tensor 引用。直接构造 `ConsistentDecodePlan` 不受支持。
- `plan.run()` 使用当前 stream。调用方必须选中与绑定 tensor 相同的 CUDA device，并负责跨 stream 同步。
- plan 生命周期内不得改变绑定 tensor 的 storage、shape、stride、dtype、device、`requires_grad` 或标量。
  可以在同一 stream 上改写 tensor 内容。重绑定地址、改变 batch/head 数或标量时创建新 plan。
- 不同 B/H 的 plan 复用同 device/架构的编译产物；请求数据、指针、stream 不进入 compile key。
- 缓冲区均归调用方所有；没有额外 workspace。状态输入/输出分离时，调用方须在下一步绑定更新后的状态；
  原地状态可直接连续 `run()`。不能让并发调用写同一输出/状态。
- 可 capture `plan.run()`；Graph 固定地址和形状，允许更新缓冲区内容。形状改变需要重新 capture。
- `consistent_decode_fwd` 接受相同参数，等价于准备后运行，适合便利调用；热循环使用预先准备的 plan，避免重复主机检查。

## Prefill、请求重排与完整 head 分片

Dense 等长 prefill 使用 [`prepare_consistent_prefill`](consistent-prefill.md)；变长 prefill 使用
`consistent_fwd`。后者的 `[1,T,H,128]` packed 输出布局与本接口不同，两者均共享 BF16 状态布局和数值契约，
无需转换状态数值即可续算。只迁移状态不能迁移请求身份：调用方须同步重排 token 和状态的 batch 维。
完整 head 分片时同步切分所有 head 相关 tensor，并保证新的连续缓冲区对齐；不支持拆分 K/V 维。
本接口没有 `cu_seqlens`、slot gather 或 inactive mask；空请求应由调用方从 dense batch 中移除。
通信与框架接入不包含在本接口中。

## 测试与 benchmark

```bash
python -m pytest -q tests/consistency/test_decode.py
FLASH_KDA_LONG_TESTS=1 python -m pytest -q tests/consistency
python -m tests.consistency.accuracy --decode
python -m benchmarks.bench_consistent_decode --output /tmp/decode-benchmark
```

输出目录必须不存在。benchmark 将完整 64 次 T=1 调用与固定 FlashInfer 提取 kernel 配对，输出 Graph 节点、
逐步原始位检查、原始计时样本和软件信息。Graph/eager 分开报告；编译、验证、预分配不计入热路径。
采用固定地址预热协议，不是 cold-L2。FlashInfer 对照不是其安装包的完整 Python dispatcher。
目前不预设性能达标阈值，不将单卡算子结果作为完整模型或多卡通信性能结论。
