# Dense decode 验证与性能记录

[English](consistent-decode-validation.en.md) · [API 契约](consistent-decode.md)

2026-11-05，在 B300 上比较 `prepare_consistent_decode(...).run()` 与固定 FlashInfer
`c3af0c13328fd23b256c53b9b4f94d9e160d118c` 提取的单 token kernel。
新版 Graph 延迟相对参考变化为 **−7.58% 到 +1.37%**。这是本矩阵的实测结果；性能验收阈值仍待定，未声称 SOL。

## 环境与比较边界

- 两张 B300、同节点 NVLink 互联；正确性涉及两张卡，性能配对使用同一张卡、同一进程。
- Python 3.12.3，PyTorch 2.13.0a0+9186a08b2c.nv26.07，CUDA 13.3，驱动 615.71.10，CuTe DSL 4.6.2。
- BF16 输入、输出及原地续算状态，K=V=128，非零初态，`lower_bound=-5`。
- 每个周期包含 64 次独立 T=1 调用和 64 组不同 token 输入，没有跨 token 融合。
- 双方 Graph 都恰好包含 64 个 kernel、零 memcpy。计时前逐步检查所有输出和完整状态的原始位模式。
- 9 组交替顺序的配对样本；每组 Graph 重放 25 周期，eager 执行 5 周期；报告中位数。
  固定地址预热协议，未声明 cold-L2；未手动锁频。每组计时前恢复同一初态，恢复不计时。
- 新版计入完整 `plan.run()` 热路径。双方编译、输入生成、分配及参数绑定在计时外。
  FlashInfer 对照是提取 kernel，不是安装包 Python dispatcher。便利接口每次重新验证的 CPU 成本不包含在 plan 数据中。

## 结果

单位为 µs/step：整个 batch 生成一个 token 的耗时，未按 B/H 除算。
Graph 为 GPU event 时间；eager 为包含发射和末尾同步的主机 wall time。
“变化”=`新版 Graph / FlashInfer Graph − 1`，负值表示新版更快。

| H | B | FI Graph | 新版 Graph | 变化 | FI eager wall | 新版 eager wall |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 1 | 2.242 | 2.273 | +1.37% | 4.224 | 3.865 |
| 6 | 64 | 6.110 | 5.986 | -2.02% | 6.548 | 6.464 |
| 12 | 16 | 3.648 | 3.587 | -1.67% | 4.259 | 4.052 |
| 12 | 32 | 6.106 | 5.893 | -3.48% | 6.561 | 6.406 |
| 12 | 64 | 9.935 | 9.476 | -4.62% | 10.424 | 9.959 |
| 12 | 128 | 17.600 | 16.267 | -7.58% | 18.132 | 16.764 |
| 16 | 1 | 2.339 | 2.342 | +0.14% | 4.297 | 3.879 |
| 16 | 64 | 12.627 | 11.855 | -6.11% | 13.101 | 12.344 |
| 32 | 1 | 2.402 | 2.401 | -0.04% | 4.246 | 3.891 |
| 32 | 64 | 23.496 | 21.964 | -6.52% | 24.149 | 22.546 |

[完整原始样本与源码 hash](../benchmarks/data/consistent-decode-b300.json)包含 min/max、中位数及全部样本。
小差异不应外推为跨平台保证。复现命令见 API 文档。

## 正确性与精度

`FLASH_KDA_LONG_TESTS=1 python -m pytest -q tests/consistency`：**154 passed**。
新增 decode 测试覆盖 H=12/16/32 及更小本地 H、空 batch、非零初态、原地/分离状态、
gate 饱和、请求重排、任意完整 head 分片、prefill/decode 边界、4K/32K、Graph 动态内容与编译缓存。
两 GPU 测试包含 H=12/16/32 的完整 head 分片，以及 GPU 间迁移 BF16 状态后续算；不验证框架通信实现。
有效输出和完整 BF16 续算状态逐位比较，NaN/Inf 失败。公开示例已运行。

独立 FP64 诊断使用 B=1、H=12、129 tokens，并保留每 token BF16 状态舍入：

| 数据 | 最大绝对误差 | RMS |
|---|---:|---:|
| 输出 | 2.7156554e-5 | 1.3608709e-6 |
| 最终状态 | 6.1035156e-5 | 1.9618364e-7 |

新版与通用 recurrent 路径诊断结果相同。这些误差包含正常舍入，只作精度诊断，不代替逐位一致性验收。

本版本延续原 BF16 recurrent 数值契约。本记录中的 prefill 续算验证使用 `consistent_fwd`；
融合 prefill 的独立验证与性能见 [prefill 文档](consistent-prefill.md)。
本记录不声明 CF-BI chunk 实现、完整模型 logits 一致性或多卡通信吞吐。
