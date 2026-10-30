# Recurrent 基线验收记录

2026-10-30 UTC，在下述 B300 软件栈上完成 recurrent 前向基线验收。
单卡 92 项测试及两节点八卡的 36 组 CP/PD/Graph 测试全部通过。
这些结果验证本接口的有效输出与续算状态；不代表 backward、完整模型或其他软件栈已验收。

## 固定版本与环境

- FlashKDA 开发基线：`d7e3b3a9f448500630fa60fb5200f4f9c9b1fc72`。
- FlashInfer 单 token 参考：`c3af0c13328fd23b256c53b9b4f94d9e160d118c`。
  测试参考的 gate、归约、kernel、launch 共五个函数与该版本 AST 一致，独立于生产算术实现。
- GPU：NVIDIA B300 SXM6 AC。分布式测试使用两节点、每节点四张分配到的 GPU；节点内两两 NV18。
  CP=1/2/4 位于首节点；CP=8 跨节点。NCCL 初始化日志确认节点内 P2P/CUMEM、跨节点 NET/Socket（TCP）。
- 各 rank 版本一致：driver 615.71.10；Python 3.12.3；PyTorch `2.14.0a0+4fdf77b940.nv26.08`；
  CUDA 13.4；CuTe DSL 4.6.2；TVM FFI 0.1.13.post3；cuda-python 13.3.1；NCCL 2.30.7。
- 公开镜像：`nvcr.io/nvidia/pytorch@sha256:3becd068f49bd2ad38f90db5f9a4803019a76933a24e63d821376c44e7a9200a`。
- 计算 kernel 文件 SHA256：`0612a6147440aa687b509b4c60b3bef540c932ad200fea55e3fdb56442d9eea5`。
- 性能 runner 所覆盖 Python 源码 SHA256：`0ba2148e74f95c941a1e8a5c5ce36a526044c651e37c380ce07d80bf2f3a7f9b`。
  算法见 `benchmarks/bench_consistency.py` 的 `provenance`；文档更新不改变该值。

测试前冻结源码，性能运行前后核验文件摘要；各 rank 核对相同源码、GPU 型号及软件版本。
本次没有改变 GPU 时钟，不把其他架构或依赖版本列为已验证平台。

## 正确性

| 验收项 | 结果与覆盖 |
|---|---|
| 单 token 参考 | H=12/16/32，长度 0/1/2/15/16/17/127/128/129，逐步输出及状态原始位模式相同 |
| 长序列 | H=12/16/32 的 4K/32K 随机切段；H=12 的 4K/32K 每步独立参考对齐 |
| Batch | batch 1/4/7/8/9/16/32，独立/混合请求、排列、变长 packing、动态增删、padding |
| PD | 17-token 请求的全部 18 个切点、空/不等长请求、按身份重新组 batch、继续 decode |
| CP | CP=1/2/4/8 × H=12/16/32 × 三组长度，共 36 组；检查每 rank 输出、边界状态和最终状态 |
| CP/Graph/PD | recv → 本地 Graph replay → send；CP prefill 后重新组 batch decode，全部通过 |
| 执行与缓存 | 重复执行、eager/Graph、容量内 metadata 更新及 batch/长度改变均复用计算编译缓存 |
| 边界与接口 | 非零初态、零 Q/K、饱和 gate、lower_bound=-5/-2.5/0、原地状态、未对齐连续 view、N/T=0、非法 metadata 与 autograd 拒绝 |

最终 Python-only wheel 在源码目录外独立安装，公开 example 执行通过；
sdist 包含测试、benchmark、数据及 FlashInfer 许可证/NOTICE，已检查归档内容。

单卡完整命令 `FLASH_KDA_LONG_TESTS=1 python -m pytest -q tests/consistency`：**92 passed**。
分布式默认矩阵的三组请求长度为 `[0,1,17,129]`、`[0,0,0,0]`、`[1,2,0,0]`，包含全空 rank。
采用按风险组合的覆盖矩阵，并非所有长度、batch、head 数与 CP 的笛卡尔积。
所有一致性断言检查有限值及 BF16 原始位模式；没有用 allclose 替代。

独立 FP64 诊断（batch=1、H=12、长度=129，仍在每步状态处舍入为 BF16）：

| 数据 | 最大绝对误差 | RMS |
|---|---:|---:|
| 输出 | 2.7156554e-5 | 1.3608709e-6 |
| 最终状态 | 6.1035156e-5 | 1.9618364e-7 |

这些误差包含所选数值路径的正常舍入，仅供诊断，不作为逐位一致验收阈值。

## 性能记录

固定输入 batch=4、H=12、每请求 129 tokens、D=128；交接状态共 1,572,864 bytes。
以下为各活跃 rank 耗时的最大值，单位 µs，单元格为 **warm / read-evicted**。

| CP | 本地计算 Graph | 仅状态传递 | eager CP 调用 |
|---:|---:|---:|---:|
| 1 | 217.42 / 270.69 | 无传递 | 218.39 / 353.44 |
| 2 | 119.85 / 148.67 | 16.48 / 39.74 | 139.33 / 409.68 |
| 4 | 71.75 / 91.22 | 57.05 / 84.22 | 270.73 / 518.11 |
| 8 | 47.28 / 61.18 | 13368.04 / 14109.36 | 13366.02 / 14254.98 |

本地 Graph 包含完整 API 的 metadata 检查与计算：导出的图有八个 kernel 节点，
其中一个为 recurrent 计算，其余为 metadata 检查；无通信节点。
warm 为 25 次 warmup 后在一对 CUDA event 内执行 100 次，取每次均值。
read-evicted 为 20 次样本的中位数，每次计时外读取 256 MiB 驱逐缓冲区并同步；实测 L2 为 132,644,864 bytes。
该名称描述执行协议，不保证每次访存都 miss L2；接收状态在计时区域内由通信产生。
编译、分配、capture、初始化及驱逐不计时。Graph 计时含 API 设备检查；eager CP 还包括 host dispatch、通信和等待。
三个指标独立测量，不应相加，亦不是整个分布式任务的 wall-clock makespan。

CP=8 的跨节点 TCP 状态传递约 13 ms；它与单节点八卡 NVLink 的性能不等价。
每节点使用分配到的四张 GPU，未控制节点上其他共享资源负载。本基线没有提速门槛。
完整数据见 [JSON](../benchmarks/data/consistency-b300-20261030.json)，运行方法见 [benchmark 说明](../benchmarks/CONSISTENCY.md)。
安装和接口见 [中文](consistency.md) / [English](consistency.en.md)，测试命令见 [测试说明](../tests/consistency/README.md)。
