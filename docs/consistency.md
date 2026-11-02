# KDA recurrent 训推一致性接口

[English](consistency.en.md) · [返回首页](../README.md)

`flash_kda.consistent_fwd` 用同一 recurrent 数值路径处理训练 forward、prefill 和 decode。
首版仅支持 no_grad 前向；需要梯度时明确报错。现有 `flash_kda.fwd` 保留原有行为。
此处的一致性仅限同型号 GPU、同编译器与软件版本、同参数及输入位模式；不能据此推导完整模型一致性。
Backward、CPR、Ulysses、并行前缀树、卷积层状态及完整模型集成不在范围内。

## 安装与验证范围

需要 Python ≥ 3.10、CUDA PyTorch、CUDA Toolkit，以及以下可选依赖。
首版目标为 B300/GB300；实际验证情况见 [验收记录](consistency-validation.md)。
其他架构或软件组合需要重新验收，不承诺跨版本逐位一致。

```bash
python -m pip install 'nvidia-cutlass-dsl==4.6.2' 'apache-tvm-ffi>=0.1.11,<0.2' 'cuda-python>=12.8' pytest
FLASH_KDA_SKIP_CUDA_BUILD=1 python -m pip install --no-build-isolation '.[consistency,test]'
python examples/consistent_fwd.py
```

`FLASH_KDA_SKIP_CUDA_BUILD=1` 仅安装 Python/CuTe 路径；这种安装不提供 `fwd` 所需的 C++ 扩展。
如需同时使用 `fwd`，按首页构建步骤安装，并安装 `consistency` extra。
CuTe DSL 在第一次调用时导入和编译；运行时无需 FlashInfer 或外部参考仓。

已测环境可用固定的公开镜像摘要复现（从仓库根目录执行）：

```bash
docker run --rm --gpus all --ipc=host -e PYTHONPATH=/work -e PYTHONNOUSERSITE=1 \
  -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 -v "$PWD:/work:ro" -w /work \
  nvcr.io/nvidia/pytorch@sha256:3becd068f49bd2ad38f90db5f9a4803019a76933a24e63d821376c44e7a9200a \
  bash -c 'python -m pip install pytest && FLASH_KDA_LONG_TESTS=1 python -m pytest -q -p no:cacheprovider tests/consistency'
```

## API 与数据契约

```python
from flash_kda import consistent_fwd
out, state_out = consistent_fwd(
    q, k, v, g, beta, A_log, dt_bias, cu_seqlens, state_in, out, state_out,
    scale=128 ** -0.5, eps=1e-6, lower_bound=-5.0,
)
```

| 参数 | dtype | shape / 语义 |
|---|---|---|
| q/k/v/g | BF16 | `[1,T,H,128]`；g 为激活前输入 |
| beta | BF16 | `[1,T,H]`；激活前 logits |
| A_log | FP32 | `[H]` |
| dt_bias | FP32 | `[H,128]` |
| cu_seqlens | CUDA int64 | `[N+1]`；从零开始、非递减、末值 ≤ T |
| state_in/state_out | BF16 | `[N,H,128,128]`；维度顺序 `[N,H,V,K]` |
| out | BF16 | `[1,T,H,128]` |

所有张量同 CUDA 设备、连续；无需额外地址对齐。H > 0，Q/K/V head 数相同。
T 表示容量，末偏移之后的 padding 不读写；空序列保持状态，支持 N=0、T=0。
调用方分配输出与状态。仅允许 state_in 与 state_out 精确同区间原地更新；其余重叠均拒绝。
scale、eps、lower_bound 为有限且可表示为 FP32 的 host 标量；eps 转成 FP32 后 > 0，lower_bound ∈ [-5,0]。
输入有效元素与初态应为有限值；验收遇到 NaN/Inf 必须失败。

形状、dtype、设备、连续性、重叠及标量错误同步报错。cu_seqlens 内容在设备上异步检查，
错误会产生 CUDA device assertion，并可能使 CUDA context 不再可用；调用方须保证合法 metadata。
接口不做设备到主机读取。梯度开启且任一浮点张量 requires_grad 时抛出 RuntimeError。

## 数值语义

每步先将输入转换为 FP32，按固定归约顺序做 Q/K L2 归一化（分母 `sqrt(sum(x²)+eps)`），
Q 再乘 scale。beta 使用 sigmoid；gate 的数学表达为：

`exp(lower_bound * sigmoid(exp(A_log) * (g + dt_bias)))`。

先对状态的 K 维施加 gate，再按 delta rule 更新。输出从更新后、尚未舍入的 FP32 状态计算并转为 BF16。
随后状态转为 BF16；下一 token 必须从这个 BF16 值恢复。数学表达不能代替实际浮点运算顺序。
输出及状态不承诺与现有 chunk `fwd` 相同。

数值契约标识由 `flash_kda.NUMERICAL_CONTRACT` 导出。交接还必须核对 scale、eps、lower_bound、
head 数、GPU 型号及软件版本。batch、长度、metadata 内容、请求排列、指针及 stream 不进入编译缓存 key。

## 生命周期与 CUDA Graph

调用异步提交到当前 PyTorch CUDA stream。输入、metadata、输出及状态必须存活到该 stream 完成。
跨 stream 使用需通过 event / wait_stream 显式建立依赖，并按 PyTorch 规则维护分配器生命周期。
原地状态再次调用会继续递推；重复相同实验时必须恢复同一初态。

先在 capture 外完成导入、编译和 warmup，再捕获本地计算。Graph replay 使用稳定地址和固定 tensor shape。
在容量内可更新 cu_seqlens：以空序列表示未使用的请求槽位，用末偏移控制有效 token 数。
metadata 和输入的更新必须在 replay 前完成；不得在执行过程中修改。改变 tensor shape 需要重新 capture，
但不会因 N/T 改变而生成另一份计算 kernel。CP 通信必须在 capture 外执行。

## CP 与 PD

`flash_kda.continuation.cp_consistent_fwd` 接收相同参数，并增加 `group=None`。
每条请求按时间切成 group rank 有序的连续区间，允许不等长、空区间；各 rank 保持相同请求顺序和状态 shape。
rank 0 提供初态；后续 rank 的 state_in 是接收缓冲区，会被改写。NCCL 传输 BF16 原始数据，
本地计算完成后发送 state_out；末 rank 持有最终状态，输出留在各自 rank。
调用方负责正确切片及所有 rank 的一致参与；此方案没有序列并行提速目标。

`Continuation(request_ids, positions, state, ...)` 保存请求身份、host token 位置、借用的 BF16 状态和数值参数。
`rebase_continuations(parts, request_ids, state_out)` 按身份重新组 batch，检查契约兼容且不改变状态位模式。
scale、eps、lower_bound 按转换后的 FP32 位模式匹配（包括区分 `+0` / `-0`）。
新请求由调用方提供位置 0 和初态；删除请求时从目标 ID 列表省略。重排目标不得与源 storage 重叠。
此工具不执行网络传输；PD 调用方需传送 metadata 与原始 BF16 状态并建立 stream 依赖。
KDA 核心不需要历史 K/V；完整层还需要另外管理卷积等状态。

## 验收

```bash
python -m pytest -q tests/consistency
FLASH_KDA_LONG_TESTS=1 python -m pytest -q tests/consistency
python -m tests.consistency.accuracy
# Eight GPUs of the same model, NCCL; standalone rendezvous owns its port.
torchrun --standalone --nproc-per-node=8 -m tests.consistency.distributed
```

[测试说明](../tests/consistency/README.md) 列出参考来源、覆盖范围及首个差异定位。
高精度误差只作诊断；一致性验收比较有效元素原始位模式，不能用 allclose 替代。
