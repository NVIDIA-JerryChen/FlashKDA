# FP32 state recurrent forward

[English](fp32-state.en.md)

`prepare_fp32_prefill` 和 `prepare_fp32_decode` 提供同一数值契约的 no_grad
forward：BF16 输入/输出，FP32 计算、状态与 checkpoint。适用于 packed varlen
prefill、训练 workload 的 forward、单 token decode 和 Q=8 speculative decode。
不支持 backward、其他 Q 的专用 decode、自动接受数选择或框架通信。
完整模型 logits 的一致性不在接口承诺内。

## 安装与支持范围

目标为 SM100/SM103 Blackwell；验证平台为 B300，Python 3.12、PyTorch 2.12.0+cu132、
CUDA 13.2、CuTe DSL 4.8.1。其他软件组合需单独验证。先安装匹配 CUDA 的 PyTorch，
在项目根目录安装纯 Python recurrent 路径：

```bash
pip install 'nvidia-cutlass-dsl==4.8.1' 'nvidia-cutlass-dsl-libs-base==4.8.1' \
  'nvidia-cutlass-dsl-libs-core==4.8.1' 'nvidia-cutlass-dsl-libs-cu13==4.8.1' \
  'apache-tvm-ffi==0.1.10' 'cuda-python>=12.8'
FLASH_KDA_SKIP_CUDA_BUILD=1 pip install --no-build-isolation '.[fp32,test]'
python -m examples.fp32_state
```

安装原 FlashKDA 扩展见根 README。不安装扩展也可调用本页两个接口。

## 输入与输出

两个 prepare 均以 `q,k,v,g,beta,A_log,dt_bias,state_in,out` 开始，
`scale=128**-0.5`、`eps=1e-6`、`lower_bound=-5.0` 为默认值。
所有 tensor 同一 CUDA device、连续、非空指针 32-byte 对齐。`D=128`、`H>0`。
Q/K/V/g/out 为 BF16，beta 为 BF16 logits，A_log/dt_bias/state/checkpoint 为 FP32。
g 为激活前 gate；beta 在 kernel 内执行 sigmoid。

| 参数 | Prefill | Decode |
|---|---|---|
| q/k/v/g/out | dense `[B,L,H,128]` 或 packed `[1,T,H,128]` | `[B,Q,H,128]`，Q=1 或 8 |
| beta | 对应前三维 | `[B,Q,H]` |
| A_log / dt_bias | `[H]` / `[H,128]` | 同左 |
| state_in | `[N,H,128,128]`，最后两维为 V,K | `[B,H,128,128]` |
| state_out | 必需，shape 同 state_in | Q1 必需；Q8 必须为 None |
| checkpoint_states | 不支持 | Q1 必须为 None；Q8 必需 `[B,8,H,128,128]` |
| cu_seqlens | packed 时 Int32 `[N+1]`；dense 时省略，N=B | 不使用，每请求固定 Q |

Prefill 使用统一入口，L=1 也不会转入 decode。最多 1023 请求；总 token 容量、
调度容量与 grid 必须满足 API 的 Int32 限制。空 batch 和空请求有效。
packed metadata 必须从 0 开始、非递减、末值不超过 T；有效 token 连续存放，
未使用的尾部输出不写入。该内容约束由调用方保证，prepare/run 均不将设备长度读回 host。
历史 KV 由完整 FP32 state 表示，不需要传入历史 KV 长度。

`.run()` 返回 `(out, state_out)`；Q8 的返回终态是 `checkpoint_states[:, -1]`
的 view，通常不连续。作为下一调用入口前用 `.contiguous()` 或拷贝到连续 state buffer。
若每请求接受 a 个 token，a=0 使用原 state_in，否则选择该请求 checkpoint 的 a−1 行；
接受数选择及必要拷贝由调用方执行，其成本须在相应端到端测量中计入。

## 数值契约

Q/K 归一化、gate、beta、投影与状态更新均保持 FP32。每步先对状态施加衰减，
再计算预测和 delta，更新 FP32 状态，最后用更新后的状态计算 BF16 输出。
状态在 token 间及 API 边界均不舍入 BF16；Q8 每步保存完整 FP32 checkpoint。
初等函数包含近似 exp2/reciprocal，不能等同于正确舍入数学函数。

同 GPU 型号、软件栈及相同标量配置下，有效输出和完整 FP32 续算状态要求在
整段/任意分段、Q1/Q8、请求重排、batch 组合和完整 head 分片间逐位相同。
与 BF16 recurrent、FlashInfer 或 FlashKDA 不要求位模式相同。切换 BF16/FP32
契约不保证已有续算轨迹一致，调用方必须明确选择契约并保持到请求结束。
独立 FP64 误差诊断用于衡量数学精度；内部位相同不能替代该诊断。

## 生命周期与错误

prepare 校验 tensor 类型、dtype、shape、device、连续性、对齐、重叠及标量范围，
编译并绑定固定容量。prepare 必须在 CUDA Graph capture 外执行。
需要梯度且启用 autograd 时抛出 RuntimeError；其他非法结构/标量抛出 TypeError 或 ValueError。
`eps` 必须在 FP32 中大于零，`lower_bound` 在 `[-5,0]`，标量须可表示为有限 FP32。

prefill 的 `workspace_limit_bytes` 默认 256 MiB，包含 plan 持有的工作区和 metadata，
不包含调用方输入/输出。容量不足报错，不自动无限增长。Decode 无额外 workspace。
除了 Q1/prefill 可让 state_in 与 state_out 精确原地重合，其他 tensor 禁止重叠。
Q8 checkpoints 必须与入口状态独立，保留接受 0 个 token 的原状态。

run 不分配设备内存、不编译、不 D2H，在 tensor 所属 device 的当前 stream 执行。
plan 绑定地址/容量/标量，允许原地址内容与合法设备 metadata 改变；改变 shape、地址
或标量需重新 prepare。调用方保持 plan 和所有 buffer 存活直至完成；跨 stream 先同步，
同一 plan/output/workspace 不得并发复用。跨 stream 更新输入/metadata 同样须显式同步。
Graph replay 前维护这些约束；终态原地更新时 replay 会继续推进状态，而不是重算同一前缀。

## 验证与 benchmark

```bash
python -m pytest -q tests/consistency/test_fp32.py
python -m benchmarks.bench_fp32 --mode prefill --lengths 0,128,1024,4096 --heads 16
python -m benchmarks.bench_fp32 --mode decode --batch 64 --query-length 1 --heads 16
python -m benchmarks.bench_fp32 --mode decode --batch 64 --query-length 8 --heads 16
python -m tests.consistency.fp32_accuracy --length 32768 --gate open
```

benchmark 计入 prepared API 的全部 GPU 工作，Q8 包含八个 checkpoint 的写出；
prepare/编译、前缀准备、接受数选择和 host 开销在计时外。使用固定输入地址的 CUDA Graph
重复计时，为热地址测量，不宣称 cold-L2。自定义长度用于复现与诊断；正式性能比较应使用
双方一致的明确长度清单、输入状态、输出义务与同节点配对计时，不用等长输入替代 varlen。
