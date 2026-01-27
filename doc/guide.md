# 上手指南 — msModeling

## 目录

- [1. 环境配置](#1-环境配置)
- [2. 快速开始：TensorCast 单模型仿真](#2-快速开始tensorcast-单模型仿真)
- [3. 快速开始：ServingCast 服务仿真](#3-快速开始servingcast-服务仿真)
- [4. 进阶用法](#4-进阶用法)
- [5. 常见问题](#5-常见问题)

---

## 1. 环境配置

### 1.1 系统要求

- **Python**：>= 3.9
- **操作系统**：Linux（推荐）/ macOS / Windows
- **硬件**：仅需 CPU（本平台为纯仿真，不需要 NPU/GPU 硬件）

### 1.2 安装步骤

#### Step 1：克隆仓库

```bash
git clone <repository_url> msModeling
cd msModeling
```

#### Step 2：创建虚拟环境（推荐）

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows
```

#### Step 3：安装 TensorCast 依赖

```bash
pip install -r tensor_cast/requirements.txt
```

主要依赖包括：

| 依赖 | 版本要求 | 用途 |
|------|----------|------|
| `torch` | >= 2.7, < 2.9 | PyTorch 核心框架 |
| `transformers` | >= 4.57, < 5.0 | HuggingFace 模型加载 |
| `overrides` | - | 方法覆写装饰器 |
| `optree` | - | PyTree 操作 |
| `pyyaml` | - | YAML 配置解析 |
| `strenum` | - | 字符串枚举（Python < 3.11） |
| `pandas` | - | 数据分析 |
| `diffusers` | - | Diffusion 模型支持 |
| `modelscope` | - | ModelScope 模型下载 |
| `compressed-tensors` | - | 压缩张量支持 |

#### Step 4：安装 ServingCast 依赖（如需服务仿真）

```bash
pip install -r serving_cast/requirements.txt
```

额外依赖：

| 依赖 | 用途 |
|------|------|
| `salabim` | 离散事件仿真引擎 |
| `blinker` | 信号/回调机制 |
| `scikit-learn` | 插值预测 |
| `scipy` | 科学计算 |
| `prettytable` | 表格格式化输出 |
| `greenlet` | 协程支持 |

#### Step 5：安装根目录依赖

```bash
pip install -r requirements.txt
```

### 1.3 模型权重说明

msModeling 仅需要模型的**配置文件**（`config.json`），不需要下载完整的模型权重。所有张量运算在 `meta` 设备上执行（零内存占用）。但 `transformers` 库在首次使用时会自动下载模型配置。

如果无法访问 HuggingFace，可以使用 ModelScope 镜像或提前下载 `config.json` 到本地路径。

---

## 2. 快速开始：TensorCast 单模型仿真

### 2.1 基础文本生成仿真

以下示例展示如何对 Qwen3-32B 模型进行推理性能仿真：

```bash
python -m tensor_cast.scripts.text_generate \
    Qwen/Qwen3-32B \
    --num-queries 2 \
    --query-length 3500 \
    --context-length 4500 \
    --device TEST_DEVICE \
    --quantize-linear-action W8A8_DYNAMIC \
    --tp-size 4 \
    --world-size 4
```

**参数说明：**

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `model_id` | HuggingFace 模型标识符（位置参数） | `Qwen/Qwen3-32B` |
| `--num-queries` | 并发请求数 | `2` |
| `--query-length` | 每个请求的查询 token 数 | `3500` |
| `--context-length` | 上下文长度（KV Cache 已缓存的 token 数） | `4500`（0 表示 Prefill） |
| `--device` | 目标设备名称 | `TEST_DEVICE` |
| `--quantize-linear-action` | 线性层量化方案 | `W8A8_DYNAMIC` |
| `--tp-size` | 张量并行大小 | `4` |
| `--world-size` | 总设备数 | `4` |

### 2.2 输出解读

仿真结束后，控制台将输出类似以下信息：

```
Initializing model on 'meta' device...
Number of Queries per DP rank: 2
Preparing dummy input tensors...
Running simulated inference...
Model compilation and execution time: 1.23s

-------------------------------------------------------------
              Name              analytic total  analytic avg  # of Calls
-------------------------------------------------------------
tensor_cast.static_quant_linear   12.345ms       0.193ms        64
tensor_cast.attention              8.765ms       0.137ms        64
aten.mm.default                    2.345ms       0.073ms        32
tensor_cast.all_reduce             1.234ms       0.019ms        64
...
-------------------------------------------------------------
Total time for analytic: 25.678ms

TPS/Device: 273.2 token/s

Total device memory: 64.000 GB
  Model weight size: 16.543 GB
  KV cache: 8.234 GB
  Model activation size: 2.156 GB
  Reserved memory: 10 GB
  Memory available: 27.067 GB

Stats breakdowns:
  analytic_OpBound: ['memory_bound: 45.23', 'communication_bound: 12.34',
                     'compute_bound_mma: 38.76', 'compute_bound_gp: 3.67']
```

**输出说明：**

| 项目 | 含义 |
|------|------|
| **算子性能表** | 按总耗时排序的算子级性能统计，包含总耗时、平均耗时和调用次数 |
| **TPS/Device** | 单卡每秒处理 token 数 |
| **内存分布** | 模型权重、KV Cache、激活值和可用内存的详细分布 |
| **OpBound 分析** | 按算子瓶颈分类的时间占比（计算密集/访存密集/通信密集） |

### 2.3 生成 Chrome Trace

添加 `--chrome-trace` 参数可导出可视化时间线：

```bash
python -m tensor_cast.scripts.text_generate \
    Qwen/Qwen3-32B \
    --num-queries 2 \
    --query-length 3500 \
    --device TEST_DEVICE \
    --quantize-linear-action W8A8_DYNAMIC \
    --tp-size 4 \
    --world-size 4 \
    --chrome-trace output_trace.json
```

生成的 `output_trace.json` 可在 Chrome 浏览器中打开：
1. 打开 Chrome 浏览器
2. 访问 `chrome://tracing`
3. 点击 "Load" 按钮，选择 `output_trace.json`

### 2.4 Decode 阶段仿真

通过 `--decode` 参数切换到 Decode 模式（自动将 `query-length` 设为 1）：

```bash
python -m tensor_cast.scripts.text_generate \
    Qwen/Qwen3-32B \
    --num-queries 8 \
    --query-length 1 \
    --context-length 4096 \
    --decode \
    --device ATLAS_800_A2_376T_64G \
    --quantize-linear-action W8A8_DYNAMIC \
    --tp-size 8 \
    --world-size 8
```

### 2.5 吞吐 Benchmark

使用 `benchmark.py` 在 SLO 约束下搜索最大吞吐：

```bash
python -m tensor_cast.scripts.benchmark \
    Qwen/Qwen3-32B \
    --device TEST_DEVICE \
    --input-length 3500 \
    --output-length 1500 \
    --quantize-linear-action W8A8_DYNAMIC \
    --tp-size 4 \
    --world-size 4
```

---

## 3. 快速开始：ServingCast 服务仿真

### 3.1 配置文件准备

ServingCast 需要两个 YAML 配置文件：

**`instances.yaml` — 实例配置**

```yaml
instance_groups:
  - num_instances: 1           # 实例数
    num_devices_per_instance: 1 # 每实例设备数
    device_type: TEST_DEVICE   # 设备类型
    pd_role: both              # both / prefill / decode
    parallel_config:
      world_size: 1
      tp_size: 1
      dp_size: 1
      mlp_tp_size: null        # MLP 专用 TP（可选）
      mlp_dp_size: null
      lmhead_tp_size: null
      lmhead_dp_size: null
      ep: False                # 是否启用专家并行
    communication_config:
      host2device_bandwidth: 10000000000  # 10 GB/s
      host2device_rate: 0.5
      device2device_bandwidth: 4000000000 # 4 GB/s
      device2device_rate: 0.5
```

**`common.yaml` — 公共配置**

```yaml
model_config:
  name: Qwen/Qwen3-32B
  num_mtp_tokens: 0
  quantize_linear_action: W8A8_DYNAMIC
  quantize_lmhead: False
  mxfp4_group_size: 32
  quantize_attention_action: DISABLED
  do_compile: False
  enable_preprocessing_modeling: True    # 是否建模预处理开销
  enable_kv_transfer_modeling: True      # 是否建模 KV 传输开销

load_gen:
  load_gen_type: fixed_length
  num_requests: 500           # 总请求数
  num_input_tokens: 3500      # 每请求输入 token 数
  num_output_tokens: 1500     # 每请求输出 token 数
  request_rate: 2.0           # 请求到达速率（QPS）

serving_config:
  max_concurrency: 100        # 最大并发
  block_size: 128             # KV Cache block 大小
  max_tokens_budget: 8192     # 每轮调度最大 token 数
```

### 3.2 运行服务仿真

```bash
cd serving_cast
python main.py \
    --instance_config_path ./example/instances.yaml \
    --common_config_path ./example/common.yaml \
    --enable_profiling \
    --profiling_output_path ./results
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--instance_config_path` | 实例配置文件路径 |
| `--common_config_path` | 公共配置文件路径 |
| `--enable_profiling` | 启用性能 Profiling |
| `--profiling_output_path` | Profiling 结果输出目录 |

### 3.3 PD 分离部署仿真

将 `instances.yaml` 中的 `pd_role` 设置为 `prefill` 或 `decode` 即可模拟 PD 分离架构：

```yaml
instance_groups:
  # Prefill 实例组
  - num_instances: 2
    num_devices_per_instance: 4
    device_type: ATLAS_800_A2_376T_64G
    pd_role: prefill
    parallel_config:
      world_size: 4
      tp_size: 4
      dp_size: 1
      ep: False
    communication_config:
      host2device_bandwidth: 10000000000
      host2device_rate: 0.5
      device2device_bandwidth: 4000000000
      device2device_rate: 0.5

  # Decode 实例组
  - num_instances: 4
    num_devices_per_instance: 4
    device_type: ATLAS_800_A2_376T_64G
    pd_role: decode
    parallel_config:
      world_size: 4
      tp_size: 4
      dp_size: 1
      ep: False
    communication_config:
      host2device_bandwidth: 10000000000
      host2device_rate: 0.5
      device2device_bandwidth: 4000000000
      device2device_rate: 0.5
```

---

## 4. 进阶用法

### 4.1 完整 CLI 参数参考（TensorCast）

运行以下命令查看完整参数列表：

```bash
python -m tensor_cast.scripts.text_generate --help
```

**关键参数分类：**

#### 模型与设备

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_id` | str | 必填 | HuggingFace 模型 ID 或本地路径 |
| `--device` | str | - | 设备名称，可选值见 `DeviceProfile.all_device_profiles` |

#### 输入配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num-queries` | int | 必填 | 并发请求数 |
| `--query-length` | int | 必填 | 每请求查询 token 数 |
| `--context-length` | int | 0 | 上下文长度（0 表示纯 Prefill） |
| `--decode` | flag | False | Decode 模式（query-length 自动设为 1） |

#### 量化

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--quantize-linear-action` | enum | - | 线性层量化方案：W8A8_STATIC, W8A8_DYNAMIC, W4A8_STATIC, W4A8_DYNAMIC, FP8, MXFP4 |
| `--quantize-lmhead` | flag | False | 是否量化 LM Head |
| `--mxfp4-group-size` | int | 32 | MXFP4 量化分组大小 |
| `--quantize-attention-action` | enum | DISABLED | 注意力量化方案 |

#### 并行化

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--world-size` | int | 1 | 总设备数 |
| `--tp-size` | int | 1 | 张量并行大小 |
| `--dp-size` | int | 自动 | 数据并行大小（默认 = world-size / tp-size） |
| `--ep` | flag | False | 启用专家并行（MoE 模型） |

#### 高级特性

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--compile` | flag | False | 启用 `torch.compile()` 图优化 |
| `--compile-allow-graph-break` | flag | False | 允许图断裂 |
| `--num-mtp-tokens` | int | 0 | 多 Token 预测的投机 token 数 |
| `--num-hidden-layers-override` | int | 0 | 覆盖模型层数（用于快速测试） |
| `--disable-repetition` | flag | False | 禁用区域展开（逐层执行） |

#### 输出

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--chrome-trace` | str | - | Chrome Trace 输出文件路径 |
| `--dump-input-shapes` | flag | False | 在性能表中按输入形状分组 |

### 4.2 可用设备列表

以下设备名称可用于 `--device` 参数：

```
TEST_DEVICE                    - 测试设备（开发调试用）
ATLAS_800_A2_376T_64G          - 昇腾 800 A2 (376T, 64GB)
ATLAS_800_A2_313T_64G          - 昇腾 800 A2 (313T, 64GB)
ATLAS_800_A2_280T_64G          - 昇腾 800 A2 (280T, 64GB)
ATLAS_800_A2_280T_64G_PCIE     - 昇腾 800 A2 (280T, 64GB, PCIE)
ATLAS_800_A2_280T_32G_PCIE     - 昇腾 800 A2 (280T, 32GB, PCIE)
ATLAS_800_A3_752T_128G_DIE     - 昇腾 800 A3 (752T, 128GB, DIE)
ATLAS_800_A3_560T_128G_DIE     - 昇腾 800 A3 (560T, 128GB, DIE)
```

### 4.3 典型仿真场景

#### 场景 1：Prefill 性能评估

```bash
python -m tensor_cast.scripts.text_generate \
    Qwen/Qwen3-32B \
    --num-queries 1 \
    --query-length 8192 \
    --context-length 0 \
    --device ATLAS_800_A2_376T_64G \
    --quantize-linear-action W8A8_DYNAMIC \
    --tp-size 8 \
    --world-size 8 \
    --chrome-trace prefill_trace.json
```

#### 场景 2：MoE 模型 + 专家并行

```bash
python -m tensor_cast.scripts.text_generate \
    deepseek-ai/DeepSeek-V3 \
    --num-queries 4 \
    --query-length 1 \
    --context-length 4096 \
    --decode \
    --device ATLAS_800_A2_376T_64G \
    --quantize-linear-action FP8 \
    --tp-size 8 \
    --world-size 16 \
    --ep \
    --num-mtp-tokens 4
```

#### 场景 3：多量化方案对比

```bash
# W8A8 量化
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
    --num-queries 4 --query-length 1 --context-length 2048 --decode \
    --device TEST_DEVICE --quantize-linear-action W8A8_DYNAMIC \
    --tp-size 4 --world-size 4

# FP8 量化
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
    --num-queries 4 --query-length 1 --context-length 2048 --decode \
    --device TEST_DEVICE --quantize-linear-action FP8 \
    --tp-size 4 --world-size 4

# MXFP4 量化
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
    --num-queries 4 --query-length 1 --context-length 2048 --decode \
    --device TEST_DEVICE --quantize-linear-action MXFP4 \
    --tp-size 4 --world-size 4
```

### 4.4 运行测试

```bash
# 运行 TensorCast 全部测试
pytest tensor_cast/tests/ -v

# 运行 ServingCast 单元测试
pytest serving_cast/tests/ut/ -v

# 运行 ServingCast 集成测试
pytest serving_cast/tests/st/ -v

# 运行特定测试
pytest tensor_cast/tests/test_runtime.py -v
```

---

## 5. 常见问题

### Q1: 仿真不需要 GPU/NPU 硬件吗？

是的。msModeling 的核心理念是**纯 CPU 仿真**。所有张量运算在 PyTorch 的 `meta` 设备上执行，只计算形状（shape）不计算数值。性能数据完全由解析模型（Roofline Model）计算得出。

### Q2: 如何添加自定义硬件设备？

在 `tensor_cast/device_profiles/` 目录下创建新的设备配置文件，参考 `tensor_cast/device.py` 中已有设备的定义格式：

```python
MY_DEVICE = DeviceProfile(
    name="MY_CUSTOM_DEVICE",
    vendor="MY_VENDOR",
    mma_ops={
        torch.float32: 100 * 1e12,    # 100 TFLOPS
        torch.bfloat16: 200 * 1e12,   # 200 TFLOPS
        # ... 其他数据类型
    },
    gp_ops={
        torch.float32: 10 * 1e12,
        torch.bfloat16: 20 * 1e12,
    },
    memory_size_bytes=80 * (1024**3),          # 80 GB
    memory_bandwidth_bytes_ps=2.0 * (1024**4), # 2 TB/s
    compute_efficiency=0.7,
    memory_efficiency=0.6,
    comm_grid=MY_INTERCONNECT,
    static_cost=StaticCost(mma_op_cost_s=5e-6, gp_op_cost_s=2e-6),
)
```

### Q3: 仿真精度如何？

仿真精度取决于以下因素：
- **效率参数的校准程度**：计算效率、内存效率、通信效率
- **算子属性的准确性**：计算量与访存量的估算
- **静态开销的精确度**：算子调度固定开销

当前版本使用硬编码的效率参数（70% 计算 / 60% 内存），适合**相对性能对比**（如不同并行策略的比较）。对于**绝对延迟预测**，建议在真实硬件上校准效率参数。

### Q4: 支持哪些量化方案？

| 量化方案 | CLI 参数值 | 说明 |
|----------|-----------|------|
| W8A8 静态量化 | `W8A8_STATIC` | 权重 INT8 + 激活 INT8（静态标定） |
| W8A8 动态量化 | `W8A8_DYNAMIC` | 权重 INT8 + 激活 INT8（运行时量化） |
| W4A8 静态量化 | `W4A8_STATIC` | 权重 INT4 + 激活 INT8 |
| W4A8 动态量化 | `W4A8_DYNAMIC` | 权重 INT4 + 激活 INT8（运行时量化） |
| FP8 | `FP8` | FP8 浮点量化 |
| MXFP4 | `MXFP4` | 微缩 FP4 量化（分组） |

### Q5: 如何理解 OpBound Breakdown 输出？

OpBound Breakdown 展示了所有算子在不同瓶颈类型上的时间占比：

- **memory_bound**：受内存带宽限制（如小 batch 的 Decode 阶段）
- **compute_bound_mma**：受矩阵乘算力限制（如大 batch 的 Prefill 阶段）
- **compute_bound_gp**：受向量/标量算力限制（如 Softmax、LayerNorm）
- **communication_bound**：受通信带宽限制（如多卡 AllReduce）

当 `memory_bound` 占比过高时，考虑增大 batch size 或使用更低精度的量化方案；当 `communication_bound` 占比过高时，考虑减少 TP 大小或使用更快的互联拓扑。
