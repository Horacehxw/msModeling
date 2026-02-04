# 性能对比工具

一个用于比较 VLLM 性能分析数据与 TensorCast 模拟结果的模块化工具。

## 概述

本工具通过将 TensorCast 性能预测与来自 Ascend NPU 执行的真实 VLLM 性能分析数据进行比较，实现 TensorCast 性能预测的验证。支持以下功能：

- **自动阶段检测**：适用于 PD 聚合场景
- **多对多算子映射**：VLLM 与 TensorCast 之间的算子映射
- **YAML 配置文件**：模型配置和算子映射
- **Excel 报告生成**：包含详细的对比指标

## 快速开始

### 使用模型配置文件（推荐）

```bash
# 对比 Qwen3-32B 性能分析与 TensorCast 模拟
python -m tensor_cast.scripts.profiling_comparison.cli compare \
    --profile qwen3_32b \
    --vllm-dir /path/to/ASCEND_PROFILER_OUTPUT \
    --output comparison.xlsx
```

### 自定义配置

```bash
# 使用显式的 TensorCast 配置进行对比
python -m tensor_cast.scripts.profiling_comparison.cli compare \
    --vllm-dir /path/to/ASCEND_PROFILER_OUTPUT \
    --model-id Qwen/Qwen3-32B \
    --device ATLAS_800_A3_752T_128G_DIE \
    --world-size 16 --tp-size 16 \
    --num-queries 136 --context-length 4096 \
    --output comparison.xlsx
```

### 查看可用资源

```bash
# 列出可用的模型配置文件
python -m tensor_cast.scripts.profiling_comparison.cli list-profiles

# 列出可用的算子映射
python -m tensor_cast.scripts.profiling_comparison.cli list-mappings
```

## 目录结构

```
tensor_cast/scripts/profiling_comparison/
├── cli.py                      # 统一的 CLI 入口
├── config/
│   ├── schema.py               # 配置数据类
│   ├── loader.py               # YAML 配置加载
│   └── profiles/               # 模型配置文件
│       ├── qwen3_32b.yaml      # Qwen3-32B 配置
│       └── deepseek_v3.yaml    # DeepSeek-V3 配置
├── parsers/
│   ├── base.py                 # 解析器协议
│   ├── kernel_details_parser.py # VLLM kernel_details.csv 解析器
│   ├── tensorcast_adapter.py   # TensorCast 模拟适配器
│   └── phase_detector.py       # 自动阶段检测
├── alignment/
│   ├── op_mapper.py            # 算子映射逻辑
│   ├── mapping_loader.py       # YAML 映射加载器
│   └── mappings/               # 算子映射文件
│       ├── default.yaml        # 默认映射
│       └── qwen3.yaml          # Qwen3 特定映射
├── output/
│   ├── base.py                 # 格式化器协议
│   └── excel.py                # Excel 报告生成器
├── tests/                      # 单元测试
├── README.md                   # 英文文档
└── README_zh.md                # 中文文档（本文件）
```

## 配置说明

### 模型配置文件

模型配置文件定义了特定模型的默认 TensorCast 配置。在 `config/profiles/` 目录下创建配置文件：

```yaml
# config/profiles/my_model.yaml
name: my-model
description: "我的自定义模型配置"

tensorcast:
  model_id: organization/model-name
  device: ATLAS_800_A3_752T_128G_DIE
  world_size: 16
  tp_size: 16
  dp_size: 1
  ep: false
  quantize_linear_action: DISABLED

decode_defaults:
  num_queries: 136
  query_length: 1
  context_length: 4096

prefill_defaults:
  num_queries: 136
  query_length: 4096
  context_length: 0
```

### 算子映射

算子映射定义了 VLLM 算子与 TensorCast 算子之间的对应关系：

```yaml
# alignment/mappings/default.yaml
version: "1.0"

# VLLM 融合算子 -> 多个 TC 算子
vllm_fusions:
  - name: "AddRmsNorm"
    vllm_ops: ["AddRmsNorm", "InplaceAddRmsNorm"]
    tc_ops: ["aten.add", "tensor_cast.rmsnorm"]
    aggregation: sum

# 多个 VLLM 算子 -> TC 融合算子
tc_fusions:
  - name: "attention_block"
    vllm_ops: ["FusedInferAttentionScore", "ReshapeAndCacheNdKernel"]
    tc_ops: ["tensor_cast.attention"]
    aggregation: sum

# 直接 1:1 映射
direct_mappings:
  MatMulV2:
    - aten.mm
    - aten.matmul
```

## 阶段检测

对于 PD 聚合场景（prefill 和 decode 在同一设备上运行），工具可以自动检测阶段：

### 检测算法

1. **主要方法**：`ReshapeAndCacheNdKernel` 输入形状中的 query_len
   - `query_len == 1` → DECODE（99% 置信度）
   - `query_len > 100` → PREFILL（98% 置信度）

2. **辅助方法**：MatMul 批次维度
   - `M <= 256` → DECODE（85% 置信度）
   - `M >= 512` → PREFILL（90% 置信度）

### 手动指定阶段

```bash
# 强制使用 decode 阶段
python -m tensor_cast.scripts.profiling_comparison.cli compare \
    --profile qwen3_32b \
    --vllm-dir /path/to/profiling \
    --phase decode
```

## 输出说明

### Excel 报告结构

生成的 Excel 报告包含四个工作表：

1. **VLLM Operations**：所有 VLLM 算子及其时间数据
2. **TensorCast Operations**：所有 TensorCast 算子及其时间数据
3. **Comparison**：并排对比及差异指标
4. **Summary**：配置信息和汇总统计

### 对比指标

- **时间覆盖率**：匹配到 TensorCast 算子的 VLLM 时间百分比
- **平均差异**：平均绝对百分比差异
- **匹配状态**：
  - `exact`：差异在 20% 以内的直接匹配
  - `partial`：差异在 50% 以内的匹配
  - `missing_in_tc`：在 TensorCast 中未找到的 VLLM 算子
  - `missing_in_vllm`：在 VLLM 中未找到的 TensorCast 算子

## CLI 参考

### compare 命令

运行性能对比。

```
python -m tensor_cast.scripts.profiling_comparison.cli compare [选项]

必需参数:
  --vllm-dir PATH          ASCEND_PROFILER_OUTPUT 目录路径

基于配置文件（推荐）:
  --profile NAME           使用的模型配置文件（如 qwen3_32b）

自定义配置:
  --model-id ID            HuggingFace 模型 ID
  --device NAME            设备配置名称
  --world-size N           设备总数
  --tp-size N              张量并行大小
  --dp-size N              数据并行大小
  --ep                     启用专家并行
  --quantize-linear-action 启用量化
  --num-queries N          批次大小
  --query-length N         查询长度
  --context-length N       上下文长度

阶段检测:
  --phase {auto,prefill,decode}  阶段类型（默认：auto）
  --step-index N           decode 步骤索引（默认：100）

输出:
  --output PATH            输出文件路径
  --format {excel,json}    输出格式（默认：excel）
  --mapping NAME           自定义映射文件
```

### list-profiles 命令

列出可用的模型配置文件。

```
python -m tensor_cast.scripts.profiling_comparison.cli list-profiles
```

### list-mappings 命令

列出可用的算子映射。

```
python -m tensor_cast.scripts.profiling_comparison.cli list-mappings
```

## 测试

使用 pytest 运行测试：

```bash
# 运行所有性能对比测试
pytest tensor_cast/scripts/profiling_comparison/tests/ -v

# 运行特定测试模块
pytest tensor_cast/scripts/profiling_comparison/tests/test_config.py -v

# 运行并生成覆盖率报告
pytest tensor_cast/scripts/profiling_comparison/tests/ --cov=tensor_cast.scripts.profiling_comparison
```

## 扩展工具

### 添加新的模型配置文件

1. 在 `config/profiles/` 目录下创建 YAML 文件：

```yaml
name: new-model
tensorcast:
  model_id: org/new-model
  device: ATLAS_800_A3_752T_128G_DIE
  # ... 配置项
decode_defaults:
  num_queries: 64
  query_length: 1
  context_length: 2048
```

2. 可选：在 `alignment/mappings/` 目录下创建模型特定的映射文件

### 添加自定义算子映射

1. 在 `alignment/mappings/` 目录下创建 YAML 文件：

```yaml
version: "1.0"
vllm_fusions:
  - name: "CustomFusion"
    vllm_ops: ["CustomOp"]
    tc_ops: ["aten.custom1", "aten.custom2"]
    aggregation: sum
```

2. 使用 `--mapping` 参数指定：

```bash
python -m tensor_cast.scripts.profiling_comparison.cli compare \
    --mapping custom_mapping --vllm-dir ...
```

### 创建自定义输出格式化器

实现 `FormatterProtocol`：

```python
from tensor_cast.scripts.profiling_comparison.output.base import (
    BaseFormatter,
    ComparisonResult,
)

class MyFormatter(BaseFormatter):
    def format(self, result: ComparisonResult, output_path: Path) -> None:
        # 自定义格式化逻辑
        pass
```

## 故障排除

### 常见问题

1. **"kernel_details.csv not found"**
   - 确保 VLLM 性能分析启用了内核级分析
   - 检查性能分析目录路径是否正确

2. **"Profile not found"**
   - 运行 `list-profiles` 查看可用的配置文件
   - 检查配置文件名称拼写（使用下划线或连字符）

3. **阶段检测不正确**
   - 使用 `--phase decode` 或 `--phase prefill` 手动指定
   - 检查性能分析数据是否包含预期的算子

4. **时间差异过大**
   - 验证 TensorCast 配置与 VLLM 部署配置匹配
   - 检查量化设置是否一致
   - 检查算子映射是否缺少融合配置

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                          CLI (cli.py)                           │
│                    统一入口，支持子命令                           │
└─────────────────────────────────────────────────────────────────┘
                                  │
           ┌──────────────────────┼──────────────────────┐
           ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    配置层       │    │    解析层       │    │    对齐层       │
│ - schema.py     │    │ - VLLM 解析器   │    │ - op_mapper.py  │
│ - loader.py     │    │ - TC 适配器     │    │ - YAML 加载器   │
│ - profiles/     │    │ - 阶段检测      │    │ - mappings/     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                  │
                                  ▼
                       ┌─────────────────┐
                       │    输出层       │
                       │ - Excel 格式    │
                       │ - JSON 格式     │
                       └─────────────────┘
```

## Qwen3-32B 性能分析示例

### 配置信息

基于实际 VLLM 部署配置（来自 VLLM.sh 和 benchmark.sh）：
- 模型：Qwen3-32B（稠密模型，非 MoE）
- TP：16 设备
- 输入：4096 tokens
- 输出：1 token（decode 模式）
- 批次：136 请求
- 精度：BF16

### VLLM 主要算子

| 算子 | 时间占比 | 总时间 (us) |
|------|---------|------------|
| MatMulV2 | 42.4% | 8634.91 |
| FusedInferAttentionScore | 18.2% | 3703.04 |
| TensorMove | 10.7% | 2184.45 |
| AddRmsNorm | 8.0% | 1633.95 |
| split_qkv_rmsnorm_rope_kernel | 5.2% | 1053.02 |
| SwiGlu | 4.8% | 983.84 |

### 运行对比

```bash
python -m tensor_cast.scripts.profiling_comparison.cli compare \
    --profile qwen3_32b \
    --vllm-dir "/mnt/d/Data/Profiling/profiling-qwen3-30b-pd_tegether/profiling/..." \
    --output qwen3_comparison.xlsx
```

## 许可证

参见仓库 LICENSE 文件。
