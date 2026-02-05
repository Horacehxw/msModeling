# 性能对比工具

基于执行顺序的 3 阶段流水线，用于比较 VLLM 性能分析数据与 TensorCast 模拟结果。

## 概述

本工具通过**基于执行顺序的序列匹配**，将 TensorCast 性能预测与来自 Ascend NPU 的 VLLM 性能分析数据进行比较验证。

### 3 阶段流水线

| 阶段 | 命令 | 输入 | 输出 |
|------|------|------|------|
| 1. 分析 | `analyze` | VLLM 分析目录 + 配置 | 打印 TC 执行命令 |
| 2. 模拟 | `simulate` | 配置 + 阶段 | Chrome trace JSON（子进程） |
| 3. 对比 | `compare` | VLLM 目录 + TC chrome trace + 映射 | Excel 对比报告 |
| 全部 | `run-all` | VLLM 目录 + 配置 + 输出目录 | 依次执行 1→2→3 |

### 序列匹配算法

核心匹配算法按**执行顺序**同步遍历 VLLM 和 TensorCast 操作序列，根据分解映射为每个 VLLM 算子消耗对应的 TC 算子。每个 TC 算子只消耗一次，完全消除了重复计算。

## 快速开始

### 一键运行（推荐）

```bash
.conda/bin/python -m tensor_cast.scripts.profiling_comparison.cli run-all \
    --vllm-dir /path/to/ASCEND_PROFILER_OUTPUT \
    --profile qwen3_32b \
    --output-dir /tmp/results/
```

### 分阶段运行

```bash
# 阶段 1：分析 VLLM 性能数据（打印 TC 命令）
.conda/bin/python -m tensor_cast.scripts.profiling_comparison.cli analyze \
    --vllm-dir /path/to/ASCEND_PROFILER_OUTPUT \
    --profile qwen3_32b

# 阶段 2：运行 TensorCast 模拟
.conda/bin/python -m tensor_cast.scripts.profiling_comparison.cli simulate \
    --profile qwen3_32b \
    --phase decode \
    --output-dir /tmp/tc_results/

# 阶段 3：按执行顺序对比
.conda/bin/python -m tensor_cast.scripts.profiling_comparison.cli compare \
    --vllm-dir /path/to/ASCEND_PROFILER_OUTPUT \
    --tc-trace /tmp/tc_results/chrome_trace.json \
    --profile qwen3_32b \
    --output /tmp/comparison.xlsx
```

## 目录结构

```
tensor_cast/scripts/profiling_comparison/
├── cli.py                      # CLI 入口（6 个子命令）
├── stages/
│   ├── analyze.py              # 阶段 1：VLLM 分析 + TC 命令
│   ├── simulate.py             # 阶段 2：TC 子进程执行
│   └── compare.py              # 阶段 3：序列匹配 + Excel
├── config/
│   ├── schema.py               # 配置数据类
│   ├── loader.py               # YAML 配置加载
│   └── profiles/               # 模型配置文件
│       ├── qwen3_32b.yaml
│       └── deepseek_v3.yaml
├── parsers/
│   ├── kernel_details_parser.py # VLLM kernel_details.csv 解析器
│   ├── chrome_trace_parser.py  # TC chrome trace JSON 解析器
│   └── phase_detector.py       # 自动阶段检测
├── alignment/
│   ├── sequence_matcher.py     # 核心序列匹配算法
│   └── mappings/               # 分解映射文件
│       ├── default.yaml        # 默认分解映射
│       └── qwen3.yaml          # Qwen3 特定覆盖
├── output/
│   ├── base.py                 # 输出数据结构
│   └── excel.py                # Excel 报告生成器
└── tests/                      # 单元测试
```

## 配置说明

### 分解映射

分解映射定义了 VLLM 融合算子如何分解为有序的 TC 算子序列：

```yaml
version: "2.0"

decompositions:
  AddRmsNorm:
    tc_ops: ["aten.add", "aten.pow", "aten.mean", "aten.rsqrt"]
  MatMulV2:
    tc_ops: ["aten.mm"]

ignored_vllm_ops:
  - TensorMove
  - Fill

ignored_tc_ops:
  - aten.view
  - aten.reshape
```

## 输出说明

### Excel 报告结构

生成的 Excel 报告包含四个工作表：

1. **VLLM Operations**：所有 VLLM 算子及其时间数据
2. **TensorCast Operations**：所有 TC chrome trace 算子
3. **Sequence Comparison**：逐位置匹配，带颜色编码
4. **Summary**：配置信息和汇总统计

## 测试

```bash
.conda/bin/python -m pytest tensor_cast/scripts/profiling_comparison/tests/ -v
```
