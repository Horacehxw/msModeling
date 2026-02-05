# 性能对比流水线演示

本文档通过两个实际模型，完整演示 3 阶段性能对比流水线的使用流程。

## 流水线概览

```
阶段 1：分析                阶段 2：模拟                 阶段 3：对比
┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│ VLLM 性能分析     │         │ text_generate.py  │         │ 序列匹配器        │
│ kernel_details   │────────▶│ （子进程）         │────────▶│                  │
│ .csv             │  阶段   │                   │ chrome  │ VLLM 有序算子     │
│                  │  + 配置 │ --chrome-trace    │ trace   │ TC 有序事件       │
│ Profile YAML     │         │ --model-id ...    │ .json   │ 分解映射          │
│ 阶段检测器        │         │                   │         │                  │
└──────────────────┘         └──────────────────┘         └──────────────────┘
       │                            │                            │
       ▼                            ▼                            ▼
  打印 TC 命令              Chrome trace 文件            Excel 对比报告
  + 检测到的阶段             + 算子汇总表                  含逐算子差异
```

每个阶段都产生**可人工检查的中间输出**，使流水线透明可调试。

---

## 核心概念：Step = 一次完整的前向传播

一个 "step" 被定义为**模型所有层的一次完整前向传播**：
- Qwen3-32B（64 层）：64 个 `FusedInferAttentionScore` 算子 = 1 个 step
- DeepSeek-V3（61 层）：61 个 `FusedInferAttentionScore` 算子 = 1 个 step

`num_layers` 参数对于正确分组算子至关重要。

## 核心概念：基于执行顺序的序列匹配

与基于名称的聚合（可能导致重复计算）不同，序列匹配器按**执行顺序**同步遍历 VLLM 和 TensorCast 的算子列表：

1. 对于每个 VLLM 算子，从分解映射中查找期望的 TC 算子
2. 沿 TC 事件序列前进，跳过忽略的算子（view、reshape 等）
3. 按顺序消耗匹配的 TC 事件
4. 每个 TC 事件**只消耗一次** -- 完全消除重复计算

这产生了逐位置的对比结果，每个算子都被精确对应。

---

## 案例 1：Qwen3-32B（稠密模型，TP=16）

### 1.1 分析数据

| 参数 | 值 |
|------|------|
| 模型 | Qwen/Qwen3-32B（稠密，64 层） |
| 硬件 | 16x ATLAS A3 Dies |
| 并行策略 | TP=16 |
| 量化 | BF16（未量化） |
| 批大小 | 136 请求 |
| 分析数据路径 | `/mnt/d/Data/Profiling/profiling-qwen3-30b-pd_tegether/` |

**判断依据**：
- ASCEND_PROFILER_OUTPUT 中有 16 个 rank 目录 → TP=16
- `MatMulV2` 是耗时最多的算子（非 `QuantBatchMatmulV3`）→ BF16，未量化
- 无 `GroupedMatmul` 或 `MoeDistribute*` → 稠密模型，非 MoE
- `ReshapeAndCacheNdKernel` 输入 shape 包含 `query_len=1` → Decode 阶段

### 1.2 配置文件

配置文件 `config/profiles/qwen3_32b.yaml` 编码了这些参数：

```yaml
name: qwen3-32b
description: "Qwen3-32B dense model on 16x ATLAS A3 Dies with TP=16"
num_layers: 64

tensorcast:
  model_id: Qwen/Qwen3-32B
  device: ATLAS_800_A3_752T_128G_DIE
  world_size: 16
  tp_size: 16
  quantize_linear_action: DISABLED
  lmhead_tp_size: 16

decode_defaults:
  num_queries: 136
  query_length: 1
  context_length: 4096

mapping_file: qwen3    # 合并 default.yaml + qwen3.yaml
```

Qwen3 使用了模型特定的映射（`qwen3.yaml`），增加了 `split_qkv_rmsnorm_rope_kernel` 分解，将单个融合 VLLM 内核映射到 6 个 TC 算子：

```yaml
decompositions:
  split_qkv_rmsnorm_rope_kernel:
    tc_ops: ["aten.mm", "aten.mm", "aten.mm", "aten.split", "aten.neg", "aten.clone"]
```

### 1.3 运行流水线

#### 方式 A：一键运行全部 3 个阶段

```bash
VLLM_DIR="/mnt/d/Data/Profiling/profiling-qwen3-30b-pd_tegether/profiling/623bd92ebdd0_417949_20260203150032546_ascend_pt/ASCEND_PROFILER_OUTPUT"

.conda/bin/python -m tensor_cast.scripts.profiling_comparison.cli run-all \
    --vllm-dir "$VLLM_DIR" \
    --profile qwen3_32b \
    --output-dir /tmp/qwen3_results/
```

#### 方式 B：分阶段运行

**阶段 1：分析** -- 解析 VLLM 数据，检测阶段，打印 TC 命令：

```bash
.conda/bin/python -m tensor_cast.scripts.profiling_comparison.cli analyze \
    --vllm-dir "$VLLM_DIR" \
    --profile qwen3_32b
```

预期输出：
```
Loaded profile: qwen3-32b
  Description: Qwen3-32B dense model on 16x ATLAS A3 Dies with TP=16
  num_layers: 64

  Detected phase: decode (confidence: 99%)
  Detected query length: 1
  Complete forward passes: 1

=== TensorCast Command ===
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --dp-size 1 --num-queries 136 --query-length 1 --context-length 4096 \
  --lmhead-tp-size 16

=== Analyze Complete ===
Phase: decode
Steps detected: 1
```

**阶段 2：模拟** -- 运行 TensorCast 并生成 chrome trace：

```bash
.conda/bin/python -m tensor_cast.scripts.profiling_comparison.cli simulate \
    --profile qwen3_32b \
    --phase decode \
    --output-dir /tmp/qwen3_results/
```

此命令以子进程方式运行 `text_generate.py`（带 `--chrome-trace` 参数），产出：
- `/tmp/qwen3_results/chrome_trace.json` -- Chrome trace 文件

**阶段 3：对比** -- 基于执行顺序匹配 VLLM 算子与 TC trace：

```bash
.conda/bin/python -m tensor_cast.scripts.profiling_comparison.cli compare \
    --vllm-dir "$VLLM_DIR" \
    --tc-trace /tmp/qwen3_results/chrome_trace.json \
    --profile qwen3_32b \
    --output /tmp/qwen3_results/qwen3_comparison.xlsx
```

预期输出：
```
=== Loading Mappings ===
  Base mapping: default (17 decompositions)
  Model mapping: qwen3 (merged)
  Total decompositions: 19

=== Parsing VLLM Profiling ===
  Step 0: 1479 operations, 206742.53 us

=== Parsing TensorCast Trace ===
  847 events, 39622.54 us total

=== Running Sequence Matching ===
  Matched: 20
  Unmatched VLLM: 3
  Unmatched TC: 5
  Mismatches: 0

============================================================
SUMMARY
============================================================
VLLM total time:     206742.53 us
TensorCast total:    39622.54 us
Difference:          -167119.99 us (-80.8%)
Matched operations:  20
Coverage:            98.5%
Output saved to:     /tmp/qwen3_results/qwen3_comparison.xlsx
```

### 1.4 结果解读

| 指标 | 值 |
|------|------|
| VLLM 总时间 | ~207 ms |
| TensorCast 总时间 | ~40 ms |
| 差异 | -80.8% |
| 匹配算子数 | 20 |
| 覆盖率 | 98.5% |

**为何时间差异如此之大？** Qwen3-32B 的 `num_key_value_heads=8`，而 `tp_size=16`。每个 TP rank 模拟时只有 1 个 KV head（8/16=0.5，向上取整为 1），使模拟中的注意力计算比实际硬件（KV head 被复制）快约 8 倍。**覆盖率指标（98.5%）**才是验证映射正确性的关键，而非绝对时间。

**关键算子级对比**：

| VLLM 算子 | TC 等价物 | VLLM 耗时 | TC 耗时 | 差异 |
|-----------|-----------|-----------|---------|------|
| FusedInferAttentionScore | tensor_cast.attention | 160,547 us | 17,638 us | -89% |
| MatMulV2 | aten.mm | 6,854 us | 6,848 us | -0.1% |
| hcom_allReduce_ | tensor_cast.all_reduce | 5,543 us | 4,543 us | -18% |
| AddRmsNorm | aten.add + aten.pow + aten.mean + aten.rsqrt | 1,634 us | 1,580 us | -3.3% |
| SwiGlu | aten.silu + aten.mul | 984 us | 920 us | -6.5% |

线性算子（MatMulV2）匹配误差在 0.1% 以内，验证了 roofline 模型对计算密集型算子的准确性。

---

## 案例 2：DeepSeek-V3（MoE 模型，DP=32，EP）

### 2.1 分析数据

| 参数 | 值 |
|------|------|
| 模型 | deepseek-ai/DeepSeek-V3.1（MoE + MLA，61 层） |
| 硬件 | 32x ATLAS A3 Dies |
| 并行策略 | DP=32, TP=1, EP=32 |
| 量化 | W8A8_DYNAMIC |
| 特性 | EmbedTP=8, LMHead=8, SharedExpMS |
| 分析数据路径 | `/mnt/d/Data/Profiling/prof-torchair-deepseekv3-decode/` |

**判断依据**（来自 `VLLM_features.txt`）：
```
torchair
DP32,TP1,EP32,NSA4,
EmbedTP_8,LMHead_8,
KV_NZ,SharedExpMS,NO MTP,
RmTorchAirCost,TSQ0
```

- `GroupedMatmul` 是耗时最多的算子 → MoE 专家计算
- 存在 `QuantBatchMatmulV3` → W8A8 量化
- 存在 `MoeDistributeDispatchV2/CombineV2` → 基于 EP 的 MoE 路由
- 32 个 rank 目录 → world_size=32

### 2.2 配置文件

```yaml
name: deepseek-v3
description: "DeepSeek-V3 MoE model on 32x ATLAS A3 Dies with DP=32 + EP"
num_layers: 61

tensorcast:
  model_id: deepseek-ai/DeepSeek-V3.1
  device: ATLAS_800_A3_752T_128G_DIE
  world_size: 32
  tp_size: 1
  dp_size: 32
  ep: true
  quantize_linear_action: W8A8_DYNAMIC
  word_embedding_tp: 8
  lmhead_tp_size: 8
  enable_external_shared_experts: true

decode_defaults:
  num_queries: 24
  query_length: 1
  context_length: 4877
```

DeepSeek-V3 **仅使用默认映射**（无需模型特定覆盖）。默认映射已涵盖 MoE 相关的分解：

| VLLM 算子 | TC 分解 |
|-----------|---------|
| MoeDistributeDispatchV2 | tensor_cast.permute_tokens + tensor_cast.all_to_all |
| GroupedMatmul | tensor_cast.static_quant_linear |
| MoeDistributeCombineV2 | tensor_cast.all_to_all + tensor_cast.unpermute_tokens |
| KvRmsNormRopeCache | tensor_cast.concat_and_cache_mla |
| QuantBatchMatmulV3 | tensor_cast.static_quant_linear |

### 2.3 运行流水线

```bash
VLLM_DIR="/mnt/d/Data/Profiling/prof-torchair-deepseekv3-decode/feb1abe5c743_785216_20251218195608873_ascend_pt/ASCEND_PROFILER_OUTPUT"

.conda/bin/python -m tensor_cast.scripts.profiling_comparison.cli run-all \
    --vllm-dir "$VLLM_DIR" \
    --profile deepseek_v3 \
    --output-dir /tmp/deepseek_results/
```

预期输出：
```
============================================================
STAGE 1: Analyze VLLM Profiling
============================================================
Loaded profile: deepseek-v3
  Description: DeepSeek-V3 MoE model on 32x ATLAS A3 Dies with DP=32 + EP
  num_layers: 61
  Detected phase: decode (confidence: 99%)
  Complete forward passes: 41

============================================================
STAGE 2: Run TensorCast Simulation
============================================================
Command: python -m tensor_cast.scripts.text_generate deepseek-ai/DeepSeek-V3.1 ...
...
Total time for analytic: 89.055ms

============================================================
STAGE 3: Compare by Execution Order
============================================================
=== Loading Mappings ===
  Base mapping: default (17 decompositions)

=== Parsing VLLM Profiling ===
  Step 0: 2115 operations, 99508.46 us

=== Running Sequence Matching ===
  Matched: 32
  Unmatched VLLM: 4
  Unmatched TC: 8
  Mismatches: 0

============================================================
FINAL SUMMARY
============================================================
VLLM total time:     99508.46 us
TensorCast total:    89054.38 us
Difference:          -10454.09 us (-10.5%)
Matched:             32
Coverage:            93.5%

Outputs:
  Chrome trace:  /tmp/deepseek_results/chrome_trace.json
  Excel report:  /tmp/deepseek_results/deepseek_v3_comparison.xlsx
```

### 2.4 结果解读

| 指标 | 值 |
|------|------|
| VLLM 总时间 | ~100 ms |
| TensorCast 总时间 | ~89 ms |
| 差异 | -10.5% |
| 匹配算子数 | 32 |
| 覆盖率 | 93.5% |

DeepSeek-V3 的时间差异远小于 Qwen3（10.5% vs 80%），因为它不受 KV head 复制问题的影响。MoE 路由、专家计算和 all-to-all 通信都被准确捕捉。

**关键算子级对比**：

| VLLM 算子 | TC 等价物 | VLLM 耗时 | TC 耗时 | 差异 |
|-----------|-----------|-----------|---------|------|
| QuantBatchMatmulV3 | tensor_cast.static_quant_linear | 48,923 us | 45,972 us | -6.0% |
| FusedInferAttentionScore | tensor_cast.attention (MLA) | 12,376 us | 10,157 us | -17.9% |
| MoeDistributeDispatchV2 | permute_tokens + all_to_all | 4,832 us | 4,521 us | -6.4% |
| GroupedMatmul | tensor_cast.static_quant_linear | 8,645 us | 8,229 us | -4.8% |

---

## Excel 报告结构

生成的 Excel 文件包含 4 个工作表：

### 工作表 1：VLLM Operations
所有 VLLM 内核算子按类型聚合，按总耗时排序：
- 算子类型、数量、总耗时 (us)、平均耗时 (us)、核心类型、输入 shape

### 工作表 2：TensorCast Operations
所有 TensorCast 算子来自 chrome trace，按名称聚合：
- 算子名称、数量、总耗时 (us)、平均耗时 (us)

### 工作表 3：Sequence Comparison
按执行顺序逐位置匹配：
- 位置、VLLM 算子、TC 算子、VLLM 耗时 (us)、TC 耗时 (us)、差异 (%)、状态

颜色编码：
- **绿色**：差异 <= 20%（良好匹配）
- **黄色**：差异 20-50%（可接受）
- **红色**：差异 > 50%（需要调查）

### 工作表 4：Summary
配置参数和汇总指标：
- 模型 ID、设备、并行策略、量化方案
- VLLM 总时间、TC 总时间、总体差异
- 匹配/未匹配/不匹配数量、覆盖率百分比

---

## 对比总结

| 模型 | 类型 | VLLM 耗时 | TC 耗时 | 差异 | 算子数/步 | 覆盖率 |
|------|------|-----------|---------|------|----------|--------|
| Qwen3-32B | 稠密, TP=16 | 207 ms | 40 ms | -80.8%* | ~1,479 | 98.5% |
| DeepSeek-V3 | MoE, DP=32+EP | 100 ms | 89 ms | -10.5% | ~2,115 | 93.5% |

*Qwen3-32B 时间差异源于 KV head 配置（`num_kv_heads=8 < tp_size=16`）

### 关键差异

| 方面 | Qwen3-32B（稠密） | DeepSeek-V3（MoE） |
|------|-------------------|-------------------|
| 耗时最多的算子 | MatMulV2 (42%) | GroupedMatmul (21%) |
| 并行策略 | TP=16 | DP=32, EP=32 |
| 量化 | BF16（未量化） | W8A8_DYNAMIC |
| MoE 算子 | 无 | MoeDistribute*, GroupedMatmul |
| 注意力机制 | 标准注意力 | MLA（多头潜在注意力） |
| KV 缓存 | reshape_and_cache | concat_and_cache_mla |
| 映射 | default + qwen3 | 仅 default |
| 模型层数 | 64 | 61 |

---

## 常见问题排查

### "kernel_details.csv not found"
VLLM 分析目录必须指向直接包含 `kernel_details.csv` 的 `ASCEND_PROFILER_OUTPUT` 文件夹。路径通常如下：
```
.../profiling/<container_id>/ASCEND_PROFILER_OUTPUT/
```

### `/mnt/d/` 上的 PermissionError
Windows 挂载的驱动器可能导致输出失败。请改用 `/tmp/` 作为输出目录：
```bash
--output-dir /tmp/results/
```

### 阶段检测为 "prefill"，但数据实际为 decode
使用 `--phase decode` 手动指定。检测器依赖 `ReshapeAndCacheNdKernel` 的输入 shape；若缺失，请使用显式覆盖。

### 覆盖率百分比偏低
执行 `list-mappings` 确认分解映射覆盖了模型中的算子。将缺失的分解添加到模型特定的 YAML 覆盖文件中。

### 特定算子的时间差异较大
- **注意力**：大 TP 下 KV head 复制可能导致显著差异
- **通信**：模拟与实际硬件间的网络拓扑差异
- **MoE 路由**：VLLM 与 TensorCast 之间的 token 路由开销可能不同
