# 评估指标使用指南 (M1–M6)

本文档说明如何运行和解读算子性能数据库的六层评估指标。

## 快速参考

| 指标 | 一句话 | 需要什么 | 看什么 |
|------|--------|---------|--------|
| M1 | 多少次调用命中了？ | TC profiling 运行 | Debug 用，被 zero_cost 虚增 |
| M2 | 多少个逻辑算子命中了？ | TC profiling 运行 | GO/NO-GO 判定 |
| M3 | 多少个计算算子命中了？ | TC profiling 运行 | **核心进度指标** |
| M4 | 多少个 shape 变体命中了？ | TC profiling 运行 | 定位缺失 shape，指导数据采集 |
| M5 | 仿真延迟中多少有实测数据？ | TC profiling 运行 | 延迟加权覆盖率 (仿真视角) |
| M6 | Empirical 预测 vs 真实单次 forward pass | TC profiling 运行 + step_trace + kernel_details | **验收标准：0.85–1.15** |

## 运行 M1–M5 (在线指标)

M1–M5 在 TC profiling 模式运行时自动输出到日志。

### 命令

```bash
# Qwen3-32B Prefill (BF16, TP=16)
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 10 --query-length 4104 \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --quantize-linear-action DISABLED \
  --performance-model profiling --compile \
  --profiling-database tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5 \
  --log-level info

# Qwen3-32B Decode (BF16, TP=16)
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 16 --query-length 1 --context-length 4096 \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --quantize-linear-action DISABLED \
  --performance-model profiling --compile \
  --profiling-database tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5 \
  --log-level info

# DSv3 Prefill (W8A8, TP=8, DP=2, EP=16)
python3.10 -m tensor_cast.scripts.text_generate deepseek-ai/DeepSeek-V3 \
  --num-queries 1 --query-length 256 \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 8 --dp-size 2 --ep-size 16 \
  --word-embedding-tp row \
  --quantize-linear-action W8A8_STATIC \
  --performance-model profiling --compile \
  --profiling-database tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5 \
  --log-level info

# DSv3 Decode (W8A8, TP=8, DP=2, EP=16)
python3.10 -m tensor_cast.scripts.text_generate deepseek-ai/DeepSeek-V3 \
  --num-queries 16 --query-length 1 --context-length 4096 \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 8 --dp-size 2 --ep-size 16 \
  --word-embedding-tp row \
  --quantize-linear-action W8A8_STATIC \
  --performance-model profiling --compile \
  --profiling-database tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5 \
  --log-level info
```

### 输出示例

```
EmpiricalPerformanceModel: 35/46 ops matched (76.1%)              ← M1
  HITs (20 unique):
    aten.mm.default->MatMulV2 (x128)
    tensor_cast.all_reduce.default->hcom_allReduce_ (x64)
    ...
  MISSes (3 unique reasons):
    [shape_mismatch] kernel found, no matching shape in CSV: ...
    [unmapped] not in op_mapping.yaml: ...
Fused Op Match Rate: 10/20 (50.0%) [GO/NO-GO]                    ← M2
Fused Op Match Rate (excl zero_cost): 3/13 (23.1%) [Reference]   ← M3
Per-Shape Match Rate: 8/19 (42.1%)                                ← M4
  MISS shapes (11):
    aten.mm.default ((4096, 5120), (5120, 5120))
    tensor_cast.swiglu.default ((4096, 6912),)
    ...
Simulated Latency Coverage: 50.8% (12.723ms / 25.037ms)          ← M5
```

### 如何解读

- **M3 低但 M5 高**: 少量高延迟算子 (MatMul, Attention) 已匹配，但很多低延迟辅助算子未匹配。覆盖率按延迟算其实不差。
- **M4 的 MISS shapes 列表**: 直接告诉你需要为哪些 (算子, shape) 组合采集 microbenchmark 数据。
- **M5 接近 M3**: analytic 模型认为所有算子延迟差不多，没有特别突出的高延迟算子。
- **M5 远高于 M3**: 已匹配的少数算子恰好是延迟大户。

## 运行 M6 (半离线指标)

```
M6 = Empirical_HIT_total / Real_per_forward_pass
```

M6 = 1.0 表示完美预测。M6 > 1 = 高估，M6 < 1 = 低估。Phase 3 目标：0.85 ≤ M6 ≤ 1.15。

**只含 empirical HIT**（不含 analytic fallback 的 MISS ops），直接衡量已有 microbench 数据的质量。

两步：
1. 跑 TC profiling 并导出 JSON：`--export-metrics report.json`
2. 用 `compute_m6.py` 计算 M6（需要 ASCEND_PROFILER_OUTPUT 目录 + `--model` 参数）

### 前置条件

1. 已有对应场景的 profiling 数据（包含 `step_trace_time.csv` + `kernel_details.csv`）
2. 已跑过对应的 TC profiling 命令并导出了 metrics JSON

### 计算 M6

```bash
# Step 1: TC profiling + 导出 JSON (同 M1-M5 命令，加 --export-metrics)
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 10 --query-length 4104 \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --quantize-linear-action DISABLED \
  --performance-model profiling --compile \
  --profiling-database $DATA_DIR \
  --export-metrics results/qwen3_prefill_metrics.json

# Step 2: 计算 M6 (--model 用于估算 forward pass 数量)
python3.10 tools/perf_data_collection/compute_m6.py \
  --tc-report results/qwen3_prefill_metrics.json \
  --profiler-output /path/to/ASCEND_PROFILER_OUTPUT \
  --model qwen3
```

### 输出示例

```
============================================================
M6: Empirical E2E Prediction Ratio
============================================================

Empirical HIT total:    609,642.0 us (609.6 ms)
Real per-fwd:         1,146,669.8 us (1,146.7 ms)
TC full prediction:   1,407,989.9 us (1,408.0 ms)  [for reference]
  Step total:         5,733,348.9 us (Computing: 3,242,347.5 + Comm: 2,491,001.4)
  Forward passes:               5   (anchor: FusedInferAttentionScore [320])

M6 = 0.532  (TC / Real)
     underestimate by 47%
Phase 3 target: 0.85 ≤ M6 ≤ 1.15 [FAIL]
```

### 如何解读

- **M6 ≈ 1.0**: empirical 数据精准，已匹配算子的 microbench 延迟与真实运行一致
- **M6 < 1 (如 0.5)**: 低估，说明 MISS ops 贡献了大量真实延迟但没有 empirical 数据覆盖
- **M6 > 1 (如 2.0)**: 高估，microbench 数据偏高（isolation vs real workload 差异），或通信 microbench 与真实 serving 差距大
- **M6 远小于 1 但 M5 高**: M5 (analytic加权) 认为已匹配算子重要，但实际它们在真实延迟中占比小
- **unmatched 列表**: 诊断参考，显示未被 empirical 覆盖的 kernel 按时间排序

### Forward Pass 估算说明

`step_trace_time.csv` 的 `Step` 列为空时，聚合了整个 profiling 窗口（可能包含数十到数百次 forward pass）。
`compute_m6.py` 通过 `kernel_details.csv` 中的算子调用计数自动估算 forward pass 数量：

| 模型 | Anchor Kernel | 每层每 fwd pass 调用数 | 层数 |
|------|--------------|:---:|:---:|
| Qwen3 | FusedInferAttentionScore | 1 | 64 |
| DSv3 | DispatchFFNCombine | 1 | 58 (MoE layers) |

如果自动估算不准确，可使用 `--n-forward-passes N` 手动指定。

## 指标关系图

```
粗粒度                                          细粒度
(算子数量)                                      (延迟加权)

M1 ──→ M2 ──→ M3 ──→ M4
 │       │       │       │
 │    +悲观   -zero   +shape
 │    +融合    cost    独立计
 │
 └──────────────────────→ M5 (analytic 延迟加权, 在线)
                          │
                          └→ M6 (empirical-only ratio vs 真实 per-fwd, 半离线)
                              ↑
                              需要 step_trace_time.csv + kernel_details.csv
                              (估算 forward pass 数量)
```

## Phase 目标

| Phase | M3 目标 | M5 目标 | M6 目标 |
|-------|:---:|:---:|:---:|
| Phase 1 (✅) | 建立指标 | — | — |
| Phase 2 (✅) | > 50% | 建立指标 | 建立指标 |
| Phase 3 | — | > 80% | **0.85 ≤ M6 ≤ 1.15** |
