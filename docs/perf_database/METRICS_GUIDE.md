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
| M6 | 真实 E2E 中多少被覆盖？ | TC profiling 运行 + kernel_details.csv | **辅助验收标准** |

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
  --perf-database tensor_cast/performance_model/perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5 \
  --log-level info

# Qwen3-32B Decode (BF16, TP=16)
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 16 --query-length 1 --context-length 4096 \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --quantize-linear-action DISABLED \
  --performance-model profiling --compile \
  --perf-database tensor_cast/performance_model/perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5 \
  --log-level info

# DSv3 Prefill (W8A8, TP=8, DP=2, EP=16)
python3.10 -m tensor_cast.scripts.text_generate deepseek-ai/DeepSeek-V3 \
  --num-queries 1 --query-length 256 \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 8 --dp-size 2 --ep-size 16 \
  --word-embedding-tp row \
  --quantize-linear-action W8A8_STATIC \
  --performance-model profiling --compile \
  --perf-database tensor_cast/performance_model/perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5 \
  --log-level info

# DSv3 Decode (W8A8, TP=8, DP=2, EP=16)
python3.10 -m tensor_cast.scripts.text_generate deepseek-ai/DeepSeek-V3 \
  --num-queries 16 --query-length 1 --context-length 4096 \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 8 --dp-size 2 --ep-size 16 \
  --word-embedding-tp row \
  --quantize-linear-action W8A8_STATIC \
  --performance-model profiling --compile \
  --perf-database tensor_cast/performance_model/perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5 \
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

M6 = Σ(empirical HIT duration) / (Computing + Communication_NotOverlapped)

两步：
1. 跑 TC profiling 并导出 JSON：`--export-metrics report.json`
2. 用 `compute_m6.py` 计算 M6（需要 ASCEND_PROFILER_OUTPUT 目录）

### 前置条件

1. 已有对应场景的 profiling 数据（包含 `step_trace_time.csv` + `kernel_details.csv`）
2. 已跑过对应的 TC profiling 命令并导出了 metrics JSON

### Profiling 数据位置

```
# Phase 1 E2E 测试数据
/mnt/d/Data/Profiling/Profiling-0313-phase1-e2e-test/
  profilier_prefill_qwen32b-input4096-output1/   ← Qwen3 Prefill
  profilier_decode_qwen32b-input4k/              ← Qwen3 Decode
  profilier_prefill_dsv3-input4096-output1/      ← DSv3 Prefill
  profilier_decode_dsv3-input4096-output1536/    ← DSv3 Decode
```

每个目录下有 `{hash}_ascend_pt/ASCEND_PROFILER_OUTPUT/` 包含 `step_trace_time.csv` 和 `kernel_details.csv`。

### 计算 M6

```bash
# Step 1: TC profiling + 导出 JSON (同 M1-M5 命令，加 --export-metrics)
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 10 --query-length 4104 --word-embedding-tp row \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --quantize-linear-action DISABLED \
  --performance-model profiling --compile \
  --perf-database $DATA_DIR \
  --export-metrics results/qwen3_prefill_metrics.json

# Step 2: 计算 M6
python3.10 tools/perf_data_collection/compute_m6.py \
  --tc-report results/qwen3_prefill_metrics.json \
  --profiler-output /path/to/ASCEND_PROFILER_OUTPUT
```

### 输出示例

```
============================================================
M6: Empirical Prediction Coverage
============================================================

Empirical HIT duration: 3,842.5 us
Step duration:          5,733.3 us (Computing: 3,242.3 + Comm: 2,491.0)

M6 = 67.0%

Top unmatched kernels (from kernel_details.csv, excl AicpuKernel):
  Type                                     Duration(us)    %KD Total
  ---------------------------------------- -------------- ----------
  FusedInferAttentionScore                      346,767.9       6.0%
  split_qkv_rmsnorm_rope                       309,817.0       5.4%
  ReshapeAndCacheNdKernel                        48,466.1       0.8%
  TensorMove                                    10,630.3       0.2%
```

### 如何解读

- **M6 = 85%+**: empirical 模型覆盖了大部分真实执行时间
- **M6 vs M5 差异大**: analytic 模型对算子延迟权重判断有偏差
- **M6 > 100%**: microbench 数据系统性高估了 kernel 延迟（isolation vs contention 差异）
- **unmatched 列表**: 按 kernel_details.csv duration 排序的缺口（诊断参考，非 M6 分母）

### 分母选择说明

使用 `step_trace_time.csv` 的 `Computing + Communication(Not Overlapped)` 而非 `kernel_details.csv sum`：

| 场景 | kernel_details sum | step_trace_time | 差异 |
|------|-------------------|-----------------|------|
| Qwen3 Prefill | 5.74M us | 5.73M us | 0.1% |
| DSv3 Prefill | 4.30M us | 4.19M us | 2.5% |
| Qwen3 Decode | 4.39M us | 2.93M us | **49.7%** |
| DSv3 Decode | 8.16M us | 4.73M us | **72.5%** |

Decode 场景 CUDAGraph 导致计算/通信 overlap，kernel_details sum >> wall-clock。step_trace_time 已处理 overlap，是 profiler 的权威 E2E 分解。

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
                          └→ M6 (empirical 预测 vs 真实 E2E, 半离线)
                              ↑
                              需要 step_trace_time.csv
```

## Phase 目标

| Phase | M3 目标 | M5 目标 | M6 目标 |
|-------|:---:|:---:|:---:|
| Phase 1 (✅) | 建立指标 | — | — |
| Phase 2 (✅) | > 50% | 建立指标 | 建立指标 |
| Phase 3 | — | > 80% | 辅助验收 (TBD) |
