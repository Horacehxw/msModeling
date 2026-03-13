# 通信算子对齐报告

**日期**：2026-03-13
**负责人**：HDY
**数据来源**：Phase 2 — 16 组 trace（Qwen3-32B 8 组 + DSV3 8 组）
**Microbench 版本**：hccl/v8.5（C10-1 重新采集，单 session，含 tier=1/2）

---

## 1. 采集配置

| 模型 | TP | DP | 量化 | 其他 |
|------|----|----|------|------|
| Qwen3-32B | 16 | 1 | BF16 | enforce-eager + cudagraph FULL_DECODE_ONLY |
| DSV3 | 8 | 2 | W8A8 | enforce-eager + cudagraph FULL_DECODE_ONLY, max-num-seqs=64 |

**Decode sweep**：batch=1/4/8/16/32（input_len=1, output_len=3）
**Prefill sweep**：ISL=256/1024/4096（num_prompts=1, output_len=3）

每组 trace 含 kernel_details.csv + step_trace_time.csv + trace_view.json。

---

## 2. 聚合 Overhead Factor

`Factor = Stage / (Computing + Communication_Not_Overlapped)`，来自 step_trace_time.csv。

| 模型 | 配置 | Stage(ms) | Comp+CommNO(ms) | Free(ms) | Factor |
|------|------|-----------|-----------------|----------|--------|
| Qwen3 | decode batch=1 | 1463 | 1131 | 332 | 1.29x |
| Qwen3 | decode batch=4 | 2218 | 2087 | 131 | 1.06x |
| Qwen3 | decode batch=8 | 3361 | 3299 | 62 | 1.02x |
| Qwen3 | decode batch=16 | 4981 | 4950 | 30 | 1.01x |
| Qwen3 | decode batch=32 | 7822 | 7698 | 124 | 1.02x |
| Qwen3 | prefill ISL=256 | 1127 | 1100 | 27 | 1.02x |
| Qwen3 | prefill ISL=1024 | 1125 | 1105 | 20 | 1.02x |
| Qwen3 | prefill ISL=4096 | 1244 | 1132 | 111 | 1.10x |
| DSV3 | decode batch=1 | 2654 | 2355 | 297 | 1.13x |
| DSV3 | decode batch=4 | 5763 | 5516 | 244 | 1.04x |
| DSV3 | decode batch=8 | 5425 | 5356 | 69 | 1.01x |
| DSV3 | decode batch=16 | 6172 | 6105 | 67 | 1.01x |
| DSV3 | decode batch=32 | 8185 | 8127 | 55 | 1.01x |
| DSV3 | prefill ISL=256 | 3822 | 3775 | 47 | 1.01x |
| DSV3 | prefill ISL=1024 | 3856 | 3810 | 46 | 1.01x |
| DSV3 | prefill ISL=4096 | 4811 | 3316 | 1493 | 1.45x |

**结论**：
- batch >= 4 时 overhead < 6%，batch >= 8 时 < 2%，可忽略或用常数 ~1.05x 近似
- batch=1 的高 overhead（13-29%）来自 host 调度开销在小负载下的放大
- DSV3 prefill ISL=4096 的 1.45x 异常（1493ms free time）需进一步排查

---

## 3. 通信算子延迟 vs Batch Size

有效传输延迟（过滤 <100us 空调用，取 p10-p90 stable median）。

**Qwen3-32B Decode（TP=16, Dense）**：

| Batch | allGather(us) | allReduce(us) |
|-------|--------------|--------------|
| 1 | 1068 | 3251 |
| 4 | 1192 | 1587 |
| 8 | 1062 | 2379 |
| 16 | 1064 | 3234 |
| 32 | 1045 | 3715 |

- allGather：~1060us，跨 batch 极稳定（range 1045-1192us）
- allReduce：1587-3715us，双峰分布（~1550us / ~3700us），与 batch size 无明显相关性

**DSV3 Decode（TP=8 DP=2, MoE）**：

| Batch | allGather(us) | reduceScatter(us) |
|-------|--------------|------------------|
| 1 | 815 | 4150 |
| 4 | 1158 | 4124 |
| 8 | 1111 | 4066 |
| 16 | 772 | 4104 |
| 32 | 1412 | 4452 |

- allGather：772-1412us，比 Qwen3 略低（TP=8 vs TP=16）
- reduceScatter：~4100us 稳定，远高于 Qwen3（MoE expert parallel 通信量大）
- 40-60% 的 comm 调用为空操作（<100us），MoE expert parallel 特征

---

## 4. communication_profiled 当前值

基于 Phase 2 cross-batch 数据，op_mapping.yaml 中的内联值：

| 算子 | latency_us | 数据来源 |
|------|-----------|---------|
| hcom_allGather_ | 1062 | Qwen3 Decode cross-batch median |
| hcom_reduceScatter_ | 1184 | Qwen3 Prefill ISL=4096 stable median |
| hcom_allReduce_ | 1932 | Qwen3 Decode cross-batch median（双峰取中位） |

---

## 5. 通信查询架构决策

### 5.1 Bench-Profiling Gap 量化

Bench CSV 测量的是纯通信时间，Profiling Duration 包含通信 + 调度等待（pipeline bubble）。两者语义不同，gap 不是简单常数：

| 算子 | bench→profiling ratio | 稳定性 | 原因 |
|------|----------------------|--------|------|
| allGather | 1.8-3.6x | 较稳定 | 调度 gap 小 |
| allReduce (Qwen3) | 9.6-22.5x | 不稳定 | 含前序 matmul pipeline bubble |
| reduceScatter (DSV3) | 5.4-12.7x | 不稳定 | 含跨算子等待 |

整体理想校准系数：Qwen3 ~16x，DSV3 ~7x，跨模型、跨 batch 均不稳定。

### 5.2 三种策略端到端误差对比

| 策略 | Qwen3 误差 | DSV3 误差 | 评估 |
|------|-----------|----------|------|
| Inline profiled（当前） | 20-32% | 9-34% | 短期最优 |
| Bench CSV（raw） | 69-91% | 15-48% | 严重低估 |
| Bench + 简单校准 | 58-91% | 12-41% | ratio 不稳定，无法收敛 |

### 5.3 决策

**"CSV first + 简单 bench_to_profiled_ratio" 方案不可行**。allReduce/reduceScatter 的 profiling Duration 本质是"通信 + 调度等待"，bench 值无论乘多少都无法稳定逼近。

**当前方案（inline profiled 优先）维持不变**，作为短期最优解（20-34% 误差）。

```
_lookup_comm() 查询优先级（维持现状）：
  ① communication_profiled（op_mapping.yaml 内联）→ confidence=0.85
  ② bench CSV（communication_data_ref 目录）→ confidence=0.9
  ③ CommAnalyticModel（最终 fallback）
```

**后续精度提升路径**：

**路径 1：per-config profiling CSV（高优先级）**

从 profiling kernel_details 提取 per-(model, batch_size, kernel_type) 延迟，替代 bench CSV。数据已有（Phase 2 的 16 组 trace），需要工具化入库。

预期精度（已见配置精确匹配时，误差 ≈ Free / Stage）：

| 模型 | 配置 | 当前 inline 误差 | per-config CSV 预期误差 | 改善 |
|------|------|-----------------|----------------------|------|
| Qwen3 | decode batch=1 | ~30% | ~23% | 有限（Free 占比高） |
| Qwen3 | decode batch=4 | ~25% | ~6% | 显著 |
| Qwen3 | decode batch=8~32 | ~20% | **1-2%** | 接近理论下限 |
| Qwen3 | prefill ISL=256~1024 | ~22% | ~2% | 接近理论下限 |
| DSV3 | decode batch=1 | ~20% | ~11% | 中等 |
| DSV3 | decode batch=4 | ~15% | ~4% | 显著 |
| DSV3 | decode batch=8~32 | ~10% | **<1.5%** | 接近理论下限 |
| DSV3 | prefill ISL=256~1024 | ~12% | ~1% | 接近理论下限 |

未见配置（插值）：comm 延迟在 batch=8~32 范围内极稳定（allGather ~1060us，变化 <15%），线性插值预期误差 3-5%。

结论：生产主力场景（batch >= 8）E2E 误差从 10-20% 压到 1-2%，是当前最高性价比的精度提升路径。

**路径 2：显式建模调度开销（中优先级）**

将 comm Duration 拆分为 `pure_comm + scheduling_wait`，分别预测。主要改善 batch=1 场景（Free time 占 13-29%），对 batch >= 8 场景收益有限（Free < 2%）。

---

## 6. DSV3 特有注意事项

1. **使用 AicpuKernel 而非 hcom_xxx_**：DSV3 profiling 里 `hcom_allGather_` 包含大量 overlap 执行的短耗时记录（<20us），应使用 `allgatherAicpuKernel` / `reduce_scatterAicpuKernel`
2. **DispatchFFNCombine 封装 EP 通信**：alltoall 被封装在 DispatchFFNCombine 内，无法直接对比，待 C11 拆分
3. **DSV3 无独立 allReduce**：MC2 融合，profiling 无独立 `hcom_allReduce_` 记录

**四算子可观测性**：

| 算子 | Qwen3-32B | DSV3 |
|------|-----------|------|
| allGather | hcom_allGather_ | allgatherAicpuKernel |
| reduceScatter | hcom_reduceScatter_ | reduce_scatterAicpuKernel |
| allReduce | hcom_allReduce_（含调度等待） | 不存在（MC2 融合） |
| allToAll | 不存在（dense 模型） | 不存在（DispatchFFNCombine 封装） |

---

## 7. 遗留问题

| # | 问题 | 优先级 | 行动 |
|---|------|--------|------|
| P5 | DSV3 allToAll 封装在 DispatchFFNCombine，无法直接验证 | 中 | C11：拆分子 kernel |
| P8 | Overhead factor batch=1 时 13-29%，需特殊处理 | 中 | 评估是否需要 per-batch-size factor |
| P9 | DSV3 blank time（MoE EP 协调 + DFC 内部等待） | 中 | 分析 blank 构成 |
| P11 | allReduce 双峰分布（~1550us / ~3700us），单一值不准确 | 中 | 考虑按 layer position 区分 |
| P12 | ~~bench-profiling gap 校准系数~~ | ~~高~~ | 方案验证不可行（ratio 9.6-22.5x 不稳定），改为 per-config profiling CSV 路径 |
| P13 | per-config profiling CSV 入库工具 | 高 | 从 Phase 2 trace 提取 per-(model, batch, kernel_type) 延迟，工具化入库 |
