# 通信算子对齐报告

**创建日期**：2026-03-12
**最后更新**：2026-03-27
**版本**：v3.2（增加数据汇总）
**负责人**：HDY
**数据来源**：Qwen3-32B 多 ISL profiling + DSV3 多 ISL profiling（vLLM 0.18.0 + CANN 8.5 + AIV 模式，均关闭 stack）
**Microbench 版本**：hccl/v8.5（C10-1 采集，单 session，含 tier=1/2，alternating + kernel 两模式）

---

## Executive Summary

| # | 核心结论 | 数据支撑 |
|---|---------|---------|
| C1 | **alternating 模式消除 1-5MB 预热偏高** | Qwen3 1.3-5MB 从偏高 19-63% 降到 ±3%；DSV3 3.5MB 从偏高 45-51% 降到 ±3% |
| C2 | **固定开销模型：仿真值 = bench + 固定开销** | Qwen3 AR +7.7us, AG +14.6us；DSV3 AG +1.2us, RS +2.0us；与 msg_bytes 无关 |
| C3 | **HCCL 协议切换点 768KB nd=8** | per_device ≈ 60KB 处 5x 跳变；bench 无法复现生产值（+107-147%）；用 profiler P50 |
| C4 | **bench CSV 纯通信时间准确** | kernel/bench ratio 0.59-0.82x；AIV 模式对 HCCL 微基准无显著影响 |
| C5 | **E2E 预测策略** | Prefill ≥1MB: 直接用 alternating bench；Decode 小消息: kernel bench + 固定开销；768KB: profiler P50 |

---

## 2. 采集配置

### 2.1 环境配置

**软件版本**：CANN 8.5 / vLLM 0.18.0 / PyTorch 2.9.0

| 模型 | TP | DP | 量化 | 关键配置 |
|------|----|----|------|---------|
| Qwen3-32B | 16 | 1 | BF16 | async-scheduling, CUDAGraph FULL_DECODE_ONLY, max-num-batched-tokens=65536, block-size=128 |
| DSV3 | 8 | 2 | W8A8 | async-scheduling, CUDAGraph FULL_DECODE_ONLY, max-num-seqs=8, max-num-batched-tokens=8192, EP |

关键环境变量：

```bash
# 两个模型共用
HCCL_OP_EXPANSION_MODE="AIV"
TASK_QUEUE_ENABLE=1
VLLM_ASCEND_ENABLE_NZ=1

# Qwen3 特有
VLLM_ASCEND_ENABLE_FLASHCOMM1=1    # Sequence Parallelism

# DSV3 特有
VLLM_ASCEND_ENABLE_FUSED_MC2=1     # MC2 融合（TP 通信）
VLLM_ASCEND_ENABLE_FLASHCOMM1=0    # DSV3 关闭
VLLM_ASCEND_ENABLE_MLAPO=1         # MLA 优化
```

### 2.2 负载矩阵

**Qwen3-32B**（8 组场景）：

| 场景 | input_len | output_len | concurrency | request_rate | 说明 |
|------|-----------|------------|-------------|-------------|------|
| Prefill | 1024 | 1 | 16 | 10 | 短 ISL prefill |
| Prefill | 2048 | 1 | 16 | 10 | 中 ISL prefill |
| Prefill | 4096 | 1 | 16 | 10 | 标准 ISL prefill |
| Prefill | 8192 | 1 | 16 | 10 | 长 ISL prefill |
| Decode | 4096 | 1536 | 1 | 1 | 低并发 decode |
| Decode | 4096 | 1536 | 4 | 2 | 中低并发 decode |
| Decode | 4096 | 1536 | 16 | 2 | 中并发 decode |
| Decode | 4096 | 1536 | 32 | 4 | 高并发 decode |

**DSV3**（10 组场景）：

| # | 场景 | input_len | output_len | concurrency | request_rate | 说明 |
|---|------|-----------|------------|-------------|-------------|------|
| 1 | Prefill | 512 | 1 | 8 | 10 | 短 ISL，batch 4 条 |
| 2 | Prefill | 1024 | 1 | 8 | 10 | batch 2 条 |
| 3 | Prefill | 2048 | 1 | 8 | 10 | batch 1 条 |
| 4 | Prefill | 4096 | 1 | 8 | 10 | 触发 chunked prefill |
| 5 | Prefill | 4096 | 1 | 1 | 1 | chunked prefill 串行基线 |
| 6 | Decode | 4096 | 1536 | 1 | 1 | 串行基线 |
| 7 | Decode | 4096 | 1536 | 2 | 1 | 低并发 |
| 8 | Decode | 4096 | 1536 | 4 | 2 | 中并发 |
| 9 | Decode | 4096 | 1536 | 8 | 4 | 满载 |
| 10 | Decode | 2048 | 1536 | 8 | 4 | 短 ISL 满载 |

---

## 3. 生产 Profiling 数据

### 3.1 Step Trace 概览

`Factor = Stage / (Computing + Communication_Not_Overlapped)`，来自 step_trace_time.csv。

| 模型 | 场景 | Stage(ms) | Computing(ms) | Comm_NO(ms) | Free(ms) | Factor | Free% |
|------|------|-----------|--------------|-------------|----------|--------|-------|
| Qwen3 | Prefill (ISL=8192) | 4375 | 2372 | 1594 | 409 | 1.10x | 9.3% |
| Qwen3 | Decode (c32r4) | 3196 | 2190 | 849 | 156 | 1.05x | 4.9% |
| DSV3 | Prefill (ISL=4096 c8) | 3540 | 3055 | 353 | 132 | 1.039x | 3.7% |
| DSV3 | Decode (c1) | 3407 | 1454 | 1798 | 156 | 1.048x | 4.6% |

**结论**：生产环境 overhead factor 整体 1.04x-1.10x，调度开销可控。

#### Qwen3-32B 多 ISL Step Trace

| 场景 | Stage(ms) | Computing(ms) | Comm_NO(ms) | Free(ms) | Factor | Free% |
|------|-----------|--------------|-------------|----------|--------|-------|
| Prefill ISL=1024 | 8200 | 3546 | 2690 | 1965 | 1.31x | 24.0% |
| Prefill ISL=2048 | 8304 | 3596 | 2696 | 2012 | 1.32x | 24.2% |
| Prefill ISL=4096 (rank3) | 3072 | 1096 | 1467 | 509 | 1.20x | 16.6% |
| Prefill ISL=8192 | 4375 | 2372 | 1594 | 409 | 1.10x | 9.3% |
| Decode c1r1 | 3115 | 2235 | 668 | 212 | 1.07x | 6.8% |
| Decode c4r2 | 3076 | 2109 | 775 | 193 | 1.07x | 6.3% |
| Decode c16r2 | 3083 | 2055 | 826 | 203 | 1.07x | 6.6% |
| Decode c32r4 | 3196 | 2190 | 849 | 156 | 1.05x | 4.9% |

#### DSV3 多 ISL/多 Concurrency Step Trace

| 场景 | Stage(ms) | Computing(ms) | Comm_NO(ms) | Free(ms) | Factor | Free% |
|------|-----------|--------------|-------------|----------|--------|-------|
| Prefill ISL=512 | 3352 | 2697 | 481 | 173 | 1.055x | 5.2% |
| Prefill ISL=1024 | 3241 | 2512 | 638 | 91 | 1.029x | 2.8% |
| Prefill ISL=2048 | 3464 | 2945 | 422 | 97 | 1.029x | 2.8% |
| Prefill ISL=4096 c8 | 3540 | 3055 | 353 | 132 | 1.039x | 3.7% |
| Prefill ISL=4096 c1 | 3437 | 2276 | 335 | 826 | 1.317x | 24.0% |
| Decode c1 | 3407 | 1454 | 1798 | 156 | 1.048x | 4.6% |
| Decode c2 | 3332 | 2063 | 917 | 352 | 1.118x | 10.6% |
| Decode c4 | 3487 | 2269 | 1028 | 191 | 1.058x | 5.5% |
| Decode c8 | 3466 | 1883 | 652 | 931 | 1.367x | 26.9% |
| Decode ISL=2048 c8 | 3432 | 2308 | 980 | 144 | 1.044x | 4.2% |

---

### 3.2 通信算子耗时（kernel_details）

取 p10-p90 stable median，kernel_details.csv 中 `hcom_*` 算子。

#### Qwen3-32B（TP=16, Dense, BF16, num_devices=16）

**Prefill 多 ISL kernel_details**：

| ISL | 算子 | Count | Stable Median(us) | P10(us) | P90(us) |
|-----|------|-------|-------------------|---------|---------|
| 1024 | allGather | 390 | 3,141 | 3,125 | 3,165 |
| 1024 | reduceScatter | 387 | 3,796 | 3,775 | 3,822 |
| 4096 (rank3) | allGather | 2,080 | 186 | 176 | 368 |
| 4096 (rank3) | reduceScatter | 2,064 | 433 | 223 | 703 |
| 8192 | allGather | 390 | 1,657 | 708 | 3,150 |
| 8192 | reduceScatter | 387 | 2,115 | 877 | 3,832 |

**Decode 多 Concurrency kernel_details**：

| Concurrency | 算子 | Count | Stable Median(us) |
|-------------|------|-------|-------------------|
| c1r1 | allReduce | 25,671 | 19.8 |
| c4r2 | allReduce | 25,800 | 21.7 |
| c16r2 | allReduce | 20,511 | 20.2 |
| c32r4 | allReduce | 14,190 | 24.1 |

#### DSV3（TP=8, DP=2, EP, W8A8, num_devices=8）

**Prefill kernel_details**（主力 ISL）：

| ISL | 算子 | Count | Stable Median(us) | P10(us) | P90(us) |
|-----|------|-------|-------------------|---------|---------|
| 2048 | allGather | 1820 | 63.9 | 28.4 | 90.1 |
| 2048 | reduceScatter | 910 | 208.0 | 197.9 | 644.6 |
| 4096 c8 | allGather | 1820 | 63.1 | 28.2 | 88.5 |
| 4096 c8 | reduceScatter | 910 | 207.5 | 198.7 | 226.4 |

**Decode kernel_details**（c8 满载）：

| 场景 | 算子 | Count | Stable Median(us) | P10(us) | P90(us) |
|------|------|-------|-------------------|---------|---------|
| c8 | allGather | 2600 | 8.8 | 6.4 | 385.8 |
| c8 | reduceScatter | 1300 | 17.3 | 8.1 | 795.2 |

---

## 4. Bench 对齐分析

### 4.1 全景对比

| 模型 | 阶段 | 算子 | msg_bytes | bench(us) | profiler(us) | 偏差 |
|------|------|------|-----------|-----------|-------------|------|
| Qwen3 | prefill | allGather | 1.3MB | 176.7 | 182.5 | -3% |
| Qwen3 | prefill | allGather | 3.8MB | 366.6 | 366.6 | 0% |
| Qwen3 | prefill | allGather | 5.0MB | 435.1 | 432.3 | +1% |
| Qwen3 | prefill | allGather | 8.8MB | 679.0 | 722.4 | -6% |
| Qwen3 | prefill | allGather | 21.4MB | 1,628.0 | 1,657.2 | -2% |
| Qwen3 | prefill | allGather | 40.0MB | 3,080.2 | 3,141.1 | -2% |
| Qwen3 | prefill | reduceScatter | 1.3MB | 216.3 | 221.2 | -2% |
| Qwen3 | prefill | reduceScatter | 3.8MB | 440.7 | 442.1 | 0% |
| Qwen3 | prefill | reduceScatter | 8.8MB | 808.5 | 887.2 | -9% |
| Qwen3 | prefill | reduceScatter | 21.4MB | 2,044.3 | 2,115.1 | -3% |
| Qwen3 | prefill | reduceScatter | 40.0MB | 3,664.6 | 3,795.8 | -3% |
| Qwen3 | decode | allReduce | 160KB | 12.1 | 19.8 | +7.7us |
| Qwen3 | decode | allGather | 74KB | 12.5 | 25.9 | +13.3us |
| Qwen3 | decode | allGather | 278KB | 31.3 | 46.5 | +15.3us |
| Qwen3 | decode | allGather | 297KB | 33.0 | 48.2 | +15.2us |
| DSV3 | prefill | allGather | 3.5MB | 166.9 | 172.0 | -3% |
| DSV3 | prefill | reduceScatter | 3.5MB | 202.1 | 197.9 | +2% |
| DSV3 | decode | allGather | 1KB | 5.4 | 6.4 | +1.0us |
| DSV3 | decode | allGather | 7KB | 5.5 | 6.8 | +1.4us |
| DSV3 | decode | allGather | 14KB | 5.6 | 7.0 | +1.3us |
| DSV3 | decode | reduceScatter | 14KB | 6.1 | 8.1 | +2.0us |

Prefill 大消息偏差 ±6%（比例一致），Decode 小消息偏差为固定绝对值（与 msg_bytes 无关）。

### 4.2 异常点与修正

| 异常点 | 现象 | 原因 | 修正方式 | 修正后偏差 |
|--------|------|------|---------|-----------|
| DSV3 allGather 768KB nd=8 | bench 偏高 107-147% | HCCL 协议切换点，bench 与生产选择不同传输协议 | 使用 profiler P50（75.4us） | 0% |
| Qwen3 decode allReduce | bench 12.1us vs profiler 19.8us | 生产环境调度链路固定延迟 | bench + 7.7us | ±1us |
| Qwen3 decode allGather | bench 12.5-33us vs profiler 25.9-48.2us | 同上，TP=16 开销更大 | bench + 14.6us | ±1us |
| DSV3 decode allGather | bench 5.4-5.6us vs profiler 6.4-7.0us | 同上，TP=8 开销较小 | bench + 1.2us | ±0.2us |

### 4.3 分场景明细

bench 实测结果与生产环境 profiler 的逐场景对比。bench 数据来自 alternating / kernel 两种采集模式，profiler 数据来自 kernel_details 的 hcom_* Duration。

**Qwen3-32B Prefill**（bench 模式：alternating, TP=16, nd=16）：

| ISL | 算子 | msg_bytes | bench(us) | profiler(us) | 偏差 |
|-----|------|-----------|-----------|-------------|------|
| 4096 | allGather | 1.3MB | 176.7 | 182.5 | -3% |
| 4096 | allGather | 3.8MB | 366.6 | 366.6 | 0% |
| 4096 | allGather | 5.0MB | 435.1 | 432.3 | +1% |
| 8192 | allGather | 8.8MB | 679.0 | 722.4 | -6% |
| 8192 | allGather | 21.4MB | 1,628.0 | 1,657.2 | -2% |
| 1024 | allGather | 40.0MB | 3,080.2 | 3,141.1 | -2% |
| 4096 | reduceScatter | 1.3MB | 216.3 | 221.2 | -2% |
| 4096 | reduceScatter | 3.8MB | 440.7 | 442.1 | 0% |
| 8192 | reduceScatter | 8.8MB | 808.5 | 887.2 | -9% |
| 8192 | reduceScatter | 21.4MB | 2,044.3 | 2,115.1 | -3% |
| 1024 | reduceScatter | 40.0MB | 3,664.6 | 3,795.8 | -3% |

**Qwen3-32B Decode**（bench 模式：kernel, TP=16, nd=16）：

| Concurrency | 算子 | msg_bytes | bench(us) | profiler(us) | diff(us) | 判定 |
|-------------|------|-----------|-----------|-------------|----------|------|
| c=1 | allReduce | 160KB | 12.1 | 19.8 | +7.7 | 固定开销 |
| c=4 | allGather | 74KB | 12.5 | 25.9 | +13.3 | 固定开销 |
| c=16 | allGather | 278KB | 31.3 | 46.5 | +15.3 | 固定开销 |
| c=16 | allGather | 297KB | 33.0 | 48.2 | +15.2 | 固定开销 |

**DSV3 Prefill**（bench 模式：alternating, TP=8, nd=8）：

| ISL | 算子 | msg_bytes | bench(us) | profiler(us) | 偏差 |
|-----|------|-----------|-----------|-------------|------|
| 2048 | allGather | 3.5MB | 166.9 | 172.0 | -3% |
| 2048 | reduceScatter | 3.5MB | 202.1 | 197.9 | +2% |

**DSV3 Decode**（bench 模式：kernel, c=8 满载, TP=8, nd=8）：

| 算子 | msg_bytes | bench(us) | profiler(us) | diff(us) | 判定 |
|------|-----------|-----------|-------------|----------|------|
| allGather | 1KB | 5.4 | 6.4 | +1.0 | 固定开销 |
| allGather | 7KB | 5.5 | 6.8 | +1.4 | 固定开销 |
| allGather | 14KB | 5.6 | 7.0 | +1.3 | 固定开销 |
| reduceScatter | 14KB | 6.1 | 8.1 | +2.0 | 固定开销 |

### 4.4 固定开销

Decode 小消息场景，bench 与 profiler 之间存在与 msg_bytes 无关的固定差值，仿真时应在 bench 值上叠加。

| 算子 | TP | 固定开销(us) | 数据点 | 来源 |
|------|----|-------------|--------|------|
| allReduce | 16 | +7.7 | 4（Qwen3 c1-c32） | profiler - bench 差值 |
| allGather | 16 | +14.6 | 3（Qwen3 c4/c16） | profiler - bench 差值 |
| allGather | 8 | +1.2 | 4（DSV3 c8, 1-14KB） | profiler - bench 差值 |
| reduceScatter | 8 | +2.0 | 1（DSV3 c8, 14KB） | profiler - bench 差值 |

核心结论：

1. **固定开销与 TP 数量正相关，与具体模型无关**：TP=16 的开销（7.7-14.6us）显著高于 TP=8（1.2-2.0us），因为更多 rank 参与通信时 group lookup 和 barrier 同步链路更长
2. 开销与 msg_bytes 无关，是生产环境 vLLM scheduler dispatch → c10d 包装 → HCCL group lookup → stream sync 等调度链路的固定延迟
3. 仿真公式：`仿真值 = bench(msg_bytes) + 固定开销(TP)`

---

### 4.5 协议切换异常

> DSV3 nd=8 allGather 768KB 在所有 bench 模式下都远高于生产 profiler，属于 HCCL 内部协议切换导致的不可复现区间。

**768KB 三模式 bench vs 生产 profiler**：

| bench 模式 | 768KB nd=8 (us) | 生产 profiler (us) | 偏高 |
|-----------|----------------|-------------------|------|
| alternating | 186.4 | 75.4 | +147% |
| kernel (profiler) | 156.4 | 75.4 | +107% |
| event | 160.0 | 75.4 | +112% |

三种模式均偏高 107-147%，说明这不是 bench 模式问题，而是 HCCL 在独立 microbench 与生产环境中选择了不同的传输协议。

**nd=8 allGather kernel Duration 跳变**：

| msg_bytes | per_device | kernel Duration(us) |
|-----------|-----------|---------------------|
| 465KB | 58KB | 30.4 |
| 596KB | 75KB | 155.3 ← 5x 跳变 |
| 768KB | 96KB | 156.4 |
| 946KB | 118KB | 171.0 |

per_device ≈ 60KB 处发生 5x 跳变，对应 HCCL 从小消息协议（如 recursive halving-doubling）切换到大消息协议（如 ring）。生产环境中 768KB 可能仍走小消息协议（因 pipeline 上下文不同），导致 bench 无法复现。

**处理策略**：768KB 属于协议切换不可复现区间，**不使用 bench 值，改用 profiler P50（75.4us）**。

---

---

## 附录

### A. AIV 模式验证

AIV 模式微基准 vs 原非 AIV 微基准（num_devices=16, tier=1）：

| 算子 | 小消息量(≤1MB) | 大消息量(≥16MB) | 结论 |
|------|---------------|----------------|------|
| allReduce | 变化 <12% | 变化 <1% | 无显著差异 |
| allGather | 变化 <23% | 变化 <1% | 小消息量有波动，大消息量一致 |
| reduceScatter | 变化 <10% | 变化 <1% | 无显著差异 |
| alltoallv | 变化 <27% | 变化 <18% | alltoallv 小消息量 AIV 略快 |

**结论：AIV 模式对独立 HCCL 微基准无显著加速效果。**

### B. bench CSV 插值验证

| 算子 | msg_bytes | 实测(us) | 插值(us) | 偏差 |
|------|-----------|---------|---------|------|
| reduceScatter TP=8 | 3.5MB | 255.1 | 262.3 | 3% |
| allGather TP=8 | 3.5MB | 252.1 | 249.0 | 1% |

### C. DSV3 特有注意事项

1. DSV3 通信算子为 `hcom_allGather_` / `hcom_reduceScatter_`，无 `allgatherAicpuKernel` 路径
2. MC2 融合：所有场景无独立 allReduce，TP 通信走 MC2 融合算子
3. DispatchFFNCombine 封装 EP 通信：alltoall 仍被封装，无法直接对比
4. Decode 通信随并发变化剧烈：reduceScatter 14KB 从 c1=789us 到 c8=17us（差 46x）
5. Prefill ISL=2048 和 ISL=4096 通信耗时一致：因 max-num-batched-tokens=2048 限制
