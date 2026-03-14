# 通信算子对齐报告

**创建日期**：2026-03-12
**最后更新**：2026-03-14（基于 0313 生产环境 profiling 数据 + 0314 AIV 微基准验证）
**负责人**：HDY
**数据来源**：0313 生产环境 profiling（vLLM 0.15.0 + CANN 8.5 + AIV 模式）
**Microbench 版本**：hccl/v8.5（C10-1 采集，单 session，含 tier=1/2）

> **重要**：本报告替代原 Phase 2 报告。原报告基于 enforce-eager 非生产环境采集，
> 通信耗时与生产环境差异巨大（allReduce 差 77-181x），结论已失效。

---

## 1. 采集配置

### 1.1 vLLM 生产环境

| 模型 | TP | DP | 量化 | 关键配置 |
|------|----|----|------|---------|
| Qwen3-32B | 16 | 1 | BF16 | async-scheduling, CUDAGraph FULL_DECODE_ONLY, max-num-batched-tokens=65536 |
| DSV3 | 8 | 2 | W8A8 | async-scheduling, CUDAGraph FULL_DECODE_ONLY, max-num-seqs=8, max_num_batched_tokens=2048, EP |

**关键环境变量**（两个模型均开启）：

```bash
HCCL_OP_EXPANSION_MODE="AIV"      # AIV 加速通信
TASK_QUEUE_ENABLE=1                # 任务队列优化
VLLM_ASCEND_ENABLE_FUSED_MC2=1    # MC2 融合（DSV3 TP 通信）
VLLM_ASCEND_ENABLE_FLASHCOMM1=1   # FlashComm 优化
```

### 1.2 Bench 负载

| 场景 | 模型 | input_len | output_len | num_prompts | 说明 |
|------|------|-----------|------------|-------------|------|
| Prefill | Qwen3-32B | 4096 | 1 | 400 | 纯 prefill |
| Decode | Qwen3-32B | 4096 | 1536 | 400 | 长输出 decode |
| Prefill | DSV3 | 4096 | 1 | 400 | 纯 prefill |
| Decode | DSV3 | 4096 | 1536 | 400 | 长输出 decode |

### 1.3 与原 Phase 2 采集的关键差异

| 配置项 | 原 Phase 2 | 本次生产环境 | 影响 |
|--------|-----------|-------------|------|
| eager 模式 | enforce-eager | CUDAGraph FULL_DECODE_ONLY | decode 通信耗时大幅降低 |
| AIV 模式 | 未确认 | `HCCL_OP_EXPANSION_MODE="AIV"` | 通信 kernel 加速 |
| TASK_QUEUE | 未确认 | `TASK_QUEUE_ENABLE=1` | kernel launch 优化 |
| async-scheduling | 无 | 开启 | 调度与执行异步 |
| DSV3 max-num-seqs | 64 | 8 | batch 规模不同 |

---

## 2. Step Trace 概览

`Factor = Stage / (Computing + Communication_Not_Overlapped)`，来自 step_trace_time.csv。

| 模型 | 场景 | Stage(ms) | Computing(ms) | Comm_NO(ms) | Free(ms) | Factor | Free% |
|------|------|-----------|--------------|-------------|----------|--------|-------|
| Qwen3 | Prefill | 5816 | 3242 | 2491 | 83 | 1.014x | 1.4% |
| Qwen3 | Decode | 3098 | 1532 | 1399 | 167 | 1.057x | 5.4% |
| DSV3 | Prefill | 4555 | 2829 | 1361 | 365 | 1.087x | 8.0% |
| DSV3 | Decode | 4880 | 1306 | 3425 | 149 | 1.031x | 3.1% |

**结论**：
- 生产环境 overhead factor 整体 1.01x-1.09x，比原 Phase 2 数据（1.01x-1.45x）更低更稳定
- Qwen3 Prefill Free 仅 1.4%，CUDAGraph + async-scheduling 下调度开销极小
- DSV3 Prefill Free=8.0%，高于其他场景，可能与 MoE EP 协调有关
- DSV3 Decode 通信占比极高（Comm_NO=3425ms，占 Stage 70%），通信是主要瓶颈

---

## 3. 通信算子 Profiling 耗时

取 p10-p90 stable median，kernel_details.csv 中 `hcom_*` 算子。

### 3.1 Qwen3-32B（TP=16, Dense, BF16, num_devices=16）

**Prefill**（allGather + reduceScatter，SP 模式）：

| 算子 | Count | Stable Median(us) | P10(us) | P50(us) | P90(us) |
|------|-------|-------------------|---------|---------|---------|
| allGather | 650 | 1975 | 1256 | 1975 | 2001 |
| reduceScatter | 645 | 2475 | 1596 | 2475 | 2538 |

按 message size 拆分（从 operator_details Input Shapes 推算）：

| 算子 | msg_bytes | 大小 | Count | Kernel Median(us) |
|------|-----------|------|-------|-------------------|
| allGather | 1,823,232 | 1.7MB | 2 | 37 |
| allGather | 3,038,720 | 2.9MB | 3 | 37 |
| allGather | 252,149,760 | 240MB | 258 | 1,264 |
| allGather | 420,249,600 | 401MB | 387 | 1,989 |
| reduceScatter | 15,759,360 | 15MB | 258 | 1,606 |
| reduceScatter | 26,265,600 | 25MB | 387 | 2,498 |

**Decode**（allReduce 为主）：

| 算子 | Count | Stable Median(us) | P10(us) | P50(us) | P90(us) |
|------|-------|-------------------|---------|---------|---------|
| allReduce | 17286 | 20.5 | 16.1 | 20.5 | 30.2 |
| allGather | 134 | 49.2 | 45.4 | 49.2 | 81.6 |

allReduce 97% 的调用 < 50us，msg 估算 40KB-640KB（nq=4~64, hidden=5120, BF16）。

### 3.2 DSV3（TP=8, DP=2, EP, W8A8, num_devices=8）

**Prefill**（allGather + reduceScatter）：

| 算子 | Count | Stable Median(us) | P10(us) | P50(us) | P90(us) |
|------|-------|-------------------|---------|---------|---------|
| allGather | 520 | 152 | 29 | 153 | 2785 |
| reduceScatter | 260 | 296 | 202 | 298 | 8314 |

DSV3 prefill 通信耗时方差极大（P10 vs P90 差 100x），与 MoE EP 的不均匀通信模式有关。

按 message size 拆分：

| 算子 | msg_bytes | 大小 | Count |
|------|-----------|------|-------|
| allGather | 258,560 | 0.2MB | 4 |
| allGather | 2,359,296 | 2.2MB | 244 |
| allGather | 6,291,456 | 6.0MB | 244 |
| allGather | 29,360,128 | 28MB | 16 |
| reduceScatter | 3,670,016 | 3.5MB | 260 |

**Decode**（allGather + reduceScatter，无独立 allReduce，MC2 融合）：

| 算子 | Count | Stable Median(us) | P10(us) | P50(us) | P90(us) |
|------|-------|-------------------|---------|---------|---------|
| allGather | 650 | 1369 | 987 | 1405 | 4328 |
| reduceScatter | 325 | 5591 | 5471 | 5592 | 5721 |

reduceScatter 极稳定（P10-P90 仅 250us 范围），msg=14KB。allGather 有双峰分布。

---

## 4. Profiling vs HCCL Microbench 对比

> **v3.1 更新（0314）**：AIV 模式微基准已采集完成（含 num_devices=8）。
> 结果表明 AIV 对独立微基准无显著影响（加速比 0.9-1.3x），bench 数据本身是准确的。
> Profiling vs bench 的 gap 根因是 **profiling Duration 语义差异**，而非 bench 环境问题。

### 4.1 AIV 模式验证

AIV 模式微基准 vs 原非 AIV 微基准（num_devices=16, tier=1）：

| 算子 | 小消息量(≤1MB) | 大消息量(≥16MB) | 结论 |
|------|---------------|----------------|------|
| allReduce | 变化 <12% | 变化 <1% | 无显著差异 |
| allGather | 变化 <23% | 变化 <1% | 小消息量有波动，大消息量一致 |
| reduceScatter | 变化 <10% | 变化 <1% | 无显著差异 |
| alltoallv | 变化 <27% | 变化 <18% | alltoallv 小消息量 AIV 略快 |

**结论：AIV 模式对独立 HCCL 微基准无显著加速效果。** 小消息量的波动属于正常采集抖动。

### 4.2 Qwen3-32B（num_devices=16）

使用 operator_details 的 Device Total Duration 作为 profiling 耗时（比 kernel_details 更准确）。

**Prefill**：

| 算子 | msg_bytes | Profiling(us) | Bench(us) | Ratio | 判定 |
|------|-----------|--------------|-----------|-------|------|
| allGather | 240MB | 5,477 | ~25,115 | 0.22x | FAIL |
| allGather | 401MB | 8,942 | ~42,650 | 0.21x | FAIL |
| allGather | 1.7MB | 34 | ~380 | 0.09x | FAIL |
| reduceScatter | 15MB | 3,150 | ~2,001 | 1.57x | WARN |
| reduceScatter | 25MB | 4,926 | ~3,861 | 1.28x | PASS |

> Bench 值通过对数线性插值计算（方法论 §3.2），非精确匹配，以 `~` 标注。

**Decode**：

| 算子 | msg 估算 | Profiling P50(us) | Bench(us) | Ratio | 判定 |
|------|---------|------------------|-----------|-------|------|
| allReduce | 40-640KB | 20.5 | 112-128 | 0.16-0.17x | FAIL |
| allGather | 4.6MB | 49 | ~637 | 0.08x | FAIL |

### 4.3 DSV3（num_devices=8，新增参考数据）

| 算子 | 场景 | msg_bytes | Profiling(us) | Bench(us) | Ratio | 判定 |
|------|------|-----------|--------------|-----------|-------|------|
| allGather | Prefill | 2.2MB | 31 | ~235 | 0.13x | FAIL |
| allGather | Prefill | 6.0MB | 1,870 | ~417 | 4.49x | FAIL |
| allGather | Decode | 9KB | 1,010 | ~458 | 2.21x | WARN |
| allGather | Decode | 24KB | 4,244 | ~459 | 9.24x | FAIL |
| reduceScatter | Prefill | 3.5MB | 989 | ~262 | 3.77x | WARN |
| reduceScatter | Decode | 14KB | 5,592 | ~284 | 19.66x | FAIL |

### 4.4 Gap 根因分析

**AIV 假设已排除**：AIV 模式对微基准无显著影响，bench 数据本身是准确的。

Gap 呈现两种截然相反的模式：

| 模式 | 场景 | Ratio | 含义 |
|------|------|-------|------|
| Profiling 远快于 bench | Qwen3 allGather/allReduce | 0.05-0.22x | profiling Duration 短于实际通信时间 |
| Profiling 远慢于 bench | DSV3 decode reduceScatter | 9-20x | profiling Duration 包含大量隐式等待 |

**根因：profiling kernel_details Duration 的语义在不同执行模式下不同**：

1. **CUDAGraph + async-scheduling 下**（Qwen3 decode）：通信 kernel 可能被 pipeline 化或与 compute overlap，kernel_details 记录的 Duration 不代表完整集合通信的 wall-clock 时间，而是 NPU 上该 kernel 的实际执行片段
2. **MoE EP 场景**（DSV3 decode）：reduceScatter Duration 包含 expert parallel 协调等待（跨 DP group 的 token dispatch/combine 同步），14KB 消息量不应需要 5.6ms 纯通信（bench CSV 实测 16KB/n=8 仅 284us，差 19.7x）
3. **SP prefill reduceScatter**（Qwen3）：ratio 1.28-1.57x 最接近合理范围，因为 SP 模式下 reduceScatter 在 compute 之后串行执行，Duration 语义最接近纯通信时间

**结论：bench CSV 数据质量可靠，但不能直接与 profiling Duration 对比。** 两者测量的不是同一个物理量。

---

## 5. 通信查询架构决策（更新）

### 5.1 原方案失效

原报告的三个策略评估基于非生产 profiling 数据，结论全部失效：

| 原结论 | 状态 | 原因 |
|--------|------|------|
| "Bench CSV raw 严重低估（69-91% 误差）" | **失效** | 基于非生产 profiling 对比 |
| "Bench + 校准不可行（ratio 不稳定）" | **部分成立** | ratio 确实不稳定，但原因是 Duration 语义差异 |
| "Inline profiled 是短期最优（20-34% 误差）" | **失效** | inline 值基于非生产数据 |
| communication_profiled 内联值 | **全部失效** | 值偏高 5-96x（decode）或偏低（prefill） |

### 5.2 当前状态

**bench CSV 数据质量已验证**：AIV 模式微基准与原数据一致（加速比 0.9-1.3x），bench 数据可靠。新数据已补充 num_devices=8。

**但 bench CSV 不能直接用于预测 profiling Duration**：两者测量的物理量不同（纯通信时间 vs 含 pipeline/等待的 kernel Duration）。

**运行时影响说明**：当前 `ProfilingDataSource._lookup_comm()` 优先查询 bench CSV（按 message_bytes + num_devices + topology_tier 匹配），`communication_profiled` 内联值仅作为 bench CSV 缺失时的 fallback。因此内联值失效不影响已有 bench CSV 覆盖的查询路径。op_mapping.yaml 中的内联值暂保留但标记为 deprecated，待 per-config profiling CSV 方案落地后移除。

**communication_profiled 内联值 vs 生产实测**：

| 算子 | 内联值(us) | 生产 Qwen3 Decode(us) | 生产 Qwen3 Prefill(us) | 偏差 |
|------|-----------|----------------------|----------------------|------|
| allGather | 1062 | 49 | 5477-8942 | Decode 高 22x，Prefill 低 5-8x |
| reduceScatter | 1184 | — | 3150-4926 | Prefill 低 2.7-4.2x |
| allReduce | 1932 | 20.5 | — | Decode 高 94x |

### 5.3 策略建议（v3.1 更新）

**核心问题**：bench CSV 测量纯通信时间，profiling Duration 包含 pipeline overlap / 隐式等待，两者语义不同。需要选择正确的对标物。

**短期**：

1. ~~重新采集 AIV 模式微基准~~ → **已完成**（0314），含 num_devices=2/4/8/16
2. bench CSV 作为纯通信时间的参考基线，用于 analytic model 校准
3. communication_profiled 内联值暂时保留但标记为 deprecated

**中期**：

4. **per-config profiling CSV**（从 kernel_details 提取 per-(model, phase, batch, kernel_type) 延迟）仍然是最高性价比路径——直接用 profiling Duration 预测 profiling Duration，避免语义转换
5. 需要更多 profiling 数据点（不同 batch size、不同 ISL）来建立插值表

**长期**：

6. 建模 profiling Duration = f(bench_pure_comm, compute_overlap, scheduling_wait)，显式拆分三个分量

---

## 6. DSV3 特有注意事项（更新）

1. **通信算子路径**：本次生产 profiling 中 DSV3 通信算子为 `hcom_allGather_` / `hcom_reduceScatter_`，未出现原报告提到的 `allgatherAicpuKernel` / `reduce_scatterAicpuKernel` 路径。可能是 vLLM 0.15.0 + CANN 8.5 的行为变化
2. **MC2 融合**：DSV3 decode 无独立 allReduce，TP 通信走 MC2 融合算子，仍然成立
3. **DispatchFFNCombine 封装 EP 通信**：alltoall 仍被封装，无法直接对比
4. **DSV3 Decode 通信占比极高**：Comm_NO=3425ms 占 Stage 70%，通信是绝对瓶颈

---

## 7. 遗留问题

| # | 问题 | 优先级 | 行动 | 状态 |
|---|------|--------|------|------|
| ~~P14~~ | ~~HCCL 微基准需在 AIV 模式下重新采集~~ | ~~最高~~ | ~~开启 AIV + TASK_QUEUE 重采~~ | **已完成**（0314，AIV 对独立微基准无显著影响） |
| ~~P15~~ | ~~补采 num_devices=8 微基准~~ | ~~高~~ | ~~DSV3 TP=8 场景无参考数据~~ | **已完成**（0314，4 个 CSV 均已补充 n=8） |
| P16 | communication_profiled 内联值全部失效 | **高** | 等 per-config profiling CSV 替代；当前运行时已走 bench CSV 路径，内联值不影响查询结果（见 §5.2） | Open |
| P5 | DSV3 allToAll 封装在 DispatchFFNCombine | 中 | C11：拆分子 kernel | Open |
| P9 | DSV3 Prefill Free=8.0% 偏高 | 中 | 分析 MoE EP 协调开销 | Open |
| P17 | DSV3 Decode 通信占 70%，需验证 reduceScatter 5591us 是否合理 | 中 | msg=14KB 但耗时 5.6ms，bench CSV 实测 16KB/n=8 仅 284us（差 19.7x），确认含隐式 EP 协调等待 | Open |

