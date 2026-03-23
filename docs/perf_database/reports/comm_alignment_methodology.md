# 通信算子耗时建模方法论

**版本**：v6.0
**日期**：2026-03-19
**作者**：HDY
**适用场景**：HCCL 通信算子耗时预测建模，bench 对齐 Comm_NO
**数据来源**：profiler-qwen3-0314 + profiler-dsv3-0316 + hccl_bench_v8.5（alternating/kernel/event/pipeline/profiler 五模式）

> **v6.0 更新**：基于 0318 bench 五模式全面验证，确立 alternating 模式为默认推荐。
> alternating 模式消除 1-5MB 预热偏高（从 19-63% 高估降至 ±3%）；
> kernel 模式用于 <1MB 小消息（event 底噪 ~60us 淹没真实值）；
> 新增固定开销修正用于 decode 小消息场景。

## 1. E2E 时间模型

### 1.1 TensorCast 串行求和

TensorCast 通过 `Runtime` 拦截所有算子，逐个查询 `PerformanceModel` 获取耗时，纯串行求和：

```
T_total = Σ T_compute + Σ T_comm
```

其中 `T_comm` 由 `DataSource.lookup()` 查 bench CSV 获得。

### 1.2 实际 Stage 时间

step_trace 记录的实际 Stage 时间：

```
Stage = Computing + Comm_NO + Free
      = T_total × overhead_factor

overhead_factor = 1 + Free / (Computing + Comm_NO)
```

| 场景 | overhead_factor | Free 占比 | 特征 |
|------|----------------|----------|------|
| Qwen3 Prefill ISL=1024 | 1.315x | 24.0% | Dense, 短 ISL, 调度开销大 |
| Qwen3 Prefill ISL=4096 | 1.198x | 16.6% | Dense, 长 ISL |
| Qwen3 Decode c1 | 1.073x | 6.8% | 图模式, 调度开销小 |
| DSV3 Prefill ISL=1024 | 1.029x | 2.8% | MoE, 计算密集 |
| DSV3 Prefill ISL=4096 | 1.039x | 3.7% | MoE, 计算密集 |
| DSV3 Decode c1 | 1.048x | 4.6% | MoE, 图模式 |

规律：Decode < Prefill，MoE < Dense。overhead_factor 在 ModelRunner 层应用，不进入单算子建模。

---

## 2. 三层 Duration 模型

通信算子的耗时存在三个测量层级：

| 层级 | 测量方式 | 包含内容 | 数据源 |
|------|---------|---------|--------|
| kernel_details | NPU timeline | hcom_kernel（HCCL 数据传输） | `kernel_details.csv` |
| operator_details | operator timeline | AicpuKernel + hcom_kernel | `operator_details.csv` c10d::* |
| bench (alternating) | host perf_counter | peer 流水执行，消除预热偏高 | microbench CSV |

### 2.1 operator_details 的物理分解

```
operator_details (Device Total Duration) = AicpuKernel + hcom_kernel
```

| 模型 | AicpuKernel | 关系 |
|------|------------|------|
| DSV3 (CANN 8.5) | = 0 | operator = kernel |
| Qwen3 (CANN 8.5) | > 0 (reduceScatter avg 319us, allGather avg 618us) | operator > kernel (1.4x~6.7x) |

AicpuKernel 是否存在取决于模型/CANN 版本/通信算子实现，不能假设为 0。

---

## 3. 恒等关系验证

### 3.1 恒等关系 1：Communication = Σ kernel_details hcom_*（去 AivKernel）— 精确成立

全部 18 个场景（Qwen3 × 8 + DSV3 × 10）hcom/Comm = 1.0000，无一例外。

AivKernel 条目的 Type 列也标记为 hcom_*，朴素求和会双重计数。去重后精确等于 Communication。

### 3.2 恒等关系 2：Comm_NO = Σ operator_details — 仅部分成立

operator_details 存在三层嵌套：`c10d::_allgather_base_` → `HcclAllGatherBase` → `HcclAllGather`，每层记录几乎相同的 Device Total Duration。取最底层 `Hccl*(不含Base)` 去重后：

| 场景 | Hccl*(去重) | COMM_NO | ratio | 原因 |
|------|------------|---------|------:|------|
| DSV3 Decode conc=8 | 652.6ms | 652.1ms | 1.001x | Overlap≈0，精确成立 |
| DSV3 Decode conc=1 | 1,799ms | 1,798ms | 2.001x | 嵌套 double counting |
| Qwen3 Prefill input4096 | 3.40s | 1.47s | 2.319x | operator Duration 含被 compute overlap 遮盖的部分 |
| Qwen3 Prefill input1024 | 17.81s | 2.69s | 6.623x | 同上，overlap 比例更高 |
| Qwen3 Decode conc=1 | 3.5ms | 667.6ms | 0.005x | allReduce 走 CUDAGraph 不在 operator_details 中 |

**根因**：operator_details Device Total Duration 是每次调用的完整 wall-clock Duration（含被 overlap 遮盖的部分），而 COMM_NO 是 step_trace 级别的未被 overlap 通信时间。两者语义不同，仅在 Overlap≈0 时相等。

### 3.3 正确的关系链

```
Communication (step_trace) = Σ kernel_details hcom_* (去 AivKernel)  [精确，全部 18 场景]
COMM_NO = Communication - Overlapped                                  [精确，step_trace 定义]
operator_details ≠ COMM_NO                                            [仅 Overlap≈0 时相等]
```

---

## 4. Bench 采集模式分析（v6.0 新增）

### 4.1 五种 bench 模式概述

| 模式 | 计时方式 | 特点 |
|------|---------|------|
| event | NPU Event 包围单次调用 | 含同步开销，小消息底噪 ~60us |
| kernel | profiler 采集 hcom_kernel Duration | 纯 HCCL 传输时间，无同步开销 |
| pipeline | host perf_counter 100 次无逐次 sync | 稳态吞吐，预热偏高 |
| profiler | profiler 采集 operator_details Duration | 含 AicpuKernel |
| alternating | peer 算子交替流水执行 | 消除预热偏高，推荐默认模式 |

### 4.2 alternating 模式核心优势

alternating 模式让目标算子与 peer 算子交替执行（如 allGather + reduceScatter 流水），消除了 pipeline 模式中 1-5MB 消息的预热偏高问题：

- pipeline 模式在 1-5MB 范围高估 19-63%
- alternating 模式在相同范围误差 ±3%
- 大消息（≥3.5MB）两种模式趋同

### 4.3 kernel 模式用于小消息

对于 <1MB 的小消息，event 模式底噪 ~60us 远大于实际 kernel 时间（如 allReduce nd=16 kernel 仅 ~13us），因此小消息必须使用 kernel 模式获取纯 HCCL 传输时间。

### 4.4 固定开销修正

decode 场景小消息的 profiling P50 = bench kernel + 固定开销（调度/同步/AicpuKernel）：

| 模型 | 算子 | nd | 固定开销 |
|------|------|---:|--------:|
| Qwen3 | allReduce | 16 | +7.7us |
| Qwen3 | allGather | 16 | +14.6us |
| DSV3 | allGather | 8 | +1.2us |
| DSV3 | reduceScatter | 8 | +2.0us |

### 4.5 HCCL 协议切换异常

DSV3 prefill allGather 768KB（nd=8, per_device ≈ 60KB）处于 HCCL 协议切换点，bench 无法复现此行为，需直接使用 profiler P50。

---

## 5. 仿真策略总表（v6.0 新增）

| 模型 | 阶段 | 算子 | msg_bytes 范围 | 策略 | 数据来源 |
|------|------|------|--------------|------|---------|
| Qwen3 | decode | allReduce | 所有 | bench + 固定开销 | alternating (profiler fallback) + 7.7us |
| Qwen3 | decode | allGather | 所有 | bench + 固定开销 | kernel (profiler) + 14.6us |
| Qwen3 | prefill | allGather | ≥1.26MB | 直接用 bench | alternating（误差 ±6%）|
| Qwen3 | prefill | reduceScatter | ≥1.26MB | 直接用 bench | alternating（误差 ±3%）|
| DSV3 | decode | allGather | ≤126KB (c=8) | bench + 固定开销 | kernel (profiler) + 1.2us |
| DSV3 | decode | reduceScatter | 14KB (c=8) | bench + 固定开销 | kernel (profiler) + 2.0us |
| DSV3 | prefill | allGather | 768KB | 用 profiler P50 | HCCL 协议切换点，bench 无法复现 |
| DSV3 | prefill | allGather | ≥3.5MB | 直接用 bench | alternating（误差 ±3%）|
| DSV3 | prefill | reduceScatter | ≥3.5MB | 直接用 bench | alternating（误差 ±2%）|

---

## 6. Bench 模式选择指南（v6.0 新增）

| 算子 | 推荐模式 | 原因 |
|------|---------|------|
| allGather | alternating | peer=reduceScatter 流水执行，消除预热偏高 |
| reduceScatter | alternating | peer=allGather 流水执行，消除预热偏高 |
| allReduce | alternating (自动 profiler fallback) | 无 peer，NPU Event 底噪 ~270us 远大于 kernel 时间 ~13us |
| all_to_all | kernel 或 event | 无 peer，按需选择 |

关键结论：

1. alternating 模式消除 1-5MB 预热偏高（从 19-63% 高估降至 ±3%）
2. kernel 模式用于 <1MB 小消息（event 底噪 ~60us 淹没真实值）
3. 固定开销修正用于 decode 小消息（bench kernel + offset = profiling P50）
4. HCCL 协议切换点（768KB nd=8, per_device ≈ 60KB）需用 profiler P50
5. allReduce 无 peer 算子，alternating 自动 fallback 到 profiler 模式

---

## 7. 环境一致性要求

```bash
export HCCL_OP_EXPANSION_MODE="AIV"   # 建议
export TASK_QUEUE_ENABLE=1             # 建议
```

---

## 8. 对比 Checklist（v6.0 更新）

- [ ] 恒等关系 1：验证 Communication = Σ kd hcom（去 AivKernel），应精确 1.0000
- [ ] 恒等关系 2：检查 Overlap 是否≈0，仅此时 operator_details ≈ COMM_NO
- [ ] bench 模式选择：allGather/reduceScatter 用 alternating，allReduce 用 alternating (profiler fallback)
- [ ] 小消息（<1MB）：使用 kernel 模式 + 固定开销修正
- [ ] 大消息（≥3.5MB）：直接用 alternating bench 值
- [ ] DSV3 768KB allGather：使用 profiler P50（HCCL 协议切换异常）
- [ ] 确认 operator_details 嵌套去重（取 Hccl* 不含 Base）
- [ ] Qwen3 Decode：allReduce 可能不在 operator_details 中（CUDAGraph）
- [ ] DSV3：注意 MoE 通信方差大（P10 vs P90 差 10-100x）
- [ ] 详细数据见 [bench_vs_profiler_comm_20260318.md](bench_vs_profiler_comm_20260318.md)

