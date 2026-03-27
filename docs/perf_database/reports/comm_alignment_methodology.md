# 通信算子耗时建模方法论

**版本**：v7.0
**日期**：2026-03-27
**作者**：HDY
**适用场景**：HCCL 通信算子耗时预测建模，bench 对齐 Comm_NO
**数据来源**：profiler-qwen3-0314 + profiler-dsv3-0316 + hccl_bench_v8.5（event/profiler 两模式，profiler 下含 alternating/kernel only 方案）

> **v7.0 更新**：合并三层 Duration 模型与恒等关系为统一的 Profiler 时间体系章节；
> 新增 §2.5 bench kernel 模式 vs 生产环境 profiler 的差异分析（pipeline 重叠机制）。

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

## 2. 通信耗时模型

### 2.1 三层 Duration 模型

通信算子的耗时在 profiler 中存在三个测量层级，分别对应不同的 CSV 文件：

| 层级 | CSV 文件 | 关键字段 | 物理含义 |
|------|---------|---------|---------|
| kernel_details | `kernel_details.csv` | hcom_* 的 Duration（去 AivKernel） | 纯 HCCL 数据传输时间 |
| operator_details | `operator_details.csv` | c10d::* 的 Device Total Duration | AicpuKernel + hcom_kernel，含调度开销 |
| step_trace | `step_trace.csv` | Communication / Comm_NO | 全 step 级通信时间 / 未被 overlap 的通信时间 |

HCCL bench 采集的 hcom_kernel Duration 对应 kernel_details 层级。仿真目标：用 bench kernel Duration 近似生产环境 kernel_details Duration。

### 2.2 operator_details 的物理分解

```
operator_details (Device Total Duration) = AicpuKernel + hcom_kernel
```

| 模型 | AicpuKernel | 关系 |
|------|------------|------|
| DSV3 (CANN 8.5) | = 0 | operator = kernel |
| Qwen3 (CANN 8.5) | > 0 (reduceScatter avg 319us, allGather avg 618us) | operator > kernel (1.4x~6.7x) |

AicpuKernel 是否存在取决于模型/CANN 版本/通信算子实现，不能假设为 0。

### 2.3 恒等关系

**恒等关系 1：Communication = Σ kernel_details hcom_*（去 AivKernel）— 精确成立**

全部 18 个场景（Qwen3 × 8 + DSV3 × 10）hcom/Comm = 1.0000，无一例外。AivKernel 条目的 Type 列也标记为 hcom_*，朴素求和会双重计数，去重后精确等于 Communication。

**恒等关系 2：Comm_NO = Σ operator_details — 仅部分成立**

operator_details 存在三层嵌套：`c10d::_allgather_base_` → `HcclAllGatherBase` → `HcclAllGather`，每层记录几乎相同的 Device Total Duration。取最底层 `Hccl*(不含Base)` 去重后：

| 场景 | Hccl*(去重) | COMM_NO | ratio | 原因 |
|------|------------|---------|------:|------|
| DSV3 Decode conc=8 | 652.6ms | 652.1ms | 1.001x | Overlap≈0，精确成立 |
| DSV3 Decode conc=1 | 1,799ms | 1,798ms | 2.001x | 嵌套 double counting |
| Qwen3 Prefill input4096 | 3.40s | 1.47s | 2.319x | operator Duration 含被 compute overlap 遮盖的部分 |
| Qwen3 Prefill input1024 | 17.81s | 2.69s | 6.623x | 同上，overlap 比例更高 |
| Qwen3 Decode conc=1 | 3.5ms | 667.6ms | 0.005x | allReduce 走 CUDAGraph 不在 operator_details 中 |

**根因**：operator_details Device Total Duration 是每次调用的完整 wall-clock Duration（含被 overlap 遮盖的部分），而 COMM_NO 是 step_trace 级别的未被 overlap 通信时间。两者语义不同，仅在 Overlap≈0 时相等。

### 2.4 关系链

```
Communication (step_trace) = Σ kernel_details hcom_* (去 AivKernel)  [精确，全部 18 场景]
COMM_NO = Communication - Overlapped                                  [精确，step_trace 定义]
operator_details ≠ COMM_NO                                            [仅 Overlap≈0 时相等]
```

---

## 3. HCCL Bench 方案

### 3.1 Bench vs 生产 Profiler

bench 的 kernel 模式和生产环境 profiler 采集使用完全相同的机制（CANN profiler → kernel_details.csv → hcom_* Duration），每次通信调用只产生 1 个 hcom 子 kernel（已验证）。但大消息下 bench 环境中同一 hcom kernel 比生产环境慢 ~90us。根因在于两种环境的算子执行模式截然不同：

**bench kernel 模式**：

```
profiler session 开启
  → allReduce 16MB (1次)
  → allReduce 16MB (1次)
  → ... (共 BENCH_ITERS 次，同一个 msg_bytes)
profiler session 关闭 → flush
```

一个 session 内只有同一个大消息算子反复执行。每次 allReduce 16MB 产生大量 trace 数据（DMA 传输记录、HCCL 内部分片记录等），profiler 的 ring buffer 快速填满，触发同步 flush，flush 过程与下一次 allReduce 的 DMA 竞争 HBM 带宽。

**生产环境 profiler**：

```
profiler session 开启
  → RmsNorm (5us)
  → MatMulV2 (180us)
  → FusedInferAttentionScore (320us)
  → MatMulV2 (140us)
  → hcom_allReduce 7MB (388us)    ← 通信
  → MatMulV2 (700us)
  → SwiGlu (40us)
  → MatMulV2 (400us)
  → hcom_allReduce 7MB (388us)    ← 通信
  → ... (数百个算子，64层)
profiler session 关闭 → flush
```

一个 session 内有数百个不同类型的算子。通信算子之间夹着大量计算算子，这些计算算子执行期间：
1. profiler 有时间将 ring buffer 中的 trace 数据异步 flush，不会积压
2. 计算算子占用 AI Core，不竞争 HCCL 使用的 DMA/SDMA 通道
3. 到下一个通信算子执行时，buffer 已清理，不触发同步 flush

这也解释了为什么 alternating bench（用 event timing 而非 profiler）反而更接近生产值 — event timing 测的是端到端 wall-clock，不受 profiler 对 kernel 边界的切分方式影响。

### 3.2 采集方案

针对上述差异，profiler 模式下根据算子类型和 message bytes 范围采用不同方案：

#### 3.2.1 alternating 方案（有 peer 算子场景）

适用于 allGather / reduceScatter 等存在互补 peer 算子的场景。目标算子与 peer 算子交替流水执行（如 allGather + reduceScatter），消除 1-5MB 消息的预热偏高问题：

- 单独执行 pipeline 模式在 1-5MB 范围高估 19-63%
- alternating 方案在相同范围误差 ±3%
- 大消息（≥3.5MB）两种方案趋同

#### 3.2.2 kernel only 方案（无 peer 算子 / 小消息场景）

适用于以下场景：
- allReduce 等无互补 peer 算子的通信算子
- <1MB 的小消息

直接采集 hcom_kernel Duration，获取纯 HCCL 传输时间。

### 3.3 固定开销

decode 场景小消息的 profiling P50 = bench kernel + 固定开销（调度/同步/AicpuKernel）：

| 模型 | 算子 | nd | 固定开销 |
|------|------|---:|--------:|
| Qwen3 | allReduce | 16 | +7.7us |
| Qwen3 | allGather | 16 | +14.6us |
| DSV3 | allGather | 8 | +1.2us |
| DSV3 | reduceScatter | 8 | +2.0us |

### 3.4 协议切换异常

DSV3 prefill allGather 768KB（nd=8, per_device ≈ 60KB）处于 HCCL 协议切换点，bench 无法复现此行为，需直接使用 profiler P50 回填。

---

## 4. 仿真策略

最终方案：全部使用 kernel 方案采集 hcom_kernel Duration 作为仿真基准数据，针对异常点使用 profiler 实测数据回填。

| 模型 | 阶段 | 算子 | msg_bytes 范围 | 策略 | 说明 |
|------|------|------|--------------|------|------|
| Qwen3 | decode | allReduce | 所有 | kernel + 固定开销 | kernel only 采集 + 7.7us |
| Qwen3 | decode | allGather | 所有 | kernel + 固定开销 | kernel only 采集 + 14.6us |
| Qwen3 | prefill | allGather | ≥1.26MB | kernel | alternating 采集（误差 ±6%）|
| Qwen3 | prefill | reduceScatter | ≥1.26MB | kernel | alternating 采集（误差 ±3%）|
| DSV3 | decode | allGather | ≤126KB (c=8) | kernel + 固定开销 | kernel only 采集 + 1.2us |
| DSV3 | decode | reduceScatter | 14KB (c=8) | kernel + 固定开销 | kernel only 采集 + 2.0us |
| DSV3 | prefill | allGather | 768KB | profiler P50 回填 | HCCL 协议切换异常点，bench 无法复现 |
| DSV3 | prefill | allGather | ≥3.5MB | kernel | alternating 采集（误差 ±3%）|
| DSV3 | prefill | reduceScatter | ≥3.5MB | kernel | alternating 采集（误差 ±2%）|

---

## 5. 环境要求

```bash
export HCCL_OP_EXPANSION_MODE="AIV"   # 建议
export TASK_QUEUE_ENABLE=1             # 建议
```

---

## 6. Checklist

- [ ] 恒等关系 1：验证 Communication = Σ kd hcom（去 AivKernel），应精确 1.0000
- [ ] 恒等关系 2：检查 Overlap 是否≈0，仅此时 operator_details ≈ COMM_NO
- [ ] bench 采集方案：allGather/reduceScatter 用 profiler-alternating，allReduce 用 profiler-kernel only
- [ ] 小消息（<1MB）：使用 kernel only 方案 + 固定开销修正
- [ ] 大消息（≥3.5MB）：直接用 alternating 方案 kernel 值
- [ ] DSV3 768KB allGather：使用 profiler P50 回填（HCCL 协议切换异常）
- [ ] 确认 operator_details 嵌套去重（取 Hccl* 不含 Base）
- [ ] Qwen3 Decode：allReduce 可能不在 operator_details 中（CUDAGraph）
- [ ] DSV3：注意 MoE 通信方差大（P10 vs P90 差 10-100x）
- [ ] 详细数据见 [bench_vs_profiler_comm_20260318.md](bench_vs_profiler_comm_20260318.md)

