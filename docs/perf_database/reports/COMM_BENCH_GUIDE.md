# 通信算子 Bench 工具链指南

**日期**：2026-03-19
**作者**：HDY

---

## 1. 概述

通信算子 bench 工具链由三个脚本组成，用于采集 HCCL 通信算子的性能数据并生成最终 CSV：

| 脚本 | 职责 |
|------|------|
| `generate_comm_microbench.py` | 5 模式采集引擎，生成并执行 HCCL 通信微基准 |
| `run_comm_bench.sh` | 统一采集脚本，编排两轮采集（alternating + kernel） |
| `build_comm_csv.py` | 后处理合并脚本，固定开销修正 + profiler P50 替换 + bandwidth 重算 |

工作流：`run_comm_bench.sh` → 调用 `generate_comm_microbench.py` 两轮采集 → `build_comm_csv.py` 合并输出最终 CSV。

---

## 2. generate_comm_microbench.py 五种 Bench 模式

### 2.1 profiler 模式

- **测量方式**：`torch_npu.profiler` → `operator_details.csv` 中 `c10d::*` 算子的 `Device Total Duration`
- **包含内容**：AicpuKernel + hcom_kernel（对齐 Comm_NO 语义）
- **参数**：WARMUP=5, ACTIVE=10，取 10 次 Duration 的 median
- **特点**：CANN profiler 不可重启，所有 message size 批量放入一个 profiler session
- **Leader/Follower**：仅 group_ranks[0] 开启 profiler，其余 rank 执行相同调用但不采集

### 2.2 kernel 模式

- **测量方式**：`torch_npu.profiler` → `kernel_details.csv` 中 `hcom_*` 算子的 Duration（去 AivKernel）
- **包含内容**：纯 HCCL kernel 时间，不含 AicpuKernel 调度开销
- **适用场景**：小消息（<1MB），NPU Event 底噪 ~60us 会淹没真实值
- **同 profiler 模式的 batching 和 Leader/Follower 策略**

### 2.3 alternating 模式（推荐）

- **测量方式**：peer→target 无 sync 流水执行，NPU Event 仅测 target op
- **原理**：模拟生产 MC2 交替执行模式（allGather↔reduceScatter 在同一 HCCL stream 上背靠背执行）
- **peer 映射**：allGather↔reduceScatter（双向）；allReduce/alltoall 无 peer → fallback 到 event 模式
- **参数**：100 次迭代，取 median
- **核心优势**：消除 1-5MB 区间的预热偏高（从 19-63% 降到 ±3%）

### 2.4 event 模式

- **测量方式**：逐次 NPU Event 计时，100 次取 median
- **特点**：快速但有 ~60us 底噪，不适合小消息（<1MB）
- **用途**：快速 sanity check；alternating 模式中 allReduce 的 fallback

### 2.5 pipeline 模式

- **测量方式**：100 次无逐次 sync，取平均
- **特点**：测量 pipeline 稳态吞吐，不对齐 Comm_NO
- **用途**：向后兼容，硬件通信能力上界参考

---

## 3. 模式选择指南

### 3.1 按算子推荐

| 算子 | 推荐模式 | 原因 |
|------|---------|------|
| allGather | alternating | peer=reduceScatter 流水执行，消除预热偏高 |
| reduceScatter | alternating | peer=allGather 流水执行，消除预热偏高 |
| allReduce | alternating（自动 profiler fallback） | 无 peer，NPU Event 底噪 ~270us 远大于 kernel 时间 ~13us |
| all_to_all | kernel 或 event | 无 peer，按需选择 |

### 3.2 按消息大小推荐

| 消息大小 | 推荐模式 | 原因 |
|---------|---------|------|
| <1MB | kernel | NPU Event 底噪 ~60us 淹没真实值；profiler kernel_details 无此问题 |
| ≥1MB | alternating | peer 流水消除预热偏高，误差 ±3% |
| 768KB nd=8 | profiler P50 | HCCL 协议切换点异常，bench 无法复现生产值 |

### 3.3 关键结论

1. **alternating 模式是 allGather/reduceScatter 的首选**：通过 peer→target 流水执行模拟生产 MC2 交替模式，消除 1-5MB 预热偏高
2. **小消息必须用 kernel 模式**：NPU Event 底噪 ~60us 对 <30us 的 kernel 时间影响巨大
3. **allReduce 用 alternating 的 profiler fallback**：无 peer op，自动回退到 profiler batch 采集 kernel_details
4. **固定开销需后处理修正**：kernel 模式数据需加固定开销才能对齐生产 Comm_NO
5. **HCCL 协议切换点需特殊处理**：768KB nd=8 allGather 用 profiler P50 替代 bench 值

---

## 4. run_comm_bench.sh 使用说明

### 4.1 两轮采集策略

**Round 1: alternating 模式**
- allReduce：所有 message size（无 peer → event fallback，可接受噪声底）
- allGather/reduceScatter：≥1MB（peer 流水消除预热偏高）
- 覆盖 nd=16, 8, 4, 2

**Round 2: kernel 模式**
- allGather/reduceScatter：<1MB（event 底噪淹没真实值）
- 覆盖 nd=16, 8, 4, 2

### 4.2 运行命令

```bash
bash tools/perf_data_collection/run_comm_bench.sh ./output_dir
```

### 4.3 输出结构

```
output_dir/
  alternating/
    hcom_allReduce_.csv      # 所有 msg size
    hcom_allGather_.csv      # ≥1MB
    hcom_reduceScatter_.csv  # ≥1MB
  kernel/
    hcom_allGather_.csv      # <1MB
    hcom_reduceScatter_.csv  # <1MB
```

### 4.4 Message Grid

- 标准 grid：1KB~512MB，powers of 2（20 个点）
- 生产 msg_bytes：Qwen3（nd=16, TP=16）和 DSV3（nd=8, TP=8）的实际通信消息大小

---

## 5. build_comm_csv.py 后处理

### 5.1 数据源选择规则

| 算子 | 消息大小 | 数据来源 | 原因 |
|------|---------|---------|------|
| allReduce | 所有 | alternating | 无 peer，event fallback 可接受 |
| allGather | <1MB | kernel + 固定开销 | event 底噪淹没真实值 |
| allGather | ≥1MB | alternating | peer 流水消除预热偏高 |
| allGather | 768KB nd=8 | profiler P50 | HCCL 协议切换点异常 |
| reduceScatter | <1MB | kernel + 固定开销 | event 底噪淹没真实值 |
| reduceScatter | ≥1MB | alternating | peer 流水消除预热偏高 |
| alltoallv | — | 空（仅 header） | 无 bench 数据 |

### 5.2 固定开销修正

kernel 模式数据需加固定开销以对齐生产 Comm_NO：

```python
OVERHEAD = {
    "allReduce": {16: 7.7},        # Qwen3 TP=16
    "allGather": {16: 14.6, 8: 1.2},  # Qwen3 TP=16, DSV3 TP=8
    "reduceScatter": {16: 14.6, 8: 2.0},  # Qwen3 TP=16, DSV3 TP=8
}
```

来源：bench vs 生产 profiler 的固定差值（与 msg_bytes 无关，仅与调度链路深度相关）。

### 5.3 处理流程

1. 读取 alternating CSV（allReduce 全量，allGather/reduceScatter ≥1MB）
2. 读取 kernel CSV（allGather/reduceScatter <1MB）
3. 解析 profiler trace 获取 DSV3 allGather 768KB P50（特殊处理）
4. 按 (message_bytes, num_devices) 合并，source priority 去重
5. 对 kernel 数据应用固定开销修正
6. 重算 bandwidth_gbps = message_bytes / (duration_us × 1e-6) / 1e9
7. 输出最终 CSV

### 5.4 输出格式

```csv
message_bytes,num_devices,dtype,topology_tier,Duration(us),bandwidth_gbps
```

最终输出：
```
output_dir/
  hcom_allReduce_.csv       # 136 行数据
  hcom_allGather_.csv       # 136 行数据
  hcom_reduceScatter_.csv   # 136 行数据
  hcom_alltoallv_.csv       # 仅 header
```

---

## 6. 参考

- [通信算子对齐报告](./comm_alignment_report_20260312.md) — 详细数据验证
- [通信算子建模方法论](./comm_alignment_methodology.md) — 理论框架
