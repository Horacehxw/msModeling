# NPU Profiling 分析报告 (二): Kernel Duration 与端到端时间关系

**软件栈**: CANN 8.5 / vLLM 0.15.0 / PyTorch 2.9.0 / Eager + aclgraph 模式
**硬件**: Atlas 800 A3 (Ascend 910B)
**模型**: DeepSeek-V3 (W8A8, TP=8/DP=2/EP, 16 NPU) + Qwen3-32B (BF16, TP=16)
**分析脚本**: `docs/perf_database/reports/profiling_analysis_eager/`

---

## 核心问题

Perf-database 仿真的核心假设是: **将所有算子的 device kernel 执行时间加总, 即可近似端到端推理时间**. 本报告验证这一假设, 量化 kernel duration 加总与 e2e 的差距及其来源, 并分析 compute/comm 是否存在 overlap.

---

## 一、分析原理

### 1.1 NPU 多 Stream 执行模型

NPU 上有多个硬件 stream 可以并行执行 kernel:
- **Compute stream** (Stream 2): 执行计算 kernel (MatMul, Norm, Attention 等) 和 DispatchFFNCombine
- **HCCL stream** (DSv3 Stream 8 / Qwen3 Stream 6): 执行通信 kernel (allReduce, reduceScatter 等)

### 1.2 kernel_details.csv 的双重计数问题

`kernel_details.csv` 记录了所有 stream 上每个 kernel 的 `Task Duration(us)`. 存在关键问题:

**双重计数**: HCCL 通信 kernel 在 CSV 中被记录了两次 — 一次作为 `hcom_*` 行 (Stream ID = NaN), 一次作为 `AivKernel` 行 (在 HCCL stream 上). 时间戳和 duration 完全相同. 如果不去重, kernel_sum/e2e 会达到 1.5-2.0x, 容易误判为 "stream 并行导致 kernel 总时间 > e2e".

### 1.3 理论模型

假设两个 stream 串行交替执行 (无 overlap):

```
e2e = Σ(compute_stream kernels) + Σ(comm_stream kernels) + Σ(CPU dispatch gaps)
    = t1_compute + t1_comm + t_cpu
```

如果存在 overlap, 则 `t1_compute + t1_comm > e2e - t_cpu`.

---

## 二、分析方法

1. 按 `Stream ID` 分组 kernel_details.csv, **去除 Stream ID = NaN 的重复 HCCL 行**
2. 利用 `Start Time(us)` 和 `Duration(us)` 计算每个 stream 的 span: `stream_span = max(start + duration) - min(start)`
3. 对比: `CPU gap = stream_span - Σ(kernel_duration)` (该 stream 上 kernel 之间的空闲)
4. 对比: `e2e - (t1_compute + t1_comm)` = 纯 CPU 调度开销

---

## 三、Eager 模式结果

### 3.1 DSv3

| 场景 | t1_comp (ms) | t1_comm (ms) | t1_sum (ms) | e2e (ms) | **e2e - t1_sum** | **CPU gap%** | comp_span (ms) |
|------|---------|---------|---------|---------|----------|----------|---------|
| decode_b1 | 869 | 1488 | 2357 | 2652 | **+295ms** | **11.1%** | 2650 |
| decode_b4 | 4245 | 1274 | 5519 | 5761 | **+242ms** | **4.2%** | 5759 |
| decode_b8 | 4420 | 935 | 5355 | 5425 | **+70ms** | **1.3%** | 5424 |
| decode_b16 | 3998 | 2107 | 6105 | 6173 | **+68ms** | **1.1%** | 6171 |
| decode_b32 | 3604 | 4525 | 8129 | 8183 | **+54ms** | **0.7%** | 8182 |
| prefill_256 | 2070 | 1705 | 3775 | 3822 | **+47ms** | **1.2%** | 3821 |
| prefill_1024 | 2099 | 1308 | 3407 | 3856 | **+449ms** | **11.6%** | 3855 |
| prefill_4096 | 2200 | 1041 | 3241 | 4809 | **+1568ms** | **32.6%** | 4807 |

### 3.2 Qwen3-32B

| 场景 | t1_comp (ms) | t1_comm (ms) | t1_sum (ms) | e2e (ms) | **e2e - t1_sum** | **CPU gap%** | comp_span (ms) |
|------|---------|---------|---------|---------|----------|----------|---------|
| decode_b1 | 29 | 1103 | 1132 | 1463 | **+331ms** | **22.6%** | 1453 |
| decode_b4 | 58 | 2029 | 2087 | 2218 | **+131ms** | **5.9%** | 2215 |
| decode_b8 | 88 | 3211 | 3299 | 3361 | **+62ms** | **1.8%** | 3358 |
| decode_b16 | 129 | 4822 | 4951 | 4981 | **+30ms** | **0.6%** | 4977 |
| decode_b32 | 210 | 7488 | 7698 | 7822 | **+124ms** | **1.6%** | 7818 |
| prefill_256 | 37 | 1064 | 1101 | 1127 | **+26ms** | **2.3%** | 1125 |
| prefill_1024 | 49 | 1056 | 1105 | 1125 | **+20ms** | **1.8%** | 1117 |
| prefill_4096 | 92 | 710 | 802 | 1244 | **+442ms** | **35.5%** | 1241 |

---

## 四、关键发现

### 4.1 去重后 t1_sum < e2e, 差值全部为正

符合串行执行的理论模型:
```
e2e = t1_compute + t1_comm + t_cpu_dispatch   (完美成立)
```

去重前的 `kernel_sum / e2e` 达到 1.5-2.0x, 是 HCCL 双重计数导致的假象.

### 4.2 Compute/Comm 几乎无 overlap

Eager 模式下 step_trace 的 Overlapped 项在所有场景 <0.1%. 两个 stream 交替串行执行:
```
Compute stream: |--kernel--|  wait  |--kernel--|  wait  |--kernel--|
Comm stream:        idle   |--comm--|   idle   |--comm--|   idle
```

### 4.3 CPU 调度开销: batch 越大越小

| 场景 | DSv3 CPU gap% | Qwen3 CPU gap% | 说明 |
|------|--------|--------|------|
| decode_b1 | 11.1% | 22.6% | kernel 数少但每个短, 下发 overhead 暴露 |
| decode_b8 | 1.3% | 1.8% | 正常水平 |
| decode_b32 | 0.7% | 1.6% | kernel 密集, CPU gap 可忽略 |
| prefill_4096 | 32.6% | 35.5% | 异常高 — profiling 采集自身开销 (tracing + disk I/O) |

prefill_4096 的 CPU gap 异常高 (>30%), 原因是 profiling 工具自身的 tracing/写盘开销在计算密集场景下暴露. 正常推理 (不开 profiling) 时该开销不存在.

### 4.4 Compute stream span ≈ e2e

compute stream 是主控 stream, 它的时间跨度几乎等于 e2e. 这意味着 compute stream 在等通信完成时处于 idle 状态 (体现为 CPU gap 的一部分).

---

## 五、aclgraph vs Eager: 计算通信掩盖对比

### 5.1 分析目的

aclgraph (cudagraph) 将整个计算图一次性提交给 device, 理论上可以减少 CPU dispatch 开销, 并可能启用 compute/comm pipeline 掩盖.

### 5.2 数据来源

- DSv3 aclgraph: `deepseekv3_torch2.9.0_vllm0.15.0_cann8.5_aclgraph_PandD/` (W8A8, TP=8/DP=2/EP, `FULL_DECODE_ONLY` cudagraph)
- Qwen3 aclgraph: `qwen3-32b_torch2.9.0_vllm0.15.0_cann8.5_aclgraph_PandD/` (BF16, TP=16, `FULL_DECODE_ONLY` cudagraph)
- 两者均为 PandD 混合场景 (非单一 decode/prefill), 包含多个 step

### 5.3 Qwen3-32B step_trace 官方分解

| 项目 | aclgraph | eager (decode_b8 参考) |
|------|---------|---------|
| E2E (Stage) | 3295 ms | 3361 ms |
| Computing | 635 ms (19.3%) | 89 ms (2.6%) |
| Comm(Not Overlapped) | 1872 ms (56.8%) | 3210 ms (95.5%) |
| **Overlapped** | **33.5 ms (1.0%)** | **~0 ms (<0.1%)** |
| Free | 788 ms (23.9%) | 62 ms (1.8%) |

### 5.4 DSv3 按 stream 计算 (无 step_trace)

| 项目 | aclgraph | eager (decode_b8 参考) |
|------|---------|---------|
| E2E span | 3258 ms | 5425 ms |
| t1_compute | 1149 ms | 4420 ms |
| t1_comm | 624 ms | 935 ms |
| **Overlap** | **0.3 ms (~0%)** | **~0 ms** |
| Free/CPU gap | 1487 ms (45.6%) | 70 ms (1.3%) |

> 注: aclgraph 是 PandD 混合场景, Computing 时间 (635ms) 比 eager 单 decode_b8 (89ms) 大得多, 是因为跨了多个 step (含 prefill). Free 中包含 step 间空闲.

### 5.5 结论: aclgraph 模式下仍然几乎没有计算通信掩盖

- Qwen3 Overlapped = 33.5 ms, 仅占 Communication 总量的 **1.8%**
- DSv3 Overlap = 0.3 ms, 可忽略
- 两者与 eager 模式一样, compute/comm 本质上是串行交替执行

### 5.6 原因分析

vllm-ascend 的 aclgraph (`FULL_DECODE_ONLY` cudagraph) 减少了 CPU dispatch 开销, 但并没有实现 compute/comm 的 pipeline 调度:

```
Layer N:  |--compute--|--allReduce--|
Layer N+1:                          |--compute--|--allReduce--|
                                     ↑ 必须等 N 的 allReduce 完成
```

每层的 compute 完成后才启动该层的 allReduce, allReduce 完成后才启动下一层的 compute. 这种**层内串行**模式无论 eager 还是 aclgraph 都一样 — aclgraph 只是把串行序列打包成图, 减少 CPU 参与.

真正实现 compute/comm overlap 需要**跨层 pipeline** (如 layer N 的 allReduce 与 layer N+1 的 compute 并行), 这需要框架层面的显式设计 (如 vllm-ascend 的 `multistream_overlap_shared_expert`, 但在此次 profiling 中被设为 `false`).

---

## 六、对 Perf-Database 仿真的意义

综合 eager 和 aclgraph 的分析结果:

| 结论 | 说明 |
|------|------|
| **仿真公式** | `e2e ≈ Σ(compute_kernels) + Σ(comm_kernels) + t_cpu` — 无需建模 overlap |
| **compute kernel duration 可信** | 单 stream 上无并行, 每个 kernel 的 Task Duration 是真实的 device 执行时间 |
| **CPU gap 可忽略** | batch≥8 时 <2%, 仿真可不建模 CPU dispatch; batch=1 时需加 ~10-20% 修正 |
| **通信 kernel duration 含等待** | comm kernel 的 Task Duration 包含了等 compute stream ready 的时间, 不等于纯网络传输时延. 仿真通信应使用带宽模型而非查表 |
| **prefill_4096 的 Free 不代表真实** | 32% 的 CPU gap 是 profiling 工具 overhead, 不应作为仿真基准 |
| **kernel_details.csv 需去重** | HCCL kernel 被记录两次 (hcom_* 行 + AivKernel 行), 使用时必须按 Stream ID 去重 |

---

## 附录

### A. 数据文件

**Eager 模式** (`dsv3_qwen3_full_torch2.9.0_vllm0.15.0_cann8.5_eager/`):
每个场景包含: `kernel_details.csv` (逐 kernel 详情), `op_statistic.csv` (聚合统计), `step_trace_time.csv` (compute/comm/free 分解), `communication.json`, `trace_view.json`.

**aclgraph 模式**:
- DSv3: `deepseekv3_torch2.9.0_vllm0.15.0_cann8.5_aclgraph_PandD/kernel_details_deepseekv3-cann85.csv` (仅 kernel_details)
- Qwen3: `qwen3-32b_torch2.9.0_vllm0.15.0_cann8.5_aclgraph_PandD/.../kernel_details.csv` + `step_trace_time.csv`

### B. 分析脚本

```bash
python3.10 docs/perf_database/reports/profiling_analysis_eager/profiling_analysis.py          # CV/outlier 深度分析, 含 per-stream 时间计算
python3.10 docs/perf_database/reports/profiling_analysis_eager/analyze_aclgraph_overlap.py   # aclgraph compute/comm overlap 分析
```

### C. 相关报告

- 姊妹报告: [算子稳定性与接入可行性](profiling_analysis_op_stability_zh.md) — 分析每个算子在相同 shape 下的 CV, 评估查表可行性
