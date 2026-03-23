# Phase 1 Profiling 分析: Kernel Duration 与端到端时间关系 (aclgraph)

**软件栈**: CANN 8.5 / vLLM 0.15.0 / PyTorch 2.9.0 / **aclgraph** (FULL_DECODE_ONLY cudagraph)
**硬件**: Atlas 800 A3 (Ascend 910B)
**模型**: DeepSeek-V3 (W8A8, TP=8/DP=2/EP) + Qwen3-32B (BF16, TP=16)
**数据**: `/mnt/d/Data/Profiling/Profiling-0313-phase1-e2e-test/`

---

## 核心结论

1. **e2e = Computing + Comm(Not Overlapped) + Free 在所有场景精确成立** (step_trace 验证)
2. **Compute/Comm Overlap 仍然可忽略** — 最大 13.7ms (Qwen3 Decode), 占 comm 仅 1%
3. **aclgraph 的 Free/CPU gap** 与 eager 模式不同: PandD 场景的 Free 包含 step 间空闲, 不能直接对比
4. **aclgraph 下 HCCL 通信以 AivKernel 形式出现在独立 stream 上**, kernel_details.csv 需按此识别

---

## 一、aclgraph 模式的 Stream 结构

aclgraph 模式下 kernel_details.csv 的 stream 分布与 eager 模式有重要差异:

### 1.1 DSv3 (Prefill + Decode)

| Stream | 内容 | Prefill 时间 | Decode 时间 |
|--------|------|-------------|-------------|
| **Stream 2** | 主计算 (MatMul, Norm, Attention, DispatchFFNCombine) | 2,829 ms | 1,306 ms |
| **Stream 37** | AivKernel = **HCCL 通信执行** | 100 ms | 3,425 ms |
| Stream 40 | Sampling (Sort, ArgMax, Neg) | 6 ms | 6 ms |
| Stream 69 | AICPU comm (reduce_scatter, allgather) | 275 ms | — |
| **NaN** | hcom_* 行 (HCCL 官方记录, 与 Stream 37 对应) | 1,361 ms | 3,425 ms |

**关键发现: Stream 37 上的 AivKernel = HCCL 通信**
- DSv3 Decode: Stream 37 (977 kernels, 3,425.1 ms) ≈ NaN hcom (975 kernels, 3,425.1 ms) — **时间完全匹配**
- DSv3 Prefill: Stream 37 仅 262 AivKernel (100ms), 因为 prefill 的通信走 AICPU 路径 (Stream 69)

### 1.2 Qwen3-32B (Prefill + Decode)

| Stream | 内容 | Prefill 时间 | Decode 时间 |
|--------|------|-------------|-------------|
| **Stream 6/62** | 主计算 (eager 路径) | 3,242 ms | 88 ms |
| **Stream 148** | Graph-compiled 计算 (decode 主体) | — | 1,429 ms |
| **Stream 149** | AivKernel = **HCCL 通信执行** | — | 1,403 ms |
| Stream 67 | AivKernel (少量, prefill) | 0.2 ms | 10 ms |
| Stream 71 | Sampling | 3 ms | 47 ms |
| Stream 99 | AICPU comm (prefill only) | 5,572 ms | — |
| **NaN** | hcom_* 行 | 2,491 ms | 1,413 ms |

**注意:**
- Qwen3 Prefill: AICPU comm stream (5,572 ms) >> NaN hcom (2,491 ms). AICPU 时间含等待 HCCL 完成的 idle。**step_trace 使用 NaN hcom 时间 (2,491 ms) 作为权威 Communication 值。**
- Qwen3 Decode: Stream 149 AivKernel (1,403 ms) ≈ NaN hcom (1,413 ms)

### 1.3 与 eager 模式的 Stream 差异

| 特征 | Eager 模式 | aclgraph 模式 |
|------|-----------|--------------|
| HCCL 执行 stream | 独立 HCCL stream (有 Stream ID) | AivKernel 在独立 stream (37/149) |
| hcom_* 双重计数 | hcom (NaN) + AivKernel (HCCL stream) | hcom (NaN) + AivKernel (新 stream) |
| Graph-compiled kernel | 无 | Decode 有 hash 后缀 (Stream 148) |
| AICPU comm | Prefill 有 | 同 eager |

---

## 二、step_trace 官方分解

step_trace_time.csv 直接给出 Computing / Communication / Overlapped / Free 的权威分解:

| 场景 | Computing (ms) | Comm(Not Overlapped) (ms) | **Overlapped (ms)** | Free (ms) | **Stage/e2e (ms)** |
|------|-----------|-----------|-----------|------|------|
| DSv3 Prefill | 2,829 (62.1%) | 1,361 (29.9%) | **0.07 (~0%)** | 365 (8.0%) | **4,555** |
| DSv3 Decode | 1,306 (26.8%) | 3,425 (70.2%) | **0.27 (~0%)** | 149 (3.1%) | **4,880** |
| Qwen3 Prefill | 3,242 (55.7%) | 2,491 (42.8%) | **0.0 (0%)** | 83 (1.4%) | **5,816** |
| Qwen3 Decode | 1,532 (49.5%) | 1,399 (45.1%) | **13.7 (0.4%)** | 166 (5.4%) | **3,098** |

**验证: e2e = Computing + Comm(Not Overlapped) + Free**
- DSv3 Prefill: 2829 + 1361 + 365 = 4,555 ✓
- DSv3 Decode: 1306 + 3425 + 149 = 4,880 ✓
- Qwen3 Prefill: 3242 + 2491 + 83 = 5,816 ✓
- Qwen3 Decode: 1532 + 1399 + 166 = 3,098 ✓ (Overlap 13.7ms 被分别扣减)

---

## 三、Compute / Comm Overlap 分析

**所有场景 Overlap 均可忽略:**

| 场景 | Overlapped | 占 Comm 比例 | 结论 |
|------|-----------|-------------|------|
| DSv3 Prefill | 0.07 ms | 0.005% | 无 overlap |
| DSv3 Decode | 0.27 ms | 0.008% | 无 overlap |
| Qwen3 Prefill | 0.0 ms | 0% | 无 overlap |
| Qwen3 Decode | 13.7 ms | **0.97%** | 微量 overlap, 可忽略 |

与之前 eager 分析结论一致: **aclgraph 没有启用 compute/comm pipeline**。每层仍然是 compute → allReduce → 下一层 compute 的串行模式。

原因不变: `multistream_overlap_shared_expert=false`, aclgraph 只打包串行序列, 减少 CPU dispatch, 但不改变 stream 间依赖关系。

---

## 四、CPU Gap / Free 时间分析

| 场景 | Free (ms) | Free% | 说明 |
|------|----------|-------|------|
| DSv3 Prefill | 365 | 8.0% | 包含 PandD step 间空闲 + profiling overhead |
| DSv3 Decode | 149 | 3.1% | 正常 CPU dispatch gap |
| Qwen3 Prefill | 83 | 1.4% | 极低, 几乎无 gap |
| Qwen3 Decode | 166 | 5.4% | 包含 PandD step 间空闲 |

与之前 eager 数据对比:

| 场景 | eager Free% | aclgraph Free% | 说明 |
|------|-----------|---------------|------|
| DSv3 decode_b8 | 1.3% | 3.1% | aclgraph PandD 含 step 间 idle |
| Qwen3 decode_b8 | 1.8% | 5.4% | 同上 |
| Qwen3 prefill_4096 | 35.5% | 1.4% | eager 的 35% 是 profiling overhead |

> **注意: aclgraph PandD 场景的 Free 不能直接与 eager 单场景对比**, 因为 PandD 的 profiling trace 跨多个 step, Free 包含了 step 间的调度空闲。Qwen3 Prefill 的 1.4% 更能代表实际 CPU gap 下限。

---

## 五、Per-Stream 验证

通过按 Stream ID 分组 kernel_details.csv 验证 step_trace:

### 5.1 DSv3 Prefill

| 来源 | Compute (ms) | Comm (ms) |
|------|-------------|-----------|
| step_trace | 2,829 | 1,361 (NaN hcom) |
| Stream 分组 | 2,829 (Stream 2) | 1,361 (NaN hcom) |
| ✅ **匹配** | | |

分解: Stream 2 = 计算 0.315s + DispatchFFNCombine 2.515s = 2.830s

### 5.2 DSv3 Decode

| 来源 | Compute (ms) | Comm (ms) |
|------|-------------|-----------|
| step_trace | 1,306 | 3,425 (NaN hcom) |
| Stream 分组 | 1,306 (Stream 2) + 6 (Stream 40) | 3,425 (Stream 37 AivKernel = NaN hcom) |
| ✅ **匹配** | | |

### 5.3 Qwen3 Prefill

| 来源 | Compute (ms) | Comm (ms) |
|------|-------------|-----------|
| step_trace | 3,242 | 2,491 (NaN hcom) |
| Stream 分组 | 3,242 (Stream 6) | **5,572 (Stream 99 AICPU)** ≠ 2,491 |

> AICPU stream 上的 allgatherAicpuKernel + reduce_scatterAicpuKernel 总时间 (5,572 ms) 远大于实际 HCCL 执行 (2,491 ms). AICPU kernel 的 Duration 包含了等待 HCCL 完成的时间。**使用 NaN hcom 时间 (= step_trace Communication) 才是正确的通信时间。**

### 5.4 Qwen3 Decode

| 来源 | Compute (ms) | Comm (ms) |
|------|-------------|-----------|
| step_trace | 1,532 | 1,413 (NaN hcom) |
| Stream 分组 | 1,429 (Stream 148) + 88 (Stream 62) + 47 (Stream 71) + 10 (Stream 67) = 1,574 | 1,403 (Stream 149 AivKernel) |

> Compute stream 合计 (1,574ms) 略大于 step_trace Computing (1,532ms), 差异来自 Stream 71 上的 sampling 算子 (47ms) 可能被 step_trace 归入 Free。

---

## 六、对 Perf-Database 仿真的意义

| 结论 | 说明 |
|------|------|
| **仿真公式** | `e2e ≈ Σ(compute_kernels) + Σ(comm) + t_free` — aclgraph 下依然成立 |
| **不需要建模 overlap** | Overlap < 1% |
| **comm 时间使用 NaN hcom** | Stream 上的 AICPU kernel 含等待时间, 不等于实际通信 |
| **aclgraph Free 含 step 间 idle** | 仿真不应直接用 PandD trace 的 Free 作为 CPU gap |
| **Graph kernel 名称需归一化** | `MatMulV2_NDNZ_..._229955` → `MatMulV2` |
| **AivKernel = HCCL comm** | 在 kernel_details.csv 中需识别并正确分类 |
| **单次 PandD trace 可提供 baseline** | 无需多个单场景 profiling, PandD 即可覆盖 prefill + decode |

---

## 附录

### A. 数据文件

每个场景包含: `kernel_details.csv`, `step_trace_time.csv`, `op_statistic.csv`, `communication.json`, `communication_matrix.json`, `operator_details.csv`, `api_statistic.csv`, `trace_view.json`.

配置文件:
- `deepseekv3_vllm+bench配置.txt`
- `qwen3-32b_vllm+bench配置.txt`

### B. aclgraph Stream 识别规则

```
DSv3:
  Stream 2  → Main compute (含 DispatchFFNCombine)
  Stream 37 → AivKernel = HCCL comm
  Stream 40 → Sampling
  Stream 69 → AICPU comm (prefill only)
  NaN       → hcom_* (authoritative comm timing)

Qwen3:
  Stream 6/62  → Main compute (eager path)
  Stream 148   → Graph-compiled compute (decode main)
  Stream 149   → AivKernel = HCCL comm (decode)
  Stream 67    → AivKernel (minimal, prefill)
  Stream 71    → Sampling
  Stream 99    → AICPU comm (prefill only)
  NaN          → hcom_* (authoritative comm timing)
```

### C. 分析脚本

```bash
python3.10 docs/perf_database/reports/profiling_analysis_aclgraph/analyze_phase1.py    # Stream 分组 + 算子分类
```

### D. 相关报告

- 姊妹报告: [算子稳定性与接入可行性](report_op_stability_zh.md)
- 前序报告 (eager): [kernel duration 与 e2e](../profiling_analysis_kernel_vs_e2e_zh.md)
