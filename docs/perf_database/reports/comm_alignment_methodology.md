# 通信算子数据对齐方法论

**版本**：v3.1
**日期**：2026-03-14
**作者**：HDY
**适用场景**：验证 HCCL microbenchmark 采集数据与 vLLM-ascend 生产环境 Profiling 数据的一致性

> **v3.1 更新**：AIV 模式微基准验证完成，确认 AIV 对独立微基准无显著影响（0.9-1.3x）。
> bench-profiling gap 根因为 profiling Duration 语义差异，而非环境配置问题。
> v2.0 基于 enforce-eager 非生产环境，结论已失效。

---

## 1. 背景与目标

HCCL microbenchmark（`generate_comm_microbench.py`）采集的是**纯通信时间**（单独 torchrun，无 compute 负载）。vLLM-ascend Profiling 采集的是**端到端推理中通信 kernel 的 Duration**。

**v3.1 关键发现**：AIV 模式对独立 HCCL 微基准无显著加速效果（加速比 0.9-1.3x）。bench-profiling gap 的根因是 **profiling Duration 语义差异**——CUDAGraph/async-scheduling 下 Duration 不代表完整集合通信 wall-clock 时间，MoE EP 场景下 Duration 包含隐式协调等待。bench CSV 测量纯通信时间，与 profiling Duration 测量的不是同一物理量。

对齐目标：确认 microbenchmark 数据在合理范围内，可作为 `EmpiricalPerformanceModel` 的通信查询数据源。

---

## 2. 环境一致性要求（v3.1 更新）

**微基准采集建议开启与 vLLM 生产环境一致的环境变量**：

```bash
export HCCL_OP_EXPANSION_MODE="AIV"   # 建议（实测对独立微基准影响 <30%）
export TASK_QUEUE_ENABLE=1             # 建议
```

**v3.1 验证结论**：AIV 模式对独立 HCCL 微基准无显著加速效果（加速比 0.9-1.3x，详见 [对齐报告 §4.1](comm_alignment_report_20260312.md#41-aiv-模式验证)）。

开启 AIV 仍然是最佳实践（保持环境一致），但**不开启不会导致数据不可用**。

---

## 3. 对齐方法

### 3.1 整体流程

```
Profiling kernel_details.csv
        │
        ▼
① 提取通信算子耗时（hcom_* kernel，去 AivKernel 重复）
        │
        ▼
② 推算 message_bytes（从 operator_details Input Shapes）
        │
        ▼
③ 从 microbench CSV 插值得到对应 message_bytes 的预测耗时
        │
        ▼
④ 计算 ratio = Profiling / Microbench，判断 PASS/WARN/FAIL
```

### 3.2 步骤详解

#### ① 提取 Profiling 通信算子耗时

从 `kernel_details.csv` 按 `Name` 前缀过滤：

| Name 前缀 | 对应算子 |
|-----------|---------|
| `hcom_allReduce_` | AllReduce |
| `hcom_allGather_` | AllGather |
| `hcom_reduceScatter_` | ReduceScatter |
| `hcom_alltoallv_` | AllToAll |

**名称归一化**：kernel 名称带唯一后缀（如 `hcom_allGather__503_0_1`），需用正则 `hcom_\w+?_\d+_\d+_\d+` 归一化为基础名。

**稳态耗时**：取 p10-p90 区间的中位数：

```python
def stable_median(durs):
    s = sorted(durs)
    lo, hi = int(len(s) * 0.1), int(len(s) * 0.9)
    return statistics.median(s[lo:hi])
```

#### ② 推算 message_bytes

**优先从 operator_details.csv 获取**：`c10d::_allgather_base_` / `c10d::_reduce_scatter_base_` 的 `Input Shapes` 字段包含实际 tensor shape。

```python
# operator_details.csv 中 c10d 算子的 Input Shapes 格式：
# "2565,5120;41040,5120;;\n;\n"  (分号分隔多个输入)
# 取第一个 shape，计算 elements * bytes_per_element
first_shape = row['Input Shapes'].split(';')[0].strip()
dims = [int(x) for x in first_shape.split(',')]
msg_bytes = math.prod(dims) * 2  # BF16 = 2 bytes
```

**kernel_details.csv 中通信算子的 Input Shapes 为 N/A**，不可用。

**备选：从模型配置反推**（当 operator_details 不可用时）：

| 算子 | message_bytes | 说明 |
|------|-------------|------|
| AllGather (SP prefill) | `(seq_len/tp) × hidden × 2` | per-rank chunk |
| ReduceScatter (SP prefill) | `seq_len × hidden × 2` | 全量 tensor |
| AllReduce (decode) | `batch × hidden × 2` | RowParallelLinear 输出 |

#### ③ microbench CSV 插值

对目标 message_bytes 做**对数线性插值**：

```python
def interp_bench(rows, msg_bytes, n_dev, tier):
    candidates = [(mb, d) for mb, nd, t, d in rows if nd == n_dev and t == tier]
    candidates.sort()
    for i in range(len(candidates) - 1):
        mb0, d0 = candidates[i]
        mb1, d1 = candidates[i + 1]
        if mb0 <= msg_bytes <= mb1:
            t = (math.log(msg_bytes) - math.log(mb0)) / (math.log(mb1) - math.log(mb0))
            return d0 + t * (d1 - d0)
```

#### ④ 判断标准

| ratio = Profiling / Microbench | 结论 |
|-------------------------------|------|
| 0.5x ～ 2.0x | **PASS** |
| 0.25x ～ 4.0x | **WARN** |
| 其他 | **FAIL**：需排查根因 |

**v3.1 注意**：ratio < 0.5x（profiling 比 bench 快）通常意味着 profiling Duration 受 CUDAGraph pipeline/overlap 影响，不代表完整集合通信时间。ratio > 2.0x 通常意味着 profiling Duration 包含隐式等待（如 MoE EP 协调）。

---

## 4. 已知干扰因素

### 4.1 Profiling Duration 语义差异（v3.1 更新，最重要）

profiling kernel_details 中通信 kernel 的 Duration 在不同执行模式下语义不同，这是 bench-profiling gap 的根因：

1. **CUDAGraph + async-scheduling 下**（如 Qwen3 decode）：通信 kernel 可能被 pipeline 化或与 compute overlap，Duration 不代表完整集合通信 wall-clock 时间，而是 NPU 上该 kernel 的实际执行片段。表现为 profiling 远快于 bench（ratio 0.05-0.22x）。

2. **MoE EP 场景**（如 DSV3 decode）：reduceScatter Duration 包含 expert parallel 协调等待（跨 DP group 的 token dispatch/combine 同步）。14KB 消息量不应需要 5.6ms 纯通信。表现为 profiling 远慢于 bench（ratio 9-20x）。

3. **SP prefill reduceScatter**（如 Qwen3 prefill）：在 compute 之后串行执行，Duration 语义最接近纯通信时间。ratio 1.28-1.57x，在合理范围内。

**结论：bench CSV 数据质量可靠，但不能直接与 profiling Duration 对比。** 两者测量的不是同一个物理量。

> ~~v3.0 原结论~~：AIV 模式差异是最重要的干扰因素。**已证伪**——AIV 对独立微基准无显著影响（0.9-1.3x）。

### 4.2 CUDAGraph 对 Decode 通信的影响

生产环境使用 `FULL_DECODE_ONLY` CUDAGraph，decode 阶段的 kernel launch overhead 被消除，通信 kernel 的 Duration 更接近纯数据传输时间。非 CUDAGraph（enforce-eager）模式下 Duration 包含更多调度等待。

### 4.3 AivKernel 重复记录

同一次通信可能产生两条记录（主 kernel + AivKernel），Duration 相同。统计时需去重。

### 4.4 HCCL JIT 编译开销

首次运行某个 message_size 时触发 JIT 编译，耗时可达正常值 10x。microbench 通过 `WARMUP_ITERS=20` 规避，Profiling 取 p10-p90 稳态中位数去除。

### 4.5 topology_tier 推断

- ATLAS_800_A3 单节点 16 卡：tier=1（intra_pod）
- 同节点 2 卡（同 die）：tier=2（die_level）
- 跨节点：tier=0（inter_pod）

### 4.6 DSV3 特有因素

1. **MC2 融合**：DSV3 TP 通信走 `matmul_allreduce` 融合算子，无独立 allReduce
2. **DispatchFFNCombine 封装 EP 通信**：alltoall 无独立记录
3. **MoE 通信方差大**：prefill 阶段 allGather P10=29us vs P90=2785us，与 expert 负载不均有关

### 4.7 四算子可观测性（v3.0 更新）

| 算子 | Qwen3-32B | DSV3 |
|------|-----------|------|
| allGather | hcom_allGather_ | hcom_allGather_ |
| reduceScatter | hcom_reduceScatter_ | hcom_reduceScatter_ |
| allReduce | hcom_allReduce_ | 不存在（MC2 融合） |
| allToAll | 不存在（dense 模型） | 不存在（DispatchFFNCombine 封装） |

> v2.0 报告中 DSV3 使用 `allgatherAicpuKernel` 路径，在 vLLM 0.15.0 + CANN 8.5 生产环境中未观察到，已统一为 `hcom_*` 路径。

---

## 5. Wall-Clock 端到端分析方法

### 5.1 E2E 时间轴模型

```
 NPU Timeline (一个完整 step)
 ═══════════════════════════════════════════════════════════════
 ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
 │Compute₁ │ │ Comm₁   │ │Compute₂ │ │ Comm₂   │  ...×N layers
 └─────────┘ └─────────┘ └─────────┘ └─────────┘

 Stage = Computing + Communication_Not_Overlapped + Free
```

| 指标 | 含义 |
|------|------|
| Stage | wall-clock 时间 |
| Computing | compute kernel 执行时间总和 |
| Communication_Not_Overlapped | 通信中未与 compute 重叠的部分 |
| Free | 调度、kernel launch、stream sync 等空闲时间 |

### 5.2 Overhead Factor

```
Factor = Stage / (Computing + Communication_Not_Overlapped)
       = 1 + Free / (Computing + Communication_Not_Overlapped)
```

- Factor ≈ 1.0x → 调度开销可忽略
- Factor > 1.1x → 需关注

### 5.3 生产环境实测（0313 数据）

| 模型 | 场景 | Stage(ms) | Comp+Comm(ms) | Free(ms) | Factor | Free% |
|------|------|-----------|--------------|----------|--------|-------|
| Qwen3 | Prefill | 5816 | 5733 | 83 | 1.014x | 1.4% |
| Qwen3 | Decode | 3098 | 2931 | 167 | 1.057x | 5.4% |
| DSV3 | Prefill | 4555 | 4190 | 365 | 1.087x | 8.0% |
| DSV3 | Decode | 4880 | 4731 | 149 | 1.031x | 3.1% |

生产环境 Factor 整体 1.01x-1.09x，CUDAGraph + async-scheduling 有效压低了调度开销。

---

## 6. 对比 Checklist

- [ ] 确认微基准采集时开启了 `HCCL_OP_EXPANSION_MODE="AIV"` 和 `TASK_QUEUE_ENABLE=1`
- [ ] 确认 profiling 的 device 数量和 TP 配置
- [ ] 从 operator_details `Input Shapes` 获取 message_bytes（优先于模型配置反推）
- [ ] 区分 prefill / decode 的通信耗时，分别对比
- [ ] 归一化 kernel 名称（去掉 `_\d+_\d+_\d+` 后缀）
- [ ] 去掉 AivKernel 重复记录
- [ ] DSV3：确认 MC2 融合（无独立 allReduce）
- [ ] DSV3：注意 MoE 通信方差大，使用 stable median
- [ ] ratio < 0.5x 时优先排查微基准 AIV 模式是否开启
- [ ] 计算 overhead factor 验证 E2E 一致性
