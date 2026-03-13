# 通信算子数据对齐方法论

**版本**：v2.0
**日期**：2026-03-13
**作者**：HDY
**适用场景**：验证 HCCL microbenchmark 采集数据与 vLLM-ascend Profiling 数据的一致性

---

## 1. 背景与目标

HCCL microbenchmark（`generate_comm_microbench.py`）采集的是**纯通信时间**（单独 torchrun，无 compute 负载）。vLLM-ascend Profiling 采集的是**端到端推理中通信 kernel 的 Duration**，可能包含调度等待开销。

对齐目标：确认 microbenchmark 数据在合理范围内，可作为 `EmpiricalPerformanceModel` 的通信查询主数据源（配合校准系数）。

---

## 2. 对齐方法

### 2.1 整体流程

```
Profiling kernel_details.csv
        │
        ▼
① 提取通信算子耗时（按 Type 过滤，去 AivKernel 重复）
        │
        ▼
② 推算 message_bytes（从模型配置 + Input Shapes 反推）
        │
        ▼
③ 从 microbench CSV 插值得到对应 message_bytes 的预测耗时
        │
        ▼
④ 计算 ratio = Profiling / Microbench，判断 PASS/WARN/FAIL
```

### 2.2 步骤详解

#### ① 提取 Profiling 通信算子耗时

从 `kernel_details.csv` 按 `Type` 字段过滤：

| Type 字段 | 对应算子 |
|-----------|---------|
| `hcom_allReduce_` | AllReduce |
| `hcom_allGather_` | AllGather |
| `hcom_reduceScatter_` | ReduceScatter |
| `hcom_alltoallv_` | AllToAll |

**去重**：过滤掉 `Name == "AivKernel"` 的行（同一次通信产生主 kernel + AivKernel 两条记录）。

**稳态耗时**：取 p10-p90 区间的中位数，去掉 warmup 和偶发抖动：

```python
def stable_median(durs):
    s = sorted(durs)
    lo, hi = int(len(s) * 0.1), int(len(s) * 0.9)
    return statistics.median(s[lo:hi])
```

#### ② 推算 message_bytes

Profiling 里通信算子的 `Input Shapes` 字段为 `N/A`，需从模型配置反推。

| 算子 | message_bytes | 说明 |
|------|-------------|------|
| AllReduce | `seq_len × hidden × 2` | RowParallelLinear 输出后 all-reduce |
| AllGather | `seq_len × (hidden / tp) × 2` | ColumnParallelLinear 前 all-gather |
| ReduceScatter | `seq_len × (hidden / tp) × 2` | SP 模式下替代 AllReduce |
| AllToAll | `seq_len × (hidden / tp) × 2` | MoE EP dispatch/combine |

**从 Profiling 确认 hidden_size**：查看 `MatMulV2` 的 `Input Shapes`，第一个矩阵的第二维即为 `hidden`。

#### ③ microbench CSV 插值

microbench CSV 的 message_bytes 是离散的（1KB~512MB），需对目标 message_bytes 做**对数线性插值**：

```python
def interp_bench(rows, msg_bytes, n_dev, tier):
    rows = [(mb, d) for mb, d in rows if n_dev == target_n and tier == target_tier]
    rows.sort()
    lo = [(mb, d) for mb, d in rows if mb < msg_bytes]
    hi = [(mb, d) for mb, d in rows if mb > msg_bytes]
    if lo and hi:
        mb0, d0 = lo[-1]; mb1, d1 = hi[0]
        t = (math.log(msg_bytes) - math.log(mb0)) / (math.log(mb1) - math.log(mb0))
        return d0 + t * (d1 - d0)
```

#### ④ 判断标准

| ratio = Profiling / Microbench | 结论 |
|-------------------------------|------|
| 0.5x ～ 2.0x | **PASS** |
| 0.25x ～ 4.0x | **WARN** |
| 其他 | **FAIL**：需排查根因 |

---

## 3. 已知干扰因素

### 3.1 Profiling Duration 包含调度等待时间

HCCL 通信 kernel 的 Duration 从 host 下发算子时开始计时，包含前序 compute kernel 的等待时间（pipeline bubble）。AllReduce 受此影响最大（9.9x），AllGather / ReduceScatter 影响较小。

### 3.2 AivKernel 重复记录

同一次通信产生两条记录（主 kernel + AivKernel），Duration 相同。统计时需去重。

### 3.3 HCCL JIT 编译开销

首次运行某个 message_size 时触发 JIT 编译，耗时可达正常值 10x。microbench 脚本通过全局预热 + `WARMUP_ITERS=20` 规避，Profiling 数据取 p10-p90 稳态中位数去除。

### 3.4 topology_tier 推断

- ATLAS_800_A3 单节点 16 卡：tier=1（intra_pod）
- 同节点 2 卡（同 die）：tier=2（die_level）
- 跨节点：tier=0（inter_pod）

### 3.5 DSV3 特有干扰因素

1. **使用 AicpuKernel 而非 hcom_xxx_**：DSV3 的 `hcom_allGather_` 含大量 <20us 空调用，应使用 `allgatherAicpuKernel` / `reduce_scatterAicpuKernel`
2. **DispatchFFNCombine 封装 EP 通信**：alltoall 无独立记录
3. **DSV3 无独立 allReduce**：MC2 融合
4. **MoE 双峰分布**：40-60% comm 调用为空操作（<100us），分析时需过滤

### 3.6 四算子可观测性

| 算子 | Qwen3-32B | DSV3 |
|------|-----------|------|
| allGather | hcom_allGather_ | allgatherAicpuKernel |
| reduceScatter | hcom_reduceScatter_ | reduce_scatterAicpuKernel |
| allReduce | hcom_allReduce_（含调度等待） | 不存在（MC2 融合） |
| allToAll | 不存在（dense 模型） | 不存在（DispatchFFNCombine 封装） |

---

## 4. 自动化工具

```bash
python tools/perf_data_collection/validate_comm_alignment.py \
    --csv-dir tensor_cast/performance_model/perf_database/data/ATLAS_800_A3_752T_128G_DIE/hccl/v8.5/ \
    --tolerance 2.0 --verbose
```

---

## 5. 对比 Checklist

- [ ] 确认 profiling 的 device 数量（`Device_id` 字段）
- [ ] 从 `MatMulV2` Input Shapes 确认 `hidden_size` 和 `seq_len`
- [ ] 区分 decode / prefill 的通信耗时，分别对比
- [ ] 去掉 `AivKernel` 重复记录
- [ ] DSV3：使用 AicpuKernel 路径
- [ ] DSV3：确认 forward pass 数量，按 pass 隔离分析
- [ ] DSV3：过滤 <100us 空 comm 调用（MoE 双峰分布）
- [ ] allReduce FAIL 时，优先判断是否是调度等待导致
- [ ] 计算 wall-clock E2E 和 overhead factor（见 §6）

---

## 6. Wall-Clock 端到端分析方法

### 6.1 背景

单算子 bench vs profiling 对比只能验证数据质量，不能预测端到端精度。算子间存在 inter-operator overhead（kernel launch、stream sync、host 调度），不被任何算子 Duration 覆盖。

### 6.2 E2E 时间轴模型

#### 6.2.1 单 Step 时间轴分解

一个推理 step 在 NPU 上的执行时间轴如下：

```
 Host (CPU)
 ─────────────────────────────────────────────────────────────────────────────
 │ dispatch  │ dispatch  │ dispatch  │ dispatch  │ dispatch  │              │
 │ compute₁  │ comm₁     │ compute₂  │ comm₂     │ compute₃  │  waiting...  │
 ─────────────────────────────────────────────────────────────────────────────

 NPU (Device)                                                    ◄── Stage ──►
 ─────────────────────────────────────────────────────────────────────────────
 │ Compute₁  │▒▒▒│  Comm₁  │▒│ Compute₂  │▒▒▒▒▒│  Comm₂  │▒│ Compute₃  │▒│
 ─────────────────────────────────────────────────────────────────────────────
 │           │   │         │ │           │     │         │ │           │ │
 │ Computing │   │  Comm   │ │ Computing │     │  Comm   │ │ Computing │ │
 │           │▒▒▒│         │▒│           │▒▒▒▒▒│         │▒│           │▒│
 │           │Free         │F│           │Free  │         │F│           │F│
 ─────────────────────────────────────────────────────────────────────────────

 图例:  █ Computing    █ Communication    ▒ Free (调度/launch/sync)
```

**关键关系**：

```
Stage = Computing + Communication_Not_Overlapped + Free
                                                    │
                                    ┌───────────────┘
                                    ▼
                    kernel launch + stream sync + host 调度
                    + pipeline bubble (前序 kernel 等待)
```

#### 6.2.2 单次通信算子的 Duration 语义差异

```
                    Profiling Duration
          ◄─────────────────────────────────────►
          │                                     │
 ─────────┬──────────────┬──────────────────────┬─────────
          │  Scheduling  │                      │
          │    Wait      │  HCCL Communication  │
          │ (pipeline    │  (actual data xfer)  │
          │  bubble)     │                      │
 ─────────┴──────────────┴──────────────────────┴─────────
          │              │                      │
          │              ◄──────────────────────►
          │                 Bench Duration
          │                (pure comm only)
          │
          ▼
    host dispatch 时刻
    (前序 compute 可能还在执行)
```

**这是 Bench-Profiling Gap 的根本原因**：

| 度量 | 起点 | 终点 | 包含内容 |
|------|------|------|---------|
| Bench Duration | HCCL 通信开始 | 通信结束 | 纯数据传输 |
| Profiling Duration | host dispatch comm | 通信结束 | 调度等待 + 数据传输 |

Gap 大小取决于前序 compute kernel 的剩余执行时间，因此：
- AllGather（通常在 layer 开头，前序 kernel 短）→ gap 小（1.8-3.6x）
- AllReduce（在 MatMul 之后，前序 kernel 长）→ gap 大且不稳定（9.6-22.5x）

#### 6.2.3 干扰因素在时间轴上的位置

```
 NPU Timeline (一个完整 step)
 ═══════════════════════════════════════════════════════════════════════════

 ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
 │Compute₁ │ │ Comm₁   │ │Compute₂ │ │ Comm₂   │ │Compute₃ │  ...×64 layers
 └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘

      ▲            ▲            ▲           ▲
      │            │            │           │
      │     ┌──────┴──────┐    │    ┌──────┴──────┐
      │     │ 3.1 调度等待 │    │    │ 3.1 调度等待 │
      │     │ Profiling    │    │    │ AllReduce    │
      │     │ Duration     │    │    │ 受影响最大   │
      │     │ 包含此部分   │    │    │ (9.6-22.5x) │
      │     └─────────────┘    │    └─────────────┘
      │                        │
 ┌────┴────────────────────────┴────┐
 │ 3.3 JIT 编译开销                  │
 │ 首次 message_size → ~10x 正常值   │
 │ Bench: WARMUP=20 规避             │
 │ Profiled: p10-p90 去除            │
 └──────────────────────────────────┘

 kernel_details.csv 中每次 Comm 产生:
 ┌──────────────────┐  ┌──────────────────┐
 │ hcom_allGather_  │  │ AivKernel        │  ← 3.2 重复记录
 │ Duration=1062us  │  │ Duration=1062us  │    统计时需去重
 └──────────────────┘  └──────────────────┘
```

### 6.3 分析流程

```
Profiling step_trace_time.csv / kernel_details.csv
        │
        ▼
① 从 step_trace_time.csv 提取 Computing / Communication_Not_Overlapped / Free
        │
        ▼
② overhead_factor = Stage / (Computing + Communication_Not_Overlapped)
```

### 6.4 Phase 2 实测结果（16 组 trace，2026-03-13）

**指标定义**（均来自 Ascend Profiler 自动生成的 `step_trace_time.csv`）：

| 指标 | 含义 |
|------|------|
| Stage | 整个 profiling 区间的 wall-clock 时间（第一个 kernel 开始 → 最后一个 kernel 结束） |
| Computing | 所有 compute kernel（MatMulV2、SwiGlu 等）在 NPU 上的实际执行时间总和 |
| Communication_Not_Overlapped | 通信 kernel 中**未与 compute 重叠**的部分（Ascend Profiler 自动分析时间线重叠关系） |
| Free | Stage 中既非 Computing 也非 Communication 的空闲时间（host 调度、kernel launch、stream sync 等） |

它们的关系：`Stage = Computing + Communication_Not_Overlapped + Free`

```
Factor = Stage / (Computing + Communication_Not_Overlapped)
       = 1 + Free / (Computing + Communication_Not_Overlapped)
```

- Factor ≈ 1.0x → 几乎没有调度开销，kernel 执行紧密衔接
- Factor > 1.1x → host 调度开销显著（batch 小时 kernel 短而密集，调度占比上升）

**实测数据**：

```
 Qwen3-32B (TP=16, Dense, BF16)
 ──────────────────────────────────────────────────────────────────────────────
 配置              │ Stage  │ Comp+Comm │ Free   │ Factor │ Free%  │ 状态
 ──────────────────┼────────┼───────────┼────────┼────────┼────────┼─────────
 decode batch=1    │ 1463ms │  1131ms   │ 332ms  │ 1.29x  │ 22.7%  │ ⚠ 高
 decode batch=4    │ 2218ms │  2087ms   │ 131ms  │ 1.06x  │  5.9%  │ ⚠ 边界
 decode batch=8    │ 3361ms │  3299ms   │  62ms  │ 1.02x  │  1.8%  │ ✓ 正常
 decode batch=16   │ 4981ms │  4950ms   │  30ms  │ 1.01x  │  0.6%  │ ✓ 正常
 decode batch=32   │ 7822ms │  7698ms   │ 124ms  │ 1.02x  │  1.6%  │ ✓ 正常
 prefill ISL=256   │ 1127ms │  1100ms   │  27ms  │ 1.02x  │  2.4%  │ ✓ 正常
 prefill ISL=1024  │ 1125ms │  1105ms   │  20ms  │ 1.02x  │  1.8%  │ ✓ 正常
 prefill ISL=4096  │ 1244ms │  1132ms   │ 111ms  │ 1.10x  │  8.9%  │ ⚠ 偏高
 ──────────────────┴────────┴───────────┴────────┴────────┴────────┴─────────

 DSV3 (TP=8, DP=2, MoE, W8A8)
 ──────────────────────────────────────────────────────────────────────────────
 配置              │ Stage  │ Comp+Comm │ Free   │ Factor │ Free%  │ 状态
 ──────────────────┼────────┼───────────┼────────┼────────┼────────┼─────────
 decode batch=1    │ 2654ms │  2355ms   │ 297ms  │ 1.13x  │ 11.2%  │ ⚠ 高
 decode batch=4    │ 5763ms │  5516ms   │ 244ms  │ 1.04x  │  4.2%  │ ✓ 正常
 decode batch=8    │ 5425ms │  5356ms   │  69ms  │ 1.01x  │  1.3%  │ ✓ 正常
 decode batch=16   │ 6172ms │  6105ms   │  67ms  │ 1.01x  │  1.1%  │ ✓ 正常
 decode batch=32   │ 8185ms │  8127ms   │  55ms  │ 1.01x  │  0.7%  │ ✓ 正常
 prefill ISL=256   │ 3822ms │  3775ms   │  47ms  │ 1.01x  │  1.2%  │ ✓ 正常
 prefill ISL=1024  │ 3856ms │  3810ms   │  46ms  │ 1.01x  │  1.2%  │ ✓ 正常
 prefill ISL=4096  │ 4811ms │  3316ms   │1493ms  │ 1.45x  │ 31.0%  │ ✗ 异常
 ──────────────────┴────────┴───────────┴────────┴────────┴────────┴─────────
```

**E2E 耗时分解可视化**（按 Free 占比排序）：

```
 Free Time 占比 (%)
 0%          5%         10%         15%         20%         25%         30%
 ├───────────┼──────────┼───────────┼───────────┼───────────┼───────────┤

 DSV3  D-b32  ▓ 0.7%
 Qwen3 D-b16  ▓ 0.6%
 DSV3  D-b16  ▓░ 1.1%
 DSV3  P-256  ▓░ 1.2%
 DSV3  P-1024 ▓░ 1.2%
 DSV3  D-b8   ▓░ 1.3%
 Qwen3 D-b32  ▓░ 1.6%
 Qwen3 D-b8   ▓░ 1.8%
 Qwen3 P-1024 ▓░ 1.8%
 Qwen3 P-256  ▓░░ 2.4%
 DSV3  D-b4   ▓░░░░ 4.2%
              ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  5% 阈值
 Qwen3 D-b4   ▓░░░░░ 5.9%                                    ⚠
 Qwen3 P-4096 ▓░░░░░░░░ 8.9%                                 ⚠
 DSV3  D-b1   ▓░░░░░░░░░░░ 11.2%                              ⚠
 Qwen3 D-b1   ▓░░░░░░░░░░░░░░░░░░░░░░ 22.7%                  ⚠
 DSV3  P-4096 ▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 31.0%         ✗ 异常
```

### 6.5 当前问题总结

#### 问题 1：Bench-Profiling 语义 Gap 导致简单校准不可行

```
 数据源对比:
 ┌─────────────────┬──────────────────────┬──────────────────────────────────┐
 │                 │ Bench (microbench)   │ Profiling (kernel_details)       │
 ├─────────────────┼──────────────────────┼──────────────────────────────────┤
 │ 测量内容        │ 纯 HCCL 通信时间     │ host dispatch → comm 结束        │
 │ 包含调度等待    │ ✗ 不包含             │ ✓ 包含 (pipeline bubble)         │
 │ 运行环境        │ 独立 torchrun        │ vLLM 推理 (compute+comm 交替)    │
 │ 预热            │ WARMUP=20            │ p10-p90 过滤                     │
 ├─────────────────┼──────────────────────┼──────────────────────────────────┤
 │ allGather ratio │ 1.8-3.6x            │ 基准                             │
 │ allReduce ratio │ 9.6-22.5x           │ 基准                             │
 │ reduceScatter   │ 5.4-12.7x           │ 基准                             │
 └─────────────────┴──────────────────────┴──────────────────────────────────┘

 → ratio 跨模型、跨 batch、跨算子均不稳定，无法用单一系数校准
```

#### 问题 2：Free Time 在小 batch 下不可忽略

```
 Free Time 构成分析:
 ┌──────────────────────────────────────────────────────────────────────┐
 │                                                                      │
 │  batch=1:  kernel 短 → launch overhead 占比高 → Free 13-23%         │
 │            每个 kernel ~10us, launch ~1us, 64 layers × 6 kernels    │
 │                                                                      │
 │  batch≥8:  kernel 长 → launch overhead 被摊薄 → Free <2%            │
 │            每个 kernel ~100us+, launch ~1us, 占比 <1%               │
 │                                                                      │
 │  DSV3 P-4096: 31% Free → 异常，疑似 MoE EP 协调 + DFC 内部等待     │
 │                                                                      │
 └──────────────────────────────────────────────────────────────────────┘
```

#### 问题 3：当前 inline profiled 方案的精度瓶颈

```
 E2E 误差来源分解:
 ┌──────────────────────────────────────────────────────────────────────┐
 │                                                                      │
 │  误差 = |预测 Stage - 实测 Stage| / 实测 Stage                      │
 │                                                                      │
 │  ┌─ 可消除 ─────────────────────────────────────────────────────┐   │
 │  │ ① comm 算子用固定 inline 值 vs 实际 per-config 值的偏差     │   │
 │  │   → per-config profiling CSV 可解决 (路径 1)                 │   │
 │  └──────────────────────────────────────────────────────────────┘   │
 │                                                                      │
 │  ┌─ 理论下限 (Free / Stage) ────────────────────────────────────┐   │
 │  │ ② Free time 不被任何算子 Duration 覆盖                       │   │
 │  │   batch≥8: <2% (可接受)                                      │   │
 │  │   batch=1: 13-23% (需显式建模, 路径 2)                       │   │
 │  └──────────────────────────────────────────────────────────────┘   │
 │                                                                      │
 │  当前 inline 误差 20-34% = ①误差 + ②误差                           │
 │  路径 1 消除 ① 后: 误差 ≈ ② ≈ Free/Stage (1-23%)                  │
 │                                                                      │
 └──────────────────────────────────────────────────────────────────────┘
```

### 6.6 通信查询策略

#### Bench + 简单校准方案验证结果

Bench CSV 测量纯通信时间，Profiling Duration 包含通信 + 调度等待。两者语义不同，gap 量化如下：

| 算子 | bench→profiling ratio | 稳定性 |
|------|----------------------|--------|
| allGather | 1.8-3.6x | 较稳定 |
| allReduce (Qwen3) | 9.6-22.5x | 不稳定 |
| reduceScatter (DSV3) | 5.4-12.7x | 不稳定 |

端到端误差对比：

| 策略 | Qwen3 误差 | DSV3 误差 |
|------|-----------|----------|
| Inline profiled（当前） | 20-32% | 9-34% |
| Bench CSV（raw） | 69-91% | 15-48% |
| Bench + 简单校准 | 58-91% | 12-41% |

**结论**："CSV first + bench_to_profiled_ratio" 方案不可行，ratio 跨模型、跨 batch 不稳定（9.6x-22.5x）。

#### 当前策略（维持 inline profiled 优先）

```
_lookup_comm() 查询优先级：
  ① communication_profiled（op_mapping.yaml 内联）→ confidence=0.85
  ② bench CSV（communication_data_ref 目录）→ confidence=0.9
  ③ CommAnalyticModel（最终 fallback）
```

#### 后续精度提升路径

##### 路径 1：per-config profiling CSV（高优先级）

从 profiling kernel_details 提取 per-(model, batch_size, kernel_type) 延迟，替代 bench CSV。Phase 2 的 16 组 trace 数据已有，需工具化入库。

**预期精度（已见配置精确匹配时，误差 ≈ Free / Stage）**：

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

**未见配置（插值）**：comm 延迟在 batch=8~32 范围内极稳定（allGather ~1060us，变化 <15%），线性插值预期误差 3-5%。

**结论**：生产主力场景（batch >= 8）E2E 误差从 10-20% 压到 1-2%，是当前最高性价比的精度提升路径。

##### 路径 2：显式建模调度开销（中优先级）

将 comm Duration 拆分为 `pure_comm + scheduling_wait`，scheduling_wait 与前序 compute kernel 执行时间相关，可从 trace_view 提取。主要改善 batch=1 场景（Free time 占 13-29%），对 batch >= 8 场景收益有限（Free < 2%）。
