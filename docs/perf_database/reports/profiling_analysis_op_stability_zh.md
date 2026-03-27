# NPU Profiling 分析报告 (一): 算子稳定性与 Perf-Database 接入可行性

**软件栈**: CANN 8.5 / vLLM 0.15.0 / PyTorch 2.9.0 / Eager 模式
**硬件**: Atlas 800 A3 (Ascend 910B)
**模型**: DeepSeek-V3 (W8A8, TP=8/DP=2/EP, 16 NPU) + Qwen3-32B (BF16, TP=16)
**数据路径**: `/mnt/d/Data/Profiling/dsv3_qwen3_full_torch2.9.0_vllm0.15.0_cann8.5_eager/`
**分析脚本**: `docs/perf_database/reports/profiling_analysis_eager/`

---

## 核心问题

对 profiling 中每个算子, 在相同 shape 下的执行时间是否稳定? 能否通过查表 (mean/P50) 来估算 kernel duration? 哪些算子需要特殊处理?

**分析方法**: 按算子类型 + input shape 分组, 计算组内 CV (Coefficient of Variation = std/mean). CV<0.1 认为稳定可查表; CV>0.1 需深入分析抖动来源.

---

## 一、全算子接入可行性总表

下表汇总了 **所有** profiling 中出现的算子, 按通信/通算融合/计算分类, 给出 perf-database 接入可行性评估.

### 1.1 通信算子 (不可查表, 需带宽建模)

| 算子 | 模型 | 占 e2e (代表性场景) | 平均耗时 | CV 范围 | 接入策略 |
|------|------|---------|---------|---------|---------|
| hcom_allReduce_ | Qwen3 | **75-97%** (batch↑则↑) | 2764 us | >0.5 | 带宽模型: `latency + bytes/BW` |
| hcom_reduceScatter_ | DSv3 | 与 allGather 合计 **17-56%** | 3208 us | 0.3-2.1 | 带宽模型 |
| hcom_allGather_ | DSv3/Qwen3 | (同上, 含 DP 通信) | 1336 us | 0.6-1.9 | 带宽模型 |
| RINGMLAPrefillBF16Kernel | DSv3 | <1% (仅 prefill) | 137 us | 0.01-0.05 | 带宽模型 (MLA Ring Attn) |
| reduce_scatterAicpuKernel | DSv3/Qwen3 | <1% (仅 prefill_4096) | 175-221 us | — | 带宽模型 (AICPU 路径) |
| allgatherAicpuKernel | DSv3/Qwen3 | <1% (仅 prefill_4096) | 183-263 us | — | 带宽模型 (AICPU 路径) |

> **通信算子 CV 天然 >0.3**, 受 HCCL 拓扑、网络竞争、同步等待影响, 不适合查表. communication.json 带宽字段全为 0, 需通过 HCCL dump 或 microbench 获取.
> 注: "占 e2e"基于 step_trace_time.csv 的正确分解, 而非 kernel_details.csv 简单求和. 详见姊妹报告《kernel duration 与 e2e 时间关系》.

### 1.2 通算融合算子 (不可直接查表)

| 算子 | 模型 | 占 e2e (代表性场景) | 平均耗时 | CV 范围 | 接入策略 |
|------|------|---------|---------|---------|---------|
| DispatchFFNCombine | DSv3 | **含在 Computing 内, 占 e2e 18-82%** | 5467 us | **0.14-2.1** | 分解为 sub_kernels 分别建模 |

> DispatchFFNCombine 内含 all-to-all + GroupedMatmul + SwiGlu, AIV time ≈ Duration (mac 仅 17us), 方差来源是内嵌通信和 MoE 路由不均. 极端 outlier 达 275ms (正常值 600 倍).
> 注: DispatchFFNCombine 在 profiling 中被归类在 compute stream, 因此 step_trace 的 "Computing" 时间包含了它.

### 1.3 计算算子 (占纯计算时间累计 >99%)

**DSv3 纯计算总时间: 1,033,857 us**
> 注: 下表"占计算%"是相对于纯计算算子总时间的占比. DSv3 的计算 stream 还包含 DispatchFFNCombine (通算融合), 因此 compute stream 总时间远大于此.

| 算子 | 占计算% | 累计% | N | 平均(us) | CV 范围 | 稳定性 | 接入策略 |
|------|---------|------|---|---------|---------|--------|---------|
| QuantBatchMatmulV3 | 39.4% | 39.4 | 15616 | 26.1 | **0.005-0.069** | 极稳定 | 直接查表取 mean |
| ~~Neg~~ | ~~8.9%~~ | — | 38 | 2414 | 0.79-1.73 | 异常 | **忽略** (warmup artifact, wait_time 占 99%) |
| AscendQuantV2 | 6.8% | 46.2 | 6954 | 10.1 | 0.007-0.126 | 稳定 | 查表取 mean |
| MatMulV2 | 6.7% | 52.9 | 4583 | 15.1 | 0.002-0.169 | 中等 | 查表取 P50 |
| AddRmsNormBias | 3.9% | 56.8 | 8662 | 4.7 | 0.030-0.094 | 稳定 | 直接查表 |
| DynamicQuant | 3.6% | 60.4 | 8662 | 4.3 | 0.002-0.152 | 稳定 | 查表取 mean |
| TransData | 3.4% | 63.8 | 2074 | 16.9 | **0.009** | 极稳定 | 直接查表 |
| FusedInferAttentionScore | 3.1% | 66.9 | 2074 | 15.5 | 0.030-**0.503** | ⚠️ batch=1 高CV | 查表取 P50, 需补采集 |
| MoeGatingTopK | 2.6% | 69.5 | 4118 | 6.6 | 0.090-0.126 | 中等 | 查表取 P50 |
| TransposeBatchMatMul | 2.5% | 72.0 | 2074 | 12.5 | 0.065-0.077 | 稳定 | 直接查表 |
| Slice | 2.3% | 74.3 | 5551 | 4.2 | 0.047-0.241 | 中等 | 查表取 P50 |
| InterleaveRope | 2.2% | 76.5 | 2318 | 9.9 | 0.019-0.045 | 稳定 | 直接查表 |
| Add | 2.1% | 78.6 | 6545 | 3.3 | 0.005-0.205 | 中等 | 查表取 mean |
| BatchMatMulV2 | 1.7% | 80.3 | 2074 | 8.4 | 0.025-0.033 | 极稳定 | 直接查表 |
| SwiGlu | 1.7% | 82.0 | 4331 | 4.0 | 0.002-0.127 | 稳定 | 直接查表 |
| Transpose | 1.4% | 83.4 | 868 | 16.8 | 0.013-0.251 | 中等 | 查表取 P50 |
| KvRmsNormRopeCache | 1.3% | 84.7 | 2318 | 5.9 | 0.038-0.099 | 稳定 | 直接查表 |
| RmsNorm | 1.3% | 86.0 | 2389 | 5.6 | 0.026-0.066 | 稳定 | 直接查表 |
| Muls | 1.0% | 87.0 | 4118 | 2.4 | 0.015-0.045 | 稳定 | 直接查表 |
| Cast | 0.9% | 87.9 | 7070 | 1.3 | 0.002-0.272 | 小算子 | zero_cost 或查表 |
| BroadcastTo | 0.5% | 88.4 | 509 | 9.2 | 0.055-0.133 | 中等 | 查表取 P50 |
| TensorMove | 0.4% | 88.8 | 492 | 8.8 | 0.080-0.231 | 中等 | 查表取 P50 |
| GreaterEqual | 0.4% | 89.2 | 109 | 36.8 | 0.005-**4.573** | ⚠️ 异常 | **忽略** (sampling artifact) |
| Fill | 0.4% | 89.6 | 2227 | 1.7 | 0.044-0.185 | 小算子 | zero_cost |
| AsStrided | 0.3% | 89.9 | 854 | 4.0 | 0.075-0.096 | 稳定 | 直接查表 |
| PagedCacheLoadNdKernel | 0.2% | 90.1 | 183 | 11.7 | 0.061-0.117 | 稳定 | 查表取 P50 |
| MaskedFill | 0.2% | 90.3 | 109 | 16.9 | 0.006-0.156 | 中等 | 查表取 P50 |
| 其余 18 个算子 | 0.8% | — | — | — | — | — | zero_cost |

**Qwen3-32B 纯计算总时间: 700,212 us**
> 注: Qwen3 无通算融合算子. 按 step_trace 分解, 计算占 e2e 的 2-7% (batch↑则↑), 通信占 75-97%.

| 算子 | 占计算% | 累计% | N | 平均(us) | CV 范围 | 稳定性 | 接入策略 |
|------|---------|------|---|---------|---------|--------|---------|
| MatMulV2 | 42.0% | 42.0 | 15549 | 18.9 | 0.002-**0.125** | 中等 | 查表取 P50 (mte2 方差) |
| FusedInferAttentionScore | 14.1% | 56.1 | 3904 | 25.2 | 0.019-**0.342** | ⚠️ 特定shape高CV | 查表取 P50, 需补采集 |
| RmsNorm | 13.3% | 69.4 | 7869 | 11.9 | 0.032-0.084 | 稳定 | 直接查表 |
| AddRmsNormBias | 6.8% | 76.3 | 7808 | 6.1 | 0.065-0.094 | 稳定 | 直接查表 |
| MatMulV3 | 5.1% | 81.3 | 128 | 277.7 | **0.008-0.022** | 极稳定 | 直接查表 (仅 prefill) |
| Slice | 4.1% | 85.4 | 11066 | 2.6 | 0.007-0.150 | 稳定 | 直接查表 |
| SwiGlu | 3.1% | 88.5 | 3904 | 5.6 | 0.038-0.120 | 稳定 | 直接查表 |
| _triton_rope | 3.1% | 91.6 | 3904 | 5.5 | 0.014-0.079 | 稳定 | 直接查表 |
| ReshapeAndCacheNdKernel | 2.1% | 93.8 | 3904 | 3.9 | 0.014-0.138 | 稳定 | 直接查表 |
| Sort | 1.9% | 95.7 | 61 | 223.7 | 0.003-0.013 | 极稳定 | 直接查表 |
| ArgMaxV2 | 0.8% | 96.5 | 61 | 88.2 | <0.01 | 极稳定 | 直接查表 |
| DSARandomUniform | 0.8% | 97.2 | 61 | 88.2 | — | — | 直接查表 |
| SoftmaxV2 | 0.5% | 97.7 | 61 | 52.3 | 0.006-0.034 | 极稳定 | 直接查表 |
| MaskedFill | 0.3% | 98.0 | 122 | 17.8 | 0.013-0.129 | 中等 | 查表取 P50 |
| ApplyTopKTopPCustom | 0.3% | 98.3 | 61 | 30.0 | 0.013-0.030 | 稳定 | 直接查表 |
| RealDiv | 0.2% | 98.5 | 122 | 12.4 | 0.000-0.140 | 中等 | 查表取 P50 |
| TensorMove | 0.2% | 98.7 | 128 | 10.0 | 0.114-0.117 | 中等 | 查表取 P50 |
| Cast | 0.2% | 98.8 | 488 | 2.4 | 0.002-0.130 | 小算子 | zero_cost |
| Index | 0.1% | 99.0 | 93 | 10.1 | 0.004-0.264 | 中等 | 查表取 P50 |
| Mul | 0.1% | 99.1 | 183 | 4.7 | 0.001-0.089 | 稳定 | 直接查表 |
| 其余 15 个算子 | 0.9% | — | — | — | — | — | zero_cost |

### 1.4 高方差 (CV>0.1) 计算算子汇总

| 算子 | 模型 | CV max | 占计算% | 方差来源 | 仿真影响 |
|------|------|--------|---------|----------|----------|
| **DispatchFFNCombine** | DSv3 | **2.1** | 42.6% 总 | 内嵌 all-to-all 通信 + MoE 路由不均 | **大** — 需分解建模 |
| **Neg** | DSv3 | 1.73 | 8.9% | wait_time (warmup artifact) | 无 — 忽略 |
| **GreaterEqual** | DSv3 | 4.57 | 0.4% | 极少调用 (24次), sampling artifact | 无 — 忽略 |
| **FusedInferAttentionScore** | DSv3 | 0.50 | 3.1% | batch=1 shape 不稳定 | 中等 — 需补长序列采集 |
| FusedInferAttentionScore | Qwen3 | 0.34 | 14.1% | batch=2 prefill 形状 | 中等 — 需补采集 |
| MatMulV2 | Qwen3 | 0.125 | 42.0% | **mte2 带宽竞争** (mac 恒定) | 低 — P50 可消除 |
| MatMulV2 | DSv3 | 0.169 | 6.7% | 大 shape mte2 | 低 — P50 可消除 |
| Transpose | DSv3 | 0.251 | 1.4% | mte 波动 | 低 |
| Slice | DSv3 | 0.241 | 2.3% | 大 seq 场景 | 低 |
| TensorMove | DSv3 | 0.231 | 0.4% | mte 波动 | 极低 |
| DynamicQuant | DSv3 | 0.152 | 3.6% | 偶发 outlier | 低 |
| AscendQuantV2 | DSv3 | 0.126 | 6.8% | scalar + mte2 | 低 (绝对值 ±1.2us) |

---

## 二、高方差算子抖动来源分层分析

对上表中 **仿真影响为"大"或"中等"** 的高方差算子, 从 vLLM 框架 → PyTorch → CANN → 昇腾微架构四层分析抖动根因, 并给出仿真建模建议.

### 2.1 DispatchFFNCombine (CV=0.14-2.1, DSv3)

**抖动来源分层:**

| 层级 | 抖动因素 | 影响程度 | 证据 |
|------|---------|---------|------|
| **vLLM 框架** | MoE routing 不均: TopK gating 动态选 expert, 不同 step 的 token→expert 分配不同 | **主要** | 同 shape 下 Duration 波动 10x+; AIV 时间 = Duration |
| **vLLM 框架** | all-to-all 通信量与 routing 相关: 不同 expert 分布导致跨节点搬运量不同 | **主要** | batch=4 outlier 275ms = 通信拥塞 |
| **CANN** | DispatchFFNCombine 是 CANN 超级融合 kernel, 内部调度不透明 | 中等 | 无法从 profiling 分解内部子操作时间 |
| **昇腾微架构** | AIV core 执行通信数据搬运时受 HCCL stream 竞争影响 | 中等 | mac=17us 恒定, AIV=8261us 波动 |

**仿真建模建议:**
- **不可直接查表**. 必须分解为 sub_kernels: `init_routing_v2 + GroupedMatmul×2 + GroupedMatmulSwigluQuant + unpermute_tokens + all_to_all×2`
- 若分解 profiling 数据不可得, 可对 DispatchFFNCombine 取 **P25** (≈纯计算下限), 通信部分用带宽模型叠加
- 长期: 用不带 FUSED_MC2 的 eager profiling 获取分解后各子 kernel 的独立数据

### 2.2 FusedInferAttentionScore (CV 最高 0.50/0.34)

**抖动来源分层:**

| 层级 | 抖动因素 | 影响程度 | 证据 |
|------|---------|---------|------|
| **vLLM 框架** | Decode bench `input_len=1` → KV cache 仅 3-4 tokens, 属于退化场景 | **主要** | batch=1 shape CV=0.50, 其他 shape CV<0.08 |
| **PyTorch** | PagedAttention KV cache block-table 寻址: 极短序列下 block-table overhead 比例大 | 中等 | scalar time 波动 |
| **CANN** | FusedInferAttention kernel 对极小 seq 有 tile 效率退化 | 中等 | mac=0, 主要在 scalar+AIV |
| **昇腾微架构** | AIC scalar core 控制逻辑 + AIV 并行执行的同步开销 | 低 | AIC scalar std ~1us |

**仿真建模建议:**
- 当前 CV=0.50 的 shape 是 `batch=1, KV_cache=891 blocks` 的退化场景, **不代表真实推理**
- **补充采集**: decode bench 改用 `input_len=512` 或 `input_len=2048`, 确保 KV cache 有合理深度
- 正常 shape (batch>2, 合理 KV depth) CV<0.08, **可查表取 P50**
- Prefill 场景 (M=3072) 的 CV=0.019, 非常稳定

### 2.3 MatMulV2 on Qwen3 (CV 最高 0.125)

**抖动来源分层:**

| 层级 | 抖动因素 | 影响程度 | 证据 |
|------|---------|---------|------|
| **vLLM 框架** | 无直接影响 — MatMul shape 由模型结构决定, 无动态变化 | 无 | — |
| **PyTorch** | 无 — eager 模式下 MatMul 直接调度到 CANN | 无 | — |
| **CANN** | ND→FRACTAL_NZ 格式转换可能影响 L1 cache 命中率 | 低 | TransData CV=0.009 说明转换本身稳定 |
| **昇腾微架构** | **mte2 (DDR→L1 读数据) 是唯一方差源**: mac std=0.00, mte2 CV=0.13 | **主要** | TP=16 下 16 NPU 共享 HBM 总线, 带宽竞争导致 mte2 波动 |

**仿真建模建议:**
- mac 时间完全确定 (std=0.00), **计算部分完全可查表**
- mte2 方差 ~13% 来自 HBM 带宽竞争, 是微架构层面的随机因素
- 建议取 **P50**, 误差 ±15% 在 e2e 层面影响有限 (MatMulV2 仅占 e2e 0.67%)
- TP 数减少 (如 TP=4) 可显著降低带宽竞争, 预期 CV 会下降

### 2.4 其他 CV>0.1 算子的共性规律

| 抖动模式 | 涉及算子 | 根因层级 | 仿真处理 |
|---------|---------|---------|---------|
| **mte2 带宽竞争** | MatMulV2, Transpose, TensorMove, Slice (大shape) | 昇腾微架构 | P50 查表; TP 数影响 |
| **scalar 控制逻辑波动** | AscendQuantV2, DynamicQuant | CANN kernel | mean 查表 (绝对值 <2us) |
| **warmup/调度 artifact** | Neg, GreaterEqual | 框架/OS | zero_cost 忽略 |
| **动态路由不均** | DispatchFFNCombine | vLLM MoE routing | 分解建模 |
| **退化场景** | FusedInferAttentionScore (batch=1) | bench 设计 | 补采集 |

---

## 三、Profiling 覆盖度与配置评估

### 3.1 覆盖度

| 维度 | DSv3 | Qwen3-32B | 评估 |
|------|------|-----------|------|
| Prefill | input=256/1024/4096 | input=256/1024/4096 | OK |
| Decode | batch=1/4/8/16/32 | batch=1/4/8/16/32 | OK |
| 量化 | W8A8 (ascend) | BF16 (无量化) | OK |
| 并行 | TP=8, DP=2, EP | TP=16 | 注意差异 |
| 模式 | enforce-eager | enforce-eager | OK |

### 3.2 需关注的配置问题

1. **DSv3 并行配置**: profiling 用 TP=8/DP=2/EP (16 NPU), 已验证生产配置为 TP=4/EP=8 (32 NPU). 计算 kernel 数据可通用, 但通信 pattern 和 expert 分布不同.
2. **Qwen3 TP=16 通信比例极高**: 32B 模型 16 卡并行, 通信占 e2e 的 75-97%, 不代表生产 (TP=4/8). 但纯计算 kernel 数据仍可用.
3. **Decode KV cache 退化**: `input_len=1, output_len=3` → KV cache 仅 3-4 tokens, FusedInferAttentionScore 不代表真实长序列.
4. **DSv3 max_num_batched_tokens=2048**: 限制了 prefill batch size, prefill 均为 batch=1.
5. **communication.json 带宽为零**: HCCL profiler 未采集 transit bandwidth.

---

## 四、采集补充建议

| 优先级 | 建议 | 目的 |
|--------|------|------|
| P0 | Decode bench 增加 `input_len=512/2048` | 修复 FusedInferAttentionScore KV cache 退化 |
| P0 | DSv3 关闭 FUSED_MC2 采集一次 | 获取 DispatchFFNCombine 分解后子 kernel 数据 |
| P1 | DSv3 补充 TP=4/EP=8 (32 NPU) 配置 | 对齐生产配置 |
| P1 | Qwen3 补充 TP=4 或 TP=8 | 降低通信比例, 提高计算算子代表性 |
| P2 | 通信 microbench 获取 HCCL 带宽 | communication.json 不可用 |

---

## 附录

### A. 数据文件

**Eager 模式** (`dsv3_qwen3_full_torch2.9.0_vllm0.15.0_cann8.5_eager/`):
每个场景包含: `kernel_details.csv` (逐 kernel 详情), `op_statistic.csv` (聚合统计), `step_trace_time.csv` (compute/comm/free 分解), `communication.json`, `trace_view.json`.

### B. 分析脚本

```bash
python3.10 docs/perf_database/reports/profiling_analysis/analyze_profiling.py    # 分类汇总
python3.10 docs/perf_database/reports/profiling_analysis/profiling_analysis.py   # CV/outlier 深度分析
```

### C. 相关报告

- 姊妹报告: [kernel duration 与 e2e 时间关系](profiling_analysis_kernel_vs_e2e_zh.md) — 分析 kernel_duration 加总与端到端时间的差距、CPU gap、compute/comm overlap
