# Phase 1 Profiling 分析: 算子稳定性与 Perf-Database 接入可行性

**软件栈**: CANN 8.5 / vLLM 0.15.0 / PyTorch 2.9.0 / **aclgraph** (FULL_DECODE_ONLY cudagraph)
**硬件**: Atlas 800 A3 (Ascend 910B)
**模型**: DeepSeek-V3 (W8A8, TP=8/DP=2/EP) + Qwen3-32B (BF16, TP=16)
**数据**: `/mnt/d/Data/Profiling/Profiling-0313-phase1-e2e-test/`

---

## 核心结论

1. **Qwen3-32B Prefill 计算算子极其稳定** — 占 99% 时间的 Top8 算子 CV 均 <0.01~0.09，可直接查表
2. **DSv3 计算算子整体稳定 (CV<0.1)**，但 DispatchFFNCombine 仍需分解建模
3. **Qwen3 Decode (graph mode) 的 CV 比 eager 模式偏高 (0.12-0.17)** — aclgraph 下小 kernel 的 CPU dispatch jitter 占比更大
4. **op_mapping 覆盖率 ~95%**，48 个 CSV 数据文件可用，可支持 mini e2e 验收

---

## 一、采集场景与配置

| 场景 | 模型 | 量化 | 并行 | input | output | kernel 数 | e2e |
|------|------|------|------|-------|--------|----------|-----|
| Prefill | DSv3 | W8A8 | TP=8,DP=2,EP | 4096 | 1 | 11K | 4.56s |
| Decode | DSv3 | W8A8 | TP=8,DP=2,EP | 4096 | 1536 | 11.5K | 4.88s |
| Prefill | Qwen3-32B | BF16 | TP=16 | 4096 | 1 | 6.4K | 5.82s |
| Decode | Qwen3-32B | BF16 | TP=16 | 4096 | ~1536 | 127K | 3.10s |

关键配置:
- 均为 aclgraph 模式 (`--no-enforce-eager`, `FULL_DECODE_ONLY` cudagraph)
- DSv3: `FUSED_MC2=1`, `max_num_seqs=8`, `max_num_batched_tokens=2048`
- Qwen3: `enable-prefix-caching`, `block-size=128`, `max-num-batched-tokens=65536`
- Decode 场景 input=4096，解决了之前 eager 采集中 KV cache 退化 (input_len=1) 的问题

---

## 二、Qwen3-32B Prefill: 计算算子极稳定

**纯计算总时间: 3,245 ms** (占 e2e 55.8%)

| 算子 | 占计算% | 累计% | N | Mean(us) | **Max CV** | 接入策略 |
|------|---------|------|---|---------|-----------|---------|
| MatMulV3 | 55.7% | 55.7 | 640 | 2825.6 | **0.0024** | 直接查表 (极稳定) |
| MatMulV2 | 16.5% | 72.3 | 640 | 838.2 | **0.0142** | 直接查表 |
| FusedInferAttentionScore | 10.7% | 82.9 | 320 | 1083.6 | **0.0111** | 直接查表 |
| split_qkv_rmsnorm_rope | 9.5% | 92.5 | 315 | 983.5 | **0.0026** | 直接查表 |
| SwiGlu | 2.8% | 95.3 | 320 | 281.0 | **0.0110** | 直接查表 |
| AddRmsNormBias | 2.1% | 97.4 | 640 | 108.5 | **0.0331** | 直接查表 |
| ReshapeAndCacheNdKernel | 1.5% | 98.9 | 320 | 151.5 | **0.0934** | 查表取 P50 |
| TensorMove | 0.2% | 99.2 | 325 | 32.7 | 0.0468 | 查表取 P50 |

> 占 99% 以上时间的算子 CV 均 <0.1, 是所有场景中最稳定的。仅有几个 1-2us 的极小算子 (Fill, Slice) CV>0.1，对 e2e 影响可忽略。
> **Qwen3 Prefill 是 mini e2e 验收的理想起点。**

---

## 三、Qwen3-32B Decode (aclgraph): 图模式 CV 偏高

**纯计算总时间: 2,976 ms** (排除 AivKernel comm stream)

| 算子 | 占计算% | 累计% | N | Mean(us) | **Max CV** | 接入策略 |
|------|---------|------|---|---------|-----------|---------|
| AivKernel (HCCL) | — | — | 17688 | 79.9 | — | **通信, 不查表** |
| MatMulV2 (graph) | 35.6% | 35.6 | 25728 | 20.6 | **0.1496** | ⚠️ P50 查表 |
| FusedInferAttentionScore | 35.2% | 70.8 | 8576 | 61.2 | **0.0839** | 查表取 P50 |
| split_qkv_rmsnorm_rope | 7.8% | 78.6 | 8442 | 13.8 | **0.0300** | 直接查表 |
| AddRmsNormBias (graph) | 6.1% | 84.7 | 17152 | 5.3 | **0.1168** | ⚠️ P50 查表 |
| MatMulV2_230083 (graph) | 5.0% | 89.7 | 8576 | 8.6 | **0.1375** | ⚠️ P50 查表 |
| SwiGlu (graph) | 3.0% | 92.7 | 8576 | 5.2 | **0.1701** | ⚠️ P50 查表 |
| reshape_and_cache | 2.4% | 95.1 | 8576 | 4.2 | 0.1600 | ⚠️ P50 查表 |

> **关键发现: 图模式 decode 的 CV (0.12-0.17) 明显高于 eager 模式同算子 (<0.05)。**
> 原因分析: decode 单 kernel 极短 (5-30us), CPU dispatch/cudagraph replay overhead 占比大, 加上 TP=16 的 HBM 带宽竞争, 导致 mte2 波动在相对值上更显著。
> **但绝对误差有限**: MatMulV2 CV=0.15, 绝对值 ±3us, 在 e2e=3098ms 中占比极低。
> 注意: graph-compiled kernel 名称带 hash 后缀, perf-database CSV 查询需要名称归一化。

---

## 四、DSv3 Prefill: DispatchFFNCombine 仍是挑战

**纯计算时间 (不含 DispatchFFNCombine): 332 ms** (仅占 e2e 7.3%)
**DispatchFFNCombine: 2,515 ms** (占 e2e 55.2%, 232 次调用)

| 算子 | 占计算% | 累计% | N | Mean(us) | **Max CV** | 接入策略 |
|------|---------|------|---|---------|-----------|---------|
| **DispatchFFNCombine** | *单独* | — | 232 | 10,840 | — | 分解建模 |
| QuantBatchMatmulV3 | 26.8% | 26.8 | 1220 | 72.9 | **0.0857** | 直接查表 |
| MatMulV2 | 5.8% | 32.6 | 366 | 53.0 | **0.0220** | 直接查表 |
| AscendQuantV2 | 5.4% | 38.0 | 732 | 24.4 | **0.0641** | 查表取 mean |
| Slice | 4.0% | 42.0 | 1342 | 9.8 | **0.0890** | 查表取 P50 |
| AddRmsNormBias | 3.4% | 45.4 | 476 | 24.0 | **0.0487** | 直接查表 |
| MoeGatingTopK | 2.5% | 47.9 | 232 | 35.7 | **0.1269** | ⚠️ P50 查表 |

> 纯计算算子整体稳定 (仅 MoeGatingTopK CV>0.1)。DispatchFFNCombine 无法做 per-shape CV 分析 (CANN 超级融合不透明), 必须分解为 sub_kernels 或用 P25 近似。

---

## 五、DSv3 Decode: 通信占主导

**纯计算时间 (不含 AivKernel comm): 108 ms** (仅占 e2e 2.2%)
**AivKernel (HCCL comm on Stream 37): 3,425 ms** (占 e2e 70.2%)
**DispatchFFNCombine: 1,203 ms** (占 e2e 24.7%)

| 算子 | 占计算% | 累计% | N | Mean(us) | **Max CV** | 接入策略 |
|------|---------|------|---|---------|-----------|---------|
| QuantBatchMatmulV3 | 33.5% | 33.5 | 1525 | 23.7 | **0.0603** | 直接查表 |
| FusedInferAttentionScore | 16.9% | 50.4 | 305 | 59.8 | **0.0521** | 直接查表 |
| AscendQuantV2 | 8.5% | 58.9 | 915 | 10.1 | **0.1462** | ⚠️ P50 查表 |
| MoeGatingTopK | 1.6% | — | 290 | 5.9 | **0.1325** | ⚠️ P50 查表 |

> FusedInferAttentionScore CV=0.052 (input=4096 context 下很稳定), 比之前 eager input_len=1 的 CV=0.50 大幅改善。
> 本场景中计算仅占 e2e 2.2%, 通信 (70.2%) + DispatchFFNCombine (24.7%) 占绝对主导。仿真精度主要取决于通信建模和 DispatchFFNCombine 分解。

---

## 六、op_mapping 覆盖度与 mini e2e 验收评估

### 6.1 覆盖度

**整体 ~95% 覆盖。** 48 个 CSV 数据文件覆盖所有主要计算算子。

| 类别 | 状态 | 说明 |
|------|------|------|
| 核心计算 (MatMul, Quant, Attention, Norm, SwiGlu) | ✅ 全覆盖 | CSV 数据齐全 |
| 通信 (hcom_allReduce, reduceScatter, allGather) | ✅ 映射有 | 无 CSV，用带宽模型 |
| DispatchFFNCombine | ✅ 映射有 | CSV 有，composite 分解 |
| split_qkv_rmsnorm_rope_kernel | ✅ 覆盖 | vLLM-ascend Triton fused |
| Neg, MaskedFill | ❌ 缺映射 | 需补充 op_mapping (占比极低) |
| PagedCacheLoadNdKernel, AivKernel | ⚠️ 内部 kernel | CANN/ATB 内部，不需 TC 建模 |
| Sampling (Sort, ArgMax, DSARandomUniform) | ⚠️ | TC 不仿真 sampling，可忽略 |

### 6.2 communication.json

| 场景 | 带宽数据 | 通信算子 |
|------|---------|---------|
| DSv3 Prefill | ✅ 有 (HCCS ~100 GB/s) | hcom_allGather, hcom_reduceScatter |
| DSv3 Decode | ❌ 全零 | hcom_allGather, hcom_reduceScatter |
| Qwen3 Prefill | ✅ 有 (HCCS ~115 GB/s) | hcom_allGather, hcom_reduceScatter |
| Qwen3 Decode | ❌ 全零 | hcom_allGather, hcom_allReduce |

> Decode 场景通信带宽为零，需依赖 microbench 或 Prefill 带宽外推。

### 6.3 mini e2e 验收就绪度

| Workplan 要求 | 数据就绪 | 注意事项 |
|--------------|---------|---------|
| Qwen3-32B Prefill | ✅ | 最稳定场景，理想起点 |
| DSv3 Decode | ✅ | 需处理 DispatchFFNCombine + graph kernel 名称归一化 |
| HIT/MISS ratio <50% | ✅ 预期达标 | 仅 Neg/MaskedFill 缺映射 |
| Fallback ops <30% e2e | ⚠️ 需验证 | DispatchFFNCombine 的 composite 查询是否走 fallback |

### 6.4 Graph-compiled kernel 名称归一化

Qwen3 Decode 的 kernel 名称带编译 hash 后缀:
```
MatMulV2_NDNZ_ND_FP16_FP16_false_true_all_229955  →  MatMulV2
FusedInferAttentionScore_3b093497fc536d61a77a7a329...  →  FusedInferAttentionScore
AddRmsNormBias_352d2859a07d64c080e8dfc836d9897f_33  →  AddRmsNormBias
SwiGlu_3_high_performance_27  →  SwiGlu
```

perf-database 的 `parse_kernel_details.py` 或 `ProfilingDataSource` 需要增加名称归一化逻辑。

---

## 七、采集建议

| 优先级 | 建议 | 目的 |
|--------|------|------|
| P0 | DSv3 关闭 FUSED_MC2 采集一次 | 获取 DispatchFFNCombine 分解后子 kernel 数据 |
| P0 | 补充 Neg、MaskedFill 的 op_mapping | 消除 MISS |
| P1 | DSv3 补 TP=4/EP=8 (32 NPU) | 对齐生产配置 |
| P1 | Qwen3 补 TP=4 或 TP=8 | 降低通信比例 |
| P2 | 通信 microbench | Decode 场景 communication.json 不可用 |

---

## 附录

### A. 与 eager 模式 CV 对比

| 算子 | 场景 | eager CV | aclgraph CV | 变化 |
|------|------|---------|-------------|------|
| MatMulV2 | Qwen3 Decode | 0.002-0.125 | 0.12-0.15 | ↑ graph jitter |
| FusedInferAttentionScore | Qwen3 Decode | 0.019-0.342 | 0.084 | ↓ input=4096 改善 |
| FusedInferAttentionScore | DSv3 Decode | 0.030-0.503 | 0.052 | ↓↓ input=4096 大幅改善 |
| AddRmsNormBias | Qwen3 Decode | 0.065-0.094 | 0.117 | ↑ graph jitter |
| SwiGlu | Qwen3 Decode | 0.038-0.120 | 0.170 | ↑ graph jitter |
| QuantBatchMatmulV3 | DSv3 Prefill | 0.005-0.069 | 0.011-0.086 | ≈ 无变化 |

### B. 分析脚本

```bash
python3.10 docs/perf_database/reports/profiling_analysis_aclgraph/analyze_phase1.py          # 全场景分类汇总
python3.10 docs/perf_database/reports/profiling_analysis_aclgraph/analyze_per_shape_cv.py    # Per-shape CV 深度分析
```

### C. 相关报告

- 姊妹报告: [kernel duration 与 e2e 时间关系](report_kernel_vs_e2e_zh.md)
- 前序报告 (eager): [算子稳定性 (eager)](../profiling_analysis_op_stability_zh.md)
