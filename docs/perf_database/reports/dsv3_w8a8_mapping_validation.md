# DeepSeek-V3 W8A8 量化场景映射验证报告（C8）

**数据来源**: `kernel_details_deepseekv3-cann85.csv`
**硬件**: ATLAS_800_A3_752T_128G_DIE，32 卡，W8A8 aclgraph，TP=4 EP=8
**CANN 版本**: 8.5 / vLLM-ascend 0.15.0
**负责人**: HDY
**日期**: 2026-03-11
**前置**: C7 op_mapping 扩展报告（`dsv3_w8a8_op_mapping_verification.md`）

---

## 一、覆盖率汇总

| 类别 | 耗时 (us) | 占比 | 说明 |
|------|----------|------|------|
| 计算算子（直接映射） | 255,339 | 10.65% | TC op → kernel_type CSV 查询 |
| 通信算子（直接映射） | 1,248,854 | 52.08% | hcom_reduceScatter_ + hcom_allGather_ |
| DispatchFFNCombine（composite 分解） | 846,380 | 35.29% | TC 分解为 sub ops，Phase 2 优化 |
| TC 不模拟 | 47,483 | 1.98% | TransData/Transpose/Sampling 等 |
| **总覆盖** | **2,350,574** | **98.02%** | — |

**结论**：op_mapping 覆盖端到端耗时 98.02%，满足 >90% 目标。

---

## 二、W8A8 量化算子逐条验证

### 2.1 QuantBatchMatmulV3（4.81%，4880 次）

**TC op**：`tensor_cast.static_quant_linear.default`
**映射路径**：`aclnnWeightQuantBatchMatmulV2/V3` → `QuantBatchMatmulV3`

**Profiling 输入 shape（FRACTAL_NZ 解码后）**：

| 调用次数 | 激活 (M,K) | 权重 ND (K,N) | 平均耗时 | 层推断 |
|---------|-----------|-------------|---------|--------|
| 928 | (1, 7168) | (7168, 4096) | 32.39 us | shared expert gate+up 或 W_O（TP=4） |
| 976 | (1, 7168) | (7168, 2112) | 28.05 us | 待 TC trace 确认（N=2112 非标准维度） |
| 976 | (8, 2048) | (2048, 7168) | 21.62 us | 路由专家 down proj（M=top_k=8） |
| 928 | (1, 2048) | (2048, 7168) | 20.29 us | 路由专家 down proj decode（M=1） |
| 976 | (4, 1536) | (1536, 3072) | 15.72 us | MLA Q up proj（K=q_lora_rank=1536，N=3072 待确认） |
| 48 | (8, 7168) | (7168, 4608) | 34.95 us | Prefill 场景（M=8） |
| 48 | (8, 2304) | (2304, 7168) | 22.07 us | Prefill 场景 |

**FRACTAL_NZ 解码规则验证**：
- 格式 `(a, b, bh, bw)` → ND `(K=b×bh, N=a×bw)`
- 示例：`(66, 448, 16, 32)` → K=448×16=7168, N=66×32=2112 ✓
- 示例：`(128, 448, 16, 32)` → K=448×16=7168, N=128×32=4096 ✓

**验证状态**：
- FRACTAL_NZ 解码规则已确认，`profiling_data_source.py` 的 `fractal_nz_to_nd()` 适用
- N=2112 和 N=3072 的层映射需 TC dispatch trace 确认（C8 遗留项）
- 主要 shape（7168×4096, 2048×7168）与 DSV3 架构吻合

### 2.2 AscendQuantV2（1.16%，2928 次）

**TC op**：`tensor_cast.quantize.default`
**映射路径**：`aclnnAscendQuant/V3` → `AscendQuantV2`

**Profiling 输入 shape**：

| 调用次数 | 输入 (M, K) | 平均耗时 | 层推断 |
|---------|-----------|---------|--------|
| 976 | (1, 7168) | 9.47 us | 激活量化（hidden=7168） |
| 976 | (4, 1536) | — | MLA Q lora 激活量化 |
| 976 | (8, 2048) | — | 专家激活量化 |

**验证状态**：
- shape 格式为 ND，无 FRACTAL_NZ 转换，TC 直接匹配 ✓
- 3 种 shape 对应 3 类量化点（hidden/q_lora/expert），覆盖完整

### 2.3 DynamicQuant（0.28%，1904 次）

**TC op**：`tensor_cast.dynamic_quantize_symmetric.default`
**映射路径**：`aclnnDynamicQuant/V2` → `DynamicQuant`

**Profiling 输入 shape**：

| 调用次数 | 输入 (M, K) | 层推断 |
|---------|-----------|--------|
| 928 | (1, 7168) | 动态量化（hidden） |
| 928 | (1, 2048) | 动态量化（expert intermediate） |
| 48 | (8, 2304) | Prefill 场景 |

**验证状态**：shape 为 ND，TC 直接匹配 ✓

### 2.4 AddRmsNormDynamicQuant（0.01%，48 次）

**TC op**：`tensor_cast.add_rms_norm_dynamic_quant_symmetric.default`
**映射路径**：`aclnnAddRmsNormDynamicQuantV2` → `AddRmsNormDynamicQuant`

**Profiling 输入 shape**：`(1,7168)+(1,7168)+(7168)+(7168)` — 4 个 ND 输入 ✓

**验证状态**：shape 匹配，调用次数少（48次），耗时占比 0.01%，低优先级

---

## 三、MLA 专用算子验证

### 3.1 FusedInferAttentionScore（1.15%，976 次）

**TC op**：`tensor_cast.attention_quant.default`（`query_mode: attention_special`）

**Profiling 输入 shape**（唯一 shape，976 次）：
```
Q:  (4, 16, 1, 512)   — batch=4, heads=16/TP, seq=1, head_dim=512
K:  (892, 1, 128, 512) — kv_seq=892, 1, kv_heads=128, head_dim=512
V:  (892, 1, 128, 512) — 同 K
...（后续为 mask/scale 等可选参数）
output_Q: (4, 16, 1, 64) — batch=4, heads=16/TP, seq=1, qk_rope_head_dim=64
output_K: (892, 1, 128, 64)
```

**MLA 特殊性**：
- Q head_dim=512（非标准，= kv_lora_rank=512）
- K/V head_dim=512（kv_lora_rank，非 v_head_dim=128）
- 输出 Q/K head_dim=64（qk_rope_head_dim）
- 这是 MLA absorb 后的 attention，与标准 MHA 维度完全不同

**验证状态**：
- `attention_special` 查询路径需正确处理非对称 head_dim（Q=512, output=64）
- FusedInferAttentionScore.csv 中的 shape 索引维度需包含 MLA 的 kv_lora_rank
- **风险**：若 CSV 仅有标准 MHA shape，MLA 场景会 MISS → fallback analytic

### 3.2 TransposeBatchMatMul（0.51%，976 次）

**TC op**：`aten.bmm.default`（alternate: TransposeBatchMatMul）

**Profiling shape**：`(16, 4, 512)` × `(16, 512, 128)` → output `(16, 4, 128)`
- 解读：16 heads, batch=4, K=512(kv_lora_rank), N=128(v_head_dim)
- 对应 MLA 中 attn_output @ W_UV（absorb projection）

**验证状态**：
- shape 为 ND，无格式转换
- alternate 查询：先查 BatchMatMulV2.csv，若 shape 不匹配再查 TransposeBatchMatMul.csv
- 需确认 TransposeBatchMatMul.csv 中有 `(16,4,512)×(16,512,128)` 的数据

### 3.3 BatchMatMulV2（0.34%，976 次）

**TC op**：`aten.bmm.default`

**Profiling shape**：`(16, 4, 128)` × `(16, 128, 512)` → output `(16, 4, 512)`
- 解读：16 heads, batch=4, K=128(v_head_dim), N=512(kv_lora_rank)
- 对应 MLA 中 absorb projection（BMM 部分）

**验证状态**：shape 为 ND，TC 直接匹配 ✓

### 3.4 InterleaveRope（0.32%，976 次）

**TC op**：`tensor_cast.apply_rope.default`（alternate: InterleaveRope）

**Profiling shape**：`(4, 16, 1, 64)` × `(4, 1, 1, 64)` × `(4, 1, 1, 64)`
- 格式：`(batch, heads/TP, seq, qk_rope_head_dim/2)`
- 注意：head_dim=64 = qk_rope_head_dim，非完整 head_dim

**验证状态**：
- TC apply_rope 输出 shape 为 `(batch, heads, seq, head_dim)`，与 profiling 的 `head_dim=64` 不同
- alternate 查询时需确认 shape 匹配规则能处理 rope_head_dim 维度差异
- **风险**：shape 不匹配可能导致 MISS

### 3.5 KvRmsNormRopeCache（0.20%，976 次）

**TC op**：`tensor_cast.kv_rmsnorm_rope_cache.default`

**Profiling shape**：
```
input:  (4, 1, 1, 576)  — batch=4, 1, seq=1, kv_lora_rank+rope_dim=576
norm_w: (512,)           — kv_lora_rank=512
q_out:  (4, 1, 1, 64)   — qk_rope_head_dim=64
k_out:  (4, 1, 1, 64)
seq_len: (4,)
kv_cache: (892, 128, 1, 64)  — kv_seq, kv_heads, 1, rope_dim
kv_cache2: (892, 128, 1, 512) — kv_seq, kv_heads, 1, kv_lora_rank
```

**验证状态**：
- 输入 576 = kv_lora_rank(512) + qk_rope_head_dim(64) ✓
- shape 为 ND，TC 直接匹配
- KvRmsNormRopeCache.csv 需包含此 shape 的数据

---

## 四、通信算子验证

### 4.1 hcom_reduceScatter_（27.92%，2082 次）

**TC op**：`tensor_cast.reduce_scatter.default`（C7 修正为主 kernel_type）

**Profiling shape**：N/A（HCCL 通信算子无 shape 记录）
**平均耗时**：321.59 us/次

**验证状态**：
- kernel_type 已修正为 hcom_reduceScatter_（C7）
- 需 C10 采集 `hcom_reduceScatter_.csv`（message_bytes × topology_tier 网格）
- 通信查询路径：`_lookup_comm()` 按 message_bytes + topology_tier 查询

### 4.2 hcom_allGather_（24.16%，4164 次）

**TC op**：`tensor_cast.all_gather.default`

**平均耗时**：139.12 us/次（约为 reduce_scatter 的 43%，符合 allGather 数据量更小的预期）

**验证状态**：已配置，需 C10 采集 `hcom_allGather_.csv`

---

## 五、DispatchFFNCombine 验证（重点风险）

**耗时**：846,380 us（35.29%，平均 912 us/次）

**当前处理**：TC 将其分解为 sub ops，各 sub op 通过 alternate_kernel_types 回退到 DispatchFFNCombine.csv

**已知风险**：
- 若 GroupedMatmul/MoeDistributeDispatchV2 等 CSV 不存在，4 个 TC ops 均查 DispatchFFNCombine.csv → 重复计数 4×
- 若 DispatchFFNCombine.csv 也不存在 → 全部 fallback analytic → 35.3% 耗时估算不准

**Phase 2 行动项**（H1/E3）：
1. 为 DispatchFFNCombine 采集 microbenchmark（`torch.ops._C_ascend.dispatch_ffn_combine`）
2. 仅保留 `grouped_matmul_quant_swiglu` 的 DispatchFFNCombine alternate
3. 其余 sub ops 移除 alternate，改用 analytic fallback

---

## 六、验证结论与遗留项

### 已验证（可直接使用）

| kernel Type | 验证状态 | 置信度 |
|------------|---------|--------|
| QuantBatchMatmulV3 | FRACTAL_NZ 解码规则确认 | 高 |
| AscendQuantV2 | ND shape，直接匹配 | 高 |
| DynamicQuant | ND shape，直接匹配 | 高 |
| AddRmsNormBias | ND shape，直接匹配 | 高 |
| BatchMatMulV2 | ND shape，直接匹配 | 高 |
| KvRmsNormRopeCache | ND shape，576=512+64 确认 | 高 |
| hcom_reduceScatter_ | kernel_type 已修正（C7） | 高 |
| hcom_allGather_ | 已配置 | 高 |

### 遗留验证项（需 TC dispatch trace）

| 项目 | 风险等级 | 说明 |
|------|---------|------|
| QuantBatchMatmulV3 N=2112/3072 层映射 | 低 | 不影响 CSV 查询，仅影响 shape 理解 |
| FusedInferAttentionScore MLA shape | 高 | CSV 需包含 MLA 的 kv_lora_rank=512 维度 |
| InterleaveRope head_dim=64 匹配 | 中 | alternate 查询时 shape 差异可能导致 MISS |
| TransposeBatchMatMul CSV 数据 | 中 | 需确认 CSV 中有 (16,4,512)×(16,512,128) |
| DispatchFFNCombine 重复计数 | 高 | Phase 2 必须解决，否则 MoE 耗时估算偏高 4× |

### 下一步行动

1. **C10**（3.13）：采集 `hcom_reduceScatter_.csv` 和 `hcom_allGather_.csv`
2. **Phase 2 H1**（3.16）：TC dispatch trace 确认 QuantBatchMatmulV3 层映射
3. **Phase 2 E3**（3.19）：DispatchFFNCombine microbenchmark 采集 + alternate 策略修正
4. **Phase 2 E4**（3.20）：FusedInferAttentionScore MLA shape microbenchmark 补充
