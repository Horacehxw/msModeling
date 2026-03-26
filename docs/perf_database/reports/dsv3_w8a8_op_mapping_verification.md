# DeepSeek-V3 W8A8 op_mapping 验证报告（C7）

**数据来源**: `kernel_details_deepseekv3-cann85.csv`
**硬件**: ATLAS_800_A3_752T_128G_DIE，32 卡，W8A8 aclgraph
**CANN 版本**: 8.5 / vLLM-ascend 0.15.0
**op_mapping 路径**: `tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5/op_mapping.yaml`
**负责人**: HDY
**日期**: 2026-03-11

---

## 统计概览

- **总耗时**: 2,398,057 us（2.398 s）
- **kernel Type 种数**: 43 种
- **Top-15 累计覆盖**: 99.13%
- **Top-15 中 TC 可模拟**: 12 种（排除 TransData/Transpose/DispatchFFNCombine 特殊处理）

---

## DSV3 Top-15 op_mapping 覆盖状态

| 排名 | kernel Type | 耗时占比 | TC op 映射 | 状态 | 备注 |
|------|------------|---------|-----------|------|------|
| 1 | DispatchFFNCombine | 35.29% | composite 分解（见下） | 特殊处理 | 超级融合 MoE op，TC 分解为多个 sub ops |
| 2 | hcom_reduceScatter_ | 27.92% | `tensor_cast.reduce_scatter.default` | 已配置（主） | C7 修正：设为主 kernel_type |
| 3 | hcom_allGather_ | 24.16% | `tensor_cast.all_gather.default` | 已配置 | — |
| 4 | QuantBatchMatmulV3 | 4.81% | `tensor_cast.static_quant_linear.default` | 已配置 | W8A8 主力 GEMM |
| 5 | AscendQuantV2 | 1.16% | `tensor_cast.quantize.default` | 已配置 | 静态量化 |
| 6 | FusedInferAttentionScore | 1.15% | `tensor_cast.attention_quant.default` | 已配置 | MLA attention，attention_special 查询 |
| 7 | TransData | 0.69% | `profiling.TransData` | TC 不模拟 | CANN 内部格式转换，不影响估算 |
| 8 | Transpose | 0.69% | `profiling.Transpose` | TC 不模拟 | 数据布局转换，不影响估算 |
| 9 | MatMulV2 | 0.59% | `aten.mm.default` | 已配置 | shared expert / lm_head BF16 matmul |
| 10 | TransposeBatchMatMul | 0.51% | `aten.bmm.default`（alternate） | 已配置 | MLA attn_output @ W_UV |
| 11 | BatchMatMulV2 | 0.34% | `aten.bmm.default` | 已配置 | MLA BMM（absorb projection） |
| 12 | InterleaveRope | 0.32% | `tensor_cast.apply_rope.default`（alternate） | 已配置 | DeepSeek interleave RoPE |
| 13 | AddRmsNormBias | 0.31% | `tensor_cast.add_rms_norm.default` | 已配置 | CANN 8.5 fused norm |
| 14 | DynamicQuant | 0.28% | `tensor_cast.dynamic_quantize_symmetric.default` | 已配置 | 动态量化 |
| 15 | MoeGatingTopK | 0.21% | `tensor_cast.moe_gating_topk.default` | 已配置 | MoE gating + top-k routing |

**Top-15 覆盖率**：13/15 有直接 TC op 映射（87%），2 种 TC 不模拟（TransData/Transpose，合计 1.38%）

---

## C7 修改记录

### 修改1：reduce_scatter kernel_type 主次关系调整

**问题**：原配置 `kernel_type: HcomReduceScatter`，但 DSV3 CANN 8.5 profiling 中实际出现的是 `hcom_reduceScatter_`（2082次，27.9%），HcomReduceScatter 未出现。

**修改**：
```yaml
# 修改前
"tensor_cast.reduce_scatter.default":
    kernel_type: HcomReduceScatter
    alternate_kernel_types: [hcom_reduceScatter_]

# 修改后
"tensor_cast.reduce_scatter.default":
    kernel_type: hcom_reduceScatter_
    alternate_kernel_types: [HcomReduceScatter]
```

**影响**：查询时优先使用 `hcom_reduceScatter_.csv`，与 DSV3 profiling 实测对齐。

### 修改2：DispatchFFNCombine notes 补充

在 `profiling.DispatchFFNCombine` 中补充了查询策略说明和 Phase 2 建议（见下节）。

---

## DispatchFFNCombine 处理策略（重点）

### 现状

DispatchFFNCombine 是 DSV3 最重要的单一 kernel（35.3%），融合了：
- `InitRouting + DispatchV2`（all_to_all dispatch）
- `2×GroupedMatmul`（gate-up proj + down proj）
- `SwiGlu`
- `CombineV2 + Unpermute`（all_to_all combine）

TC 将其分解为独立 ops：
```
permute_tokens → all_to_all → grouped_matmul_quant_swiglu → grouped_matmul_quant → all_to_all → unpermute_tokens
```

### 当前 op_mapping 配置

多个 TC ops 设置了 `alternate_kernel_types: [DispatchFFNCombine]`：
- `tensor_cast.grouped_matmul_quant.default`
- `tensor_cast.grouped_matmul_quant_swiglu.default`
- `tensor_cast.permute_tokens.default`
- `tensor_cast.unpermute_tokens.default`

### 已知问题

若 GroupedMatmul/MoeDistributeDispatchV2 等 CSV 不存在，上述 4 个 TC ops 均会 fallback 到 `DispatchFFNCombine.csv`，导致 DispatchFFNCombine 耗时被计算 4 次（重复计数）。

### Phase 2 建议（H1/E3 阶段）

1. 为 DispatchFFNCombine 单独采集 microbenchmark（`generate_microbench.py` 支持 `torch.ops._C_ascend.dispatch_ffn_combine`）
2. 仅保留 `grouped_matmul_quant_swiglu` 的 DispatchFFNCombine alternate（它是最主要的计算部分）
3. 其余 sub ops（permute_tokens, unpermute_tokens, grouped_matmul_quant）移除 DispatchFFNCombine alternate，改用 analytic fallback
4. 或者：为 DispatchFFNCombine 添加 composite 分解比例配置（需要 profiling 数据支撑）

---

## 额外覆盖（Top-15 之外）

| kernel Type | 耗时占比 | TC op 映射 | 状态 |
|------------|---------|-----------|------|
| KvRmsNormRopeCache | 0.20% | `tensor_cast.kv_rmsnorm_rope_cache.default` | 已配置（MLA 专用） |
| Add | 0.18% | `aten.add.Tensor` | 已配置 |
| RmsNorm | 0.17% | `tensor_cast.rms_norm.default` | 已配置 |
| SwiGlu | 0.14% | `tensor_cast.swiglu.default` | 已配置 |
| Cast | 0.12% | `aten.to.dtype` | 已配置 |
| AddRmsNormDynamicQuant | 0.01% | `tensor_cast.add_rms_norm_dynamic_quant_symmetric.default` | 已配置 |

---

## 覆盖率汇总

| 类别 | kernel 数 | 耗时占比 | 状态 |
|------|----------|---------|------|
| 已直接映射（TC op → kernel_type） | 12 | 61.8% | 可查询 |
| DispatchFFNCombine（composite 分解） | 1 | 35.3% | 特殊处理，Phase 2 优化 |
| TC 不模拟（TransData/Transpose/AsStrided） | 3 | 1.5% | 不影响估算 |
| **合计 Top-15** | **15** | **99.1%** | — |

**结论**：DSV3 Top-15 op_mapping 覆盖完整。主要风险是 DispatchFFNCombine 的重复计数问题，需在 Phase 2 解决。

---

## 注意事项（C8 验证重点）

1. **QuantBatchMatmulV3 shape 差异**：
   - Profiling 输入：`(1,7168)` × `(66,448,16,32)` — 第二个输入为 FRACTAL_NZ 格式
   - TC 输出：`(M, K)` × `(K, N)` — ND 格式
   - 需确认 `profiling_data_source.py` 的 `fractal_nz_to_nd()` 能正确处理 W8A8 量化矩阵

2. **FusedInferAttentionScore（MLA）**：
   - Profiling 输入：`(4,16,1,512)` Q + `(892,1,128,512)` K + `(892,1,128,512)` V
   - MLA 的 KV 维度与标准 MHA 不同（head_dim=512 for KV, 64 for Q output）
   - 需确认 attention_special 查询路径能正确处理 MLA 的非对称 head_dim

3. **InterleaveRope shape**：
   - Profiling：`(4,16,1,64)` — 格式为 `(batch, heads, seq, head_dim/2)`
   - TC apply_rope：`(batch, heads, seq, head_dim)` — 需确认 alternate 查询时 shape 匹配规则

4. **hcom_reduceScatter_ CSV 文件**：
   - 需确认 HCCL 数据目录中存在 `hcom_reduceScatter_.csv`（C10 采集产物）
   - 文件名大小写敏感，需与 kernel_type 完全一致
