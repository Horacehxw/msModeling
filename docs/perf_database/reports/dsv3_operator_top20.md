# DeepSeek-V3 算子耗时 Top-20 清单

**数据来源**: `kernel_details_deepseekv3-cann85.csv`
**硬件**: ATLAS_800_A3_752T_128G_DIE，32 卡，W8A8 aclgraph
**CANN 版本**: 8.5
**负责人**: HDY
**日期**: 2026-03-10

---

## 统计概览

- **总耗时**: 2,398,057 us（2.398 s）
- **kernel Type 种数**: 43 种
- **Top-20 累计覆盖**: 99.30%

---

## Top-20 算子耗时排名

| 排名 | kernel Type | 调用次数 | 总耗时 (us) | 占比 | 累计占比 | 类别 | 备注 |
|------|------------|--------:|----------:|-----:|--------:|------|------|
| 1 | DispatchFFNCombine | 928 | 846,381 | 35.29% | 35.29% | 计算+通信融合 | MoE 超级融合 op：all_to_all×2 + GMM×2 + SwiGlu + routing |
| 2 | hcom_reduceScatter_ | 2,082 | 669,552 | 27.92% | 63.22% | 通信 | TP reduce-scatter |
| 3 | hcom_allGather_ | 4,164 | 579,302 | 24.16% | 87.37% | 通信 | TP all-gather |
| 4 | QuantBatchMatmulV3 | 4,880 | 115,447 | 4.81% | 92.19% | 计算 | W8A8 量化矩阵乘（主力 GEMM） |
| 5 | AscendQuantV2 | 2,928 | 27,732 | 1.16% | 93.34% | 计算 | 静态量化 |
| 6 | FusedInferAttentionScore | 976 | 27,542 | 1.15% | 94.49% | 计算 | MLA attention |
| 7 | TransData | 976 | 16,609 | 0.69% | 95.18% | 格式转换 | ND ↔ FRACTAL_NZ，TC 不模拟 |
| 8 | Transpose | 992 | 16,572 | 0.69% | 95.87% | 计算 | 数据布局转换 |
| 9 | MatMulV2 | 944 | 14,172 | 0.59% | 96.47% | 计算 | BF16 矩阵乘（shared expert / lm_head） |
| 10 | TransposeBatchMatMul | 976 | 12,226 | 0.51% | 96.98% | 计算 | MLA attn_output @ W_UV |
| 11 | BatchMatMulV2 | 976 | 8,272 | 0.34% | 97.32% | 计算 | MLA BMM（absorb projection） |
| 12 | InterleaveRope | 976 | 7,669 | 0.32% | 97.64% | 计算 | DeepSeek interleave RoPE |
| 13 | AddRmsNormBias | 1,904 | 7,410 | 0.31% | 97.95% | 计算 | Add + RmsNorm 融合（CANN 8.5） |
| 14 | DynamicQuant | 1,904 | 6,608 | 0.28% | 98.23% | 计算 | 动态量化 |
| 15 | MoeGatingTopK | 928 | 5,109 | 0.21% | 98.44% | 计算 | MoE gating + top-k routing |
| 16 | KvRmsNormRopeCache | 976 | 4,832 | 0.20% | 98.64% | 计算 | MLA KV norm + RoPE + cache 融合 |
| 17 | Add | 1,936 | 4,410 | 0.18% | 98.82% | 计算 | 残差加法 |
| 18 | RmsNorm | 992 | 4,012 | 0.17% | 98.99% | 计算 | 独立 RmsNorm（未融合层） |
| 19 | AsStrided | 976 | 3,871 | 0.16% | 99.15% | 内存 | stride 操作，TC 不模拟 |
| 20 | SwiGlu | 976 | 3,443 | 0.14% | 99.30% | 计算 | shared expert SwiGlu 激活 |

---

## 关键观察

### 耗时分布

- **通信主导**：hcom_reduceScatter_ + hcom_allGather_ 合计占 **52.1%**，TP 通信是最大瓶颈
- **DispatchFFNCombine 异常高**：单个融合 kernel 占 **35.3%**，是 DSV3 MoE EP 路由 + FFN 的全部耗时
- **三者合计 87.4%**：前 3 名已覆盖绝大部分耗时，其余 40 种 kernel 合计仅 12.6%

### 计算+通信融合算子确认

| 融合类型 | kernel Type | 是否存在 | 处理方式 |
|---------|------------|---------|---------|
| MC2（MatMul + AllReduce） | 无专用 kernel | 否 | `composite` 分解：`QuantBatchMatmulV3` + `hcom_allReduce_` |
| MoE EP 融合 | DispatchFFNCombine | **是**（35.3%） | `composite` 分解：`init_routing_v2 + grouped_matmul×2 + swiglu + unpermute_tokens + all_to_all×2` |

### op_mapping 覆盖状态（C7 参考）

| kernel Type | TC op 映射 | 状态 |
|------------|-----------|------|
| DispatchFFNCombine | `tensor_cast.grouped_matmul_quant_swiglu` + `init_routing_v2/unpermute_tokens` | composite，已配置 |
| hcom_reduceScatter_ | `tensor_cast.reduce_scatter` | 已配置 |
| hcom_allGather_ | `tensor_cast.all_gather` | 已配置 |
| QuantBatchMatmulV3 | `tensor_cast.static_quant_linear` | 已配置 |
| AscendQuantV2 | `tensor_cast.quantize` | 已配置 |
| FusedInferAttentionScore | `tensor_cast.attention_quant` | 已配置（attention_special） |
| TransData | — | TC 不模拟，CANN 内部格式转换 |
| Transpose | — | TC 不模拟 |
| MatMulV2 | `aten.mm.default` | 已配置 |
| TransposeBatchMatMul | `aten.bmm.default` | alternate，已配置 |
| BatchMatMulV2 | `aten.bmm.default` | 已配置 |
| InterleaveRope | `tensor_cast.apply_rope` | alternate，已配置 |
| AddRmsNormBias | `tensor_cast.add_rms_norm` | 已配置（CANN 8.5） |
| DynamicQuant | `tensor_cast.dynamic_quantize_*` | 已配置 |
| MoeGatingTopK | `tensor_cast.moe_gating_top_k_softmax` | 已配置 |
| KvRmsNormRopeCache | `tensor_cast.kv_rmsnorm_rope_cache` | 已配置 |
| Add | `aten.add.Tensor` | 已配置 |
| RmsNorm | `tensor_cast.rms_norm` | 已配置 |
| AsStrided | — | TC 不模拟 |
| SwiGlu | `tensor_cast.swiglu` | 已配置 |
