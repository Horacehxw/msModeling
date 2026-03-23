# Qwen3-32B 算子耗时 Top-20 清单

**数据来源**: `kernel_details_qwen3-32b_cann85.csv`
**硬件**: ATLAS_800_A3_752T_128G_DIE，16 卡，BF16 aclgraph
**CANN 版本**: 8.5
**负责人**: ZZY
**日期**: 2026-03-10

---

## 统计概览

- **总耗时**: 4,461,888 us（4.462 s）
- **kernel Type 种数**: 38 种
- **总 kernel 调用次数**: 28,641
- **Top-20 累计覆盖**: 99.76%

---

## Top-20 算子耗时排名

| 排名 | kernel Type | 调用次数 | 总耗时 (us) | 占比 | 累计占比 | 类别 | 备注 |
|------|------------|--------:|----------:|-----:|--------:|------|------|
| 1 | hcom_allReduce_ | 7,748 | 3,755,792 | 84.17% | 84.17% | 通信 | TP all-reduce，绝对主导 |
| 2 | MatMulV2 | 7,710 | 245,135 | 5.49% | 89.67% | 计算 | BF16 矩阵乘（主力 GEMM） |
| 3 | FusedInferAttentionScore | 1,920 | 118,485 | 2.66% | 92.32% | 计算 | Prefill + Decode attention |
| 4 | AddRmsNormBias | 3,840 | 70,568 | 1.58% | 93.91% | 计算 | Add + RmsNorm 融合（CANN 8.5） |
| 5 | DSARandomUniform | 30 | 54,614 | 1.22% | 95.13% | 其他 | 随机数生成，sampling pipeline，TC 不模拟 |
| 6 | Sort | 30 | 47,139 | 1.06% | 96.19% | 其他 | Sampling top-k sort，TC 不模拟 |
| 7 | SwiGlu | 1,920 | 33,273 | 0.75% | 96.93% | 计算 | SwiGlu 激活 |
| 8 | split_qkv_rmsnorm_rope_kernel | 1,890 | 31,102 | 0.70% | 97.63% | 计算 | Triton 融合：QKV split + RmsNorm + RoPE（63/64 层） |
| 9 | hcom_allGather_ | 30 | 27,921 | 0.63% | 98.26% | 通信 | all-gather（sampling 阶段） |
| 10 | ReshapeAndCacheNdKernel | 1,920 | 23,929 | 0.54% | 98.79% | 计算 | KV cache 写入 |
| 11 | allgatherAicpuKernel | 30 | 10,754 | 0.24% | 99.03% | 通信 | AICPU all-gather 变体（graph 编译路径） |
| 12 | SoftmaxV2 | 30 | 6,273 | 0.14% | 99.17% | 其他 | Sampling softmax，TC 不模拟 |
| 13 | RealDiv | 60 | 4,486 | 0.10% | 99.27% | 计算 | 除法 |
| 14 | MaskedFill | 60 | 4,387 | 0.10% | 99.37% | 其他 | Sampling mask，TC 不模拟 |
| 15 | ArgMaxV2 | 30 | 4,281 | 0.10% | 99.47% | 其他 | Sampling argmax，TC 不模拟 |
| 16 | Neg | 30 | 4,015 | 0.09% | 99.56% | 其他 | Sampling 内部，TC 不模拟 |
| 17 | ApplyTopKTopPCustom | 30 | 2,766 | 0.06% | 99.62% | 其他 | vllm-ascend 自定义 sampling op，TC 不模拟 |
| 18 | Add | 30 | 2,412 | 0.05% | 99.67% | 计算 | 残差加法 |
| 19 | GreaterEqual | 60 | 2,094 | 0.05% | 99.72% | 其他 | Sampling 比较，TC 不模拟 |
| 20 | Mul | 90 | 1,942 | 0.04% | 99.76% | 计算 | 标量乘法 |

---

## 关键观察

### 耗时分布

- **通信绝对主导**：`hcom_allReduce_` 单项占 **84.2%**，是 Qwen3-32B BF16 TP 推理的核心瓶颈
- **计算侧集中**：`MatMulV2`（5.5%）+ `FusedInferAttentionScore`（2.7%）+ `AddRmsNormBias`（1.6%）合计 9.8%
- **Sampling 开销不可忽视**：`DSARandomUniform`（1.2%）+ `Sort`（1.1%）合计 2.3%，但 TC 不模拟 sampling，不影响推理耗时估算
- **与 DSV3 对比**：Qwen3 无 MoE，无 `DispatchFFNCombine`，通信形式为 all-reduce（TP），而非 DSV3 的 reduce-scatter + all-gather（SP）

### 计算+通信融合算子确认

**结论：Qwen3-32B Profiling 中不存在计算+通信融合类 kernel。**

| 融合类型 | kernel Type | 是否存在 | 说明 |
|---------|------------|---------|------|
| MC2（MatMul + AllReduce） | 无专用 kernel | 否 | `MatMulV2` 和 `hcom_allReduce_` 分开记录 |
| MoE EP 融合（DispatchFFNCombine） | 无 | 否 | Qwen3 无 MoE 结构 |

`MatMulV2`（7710 次）和 `hcom_allReduce_`（7748 次）调用次数几乎相等，印证了 TP 模式下每个 matmul 后跟一次 all-reduce，两者独立记录，无融合。

### op_mapping 覆盖状态（C3 参考）

| kernel Type | TC op 映射 | 状态 |
|------------|-----------|------|
| hcom_allReduce_ | `tensor_cast.all_reduce` | 已配置 |
| MatMulV2 | `aten.mm.default` | 已配置 |
| FusedInferAttentionScore | `tensor_cast.attention` | 已配置（attention_special） |
| AddRmsNormBias | `tensor_cast.add_rms_norm` / `add_rms_norm2` | 已配置（CANN 8.5） |
| DSARandomUniform | — | TC 不模拟（sampling） |
| Sort | — | TC 不模拟（sampling） |
| SwiGlu | `tensor_cast.swiglu` | 已配置 |
| split_qkv_rmsnorm_rope_kernel | `tensor_cast.apply_rope`（composite） | Triton 融合 kernel，TC 分解为 rms_norm + apply_rope |
| hcom_allGather_ | `tensor_cast.all_gather` | 已配置 |
| ReshapeAndCacheNdKernel | `tensor_cast.reshape_and_cache` | 已配置 |
| allgatherAicpuKernel | `tensor_cast.all_gather`（alternate） | 已配置 |
| SoftmaxV2 | — | TC 不模拟（sampling） |
| RealDiv | `aten.div.Tensor` | 已配置（alternate RealDiv） |
| MaskedFill | — | TC 不模拟（sampling） |
| ArgMaxV2 | — | TC 不模拟（sampling） |
| Neg | — | TC 不模拟（sampling） |
| ApplyTopKTopPCustom | — | TC 不模拟（sampling） |
| Add | `aten.add.Tensor` | 已配置 |
| GreaterEqual | — | TC 不模拟（sampling） |
| Mul | `aten.mul.Tensor` | 已配置 |

### 注意事项（C3 验证重点）

- **`split_qkv_rmsnorm_rope_kernel`**（排名 8，0.70%）：vllm-ascend Triton 融合 kernel，TC 无对应单一 op，分解为 `rms_norm + apply_rope`。需确认分解后耗时估算误差是否可接受
- **`hcom_allReduce_` 占比异常高（84%）**：工作计划风险 R7 已标注，Phase 1 mini 验证需重点确认 `CommAnalytic` 在此场景下的精度
