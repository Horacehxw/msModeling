# 算子性能数据库穿刺实验总结

**日期**: 2026-03-05
**作者**: Claude (AI 辅助开发)
**关联设计文档**: `docs/perf_database/OPERATOR_PERF_DATABASE_DESIGN_zh_v1.2.md`

---

## Executive Summary

本穿刺实验验证了设计文档 v1.2 中 `EmpiricalPerformanceModel + DataSource` 架构的可行性。以 Qwen3-32B BF16 Prefill（TP=16）为测试场景，基于 Qwen3-30B 的真实 Profiling 数据，**经过 5 轮迭代优化，算子匹配率从 6.5% 提升至 87.0%（40/46），其中全部 12 个计算算子实现 100% 匹配**。

核心发现：TC dispatch trace 与 NPU Profiling 之间存在 8 类系统性的 shape 差异（batch 维度、权重格式、融合算子拆分、RoPE 布局等），均可通过 `ProfilingDataSource` 中的通用规则化处理解决，无需修改 TC 核心代码。剩余 6 个未匹配算子属于结构性差异（attention 特殊模式、通信算子、KV Cache 接口差异），需后续专项开发。

**关键结论：基于 Profiling 数据的性能估算路线可行，值得投入产品化。**

---

## 1. 穿刺范围与目标

| 项目 | 内容 |
|------|------|
| **验证目标** | 设计文档 §4.1-4.3：DataSource ABC → ProfilingDataSource → EmpiricalPerformanceModel |
| **测试模型** | Qwen/Qwen3-32B (BF16, Prefill, TP=16) |
| **测试硬件** | ATLAS_800_A3_752T_128G_DIE |
| **Profiling 来源** | Qwen3-30B Prefill（同架构维度，TP=16，seq=136） |
| **数据规模** | 45 个 kernel CSV 文件，60+ 条 op_mapping 映射 |

---

## 2. 迭代过程与结果

### 2.1 匹配率演进

| 轮次 | 匹配率 | 新增方案 | 新增 HIT |
|------|--------|---------|---------|
| v1 (基线) | 3/46 (6.5%) | `--compile` 启用融合算子 | — |
| v2 | 9/46 (19.6%) | batch 维度剥离 + SwiGlu 输入合并 | +6 |
| v3 | 36/46 (78.3%) | zero-cost 算子注册 + 备选 kernel_type | +27 |
| v4 | 38/46 (82.6%) | RoPE shape 归一化 + 对称 batch 剥离 | +2 |
| v5 | 40/46 (87.0%) | 复合算子分解（matmul_all_reduce） | +2 |

### 2.2 按类别匹配情况

| 算子类别 | 总数 | 匹配 | 匹配率 | 说明 |
|---------|------|------|--------|------|
| 计算算子（MatMul、Norm、激活、RoPE） | 10 | 10 | **100%** | 全部命中 |
| 复合计算（matmul_all_reduce） | 2 | 2 | **100%** | 分解后匹配 MatMulV2 |
| 零代价算子（view、permute、split 等） | 28 | 28 | **100%** | 标记为 zero_cost |
| KV Cache (reshape_and_cache) | 1 | 0 | 0% | TC 与 NPU 接口结构差异 |
| Embedding (GatherV2) | 1 | 0 | 0% | 词表 TP 分片差异 |
| 通信 (all_gather) | 1 | 0 | 0% | 需 CommGrid 带宽模型 |
| Attention (特殊模式) | 1 | 0 | 0% | 需专用匹配逻辑 |
| 其他 (index) | 2 | 0 | 0% | 无对应 Profiling 数据 |

### 2.3 已匹配计算算子详情

| TC 算子 | NPU Kernel | 延迟 (us) | 匹配方法 |
|---------|-----------|-----------|---------|
| `aten.mm` (QKV 投影) | MatMulV2 | 19.6 | FRACTAL_NZ 还原 + padding |
| `aten.mm` (gate_up 投影) | MatMulV2 | 59.7 | FRACTAL_NZ 还原 + padding |
| `aten.mm` (lm_head) | MatMulV2 | 91.8 | ND 转置匹配 |
| `matmul_all_reduce` (o_proj) | MatMulV2 (复合) | 14.2 | 复合分解 + FRACTAL_NZ |
| `matmul_all_reduce` (down_proj) | MatMulV2 (复合) | 25.1 | 复合分解 + FRACTAL_NZ |
| `rms_norm` x3 | RmsNorm | 21.7 / 20.1 / 7.7 | batch 剥离 + padding |
| `add_rms_norm2` | AddRmsNorm | 12.5 | batch 剥离 + padding |
| `swiglu` | SwiGlu | 14.9 | 输入合并 + batch 剥离 |
| `apply_rope` | ApplyRotaryPosEmb | 12.5 | RoPE 归一化 + 备选 kernel |
| `aten.add` | Add | 16.2 | batch 剥离 + padding |

**每层总计算延迟：316.0 us**（基于实测 Profiling 数据）

---

## 3. 关键发现

### 3.1 TC 与 NPU Profiling 的 8 类 Shape 差异

| # | 差异类型 | TC 行为 | Profiling 行为 | 解决方案 |
|---|---------|--------|---------------|---------|
| 1 | Batch 维度 | 保留 `(1, seq, dim)` | 扁平化 `(seq, dim)` | `_strip_batch_dim()` |
| 2 | Seq padding | `ceil(seq/16)*16` = 144 | 原始 seq = 136 | block-padding 容差 |
| 3 | FRACTAL_NZ | TC 用 ND `(K, N)` | 权重 `[H,W,bh,bw]` | `fractal_nz_to_nd()` |
| 4 | ND 转置 | `F.linear` 转置后 `(K,N)` | 存储 `(N,K)` | MatMul 专用转置检查 |
| 5 | SwiGlu 输入 | 2 个独立输入 `(S,D/2)` | 1 个融合输入 `(S,D)` | 按末维拼接 |
| 6 | RoPE 布局 | `(B,H,S,D)` + Q/K 顺序 | `(B,S,H,D)` + K/Q 顺序 | `_normalize_rope_inputs()` |
| 7 | RoPE kernel | 同一 TC op 对应不同 kernel | neox→ApplyRotaryPosEmb | `alternate_kernel_types` |
| 8 | 复合算子 | matmul+allReduce 融合 | 可能独立或 MC2 融合 | `_lookup_composite()` |

### 3.2 op_mapping.yaml 扩展机制

穿刺过程中为 op_mapping.yaml 引入了 3 个新字段：

```yaml
# 备选 kernel 类型（同一 TC op 在不同模型/配置下映射不同 kernel）
alternate_kernel_types: [ApplyRotaryPosEmb]

# 零代价标记（纯 shape 变换，无硬件执行）
zero_cost: true

# 复合算子分解
composite: true
sub_kernels: [MatMulV2, hcom_allReduce_]
```

### 3.3 `--compile` 对算子匹配的关键影响

不加 `--compile` 时，TC 将融合算子（RmsNorm、SwiGlu、RoPE）分解为 72+ 个 aten 原语，导致无法匹配到 Profiling 中的融合 kernel。**`--compile` 与量化无关（BF16 也需要），是正确使用 `--performance-model profiling` 的前提条件**。

---

## 4. 未解决问题分析

### 4.1 Attention 特殊模式（P1，影响最大）

`FusedInferAttentionScore` 是 Prefill 中单次延迟最高的算子（~100us 级别），但其输入结构复杂（Q、K cache、V cache、mask、seq_lens 等 7 个输入），需要专用的 shape 匹配逻辑，包括：
- seq 维度动态：依赖 num_queries × query_length
- block 维度：依赖 KV cache block_size（128）
- head 维度：需要按 TP 分片后的 num_heads 匹配

### 4.2 通信算子（P2，架构设计问题）

`hcom_allReduce_` 和 `hcom_allGather_` 的延迟不取决于 tensor shape，而取决于：
- 消息大小（bytes）
- 拓扑结构（ring/mesh/full-mesh）
- 通信组大小（world_size）

设计文档 §4.4 已规划 `CommDataSource`，需结合 `CommGrid` 的带宽参数和 HCCL 实测数据。

### 4.3 reshape_and_cache（P3，TC 接口差异）

TC 的 `reshape_and_cache` op 接口与 NPU kernel（`ReshapeAndCacheNdKernel`）差异过大：
- 输入数量不同（4 vs 5）
- KV tensor 缺少 head 维度
- cache 结构不同

**根本原因**：TC 为了通用性使用简化接口，而 NPU kernel 需要完整的 paged attention cache 信息。

### 4.4 Embedding 词表分片（P4，较易解决）

TC 发送全量词表 `(151936, 5120)`，Profiling 存储 TP 分片后的 `(9496, 5120)` = 151936/16。解决思路：在 `_inputs_match` 中加入 embedding-aware 的词表维度缩放逻辑（需要传入 TP size）。

---

## 5. 下一步行动建议

### 第一阶段：完善穿刺 → 可 Demo 状态（1 周）

| 优先级 | 任务 | 预计收益 | 说明 |
|--------|------|---------|------|
| **P0** | Attention 特殊模式匹配 | +1 HIT，覆盖最高延迟算子 | 实现 `FusedInferAttentionScore` 的 seq/head/block 感知匹配 |
| **P0** | 通信算子带宽模型 | +1 HIT | 基于 CommGrid 参数 + HCCL benchmark 数据实现 `CommDataSource` |
| P1 | Embedding TP 分片 | +1 HIT | 在匹配逻辑中传入 world_size，缩放词表维度 |
| P1 | Shape 插值（设计文档 §4.8） | 支持任意 seq length | 在 ProfilingDataSource 基础上实现 `InterpolatingDataSource` |

### 第二阶段：DSV3 Decode 支持 + 端到端验证（2 周）

| 任务 | 说明 |
|------|------|
| DSV3 Decode 数据集成 | `v0.14.0_dsv3_decode/` 数据已就绪，需 W8A8 dtype 支持和 Decode 特有算子映射 |
| MoE 算子映射 | GroupedMatmul、MoeGatingTopK、DistributeDispatch/Combine |
| 端到端精度验证 | 对比 TC 仿真结果与实际 vLLM Profiling 的端到端延迟，目标 <15% 误差 |
| 与 develop 分支集成 | 合并 gitcode/develop 的 SwiGlu 融合、GMM 融合等新 pass |

### 第三阶段：产品化（3-4 周，对应设计文档 §5）

| 任务 | 说明 |
|------|------|
| 数据采集自动化 | Profiling 数据解析 → CSV 导出 → 数据库验证的 CI 流水线 |
| 多版本管理 | 支持 CANN/vLLM-Ascend 不同版本的数据目录隔离 |
| 条件映射（conditional kernel_type） | 替代 `alternate_kernel_types`，基于 op kwargs 精确映射 |
| 直接 vLLM op graph 抓取 | 绕过 TC dispatch，直接从 vLLM 实跑抓取算子图（设计文档核心设计原则） |

### 架构建议

1. **op_mapping.yaml 应按模型/场景拆分**：当前单文件 60+ 条映射，随模型增多将膨胀。建议 `op_mapping_base.yaml` + `op_mapping_qwen3.yaml` overlay 模式。

2. **Shape 匹配管线需要更好的可观测性**：当前 MISS 只有 DEBUG 日志，建议增加结构化的 match report 输出（类似本报告的 §2.3 表格），方便快速定位新模型的匹配问题。

3. **ProfilingDataSource 不应感知 TC 的 batch/padding 行为**：当前 `_strip_batch_dim` 和 padding 容差是为了弥补 TC 与 Profiling 的差异。长期方向应在 TC 或 EmpiricalPerformanceModel 层面统一 shape 归一化，而非在 DataSource 内部逐个处理。

---

## 6. 交付物清单

| 类别 | 文件/路径 | 说明 |
|------|----------|------|
| **核心代码** | `tensor_cast/performance_model/perf_database/data_source.py` | DataSource ABC |
| | `tensor_cast/performance_model/perf_database/profiling_data_source.py` | CSV 查询 + 8 种 shape 匹配规则 |
| | `tensor_cast/performance_model/empirical.py` | EmpiricalPerformanceModel 重构 |
| | `tensor_cast/core/model_runner.py` | `--performance-model profiling` CLI |
| **数据** | `.../v0.14.0/op_mapping.yaml` | 60+ 条算子映射 |
| | `.../v0.14.0/*.csv` (45 个) | Qwen3-30B Profiling 数据 |
| **测试** | `tests/perf_database/` (34 个测试) | 全部通过 |
| **文档** | `docs/perf_database/reports/qwen3_32b_alignment_report.md` | 详细对齐报告 (v5) |
| | `docs/perf_database/reports/spike_executive_summary_zh.md` | 本文档 |
| | `docs/plans/2026-03-04-perf-database-spike.md` | 穿刺计划 |
| | `docs/plans/2026-03-05-fix-shape-matching.md` | Shape 匹配修复计划 |
