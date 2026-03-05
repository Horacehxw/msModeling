# 算子性能数据库穿刺实验总结

**日期**: 2026-03-05
**作者**: Claude (AI 辅助开发)
**关联设计文档**: `docs/perf_database/OPERATOR_PERF_DATABASE_DESIGN_zh_v1.2.md`

---

## Executive Summary

本穿刺实验验证了设计文档 v1.2 中 `EmpiricalPerformanceModel + DataSource` 架构的可行性。以 Qwen3-32B BF16 Prefill（TP=16）为测试场景，基于 Qwen3-30B 的真实 Profiling 数据，**经过 5 轮迭代优化，算子匹配率从 6.5% 提升至 87.0%（40/46），其中全部 12 个计算算子实现 100% 匹配**。

核心发现：TC dispatch trace 与 NPU Profiling 之间存在 8 类系统性的 shape 差异（batch 维度、权重格式、融合算子拆分、RoPE 布局等），均可通过 `ProfilingDataSource` 中的通用规则化处理解决，无需修改 TC 核心代码。剩余 6 个未匹配算子属于结构性差异（attention 特殊模式、通信算子、KV Cache 接口差异），需后续专项开发。

穿刺过程中发现 **17 处简化实现**（详见 §5），涉及 shape 匹配、op_mapping 映射、查询逻辑、dtype、数据等五个层面。这些简化在穿刺阶段足够验证可行性，但产品化前需逐项评估和升级。

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
- seq 维度动态：依赖 num_queries x query_length
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

## 5. 穿刺中的简化实现清单

穿刺过程中共有 17 处简化实现，按层面分类如下。每项标注产品化所需的升级方向和对应的设计文档章节。

### 5.1 Shape 匹配层面

#### S-1: Block-Padding 容差 — 硬编码对齐值

**现状**: `_BLOCK_SIZES = (16, 32, 64)` 硬编码三种对齐值，TC 维度只要是 CSV 维度按这些值 ceil 对齐的结果就算匹配。

**风险**: 实际 NPU tile 对齐策略更复杂（不同算子、不同 dtype 可能有不同 tile size），可能产生误匹配或漏匹配。

**产品化方向**: 根据 kernel_type + dtype 确定准确的 tile size（Da Vinci Cube: BF16=16x16, INT8=16x32）。可在 op_mapping.yaml 中添加 `tile_alignment` 字段。

#### S-2: Batch 维度剥离 — 无差别剥离 leading dim=1

**现状**: `_strip_batch_dim()` 对所有算子无差别地剥离 leading dim=1，不区分该维度是真正的 batch 还是其他语义。

**风险**: 如果某算子的 leading dim=1 有语义含义（不是 batch），会导致误匹配。目前对所有 TC 和 CSV 输入都做对称剥离。

**产品化方向（设计文档 §4.2）**: 应在 TC 层面或 EmpiricalPerformanceModel 统一 shape 归一化，而非在 DataSource 内部逐个处理。

#### S-3: RoPE 归一化 — 硬编码 Q/K 重排 + 转置规则

**现状**: `_normalize_rope_inputs()` 假设 TC 永远发送 `[Q(B,H,S,D), K(B,H,S,D), cos, sin]`，CSV 永远是 `[K(B,S,H,D), Q(B,S,H,D), cos, sin]`。输入顺序和维度排列规则是代码硬编码的。

**风险**: 不同版本 vLLM-ascend 或不同 RoPE 模式（如 GLM4 的 rotary_dim != head_dim）可能改变输入结构。

**产品化方向**: 在 op_mapping.yaml 中用声明式规则描述 shape 变换（如 `input_transform: [{permute: [0,2,1,3]}, {swap: [0,1]}]`），而非在 Python 代码中硬编码。

#### S-4: SwiGlu 输入合并 — 假设 2→1 合并

**现状**: 假设 TC 永远发 2 个等形状输入，CSV 永远存 1 个沿末维拼接的输入。

**风险**: W8A8 DequantSwigluQuant 等变体的输入结构不同，此规则不适用。

**产品化方向**: 与 S-3 类似，用声明式融合模式描述。

#### S-5: ND 转置匹配 — 仅对第 2+ 个输入尝试

**现状**: `i >= 1 and fmt == "ND"` — 只对非第一个 ND 格式输入做 `(K,N) <-> (N,K)` 转置。

**风险**: 假设第一个输入永远是 activation（不需转置），非标准 matmul pattern 会失效。

### 5.2 op_mapping.yaml 映射层面

#### S-6: `alternate_kernel_types` — 暴力回退而非条件分派

**现状**: RoPE 的 `alternate_kernel_types: [ApplyRotaryPosEmb]` 是"试完主类型再试备选"，不看 `is_neox` 等运行时参数。

**产品化方向（设计文档 §4.5）**: 实现 `kernel_type_variants` 条件映射：
```yaml
kernel_type_variants:
  - condition: {is_neox: true}
    kernel_type: ApplyRotaryPosEmb
  - condition: {is_neox: false}
    kernel_type: InterleaveRope
```

#### S-7: `zero_cost: true` — 笼统标记，部分存疑

**现状**: 14 个 op 标记为 zero_cost，但其中部分存疑：
- `aten.copy_.default`：实际有数据搬移（KV cache update），Profiling 中可能表现为 TensorMove kernel
- `aten.slice.Tensor`：跨步切片有实际开销
- `aten.arange.start`：有微量计算

**产品化方向**: 区分"真正零代价"（view、permute）和"近似零代价"（copy_、slice），后者应有估算逻辑或查询 TensorMove.csv。

#### S-8: 量化变体映射未经 Profiling 验证

**现状**: 以下映射基于 op-plugin 代码分析推导，无实际 Profiling 数据验证：
- `fp8_linear` → `QuantBatchMatmulV3`（FP8 专用 API 可能不存在）
- `mxfp4_linear` → `QuantBatchMatmulV3`（placeholder）
- `grouped_matmul_fp8_swiglu` → `DequantSwigluQuant`（路径不确定）
- 所有 `*_all_reduce` 复合算子的 sub_kernels 分解（实际可能走 MC2 单 kernel）

**产品化方向**: 需采集 FP8/MXFP4/W4A8 场景的 Profiling 数据，验证或修正映射。

#### S-9: MoE 路由算子映射依赖场景假设

**现状**: `permute_tokens` 映射到 `MoeDistributeDispatchV2`（EP 场景），非 EP 场景应映射到 `MoeInitRouting`。当前无条件映射。

**产品化方向**: 与 S-6 相同，需 `kernel_type_variants` 按 EP/非 EP 条件选择。

### 5.3 查询逻辑层面

#### S-10: 复合算子分解 — 只取计算部分，忽略通信延迟

**现状**: `_lookup_composite()` 跳过 `hcom_*` sub_kernel，只返回 MatMulV2 延迟，confidence=0.8。通信部分完全交给 analytic model。

**风险**: 实际 MC2 是流水线融合（matmul 和 allReduce 重叠执行），latency != matmul + allReduce，分开估算会**高估**总延迟。

**产品化方向**: MC2 kernel 应有独立的 Profiling CSV（Type = MC2 专用 kernel），而非分解。或在 `_lookup_composite` 中建模流水线重叠。

#### S-11: Attention 完全跳过（设计文档 §4.2 query_mode）

**现状**: `query_mode: attention_special` → 直接返回 None。`FusedInferAttentionScore` 是延迟最高的单算子（~100us+），对端到端精度影响最大。

**产品化方向（设计文档 §4.2）**: 实现 attention_special 查询模式：提取 (seq_len, num_heads, head_dim, block_size) 等关键维度，结合 FusedInferAttentionScore.csv 匹配。需同时处理 PA（PagedAttention）和 FA（FlashAttention）两种模式。

#### S-12: 通信算子完全跳过（设计文档 §4.4 CommDataSource）

**现状**: `category: communication` → 直接返回 None。`hcom_allReduce_.csv` 只存一个平均值 690us，无 shape 依赖。

**产品化方向（设计文档 §4.4）**: 实现 `CommDataSource`，基于消息大小 + 拓扑 + 通信组的带宽模型。需 HCCL benchmark 数据（`hccl/{cann_version}/`）。

#### S-13: CSV 逐行遍历匹配，无索引

**现状**: `_inputs_match` 对 CSV DataFrame 逐行 `iterrows()`，O(N) 暴力匹配。当前 CSV 只有几行，不影响性能。

**产品化方向**: 数据量大时（Microbenchmark 网格可达数千行）需建立 shape hash 索引或预排序结构。

### 5.4 dtype 映射层面

#### S-14: FP16 = BF16 等价处理

**现状**: `torch.float16: "DT_BF16"` — 将 FP16 视同 BF16。Ascend A3 上 BF16 和 FP16 共用相同 kernel 路径。

**风险**: 如果未来硬件或 CANN 区分 FP16/BF16 kernel 路径，会导致 dtype 不匹配。

#### S-15: 不检查 output dtype/shape

**现状**: `_inputs_match` 只检查输入 shape + dtype，完全不看输出。

**风险**: 同一输入不同输出配置（如 in-place vs out-of-place、不同输出 dtype）可能有不同性能。

### 5.5 数据层面

#### S-16: 单场景数据 + 跨模型复用

**现状**: 用 Qwen3-30B Prefill（TP=16, seq=136）的 Profiling 数据验证 Qwen3-32B。两者同架构但不同参数，隐含假设"同架构 → kernel 行为相同"。

**产品化方向**: 需要多 seq length、多 batch size 的 Profiling 数据 + 插值。对应设计文档 §4.8 InterpolatingDataSource。

#### S-17: 无插值 — 严格精确匹配

**现状**: 严格精确匹配（允许 padding 容差），不支持对未见 shape 进行插值估算。seq=200 就无法匹配。

**产品化方向（设计文档 §4.8）**: 实现 `InterpolatingDataSource`，参考 AI Configurator 的 2D+1D 混合插值 + sqrt 变换（Attention O(n^2) 算子）。

---

## 6. 下一步行动建议

结合穿刺结论、简化实现清单、以及设计文档 v1.2 的整体规划，建议按以下三个阶段推进。

### 第一阶段：消除关键盲区（1-2 周，对应设计文档 §4.2-4.8）

目标：解决穿刺中跳过的 3 类算子 + 实现插值，使端到端仿真可用。

| 优先级 | 任务 | 涉及简化项 | 设计文档 | 预计收益 |
|--------|------|-----------|---------|---------|
| **P0** | **Attention 特殊模式匹配** | S-11 | §4.2 query_mode | +1 HIT，覆盖最高延迟算子 |
| **P0** | **CommDataSource 通信带宽模型** | S-12 | §4.4 | +1 HIT，解决 allReduce/allGather |
| **P0** | **InterpolatingDataSource 插值** | S-17 | §4.8 | 支持任意 seq length，不再依赖精确匹配 |
| P1 | Embedding TP 分片 | — | — | +1 HIT |
| P1 | 条件映射（kernel_type_variants） | S-6, S-9 | §4.5 | 替代 alternate_kernel_types 暴力回退 |
| P2 | 精细化 zero_cost 分类 | S-7 | — | 区分真零代价 vs 近似零代价 |

**Attention 匹配具体方案**：
1. 从 `OpInvokeInfo` 提取 `(seq_len, num_heads, head_dim, block_size, num_blocks)` 关键维度
2. `FusedInferAttentionScore.csv` 已有数据（Qwen3 Prefill 67x），建立多维匹配
3. 区分 PA（PagedAttention, decode）和 FA（FlashAttention, prefill）两种模式的输入结构

**InterpolatingDataSource 具体方案**（参考 AI Configurator）：
1. Wrapper 模式包装 ProfilingDataSource：精确命中 → 直接返回，未命中 → 插值
2. 激活维度（seq_len, batch）做线性/双线性插值
3. Attention 算子做 sqrt 变换后再插值（O(n^2) 复杂度）
4. 只对 `interpolatable` 标记的维度做插值（op_mapping.yaml 已有 `interpolation_policy` 字段）

### 第二阶段：DSV3 Decode 支持 + 端到端验证（2-3 周，对应设计文档 §5.2）

目标：覆盖第二个目标模型 DeepSeekV3 Decode 场景，验证端到端精度 <15%。

| 任务 | 涉及简化项 | 说明 |
|------|-----------|------|
| **DSV3 Decode 数据集成** | S-16 | `v0.14.0_dsv3_decode/` 数据已就绪，需 W8A8 dtype 支持 |
| **MoE 算子映射验证** | S-8, S-9 | GroupedMatmul、MoeGatingTopK、DistributeDispatch/Combine 实际验证 |
| **MC2 融合 kernel** | S-10 | 确认 MC2 在 Profiling 中的实际表现（单 kernel vs 分离），调整复合分解逻辑 |
| **reshape_and_cache 结构适配** | — | 分析 TC 与 NPU 的 KV cache 接口差异，选择改 TC op 还是加适配层 |
| **端到端精度验证** | — | 对比 TC 仿真结果 vs 实际 vLLM Profiling 端到端延迟，目标 <15% |
| **与 develop 分支集成** | — | 合并 gitcode/develop 的 SwiGlu 融合、GMM+SwiGlu 融合等新 pass |
| **声明式 shape 变换** | S-3, S-4 | 将 RoPE/SwiGlu 的硬编码归一化改为 op_mapping.yaml 中的声明式规则 |

**DSV3 Decode 新增算子映射清单**：
- `QuantBatchMatmulV3`（15006x）— W8A8 matmul，需 INT8 dtype 支持
- `AscendQuantV2`（10004x）— static quantization
- `DequantSwigluQuant`（4879x）— GMM+SwiGlu+Quant 三合一
- `GroupedMatmul`（4756x）— MoE expert computation
- `InplaceAddRmsNorm`（5002x）— AddRmsNorm in-place 变体
- `TransposeBatchMatMul`（5002x）— MLA absorb projections
- `InterleaveRope`（2501x）— DeepSeek interleave RoPE
- `KvRmsNormRopeCache`（2501x）— KV Norm+RoPE+Cache 融合
- `MoeGatingTopK`（2378x）— MoE routing
- `MoeDistributeDispatch/CombineV2`（2378x each）— MoE token routing

### 第三阶段：产品化（3-4 周，对应设计文档 §5.3-5.5）

目标：达到可维护、可扩展的产品质量。

| 任务 | 涉及简化项 | 设计文档 | 说明 |
|------|-----------|---------|------|
| **数据采集自动化** | S-16 | §5.4 | Profiling → CSV → 验证的 CI 流水线 |
| **Microbenchmark 网格** | S-13 | §5.5 | 生成计算算子 microbench 脚本，扩充 CSV 覆盖范围 |
| **CSV 索引优化** | S-13 | — | shape hash 索引，支持数千行快速查询 |
| **多版本管理** | — | §2.3 | CANN/vLLM-Ascend 版本数据目录隔离 |
| **op_mapping 分层** | — | — | `op_mapping_base.yaml` + 模型/场景 overlay |
| **Shape 归一化层级上移** | S-1, S-2 | — | 在 TC/EmpiricalPerformanceModel 层统一归一化 |
| **直接 vLLM op graph 抓取** | — | §设计原则 | 绕过 TC dispatch，直接从 vLLM 实跑抓取算子图 |
| **CompositePerformanceModel** | — | §9.3 | 多 PerformanceModel 级联调度器 |

### 架构建议

1. **op_mapping.yaml 应按模型/场景拆分**: 当前单文件 60+ 条映射，随模型增多将膨胀。建议 `op_mapping_base.yaml` + `op_mapping_qwen3.yaml` overlay 模式。

2. **Shape 匹配管线需要更好的可观测性**: 当前 MISS 只有 DEBUG 日志，建议增加结构化的 match report 输出（类似本报告 §2.3 表格），方便快速定位新模型的匹配问题。

3. **ProfilingDataSource 不应感知 TC 的 batch/padding 行为**: 当前 `_strip_batch_dim` 和 padding 容差是为了弥补 TC 与 Profiling 的差异。长期方向应在 TC 或 EmpiricalPerformanceModel 层面统一 shape 归一化，而非在 DataSource 内部逐个处理。

4. **优先实现 InterpolatingDataSource**: 穿刺依赖精确 shape 匹配 + padding 容差，但生产环境 seq length 变化频繁。插值是实用性的关键瓶颈，应优先于其他优化项。

---

## 7. 交付物清单

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
