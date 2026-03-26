# 实测算子性能数据库 Phase 1 E2E 验证报告

**日期**: 2026-03-15
**Profiling 数据**: CANN 8.5, vLLM 0.15.0, torch 2.9.0 (2026-03-13 采集, eager prefill + cudagraph decode)

---

## 1. 背景与目标

TensorCast 正在构建基于实测 Profiling 数据的算子性能估算系统（`EmpiricalPerformanceModel + ProfilingDataSource`）。该系统通过 `op_mapping.yaml` 将 TC 算子映射到 NPU kernel，再从 CSV 数据中查询匹配 shape 的实测延迟，替代 Roofline 解析模型。

**Phase 1 目标**: 查询基础设施打通 + 全量 blocker 暴露 + 指标体系建立。不追求 <15% E2E 误差（Work Plan Phase 3 目标）。

**验证范围**: 4 个 E2E 场景 × 2 模型 (Qwen3-32B BF16, DeepSeek-V3 W8A8) × 2 阶段 (Prefill, Decode)。

---

## 2. 结论

### 2.1 GO/NO-GO: GO

Phase 1 目标达成:
1. 查询基础设施 4 场景全部可运行
2. 所有 MISS 原因已分类，解决方案路径清晰，每项有 owner
3. 指标体系建立（M1-M3 + 悲观规则 + 融合分组）

### 2.2 结果总表

| 场景 | TC 参数 | M1: Op-Count HR | M2: Fused Op HR (GO/NO-GO) | M3: Fused (不含 zc) |
|------|---------|----------------|-----------------|---------------------|
| Qwen3 Prefill | nq=10, ql=4104, tp=16, BF16 | 78.6% (44/56) | **63.3%** (19/30) | 31.2% (5/16) |
| Qwen3 Decode | nq=16, ql=1, cl=4096, tp=16, BF16 | 81.8% (45/55) | **70.0%** (21/30) | 43.8% (7/16) |
| DSv3 Prefill | nq=1, ql=256, tp=8, dp=2, ep=16, W8A8 | 60.2% (62/103) | **38.6%** (17/44) | 12.9% (4/31) |
| DSv3 Decode | nq=16, ql=1, cl=4096, tp=8, dp=2, ep=16, W8A8 | 59.6% (62/104) | **40.9%** (18/44) | 16.1% (5/31) |

### 2.3 关键发现

1. **DSv3 最大 blocker: DispatchFFNCombine (DFC) 融合 gap**。NPU 将 MoE 路径的 `all_to_all×2 + GroupedMatmul×2 + SwiGlu + routing` 融合为单一 DFC kernel，占 DSv3 延迟 ~40%。TC 无对应 fusion pass，需 Phase 2 实现。

2. **Qwen3 最大 blocker: FIA CSV 格式**。FusedInferAttentionScore CSV 来自 profiling trace，非 microbench 格式，无法按 `(batch, seq, heads, head_dim)` 结构化查询。占 Qwen3 延迟 ~9%。

3. **Shape 匹配规则已建立 7 条**（附录 A），覆盖 batch dim strip、FRACTAL_NZ 恢复、权重转置、block-padding、SwiGlu 合并、RoPE layout、flatten batch。但 RoPE 存在 dtype gap（NPU 内部 FP32 vs TC BF16），Qwen3 Prefill 的 RoPE shape 匹配正确但 dtype 阻止 HIT。

4. **TC 主分支需修复 2 个建模问题**: (a) add_rms_norm2 SP 场景未对 seq dim 除以 TP；(b) W8A8 MLA output quantize shape 与 NPU 不一致（3D per-head vs 2D hidden，后者在 CSV 中已存在）。

### 2.4 收益粗估

| 完成项 | Qwen3 M3 | DSv3 M3 |
|--------|----------|---------|
| 当前 | 31-44% | 13-16% |
| + P0 (DFC + FIA + MLA) | ~50-60% | ~40-50% |
| + P1 (shape data + dtype fix + tc_input_count) | ~60-70% | ~55-65% |
| + InterpolatingDataSource | ~75-85% | ~65-75% |

---

## 3. 指标体系

| 指标 | 定义 | 用途 |
|------|------|------|
| **M1: Raw Op-Count HR** | `HIT_invocations / total_invocations` | Debug 用，向后兼容 |
| **M2: Fused Op HR** | 按融合 op 计数（含 zero_cost），悲观规则 | **GO/NO-GO 判定** |
| **M3: Fused Op HR (不含 zc)** | 同 M2，排除 zero_cost | 真实计算覆盖 |

**悲观规则**: 同一 op 若有任一 shape MISS，整个 op 计为 MISS。

**融合分组**: DFC (6-8 个 TC ops = 1 融合 op)、MLAPO、MLA、MC2 同理。全部成员 HIT 才算 HIT。

**Phase 2 待实现**: M4 (Per-Shape Match HR)、M5 (Latency-Weighted HR)。

---

## 4. MISS 根因分析

### 4.1 根因分类汇总

| 根因分类 | 定义 | Qwen3 影响 | DSv3 影响 | 解决方案 |
|---------|------|-----------|----------|---------|
| **fused_kernel_gap** | N:1 融合 (DFC) 无 TC 实现 | 无 | ~40% 延迟 | TC fusion pass |
| **data_format_gap** | CSV 格式/dtype 不匹配 | ~9.5% (FIA + RoPE dtype) | ~1.5% (MLA FIA) | FIA microbench 格式; RoPE dtype 宽松匹配 |
| **shape_coverage_gap** | CSV 缺少匹配 M×D 组合 | ~2% | ~12% (含 quantize ~10%) | Microbenchmark 补充 |
| **tc_decomposition_mismatch** | TC 中间 shape 与 NPU 不同 | ~0.5% (KV cache) | ~2% (MLA quantize, KV cache) | TC 建模层修复 |
| **input_count_mismatch** | TC vs CSV input 数量差异 | ~1% | ~3% | tc_input_count 扩展 |
| **sp_modeling_gap** | SP 场景 TC 未对 seq dim 除以 TP | ~1% | ~0.1% | TC 主分支 Issue |
| **structural_miss** | Embedding TP 辅助 ops 等 | <0.01% | <0.01% | zero_cost 标记 |

### 4.2 DSv3 quantize ×3 MISS 深入分析

DSv3 的 3 个 quantize MISS 经 debug trace 逐 shape 验证，根因各不相同:

| TC Shape | 来源 | 根因 | 解决方案 |
|----------|------|------|---------|
| `(8,16,128)` 3D | MLA attention output (tokens=8, heads=16, head_dim=128) | `tc_decomposition_mismatch`: TC quantize 保留 3D per-head 格式，NPU 在 reshape 到 `(8,2048)` 后做 quantize，`(8,2048)` 在 CSV 中**已存在** | TC 主分支排查 MLA output quantize shape 来源并修复 |
| `(1,8,2304)` → strip → `(8,2304)` | 共享专家 dense FFN (D=18432/TP=2304) | `shape_coverage_gap`: CSV 无 D=2304 | Microbenchmark 补充 |
| `(256,2048)` 2D | MoE routed expert (M=256, D=2048) | `shape_coverage_gap`: CSV 有 `(8,2048)` 但无 `(256,2048)` | Microbenchmark 补充 |

注: `_FLATTEN_BATCH_KERNELS` 的 `(8*16,128)=(128,128)` 对第一个 case 语义错误（应合并后两维 `(8,16*128)=(8,2048)`），但不产生 false positive。

---

## 5. Phase 2 TODO

### 5.1 优先级排序

| # | 工作项 | Owner | 优先级 | 预估工作量 | 预期 M3 提升 |
|---|--------|-------|--------|-----------|-------------|
| **P0-1** | **DFC TC fusion pass** (N:1) | LJW | P0 | 1 周 | DSv3: +15-30pp |
| **P0-2** | **FIA microbench CSV 格式** | ZZY | P0 | 进行中 | Qwen3: +5-10pp |
| **P0-3** | **MLA/MLAPO composite shape 修复** | ZH | P0 | 1-2 天 | DSv3: +3-5pp |
| P1-1 | quantize/norm Microbenchmark shape 网格补充 | TCX | P1 | 数据采集 | DSv3: +3-5pp |
| P1-2 | RoPE dtype 宽松匹配 (_triton_rope FLOAT vs BF16) | 待定 | P1 | ~10 行 | Qwen3 PF: +1-2pp |
| P1-3 | tc_input_count 扩展 (add, mul, index) | 待定 | P1 | Config | 全场景: +1-2pp |
| P1-4 | SP 建模修复 (add_rms_norm2 M/TP) | TC 主分支 | P1 | 需评估 | Qwen3 PF: +1-2pp |
| P1-5 | MoE routing 辅助 ops CSV 补充 | TCX | P1 | 数据采集 | DSv3: +1-2pp |
| P2-1 | Per-Shape Match HR (M4) 实现 | ZH | P2 | ~60 行 | 诊断改善 |
| P2-2 | Latency-Weighted HR (M5) 实现 | HXW | P2 | 外部脚本 | 性能评估 |
| P2-3 | Static cost 提取为公共组件 | TC 主库 Issue | P2 | ~30 行 | E2E +1-5% |

### 5.2 TC 主分支 Issue

| Issue | 描述 | 预期收益 |
|-------|------|---------|
| add_rms_norm2 SP 维度 | SP 场景下 TC 未对 seq dim 除以 TP，导致 M=41040 vs NPU M=2565 | 修复后 Qwen3 PF norm 从 MISS→HIT |
| MLA output quantize shape | W8A8 MLA 路径 TC quantize shape `(8,16,128)` (3D) vs NPU `(8,2048)` (2D)，后者在 CSV 中已存在 | 修复后 DSv3 每层 quantize 从 MISS→HIT (~1-2%) |

---

## 6. 代码变更总结

### 6.1 计划内变更 (C1-C6)

| 变更 | 说明 |
|------|------|
| MLA/MLAPO 解除硬编码拒绝 | composite lookup 恢复 |
| moe_gating_topk op | 匹配 NPU MoeGatingTopK kernel |
| tc_input_count 配置 | 7 ops (quantize, embedding 等) |
| MISS reason 修正 | tc_input_count 双侧截断 |
| TC_ENABLE_INTERPOLATION 开关 | 默认 OFF |
| Fused Op HR 指标 | 含悲观规则修正 |

### 6.2 E2E 验证过程中发现的改进

| 变更 | 发现过程 |
|------|---------|
| 通信 alpha-beta 模型插值 | E2E 发现通信全 MISS，message_bytes 精确匹配不合理 |
| Embedding TP row 辅助 ops (zero_cost) | `--word-embedding-tp row` 添加未映射 ops |
| FRACTAL_NZ 权重转置修复 | ND 转置检查仅对 fmt=="ND" 生效，FRACTAL_NZ 恢复后未尝试转置 |
| _MATMUL_KERNELS 补全 | QuantBatchMatmulV3 等不在集合中 |
| EP=16 配置修正 | TC `--ep-size` = vLLM EP = TP×DP |
| HCCL num_devices=8 数据合入 | 修复 DSv3 TP=8 通信 MISS |
| 悲观规则修正 | quantize/mm/add/swiglu 在 HIT 和 MISS 中双重计数 |
| `_FLATTEN_BATCH_KERNELS` 3D→2D 匹配 | quantize/norm 的 `(B,M,D)→(B*M,D)` flatten |
| `_ROPE_KERNELS` 扩展 + normalize 修复 | 添加 `_triton_rope`/`split_qkv_rmsnorm_rope_kernel`，支持 tc_input_count=2 |

### 6.3 已修复的 Bug

| Bug | 影响 | 修复 |
|-----|------|------|
| MLA 硬编码拒绝 | MLA/MLAPO 无法 composite 查询 | 删除早期返回 |
| MISS reason 未截断 CSV 侧 | tc_input_count 后 reason 仍为 input_count_mismatch | 双侧截断 |
| FRACTAL_NZ 转置仅对 ND 生效 | QuantBatchMatmulV3 权重转置不匹配 | 移除 fmt=="ND" 限制 |
| _MATMUL_KERNELS 不完整 | 转置检查不对 QuantBatchMatmulV3 等生效 | 补全集合 |
| Fused Op HR 双重计数 | 同名 op 部分 HIT/MISS 时 inflate 分子分母 | 悲观规则 |

### 6.4 测试

137 passed, 9 skipped, 0 failures。

---

## 7. 工作计划和设计文档更新建议

### 7.1 Work Plan 更新

1. **Phase 1 目标** (已达成): "查询基础设施打通 + blocker 全量暴露 + 指标体系建立"
2. **Phase 2 主指标**: M3 (Fused Op HR 不含 zc) > 50%
3. **Phase 3 目标**: M5 (Latency-Weighted HR) > 80%
4. **废弃 Op-Count HR**: M1 仅用于 debug，不作为评估指标

### 7.2 设计文档更新

1. **§4.2 查询分派**: 增加通信 alpha-beta 模型插值描述
2. **§4.5 op_mapping.yaml**: 增加 kernel dispatch 条件记录约定
3. **§4.9 FRACTAL_NZ**: 补充 "FRACTAL_NZ 恢复后仍需尝试权重转置" 规则
4. **§4.10 Flatten Batch 规则**: `_FLATTEN_BATCH_KERNELS` 的 `(B,M,D)→(B*M,D)` 匹配
5. **§7 评估指标**: 替换为 M1-M5 五层指标体系，含悲观规则说明
6. **新增**: EP 配置说明 (TC `--ep-size` = vLLM EP = TP × DP)
7. **新增**: `_MATMUL_KERNELS` 应包含所有 matmul 变体
8. **已完成**: `_ROPE_KERNELS` 已扩展，normalize 支持 tc_input_count=2；dtype gap 待 Phase 2 处理

---

## 附录 A: Shape 匹配规则总览

| # | 规则 | Kernel 范围 | 变换 | 状态 |
|---|------|-----------|------|------|
| 1 | Batch dim=1 strip | 所有 | `(1,M,D)→(M,D)` | 正常 |
| 2 | FRACTAL_NZ 恢复 | 所有 | `[H,W,bh,bw]→(H*bw,W*bh)` | 正常 |
| 3 | ND 权重转置 | `_MATMUL_KERNELS` | `(K,N)↔(N,K)` | 正常 |
| 4 | Block-padding 容忍 | 所有 | `ceil(M/bs)*bs, bs∈{16,32,64}` | 正常 |
| 5 | SwiGlu 输入合并 | `_SWIGLU_KERNELS` | `2×(M,D/2)→(M,D)` | 正常 |
| 6 | RoPE layout 归一化 | `_ROPE_KERNELS` | `(B,H,S,D)→(B,S,H,D)` + Q↔K 重排 | Shape 正确，dtype gap 阻止 HIT |
| 7 | Flatten batch | `_FLATTEN_BATCH_KERNELS` | `(B,M,D)→(B*M,D)` | 正常，待数据补充 |

## 附录 B: TC 命令

```bash
DATA_DIR="$(pwd)/tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5"

# Qwen3 Prefill
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 10 --query-length 4104 --word-embedding-tp row \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --quantize-linear-action DISABLED \
  --performance-model profiling --compile --profiling-database "$DATA_DIR"

# Qwen3 Decode
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 16 --query-length 1 --context-length 4096 --word-embedding-tp row \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --quantize-linear-action DISABLED \
  --performance-model profiling --compile --profiling-database "$DATA_DIR"

# DSv3 Prefill
python3.10 -m tensor_cast.scripts.text_generate deepseek-ai/DeepSeek-V3 \
  --num-queries 1 --query-length 256 --word-embedding-tp row \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 8 --dp-size 2 --ep-size 16 \
  --quantize-linear-action W8A8_STATIC \
  --performance-model profiling --compile --profiling-database "$DATA_DIR"

# DSv3 Decode
python3.10 -m tensor_cast.scripts.text_generate deepseek-ai/DeepSeek-V3 \
  --num-queries 16 --query-length 1 --context-length 4096 --word-embedding-tp row \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 8 --dp-size 2 --ep-size 16 \
  --quantize-linear-action W8A8_STATIC \
  --performance-model profiling --compile --profiling-database "$DATA_DIR"
```

TC 参数从 CSV shapes 反推（非直接使用 vLLM 配置）。EP 配置: vLLM `--enable-expert-parallel` with TP=8, DP=2 → TC `--ep-size 16`。Embedding TP: `--word-embedding-tp row`（profiling 显示 vocab/TP 分片）。加 `--log-level debug` 可查看每个 MISS 的 TC shape 和 CSV shape 对比。

## 附录 C: 全场景逐 Shape MISS 分析

本附录对 4 个 E2E 场景中每一条 MISS 记录进行逐 shape 根因分析。

**根因分类**:

| 代号 | 含义 |
|------|------|
| `data_format_gap` | CSV 格式/dtype 不匹配 |
| `shape_coverage_gap` | CSV 有正确 kernel 但缺少 M/D 组合 |
| `tc_decomposition_mismatch` | TC 分解方式与 NPU 不同 |
| `input_count_mismatch` | TC 与 CSV 输入数量不一致 |
| `fused_kernel_gap` | NPU N:1 融合，TC 无对应 pass |
| `sp_modeling_gap` | SP 场景 TC 未对 seq dim 除以 TP |
| `structural_miss` | 根本语义不同 |

---

### C.1 Qwen3 Decode (M1=81.8%, M2=70.0%, M3=43.8%)

| # | TC Op | NPU Kernel | TC Shape | CSV 最近似 | 根因 | 分析 |
|---|-------|-----------|----------|-----------|------|------|
| 1 | `where.self` | SelectV2 | `(1,16)×3` | 无 CSV | `structural_miss` | Embedding TP row 辅助 op，已标记 zero_cost |
| 2 | `embedding` | GatherV2 | `(9496,5120),(1,16)` | `(9496,5120;336;1)` | `shape_coverage_gap` | tc_input_count=2 已配置，indices M=16 vs CSV M=336 |
| 3 | `mul.Tensor` | Mul | `(1,16,5120),(1,16,1)` | `"336;"` | `input_count_mismatch` | TC 2 inputs vs CSV 1 input |
| 4 | `index.Tensor` ×2 | Index | `(40960,256),(16,)` | `(336,5120;1;2;128)` | `input_count_mismatch` | TC 2 inputs vs CSV 4 inputs |
| 5 | `apply_rope` | _triton_rope | `(1,1,16,128),(1,4,16,128)` | `(336,4,128;336,1,128;...)` | `data_format_gap` | shape normalize 正确但 CSV dtype 不匹配 (FLOAT vs BF16) + decode M=16 无 CSV |
| 6 | `reshape_and_cache` | ReshapeAndCacheNdKernel | `(16,128),(16,128),...` | `(333,1,128;...)` | `tc_decomposition_mismatch` | TC KV 2D 缺 head_dim，CSV 3D 含 head_dim=1 |
| 7 | `attention` | FusedInferAttentionScore | (特殊查询) | CSV 缺 microbench 列 | `data_format_gap` | FIA CSV 非 microbench 格式 |
| 8 | `add.Tensor` | Add | `(1,16,5120),(16,5120)` | 无匹配 | `input_count_mismatch` | TC 2 inputs，shape 语义不匹配 |
| 9 | `index.Tensor` | Index | `(1,16,5120),(16,)` | `(16,5120;1;2;16)` | `input_count_mismatch` | TC 2 inputs vs CSV 4 inputs |
| 10 | `copy_` | TensorMove | `(2,513,128,1,128)` 5D | `(336,5120)` 2D | `structural_miss` | KV cache copy vs activation copy |

---

### C.2 Qwen3 Prefill (M1=78.6%, M2=63.3%, M3=31.2%)

与 Decode 共有的 MISS 不重复，仅列 Prefill 特有项。

| # | TC Op | NPU Kernel | TC Shape | CSV 最近似 | 根因 | 分析 |
|---|-------|-----------|----------|-----------|------|------|
| 1 | `rms_norm` | RmsNorm | `(1,41040,5120),(5120,)` | `(2565,5120;5120)` | `sp_modeling_gap` | M=41040 vs CSV M=2565=41040/16(TP)，TC 未除以 TP |
| 2 | `apply_rope` | _triton_rope | `(1,1,41040,128),(1,4,41040,128)` | `(41040,4,128;41040,1,128;...)` | `data_format_gap` | shape 匹配正确，但 CSV 第二 input dtype 为 FLOAT (NPU FP32)，TC 为 BF16 |
| 3 | `add_rms_norm2` | AddRmsNormBias | `(1,41040,5120),(41040,5120),(5120,)` | `(2565,5120;...)` | `sp_modeling_gap` | 同 rms_norm，M=41040 vs M/TP=2565 |

---

### C.3 DSv3 Decode (M1=59.6%, M2=40.9%, M3=16.1%)

DSv3 MISS 数量显著高于 Qwen3，主要因 MoE routing 辅助 ops 和 DFC 融合 gap。

#### C.3.1 Embedding & Shared Attention

| # | TC Op | NPU Kernel | TC Shape | CSV 最近似 | 根因 | 分析 |
|---|-------|-----------|----------|-----------|------|------|
| 1 | `where.self` | SelectV2 | `(1,8)×3` | 无 CSV | `structural_miss` | Embedding TP row 辅助 op |
| 2 | `mul.Tensor` | Mul | `(1,8,7168),(1,8,1)` | `"8;"` | `input_count_mismatch` | TC 2 inputs vs CSV 1 input |
| 3 | `index.Tensor` | Index | `(163840,128),(8,)` | `(163840,64;1;2;4)` | `input_count_mismatch` | TC 2 vs CSV 4 inputs |
| 4 | `rms_norm` | RmsNorm | `(1,8,7168),(7168,)` | `(256,7168;7168)` | `shape_coverage_gap` | M=8 无 CSV，最近 M=256 |
| 5 | `reshape_and_cache` | ReshapeAndCacheNdKernel | `(8,512),(8,64)` | `(333,1,128;...)` | `tc_decomposition_mismatch` | MLA KV 2D (kv_lora_rank=512) vs 标准 KV cache 3D |

#### C.3.2 MLA/MLAPO Composite & Quantize

| # | TC Op | NPU Kernel | TC Shape | CSV 最近似 | 根因 | 分析 |
|---|-------|-----------|----------|-----------|------|------|
| 6 | `quantize` (MLA) | AscendQuantV2 | `(8,16,128)` 3D | `(8,2048)` **TC 修复后可匹配** | `tc_decomposition_mismatch` | TC quantize 保留 3D per-head，NPU reshape 到 2D 后 quantize |
| 7 | `quantize` (shared FFN) | AscendQuantV2 | `(1,8,2304)` | 无 D=2304 | `shape_coverage_gap` | D=2304=18432/8(TP)，共享专家 FFN |
| 8 | `quantize` (MoE) | AscendQuantV2 | `(256,2048)` | `(8,2048)` | `shape_coverage_gap` | M=256 (全 expert)，CSV 仅 M=8 |

#### C.3.3 Shared Expert Path

| # | TC Op | NPU Kernel | TC Shape | CSV 最近似 | 根因 | 分析 |
|---|-------|-----------|----------|-----------|------|------|
| 9 | `add_rms_norm2` | AddRmsNormQuant | `(1,8,7168),(8,7168),(7168,)` | 无 CSV | `shape_coverage_gap` | W8A8 产生 AddRmsNormQuant，该 kernel 无 CSV |
| 10 | `add_rms_norm2` | AddRmsNormBias | `(1,8,7168),(8,7168),(7168,)` | `(1,7168;...)` | `shape_coverage_gap` | M=8 无 CSV |
| 11 | `copy_` | TensorMove | `(8,7168)` | `(256,7168)` | `shape_coverage_gap` | M=8 无 CSV |

#### C.3.4 MoE Routing 辅助 Ops (12 个 MISS)

`sigmoid`, `topk`×3, `sum`×3, `scatter`, `bitwise_not`, `where`, `gather`, `div` — 均为 MoE expert routing 逻辑产生的辅助 op。大部分无 CSV（`shape_coverage_gap`），少数为 `input_count_mismatch`（tc_input_count 未配置）。shape 均含 `n_experts=256` 维度。

#### C.3.5 DFC 融合组 (5 个 MISS)

`permute_tokens`, `grouped_matmul`, `cat`(256×expert), `unpermute_tokens` — 均为 `fused_kernel_gap`。NPU 将整个 MoE dispatch+compute+combine 融合为 DispatchFFNCombine kernel，TC 逐 op 模拟。

#### C.3.6 Post-MoE & Embedding

`add.Tensor`×3, `index.Tensor`, `mm`(lm_head M=8), `copy_`(MLA KV 3D) — 主要为 `input_count_mismatch` 和 `shape_coverage_gap`（decode M=8 无 CSV）。

---

### C.4 DSv3 Prefill (M1=60.2%, M2=38.6%, M3=12.9%)

与 Decode 大部分相同（MoE routing、DFC 完全一致）。Prefill 特有差异:

- `embedding` GatherV2: indices M=256 无 CSV（Decode M=8 也无）→ `shape_coverage_gap`
- `rms_norm`: M=256 在 CSV 中存在 → **HIT**（Decode M=8 MISS）
- `quantize` (MLA): shape `(256,16,128)` → 同 Decode 根因
- `static_quant_linear` (shared expert): N=4608 vs CSV N=4096 → `shape_coverage_gap`
- `swiglu` (shared expert): concat→`(256,4608)` vs CSV `(256,4096)` → `shape_coverage_gap`
- `add_rms_norm2` (FFN 层): M=256 匹配但 input 数为 4 vs TC 3 → `input_count_mismatch`
- `mm` (lm_head): M=1 → **HIT**（Decode M=8 MISS）

### C.5 跨场景 MISS 根因汇总

| 根因 | Qwen3 Decode | Qwen3 Prefill | DSv3 Decode | DSv3 Prefill |
|------|-------------|--------------|------------|-------------|
| `data_format_gap` | 2 (FIA+RoPE) | 2 (FIA+RoPE) | 0 | 0 |
| `shape_coverage_gap` | 0 | 1 | 18 | 17 |
| `tc_decomposition_mismatch` | 1 | 1 | 2 | 3 |
| `input_count_mismatch` | 4 | 4 | 8 | 8 |
| `fused_kernel_gap` | 0 | 0 | 5 | 5 |
| `sp_modeling_gap` | 0 | 2 | 0 | 0 |
| `structural_miss` | 2 | 2 | 2 | 2 |

### C.6 解决优先级矩阵

| 优先级 | 根因 | 影响场景 | 预期 M3 提升 | 工作量 |
|--------|------|---------|-------------|--------|
| **P0** | `fused_kernel_gap` (DFC) | DSv3 全场景 | +15-30pp | TC fusion pass (1-2 周) |
| **P0** | `data_format_gap` (FIA) | Qwen3 全场景 | +5-10pp | FIA microbench CSV |
| **P1** | `shape_coverage_gap` (routing ops) | DSv3 全场景 | +3-5pp | Microbench 采集 |
| **P1** | `shape_coverage_gap` (quantize/norm M×D) | DSv3 全场景 | +3-5pp | Microbench 网格扩展 |
| **P1** | `data_format_gap` (RoPE dtype) | Qwen3 Prefill | +1-2pp | dtype 宽松匹配 |
| **P1** | `input_count_mismatch` | 全场景 | +1-3pp | op_mapping 配置 |
| **P1** | `sp_modeling_gap` | Qwen3 Prefill | +1-2pp | TC 主分支修复 |
| **P2** | `tc_decomposition_mismatch` (KV cache) | 全场景 | +1pp | TC 建模层 |
| **P3** | `structural_miss` | 全场景 | <0.5pp | zero_cost 标记 |
