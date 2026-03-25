# FIA KV 维度分析报告

**版本**: v0.2
**日期**: 2026-03-21
**负责人**: ZH
**前置文档**: `FIA_LOOKUP_AND_MLA_RESEARCH_20260321.md`（同目录）

---

## 1. 研究目标

验证 CANN 8.5 profiling 中 `FusedInferAttentionScore`（FIA）的 KV 维度是否反映真实 sequence length，以此决定 FIA lookup 方案选型。

本报告基于两项并行研究：
1. **编译路径分析**：从 vllm-ascend → pytorch-npu → ops-transformer 追溯 KV tensor 传递链
2. **Profiling 数据验证**：从 CANN 8.5 实际 profiling 数据中提取 FIA 行，分析 KV shape 变化规律

---

## 2. Profiling 数据来源

### 2.1 数据目录结构

profiling 数据位于 `E:\work\project\code\profiling`，包含两组模型的 CANN 8.5 profiling 数据。本次分析仅从各场景的 `kernel_details.csv` 中提取 `FusedInferAttentionScore` 行，未读取 step_trace、timeline、memory 等其他产物。

### 2.2 Qwen3-32B（`profiler-qwen3-0314/`）

配置：CANN 8.5, vLLM 0.15.0, TP=16, BF16, block_size=128, max_num_batched_tokens=65536

| 场景 | 类型 | FIA 行数 | Q shape 特征 |
|------|------|---------|-------------|
| input1024-output1 | Prefill-only | 有 | Q=(65536,4,128) |
| input2048-output1 | Prefill-only | 有 | Q=(65536,4,128) |
| input4096-output1 | Prefill-only（chunked） | 有 | Q 从 (2064,4,128) 到 (8224,4,128) 逐 chunk 增长 |
| input8192-output1 | Prefill-only（chunked） | 有 | Q 从 (14352,4,128) 到 (65536,4,128) |
| input4096-output1536-c1 | Prefill+Decode | 有 | decode 阶段 Q=(16,4,128) |
| input4096-output1536-c4 | Prefill+Decode | 有 | decode 阶段 Q=(16,4,128) |
| input4096-output1536-c16 | Prefill+Decode | 有 | decode 阶段 Q=(16/32,4,128) |
| input4096-output1536-c32 | Prefill+Decode | 有 | decode 阶段 Q=(32,4,128) |

### 2.3 DeepSeek-V3（`profiler-dsv3-0319/`）

配置：CANN 8.5, TP=8, DP=2, EP, W8A8, block_size=128, max_num_seqs=8, max_num_batched_tokens=2048

| 场景 | 类型 | FIA 行数 | 说明 |
|------|------|---------|------|
| input512-output1 | Prefill-only | 有（decode step） | DSV3 prefill 用 RINGMLAPrefillBF16Kernel，FIA 仅出现在 decode step |
| input1024-output1 | Prefill-only | 有（decode step） | 同上 |
| input2048-output1 | Prefill-only | **0** | 仅 RINGMLAPrefillBF16Kernel |
| input4096-output1 | Prefill-only | **0** | 同上 |
| input4096-output1-c1 | Prefill-only | 有 | |
| input2048-output1536-c8 | Prefill+Decode | 有 | MLA decode，Q=(B,16,1,512) |
| input4096-output1536-c1/c2/c4/c8 | Prefill+Decode | 有 | |

> **重要发现**：DSV3 prefill 阶段不使用 FusedInferAttentionScore，而是使用 `RINGMLAPrefillBF16Kernel`。FIA 仅出现在 decode step 中。

### 2.4 提取的字段

从每个场景的 `kernel_details.csv` 中 grep `FusedInferAttentionScore` 行，解析：
- **Input Shapes**（31 个槽位）：重点 slot 0 (Q), slot 1 (K), slot 2 (V), slot 5/6 (actual_seq_lengths), slot 14 (block_table), slot 24/25 (rope)
- **Duration(us)**：每次 FIA 调用的实测耗时
- **Input Data Types / Input Formats**：dtype 和格式信息

---

## 3. 编译路径分析

### 2.1 完整调用链

```
vllm-ascend (attention_v1.py, _get_fia_params())
  ├─ PrefillNoCache:  key/value = 原始输入 tensor（真实长度）
  ├─ PrefillCacheHit: key/value = self.key_cache.view(num_block, block_size, -1)  ← 整个 pool
  ├─ DecodeOnly:      key/value = self.key_cache.view(num_block, block_size, -1)  ← 整个 pool
  └─ ChunkedPrefill:  key/value = self.key_cache.view(num_block, block_size, -1)  ← 整个 pool
  +
  actual_seq_lengths_kv = [seq_len_0, seq_len_1, ...]   ← 标量列表，传递真实长度
  block_table = attn_metadata.block_tables               ← 页表间接寻址
      ↓
pytorch-npu (op-plugin binding)
  参数直传，无 shape 变换
      ↓
ops-transformer (aclnnFusedInferAttentionScore)
  NPU kernel 接收完整 pool + block_table + actual_seq_lengths_kv
  内部通过 block_table 间接寻址 + actual_seq_lengths_kv 限制计算范围
```

### 2.2 关键结论

- NPU kernel 接收的是**完整 KV cache pool**，不是按实际 seq length 切片的 tensor
- `actual_seq_lengths_kv` 作为标量列表传入，kernel 内部用它限制计算范围
- `block_table` 提供页表间接寻址，kernel 通过它访问相关 block
- 唯一例外：`PrefillNoCache` 路径下 key/value 是原始输入（真实长度）

### 2.3 关键源码位置

| 文件 | 行号 | 内容 |
|------|------|------|
| `vllm-ascend/vllm_ascend/attention/attention_v1.py` | 680-719 | `_get_fia_params()` 构造 key/value |
| 同上 | 496-512, 592-608, 813-827 | 各场景调用 FIA，传递 `actual_seq_lengths_kv` |

---

## 4. Profiling 数据验证

### 4.1 核心结论详解

**"CANN 8.3 和 8.5 的 FIA KV 维度都是预分配 pool 大小，不是真实 seq length"**

这句话的含义是：vLLM 调用 `npu_fused_infer_attention_score` 时，传给 NPU kernel 的 key/value tensor 不是"这个 batch 实际用到的 KV 数据"，而是"vLLM 启动时预分配的整个 KV cache 内存池"。

vLLM 启动时根据 `gpu_memory_utilization` 和 `block_size` 预分配一大块 KV cache 内存。例如 Qwen3 TP=16 场景下分配了约 12308 个 block（每个 block 128 token）。之后无论实际请求是 1 条还是 32 条、context 是 100 token 还是 8000 token，传给 FIA kernel 的 key/value shape 始终是 `(12308, 128, 128)` — 这就是 pool 大小。

NPU kernel 通过另外两个参数来知道"实际该算哪些数据"：
- `block_table`（slot 14）：页表，告诉 kernel 每个请求的 KV 数据存在 pool 的哪些 block 里
- `actual_seq_lengths_kv`（slot 6）：标量列表，告诉 kernel 每个请求的真实 KV 长度

因此 profiling 记录的 K/V Input Shapes 是 pool 容量，不携带 seq length 语义信息。

### 4.2 CSV 槽位语义

FIA profiling CSV 按 `torch_npu.npu_fused_infer_attention_score()` 的 tensor 参数展开为 31 个固定槽位：

| 槽位 | 参数名 | 含义 | 是否随 seq length 变化 |
|------|--------|------|----------------------|
| 0 | query | Q tensor | **是** — 反映实际 batch×seq tokens |
| 1 | key | KV cache pool | **否** — 预分配 pool 大小 |
| 2 | value | KV cache pool | **否** — 同上 |
| 5 | actual_seq_lengths | Q 长度列表 | **是** — 标量 |
| 6 | actual_seq_lengths_kv | KV 长度列表 | **是** — 标量 |
| 14 | block_table | 页表 | **是** — (N, max_blocks) |
| 24 | query_rope | MLA rope Q | 仅 MLA 场景 |
| 25 | key_rope | MLA rope K | 仅 MLA 场景 |

### 4.3 实测数据验证

**Qwen3 CANN 8.5（TP=16）— KV shape 跨场景完全不变**：

| 场景 | Q shape (slot 0) | K shape (slot 1) | actual_seq_len (slot 5) | Duration(us) |
|------|-----------------|-----------------|------------------------|-------------|
| prefill input-1024 | 65536,4,128 | **12307**,128,128 | — | ~780-1286 |
| prefill input-2048 | 65536,4,128 | **12307**,128,128 | — | ~780-1286 |
| prefill input-4096 chunk 0 | 2064,4,128 | **12307**,128,128 | 2 | — |
| prefill input-4096 chunk 3 | 8224,4,128 | **12307**,128,128 | 5 | — |
| prefill input-8192 | 65536,4,128 | **12308**,128,128 | — | — |
| decode c=1 | 16,4,128 | **12307**,128,128 | 16 | ~58-64 |
| decode c=32 | 32,4,128 | **12307**,128,128 | 32 | ~58-64 |

Q shape 随负载显著变化（65536 vs 16），但 K/V 始终是 ~12307-12308 — 这就是 pool 的 block 总数。

**DSv3 CANN 8.5（TP=8, MLA 架构）— KV shape 同样固定**：

| 场景 | Q shape (slot 0) | K shape (slot 1) | actual_seq_len_kv (slot 6) | Duration(us) |
|------|-----------------|-----------------|---------------------------|-------------|
| input-512 decode | 2,16,1,512 | **1135**,1,128,512 | 2 | ~28 |
| input-1024 decode | 3,16,1,512 | **1135**,1,128,512 | 3 | ~28-40 |
| input-4096 decode c=4 | 4,16,1,512 | **1135**,1,128,512 | 4 | ~40-50 |
| input-4096 decode c=8 | 5,16,1,512 | **1135**,1,128,512 | 5 | ~50-60 |

K shape 始终 `1135,1,128,512`，1135 是 DSV3 场景下 pool 的 block 总数。

### 4.4 CANN 8.3 vs 8.5 对比

会议纪要（3.20）中提到"8.5 版本 profiling 结果显示 k 和 v 的第零个维度与 actual sequence length 能对上"。

**实测结论：此假设不成立。** CANN 8.3 和 8.5 在 KV 维度上行为一致 — 均为预分配 pool 大小，不反映真实 seq length。以下三个具体例子说明这一结论。

#### 例 1：Qwen3 不同输入长度，KV shape 完全不变

同一模型（Qwen3-32B, TP=16, CANN 8.5），仅改变输入长度：

| 输入长度 | Q shape (slot 0) | K shape (slot 1) | actual_seq_len (slot 5) | Duration |
|---------|-----------------|-----------------|------------------------|---------|
| 1024 tokens | 65536,4,128 | **12308**,128,128 | 66 | 1073us |
| 4096 tokens (chunk 1) | 6176,4,128 | **12307**,128,128 | 4 | 325us |
| 4096 tokens (chunk 5) | 8224,4,128 | **12307**,128,128 | 5 | 402us |
| 8192 tokens | 65536,4,128 | **12307**,128,128 | 11 | 3845us |

输入从 1024 变到 8192，Q shape 和 Duration 都在剧烈变化，但 K 的第一维始终是 **12307-12308** — 这是 vLLM 在该 NPU 上预分配的 KV cache block 总数。如果 K 反映真实 seq length，它应该随输入长度成比例变化，但实际完全不变。

#### 例 2：Qwen3 Prefill vs Decode，KV shape 还是不变

同一次 profiling run（input4096-output1536-c1），prefill 和 decode 阶段的 FIA 调用：

| 阶段 | Q shape | K shape | actual_seq_len | block_table | Duration |
|------|---------|---------|----------------|-------------|---------|
| Prefill (chunked) | 8224,4,128 | **12307**,128,128 | 5 | 5,512 | 402us |
| Decode | 16,4,128 | **12307**,128,128 | 16 | 16,512 | 64us |

Prefill 处理 8224 个 token，Decode 只处理 16 个 token，Q shape 差 500 倍，但 K shape 完全一样。真正区分工作量的是 `actual_seq_len`（5 vs 16）和 `block_table` 第一维（5 vs 16）。

#### 例 3：Non-paged Prefill 是唯一例外 — K 确实是真实长度

CANN 8.5 数据中有两行 non-paged prefill（无 block_table），与 paged decode 对比：

| 场景 | Q shape | K shape | V shape | block_table | Duration |
|------|---------|---------|---------|-------------|---------|
| Non-paged prefill | 41040,4,128 | **41040**,1,128 | **41040**,1,128 | 无 | 1286us |
| Non-paged prefill | 24624,4,128 | **24624**,1,128 | **24624**,1,128 | 无 | 780us |
| Paged decode | 16,4,128 | **12307**,128,128 | **12307**,128,128 | 16,512 | 64us |

Non-paged 场景下 K 的第一维 = Q 的第一维（41040 / 24624），这才是真实 token 数。而 paged 场景下 K 是 12307（pool 大小），和实际负载无关。

**结论**：只有 non-paged prefill 路径的 KV 是真实长度；所有 paged 路径（decode + chunked prefill）的 KV 都是预分配 pool 大小。

---

## 5. 对方案选型的影响

### 5.1 方案 A（Raw CSV 直接匹配）可行性评估

原始方案 A 设想：从 TC args 提取 query/key/value shape，映射到 CSV 31 槽位做 shape 匹配。

**问题**：KV shape 是 pool 容量，不携带 seq length 语义。直接匹配 KV shape 退化为"按部署配置匹配"而非"按性能特征匹配"。

**修正后的方案 A**：仍用 raw CSV 存储，但匹配逻辑需 FIA 专属规则：

| 匹配维度 | 来源 | 匹配方式 |
|----------|------|----------|
| Q shape (slot 0) | TC args[0] query | 精确匹配（需 2D→3D 转换） |
| actual_seq_lengths (slot 5/6) | TC args[6] seq_lens | 标量匹配 |
| block_table shape (slot 14) | TC args[4] block_table | 宽松匹配（batch_size 维度） |
| K/V shape (slot 1/2) | — | **忽略或宽松匹配**（pool 大小无语义） |
| dtype | TC query dtype | 精确匹配 |

### 5.2 三种 Case 的匹配策略

| Case | 场景 | Q shape | KV 匹配 | 关键匹配维度 |
|------|------|---------|---------|-------------|
| A | Qwen3 Decode (Paged TND) | 3D (T,N,D) | 忽略 pool shape | Q + actual_seq_lens + block_table |
| B | Qwen3 Prefill (Non-paged TND) | 3D (T,N,D) | 可精确匹配（真实长度） | Q + K/V（真实长度） |
| C | DSV3 MLA Decode (BNSD + rope) | 4D (B,N,S,D) | 忽略 pool shape | Q + actual_seq_lens + rope tensors |

### 5.3 Decode 场景精度风险

从 Qwen3 CANN 8.5 数据（Case A，18 行）：
- Q 首维变化范围：128-496
- actual_seq_lengths 变化：126-129
- Duration 范围：58.2-64.2us（std 1.4-4.2us）

**结论**：Decode 场景 FIA 延迟对 Q token 数和 context length 都不敏感，匹配精度风险可控。

### 5.4 方案对比更新

| 维度 | A（修正版）: Raw CSV + FIA 专属规则 | B: Hybrid | C: 仅结构化 |
|------|-----------------------------------|-----------|------------|
| 短期可用性 | 立即可用 | 精确匹配立即可用 | 阻塞等 microbench |
| KV 维度处理 | 忽略/宽松匹配 | 同 A | 不涉及（用 avg_seq_len） |
| 实现复杂度 | 中（FIA 专属匹配规则） | 高 | 低 |
| 插值能力 | 弱（raw shape 多维） | 强 | 强 |
| 降级兜底 | 有 | 有 | 无 |

---

## 6. 与 3.20 会议结论的冲突分析

以下逐条列出本次研究发现与 3.20 会议纪要（`FIA lookup.docx`）中结论的冲突点及解决方案。

### 冲突 1（最关键）：8.5 版本 KV 维度能否对上

**会议原文**：
> "8.5 版本 profiling 结果显示，k 和 v 的第零个维度与输入的 actual sequence length 能对上"

**实测结论**：不成立。CANN 8.5 的 paged 场景下 K/V 第零维始终是 pool 大小（12307-12308），和 actual sequence length 无关。只有 non-paged prefill 例外。

**原因分析**：会议中这个判断可能来自对 non-paged prefill 行的观察（那两行确实 K dim0 = Q dim0），但被错误泛化到了所有场景。

**解决方案**：需在后续沟通中纠正：8.5 和 8.3 行为一致，paged 路径的 KV 都是 pool 大小。

### 冲突 2：方案 1 的前提条件

**会议原文**：
> "方案1，使用raw CSV格式（如果profiling数据准确）"

会议将"profiling 数据准确"等同于"KV 维度是真实 seq length"。现在验证结果是 KV 维度不是真实 seq length，按会议的二分逻辑应该走方案 2（自定义格式）。

**解决方案**：方案 1（raw CSV）仍然可行，但前提条件需要修正。不是"KV 维度准确就能用 raw CSV"，而是"用 raw CSV 存储 + FIA 专属匹配规则（忽略 KV pool shape，重点匹配 Q shape + actual_seq_lengths）"。这是方案 1 的修正版，既保留了 raw CSV 的优势（立即可用、不依赖 microbench），又绕开了 KV 维度不可靠的问题。

### 冲突 3：合并 lookup_attention 和普通 lookup 的可行性

**会议原文**：
> "考虑将 lookup attention 和 lookup 正常算子合到一起，保持 CSV 结构为 profiling 原始结构"

会议设想的合并方案是：FIA 和普通算子共用 `_lookup_compute()` + `_inputs_match()` 的匹配路径。

**实测结论**：无法直接合并。原因：
- 普通算子的 `_inputs_match()` 匹配所有 input shape，但 FIA 的 31 个槽位中 KV shape 无语义、actual_seq_lengths 是标量不是 tensor shape
- FIA 需要专门的槽位选择逻辑（只匹配 slot 0/5/6/14，忽略 slot 1/2）
- Non-paged 和 paged 两种 case 的匹配策略不同

**解决方案**：保留 raw CSV 统一存储格式（不需要两套 CSV），但查询路径仍需 FIA 专属方法 `_lookup_attention_raw()`，内部实现 slot 选择 + paged/non-paged 分支逻辑。存储统一，查询分离。

### 冲突 4：编译路径分析的预期结论

**会议原文**：
> "查看是否存在使用 blocksize 而非 sequence lens，或者 sequence lens 丢失的情况"

会议预期编译路径分析可能发现"某个环节用了 blocksize 替代了 sequence lens"，暗示这是一个 bug 或可修复的问题。

**实测结论**：这不是 bug，而是 by design。vLLM 的 PagedAttention 机制就是传整个 pool + block_table 间接寻址，`actual_seq_lengths_kv` 作为标量参数传递。profiling 记录的 Input Shapes 只包含 tensor shape，标量参数不体现在 shape 中。

**解决方案**：不需要修复编译路径，而是在匹配逻辑中适配这个设计 — 从 CSV 的 slot 5/6 提取 actual_seq_lengths 标量值作为匹配维度。

### 冲突 5：验证通过后的操作路径

**会议原文**：
> "若符合预期，采用 raw CSV 格式，对应修改 lookup attention 和 LOOKUP interpolation"
> "修改完 CSV 后，让TCX采集 microbench数据，同时修改查询接口匹配"

会议假设验证通过后是一条简单的路径：raw CSV → 修改 lookup → TCX 采集 microbench。

**实测结论**：验证结果是"部分通过"（non-paged 可以，paged 不行），操作路径比预期复杂：
- raw CSV 可用，但需要 FIA 专属匹配规则
- TCX 的 microbench（E4 任务）优先级可以降低（raw CSV 已能覆盖 profiling 数据），但长期仍需要结构化 CSV 做插值
- 插值路径需要重新设计 — raw CSV 的多维 shape 不适合直接做 1D 插值

**解决方案**：分两步走：
1. 短期（3.23 交付）：实现 `_lookup_attention_raw()` 做精确匹配，覆盖 profiling 已有数据
2. 长期：TCX 采集结构化 microbench CSV，用于 `InterpolatingDataSource` 的 FIA 插值路径

---

## 7. 待决策事项

1. **是否采用修正版方案 A**：用 raw CSV 存储 + FIA 专属匹配规则（忽略 KV pool shape，重点匹配 Q + actual_seq_lens）
2. **MLA composite 分解中 FIA 子内核查询如何适配**：需从 MLA args 推导出 FIA 的 Q shape 和 actual_seq_lens
3. **是否通知 TCX 调整 E4 任务**：如果方案 A 可满足短期需求，E4（FIA microbench）可降优先级

---

## 8. 下一步计划

| 任务 | 优先级 | 依赖 |
|------|--------|------|
| 与 HXW 确认方案选型 | P0 | 本报告 |
| 实现 FIA raw CSV 匹配（修正版方案 A） | P0 | 方案确定 |
| G2 MLA 分解完善 + FIA 子内核适配 | P0 | FIA 匹配实现 |
| C11-3 DFC composite 验证 | P1 | — |
| Phase 3 端到端验证（J1/J3） | P0 | 上述任务完成 |
