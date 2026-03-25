# FIA Lookup + MLA 分解研究报告

**日期**: 2026-03-21
**负责人**: ZH
**分支**: feat/interpolating-datasource（已合并 origin/feat/perf-database 最新 8 个 commit）
**更新**: v0.2 — 补充 KV 维度验证结论（详见 `FIA_KV_DIMENSION_ANALYSIS_v0.1.md`）

---

## 1. 背景

当前 `_lookup_attention()` 要求 microbench 结构化 CSV（`batch_size, avg_seq_len, num_heads, head_dim` 列），但实际 `FusedInferAttentionScore.csv` 是 raw profiling 格式（31 个 tensor 槽位），导致每次查询返回 `csv_format_raw` MISS。

本次研究目标：
1. 评估能否直接用 raw CSV 做 FIA lookup
2. 结合 MLA composite 分解完善一起设计方案

---

## 2. 现有代码分析

### 2.1 _lookup_attention() 现状

- 从 TC args 提取 5 个维度：`batch_size`（seq_lens.shape[0]）、`avg_seq_len`（seq_lens 均值）、`num_heads`（hidden_size/head_dim）、`head_dim`（key.shape[-1]）、`dtype`
- 要求 CSV 包含 `{batch_size, avg_seq_len, num_heads, head_dim}` 列
- Raw CSV 没有这些列 → 始终返回 `csv_format_raw` MISS

### 2.2 _lookup_composite() MLA 分解现状

已注册 4 个 decomposer：
- `tensor_cast.multihead_latent_attention.default` → `_decompose_mla`
- `tensor_cast.multihead_latent_attention_quant.default` → `_decompose_mla_quant`
- `tensor_cast.mlapo.default` → `_decompose_mlapo`
- `tensor_cast.mlapo_quant.default` → `_decompose_mlapo_quant`

MLA 分解逻辑：
- **Decode**: TransposeBatchMatMul(q@W_UK_T) + FIA + TransposeBatchMatMul(out@W_UV)，共 3 个子内核
- **Prefill**: MatMulV2(kv_c_normed@kv_b_proj) + FIA，共 2 个子内核
- FIA 子内核通过 `_lookup_attention_by_params()` 查询，同样依赖结构化 CSV → 同样 MISS

### 2.3 _inputs_match() shape 匹配能力

已支持：batch dim strip、FRACTAL_NZ 恢复、block-padding tolerance、matmul 转置、SwiGlu 合并、RoPE 归一化、3D→2D flatten。
不支持：FIA 的 31 槽位映射。

## 3. FIA Raw CSV 数据分析

### 3.1 CSV 结构

CSV 按 `torch_npu.npu_fused_infer_attention_score()` 的 tensor 参数顺序展开为 31 个固定槽位。标量参数（`num_heads`, `scale`, `input_layout`, `sparse_mode`, `block_size`）不在其中。

关键槽位：
| Index | 参数名 | 说明 |
|---|---|---|
| 0 | query | 主输入 Q |
| 1 | key | 主输入 K（paged 场景为 KV cache pool） |
| 2 | value | 主输入 V |
| 4 | atten_mask | attention mask |
| 5 | actual_seq_lengths | Q 长度列表 |
| 6 | actual_seq_lengths_kv | KV 长度列表 |
| 14 | block_table | page attention block 映射表 |
| 24 | query_rope | MLA rope 场景 |
| 25 | key_rope | MLA rope 场景 |

### 3.2 三种 Case 分类（CANN 8.5 数据，24 行）

**Case A: Qwen3 Decode（Paged TND）** — 数量最多
- query: 3D `(T, N, D)` 如 `(336,4,128)`，T = num_tokens（batch 内所有 query token 拼接）
- key/value: 3D `(block_num, block_size, D)` 如 `(12308,128,128)` — KV cache pool 总容量
- 有 block_table、actual_seq_lengths、actual_seq_lengths_kv
- Duration: **~58-64us，非常稳定**（std ~1-2us）
- 关键观察：不同 query 首维（128-496）和不同 actual_seq_lengths（126-129）下 duration 变化极小

**Case B: Qwen3 Prefill（Non-paged TND）** — 少量
- query: 3D `(T, N, D)` 如 `(41040,4,128)`
- key/value: 3D `(T_kv, KV_N, D)` 如 `(41040,1,128)` — 真实 KV 长度
- 无 block_table
- Duration: **~780-1286us，随 T 显著变化**

**Case C: DSV3 MLA Decode（BNSD + rope）** — 少量
- query: 4D `(B, N, S, D)` 如 `(4,16,1,512)`
- key/value: 4D `(block_num, KV_N, block_size, D)` 如 `(1135,1,128,512)`（CANN 8.5）/ `(892,1,128,512)`（CANN 8.3）— KV cache pool 总容量
- 有 query_rope `(B,N,S,64)` 和 key_rope `(block_num,KV_N,block_size,64)`
- Duration: **~28-60us**

> **v0.2 补充**：编译路径分析和 profiling 数据验证确认，Case A 和 Case C 的 KV shape 均为预分配 pool 大小（by design），不是真实 seq length。只有 Case B（non-paged prefill）的 KV 是真实长度。此外，chunked prefill 走 paged 路径，KV 同样是 pool 大小。详见 `FIA_KV_DIMENSION_ANALYSIS_v0.1.md` §4。

### 3.3 TC Args vs CSV 槽位映射关系

**tensor_cast.attention（Qwen3 GQA）**:
| TC args | CSV slot | 差异 |
|---|---|---|
| args[0] query (num_tokens, hidden_size) | slot[0] query (T, N, D) | TC 是 2D，CSV 是 3D（hidden_size = N×D） |
| args[1] key (total_blocks, block_size, kv_heads, head_dim) | slot[1] key (block_num, block_size, D) | TC 是 4D，CSV 是 3D（D = kv_heads×head_dim） |
| args[6] seq_lens (batch_size,) | slot[5]/slot[6] actual_seq_lengths | TC 是 tensor，CSV 是标量 shape |
| args[4] block_table | slot[14] block_table | shape 可匹配 |

**tensor_cast.multihead_latent_attention（DSV3 MLA）**:
| TC args | CSV slot | 差异 |
|---|---|---|
| args[0] q (num_tokens, num_heads, qk_head_dim) | slot[0] query (B, N, S, D) | TC 是 3D，CSV 是 4D（BNSD 布局） |
| args[1] kv_cache (total_blocks, block_size, kv_lora_rank+rope_dim) | slot[1] key (block_num, KV_N, block_size, D) | 维度顺序不同 |
| args[4] seq_lens | slot[6] actual_seq_lengths_kv | |
| — | slot[24]/slot[25] query_rope/key_rope | TC MLA op 不传 rope tensors |

## 4. 三种 FIA Lookup 方案对比

### 方案 A：Raw CSV 直接匹配（推荐）

新写 `_lookup_attention_raw()` 方法，从 TC args 提取关键 tensor（query/key/value/seq_lens/block_table），映射到 CSV 31 槽位中对应位置做 shape 匹配。

**优点**：
- 不需要额外 microbench 数据采集，现有 profiling CSV 立即可用
- 覆盖率高 — profiling 跑过的所有 FIA 调用都有数据
- CSV 存储格式与普通算子统一（raw profiling 格式）

**缺点/风险**：
- TC args 和 CSV 31 槽位之间需要写专门的映射逻辑
- ~~实现路径和 `_lookup_compute()` 一致，代码复用度高~~ **v0.2 修正**：无法直接复用 `_inputs_match()`，因为 KV slot 需要忽略、actual_seq_lengths 是标量。需要 FIA 专属匹配逻辑（存储统一，查询分离）
- Paged 场景（decode + chunked prefill）：key/value shape 是 KV cache pool 总容量，不反映真实 context length。但从数据看 decode duration 变化很小（std ~1-2us on ~60us），风险可控。**v0.2 补充**：匹配策略应忽略 KV shape，重点匹配 Q shape + actual_seq_lengths（slot 5/6）
- Non-paged prefill 场景无此问题：query/key/value shape 直接反映真实 token 数
- 插值较困难 — raw shape 是多维的，不像结构化 CSV 有明确插值维度

### 方案 B：Hybrid（Raw 精确 + 结构化插值）

精确匹配走 raw CSV，插值走结构化 CSV（`batch_size, avg_seq_len, num_heads, head_dim`）。

**优点**：
- 精确匹配覆盖率高（复用 profiling 数据）
- 插值精度好（结构化维度上做 1D 插值，语义清晰）

**缺点**：
- 需要维护两套 CSV 格式，代码复杂度高
- 结构化 CSV 仍依赖 microbench 数据采集（TCX E4 任务），短期内插值路径不可用

### 方案 C：仅结构化 CSV（保持现有设计）

保持 design doc 原方案，等 microbench 数据就绪后用结构化 CSV。

**优点**：
- 设计最干净，插值语义明确
- 长期最优方案

**缺点/风险**：
- **短期内 FIA 全部 MISS** — 没有 microbench 数据就完全不工作
- 依赖 TCX E4 任务完成（FIA microbench 构造 paged KV cache 输入复杂度高）
- MLA 场景需要额外覆盖 4D BNSD 布局的 FIA 调用模式，增加 E4 工作量
- **没有降级兜底** — 如果 microbench 数据 3.23 前到不了位，FIA 在最终验证中就是黑洞

### 对比总结

| 维度 | A: Raw 直接 | B: Hybrid | C: 仅结构化 |
|------|-----------|-----------|------------|
| 短期可用性 | 立即可用 | 精确匹配立即可用 | 阻塞等 microbench |
| 实现复杂度 | 中 | 高 | 低 |
| 精确匹配覆盖率 | 高 | 高 | 低 |
| 插值能力 | 弱 | 强 | 强 |
| Decode 精度风险 | 中（可控） | 低 | 低 |
| 长期维护成本 | 低 | 高 | 低 |
| 降级兜底 | 有 | 有 | 无 |

---

## 5. MLA 分解与 FIA Lookup 的关联

MLA composite 分解中，FIA 是关键子内核。当前 `_decompose_mla()` 生成 FIA 的 `SubKernelSpec` 时使用 `query_mode="attention"` + `attention_params`，走 `_lookup_attention_by_params()` 查询结构化 CSV → 同样 MISS。

如果选择方案 A（Raw CSV 直接匹配），MLA 的 FIA 子内核查询需要适配：
- MLA decode 的 FIA 是 Case C（4D BNSD + rope），需要从 MLA args 推导出 CANN FIA 的输入 shape
- ~~MLA prefill 的 FIA 是 Case B（non-paged TND），shape 推导相对直接~~ **v0.2 修正**：DSV3 prefill 阶段不使用 FusedInferAttentionScore，而是使用 `RINGMLAPrefillBF16Kernel`。FIA 仅出现在 DSV3 decode step 中。因此 MLA composite 分解的 prefill 路径中 FIA 子内核查询实际不会被触发（或应跳过）
- 需要新增 `_lookup_attention_raw_by_shapes()` 方法供 composite 分解调用

---

## 6. 关键数据观察

### Decode 场景 Duration 稳定性

从 CANN 8.5 CSV 数据（Case A，18 行）：
- query 首维变化范围：128-496（不同 batch token 数）
- actual_seq_lengths 变化：126-129
- Duration 范围：**58.2-64.2us**，std 1.4-4.2us
- **结论**：decode 场景下 FIA 延迟对 query token 数和 context length 都不敏感，raw shape 匹配精度风险可控

### Prefill 场景 Duration 变化

从 CANN 8.5 CSV 数据（Case B，2 行）：
- T=24624 → 780us，T=41040 → 1286us
- **结论**：prefill 延迟随 token 数显著变化，但 raw shape 可以精确匹配（key/value 是真实长度）

### MLA Decode 场景

从 CANN 8.5 CSV 数据（Case C，2 行）：
- batch_size=4 → 28us，batch_size=5 → 60us
- **结论**：MLA decode 延迟对 batch_size 敏感，需要精确匹配

---

## 7. 待决策事项

1. **FIA lookup 方案选择**：~~A / B / C，建议方案 A~~ **v0.2 更新**：KV 维度验证已完成，建议采用修正版方案 A（raw CSV 存储 + FIA 专属匹配规则，忽略 KV pool shape，重点匹配 Q + actual_seq_lens）
2. **MLA FIA 子内核查询适配**：~~取决于方案选择~~ **v0.2 更新**：仅需适配 decode 路径（Case C），DSV3 prefill 不走 FIA
3. **是否需要通知 TCX 调整 E4 任务优先级**：如果选方案 A，E4 可降优先级；长期仍需结构化 CSV 做插值

---

## 8. 下一步计划

| 任务 | 优先级 | 依赖 | v0.2 状态 |
|------|--------|------|----------|
| ~~确定 FIA lookup 方案~~ | ~~P0~~ | ~~本报告~~ | 验证已完成，建议修正版方案 A |
| 实现 `_lookup_attention_raw()`（修正版方案 A） | P0 | 与 HXW 确认 | 待实现 |
| G2 MLA 分解完善（仅 decode 路径） | P0 | FIA 匹配实现 | 待实现 |
| C11-3 DFC composite 验证 | P1 | — | 待验证 |
| Phase 3 端到端验证（J1/J3） | P0 | 上述任务完成 | — |
