# MoE / MLA 查询接口设计文档

**版本**：v1.0（草稿）
**作者**：ZH
**日期**：2026-03-12
**关联文档**：`OPERATOR_PERF_DATABASE_DESIGN_zh_v1.2.md` §4.2、`COMM_QUERY_DESIGN.md`
**关联任务**：G1（MoE 算子匹配，截止 3.18）、G2（MLA 分解完整实现，截止 3.20）

---

## 1. 概述

本文档描述 `ProfilingDataSource` 对 MoE 路由算子和 MLA 注意力算子的查询逻辑设计。

### 1.1 算子分类

| 类别 | TC 算子 | NPU Kernel | 查询路径 | 任务 |
|------|---------|-----------|---------|------|
| MoE 路由 | `permute_tokens` | `MoeDistributeDispatchV2` | compute（标准 shape 匹配） | G1 |
| MoE 路由 | `unpermute_tokens` | `MoeDistributeCombineV2` | compute（标准 shape 匹配） | G1 |
| MoE 路由 | `moe_gating_topk` | `MoeGatingTopK` | compute（标准 shape 匹配） | G1 |
| MLA attention | `multihead_latent_attention` | TransposeBatchMatMul + FusedInferAttentionScore | composite（B2 占位，G2 完整实现） | G2 |
| MLA attention | `multihead_latent_attention_quant` | TransposeBatchMatMul + FusedInferAttentionScore | composite（B2 占位，G2 完整实现） | G2 |
| MLA prolog | `mlapo` | MatMulV2 + KvRmsNormRopeCache | composite（G2） | G2 |
| MLA prolog | `mlapo_quant` | QuantBatchMatmulV3 + KvRmsNormRopeCache | composite（G2） | G2 |

---

## 2. MoE 算子查询设计（G1）

### 2.1 TC 算子签名

**文件**：`tensor_cast/ops/fused_moe.py`

```python
@register_tensor_cast_op("permute_tokens")
def _(
    x: torch.Tensor,           # args[0]: (num_tokens, hidden_size)
    topk_indices: torch.Tensor, # args[1]: (num_tokens, top_k)
) -> torch.Tensor:
    # 返回: (num_tokens * top_k, hidden_size)

@register_tensor_cast_op("unpermute_tokens")
def _(
    x: torch.Tensor,           # args[0]: (num_tokens * top_k, hidden_size)
    topk_indices: torch.Tensor, # args[1]: (num_tokens, top_k)
) -> torch.Tensor:
    # 返回: (num_tokens, top_k, hidden_size)
```

> `moe_gating_topk` 当前未在 develop 分支注册为 TC 算子（TC 用 `aten.topk` 实现路由）。
> op_mapping.yaml 中有 `profiling.MoeGatingTopK` 占位条目，G1 暂不实现其查询逻辑。

### 2.2 CSV 格式

现有 CSV（`vllm0.13.0_torch2.8.0_cann8.3/` 目录）为**原始 Profiling 格式**，包含完整的 NPU 性能计数器列：

```
OP State, Accelerator Core, Input Shapes, Input Data Types, Input Formats,
Output Shapes, Output Data Types, Output Formats, Average Duration(us), ...
```

这与标准 compute 查询路径（`_lookup_compute` → `_inputs_match`）**完全兼容**，无需特殊处理。

#### MoeDistributeDispatchV2 CSV 示例

```
Input Shapes: "3,7168;3,8;;3;3,8;"
Input Data Types: DT_BF16;INT32;DT_UNDEFINED;BOOL;FLOAT;DT_UNDEFINED
Average Duration(us): 1537.95
```

- `args[0]` = `x`：shape `(num_tokens, hidden_size)` = `(3, 7168)`
- `args[1]` = `topk_indices`：shape `(num_tokens, top_k)` = `(3, 8)`
- 其余输入（`;;3;3,8;`）为 NPU 内部参数，TC 不传递

#### MoeDistributeCombineV2 CSV 示例

```
Input Shapes: "384,7168;3,8;49152;1792;3,8;1;3;..."
Average Duration(us): 164.38
```

- `args[0]` = `x`：shape `(num_tokens * top_k, hidden_size)` = `(384, 7168)`
- `args[1]` = `topk_indices`：shape `(num_tokens, top_k)` = `(3, 8)`

#### MoeGatingTopK CSV 示例

```
Input Shapes: "3,256;256"
Input Data Types: DT_BF16;DT_BF16
Average Duration(us): 6.94
```

- `args[0]` = logits：shape `(num_tokens, num_experts)` = `(3, 256)`
- `args[1]` = expert_bias：shape `(num_experts,)` = `(256,)`

### 2.3 Shape 匹配挑战

MoE CSV 的 Input Shapes 包含 TC 不传递的 NPU 内部参数（空字段 `;;`），导致 `_inputs_match` 中 `len(tc_inputs) != len(csv_shapes)` 直接返回 False。

**解决方案**：在 `_inputs_match` 中增加 MoE kernel 的特殊处理，或在 `_lookup_compute` 中对 MoE kernel 做 shape 前缀匹配（只比较 TC 传递的前 N 个输入）。

具体策略：

| 方案 | 描述 | 优缺点 |
|------|------|--------|
| A：前缀匹配 | 只比较 TC 输入数量对应的前 N 个 CSV shape | 简单，但可能误匹配 |
| B：op_mapping 配置 `tc_input_count` | 在 op_mapping.yaml 中声明 TC 侧输入数量，匹配时截断 CSV shapes | 显式，可扩展 |
| C：microbenchmark CSV | 重新采集只含 TC 输入的 microbenchmark 格式 CSV | 最干净，但需要数据采集 |

**推荐方案 B**：在 op_mapping.yaml 中为 MoE kernel 添加 `tc_input_count` 字段，`_inputs_match` 读取该字段后截断 CSV shapes 再比较。

### 2.4 op_mapping.yaml 配置（建议）

```yaml
"tensor_cast.permute_tokens.default":
  kernel_type: MoeDistributeDispatchV2
  tc_input_count: 2   # TC 只传 x + topk_indices，CSV 有更多 NPU 内部参数

"tensor_cast.unpermute_tokens.default":
  kernel_type: MoeDistributeCombineV2
  tc_input_count: 2   # TC 只传 x + topk_indices
```

### 2.5 查询流程

```
_lookup_compute(op_invoke_info, mapping)
  │
  ├─ 读取 mapping.get("tc_input_count")
  │    若存在 → tc_inputs = tc_inputs[:tc_input_count]
  │
  ├─ _load_csv("MoeDistributeDispatchV2")
  │
  └─ _inputs_match(tc_inputs[:N], row, kernel_type)
       ├─ csv_shapes = _parse_shape_str(row["Input Shapes"])[:N]  ← 截断
       └─ 逐维比较（含 block-padding 容忍）
```

### 2.6 miss_reason

| miss_reason | 含义 |
|-------------|------|
| `csv_not_found` | MoE CSV 文件不存在 |
| `shape_mismatch` | num_tokens / hidden_size / top_k 不匹配 |
| `input_count_mismatch` | tc_input_count 配置错误 |

---

## 3. MLA 算子查询设计（G2）

### 3.1 TC 算子签名

**文件**：`tensor_cast/ops/mla.py`

#### multihead_latent_attention

```python
@register_tensor_cast_op("multihead_latent_attention")
def _(
    q: torch.Tensor,                    # args[0]: (num_tokens, num_heads, qk_head_dim)
    kv_cache: torch.Tensor,             # args[1]: (total_blocks, block_size, kv_lora_rank + qk_rope_head_dim)
    block_table: torch.Tensor,          # args[2]: (batch_size, max_blocks_per_seq)
    query_start_loc: torch.Tensor,      # args[3]: (batch_size + 1,)
    seq_lens: torch.Tensor,             # args[4]: (batch_size,) — KV 序列长度
    query_lens: Optional[torch.Tensor], # args[5]: (batch_size,) — query 长度，None 时为 decode
    W_UK_T: Optional[torch.Tensor],     # args[6]: (num_heads, qk_nope_head_dim, kv_lora_rank)，decode 专用
    W_UV: Optional[torch.Tensor],       # args[7]: (num_heads, kv_lora_rank, v_head_dim)，decode 专用
    kv_b_proj: Optional[torch.Tensor],  # args[8]: (kv_lora_rank, num_heads*(qk_nope_head_dim+v_head_dim))，prefill 专用
    v_head_dim: int,                    # args[9]: scalar
) -> torch.Tensor:
    # 返回: (num_tokens, num_heads, v_head_dim)
```

#### mlapo（MLA Prolog）

```python
@register_tensor_cast_op("mlapo")
def _(
    hidden_states: torch.Tensor,        # args[0]: (num_tokens, hidden_size)
    cos: torch.Tensor,                  # args[1]: (1, seq_len, qk_rope_head_dim)
    sin: torch.Tensor,                  # args[2]: (1, seq_len, qk_rope_head_dim)
    q_a_proj_weight: Optional[...],     # args[3]: (hidden_size, q_lora_rank)
    q_a_layernorm_weight: Optional[...],# args[4]: (q_lora_rank,)
    q_b_proj_weight: Optional[...],     # args[5]: (q_lora_rank, num_heads * qk_head_dim)
    kv_a_proj_weight: Optional[...],    # args[6]: (hidden_size, kv_lora_rank + qk_rope_head_dim)
    kv_a_layernorm_weight: torch.Tensor,# args[7]: (kv_lora_rank,)
    num_heads: int,                     # args[8]
    qk_head_dim: int,                   # args[9]
    qk_nope_head_dim: int,              # args[10]
    qk_rope_head_dim: int,              # args[11]
    kv_lora_rank: int,                  # args[12]
    q_lora_rank: int,                   # args[13]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 返回: (q_states, kv_c_normed, k_rot)
```

### 3.2 Prefill vs Decode 路径判断

`multihead_latent_attention` 通过 `query_lens`（args[5]）区分路径：

| 条件 | 路径 | NPU Kernel 分解 |
|------|------|----------------|
| `query_lens is None` 或全为 1 | **Decode** | TransposeBatchMatMul × 2 + FusedInferAttentionScore |
| `query_lens` 存在且 > 1 | **Prefill** | MatMulV2 + FusedInferAttentionScore |

### 3.3 Decode 路径分解

Decode 计算：`softmax(q @ W_UK_T @ k_cache) @ v_cache @ W_UV`

NPU 上分解为：
1. **TransposeBatchMatMul**（q @ W_UK_T）：`(batch, num_heads, qk_nope_head_dim) @ (num_heads, qk_nope_head_dim, kv_lora_rank)`
2. **FusedInferAttentionScore**：标准 attention，kv_cache 中存储压缩 KV
3. **TransposeBatchMatMul**（attn_out @ W_UV）：`(batch, num_heads, kv_lora_rank) @ (num_heads, kv_lora_rank, v_head_dim)`

查询维度（TransposeBatchMatMul）：
- Input Shapes：`(batch, num_heads, qk_nope_head_dim); (num_heads, qk_nope_head_dim, kv_lora_rank)`
- 从 `args[0]`（q）和 `args[6]`（W_UK_T）提取

查询维度（FusedInferAttentionScore）：
- 走 `attention_special` 路径（batch_size, avg_seq_len, num_heads, head_dim）
- 从 `args[4]`（seq_lens）和 `args[0]`（q）提取

### 3.4 Prefill 路径分解

Prefill 计算：`kv_c_normed @ kv_b_proj → k_nope, v → softmax(q @ [k_nope, k_rot]) @ v`

NPU 上分解为：
1. **MatMulV2**（kv_c_normed @ kv_b_proj）：`(num_tokens, kv_lora_rank) @ (kv_lora_rank, num_heads*(qk_nope_head_dim+v_head_dim))`
2. **FusedInferAttentionScore**：标准 attention

查询维度（MatMulV2）：
- 从 kv_cache shape 推导 kv_lora_rank，从 `args[8]`（kv_b_proj）提取权重 shape
- 注意：prefill 时 kv_c_normed 不在 args 中，需从 kv_cache 推导

> **阻塞项**：prefill 路径的 MatMulV2 shape 推导依赖 kv_c_normed，而 TC 的 `multihead_latent_attention` 接收的是已写入 kv_cache 的数据，无法直接获取 kv_c_normed shape。需要确认 kv_cache shape 与 kv_lora_rank 的对应关系。

### 3.5 mlapo 分解

`mlapo` 分解为：
1. **MatMulV2**：hidden_states @ q_a_proj_weight（Q projection）
2. **KvRmsNormRopeCache**：KV projection + norm + RoPE + cache write

查询维度（MatMulV2）：
- `args[0]`（hidden_states）：`(num_tokens, hidden_size)`
- `args[3]`（q_a_proj_weight）：`(hidden_size, q_lora_rank)`

查询维度（KvRmsNormRopeCache）：
- `args[0]`（hidden_states）：`(num_tokens, hidden_size)`
- `args[6]`（kv_a_proj_weight）：`(hidden_size, kv_lora_rank + qk_rope_head_dim)`

### 3.6 G2 实现前置条件

| 条件 | 状态 | 负责人 |
|------|------|--------|
| `FusedInferAttentionScore` microbenchmark CSV | 待交付 | TCX |
| `TransposeBatchMatMul.csv`（microbenchmark 格式） | 待交付 | HDY |
| `KvRmsNormRopeCache.csv` | 已有（原始 profiling 格式） | — |
| `MatMulV2.csv` | 已有 | — |

> 当前 `_lookup_composite` 对 MLA 返回 `None`（`miss_reason = "csv_not_found"`），fallback 到 AnalyticPerformanceModel。G2 到位后完整实现。

---

## 4. 数据采集需求（microbenchmark 格式）

### 4.1 MoE CSV 格式建议

当前 MoE CSV 为原始 profiling 格式（含大量 NPU 内部参数列）。G1 通过 `tc_input_count` 截断匹配，无需重新采集。

若后续需要 microbenchmark 格式，建议列定义：

**MoeDistributeDispatchV2.csv**：

| 列名 | 类型 | 说明 |
|------|------|------|
| `num_tokens` | int | token 数量（batch × seq） |
| `hidden_size` | int | 隐藏层维度 |
| `top_k` | int | 每 token 选择的 expert 数 |
| `num_experts` | int | 总 expert 数 |
| `dtype` | str | 数据类型 |
| `Duration(us)` | float | 平均耗时 |

**MoeGatingTopK.csv**：

| 列名 | 类型 | 说明 |
|------|------|------|
| `num_tokens` | int | token 数量 |
| `num_experts` | int | expert 数量 |
| `top_k` | int | 选择的 expert 数 |
| `dtype` | str | 数据类型 |
| `Duration(us)` | float | 平均耗时 |

### 4.2 MLA CSV 格式建议

**TransposeBatchMatMul.csv**（microbenchmark 格式，G2 需要）：

| 列名 | 类型 | 说明 |
|------|------|------|
| `batch_size` | int | batch 大小 |
| `num_heads` | int | 注意力头数 |
| `m` | int | 矩阵 M 维度 |
| `k` | int | 矩阵 K 维度（收缩维） |
| `n` | int | 矩阵 N 维度 |
| `dtype` | str | 数据类型 |
| `Duration(us)` | float | 平均耗时 |

---

## 5. 待办与阻塞项

| 项目 | 状态 | 截止 | 负责人 |
|------|------|------|--------|
| G1：`tc_input_count` 机制实现 | 待开始 | 3.18 | ZH |
| G1：MoE 算子查询逻辑 + 单元测试 | 待开始 | 3.18 | ZH |
| G1：`moe_gating_topk` TC 算子注册 | 待确认是否 G1 范围 | 3.18 | ZH/TCX |
| G2：`multihead_latent_attention` prefill/decode 分解 | 阻塞（等 CSV） | 3.20 | ZH |
| G2：`mlapo` / `mlapo_quant` 分解 | 阻塞（等 CSV） | 3.20 | ZH |
| FIA microbenchmark CSV 采集 | 待交付 | 3.18 | TCX |
| TransposeBatchMatMul.csv 采集 | 待交付 | 3.18 | HDY |

---

## 6. 开放问题

1. **MoE CSV 格式**：现有原始 profiling CSV 的 Input Shapes 含 NPU 内部参数，`tc_input_count` 截断方案是否足够，还是需要重新采集 microbenchmark 格式？

2. **MLA prefill kv_c_normed shape**：prefill 路径的 MatMulV2 第一个输入 kv_c_normed 不在 `multihead_latent_attention` 的 args 中，需要从 kv_cache shape 反推 kv_lora_rank。确认 kv_cache shape 约定：`(total_blocks, block_size, kv_lora_rank + qk_rope_head_dim)`，kv_lora_rank 可从 `kv_cache.shape[-1] - qk_rope_head_dim` 计算，但 qk_rope_head_dim 是 scalar 参数，不在 args 中。**需要确认 prefill 路径的 shape 提取方式。**

3. **moe_gating_topk 是否纳入 G1**：TC 当前用 `aten.topk` 实现，NPU 有专用 `MoeGatingTopK` kernel。G1 是否需要新增 TC 算子注册，还是仅做 op_mapping 占位？
