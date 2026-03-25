# SUPERSEDED — G2 MLA 分解查询 + InterpolatingDataSource 改进设计

> **Status**: SUPERSEDED by actual implementation in `profiling_data_source.py` (`_decompose_mla_common`, `_decompose_mlapo`) and `interpolating_data_source.py`. Key changes from this spec:
> - §2.4 MLA prefill now decomposes to MatMulV2 + RINGMLAPrefillBF16Kernel (not MatMulV2 + FIA)
> - §2.5-2.6 MLAPO uses composite decomposition (3x MatMulV2 + KvRmsNormRopeCache), not direct compute lookup
> - See Design Doc v1.5 §4.8.7 and PR #52 review for rationale.

**日期**: 2026-03-19
**负责人**: ZH
**范围**: profiling_data_source.py MLA 专用分解 + interpolating_data_source.py composite 插值支持

---

## 1. 背景

当前 `_lookup_composite()` 是通用逻辑：拿 TC op 的原始 inputs 去匹配 sub-kernel CSV。MLA 的 sub-kernel shapes 与 TC op args 不同，需要专用分解函数从 args 推导每个子内核的查询 shape。

InterpolatingDataSource 基础版（TCX）覆盖 compute/comm/attention 三条路径，但 `composite` 直接跳过（`return None`）。MLA 作为 composite op 无法走插值。

## 2. Part 1：MLA 专用分解

### 2.1 分解函数注册机制

在 `profiling_data_source.py` 中新增 `COMPOSITE_DECOMPOSERS` 字典：

```python
COMPOSITE_DECOMPOSERS: Dict[str, Callable[[OpInvokeInfo, dict], Optional[List[SubKernelSpec]]]] = {
    "tensor_cast.multihead_latent_attention.default": _decompose_mla,
    "tensor_cast.multihead_latent_attention_quant.default": _decompose_mla_quant,
    "tensor_cast.mlapo.default": _decompose_mlapo,
    "tensor_cast.mlapo_quant.default": _decompose_mlapo_quant,
}
```

`_lookup_composite` 修改：检测到 func_name 在 COMPOSITE_DECOMPOSERS 中时，调用专用分解函数，否则走现有通用逻辑。

### 2.2 SubKernelSpec 数据结构

```python
@dataclass
class SubKernelSpec:
    kernel_type: str
    input_shapes: List[Tuple[int, ...]]
    dtype: str
    query_mode: str = "compute"  # "compute" | "attention"
    attention_params: Optional[Dict] = None  # batch_size, avg_seq_len, num_heads, head_dim
```

### 2.3 MLA Decode 分解（query_lens is None 或全为 1）

计算：`softmax(q @ W_UK_T @ k_cache) @ v_cache @ W_UV`

3 个子内核：
1. **TransposeBatchMatMul**（q @ W_UK_T）
   - shapes: `(batch, num_heads, qk_nope_head_dim)` @ `(num_heads, qk_nope_head_dim, kv_lora_rank)`
   - 来源: `q=args[0]`, `W_UK_T=args[6]`
2. **FusedInferAttentionScore**
   - query_mode: attention
   - params: batch_size=seq_lens.shape[0], avg_seq_len=mean(seq_lens), num_heads, head_dim
   - 来源: `seq_lens=args[4]`, `q=args[0]`
3. **TransposeBatchMatMul**（attn_out @ W_UV）
   - shapes: `(batch, num_heads, kv_lora_rank)` @ `(num_heads, kv_lora_rank, v_head_dim)`
   - 来源: `W_UV=args[7]`, `v_head_dim=args[9]`

### 2.4 MLA Prefill 分解（query_lens 存在且 > 1）

计算：`kv_c_normed @ kv_b_proj → k_nope, v → softmax(q @ [k_nope, k_rot]) @ v`

2 个子内核：
1. **MatMulV2**（kv_c_normed @ kv_b_proj）
   - shapes: `(num_tokens, kv_lora_rank)` @ `(kv_lora_rank, num_heads*(qk_nope_head_dim+v_head_dim))`
   - kv_lora_rank = `kv_b_proj.shape[0]`（args[8]）
   - num_tokens = `q.shape[0]`（args[0]）
2. **FusedInferAttentionScore**
   - 同 decode 路径

### 2.5 MLAPO 分解

2 个子内核：
1. **MatMulV2**（hidden_states @ q_a_proj_weight）
   - shapes: `args[0]` @ `args[3]`
2. **KvRmsNormRopeCache**
   - shapes: `args[0]` @ `args[6]`

### 2.6 MLAPO Quant 分解

同 MLAPO 但 MatMulV2 → QuantBatchMatmulV3。

## 3. Part 2：InterpolatingDataSource composite 插值

### 3.1 修改点

移除第 80 行 `if mapping.get("composite"): return None`，新增 `_interpolate_composite()`。

### 3.2 逻辑

```
_interpolate_composite(op_invoke_info, mapping)
  → COMPOSITE_DECOMPOSERS[func_name](op_invoke_info) → sub-kernels
  → for each sub-kernel:
      if query_mode == "attention" → _interpolate_attention_by_params(...)
      else → _interpolate_compute_by_shapes(...)
  → 任一子内核无法插值 → return None
  → 全部成功 → sum latencies, confidence=0.5
```

### 3.3 重构：提取 by_shapes/by_params 内部方法

从现有 `_interpolate_compute` 和 `_interpolate_attention` 中提取：
- `_interpolate_compute_by_shapes(kernel_type, input_shapes, dtype)` — 可被 composite 复用
- `_interpolate_attention_by_params(kernel_type, batch_size, avg_seq_len, num_heads, head_dim, dtype)` — 可被 composite 复用

## 4. 测试用例

### 4.1 MLA 分解测试
1. MLA decode：query_lens=None → 3 个子内核，shape 正确
2. MLA prefill：query_lens>1 → 2 个子内核，kv_c_normed shape = (num_tokens, kv_b_proj.shape[0])
3. MLAPO：→ MatMulV2 + KvRmsNormRopeCache
4. MLA quant：→ QuantBatchMatmulV3 替代
5. 端到端 composite lookup：CSV 匹配 → 返回求和延迟
6. 子内核 miss → 整体 None

### 4.2 InterpolatingDataSource 改进测试
7. Composite 插值：MLA decode 精确 miss 但子内核可插值 → 返回插值结果
8. 部分子内核无法插值 → None
9. 现有 compute/comm/attention 回归不受影响
