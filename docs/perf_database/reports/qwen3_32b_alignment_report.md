# Qwen3-32B Profiling Alignment Report (v3)

**Date:** 2026-03-05
**Model:** Qwen/Qwen3-32B (BF16 Prefill)
**Device:** ATLAS_800_A3_752T_128G_DIE (TP=16)
**Profiling Source:** Qwen3-30B Prefill (same architecture dims, TP=16, seq=136)

## TC Command

```bash
PYTHONPATH=<worktree>:$PYTHONPATH python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --device ATLAS_800_A3_752T_128G_DIE \
  --world-size 16 --tp-size 16 \
  --num-queries 1 --query-length 136 \
  --quantize-linear-action DISABLED --compile \
  --performance-model profiling \
  --perf-database .../v0.14.0 --log-level debug
```

## Summary — Iteration History

| Metric | v1 (baseline) | v2 (batch+SwiGlu) | v3 (zero-cost+alternate) |
|--------|--------------|-------------------|--------------------------|
| Total TC ops | 46 | 46 | 46 |
| HITs | **3 (6.5%)** | **9 (19.6%)** | **36 (78.3%)** |
| Mapped but MISS | 6 | 6 | 10 |
| Skipped (composite/comm/attn) | 4 | 4 | 0 (now counted in MISS) |
| Unmapped (no kernel) | 33 | 27 | 0 (now zero_cost HITs) |

## 1. Solutions Applied (Cumulative)

### Solution 1: `--compile` Flag (v1)

**Problem**: Without `--compile`, TC decomposes fused ops (RmsNorm, SwiGlu, RoPE) into 72+ primitive aten ops.
**Fix**: Use `--compile` to enable `torch.compile()` pattern matching.
**Impact**: Reduced op count from 86 to 46.

### Solution 2: Batch-Dim Stripping (v2)

**Problem**: TC keeps explicit batch dim `(1, seq, dim)`, profiling flattens to `(seq, dim)`.
**Fix**: `_strip_batch_dim()` strips leading dim=1 before shape comparison.
**Impact**: +6 HITs (RmsNorm x3, AddRmsNorm x1, Add x1, SwiGlu x1).

### Solution 3: SwiGlu Input Concatenation (v2)

**Problem**: TC dispatches SwiGlu with 2 inputs `(seq, D/2), (seq, D/2)`, profiling stores 1 fused input `(seq, D)`.
**Fix**: SwiGlu-specific normalization concatenates TC inputs along last dim.
**Impact**: +1 HIT (SwiGlu).

### Solution 4: Zero-Cost Op Registry (v3 — NEW)

**Problem**: 27 unmapped shape-only ops (view, permute, split, select, etc.) inflated miss count.
**Fix**: Added `zero_cost: true` entries in op_mapping.yaml for 13 shape-only op types. `lookup()` returns `QueryResult(latency_us=0.0)` immediately.
**Impact**: +27 HITs (all shape-only ops now correctly matched as zero-cost).

Zero-cost ops added:
| Op | Count | Notes |
|----|-------|-------|
| `aten.view.default` | 16 | Reshape, no data movement |
| `aten.permute.default` | 4 | View-based transpose |
| `aten.split_with_sizes.default` | 2 | QKV/gate_up split |
| `aten.select.int` | 2 | KV cache indexing |
| `aten.split.Tensor` | 1 | View-based split |
| `aten.alias.default` | 1 | No-op |
| `aten.copy_.default` | 1 | KV cache update (counted as zero-cost for now) |

### Solution 5: `alternate_kernel_types` Mechanism (v3 — NEW)

**Problem**: Some TC ops map to different NPU kernels depending on runtime parameters. For example, `tensor_cast.apply_rope` maps to `InterleaveRope` (DeepSeek) or `ApplyRotaryPosEmb` (Qwen3/neox).
**Fix**: Added `alternate_kernel_types` field in op_mapping.yaml. `_lookup_compute()` tries primary kernel_type first, then alternates.
**Impact**: RoPE now correctly tries `ApplyRotaryPosEmb.csv`, but still MISSes due to shape structure mismatch (see §3.1).

## 2. HITs (36 matched ops)

### 2.1 Compute HITs (9 ops)

| TC Op | kernel_type | TC Shape | CSV Shape | Latency | Match Method |
|-------|-------------|----------|-----------|---------|-------------|
| `tensor_cast.rms_norm` | RmsNorm | (1,144,5120),(5120,) | (136,5120),(5120) | 21.7 us | batch-strip + padding |
| `aten.mm.default` | MatMulV2 | (144,5120)x(5120,768) | 136,5120;320,48,16,16 | 19.6 us | FRACTAL_NZ + padding |
| `tensor_cast.rms_norm` | RmsNorm | (1,144,4,128),(128,) | (136,4,128),(128) | 20.1 us | batch-strip + padding |
| `tensor_cast.rms_norm` | RmsNorm | (1,144,1,128),(128,) | (136,1,128),(128) | 7.7 us | batch-strip + padding |
| `tensor_cast.add_rms_norm2` | AddRmsNorm | (1,144,5120),(144,5120),(5120,) | (136,5120),(136,5120),(5120) | 12.5 us | batch-strip + padding |
| `aten.mm.default` | MatMulV2 | (144,5120)x(5120,3200) | 136,5120;320,200,16,16 | 59.7 us | FRACTAL_NZ + padding |
| `tensor_cast.swiglu` | SwiGlu | (1,144,1600),(1,144,1600) | (136,3200) | 14.9 us | SwiGlu concat + batch-strip + padding |
| `aten.add.Tensor` | Add | (1,144,5120),(144,5120) | (136,5120),(136,5120) | 16.2 us | batch-strip + padding |
| `aten.mm.default` | MatMulV2 | (1,5120)x(5120,9496) | 1,5120;9496,5120 | 91.8 us | ND transpose |

### 2.2 Zero-Cost HITs (27 ops)

All shape-only operations: `aten.view.default` (16), `aten.permute.default` (4), `aten.split_with_sizes.default` (2), `aten.select.int` (2), `aten.split.Tensor` (1), `aten.alias.default` (1), `aten.copy_.default` (1).

## 3. Remaining MISSes (10 ops)

### 3.1 RoPE — Shape Layout Mismatch

**TC**: `(1,1,144,128), (1,4,144,128), (1,144,128), (1,144,128)` — `(batch, heads, seq, dim)` format
**CSV**: `(1,136,4,128), (1,136,1,128), (1,136,1,128), (1,136,1,128)` — `(batch, seq, heads, dim)` format

The `alternate_kernel_types` mechanism correctly falls back from `InterleaveRope` (no CSV) to `ApplyRotaryPosEmb` (CSV exists), but shapes don't match because TC dispatches in `(B,H,S,D)` layout while the NPU kernel receives `(B,S,H,D)`.

**Proposed fix**: Add RoPE-specific shape permutation in `_inputs_match()` — when kernel_type is `ApplyRotaryPosEmb`, try transposing `(B,H,S,D)` → `(B,S,H,D)` before comparison.

### 3.2 ReshapeAndCacheNdKernel — Shape Structure

**TC**: `(144,128), (144,128), (2,2,128,1,128), (136,)` — 4 inputs
**CSV**: `(136,1,128), (136,1,128), (10873,128,1,128), (10873,128,1,128), (136)` — 5 inputs

Differences:
- TC missing `num_heads=1` dim in KV tensors: `(144,128)` vs `(136,1,128)`
- TC KV cache has different block structure: `(2,2,128,1,128)` vs `(10873,128,1,128)`
- Input count mismatch: 4 vs 5

**Root cause**: TC's `reshape_and_cache` abstraction differs significantly from the NPU kernel interface.

### 3.3 Embedding (GatherV2) — Vocab Sharding

**TC**: `(151936, 5120)` full vocab + `(1, 144)` indices
**CSV**: `(9496, 5120)` = 151936/16 TP-sharded + `(136), (1)` indices

TC doesn't shard the embedding table by TP; profiling captures the per-rank sharded shape.

### 3.4 Skipped by Design (counted as MISS)

| TC Op | Reason | Notes |
|-------|--------|-------|
| `tensor_cast.attention.default` | `query_mode=attention_special` | Needs dedicated attention matching logic |
| `tensor_cast.all_gather.default` | `category=communication` | Needs CommGrid topology for bandwidth estimation |
| `tensor_cast.matmul_all_reduce.default` x2 | `composite=true` | MatMulV2 + hcom_allReduce_ fusion |

### 3.5 Minor MISSes

| TC Op | Shape | Issue |
|-------|-------|-------|
| `aten.index.Tensor` | `(40960, 256)` | Position embedding indexing, no mapping |
| `aten.slice.Tensor` | `(1, 144, 5120)` | Not marked zero_cost (has actual data access) |
| `aten.index.Tensor` | `(1, 144, 5120)` | Not marked zero_cost (has actual data access) |

## 4. Design Analysis: `alternate_kernel_types` vs Conditional Mapping

The `alternate_kernel_types` mechanism was added as a pragmatic fallback:

```yaml
"tensor_cast.apply_rope.default":
    kernel_type: InterleaveRope
    alternate_kernel_types: [ApplyRotaryPosEmb]
```

**How it works**: `_lookup_compute()` tries each kernel_type's CSV in order until a shape match is found.

**Limitation**: This is a brute-force approach — it tries all alternates regardless of context. The ideal design would use **conditional mapping** based on op kwargs:

```yaml
"tensor_cast.apply_rope.default":
    kernel_type_variants:
      - condition: {is_neox: true}
        kernel_type: ApplyRotaryPosEmb
      - condition: {is_neox: false}
        kernel_type: InterleaveRope
```

For the spike, `alternate_kernel_types` is sufficient since it's O(small) alternates. A production implementation should use conditional dispatch.

## 5. Coverage Analysis

### By Category

| Category | Ops | Matched | Rate |
|----------|-----|---------|------|
| Compute (mm, norm, activation) | 9 | 9 | 100% |
| Zero-cost (view, permute, split) | 27 | 27 | 100% |
| RoPE | 1 | 0 | 0% (shape layout) |
| KV Cache | 1 | 0 | 0% (structure) |
| Embedding | 1 | 0 | 0% (sharding) |
| Communication | 1 | 0 | 0% (by design) |
| Composite | 2 | 0 | 0% (by design) |
| Attention | 1 | 0 | 0% (by design) |
| Other (index, slice) | 3 | 0 | 0% |
| **Total** | **46** | **36** | **78.3%** |

### Effective Hit Rate

Excluding by-design skips (attention, communication, composite = 4 ops) and zero-cost ops (27 ops):
- Matchable compute ops: 15
- Matched: 9
- **Effective hit rate: 9/15 = 60%**

### Latency Coverage

For the 9 compute HITs, total measured latency = 264.2 us per layer iteration.
The 6 compute MISSes (RoPE, ReshapeAndCache, embedding, index x2, slice) fall back to analytic model estimation.

## 6. Remaining Improvement Opportunities

### P1: RoPE Shape Permutation
Add `(B,H,S,D)` → `(B,S,H,D)` normalization for `ApplyRotaryPosEmb` kernel_type. Would add 1 HIT.

### P2: `aten.slice.Tensor` Zero-Cost
Slice on contiguous dims is often zero-cost (view). Could mark as `zero_cost` with a note. Would add 1 HIT.

### P3: Composite Op Decomposition
`tensor_cast.matmul_all_reduce.default` = MatMulV2 + hcom_allReduce_. Could decompose and query each sub-kernel. Would add 2 HITs.

### P4: Attention Special Mode
Implement dedicated attention shape matching for `FusedInferAttentionScore.csv`. Would add 1 HIT.

### P5: Embedding Sharding
Apply TP sharding to embedding vocab dim before lookup. Would add 1 HIT.
