# Qwen3-32B Profiling Alignment Report (v4)

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

| Metric | v1 (baseline) | v2 (batch+SwiGlu) | v3 (zero-cost) | v4 (RoPE+slice) |
|--------|--------------|-------------------|-----------------|------------------|
| Total TC ops | 46 | 46 | 46 | 46 |
| HITs | **3 (6.5%)** | **9 (19.6%)** | **36 (78.3%)** | **38 (82.6%)** |
| MISSes | 43 | 37 | 10 | 8 |

## 1. Solutions Applied (Cumulative)

### Solution 1: `--compile` Flag (v1)
**Problem**: Without `--compile`, TC decomposes fused ops into 72+ primitive aten ops.
**Fix**: `--compile` enables `torch.compile()` pattern matching.
**Impact**: 86 → 46 ops.

### Solution 2: Batch-Dim Stripping (v2)
**Problem**: TC `(1, seq, dim)` vs profiling `(seq, dim)`.
**Fix**: `_strip_batch_dim()` strips leading dim=1.
**Impact**: +6 HITs. Also strip CSV batch dim for symmetry (fixed in v4).

### Solution 3: SwiGlu Input Concatenation (v2)
**Problem**: TC 2 inputs `(seq, D/2)` vs profiling 1 fused `(seq, D)`.
**Fix**: SwiGlu-specific concat along last dim.
**Impact**: +1 HIT.

### Solution 4: Zero-Cost Op Registry (v3)
**Problem**: 27 unmapped shape-only ops inflated miss count.
**Fix**: `zero_cost: true` in op_mapping.yaml → returns `QueryResult(latency_us=0.0)`.
**Impact**: +27 HITs (view, permute, split, select, alias, copy_).

### Solution 5: `alternate_kernel_types` (v3)
**Problem**: `tensor_cast.apply_rope` maps to `InterleaveRope` (DeepSeek) but Qwen3 uses `ApplyRotaryPosEmb`.
**Fix**: `alternate_kernel_types: [ApplyRotaryPosEmb]` tries fallback CSV files.
**Impact**: Enabled RoPE matching (combined with Solution 6).

### Solution 6: RoPE Shape Normalization (v4 — NEW)
**Problem**: TC dispatches RoPE as `[Q(B,H,S,D), K(B,H,S,D), cos(1,S,D), sin(1,S,D)]` but NPU kernel receives `[K(B,S,H,D), Q(B,S,H,D), cos(B,S,1,D), sin(B,S,1,D)]`.
**Fix**: `_normalize_rope_inputs()` reorders Q/K, transposes `(B,H,S,D)→(B,S,H,D)`, inserts head dim in cos/sin.
**Impact**: +1 HIT (ApplyRotaryPosEmb, 12.5 us).

### Solution 7: `aten.slice.Tensor` Zero-Cost (v4 — NEW)
**Problem**: Contiguous slice returns a view (no data movement) but was counted as MISS.
**Fix**: Added `zero_cost: true` for `aten.slice.Tensor` in op_mapping.yaml.
**Impact**: +1 HIT.

### Solution 8: Symmetric Batch-Dim Stripping (v4 — NEW)
**Problem**: When both TC and CSV have leading batch dim=1 (e.g., RoPE `(1,S,H,D)`), stripping only TC produced length mismatch.
**Fix**: Also strip CSV batch dim before comparison and padding checks.
**Impact**: Required for RoPE matching.

## 2. HITs (38 matched ops)

### 2.1 Compute HITs (10 ops)

| TC Op | kernel_type | TC Shape | CSV Shape | Latency | Match Method |
|-------|-------------|----------|-----------|---------|-------------|
| `tensor_cast.rms_norm` | RmsNorm | (1,144,5120),(5120,) | (136,5120),(5120) | 21.7 us | batch-strip + padding |
| `aten.mm.default` | MatMulV2 | (144,5120)x(5120,768) | 136,5120;320,48,16,16 | 19.6 us | FRACTAL_NZ + padding |
| `tensor_cast.rms_norm` | RmsNorm | (1,144,4,128),(128,) | (136,4,128),(128) | 20.1 us | batch-strip + padding |
| `tensor_cast.rms_norm` | RmsNorm | (1,144,1,128),(128,) | (136,1,128),(128) | 7.7 us | batch-strip + padding |
| `tensor_cast.apply_rope` | ApplyRotaryPosEmb | (1,1,144,128),(1,4,144,128),(1,144,128),(1,144,128) | (1,136,4,128),(1,136,1,128),(1,136,1,128),(1,136,1,128) | **12.5 us** | **RoPE normalization + alternate_kernel_types + padding** |
| `tensor_cast.add_rms_norm2` | AddRmsNorm | (1,144,5120),(144,5120),(5120,) | (136,5120),(136,5120),(5120) | 12.5 us | batch-strip + padding |
| `aten.mm.default` | MatMulV2 | (144,5120)x(5120,3200) | 136,5120;320,200,16,16 | 59.7 us | FRACTAL_NZ + padding |
| `tensor_cast.swiglu` | SwiGlu | (1,144,1600),(1,144,1600) | (136,3200) | 14.9 us | SwiGlu concat + batch-strip + padding |
| `aten.add.Tensor` | Add | (1,144,5120),(144,5120) | (136,5120),(136,5120) | 16.2 us | batch-strip + padding |
| `aten.mm.default` | MatMulV2 | (1,5120)x(5120,9496) | 1,5120;9496,5120 | 91.8 us | ND transpose |

### 2.2 Zero-Cost HITs (28 ops)
`aten.view.default` (16), `aten.permute.default` (4), `aten.split_with_sizes.default` (2), `aten.select.int` (2), `aten.split.Tensor` (1), `aten.slice.Tensor` (1), `aten.alias.default` (1), `aten.copy_.default` (1).

## 3. Remaining MISSes (8 ops)

### 3.1 ReshapeAndCacheNdKernel — Structural Mismatch
**TC**: `(144,128), (144,128), (2,2,128,1,128), (136,)` — 4 inputs
**CSV**: `(136,1,128), (136,1,128), (10873,128,1,128), (10873,128,1,128), (136)` — 5 inputs

Root cause: TC's `reshape_and_cache` abstraction differs from NPU kernel interface (missing head dim, different cache shape, different input count).

### 3.2 Embedding (GatherV2) — Vocab Sharding
**TC**: `(151936, 5120)` full vocab
**CSV**: `(9496, 5120)` = 151936/16 TP-sharded

TC doesn't shard embedding; profiling captures per-rank shape.

### 3.3 Skipped by Design
| TC Op | Reason | Potential Fix |
|-------|--------|---------------|
| `tensor_cast.attention.default` | `query_mode=attention_special` | Dedicated attention matching |
| `tensor_cast.all_gather.default` | `category=communication` | CommGrid bandwidth estimation |
| `tensor_cast.matmul_all_reduce.default` x2 | `composite=true` | Decompose into MatMulV2 + hcom_allReduce_ |

### 3.4 Minor
| TC Op | Shape | Issue |
|-------|-------|-------|
| `aten.index.Tensor` | `(40960, 256)` | Position embedding, no mapping |
| `aten.index.Tensor` | `(1, 144, 5120)` | No mapping |

## 4. Coverage Analysis

| Category | Ops | Matched | Rate |
|----------|-----|---------|------|
| Compute (mm, norm, activation, RoPE) | 10 | **10** | **100%** |
| Zero-cost (view, permute, split, etc.) | 28 | **28** | **100%** |
| KV Cache (reshape_and_cache) | 1 | 0 | 0% |
| Embedding (gather) | 1 | 0 | 0% |
| Communication (all_gather) | 1 | 0 | 0% |
| Composite (matmul_all_reduce) | 2 | 0 | 0% |
| Attention (special) | 1 | 0 | 0% |
| Other (index) | 2 | 0 | 0% |
| **Total** | **46** | **38** | **82.6%** |

**Effective hit rate** (excluding by-design skips + zero-cost): **10/13 = 76.9%**

### Latency Coverage
Total measured compute latency: **276.7 us** per layer (10 compute HITs).
The 3 compute MISSes (ReshapeAndCache, embedding, index x2) fall back to analytic model.

## 5. Remaining Improvement Opportunities

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| P1 | Composite decomposition (matmul_all_reduce → MatMulV2 + hcom) | Medium | +2 HITs |
| P2 | Attention special mode (FusedInferAttentionScore.csv) | Medium | +1 HIT |
| P3 | Embedding sharding (vocab / TP) | Low | +1 HIT |
| P4 | ReshapeAndCache structural mapping | High | +1 HIT |
| P5 | aten.index.Tensor mapping | Low | +2 HITs |

## 6. Implementation Summary

### Files Modified
- `tensor_cast/performance_model/perf_database/profiling_data_source.py`: RoPE normalization, symmetric batch-dim stripping, alternate_kernel_types, zero_cost handling
- `tensor_cast/performance_model/perf_database/data/.../v0.14.0/op_mapping.yaml`: alternate_kernel_types for RoPE, 14 zero_cost entries
- `tests/perf_database/test_profiling_data_source.py`: 32 tests (all passing)

### Key Design Decisions
1. **`alternate_kernel_types`**: Brute-force fallback — tries each CSV in order. Sufficient for spike; production should use conditional dispatch based on op kwargs.
2. **RoPE normalization**: Kernel-specific input transformation in `_inputs_match()`. Pattern: detect kernel_type, reorder/reshape TC inputs to match NPU convention.
3. **Symmetric batch-dim stripping**: Both TC and CSV may have leading batch=1. Strip both before comparison.
4. **Zero-cost registry**: Simple flag in op_mapping.yaml. Could be extended with cost estimation (e.g., copy_ might have nonzero cost in some cases).
