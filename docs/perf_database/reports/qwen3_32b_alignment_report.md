# Qwen3-32B Profiling Alignment Report (v5)

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
  --profiling-database .../v0.14.0 --log-level debug
```

## Summary — Iteration History

| Metric | v1 | v2 | v3 | v4 | v5 |
|--------|----|----|----|----|-----|
| HITs | 3 (6.5%) | 9 (19.6%) | 36 (78.3%) | 38 (82.6%) | **40 (87.0%)** |
| MISSes | 43 | 37 | 10 | 8 | **6** |
| Key Fix | --compile | batch-strip, SwiGlu | zero-cost | RoPE | composite decomp |

## 1. All Solutions Applied

### S1: `--compile` Flag (v1)
Enables `torch.compile()` pattern matching for fused ops. 86 → 46 ops.

### S2: Batch-Dim Stripping (v2)
`_strip_batch_dim()` strips leading dim=1 from TC shapes. +6 HITs.

### S3: SwiGlu Input Concatenation (v2)
TC 2 inputs → CSV 1 fused input, concat along last dim. +1 HIT.

### S4: Zero-Cost Op Registry (v3)
`zero_cost: true` in op_mapping.yaml for shape-only ops (14 types). +28 HITs.

### S5: `alternate_kernel_types` (v3)
Fallback kernel type lookup. Enabled RoPE matching with ApplyRotaryPosEmb.

### S6: RoPE Shape Normalization (v4)
`_normalize_rope_inputs()`: reorder Q/K, transpose `(B,H,S,D)→(B,S,H,D)`, insert head dim in cos/sin. +1 HIT.

### S7: Symmetric Batch-Dim Stripping (v4)
Strip both TC and CSV leading dim=1 for consistent comparison.

### S8: Composite Op Decomposition (v5 — NEW)
**Problem**: `matmul_all_reduce` marked `composite: true` → returned None immediately, losing 2 HITs.
**Fix**: `_lookup_composite()` extracts compute sub-kernels from `sub_kernels` list, skips `hcom_*` communication kernels, looks up remaining against CSV. Returns with `confidence=0.8` (partial match — comm portion handled by analytic model).
**Impact**: +2 HITs (MatMulV2 14.2 us + 25.1 us).

## 2. HITs (40 matched ops)

### 2.1 Compute HITs (12 ops)

| TC Op | kernel_type | TC Shape | CSV Shape | Latency | Match |
|-------|-------------|----------|-----------|---------|-------|
| `tensor_cast.rms_norm` | RmsNorm | (1,144,5120),(5120,) | (136,5120),(5120) | 21.7 us | batch-strip+pad |
| `aten.mm.default` | MatMulV2 | (144,5120)x(5120,768) | FRACTAL_NZ | 19.6 us | NZ+pad |
| `tensor_cast.rms_norm` | RmsNorm | (1,144,4,128),(128,) | (136,4,128),(128) | 20.1 us | batch-strip+pad |
| `tensor_cast.rms_norm` | RmsNorm | (1,144,1,128),(128,) | (136,1,128),(128) | 7.7 us | batch-strip+pad |
| `tensor_cast.apply_rope` | ApplyRotaryPosEmb | (1,1,144,128),... | (1,136,4,128),... | 12.5 us | RoPE norm+alt+pad |
| `tensor_cast.matmul_all_reduce` | MatMulV2 (composite) | (144,512)x(512,5120) | FRACTAL_NZ | **14.2 us** | **composite decomp** |
| `tensor_cast.add_rms_norm2` | AddRmsNorm | (1,144,5120),(144,5120),(5120,) | (136,5120),... | 12.5 us | batch-strip+pad |
| `aten.mm.default` | MatMulV2 | (144,5120)x(5120,3200) | FRACTAL_NZ | 59.7 us | NZ+pad |
| `tensor_cast.swiglu` | SwiGlu | (1,144,1600),(1,144,1600) | (136,3200) | 14.9 us | SwiGlu concat |
| `tensor_cast.matmul_all_reduce` | MatMulV2 (composite) | (144,1600)x(1600,5120) | FRACTAL_NZ | **25.1 us** | **composite decomp** |
| `aten.add.Tensor` | Add | (1,144,5120),(144,5120) | (136,5120),... | 16.2 us | batch-strip+pad |
| `aten.mm.default` | MatMulV2 | (1,5120)x(5120,9496) | (1,5120;9496,5120) | 91.8 us | ND transpose |

### 2.2 Zero-Cost HITs (28 ops)
view(16), permute(4), split_with_sizes(2), select(2), split(1), slice(1), alias(1), copy_(1).

**Total measured compute latency: 316.0 us** per layer (12 compute HITs).

## 3. Remaining MISSes (6 ops)

### 3.1 ReshapeAndCacheNdKernel — Structural Mismatch
TC: 4 inputs `(144,128), (144,128), (2,2,128,1,128), (136,)` — missing head dim, different cache shape.
CSV: 5 inputs `(136,1,128), (136,1,128), (10873,128,1,128), (10873,128,1,128), (136)`.
**Status**: Requires TC reshape_and_cache op to match NPU kernel interface. Not fixable at matching level.

### 3.2 Embedding (GatherV2) — Vocab Sharding
TC: `(151936, 5120)` full vocab. CSV: `(9496, 5120)` = 151936/16 TP-sharded.
**Status**: Requires TP-aware embedding lookup or dividing vocab by world_size before matching.

### 3.3 By-Design Skips
| TC Op | Reason | Future Fix |
|-------|--------|------------|
| `tensor_cast.attention.default` | `query_mode=attention_special` | Dedicated attention matching with seq/head-aware logic |
| `tensor_cast.all_gather.default` | `category=communication` | CommGrid bandwidth model |

### 3.4 Minor Unmapped
| TC Op | Shape | Notes |
|-------|-------|-------|
| `aten.index.Tensor` | `(40960, 256)` | Position embedding indexing |
| `aten.index.Tensor` | `(1, 144, 5120)` | Token selection |

## 4. Coverage Analysis

| Category | Ops | Matched | Rate |
|----------|-----|---------|------|
| Compute (mm, norm, activation, RoPE) | 10 | **10** | **100%** |
| Composite compute (matmul_all_reduce) | 2 | **2** | **100%** |
| Zero-cost (view, permute, split, etc.) | 28 | **28** | **100%** |
| KV Cache (reshape_and_cache) | 1 | 0 | 0% |
| Embedding (gather) | 1 | 0 | 0% |
| Communication (all_gather) | 1 | 0 | 0% |
| Attention (special) | 1 | 0 | 0% |
| Other (index) | 2 | 0 | 0% |
| **Total** | **46** | **40** | **87.0%** |

**Effective hit rate** (excluding by-design skips + zero-cost): **12/16 = 75.0%**

## 5. Architecture Summary

### Shape Matching Pipeline in `_inputs_match()`
```
TC inputs → RoPE normalization → SwiGlu normalization → per-tensor:
  → exact match? → batch-strip (both TC+CSV)? → FRACTAL_NZ restore?
  → ND weight transpose? → block-padding tolerance?
```

### Dispatch Logic in `lookup()`
```
func → op_mapping.yaml → composite? → _lookup_composite(sub_kernels)
                        → communication? → None (analytic)
                        → attention_special? → None (analytic)
                        → zero_cost? → QueryResult(0.0)
                        → default → _lookup_compute(kernel_types)
```

### Key Schema Extensions to op_mapping.yaml
```yaml
# Fallback kernel types
alternate_kernel_types: [Type1, Type2]

# Shape-only ops
zero_cost: true

# Composite decomposition
composite: true
sub_kernels: [MatMulV2, hcom_allReduce_]
```

## 6. Remaining Improvement Opportunities

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| P1 | Attention special mode (FusedInferAttentionScore) | High | +1 HIT, high latency coverage |
| P2 | Embedding sharding (vocab / TP) | Medium | +1 HIT |
| P3 | Communication bandwidth model | Medium | +1 HIT |
| P4 | ReshapeAndCache structural mapping | High | +1 HIT |
| P5 | aten.index.Tensor mapping | Low | +2 HITs, low latency |

## 7. Test Coverage

34 tests covering:
- FRACTAL_NZ restoration (3 tests)
- dtype mapping (1 test)
- Exact match, wrong shape, unmapped op (3 tests)
- Composite, communication, attention_special returns None (3 tests)
- ND weight transpose (2 tests)
- Block-padding tolerance (3 tests)
- Batch-dim stripping (4 tests)
- SwiGlu input concatenation (2 tests)
- RoPE shape normalization (2 tests)
- **Composite decomposition (2 tests)** — NEW
