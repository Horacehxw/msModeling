# Composite Lookup Refactor — Design Spec

**Date**: 2026-03-27
**Scope**: `profiling_data_source.py` query path refactor + DFC EP Size matching + composite partial match
**Motivation**: DSV3 MLA profiling MISS — 12 (prefill) / 16 (decode), coverage ~78%

## Problem

Three root causes behind DSV3 MLA MISSes:

1. **Decomposer → CSV shape matching gap**: `_lookup_compute_by_shapes` lacks `tc_input_count` and `alternate_kernel_types`, causing QuantBatchMatmulV3 MISS (decomposer passes 2 shapes, CSV has 4 inputs).
2. **Duplicated query logic**: `_lookup_compute`, `_lookup_compute_by_shapes`, and generic composite inline CSV traversal all do the same "shapes → CSV match" work independently, with inconsistent capabilities.
3. **Missing EP Size matching**: DFC (`DispatchFFNCombine`) ignores the `EP Size` CSV column. Currently masked because all data is EP=16, but will break with multi-EP data.

## Design

### Principle

Extract shared query cores only where 2+ callers exist. Single-caller paths stay as standalone methods.

### Architecture

```
lookup(OpInvokeInfo)
 │
 ├─► composite? → _lookup_composite
 │     ├─► decomposed:
 │     │     ├─ "compute"   ─► _query_by_shapes()      ◄─ shared core (3 callers)
 │     │     ├─ "attention" ─► _query_by_attn_params()  ◄─ shared core (2 callers)
 │     │     └─ partial match: accumulate + mark MISS
 │     └─► generic (MC2):
 │           ├─ compute ─► _query_by_shapes()
 │           └─ comm ─► _query_comm_csv()               ◄─ shared core (2 callers, existing)
 │
 ├─► moe_fused? → _lookup_moe()                         ◄─ standalone (1 caller)
 ├─► attention? → _lookup_attention() → _query_by_attn_params()
 ├─► elementwise? → _lookup_elementwise()                  (unchanged)
 ├─► comm? → _lookup_comm() → _query_comm_csv()
 ├─► zero_cost → return 0
 └─► default → _lookup_compute() → _query_by_shapes()
```

### Shared Core Methods (new)

#### `_query_by_shapes`

```python
def _query_by_shapes(
    self,
    kernel_types: List[str],
    tc_inputs: List[Tuple[Tuple[int, ...], torch.dtype]],
    tc_input_count: Optional[int] = None,
) -> Optional[float]:
```

- Iterates `kernel_types` (primary + alternates), loads CSV for each.
- Iterates CSV rows, calls `_inputs_match(tc_inputs, row, kernel_type, tc_input_count)`.
- On MISS: sets `self.last_miss_reason`, logs debug diagnostics.
- `tc_input_count=None` preserves current behavior (no auto-truncation).

Replaces: `_lookup_compute_by_shapes` (deleted), inline CSV traversal in generic composite (L918-941).

Callers:
- `_lookup_compute` (from OpInvokeInfo)
- `_lookup_composite` generic path (from OpInvokeInfo)
- `_lookup_composite_decomposed` (from SubKernelSpec)

#### `_query_by_attn_params`

```python
def _query_by_attn_params(
    self,
    kernel_types: List[str],
    params: Dict[str, Any],  # q_shape_3d, avg_seq_len, sparse_mode, num_kv_heads
    dtype: str,
) -> Optional[float]:
```

- Iterates `kernel_types`, loads enriched CSV.
- Matches `(N, D, dtype, avg_seq_len)` required, `(sparse_mode, num_kv_heads)` optional (CSV column presence), `input_layout` as optional tie-breaker.
- Block-padding tolerance on T dimension.

Replaces: `_lookup_attention_by_params` (deleted).

Callers:
- `_lookup_attention` (from OpInvokeInfo)
- `_lookup_composite_decomposed` (from SubKernelSpec)

### SubKernelSpec Extension

```python
@dataclass
class SubKernelSpec:
    kernel_type: str
    input_shapes: List[Tuple[int, ...]]
    dtype: str
    query_mode: str = "compute"
    attention_params: Optional[Dict[str, Any]] = None
    tc_input_count: Optional[int] = None           # new
    alternate_kernel_types: Optional[List[str]] = None  # new
```

### DFC EP Size Matching

op_mapping.yaml:
```yaml
"tensor_cast.dispatch_ffn_combine.default":
    kernel_type: DispatchFFNCombine
    query_mode: moe_fused
    tc_input_count: 1
```

Standalone method `_lookup_moe(OpInvokeInfo, mapping)`:
- Shape matching via `_inputs_match` (reuses existing logic).
- `EP Size` column exact match. CSV has column but `ep_size` not configured → MISS with reason `ep_size_not_configured`.

EP Size source: `ProfilingDataSource.__init__` accepts `parallel_config: Optional[ParallelConfig]`, stores `self.ep_size = parallel_config.expert_parallel_size`.

```python
class ProfilingDataSource(DataSourcePerformanceModel):
    def __init__(
        self,
        data_dir: str | Path,
        device_profile: Optional[DeviceProfile] = None,
        parallel_config: Optional[ParallelConfig] = None,
    ):
        self.comm_grid = device_profile.comm_grid if device_profile else None
        self.ep_size = parallel_config.expert_parallel_size if parallel_config else None
```

### Composite Partial Match

When a decomposed composite has some sub-kernels HIT and others MISS:

- Accumulate latency from matched sub-kernels (do not early-return on first MISS).
- Return `QueryResult` with `source=QuerySource.PARTIAL`.
- `confidence = len(hit_kernels) / len(specs)` (reflects match ratio).
- Details include `hit_kernels` and `missed_kernels` lists.

```python
class QuerySource(Enum):
    MEASURED = "measured"
    INTERPOLATED = "interpolated"
    PARTIAL = "partial"  # new
```

Upper layer (`EmpiricalPerformanceModel`):
- PARTIAL latency participates in end-to-end sum (business value).
- PARTIAL counts as MISS in match rate metrics (accuracy reporting).
- Display format: `HIT: 34/42, PARTIAL: 3/42 (mlapo_quant×2, mla×1), MISS: 5/42`.

Generic composite (MC2) does NOT use partial match — only decomposed path.

## Changes Summary

| Type | Target | Detail |
|------|--------|--------|
| New method | `_query_by_shapes` | Shared shape-matching core |
| New method | `_query_by_attn_params` | Shared FIA params core |
| New method | `_lookup_moe` | DFC shape + EP Size |
| New enum | `QuerySource.PARTIAL` | Partial composite match |
| Delete | `_lookup_compute_by_shapes` | Replaced by `_query_by_shapes` |
| Delete | `_lookup_attention_by_params` | Replaced by `_query_by_attn_params` |
| Refactor | `_lookup_compute` | Extract params → call `_query_by_shapes` |
| Refactor | `_lookup_attention` | Extract params → call `_query_by_attn_params` |
| Refactor | `_lookup_composite` generic | Inline CSV loop → call `_query_by_shapes` |
| Refactor | `_lookup_composite_decomposed` | Partial match + call shared cores |
| Extend | `SubKernelSpec` | Add `tc_input_count`, `alternate_kernel_types` |
| Extend | `ProfilingDataSource.__init__` | Accept `parallel_config` for `ep_size` |
| Config | op_mapping.yaml | DFC: add `query_mode: moe_fused` |
| Config | model_runner.py | Pass `parallel_config` to `ProfilingDataSource` |

## Unchanged

- `_inputs_match` (per-row shape matching logic)
- `_lookup_elementwise` (output-shape matching, independent)
- `_query_comm_csv` (already shared)
- `_lookup_comm`, `_lookup_comm_for_composite` (comm paths)
- `_shapes_match_with_padding`, `fractal_nz_to_nd` (utilities)
- All decomposer functions (shape fixes are separate from this refactor)

## Out of Scope (separate work)

- Decomposer shape bugs (MLA head_dim=576→512, T=batch_size→num_tokens, TransposeBMM dim order)
- `_BLOCK_SIZES` adding block size 8
- MLA zero_cost ops (quantize dim=2304/256, concat_and_cache_mla, elementwise Mul/Add)
- FIA TP=8 prefill data collection
