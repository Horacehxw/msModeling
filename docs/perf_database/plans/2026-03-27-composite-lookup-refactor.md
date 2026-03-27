# Composite Lookup Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor profiling query paths to eliminate duplicated logic, fix DSV3 MLA MISSes, add DFC EP Size matching, and support composite partial match.

**Architecture:** Extract two shared query cores (`_query_by_shapes`, `_query_by_attn_params`) from duplicated code in `_lookup_compute`, `_lookup_compute_by_shapes`, and `_lookup_attention`/`_lookup_attention_by_params`. Add standalone `_lookup_moe` for DFC EP Size matching. Change composite decomposed path to accumulate partial results instead of early-returning on first sub-kernel MISS.

**Tech Stack:** Python 3.10+, pytest, PyTorch (meta tensors), pandas (CSV loading)

**Spec:** `docs/perf_database/specs/2026-03-27-composite-lookup-refactor-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `tensor_cast/performance_model/profiling_database/data_source.py` | Modify | Add `QuerySource.PARTIAL` |
| `tensor_cast/performance_model/profiling_database/profiling_data_source.py` | Modify | Core refactor: extract shared cores, add `_lookup_moe`, partial match, extend `SubKernelSpec`, extend `__init__` |
| `tensor_cast/performance_model/empirical.py` | Modify | Handle `PARTIAL` results in metrics |
| `tensor_cast/core/model_runner.py` | Modify | Pass `parallel_config` to `ProfilingDataSource` |
| `tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5/op_mapping.yaml` | Modify | DFC `query_mode: moe_fused` |
| `tests/perf_database/test_profiling_data_source.py` | Modify | Tests for `_query_by_shapes`, `_lookup_moe`, partial match |
| `tests/perf_database/test_mla_decomposition.py` | Modify | Tests for `SubKernelSpec.tc_input_count` |
| `tests/perf_database/test_empirical_metrics.py` | Modify | Tests for PARTIAL metric handling |

---

### Task 1: Add `QuerySource.PARTIAL` enum

**Files:**
- Modify: `tensor_cast/performance_model/profiling_database/data_source.py:10-13`
- Test: `tests/perf_database/test_profiling_data_source.py`

- [ ] **Step 1: Write test for PARTIAL enum existence**

```python
# Add to tests/perf_database/test_profiling_data_source.py, after existing imports
def test_query_source_partial_exists():
    """QuerySource.PARTIAL enum value exists for composite partial match."""
    assert hasattr(QuerySource, "PARTIAL")
    assert QuerySource.PARTIAL.value is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.10 -m pytest tests/perf_database/test_profiling_data_source.py::test_query_source_partial_exists -v`
Expected: FAIL with `AttributeError: PARTIAL`

- [ ] **Step 3: Add PARTIAL to QuerySource**

In `tensor_cast/performance_model/profiling_database/data_source.py`, change:

```python
class QuerySource(Enum):
    MEASURED = auto()
    INTERPOLATED = auto()
    EXTRAPOLATED = auto()
    PARTIAL = auto()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3.10 -m pytest tests/perf_database/test_profiling_data_source.py::test_query_source_partial_exists -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tensor_cast/performance_model/profiling_database/data_source.py tests/perf_database/test_profiling_data_source.py
git commit -m "feat(perf-db): add QuerySource.PARTIAL for composite partial match"
```

---

### Task 2: Extend `SubKernelSpec` with `tc_input_count` and `alternate_kernel_types`

**Files:**
- Modify: `tensor_cast/performance_model/profiling_database/profiling_data_source.py:332-339`
- Test: `tests/perf_database/test_mla_decomposition.py`

- [ ] **Step 1: Write test for new SubKernelSpec fields**

Add to `tests/perf_database/test_mla_decomposition.py`:

```python
from tensor_cast.performance_model.profiling_database.profiling_data_source import (
    SubKernelSpec,
)


class TestSubKernelSpecExtension:
    def test_default_tc_input_count_is_none(self):
        spec = SubKernelSpec(
            kernel_type="MatMulV2",
            input_shapes=[(128, 5120)],
            dtype="DT_BF16",
        )
        assert spec.tc_input_count is None

    def test_tc_input_count_set(self):
        spec = SubKernelSpec(
            kernel_type="QuantBatchMatmulV3",
            input_shapes=[(4099, 7168), (2112, 7168)],
            dtype="DT_BF16",
            tc_input_count=2,
        )
        assert spec.tc_input_count == 2

    def test_alternate_kernel_types_default_none(self):
        spec = SubKernelSpec(
            kernel_type="MatMulV2",
            input_shapes=[(128, 5120)],
            dtype="DT_BF16",
        )
        assert spec.alternate_kernel_types is None

    def test_alternate_kernel_types_set(self):
        spec = SubKernelSpec(
            kernel_type="MatMulV2",
            input_shapes=[(128, 5120)],
            dtype="DT_BF16",
            alternate_kernel_types=["MatMulV3", "MatMulCommon"],
        )
        assert spec.alternate_kernel_types == ["MatMulV3", "MatMulCommon"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.10 -m pytest tests/perf_database/test_mla_decomposition.py::TestSubKernelSpecExtension -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'tc_input_count'`

- [ ] **Step 3: Extend SubKernelSpec dataclass**

In `profiling_data_source.py`, change the `SubKernelSpec` class at line 332:

```python
@dataclass
class SubKernelSpec:
    """Specification for a sub-kernel in composite decomposition."""

    kernel_type: str
    input_shapes: List[Tuple[int, ...]]
    dtype: str  # Profiling dtype string, e.g. "DT_BF16"
    query_mode: str = "compute"  # "compute" | "attention"
    attention_params: Optional[Dict[str, Any]] = field(default=None)
    tc_input_count: Optional[int] = None
    alternate_kernel_types: Optional[List[str]] = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3.10 -m pytest tests/perf_database/test_mla_decomposition.py::TestSubKernelSpecExtension -v`
Expected: PASS

- [ ] **Step 5: Run existing MLA decomposition tests to verify no regression**

Run: `python3.10 -m pytest tests/perf_database/test_mla_decomposition.py -v`
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add tensor_cast/performance_model/profiling_database/profiling_data_source.py tests/perf_database/test_mla_decomposition.py
git commit -m "feat(perf-db): extend SubKernelSpec with tc_input_count and alternate_kernel_types"
```

---

### Task 3: Extract `_query_by_shapes` shared core

**Files:**
- Modify: `tensor_cast/performance_model/profiling_database/profiling_data_source.py`
- Test: `tests/perf_database/test_profiling_data_source.py`

- [ ] **Step 1: Write tests for `_query_by_shapes`**

Add to `tests/perf_database/test_profiling_data_source.py`:

```python
# --- _query_by_shapes tests ---

QUERY_BY_SHAPES_OP_MAPPING = """
version: "test"
device: TEST_DEVICE

operator_mappings:
  "aten.mm.default":
    kernel_type: MatMulV2
    alternate_kernel_types: [MatMulV3]
"""

QUERY_BY_SHAPES_MATMULV2_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"136,5120;5120,768","DT_BF16;DT_BF16","ND;ND","136,768","DT_BF16","ND",45.3
"""

QUERY_BY_SHAPES_MATMULV3_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"256,5120;5120,768","DT_BF16;DT_BF16","ND;ND","256,768","DT_BF16","ND",55.0
"""

# CSV with 4 inputs (like QuantBatchMatmulV3: activation, FRACTAL_NZ weight, bias, bias)
QUERY_BY_SHAPES_QBMV3_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"4099,7168;66,448,16,32;2112;2112","DT_BF16;DT_BF16;DT_BF16;DT_BF16","ND;FRACTAL_NZ;ND;ND","4099,2112","DT_BF16","ND",100.5
"""


@pytest.fixture
def query_shapes_data_dir(tmp_path):
    d = tmp_path / "qbs"
    d.mkdir()
    (d / "op_mapping.yaml").write_text(QUERY_BY_SHAPES_OP_MAPPING)
    (d / "MatMulV2.csv").write_text(QUERY_BY_SHAPES_MATMULV2_CSV.strip())
    (d / "MatMulV3.csv").write_text(QUERY_BY_SHAPES_MATMULV3_CSV.strip())
    (d / "QuantBatchMatmulV3.csv").write_text(QUERY_BY_SHAPES_QBMV3_CSV.strip())
    return d


class TestQueryByShapes:
    def test_primary_kernel_hit(self, query_shapes_data_dir):
        ds = ProfilingDataSource(query_shapes_data_dir)
        tc_inputs = [
            ((136, 5120), torch.bfloat16),
            ((5120, 768), torch.bfloat16),
        ]
        lat = ds._query_by_shapes(["MatMulV2"], tc_inputs)
        assert lat is not None
        assert abs(lat - 45.3) < 0.01

    def test_alternate_kernel_fallback(self, query_shapes_data_dir):
        """Primary misses, alternate hits."""
        ds = ProfilingDataSource(query_shapes_data_dir)
        tc_inputs = [
            ((256, 5120), torch.bfloat16),
            ((5120, 768), torch.bfloat16),
        ]
        lat = ds._query_by_shapes(["MatMulV2", "MatMulV3"], tc_inputs)
        assert lat is not None
        assert abs(lat - 55.0) < 0.01

    def test_all_miss_returns_none(self, query_shapes_data_dir):
        ds = ProfilingDataSource(query_shapes_data_dir)
        tc_inputs = [
            ((999, 5120), torch.bfloat16),
            ((5120, 768), torch.bfloat16),
        ]
        lat = ds._query_by_shapes(["MatMulV2", "MatMulV3"], tc_inputs)
        assert lat is None

    def test_tc_input_count_truncates_csv(self, query_shapes_data_dir):
        """tc_input_count=2 allows matching CSV with 4 inputs using only first 2."""
        ds = ProfilingDataSource(query_shapes_data_dir)
        # Decomposer provides 2 shapes; CSV has 4 inputs (act, FRACTAL_NZ weight, bias, bias)
        # FRACTAL_NZ (66,448,16,32) -> ND (2112,7168)
        # TC weight (2112,7168) matches CSV weight (2112,7168) after FRACTAL_NZ restore
        tc_inputs = [
            ((4099, 7168), torch.bfloat16),
            ((2112, 7168), torch.bfloat16),
        ]
        lat = ds._query_by_shapes(
            ["QuantBatchMatmulV3"], tc_inputs, tc_input_count=2
        )
        assert lat is not None
        assert abs(lat - 100.5) < 0.01

    def test_without_tc_input_count_misses_4_input_csv(self, query_shapes_data_dir):
        """Without tc_input_count, 2 TC inputs vs 4 CSV inputs → MISS."""
        ds = ProfilingDataSource(query_shapes_data_dir)
        tc_inputs = [
            ((4099, 7168), torch.bfloat16),
            ((2112, 7168), torch.bfloat16),
        ]
        lat = ds._query_by_shapes(["QuantBatchMatmulV3"], tc_inputs)
        assert lat is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3.10 -m pytest tests/perf_database/test_profiling_data_source.py::TestQueryByShapes -v`
Expected: FAIL with `AttributeError: 'ProfilingDataSource' object has no attribute '_query_by_shapes'`

- [ ] **Step 3: Implement `_query_by_shapes`**

Add new method to `ProfilingDataSource` class. Extract the core logic from `_lookup_compute` (L1622-1718):

```python
def _query_by_shapes(
    self,
    kernel_types: List[str],
    tc_inputs: List[Tuple[Tuple[int, ...], torch.dtype]],
    tc_input_count: Optional[int] = None,
) -> Optional[float]:
    """Shared shape-matching query core.

    Iterates kernel_types (primary + alternates), loads CSV for each,
    matches tc_inputs against CSV rows via _inputs_match.

    Args:
        kernel_types: List of kernel type names to try in order.
        tc_inputs: List of (shape, dtype) tuples to match.
        tc_input_count: If set, truncate CSV comparison to first N inputs.
    Returns:
        Latency in microseconds, or None if no match.
    """
    for kernel_type in kernel_types:
        df = self._load_csv(kernel_type)
        if df is None:
            continue
        for _, row in df.iterrows():
            if self._inputs_match(
                tc_inputs, row, kernel_type=kernel_type, tc_input_count=tc_input_count
            ):
                lat = float(row[self._latency_col(df)])
                logger.debug(
                    "HIT (query_by_shapes) %s: shapes=%s -> %.1f us",
                    kernel_type,
                    [s for s, _ in tc_inputs],
                    lat,
                )
                return lat

    # MISS diagnostics
    primary = kernel_types[0] if kernel_types else "unknown"
    df = self._load_csv(primary)
    csv_shapes_list = []
    if df is not None:
        for _, row in df.iterrows():
            csv_shapes_list.append(str(row.get("Input Shapes", "")))
    if df is not None and len(df) > 0:
        csv_first_shapes = _parse_shape_str(str(df.iloc[0].get("Input Shapes", "")))
        effective_csv_count = len(csv_first_shapes)
        effective_tc_count = len(tc_inputs)
        if tc_input_count is not None:
            effective_csv_count = min(effective_csv_count, tc_input_count)
            effective_tc_count = min(effective_tc_count, tc_input_count)
        if primary in _SWIGLU_KERNELS and effective_tc_count == 2 and effective_csv_count == 1:
            effective_tc_count = 1
        if effective_tc_count != effective_csv_count:
            self.last_miss_reason = "input_count_mismatch"
        else:
            self.last_miss_reason = "shape_mismatch"
    else:
        self.last_miss_reason = "csv_not_found"
    logger.debug(
        "MISS %s: tc_shapes=%s, csv_shapes=%s",
        primary,
        [s for s, _ in tc_inputs],
        csv_shapes_list,
    )
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3.10 -m pytest tests/perf_database/test_profiling_data_source.py::TestQueryByShapes -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tensor_cast/performance_model/profiling_database/profiling_data_source.py tests/perf_database/test_profiling_data_source.py
git commit -m "feat(perf-db): extract _query_by_shapes shared core"
```

---

### Task 4: Refactor `_lookup_compute` to use `_query_by_shapes`

**Files:**
- Modify: `tensor_cast/performance_model/profiling_database/profiling_data_source.py:1622-1718`
- Test: `tests/perf_database/test_profiling_data_source.py`

- [ ] **Step 1: Refactor `_lookup_compute`**

Replace `_lookup_compute` body (L1622-1718) with:

```python
def _lookup_compute(
    self, op_invoke_info: "OpInvokeInfo", mapping: dict
) -> Optional[QueryResult]:
    kernel_types = [mapping["kernel_type"]]
    for alt in mapping.get("alternate_kernel_types", []):
        if alt not in kernel_types:
            kernel_types.append(alt)

    tc_inputs = self._extract_tensor_inputs(op_invoke_info)
    tc_input_count = mapping.get("tc_input_count")
    if tc_input_count is not None:
        tc_inputs = tc_inputs[:tc_input_count]

    lat = self._query_by_shapes(kernel_types, tc_inputs, tc_input_count)
    if lat is None:
        return None

    return QueryResult(
        latency_us=lat,
        confidence=1.0,
        source=QuerySource.MEASURED,
        details={"kernel_type": kernel_types[0]},
    )
```

- [ ] **Step 2: Run all existing tests to verify no regression**

Run: `python3.10 -m pytest tests/perf_database/test_profiling_data_source.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tensor_cast/performance_model/profiling_database/profiling_data_source.py
git commit -m "refactor(perf-db): _lookup_compute delegates to _query_by_shapes"
```

---

### Task 5: Refactor generic composite to use `_query_by_shapes`

**Files:**
- Modify: `tensor_cast/performance_model/profiling_database/profiling_data_source.py:881-985` (`_lookup_composite`)

- [ ] **Step 1: Replace inline CSV loop in `_lookup_composite`**

Replace the compute sub-kernel loop (L913-941) with a call to `_query_by_shapes`:

```python
def _lookup_composite(
    self, op_invoke_info: "OpInvokeInfo", mapping: dict
) -> Optional[QueryResult]:
    """Decompose composite ops and sum sub-kernel latencies."""
    func_str = _normalize_func_name(op_invoke_info.func)
    decomposer = COMPOSITE_DECOMPOSERS.get(func_str)
    if decomposer is not None:
        return self._lookup_composite_decomposed(
            op_invoke_info, mapping, decomposer
        )

    # Generic composite path (MC2 etc.)
    sub_kernels = mapping.get("sub_kernels", [])
    if not sub_kernels:
        self.last_miss_reason = "no_sub_kernels"
        return None

    tc_inputs = self._extract_tensor_inputs(op_invoke_info)
    tc_input_count = mapping.get("tc_input_count")
    if tc_input_count is not None:
        tc_inputs = tc_inputs[:tc_input_count]

    # --- Compute sub-kernels: try each until one matches ---
    compute_kernels = [k for k in sub_kernels if not k.startswith("hcom_")]
    compute_latency = self._query_by_shapes(compute_kernels, tc_inputs, tc_input_count)

    if compute_latency is None:
        return None

    # --- Communication sub-kernels ---
    comm_latency = 0.0
    has_comm = False
    for kernel_type in sub_kernels:
        if not kernel_type.startswith("hcom_"):
            continue
        has_comm = True
        lat = self._lookup_comm_for_composite(op_invoke_info, kernel_type)
        if lat is None:
            self.last_miss_reason = "comm_sub_kernel_miss"
            return None
        comm_latency += lat
        logger.debug("HIT (composite comm) %s: %.1f us", kernel_type, lat)

    return QueryResult(
        latency_us=compute_latency + comm_latency,
        confidence=0.9 if has_comm else 0.8,
        source=QuerySource.MEASURED,
        details={
            "kernel_type": compute_kernels[0] if compute_kernels else "unknown",
            "composite": True,
            "note": "compute + comm sub-kernels" if has_comm else "compute sub-kernel only",
        },
    )
```

- [ ] **Step 2: Run existing composite tests**

Run: `python3.10 -m pytest tests/perf_database/test_profiling_data_source.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tensor_cast/performance_model/profiling_database/profiling_data_source.py
git commit -m "refactor(perf-db): generic composite delegates to _query_by_shapes"
```

---

### Task 6: Extract `_query_by_attn_params` and refactor `_lookup_attention`

**Files:**
- Modify: `tensor_cast/performance_model/profiling_database/profiling_data_source.py`
- Test: `tests/perf_database/test_fia_enriched_lookup.py`

- [ ] **Step 1: Write test for `_query_by_attn_params`**

Add to `tests/perf_database/test_fia_enriched_lookup.py` (check existing test patterns first — the file likely has enriched CSV fixtures already):

```python
class TestQueryByAttnParams:
    """Test the shared _query_by_attn_params core directly."""

    def test_basic_hit(self, enriched_data_dir):
        """Direct call to core with matching params should hit."""
        ds = ProfilingDataSource(enriched_data_dir)
        # Use params matching whatever enriched CSV fixture exists
        # Adapt q_shape_3d, avg_seq_len, etc. to match the fixture
        params = {
            "q_shape_3d": (136, 4, 128),  # adapt to fixture
            "avg_seq_len": 68,
            "sparse_mode": 3,
            "num_kv_heads": 4,
        }
        lat = ds._query_by_attn_params(
            ["FusedInferAttentionScore"], params, "DT_BF16"
        )
        # Verify returns a float (exact value depends on fixture)
        assert lat is not None
        assert isinstance(lat, float)

    def test_alternate_kernel_fallback(self, enriched_data_dir):
        """If primary kernel CSV doesn't exist, try alternate."""
        ds = ProfilingDataSource(enriched_data_dir)
        params = {
            "q_shape_3d": (136, 4, 128),
            "avg_seq_len": 68,
            "sparse_mode": 3,
            "num_kv_heads": 4,
        }
        # NonExistent primary, FusedInferAttentionScore as alternate
        lat = ds._query_by_attn_params(
            ["NonExistentKernel", "FusedInferAttentionScore"], params, "DT_BF16"
        )
        assert lat is not None
```

Note: Adapt the params to match existing fixtures in `test_fia_enriched_lookup.py`. Read the file first and use its fixture data.

- [ ] **Step 2: Implement `_query_by_attn_params`**

Extract the CSV iteration loop from `_lookup_attention` (L1407-1488) into a standalone method. The new method iterates `kernel_types`:

```python
def _query_by_attn_params(
    self,
    kernel_types: List[str],
    params: Dict[str, Any],
    dtype_str: str,
) -> Optional[float]:
    """Shared FIA attention params query core.

    Args:
        kernel_types: Kernel type names to try in order.
        params: Must contain q_shape_3d, avg_seq_len.
            Optional: sparse_mode, num_kv_heads.
        dtype_str: Profiling dtype string, e.g. "DT_BF16".
    Returns:
        Latency in microseconds, or None.
    """
    q_shape_3d = params.get("q_shape_3d")
    target_avg_seq = params.get("avg_seq_len")
    if q_shape_3d is None or target_avg_seq is None:
        return None

    target_sparse = params.get("sparse_mode")
    target_kv_heads = params.get("num_kv_heads")
    tc_N, tc_D = q_shape_3d[1], q_shape_3d[2]
    head_dim = tc_D

    for kernel_type in kernel_types:
        df = self._load_csv(kernel_type)
        if df is None:
            continue

        avg_seq_col = None
        if "Runtime avg_seq_len" in df.columns:
            avg_seq_col = "Runtime avg_seq_len"
        elif "avg_seq_len" in df.columns:
            avg_seq_col = "avg_seq_len"
        else:
            continue

        if "Input Shapes" not in df.columns:
            continue

        has_sparse_col = "Runtime sparse_mode" in df.columns
        has_kv_heads_col = "Runtime num_key_value_heads" in df.columns
        latency_col = self._latency_col(df)

        for _, row in df.iterrows():
            csv_avg_seq = int(row[avg_seq_col])
            if csv_avg_seq < 0:
                continue

            shapes_str = str(row.get("Input Shapes", "")).strip('"')
            csv_q_raw = _parse_fia_q_shape(shapes_str)
            if csv_q_raw is None:
                continue
            csv_q_3d = _normalize_fia_q_shape(csv_q_raw, head_dim)
            if csv_q_3d is None:
                continue

            csv_N, csv_D = csv_q_3d[1], csv_q_3d[2]

            csv_dtypes_str = str(row.get("Input Data Types", ""))
            csv_first_dtype = csv_dtypes_str.split(";")[0].strip() if csv_dtypes_str else ""
            if dtype_str != csv_first_dtype:
                continue

            if tc_N != csv_N or tc_D != csv_D:
                continue

            if has_sparse_col and target_sparse is not None and int(row["Runtime sparse_mode"]) != target_sparse:
                continue

            if has_kv_heads_col and target_kv_heads is not None and int(row["Runtime num_key_value_heads"]) != target_kv_heads:
                continue

            if target_avg_seq != csv_avg_seq:
                continue

            tc_T = q_shape_3d[0]
            csv_T = csv_q_3d[0]
            if tc_T != csv_T and not _is_block_padded(tc_T, csv_T) and not _is_block_padded(csv_T, tc_T):
                continue

            lat = float(row[latency_col])
            logger.debug(
                "HIT (query_by_attn_params) %s: params=%s -> %.1f us",
                kernel_type, params, lat,
            )
            return lat

    self.last_miss_reason = "shape_mismatch"
    return None
```

- [ ] **Step 3: Refactor `_lookup_attention` to delegate**

Replace `_lookup_attention` body with parameter extraction + delegation:

```python
def _lookup_attention(
    self, op_invoke_info: "OpInvokeInfo", mapping: dict
) -> Optional[QueryResult]:
    """Query FIA enriched CSV."""
    kernel_types = [mapping.get("kernel_type")]
    for alt in mapping.get("alternate_kernel_types", []):
        if alt not in kernel_types:
            kernel_types.append(alt)

    args = op_invoke_info.args
    if len(args) < 7:
        self.last_miss_reason = "insufficient_args"
        return None

    query = args[0]
    key = args[1]
    seq_lens = args[6] if len(args) > 6 else None
    query_lens = args[7] if len(args) > 7 else None

    if not isinstance(query, torch.Tensor):
        self.last_miss_reason = "query_not_tensor"
        return None

    tc_dtype_str = DTYPE_MAP.get(query.dtype)
    if tc_dtype_str is None:
        self.last_miss_reason = "dtype_unmapped"
        return None

    head_dim = key.shape[-1] if isinstance(key, torch.Tensor) and key.ndim >= 1 else 0
    tc_q_3d = _normalize_fia_q_shape(tuple(query.shape), head_dim)
    if tc_q_3d is None:
        self.last_miss_reason = "q_shape_normalize_failed"
        return None

    if seq_lens is not None and isinstance(seq_lens, torch.Tensor):
        try:
            tc_avg_seq_len = int(seq_lens.float().mean().item())
        except Exception:
            self.last_miss_reason = "invalid_seq_lens"
            return None
    else:
        self.last_miss_reason = "missing_seq_lens"
        return None

    tc_sparse_mode = _infer_sparse_mode(query_lens)
    tc_num_kv_heads = key.shape[-2] if isinstance(key, torch.Tensor) and key.ndim >= 2 else None

    params = {
        "q_shape_3d": tc_q_3d,
        "avg_seq_len": tc_avg_seq_len,
        "sparse_mode": tc_sparse_mode,
        "num_kv_heads": tc_num_kv_heads,
    }

    lat = self._query_by_attn_params(kernel_types, params, tc_dtype_str)
    if lat is None:
        return None

    return QueryResult(
        latency_us=lat,
        confidence=0.9,
        source=QuerySource.MEASURED,
        details={
            "kernel_type": kernel_types[0],
            "avg_seq_len": tc_avg_seq_len,
            "sparse_mode": tc_sparse_mode,
            "num_kv_heads": tc_num_kv_heads,
        },
    )
```

- [ ] **Step 4: Delete `_lookup_attention_by_params`**

Remove the entire `_lookup_attention_by_params` method (L1074-1178).

- [ ] **Step 5: Run FIA tests**

Run: `python3.10 -m pytest tests/perf_database/test_fia_enriched_lookup.py -v`
Expected: All PASS

- [ ] **Step 6: Run full test suite**

Run: `python3.10 -m pytest tests/perf_database/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add tensor_cast/performance_model/profiling_database/profiling_data_source.py tests/perf_database/test_fia_enriched_lookup.py
git commit -m "refactor(perf-db): extract _query_by_attn_params, _lookup_attention delegates"
```

---

### Task 7: Refactor `_lookup_composite_decomposed` for partial match + shared cores

**Files:**
- Modify: `tensor_cast/performance_model/profiling_database/profiling_data_source.py:987-1038`
- Test: `tests/perf_database/test_profiling_data_source.py`

- [ ] **Step 1: Write tests for partial match**

Add to `tests/perf_database/test_profiling_data_source.py`:

```python
# --- Composite partial match tests ---

PARTIAL_MATCH_OP_MAPPING = """
version: "test"
device: TEST_DEVICE

operator_mappings:
  "tensor_cast.mlapo_quant.default":
    composite: true
    sub_kernels: [QuantBatchMatmulV3, KvRmsNormRopeCache]
"""

PARTIAL_MATCH_QBMV3_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"4099,7168;2112,7168","DT_BF16;DT_BF16","ND;ND","4099,2112","DT_BF16","ND",80.0
"""


@pytest.fixture
def partial_match_data_dir(tmp_path):
    d = tmp_path / "partial"
    d.mkdir()
    (d / "op_mapping.yaml").write_text(PARTIAL_MATCH_OP_MAPPING)
    (d / "QuantBatchMatmulV3.csv").write_text(PARTIAL_MATCH_QBMV3_CSV.strip())
    # No KvRmsNormRopeCache.csv — will MISS
    return d


class TestCompositePartialMatch:
    def test_partial_returns_partial_source(self, partial_match_data_dir):
        """When some sub-kernels hit and others miss, return PARTIAL."""
        ds = ProfilingDataSource(partial_match_data_dir)

        # Simulate: decomposer returns 2 specs, first matches, second doesn't
        from tensor_cast.performance_model.profiling_database.profiling_data_source import (
            SubKernelSpec,
        )

        specs = [
            SubKernelSpec(
                kernel_type="QuantBatchMatmulV3",
                input_shapes=[(4099, 7168), (2112, 7168)],
                dtype="DT_BF16",
            ),
            SubKernelSpec(
                kernel_type="KvRmsNormRopeCache",
                input_shapes=[(4099, 576)],
                dtype="DT_BF16",
            ),
        ]

        # Manually call the decomposed lookup with a mock decomposer
        op = _make_op_info(
            torch.ops.tensor_cast.mlapo_quant.default,
            [torch.empty(4099, 7168, device="meta", dtype=torch.bfloat16)],
        )
        result = ds._lookup_composite_decomposed(
            op, {}, lambda op, m: specs
        )
        assert result is not None
        assert result.source == QuerySource.PARTIAL
        assert result.latency_us == 80.0  # only first sub-kernel
        assert "missed_kernels" in result.details
        assert "KvRmsNormRopeCache" in result.details["missed_kernels"]
        assert result.confidence == pytest.approx(0.5)  # 1/2

    def test_all_hit_returns_measured(self, partial_match_data_dir):
        """When all sub-kernels hit, return MEASURED with full confidence."""
        ds = ProfilingDataSource(partial_match_data_dir)

        specs = [
            SubKernelSpec(
                kernel_type="QuantBatchMatmulV3",
                input_shapes=[(4099, 7168), (2112, 7168)],
                dtype="DT_BF16",
            ),
        ]

        op = _make_op_info(
            torch.ops.tensor_cast.mlapo_quant.default,
            [torch.empty(4099, 7168, device="meta", dtype=torch.bfloat16)],
        )
        result = ds._lookup_composite_decomposed(
            op, {}, lambda op, m: specs
        )
        assert result is not None
        assert result.source == QuerySource.MEASURED
        assert result.confidence == 0.8

    def test_all_miss_returns_partial_zero_latency(self, partial_match_data_dir):
        """When all sub-kernels miss, return PARTIAL with 0 latency."""
        ds = ProfilingDataSource(partial_match_data_dir)

        specs = [
            SubKernelSpec(
                kernel_type="KvRmsNormRopeCache",
                input_shapes=[(4099, 576)],
                dtype="DT_BF16",
            ),
        ]

        op = _make_op_info(
            torch.ops.tensor_cast.mlapo_quant.default,
            [torch.empty(4099, 7168, device="meta", dtype=torch.bfloat16)],
        )
        result = ds._lookup_composite_decomposed(
            op, {}, lambda op, m: specs
        )
        assert result is not None
        assert result.source == QuerySource.PARTIAL
        assert result.latency_us == 0.0
        assert result.confidence == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3.10 -m pytest tests/perf_database/test_profiling_data_source.py::TestCompositePartialMatch -v`
Expected: FAIL — current code returns None on first MISS

- [ ] **Step 3: Rewrite `_lookup_composite_decomposed`**

```python
def _lookup_composite_decomposed(
    self,
    op_invoke_info: "OpInvokeInfo",
    mapping: dict,
    decomposer: Callable,
) -> Optional[QueryResult]:
    """Query composite op using registered decomposer (MLA/MLAPO).

    Accumulates sub-kernel latencies. If some sub-kernels miss,
    returns PARTIAL result with partial latency and confidence
    proportional to hit ratio.
    """
    specs = decomposer(op_invoke_info, mapping)
    if not specs:
        self.last_miss_reason = "decompose_failed"
        return None

    total_latency = 0.0
    hit_kernels = []
    missed_kernels = []

    for spec in specs:
        kernel_types = [spec.kernel_type] + (spec.alternate_kernel_types or [])

        if spec.query_mode == "attention" and spec.attention_params:
            lat = self._query_by_attn_params(
                kernel_types, spec.attention_params, spec.dtype
            )
        else:
            torch_dtype = None
            for k, v in DTYPE_MAP.items():
                if v == spec.dtype:
                    torch_dtype = k
                    break
            if torch_dtype is None:
                missed_kernels.append(spec.kernel_type)
                continue

            tc_inputs = [(shape, torch_dtype) for shape in spec.input_shapes]
            lat = self._query_by_shapes(
                kernel_types, tc_inputs, spec.tc_input_count
            )

        if lat is not None:
            total_latency += lat
            hit_kernels.append(spec.kernel_type)
        else:
            missed_kernels.append(spec.kernel_type)

    if missed_kernels:
        self.last_miss_reason = f"sub_kernel_miss:{','.join(missed_kernels)}"
        confidence = len(hit_kernels) / len(specs) if specs else 0.0
        logger.debug(
            "PARTIAL (composite decomposed) %s: hit=%s, missed=%s, latency=%.1f us",
            _normalize_func_name(op_invoke_info.func),
            hit_kernels,
            missed_kernels,
            total_latency,
        )
        return QueryResult(
            latency_us=total_latency,
            confidence=confidence,
            source=QuerySource.PARTIAL,
            details={
                "kernel_type": hit_kernels,
                "missed_kernels": missed_kernels,
                "composite": True,
                "partial": True,
            },
        )

    logger.debug(
        "HIT (composite decomposed) %s: sub_kernels=%s, total=%.1f us",
        _normalize_func_name(op_invoke_info.func),
        hit_kernels,
        total_latency,
    )
    return QueryResult(
        latency_us=total_latency,
        confidence=0.8,
        source=QuerySource.MEASURED,
        details={
            "kernel_type": hit_kernels,
            "composite": True,
        },
    )
```

- [ ] **Step 4: Delete `_lookup_compute_by_shapes`**

Remove the entire `_lookup_compute_by_shapes` method (L1040-1072). It is now fully replaced by `_query_by_shapes`.

- [ ] **Step 5: Run tests**

Run: `python3.10 -m pytest tests/perf_database/test_profiling_data_source.py::TestCompositePartialMatch -v`
Expected: All PASS

Run: `python3.10 -m pytest tests/perf_database/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add tensor_cast/performance_model/profiling_database/profiling_data_source.py tests/perf_database/test_profiling_data_source.py
git commit -m "feat(perf-db): composite partial match + delete _lookup_compute_by_shapes"
```

---

### Task 8: Handle `PARTIAL` in `EmpiricalPerformanceModel` metrics

**Files:**
- Modify: `tensor_cast/performance_model/empirical.py:244-283` (`process_op`)
- Modify: `tensor_cast/performance_model/empirical.py:301-392` (`log_stats`)
- Test: `tests/perf_database/test_empirical_metrics.py`

- [ ] **Step 1: Write test for PARTIAL handling in process_op**

Add to `tests/perf_database/test_empirical_metrics.py`:

```python
class TestPartialMetrics:
    def test_partial_uses_latency_but_counts_as_miss(self):
        """PARTIAL result: latency is used in E2E, but counted as MISS in metrics."""
        mock_ds = MagicMock(spec=DataSourcePerformanceModel)
        mock_ds.lookup.return_value = QueryResult(
            latency_us=100.0,
            confidence=0.5,
            source=QuerySource.PARTIAL,
            details={
                "kernel_type": ["QuantBatchMatmulV3"],
                "missed_kernels": ["KvRmsNormRopeCache"],
                "composite": True,
                "partial": True,
            },
        )

        mock_device = MagicMock()
        mock_device.flops = 1e12
        mock_device.bandwidth = 1e12

        mock_fallback = MagicMock(spec=PerformanceModel)
        mock_fallback.process_op.return_value = PerformanceModel.Result(
            execution_time_s=200e-6,
            statistics={},
        )
        mock_fallback.get_classifiers.return_value = []

        pm = EmpiricalPerformanceModel(mock_device, mock_ds, mock_fallback)

        op = MagicMock()
        op.func = torch.ops.tensor_cast.mlapo_quant.default
        op.args = (torch.empty(4099, 7168, device="meta", dtype=torch.bfloat16),)

        result = pm.process_op(op)

        # PARTIAL uses empirical latency
        assert abs(result.execution_time_s - 100e-6) < 1e-9

        # But counts as MISS in stats
        stats = pm.get_stats()
        assert stats["miss"] == 1
        assert stats["hit"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.10 -m pytest tests/perf_database/test_empirical_metrics.py::TestPartialMetrics -v`
Expected: FAIL — current code treats any non-None result as HIT

- [ ] **Step 3: Modify `process_op` to handle PARTIAL**

In `empirical.py`, update `process_op` (L244-283):

```python
@override
def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
    result = self.data_source.lookup(op_invoke_info)
    func_name = str(op_invoke_info.func).removeprefix("torch.ops.")

    # M5: always compute analytic latency as weight
    analytic_result = self.fallback_model.process_op(op_invoke_info)
    self._total_latency_sum += analytic_result.execution_time_s

    if result is not None and result.source != QuerySource.PARTIAL:
        # Full HIT
        self._stats["hit"] += 1
        self._hit_latency_sum += analytic_result.execution_time_s
        empirical_s = result.latency_us * 1e-6
        self._empirical_hit_total_s += empirical_s
        kernel_type = result.details.get("kernel_type", "?")
        tc_shapes = [
            tuple(a.shape)
            for a in op_invoke_info.args
            if isinstance(a, torch.Tensor)
        ]
        shape_sig = tuple(tc_shapes)
        self._hit_details.append((func_name, kernel_type, shape_sig, empirical_s))
        return PerformanceModel.Result(
            execution_time_s=empirical_s,
            statistics={
                "source": result.source.name,
                "confidence": result.confidence,
                **result.details,
            },
        )

    if result is not None and result.source == QuerySource.PARTIAL:
        # PARTIAL: use empirical latency but count as MISS
        self._stats["miss"] += 1
        empirical_s = result.latency_us * 1e-6
        tc_shapes = [
            tuple(a.shape) for a in op_invoke_info.args if isinstance(a, torch.Tensor)
        ]
        missed_kernels = result.details.get("missed_kernels", [])
        reason = f"partial:{','.join(missed_kernels)}"
        self._miss_details.append(
            (func_name, reason, tc_shapes, analytic_result.execution_time_s)
        )
        return PerformanceModel.Result(
            execution_time_s=empirical_s,
            statistics={
                "source": result.source.name,
                "confidence": result.confidence,
                **result.details,
            },
        )

    # Full MISS
    self._stats["miss"] += 1
    tc_shapes = [
        tuple(a.shape) for a in op_invoke_info.args if isinstance(a, torch.Tensor)
    ]
    reason = getattr(self.data_source, "last_miss_reason", "unknown")
    self._miss_details.append(
        (func_name, reason, tc_shapes, analytic_result.execution_time_s)
    )
    return analytic_result
```

- [ ] **Step 4: Run tests**

Run: `python3.10 -m pytest tests/perf_database/test_empirical_metrics.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tensor_cast/performance_model/empirical.py tests/perf_database/test_empirical_metrics.py
git commit -m "feat(perf-db): handle PARTIAL results in EmpiricalPerformanceModel metrics"
```

---

### Task 9: Add `_lookup_moe` for DFC EP Size matching

**Files:**
- Modify: `tensor_cast/performance_model/profiling_database/profiling_data_source.py`
- Modify: op_mapping.yaml (DFC entry)
- Test: `tests/perf_database/test_profiling_data_source.py`

- [ ] **Step 1: Write tests for `_lookup_moe`**

Add to `tests/perf_database/test_profiling_data_source.py`:

```python
# --- DFC EP Size matching tests ---

MOE_OP_MAPPING = """
version: "test"
device: TEST_DEVICE

operator_mappings:
  "tensor_cast.dispatch_ffn_combine.default":
    kernel_type: DispatchFFNCombine
    query_mode: moe_fused
    tc_input_count: 1
"""

MOE_DFC_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,EP Size,Average Duration(us)
"513,7168","DT_BF16","ND","513,7168","DT_BF16","ND",16,235.0
"513,7168","DT_BF16","ND","513,7168","DT_BF16","ND",8,180.0
"1024,7168","DT_BF16","ND","1024,7168","DT_BF16","ND",16,400.0
"""


@pytest.fixture
def moe_data_dir(tmp_path):
    d = tmp_path / "moe"
    d.mkdir()
    (d / "op_mapping.yaml").write_text(MOE_OP_MAPPING)
    (d / "DispatchFFNCombine.csv").write_text(MOE_DFC_CSV.strip())
    return d


class TestLookupMoe:
    def test_ep_size_exact_match(self, moe_data_dir):
        """Same shape, different EP sizes → match the right one."""
        ds = ProfilingDataSource(moe_data_dir, ep_size=16)
        op = _make_op_info(
            torch.ops.tensor_cast.dispatch_ffn_combine.default,
            [
                torch.empty(513, 7168, device="meta", dtype=torch.bfloat16),
                torch.empty(513, dtype=torch.int64, device="meta"),  # expert_indices
            ],
        )
        result = ds.lookup(op)
        assert result is not None
        assert abs(result.latency_us - 235.0) < 0.01

    def test_ep_size_8_matches_different_row(self, moe_data_dir):
        ds = ProfilingDataSource(moe_data_dir, ep_size=8)
        op = _make_op_info(
            torch.ops.tensor_cast.dispatch_ffn_combine.default,
            [
                torch.empty(513, 7168, device="meta", dtype=torch.bfloat16),
                torch.empty(513, dtype=torch.int64, device="meta"),
            ],
        )
        result = ds.lookup(op)
        assert result is not None
        assert abs(result.latency_us - 180.0) < 0.01

    def test_ep_size_not_configured_misses(self, moe_data_dir):
        """CSV has EP Size column but ProfilingDataSource has no ep_size → MISS."""
        ds = ProfilingDataSource(moe_data_dir)  # no ep_size
        op = _make_op_info(
            torch.ops.tensor_cast.dispatch_ffn_combine.default,
            [
                torch.empty(513, 7168, device="meta", dtype=torch.bfloat16),
                torch.empty(513, dtype=torch.int64, device="meta"),
            ],
        )
        result = ds.lookup(op)
        assert result is None
        assert ds.last_miss_reason == "ep_size_not_configured"

    def test_shape_miss(self, moe_data_dir):
        """Shape doesn't match any CSV row."""
        ds = ProfilingDataSource(moe_data_dir, ep_size=16)
        op = _make_op_info(
            torch.ops.tensor_cast.dispatch_ffn_combine.default,
            [
                torch.empty(999, 7168, device="meta", dtype=torch.bfloat16),
                torch.empty(999, dtype=torch.int64, device="meta"),
            ],
        )
        result = ds.lookup(op)
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3.10 -m pytest tests/perf_database/test_profiling_data_source.py::TestLookupMoe -v`
Expected: FAIL

- [ ] **Step 3: Extend `ProfilingDataSource.__init__` to accept `parallel_config`**

Modify `__init__` (L650-667):

```python
def __init__(
    self,
    data_dir: str | Path,
    device_profile: Optional[DeviceProfile] = None,
    parallel_config: Optional["ParallelConfig"] = None,
    *,
    ep_size: Optional[int] = None,
):
    self.data_dir = Path(data_dir)
    self.comm_grid = device_profile.comm_grid if device_profile else None
    # EP size: prefer parallel_config, fallback to explicit kwarg
    if parallel_config is not None:
        self.ep_size = parallel_config.expert_parallel_size
    else:
        self.ep_size = ep_size
    self._op_mapping = self._load_op_mapping()
    self._csv_cache: Dict[str, Optional[pd.DataFrame]] = {}
    comm_ref = self._op_mapping.get("communication_data_ref")
    if comm_ref:
        self._comm_data_dir = (self.data_dir / comm_ref).resolve()
    else:
        self._comm_data_dir = self.data_dir
    self.last_miss_reason: str = ""
```

Note: `ep_size` keyword-only arg is for test convenience — tests can pass `ep_size=16` without constructing a full `ParallelConfig`.

- [ ] **Step 4: Implement `_lookup_moe`**

```python
def _lookup_moe(
    self, op_invoke_info: "OpInvokeInfo", mapping: dict
) -> Optional[QueryResult]:
    """Query DFC CSV: shape match + EP Size exact match."""
    kernel_type = mapping.get("kernel_type")
    if not kernel_type:
        self.last_miss_reason = "unmapped"
        return None

    df = self._load_csv(kernel_type)
    if df is None:
        self.last_miss_reason = "csv_not_found"
        return None

    has_ep_col = "EP Size" in df.columns

    # EP Size required when CSV has the column
    if has_ep_col and self.ep_size is None:
        self.last_miss_reason = "ep_size_not_configured"
        return None

    tc_inputs = self._extract_tensor_inputs(op_invoke_info)
    tc_input_count = mapping.get("tc_input_count")
    if tc_input_count is not None:
        tc_inputs = tc_inputs[:tc_input_count]

    latency_col = self._latency_col(df)

    for _, row in df.iterrows():
        if not self._inputs_match(
            tc_inputs, row, kernel_type=kernel_type, tc_input_count=tc_input_count
        ):
            continue
        if has_ep_col and self.ep_size is not None:
            csv_ep = int(row["EP Size"])
            if csv_ep != self.ep_size:
                continue
        lat = float(row[latency_col])
        logger.debug(
            "HIT (moe) %s: ep_size=%s -> %.1f us",
            kernel_type, self.ep_size, lat,
        )
        return QueryResult(
            latency_us=lat,
            confidence=1.0,
            source=QuerySource.MEASURED,
            details={"kernel_type": kernel_type, "ep_size": self.ep_size},
        )

    self.last_miss_reason = "shape_mismatch"
    return None
```

- [ ] **Step 5: Add `moe_fused` dispatch in `lookup()`**

In the `lookup` method (L838-877), add the `moe_fused` branch after `elementwise`:

```python
if mapping.get("query_mode") == "elementwise":
    return self._lookup_elementwise(op_invoke_info, mapping)
if mapping.get("query_mode") == "moe_fused":
    return self._lookup_moe(op_invoke_info, mapping)
```

- [ ] **Step 6: Run tests**

Run: `python3.10 -m pytest tests/perf_database/test_profiling_data_source.py::TestLookupMoe -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add tensor_cast/performance_model/profiling_database/profiling_data_source.py tests/perf_database/test_profiling_data_source.py
git commit -m "feat(perf-db): add _lookup_moe for DFC EP Size matching"
```

---

### Task 10: Update op_mapping.yaml and model_runner.py

**Files:**
- Modify: `tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5/op_mapping.yaml`
- Modify: `tensor_cast/core/model_runner.py:75-78`

- [ ] **Step 1: Update op_mapping.yaml DFC entry**

Find the `tensor_cast.dispatch_ffn_combine.default` entry and add `query_mode: moe_fused`:

```yaml
  "tensor_cast.dispatch_ffn_combine.default":
    kernel_type: DispatchFFNCombine
    query_mode: moe_fused
    tc_input_count: 1
    notes: >
      [HIGH] ... (keep existing notes)
```

- [ ] **Step 2: Update model_runner.py to pass parallel_config**

Change L75-78:

```python
data_source = ProfilingDataSource(
    profiling_database,
    self.device_profile,
    parallel_config=user_input.get_parallel_config(),
)
```

- [ ] **Step 3: Run full test suite**

Run: `python3.10 -m pytest tests/perf_database/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5/op_mapping.yaml tensor_cast/core/model_runner.py
git commit -m "feat(perf-db): DFC query_mode moe_fused + pass parallel_config to ProfilingDataSource"
```

---

### Task 11: Final regression test and lint

- [ ] **Step 1: Run full perf_database test suite**

Run: `python3.10 -m pytest tests/perf_database/ -v`
Expected: All PASS

- [ ] **Step 2: Lint**

Run: `lintrunner -a`
Expected: Clean or auto-fixed

- [ ] **Step 3: Commit lint fixes if any**

```bash
git add -u
git commit -m "style: lint fixes"
```

---

Plan complete and saved to `docs/perf_database/plans/2026-03-27-composite-lookup-refactor.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?