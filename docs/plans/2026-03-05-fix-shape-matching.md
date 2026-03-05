# Fix ProfilingDataSource Shape Matching — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Achieve >0% hit rate in ProfilingDataSource by fixing the TC command, shape matching, and input filtering — validated against Qwen3-32B BF16 profiling data.

**Architecture:** Fix the E2E flow in 3 layers: (1) correct TC CLI args to match profiling config, (2) add flexible shape matching in ProfilingDataSource (weight transpose, block-padding tolerance), (3) add `input_indices` support in op_mapping.yaml for ops with extra args (W8A8 scale tensors). All changes are in ProfilingDataSource — no changes to TC core dispatch.

**Tech Stack:** Python, PyTorch, pandas, YAML

**Scope:** Qwen3-32B BF16 prefill only. DSv3 W8A8 decode deferred (requires `input_indices` + MLA decomposition — separate plan).

---

## Background: Why 0% Hit Rate

Two root causes:

1. **Wrong TC quantization default**: TC defaults to `W8A8_DYNAMIC` (`user_config.py:33`), so ALL linears dispatch `static_quant_linear` → mapped to `QuantBatchMatmulV3`. But Qwen3 profiling is BF16 → uses `MatMulV2`. Fix: `--quantize-linear-action DISABLED`.

2. **Shape mismatches** (after kernel type is correct):
   - Block padding: TC pads seq 136→144 (`ceil(136/16)×16`)
   - Weight transpose: ND-format matmul weights use (N,K) in profiling vs (K,N) in TC
   - Embedding vocab sharding: TC uses full vocab, profiling uses TP-sharded

## Key Files

| File | Role |
|------|------|
| `tensor_cast/performance_model/perf_database/profiling_data_source.py` | Shape matching logic — main changes here |
| `tensor_cast/performance_model/perf_database/data/.../v0.14.0/op_mapping.yaml` | Op mapping config |
| `tests/perf_database/test_profiling_data_source.py` | Unit tests |
| `tensor_cast/core/user_config.py:33` | Default quant (W8A8_DYNAMIC) — read-only context |

## Dependency Graph

```
Task 1 (weight transpose)  ──┐
Task 2 (block padding)     ──┼── Task 4 (E2E validation)
Task 3 (input count)       ──┘
```

Tasks 1-3 are independent and can be done in parallel.

---

### Task 1: Weight Transpose Matching for ND-format Matmul Weights

**Problem:** For `aten.mm(x, w)`, TC weight shape is (K,N) but profiling CSV sometimes stores weight as (N,K) in ND format. Example: lm_head CSV `(9496,5120)` vs TC `(5120,9496)`.

Note: FRACTAL_NZ weights already match correctly after `fractal_nz_to_nd()` restoration. This only affects ND-format weights.

**Files:**
- Modify: `tensor_cast/performance_model/perf_database/profiling_data_source.py` — `_inputs_match()`
- Test: `tests/perf_database/test_profiling_data_source.py`

**Step 1: Write failing test**

Add to `tests/perf_database/test_profiling_data_source.py`:

```python
def test_matmul_weight_transpose_nd_match(tmp_path):
    """ND-format matmul weight (N,K) should match TC's (K,N) when transposed."""
    # Create MatMulV2.csv with lm_head shape: weight is (N,K) = (9496,5120)
    csv_content = (
        "Name,Type,Input Shapes,Input Data Types,Input Formats,Output Shapes,Duration(us)\n"
        'aclnnMm_MatMulV2,MatMulV2,"1,5120;9496,5120",DT_BF16;DT_BF16,ND;ND,"1,9496",10.5\n'
    )
    (tmp_path / "MatMulV2.csv").write_text(csv_content)
    # op_mapping
    op_mapping = {
        "version": "test",
        "operator_mappings": {
            "aten.mm.default": {"kernel_type": "MatMulV2"},
        },
    }
    import yaml
    (tmp_path / "op_mapping.yaml").write_text(yaml.dump(op_mapping))

    ds = ProfilingDataSource(tmp_path)

    # TC dispatches mm with weight as (K,N) = (5120,9496)
    op = _make_op_invoke_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(1, 5120, dtype=torch.bfloat16),    # activation (M,K)
            torch.empty(5120, 9496, dtype=torch.bfloat16),  # weight (K,N) — transposed vs CSV
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match with weight transpose"
    assert result.latency_us == 10.5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/perf_database/test_profiling_data_source.py::test_matmul_weight_transpose_nd_match -v`
Expected: FAIL — shapes `(5120,9496)` != `(9496,5120)`

**Step 3: Implement weight transpose matching**

In `profiling_data_source.py`, modify `_inputs_match()` to try transposing 2D weight tensors when the kernel type is a matmul variant:

```python
def _inputs_match(
    self,
    tc_inputs: List[Tuple[Tuple[int, ...], torch.dtype]],
    csv_row: pd.Series,
    kernel_type: str = "",
) -> bool:
    """Match TensorCast input shapes/dtypes against a CSV row.
    Handles FRACTAL_NZ restoration (design doc S4.9) and weight transpose."""
    csv_shapes = _parse_shape_str(str(csv_row.get("Input Shapes", "")))
    csv_dtypes = _parse_str_list(str(csv_row.get("Input Data Types", "")))
    csv_formats = _parse_str_list(str(csv_row.get("Input Formats", "")))

    if len(tc_inputs) != len(csv_shapes):
        return False

    # Matmul kernel types where weight transpose is expected
    _MATMUL_KERNELS = {"MatMulV2", "MatMul", "TransposeBatchMatMul"}

    for i, (tc_shape, tc_dtype) in enumerate(tc_inputs):
        # Check dtype
        expected_dtype = DTYPE_MAP.get(tc_dtype)
        if expected_dtype is None or i >= len(csv_dtypes):
            return False
        if expected_dtype != csv_dtypes[i]:
            return False

        # Get CSV shape, restore FRACTAL_NZ if needed
        csv_shape = csv_shapes[i]
        fmt = csv_formats[i] if i < len(csv_formats) else "ND"
        if fmt == "FRACTAL_NZ":
            csv_shape = fractal_nz_to_nd(csv_shape)

        if tc_shape == csv_shape:
            continue

        # Try weight transpose for 2D ND matmul weights (i >= 1)
        if (
            kernel_type in _MATMUL_KERNELS
            and i >= 1
            and fmt == "ND"
            and len(tc_shape) == 2
            and len(csv_shape) == 2
            and tc_shape == (csv_shape[1], csv_shape[0])
        ):
            continue

        return False

    return True
```

Update `_lookup_compute()` to pass `kernel_type` to `_inputs_match()`:

```python
if self._inputs_match(tc_inputs, row, kernel_type=kernel_type):
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/perf_database/test_profiling_data_source.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add tensor_cast/performance_model/perf_database/profiling_data_source.py tests/perf_database/test_profiling_data_source.py
git commit -m "feat(perf-db): add weight transpose matching for ND matmul weights"
```

---

### Task 2: Block-Padding Tolerance in Shape Matching

**Problem:** TC pads sequence length to block_size multiples: `ceil(136/16)×16 = 144`. CSV has actual seq=136. Shapes like `(144,5120)` vs `(136,5120)` cause MISS.

**Approach:** When exact match fails, check if TC shape is a block-padded version of the CSV shape. A dimension is "block-padded" if it's the smallest multiple of a block size (16, 32, 64) that is >= the CSV dimension. Only allow padding tolerance on non-weight dimensions (first tensor / activation shapes).

**Files:**
- Modify: `tensor_cast/performance_model/perf_database/profiling_data_source.py`
- Test: `tests/perf_database/test_profiling_data_source.py`

**Step 1: Write failing test**

```python
def test_block_padding_tolerance(tmp_path):
    """TC seq=144 (padded from 136) should match CSV seq=136."""
    csv_content = (
        "Name,Type,Input Shapes,Input Data Types,Input Formats,Output Shapes,Duration(us)\n"
        'aclnnAdd,Add,"136,5120;136,5120",DT_BF16;DT_BF16,ND;ND,"136,5120",2.1\n'
    )
    (tmp_path / "Add.csv").write_text(csv_content)
    op_mapping = {
        "version": "test",
        "operator_mappings": {
            "aten.add.Tensor": {"kernel_type": "Add"},
        },
    }
    import yaml
    (tmp_path / "op_mapping.yaml").write_text(yaml.dump(op_mapping))

    ds = ProfilingDataSource(tmp_path)

    # TC dispatches with padded seq=144
    op = _make_op_invoke_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(144, 5120, dtype=torch.bfloat16),
            torch.empty(144, 5120, dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match with block-padding tolerance"
    assert result.latency_us == 2.1


def test_block_padding_no_false_positive(tmp_path):
    """Shapes that aren't block-padding related should NOT match."""
    csv_content = (
        "Name,Type,Input Shapes,Input Data Types,Input Formats,Output Shapes,Duration(us)\n"
        'aclnnAdd,Add,"136,5120;136,5120",DT_BF16;DT_BF16,ND;ND,"136,5120",2.1\n'
    )
    (tmp_path / "Add.csv").write_text(csv_content)
    op_mapping = {
        "version": "test",
        "operator_mappings": {
            "aten.add.Tensor": {"kernel_type": "Add"},
        },
    }
    import yaml
    (tmp_path / "op_mapping.yaml").write_text(yaml.dump(op_mapping))

    ds = ProfilingDataSource(tmp_path)

    # Completely different shape — should NOT match
    op = _make_op_invoke_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(256, 5120, dtype=torch.bfloat16),
            torch.empty(256, 5120, dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "256 is not a block-padding of 136"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/perf_database/test_profiling_data_source.py::test_block_padding_tolerance -v`
Expected: FAIL

**Step 3: Implement block-padding tolerance**

Add helper function and integrate into `_inputs_match()`:

```python
# Common block sizes used by NPU kernels
_BLOCK_SIZES = (16, 32, 64)


def _is_block_padded(tc_dim: int, csv_dim: int) -> bool:
    """Check if tc_dim is a block-padded version of csv_dim.

    Returns True if tc_dim == ceil(csv_dim / block_size) * block_size
    for any common block size.
    """
    if tc_dim <= csv_dim:
        return False
    for bs in _BLOCK_SIZES:
        padded = ((csv_dim + bs - 1) // bs) * bs
        if tc_dim == padded:
            return True
    return False


def _shapes_match_with_padding(
    tc_shape: Tuple[int, ...], csv_shape: Tuple[int, ...]
) -> bool:
    """Check if shapes match allowing block-padding on any dimension."""
    if len(tc_shape) != len(csv_shape):
        return False
    for tc_dim, csv_dim in zip(tc_shape, csv_shape):
        if tc_dim == csv_dim:
            continue
        if _is_block_padded(tc_dim, csv_dim):
            continue
        return False
    return True
```

Then in `_inputs_match()`, after `if tc_shape == csv_shape: continue`, add:

```python
        # Try block-padding tolerance
        if _shapes_match_with_padding(tc_shape, csv_shape):
            continue
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/perf_database/test_profiling_data_source.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add tensor_cast/performance_model/perf_database/profiling_data_source.py tests/perf_database/test_profiling_data_source.py
git commit -m "feat(perf-db): add block-padding tolerance in shape matching"
```

---

### Task 3: Input Count Flexibility via `input_indices` in op_mapping

**Problem:** Some TC ops have more tensor args than profiling. For example, `static_quant_linear` passes (x, w, w_scale, w_offset, x_scale, x_offset, bias, out_dtype) = 7+ tensors, but profiling `QuantBatchMatmulV3` CSV has a different number of inputs. Similarly, `aten.embedding` passes (weight, indices, ...) but CSV `GatherV2` has (weight, indices, axis_scalar).

**Approach:** Add optional `input_indices` field to op_mapping.yaml entries. When present, only extract the specified tensor arg positions for shape matching. Default: use all tensor args (current behavior).

**Files:**
- Modify: `tensor_cast/performance_model/perf_database/profiling_data_source.py`
- Modify: `tensor_cast/performance_model/perf_database/data/.../v0.14.0/op_mapping.yaml`
- Test: `tests/perf_database/test_profiling_data_source.py`

**Step 1: Write failing test**

```python
def test_input_indices_filtering(tmp_path):
    """input_indices in op_mapping should select which tensor args to match."""
    # CSV has 2 inputs: activation (M,K) and weight FRACTAL_NZ
    csv_content = (
        "Name,Type,Input Shapes,Input Data Types,Input Formats,Output Shapes,Duration(us)\n"
        'aclnnWQBMV3,QuantBatchMatmulV3,"42,7168;448,224,16,32",INT8;INT8,ND;FRACTAL_NZ,"42,3584",15.2\n'
    )
    (tmp_path / "QuantBatchMatmulV3.csv").write_text(csv_content)
    op_mapping = {
        "version": "test",
        "operator_mappings": {
            "tensor_cast.static_quant_linear.default": {
                "kernel_type": "QuantBatchMatmulV3",
                "input_indices": [0, 1],  # only match activation and weight
            },
        },
    }
    import yaml
    (tmp_path / "op_mapping.yaml").write_text(yaml.dump(op_mapping))

    ds = ProfilingDataSource(tmp_path)

    # TC dispatches with 7 tensor args: x, w, w_scale, w_offset, x_scale, x_offset, bias
    op = _make_op_invoke_info(
        torch.ops.tensor_cast.static_quant_linear.default,
        [
            torch.empty(42, 7168, dtype=torch.int8),     # x (activation)
            torch.empty(7168, 3584, dtype=torch.int8),    # w (weight, K×N)
            torch.empty(1, dtype=torch.float32),           # w_scale
            None,                                           # w_offset
            torch.empty(1, dtype=torch.float32),           # x_scale
            None,                                           # x_offset
            None,                                           # bias
            None,                                           # out_dtype
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match using only input_indices [0,1]"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/perf_database/test_profiling_data_source.py::test_input_indices_filtering -v`
Expected: FAIL — input count mismatch (7 tensors vs 2)

**Step 3: Implement input_indices support**

In `_lookup_compute()`, pass `input_indices` from mapping to `_extract_tensor_inputs()`:

```python
def _lookup_compute(
    self, op_invoke_info: "OpInvokeInfo", mapping: dict
) -> Optional[QueryResult]:
    kernel_type = mapping["kernel_type"]
    df = self._load_csv(kernel_type)
    if df is None:
        return None

    input_indices = mapping.get("input_indices")
    tc_inputs = self._extract_tensor_inputs(op_invoke_info, input_indices)
    # ... rest unchanged
```

Modify `_extract_tensor_inputs()`:

```python
def _extract_tensor_inputs(
    self,
    op_invoke_info: "OpInvokeInfo",
    input_indices: Optional[List[int]] = None,
) -> List[Tuple[Tuple[int, ...], torch.dtype]]:
    """Extract (shape, dtype) for each tensor arg.

    If input_indices is provided, only extract args at those positions
    (counting only tensor args, skipping None/scalar).
    """
    all_inputs = []
    for arg in op_invoke_info.args:
        if isinstance(arg, torch.Tensor):
            all_inputs.append((tuple(arg.shape), arg.dtype))
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, torch.Tensor):
                    all_inputs.append((tuple(item.shape), item.dtype))

    if input_indices is not None:
        return [all_inputs[i] for i in input_indices if i < len(all_inputs)]
    return all_inputs
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/perf_database/test_profiling_data_source.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add tensor_cast/performance_model/perf_database/profiling_data_source.py tests/perf_database/test_profiling_data_source.py
git commit -m "feat(perf-db): add input_indices support for selective tensor matching"
```

---

### Task 4: E2E Validation — Qwen3-32B BF16 Prefill

**Problem:** Validate that Tasks 1-3 produce actual HITs in a real TC run.

**Step 1: Run TC with corrected BF16 command**

The profiling was collected with this vLLM config:
- Model: Qwen3-32B, TP=16, DP=1, BF16 (no quantization)
- Bench: `--max-concurrency 1`, `--num_prompts 1`
- Profiling shows: seq=136, eagle3 speculative decoding

TC command (matching profiling config):

```bash
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 1 --query-length 136 \
  --device ATLAS_800_A3_752T_128G_DIE \
  --world-size 16 --tp-size 16 --dp-size 1 \
  --quantize-linear-action DISABLED \
  --performance-model profiling \
  --perf-database tensor_cast/performance_model/perf_database/data/atlas_800_a3_752t_128g_die/vllm_ascend/v0.14.0/ \
  2>&1 | tee /tmp/qwen3_bf16_e2e.log
```

**Step 2: Check hit rate**

```bash
grep -E "EmpiricalPerformanceModel|HIT|MISS" /tmp/qwen3_bf16_e2e.log | tail -20
```

Expected: hit rate > 0%. Specifically:
- `aten.mm` → `MatMulV2` should HIT for linear layers (with FRACTAL_NZ restoration + weight transpose)
- `aten.add.Tensor` → `Add` should HIT (with block-padding tolerance)
- Element-wise ops (mul, pow, rsqrt) will still MISS (fused in profiling) — acceptable

**Step 3: Analyze remaining MISSes**

Categorize any remaining MISSes:
- Acceptable: fused ops (AddRmsNorm covers mul+pow+rsqrt+mean), SwiGlu, split_qkv_rmsnorm_rope
- Fixable: shape mismatches requiring further investigation
- Deferred: communication, attention (Phase 2)

**Step 4: Update spike alignment report**

Update `docs/perf_database/reports/spike_alignment_report.md` with:
- Run 4 results (BF16 DISABLED)
- Actual hit rate and HIT/MISS breakdown
- Comparison with Run 2 (0% baseline)

**Step 5: Commit**

```bash
git add docs/perf_database/reports/spike_alignment_report.md
git commit -m "feat(perf-db): validate BF16 shape matching with DISABLED quant"
```

---

## Success Criteria

| Metric | Target |
|--------|--------|
| All existing tests pass | 19/19 |
| New unit tests pass | 5+ new tests |
| Qwen3 BF16 hit rate | > 30% of compute ops (MatMulV2 linears) |
| No TC core changes | Only ProfilingDataSource + op_mapping changes |

## Out of Scope (future plan)

- DSv3 W8A8 decode matching (needs `input_indices` values tuned + MLA decomposition)
- Interpolation across shapes (InterpolatingDataSource)
- Communication ops matching
- Attention ops matching
