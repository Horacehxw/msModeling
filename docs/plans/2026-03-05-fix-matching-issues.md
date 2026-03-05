# Fix Profiling Matching Issues Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix shape matching issues in ProfilingDataSource to improve hit rate from 3/46 (6.5%) to near-full coverage for mapped ops.

**Architecture:** Add batch-dim stripping, SwiGlu input normalization, and block-padding tolerance improvements to `_inputs_match()` in ProfilingDataSource.

**Tech Stack:** Python, pytest, ProfilingDataSource

---

## Identified Issues (from compile run)

| Op | TC Shape | CSV Shape | Fix Needed |
|----|----------|-----------|-----------|
| RmsNorm | (1,144,5120),(5120,) | (136,5120),(5120) | Strip batch dim + padding |
| AddRmsNorm | (1,144,5120),(144,5120),(5120,) | (136,5120),(136,5120),(5120) | Strip batch dim + padding |
| SwiGlu | (1,144,1600),(1,144,1600) | (136,3200) | Strip batch dim + concat inputs |
| Add | (1,144,5120),(144,5120) | (136,5120),(136,5120) | Strip batch dim + padding |
| GatherV2 | (151936,5120),(1,144) | (9496,5120),(136),(1) | Vocab sharding + input count |

---

### Task 1: Add batch-dim stripping to shape matching

**Files:**
- Modify: `tensor_cast/performance_model/perf_database/profiling_data_source.py`
- Test: `tests/perf_database/test_profiling_data_source.py`

**Step 1: Write failing tests**

```python
def test_batch_dim_stripping_rmsnorm(rmsnorm_data_dir):
    """TC RmsNorm sends (1,144,5120),(5120,) — should match CSV (136,5120),(5120) after stripping batch dim."""
    ds = ProfilingDataSource(rmsnorm_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.rms_norm.default,
        [
            torch.empty(1, 144, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match after stripping batch dim=1 + padding tolerance"

def test_batch_dim_stripping_add(add_data_dir):
    """TC Add sends (1,144,5120),(1,144,5120) — should match CSV (136,5120),(136,5120)."""
    ds = ProfilingDataSource(add_data_dir)
    op = _make_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(1, 144, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 144, 5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match after stripping batch dim=1 + padding"

def test_batch_dim_stripping_add_rms_norm(add_rmsnorm_data_dir):
    """TC AddRmsNorm sends (1,144,5120),(144,5120),(5120,) — should match CSV (136,5120),(136,5120),(5120)."""
    ds = ProfilingDataSource(add_rmsnorm_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.add_rms_norm2.default,
        [
            torch.empty(1, 144, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(144, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match after stripping batch dim + padding"
```

**Step 2: Implement batch-dim stripping**

In `_inputs_match()`, before comparing shapes, strip leading dim=1 from TC shapes:

```python
def _strip_batch_dim(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Strip leading batch dim=1 from TC shapes.
    TC keeps explicit batch: (1, seq, dim). Profiling flattens: (seq, dim).
    """
    if len(shape) > 1 and shape[0] == 1:
        return shape[1:]
    return shape
```

Apply in the per-tensor matching loop before comparing tc_shape with csv_shape.

**Step 3: Run tests, verify pass**

**Step 4: Commit**

---

### Task 2: Add SwiGlu input normalization

**Files:**
- Modify: `tensor_cast/performance_model/perf_database/profiling_data_source.py`
- Test: `tests/perf_database/test_profiling_data_source.py`

**Step 1: Write failing test**

```python
def test_swiglu_input_concat(swiglu_data_dir):
    """TC SwiGlu sends 2 inputs (gate, up), CSV has 1 fused input.
    TC: (1,144,1600),(1,144,1600) → strip batch → (144,1600),(144,1600) → concat → (144,3200) → padding → (136,3200)
    """
    ds = ProfilingDataSource(swiglu_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.swiglu.default,
        [
            torch.empty(1, 144, 1600, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 144, 1600, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match SwiGlu after concatenating 2 inputs into 1"
```

**Step 2: Implement SwiGlu-specific matching**

In `_lookup_compute()` or `_inputs_match()`, add SwiGlu-specific logic:
- If kernel_type is "SwiGlu" and TC has 2 inputs but CSV has 1
- Concatenate TC input dims along last axis: (seq, D/2) + (seq, D/2) → (seq, D)

**Step 3: Run tests, verify pass**

**Step 4: Commit**

---

### Task 3: Re-run verification and generate updated report

**Step 1: Run TC with --compile and capture results**

```bash
PYTHONPATH=... python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --device ATLAS_800_A3_752T_128G_DIE \
  --world-size 16 --tp-size 16 \
  --num-queries 1 --query-length 136 \
  --quantize-linear-action DISABLED --compile \
  --performance-model profiling \
  --perf-database .../v0.14.0 --log-level debug
```

**Step 2: Analyze results and update report**

Update `docs/perf_database/reports/qwen3_32b_alignment_report.md` with:
- New hit rate
- Before/after comparison
- Remaining gaps
- Solutions applied

---

### Task 4: Run all tests

```bash
python3.10 -m pytest tests/perf_database/ -v
```
