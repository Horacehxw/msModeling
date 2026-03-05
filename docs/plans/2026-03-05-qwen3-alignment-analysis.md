# Qwen3-32B Profiling Alignment Analysis Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run TensorCast with correct parameters matching the Qwen3-32B vLLM deployment, then produce a detailed alignment report analyzing op mapping coverage and shape matching against real profiling data.

**Architecture:** Run TC in `--performance-model profiling` mode with BF16 (no quantization) matching the vLLM server config, capture per-op HIT/MISS details, then systematically analyze (1) which TC ops have op_mapping entries vs not, (2) which mapped ops match shapes vs not and why.

**Tech Stack:** TensorCast CLI, ProfilingDataSource, EmpiricalPerformanceModel, existing CSV data in `v0.14.0/`

---

## Background

### vLLM Server Config (Qwen3-32B)
- `--dtype bfloat16` → TC: `--quantize-linear-action DISABLED`
- `--tensor-parallel-size 16` → TC: `--world-size 16 --tp-size 16`
- `--block-size 128` → TC: `--block-size 128`
- `--speculative-config eagle3 (num_speculative_tokens=3)` → TC: `--num-mtp-tokens 3` (approximate)
- `--max-num-batched-tokens 65536` → TC: `--num-queries` + `--query-length` product
- `--max-concurrency 1` → single request at a time

### Profiling Data Source
- Qwen3-30B Prefill profiling (same architecture dims as Qwen3-32B with TP=16)
- CSV files in `v0.14.0/`: 45 kernel types, seq=136 shapes (from actual bench)
- Key shapes: hidden=5120, QKV/rank=768, Q/rank=512, gate_up/rank=3200, down/rank=1600, vocab/rank=9496

### Key TC Behavior
- TC defaults `quantize_linear_action=W8A8_DYNAMIC` (user_config.py:33), must use `DISABLED` for BF16
- TC pads seq to `ceil(seq/16)*16` (e.g., 136→144), profiling stores unpadded shapes
- TC dispatches Q, K, V as separate projections; profiling fuses QKV via `split_qkv_rmsnorm_rope_kernel`
- TC dispatches gate, up as separate projections; profiling fuses gate_up into single MatMulV2

---

### Task 1: Remove Old Report

**Files:**
- Delete: `docs/perf_database/reports/spike_alignment_report.md`

**Step 1: Delete the old report**

```bash
rm docs/perf_database/reports/spike_alignment_report.md
```

**Step 2: Verify deletion**

```bash
ls docs/perf_database/reports/
```
Expected: empty directory or only new files

---

### Task 2: Run TC with Correct BF16 Command

**Files:**
- Read: `tensor_cast/core/model_runner.py` (understand profiling output)
- Read: `tensor_cast/performance_model/empirical.py` (understand HIT/MISS logging)

**Step 1: Run TC matching vLLM config**

Run TensorCast with parameters matching the vLLM server deployment:

```bash
cd /home/horacehxw/Projects/msmodeling && \
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --device ATLAS_800_A3_752T_128G_DIE \
  --world-size 16 --tp-size 16 \
  --num-queries 1 --query-length 136 \
  --block-size 128 \
  --quantize-linear-action DISABLED \
  --performance-model profiling \
  --perf-database tensor_cast/performance_model/perf_database/data/atlas_800_a3_752t_128g_die/vllm_ascend/v0.14.0 \
  --log-level DEBUG 2>&1 | tee /tmp/qwen3_32b_bf16_alignment.log
```

Notes:
- `--quantize-linear-action DISABLED` ensures BF16 mm ops (not W8A8 quant ops)
- `--query-length 136` matches profiling seq length
- `--num-queries 1` matches `--max-concurrency 1`
- `--log-level DEBUG` enables per-op HIT/MISS details from EmpiricalPerformanceModel

**Step 2: Verify the run completes**

Check the log for:
- `EmpiricalPerformanceModel: X/Y ops matched (Z%)`
- `HITs: ...` and `MISSes: ...` detail lines

If the run fails (e.g., model download issue), try with `--num-hidden-layers-override 2` to test with fewer layers, or check if torch is available.

---

### Task 3: Extract and Analyze Op Mapping Coverage

**Files:**
- Create: `docs/perf_database/reports/qwen3_32b_alignment_report.md`

**Step 1: Parse TC dispatch trace**

From the log output, extract:
1. All unique TC func names dispatched (from MISS/HIT lines)
2. For each func, whether it has an op_mapping.yaml entry
3. Group into: mapped-compute, mapped-composite, mapped-communication, unmapped

```bash
# Extract all unique ops from the log
grep -E "(HIT|MISS)" /tmp/qwen3_32b_bf16_alignment.log | \
  sed 's/.*HITs: //; s/.*MISSes: //' | tr '|' '\n' | \
  sed 's/->.*//; s/ \[.*//; s/^ *//; s/ *$//' | sort -u
```

**Step 2: Cross-reference with op_mapping.yaml**

For each dispatched op, check:
- Does it have an `operator_mappings` entry?
- If yes: what `kernel_type`? Is it `composite`? Is it `communication`?
- If no: what is this op? (likely aten element-wise, reshape, etc.)

**Step 3: Summarize op mapping coverage**

Create a table:

| TC Op | Mapped? | kernel_type | Category | Notes |
|-------|---------|-------------|----------|-------|
| aten.mm.default | Yes | MatMulV2 | compute | Main linear projection |
| tensor_cast.add_rms_norm.default | Yes | AddRmsNorm | compute | Fused norm |
| aten.reshape.default | No | - | - | Shape-only, no kernel |
| ... | | | | |

---

### Task 4: Analyze Shape Matching Results

**Files:**
- Modify: `docs/perf_database/reports/qwen3_32b_alignment_report.md`

**Step 1: Parse HIT details**

From the log, extract all HITs with their kernel types:
```
func_name->kernel_type
```

Group by kernel_type and count.

**Step 2: Parse MISS details**

From the log, extract all MISSes with their shapes:
```
func_name [(shape1), (shape2), ...]
```

For each MISS that HAS an op_mapping entry (mapped but shape-mismatched):
- What shape does TC send?
- What shapes exist in the CSV?
- Why don't they match? (batch dim, padding, fused vs split, transpose, etc.)

**Step 3: Classify shape mismatches**

Group mismatches into categories:
1. **Batch dim**: TC sends (1, seq, dim), CSV has (seq, dim)
2. **Padding**: TC sends (144, dim), CSV has (136, dim) — should match with tolerance
3. **Layer decomposition**: TC sends separate Q/K/V shapes, CSV has fused QKV shape
4. **Vocab sharding**: TC sends (seq, 9496), CSV has different vocab shape
5. **Format**: TC sends ND, CSV has FRACTAL_NZ (should be handled by restoration)
6. **Missing CSV entry**: Shape simply not in profiling data

---

### Task 5: Write Final Alignment Report

**Files:**
- Write: `docs/perf_database/reports/qwen3_32b_alignment_report.md`

**Step 1: Write the report**

Structure:

```markdown
# Qwen3-32B Profiling Alignment Report

**Date:** 2026-03-05
**Model:** Qwen/Qwen3-32B (BF16 Prefill)
**Device:** ATLAS_800_A3_752T_128G_DIE (TP=16)
**Profiling Source:** Qwen3-30B Prefill (same arch dims, TP=16)

## TC Command
[exact command used]

## Summary
- Total TC ops dispatched: X
- Op mapping coverage: Y/X (Z%)
- Shape match (HITs): A/B (C%) of mapped ops
- Overall hit rate: A/X (D%)

## 1. Op Mapping Coverage

### Mapped Ops (Hit or Miss)
[table of ops with mapping entries]

### Unmapped Ops
[table of ops without mapping entries, with explanation]

### Composite/Communication Ops (Skipped)
[table of composite/comm ops that return None by design]

## 2. Shape Matching Analysis

### HITs (Successful Matches)
[table of matched ops with shapes and latencies]

### MISSes by Category
#### Category 1: Batch Dimension Mismatch
[details]

#### Category 2: Layer Decomposition Gap
[details: TC splits Q/K/V/gate/up, profiling fuses them]

#### Category 3: Other
[details]

## 3. Gap Analysis & Recommendations
[prioritized list of what to fix to improve hit rate]
```

**Step 2: Verify report is complete**

Check that report covers:
- [ ] Exact TC command used
- [ ] Op mapping coverage table
- [ ] Shape matching results table
- [ ] Mismatch categorization
- [ ] Recommendations

---

### Task 6: Verify All Tests Still Pass

**Step 1: Run existing tests**

```bash
cd /home/horacehxw/Projects/msmodeling && \
pytest tests/perf_database/ -v
```

Expected: All 24 tests pass (no regressions from prior work)
