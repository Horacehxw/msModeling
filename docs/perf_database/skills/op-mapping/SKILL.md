---
name: op-mapping-generator
description: Use when creating or updating op_mapping.yaml to map TensorCast simulation ops to NPU profiling kernel types, given a model, device, profiling data, and software stack versions
---

# op_mapping Generator

## Overview

Generate a complete `op_mapping.yaml` that maps TensorCast (TC) virtual runtime operators to real NPU profiling kernel types. The mapping bridges TC simulation and profiling-based performance estimation by matching each TC op to its corresponding NPU kernel (identified by the Profiling Type column in `kernel_details.csv`).

**Core approach:** Dispatch parallel sub-agent teams 鈥?one agent per operator 鈥?each independently traces the full vLLM鈫扖ANN call chain using grep/read/web search. This avoids context contamination between operators.

## Required Inputs

Collect ALL of these from the user before proceeding:

- [ ] **Target model** 鈥?HuggingFace ID (e.g., `Qwen/Qwen3-32B`)
- [ ] **Device profile** 鈥?(e.g., `ATLAS_800_A3_752T_128G_DIE`)
- [ ] **Parallelism config** 鈥?world-size, tp-size, dp-size, ep flag
- [ ] **Quantization** 鈥?none, W8A8_STATIC, W4A8_STATIC, FP8, MXFP4
- [ ] **Profiling CSV path** 鈥?`kernel_details.csv` from an NPU profiling run
- [ ] **Software stack versions + repo sources** (ask user for each):

| Repo | Default URL | Version/Tag |
|------|-------------|-------------|
| vLLM | `github.com/vllm-project/vllm` | (ask user) |
| vLLM-ascend | `github.com/vllm-project/vllm-ascend` | (ask user) |
| op-plugin | `github.com/Ascend/op-plugin` | (ask user) |
| pytorch-npu | `github.com/Ascend/pytorch` | (ask user) |
| CANN ops-nn | `gitee.com/ascend/cann-ops-nn` | (ask user) |
| CANN ops-transformer | `gitee.com/ascend/cann-ops-transformer` | (ask user) |
| CANN ops-math | `gitee.com/ascend/cann-ops-math` | (ask user) |
| CANN ATB | `gitee.com/ascend/ascend-transformer-boost` | (ask user) |

- [ ] **Local repo paths** (optional) 鈥?if repos are already cloned locally
- [ ] **msmodeling project root** 鈥?path to this repo
- [ ] **Python path** 鈥?Python 3.10 executable with torch installed
- [ ] **Existing op_mapping.yaml** (optional) 鈥?for update mode; skip already-mapped ops

## Key Principles

Teach these to all sub-agents:

1. **Profiling Name 3-part structure**: `aclnnAPI_DispatchFunc_L0OpType` 鈥?the 3rd segment = Profiling Type = our lookup key
2. **Type column = OPTYPE** from CANN `op_host/CMakeLists.txt` = CSV filename for database query
3. **Three paths**: A (aten鈫抩p-plugin鈫抋clnn), B (torch_npu.npu_*鈫抩p-plugin鈫抋clnn), C (vllm-ascend custom/triton)
4. **10 shape differences** between TC tensors and NPU profiling shapes (see worker prompt + `ref/shape_matching_catalog.md`)
5. **Mutually exclusive**: `kernel_type` vs `composite` vs `zero_cost` 鈥?exactly one per entry
6. **Communication ops** use message_bytes + num_devices, NOT shape matching
7. **tc_input_count safety**: Only safe for truncating NPU-internal params (axis, scale), NOT for elementwise broadcast ops. See `ref/tc_input_count_rules.md`
8. **zero_cost classification**: Must verify kernel Type never appears in profiling AND latency is captured by a fused kernel. See `ref/zero_cost_classification.md`
9. **Elementwise query_mode**: For memory-bound elementwise ops (add, mul, div), use `query_mode: elementwise` to match on output shape with dtype-relaxed byte-ratio scaling. See `ref/shape_matching_catalog.md` Type 11.

---

## Phase 1: GATHER

### 1a: Prepare Repos

Verify local repo checkouts are at correct versions, or clone fresh:
```bash
# For each repo in the version table above:
cd $LOCAL_PATH && git checkout $VERSION
# Or: git clone $URL --branch $VERSION --depth 1 /tmp/$REPO_NAME
```

### 1b: Run TC Simulation

```bash
$PYTHON -m tensor_cast.scripts.text_generate $MODEL \
  --num-queries 2 --query-length 3500 \
  --device $DEVICE --world-size $WS --tp-size $TP \
  [--dp-size $DP] [--ep] [--quantize-linear-action $QUANT] \
  --performance-model profiling --compile \
  --chrome-trace /tmp/tc_gather_trace.json 2>&1 | tee /tmp/tc_gather.log
```

**IMPORTANT**: `--compile` is REQUIRED 鈥?without it, fused ops decompose to aten primitives that can't match profiling kernels.

### 1c: Extract TC Ops

Use the current trace-to-op extraction workflow in this repo to produce
`/tmp/tc_ops.json` from `/tmp/tc_gather_trace.json`. Do not reference the
removed legacy helper by filename.

### 1d: Parse Profiling Data

Extract unique profiling Types with counts:
```bash
awk -F',' 'NR>1 {count[$2]++} END {for (t in count) print count[t], t}' \
  "$PROFILING_CSV" | sort -rn > /tmp/profiling_types.txt
```

### 1e: Compute Work List

- Read `/tmp/tc_ops.json` 鈫?list of TC op names (with `unmapped_ops` if existing mapping provided)
- Read `/tmp/profiling_types.txt` 鈫?list of profiling Types
- **Forward work**: TC ops not yet mapped (all if from scratch, unmapped_ops if updating)
- **Reverse work**: profiling Types not covered by any forward mapping

---

## Phase 2: FORWARD MAPPING (Parallel Sub-Agents)

For each unmapped TC op, dispatch a worker agent:

```
DISPATCH: Agent tool
  subagent_type: general-purpose
  prompt: |
    Read the file <skill_dir>/single-op-worker-prompt.md for instructions.

    Your task:
      op_name: "<tc_op_name>"
      direction: forward
      profiling_csv_path: "<path>"
      local_repo_paths:
        op_plugin: "<path>"
        vllm_ascend: "<path>"
        vllm: "<path>"
        pytorch_npu: "<path>"
        cann_ops_nn: "<path>"
        cann_ops_transformer: "<path>"
        cann_ops_math: "<path>"
        cann_atb: "<path>"
      repo_urls:
        op_plugin: "<url>@<version>"
        vllm_ascend: "<url>@<version>"
        ...

    Return the YAML snippet and RESULT summary line.
```

**Batching**: Dispatch 5-10 agents in parallel. Wait for completion. Collect YAML snippets.

**Error handling**: If a worker fails or returns LOW confidence, queue for manual review in Phase 6.

---

## Phase 3: REVERSE MAPPING (Parallel Sub-Agents)

For each profiling Type NOT covered by Phase 2 mappings:

```
DISPATCH: Agent tool (same pattern as Phase 2)
  prompt: |
    Read <skill_dir>/single-op-worker-prompt.md.
    op_name: "<ProfilingType>"
    direction: reverse
    ...
```

These typically become `profiling.<Type>` placeholder entries 鈥?NPU fusion kernels not modeled in TC.

---

## Phase 4: ASSEMBLE

### 4a: Merge Snippets

Combine all YAML snippets from Phases 2-3 into one file. Deduplicate by op_name.

### 4b: Add Metadata Header

```yaml
version: "<vllm_ascend_version>"
device: <DEVICE>
cann_version: "<cann_version>"
collection_date: "<today>"

communication_data_ref: "../../hccl/<cann_version>/"
communication_fallback: analytic

interpolation_policy:
  default_method: linear
  kernel_overrides:
    FusedInferAttentionScore:
      shape_transform: sqrt    # O(seq^2) 鈫?interpolate in sqrt(seq) space
```

### 4c: Organize by Category

Group operator_mappings entries by section (match existing op_mapping.yaml convention):
1. Standard aten ops
2. Quantized linear variants
3. GroupedMatmul (MoE)
4. GroupedMatmul+SwiGlu fusion
5. MoE routing
6. Attention
7. MLA
8. KV Cache
9. Norm basics
10. Norm+Quant fusions
11. Quantization ops
12. RoPE
13. Communication
14. MC2 fusion
15. Utility ops
16. Profiling-only placeholders

### 4d: Add torch_npu_reference Section

Merge all torch_npu_reference entries from workers. Deduplicate by kernel_type.

### 4e: Write Output

Save to: `$MSMODELING/tensor_cast/performance_model/perf_database/data/$DEVICE/vllm_ascend/$VERSION/op_mapping.yaml`

---

## Phase 5: VERIFY

Verification requires deriving the correct TensorCast simulation parameters from the profiling data itself. Do NOT guess parameters 鈥?extract them from the CSV shapes.

### 5a: Analyze Profiling Data to Derive TC Parameters

For each profiling dataset, extract these from the per-kernel CSVs:

**Step 1: Determine workload type (prefill vs decode)**
```bash
# Check batch dimensions in compute kernel CSVs
awk -F',' 'NR>1 {print $3}' $DATA_DIR/MatMulV2.csv | head -20
awk -F',' 'NR>1 {print $3}' $DATA_DIR/AddRmsNorm.csv | head -10
```
- Small batch dims (1-50) = **decode** workload 鈫?use `--query-length 1 --context-length X`
- Large batch dims (100+) = **prefill** workload 鈫?use `--query-length X`
- Mixed = PandD trace 鈫?verify both separately

**Step 2: Determine quantization mode**
```bash
ls $DATA_DIR/*.csv | grep -i -E "quant|int8"
```
- `QuantBatchMatmulV3.csv` exists with INT8 dtypes 鈫?`--quantize-linear-action W8A8_STATIC`
- `DynamicQuant.csv` exists 鈫?may indicate W8A8_DYNAMIC
- Only BF16 MatMulV2 shapes 鈫?`--quantize-linear-action DISABLED`

**Step 3: Determine parallelism from hidden dimensions**
```bash
# Extract hidden/intermediate dims from MatMulV2 or QuantBatchMatmulV3
head -5 $DATA_DIR/MatMulV2.csv | cut -d',' -f3,4
# Look at SwiGlu dims
head -5 $DATA_DIR/SwiGlu.csv | cut -d',' -f3,4
# Check FIA head counts
head -5 $DATA_DIR/FusedInferAttentionScore.csv | cut -d',' -f3
```
- Compare CSV dims to model's full hidden_size to compute TP:
  - `intermediate_per_card = model.intermediate_size / TP`
  - `q_heads_per_card = model.num_attention_heads / TP`
  - Match against SwiGlu and FIA shapes to find correct TP
- Check MoE ops (GroupedMatmul*, MoeGatingTopK) 鈫?EP config
- Derive DP from: `world_size = TP 脳 DP 脳 EP`

**Step 4: Determine batch size**
```bash
# Find most common batch dims across compute kernels
for f in MatMulV2.csv AddRmsNorm.csv SwiGlu.csv QuantBatchMatmulV3.csv; do
  echo "=== $f ===" && awk -F',' 'NR>1 {print $3}' $DATA_DIR/$f | sort | uniq -c | sort -rn | head -5
done
```
- The most frequent batch dim = target `--num-queries`
- For decode: `--num-queries=<batch_dim> --query-length=1`
- For prefill: `--num-queries=1 --query-length=<batch_dim>` (or split: nq=2, ql=batch/2)

### 5b: Run TensorCast Simulation

```bash
$PYTHON -m tensor_cast.scripts.text_generate $MODEL \
  --num-queries $NQ --query-length $QL [--context-length $CL] \
  --device $DEVICE --world-size $WS --tp-size $TP [--dp-size $DP] [--ep-size $EP] \
  --quantize-linear-action $QUANT \
  --performance-model profiling --compile \
  --perf-database $DATA_DIR 2>&1 | tee /tmp/verify_run.log
```

### 5c: Analyze Gaps

From the output, classify every MISS into one of these categories:

| Gap Category | Symptom | Action |
|---|---|---|
| **Op mapping error** | Op has wrong kernel_type or no mapping | Fix op_mapping.yaml entry |
| **Shape coverage gap** | Op mapped correctly but CSV lacks matching shape | Add profiling data for that shape (re-profile or microbenchmark) |
| **TC decomposition mismatch** | TC produces different intermediate shapes than real vLLM | Known limitation 鈥?TC compile pass doesn't match vLLM exactly |
| **Structural miss** | Embedding, KV cache, comm ops differ structurally | Expected 鈥?these ops have fundamentally different TC vs NPU interfaces |
| **Param mismatch** | Wrong batch/seq/TP caused shape miss | Re-derive params from Step 5a |

**Key distinction:** A shape MISS with correct kernel_type = data coverage gap (not an op_mapping bug). A shape MISS with wrong kernel_type = op_mapping error.

### 5d: Iterate

1. Fix any op_mapping errors found in 5c
2. If param mismatch: re-run with corrected params
3. Re-run simulation until no new op_mapping errors remain
4. Document remaining gaps with categories

### 5e: Report

Generate a verification report with:
- TC command used (copy-pasteable)
- Match rate: `X/Y ops matched (Z%)`
- Gap breakdown by category (table)
- For each MISS: op name, TC shape, expected kernel_type, gap category, action needed

---

## Phase 6: CORRECT (Loop)

For each correction in corrections_needed:

| Issue Type | Action |
|---|---|
| MISS 鈥?no mapping | Dispatch forward-mapping worker for the op |
| Shape mismatch | Investigate shape transform; may need new flag in profiling_data_source.py |
| Latency outlier (>2x) | Check if wrong kernel_type or missing alternate_kernel_types |
| Wrong confidence | Re-verify evidence chain with updated profiling data |
| TC decomposition mismatch | Document as known limitation; no op_mapping fix needed |

After all corrections applied, re-run Phase 5. Repeat until verification passes.

---

## Handling CANN Version Differences

Kernel types can change between CANN versions (renames, fusions, removals). The skill handles this naturally:

1. **Always verify against profiling data** 鈥?the `Type` column in `kernel_details.csv` is ground truth for the current CANN version
2. **Use `alternate_kernel_types`** 鈥?when a kernel has been renamed across versions, list both old and new names so the mapping works with either
3. **Check for fused kernels** 鈥?newer CANN versions may fuse previously separate ops into a single kernel (use `composite: true` or update `kernel_type`)
4. **Check for removed kernels** 鈥?Triton kernels may be replaced by native CANN fusions; profiling-only placeholders may no longer appear

**How to discover version changes:** Compare `profiling_types.txt` from two profiling runs on different CANN versions. Types that appear in one but not the other indicate renames, fusions, or removals. Trace each through the 5-layer pipeline to determine the correct mapping.

**aclgraph parity:** vllm-ascend's `aclgraph` compiler ensures eager mode and graph mode produce exactly the same ops, including fusion passes. Profiling from either mode is valid for op_mapping.

---

## Completion Criteria

- [ ] All TC compute ops have mappings (unmapped_ops = [])
- [ ] Hit rate > 95% for compute ops
- [ ] End-to-end latency within 卤30% of profiling (if data available)
- [ ] All notes fields have complete evidence chains
- [ ] torch_npu_reference section covers all unique kernel_types


