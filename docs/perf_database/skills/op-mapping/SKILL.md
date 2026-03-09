---
name: op-mapping-generator
description: Use when creating or updating op_mapping.yaml to map TensorCast simulation ops to NPU profiling kernel types, given a model, device, profiling data, and software stack versions
---

# op_mapping Generator

## Overview

Generate a complete `op_mapping.yaml` that maps TensorCast (TC) virtual runtime operators to real NPU profiling kernel types. The mapping bridges TC simulation and profiling-based performance estimation by matching each TC op to its corresponding NPU kernel (identified by the Profiling Type column in `kernel_details.csv`).

**Core approach:** Dispatch parallel sub-agent teams — one agent per operator — each independently traces the full vLLM→CANN call chain using grep/read/web search. This avoids context contamination between operators.

## Required Inputs

Collect ALL of these from the user before proceeding:

- [ ] **Target model** — HuggingFace ID (e.g., `Qwen/Qwen3-32B`)
- [ ] **Device profile** — (e.g., `ATLAS_800_A3_752T_128G_DIE`)
- [ ] **Parallelism config** — world-size, tp-size, dp-size, ep flag
- [ ] **Quantization** — none, W8A8_STATIC, W4A8_STATIC, FP8, MXFP4
- [ ] **Profiling CSV path** — `kernel_details.csv` from an NPU profiling run
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

- [ ] **Local repo paths** (optional) — if repos are already cloned locally
- [ ] **msmodeling project root** — path to this repo
- [ ] **Python path** — Python 3.10 executable with torch installed
- [ ] **Existing op_mapping.yaml** (optional) — for update mode; skip already-mapped ops

## Key Principles

Teach these to all sub-agents:

1. **Profiling Name 3-part structure**: `aclnnAPI_DispatchFunc_L0OpType` — the 3rd segment = Profiling Type = our lookup key
2. **Type column = OPTYPE** from CANN `op_host/CMakeLists.txt` = CSV filename for database query
3. **Three paths**: A (aten→op-plugin→aclnn), B (torch_npu.npu_*→op-plugin→aclnn), C (vllm-ascend custom/triton)
4. **8 shape differences** between TC tensors and NPU profiling shapes (see worker prompt)
5. **Mutually exclusive**: `kernel_type` vs `composite` vs `zero_cost` — exactly one per entry
6. **Communication ops** use message_bytes + num_devices, NOT shape matching

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

**IMPORTANT**: `--compile` is REQUIRED — without it, fused ops decompose to aten primitives that can't match profiling kernels.

### 1c: Extract TC Ops

```bash
$PYTHON tools/perf_data_collection/extract_tc_ops.py \
  --chrome-trace /tmp/tc_gather_trace.json \
  --output /tmp/tc_ops.json \
  [--op-mapping $EXISTING_OP_MAPPING]
```

### 1d: Parse Profiling Data

Extract unique profiling Types with counts:
```bash
awk -F',' 'NR>1 {count[$2]++} END {for (t in count) print count[t], t}' \
  "$PROFILING_CSV" | sort -rn > /tmp/profiling_types.txt
```

### 1e: Compute Work List

- Read `/tmp/tc_ops.json` → list of TC op names (with `unmapped_ops` if existing mapping provided)
- Read `/tmp/profiling_types.txt` → list of profiling Types
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

These typically become `profiling.<Type>` placeholder entries — NPU fusion kernels not modeled in TC.

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
      shape_transform: sqrt    # O(seq^2) → interpolate in sqrt(seq) space
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

```
DISPATCH: Agent tool
  subagent_type: general-purpose
  prompt: |
    Read <skill_dir>/verifier-prompt.md for instructions.

    op_mapping_path: "<assembled yaml path>"
    msmodeling_dir: "<path>"
    model: "<MODEL>"
    device: "<DEVICE>"
    tc_config:
      world_size: <WS>
      tp_size: <TP>
      dp_size: <DP>
      ep: <true|false>
      quantize_linear_action: "<QUANT>"
    profiling_csv_path: "<path>"
    python_path: "<PYTHON>"
```

Collect verification_report.md and corrections_needed list.

---

## Phase 6: CORRECT (Loop)

For each correction in corrections_needed:

| Issue Type | Action |
|---|---|
| MISS — no mapping | Dispatch forward-mapping worker for the op |
| Shape mismatch | Investigate shape transform; may need new flag in profiling_data_source.py |
| Latency outlier (>2x) | Check if wrong kernel_type or missing alternate_kernel_types |
| Wrong confidence | Re-verify evidence chain with updated profiling data |

After all corrections applied, re-run Phase 5. Repeat until verification passes.

---

## Completion Criteria

- [ ] All TC compute ops have mappings (unmapped_ops = [])
- [ ] Hit rate > 95% for compute ops
- [ ] End-to-end latency within ±30% of profiling (if data available)
- [ ] All notes fields have complete evidence chains
- [ ] torch_npu_reference section covers all unique kernel_types
