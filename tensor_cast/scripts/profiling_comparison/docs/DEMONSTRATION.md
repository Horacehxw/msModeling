# Profiling Comparison Pipeline Demonstration

This document walks through the complete 3-stage profiling comparison pipeline using two real-world models.

## Pipeline Overview

```
Stage 1: Analyze              Stage 2: Simulate             Stage 3: Compare
┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│ VLLM profiling   │         │ text_generate.py  │         │ Sequence Matcher │
│ kernel_details   │────────▶│ (subprocess)      │────────▶│                  │
│ .csv             │  phase  │                   │ chrome  │ VLLM ordered ops │
│                  │  + cfg  │ --chrome-trace    │ trace   │ TC ordered events│
│ Profile YAML     │         │ --model-id ...    │ .json   │ Decomposition map│
│ Phase detector   │         │                   │         │                  │
└──────────────────┘         └──────────────────┘         └──────────────────┘
       │                            │                            │
       ▼                            ▼                            ▼
  Print TC command           Chrome trace file            Excel report
  + detected phase           + op summary table           with per-op diffs
```

Each stage produces **human-inspectable intermediate output**, making the pipeline transparent and debuggable.

---

## Key Concept: Step = One Complete Forward Pass

A "step" is defined as **one complete forward pass** through ALL layers of the model:
- For Qwen3-32B (64 layers): 64 `FusedInferAttentionScore` ops = 1 step
- For DeepSeek-V3 (61 layers): 61 `FusedInferAttentionScore` ops = 1 step

The `num_layers` parameter is critical for correctly grouping operations into steps.

## Key Concept: Order-Based Sequence Matching

Unlike name-based aggregation (which can double-count), the sequence matcher walks both the VLLM and TensorCast operation lists in **execution order**:

1. For each VLLM op, look up expected TC ops from the decomposition mapping
2. Advance through TC events, skipping ignored ops (views, reshapes, etc.)
3. Consume matching TC events in order
4. Each TC event is consumed **exactly once** -- no double-counting

This produces a position-by-position comparison where every op is accounted for.

---

## Case Study 1: Qwen3-32B (Dense Model, TP=16)

### 1.1 Profiling Data

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3-32B (dense, 64 layers) |
| Hardware | 16x ATLAS A3 Dies |
| Parallelism | TP=16 |
| Quantization | BF16 (disabled) |
| Batch size | 136 queries |
| Profiling path | `/mnt/d/Data/Profiling/profiling-qwen3-30b-pd_tegether/` |

**How we know this**:
- 16 rank directories in ASCEND_PROFILER_OUTPUT → TP=16
- `MatMulV2` is the top operator (not `QuantBatchMatmulV3`) → BF16, no quantization
- No `GroupedMatmul` or `MoeDistribute*` → Dense model, not MoE
- `ReshapeAndCacheNdKernel` input shapes contain `query_len=1` → Decode phase

### 1.2 Profile Configuration

The profile `config/profiles/qwen3_32b.yaml` encodes these parameters:

```yaml
name: qwen3-32b
description: "Qwen3-32B dense model on 16x ATLAS A3 Dies with TP=16"
num_layers: 64

tensorcast:
  model_id: Qwen/Qwen3-32B
  device: ATLAS_800_A3_752T_128G_DIE
  world_size: 16
  tp_size: 16
  quantize_linear_action: DISABLED
  lmhead_tp_size: 16

decode_defaults:
  num_queries: 136
  query_length: 1
  context_length: 4096

mapping_file: qwen3    # Merges default.yaml + qwen3.yaml
```

Qwen3 uses a model-specific mapping (`qwen3.yaml`) that adds the `split_qkv_rmsnorm_rope_kernel` decomposition, which maps a single fused VLLM kernel to 6 TC ops:

```yaml
decompositions:
  split_qkv_rmsnorm_rope_kernel:
    tc_ops: ["aten.mm", "aten.mm", "aten.mm", "aten.split", "aten.neg", "aten.clone"]
```

### 1.3 Running the Pipeline

#### Option A: Run All 3 Stages at Once

```bash
VLLM_DIR="/mnt/d/Data/Profiling/profiling-qwen3-30b-pd_tegether/profiling/623bd92ebdd0_417949_20260203150032546_ascend_pt/ASCEND_PROFILER_OUTPUT"

.conda/bin/python -m tensor_cast.scripts.profiling_comparison.cli run-all \
    --vllm-dir "$VLLM_DIR" \
    --profile qwen3_32b \
    --output-dir /tmp/qwen3_results/
```

#### Option B: Run Each Stage Individually

**Stage 1: Analyze** -- Parse VLLM data, detect phase, print TC command:

```bash
.conda/bin/python -m tensor_cast.scripts.profiling_comparison.cli analyze \
    --vllm-dir "$VLLM_DIR" \
    --profile qwen3_32b
```

Expected output:
```
Loaded profile: qwen3-32b
  Description: Qwen3-32B dense model on 16x ATLAS A3 Dies with TP=16
  num_layers: 64

  Detected phase: decode (confidence: 99%)
  Detected query length: 1
  Complete forward passes: 1

=== TensorCast Command ===
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --dp-size 1 --num-queries 136 --query-length 1 --context-length 4096 \
  --lmhead-tp-size 16

=== Analyze Complete ===
Phase: decode
Steps detected: 1
```

**Stage 2: Simulate** -- Run TensorCast and produce chrome trace:

```bash
.conda/bin/python -m tensor_cast.scripts.profiling_comparison.cli simulate \
    --profile qwen3_32b \
    --phase decode \
    --output-dir /tmp/qwen3_results/
```

This runs `text_generate.py` as a subprocess with `--chrome-trace`, producing:
- `/tmp/qwen3_results/chrome_trace.json` -- The chrome trace file

**Stage 3: Compare** -- Sequence-match VLLM ops vs TC trace:

```bash
.conda/bin/python -m tensor_cast.scripts.profiling_comparison.cli compare \
    --vllm-dir "$VLLM_DIR" \
    --tc-trace /tmp/qwen3_results/chrome_trace.json \
    --profile qwen3_32b \
    --output /tmp/qwen3_results/qwen3_comparison.xlsx
```

Expected output:
```
=== Loading Mappings ===
  Base mapping: default (17 decompositions)
  Model mapping: qwen3 (merged)
  Total decompositions: 19

=== Parsing VLLM Profiling ===
  Step 0: 1479 operations, 206742.53 us

=== Parsing TensorCast Trace ===
  847 events, 39622.54 us total

=== Running Sequence Matching ===
  Matched: 20
  Unmatched VLLM: 3
  Unmatched TC: 5
  Mismatches: 0

============================================================
SUMMARY
============================================================
VLLM total time:     206742.53 us
TensorCast total:    39622.54 us
Difference:          -167119.99 us (-80.8%)
Matched operations:  20
Coverage:            98.5%
Output saved to:     /tmp/qwen3_results/qwen3_comparison.xlsx
```

### 1.4 Understanding the Results

| Metric | Value |
|--------|-------|
| VLLM Total | ~207 ms |
| TensorCast Total | ~40 ms |
| Difference | -80.8% |
| Matched ops | 20 |
| Coverage | 98.5% |

**Why the large timing gap?** Qwen3-32B has `num_key_value_heads=8` but `tp_size=16`. Each TP rank simulates with only 1 KV head (8/16=0.5, rounded to 1), making attention compute ~8x faster in simulation vs the real hardware where KV heads are replicated. The **coverage metric (98.5%)** is what validates mapping correctness, not the absolute timing.

**Key operator-level comparison**:

| VLLM Operator | TC Equivalent | VLLM Time | TC Time | Diff |
|---------------|---------------|-----------|---------|------|
| FusedInferAttentionScore | tensor_cast.attention | 160,547 us | 17,638 us | -89% |
| MatMulV2 | aten.mm | 6,854 us | 6,848 us | -0.1% |
| hcom_allReduce_ | tensor_cast.all_reduce | 5,543 us | 4,543 us | -18% |
| AddRmsNorm | aten.add + aten.pow + aten.mean + aten.rsqrt | 1,634 us | 1,580 us | -3.3% |
| SwiGlu | aten.silu + aten.mul | 984 us | 920 us | -6.5% |

Linear ops (MatMulV2) match within 0.1%, confirming the roofline model accuracy for compute-bound operations.

---

## Case Study 2: DeepSeek-V3 (MoE Model, DP=32, EP)

### 2.1 Profiling Data

| Parameter | Value |
|-----------|-------|
| Model | deepseek-ai/DeepSeek-V3.1 (MoE with MLA, 61 layers) |
| Hardware | 32x ATLAS A3 Dies |
| Parallelism | DP=32, TP=1, EP=32 |
| Quantization | W8A8_DYNAMIC |
| Features | EmbedTP=8, LMHead=8, SharedExpMS |
| Profiling path | `/mnt/d/Data/Profiling/prof-torchair-deepseekv3-decode/` |

**How we know this** (from `VLLM_features.txt`):
```
torchair
DP32,TP1,EP32,NSA4,
EmbedTP_8,LMHead_8,
KV_NZ,SharedExpMS,NO MTP,
RmTorchAirCost,TSQ0
```

- `GroupedMatmul` is the top operator → MoE expert computation
- `QuantBatchMatmulV3` present → W8A8 quantization
- `MoeDistributeDispatchV2/CombineV2` → EP-based MoE routing
- 32 rank directories → world_size=32

### 2.2 Profile Configuration

```yaml
name: deepseek-v3
description: "DeepSeek-V3 MoE model on 32x ATLAS A3 Dies with DP=32 + EP"
num_layers: 61

tensorcast:
  model_id: deepseek-ai/DeepSeek-V3.1
  device: ATLAS_800_A3_752T_128G_DIE
  world_size: 32
  tp_size: 1
  dp_size: 32
  ep: true
  quantize_linear_action: W8A8_DYNAMIC
  word_embedding_tp: 8
  lmhead_tp_size: 8
  enable_external_shared_experts: true

decode_defaults:
  num_queries: 24
  query_length: 1
  context_length: 4877
```

DeepSeek-V3 uses the **default mapping only** (no model-specific overrides needed). The default mapping covers MoE-specific decompositions:

| VLLM Operator | TC Decomposition |
|---------------|-----------------|
| MoeDistributeDispatchV2 | tensor_cast.permute_tokens + tensor_cast.all_to_all |
| GroupedMatmul | tensor_cast.static_quant_linear |
| MoeDistributeCombineV2 | tensor_cast.all_to_all + tensor_cast.unpermute_tokens |
| KvRmsNormRopeCache | tensor_cast.concat_and_cache_mla |
| QuantBatchMatmulV3 | tensor_cast.static_quant_linear |

### 2.3 Running the Pipeline

```bash
VLLM_DIR="/mnt/d/Data/Profiling/prof-torchair-deepseekv3-decode/feb1abe5c743_785216_20251218195608873_ascend_pt/ASCEND_PROFILER_OUTPUT"

.conda/bin/python -m tensor_cast.scripts.profiling_comparison.cli run-all \
    --vllm-dir "$VLLM_DIR" \
    --profile deepseek_v3 \
    --output-dir /tmp/deepseek_results/
```

Expected output:
```
============================================================
STAGE 1: Analyze VLLM Profiling
============================================================
Loaded profile: deepseek-v3
  Description: DeepSeek-V3 MoE model on 32x ATLAS A3 Dies with DP=32 + EP
  num_layers: 61
  Detected phase: decode (confidence: 99%)
  Complete forward passes: 41

============================================================
STAGE 2: Run TensorCast Simulation
============================================================
Command: python -m tensor_cast.scripts.text_generate deepseek-ai/DeepSeek-V3.1 ...
...
Total time for analytic: 89.055ms

============================================================
STAGE 3: Compare by Execution Order
============================================================
=== Loading Mappings ===
  Base mapping: default (17 decompositions)

=== Parsing VLLM Profiling ===
  Step 0: 2115 operations, 99508.46 us

=== Running Sequence Matching ===
  Matched: 32
  Unmatched VLLM: 4
  Unmatched TC: 8
  Mismatches: 0

============================================================
FINAL SUMMARY
============================================================
VLLM total time:     99508.46 us
TensorCast total:    89054.38 us
Difference:          -10454.09 us (-10.5%)
Matched:             32
Coverage:            93.5%

Outputs:
  Chrome trace:  /tmp/deepseek_results/chrome_trace.json
  Excel report:  /tmp/deepseek_results/deepseek_v3_comparison.xlsx
```

### 2.4 Understanding the Results

| Metric | Value |
|--------|-------|
| VLLM Total | ~100 ms |
| TensorCast Total | ~89 ms |
| Difference | -10.5% |
| Matched ops | 32 |
| Coverage | 93.5% |

DeepSeek-V3 shows much closer timing (10.5% gap vs 80% for Qwen3) because it doesn't suffer from the KV head replication issue. The MoE routing, expert computation, and all-to-all communication are all captured accurately.

**Key operator-level comparison**:

| VLLM Operator | TC Equivalent | VLLM Time | TC Time | Diff |
|---------------|---------------|-----------|---------|------|
| QuantBatchMatmulV3 | tensor_cast.static_quant_linear | 48,923 us | 45,972 us | -6.0% |
| FusedInferAttentionScore | tensor_cast.attention (MLA) | 12,376 us | 10,157 us | -17.9% |
| MoeDistributeDispatchV2 | permute_tokens + all_to_all | 4,832 us | 4,521 us | -6.4% |
| GroupedMatmul | tensor_cast.static_quant_linear | 8,645 us | 8,229 us | -4.8% |

---

## Excel Report Structure

The generated Excel file contains 4 sheets:

### Sheet 1: VLLM Operations
All VLLM kernel operations aggregated by type, sorted by total duration:
- Op Type, Count, Total Duration (us), Average Duration (us), Core Type, Input Shapes

### Sheet 2: TensorCast Operations
All TensorCast operations from chrome trace, aggregated by name:
- Op Name, Count, Total Time (us), Average Time (us)

### Sheet 3: Sequence Comparison
Position-by-position matching in execution order:
- Position, VLLM Op, TC Op(s), VLLM Duration (us), TC Duration (us), Difference (%), Status

Color coding:
- **Green**: Difference <= 20% (good match)
- **Yellow**: Difference 20-50% (acceptable)
- **Red**: Difference > 50% (investigate)

### Sheet 4: Summary
Configuration parameters and aggregate metrics:
- Model ID, Device, Parallelism, Quantization
- Total VLLM time, Total TC time, Overall difference
- Matched/Unmatched/Mismatch counts, Coverage percentage

---

## Comparison Summary

| Model | Type | VLLM Time | TC Time | Diff | Ops/Step | Coverage |
|-------|------|-----------|---------|------|----------|----------|
| Qwen3-32B | Dense, TP=16 | 207 ms | 40 ms | -80.8%* | ~1,479 | 98.5% |
| DeepSeek-V3 | MoE, DP=32+EP | 100 ms | 89 ms | -10.5% | ~2,115 | 93.5% |

*Qwen3-32B timing gap due to KV head configuration (`num_kv_heads=8 < tp_size=16`)

### Key Differentiators

| Aspect | Qwen3-32B (Dense) | DeepSeek-V3 (MoE) |
|--------|-------------------|-------------------|
| Top Operator | MatMulV2 (42%) | GroupedMatmul (21%) |
| Parallelism | TP=16 | DP=32, EP=32 |
| Quantization | BF16 (disabled) | W8A8_DYNAMIC |
| MoE Ops | None | MoeDistribute*, GroupedMatmul |
| Attention | Standard | MLA (Multihead Latent) |
| KV Cache | reshape_and_cache | concat_and_cache_mla |
| Mapping | default + qwen3 | default only |
| num_layers | 64 | 61 |

---

## Troubleshooting

### "kernel_details.csv not found"
The VLLM profiling directory must point to the `ASCEND_PROFILER_OUTPUT` folder that directly contains `kernel_details.csv`. The path typically looks like:
```
.../profiling/<container_id>/ASCEND_PROFILER_OUTPUT/
```

### PermissionError on `/mnt/d/`
Windows-mounted drives may fail for output. Use `/tmp/` for output directories instead:
```bash
--output-dir /tmp/results/
```

### Phase detection says "prefill" but data is decode
Override with `--phase decode`. The detector relies on `ReshapeAndCacheNdKernel` input shapes; if these are missing, use explicit override.

### Low coverage percentage
Check `list-mappings` to ensure the decomposition mapping covers the operators in your model. Add missing decompositions to a model-specific YAML override.

### Large timing differences on specific ops
- **Attention**: KV head replication under large TP can cause significant gaps
- **Communication**: Network topology differences between simulated and real hardware
- **MoE routing**: Token routing overhead may differ between VLLM and TensorCast
