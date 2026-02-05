# Profiling Comparison Tool

A 3-stage pipeline for comparing VLLM profiling data with TensorCast simulation results.

## Overview

This tool validates TensorCast performance predictions by comparing them against real VLLM profiling data from Ascend NPU execution using **order-based sequence matching**.

### 3-Stage Pipeline

| Stage | Command | Input | Output |
|-------|---------|-------|--------|
| 1. Analyze | `analyze` | VLLM profiling dir + profile | TC execution command printed to console |
| 2. Simulate | `simulate` | Profile/config + phase | Chrome trace JSON (subprocess) |
| 3. Compare | `compare` | VLLM dir + TC chrome trace + mapping | Excel comparison report |
| All | `run-all` | VLLM dir + profile + output dir | Runs 1→2→3 sequentially |

### Sequence-Based Matching

The core matching algorithm walks VLLM and TensorCast operation sequences in **lockstep execution order**, consuming TC ops for each VLLM op according to a decomposition mapping. This eliminates double-counting entirely since each TC op is consumed exactly once.

## Quick Start

### Run All 3 Stages (Recommended)

```bash
.conda/bin/python -m tensor_cast.scripts.profiling_comparison.cli run-all \
    --vllm-dir /path/to/ASCEND_PROFILER_OUTPUT \
    --profile qwen3_32b \
    --output-dir /tmp/results/
```

### Run Individual Stages

```bash
# Stage 1: Analyze VLLM profiling (prints TC command)
.conda/bin/python -m tensor_cast.scripts.profiling_comparison.cli analyze \
    --vllm-dir /path/to/ASCEND_PROFILER_OUTPUT \
    --profile qwen3_32b

# Stage 2: Run TensorCast simulation
.conda/bin/python -m tensor_cast.scripts.profiling_comparison.cli simulate \
    --profile qwen3_32b \
    --phase decode \
    --output-dir /tmp/tc_results/

# Stage 3: Compare by execution order
.conda/bin/python -m tensor_cast.scripts.profiling_comparison.cli compare \
    --vllm-dir /path/to/ASCEND_PROFILER_OUTPUT \
    --tc-trace /tmp/tc_results/chrome_trace.json \
    --profile qwen3_32b \
    --output /tmp/comparison.xlsx
```

### List Available Resources

```bash
.conda/bin/python -m tensor_cast.scripts.profiling_comparison.cli list-profiles
.conda/bin/python -m tensor_cast.scripts.profiling_comparison.cli list-mappings
```

## Directory Structure

```
tensor_cast/scripts/profiling_comparison/
├── cli.py                      # CLI entry point (6 subcommands)
├── stages/
│   ├── analyze.py              # Stage 1: VLLM analysis + TC command
│   ├── simulate.py             # Stage 2: TC subprocess execution
│   └── compare.py              # Stage 3: Sequence matching + Excel
├── config/
│   ├── schema.py               # Configuration dataclasses
│   ├── loader.py               # YAML profile loading
│   └── profiles/               # Model profiles
│       ├── qwen3_32b.yaml
│       └── deepseek_v3.yaml
├── parsers/
│   ├── kernel_details_parser.py # VLLM kernel_details.csv parser
│   ├── chrome_trace_parser.py  # TC chrome trace JSON parser
│   └── phase_detector.py       # Automatic phase detection
├── alignment/
│   ├── sequence_matcher.py     # Core lockstep matching algorithm
│   └── mappings/               # Decomposition mapping files
│       ├── default.yaml        # Default decomposition mappings
│       └── qwen3.yaml          # Qwen3-specific overrides
├── output/
│   ├── base.py                 # Output data structures
│   └── excel.py                # Excel report generator
└── tests/                      # Unit tests
```

## Configuration

### Model Profiles

Model profiles define default TensorCast configurations. Create profiles in `config/profiles/`:

```yaml
name: my-model
description: "My custom model profile"
num_layers: 64  # Critical for correct step detection

tensorcast:
  model_id: organization/model-name
  device: ATLAS_800_A3_752T_128G_DIE
  world_size: 16
  tp_size: 16

decode_defaults:
  num_queries: 136
  query_length: 1
  context_length: 4096
```

### Decomposition Mappings

Decomposition mappings define how VLLM fused ops decompose into ordered TC op sequences:

```yaml
version: "2.0"

decompositions:
  AddRmsNorm:
    tc_ops: ["aten.add", "aten.pow", "aten.mean", "aten.rsqrt"]
  MatMulV2:
    tc_ops: ["aten.mm"]
  SwiGlu:
    tc_ops: ["aten.silu", "aten.mul"]

ignored_vllm_ops:
  - TensorMove
  - Fill

ignored_tc_ops:
  - aten.view
  - aten.reshape
```

## Output

### Excel Report Structure

The generated Excel report contains four sheets:

1. **VLLM Operations**: All VLLM operations with timing data
2. **TensorCast Operations**: All TC operations from chrome trace
3. **Sequence Comparison**: Position-by-position matching with color coding
4. **Summary**: Configuration and summary statistics

## Testing

```bash
.conda/bin/python -m pytest tensor_cast/scripts/profiling_comparison/tests/ -v
```
