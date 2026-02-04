# Profiling Comparison Tool

A modular tool for comparing VLLM profiling data with TensorCast simulation results.

## Overview

This tool enables validation of TensorCast performance predictions by comparing them against real VLLM profiling data from Ascend NPU execution. It supports:

- **Automatic phase detection** for PD Aggregation scenarios
- **Many-to-many operator mapping** between VLLM and TensorCast
- **YAML-based configuration** for model profiles and operator mappings
- **Excel report generation** with detailed comparison metrics

## Quick Start

### Using a Model Profile (Recommended)

```bash
# Compare Qwen3-32B profiling with TensorCast simulation
python -m tensor_cast.scripts.profiling_comparison.cli compare \
    --profile qwen3_32b \
    --vllm-dir /path/to/ASCEND_PROFILER_OUTPUT \
    --output comparison.xlsx
```

### Custom Configuration

```bash
# Compare with explicit TensorCast configuration
python -m tensor_cast.scripts.profiling_comparison.cli compare \
    --vllm-dir /path/to/ASCEND_PROFILER_OUTPUT \
    --model-id Qwen/Qwen3-32B \
    --device ATLAS_800_A3_752T_128G_DIE \
    --world-size 16 --tp-size 16 \
    --num-queries 136 --context-length 4096 \
    --output comparison.xlsx
```

### List Available Resources

```bash
# List available model profiles
python -m tensor_cast.scripts.profiling_comparison.cli list-profiles

# List available operator mappings
python -m tensor_cast.scripts.profiling_comparison.cli list-mappings
```

## Directory Structure

```
tensor_cast/scripts/profiling_comparison/
├── cli.py                      # Unified CLI entry point
├── config/
│   ├── schema.py               # Configuration dataclasses
│   ├── loader.py               # YAML profile loading
│   └── profiles/               # Model profiles
│       ├── qwen3_32b.yaml      # Qwen3-32B configuration
│       └── deepseek_v3.yaml    # DeepSeek-V3 configuration
├── parsers/
│   ├── base.py                 # Parser protocol
│   ├── kernel_details_parser.py # VLLM kernel_details.csv parser
│   ├── tensorcast_adapter.py   # TensorCast simulation adapter
│   └── phase_detector.py       # Automatic phase detection
├── alignment/
│   ├── op_mapper.py            # Operator mapping logic
│   ├── mapping_loader.py       # YAML mapping loader
│   └── mappings/               # Operator mapping files
│       ├── default.yaml        # Default mappings
│       └── qwen3.yaml          # Qwen3-specific mappings
├── output/
│   ├── base.py                 # Formatter protocol
│   └── excel.py                # Excel report generator
├── tests/                      # Unit tests
├── README.md                   # This file (English)
└── README_zh.md                # Chinese documentation
```

## Configuration

### Model Profiles

Model profiles define default TensorCast configurations for specific models. Create profiles in `config/profiles/`:

```yaml
# config/profiles/my_model.yaml
name: my-model
description: "My custom model profile"

tensorcast:
  model_id: organization/model-name
  device: ATLAS_800_A3_752T_128G_DIE
  world_size: 16
  tp_size: 16
  dp_size: 1
  ep: false
  quantize_linear_action: DISABLED

decode_defaults:
  num_queries: 136
  query_length: 1
  context_length: 4096

prefill_defaults:
  num_queries: 136
  query_length: 4096
  context_length: 0
```

### Operator Mappings

Operator mappings define how VLLM operations correspond to TensorCast operations:

```yaml
# alignment/mappings/default.yaml
version: "1.0"

# VLLM fused op -> multiple TC ops
vllm_fusions:
  - name: "AddRmsNorm"
    vllm_ops: ["AddRmsNorm", "InplaceAddRmsNorm"]
    tc_ops: ["aten.add", "tensor_cast.rmsnorm"]
    aggregation: sum

# Multiple VLLM ops -> TC fused op
tc_fusions:
  - name: "attention_block"
    vllm_ops: ["FusedInferAttentionScore", "ReshapeAndCacheNdKernel"]
    tc_ops: ["tensor_cast.attention"]
    aggregation: sum

# Direct 1:1 mappings
direct_mappings:
  MatMulV2:
    - aten.mm
    - aten.matmul
```

## Phase Detection

For PD Aggregation scenarios where both prefill and decode run on the same device, the tool can automatically detect the phase:

### Detection Algorithm

1. **Primary Method**: Query length in `ReshapeAndCacheNdKernel` Input Shapes
   - `query_len == 1` → DECODE (99% confidence)
   - `query_len > 100` → PREFILL (98% confidence)

2. **Secondary Method**: MatMul batch dimension
   - `M <= 256` → DECODE (85% confidence)
   - `M >= 512` → PREFILL (90% confidence)

### Override Phase Detection

```bash
# Force decode phase
python -m tensor_cast.scripts.profiling_comparison.cli compare \
    --profile qwen3_32b \
    --vllm-dir /path/to/profiling \
    --phase decode
```

## Output

### Excel Report Structure

The generated Excel report contains four sheets:

1. **VLLM Operations**: All VLLM operations with timing data
2. **TensorCast Operations**: All TensorCast operations with timing data
3. **Comparison**: Side-by-side comparison with difference metrics
4. **Summary**: Configuration and summary statistics

### Comparison Metrics

- **Time Coverage**: Percentage of VLLM time matched to TensorCast ops
- **Average Difference**: Average absolute percentage difference
- **Match Status**:
  - `exact`: Direct match within 20%
  - `partial`: Match within 50%
  - `missing_in_tc`: VLLM op not found in TensorCast
  - `missing_in_vllm`: TensorCast op not found in VLLM

## CLI Reference

### compare

Run profiling comparison.

```
python -m tensor_cast.scripts.profiling_comparison.cli compare [OPTIONS]

Required:
  --vllm-dir PATH          Path to ASCEND_PROFILER_OUTPUT directory

Profile-based (recommended):
  --profile NAME           Model profile to use (e.g., qwen3_32b)

Custom configuration:
  --model-id ID            HuggingFace model ID
  --device NAME            Device profile name
  --world-size N           Total number of devices
  --tp-size N              Tensor parallelism size
  --dp-size N              Data parallelism size
  --ep                     Enable expert parallelism
  --quantize-linear-action Enable quantization
  --num-queries N          Batch size
  --query-length N         Query length
  --context-length N       Context length

Phase detection:
  --phase {auto,prefill,decode}  Phase type (default: auto)
  --step-index N           Decode step index (default: 100)

Output:
  --output PATH            Output file path
  --format {excel,json}    Output format (default: excel)
  --mapping NAME           Custom mapping file
```

### list-profiles

List available model profiles.

```
python -m tensor_cast.scripts.profiling_comparison.cli list-profiles
```

### list-mappings

List available operator mappings.

```
python -m tensor_cast.scripts.profiling_comparison.cli list-mappings
```

## Testing

Run tests with pytest:

```bash
# Run all profiling comparison tests
pytest tensor_cast/scripts/profiling_comparison/tests/ -v

# Run specific test module
pytest tensor_cast/scripts/profiling_comparison/tests/test_config.py -v

# Run with coverage
pytest tensor_cast/scripts/profiling_comparison/tests/ --cov=tensor_cast.scripts.profiling_comparison
```

## Extending the Tool

### Adding a New Model Profile

1. Create YAML file in `config/profiles/`:

```yaml
name: new-model
tensorcast:
  model_id: org/new-model
  device: ATLAS_800_A3_752T_128G_DIE
  # ... configuration
decode_defaults:
  num_queries: 64
  query_length: 1
  context_length: 2048
```

2. Optionally create model-specific mappings in `alignment/mappings/`

### Adding Custom Operator Mappings

1. Create YAML file in `alignment/mappings/`:

```yaml
version: "1.0"
vllm_fusions:
  - name: "CustomFusion"
    vllm_ops: ["CustomOp"]
    tc_ops: ["aten.custom1", "aten.custom2"]
    aggregation: sum
```

2. Use with `--mapping` flag:

```bash
python -m tensor_cast.scripts.profiling_comparison.cli compare \
    --mapping custom_mapping --vllm-dir ...
```

### Creating a Custom Output Formatter

Implement the `FormatterProtocol`:

```python
from tensor_cast.scripts.profiling_comparison.output.base import (
    BaseFormatter,
    ComparisonResult,
)

class MyFormatter(BaseFormatter):
    def format(self, result: ComparisonResult, output_path: Path) -> None:
        # Custom formatting logic
        pass
```

## Troubleshooting

### Common Issues

1. **"kernel_details.csv not found"**
   - Ensure VLLM profiling was run with kernel-level profiling enabled
   - Check the profiling directory path is correct

2. **"Profile not found"**
   - Run `list-profiles` to see available profiles
   - Check profile name spelling (use underscores or hyphens)

3. **Phase detection incorrect**
   - Use `--phase decode` or `--phase prefill` to override
   - Check profiling data contains expected operations

4. **Large time differences**
   - Verify TensorCast configuration matches VLLM deployment
   - Check quantization settings match
   - Review operator mappings for missing fusions

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          CLI (cli.py)                           │
│                  Unified entry point with subcommands           │
└─────────────────────────────────────────────────────────────────┘
                                  │
           ┌──────────────────────┼──────────────────────┐
           ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Config Layer  │    │  Parser Layer   │    │ Alignment Layer │
│ - schema.py     │    │ - VLLM parser   │    │ - op_mapper.py  │
│ - loader.py     │    │ - TC adapter    │    │ - YAML loader   │
│ - profiles/     │    │ - phase detect  │    │ - mappings/     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                  │
                                  ▼
                       ┌─────────────────┐
                       │  Output Layer   │
                       │ - Excel format  │
                       │ - JSON format   │
                       └─────────────────┘
```

## License

See repository LICENSE file.
