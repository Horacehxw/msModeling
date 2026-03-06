# CLAUDE.md

**Wiki**: https://deepwiki.com/Horacehxw/msModeling

## Project Overview

MindStudio-Modeling (msmodeling) is a performance simulation framework for LLMs and diffusion models. It predicts neural network performance on specific hardware without physical accelerator access.

Three main subsystems:
- **TensorCast**: Operator-level performance simulation using PyTorch's `TorchDispatchMode`
- **ServingCast**: Service-level discrete event simulation using salabim
- **Perf Database**: Profiling-data-driven performance estimation (`EmpiricalPerformanceModel + DataSource`)

## Git Workflow

- Default branch (PR target): `develop`
- Feature branches: `feat/*`
- Perf database branch: `feat/perf-database` (from `feat/database`)

## Build & Development

```bash
# Setup
pip install lintrunner && lintrunner init

# Lint (run before every commit)
lintrunner -a

# Install deps
pip install -r ./tensor_cast/requirements.txt
pip install -r ./serving_cast/requirements.txt

# Tests
pip install pytest-xdist
pytest ./tensor_cast/tests -n auto
pytest ./serving_cast/tests -n auto
pytest ./tests
pytest ./tests/perf_database/ -v  # perf database tests

# Single test
pytest ./tensor_cast/tests/test_specific.py::test_function_name -v
```

- **Python**: TensorCast 3.10+, ServingCast 3.9+
- **Style**: black (line-length 88), isort (profile "black"), ruff + flake8 via lintrunner

## Key CLI Commands

### TensorCast - Analytic Mode (default)
```bash
# Prefill
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 2 --query-length 3500 --device TEST_DEVICE

# Decode
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 10 --query-length 1 --context-length 4500 \
  --device TEST_DEVICE --quantize-linear-action W8A8_STATIC

# Kimi-K2 (TP=4, DP=8, EP, W4A8)
python -m tensor_cast.scripts.text_generate moonshotai/Kimi-K2-Instruct \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 32 --tp-size 4 --dp-size 8 --ep \
  --quantize-linear-action W4A8_STATIC --num-queries 18 --query-length 4096 \
  --chrome-trace ./kimi_k2_prefill_trace.json
```

### TensorCast - Profiling Mode (perf database)
```bash
# --compile is REQUIRED for profiling mode (even BF16) — without it,
# fused ops decompose to 72+ aten primitives that can't match profiling kernels
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 2 --query-length 3500 \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --performance-model profiling --compile

# With quantization
python -m tensor_cast.scripts.text_generate deepseek-ai/DeepSeek-V3 \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 32 --tp-size 4 --dp-size 8 --ep \
  --quantize-linear-action W8A8_STATIC \
  --performance-model profiling --compile --perf-database ./path/to/data
```

### TensorCast - Benchmarking
```bash
python -m tensor_cast.scripts.benchmark \
  --model-id Qwen/Qwen3-235B-A22B --device ATLAS_800_A2_280T_64G \
  --num-devices 16 --input-length 3500 --output-length 1500 \
  --ttft-limits 3 --tpot-limits 0.1 0.05
```

### ServingCast
```bash
export PYTHONPATH=/path/to/msmodeling:$PYTHONPATH
python serving_cast/main.py \
  --instance_config_path=./serving_cast/example/instances.yaml \
  --common_config_path=./serving_cast/example/common.yaml
```

## Mapping vLLM to TensorCast

| vLLM Parameter | TensorCast Equivalent |
|---|---|
| `--data-parallel-size` | `--dp-size` |
| `--tensor-parallel-size` | `--tp-size` |
| `--enable-expert-parallel` | `--ep` |
| `--max-num-seqs` | `--num-queries` |
| `--max-num-batched-tokens` (prefill) | `--query-length` |
| `--quantization ascend` W4A8 | `--quantize-linear-action W4A8_STATIC` |
| Prefill workload | `query-length > 1000`, `context-length = 0` |
| Decode workload | `query-length = 1`, `context-length > 0` |

## TensorCast Architecture

### Seven-Layer Design

1. **Configuration**: `ModelConfig`, `ParallelConfig`, `QuantConfig`
2. **Model Transformation**: `TransformerModel` pipeline (7 stages in `tensor_cast/transformers/model.py`)
3. **Execution**: `Runtime` intercepts ops via `TorchDispatchMode`
4. **Operation Analysis**: `OpInvokeInfo` captures metadata; properties via `@OpInvokeInfo.register_op_properties()`
5. **Performance Estimation**: `AnalyticPerformanceModel` (Roofline) or `EmpiricalPerformanceModel` (profiling data)
6. **Memory Analysis**: `MemoryTracker` simulates allocation patterns
7. **Hardware Abstraction**: `DeviceProfile` + `CommGrid` (compute/memory/network specs)

### Custom Operations (`torch.ops.tensor_cast`)

- **Quantization**: `quantize`, `dequantize`, `static_quant_linear`, `fp8_linear`, `mxfp4_linear`
- **Attention**: `attention`, `attention_quant`, `multihead_latent_attention`
- **Communication**: `all_reduce`, `all_gather`, `all_to_all`
- **MoE**: `permute_tokens`, `unpermute_tokens`
- **Cache**: `reshape_and_cache`, `concat_and_cache_mla`
- **Fusion**: `matmul_all_reduce` (MC2), `swiglu`, `add_rms_norm2`, `apply_rope`

### Compilation / Fusion Passes

`--compile` enables pattern-based fusion in `tensor_cast/compilation/patterns/`:
- `swiglu.py` — SwiGlu fusion
- `rms_norm.py` — RmsNorm / AddRmsNorm fusion
- `rotary_embedding.py` — RoPE fusion
- `freezing_passes/grouped_matmul_swiglu_pass.py` — GroupedMatmul+SwiGlu (5 quant variants)
- `freezing_passes/patterns/matmul_allreduce.py` — MC2 MatMul+AllReduce fusion

## Perf Database Subsystem

Design doc: `docs/perf_database/OPERATOR_PERF_DATABASE_DESIGN_zh_v1.2.md`
Work plan: `docs/perf_database/WORK_PLAN_Q1.md`
Spike report: `docs/perf_database/reports/spike_executive_summary_zh.md`

### Architecture

```
Runtime (OpInvokeInfo) → EmpiricalPerformanceModel → DataSource.lookup()
                                                      ├── ProfilingDataSource (op_mapping + CSV query)
                                                      ├── InterpolatingDataSource (wrapper: miss → interpolate)
                                                      └── fallback → AnalyticPerformanceModel
```

### Query Dispatch (ProfilingDataSource)

| Category | Method | Data Source |
|----------|--------|-------------|
| `compute` | op_mapping → kernel_type → CSV shape match | `{kernel_type}.csv` |
| `communication` | message_bytes + topology_tier | comm CSV |
| `composite` | decompose sub_kernels → query each → sum | sub-kernel CSVs |
| `attention_special` | (batch, seq, heads, head_dim) | `FusedInferAttentionScore.csv` |
| `zero_cost` | return 0 | — (view, permute, etc.) |
| MISS | fallback | AnalyticPerformanceModel |

### Data Directory Layout

```
tensor_cast/performance_model/perf_database/
├── data_source.py              # DataSource ABC: lookup(OpInvokeInfo) → LookupResult
├── profiling_data_source.py    # CSV query + 8 shape matching rules + op_mapping
├── interpolating_data_source.py # Wrapper: nearest-neighbor + linear interpolation
├── data/
│   └── {device}/{backend}/{version}/
│       ├── op_mapping.yaml     # TC op → NPU kernel_type mapping (60+ entries)
│       ├── MatMulV2.csv        # Per-kernel profiling data
│       ├── FusedInferAttentionScore.csv
│       └── hcom_allReduce_.csv
└── __init__.py
```

### op_mapping.yaml Structure

```yaml
tensor_cast.swiglu:
  kernel_type: SwiGlu              # Primary NPU kernel type
  alternate_kernel_types: [...]     # Fallback kernel types to try
  zero_cost: true                   # Pure shape ops (view, permute)
  composite: true                   # Decompose into sub_kernels
  sub_kernels: [MatMulV2, hcom_allReduce_]
  interpolation_policy:
    kernel_overrides:
      FusedInferAttentionScore: {transform: sqrt}  # O(n^2) ops need sqrt before interpolation
```

References: `docs/perf_database/examples/op_mapping_example.yaml`, `docs/perf_database/tutorial/OP_PLUGIN_MAPPING_TUTORIAL.md`

### TC vs NPU Shape Differences (8 types)

| # | Type | TC | NPU Profiling | Handling |
|---|------|----|----|---|
| 1 | Batch dim | `(1,S,D)` | `(S,D)` | `_strip_batch_dim()` |
| 2 | Seq padding | `ceil(S/16)*16` | raw S | block-padding tolerance |
| 3 | FRACTAL_NZ | ND `(K,N)` | `[H,W,bh,bw]` | `fractal_nz_to_nd()` |
| 4 | ND transpose | `(K,N)` | `(N,K)` | MatMul-specific check |
| 5 | SwiGlu inputs | 2x`(S,D/2)` | 1x`(S,D)` | concat on last dim |
| 6 | RoPE layout | `(B,H,S,D)` Q,K | `(B,S,H,D)` K,Q | normalize + reorder |
| 7 | RoPE kernel | single TC op | multiple NPU kernels | `alternate_kernel_types` |
| 8 | Composite ops | fused (matmul+allreduce) | may be separate | `_lookup_composite()` |

### Data Collection Tools (`tools/perf_data_collection/`)

| Tool | Purpose |
|------|---------|
| `parse_kernel_details.py` | Parse NPU kernel_details.csv → per-kernel CSVs |
| `discover_operators.py` | Compare Profiling vs op_mapping coverage |
| `generate_shape_grid.py` | Generate microbenchmark shape grids |
| `generate_microbench.py` | Generate torch_npu benchmark scripts |
| `generate_comm_microbench.py` | Generate HCCL communication benchmarks |
| `build_database.py` | Build final CSV database from microbenchmark results |
| `validate.py` | Per-operator + end-to-end accuracy validation |

## Quantization Framework

| Scheme | Weights | Activations | Use Case |
|--------|---------|-------------|----------|
| W8A8_STATIC | INT8 | INT8 (static) | Maximum performance |
| W8A8_DYNAMIC | INT8 | INT8 (dynamic) | Balanced accuracy/speed |
| W4A8_STATIC | INT4 | INT8 (static) | Memory constrained |
| W4A8_DYNAMIC | INT4 | INT8 (dynamic) | Memory constrained |
| FP8 | FP8 | FP8 | Native FP8 hardware |
| MXFP4 | MXFP4 | MXFP4 | Microscaling float |

## Parallelization

| Type | Implementation | Communication |
|------|----------------|---------------|
| **TP** | `ColumnParallelLinear`, `RowParallelLinear` | all-reduce, all-gather |
| **EP** | `ParallelMoELayer` | all-to-all |
| **DP** | Batch distribution | — |
| **SP** | Ulysses-style | all-to-all |

Configured via `--tp-size`, `--dp-size`, `--ep`, `--world-size`.

## Critical Files

| File | Purpose |
|------|---------|
| `tensor_cast/runtime.py` | Core `Runtime` using `TorchDispatchMode` |
| `tensor_cast/device.py` | `DeviceProfile` and `CommGrid` |
| `tensor_cast/model_config.py` | `ModelConfig`, `ParallelConfig`, `QuantConfig` |
| `tensor_cast/core/model_runner.py` | Inference API, profiling hooks |
| `tensor_cast/core/config_resolver.py` | Config → model transformations |
| `tensor_cast/transformers/utils.py` | Model type detection, config loading |
| `tensor_cast/layers/quant_linear.py` | Quantization (W4A8, W8A8, FP8) |
| `tensor_cast/performance_model/analytic.py` | Roofline-based performance model |
| `tensor_cast/performance_model/empirical.py` | Profiling-based performance model |
| `tensor_cast/performance_model/perf_database/profiling_data_source.py` | CSV query + shape matching engine |
| `tensor_cast/performance_model/perf_database/data_source.py` | DataSource ABC interface |
| `tensor_cast/performance_model/perf_database/interpolating_data_source.py` | Interpolation wrapper |
| `serving_cast/main.py` | ServingCast entry point |
| `serving_cast/engine.py` | Batch scheduling, KV cache preemption |
| `stime.py` | DES time management (`Task`, `elapse()`, `now()`) |

## Common Issues

### `--compile` Required for Profiling Mode
Without `--compile`, fused ops (RmsNorm, SwiGlu, RoPE) decompose to 72+ aten primitives that can't match profiling kernels. `--compile` is orthogonal to `--quantize-linear-action`.

### W4A8 Meta Device Quantization Error
W4A8_STATIC fails with infinite recursion in `pack_int4()` (`tensor_cast/layers/quant_linear.py:160`). Meta tensors don't support `clamp()`. Workaround: `if tensor.device.type != 'meta': tensor = tensor.clamp(-8, 7)`

### Model Type Mismatches (Kimi-K2 as DeepSeek)
Kimi-K2 auto-detected and reloaded as DeepSeek-V3 in `tensor_cast/transformers/utils.py:287-374`.

## Design Principles

### EmpiricalPerformanceModel + DataSource

EmpiricalPerformanceModel accepts a generic DataSource interface (e.g., ProfilingDataSource) for profiling-data-based estimation. Design should align with actual NPU kernels — match real kernel types and shapes, don't compromise for TC's op abstractions. Long-term: capture op graphs directly from vLLM runs rather than relying on TC dispatch traces. See `docs/perf_database/OPERATOR_PERF_DATABASE_DESIGN_zh_v1.2.md`.
