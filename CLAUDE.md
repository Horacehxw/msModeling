# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Wiki Reference**: https://deepwiki.com/Horacehxw/msModeling

## Project Overview

MindStudio-Modeling (msmodeling) is a performance simulation framework for LLMs and diffusion models. It enables developers to predict neural network performance on specific hardware without requiring physical accelerator access by functioning as a virtual execution environment that intercepts PyTorch operations.

Two main components:
- **TensorCast**: Operator-level performance simulation using PyTorch's `TorchDispatchMode`
- **ServingCast**: Service-level discrete event simulation using salabim

## Python Environment

**IMPORTANT**: Always use the project's virtual environment located at `.venv/`:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Or run commands directly with the venv Python
.venv/bin/python -m tensor_cast.scripts.text_generate ...
```

When running any Python command in this project, use `.venv/bin/python` instead of the system Python.

## Build & Development Commands

### Setup
```bash
pip install lintrunner
lintrunner init  # one-time setup
```

### Linting (run before every commit)
```bash
lintrunner -a  # check and auto-fix style issues
```

### Install Dependencies
```bash
pip install -r ./tensor_cast/requirements.txt
pip install -r ./serving_cast/requirements.txt
```

### Run Tests
```bash
pip install pytest-xdist
pytest ./tensor_cast/tests -n auto
pytest ./serving_cast/tests -n auto
pytest ./tests  # root-level tests
```

### Run Single Test
```bash
pytest ./tensor_cast/tests/test_specific.py -v
pytest ./tensor_cast/tests/test_specific.py::test_function_name -v
```

## Key CLI Commands

### TensorCast - Text Generation Simulation
```bash
# Prefill simulation
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 2 --query-length 3500 --device TEST_DEVICE

# With quantization
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 2 --query-length 3500 --context-length 4500 \
  --device TEST_DEVICE --quantize-linear-action W8A8_DYNAMIC

# Decode simulation
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 10 --query-length 1 --context-length 4500 \
  --device TEST_DEVICE --quantize-linear-action W8A8_STATIC

# Real-world example: Kimi-K2 on 32 A3 Dies (TP=4, DP=8, EP, W4A8)
python -m tensor_cast.scripts.text_generate moonshotai/Kimi-K2-Instruct \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 32 --tp-size 4 --dp-size 8 --ep \
  --quantize-linear-action W4A8_STATIC --num-queries 18 --query-length 4096 \
  --chrome-trace ./kimi_k2_prefill_trace.json
```

### TensorCast - Throughput Benchmarking
```bash
python -m tensor_cast.scripts.benchmark \
  --model-id Qwen/Qwen3-235B-A22B --device ATLAS_800_A2_280T_64G \
  --num-devices 16 --input-length 3500 --output-length 1500 \
  --ttft-limits 3 --tpot-limits 0.1 0.05
```

### TensorCast - Video Generation Simulation
```bash
python -m tensor_cast.scripts.video_generate <model_path> \
  --device TEST_DEVICE --num-devices 8
```

### ServingCast - Service Simulation
```bash
export PYTHONPATH=/path/to/msmodeling:$PYTHONPATH
python serving_cast/main.py \
  --instance_config_path=./serving_cast/example/instances.yaml \
  --common_config_path=./serving_cast/example/common.yaml

# With profiling
python serving_cast/main.py \
  --instance_config_path=./serving_cast/example/instances.yaml \
  --common_config_path=./serving_cast/example/common.yaml \
  --enable_profiling --profiling_output_path ./results/profiling
```

### ServingCast - Performance Analysis (PD Aggregation)
```bash
python -m serving_cast.scripts.performance_analyze \
  --model-id Qwen/Qwen3-32B --device TEST_DEVICE --num-devices 8 \
  --input-length 3500 --output-length 1500 --tpot-limits 50
```

## TensorCast Architecture

### Seven-Layer Design

1. **Configuration Layer**: `ModelConfig`, `ParallelConfig`, `QuantConfig` define optimization strategies
2. **Model Transformation Layer**: `TransformerModel` orchestrates the optimization pipeline
3. **Execution Layer**: `Runtime` intercepts operations via `TorchDispatchMode`
4. **Operation Analysis Layer**: `OpInvokeInfo` captures operation metadata and tensor characteristics
5. **Performance Estimation Layer**: Multiple performance models estimate execution costs
6. **Memory Analysis Layer**: `MemoryTracker` simulates allocation patterns and tracks peak usage
7. **Hardware Abstraction Layer**: `DeviceProfile` specifies compute/memory/network characteristics

### Model Transformation Pipeline (7 Stages)

Executed in strict order in `tensor_cast/transformers/model.py`:

1. **Meta Device Initialization**: Load model structure without memory allocation
2. **Model Wrapping**: Normalize interfaces via `CausalLmWrapper` or `VLModelWrapper`
3. **Multi-Token Prediction**: Add optional MTP layers for parallel token prediction
4. **Layer Repetition Optimization**: Mark repetitive decoder layers for analysis reuse
5. **Module Patching**: Replace attention, MLA, and MoE modules with optimized implementations
6. **Quantization Application**: Apply precision reduction via `TensorCastQuantLinear`
7. **Parallelism Distribution**: Apply TP/EP/DP sharding across devices

### Key Files

- **runtime.py**: Core `Runtime` class using `TorchDispatchMode` to intercept all PyTorch operations
- **device.py**: `DeviceProfile` (compute TFLOPS, memory bandwidth) and `CommGrid` (network topology)
- **model_config.py**: `ModelConfig`, `ParallelConfig`, `QuantConfig` configuration dataclasses
- **performance_model/analytic.py**: Roofline-based `AnalyticPerformanceModel`
- **performance_model/comm_analytic.py**: `CommAnalyticModel` for collective communication costs
- **performance_model/memory_tracker.py**: `MemoryTracker` for allocation simulation

### OpInvokeInfo System

Operations are captured as `OpInvokeInfo` objects containing:
- Function reference and arguments
- Output tensors and metadata
- Cache key for performance model caching

Property extractors are registered via decorator:
```python
@OpInvokeInfo.register_op_properties(torch.ops.aten.mm.default)
```

### Custom Operations

Registered in `torch.ops.tensor_cast` namespace:
- **Quantization**: `quantize`, `dequantize`, `static_quant_linear`, `fp8_linear`, `mxfp4_linear`
- **Attention**: `attention`, `attention_quant`, `multihead_latent_attention`
- **Communication**: `all_reduce`, `all_gather`, `all_to_all`
- **MoE**: `permute_tokens`, `unpermute_tokens`
- **Cache**: `reshape_and_cache`, `concat_and_cache_mla`

## ServingCast Architecture

### Discrete Event Simulation

Built on **salabim** framework with logical time management in `stime.py`:

- **SimulationEnv**: Singleton wrapper around `salabim.Environment`
- **Task**: Base class extending `salabim.Component` with `process()`, `wait()`, `notify()`
- **Time Functions**: `now()` queries logical time, `elapse(ts)` advances time for current task

### Task Lifecycle States

`Created` → `Scheduled` → `Running` → `Holding`/`Passive` → `Terminated`

### Key Files

- **stime.py**: `SimulationEnv`, `Task`, `CallableTask`, time utilities
- **engine.py**: `BatchScheduler` for token-aware batch scheduling
- **serving.py**: PD (Prefill-Decode) aggregation and disaggregation modes
- **kv_cache_manager.py**: Block-based KV cache memory management
- **config.py**: `ParallelConfig`, `InstanceConfig`, `ModelConfig`, `ServingConfig`
- **request.py**: Request states (PREFILLING → DECODING → COMPLETED)

## Quantization Framework

| Scheme | Weights | Activations | Use Case |
|--------|---------|-------------|----------|
| W8A8_STATIC | INT8 | INT8 (static) | Maximum performance |
| W8A8_DYNAMIC | INT8 | INT8 (dynamic) | Balanced accuracy/speed |
| W4A8_STATIC | INT4 | INT8 (static) | Memory constrained |
| W4A8_DYNAMIC | INT4 | INT8 (dynamic) | Memory constrained |
| FP8 | FP8 | FP8 | Native FP8 hardware |
| MXFP4 | MXFP4 | MXFP4 | Microscaling float |
| INT8 KV Cache | — | INT8 | Attention memory reduction |

Configuration via `QuantConfig` mapping module paths to `LinearQuantConfig`/`AttentionQuantConfig`.

## Parallelization Framework

| Type | Implementation | Communication |
|------|----------------|---------------|
| **Tensor Parallelism (TP)** | `ColumnParallelLinear`, `RowParallelLinear` | all-reduce, all-gather |
| **Expert Parallelism (EP)** | `ParallelMoELayer` | all-to-all for token routing |
| **Data Parallelism (DP)** | Batch distribution | — |
| **Sequence Parallelism** | Ulysses-style | all-to-all |

Configured via `--tp-size`, `--dp-size`, `--ep`, `--world-size` flags.

## Advanced Model Components

### Mixture of Experts (MoE)

`MoELayer` in `tensor_cast/ops/fused_moe.py`:
1. Gating: Compute routing logits, select top-k experts
2. Token permutation via `permute_tokens()`
3. Expert MLP computation
4. Token unpermutation via `unpermute_tokens()`
5. Weighted output combination

`ParallelMoELayer` adds EP with all-to-all communication and optional redundant/external shared experts.

### Multihead Latent Attention (MLA)

`tensor_cast/ops/mla.py`: Compresses KV cache by projecting into lower-dimensional latent space:
- Prefill: Full decompression via `kv_b_proj`
- Decode: Compressed computation using `W_UK_T` and `W_UV` projections

Supports INT8 quantization for query states, compressed KV, attention probabilities.

## Hardware Abstraction

`DeviceProfile` specifies:
- **Compute**: TFLOPS by data type (INT8, FP8, FP16, FP32) and operation type (MMA vs general)
- **Memory**: Capacity, bandwidth, cache hierarchy
- **Network**: `CommGrid` topology, bandwidth, latency

Built-in profiles for Ascend ATLAS accelerators. Custom devices: drop Python files in `tensor_cast/device_profiles/` for auto-loading.

## Output & Reporting

- **Summary Tables**: Operator-level breakdown (execution time, memory, FLOPs, bound classification)
- **Chrome Trace**: Timeline JSON for `chrome://tracing` visualization
- **Memory Profile**: Peak usage tracking with allocation/deallocation events
- **Metrics**: Total time, throughput (TPS), TTFT, TPOT, peak memory

## Data Flow Pipelines

### TensorCast Pipeline
- **Entry**: `tensor_cast/scripts/text_generate.py` (prefill/decode) or `tensor_cast/scripts/benchmark.py` (throughput)
- **Model Loading**: `tensor_cast/core/model_builder.py` → HuggingFace transformers
- **Simulation**: `tensor_cast/performance_model/analytic.py` estimates operator time using roofline model
- **Hardware**: Device profiles in `tensor_cast/device_profiles/` (TFLOPS, memory bandwidth, interconnect)
- **Optimization**: `tensor_cast/compilation/passes/` applies quantization, merge linear, freezing via custom PyTorch backend
- **Output**: Execution metrics + optional Chrome Trace JSON

### ServingCast Pipeline
- **Entry**: `serving_cast/main.py` with YAML config (instances.yaml + common.yaml)
- **Config**: `serving_cast/config.py` dataclasses (ParallelConfig, ModelConfig, LoadGenConfig, ServingConfig)
- **Simulation Loop**: `stime.Task`-based `BatchScheduler` in `serving_cast/engine.py`:
  - Request arrival simulation
  - Prefill/decode scheduling with token budget constraints
  - KV cache allocation via `kv_cache_manager.py`
  - Device communication modeling
- **Backend**: `serving_cast/service/backend_factory.py` creates MindIE backend
- **Output**: Request latency metrics (E2E, TTFT, TPOT) + optional profiling

## Key Conventions & Patterns

### Configuration Management
- **Dataclass-based config**: `ParallelConfig`, `ModelConfig`, `LoadGenConfig`, `ServingConfig` in `serving_cast/config.py`
- **Device profiles**: Each file in `tensor_cast/device_profiles/` registers device class with `@register_device`
- **YAML schemas**: ServingCast expects flat YAML with nested dicts for model_config, serving_config, load_gen

### Performance Modeling
- **Analytic model**: `OperatorProfile` + hardware specs → execution time (no ML predictor needed)
- **Memory tracking**: `MemoryTracker` tracks peak & total allocations per operator
- **Quantization**: W8A8_DYNAMIC, W4A8_STATIC, MXFP4 applied uniformly to linear layers
- **Parallelism**: Tensor/Sequence/Data parallelism stored in `model_config.parallel_config`

### Code Organization
- **tensor_cast/layers/**: Custom layer implementations (attention, MoE, parallel linear)
- **tensor_cast/core/**: Config resolution, model building, input generation
- **tensor_cast/compilation/passes/**: Graph optimization with pattern matching
- **serving_cast/service/**: Backend abstraction (MindIE aggregation/disaggregation modes)
- **serving_cast/profiler/**: Optional profiling hooks (stime-based instrumentation)

## Mapping vLLM to TensorCast

When translating vLLM deployment to TensorCast simulation:

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

## Common Issues & Solutions

### W4A8 Meta Device Quantization Error
**Issue**: W4A8_STATIC quantization fails with infinite recursion in `tensor.clamp()`
**Location**: `tensor_cast/layers/quant_linear.py:160` in `pack_int4()`
**Cause**: PyTorch meta tensors don't support `clamp()` (triggers `isnan()` recursion)
**Fix**: Add device check before clamp or use W8A8_DYNAMIC instead
```python
if tensor.device.type != 'meta':
    tensor = tensor.clamp(-8, 7)
```

### Model Type Mismatches (Kimi-K2 as DeepSeek)
**Behavior**: Warning about `kimi_k2` → `deepseek_v3` type conversion
**Cause**: Kimi-K2 based on DeepSeek-V3 architecture
**Handling**: `tensor_cast/transformers/utils.py:287-374` auto-detects and reloads with native DeepSeek config
**Benefit**: Enables MoE/MLA optimizations without `trust_remote_code=True`

## Critical Files for New Contributors

| File | Purpose |
|------|---------|
| `stime.py` | Discrete event simulation (understand `Task`, `elapse()`, `now()` first) |
| `tensor_cast/core/model_runner.py` | Inference API, profiling hooks |
| `tensor_cast/core/config_resolver.py` | Config → model transformations |
| `tensor_cast/transformers/utils.py` | Model type detection, config loading |
| `tensor_cast/layers/quant_linear.py` | Quantization implementations (W4A8, W8A8, etc.) |
| `serving_cast/main.py` | Entry point, CLI args, simulation orchestration |
| `serving_cast/engine.py` | Batch scheduling logic, KV cache preemption |
| `serving_cast/config.py` | YAML schema definitions |

## Design Principles

### EmpiricalPerformanceModel + DataSource 模式

EmpiricalPerformanceModel 接受通用的 DataSource 抽象接口（如 ProfilingDataSource），基于实测数据估算算子性能。设计应尽量贴合实测的 Profiling 算子，需要能和实际的 NPU Kernel 对齐算子和 Shape，不应为了迁就 TensorCast 当前的算子抽象而妥协。未来最好直接从 VLLM 实跑抓取算子图（而非依赖 TensorCast 的 dispatch trace）。详见 `docs/OPERATOR_PERF_DATABASE_DESIGN_zh_v1.2.md`。

## Python Version

- TensorCast: Python 3.10+
- ServingCast: Python 3.9+
