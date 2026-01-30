# AI Coding Assistant Instructions for msmodeling

## Project Overview
**msmodeling** (MindStudio-Modeling) is a lightweight full-system simulation framework that evaluates model and service deployment performance. It consists of two main components:

1. **TensorCast**: Performance simulation for PyTorch models on specific hardware. Simulates operator-level execution, memory allocation, and provides bottleneck analysis via Chrome Trace.
2. **ServingCast**: Service-level simulation with request scheduling, KV cache management, and multi-instance/multi-role deployment patterns.

ServingCast leverage **stime** (a discrete event simulation framework built on salabim) for timeline-based simulation.

## Architecture & Data Flows

### TensorCast Pipeline
- **Entry**: `tensor_cast/scripts/text_generate.py` (prefill/decode simulation) or `tensor_cast/scripts/benchmark.py` (throughput search)
- **Core**: `tensor_cast/core/model_runner.py` → `model_builder.py` → loads HuggingFace transformers
- **Simulation**: `tensor_cast/performance_model/analytic.py` estimates operator execution time using roofline model
- **Hardware**: Device profiles in `tensor_cast/device_profiles/` define hardware (TFLOPS, memory bandwidth, cache)
- **Optimization**: `tensor_cast/compilation/` applies graph passes (quantization, merge linear, freezing) via custom PyTorch backend
- **Output**: Execution metrics + optional Chrome Trace files

### ServingCast Pipeline  
- **Entry**: `serving_cast/main.py` with YAML config files (instances.yaml + common.yaml)
- **Config Loading**: `serving_cast/config.py` defines dataclasses for parallelism, models, load generation
- **Simulation Loop**: `stime.Task`-based scheduler in `serving_cast/engine.py` (BatchScheduler) orchestrates:
  - Request arrival (load_gen)
  - Prefill/decode scheduling with token budget constraints
  - KV cache allocation (`kv_cache_manager.py`)
  - Device communication modeling
- **Backend**: `serving_cast/service/backend_factory.py` creates MindIE backend for operator simulation
- **Output**: Request latency metrics (E2E, TTFT, TPOT) + optional profiling

### stime (Simulation Time) Framework
- **Core**: [stime.py](stime.py) provides discrete event simulation via salabim
- **Key Classes**: `SimulationEnv` (singleton), `Task` (base class for simulation processes), `now()`, `elapse()`
- **Pattern**: All simulation actors inherit from `stime.Task` and use `elapse()` to advance logical time

## Key Conventions & Patterns

### Configuration Management
- **Dataclass-based config** (see [serving_cast/config.py](serving_cast/config.py)): ParallelConfig, ModelConfig, LoadGenConfig, ServingConfig
- **Singleton pattern**: `Config.get_instance()` for global access in ServingCast
- **Device profiles**: Each file in `tensor_cast/device_profiles/` registers a device class with `@register_device`
- **YAML schemas**: ServingCast expects flat YAML with nested dicts for model_config, serving_config, load_gen

### Performance Modeling
- **Analytic model** (TensorCast): `OperatorProfile` + hardware specs → execution time (no ML predictor needed for simulation)
- **Memory tracking**: `MemoryTracker` tracks peak & total allocations per operator via PyTorch hooks
- **Quantization configs**: W8A8_DYNAMIC, W4A8_STATIC, MXFP4 - applied uniformly to linear layers unless disabled
- **Parallelism annotations**: Tensor/Sequence/Data parallelism stored in `model_config.parallel_config`; tensor ops use `parallel_group.py` to compute local shapes

### Code Organization
- **tensor_cast/layers/**: Custom layer implementations (attention adapters, MoE, parallel embedding/linear)
- **tensor_cast/core/**: Config resolution, model building, input generation, user-facing API
- **tensor_cast/compilation/passes/**: Graph optimization (pattern matching in `patterns/`, base class in `pass_base.py`)
- **serving_cast/service/**: Backend abstraction (MindIE aggregation/disaggregation modes)
- **serving_cast/profiler/**: Optional profiling hooks (stime-based instrumentation)

## Developer Workflows

### Environment Setup
- **Python Environment**: Project uses local conda environment at `.conda/` (Python 3.11.14)
- **Windows PowerShell**: Use `conda run -p .\.conda --no-capture-output python <script>`
- **Linux/Mac**: Activate with `conda activate .conda` or use standard `python -m` commands
- **Dependencies**: Install via `pip install -r tensor_cast/requirements.txt` (torch, transformers, diffusers, etc.)

### Running TensorCast Simulations
```bash
# Prefill simulation (3500 tokens input, 2 requests, TEST_DEVICE)
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 2 --query-length 3500 --device TEST_DEVICE

# With quantization (W8A8 dynamic)
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 2 --query-length 3500 --device TEST_DEVICE \
  --quantize-linear-action W8A8_DYNAMIC

# Decode simulation (1 token, 4500-token context)
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 10 --query-length 1 --context-length 4500 --device TEST_DEVICE

# Real-world example: Kimi-K2 Prefill on 32 A3 Dies (TP=4, DP=8, EP)
python -m tensor_cast.scripts.text_generate moonshotai/Kimi-K2-Instruct \
  --device ATLAS_800_A3_752T_128G_DIE \
  --world-size 32 --tp-size 4 --dp-size 8 --ep \
  --quantize-linear-action W4A8_STATIC \
  --num-queries 18 --query-length 4096 \
  --chrome-trace ./kimi_k2_prefill_trace.json

# Throughput benchmark (search optimal TPOT/TTFT under SLO)
python -m tensor_cast.scripts.benchmark.py --input-length 3500 --output-length 1500
```

### Running ServingCast Simulations
```bash
# Set PYTHONPATH for imports
export PYTHONPATH=/path/to/msmodeling:$PYTHONPATH

# Basic simulation
cd serving_cast
python main.py --instance_config_path example/instances.yaml --common_config_path example/common.yaml

# With profiling
python main.py --instance_config_path example/instances.yaml --common_config_path example/common.yaml \
  --enable_profiling --profiling_output_path ./results/profiling
```

### Testing
- **Unit tests**: `tests/` and `tensor_cast/tests/` use unittest framework
- **Test pattern** (see [tests/test_stime.py](tests/test_stime.py)): setUp/tearDown with `init_simulation()`, mock sim.Component for events
- **Run TensorCast tests**: `cd tensor_cast && python -m pytest tests/`
- **Run ServingCast tests**: Check `serving_cast/tests/run_test.sh`

### Linting & Formatting
- **Black** (line length 88), **ruff** (src: [".", "tensor_cast", "tests"]), **isort** (3-line imports)
- Run: `ruff check . && black . && isort .`

## Integration Points & Dependencies

### External Libraries
- **transformers/diffusers**: HuggingFace model loading (TensorCast wraps in `TransformerModel`)
- **salabim**: Discrete event simulation backend (wrapped by stime)
- **torch**: PyTorch—intercepted via custom compilation backend in `tensor_cast/compilation/compile_backend.py`

### Cross-Component Communication
- **TensorCast → ServingCast**: Reuses model loading, device profiles, quantization configs (import shared modules)
- **Operator simulation**: `tensor_cast/performance_model/` used by ServingCast's backend for per-request execution time
- **Request scheduling**: ServingCast's `BatchScheduler` uses TensorCast's `ModelRunner` API indirectly through backend

## Common Issues & Solutions

### Meta Device Quantization Errors
**Issue**: W4A8_STATIC quantization fails with infinite recursion in `tensor.clamp()` on meta device
- **Location**: [tensor_cast/layers/quant_linear.py:160](tensor_cast/layers/quant_linear.py#L160) `pack_int4()` method
- **Root Cause**: PyTorch meta tensors don't support `clamp()` operations (triggers `isnan()` → recursive call chain)
- **Workaround**: Skip clamp on meta device or use W8A8_DYNAMIC quantization instead
- **Fix Pattern**: Add device type check before clamp:
  ```python
  if tensor.device.type != 'meta':
      tensor = tensor.clamp(-8, 7)
  ```

### Model Type Mismatches (Kimi-K2 → DeepSeek)
**Behavior**: Kimi-K2 models show warning about `kimi_k2` → `deepseek_v3` type conversion
- **Expected**: Kimi-K2 architecture is based on DeepSeek-V3, configuration declares `model_type = "deepseek_v3"`
- **Handling**: [transformers/utils.py:287-374](tensor_cast/transformers/utils.py#L287-L374) detects mismatch via `is_model_type_different()`, reloads with native DeepSeek config
- **Benefit**: Enables MoE/MLA optimizations and avoids `trust_remote_code=True`

### Mapping vLLM Deployment to TensorCast Simulation
When translating vLLM serving scripts to TensorCast commands:
- `--data-parallel-size` → `--dp-size`
- `--tensor-parallel-size` → `--tp-size`
- `--enable-expert-parallel` → `--ep`
- `--max-num-seqs` → `--num-queries`
- `--max-num-batched-tokens` (prefill) → `--query-length` approximation
- `--quantization ascend` + W4A8 model → `--quantize-linear-action W4A8_STATIC`
- Prefill instance: `query-length > 1000`, `context-length = 0`
- Decode instance: `query-length = 1`, `context-length > 0`

## Critical Files for New Contributors
- [serving_cast/main.py](serving_cast/main.py) - entry point, CLI args, simulation orchestration
- [serving_cast/engine.py](serving_cast/engine.py) - batch scheduling logic, KV cache preemption
- [tensor_cast/core/model_runner.py](tensor_cast/core/model_runner.py) - inference API, profiling hooks
- [tensor_cast/core/config_resolver.py](tensor_cast/core/config_resolver.py) - config → model transformations
- [tensor_cast/transformers/utils.py](tensor_cast/transformers/utils.py) - model type detection, config loading
- [tensor_cast/layers/quant_linear.py](tensor_cast/layers/quant_linear.py) - quantization implementations (W4A8, W8A8, etc.)
- [stime.py](stime.py) - discrete event simulation primitives (understand `Task`, `elapse()`, `now()` first)
- [serving_cast/config.py](serving_cast/config.py) - expected YAML schema definitions
