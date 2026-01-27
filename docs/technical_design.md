# Technical Design

This document describes the internal architecture and key design decisions of msModeling's two subsystems: **TensorCast** and **ServingCast**.

---

## 1. TensorCast Architecture

### 1.1 Execution Flow

A typical TensorCast simulation follows this pipeline:

```
User CLI (text_generate / benchmark)
  │
  ▼
UserInputConfig          ← CLI args: model ID, device, quantization, parallelism
  │
  ▼
ConfigResolver           ← Resolves HuggingFace model config from Hub or local path
  │
  ▼
ModelBuilder             ← Loads model on meta device (no real weights)
  │                        Applies quantization (QuantLinear layers)
  │                        Applies tensor-parallel sharding (RowParallel / ColParallel)
  │                        Optionally applies torch.compile with custom backend
  │
  ▼
InputGenerator           ← Generates synthetic inputs (token IDs, attention metadata,
  │                        paged KV cache block tables) on meta device
  │
  ▼
Runtime                  ← TorchDispatchMode intercepting every PyTorch op
  │                        Records OpInvokeInfo for each operation
  │                        Delegates to PerformanceModel for cost estimation
  │
  ▼
PerformanceModel         ← Analytic roofline model or empirical model
  │                        Classifies ops as compute-bound / memory-bound / comm-bound
  │
  ▼
Output                   ← Performance summary table
                           Chrome Trace JSON
                           ModelRunnerMetrics dataclass
```

### 1.2 Runtime Interception: TorchDispatchMode

The core simulation mechanism lives in `tensor_cast/runtime.py`. The `Runtime` class extends PyTorch's `TorchDispatchMode`, which intercepts every dispatched operation during model execution:

```python
class Runtime(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 1. Execute the op on meta tensors (no real compute)
        # 2. Record OpInvokeInfo: op name, input/output shapes, dtypes
        # 3. Query PerformanceModel for estimated execution time
        # 4. Track memory allocation/deallocation
        return result
```

This approach means that the actual model code runs unmodified — the simulation is transparent to the model implementation. All tensors live on PyTorch's `meta` device, so no GPU memory or compute is consumed.

### 1.3 Performance Modeling

#### Analytic Roofline Model (`performance_model/analytic.py`)

The primary performance model uses a roofline analysis. For each operation, it computes:

- **FLOPs**: floating-point operations required (e.g., `2 × M × N × K` for matrix multiply).
- **Memory bytes**: total bytes read + written (weights, activations, outputs).
- **Compute time**: `FLOPs / device_peak_ops` (using MMA ops for matmuls, GP ops for element-wise).
- **Memory time**: `bytes / device_memory_bandwidth`.
- **Execution time**: `max(compute_time, memory_time)` — the roofline bound.

Each operation is classified into one of four categories:

| Category | Bottleneck | Typical Operations |
|---|---|---|
| `compute_bound_mma` | Matrix-multiply ALUs | Linear, BMM, attention QK/PV |
| `compute_bound_gp` | General-purpose ALUs | Softmax, GELU, LayerNorm |
| `memory_bound` | HBM bandwidth | Embedding lookup, small activations |
| `communication_bound` | Interconnect bandwidth | AllReduce, AllGather, ReduceScatter |

Hardware efficiency factors (`compute_efficiency`, `memory_efficiency`, `comm_efficiency`) are applied to account for the gap between peak and achievable throughput.

#### Empirical Model (`performance_model/empirical.py`)

An alternative model uses benchmarked operator latencies collected from real hardware. When empirical data is available, it provides a lookup table keyed by operation type and tensor shapes.

#### Communication Model (`performance_model/comm_analytic.py`)

Communication operations (AllReduce, AllGather, AllToAll) are modeled using the device's `CommGrid`, which defines a hierarchical interconnect topology. Bandwidth and latency depend on:

- The number of participating devices.
- Which level of the hierarchy is traversed (e.g., intra-node vs. inter-rack).
- The interconnect type (Full Mesh, Clos network).

#### Memory Tracker (`performance_model/memory_tracker.py`)

Tracks tensor allocations and deallocations throughout the forward pass. Reports:

- **Model weight size** (after quantization and sharding).
- **KV cache size** (per-token and total, based on batch and sequence length).
- **Peak activation memory** (high-water mark during execution).
- **Total device memory** available minus reserved memory.

### 1.4 Compilation and Graph Optimization

TensorCast uses `torch.compile()` with a custom backend (`tensor_cast/compilation/compile_backend.py`) to optimize the FX graph before simulation. The optimization pipeline applies a sequence of passes:

1. **Redundant Node Elimination** — removes unused graph nodes.
2. **Lift and Combine Quantization** (`lift_quant`) — hoists quantization ops and merges adjacent quantize/dequantize pairs.
3. **Pattern Matching Fusion** — recognizes and fuses common subgraphs:
   - **RMSNorm fusion**: replaces separate variance → rsqrt → multiply nodes with a single fused RMSNorm op.
   - **Rotary Embedding fusion**: combines sin/cos lookup and rotation into a fused RoPE op.
4. **Merge Linear** — combines consecutive linear layers that can share the same matmul (e.g., QKV projection).
5. **Sink Split** — pushes split operations downstream to reduce intermediate tensor count.
6. **Constant Folding** — evaluates constant subexpressions at compile time.
7. **Topological Sort** — re-orders the graph for optimal execution ordering.

### 1.5 Model Sharding (Parallelism)

Sharding is managed by `ParallelGroupManager` (`tensor_cast/parallel_group.py`), which defines communication groups for each parallelism strategy. During model construction, layers are replaced with their parallel variants:

- **ColwiseParallelLinear**: partitions the weight matrix along the output dimension. Each device computes a slice of the output. An AllGather collects the full output.
- **RowParallelLinear**: partitions the weight matrix along the input dimension. Each device receives a slice of the input. An AllReduce aggregates partial sums.
- **ParallelEmbedding**: partitions the vocabulary embedding table across devices.
- **MoELayer**: assigns experts to devices with support for redundant experts and external shared experts.

Fine-grained parallelism allows different TP/DP sizes for:
- MLP layers (`mlp_tp_size`, `mlp_dp_size`)
- LM head (`lmhead_tp_size`, `lmhead_dp_size`)
- Output projection (`o_proj_tp_size`, `o_proj_dp_size`)

### 1.6 Quantization Pipeline

Quantization replaces standard `torch.nn.Linear` layers with `QuantLinear` variants that simulate lower-precision arithmetic:

```
Original Linear (FP16/BF16 weights)
  │
  ▼
QuantLinearBase
  ├── Quantize weight to target dtype (INT8/INT4/FP8/FP4)
  ├── Store scale and offset tensors
  ├── For W4A8: pack two INT4 values into one INT8 byte
  └── Forward pass simulates: dequant → matmul → requant
```

Configuration is controlled via `LinearQuantConfig`:
- **Granularity**: per-tensor, per-channel, or per-group.
- **Scheme**: symmetric or asymmetric.
- **Dynamic vs. static**: dynamic computes activation scales at runtime; static uses pre-computed scales.

Attention quantization (`AttentionQuantConfig`) applies INT8 quantization to KV cache storage, reducing memory footprint per token.

### 1.7 Specialized Layer Implementations

#### Multi-Head Latent Attention (MLA)

`tensor_cast/layers/mla.py` implements the MLA mechanism used in DeepSeek models. Instead of standard multi-head attention with separate Q/K/V projections, MLA uses a low-rank latent projection for keys and values, reducing KV cache size.

#### Multi-Token Prediction (MTP)

`tensor_cast/layers/mtp.py` implements speculative decoding where the model predicts multiple tokens per step. Each MTP layer predicts the next token conditioned on the previous prediction. Acceptance rates (configurable per layer, e.g., `[0.9, 0.6, 0.4, 0.2]`) model the probability that each speculated token is correct.

#### Mixture of Experts (MoE)

`tensor_cast/layers/moe_layer.py` implements sparse MoE routing with:
- Top-K expert selection via a learned gate.
- Expert-parallel distribution across devices.
- Support for redundant experts (each device hosts extra experts to reduce communication).
- External shared experts (dedicated devices for a shared expert used by all tokens).

### 1.8 Device Profile System

Hardware is described by the `DeviceProfile` dataclass (`tensor_cast/device.py`):

```python
@dataclass
class DeviceProfile:
    name: str
    vendor: str
    comm_grid: CommGrid              # Hierarchical interconnect topology

    mma_ops: Dict[torch.dtype, float]     # Matrix-multiply peak ops/s per dtype
    gp_ops: Dict[torch.dtype, float]      # General-purpose peak ops/s per dtype

    compute_efficiency: float        # Fraction of peak compute achievable
    memory_size_bytes: float         # Total device memory (bytes)
    memory_bandwidth_bytes_ps: float # Peak memory bandwidth (bytes/s)
    memory_efficiency: float         # Fraction of peak bandwidth achievable

    static_cost: StaticCost          # Per-op scheduling overhead
```

The `CommGrid` represents a multi-level device topology. For example, 64 devices arranged as `[4, 8, 2]` means 2 devices per MCP pair (fastest link), 8 MCP pairs per node, and 4 nodes in the cluster. Each level maps to an `InterconnectTopology` with its own bandwidth, latency, and type (Full Mesh or Clos).

Custom device profiles are Python files placed in `tensor_cast/device_profiles/`. They are loaded automatically at import time.

---

## 2. ServingCast Architecture

### 2.1 Discrete-Event Simulation

ServingCast uses `salabim`, a discrete-event simulation library, through the `stime.py` module. The simulation advances logical time (not wall-clock time) as events occur:

- **`stime.elapse(duration)`** — advances simulation time by `duration` seconds.
- **`stime.Duration` context manager** — tracks elapsed time within a block.
- **`stime.Task`** — a coroutine-like abstraction for concurrent simulation actors.

This allows the simulation to model thousands of requests and multi-instance coordination in seconds of real time.

### 2.2 Component Architecture

```
main.py
  │
  ├── CommonConfig (YAML)          ← model, load generation, serving parameters
  ├── InstanceConfig (YAML)        ← device pools, parallelism, communication
  │
  ▼
Serving Layer
  ├── PdAggregation               ← Single-pool: prefill + decode on same instances
  └── PdDisaggregation            ← Separate prefill and decode instance pools
      │
      ▼
  Instance (per compute node)
  ├── ModelRunner                  ← Wraps TensorCast for latency prediction
  ├── BatchScheduler (Engine)     ← Continuous batching with token budget
  ├── KVCacheManager              ← Block-based paged KV cache allocation
  └── CommunicationManager        ← Inter-instance KV cache transfer
      │
      ▼
  LoadGenerator                    ← Generates requests at configured rate
  ├── FixedLengthLoadGen          ← Fixed input/output token lengths
  └── (extensible)
      │
      ▼
  Request Lifecycle
  ├── WAITING → PREFILLING → DECODING → FINISHED
  └── Metrics: E2E_TIME, TTFT, TPOT, throughput
```

### 2.3 Batch Scheduling

The `BatchScheduler` (in `serving_cast/engine.py`) implements continuous batching:

1. **Prefill phase**: New requests are batched up to `max_tokens_budget` tokens per batch. The prefill latency is obtained from TensorCast.
2. **Decode phase**: After prefill, requests join the decode pool. Each decode step generates one token per request (or multiple with MTP). Requests are batched up to `max_concurrency`.
3. **Preemption**: When memory is exhausted, lower-priority requests can be preempted to free KV cache blocks.

### 2.4 KV Cache Management

`KVCacheManager` manages a block-based paged attention cache:

- **Block size**: configurable (default 128 tokens per block).
- **Allocation**: blocks are allocated as sequences grow during decode.
- **Deallocation**: blocks are freed when requests complete.
- **Transfer**: in P/D disaggregation mode, KV cache blocks are transferred between prefill and decode instances through the `CommunicationManager`, which models transfer latency based on configured bandwidth.

### 2.5 Service Backends

The `BackendFactory` creates backend implementations for different serving frameworks:

- **MindIE Backend** (`service/mindie_backend.py`): models the scheduling and execution behavior of the MindIE inference engine.
- **Base Backend** (`service/base_backend.py`): abstract interface for adding new backends.

### 2.6 Output Metrics

After simulation completes, `ReportAndSave` collects per-request metrics and computes summary statistics:

| Metric | Description |
|---|---|
| E2E_TIME | Total latency from request issue to last token |
| TTFT | Time to first token (includes queuing + prefill) |
| TPOT | Average time per output token after the first |
| OUTPUT_TOKEN_THROUGHPUT | Per-request output token rate |
| request_throughput | System-wide requests per second |
| input_token_throughput | Aggregate input tokens per second |
| output_token_throughput | Aggregate output tokens per second |

Statistics are reported at AVERAGE, MIN, MAX, MEDIAN, P75, P90, and P99 percentiles.

---

## 3. Cross-Cutting Concerns

### 3.1 Simulation Time (`stime.py`)

Both TensorCast and ServingCast share the simulation time library. It provides:

- A singleton `salabim.Environment` for the event loop.
- `Task` class for spawning concurrent simulation actors.
- `elapse()` for advancing logical time.
- `Duration` context manager for measuring simulated durations.
- `DurationDecorator` for wrapping entire functions with timing.
- Logging integration that includes simulation timestamps.

### 3.2 Custom PyTorch Operations

TensorCast registers custom ops under the `torch.ops.tensor_cast` namespace (in `tensor_cast/ops/`). These ops have no real implementation — they exist to provide symbolic shapes and allow the performance model to recognize and cost specific operations:

- `tensor_cast.attention` / `tensor_cast.attention_quant` — attention with/without quantization.
- `tensor_cast.reshape_and_cache` — KV cache update.
- `tensor_cast.quantize` / `tensor_cast.dequantize` — quantization round-trip.
- `tensor_cast.linear_quant` — fused quantized linear.
- `tensor_cast.all_reduce` / `tensor_cast.all_gather` — communication ops.
- `tensor_cast.rms_norm` — fused RMS normalization.
- `tensor_cast.fused_moe` — fused mixture-of-experts.

### 3.3 Configuration Resolution

`ConfigResolver` (`tensor_cast/core/config_resolver.py`) loads model configurations from multiple sources:

1. HuggingFace Hub (via `transformers.AutoConfig`).
2. ModelScope Hub (Alibaba's model registry).
3. Local file paths.
4. Compressed-tensors metadata (for pre-quantized models).

It also handles model-specific adaptations (e.g., DeepSeek MLA parameters, Qwen3 MoE routing configuration).
