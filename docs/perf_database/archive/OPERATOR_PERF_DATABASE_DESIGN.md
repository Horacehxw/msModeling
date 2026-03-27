# Operator Performance Database for TensorCast: Technical Design Document (UNDER REVIEW !!!!!)

**Version**: 1.0
**Date**: February 2026
**Scope**: Extensible profiling-based cost model for LLM simulation on Huawei Ascend A3
**Initial Target Models**: DeepSeek-V3, Qwen3-32B

---

## 1. Technical Analysis

### 1.1 Problem Statement

TensorCast currently uses a **roofline-based analytical model** (`AnalyticPerformanceModel`) to estimate operator execution time. This model computes `max(compute_time, memory_time)` based on FLOPs and memory access bytes, which produces systematic errors when compared to actual VLLM-Ascend profiling on Atlas 800 A3 hardware.

**Root causes of inaccuracy**:

| Gap Source | Description | Impact |
|-----------|-------------|--------|
| **Kernel fusion** | VLLM fuses multiple ops into single kernels (e.g., `DequantSwigluQuant`, `AddRmsNorm`) | 10-30% time difference |
| **Hardware utilization** | Actual cube utilization ~44-68%, not 100% as roofline assumes | Overestimates compute-bound ops |
| **Shape-dependent overhead** | Small-batch kernel launch overhead not modeled | Underestimates decode phase |
| **Version-dependent optimization** | VLLM-Ascend versions add new fused kernels | Model becomes stale across versions |

### 1.2 Profiling Data Analysis

From actual Ascend profiler output (`kernel_details.csv`, `op_statistic.csv`):

**Qwen3-32B (41 unique operators)**:
| Top Operators | Time % | Core Type |
|--------------|--------|-----------|
| MatMulV2 | 42.4% | AI_CORE |
| FusedInferAttentionScore | 18.2% | MIX_AIC |
| TensorMove | 10.7% | AI_VECTOR_CORE |
| AddRmsNorm | 8.0% | AI_VECTOR_CORE |
| split_qkv_rmsnorm_rope_kernel | 5.2% | MIX_AIC |
| SwiGlu | 4.8% | AI_VECTOR_CORE |

**DeepSeek-V3 (39 unique operators, MoE-specific)**:
| Top Operators | Time % | Core Type |
|--------------|--------|-----------|
| GroupedMatmul | 20.9% | AI_CORE |
| FusedInferAttentionScore | 18.5% | MIX_AIC |
| QuantBatchMatmulV3 | 16.9% | AI_CORE |
| MoeDistributeDispatch/Combine | 11.8% | MIX_AIC |
| AscendQuantV2 | 4.4% | AI_VECTOR_CORE |
| TransposeBatchMatMul | 4.2% | AI_CORE |

**Key finding**: ~15 operators account for >95% of execution time. The remaining ~65 operators (arithmetic, indexing, logic) contribute <5% and can use roofline fallback.

### 1.3 Version Impact Analysis

VLLM-Ascend versions significantly affect which fused kernels exist and their performance:

| Version | New Fused Kernels | Performance Impact |
|---------|------------------|--------------------|
| v0.14.0 | Triton rope, MatMul-AllReduce-RMSNorm fusion | New fusion patterns |
| v0.12.0 | AddRmsNormQuant, many triton kernels | Different op graph |
| v0.10.x | MLP TP, allgather-expert fusion | Expert routing changes |

CANN versions also affect kernel efficiency (different compiler optimizations). Both must be tracked.

### 1.4 Existing Infrastructure

**`msmodeling-profiling_compare`** (already built):
- Fusion-aware operator mapping (YAML config, 100+ mappings)
- VLLM kernel_details.csv parser with phase detection
- TensorCast simulation adapter
- Excel comparison reports
- Location: `tensor_cast/scripts/profiling_comparison/`

**`EmpiricalPerformanceModel`** (exists but for JIT benchmarks):
- `tensor_cast/performance_model/empirical.py` - runs benchmarks on actual device
- Uses `OpBenchmark` class
- Only works with device physically present

### 1.5 AI Configurator Reference (NVIDIA)

Key design patterns adapted from `/home/horacehxw/Projects/aiconfigurator`:
- **Nested dict indexing**: O(1) exact lookup for profiled shapes
- **scipy.griddata interpolation**: Multi-dimensional shape matching
- **DatabaseMode fallback**: SILICON → HYBRID → EMPIRICAL → SOL
- **PerformanceResult(float)**: Backward-compatible result type
- **CSV storage**: Human-readable, version-controllable
- **Lazy loading with caching**: Module-level cache avoids reloading

---

## 2. Architecture

### 2.1 System Architecture

* TensorCast Runtime 通过 `PerformanceModel` 插件化架构接入 Profiling 数据库，选择哪种建模应该是可以配置的，都必须支持。


```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           TensorCast Runtime                                  │
│                                                                               │
│  ┌────────────────────┐     ┌─────────────────────────────────────────────┐  │
│  │  Runtime            │     │  ProfilingPerformanceModel (NEW)            │  │
│  │  (TorchDispatchMode)│────▶│  ┌─────────────────────────────────────┐   │  │
│  │                     │     │  │ 1. Match op → OperatorSchema        │   │  │
│  │  Intercepts all ops │     │  │ 2. Extract shape from OpInvokeInfo  │   │  │
│  │  Creates OpInvokeInfo│     │  │ 3. Query PerfDatabase              │   │  │
│  └────────────────────┘     │  │ 4. Fallback to AnalyticModel        │   │  │
│                              │  └─────────────────────────────────────┘   │  │
│                              └─────────────────────────────────────────────┘  │
│                                         │                                     │
└─────────────────────────────────────────┼─────────────────────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     ▼                     │
                    │              PerfDatabase                 │
                    │  ┌──────────────────────────────────┐    │
                    │  │  QueryEngine                      │    │
                    │  │  ┌──────────────────────────────┐ │    │
                    │  │  │ Exact Match → Interpolate    │ │    │
                    │  │  │ → Extrapolate → Roofline     │ │    │
                    │  │  └──────────────────────────────┘ │    │
                    │  └──────────────────────────────────┘    │
                    │                     │                     │
                    │  ┌──────────────────┴──────────────────┐ │
                    │  │ OperatorSchema Registry              │ │
                    │  │ ┌──────┐ ┌─────────┐ ┌───┐ ┌─────┐ │ │
                    │  │ │ GEMM │ │Attention│ │MoE│ │Fused│ │ │
                    │  │ └──────┘ └─────────┘ └───┘ └─────┘ │ │
                    │  └─────────────────────────────────────┘ │
                    │                     │                     │
                    │  ┌──────────────────┴──────────────────┐ │
                    │  │ Storage (Parquet/CSV per version)    │ │
                    │  │ atlas_a3/vllm_ascend/0.14.0/        │ │
                    │  │   gemm.parquet, attention.parquet    │ │
                    │  └─────────────────────────────────────┘ │
                    └──────────────────────────────────────────┘

Data Collection Pipeline (Offline)
┌──────────────────────────────────────────────────────────────┐
│  Level 1: Full-Model Profiling                                │
│  VLLM serve + bench → kernel_details.csv → parse → database  │
├──────────────────────────────────────────────────────────────┤
│  Level 2: Isolated Microbenchmarks                            │
│  torch_npu scripts → direct measurement → database           │
├──────────────────────────────────────────────────────────────┤
│  Calibration: Level 1 ground truth validates Level 2 data     │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Module Structure

```
tensor_cast/perf_database/              # NEW package
├── __init__.py
├── core/
│   ├── database.py                     # PerfDatabase main class
│   ├── query.py                        # QueryEngine with interpolation
│   ├── storage.py                      # Parquet/CSV IO backend
│   └── versioning.py                   # Version resolution
├── operators/
│   ├── __init__.py                     # Auto-discovery via importlib
│   ├── base.py                         # OperatorSchema ABC + registry
│   ├── gemm.py                         # GEMM/MatMul schema
│   ├── attention.py                    # MHA, GQA, MLA schema
│   ├── moe.py                          # MoE routing + expert compute
│   ├── communication.py                # allreduce, allgather, alltoall
│   ├── normalization.py                # RMSNorm, AddRmsNorm
│   ├── fused.py                        # DequantSwigluQuant, etc.
│   └── elementwise.py                  # Cast, activation, arithmetic
├── shape_generators/
│   ├── base.py                         # ShapeGridGenerator ABC
│   ├── model_driven.py                 # Extract from HuggingFace configs
│   └── universal.py                    # Power-of-2 + common patterns
├── interpolation/
│   ├── base.py                         # InterpolationStrategy ABC
│   └── linear.py                       # scipy LinearNDInterpolator
├── mappings/vllm_ascend/
│   ├── v0.10.yaml                      # VLLM kernel → schema mappings
│   ├── v0.12.yaml
│   └── v0.14.yaml
├── data/systems/                       # Profiling data (gitignored)
│   └── ATLAS_800_A3_752T_128G_DIE/
│       └── vllm_ascend/{version}/
│           ├── metadata.yaml
│           ├── gemm.parquet
│           ├── attention.parquet
│           └── ...
└── scripts/
    ├── collect_full_model.py           # Level 1: VLLM-based profiling
    ├── collect_microbench.py           # Level 2: torch_npu benchmarks
    ├── parse_ascend_output.py          # Parse kernel_details.csv
    ├── generate_shape_grid.py          # Generate shape grids from models
    ├── calibrate.py                    # Calibrate L2 against L1
    └── validate.py                     # Validate database accuracy

tensor_cast/performance_model/
└── profiling.py                        # ProfilingPerformanceModel (NEW)
```

---

## 3. Key Module Design

### 3.1 OperatorSchema (Plugin Architecture)

Each operator type defines its shape space, extraction logic, and roofline fallback:

```python
# tensor_cast/perf_database/operators/base.py

class DimensionType(Enum):
    BATCH = auto()      # Varies with deployment (batch_size, num_tokens)
    SEQUENCE = auto()   # Varies with input (query_len, context_len)
    FEATURE = auto()    # Fixed per model (hidden_size, head_dim)
    EXPERT = auto()     # MoE-specific (num_experts, top_k)
    DEVICE = auto()     # Parallelism-dependent (num_devices)

@dataclass
class DimensionSpec:
    name: str
    dim_type: DimensionType
    typical_range: Tuple[int, int]
    typical_values: Optional[List[int]] = None
    is_required: bool = True

class OperatorSchema(ABC):
    """
    Base class for operator performance schemas.
    Auto-discovered via registry pattern (similar to DeviceProfile).
    """
    _registry: ClassVar[Dict[str, Type["OperatorSchema"]]] = {}

    @classmethod
    def register(cls, op_name: str):
        """Decorator to register operator schema"""
        def decorator(schema_cls):
            cls._registry[op_name] = schema_cls
            return schema_cls
        return decorator

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def dimensions(self) -> List[DimensionSpec]: ...

    @property
    @abstractmethod
    def tensorcast_ops(self) -> List[str]:
        """TensorCast operation names this schema handles"""
        ...

    @abstractmethod
    def extract_shape_from_op(self, op_invoke_info: OpInvokeInfo) -> Dict[str, Any]:
        """Extract shape dimensions from intercepted operation"""
        ...

    @abstractmethod
    def compute_roofline_estimate(self, shape, device_profile) -> float:
        """Compute roofline (speed-of-light) estimate in microseconds"""
        ...

    def get_interpolation_dimensions(self) -> List[str]:
        """Dimensions suitable for continuous interpolation"""
        return [d.name for d in self.dimensions
                if d.dim_type in (DimensionType.BATCH, DimensionType.SEQUENCE)]
```

**Concrete schemas**:

| Schema | Dimensions | Key TensorCast Ops |
|--------|-----------|-------------------|
| `GEMMSchema` | m, n, k, quant_mode | `aten.mm`, `static_quant_linear`, `fp8_linear`, `grouped_matmul_quant` |
| `AttentionSchema` | batch, query_len, context_len, num_heads, num_kv_heads, head_dim, kv_lora_rank | `attention`, `attention_quant`, `multihead_latent_attention` |
| `MoESchema` | num_tokens, num_experts, top_k, hidden_size, intermediate_size | `grouped_matmul`, `init_routing_v2`, `unpermute_tokens` |
| `NormalizationSchema` | num_tokens, hidden_size | `rms_norm`, `add_rms_norm` |
| `FusedSchema` | (varies by fused kernel) | Maps to component ops |
| `CommunicationSchema` | op_type, num_devices, message_size | `all_reduce`, `all_gather`, `all_to_all` |

### 3.2 PerfDatabase

```python
# tensor_cast/perf_database/core/database.py

class PerfDatabase:
    """Main database class. Loads profiling data, provides query interface."""

    def __init__(
        self,
        system: str = "ATLAS_800_A3_752T_128G_DIE",
        backend: str = "vllm_ascend",
        version: str = "latest",
        data_root: Optional[Path] = None,
    ):
        self.version_mgr = VersionManager(data_root)
        self.resolved_version, self.data_path = self.version_mgr.resolve(system, backend, version)
        self.query_engine = QueryEngine(self)
        self._data_cache: Dict[str, pd.DataFrame] = {}

    def query(self, schema_name: str, shape: Dict, mode: QueryMode = QueryMode.HYBRID) -> QueryResult:
        """Main query entry point. Delegates to QueryEngine."""
        return self.query_engine.query(schema_name, shape, mode)

    def get_data(self, schema_name: str, quant_mode: str) -> Optional[pd.DataFrame]:
        """Lazy-load and cache operator data from Parquet"""
        cache_key = f"{schema_name}_{quant_mode}"
        if cache_key not in self._data_cache:
            path = self.data_path / f"{schema_name}.parquet"
            if not path.exists():
                return None
            df = pd.read_parquet(path)
            self._data_cache[cache_key] = df[df["quant_mode"] == quant_mode]
        return self._data_cache[cache_key]
```

### 3.3 QueryEngine (Interpolation + Fallback)

```python
# tensor_cast/perf_database/core/query.py

class QueryMode(Enum):
    EXACT = auto()
    INTERPOLATE = auto()
    HYBRID = auto()       # Exact → Interpolate → Roofline
    ROOFLINE = auto()

class QuerySource(Enum):
    MEASURED = auto()          # Exact profiled data (confidence: 1.0)
    INTERPOLATED = auto()      # scipy interpolation (confidence: 0.7-0.95)
    EXTRAPOLATED = auto()      # Outside convex hull (confidence: 0.3-0.6)
    ROOFLINE = auto()          # Analytical model (confidence: 0.4)
    ROOFLINE_CALIBRATED = auto()  # Roofline × efficiency factor (confidence: 0.6)

@dataclass
class QueryResult:
    latency_us: float
    confidence: float
    source: QuerySource
    details: Dict[str, Any] = field(default_factory=dict)

class QueryEngine:
    """Handles exact lookup, interpolation, and fallback strategies."""

    def query(self, schema_name, shape, mode=QueryMode.HYBRID) -> QueryResult:
        schema = OperatorSchema.get_schema(schema_name)

        if mode == QueryMode.HYBRID:
            # Try in order: exact → interpolate → roofline
            result = self._try_exact(schema, shape)
            if result: return result

            result = self._try_interpolate(schema, shape)
            if result: return result

            return self._roofline_fallback(schema, shape)

    def _try_interpolate(self, schema, shape) -> Optional[QueryResult]:
        """scipy.interpolate.LinearNDInterpolator over measured points"""
        # Get continuous dimensions for interpolation
        interp_dims = schema.get_interpolation_dimensions()
        # Build point cloud from measured data
        # Interpolate, compute confidence from distance to nearest points
        ...
```

### 3.4 Operator-to-Schema Mapping (YAML)

```yaml
# tensor_cast/perf_database/mappings/vllm_ascend/v0.14.yaml
version: "0.14"
device: ATLAS_800_A3_752T_128G_DIE

operator_schemas:
  gemm:
    vllm_kernels:
      - name: "MatMulV2"
        shape_parser: "matmul_v2"
      - name: "GroupedMatmul"
        shape_parser: "grouped_matmul"
      - name: "QuantBatchMatmulV3"
        shape_parser: "quant_batch_matmul"
      - name: "TransposeBatchMatMul"
        shape_parser: "transpose_batch_matmul"
    tensorcast_ops:
      - "aten.mm.default"
      - "tensor_cast.static_quant_linear.default"
      - "tensor_cast.static_quant_linear_int4.default"
      - "tensor_cast.fp8_linear.default"
      - "tensor_cast.grouped_matmul_quant.default"

  attention:
    vllm_kernels:
      - name: "FusedInferAttentionScore"
        shape_parser: "fused_attention"
    tensorcast_ops:
      - "tensor_cast.attention.default"
      - "tensor_cast.attention_quant.default"
      - "tensor_cast.multihead_latent_attention.default"
      - "tensor_cast.multihead_latent_attention_quant.default"

  moe_routing:
    vllm_kernels:
      - name: "MoeDistributeDispatchV2"
      - name: "MoeDistributeCombineV2"
      - name: "MoeGatingTopK"
    tensorcast_ops:
      - "tensor_cast.init_routing_v2.default"
      - "tensor_cast.unpermute_tokens.default"

  fused:
    vllm_kernels:
      - name: "DequantSwigluQuant"
        components: [dequantize, silu, mul, quantize]
      - name: "AddRmsNorm"
        components: [add, rms_norm]
      - name: "InplaceAddRmsNorm"
        components: [add, rms_norm]
      - name: "split_qkv_rmsnorm_rope_kernel"
        components: [rms_norm, qkv_split, apply_rope]
      - name: "KvRmsNormRopeCache"
        components: [rms_norm, apply_rope, reshape_and_cache]
```

---

## 4. Interface Design: TensorCast Integration

### 4.1 ProfilingPerformanceModel

This is the main integration point. It extends `PerformanceModel` (the same abstract class used by `AnalyticPerformanceModel` and `EmpiricalPerformanceModel`).

```python
# tensor_cast/performance_model/profiling.py

class ProfilingPerformanceModel(PerformanceModel):
    """
    Performance model using profiled operator data.

    Integration with TensorCast:
    - Extends PerformanceModel (tensor_cast/performance_model/__init__.py:169)
    - Implements process_op(OpInvokeInfo) → PerformanceModel.Result
    - Plugs into Runtime (tensor_cast/runtime.py:41) via perf_models list
    - Wrapped in CachingPerformanceModel automatically by Runtime
    - Supports get_classifiers() for bound classification
    """

    def __init__(
        self,
        device_profile: DeviceProfile,            # From tensor_cast/device.py
        database_mode: QueryMode = QueryMode.HYBRID,
        vllm_version: str = "latest",
    ):
        super().__init__("profiling", device_profile)  # name="profiling"
        self.db = PerfDatabase(
            system=self._device_to_system(device_profile),
            version=vllm_version,
        )
        self.query_mode = database_mode
        self.fallback = AnalyticPerformanceModel(device_profile)  # Existing model
        self._schema_cache: Dict[str, Optional[OperatorSchema]] = {}

    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        """
        Core interface method.
        Called by Runtime.__torch_dispatch__ for each intercepted operation.

        Args:
            op_invoke_info: Contains func, args, kwargs, out, cache_key
                           (defined in tensor_cast/performance_model/__init__.py:52)

        Returns:
            PerformanceModel.Result with execution_time_s and statistics dict
            (defined in tensor_cast/performance_model/__init__.py:175)
        """
        schema = self._get_schema(op_invoke_info.func)

        if schema is not None:
            try:
                shape = schema.extract_shape_from_op(op_invoke_info)
                query_result = self.db.query(schema.name, shape, self.query_mode)
                return PerformanceModel.Result(
                    execution_time_s=query_result.latency_us * 1e-6,
                    statistics={
                        "source": query_result.source.name,
                        "confidence": query_result.confidence,
                    }
                )
            except Exception:
                pass  # Fall through to analytic

        return self.fallback.process_op(op_invoke_info)

    def get_classifiers(self) -> List[PerformanceModel.OpClassifier]:
        return self.fallback.get_classifiers()
```

### 4.2 Runtime Integration

No changes needed to `Runtime` class. It already accepts a list of `PerformanceModel` instances:

```python
# tensor_cast/runtime.py:41-56 (EXISTING, NO CHANGES)
class Runtime(TorchDispatchMode):
    def __init__(
        self,
        perf_models: Union[PerformanceModel, List[PerformanceModel]],
        device_profile: DeviceProfile,
        memory_tracker: Optional[MemoryTracker] = None,
    ):
        self.perf_models = [
            CachingPerformanceModel(m) if not isinstance(m, CachingPerformanceModel) else m
            for m in (perf_models if isinstance(perf_models, list) else [perf_models])
        ]
```

To use profiling model, simply pass it as the perf_model:
```python
# In model_runner.py or config_resolver.py:
perf_model = ProfilingPerformanceModel(device_profile, vllm_version="0.14.0")
runtime = Runtime(perf_models=perf_model, device_profile=device_profile)
```

### 4.3 CLI Interface

Add to `tensor_cast/scripts/text_generate.py`:

```python
parser.add_argument("--performance-model", choices=["analytic", "profiling", "empirical"],
                    default="analytic", help="Performance model type")
parser.add_argument("--database-mode", choices=["exact", "interpolate", "hybrid", "roofline"],
                    default="hybrid", help="Database query mode")
parser.add_argument("--database-version", type=str, default="latest",
                    help="Profiling database version")
```

### 4.4 Data Flow

```
User CLI                    TensorCast                    Database
─────────                   ──────────                    ────────
text_generate.py
  --performance-model profiling
  --database-version 0.14.0
       │
       ▼
  ModelRunner creates
  ProfilingPerformanceModel
       │
       ▼
  Runtime.__torch_dispatch__
  intercepts each op
       │
       ▼
  OpInvokeInfo(func, args, kwargs, out)
       │
       ▼
  ProfilingPerformanceModel.process_op()
       │
       ├──▶ Match func to OperatorSchema
       │    (e.g., aten.mm → GEMMSchema)
       │
       ├──▶ schema.extract_shape_from_op()
       │    → {m: 136, n: 4096, k: 5120, quant: "int8"}
       │
       ├──▶ db.query("gemm", shape, mode=HYBRID)
       │         │
       │         ├──▶ Exact match in gemm.parquet? → Return
       │         ├──▶ Interpolate from nearby points? → Return
       │         └──▶ Roofline fallback → Return
       │
       └──▶ Return PerformanceModel.Result(execution_time_s=...)
```

---

## 5. Automated Profiling Pipeline

### 5.1 Two-Level Strategy

#### Level 1: Full-Model Profiling (Ground Truth)

Runs VLLM inference with Ascend profiler enabled. Captures real fused kernel timing.

**Trigger mechanism** (from actual profiling scripts):
```bash
# Environment setup
export VLLM_TORCH_PROFILER_DIR=/path/to/output
export PROFILING_SAVE_PATH=/path/to/output

# Start VLLM server (expensive: model loading ~minutes)
vllm serve $MODEL --tensor-parallel-size $TP --dtype bfloat16 ...

# Run benchmarks with controlled shapes (cheap: seconds each)
vllm bench serve --profile \
    --dataset-name random \
    --random-input-len $INPUT_LEN \
    --random-output-len $OUTPUT_LEN \
    --max-concurrency $BATCH_SIZE
```

**Efficiency**: Start server ONCE, send multiple benchmark rounds with different shapes.

**Shape sweep per server instance**:
```python
# Decode scenarios (query_len=1)
for batch in [1, 8, 16, 32, 64, 128, 256]:
    bench(input_len=1, output_len=1, concurrency=batch)

# Prefill scenarios
for isl in [256, 512, 1024, 2048, 4096, 8192]:
    for batch in [1, 8, 32, 64, 128]:
        bench(input_len=isl, output_len=1, concurrency=batch)
```

**Server configurations requiring separate instances**:
```python
SERVER_CONFIGS = [
    {"model": "Qwen3-32B", "tp": 4, "quant": "none"},
    {"model": "Qwen3-32B", "tp": 8, "quant": "W8A8"},
    {"model": "Qwen3-32B", "tp": 16, "quant": "W4A8"},
    {"model": "DeepSeek-V3", "tp": 4, "dp": 8, "ep": True, "quant": "W8A8"},
    {"model": "DeepSeek-V3", "tp": 4, "dp": 8, "ep": True, "quant": "W4A8"},
]
```

**Output parsing**:
```python
# Parse ASCEND_PROFILER_OUTPUT/kernel_details.csv
# Columns: Name, Duration(us), Input Shapes, cube_utilization(%)
# Group by step (use FusedInferAttentionScore as boundary)
# Extract per-operator aggregate timing for each shape config
```

#### Level 2: Isolated Microbenchmarks (Shape Grid Fill)

Direct `torch_npu` kernel benchmarks for fine-grained shape coverage:

```python
# scripts/collect_microbench.py
import torch, torch_npu

def benchmark_matmul(m, n, k, dtype, warmup=5, runs=20):
    a = torch.randn(m, k, dtype=dtype, device='npu')
    b = torch.randn(k, n, dtype=dtype, device='npu')
    # Warmup
    for _ in range(warmup):
        torch.mm(a, b)
    torch.npu.synchronize()
    # Measure
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()
    for _ in range(runs):
        torch.mm(a, b)
    end_event.record()
    torch.npu.synchronize()
    return start_event.elapsed_time(end_event) / runs * 1000  # microseconds
```

#### Level 1 vs Level 2 Comparison

| Aspect | Level 1 (Full-Model) | Level 2 (Microbench) |
|--------|---------------------|---------------------|
| Captures fused kernels | Yes (ground truth) | Partial |
| Shape control | Indirect (batch, seq_len) | Direct (m, n, k) |
| Overhead | High (server startup) | Low (kernel only) |
| Coverage | ~100 configs per model | ~2500 shapes per operator |
| Use case | Validation + calibration | Shape grid population |

**Calibration**: Compare Level 1 and Level 2 data for overlapping shapes. Compute per-operator correction factor to align Level 2 with real deployment.

### 5.2 Profiling Output Parser

```python
# scripts/parse_ascend_output.py

class AscendProfilerParser:
    def parse_kernel_details(self, csv_path: Path) -> pd.DataFrame:
        """Parse kernel_details.csv → structured DataFrame"""
        # Handles flexible column names, missing fields
        # Returns: name, duration_us, input_shapes, data_types,
        #          aicore_time_us, aiv_time_us, cube_utilization_pct

    def extract_single_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract one decode/prefill step using attention kernel as boundary"""
        # Reuses logic from profiling_compare: kernel_details_parser.py

    def map_to_schema(self, kernel_name: str, input_shapes: str) -> Tuple[str, Dict]:
        """Map VLLM kernel → (schema_name, shape_dict)"""
        # "MatMulV2" + "136,4096; 4096,4096" → ("gemm", {m:136, k:4096, n:4096})
        # Uses version-specific YAML mappings
```

### 5.3 Operator Discovery for New Models

```python
# scripts/discover_operators.py

def discover_operators(profiling_output: Path, mapping_yaml: Path) -> Dict:
    """Discover operators in profiling that aren't in current mappings"""
    parser = AscendProfilerParser()
    kernels = parser.parse_kernel_details(profiling_output / "kernel_details.csv")

    known_ops = load_yaml_mappings(mapping_yaml)
    unknown = []

    for kernel_name in kernels["name"].unique():
        if not any(kernel_name in schema_kernels for schema_kernels in known_ops.values()):
            # Auto-classify by name pattern
            suggested_schema = auto_classify(kernel_name)
            unknown.append({"kernel": kernel_name, "suggested_schema": suggested_schema})

    return {"known": len(kernels) - len(unknown), "unknown": unknown}
```

---

## 6. Comprehensive Operator Coverage

### 6.1 Operator Tiers

| Tier | Criteria | Action | Count |
|------|----------|--------|-------|
| **Tier 1** | >2% execution time | MUST profile with full shape grid | ~11 operators |
| **Tier 2** | 0.5-2% execution time | Profile with reduced shape grid | ~8 operators |
| **Tier 3** | <0.5% execution time | Use roofline fallback | ~60+ operators |

### 6.2 Complete Tier 1 Operator List

| VLLM Kernel | DB Schema | Qwen3 % | DSV3 % |
|-------------|-----------|---------|--------|
| MatMulV2 | gemm | 42.4% | - |
| GroupedMatmul | gemm | - | 20.9% |
| QuantBatchMatmulV3 | gemm | - | 16.9% |
| FusedInferAttentionScore | attention | 18.2% | 18.5% |
| TensorMove | memory_move | 10.7% | ~3% |
| AddRmsNorm/InplaceAddRmsNorm | normalization | 8.0% | 2.0% |
| split_qkv_rmsnorm_rope_kernel | fused | 5.2% | - |
| SwiGlu/DequantSwigluQuant | fused | 4.8% | 2.7% |
| ReshapeAndCacheNdKernel | cache | 3.3% | - |
| MoeDistributeDispatch/Combine | moe_routing | - | 11.8% |
| AscendQuantV2/DynamicQuant | quantization | - | 5.5% |
| TransposeBatchMatMul | gemm | - | 4.2% |
| InterleaveRope | rope | - | 2.8% |

### 6.3 Shape Grid (Hybrid: Model-Driven + Generic)

**Step 1**: Extract dimensions from HuggingFace model configs:
```python
PRIORITY_MODELS = [
    "Qwen/Qwen3-32B", "Qwen/Qwen3-235B-A22B",
    "deepseek-ai/DeepSeek-V3", "moonshotai/Kimi-K2-Instruct",
    "meta-llama/Llama-3.1-70B", "meta-llama/Llama-3.1-405B",
    "Qwen/Qwen2.5-72B",
]
# Auto-extract: hidden_size, num_heads, num_kv_heads, head_dim,
#               intermediate_size, num_experts, kv_lora_rank, etc.
```

**Step 2**: Supplement with generic power-of-2 grid for interpolation gaps.

**Total estimated shapes**: ~5,000 per version

---

## 7. Development Plan

### Phase 1: Core Infrastructure
- Create `tensor_cast/perf_database/` package
- Implement `OperatorSchema` base class + registry
- Implement `GEMMSchema` and `AttentionSchema`
- Implement `PerfDatabase` with Parquet loading
- Implement exact-match query

### Phase 2: Interpolation & Fallback
- Implement `QueryEngine` with scipy interpolation
- Add confidence scoring
- Implement roofline fallback integration
- Add version resolution logic (`VersionManager`)

### Phase 3: Data Collection Pipeline
- Implement `generate_shape_grid.py` (model-driven + generic)
- Implement `collect_full_model.py` (Level 1 orchestration)
- Implement `collect_microbench.py` (Level 2 torch_npu)
- Implement `parse_ascend_output.py` (kernel_details.csv parser)
- Implement `calibrate.py` (L1 vs L2 alignment)
- Collect initial data: GEMM and Attention on A3

### Phase 4: TensorCast Integration
- Implement `ProfilingPerformanceModel`
- Add CLI arguments to `text_generate.py`
- Add to `model_runner.py` / `config_resolver.py`
- Test with Qwen3-32B and DeepSeek-V3

### Phase 5: Extended Operators & Validation
- Add MoE, Normalization, Fused, Cache schemas
- Validate against full VLLM profiling end-to-end
- Test PD aggregation and disaggregation scenarios
- Add operator discovery pipeline for new models/versions
- Documentation and usage guide

### Validation Criteria

| Metric | Target |
|--------|--------|
| End-to-end time error | <15% vs actual VLLM |
| Per-operator error (matched) | <20% |
| Time coverage | >90% of VLLM execution |

### Test Cases
1. Qwen3-32B Prefill: 136 queries × 4096 tokens, TP=16
2. Qwen3-32B Decode: 136 queries × 1 token, context=4096, TP=16
3. DeepSeek-V3 Prefill: 18 queries × 4096 tokens, TP=4, DP=8, EP
4. DeepSeek-V3 Decode: 18 queries × 1 token, context=4096, TP=4, DP=8, EP
5. PD Aggregation/Disaggregation: Both modes

---

## 8. Dependencies

**New**: `scipy` (interpolation), `pyarrow` (Parquet), `packaging` (version parsing)
**Existing**: `pandas`, `numpy`, `pyyaml`, `torch`

---

## 9. References

- [vLLM Ascend GitHub](https://github.com/vllm-project/vllm-ascend)
- [vLLM Ascend Release Notes](https://docs.vllm.ai/projects/ascend/en/main/user_guide/release_notes.html)
- [vLLM Ascend Profiling Guide](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/service_profiling_guide.html)
- [Huawei Ascend Profiler](https://support.huaweicloud.com/intl/en-us/bestpractice-modelarts/modelarts_llm_infer_5906034.html)
- [Intel NPU Cost Model](https://github.com/intel/npu-nn-cost-model)
- NVIDIA AI Configurator (internal reference at `/home/horacehxw/Projects/aiconfigurator`)
- Existing profiling comparison: `tensor_cast/scripts/profiling_comparison/`
