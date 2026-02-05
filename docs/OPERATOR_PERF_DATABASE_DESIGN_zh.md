# TensorCast 算子性能数据库：技术设计文档 （审核中！！！！）

**版本**: 1.0
**日期**: 2026 年 2 月
**范围**: 面向华为昇腾 A3 的 LLM 仿真可扩展 Profiling 代价模型
**初期目标模型**: DeepSeek-V3、Qwen3-32B

---

## 1. 技术分析

### 1.1 问题陈述

TensorCast 当前采用**基于 Roofline 的解析模型**（`AnalyticPerformanceModel`）估算算子执行耗时。该模型基于浮点运算量（FLOPs）与访存字节数计算 `max(计算耗时, 访存耗时)`，与 Atlas 800 A3 硬件上的实际 VLLM-Ascend Profiling 结果对比存在系统性偏差。

**偏差根因分析**：

| 偏差来源 | 具体描述 | 影响程度 |
|---------|---------|---------|
| **算子融合** | VLLM 将多个算子融合为单一内核执行（如 `DequantSwigluQuant`、`AddRmsNorm`） | 10-30% 时间差异 |
| **硬件利用率** | 实际 Cube 利用率约 44-68%，而非 Roofline 假设的 100% | 高估计算密集型算子 |
| **Shape 相关开销** | 小 batch 场景下内核启动开销未建模 | 低估 Decode 阶段耗时 |
| **版本相关优化** | VLLM-Ascend 各版本引入不同的融合内核 | 模型在版本迭代中失效 |

### 1.2 Profiling 数据分析

基于实际昇腾 Profiler 输出（`kernel_details.csv`、`op_statistic.csv`）：

**Qwen3-32B（共 41 个独立算子）**：
| 核心算子 | 耗时占比 | 核心类型 |
|---------|---------|---------|
| MatMulV2 | 42.4% | AI_CORE |
| FusedInferAttentionScore | 18.2% | MIX_AIC |
| TensorMove | 10.7% | AI_VECTOR_CORE |
| AddRmsNorm | 8.0% | AI_VECTOR_CORE |
| split_qkv_rmsnorm_rope_kernel | 5.2% | MIX_AIC |
| SwiGlu | 4.8% | AI_VECTOR_CORE |

**DeepSeek-V3（共 39 个独立算子，含 MoE 特有算子）**：
| 核心算子 | 耗时占比 | 核心类型 |
|---------|---------|---------|
| GroupedMatmul | 20.9% | AI_CORE |
| FusedInferAttentionScore | 18.5% | MIX_AIC |
| QuantBatchMatmulV3 | 16.9% | AI_CORE |
| MoeDistributeDispatch/Combine | 11.8% | MIX_AIC |
| AscendQuantV2 | 4.4% | AI_VECTOR_CORE |
| TransposeBatchMatMul | 4.2% | AI_CORE |

**关键发现**：约 15 个算子贡献了超过 95% 的执行时间。其余约 65 个算子（算术、索引、逻辑运算）贡献不足 5%，可采用 Roofline 兜底估算。

### 1.3 版本影响分析

VLLM-Ascend 不同版本对融合内核的种类及其性能有显著影响：

| 版本 | 新增融合内核 | 性能影响 |
|-----|------------|---------|
| v0.14.0 | Triton RoPE 内核、MatMul-AllReduce-RMSNorm 融合 | 产生全新融合模式 |
| v0.12.0 | AddRmsNormQuant 融合、大量 Triton 内核 | 算子执行图发生变化 |
| v0.10.x | MLP 张量并行、AllGather-Expert 融合 | 专家路由方式变更 |

CANN 版本同样影响内核执行效率（不同编译优化策略），两者均需纳入版本追踪。

### 1.4 现有基础设施

**`msmodeling-profiling_compare`**（已实现）：
- 融合感知的算子映射（YAML 配置，100+ 映射规则）
- VLLM kernel_details.csv 解析器，支持阶段检测
- TensorCast 仿真适配器
- Excel 对比报告
- 代码位置：`tensor_cast/scripts/profiling_comparison/`

**`EmpiricalPerformanceModel`**（已实现，仅支持 JIT 基准测试）：
- `tensor_cast/performance_model/empirical.py` — 在实际设备上运行基准测试
- 使用 `OpBenchmark` 类
- 仅在物理设备可用时工作

### 1.5 AI Configurator 参考（NVIDIA）

从 `/home/horacehxw/Projects/aiconfigurator` 项目中提炼的核心设计模式：
- **嵌套字典索引**：对已 Profiling 的 Shape 实现 O(1) 精确查找
- **scipy.griddata 插值**：多维 Shape 空间匹配
- **DatabaseMode 降级策略**：SILICON → HYBRID → EMPIRICAL → SOL
- **PerformanceResult(float)**：后向兼容的结果类型
- **CSV 存储**：可读性强、便于版本管理
- **延迟加载与缓存**：模块级缓存避免重复加载

---

## 2. 系统架构

### 2.1 整体架构

* TensorCast Runtime 通过 `PerformanceModel` 插件化架构接入 Profiling 数据库，选择哪种建模应该是可以配置的，都必须支持。

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           TensorCast Runtime                                  │
│                                                                               │
│  ┌────────────────────┐     ┌─────────────────────────────────────────────┐  │
│  │  Runtime            │     │  ProfilingPerformanceModel（新增）          │  │
│  │  (TorchDispatchMode)│────▶│  ┌─────────────────────────────────────┐   │  │
│  │                     │     │  │ 1. 匹配算子 → OperatorSchema        │   │  │
│  │  拦截所有算子调用   │     │  │ 2. 从 OpInvokeInfo 提取 Shape       │   │  │
│  │  生成 OpInvokeInfo  │     │  │ 3. 查询 PerfDatabase               │   │  │
│  └────────────────────┘     │  │ 4. 降级至 AnalyticModel             │   │  │
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
                    │  │  │ 精确匹配 → 插值估算          │ │    │
                    │  │  │ → 外推估算 → Roofline 兜底   │ │    │
                    │  │  └──────────────────────────────┘ │    │
                    │  └──────────────────────────────────┘    │
                    │                     │                     │
                    │  ┌──────────────────┴──────────────────┐ │
                    │  │ OperatorSchema 注册表               │ │
                    │  │ ┌──────┐ ┌─────────┐ ┌───┐ ┌─────┐ │ │
                    │  │ │ GEMM │ │Attention│ │MoE│ │Fused│ │ │
                    │  │ └──────┘ └─────────┘ └───┘ └─────┘ │ │
                    │  └─────────────────────────────────────┘ │
                    │                     │                     │
                    │  ┌──────────────────┴──────────────────┐ │
                    │  │ 存储层（按版本分目录，Parquet/CSV）  │ │
                    │  │ atlas_a3/vllm_ascend/0.14.0/        │ │
                    │  │   gemm.parquet, attention.parquet    │ │
                    │  └─────────────────────────────────────┘ │
                    └──────────────────────────────────────────┘

数据采集流水线（离线执行）
┌──────────────────────────────────────────────────────────────┐
│  Level 1: 全模型 Profiling                                    │
│  VLLM serve + bench → kernel_details.csv → 解析 → 入库       │
├──────────────────────────────────────────────────────────────┤
│  Level 2: 单算子微基准测试                                    │
│  torch_npu 脚本 → 直接测量 → 入库                            │
├──────────────────────────────────────────────────────────────┤
│  校准：以 Level 1 数据为基准，校正 Level 2 测量偏差           │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 模块结构

```
tensor_cast/perf_database/              # 新增包
├── __init__.py
├── core/
│   ├── database.py                     # PerfDatabase 主类
│   ├── query.py                        # QueryEngine（含插值逻辑）
│   ├── storage.py                      # Parquet/CSV IO 后端
│   └── versioning.py                   # 版本解析与管理
├── operators/
│   ├── __init__.py                     # 通过 importlib 自动发现
│   ├── base.py                         # OperatorSchema 抽象基类 + 注册表
│   ├── gemm.py                         # GEMM/MatMul 算子 Schema
│   ├── attention.py                    # MHA、GQA、MLA 算子 Schema
│   ├── moe.py                          # MoE 路由 + 专家计算 Schema
│   ├── communication.py                # allreduce、allgather、alltoall
│   ├── normalization.py                # RMSNorm、AddRmsNorm
│   ├── fused.py                        # DequantSwigluQuant 等融合算子
│   └── elementwise.py                  # Cast、激活函数、逐元素运算
├── shape_generators/
│   ├── base.py                         # ShapeGridGenerator 抽象基类
│   ├── model_driven.py                 # 从 HuggingFace 配置自动提取
│   └── universal.py                    # 2 的幂次 + 常见 Shape 模板
├── interpolation/
│   ├── base.py                         # InterpolationStrategy 抽象基类
│   └── linear.py                       # 基于 scipy 的线性插值
├── mappings/vllm_ascend/
│   ├── v0.10.yaml                      # VLLM 内核 → Schema 映射
│   ├── v0.12.yaml
│   └── v0.14.yaml
├── data/systems/                       # Profiling 数据存储（gitignore）
│   └── atlas_a3_752t_128g/
│       └── vllm_ascend/{version}/
│           ├── metadata.yaml
│           ├── gemm.parquet
│           ├── attention.parquet
│           └── ...
└── scripts/
    ├── collect_full_model.py           # Level 1: 基于 VLLM 的 Profiling
    ├── collect_microbench.py           # Level 2: torch_npu 微基准测试
    ├── parse_ascend_output.py          # 解析 kernel_details.csv
    ├── generate_shape_grid.py          # 根据模型配置生成 Shape 网格
    ├── calibrate.py                    # L2 对齐 L1 的校准
    └── validate.py                     # 数据库精度验证

tensor_cast/performance_model/
└── profiling.py                        # ProfilingPerformanceModel（新增）
```

---

## 3. 核心模块设计

### 3.1 OperatorSchema（插件化架构）

每种算子类型定义其 Shape 空间、提取逻辑和 Roofline 兜底估算：

```python
# tensor_cast/perf_database/operators/base.py

class DimensionType(Enum):
    BATCH = auto()      # 随部署变化（batch_size、num_tokens）
    SEQUENCE = auto()   # 随输入变化（query_len、context_len）
    FEATURE = auto()    # 模型固有参数（hidden_size、head_dim）
    EXPERT = auto()     # MoE 专有（num_experts、top_k）
    DEVICE = auto()     # 并行策略相关（num_devices）

@dataclass
class DimensionSpec:
    name: str
    dim_type: DimensionType
    typical_range: Tuple[int, int]
    typical_values: Optional[List[int]] = None
    is_required: bool = True

class OperatorSchema(ABC):
    """
    算子性能 Schema 基类。
    采用注册表模式自动发现（与 DeviceProfile 设计一致）。
    """
    _registry: ClassVar[Dict[str, Type["OperatorSchema"]]] = {}

    @classmethod
    def register(cls, op_name: str):
        """注册装饰器"""
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
        """该 Schema 处理的 TensorCast 算子名称列表"""
        ...

    @abstractmethod
    def extract_shape_from_op(self, op_invoke_info: OpInvokeInfo) -> Dict[str, Any]:
        """从拦截到的算子调用中提取 Shape 维度"""
        ...

    @abstractmethod
    def compute_roofline_estimate(self, shape, device_profile) -> float:
        """计算 Roofline（Speed-of-Light）估算值，单位微秒"""
        ...

    def get_interpolation_dimensions(self) -> List[str]:
        """返回适用于连续插值的维度"""
        return [d.name for d in self.dimensions
                if d.dim_type in (DimensionType.BATCH, DimensionType.SEQUENCE)]
```

**具体 Schema 一览**：

| Schema | 关键维度 | 对应 TensorCast 算子 |
|--------|---------|---------------------|
| `GEMMSchema` | m, n, k, quant_mode | `aten.mm`、`static_quant_linear`、`fp8_linear`、`grouped_matmul_quant` |
| `AttentionSchema` | batch, query_len, context_len, num_heads, num_kv_heads, head_dim, kv_lora_rank | `attention`、`attention_quant`、`multihead_latent_attention` |
| `MoESchema` | num_tokens, num_experts, top_k, hidden_size, intermediate_size | `grouped_matmul`、`permute_tokens`、`unpermute_tokens` |
| `NormalizationSchema` | num_tokens, hidden_size | `rms_norm`、`add_rms_norm` |
| `FusedSchema` | （随融合内核而异） | 映射至组成算子 |
| `CommunicationSchema` | op_type, num_devices, message_size | `all_reduce`、`all_gather`、`all_to_all` |

### 3.2 PerfDatabase

```python
# tensor_cast/perf_database/core/database.py

class PerfDatabase:
    """性能数据库主类。加载 Profiling 数据，提供查询接口。"""

    def __init__(
        self,
        system: str = "atlas_a3_752t_128g",
        backend: str = "vllm_ascend",
        version: str = "latest",
        data_root: Optional[Path] = None,
    ):
        self.version_mgr = VersionManager(data_root)
        self.resolved_version, self.data_path = self.version_mgr.resolve(system, backend, version)
        self.query_engine = QueryEngine(self)
        self._data_cache: Dict[str, pd.DataFrame] = {}

    def query(self, schema_name: str, shape: Dict, mode: QueryMode = QueryMode.HYBRID) -> QueryResult:
        """主查询入口，委托给 QueryEngine。"""
        return self.query_engine.query(schema_name, shape, mode)

    def get_data(self, schema_name: str, quant_mode: str) -> Optional[pd.DataFrame]:
        """延迟加载并缓存算子 Parquet 数据"""
        cache_key = f"{schema_name}_{quant_mode}"
        if cache_key not in self._data_cache:
            path = self.data_path / f"{schema_name}.parquet"
            if not path.exists():
                return None
            df = pd.read_parquet(path)
            self._data_cache[cache_key] = df[df["quant_mode"] == quant_mode]
        return self._data_cache[cache_key]
```

### 3.3 QueryEngine（插值 + 降级策略）

```python
# tensor_cast/perf_database/core/query.py

class QueryMode(Enum):
    EXACT = auto()
    INTERPOLATE = auto()
    HYBRID = auto()       # 精确匹配 → 插值 → Roofline
    ROOFLINE = auto()

class QuerySource(Enum):
    MEASURED = auto()          # 精确 Profiling 数据（置信度: 1.0）
    INTERPOLATED = auto()      # scipy 插值（置信度: 0.7-0.95）
    EXTRAPOLATED = auto()      # 凸包外推（置信度: 0.3-0.6）
    ROOFLINE = auto()          # 解析模型（置信度: 0.4）
    ROOFLINE_CALIBRATED = auto()  # Roofline × 经验效率系数（置信度: 0.6）

@dataclass
class QueryResult:
    latency_us: float
    confidence: float
    source: QuerySource
    details: Dict[str, Any] = field(default_factory=dict)

class QueryEngine:
    """处理精确查找、插值估算和降级策略。"""

    def query(self, schema_name, shape, mode=QueryMode.HYBRID) -> QueryResult:
        schema = OperatorSchema.get_schema(schema_name)

        if mode == QueryMode.HYBRID:
            # 按优先级依次尝试：精确匹配 → 插值 → Roofline 兜底
            result = self._try_exact(schema, shape)
            if result: return result

            result = self._try_interpolate(schema, shape)
            if result: return result

            return self._roofline_fallback(schema, shape)

    def _try_interpolate(self, schema, shape) -> Optional[QueryResult]:
        """基于 scipy.interpolate.LinearNDInterpolator 的多维插值"""
        # 获取适用于连续插值的维度
        interp_dims = schema.get_interpolation_dimensions()
        # 从已有测量数据构建点云
        # 执行插值计算，根据到最近测量点的距离估算置信度
        ...
```

### 3.4 算子到 Schema 的映射（YAML 配置）

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
      - "tensor_cast.permute_tokens.default"
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

## 4. 接口设计：TensorCast 集成

### 4.1 ProfilingPerformanceModel

这是与 TensorCast 集成的核心入口。它继承自 `PerformanceModel`（与 `AnalyticPerformanceModel`、`EmpiricalPerformanceModel` 共用同一抽象基类）。

```python
# tensor_cast/performance_model/profiling.py

class ProfilingPerformanceModel(PerformanceModel):
    """
    基于 Profiling 数据的性能模型。

    与 TensorCast 的集成方式：
    - 继承 PerformanceModel（tensor_cast/performance_model/__init__.py:169）
    - 实现 process_op(OpInvokeInfo) → PerformanceModel.Result
    - 通过 perf_models 列表接入 Runtime（tensor_cast/runtime.py:41）
    - 由 Runtime 自动包装为 CachingPerformanceModel
    - 支持 get_classifiers() 用于瓶颈分类统计
    """

    def __init__(
        self,
        device_profile: DeviceProfile,            # 来自 tensor_cast/device.py
        database_mode: QueryMode = QueryMode.HYBRID,
        vllm_version: str = "latest",
    ):
        super().__init__("profiling", device_profile)  # name="profiling"
        self.db = PerfDatabase(
            system=self._device_to_system(device_profile),
            version=vllm_version,
        )
        self.query_mode = database_mode
        self.fallback = AnalyticPerformanceModel(device_profile)  # 现有解析模型
        self._schema_cache: Dict[str, Optional[OperatorSchema]] = {}

    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        """
        核心接口方法。
        由 Runtime.__torch_dispatch__ 在每次算子拦截时调用。

        参数:
            op_invoke_info: 包含 func、args、kwargs、out、cache_key
                           （定义于 tensor_cast/performance_model/__init__.py:52）

        返回:
            PerformanceModel.Result，包含 execution_time_s 和 statistics 字典
            （定义于 tensor_cast/performance_model/__init__.py:175）
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
                pass  # 降级至解析模型

        return self.fallback.process_op(op_invoke_info)

    def get_classifiers(self) -> List[PerformanceModel.OpClassifier]:
        return self.fallback.get_classifiers()
```

### 4.2 Runtime 集成

`Runtime` 类无需任何修改。它已支持接收 `PerformanceModel` 实例列表：

```python
# tensor_cast/runtime.py:41-56（现有代码，无需改动）
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

使用 Profiling 模型只需将其作为 perf_model 传入：
```python
# 在 model_runner.py 或 config_resolver.py 中：
perf_model = ProfilingPerformanceModel(device_profile, vllm_version="0.14.0")
runtime = Runtime(perf_models=perf_model, device_profile=device_profile)
```

### 4.3 CLI 接口

在 `tensor_cast/scripts/text_generate.py` 中新增以下参数：

```python
parser.add_argument("--performance-model", choices=["analytic", "profiling", "empirical"],
                    default="analytic", help="性能模型类型")
parser.add_argument("--database-mode", choices=["exact", "interpolate", "hybrid", "roofline"],
                    default="hybrid", help="数据库查询模式")
parser.add_argument("--database-version", type=str, default="latest",
                    help="Profiling 数据库版本号")
```

### 4.4 数据流

```
用户 CLI                     TensorCast                    数据库
────────                     ──────────                    ──────
text_generate.py
  --performance-model profiling
  --database-version 0.14.0
       │
       ▼
  ModelRunner 创建
  ProfilingPerformanceModel
       │
       ▼
  Runtime.__torch_dispatch__
  拦截每个算子调用
       │
       ▼
  OpInvokeInfo(func, args, kwargs, out)
       │
       ▼
  ProfilingPerformanceModel.process_op()
       │
       ├──▶ 匹配算子到 OperatorSchema
       │    （例如 aten.mm → GEMMSchema）
       │
       ├──▶ schema.extract_shape_from_op()
       │    → {m: 136, n: 4096, k: 5120, quant: "int8"}
       │
       ├──▶ db.query("gemm", shape, mode=HYBRID)
       │         │
       │         ├──▶ gemm.parquet 中精确匹配？ → 返回
       │         ├──▶ 从邻近点插值？ → 返回
       │         └──▶ Roofline 兜底 → 返回
       │
       └──▶ 返回 PerformanceModel.Result(execution_time_s=...)
```

---

## 5. 自动化 Profiling 流水线

### 5.1 两级采集策略

#### Level 1: 全模型 Profiling（获取基准真值）

启用昇腾 Profiler 运行 VLLM 推理，捕获真实融合内核的执行耗时。

**触发方式**（参考实际 Profiling 脚本）：
```bash
# 环境变量配置
export VLLM_TORCH_PROFILER_DIR=/path/to/output
export PROFILING_SAVE_PATH=/path/to/output

# 启动 VLLM 服务（高开销：模型加载需数分钟）
vllm serve $MODEL --tensor-parallel-size $TP --dtype bfloat16 ...

# 发送受控请求进行基准测试（低开销：每轮仅数秒）
vllm bench serve --profile \
    --dataset-name random \
    --random-input-len $INPUT_LEN \
    --random-output-len $OUTPUT_LEN \
    --max-concurrency $BATCH_SIZE
```

**效率优化**：每个服务实例只需启动一次，随后发送多轮不同 Shape 配置的基准测试请求。

**单实例内 Shape 遍历方案**：
```python
# Decode 场景（query_len=1）
for batch in [1, 8, 16, 32, 64, 128, 256]:
    bench(input_len=1, output_len=1, concurrency=batch)

# Prefill 场景
for isl in [256, 512, 1024, 2048, 4096, 8192]:
    for batch in [1, 8, 32, 64, 128]:
        bench(input_len=isl, output_len=1, concurrency=batch)
```

**需要独立服务实例的配置**：
```python
SERVER_CONFIGS = [
    {"model": "Qwen3-32B", "tp": 4, "quant": "none"},
    {"model": "Qwen3-32B", "tp": 8, "quant": "W8A8"},
    {"model": "Qwen3-32B", "tp": 16, "quant": "W4A8"},
    {"model": "DeepSeek-V3", "tp": 4, "dp": 8, "ep": True, "quant": "W8A8"},
    {"model": "DeepSeek-V3", "tp": 4, "dp": 8, "ep": True, "quant": "W4A8"},
]
```

**输出解析**：
```python
# 解析 ASCEND_PROFILER_OUTPUT/kernel_details.csv
# 列字段：Name, Duration(us), Input Shapes, cube_utilization(%)
# 以 FusedInferAttentionScore 作为步骤边界进行分组
# 提取每个 Shape 配置下各算子的聚合耗时
```

#### Level 2: 单算子微基准测试（Shape 网格填充）

通过 `torch_npu` 直接对单个算子进行细粒度 Shape 覆盖测试：

```python
# scripts/collect_microbench.py
import torch, torch_npu

def benchmark_matmul(m, n, k, dtype, warmup=5, runs=20):
    a = torch.randn(m, k, dtype=dtype, device='npu')
    b = torch.randn(k, n, dtype=dtype, device='npu')
    # 预热
    for _ in range(warmup):
        torch.mm(a, b)
    torch.npu.synchronize()
    # 计时
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()
    for _ in range(runs):
        torch.mm(a, b)
    end_event.record()
    torch.npu.synchronize()
    return start_event.elapsed_time(end_event) / runs * 1000  # 返回微秒
```

#### Level 1 与 Level 2 对比

| 维度 | Level 1（全模型 Profiling） | Level 2（微基准测试） |
|-----|--------------------------|---------------------|
| 融合内核捕获 | 完整捕获（基准真值） | 部分捕获 |
| Shape 控制 | 间接控制（batch、seq_len） | 直接控制（m、n、k） |
| 执行开销 | 高（服务启动耗时长） | 低（仅内核执行） |
| 覆盖范围 | 每个模型约 100 种配置 | 每个算子约 2500 种 Shape |
| 适用场景 | 验证与校准 | Shape 网格大规模填充 |

**校准方法**：在 Shape 重叠区域对比 Level 1 和 Level 2 的数据，计算每类算子的修正系数，使 Level 2 数据对齐实际部署场景。

### 5.2 Profiling 输出解析器

```python
# scripts/parse_ascend_output.py

class AscendProfilerParser:
    def parse_kernel_details(self, csv_path: Path) -> pd.DataFrame:
        """解析 kernel_details.csv → 结构化 DataFrame"""
        # 处理灵活的列名格式和缺失字段
        # 返回: name, duration_us, input_shapes, data_types,
        #       aicore_time_us, aiv_time_us, cube_utilization_pct

    def extract_single_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """以 Attention 内核为边界提取单个 Decode/Prefill 步骤"""
        # 复用 profiling_compare 中的逻辑: kernel_details_parser.py

    def map_to_schema(self, kernel_name: str, input_shapes: str) -> Tuple[str, Dict]:
        """将 VLLM 内核名映射到 (schema_name, shape_dict)"""
        # "MatMulV2" + "136,4096; 4096,4096" → ("gemm", {m:136, k:4096, n:4096})
        # 使用版本相关的 YAML 映射配置
```

### 5.3 新模型算子发现机制

```python
# scripts/discover_operators.py

def discover_operators(profiling_output: Path, mapping_yaml: Path) -> Dict:
    """发现 Profiling 中存在但当前映射表中缺失的算子"""
    parser = AscendProfilerParser()
    kernels = parser.parse_kernel_details(profiling_output / "kernel_details.csv")

    known_ops = load_yaml_mappings(mapping_yaml)
    unknown = []

    for kernel_name in kernels["name"].unique():
        if not any(kernel_name in schema_kernels for schema_kernels in known_ops.values()):
            # 基于内核名称模式自动分类
            suggested_schema = auto_classify(kernel_name)
            unknown.append({"kernel": kernel_name, "suggested_schema": suggested_schema})

    return {"known": len(kernels) - len(unknown), "unknown": unknown}
```

---

## 6. 全面算子覆盖策略

### 6.1 算子分级

| 层级 | 判定标准 | 处理方式 | 算子数量 |
|-----|---------|---------|---------|
| **Tier 1** | 执行耗时占比 >2% | 必须使用完整 Shape 网格进行 Profiling | 约 11 个 |
| **Tier 2** | 执行耗时占比 0.5-2% | 使用精简 Shape 网格进行 Profiling | 约 8 个 |
| **Tier 3** | 执行耗时占比 <0.5% | 采用 Roofline 兜底估算 | 60+ 个 |

### 6.2 Tier 1 完整算子列表

| VLLM 内核名称 | 数据库 Schema | Qwen3 占比 | DSV3 占比 |
|-------------|-------------|-----------|---------|
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

### 6.3 Shape 网格（混合策略：模型驱动 + 通用网格）

**第一步**：从 HuggingFace 模型配置自动提取维度参数：
```python
PRIORITY_MODELS = [
    "Qwen/Qwen3-32B", "Qwen/Qwen3-235B-A22B",
    "deepseek-ai/DeepSeek-V3", "moonshotai/Kimi-K2-Instruct",
    "meta-llama/Llama-3.1-70B", "meta-llama/Llama-3.1-405B",
    "Qwen/Qwen2.5-72B",
]
# 自动提取：hidden_size、num_heads、num_kv_heads、head_dim、
#           intermediate_size、num_experts、kv_lora_rank 等
```

**第二步**：以通用的 2 的幂次网格补充模型间的插值间隙。

**预估 Shape 总量**：每个版本约 5,000 个

---

## 7. 开发计划

### 阶段一：核心基础设施
- 创建 `tensor_cast/perf_database/` 包结构
- 实现 `OperatorSchema` 基类及注册表机制
- 实现 `GEMMSchema` 和 `AttentionSchema`
- 实现 `PerfDatabase`（Parquet 加载）
- 实现精确匹配查询

### 阶段二：插值与降级策略
- 实现 `QueryEngine`（基于 scipy 的多维插值）
- 添加置信度评分机制
- 集成 Roofline 兜底估算
- 实现版本解析逻辑（`VersionManager`）

### 阶段三：数据采集流水线
- 实现 `generate_shape_grid.py`（模型驱动 + 通用网格）
- 实现 `collect_full_model.py`（Level 1 编排脚本）
- 实现 `collect_microbench.py`（Level 2 torch_npu 微基准测试）
- 实现 `parse_ascend_output.py`（kernel_details.csv 解析器）
- 实现 `calibrate.py`（L1 与 L2 校准对齐）
- 完成 GEMM 和 Attention 在 A3 上的初始数据采集

### 阶段四：TensorCast 集成
- 实现 `ProfilingPerformanceModel`
- 在 `text_generate.py` 中添加 CLI 参数
- 在 `model_runner.py` / `config_resolver.py` 中接入
- 使用 Qwen3-32B 和 DeepSeek-V3 进行端到端测试

### 阶段五：扩展算子与精度验证
- 新增 MoE、归一化、融合算子、缓存算子的 Schema
- 与实际 VLLM Profiling 进行端到端精度验证
- 测试 PD 聚合与分离（Aggregation/Disaggregation）场景
- 完善新模型算子发现流水线
- 编写使用文档

### 验证标准

| 指标 | 目标值 |
|-----|-------|
| 端到端耗时误差 | 与实际 VLLM 对比 <15% |
| 单算子误差（已匹配算子） | <20% |
| 时间覆盖率 | 覆盖 VLLM 执行时间的 >90% |

### 测试用例
1. Qwen3-32B Prefill：136 请求 x 4096 tokens，TP=16
2. Qwen3-32B Decode：136 请求 x 1 token，context=4096，TP=16
3. DeepSeek-V3 Prefill：18 请求 x 4096 tokens，TP=4，DP=8，EP
4. DeepSeek-V3 Decode：18 请求 x 1 token，context=4096，TP=4，DP=8，EP
5. PD 聚合/分离模式：分别测试两种模式

---

## 8. 依赖项

**新增**：`scipy`（插值）、`pyarrow`（Parquet）、`packaging`（版本解析）
**现有**：`pandas`、`numpy`、`pyyaml`、`torch`

---

## 9. 参考资料

- [vLLM Ascend GitHub](https://github.com/vllm-project/vllm-ascend)
- [vLLM Ascend 发布说明](https://docs.vllm.ai/projects/ascend/en/main/user_guide/release_notes.html)
- [vLLM Ascend Profiling 指南](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/service_profiling_guide.html)
- [华为昇腾 Profiler 文档](https://support.huaweicloud.com/intl/en-us/bestpractice-modelarts/modelarts_llm_infer_5906034.html)
- [Intel NPU Cost Model](https://github.com/intel/npu-nn-cost-model)
- NVIDIA AI Configurator（内部参考，位于 `/home/horacehxw/Projects/aiconfigurator`）
- 现有 Profiling 对比工具：`tensor_cast/scripts/profiling_comparison/`
