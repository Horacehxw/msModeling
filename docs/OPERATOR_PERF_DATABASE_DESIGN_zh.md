# TensorCast 算子性能数据库：技术设计文档 （审核中）

**版本**: 1.1
**日期**: 2026 年 2 月
**范围**: 面向 LLM 仿真的可扩展 Profiling Cost Model，不绑定具体算力卡，支持基于实测 Profiling 数据的算子性能估算。
**初期目标模型**: DeepSeek-V3、Qwen3-32B

作者：贺骁武

审核人：龚炯

---

## 1. 功能概述

### 1.1 目标

为 TensorCast 仿真器构建基于实测数据的算子性能估算系统。通过扩展现有 `EmpiricalPerformanceModel` 框架，引入 **DataSource 抽象层**，使其支持从预构建的性能数据库中查询算子耗时，替代纯 Roofline 解析模型，提升仿真精度。

### 1.2 核心功能（本方案范围）

1. **算子性能数据库**：定义标准化数据格式和查询接口，支持按算子类型、Shape 维度、量化模式进行性能查询，与数据来源解耦（支持 micro-benchmark、全模型 Profiling 等多种数据源）
2. **EmpiricalPerformanceModel 扩展**：引入 `DataSource` 抽象层（`DatabaseSource`、`BenchmarkSource`、`CachedSource`），使 `EmpiricalPerformanceModel` 按优先级查询多种数据源，不再仅限于 JIT 基准测试
3. **查询引擎**：实现精确匹配 → 插值估算 → 外推估算的多级查询策略；对于未收录算子，内部兜底回退至 Roofline 模型
4. **数据采集流水线**（独立子系统，不在 TensorCast 包内）：自动化执行全模型 Profiling 和单算子微基准测试，解析输出并构建性能数据库

### 1.3 不在本方案范围内（建议后续支持）

- **CompositePerformanceModel 顶层调度器**：统一编排多种 PerformanceModel（Empirical + Analytic + Communication），实现可配置的降级策略与模型组合（详见第 9 节 Future Work）
- **跨硬件泛化**：当前仅支持昇腾 A3，其他硬件需独立采集数据并注册 DeviceProfile
- **自动化持续集成**：随 VLLM-Ascend / CANN 版本发布自动触发数据采集
- **能耗建模**：AI Configurator 的 `PerformanceResult(float)` 支持能耗追踪，本方案暂不涉及

### 1.4 初期目标

- **目标模型**：DeepSeek-V3、Qwen3-32B
- **目标硬件**：Atlas 800 A3（752T，128G DIE）
- **目标后端**：VLLM-Ascend v0.13.0+
- **精度目标**：端到端仿真误差 <15%（对比实际 VLLM Profiling）
- **交付时间**：Q3（2026.3.20）完全跑通并完成初始数据采集和集成测试

---

## 2. 技术分析

### 2.1 问题陈述

TensorCast 当前采用**基于 Roofline 的解析模型**（`AnalyticPerformanceModel`）估算算子执行耗时。该模型基于浮点运算量（FLOPs）与访存字节数计算 `max(计算耗时, 访存耗时)`，与 Atlas 800 A3 硬件上的实际 VLLM-Ascend Profiling 结果对比存在系统性偏差。

现有 `EmpiricalPerformanceModel`（`tensor_cast/performance_model/empirical.py`）已提供对接实测数据的初步框架：

- 继承 `PerformanceModel` 抽象基类，实现 `process_op(OpInvokeInfo) → Result` 接口
- 当前仅通过 `OpBenchmark` 类在物理设备上 JIT 执行并计时
- 代码中已标注 `TODO: add a mode so that we can do JIT or offline benchmarks`
- `OpBenchmarkBase` 抽象基类为多种数据源预留了扩展点

**本方案的核心思路**：基于现有 `EmpiricalPerformanceModel` 框架进行扩展，引入 DataSource 抽象层，使其支持从数据库读取预存数据，同时保留 JIT 基准测试能力。

**偏差根因分析**：

| 偏差来源 | 具体描述 | 影响程度 |
|---------|---------|---------|
| **算子融合** | VLLM 将多个算子融合为单一内核执行（如 `DequantSwigluQuant`、`AddRmsNorm`） | 10-30% 时间差异 |
| **硬件利用率** | 实际 Cube 利用率约 44-68%，而非 Roofline 假设的 100% | 高估计算密集型算子 |
| **Shape 相关开销** | 小 batch 场景下内核启动开销未建模 | 低估 Decode 阶段耗时 |
| **版本相关优化** | VLLM-Ascend 各版本引入不同的融合内核 | 模型在版本迭代中失效 |

### 2.2 Profiling 数据分析

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

### 2.3 版本影响分析

VLLM-Ascend 不同版本对融合内核的种类及其性能有显著影响。以下信息来自 [vLLM-Ascend 发布说明](https://docs.vllm.ai/projects/ascend/en/main/user_guide/release_notes.html) 和 [GitHub Releases](https://github.com/vllm-project/vllm-ascend/releases)：

| 版本 | 发布日期 | 关键融合内核变化 |
|-----|---------|---------------|
| v0.13.0（最新稳定版） | 2026.02.06 | AddRmsnormQuant 融合+SP 支持、fused matmul/reduce-scatter 内核、Triton chunk_gated_delta_rule（Qwen3-Next）、Triton RoPE 优化 |
| v0.14.0rc1（预发布） | 2026.01.26 | MatMul-AllReduce-RMSNorm 融合 Pass（默认关闭，需配置 `fuse_allreduce_rms=True`）、310P 基础支持 |
| v0.12.0rc1 | 2025.12.13 | 大量 Triton 内核（Qwen3-Next, DeepSeek 3.2）、Full Decode-Only Graph Mode（实验性）、W4A4 量化支持 |
| v0.11.0 | 2025.12.16 | W8A16 量化、Full Graph Mode (ACLGraph) + GQA、多 Token 预测 + Chunked Prefill、Context Parallel（实验性） |

CANN 版本同样影响内核执行效率（不同编译优化策略）。例如 CANN 8.5 针对 FIA 算子做了 flash decoding 优化。两者版本号均需纳入数据库版本追踪。

### 2.4 现有基础设施

**`EmpiricalPerformanceModel`**（已实现，位于 `tensor_cast/performance_model/empirical.py`）：

当前实现仅 20 行代码，是一个轻量级框架：
```python
class EmpiricalPerformanceModel(PerformanceModel):
    def __init__(self, device_profile: DeviceProfile):
        super().__init__("empirical", device_profile)
        self.op_benchmark = OpBenchmark(device_profile)  # JIT 基准测试

    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        return self.op_benchmark.benchmark(op_invoke_info)
```

`OpBenchmark`（`op_benchmark.py`）提供了关键扩展点：
- `OpBenchmarkBase` 抽象基类定义了 `benchmark()` 接口
- 支持 meta tensor → real tensor 的自动转换
- `register_op_impl` 注册表为 TensorCast 自定义算子提供设备特定实现
- 配置项位于 `tensor_cast/config.py` 的 `performance_model.empirical` 命名空间

**`msmodeling-profiling_compare`**（已实现，独立模块）：
- 融合感知的算子映射（YAML 配置，100+ 映射规则）
- VLLM kernel_details.csv 解析器，支持阶段检测
- TensorCast 仿真适配器
- Excel 对比报告

### 2.5 AI Configurator 参考

参考 [AI Configurator](https://github.com/ai-dynamo/aiconfigurator) 项目（NVIDIA 的 LLM 推理性能预估工具）提炼的核心设计模式：

- **嵌套字典索引**：对已 Profiling 的 Shape 实现 O(1) 精确查找，如 `gemm_data[quant_mode][m][n][k]`
- **逐轴 1D 插值**：在嵌套字典结构上按维度依次进行线性插值（基于 `scipy.interpolate.griddata`），支持 sqrt 变换处理 O(n²) 复杂度的 Attention 序列维度
- **DatabaseMode 降级策略**：
  - **SILICON**：仅使用实测数据（精确匹配或插值），无数据时抛异常
  - **HYBRID**：优先使用实测数据，无数据时回退至 EMPIRICAL
  - **EMPIRICAL**：基于 Roofline 理论值乘以经验效率系数（0.6-0.8）
  - **SOL**（Speed-of-Light）：纯理论峰值，`max(compute_time, memory_time)` Roofline 模型
- **PerformanceResult(float)**：继承 `float` 的后向兼容结果类型，附带 energy 属性
- **CSV 存储**：可读性强、Git-friendly，按 `systems/{device}/{backend}/{version}/` 分目录存储
- **延迟加载与多级缓存**：全局 DB 缓存、LRU 查询缓存、metric 提取缓存；初始化时预插值常用网格点
- **SOL 数据校正**：用理论下界校正异常测量值，确保 `measured >= SOL`

---

## 3. 系统架构

### 3.1 整体架构

本方案聚焦于 `EmpiricalPerformanceModel` 的扩展设计。TensorCast Runtime 已支持 `PerformanceModel` 插件化架构，`EmpiricalPerformanceModel` 通过引入 DataSource 抽象层接入性能数据库。

> **注意**：关于顶层 CompositePerformanceModel（组合 Empirical + Analytic + Communication 模型，实现可配置降级策略）的设计，详见第 9 节建议与 Future Work。当前方案中 `EmpiricalPerformanceModel` 对未收录算子内部兜底回退至 `AnalyticPerformanceModel`，作为 v1.0 的务实方案。

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           TensorCast Runtime                                  │
│                                                                               │
│  ┌────────────────────┐     ┌─────────────────────────────────────────────┐  │
│  │  Runtime            │     │  EmpiricalPerformanceModel（扩展）          │  │
│  │  (TorchDispatchMode)│────▶│  ┌─────────────────────────────────────┐   │  │
│  │                     │     │  │ DataSource 优先级链:                 │   │  │
│  │  拦截所有算子调用   │     │  │  1. DatabaseSource（查询性能数据库） │   │  │
│  │  生成 OpInvokeInfo  │     │  │  2. CachedSource（本地执行缓存）    │   │  │
│  └────────────────────┘     │  │  3. BenchmarkSource（JIT 基准测试） │   │  │
│                              │  │  4. Roofline 兜底                   │   │  │
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
                    │  │  │ → 外推估算                    │ │    │
                    │  │  └──────────────────────────────┘ │    │
                    │  └──────────────────────────────────┘    │
                    │                     │                     │
                    │  ┌──────────────────┴──────────────────┐ │
                    │  │ OperatorSchema 注册表               │ │
                    │  │  TensorCast 算子 → Schema 映射      │ │
                    │  │ ┌──────┐ ┌─────────┐ ┌───┐ ┌─────┐ │ │
                    │  │ │ GEMM │ │Attention│ │MoE│ │Fused│ │ │
                    │  │ └──────┘ └─────────┘ └───┘ └─────┘ │ │
                    │  └─────────────────────────────────────┘ │
                    │                     │                     │
                    │  ┌──────────────────┴──────────────────┐ │
                    │  │ 存储层（按版本分目录，CSV/Parquet）  │ │
                    │  │ data/{device}/{backend}/{version}/   │ │
                    │  │   gemm.csv, attention.csv, ...       │ │
                    │  └─────────────────────────────────────┘ │
                    └──────────────────────────────────────────┘

数据采集流水线（离线执行，独立子系统）
┌──────────────────────────────────────────────────────────────┐
│  方案 A（默认）: 全模型 Profiling + 微基准测试               │
│    VLLM serve + bench → kernel_details.csv → 解析 → 入库    │
│    torch_npu 微基准测试 → 直接测量 → 入库                    │
│    校准：以 Level 1 数据为基准，校正 Level 2 测量偏差        │
├──────────────────────────────────────────────────────────────┤
│  方案 B: 仿真驱动 + 微基准测试                               │
│    VLLM 拉取一次 → 获取算子列表和名称                        │
│    TensorCast 仿真 → 枚举不同配置下的 Input Shape            │
│    torch_npu 微基准测试 → 按需采集这些 Shape 的性能         │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 模块结构

数据库查询逻辑放在 `tensor_cast/performance_model/empirical/` 下统一管理，数据采集流水线作为独立子系统放在仓库顶层 `perf_database/` 目录。

```
tensor_cast/performance_model/
├── __init__.py                        # PerformanceModel, OpInvokeInfo, CachingPerformanceModel（现有）
├── analytic.py                        # AnalyticPerformanceModel（现有）
├── comm_analytic.py                   # CommAnalyticModel（现有）
├── empirical.py                       # EmpiricalPerformanceModel（扩展入口）
├── op_benchmark.py                    # OpBenchmark（现有 JIT 基准测试）
├── memory_tracker.py                  # MemoryTracker（现有）
└── empirical/                         # 新增：数据库支持子包
    ├── __init__.py
    ├── data_source.py                 # DataSource 抽象基类 + DatabaseSource, CachedSource
    ├── database.py                    # PerfDatabase 主类（加载、缓存、查询委托）
    ├── query_engine.py                # QueryEngine（精确匹配 + 插值 + 外推）
    ├── storage.py                     # CSV/Parquet IO 后端
    ├── versioning.py                  # 版本解析与管理（VersionManager）
    └── schemas/                       # OperatorSchema 注册表
        ├── __init__.py                # 通过 importlib 自动发现
        ├── base.py                    # OperatorSchema 抽象基类 + 注册表
        ├── gemm.py                    # GEMM/MatMul Schema
        ├── attention.py               # MHA、GQA、MLA Schema
        ├── moe.py                     # MoE 路由 + 专家计算 Schema
        ├── communication.py           # allreduce、allgather、alltoall Schema
        ├── normalization.py           # RMSNorm、AddRmsNorm Schema
        └── fused.py                   # 融合算子 Schema（DequantSwigluQuant 等）

perf_database/                         # 独立子系统：数据采集流水线
├── scripts/
│   ├── collect_full_model.py          # Level 1: 基于 VLLM 的全模型 Profiling
│   ├── collect_microbench.py          # Level 2: torch_npu 微基准测试
│   ├── parse_ascend_output.py         # 解析 kernel_details.csv
│   ├── generate_shape_grid.py         # 根据模型配置 / 仿真结果生成 Shape 网格
│   ├── calibrate.py                   # L1 与 L2 校准对齐
│   ├── validate.py                    # 数据库精度验证
│   └── discover_operators.py          # 新模型算子发现
├── mappings/                          # VLLM 内核 → TensorCast 算子映射表
│   └── vllm_ascend/
│       ├── v0.11.yaml
│       ├── v0.12.yaml
│       └── v0.13.yaml
└── data/                              # 性能数据存储（.gitignore）
    └── atlas_a3_752t_128g/
        └── vllm_ascend/{version}/
            ├── metadata.yaml          # 采集环境信息
            ├── gemm.csv
            ├── attention.csv
            └── ...
```

---

## 4. 核心模块设计

### 4.1 DataSource 抽象层

扩展 `EmpiricalPerformanceModel` 的核心设计。每种 DataSource 实现不同的数据获取策略，`EmpiricalPerformanceModel` 按优先级依次查询。

```python
# tensor_cast/performance_model/empirical/data_source.py

class DataSource(ABC):
    """数据源抽象基类，与现有 OpBenchmarkBase 设计一致。"""

    @abstractmethod
    def query(self, op_invoke_info: OpInvokeInfo) -> Optional[PerformanceModel.Result]:
        """
        尝试从该数据源获取算子性能估算。
        返回 None 表示该数据源无法处理此算子，由下一个数据源接管。
        """
        ...

class DatabaseSource(DataSource):
    """从预构建的性能数据库查询。"""

    def __init__(self, database: PerfDatabase):
        self.database = database
        self._schema_map = self._build_op_to_schema_map()

    def query(self, op_invoke_info: OpInvokeInfo) -> Optional[PerformanceModel.Result]:
        schema = self._schema_map.get(str(op_invoke_info.func))
        if schema is None:
            return None  # 该算子不在数据库覆盖范围内
        try:
            shape = schema.extract_shape_from_op(op_invoke_info)
            result = self.database.query(schema.name, shape)
            return PerformanceModel.Result(
                execution_time_s=result.latency_us * 1e-6,
                statistics={"source": result.source.name, "confidence": result.confidence}
            )
        except Exception:
            return None  # 查询失败（如 Shape 超出插值范围），由下一个数据源处理

class BenchmarkSource(DataSource):
    """现有 JIT 基准测试，封装 OpBenchmark。"""

    def __init__(self, op_benchmark: OpBenchmark):
        self.op_benchmark = op_benchmark

    def query(self, op_invoke_info: OpInvokeInfo) -> Optional[PerformanceModel.Result]:
        try:
            return self.op_benchmark.benchmark(op_invoke_info)
        except Exception:
            return None

class CachedSource(DataSource):
    """从本地文件缓存读取之前的执行结果。"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self._cache: Dict[str, float] = {}

    def query(self, op_invoke_info: OpInvokeInfo) -> Optional[PerformanceModel.Result]:
        if op_invoke_info.cache_key in self._cache:
            return PerformanceModel.Result(
                execution_time_s=self._cache[op_invoke_info.cache_key],
                statistics={"source": "cached"}
            )
        return None
```

### 4.2 EmpiricalPerformanceModel 扩展

```python
# tensor_cast/performance_model/empirical.py（扩展后）

class EmpiricalPerformanceModel(PerformanceModel):
    """
    基于实测数据的性能模型。

    扩展设计：通过 DataSource 优先级链查询算子性能。
    - 继承 PerformanceModel（tensor_cast/performance_model/__init__.py:169）
    - 实现 process_op(OpInvokeInfo) → PerformanceModel.Result
    - 通过 perf_models 列表接入 Runtime（tensor_cast/runtime.py:41）
    - 由 Runtime 自动包装为 CachingPerformanceModel
    """

    def __init__(
        self,
        device_profile: DeviceProfile,
        data_sources: Optional[List[DataSource]] = None,
    ):
        super().__init__("empirical", device_profile)

        if data_sources is not None:
            self.data_sources = data_sources
        else:
            # 默认行为：仅 JIT 基准测试（保持后向兼容）
            self.data_sources = [BenchmarkSource(OpBenchmark(device_profile))]

        # 内部 Roofline 兜底（用于未被任何 DataSource 覆盖的算子）
        self._fallback = AnalyticPerformanceModel(device_profile)

    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        # 按优先级依次查询各 DataSource
        for source in self.data_sources:
            result = source.query(op_invoke_info)
            if result is not None:
                return result

        # 所有 DataSource 均无法处理，回退至 Roofline
        return self._fallback.process_op(op_invoke_info)

    def get_classifiers(self) -> List[PerformanceModel.OpClassifier]:
        return self._fallback.get_classifiers()
```

**使用示例**：
```python
# 仅使用数据库（无物理设备）
db = PerfDatabase(system="atlas_a3_752t_128g", backend="vllm_ascend", version="0.13.0")
perf_model = EmpiricalPerformanceModel(
    device_profile,
    data_sources=[DatabaseSource(db)]
)

# 数据库优先，JIT 兜底（有物理设备时）
perf_model = EmpiricalPerformanceModel(
    device_profile,
    data_sources=[DatabaseSource(db), BenchmarkSource(OpBenchmark(device_profile))]
)

# 后向兼容：不传 data_sources 时与现有行为一致
perf_model = EmpiricalPerformanceModel(device_profile)
```

### 4.3 OperatorSchema（插件化架构）

每种算子类型定义一个 Schema，负责：
1. 声明该算子的 Shape 维度空间（哪些维度影响性能）
2. 提供从 `OpInvokeInfo` 提取 Shape 的逻辑（连接 TensorCast 运行时）
3. 声明该 Schema 处理的 TensorCast 算子名称列表（映射表）

```python
# tensor_cast/performance_model/empirical/schemas/base.py

class DimensionType(Enum):
    """维度类型，用于区分哪些维度适合连续插值。"""
    BATCH = auto()      # 随部署变化（batch_size、num_tokens）— 适合插值
    SEQUENCE = auto()   # 随输入变化（query_len、context_len）— 适合插值
    FEATURE = auto()    # 模型固有参数（hidden_size、head_dim）— 离散匹配
    EXPERT = auto()     # MoE 专有（num_experts、top_k）— 离散匹配
    QUANT = auto()      # 量化模式（int8、fp8）— 离散匹配

@dataclass
class DimensionSpec:
    name: str
    dim_type: DimensionType
    typical_range: Optional[Tuple[int, int]] = None

class OperatorSchema(ABC):
    """
    算子性能 Schema 基类。
    采用注册表模式自动发现（与 DeviceProfile、OpInvokeInfo.register_op_properties 设计一致）。
    """
    _registry: ClassVar[Dict[str, "OperatorSchema"]] = {}
    _op_to_schema: ClassVar[Dict[str, "OperatorSchema"]] = {}

    def __init_subclass__(cls, **kwargs):
        """自动注册子类实例。"""
        super().__init_subclass__(**kwargs)
        if not getattr(cls, '__abstractmethods__', None):
            instance = cls()
            OperatorSchema._registry[instance.name] = instance
            for op_name in instance.tensorcast_ops:
                OperatorSchema._op_to_schema[op_name] = instance

    @classmethod
    def get_schema_for_op(cls, op_name: str) -> Optional["OperatorSchema"]:
        """根据 TensorCast 算子名称查找对应的 Schema。"""
        return cls._op_to_schema.get(op_name)

    @property
    @abstractmethod
    def name(self) -> str:
        """Schema 名称，对应数据文件名（如 "gemm" → gemm.csv）。"""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> List[DimensionSpec]:
        """该算子的 Shape 维度定义。"""
        ...

    @property
    @abstractmethod
    def tensorcast_ops(self) -> List[str]:
        """该 Schema 处理的 TensorCast 算子名称列表。"""
        ...

    @abstractmethod
    def extract_shape_from_op(self, op_invoke_info: OpInvokeInfo) -> Dict[str, Any]:
        """从拦截到的算子调用中提取 Shape 维度字典。"""
        ...

    def get_interpolation_dimensions(self) -> List[str]:
        """返回适用于连续插值的维度（BATCH 和 SEQUENCE 类型）。"""
        return [d.name for d in self.dimensions
                if d.dim_type in (DimensionType.BATCH, DimensionType.SEQUENCE)]

    def get_discrete_dimensions(self) -> List[str]:
        """返回需要精确匹配的离散维度（FEATURE、EXPERT、QUANT 类型）。"""
        return [d.name for d in self.dimensions
                if d.dim_type in (DimensionType.FEATURE, DimensionType.EXPERT, DimensionType.QUANT)]
```

**具体 Schema 一览**：

| Schema | 名称 | 关键维度 | 对应 TensorCast 算子 |
|--------|------|---------|---------------------|
| `GEMMSchema` | `gemm` | m, n, k, quant_mode | `aten.mm.default`、`tensor_cast.static_quant_linear.default`、`tensor_cast.static_quant_linear_int4.default`、`tensor_cast.fp8_linear.default`、`tensor_cast.mxfp4_linear.default` |
| `AttentionSchema` | `attention` | batch, query_len, context_len, num_heads, num_kv_heads, head_dim | `tensor_cast.attention.default`、`tensor_cast.attention_quant.default` |
| `MLASchema` | `mla` | batch, query_len, context_len, num_heads, kv_lora_rank, qk_rope_head_dim | `tensor_cast.multihead_latent_attention.default`、`tensor_cast.multihead_latent_attention_quant.default` |
| `MoERoutingSchema` | `moe_routing` | num_tokens, num_experts, top_k | `tensor_cast.permute_tokens.default`、`tensor_cast.unpermute_tokens.default` |
| `GroupedGEMMSchema` | `grouped_gemm` | num_tokens, hidden_size, intermediate_size, num_experts, quant_mode | `tensor_cast.grouped_matmul.default`、`tensor_cast.grouped_matmul_quant.default`、`tensor_cast.grouped_matmul_fp8.default` |
| `NormalizationSchema` | `normalization` | num_tokens, hidden_size | `tensor_cast.rms_norm.default`、`tensor_cast.add_rms_norm.default` 及其量化变体 |
| `CommunicationSchema` | `communication` | op_type, num_devices, message_bytes | `tensor_cast.all_reduce.default`、`tensor_cast.all_gather.default`、`tensor_cast.all_to_all.default` |
| `FusedSchema` | `fused` | （随融合内核而异） | 融合算子映射至其组成的基础算子 |

**GEMM Schema 示例实现**：
```python
# tensor_cast/performance_model/empirical/schemas/gemm.py

class GEMMSchema(OperatorSchema):
    @property
    def name(self) -> str:
        return "gemm"

    @property
    def dimensions(self) -> List[DimensionSpec]:
        return [
            DimensionSpec("m", DimensionType.BATCH),        # token 维度，适合插值
            DimensionSpec("n", DimensionType.FEATURE),       # 模型参数，离散匹配
            DimensionSpec("k", DimensionType.FEATURE),       # 模型参数，离散匹配
            DimensionSpec("quant_mode", DimensionType.QUANT),# 量化模式，离散匹配
        ]

    @property
    def tensorcast_ops(self) -> List[str]:
        return [
            "aten.mm.default",
            "tensor_cast.static_quant_linear.default",
            "tensor_cast.static_quant_linear_int4.default",
            "tensor_cast.fp8_linear.default",
            "tensor_cast.mxfp4_linear.default",
        ]

    def extract_shape_from_op(self, op_invoke_info: OpInvokeInfo) -> Dict[str, Any]:
        func_name = str(op_invoke_info.func)
        if "aten.mm" in func_name:
            x, w = op_invoke_info.args[0], op_invoke_info.args[1]
            return {"m": x.shape[0], "k": x.shape[1], "n": w.shape[1], "quant_mode": "fp16"}
        elif "static_quant_linear" in func_name:
            x, w = op_invoke_info.args[0], op_invoke_info.args[1]
            quant = "int4" if "int4" in func_name else "int8"
            return {"m": x.shape[0], "k": x.shape[1], "n": w.shape[1], "quant_mode": quant}
        elif "fp8_linear" in func_name:
            x, w = op_invoke_info.args[0], op_invoke_info.args[1]
            return {"m": x.shape[0], "k": x.shape[1], "n": w.shape[1], "quant_mode": "fp8"}
        elif "mxfp4_linear" in func_name:
            x, w = op_invoke_info.args[0], op_invoke_info.args[1]
            return {"m": x.shape[0], "k": x.shape[1], "n": w.shape[1], "quant_mode": "mxfp4"}
        raise ValueError(f"Unsupported op: {func_name}")
```

### 4.4 融合算子处理机制

VLLM-Ascend 会将多个基础算子融合为单一内核执行（如 `DequantSwigluQuant` = dequantize + silu + mul + quantize）。处理方式分两种：

**方式一：直接建模融合内核**（推荐用于 Tier 1 融合算子）

对于 Profiling 中耗时占比较高的融合内核（如 `AddRmsNorm`、`DequantSwigluQuant`），直接设计专门的 Schema，采集融合后的整体性能数据。TensorCast 仿真中对应的多个基础算子（如 `add` + `rms_norm`）的耗时被替换为融合内核的实测耗时。

**方式二：组合基础算子**（用于 Tier 2/3 融合算子）

对于不常见或新出现的融合内核，将其拆解为组成的基础算子分别查询，取总和作为估算值。

映射关系通过 YAML 配置管理：

```yaml
# perf_database/mappings/vllm_ascend/v0.13.yaml
version: "0.13"
device: ATLAS_800_A3_752T_128G_DIE

# TensorCast 算子 → Schema 映射
tensorcast_op_schemas:
  "aten.mm.default": gemm
  "tensor_cast.static_quant_linear.default": gemm
  "tensor_cast.static_quant_linear_int4.default": gemm
  "tensor_cast.fp8_linear.default": gemm
  "tensor_cast.attention.default": attention
  "tensor_cast.attention_quant.default": attention
  "tensor_cast.multihead_latent_attention.default": mla
  "tensor_cast.multihead_latent_attention_quant.default": mla
  "tensor_cast.permute_tokens.default": moe_routing
  "tensor_cast.unpermute_tokens.default": moe_routing
  "tensor_cast.grouped_matmul.default": grouped_gemm
  "tensor_cast.grouped_matmul_quant.default": grouped_gemm
  "tensor_cast.rms_norm.default": normalization
  "tensor_cast.add_rms_norm.default": normalization
  "tensor_cast.all_reduce.default": communication
  "tensor_cast.all_gather.default": communication
  "tensor_cast.all_to_all.default": communication

# 融合内核映射（用于 Profiling 数据解析）
fused_kernel_mappings:
  AddRmsNorm:
    schema: normalization      # 直接建模
    replaces: [add, rms_norm]
  DequantSwigluQuant:
    schema: fused_swiglu       # 直接建模
    replaces: [dequantize, silu, mul, quantize]
  split_qkv_rmsnorm_rope_kernel:
    components: [rms_norm, qkv_split, apply_rope]  # 拆解为基础算子
```

### 4.5 PerfDatabase

性能数据库负责数据加载、缓存管理和查询委托。**核心设计原则：与数据来源解耦**。数据库定义标准化的数据格式和查询接口，可以接入来自 micro-benchmark、全模型 Profiling 或其他任何工具产出的数据。

```python
# tensor_cast/performance_model/empirical/database.py

class PerfDatabase:
    """
    性能数据库主类。

    职责：
    - 按 (device, backend, version) 三元组定位数据目录
    - 延迟加载并缓存算子性能数据（CSV/Parquet）
    - 委托 QueryEngine 执行查询（精确匹配 + 插值）

    数据格式约定：
    - 每个 Schema 对应一个数据文件（如 gemm.csv、attention.csv）
    - CSV 列 = Schema.dimensions 定义的维度名 + "latency_us" 列
    - 离散维度（quant_mode 等）作为过滤条件
    - 连续维度（m、batch 等）用于插值
    """

    def __init__(
        self,
        system: str = "atlas_a3_752t_128g",
        backend: str = "vllm_ascend",
        version: str = "latest",
        data_root: Optional[Path] = None,
    ):
        self.version_mgr = VersionManager(data_root or self._default_data_root())
        self.resolved_version, self.data_path = self.version_mgr.resolve(system, backend, version)
        self.query_engine = QueryEngine()
        self._data_cache: Dict[str, Dict] = {}  # schema_name → nested dict

    def query(self, schema_name: str, shape: Dict[str, Any]) -> QueryResult:
        """主查询入口。"""
        data = self._load_data(schema_name)
        if data is None:
            raise PerfDataNotAvailableError(f"No data for schema: {schema_name}")
        schema = OperatorSchema._registry[schema_name]
        return self.query_engine.query(schema, data, shape)

    def _load_data(self, schema_name: str) -> Optional[Dict]:
        """延迟加载并缓存。将 CSV 数据加载为嵌套字典结构（参考 AI Configurator）。"""
        if schema_name not in self._data_cache:
            path = self.data_path / f"{schema_name}.csv"
            if not path.exists():
                return None
            self._data_cache[schema_name] = self._csv_to_nested_dict(path, schema_name)
        return self._data_cache[schema_name]
```

**数据文件格式示例（gemm.csv）**：
```csv
quant_mode,m,n,k,latency_us
fp16,1,4096,5120,12.5
fp16,8,4096,5120,13.2
fp16,32,4096,5120,18.7
int8,1,4096,5120,8.1
int8,8,4096,5120,8.9
fp8,1,4096,5120,7.3
...
```

### 4.6 QueryEngine（精确匹配 + 插值）

`EmpiricalPerformanceModel` 内部的 QueryEngine **仅处理实测数据**：精确匹配 → 插值 → 外推。对于未收录的算子，不在 QueryEngine 内做 Roofline 兜底，而是返回查询失败，由 `EmpiricalPerformanceModel.process_op()` 的 DataSource 链或 Roofline fallback 处理。

```python
# tensor_cast/performance_model/empirical/query_engine.py

class QuerySource(Enum):
    MEASURED = auto()          # 精确匹配（置信度: 1.0）
    INTERPOLATED = auto()      # 插值估算（置信度: 0.7-0.95）
    EXTRAPOLATED = auto()      # 凸包外推（置信度: 0.3-0.6）

@dataclass
class QueryResult:
    latency_us: float
    confidence: float
    source: QuerySource
    details: Dict[str, Any] = field(default_factory=dict)

class QueryEngine:
    """处理精确查找和插值估算。"""

    def query(self, schema: OperatorSchema, data: Dict, shape: Dict) -> QueryResult:
        # 分离离散维度和连续维度
        discrete_dims = schema.get_discrete_dimensions()
        interp_dims = schema.get_interpolation_dimensions()

        # Step 1: 按离散维度过滤到子集
        subset = self._filter_by_discrete(data, shape, discrete_dims)
        if subset is None:
            raise PerfDataNotAvailableError(f"No data for discrete dims: {shape}")

        # Step 2: 在子集上尝试精确匹配
        result = self._try_exact(subset, shape, interp_dims)
        if result is not None:
            return result

        # Step 3: 在子集上尝试插值
        result = self._try_interpolate(subset, shape, interp_dims)
        if result is not None:
            return result

        # Step 4: 外推（最近邻 + 线性外推）
        return self._try_extrapolate(subset, shape, interp_dims)

    def _try_exact(self, data, shape, dims) -> Optional[QueryResult]:
        """O(1) 嵌套字典精确查找。"""
        try:
            node = data
            for dim in dims:
                node = node[shape[dim]]
            return QueryResult(
                latency_us=node["latency_us"],
                confidence=1.0,
                source=QuerySource.MEASURED
            )
        except KeyError:
            return None

    def _try_interpolate(self, data, shape, dims) -> Optional[QueryResult]:
        """
        逐轴 1D 线性插值（参考 AI Configurator 的 _interp_3d 设计）。
        对每个维度找到最近的左右邻居，逐维度进行线性插值。
        置信度根据到最近测量点的相对距离估算。
        """
        ...
```

---

## 5. TensorCast 集成接口

### 5.1 Runtime 集成

`Runtime` 类无需任何修改。它已支持接收 `PerformanceModel` 实例列表，并自动包装为 `CachingPerformanceModel`：

```python
# tensor_cast/runtime.py（现有代码，无需改动）
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

使用扩展后的 EmpiricalPerformanceModel：
```python
# 在 model_runner.py 或 config_resolver.py 中：
db = PerfDatabase(system="atlas_a3_752t_128g", backend="vllm_ascend", version="0.13.0")
perf_model = EmpiricalPerformanceModel(
    device_profile,
    data_sources=[DatabaseSource(db)]
)
runtime = Runtime(perf_models=perf_model, device_profile=device_profile)
```

### 5.2 CLI 接口

在 `tensor_cast/scripts/text_generate.py` 中新增参数：

```python
parser.add_argument("--performance-model", choices=["analytic", "empirical"],
                    default="analytic", help="性能模型类型")
parser.add_argument("--database-version", type=str, default="latest",
                    help="性能数据库版本号（仅 empirical 模式生效）")
```

用户选择 `--performance-model empirical` 时，系统自动创建 `DatabaseSource` 并加载对应版本的性能数据库。查询策略（精确匹配 → 插值 → Roofline 兜底）自动执行，用户无需关心降级细节。

### 5.3 数据流

```
用户 CLI                     TensorCast                    数据库
────────                     ──────────                    ──────
text_generate.py
  --performance-model empirical
  --database-version 0.13.0
       │
       ▼
  创建 EmpiricalPerformanceModel
  + DatabaseSource(PerfDatabase)
       │
       ▼
  Runtime.__torch_dispatch__
  拦截每个算子调用
       │
       ▼
  OpInvokeInfo(func, args, kwargs, out)
       │
       ▼
  EmpiricalPerformanceModel.process_op()
       │
       ├──▶ DatabaseSource.query()
       │    ├──▶ OperatorSchema.get_schema_for_op("aten.mm.default")
       │    │    → GEMMSchema
       │    ├──▶ schema.extract_shape_from_op(op_invoke_info)
       │    │    → {m: 136, n: 4096, k: 5120, quant_mode: "int8"}
       │    ├──▶ database.query("gemm", shape)
       │    │    ├──▶ 精确匹配？ → 返回 QueryResult
       │    │    ├──▶ 插值估算？ → 返回 QueryResult
       │    │    └──▶ 外推估算？ → 返回 QueryResult
       │    └──▶ 返回 PerformanceModel.Result
       │
       ├──▶ 若 DatabaseSource 返回 None（算子未收录）
       │    └──▶ Roofline 兜底（内部 AnalyticPerformanceModel）
       │
       └──▶ 返回 PerformanceModel.Result(execution_time_s=...)
```

---

## 6. 自动化 Profiling 流水线

> **注意**：本节描述的数据采集流水线作为独立子系统，代码位于仓库顶层 `perf_database/` 目录，不包含在 TensorCast 包内。

### 6.1 数据库构建策略

用户可根据可用资源选择不同的数据库构建方式：

| 策略 | 适用场景 | 优势 | 劣势 |
|------|---------|------|------|
| **方案 A**（默认）：全模型 Profiling + 微基准测试 | 有完整 VLLM 部署环境 | 精确捕获融合内核，校准数据可靠 | VLLM 服务启动开销大 |
| **方案 B**：仿真驱动 + 微基准测试 | 仅有裸机 NPU 环境 | 无需 VLLM 部署，采集效率高 | 无法覆盖融合内核 |
| **方案 C**：纯 Profiling 导入 | 已有现成的 Profiling 数据 | 零额外采集成本 | Shape 覆盖受限 |

### 6.2 两级采集策略（方案 A）

#### Level 1: 全模型 Profiling（获取基准真值）

启用昇腾 Profiler 运行 VLLM 推理，捕获真实融合内核的执行耗时。

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

#### Level 2: 单算子微基准测试（Shape 网格填充）

通过 `torch_npu` 直接对单个算子进行细粒度 Shape 覆盖测试：

```python
# perf_database/scripts/collect_microbench.py
import torch, torch_npu

def benchmark_matmul(m, n, k, dtype, warmup=5, runs=20):
    a = torch.randn(m, k, dtype=dtype, device='npu')
    b = torch.randn(k, n, dtype=dtype, device='npu')
    for _ in range(warmup):
        torch.mm(a, b)
    torch.npu.synchronize()
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
| 融合内核捕获 | 完整捕获（基准真值） | 无法捕获 |
| Shape 控制 | 间接控制（batch、seq_len） | 直接控制（m、n、k） |
| 执行开销 | 高（服务启动耗时长） | 低（仅内核执行） |
| 覆盖范围 | 每个模型约 100 种配置 | 每个算子约 2500 种 Shape |
| 适用场景 | 验证与校准 | Shape 网格大规模填充 |

**校准方法**：在 Shape 重叠区域对比 Level 1 和 Level 2 的数据，计算每类算子的修正系数，使 Level 2 数据对齐实际部署场景。

### 6.3 Profiling 输出解析器

```python
# perf_database/scripts/parse_ascend_output.py

class AscendProfilerParser:
    def parse_kernel_details(self, csv_path: Path) -> pd.DataFrame:
        """解析 kernel_details.csv → 结构化 DataFrame"""
        # 处理灵活的列名格式和缺失字段
        # 返回: name, duration_us, input_shapes, data_types,
        #       aicore_time_us, aiv_time_us, cube_utilization_pct

    def extract_single_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """以 Attention 内核为边界提取单个 Decode/Prefill 步骤"""

    def map_to_schema(self, kernel_name: str, input_shapes: str,
                      mapping_yaml: Path) -> Tuple[str, Dict]:
        """将 VLLM 内核名映射到 (schema_name, shape_dict)"""
        # "MatMulV2" + "136,4096; 4096,4096" → ("gemm", {m:136, k:4096, n:4096})
        # 使用版本相关的 YAML 映射配置
```

### 6.4 新模型算子发现机制

```python
# perf_database/scripts/discover_operators.py

def discover_operators(profiling_output: Path, mapping_yaml: Path) -> Dict:
    """发现 Profiling 中存在但当前映射表中缺失的算子"""
    parser = AscendProfilerParser()
    kernels = parser.parse_kernel_details(profiling_output / "kernel_details.csv")

    known_ops = load_yaml_mappings(mapping_yaml)
    unknown = []

    for kernel_name in kernels["name"].unique():
        if not any(kernel_name in schema_kernels for schema_kernels in known_ops.values()):
            suggested_schema = auto_classify(kernel_name)
            unknown.append({"kernel": kernel_name, "suggested_schema": suggested_schema})

    return {"known": len(kernels) - len(unknown), "unknown": unknown}
```

---

## 7. 全面算子覆盖策略

### 7.1 算子分级

| 层级 | 判定标准 | 处理方式 | 算子数量 |
|-----|---------|---------|---------|
| **Tier 1** | 执行耗时占比 >2% | 必须使用完整 Shape 网格进行 Profiling | 约 11 个 |
| **Tier 2** | 执行耗时占比 0.5-2% | 使用精简 Shape 网格进行 Profiling | 约 8 个 |
| **Tier 3** | 执行耗时占比 <0.5% | 采用 Roofline 兜底估算 | 60+ 个 |

### 7.2 Tier 1 完整算子列表

| VLLM 内核名称 | 数据库 Schema | Qwen3 占比 | DSV3 占比 |
|-------------|-------------|-----------|---------|
| MatMulV2 | gemm | 42.4% | - |
| GroupedMatmul | grouped_gemm | - | 20.9% |
| QuantBatchMatmulV3 | gemm | - | 16.9% |
| FusedInferAttentionScore | attention / mla | 18.2% | 18.5% |
| TensorMove | memory_move | 10.7% | ~3% |
| AddRmsNorm / InplaceAddRmsNorm | normalization | 8.0% | 2.0% |
| split_qkv_rmsnorm_rope_kernel | fused | 5.2% | - |
| SwiGlu / DequantSwigluQuant | fused | 4.8% | 2.7% |
| ReshapeAndCacheNdKernel | cache | 3.3% | - |
| MoeDistributeDispatch/Combine | moe_routing | - | 11.8% |
| AscendQuantV2 / DynamicQuant | quantization | - | 5.5% |
| TransposeBatchMatMul | gemm | - | 4.2% |
| InterleaveRope | rope | - | 2.8% |

### 7.3 Shape 网格策略

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

## 8. 开发计划

当前计划由两组团队分工协作：
- **六壬工具团队**：TensorCast 内部接口（DataSource 抽象、EmpiricalPerformanceModel 扩展、OperatorSchema、QueryEngine）
- **小巧灵团队**：性能数据库和数据采集流水线（perf_database/、CSV 数据生成、VLLM Profiling 脚本）

**关键接口约定**：两组团队共享 `OperatorSchema` 定义和 CSV 数据格式规范，确保模块无缝对接。

**目标**：Q3（2026.3.20）完全跑通，在 DeepSeek-V3、Qwen3-32B 上完成初始数据采集和端到端集成测试。

### 阶段一：接口定义与核心基础设施（2 周，~1500 行）

**六壬团队**：
- 实现 `DataSource` 抽象基类和 `DatabaseSource`（~200 行）
- 扩展 `EmpiricalPerformanceModel`，保持后向兼容（~100 行）
- 实现 `OperatorSchema` 基类、注册表机制和自动发现（~300 行）
- 实现 `GEMMSchema`、`AttentionSchema`、`MLASchema`（~400 行）

**小巧灵团队**：
- 设计 CSV 数据格式规范，编写示例数据文件
- 实现 `PerfDatabase`（CSV 加载、嵌套字典构建）（~300 行）
- 实现精确匹配查询和 `VersionManager`（~200 行）

### 阶段二：插值引擎与数据采集（2 周，~2000 行）

**六壬团队**：
- 实现 `QueryEngine`（逐轴 1D 插值、置信度评分）（~500 行）
- 集成 Roofline 内部兜底逻辑（~100 行）
- 在 `text_generate.py` 中添加 CLI 参数（~50 行）

**小巧灵团队**：
- 实现 `generate_shape_grid.py`（模型驱动 + 通用网格）（~300 行）
- 实现 `collect_full_model.py`（Level 1 编排脚本）（~400 行）
- 实现 `collect_microbench.py`（Level 2 torch_npu 微基准测试）（~300 行）
- 实现 `parse_ascend_output.py`（kernel_details.csv 解析器）（~400 行）

### 阶段三：数据采集与集成测试（2 周）

**联合**：
- 完成 GEMM 和 Attention 在 A3 上的初始数据采集
- 实现 `calibrate.py`（L1 与 L2 校准对齐）
- 使用 Qwen3-32B 和 DeepSeek-V3 进行端到端测试
- 与实际 VLLM Profiling 对比验证精度

### 阶段四：扩展算子与精度验证（持续迭代）

- 新增 MoE、归一化、融合算子、缓存算子的 Schema
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

## 9. 建议与 Future Work

### 9.1 CompositePerformanceModel 顶层调度器

当前方案中 `EmpiricalPerformanceModel` 内部兜底回退至 `AnalyticPerformanceModel`，这是 v1.0 的务实做法。长期来看，建议在 TensorCast 框架层面设计一个 `CompositePerformanceModel`：

```python
class CompositePerformanceModel(PerformanceModel):
    """
    顶层性能模型调度器。
    按优先级组合多个 PerformanceModel，实现可配置的降级策略。
    """
    def __init__(self, models: List[PerformanceModel], strategy: str = "first_valid"):
        self.models = models
        self.strategy = strategy

    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        for model in self.models:
            try:
                result = model.process_op(op_invoke_info)
                if result.execution_time_s > 0:
                    return result
            except Exception:
                continue
        raise RuntimeError("All performance models failed")
```

**使用方式**：
```python
composite = CompositePerformanceModel([
    EmpiricalPerformanceModel(device_profile, data_sources=[DatabaseSource(db)]),
    AnalyticPerformanceModel(device_profile),
])
runtime = Runtime(perf_models=composite, device_profile=device_profile)
```

这样 `EmpiricalPerformanceModel` 不再需要内部持有 `AnalyticPerformanceModel` 引用，实现了性能模型之间的完全解耦。

### 9.2 其他 Future Work

- **跨硬件泛化**：支持更多 DeviceProfile（如 Atlas A2、GPU），需为每种硬件独立采集数据
- **自动化 CI**：随 VLLM-Ascend / CANN 版本发布自动触发数据采集流水线
- **能耗建模**：参考 AI Configurator 的 `PerformanceResult(float)` 模式，在 `PerformanceModel.Result` 中增加 energy 字段
- **预插值优化**：参考 AI Configurator 的 `_extrapolate_data_grid`，在数据库加载时预填充常用网格点，减少运行时插值开销
- **SOL 数据校正**：用 Roofline 理论下界校正异常测量值，确保 `measured >= SOL`

---

## 10. 依赖项

**新增**：`scipy`（插值）、`packaging`（版本解析）
**现有**：`pandas`、`numpy`、`pyyaml`、`torch`
**可选**：`pyarrow`（Parquet 格式支持，初期可仅用 CSV）

---

## 11. 参考资料

- [vLLM Ascend GitHub](https://github.com/vllm-project/vllm-ascend)
- [vLLM Ascend 发布说明](https://docs.vllm.ai/projects/ascend/en/main/user_guide/release_notes.html)
- [vLLM Ascend GitHub Releases](https://github.com/vllm-project/vllm-ascend/releases)
- [vLLM Ascend Profiling 指南](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/service_profiling_guide.html)
- [华为昇腾 Profiler 文档](https://support.huaweicloud.com/intl/en-us/bestpractice-modelarts/modelarts_llm_infer_5906034.html)
- [AI Configurator](https://github.com/ai-dynamo/aiconfigurator)（NVIDIA LLM 推理性能预估工具）
- [Intel NPU Cost Model](https://github.com/intel/npu-nn-cost-model)

---

## 附录：讨论会议纪要

1. **算子性能 load 函数设计**：AI Configurator 为每个算子类型提供独立查询接口（`query_gemm`、`query_attention` 等）。本方案为泛化性考虑，采用统一的 `PerfDatabase.query(schema_name, shape)` 接口，Shape 维度通过 `OperatorSchema.dimensions` 定义，数据库查询按输入 Shape 进行匹配。

2. **算子覆盖列表**：初步需支持 DeepSeek-V3、Qwen3-32B 在 910C、VLLM-Ascend 上的算子覆盖。需根据 Profiling 结果确定优先级（见第 7 节算子分级），并提供算子的 Input Shape 维度说明和接口配置文件（`perf_database/mappings/` 目录下的 YAML 文件）。

3. **VLLM 融合算子映射**：需进行 VLLM 融合算子到 TensorCast 算子的映射（可能一对多或多对一），映射表格式见第 4.4 节的 YAML 配置示例。TensorCast 的算子列表可通过运行仿真测试脚本获取。

4. **存储结构**：按 `{device}/{backend}/{version}/` 分目录存储，查询时通过 `(system, backend, version)` 三元组定位数据目录。

---

## Change Log

### v1.0 → v1.1（2026 年 2 月）

| 变更项 | 详细说明 |
|-------|---------|
| **新增第 1 节：功能概述** | 明确目标、核心功能、范围界定（做什么/不做什么），回应审核意见 |
| **命名统一** | 全文 `ProfilingPerformanceModel` → `EmpiricalPerformanceModel` 扩展，基于现有框架而非新建类 |
| **架构重设计** | 引入 `DataSource` 抽象层（`DatabaseSource`、`BenchmarkSource`、`CachedSource`），保持后向兼容 |
| **模块目录调整** | 数据库查询逻辑移至 `tensor_cast/performance_model/empirical/`；数据采集流水线移至顶层 `perf_database/` |
| **OperatorSchema 细化** | 按 TensorCast 算子粒度注册，每个 Schema 声明 `tensorcast_ops` 列表作为映射表；新增融合算子处理机制（直接建模 vs 拆解组合） |
| **PerfDatabase 解耦** | 明确数据库仅定义格式和接口，与数据来源（micro-benchmark / profiling / 手动录入）解耦 |
| **QueryEngine 简化** | EmpiricalPerformanceModel 内部仅处理实测数据（精确/插值/外推），不在 QueryEngine 内做 Roofline；未收录算子的 Roofline 兜底由上层处理 |
| **CLI 简化** | 移除 `--database-mode` 参数，用户无需关心降级策略细节 |
| **版本信息核实** | 基于 vLLM-Ascend 官方发布说明和 GitHub Releases 核实版本影响分析，更新版本号和日期 |
| **数据库构建多策略** | 新增方案 B（仿真驱动 + 微基准测试）和方案 C（纯 Profiling 导入），用户可按需选择 |
| **新增第 9 节：建议与 Future Work** | CompositePerformanceModel 顶层调度器设计建议、跨硬件泛化、CI 自动化、能耗建模、预插值优化等 |
| **开发计划更新** | 明确两组团队分工、Q3 倒排时间线、代码量估计；补充接口约定要求 |
