# TensorCast 算子性能数据库：技术设计文档

**版本**: 1.1
**日期**: 2026.2.12
**范围**: 面向 LLM 仿真的可扩展 Profiling Cost Model，不绑定具体算力卡，支持基于实测 Profiling 数据的算子性能估算。
**初期目标模型**: DeepSeek-V3、Qwen3-32B

作者：贺骁武
审核人：龚炯

---

## 1. 功能概述

### 1.1 目标

为 TensorCast 仿真器构建基于实测数据的算子性能估算系统。新增独立的 `ProfilingPerformanceModel`，与现有的 `EmpiricalPerformanceModel`（JIT 基准测试）和 `AnalyticPerformanceModel`（Roofline）并列，作为用户可选的第三种性能模型。

`ProfilingPerformanceModel` 和 `EmpiricalPerformanceModel`可共享 **`PerfDatabase` 数据层** 作为标准化的算子性能数据存储与查询接口。

### 1.2 核心功能（本方案范围）

1. **算子性能数据库（PerfDatabase）（新增）**：定义标准化数据格式和查询接口，支持按算子类型、Shape 维度、量化模式进行性能查询，与数据来源解耦（支持 micro-benchmark、全模型 Profiling 等多种数据源）
2. **ProfilingPerformanceModel（新增）**：独立的 `PerformanceModel` 子类，从预构建的 `PerfDatabase` 中查询算子耗时；对未收录算子内部兜底回退至 `AnalyticPerformanceModel`
3. **EmpiricalPerformanceModel 增强**：可选接入 `PerfDatabase` 作为 JIT benchmark 的持久化缓存，实现跨 session 的结果复用
4. **查询引擎**：实现精确匹配 → 插值估算 → 外推估算的多级查询策略
5. **数据采集流水线**（独立子系统，不在 TensorCast 包内）：自动化执行全模型 Profiling 和单算子微基准测试，解析输出并构建性能数据库

### 1.3 不在本方案范围内（建议后续支持）

- **CompositePerformanceModel 顶层调度器**：统一编排多种 PerformanceModel，实现可配置的降级或者组合策略（详见第 9 节）
- **跨硬件泛化**：当前仅支持昇腾 A3，其他硬件需独立采集数据
- **自动化持续集成**：随 VLLM-Ascend / CANN 版本发布自动触发数据采集

### 1.4 初期目标 （待对齐）

- **目标模型**：DeepSeek-V3、Qwen3-32B
- **目标硬件**：Atlas 800 A3（752T，128G DIE）
- **目标后端**：vllm-0.13.0 (内部镜像)
- **精度目标**：端到端仿真误差 <15%（对比实际 VLLM Profiling）
- **交付时间**：Q1（2026.3.20）完全跑通并完成初始数据采集和集成测试

---

## 2. 技术分析

### 2.1 问题陈述

TensorCast 当前采用**基于 Roofline 的解析模型**（`AnalyticPerformanceModel`）估算算子执行耗时。该模型基于浮点运算量（FLOPs）与访存字节数计算 `max(计算耗时, 访存耗时)`，主要用作理论性能上线评估，实际耗时可能存在偏差。

`EmpiricalPerformanceModel`（`tensor_cast/performance_model/empirical.py`）已提供对接实测数据的初步框架，当前仅通过 `OpBenchmark` 类在物理设备上 JIT 执行并计时

**本方案的核心思路**：新增独立的 `ProfilingPerformanceModel`，专职从预构建的性能数据库中查询算子耗时。同时抽取 `PerfDatabase` 作为共享数据层，供 `ProfilingPerformanceModel`（只读查询）和 `EmpiricalPerformanceModel`（读写缓存）共同使用。三种 PerformanceModel 保持独立，用户通过 CLI 配置选择。

**偏差根因分析**：

| 偏差来源 | 具体描述 | 影响程度 |
|---------|---------|---------|
| **算子融合** | VLLM 将多个算子融合为单一内核执行（如 `DequantSwigluQuant`、`AddRmsNorm`） | 10-30% 时间差异 |
| **硬件利用率** | 实际 Cube 利用率约 44-68%，而非 Roofline 假设的 100% | 高估计算密集型算子 |
| **Shape 相关开销** | 小 batch 场景下内核启动开销未建模 | 低估 Decode 阶段耗时 |
| **版本相关优化** | VLLM-Ascend 各版本引入不同的融合内核 | 模型在版本迭代中失效 |

### 2.2 Profiling 数据分析

基于实际昇腾 Profiler 输出（`kernel_details.csv`、`op_statistic.csv`）：

> **注意**：以下占比仅统计**计算内核**，不含通信内核（如 AllReduce）。与第 7.2 节包含通信的全量内核占比不同。

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
| v0.14.0rc1（预发布） | 2026.01.26 | MatMul-AllReduce-RMSNorm 融合 Pass（默认关闭）、310P 基础支持 |
| v0.12.0rc1 | 2025.12.13 | 大量 Triton 内核（Qwen3-Next, DeepSeek 3.2）、Full Decode-Only Graph Mode（实验性）、W4A4 量化支持 |
| v0.11.0 | 2025.12.16 | W8A16 量化、Full Graph Mode (ACLGraph) + GQA、多 Token 预测 + Chunked Prefill |

CANN 版本同样影响内核执行效率（如 CANN 8.5 针对 FIA 算子做了 flash decoding 优化），两者版本号均需纳入数据库版本追踪。

### 2.4 现有基础设施

**`EmpiricalPerformanceModel`**（已实现，位于 `tensor_cast/performance_model/empirical.py`）：当前仅 20 行代码，继承 `PerformanceModel`，通过 `OpBenchmark` 在物理设备上 JIT 执行并计时。

`OpBenchmark`（`op_benchmark.py`）提供了关键扩展点：
- `OpBenchmarkBase` 抽象基类定义了 `benchmark()` 接口
- 支持 meta tensor → real tensor 的自动转换
- `register_op_impl` 注册表为 TensorCast 自定义算子提供设备特定实现

### 2.5 AI Configurator 参考

参考 [AI Configurator](https://github.com/ai-dynamo/aiconfigurator) 项目（NVIDIA 的 LLM 推理性能预估工具）提炼的核心设计模式：

- **嵌套字典索引 + 混合插值**：对已 Profiling 的 Shape 实现 O(1) 精确查找（如 `gemm_data[quant_mode][m][n][k]`）；采用 2D+1D 混合插值策略（先对两个维度做双线性插值，再对第三个维度做 1D 插值），支持 sqrt 变换处理 O(n²) 复杂度的 Attention 序列维度
- **DatabaseMode 降级策略**：SILICON（仅实测数据）→ HYBRID（实测优先，回退经验值）→ EMPIRICAL（Roofline × 效率系数）→ SOL（纯理论峰值）
- **CSV 存储 + 延迟加载**：可读性强、Git-friendly，按 `systems/{device}/{backend}/{version}/` 分目录存储；全局 DB 缓存 + LRU 查询缓存
- **SOL 数据校正**：用理论下界校正异常测量值，确保 `measured >= SOL`

---

## 3. 系统架构

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TensorCast Runtime                                │
│                                                                             │
│  ┌────────────────────┐     ┌────────────────────────────────────────────┐  │
│  │  Runtime            │     │  用户可配置选择以下任一 PerformanceModel   │  │
│  │  (TorchDispatchMode)│────▶│                                            │  │
│  │                     │     │  ┌────────────────────────────────────┐    │  │
│  │  拦截所有算子调用   │     │  │ ProfilingPerformanceModel（新增）  │    │  │
│  │  生成 OpInvokeInfo  │     │  │  PerfDatabase 只读查询             │    │  │
│  └────────────────────┘     │  │  未命中 → fallback (Analytic)      │    │  │
│                              │  └────────────────────────────────────┘    │  │
│                              │  ┌────────────────────────────────────┐    │  │
│                              │  │ EmpiricalPerformanceModel（现有）  │    │  │
│                              │  │  可选: PerfDatabase 持久化缓存     │    │  │
│                              │  │  查缓存 → JIT benchmark → 写缓存  │    │  │
│                              │  └────────────────────────────────────┘    │  │
│                              │  ┌────────────────────────────────────┐    │  │
│                              │  │ AnalyticPerformanceModel（现有）   │    │  │
│                              │  │  Roofline 理论模型                 │    │  │
│                              │  └────────────────────────────────────┘    │  │
│                              └────────────────────────────────────────────┘  │
│                                           │                                  │
└───────────────────────────────────────────┼──────────────────────────────────┘
                                            │
                      ┌─────────────────────┼─────────────────────┐
                      │                     ▼                     │
                      │     PerfDatabase（共享数据层）             │
                      │  ┌──────────────────────────────────┐    │
                      │  │  QueryEngine                      │    │
                      │  │  精确匹配 → 插值 → 外推            │    │
                      │  └──────────────────────────────────┘    │
                      │  ┌──────────────────────────────────┐    │
                      │  │  OperatorSchema 注册表             │    │
                      │  │  OperatorKey 提取逻辑              │    │
                      │  │ ┌──────┐ ┌─────────┐ ┌───┐ ┌───┐ │    │
                      │  │ │ GEMM │ │Attention│ │MoE│ │...│ │    │
                      │  │ └──────┘ └─────────┘ └───┘ └───┘ │    │
                      │  └──────────────────────────────────┘    │
                      │  ┌──────────────────────────────────┐    │
                      │  │  存储层（按版本分目录，CSV/Parquet）│    │
                      │  │  data/{device}/{backend}/{version}│    │
                      │  └──────────────────────────────────┘    │
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

共享数据层 `PerfDatabase` 放在 `tensor_cast/performance_model/perf_database/` 下，与 `analytic.py`、`empirical.py` 并列。新增 `profiling.py` 作为 `ProfilingPerformanceModel` 的实现文件。数据采集流水线作为独立子系统放在仓库顶层 `perf_database/` 目录。

```
tensor_cast/performance_model/
├── __init__.py                        # PerformanceModel, OpInvokeInfo, CachingPerformanceModel（现有）
├── analytic.py                        # AnalyticPerformanceModel（现有）
├── profiling.py                       # ProfilingPerformanceModel（新增）
├── empirical.py                       # EmpiricalPerformanceModel（增强：可选 cache_db）
├── comm_analytic.py                   # CommAnalyticModel（现有）
├── op_benchmark.py                    # OpBenchmark（现有 JIT 基准测试）
├── memory_tracker.py                  # MemoryTracker（现有）
└── perf_database/                     # 新增：共享数据层
    ├── __init__.py
    ├── database.py                    # PerfDatabase 主类（加载、缓存、查询委托）
    ├── operator_key.py                # OperatorKey（OpInvokeInfo → 查询 key 转换）
    ├── query_engine.py                # QueryEngine（精确匹配 + 插值 + 外推）
    ├── storage.py                     # CSV/Parquet IO 后端
    └── schemas/                       # OperatorSchema 注册表
        ├── __init__.py                # 通过 importlib 自动发现
        ├── base.py                    # OperatorSchema 抽象基类 + 注册表
        ├── matmul.py                  # MatMulSchema, GroupedMatMulSchema, QuantBatchMatMulSchema, TransposeBatchMatMulSchema
        ├── attention.py               # FusedAttentionSchema
        ├── rope.py                    # InterleaveRopeSchema
        ├── moe.py                     # MoeDispatchSchema, MoeCombineSchema, MoeGatingSchema
        ├── communication.py           # AllReduceSchema, AllGatherSchema, AllToAllSchema
        ├── normalization.py           # AddRmsNormSchema
        ├── activation.py              # SwiGluSchema
        ├── quantization.py            # AscendQuantSchema
        ├── cache.py                   # ReshapeAndCacheSchema
        └── data_movement.py           # TensorMoveSchema

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
            ├── matmul.csv
            ├── fused_attention.csv
            └── ...
```

---

## 4. 核心模块设计

### 4.1 PerfDatabase（共享数据层）

性能数据库作为共享数据层，为 `ProfilingPerformanceModel`（只读查询）和 `EmpiricalPerformanceModel`（读写缓存）提供统一的数据存储与查询能力。

**核心设计原则：与数据来源解耦**。数据库定义标准化的数据格式和查询接口，可以接入来自 micro-benchmark、全模型 Profiling 或其他任何工具产出的数据。

```python
# tensor_cast/performance_model/perf_database/database.py

class PerfDatabase:
    """
    性能数据库主类。

    职责：
    - 按 data_path 定位数据目录，从 metadata.yaml 读取版本信息和 YAML 映射表路径
      metadata.yaml 包含 device、backend、version、mapping_yaml（指向 YAML 映射表的路径）、collection_date 等字段
    - 延迟加载并缓存算子性能数据（CSV/Parquet）
    - 提供 lookup() / store() / save() 接口
    - 委托 QueryEngine 执行查询（精确匹配 + 插值）

    数据格式约定：
    - 每个 Schema 对应一个数据文件（如 matmul.csv、fused_attention.csv）
    - CSV 列 = Schema.dimensions 定义的维度名 + "latency_us" 列
    - 离散维度（quant_mode 等）作为过滤条件
    - 连续维度（m、batch 等）用于插值
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        writable: bool = False,
    ):
        self.data_path = Path(data_path)
        self.query_engine = QueryEngine()
        self.writable = writable
        self._data_cache: Dict[str, Dict] = {}  # schema_name → nested dict
        # writable=True 时若 data_path 不存在或无 metadata.yaml，自动创建目录和默认 metadata
        if self.writable and not (self.data_path / "metadata.yaml").exists():
            self.data_path.mkdir(parents=True, exist_ok=True)
            self._create_default_metadata()
        self.metadata = self._load_metadata()
        self.op_mapping: Dict[str, str] = self._load_op_mapping()

    def lookup(self, key: "OperatorKey") -> Optional[QueryResult]:
        """查询算子性能数据。返回 None 表示无数据。"""
        data = self._load_data(key.schema_name)
        if data is None:
            return None
        schema = OperatorSchema._registry[key.schema_name]
        try:
            return self.query_engine.query(schema, data, key.shape)
        except PerfDataNotAvailableError:
            return None

    def store(self, key: "OperatorKey", result: QueryResult) -> None:
        """写入算子性能数据（仅 writable=True 时可用）。"""
        if not self.writable:
            raise RuntimeError("PerfDatabase is read-only")
        ...

    def save(self, output_path: Optional[Path] = None) -> None:
        """将内存中的数据持久化到磁盘。"""
        ...
```

**离散维度设计说明**：

数据库的离散维度应与 TensorCast 的感知能力对齐。参考 [AI Configurator](https://github.com/ai-dynamo/aiconfigurator) 的设计：

| 维度类型 | 是否纳入 | 理由 |
|---------|---------|------|
| **quant_mode** | 是（离散） | TensorCast 通过不同算子名区分量化模式（如 `static_quant_linear` vs `fp8_linear`），映射到数据库的 `quant_mode` 列。AI Configurator 同样以 `quant_mode` 作为嵌套字典的第一级 key |
| **BF16 vs FP16** | 统一为 `fp16` | AI Configurator 不区分 BF16/FP16。昇腾硬件上 BF16/FP16 的 Cube 算力相同，实测差异可忽略 |
| **stride / layout** | 不纳入 | TensorCast 使用 meta tensor，不感知内存排布。AI Configurator 同样不追踪 layout |
| **kv_cache_quant_mode** | 是（离散，仅 Attention） | AI Configurator 对 Attention 分别追踪 `fmha_quant_mode` 和 `kv_cache_quant_mode`。本方案在 `fused_attention` Schema 中增加 `kv_cache_dtype` 离散维度 |
| **head_size** | 是（离散，仅 Attention） | 不同 head_dim 的 Attention 内核行为差异大（如 128 vs 576 for MLA），应作为离散维度而非插值 |

> **注意**：`quant_mode` 的具体取值需与 TensorCast 的量化配置（`QuantConfig`）和 YAML 映射表中的算子分类保持一致。例如 `tensor_cast.static_quant_linear.default` 对应 `quant_mode=int8`，`tensor_cast.fp8_linear.default` 对应 `quant_mode=fp8`。

**数据文件格式示例（matmul.csv）**：
```csv
quant_mode,m,n,k,latency_us
fp16,1,4096,5120,12.5
fp16,8,4096,5120,13.2
...
```

### 4.2 OperatorKey（共享查询键）

`OperatorKey` 封装从 `OpInvokeInfo` 到 schematized 查询 key 的共享转换逻辑，供 `ProfilingPerformanceModel` 和 `EmpiricalPerformanceModel` 共同使用。

```python
# tensor_cast/performance_model/perf_database/operator_key.py

@dataclass
class OperatorKey:
    """
    标准化的算子查询键。

    与 CachingPerformanceModel 的 SHA256 cache_key 的区别：
    - CachingPerformanceModel: session 级精确匹配，func + 完整 tensor shape/stride/dtype
    - OperatorKey: 跨 session 语义匹配，按 Schema 定义的性能相关维度提取
    """
    schema_name: str          # Schema 名称（如 "matmul"、"fused_attention"）
    shape: Dict[str, Any]     # 性能相关维度（如 {m: 136, n: 4096, k: 5120, quant_mode: "int8"}）

    @classmethod
    def from_op_invoke_info(
        cls, op_invoke_info: OpInvokeInfo, op_mapping: Dict[str, str]
    ) -> Optional["OperatorKey"]:
        """
        从 OpInvokeInfo 提取 OperatorKey。
        op_mapping: YAML 加载的 TensorCast 算子 → Schema 名称映射表（见 4.6 节）
        返回 None 表示该算子不在任何 Schema 的覆盖范围内。
        """
        op_name = str(op_invoke_info.func)
        schema_name = op_mapping.get(op_name)
        if schema_name is None:
            return None
        schema = OperatorSchema._registry.get(schema_name)
        if schema is None:
            return None
        shape = schema.extract_shape_from_op(op_invoke_info)
        return cls(schema_name=schema_name, shape=shape)
```

### 4.3 ProfilingPerformanceModel（新增）

独立的 `PerformanceModel` 子类，专职从预构建的 `PerfDatabase` 中查询算子耗时。

```python
# tensor_cast/performance_model/profiling.py

class ProfilingPerformanceModel(PerformanceModel):
    """
    基于预采集 Profiling 数据的性能模型。

    - 只读查询 PerfDatabase，不进行任何写入
    - 未命中时内部 fallback 至 AnalyticPerformanceModel
    - 通过 perf_models 列表接入 Runtime（tensor_cast/runtime.py:41）
    - 由 Runtime 自动包装为 CachingPerformanceModel
    """

    def __init__(
        self,
        device_profile: DeviceProfile,
        database: PerfDatabase,
        fallback_model: Optional[PerformanceModel] = None,
    ):
        super().__init__("profiling", device_profile)
        self.database = database
        self.fallback_model = fallback_model or AnalyticPerformanceModel(device_profile)

    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        key = OperatorKey.from_op_invoke_info(op_invoke_info, self.database.op_mapping)
        if key is not None:
            result = self.database.lookup(key)
            if result is not None:
                return PerformanceModel.Result(
                    execution_time_s=result.latency_us * 1e-6,
                    statistics={"source": result.source.name, "confidence": result.confidence}
                )
        # 未命中：回退至 Roofline
        return self.fallback_model.process_op(op_invoke_info)

    def get_classifiers(self) -> List[PerformanceModel.OpClassifier]:
        return self.fallback_model.get_classifiers()
```

### 4.4 EmpiricalPerformanceModel 增强

增强现有 `EmpiricalPerformanceModel`，可选接入 `PerfDatabase` 作为跨 session 的持久化缓存。不传 `cache_db` 时与现有行为完全一致（后向兼容）。

```python
# tensor_cast/performance_model/empirical.py（增强后）

class EmpiricalPerformanceModel(PerformanceModel):
    """
    基于 JIT 实测的性能模型，可选接入 PerfDatabase 作为持久化缓存。
    流程：查缓存（若有） → JIT benchmark → 写缓存（若有）
    """

    def __init__(
        self,
        device_profile: DeviceProfile,
        cache_db: Optional[PerfDatabase] = None,
    ):
        super().__init__("empirical", device_profile)
        self.op_benchmark = OpBenchmark(device_profile)
        self.cache_db = cache_db

    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        # 1. 若有持久化缓存，先查缓存
        key = None
        if self.cache_db is not None:
            key = OperatorKey.from_op_invoke_info(op_invoke_info, self.cache_db.op_mapping)
            if key is not None:
                cached = self.cache_db.lookup(key)
                if cached is not None:
                    return PerformanceModel.Result(
                        execution_time_s=cached.latency_us * 1e-6,
                        statistics={"source": "cached_benchmark"}
                    )

        # 2. JIT benchmark（现有逻辑）
        result = self.op_benchmark.benchmark(op_invoke_info)

        # 3. 将结果写入持久化缓存
        if self.cache_db is not None and key is not None:
            self.cache_db.store(key, QueryResult(
                latency_us=result.execution_time_s * 1e6,
                confidence=1.0,
                source=QuerySource.MEASURED
            ))

        return result
```

### 4.5 OperatorSchema（插件化架构）

每种 Schema **与 kernel_details.csv 中的硬件内核类型一一对应**，负责：
1. 声明该硬件内核的 Shape 维度空间（哪些维度影响性能）
2. 提供从 `OpInvokeInfo` 提取 Shape 的逻辑（连接 TensorCast 运行时）
3. 声明该 Schema 处理的 TensorCast 算子名称列表（通过版本相关的 YAML 映射表配置）

> **设计原则**：Schema 的粒度与硬件内核对齐，而非与 TensorCast 抽象算子对齐。不同硬件内核即使功能相似（如 `MatMulV2` 和 `GroupedMatmul` 都是矩阵乘），也有不同的实现和性能曲线，必须分开建模。

```python
# tensor_cast/performance_model/perf_database/schemas/base.py

class DimensionSpec:
    """
    维度规格定义。
    - 可插值维度：数值连续，支持插值估算（如 m、num_tokens、batch_size）
    - 离散匹配维度：枚举值，必须精确匹配作为过滤条件（如 quant_mode、dtype）
    """
    name: str
    interpolatable: bool = True   # True=可插值, False=离散匹配
    typical_range: Optional[Tuple[int, int]] = None

class OperatorSchema(ABC):
    """
    算子性能 Schema 基类。
    每个 Schema 对应 kernel_details.csv 中的一种硬件内核类型。
    采用注册表模式自动发现（与 DeviceProfile、OpInvokeInfo.register_op_properties 设计一致）。
    """
    _registry: ClassVar[Dict[str, "OperatorSchema"]] = {}

    def __init_subclass__(cls, **kwargs):
        """自动注册子类实例。"""
        super().__init_subclass__(**kwargs)
        if not getattr(cls, '__abstractmethods__', None):
            instance = cls()
            OperatorSchema._registry[instance.name] = instance

    @property
    @abstractmethod
    def name(self) -> str:
        """Schema 名称，对应数据文件名和硬件内核类型（如 "matmul" → matmul.csv）。"""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> List[DimensionSpec]:
        """该硬件内核的 Shape 维度定义。"""
        ...

    @abstractmethod
    def extract_shape_from_op(self, op_invoke_info: OpInvokeInfo) -> Dict[str, Any]:
        """从拦截到的算子调用中提取 Shape 维度字典。"""
        ...

    def get_interpolation_dimensions(self) -> List[str]:
        return [d.name for d in self.dimensions if d.interpolatable]

    def get_discrete_dimensions(self) -> List[str]:
        return [d.name for d in self.dimensions if not d.interpolatable]
```

**具体 Schema 一览**（与 kernel_details.csv 硬件内核对齐）：

| Schema | 名称 | 对应硬件内核 | 关键维度 | 数据文件 |
|--------|------|------------|---------|---------|
| `MatMulSchema` | `matmul` | `MatMulV2` | quant_mode\*, m, n, k | `matmul.csv` |
| `GroupedMatMulSchema` | `grouped_matmul` | `GroupedMatmul` | quant_mode\*, num_tokens, hidden_size, inter_size, num_experts | `grouped_matmul.csv` |
| `QuantBatchMatMulSchema` | `quant_batch_matmul` | `QuantBatchMatmulV3` | quant_mode\*, m, n, k | `quant_batch_matmul.csv` |
| `TransposeBatchMatMulSchema` | `transpose_batch_matmul` | `TransposeBatchMatMul` | batch, m, n, k | `transpose_batch_matmul.csv` |
| `FusedAttentionSchema` | `fused_attention` | `FusedInferAttentionScore` | batch, query_len, context_len, num_heads, num_kv_heads, head_dim\*, kv_cache_dtype\* | `fused_attention.csv` |
| `InterleaveRopeSchema` | `interleave_rope` | `InterleaveRope` | num_tokens, num_heads, head_dim | `interleave_rope.csv` |
| `AddRmsNormSchema` | `add_rms_norm` | `AddRmsNorm` / `InplaceAddRmsNorm` | num_tokens, hidden_size | `add_rms_norm.csv` |
| `MoeDispatchSchema` | `moe_dispatch` | `MoeDistributeDispatch` | num_tokens, num_experts, hidden_size | `moe_dispatch.csv` |
| `MoeCombineSchema` | `moe_combine` | `MoeCombine` | num_tokens, num_experts, hidden_size | `moe_combine.csv` |
| `MoeGatingSchema` | `moe_gating` | `MoeGatingTopK` | num_tokens, num_experts | `moe_gating.csv` |
| `SwiGluSchema` | `swiglu` | `SwiGlu` / `DequantSwigluQuant` | num_tokens, hidden_size | `swiglu.csv` |
| `AscendQuantSchema` | `ascend_quant` | `AscendQuantV2` / `DynamicQuant` | num_tokens, hidden_size | `ascend_quant.csv` |
| `ReshapeAndCacheSchema` | `reshape_and_cache` | `ReshapeAndCacheNdKernel` | num_tokens, num_kv_heads, head_dim | `reshape_and_cache.csv` |
| `TensorMoveSchema` | `tensor_move` | `TensorMove` | total_bytes | `tensor_move.csv` |
| `AllReduceSchema` | `all_reduce` | `AllReduce` | num_devices, message_bytes | `all_reduce.csv` |
| `AllGatherSchema` | `all_gather` | `AllGather` | num_devices, message_bytes | `all_gather.csv` |
| `AllToAllSchema` | `all_to_all` | `AllToAll` | num_devices, message_bytes | `all_to_all.csv` |

> **注意**：以上为初始参考列表。标 `*` 的维度为离散匹配维度（`interpolatable=False`），其余为可插值维度。具体的 Schema 名称、维度定义和映射关系需根据实际 kernel_details.csv 的内核名称由专家确认。不同 vLLM-Ascend 版本的内核名可能不同，通过版本相关的 YAML 映射表管理。

**MatMul Schema 示例实现**：
```python
# tensor_cast/performance_model/perf_database/schemas/matmul.py

class MatMulSchema(OperatorSchema):
    """对应硬件内核 MatMulV2（FP16 标准稠密矩阵乘）。"""

    @property
    def name(self) -> str:
        return "matmul"

    @property
    def dimensions(self) -> List[DimensionSpec]:
        return [
            DimensionSpec("quant_mode", interpolatable=False),
            DimensionSpec("m", interpolatable=True),
            DimensionSpec("n", interpolatable=True),
            DimensionSpec("k", interpolatable=True),
        ]

    def extract_shape_from_op(self, op_invoke_info: OpInvokeInfo) -> Dict[str, Any]:
        x, w = op_invoke_info.args[0], op_invoke_info.args[1]
        return {"quant_mode": "fp16", "m": x.shape[0], "k": x.shape[1], "n": w.shape[1]}
```

### 4.6 TensorCast 算子 → 硬件内核 Schema 映射

映射表解决**名称翻译**问题：TensorCast dispatch 的算子名与 Profiling 的硬件内核名不同（如 `aten.mm.default` → `MatMulV2`），需要映射表将 TensorCast 算子对应到正确的 Schema。

**核心前提：通过编译 Pass 实现 1:1 对齐**。TensorCast 已有成熟的 `PatternMatcherPass` 基础设施（`tensor_cast/compilation/patterns/` 下已实现 13+ 种 RMSNorm 融合和 2 种 RoPE 融合），对于 vLLM 硬件融合内核与 TensorCast dispatch trace 不一致的情况，**统一通过新增编译 Pass 解决**，使 dispatch trace 与硬件内核列表保持 1:1 对齐。当前仍需补充的编译 Pass 详见第 9.1 节。

#### 映射表格式

映射关系通过**版本相关的 YAML 配置**管理。`tensorcast_op_to_schema` 为扁平的 1:1 映射：每个 TensorCast 算子映射到一个硬件内核 Schema。同类算子的不同量化变体可能映射到不同的硬件内核（如 `aten.mm` → `matmul`，`static_quant_linear` → `quant_batch_matmul`）。

```yaml
# perf_database/mappings/vllm_ascend/v0.13.yaml
version: "0.13"
device: ATLAS_800_A3_752T_128G_DIE

tensorcast_op_to_schema:
  # GEMM
  "aten.mm.default": matmul
  "tensor_cast.static_quant_linear.default": quant_batch_matmul
  "tensor_cast.static_quant_linear_int4.default": quant_batch_matmul
  "tensor_cast.fp8_linear.default": quant_batch_matmul
  "tensor_cast.mxfp4_linear.default": quant_batch_matmul

  # Grouped MatMul (MoE)
  "tensor_cast.grouped_matmul.default": grouped_matmul
  "tensor_cast.grouped_matmul_quant.default": grouped_matmul
  "tensor_cast.grouped_matmul_fp8.default": grouped_matmul

  # Attention
  "tensor_cast.attention.default": fused_attention
  "tensor_cast.attention_quant.default": fused_attention
  "tensor_cast.multihead_latent_attention.default": fused_attention
  "tensor_cast.multihead_latent_attention_quant.default": fused_attention

  # RoPE
  "tensor_cast.apply_rope.default": interleave_rope

  # Normalization（已有编译 Pass 融合，见 compilation/patterns/rms_norm.py）
  "tensor_cast.add_rms_norm.default": add_rms_norm
  "tensor_cast.add_rms_norm_quant.default": add_rms_norm
  "tensor_cast.add_rms_norm_dynamic_quant_symmetric.default": add_rms_norm

  # Activation（需新增编译 Pass，见 9.1 节）
  "tensor_cast.swiglu.default": swiglu                  # 待实现
  "tensor_cast.dequant_swiglu_quant.default": swiglu     # 待实现

  # MoE 路由
  "tensor_cast.permute_tokens.default": moe_dispatch
  "tensor_cast.unpermute_tokens.default": moe_combine
  "aten.topk.default": moe_gating  # 注意：aten.topk 也用于采样，此处仅近似处理 MoE 场景，待 9.1 节适配后改用专用算子

  # Communication
  "tensor_cast.all_reduce.default": all_reduce
  "tensor_cast.all_gather.default": all_gather
  "tensor_cast.all_to_all.default": all_to_all

  # 量化（linear 前的输入量化，独立 dispatch）
  "tensor_cast.quantize.default": ascend_quant
  "tensor_cast.dynamic_quantize_symmetric.default": ascend_quant
  "tensor_cast.dynamic_quantize_asymmetric.default": ascend_quant
  "tensor_cast.dynamic_quantize_mxfp4.default": ascend_quant

  # Cache
  "tensor_cast.reshape_and_cache.default": reshape_and_cache
```

> **注意**：以上映射为初始示例。标注"待实现"的算子需先完成第 9.1 节中的编译 Pass 适配。具体映射关系需根据实际 kernel_details.csv 由专家配置，不同版本的 vLLM-Ascend 映射表会不同。

### 4.7 QueryEngine（精确匹配 + 插值）

`PerfDatabase` 内部的 QueryEngine **仅处理实测数据**：精确匹配 → 插值 → 外推。未收录算子不在 QueryEngine 内做 Roofline 兜底，而是返回查询失败，由 `ProfilingPerformanceModel` 的 `fallback_model` 处理。

```python
# tensor_cast/performance_model/perf_database/query_engine.py

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
        # Step 2: 精确匹配
        # Step 3: 插值
        # Step 4: 外推（最近邻 + 线性外推）
```

> **插值策略参考**：参考 [AI Configurator](https://github.com/ai-dynamo/aiconfigurator) 的 2D+1D 混合插值方法（见第 2.5 节）：先对两个维度做双线性插值，再对第三个维度做 1D 插值。对 Attention 等 O(n²) 复杂度的算子，在插值前对序列维度做 sqrt 变换以提高拟合精度。置信度评分基于查询点到最近实测数据点的距离计算。

---

## 5. TensorCast 集成接口

### 5.1 Runtime 集成

`Runtime` 类无需修改。它已支持接收 `PerformanceModel` 实例列表，并自动包装为 `CachingPerformanceModel`。三种 PerformanceModel 的使用方式：

```python
# 方式 1: ProfilingPerformanceModel（查预构建数据库，无需物理设备）
db = PerfDatabase("perf_database/data/atlas_a3_752t_128g/vllm_ascend/v0.13.0")
perf_model = ProfilingPerformanceModel(device_profile, database=db)

# 方式 2: EmpiricalPerformanceModel + 持久化缓存（有物理设备）
cache_db = PerfDatabase("./my_benchmark_cache", writable=True)
perf_model = EmpiricalPerformanceModel(device_profile, cache_db=cache_db)

# 方式 3: AnalyticPerformanceModel（Roofline，现有默认行为）
perf_model = AnalyticPerformanceModel(device_profile)

# 统一传入 Runtime
runtime = Runtime(perf_models=perf_model, device_profile=device_profile)
```

### 5.2 CLI 接口

在 `tensor_cast/scripts/text_generate.py` 中新增参数：

```python
parser.add_argument("--performance-model",
                    choices=["analytic", "profiling", "empirical"],
                    default="analytic", help="性能模型类型")
parser.add_argument("--perf-database", type=str, default=None,
                    help="性能数据库路径（profiling / empirical 模式生效），"
                         "指向包含 metadata.yaml 和 CSV 数据文件的目录")
```

| CLI 选项 | 创建的 PerformanceModel | 是否需要物理设备 | 是否需要数据库 |
|---------|------------------------|----------------|-------------|
| `--performance-model analytic` | `AnalyticPerformanceModel` | 否 | 否 |
| `--performance-model profiling` | `ProfilingPerformanceModel` | 否 | 是（`--perf-database`） |
| `--performance-model empirical` | `EmpiricalPerformanceModel` | 是 | 可选（`--perf-database` 作缓存） |

### 5.3 数据流（以 ProfilingPerformanceModel 为例）

```
text_generate.py
  --performance-model profiling
  --perf-database ./perf_database/data/atlas_a3/.../v0.13.0/
       │
       ▼
  创建 ProfilingPerformanceModel + PerfDatabase(data_path)
       │
       ▼
  Runtime.__torch_dispatch__ → 拦截算子 → OpInvokeInfo
       │
       ▼
  ProfilingPerformanceModel.process_op()
       ├──▶ OperatorKey.from_op_invoke_info()
       │    op_mapping["aten.mm.default"] → "matmul" → MatMulSchema
       │    → extract_shape → {m: 136, n: 4096, k: 5120, quant_mode: "fp16"}
       │
       ├──▶ database.lookup(key)
       │    精确匹配 / 插值 / 外推 → QueryResult
       │
       ├──▶ 若 lookup 返回 None → fallback_model.process_op()
       │
       └──▶ 返回 PerformanceModel.Result(execution_time_s=...)
```

---

## 6. 自动化 Profiling 流水线

> **注意**：本节描述的数据采集流水线作为独立子系统，代码位于仓库顶层 `perf_database/` 目录，不包含在 TensorCast 包内。

### 6.1 数据库构建策略

| 策略 | 适用场景 | 优势 | 劣势 |
|------|---------|------|------|
| **方案 A**（默认）：全模型 Profiling + 微基准测试 | 有完整 VLLM 部署环境 | 精确捕获融合内核，校准数据可靠 | VLLM 服务启动开销大 |
| **方案 B**：仿真驱动 + 微基准测试 | 仅有裸机 NPU 环境 | 无需 VLLM 部署，采集效率高 | 无法覆盖融合内核 |
| **方案 C**：纯 Profiling 导入 | 已有现成的 Profiling 数据 | 零额外采集成本 | Shape 覆盖受限 |

### 6.2 两级采集策略（方案 A）

#### Level 1: 全模型 Profiling（获取基准真值）

启用昇腾 Profiler 运行 VLLM 推理，捕获真实融合内核的执行耗时。每个服务实例只需启动一次，随后发送多轮不同 Shape 配置的基准测试请求。

```bash
export VLLM_TORCH_PROFILER_DIR=/path/to/output
export PROFILING_SAVE_PATH=/path/to/output
vllm serve $MODEL --tensor-parallel-size $TP --dtype bfloat16 ...
vllm bench serve --profile --dataset-name random \
    --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
    --max-concurrency $BATCH_SIZE
```

**单实例内 Shape 遍历方案**：
```python
# Decode 场景（query_len=1，不同 context_len 影响 KV Cache 长度和 Attention 性能）
for context_len in [256, 512, 1024, 2048, 4096, 8192]:
    for batch in [1, 8, 16, 32, 64, 128, 256]:
        bench(input_len=1, output_len=1, concurrency=batch,
              max_model_len=context_len)

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
def benchmark_matmul(m, n, k, dtype, warmup=5, runs=20):
    """使用 NPU Event 计时的单算子微基准测试，返回平均耗时（微秒）。"""
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
        """解析 kernel_details.csv → 结构化 DataFrame。
        返回: name, duration_us, input_shapes, data_types,
              aicore_time_us, aiv_time_us, cube_utilization_pct"""

    def extract_single_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """以 Attention 内核为边界提取单个 Decode/Prefill 步骤。"""

    def map_to_schema(self, kernel_name: str, input_shapes: str,
                      mapping_yaml: Path) -> Tuple[str, Dict]:
        """将 VLLM 内核名映射到 (schema_name, shape_dict)。
        如 "MatMulV2" + "136,4096; 4096,4096" → ("matmul", {m:136, k:4096, n:4096})"""
```

### 6.4 新模型算子发现机制

```python
# perf_database/scripts/discover_operators.py

def discover_operators(profiling_output: Path, mapping_yaml: Path) -> Dict:
    """发现 Profiling 中存在但当前映射表中缺失的算子。
    返回 known 和 unknown 计数，unknown 包含 kernel 名称和建议 Schema。"""
    parser = AscendProfilerParser()
    kernels = parser.parse_kernel_details(profiling_output / "kernel_details.csv")
    known_ops = load_yaml_mappings(mapping_yaml)
    unknown = []
    for kernel_name in kernels["name"].unique():
        if not any(kernel_name in schema_kernels for schema_kernels in known_ops.values()):
            unknown.append({"kernel": kernel_name, "suggested_schema": auto_classify(kernel_name)})
    return {"known": len(kernels) - len(unknown), "unknown": unknown}
```

---

## 7. 全面算子覆盖策略

### 7.1 算子分级

| 层级 | 判定标准 | 处理方式 | 算子数量 |
|-----|---------|---------|---------|
| **Tier 1** | 执行耗时占比 >2%（或计算内核中 >5%） | 必须使用完整 Shape 网格进行 Profiling | 约 14 个 |
| **Tier 2** | 执行耗时占比 0.5-2% | 使用精简 Shape 网格进行 Profiling | 约 9 个 |
| **Tier 3** | 执行耗时占比 <0.5% | 采用 Roofline 兜底估算 | 60+ 个 |

### 7.2 Tier 1 完整算子列表

> **数据来源**：以下占比均基于 kernel_details.csv 的**全量内核耗时（包含通信内核）**，与第 2.2 节仅统计计算内核的占比口径不同。例如 Qwen3 的 FusedInferAttentionScore 在计算内核中占 18.2%（第 2.2 节），但在全量内核（含通信）中仅占 1.7%。
> - **Qwen3-32B**：Prefill 模式，16卡，通信占 89.8%，计算内核占比相应偏小
> - **DeepSeekV3**：Decode 模式，32卡，计算/通信比例更均衡
>
> 某内核在任一模型中超过 2% 即列入 Tier 1。

| VLLM 内核名称 | 数据库 Schema | Qwen3 占比 | DSV3 占比 |
|-------------|-------------|-----------|---------|
| hcom_allReduce\_ | `all_reduce` | **89.8%** | 8.1% |
| GroupedMatmul | `grouped_matmul` | - | **18.7%** |
| QuantBatchMatmulV3 | `quant_batch_matmul` | - | **18.3%** |
| FusedInferAttentionScore | `fused_attention` | 1.7% | **16.5%** |
| MoeDistributeDispatchV2 | `moe_dispatch` | - | **7.0%** |
| MatMulV2 | `matmul` | **4.1%** | 0.5% |
| AscendQuantV2 / DynamicQuant | `ascend_quant` | - | **3.9%** |
| TransposeBatchMatMul | `transpose_batch_matmul` | - | **3.8%** |
| MoeDistributeCombineV2 | `moe_combine` | - | **3.6%** |
| InterleaveRope | `interleave_rope` | - | **2.5%** |
| DequantSwigluQuant / SwiGlu | `swiglu` | 0.5% | **2.4%** |
| InplaceAddRmsNorm / AddRmsNorm | `add_rms_norm` | 0.8% | 1.8% |
| HcomAllGather | `all_gather` | 0.6% | 1.7% |

**Tier 2 参考列表**（0.5-2%）：

| VLLM 内核名称 | 数据库 Schema | Qwen3 占比 | DSV3 占比 |
|-------------|-------------|-----------|---------|
| MatMul（MLA latent proj） | `matmul` | - | 1.3% |
| MoeGatingTopK | `moe_gating` | - | 1.1% |
| TensorMove | `tensor_move` | 1.0% | 0.1% |
| RmsNorm | Roofline 兜底 | <0.1% | 0.9% |
| KvRmsNormRopeCache | 融合拆解（见 4.6 节） | - | 0.8% |
| Transpose | Roofline 兜底 | - | 0.8% |
| split\_qkv\_rmsnorm\_rope\_kernel | 融合拆解（见 4.6 节） | 0.5% | - |
| ReshapeAndCacheNdKernel | `reshape_and_cache` | 0.3% | - |
| SwiGlu | `swiglu` | 0.5% | - |

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

**预估 Shape 总量**：每个版本约 5,000 个。

---

## 8. 开发计划

**团队分工**：
- **六壬工具团队**：TensorCast 侧（PerformanceModel、OperatorSchema、CLI 集成）
- **小巧灵团队**：数据侧（PerfDatabase、QueryEngine、数据采集流水线）

**前置依赖**：TensorCast 算子追踪对齐（详见第 9.1 节），需在集成测试前完成。

**目标**：2026.3.20 完成端到端集成，DeepSeek-V3 / Qwen3-32B 能对齐的算子覆盖端到端 >90% 的时间。

### 阶段一：核心框架与端到端穿刺（2 周）

联合对齐 Qwen3-32B 和 DeepSeek-V3 在目标版本上的算子列表和 YAML 映射表。

| 六壬团队 | 小巧灵团队 |
|---------|-----------|
| `ProfilingPerformanceModel` + Roofline fallback | `PerfDatabase`（加载、查询、存储） |
| `OperatorSchema` 基类 + 初始 Schema 集合 | CSV 数据格式规范 + 示例数据 + 精确匹配查询 |
| `OperatorKey` + CLI 集成 | Profiling 解析器（`parse_ascend_output.py`） |
| 基于示例数据库穿刺端到端仿真 | 穿刺数据采集方案 |

### 阶段二：插值引擎与数据采集（1 周）

| 六壬团队 | 小巧灵团队 |
|---------|-----------|
| `EmpiricalPerformanceModel` 增强（可选 `cache_db`） | `QueryEngine`（插值、置信度评分） |
| 扩展 Schema 覆盖（MoE、归一化、融合算子等） | 数据采集脚本（全模型 Profiling + 微基准测试） |
| | A3 上初始数据库采集 |

### 阶段三：集成测试与验证（1 周）

联合使用 Qwen3-32B 和 DeepSeek-V3 进行端到端精度验证。

**验证标准**：

| 指标 | 目标值 |
|-----|-------|
| 端到端耗时误差 | <15%（对比实际 VLLM Profiling） |
| 单算子误差（已匹配算子） | <20% |
| 时间覆盖率 | >90% |

---

## 9. 外部依赖与扩展建议

### 9.1 前置依赖：TensorCast 算子追踪对齐

本方案要求 TensorCast dispatch trace 与 vLLM Profiling 的硬件内核列表在关键算子上 1:1 对齐（见 4.6 节）。对齐方式为**新增编译 Pass**（`PatternMatcherPass`），将多个 aten 算子融合为单一 TensorCast 自定义算子，与已有的 13+ 种 RMSNorm 融合 Pass（`compilation/patterns/rms_norm.py`）和 RoPE 融合 Pass（`compilation/patterns/rotary_embedding.py`）一致。

当前存在若干不匹配项需通过前置适配工作解决：

| 不匹配项 | 现状 | 所需适配 | 对应硬件内核 |
|---------|------|---------|------------|
| **SwiGlu 融合** | TensorCast dispatch `aten.silu` + `aten.mul` 两个独立算子 | 新增 `PatternMatcherPass` 将 silu+mul 融合为 `tensor_cast.swiglu`；含量化变体 `tensor_cast.dequant_swiglu_quant` | `SwiGlu` / `DequantSwigluQuant` |
| **split_qkv_rmsnorm_rope 融合** | TensorCast 分别 dispatch qkv_split、rms_norm、apply_rope | 新增编译 Pass 融合为 `tensor_cast.split_qkv_rmsnorm_rope` | `split_qkv_rmsnorm_rope_kernel` |
| **KvRmsNormRopeCache 融合** | TensorCast 分别 dispatch rms_norm、apply_rope、reshape_and_cache | 新增编译 Pass 融合为 `tensor_cast.kv_rmsnorm_rope_cache` | `KvRmsNormRopeCache` |
| `multihead_latent_attention` | 单一 dispatch 节点，包含 TransposeBatchMatMul × 2 + FusedInferAttentionScore | 拆分为独立 dispatch 算子 |  |
| `aten.topk` / MoeGatingTopK | aten.topk 在非 MoE 场景也被调用 | 区分 MoE 路由 topk 和采样 topk |  |
| 其他潜在差异 | 需逐模型比对确认 | 逐一人工适配 |  |

**建议方式**：以 DeepSeek-V3 和 Qwen3-32B 的 kernel_details.csv 为基准，导出 TensorCast dispatch trace 逐一比对，通过新增 custom op 和编译 Pass 消除差异。SwiGlu 融合优先级最高（Decode 阶段占比 2.4%）。

### 9.2 CompositePerformanceModel 顶层调度器

当前方案中 `ProfilingPerformanceModel` 对未收录算子内部兜底回退至 `AnalyticPerformanceModel`，这是 v1.1 的务实做法。长期来看，建议设计一个 `CompositePerformanceModel`，按优先级组合多个 PerformanceModel，实现可配置的降级策略（`Profiling → Empirical → Analytic`）。引入后 `ProfilingPerformanceModel` 不再需要内部持有 `fallback_model`，三种模型完全解耦。

### 9.3 其他扩展建议

- **跨硬件泛化**：支持更多 DeviceProfile（如 Atlas A2、GPU），需为每种硬件独立采集数据
- **自动化 CI**：随 VLLM-Ascend / CANN 版本发布自动触发数据采集流水线
- **预插值优化**：参考 AI Configurator 的 `_extrapolate_data_grid`，在数据库加载时预填充常用网格点
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

## 附录：架构设计分析

### ProfilingPerformanceModel vs EmpiricalPerformanceModel

**问题**：基于数据库查询的 Profiling 模型和基于 JIT microbenchmark 的 Empirical 模型应该合并还是分开？

**三种候选方案**：
- **方案 A（合并）**：在 EmpiricalPerformanceModel 内引入 DataSource 抽象层。缺点：类复杂度膨胀，内部 fallback 与 CompositePerformanceModel 职责重叠，违反开闭原则。
- **方案 B（采纳：分开）**：两个独立 PerformanceModel + 共享 PerfDatabase 数据层。优点：单一职责，独立可测试，EmpiricalPerformanceModel 无需改动核心逻辑。
- **方案 C（继承）**：DatabasePerformanceModel 继承 EmpiricalPerformanceModel。缺点：继承耦合，无法单独使用 Database 而不用 JIT。

**结论**：耦合在数据层，解耦在模型层。两者的区别仅在于数据何时产生（离线 vs JIT）、读写模式（只读 vs 读写）、数据密度（系统性采集 vs 按需触发）。更优雅的做法是抽取共享的 `PerfDatabase` 数据层，而非合并两个 Model。

---

## Change Log

### v1.0 → v1.1

1. 从单一 `ProfilingPerformanceModel`（内含 Roofline 兜底）重构为三个独立 PerformanceModel 并列，Roofline 兜底上移至 Model 层 `fallback_model`
2. `PerfDatabase` 从 Profiling 专用包提升为共享数据层，同时为 Profiling（只读）和 Empirical（读写持久化缓存）服务；`EmpiricalPerformanceModel` 新增可选 `cache_db` 参数
3. 新增 `OperatorKey` 共享抽象，封装 `OpInvokeInfo → 查询 key` 转换逻辑
4. `PerfDatabase` API 从 `(system, backend, version)` 三元组改为 `data_path` 路径直传，删除 `VersionManager`
5. Schema 粒度从按抽象算子类型分组（~6 类）改为按 kernel_details.csv 硬件内核一一对齐（~17 个），`DimensionSpec` 从枚举简化为 `interpolatable: bool`
6. YAML 映射格式从按 Schema 分组改为扁平的 `tensorcast_op_to_schema`（1:1）
7. QueryEngine 从 4 级降级（含 Roofline）精简为 3 级（精确→插值→外推）
8. 存储格式从 Parquet 为主改为 CSV 为主；数据采集从单一 L1+L2 扩展为三种可选方案（全模型 Profiling / 仿真驱动 / 纯导入）
9. 基于 DeepSeekV3/Qwen3 实测 Profiling 数据修正了多项算子映射（如 `static_quant_linear` → `quant_batch_matmul`），新增 TransposeBatchMatMul、InterleaveRope、AllGather、MoeGating 独立 Schema
10. 数据采集流水线从 TensorCast 包内移至仓库顶层 `perf_database/` 独立子系统；开发计划从 5 阶段单团队改为 3 阶段双团队分工
