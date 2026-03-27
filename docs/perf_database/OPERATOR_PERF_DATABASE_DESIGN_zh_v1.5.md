# TensorCast 算子性能数据库：技术设计文档

**版本**: 1.5
**日期**: 2026.3.24
**变更**: §6 数据库构建方案更新（op-replay pipeline + 理论导向 Shape Grid 策略）；FIA lookup 方案重设计（enriched CSV、MLA prefill 路径修正、CANN 8.3/8.5 差异处理）
**范围**: 面向 LLM 仿真的可扩展实测性能模型，不绑定具体算力卡，支持基于实测 Profiling 数据和 Microbenchmark 数据的算子性能估算。
**初期目标模型**: DeepSeek-V3、Qwen3-32B

作者：HXW
审核人：GJ

---

## 1. 功能概述

### 1.1 目标

为 TensorCast 仿真器构建基于实测数据的算子性能估算系统。重构现有 `EmpiricalPerformanceModel`，使其接受通用的 `DataSource` 抽象接口，支持多种数据来源（预采集 Profiling 数据库、JIT Benchmark 缓存等）。通过 `ProfilingDataSource` 实现基于预采集数据的查询，与现有 `AnalyticPerformanceModel`（Roofline）并列，作为用户可选的性能模型。

核心架构为 `EmpiricalPerformanceModel` + `DataSource` 模式：
- `EmpiricalPerformanceModel`：统一的性能模型入口，接受不同 `DataSource` 实例实现不同行为
- `ProfilingDataSource`：基于预采集 Profiling CSV 的只读数据源，内部处理算子映射、FRACTAL_NZ 格式转换、dtype 匹配
- `InterpolatingDataSource`：Wrapper DataSource，在底层 DataSource 基础上提供插值/近似查询能力
- `CacheKeyDataSource`：基于 `OpInvokeInfo.cache_key` 的精确匹配数据源（未来扩展）

### 1.2 核心功能（本方案范围）

1. **DataSource 抽象层（新增）**：定义标准化查询接口 `lookup(OpInvokeInfo)`，与数据来源解耦。`ProfilingDataSource` 为首要实现，内部处理 `op_mapping.yaml` 映射 + CSV 数据查询 + FRACTAL_NZ 格式转换
2. **EmpiricalPerformanceModel（重构）**：接受 `DataSource` 实例，从数据源查询算子耗时；未命中时内部 fallback 至 `AnalyticPerformanceModel`（计算算子）或 `CommAnalyticModel`（通信算子）
3. **InterpolatingDataSource（新增）**：Wrapper 模式，在底层 DataSource 的精确匹配基础上提供插值/外推能力
4. **op_mapping.yaml 配置驱动（新增）**：纯名字映射（TensorCast func → Profiling kernel Type），不含 per-op 维度提取逻辑，所有维度匹配和格式转换由 ProfilingDataSource 通用代码处理
5. **数据采集工具链**（独立子系统，位于 `tools/perf_data_collection/`）：Profiling 数据解析、Microbenchmark 脚本生成、数据库构建与验证

### 1.3 不在本方案范围内（建议后续支持）

- **CompositePerformanceModel 顶层调度器**：统一编排多种 PerformanceModel，实现可配置的降级或组合策略（详见第 9.3 节）
- **跨硬件泛化**：当前仅支持昇腾 A3，其他硬件需独立采集数据
- **自动化持续集成**：随 vLLM-Ascend / CANN 版本发布自动触发数据采集

### 1.4 初期目标

- **目标模型**：DeepSeek-V3、Qwen3-32B
- **目标硬件**：Atlas 800 A3（752T，128G DIE）
- **目标后端**：vllm-ascend 0.15.0（CANN 8.5，torch 2.9.0）
- **精度目标**：端到端仿真误差 <15%（对比实际 vLLM Profiling）
- **交付时间**：2026.3.23 完成端到端集成并完成初始数据采集和集成测试

> **v1.3.1 注**：目标版本从 vllm-ascend 0.13.0（CANN 8.3）升级至 0.15.0（CANN 8.5）。CANN 8.3 数据保留作为参考基线。

---

## 2. 技术分析

### 2.1 问题陈述

TensorCast 当前采用**基于 Roofline 的解析模型**（`AnalyticPerformanceModel`）估算算子执行耗时。该模型基于浮点运算量（FLOPs）与访存字节数计算 `max(计算耗时, 访存耗时)`，主要用作理论性能上界评估。

**Roofline 偏差根因**：

| 偏差来源 | 影响场景 | 描述 |
|---------|---------|------|
| **硬件利用率受 tiling 策略影响** | 低估小 batch 场景耗时（含 Decode 和小 batch Prefill） | 小 batch 下 AI Core 占用率不足（tile 数少于 AI Core 数量导致空闲）、tiling 粒度不匹配（矩阵维度未对齐 tile size 产生 padding 浪费）、MTE-Cube-Vector 流水线无法填满 |
| **融合内核差异** | 部分算子 | vLLM 实际部署使用深度融合的硬件内核（如 `DequantSwigluQuant`、`KvRmsNormRopeCache`），与 TensorCast 的原子算子 dispatch 存在粒度差异 |
| **硬件特性未完整建模** | Memory-bound 场景 | Cache 层级、bank conflict、内存访问 pattern 等微架构特性未在 Roofline 中体现 |

> 关于小 batch 场景的详细分析（含 Decode 和小 batch Prefill 的区别）见附录 A。

### 2.2 Profiling 数据分析

基于 DeepSeekV3 Decode（32 卡）和 Qwen3-30B Prefill（16 卡）的 kernel_details.csv 分析：

**DeepSeekV3 Decode 关键算子（按调用次数排序，Top 15）**：
QuantBatchMatmulV3: 15006, AscendQuantV2: 10004, Add: 7545, TransposeBatchMatMul: 5002, InplaceAddRmsNorm: 5002, DequantSwigluQuant: 4879, GroupedMatmul: 4756, FusedInferAttentionScore: 2501, KvRmsNormRopeCache: 2501, InterleaveRope: 2501, DynamicQuant: 2501, MoeGatingTopK: 2378, MoeDistributeDispatchV2: 2378, MoeDistributeCombineV2: 2378, MatMul: 2378, MatMulV2: 41

**Qwen3-30B Prefill 关键算子（按调用次数排序）**：
TensorMove: 386, hcom_allReduce_: 276, MatMulV2: 275, AddRmsNorm: 131, FusedInferAttentionScore: 67, SwiGlu: 67, ReshapeAndCacheNdKernel: 67, split_qkv_rmsnorm_rope_kernel: 64

> **v1.3.1 注**：上述数据基于 CANN 8.1/8.3 版本。CANN 8.5 引入了 `DispatchFFNCombine` 超级融合算子（融合 `all_to_all×2 + GroupedMatmul×2 + SwiGlu + MoE routing`），占 DSV3 Decode 端到端耗时 **35.3%**，是 DSV3 最重要的单一 kernel。TC 通过 `composite: true` 分解查询覆盖。详见 §9.1 融合 Gap 状态。

### 2.3 版本影响分析

算子性能受以下版本因素影响：
- **CANN 版本**：影响 kernel 实现、tiling 策略、通信库（HCCL）行为
- **vLLM-Ascend 版本**：影响融合算子实现（如 MC2）、kernel 调度策略
- **硬件型号**：直接决定计算/访存/通信性能参数

数据存储按 `{device}/{backend}/{version}/` 层级管理，通信数据按 `{device}/hccl/{cann_version}/` 单独管理（跨 vLLM 版本复用）。

### 2.4 现有基础设施

TensorCast 已有完善的性能模型框架：
- **`PerformanceModel` 基类**：统一接口 `process_op(OpInvokeInfo) → Result`
- **`AnalyticPerformanceModel`**：Roofline 模型（`tensor_cast/performance_model/analytic.py`）
- **`CommAnalyticModel`**：通信算子解析模型（`tensor_cast/performance_model/comm_analytic.py`）
- **`CachingPerformanceModel`**：SHA256 cache_key 级别的 session 内缓存
- **`OpInvokeInfo`**：算子元数据（func, args, cache_key），支持 `register_op_properties` 扩展
- **`DeviceProfile` + `CommGrid`**：硬件规格 + 网络拓扑描述

### 2.5 AI Configurator 参考

[AI Configurator](https://github.com/ai-dynamo/aiconfigurator) 是 NVIDIA 的 LLM 推理性能预估工具，其设计对本方案有参考价值：

- **数据格式**：CSV（`.txt` 后缀），每行含重复的 framework/version/device 列实现自描述性，支持跨版本聚合
- **插值策略**：2D+1D 混合插值，对 O(n²) 的 Attention 算子做 sqrt 变换后再插值
- **DatabaseMode**：`SILICON → HYBRID → EMPIRICAL → SOL` 级联模式
- **Git LFS 管理**：`.gitattributes` 中配置 `systems/**/*.txt filter=lfs`
- **通信数据**：NCCL 原生通信按 `nccl/{version}/` 独立存储，跨框架版本共享；Custom AllReduce 和框架版本绑定

> 通信方案的详细对比分析见附录 F。

我们的方案与 AI Configurator 的区别：
- **不复制重复列模式**：CSV 按目录分级存储，上下文信息在 `op_mapping.yaml` 中
- **参考其 Git LFS 管理方式**
- **参考其通信数据按库版本独立存储的模式**（hccl/{cann_version}/）

---

## 3. 系统架构

### 3.1 整体架构

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           TensorCast Runtime                              │
│                                                                           │
│  ┌────────────────────┐     ┌──────────────────────────────────────────┐  │
│  │  Runtime            │     │  用户可配置选择 PerformanceModel          │  │
│  │  (TorchDispatchMode)│────▶│                                          │  │
│  │                     │     │  ┌──────────────────────────────────┐    │  │
│  │  拦截所有算子调用   │     │  │ EmpiricalPerformanceModel（重构）│    │  │
│  │  生成 OpInvokeInfo  │     │  │  DataSource.lookup(OpInvokeInfo) │    │  │
│  └────────────────────┘     │  │  未命中 → fallback (Analytic)    │    │  │
│                              │  └──────────────────────────────────┘    │  │
│                              │  ┌──────────────────────────────────┐    │  │
│                              │  │ AnalyticPerformanceModel（现有） │    │  │
│                              │  │  Roofline 理论模型               │    │  │
│                              │  └──────────────────────────────────┘    │  │
│                              └──────────────────────────────────────────┘  │
│                                           │                                │
└───────────────────────────────────────────┼────────────────────────────────┘
                                            │
                      ┌─────────────────────┼─────────────────────┐
                      │                     ▼                     │
                      │       DataSource（抽象接口）              │
                      │  ┌──────────────────────────────────┐    │
                      │  │  ProfilingDataSource              │    │
                      │  │  op_mapping.yaml 映射 + CSV 查询  │    │
                      │  │  FRACTAL_NZ 恢复 + dtype 匹配     │    │
                      │  └──────────────────────────────────┘    │
                      │  ┌──────────────────────────────────┐    │
                      │  │  InterpolatingDataSource (Wrapper)│    │
                      │  │  精确匹配 → 插值 → 外推            │    │
                      │  └──────────────────────────────────┘    │
                      │  ┌──────────────────────────────────┐    │
                      │  │  CacheKeyDataSource（未来扩展）   │    │
                      │  │  JIT benchmark 缓存               │    │
                      │  └──────────────────────────────────┘    │
                      │  ┌──────────────────────────────────┐    │
                      │  │  存储层                            │    │
                      │  │  计算: vllm_ascend/{version}/*.csv │    │
                      │  │  通信: hccl/{cann_version}/*.csv   │    │
                      │  └──────────────────────────────────┘    │
                      └──────────────────────────────────────────┘

数据采集工具（离线执行，独立子系统）
┌──────────────────────────────────────────────────────────────┐
│  三步走：                                                     │
│  1. vLLM Profiling → kernel_details.csv → 按 Type 拆分 CSV  │
│  2. Microbenchmark 网格遍历 → 扩充 CSV 覆盖范围              │
│  3. 端到端验证 → 精度报告                                    │
└──────────────────────────────────────────────────────────────┘
```

**设计原则**：
- **配置优于代码**：算子映射通过 `op_mapping.yaml` 配置，新增算子只需修改 YAML，不需写 Python 代码
- **数据与逻辑解耦**：`ProfilingDataSource` 处理所有映射和格式转换，上层 `EmpiricalPerformanceModel` 只关心 `lookup(OpInvokeInfo) → Optional[QueryResult]`
- **通用 FRACTAL_NZ 处理**：格式转换由 `ProfilingDataSource` 内通用函数处理，不依赖算子类型

### 3.2 模块结构

```
tensor_cast/performance_model/
├── analytic.py                           # AnalyticPerformanceModel（现有）
├── comm_analytic.py                      # CommAnalyticModel（现有）
├── memory_tracker.py                     # MemoryTracker（现有）
├── empirical.py                          # EmpiricalPerformanceModel（重构）
└── profiling_database/                        # 新增：DataSource + 数据存储
    ├── __init__.py
    ├── data_source.py                    # DataSource ABC + QueryResult
    ├── profiling_data_source.py          # ProfilingDataSource（CSV 查询 + FRACTAL_NZ）
    ├── interpolating_data_source.py      # InterpolatingDataSource（Wrapper 插值）
    └── data/                             # 性能数据存储（Git LFS 管理 .csv）
        └── ATLAS_800_A3_752T_128G_DIE/
            ├── vllm_ascend/
            │   ├── v0.13.0/                         # CANN 8.3 legacy
            │   ├── vllm0.13.0_torch2.8.0_cann8.3/  # CANN 8.3 完整命名
            │   └── vllm0.15.0_torch2.9.0_cann8.5/  # CANN 8.5 生产目标
            │       ├── op_mapping.yaml
            │       └── {KernelType}.csv             # ~52 个 CSV
            └── hccl/v8.5/               # HCCL 通信（和 CANN 版本绑定，跨 vLLM 版本复用）
                ├── hcom_allReduce_.csv
                ├── hcom_allGather_.csv
                ├── hcom_reduceScatter_.csv
                └── hcom_alltoallv_.csv

tools/perf_data_collection/               # 数据采集与数据库构建工具
├── parse_kernel_details.py               # 解析 kernel_details.csv → 按 Type 拆分
├── generate_shape_grid.py                # 生成 Shape 遍历网格（详见 §7.3）
├── start_microbench.py                   # msprof 编排 + op_summary 解析 + CSV 回写
├── op_replay/                            # 算子重放脚本（26 个 *_run.py）
│   ├── common.py                         # 共享工具：tensor 构建、CSV 迭代
│   └── run_all_op.py                     # 编排入口：发现并执行所有 *_run.py
├── generate_comm_microbench.py           # 生成通信算子 microbenchmark 脚本
└── compute_m6.py                         # E2E 预测精度（M6 ratio）计算
```

### 3.3 数据存储结构

**设计决策**：

| 决策项 | 方案 | 理由 |
|-------|------|------|
| 数据位置 | `tensor_cast/performance_model/profiling_database/data/` | 数据是 TensorCast 功能的一部分（类似 `device_profiles/`），用户通过 `--performance-model profiling` 时需要数据随包可用 |
| 文件格式 | CSV（Profiling 原始格式） | 与 kernel_details.csv 完全对齐，保证可溯源 |
| CSV 命名 | Profiling Type 列原始大小写 | `MatMulV2.csv`、`GroupedMatmul.csv` 等，与 Profiling 直接对应 |
| 大文件管理 | Git LFS | `.gitattributes` 中配置 `tensor_cast/performance_model/profiling_database/data/**/*.csv filter=lfs diff=lfs merge=lfs -text` |
| 计算/通信分离 | 计算在 `vllm_ascend/{version}/`，通信在 `hccl/{cann_version}/` | HCCL 通信行为只取决于 CANN 版本和硬件，与 vLLM 版本无关；升级 vLLM 不需要重测通信 |
| 目录命名约定 | `vllm{ver}_torch{ver}_cann{ver}` 完整版本字符串 | 避免版本歧义，支持同一设备下多版本共存（v1.3.1 更新） |
| MC2 数据位置 | `vllm_ascend/{version}/` | MC2（`npu_mm_all_reduce_base`）和 vLLM-Ascend 实现绑定 |

---

## 4. 核心模块设计

### 4.1 DataSource 抽象基类

```python
# tensor_cast/performance_model/profiling_database/data_source.py

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

class DataSource(ABC):
    """通用数据源抽象基类。
    TensorCast 只通过 OpInvokeInfo 查询，不感知底层映射关系和数据格式。"""

    @abstractmethod
    def lookup(self, op_invoke_info: "OpInvokeInfo") -> Optional[QueryResult]:
        """从 OpInvokeInfo 查询算子性能。内部处理映射/维度提取/数据查找。
        返回 None 表示未命中。"""
        ...

    def store(self, op_invoke_info: "OpInvokeInfo", result: QueryResult) -> None:
        """存储性能数据（可选，部分子类只读）。默认不支持写入。"""
        raise NotImplementedError("This DataSource is read-only")
```

### 4.2 ProfilingDataSource

基于预采集 Profiling CSV 的只读数据源。核心职责：`op_mapping.yaml` 映射 → CSV 查询 → FRACTAL_NZ 格式转换 → dtype 匹配。

**初始化**：接受 `(db_path, device_profile)` 参数（v1.3.1：`device_profile` 替代原 `comm_grid`，统一计算+通信数据的硬件参数访问）。加载 `op_mapping.yaml` + 按 `communication_data_ref` 加载通信数据（如有）。CSV 延迟加载并缓存。

**查询分派**（`lookup()` 内部）：

```
func_name → op_mapping.yaml 查 mapping
  ├─ mapping 不存在 → return None
  ├─ composite == true → _lookup_composite()
  ├─ category == "communication" → _lookup_comm()
  ├─ query_mode == "attention_special" → _lookup_attention()
  ├─ query_mode == "elementwise" → _lookup_elementwise()
  └─ 默认 → _lookup_compute()
```

**计算算子匹配**（`_lookup_compute` + `_inputs_match`）：从 `OpInvokeInfo.args` 提取所有 tensor 的 `(shape, dtype)`，逐行匹配 CSV。对 FRACTAL_NZ 格式的 input 先调用 `fractal_nz_to_nd()` 恢复为 ND shape，再做精确匹配。Output shape 作为验证。

**Input 数量不匹配处理**（v1.3.1 新增）：TC dispatch 的 input 数量可能与 Profiling CSV 不同。`_inputs_match()` 按 kernel_type 应用过滤规则：
- **默认**：取 `min(len(tc_inputs), len(csv_inputs))` 个 input 做匹配
- **QuantBatchMatmulV3**：仅匹配前 2 个 input（scale/zero_point 由 NPU 内部生成，CSV 中多出 2 个）
- **ReshapeAndCacheNdKernel**：仅匹配前 4 个 input（CSV 中 K/V cache 拆分为 5 个）
- **TensorMove**：仅匹配 src tensor（dst 在 CSV 中不记录）

> 原因：NPU kernel 实际接收的 input 可能包含 TC 层不可见的内部参数（量化 scale、buffer 指针等）。

**3D→2D Flatten Batch 匹配**（v1.4 新增）：quantize/norm 类 kernel（`AscendQuantV2`, `DynamicQuant`, `RmsNorm`, `AddRmsNormBias`, `AddRmsNorm`）的 TC dispatch 可能保留 3D `(B, M, D)` 格式，而 profiling CSV 存储 2D `(B*M, D)`。`_inputs_match` 在常规匹配（含 batch dim=1 strip 和 block-padding）失败后，对这些 kernel 尝试 `(B, M, D) → (B*M, D)` 展平匹配。此规则区别于通用 `_strip_batch_dim`（仅处理 B=1），仅对特定 kernel 类型生效，避免 matmul 等算子的 false positive。

**通信算子匹配**（`_lookup_comm`）：从 `args[0]` 计算 `message_bytes`，从 `rank_group`（位置因算子而异，见下表）推导 `topology_tier`。查询策略：精确匹配 `(message_bytes, num_devices, topology_tier)` 优先；未命中时按 `message_bytes` 做 alpha-beta 模型插值 — 对同一 `(num_devices, topology_tier)` 分组的全部数据点做最小二乘拟合 `latency = α + β × message_bytes`，结果 clamp 到 bracket 上下界。通信插值默认开启（与计算算子插值独立），因 `message_bytes` 为连续值，精确匹配命中率极低。

| Op | rank_group 位置 |
|----|----------------|
| all_reduce | args[2] |
| all_gather / reduce_scatter | args[3] |
| all_to_all | args[4] |

**FusedAttention 匹配**（`_lookup_attention`）：从 `args[6]`（seq_lens）计算 `batch_size` 和 `avg_seq_len`，在 FusedAttention CSV 中按 `(batch_size, avg_seq_len)` 查询。

**Composite 查询**（`_lookup_composite`）：针对 1:N 映射（如 MLA），ProfilingDataSource 主动分解为多个子内核查询并求和。流程：

1. 根据 `func_name` 查找注册的分解函数（`COMPOSITE_DECOMPOSERS`）；未注册则 return None fallback
2. 分解函数从 `OpInvokeInfo.args` 推导每个子内核的 `(kernel_type, shapes, dtype)`（区分 prefill/decode）
3. 逐个子内核调用 `_lookup_compute_by_shapes()` 查询 CSV
4. 任一子内核未命中 → 整体 return None；全部命中 → 求和 Duration 返回

**MLA 分解函数**：复用 `performance_model/__init__.py` 已有的 shape 推导逻辑。MLA 按 prefill/decode 分解为不同子内核：

| 阶段 | 子内核 | Shape 推导 |
|------|--------|-----------|
| Prefill | 1× TransposeBatchMatMul | `(num_tokens, kv_lora_rank) @ kv_b_proj` |
| Prefill | 1× FusedInferAttentionScore | `(num_tokens, num_heads, qk_head_dim)` + decompressed KV |
| Decode | 1× TransposeBatchMatMul (Q@W_UK_T) | `(num_tokens, num_heads, qk_nope_head_dim) @ W_UK_T` |
| Decode | 1× FusedInferAttentionScore | compressed attention via latent space |
| Decode | 1× TransposeBatchMatMul (AV@W_UV) | `(num_tokens, num_heads, kv_lora_rank) @ W_UV` |

维度从 `OpInvokeInfo.args` 提取：`num_heads = q.size(1)`, `kv_lora_rank = W_UK_T.size(-1)`, `qk_rope_head_dim = kv_cache.size(-1) - kv_lora_rank` 等。Prefill/Decode 通过 `query_lens` 阈值判定。

> 长期方案：MLA decomposition pass 完成后（见 9.1 节），每个子 op 独立生成 OpInvokeInfo，走普通 `_lookup_compute` 路径，composite 逻辑自然废弃。

**关键辅助函数**：

```python
def fractal_nz_to_nd(nz_shape):
    """[..., H, W, block_h, block_w] → [..., H*block_w, W*block_h]
    268 行零例外验证。恢复后与 aten.mm args[1] 一致，无需转置。"""
    *batch, H, W, block_h, block_w = nz_shape
    return (*batch, H * block_w, W * block_h)

DTYPE_MAP = {  # torch dtype → Profiling dtype string
    torch.bfloat16: "DT_BF16", torch.float16: "DT_BF16",
    torch.int8: "INT8", torch.int32: "INT32", torch.int64: "INT64",
    torch.float32: "FLOAT", torch.bool: "BOOL",
}
```

> FRACTAL_NZ 布局详细分析和验证数据见附录 B。

### 4.3 EmpiricalPerformanceModel

重构后的统一入口：接受 `DataSource` 实例，`process_op()` 先查数据源，未命中回退至 `fallback_model`（默认 `AnalyticPerformanceModel`）。

```python
# tensor_cast/performance_model/empirical.py
class EmpiricalPerformanceModel(PerformanceModel):
    def __init__(self, device_profile, data_source: DataSource,
                 fallback_model: Optional[PerformanceModel] = None):
        self.data_source = data_source
        self.fallback_model = fallback_model or AnalyticPerformanceModel(device_profile)

    def process_op(self, op_invoke_info):
        result = self.data_source.lookup(op_invoke_info)
        if result is not None:
            return Result(execution_time_s=result.latency_us * 1e-6, ...)
        return self.fallback_model.process_op(op_invoke_info)
```

**使用示例**：

```python
# Profiling 数据库驱动
pm = EmpiricalPerformanceModel(device_profile,
    data_source=ProfilingDataSource("data/.../vllm_ascend/v0.13.0/", device_profile))

# Profiling + 插值
pm = EmpiricalPerformanceModel(device_profile,
    data_source=InterpolatingDataSource(ProfilingDataSource("data/...", device_profile)))
```

### 4.4 InterpolatingDataSource

Wrapper 模式：精确匹配委托给 `base_source`，未命中时 `find_neighbors()` 查找近邻数据点，然后插值/外推。

**通用插值策略**（不需要 per-operator 维度声明）：
- **dtype + format 精确匹配**：已在 `ProfilingDataSource._inputs_match()` 中实现，所有算子通用
- **shape 维度线性插值**：未精确命中时，在 CSV 中找 dtype/format 精确匹配的行，对 shape 维度做最近邻搜索 + 线性插值
- **特殊变换**：仅 `FusedInferAttentionScore` 需要 sqrt 变换（Attention 延迟 ∝ seq²，在 √seq 空间插值精度更高）

此设计基于以下观察：95% 的算子（GEMM、Norm、Elementwise、MoE 等）共享相同的插值逻辑——dtype 精确匹配 + shape 线性插值。只有 FIA 因 O(n²) 复杂度需要额外处理。参考 AI Configurator 的实现，其 12 个 `query_*` 方法中只有 Attention 系列使用了 sqrt 变换，其余均为标准线性/立方插值。

`op_mapping.yaml` 的 `interpolation_policy` 声明默认策略和少量 kernel_type override。

### 4.5 op_mapping.yaml 规格

`op_mapping.yaml` 是纯名字映射配置，不含 per-op 维度提取逻辑。完整示例见 [`docs/examples/op_mapping_example.yaml`](examples/op_mapping_example.yaml)。

**`op_mapping.yaml` 顶层结构**：

```yaml
version: "0.15.0"                              # vLLM-Ascend 版本
device: ATLAS_800_A3_752T_128G_DIE
cann_version: "8.5"
communication_data_ref: "../../hccl/v8.5/"     # 通信数据相对路径
communication_fallback: analytic                # 通信数据不存在时 fallback

interpolation_policy:
  default_method: linear                       # 所有算子默认：dtype+format 精确匹配，shape 维度线性插值
  kernel_overrides:                            # 仅列出需要特殊处理的 kernel_type
    FusedInferAttentionScore:
      shape_transform: sqrt                    # O(seq²) → 在 √seq 空间插值

operator_mappings:
  "aten.mm.default":                           # 标准 aten 算子
    kernel_type: MatMulV2
  "tensor_cast.static_quant_linear.default":   # 量化 Matmul
    kernel_type: QuantBatchMatmulV3
  "tensor_cast.attention.default":             # 特殊查询模式
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
  "tensor_cast.all_reduce.default":            # 通信算子（引用 hccl 数据库）
    kernel_type: hcom_allReduce_
    category: communication
  "tensor_cast.multihead_latent_attention.default":  # 复合映射（1:N）
    composite: true
    sub_kernels: [TransposeBatchMatMul, FusedInferAttentionScore]
  ...                                          # 完整列表约 25 条

torch_npu_reference:                           # microbenchmark 脚本生成用
  MatMulV2:
    apis: [{name: "torch.mm"}, {name: "torch_npu.npu_linear"}]
    microbench_api: "torch.mm"
  QuantBatchMatmulV3:
    apis: [{name: "torch_npu.npu_weight_quant_batchmatmul"}]
    microbench_api: "torch_npu.npu_weight_quant_batchmatmul"
  ...                                          # 完整列表约 11 条
```

> 完整示例见 [`docs/examples/op_mapping_example.yaml`](examples/op_mapping_example.yaml)。

**`comm_config.yaml` 顶层结构**（位于 `hccl/{cann_version}/` 目录）：

```yaml
device: ATLAS_800_A3_752T_128G_DIE
cann_version: "8.5"

topology:
  grid_shape: [48, 8, 2]                       # 三维拓扑网格
  tiers:
    0: {name: "inter_pod", bandwidth_gbps: 196, latency_us: 5.5, type: "CLOS"}
    1: {name: "intra_pod", bandwidth_gbps: 196, latency_us: 0.5, type: "CLOS"}
    2: {name: "die_level", bandwidth_gbps: 224, latency_us: 0.2, type: "SIO"}

comm_operator_mappings:
  "tensor_cast.all_reduce.default": hcom_allReduce_
  "tensor_cast.all_gather.default": HcomAllGather
  ...
```

> 完整示例见 [`docs/examples/comm_config_example.yaml`](examples/comm_config_example.yaml)。

**`operator_mappings` 每条记录支持的字段**：

| 字段 | 说明 | 示例 |
|-----|------|------|
| `kernel_type` | Profiling Type 列名 | `MatMulV2` |
| `category` | 算子类别（驱动查询分派） | `communication` |
| `query_mode` | 特殊查询模式 | `attention_special`, `elementwise` |
| `composite` | 复合映射（1:N） | `true` |
| `sub_kernels` | 复合映射子内核 | `[TransposeBatchMatMul, ...]` |
| `notes` | 文档说明 | |

### 4.6 查询示例

**MatMulV2 (BF16, FRACTAL_NZ weight)**：
1. Runtime 拦截 `aten.mm.default(A[136,5120], B[5120,768])`
2. op_mapping.yaml: `aten.mm.default` → `MatMulV2`
3. 加载 MatMulV2.csv, 逐行匹配:
   - CSV: Input Shapes=`"136,5120;320,48,16,16"`, Formats=`"ND;FRACTAL_NZ"`
   - input[0]: ND (136,5120) → match ✓
   - input[1]: FRACTAL_NZ (320,48,16,16) → `fractal_nz_to_nd` → (5120, 768) → match ✓
   - dtype: DT_BF16 ↔ torch.bfloat16 → match ✓
4. Duration=45.3μs → `QueryResult(latency_us=45.3)`

### 4.7 通信算子数据格式

通信数据来源是 HCCL Test / `torch.distributed` microbenchmark（非全模型 Profiling）。

**通信 CSV 格式**：

```csv
message_bytes,num_devices,dtype,topology_tier,Duration(us),bandwidth_gbps
602112,8,DT_BF16,2,125.3,4.6
602112,4,DT_BF16,2,87.2,6.6
1204224,16,DT_BF16,0,342.1,1.7
```

**topology_tier 含义**（ATLAS_800_A3 [48,8,2] 三维网格）：tier 2 = die 级 SIO（224 GB/s），tier 1 = pod 内 CLOS（196 GB/s），tier 0 = pod 间 CLOS（196 GB/s, 5.5µs latency）。

**HCCL v8.5 数据质量注意事项**（v1.3.1，C10 采集经验）：
- 初版脚本每个 op 独立 `torchrun` 导致 HCCL JIT 重新初始化，小消息（4KB/16KB）延迟异常偏高
- 修正方案：单 session 运行所有 op + 全局预热（每个 op/group 先跑 1KB 触发 JIT 编译）+ `WARMUP_ITERS=20`
- tier=2（die_level，2 卡）数据已采集；tier=1（intra_pod，16 卡）数据已采集
- tier=0（inter_pod）需多节点（>16 卡）环境，暂 fallback analytic

**通信算子 OpInvokeInfo args 完整布局**：

| Op | args[0] | args[1] | args[2] | args[3] | args[4] |
|----|---------|---------|---------|---------|---------|
| all_reduce | Tensor x | int rank | List rank_group | | |
| all_gather | Tensor x | int dim | int rank | List rank_group | |
| reduce_scatter | Tensor x | int dim | int rank | List rank_group | |
| all_to_all | Tensor x | List out_splits | List in_splits | int rank | List rank_group |

### 4.8 FusedAttention 特殊处理

#### 4.8.1 KV 维度问题

Profiling 中 `FusedInferAttentionScore` 的 KV Input Shapes 是预分配 KV cache pool 的 buffer shape，不是实际 KV 长度，无法直接匹配。这是 vLLM PagedAttention 的 by-design 行为：paged 路径（decode + chunked prefill）传给 FIA kernel 的是整个 KV cache pool + block_table 间接寻址 + `actual_seq_lengths_kv` 标量列表。唯一例外是 non-paged prefill（PrefillNoCache 路径），KV 是真实长度，但实际数据极少（CANN 8.5 中仅 2/24 行）。

CANN 8.3 和 8.5 在 KV 维度上行为一致，均为预分配 pool 大小。

#### 4.8.2 为什么需要 avg_seq_len

FIA duration 是 O(query_tokens × context_length)。raw profiling CSV 只记录 tensor shape 不记录值，`actual_seq_lengths_kv` slot 只有 shape `(batch_size,)` 信息，无法获取每个请求的真实 context length。因此 **profiling FIA 数据不能用于性能查询**（匹配上也不可信），仅用于验证 E4 microbench 精度。

`avg_seq_len = mean(actual_seq_lengths_kv)` 作为 batch 级别的 context length 聚合量，由 E4 microbenchmark 控制输入获得。

#### 4.8.3 Enriched CSV 格式（v1.5 更新）

FIA 使用 **enriched CSV** 格式：raw profiling 31 slots + 2 个新增列，不需要维护单独的结构化 CSV 格式。

| enriched 列 | 类型 | 说明 |
|-------------|------|------|
| `avg_seq_len` | INT | mean(actual_seq_lengths_kv)，单位 token。决定 O(S_kv) 计算量 |
| `sparse_mode` | INT | 0=no_mask, 1=all_mask, 2=left_up_causal, 3=right_down_causal, 4=band。causal(2/3) 跳过 ~50% 计算 vs 全量(0/1) |

```
# Enriched FIA CSV: raw profiling 列 + 2 个 enriched 列
OP State,Accelerator Core,Input Shapes,...,Duration(us),...,avg_seq_len,sparse_mode
dynamic,MIX_AIC,"336,4,128;12308,128,128;...",62.6,...,4096,3
```

**以下维度可从 raw CSV 推导，不需要 enriched 列**：num_heads（Q slot 0 第二维）、head_dim（Q slot 0 第三维）、num_kv_heads（K 最后维 / head_dim）、dtype（Input Data Types）、query tokens T（Q slot 0 第一维）。

数据来源：
- **E4 Microbenchmark**：控制输入参数，`avg_seq_len` 和 `sparse_mode` 直接已知
- **Profiling 数据**：enriched 列填 -1，不参与查询，仅做 E4 验证 ground truth

#### 4.8.4 查询逻辑（v1.5 更新）

`_lookup_attention()` 直接查询 enriched CSV，匹配 6 个维度：

1. 从 CSV Input Shapes slot 0 解析 Q shape，归一化到 3D (T, N, D)
2. 从 K shape 推导 `num_kv_heads`（K 最后维 / head_dim）
3. 固定 (N, D, num_kv_heads, dtype, sparse_mode) 精确匹配
4. `avg_seq_len` 精确匹配或插值（sqrt transform）
5. 忽略 KV slots（pool 大小无语义）

`_interpolate_attention()` 在 enriched CSV 的 `avg_seq_len` 上做 sqrt 插值，其余 5 个维度精确匹配作为 filter。

#### 4.8.5 CANN 版本 Layout 差异

同一个 MLA decode FIA，CANN 不同版本使用不同 Q layout：
- 8.3: TND `(T, 16, 512)` — 3D
- 8.5: BNSD `(B, 16, 1, 512)` — 4D

TC dispatch 始终是 3D `(num_tokens, num_heads, qk_head_dim)`。匹配时统一将 CSV 的 Q shape squeeze 到 3D 再比较。

#### 4.8.6 MLA composite 中的 FIA 子内核

MLA composite 的 decode 路径中 FIA 通过 `_lookup_attention_by_params()` 查询 enriched CSV。Prefill 路径不走 FIA（见 §4.8.7）。

#### 4.8.7 DSV3 MLA Prefill 路径修正（v1.5 新增）

DSV3 prefill 阶段不使用 `FusedInferAttentionScore`，而是使用 `RINGMLAPrefillBF16Kernel`（vllm-ascend ATB ring attention 实现，融合 ring attention + KV projection）。MLA decomposer（`_decompose_mla`）的 prefill 分支应生成 `RINGMLAPrefillBF16Kernel` SubKernelSpec，不再生成 FIA。

op_mapping.yaml 中 MLA entries 的 `sub_kernels` 更新为 decode/prefill 所有 kernel 的并集，实际路由由 decomposer 硬编码控制。

**RINGMLAPrefillBF16Kernel 和 FIA 有相同的"tensor 值控制计算量"问题**：

- `seqlen` 参数（CPU tensor, int32, shape `[batch]` 或 `[2, batch]`）包含 per-request 实际 token 数
- 调用链：`torch_npu.atb.npu_ring_mla(seqlen=query_lens)` → `op-plugin/RingMlaAtb.cpp` → `ATB/ring_mla_operation.cpp` → `RINGMLAPrefillTiling(totalTaskNum=sum(qSeqLen))`
- CSV 只记录 tensor shape，不记录 seqlen 值。同 Q shape `(2048,16,128)` 下 duration 差 3.5 倍（309us vs 89us），证明 shape 匹配不够
- **因此 RING kernel 也不能用标准 compute 路径匹配（`_inputs_match`），需要类似 FIA 的 enriched CSV 方案**（raw + seqlen 相关列）
- Microbench 回放脚本 `RINGMLAPrefillBF16Kernel_run.py` 因缺 seqlen 默认全量计算，导致与整网实测出现 ~7.5 倍 gap

> 详细分析见 `reports/FIA_KV_DIMENSION_ANALYSIS_v0.1.md`、`reports/FIA_LOOKUP_AND_MLA_RESEARCH_20260321.md`、`reports/FIA_TODO_20260322.md`；args 布局见附录 C。

### 4.9 FRACTAL_NZ 通用处理

恢复公式 `[..., H, W, block_h, block_w] → [..., H*block_w, W*block_h]` 见 4.2 节。Tile 大小从 CSV shape 最后两维直接读取（BF16: 16×16, INT8: 16×32）。恢复后与 `aten.mm args[1]` 一致。

> 详细的布局分析、源码溯源和验证数据见附录 B。

**权重转置匹配**（v1.4 补充）：FRACTAL_NZ 恢复为 ND shape 后，对 matmul 类 kernel（`_MATMUL_KERNELS` 集合，含 `MatMulV2`, `MatMulV3`, `QuantBatchMatmulV3`, `BatchMatMulV2`, `TransposeBatchMatMul` 等）仍需尝试权重转置匹配 `(K,N) ↔ (N,K)`。此规则不限于 ND 格式 — FRACTAL_NZ 恢复后的 shape 同样可能需要转置。

### 4.10 逐元素算子输出形状匹配 (Elementwise Output-Shape Matching)

对于内存带宽受限的逐元素算子 (Add, Mul, Div), 使用**输出形状**而非输入形状进行匹配:

- **输出形状确定性**: 无论标量/向量/逐元素广播, 输出形状相同
- **dtype 松弛匹配**: FP32 ≈ BF16 × (4/2), 按字节比缩放延迟 (`_dtype_byte_size()`)
- **适用条件**: `Duration ∝ output_elements × bytes_per_element / memory_bandwidth`
- **置信度**: dtype 相同 = 1.0; dtype 不同 (缩放后) = 0.9; 插值 + dtype 缩放 = 0.6

在 `op_mapping.yaml` 中标记 `query_mode: elementwise`。与 `tc_input_count` 互斥。

---

## 5. TensorCast 集成接口

### 5.1 Runtime 集成

`Runtime` 类无需修改。它已支持接收 `PerformanceModel` 实例列表，并自动包装为 `CachingPerformanceModel`。

```python
# 使用 ProfilingDataSource
data_source = ProfilingDataSource(
    "profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/v0.13.0",
    device_profile,
)
perf_model = EmpiricalPerformanceModel(device_profile, data_source)
runtime = Runtime(perf_models=perf_model, device_profile=device_profile)
```

### 5.2 CLI 接口

在 `cli/inference/text_generate.py` 中支持多性能模型并行：

```python
parser.add_argument("--performance-model",
                    action="append",
                    default=None,
                    help="性能模型类型，可多次指定。"
                         "'analytic': Roofline 模型（默认，无需数据）。"
                         "'profiling': 基于实测 Profiling 数据库的 EmpiricalPerformanceModel "
                         "（需要 --profiling-database）。")
parser.add_argument("--profiling-database", type=str, default=None,
                    help="性能数据库路径（profiling 模式生效），"
                         "指向包含 op_mapping.yaml 和 CSV 数据文件的目录")

# 默认值处理
if args.performance_model is None:
    args.performance_model = ["analytic"]
```

**使用示例**：

```bash
# 单个模型（默认）
--performance-model analytic

# 多个模型并行运行
--performance-model analytic --performance-model profiling --profiling-database /path/to/db
```

| CLI 选项 | 创建的模型 | 是否需要物理设备 | 是否需要数据库 |
|---------|----------|----------------|---------------|
| `--performance-model analytic` | `AnalyticPerformanceModel` | 否 | 否 |
| `--performance-model profiling` | `EmpiricalPerformanceModel(ProfilingDataSource(...))` | 否 | 是（`--profiling-database`） |
| 多次指定 | 多个模型并行运行，输出多份结果 | 否 | 按需 |

### 5.3 UserInputConfig 配置

`UserInputConfig.performance_model` 支持 `Union[str, List[str]]`：

```python
@dataclass
class UserInputConfig:
    performance_model: Union[str, List[str]] = "analytic"
    """性能模型类型：'analytic' | 'profiling'。
    可以是单个字符串或字符串列表以运行多个模型。"""

    def _normalize_performance_model(self):
        """将 performance_model 规范化为字符串列表。"""
        pm = self.performance_model
        if isinstance(pm, str):
            self.performance_model = [pm]
```

### 5.4 ModelRunner 多模型支持

`ModelRunner.__init__` 构建多个 `PerformanceModel` 实例：

```python
class ModelRunner:
    def __init__(self, user_input: UserInputConfig):
        perf_model_types: List[str] = user_input.performance_model
        self.perf_models: List[PerformanceModel] = []
        
        for perf_model_type in perf_model_types:
            if perf_model_type == "profiling":
                data_source = ProfilingDataSource(
                    user_input.profiling_database,
                    self.device_profile,
                )
                self.perf_models.append(
                    EmpiricalPerformanceModel(
                        self.device_profile,
                        data_source=data_source,
                        fallback_model=AnalyticPerformanceModel(self.device_profile),
                    )
                )
            elif perf_model_type == "analytic":
                self.perf_models.append(AnalyticPerformanceModel(self.device_profile))
```

### 5.5 ModelRunnerMetrics 输出

`ModelRunnerMetrics` 存储每个模型的执行时间和 TPS：

```python
@dataclass
class ModelRunnerMetrics:
    single_card_tps: float  # 第一个模型的 TPS（向后兼容）
    execution_time_s: Dict[str, float]  # 每个模型的执行时间，按模型名索引
    tps_per_model: Dict[str, float]  # 每个模型的 TPS，按模型名索引
    # ... 其他字段

    def print_info(self):
        for model_name, exec_time in self.execution_time_s.items():
            print(f"[{model_name}] Execution time: {exec_time:.6f} s")
            tps = self.tps_per_model.get(model_name)
            if tps is not None:
                print(f"[{model_name}] TPS/Device: {tps:.4g} token/s")
```

### 5.6 数据流

`--performance-model profiling` → 创建 `EmpiricalPerformanceModel(ProfilingDataSource(data_dir))` → Runtime 拦截算子生成 `OpInvokeInfo` → `data_source.lookup()` 查询（分派逻辑见 4.2 节）→ 命中返回实测耗时，未命中 fallback 至 Roofline/CommAnalytic。

---

## 6. 数据库构建策略

### 6.1 三步走策略

| 步骤 | 内容 | 输出 |
|-----|------|------|
| **步骤一**：单次 vLLM Profiling | 拉起一次 vLLM 实例，采集 kernel_details.csv | 按 Type 列拆分的 `{KernelType}.csv` + `op_mapping.yaml` 初版 |
| **步骤二**：Microbenchmark 网格遍历 | 针对每个算子，通过 op_replay 脚本 + `msprof` 遍历 shape 网格 | 扩充后的 CSV 数据库（含 MicroBench Duration 列）|
| **步骤三**：端到端验证 | 跑典型 vLLM 配置，用 `compute_m6.py` 对比 microbenchmark 预测 vs Profiling 实测 | M6 精度报告 + duration gap 热点分析 |

**步骤一详细流程**：
1. 拉起 vLLM 实例，采集 kernel_details.csv（只需能跑起来即可，EP=1 配置）
2. `parse_kernel_details.py` → 按 Type 列分组聚合，拆分为各 `{KernelType}.csv`，保留 25+ 个 NPU 性能指标列（aicore_time, aic_mac_ratio, cube_utilization 等）
3. 同时基于映射表 + 运行时环境信息生成 `op_mapping.yaml` 初版

> 注：EP=1 时 Profiling 中不会出现 AllToAll 通信，AllToAll 数据通过步骤二的通信 microbenchmark 独立采集。

**步骤二详细流程**：

计算算子（op-replay pipeline）：
1. `generate_shape_grid.py` → 基于现有 CSV 模板生成扩展 shape 行（详见 §7.3 Shape 网格策略）
2. `start_microbench.py` → 编排 `op_replay/run_all_op.py`，在 `msprof` 下运行 26 个算子重放脚本
3. `start_microbench.py` → 解析 `op_summary_*.csv` 输出，按 shape signature 聚合，回写 `MicroBench Duration(us)` 列到各算子 CSV
4. 生成报告：`profile_update_report_*.md`（更新摘要）+ `duration_gap_hotspots_full_*.csv`（microbench vs profiling 偏差热点）
5. 特殊: FusedAttention 用 `actual_seq_lengths_kv` 控制实际 KV 长度（见 §4.8）

通信算子（4-mode 采集框架）：
1. `run_comm_bench.sh` → 编排 `generate_comm_microbench.py`，分两轮采集
   - Round 1（alternating 模式）：allReduce 全量 + allGather/reduceScatter ≥1MB，peer→target 无同步流水线消除 warmup 偏差
   - Round 2（kernel 模式）：allGather/reduceScatter <1MB，用 kernel_details 提取避免 event 计时 floor（~60µs）
2. `build_comm_csv.py` → 数据源选择 + dispatch overhead 修正（vLLM scheduler→c10d→HCCL 开销），输出最终 `hcom_*.csv`
3. `validate_comm_alignment.py` → 对比 CommAnalyticModel（Ring/Tree/Recursive 算法），输出 PASS/WARN/FAIL 三级报告

> 通信数据的 4 种采集模式（profiler/kernel/alternating/event）适用于不同精度-速度权衡场景，另有 pipeline 模式作为向后兼容。详见 `docs/perf_database/COMM_BENCH_GUIDE.md`。

### 6.2 计算算子 Microbenchmark：Op-Replay 框架

通过 `torch_npu` 直接对单个算子进行细粒度 Shape 覆盖测试。核心思路：**从 CSV 读取 shape → 重建 tensor → 执行算子 → msprof 采集 device 侧数据**。

**op_replay 框架**（`tools/perf_data_collection/op_replay/`）：

| 组件 | 作用 |
|------|------|
| `common.py` | 共享工具：tensor 构建、FRACTAL_NZ 展开、CSV 迭代、dtype 映射 |
| `run_all_op.py` | 编排入口：发现并执行所有 `*_run.py` 脚本，支持 `--op` 过滤 |
| `{KernelType}_run.py` ×26 | 算子重放脚本：读 CSV 行 → `build_input_tensor()` → 执行 torch_npu API → sync |

**执行流程**：
```bash
# start_microbench.py 编排：
msprof python tools/perf_data_collection/op_replay/run_all_op.py \
  --device ATLAS_800_A3_752T_128G_DIE \
  --vllm-ascend-version 0.15.0 \
  --op MatMulV2 RmsNorm SwiGlu     # 可选：只跑指定算子

# start_microbench.py 后处理：
# 1. 解析 PROF_*/op_summary_*.csv
# 2. 按 (op_type, shape_signature) 聚合
# 3. 回写 MicroBench Duration(us) + aicore_time(us) 等列到各算子 CSV
```

**覆盖的 26 个算子**：MatMulV2, MatMulV3, MatMulCommon, QuantBatchMatmulV3, FusedInferAttentionScore, RmsNorm, AddRmsNormBias, SwiGlu, KvRmsNormRopeCache, split_qkv_rmsnorm_rope_kernel, AscendQuantV2, DynamicQuant, InterleaveRope, RINGMLAPrefillBF16Kernel, Transpose, Slice, Pad, ReshapeAndCacheNdKernel, GatherV2, Add, SoftmaxV2, MaskedFill, ArgMaxV2, Sort, TensorMove, DispatchFFNCombine

> 新增算子只需编写对应的 `{KernelType}_run.py` 并实现 `run_row(csv_path, row_index, row)` 接口。op-plugin 库可用于反查 Profiling Type → torch_npu API，详见附录 E。

### 6.3 通信算子 Microbenchmark

**统一方案**：计算和通信都用 Python 脚本 + `msprof`，HCCL Test 作为交叉验证。

```python
# tools/perf_data_collection/generate_comm_microbench.py
# 初始化 HCCL backend，遍历 message_size 网格，
# 用 torch.npu.Event 计时 warmup + N 次重复的 dist.all_reduce/all_gather/all_to_all，
# 通过 rank_group 控制测试各 topology_tier
```

**MC2 microbenchmark**：MC2（`npu_mm_all_reduce_base`）将 MatMul+AllReduce 融合，无法分开测试。用 Python 脚本直接调用 `torch_npu.npu_mm_all_reduce_base` 遍历不同 M/N/K + num_devices。MC2 数据存储在 `vllm_ascend/{version}/` 下（和框架实现绑定）。

**HCCL Test 工具**（交叉验证用）：
```bash
# 位于 CANN toolkit: /usr/local/Ascend/ascend-toolkit/latest/tools/hccl_test/
mpirun -n 8 ./bin/all_reduce_test -b 8K -e 2048M -f 2 -d fp16 -o sum -p 8
mpirun -n 8 ./bin/all_gather_test -b 8K -e 512M -f 2 -d fp16 -p 8
mpirun -n 8 ./bin/all_to_all_test -b 8K -e 256M -f 2 -d fp16 -p 8
```

> 参考文档：https://www.hiascend.com/document/detail/zh/mindstudio/70RC1/mscommandtoolug/mscommandug/auxiliarydevtool_0017.html

### 6.4 FusedAttention Microbenchmark

FusedAttention 需构造合法的 paged KV cache 输入：预分配 KV buffer + `block_table` + `actual_seq_lengths_kv` 控制实际 KV 长度。遍历 `(batch_size, seq_len, num_heads, head_dim)` 组合，调用 `torch_npu.npu_fused_infer_attention_score`。详见附录 C 的 args 布局。

E4 microbench 输出使用 enriched CSV 格式（raw 31 slots + `avg_seq_len` + `sparse_mode` 列），详见 §4.8.3。

### 6.5 Profiling 输出解析器

```python
# tools/perf_data_collection/parse_kernel_details.py

class KernelDetailsParser:
    def parse(self, csv_path: Path) -> pd.DataFrame:
        """解析 kernel_details.csv → 结构化 DataFrame。
        按 Type 列分组，保留 Profiling 原始列。
        支持单文件或目录扫描（多次 profiling 合并）。"""

    def split_by_type(self, df: pd.DataFrame, output_dir: Path):
        """按 Type 列拆分为各 {KernelType}.csv。
        聚合同 (op_type, input_shapes, output_shapes) 组的统计量：
        mean/std/median duration + 25 个 NPU 性能指标均值。
        CSV 保持 Profiling 原始格式：
        Input Shapes, Input Data Types, Input Formats,
        Output Shapes, Output Data Types, Output Formats,
        Profiling Average Duration(us), Profiling Std Duration(us),
        Profiling Median Duration(us), Accelerator Core, Block Dim,
        aicore_time(us), aic_mac_time(us), aic_mac_ratio,
        aic_scalar_time(us), aiv_time(us), aiv_vec_time(us),
        aiv_scalar_time(us), cube_utilization(%)
        """
```

### 6.6 端到端验证

```python
# tools/perf_data_collection/compute_m6.py

# M6 = Empirical_HIT_total_us / Real_per_forward_pass_us
# - 分子：HIT ops 的 empirical (microbench CSV) 延迟之和
# - 分母：step_trace_time.csv 的 (Computing + Comm_NotOverlapped) / N_forward_passes
# - N_forward_passes 通过 ArgMaxV2（token sampling 算子）出现次数自动推断

# 输出：M6 ratio、组成分析、未匹配 kernel 诊断
# 目标：0.85 ≤ M6 ≤ 1.15
```

### 6.7 工具链总览

| 工具 | 阶段 | 用途 |
|------|------|------|
| `parse_kernel_details.py` | 步骤一 | vLLM kernel_details.csv → 按 Type 拆分的 per-kernel CSV |
| `generate_shape_grid.py` | 步骤二 | 基于理论范围生成 shape 网格行（详见 §7.3） |
| `op_replay/run_all_op.py` | 步骤二 | 编排 26 个算子重放脚本 |
| `start_microbench.py` | 步骤二 | msprof 编排 + op_summary 解析 + CSV 回写 |
| `generate_comm_microbench.py` | 步骤二 | 通信 microbench 脚本生成 |
| `compute_m6.py` | 步骤三 | E2E 预测精度（M6 ratio）计算 |

---

## 7. 全面算子覆盖策略

### 7.1 算子分级

| 层级 | 判定标准 | 处理方式 |
|-----|---------|---------|
| **Tier 1** | 执行耗时占比 >2%（任一模型） | 完整 Shape 网格 Microbenchmark |
| **Tier 2** | 执行耗时占比 0.5-2% | 精简 Shape 网格或 Profiling 直接导入 |
| **Tier 3** | 执行耗时占比 <0.5% | Roofline 兜底 |

### 7.2 Tier 1 完整算子列表

> 数据来源：全量内核耗时（含通信），Qwen3-32B Prefill 16 卡 / DSV3 Decode 32 卡。

| Profiling Type | Qwen3 占比 | DSV3 占比 |
|---------------|-----------|---------|
| hcom_allReduce\_ | **89.8%** | 8.1% |
| GroupedMatmul | - | **18.7%** |
| QuantBatchMatmulV3 | - | **18.3%** |
| FusedInferAttentionScore | 1.7% | **16.5%** |
| MoeDistributeDispatchV2 | - | **7.0%** |
| MatMulV2 | **4.1%** | 0.5% |
| AscendQuantV2 / DynamicQuant | - | **3.9%** |
| TransposeBatchMatMul | - | **3.8%** |
| MoeDistributeCombineV2 | - | **3.6%** |
| InterleaveRope | - | **2.5%** |
| DequantSwigluQuant / SwiGlu | 0.5% | **2.4%** |
| InplaceAddRmsNorm / AddRmsNorm | 0.8% | 1.8% |
| HcomAllGather | 0.6% | 1.7% |

### 7.3 Shape 网格策略

#### 设计原理

每个算子的 shape 是 `f(model_config, parallel_config, runtime_config)` 的确定性函数。维度可分为两类：

- **固定维度**（N, K, heads, head_dim 等）：由模型架构 + 并行策略决定，取值为有限离散集
- **变化维度**（M/num_tokens, seq_len）：由推理 runtime 决定，在连续范围内变化

基于此，shape 网格策略为：对固定维度定义**覆盖常见 LLM 架构值的模型无关网格**，对变化维度在理论上下界内采样。

#### 固定维度：模型无关网格

固定维度的理论范围由 LLM 架构空间决定：
- **上界**：并行策略=1 时的最大值（如 `hidden_size` 原始值）
- **下界**：最大并行度下的最小值（如 `hidden_size / 32`）

`generate_shape_grid.py` 内部维护覆盖常见 LLM 架构值的离散集，无需读取具体模型配置。例如 GEMM 的 N/K 网格覆盖 `[128, 256, 512, ..., 8192, 12288, 16384, 27648, 55296]`，囊括主流模型（Qwen3, DSV3, LLaMA, Kimi-K2 等）在各 TP 配置下的实际值。

#### 变化维度：理论上下界 + 按算子特征采样

| 算子类型 | 变化维度 | 范围 | 采样策略 | 依据 |
|---------|---------|------|---------|------|
| **GEMM** | M (num_tokens) | [1, max_batch_tokens] | powers-of-2 骨架 + tiling 中间值 | 阶梯函数（tiling boundary 处性能跳变） |
| **Elementwise** | num_tokens | [1, max_batch_tokens] | 稀疏均匀（~15-20 点） | 近似线性（memory-bound，α+β×numel） |
| **Attention** | seq_len / avg_seq_len | [1, max_context_len] | sqrt 空间均匀 | O(seq²) 复杂度 → √seq 空间线性化（见 §4.4） |
| **MOE** | num_tokens_per_expert | [1, max_tokens×topk/n_local_experts] | powers-of-2 | 类似 GEMM，但范围受 routing 限制 |
| **通信** | message_bytes | [1KB, 512MB] | powers-of-2 + 生产工作点 | α-β model + protocol 切换点 |

#### 与 InterpolatingDataSource 的协同

Shape 网格策略和 InterpolatingDataSource（§4.4）形成互补：
- 网格提供**足够密度的 bracket 点**，确保 InterpolatingDataSource 的 1D 插值有上下界
- 固定维度（N, K, heads）要求**精确匹配**（不插值），因此网格需覆盖所有可能取值
- 变化维度（M, seq_len）支持**插值**，网格只需覆盖关键采样点

#### 预估 Shape 总量

参考 [AI Configurator](https://github.com/ai-dynamo/aiconfigurator) 在 H100 上的实践（~65K 数据点覆盖 GEMM + Attention + MOE），预估 NPU 侧每个设备 × 每个 CANN 版本约需 **20,000-50,000** 个 shape 点。具体各算子的网格设计详见附录 I。

### 7.4 无法 Microbenchmark 的算子影响分析

| 算子 | DSV3 Decode 占比 | Qwen3 Prefill 占比 | 原因 | 方案 |
|-----|-------------|-------------|------|------|
| TensorMove | 0.06% | 1.03% | 完全由 runtime 调度决定 | 忽略，用 Roofline 兜底 |
| split_qkv_rmsnorm_rope_kernel | 0% (DSV3 无) | 0.50% | 无公开 torch_npu API | 全模型 Profiling 获取；或待新增融合 pass |
| ReshapeAndCacheNdKernel | 0% (DSV3 无) | 0.32% | 同上 | 同上 |
| Graph Mode 融合算子 | - | - | 仅在 ACLGraph 模式下触发 | 无法独立 benchmark |
| **合计** | **0.06%** | **1.85%** | | |

**结论**：无法 microbenchmark 的算子仅占 0.06%-1.85%，对端到端精度影响极小（远在 <15% 目标内）。

### 7.5 评估指标体系（v1.4 新增，v1.4.1 扩展 M4-M6）

六层指标体系，从粗到细、从算子计数到延迟加权：

| 指标 | 分子 | 分母 | 排除 zc | 悲观 | 融合 | 计算方式 | 用途 |
|------|------|------|:---:|:---:|:---:|---------|------|
| **M1**: Raw Op-Count HR | HIT 调用次数 | 总调用次数 | — | — | — | 在线 | Debug，向后兼容 |
| **M2**: Fused Op HR | HIT 唯一 func_name 数 | 所有唯一 func_name 数 | — | ✓ | ✓ | 在线 | **GO/NO-GO 判定** |
| **M3**: Fused Op HR (不含 zc) | 同 M2 排除 zero_cost | 同 M2 排除 zero_cost | ✓ | ✓ | ✓ | 在线 | 真实计算覆盖 |
| **M4**: Per-Shape Match HR | HIT 唯一 (func, shape) 数 | 所有唯一 (func, shape) 数 | ✓ | — | — | 在线 | Shape 缺口诊断 |
| **M5**: Simulated Latency Coverage | HIT ops 的 analytic 延迟之和 | 所有 ops 的 analytic 延迟之和 | — | — | — | 在线 | 仿真视角延迟覆盖率 |
| **M6**: Empirical E2E Ratio | HIT ops 的 empirical (microbench) duration 之和 (全模型 replay) | step_trace_time / N_forward_passes (单次 forward pass) | — | — | — | 半离线 | **验收标准：M6=1.0 完美，0.85–1.15 为 Phase 3 目标** |

#### 指标详细定义

**M1–M3**: 算子计数层

- M1 按每次 `process_op()` 调用计数，被 zero_cost ops 虚增
- M2/M3 按唯一 `func_name` 计数，应用悲观规则和融合分组
- **悲观规则**: 同一 `func_name` 若有 N 次 HIT + M 次 MISS（不同 shape），整个算子计为 1 MISS
- **融合分组**: DFC（init_routing_v2 + grouped_matmul×2 + unpermute_tokens + all_to_all×2）= 1 融合 op；MLAPO、MLA、MC2 同理。全部成员 HIT 才算 HIT

**M4**: Shape 诊断层

- 每个唯一的 `(func_name, input_shape_tuple)` 独立计数
- 不应用悲观规则，不应用融合分组，排除 zero_cost
- 用于定位具体缺失哪些 shape 的数据，指导 microbenchmark 数据采集

**M5**: 仿真延迟覆盖层

- 分子分母均使用 **analytic (Roofline) 延迟**作为权重
- 本质是 M3 的延迟加权版本：高延迟算子权重大，低延迟辅助算子权重小
- 在线计算，不需要外部数据

**M6**: Empirical E2E Prediction Ratio（v1.5 重新定义）

```
M6 = Empirical_HIT_total / Real_per_forward_pass
```

- 跑一次 TC `--performance-model profiling --export-metrics report.json` 得到 empirical HIT duration
- **分子** = HIT ops 的 empirical (microbench CSV) 延迟之和，乘以 replay_multiplier 得到全模型 replay 总量。**不含 analytic fallback**（MISS ops 不计入）
- **分母** = `step_trace_time.csv` 的 `(Computing + Comm_NotOverlapped) / N_forward_passes`
  - `step_trace_time.csv` 的 `Step` 列为空时，聚合了整个 profiling 窗口的所有 forward passes
  - `N_forward_passes` 从 `kernel_details.csv` 估算：Qwen3 用 `FIA_count / 64`，DSv3 用 `DFC_count / 58`
  - 也可通过 `--n-forward-passes` 手动指定
- M6 = 1.0 表示完美预测；M6 > 1 表示 empirical 高估（microbench isolation effect）；M6 < 1 表示覆盖不足（MISS ops 未贡献）
- `*AicpuKernel` 条目为 AICPU dispatch wrapper，仅用于诊断（排除）

**为什么只看 empirical-only（不含 analytic fallback）？**
- Analytic fallback 用 Roofline 模型估算 MISS ops 延迟，其精度受限于通信模型、不可预测的 kernel fusion 等因素
- 将 analytic fallback 计入 M6 会混淆"empirical 数据质量"和"analytic 模型准确度"两个不同问题
- M6 的目的是评估 **已有 empirical 数据的质量和覆盖度**：M6 < 1 说明覆盖不足（需要更多 microbench 数据），M6 > 1 说明 microbench 数据偏高（isolation vs real workload）

**M5 vs M6**: M5 从 TC 仿真视角看（权重=analytic Roofline，在线），M6 从 empirical 预测 vs NPU 真实单次 forward pass 看（半离线）。M5 含 analytic fallback，M6 不含。

#### Phase 目标

- Phase 1: M1–M3 指标体系建立 ✅
- Phase 2: M3 > 50%, M4/M5 建立 ✅
- Phase 3: M5 > 80%, **0.85 ≤ M6 ≤ 1.15**（±15% 以内）

---

## 8. 开发计划

**团队分工**：
- **六壬工具团队**：TensorCast 侧（EmpiricalPerformanceModel 重构、DataSource 接口、CLI 集成、融合 Pass 补齐）
- **小巧灵团队**：数据侧（ProfilingDataSource 实现、CSV 解析、Microbenchmark 脚本、数据采集）

**前置依赖**：TensorCast 算子追踪对齐（详见第 9.1 节），需在集成测试前完成。

**目标**：2026.3.23 完成端到端集成，DeepSeek-V3 / Qwen3-32B 能对齐的算子覆盖端到端 >90% 的时间。

### Phase 1：计算算子穿刺（3.4 - 3.14）

联合对齐 Qwen3-32B 和 DeepSeek-V3 在目标版本上的算子列表和 op_mapping.yaml。

| 六壬团队 | 小巧灵团队 |
|---------|-----------|
| `EmpiricalPerformanceModel` 重构 + DataSource 接口 | `op_mapping.yaml` 规格定义 + CLI 集成 |
| `ProfilingDataSource` 实现（CSV 查询 + FRACTAL_NZ） | Profiling 解析器（`parse_kernel_details.py`） |
| 基于示例数据库穿刺端到端仿真 | 穿刺数据采集（单次 Profiling + 初始 CSV） |

**Phase 1 交付物**：Profiling CSV 导入 + ProfilingDataSource + 端到端查询验证（计算算子）

### Phase 2：通信算子接入 + 插值（3.14 - 3.20）

| 六壬团队 | 小巧灵团队 |
|---------|-----------|
| `InterpolatingDataSource` 实现 | 通信 microbenchmark 脚本（`generate_comm_microbench.py`） |
| 扩展融合 Pass（MC2、KvRmsNormRopeCache 等） | hccl CSV 构建 + `comm_config.yaml` |
| CommGrid 集成到 ProfilingDataSource | Shape 网格扩充（`generate_shape_grid.py`） |

**Phase 2 交付物**：通信 microbenchmark + hccl CSV + InterpolatingDataSource

### Phase 3：集成验证（3.20 - 3.23）

联合使用 Qwen3-32B 和 DeepSeek-V3 进行端到端精度验证。

**验证标准**：

| 指标 | 目标值 |
|-----|-------|
| 端到端耗时误差 | <15%（对比实际 vLLM Profiling） |
| 单算子误差（已匹配算子） | <20% |
| 时间覆盖率 | >90% |

**Phase 3 交付物**：精度对比报告 + 文档收尾

---

## 9. 外部依赖与扩展建议

### 9.1 前置依赖：TensorCast 算子追踪对齐

本方案要求 TensorCast dispatch trace 与 vLLM Profiling 的硬件内核列表在关键算子上 1:1 对齐。

**当前状态**（v1.3.1 更新，基于 feat/perf-database 分支 + Phase 1 验证）：

| Gap 项 | 状态 | 详情 | Profiling 占比 |
|--------|------|------|---------------|
| SwiGlu 融合 | **已关闭** ✓ | `patterns/swiglu.py` 支持双序 mul 模式 | DSV3 2.4% |
| GroupedMatmul+SwiGlu 融合 | **已关闭** ✓ | `freezing_passes/grouped_matmul_swiglu_pass.py`，5 种量化变体 | DSV3 含在 GroupedMatmul 中 |
| MC2（MatMul+AllReduce） | **已验证** ✓ | BF16 + W8A8 dispatch trace 确认融合正确（3f82c2b）；composite 分解查询已实现 | composite 分解 |
| KvRmsNormRopeCache | **已关闭** ✓ | mlapo op 已覆盖此融合，无需独立 pass（穿刺验证 3f82c2b） | DSV3 0.8% |
| TP Padding bug | **已修复** ✓ | 从全局 input padding 移到 MoE layer-local（59cf184），修复 DSV3 Decode 8x 高估 | P0 |
| MLA 分解 | composite 兜底 | `_lookup_composite()` 分解查询已实现（3a5af02）；长期需 decomposition pass | DSV3 相关 |
| DispatchFFNCombine | **新增，开放** | CANN 8.5 超级融合，融合 `all_to_all×2 + GroupedMatmul×2 + SwiGlu + MoE routing`。TC 用 composite 分解兜底，需子内核 CSV 数据（C11） | **DSV3 35.3%** |
| split_qkv_rmsnorm_rope | 仍开放 | 无对应 pass/op | Qwen3 0.5% |
| aten.topk / MoeGatingTopK | 仍开放 | 无区分机制 | DSV3 (2378 次调用) |

> 注：develop 分支发现一个 bug：`grouped_matmul_*_swiglu` 系列 ops 输出 shape 为 (M, N) 但应为 (M, N//2)（SwiGlu 将 gate+up 对半分后输出 half width）。

### 9.2 融合算子语义一致性分析

大部分 TensorCast op → Profiling kernel Type 映射 shape 一致（如 MatMul、RmsNorm、SwiGlu、通信算子）。需要特殊处理的有：
- **FusedAttention**：TC 用 `(num_tokens, hidden_size)`，Profiling 用 31 slot raw 格式（KV 是 pool shape）。通过 enriched CSV（raw + avg_seq_len 列）+ `attention_special` 模式处理（v1.5 重设计，详见 §4.8）
- **MLA**：1:N 映射，decode 对应 TransposeBatchMatMul + FIA + TransposeBatchMatMul，prefill 对应 MatMulV2 + RINGMLAPrefillBF16Kernel（v1.5：prefill 不走 FIA）。通过 `composite: true` + `_lookup_composite()` 分解查询覆盖
- **init_routing_v2**：TC 只含本地 permute，Profiling 的 MoeDistributeDispatchV2 含通信，通信由 all_to_all 分开计时

> 完整映射表和已发现的 bug 见附录 H。

### 9.3 CompositePerformanceModel

长期建议设计 `CompositePerformanceModel`，按优先级组合多个 PerformanceModel，实现可配置的降级策略（`Profiling → Empirical → Analytic`）。引入后 `EmpiricalPerformanceModel` 不再需要内部持有 `fallback_model`，三种模型完全解耦。

### 9.4 其他扩展建议

- **跨硬件泛化**：支持更多 DeviceProfile（如 Atlas A2、GPU），需为每种硬件独立采集数据
- **自动化 CI**：随 vLLM-Ascend / CANN 版本发布自动触发数据采集流水线
- **预插值优化**：参考 AI Configurator 的 `_extrapolate_data_grid`，在数据库加载时预填充常用网格点
- **SOL 数据校正**：用 Roofline 理论下界校正异常测量值

---

## 10. 遗留问题与 Future Work

### 10.1 vLLM 自定义算子覆盖

**问题**：vLLM 使用的 NPU pybind 算子（如 `npu_fused_infer_attention_score`）不走 PyTorch Dispatch，TensorCast 从原理上无法直接捕获。

**当前方案（中期）**：
1. **TensorCast 融合 Pass 1:1 对齐**：对于每个 pybind 算子，TensorCast 已有对应的自定义算子（如 `tensor_cast.attention` 对应 `npu_fused_infer_attention_score`），通过编译 Pass 将 aten 算子融合为对应的 TensorCast 算子
2. **MC2 新增融合 Pass**：将 `aten.mm + tensor_cast.all_reduce` 融合为新的 `tensor_cast.mm_all_reduce`
3. **手工映射表**：对于无法自动对齐的算子，在 `op_mapping.yaml` 中手工维护映射

**长期方案（Future Work）**：
1. **路径 A：vLLM FX Graph 抓取**：直接从 vLLM 运行时用 `torch.compile` 或 `torch.fx.symbolic_trace` 抓取算子图，避免依赖 TensorCast 的 dispatch trace
2. **路径 B：侵入式 DispatchMode**：侵入式修改 vLLM 添加 DispatchMode（内部湛卢团队方案，待交流），将 `torch_npu` 的 pybind 调用也纳入 trace

> 完整的 vLLM 自定义算子覆盖分析见附录 D。

### 10.2 FusedAttention KV Cache 维度

**问题**：Profiling 中 `FusedInferAttentionScore` 的 Input Shapes 包含预分配 KV Cache buffer shape，不是实际 KV 长度，与 TensorCast 理论建模无法直接匹配。

**当前方案**：通过 microbenchmark 构建数据库，以 `(batch_size, avg_seq_len, num_heads, head_dim, dtype)` 为索引，使用 `actual_seq_lengths_kv` 参数控制实际 KV 长度。

**长期方案（Future Work）**：
1. **路径 A**：从 `torch.compile` FX graph 抓取 vLLM 算子图，直接获取带真实维度的算子 trace
2. **路径 B**：侵入式修改 vLLM 添加 DispatchMode（内部湛卢团队方案），从 vLLM 实跑获取带真实 seq_lens 的 profiling 数据

> 详细分析见附录 C。

### 10.3 通信算子数据库完善

Phase 2 实施通信算子 microbenchmark 数据库：
- 通过 `torch.distributed` Python 脚本精确控制 `rank_group` 测试各 `topology_tier`
- HCCL Test 工具交叉验证
- MC2 融合通信单独处理

---

## 11. 依赖项

**新增**：`scipy`（插值）、`packaging`（版本解析）
**现有**：`pandas`、`numpy`、`pyyaml`、`torch`
**可选**：`pyarrow`（Parquet 格式支持，初期可仅用 CSV）

---

## 12. 参考资料

- [vLLM Ascend GitHub](https://github.com/vllm-project/vllm-ascend)
- [vLLM Ascend Profiling 指南](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/service_profiling_guide.html)
- [vLLM CustomOp Replacement Tracker](https://github.com/vllm-project/vllm/issues/32676)
- [torch_npu Operator Inventory](https://github.com/vllm-project/vllm-ascend/issues/1511)
- [vLLM torch_bindings.cpp](https://github.com/vllm-project/vllm/blob/main/csrc/torch_bindings.cpp)
- [op-plugin (torch_npu op mapping)](https://github.com/Ascend/op-plugin)
- [华为昇腾 Profiler 文档](https://support.huaweicloud.com/intl/en-us/bestpractice-modelarts/modelarts_llm_infer_5906034.html)
- [HCCL Test 文档](https://www.hiascend.com/document/detail/zh/mindstudio/70RC1/mscommandtoolug/mscommandug/auxiliarydevtool_0017.html)
- [AI Configurator](https://github.com/ai-dynamo/aiconfigurator)
- [Intel NPU Cost Model](https://github.com/intel/npu-nn-cost-model)
- [msModeling Wiki](https://deepwiki.com/Horacehxw/msModeling)
- MC2 参考: [vllm-ascend#6092](https://github.com/vllm-project/vllm-ascend/issues/6092), [vllm-ascend#5743](https://github.com/vllm-project/vllm-ascend/issues/5743)

---

## 附录 A：小 Batch 场景 Roofline 偏差分析

**小 batch Prefill 同样受影响，甚至更难预测**：
- **Decode (bs=1, seq=1)**：GEMM shape 为 M=1，本质是矩阵-向量乘，明确 memory-bound，Roofline 至少能正确识别瓶颈方向，但会高估可达带宽
- **小 batch Prefill (bs=1, seq=256-1024)**：M=seq_len 处于 memory-bound 到 compute-bound 的过渡区域（Roofline 曲线的"拐点"位置），此时性能高度依赖 kernel 实现质量（tiling 策略、cache 利用率），Roofline 误差最大
- **大 batch Prefill**：M 足够大，tile 能填满所有 AI Core，最接近 Roofline 预测

**NPU 微架构分析**：Da Vinci 核心的 Cube 单元期望大型 tile 输入（通常 16×16 最小），M=1 时 Cube 处理大量 padding 零值。

## 附录 B：FRACTAL_NZ 布局分析与验证

### 布局公式

标准公式（从 `torch_npu` 源码 `FormatHelper.cpp` 确认）：

对矩阵 `[rows, cols]` 做 tiling → `[ceil(cols/N0), ceil(rows/M0), M0, N0]`

其中 `M0 = 16`（固定），`N0 = BLOCKBYTES / min(itemsize, 2)`：
- BF16/FP16: N0 = 32/2 = 16
- INT8: N0 = 32/1 = 32

**恢复公式**：`[..., H, W, block_h, block_w]` → `[..., K, N]`，其中 `K = H * block_w, N = W * block_h`

### 验证数据

| 算子 | Weight 基础布局 | Tile 尺寸 | FRACTAL_NZ Shape | 恢复公式 |
|-----|---------------|---------|-----------------|---------|
| MatMulV2 (BF16) | [N,K] (同 nn.Linear) | 16×16 | [K/16, N/16, 16, 16] | K=dim0×16, N=dim1×16 |
| QuantBatchMatmulV3 (INT8) | [K,N] (转置存储) | 16×32 | [N/32, K/16, 16, 32] | K=dim1×16, N=dim0×32 |
| GroupedMatmul (INT8) | [E, K, N] | 16×32 | [E, N/32, K/16, 16, 32] | K=dim2×16, N=dim1×32 |

**实际验证**：
- Qwen3 BF16: input [136,5120] + weight FRACTAL_NZ [320,48,16,16] → K=320×16=5120, N=48×16=768 → output [136,768] ✓
- DSV3 INT8: input [42,7168] + weight FRACTAL_NZ [48,448,16,32] → K=448×32=14336? → 用通用公式: K=H×block_w=448×32, N=W×block_h=48×16=768

**268 行零例外验证**：在 Qwen3-30B Prefill 的 268 行 FRACTAL_NZ MatMulV2 数据上，使用通用恢复公式与 Output Shapes 交叉验证，全部通过。

**关键结论**：
- 恢复后 shape 和 `aten.mm` 的 `args[1]` 完全一致，**无需转置**
- 原因：`nn.Linear` 的 weight `[N,K]` dispatch 到 `aten.mm` 时已转置为 `[K,N]`，NPU 的 FRACTAL_NZ 存的也是转置后的 `[K,N]` 形式
- 没有 TransData 算子——FRACTAL_NZ 转换在 torchair 图编译时完成，不产生运行时 kernel

## 附录 C：FusedAttention KV Cache 维度分析

问题描述和解决方案见第 4.8 节。此处补充 TensorCast `OpInvokeInfo` 中 FusedAttention 的 args 布局：

| 位置 | 含义 | 备注 |
|-----|------|------|
| `args[0]` | query tensor | (num_tokens, hidden_size) |
| `args[1]` | key tensor | |
| `args[2]` | value tensor | |
| `args[6]` | `seq_lens` | 每个 request 的 KV cache 长度，对应 `actual_seq_lengths_kv` |
| `args[7]` | `query_lens` | 每个 request 的新 query token 数 |

## 附录 D：vLLM 自定义算子覆盖分析

通过分析 vLLM `torch_bindings.cpp` 和 vllm-ascend issue #1511，主要的 vLLM/Ascend 特有算子：

| 算子 | PyTorch Dispatch 可捕获？ | TensorCast 覆盖情况 |
|-----|------------------------|-------------------|
| npu_fused_infer_attention_score | 否（pybind） | 已有 `tensor_cast.attention` 对应 |
| npu_grouped_matmul | 否（pybind） | 已有 `tensor_cast.grouped_matmul` 对应 |
| npu_mm_all_reduce_base (MC2) | 否（pybind） | 未覆盖，需新增 TensorCast 融合 pass |
| npu_moe_distribute_dispatch/combine | 否（pybind） | 已有 `tensor_cast.init_routing_v2/unpermute_tokens` |
| npu_dequant_swiglu_quant | 否（pybind） | develop 分支已有融合 pass |
| npu_kv_rmsnorm_rope_cache | 否（pybind） | 未覆盖 |
| npu_dynamic_quant | 否（pybind） | 已有 `tensor_cast.dynamic_quantize_*` |
| vLLM silu_and_mul (CUDA) | 是（torch.library） | 已有融合 pass |
| vLLM paged_attention_v1/v2 (CUDA) | 是（torch.library） | 已有 `tensor_cast.attention` |

## 附录 E：op-plugin Type → torch_npu 映射分析

op-plugin 库（https://github.com/Ascend/op-plugin）的 `op_plugin/config/op_plugin_functions.yaml` 包含 7148 行、1200+ 算子映射。

**核心 kernel Type 到 torch_npu API 的映射**：

| Profiling Type | torch_npu API | 在 op_plugin_functions.yaml? | 备注 |
|---------------|--------------|---------------------------|------|
| MatMulV2 | npu_linear / aten::mm | ✓ (ACL path: OpCommand.Name("MatMulV2")) | |
| QuantBatchMatmulV3 | npu_weight_quant_batchmatmul | ✓ (OpAPI path) | |
| FusedInferAttentionScore | npu_fused_infer_attention_score | ✓ (有 V2/V3/V4 版本) | |
| GroupedMatmul | npu_grouped_matmul | ✓ | |
| AddRmsNorm | npu_add_rms_norm | ✓ | |
| DequantSwigluQuant | npu_dequant_swiglu_quant | ✓ | |
| KvRmsNormRopeCache | npu_kv_rmsnorm_rope_cache | ✓ | |
| MoeGatingTopK | npu_moe_gating_top_k | ✓ | |
| AscendQuantV2 | npu_quantize | ✓ (名字完全不同) | |
| DynamicQuant | npu_dynamic_quant | ✓ | |
| SwiGlu | npu_swiglu | ✓ | |
| InterleaveRope | npu_interleave_rope | ✓ | |
| hcom_allReduce_ | torch.distributed.all_reduce | ✗ (HCCL 通信库) | |
| split_qkv_rmsnorm_rope_kernel | 无公开 API | ✗ (vLLM-Ascend 自定义 kernel) | |

**Microbenchmark 默认 API**（`op_mapping.yaml` 中 `torch_npu_reference.{type}.microbench_api`）：

| Profiling Type | microbench_api |
|---------------|---------------|
| MatMulV2 | `torch.mm` |
| QuantBatchMatmulV3 | `torch_npu.npu_weight_quant_batchmatmul` |
| FusedInferAttentionScore | `torch_npu.npu_fused_infer_attention_score` |
| GroupedMatmul | `torch_npu.npu_grouped_matmul` |
| AddRmsNorm | `torch_npu.npu_add_rms_norm` |
| SwiGlu | `torch_npu.npu_swiglu` |
| DynamicQuant | `torch_npu.npu_dynamic_quant` |
| AscendQuantV2 | `torch_npu.npu_quantize` |
| hcom_allReduce\_ | `torch.distributed.all_reduce` |
| HcomAllGather | `torch.distributed.all_gather` |
| hcom_alltoall\_ | `torch.distributed.all_to_all` |

**关键发现**：
- 13/15 个核心 kernel Type 可通过 op-plugin 追溯到 torch_npu API
- 命名不一致：MatMulV2→npu_linear、AscendQuantV2→npu_quantize，无法自动推导
- op-plugin 有两条路径：ACL path 用 `OpCommand.Name("KernelType")` 直接匹配 Type 列，OpAPI path 用 aclnn 前缀

## 附录 F：AIConfigurator 通信方案对比

| 维度 | AIConfigurator | 本方案 |
|-----|---------------|-------|
| 数据来源 | NCCL intra-node 实测 + 带宽缩放推算 inter-node | 各 topology_tier 分别实测 |
| 存储结构 | `data/{device}/nccl/{version}/nccl_perf.txt` | `data/{device}/hccl/{cann_version}/` |
| 拓扑处理 | CSV 不存拓扑，动态推导 `_get_p2p_bandwidth()` | CSV 存 `topology_tier` 整数列 |
| 跨层通信 | 带宽比例缩放（scale_factor = base_bw / target_bw） | 直接实测各层级 |

**选择理由**：
- AIConfigurator 只能 intra-node 实测 + 解析缩放（因为在 NVIDIA 上 NCCL 的行为复杂）
- 我们在 Ascend 上用 HCCL Test / `torch.distributed` 可以精确控制 `rank_list` 来测任意拓扑层级
- 不需要解析缩放，直接用实测数据更准确

## 附录 G：AI 辅助开发实践建议

建议在本项目开发中采用以下 AI 辅助开发实践：

1. **接口先行**：先定义 `DataSource` 抽象接口和 `op_mapping.yaml` schema
2. **Planning**：使用 `/superpowers:writing-plans` 生成实现计划
3. **TDD**：使用 `/superpowers:test-driven-development`，先生成测试再实现
4. **CLAUDE.md 维护**：使用 `/claude-md-management:revise-claude-md` 保持项目上下文更新
5. **代码评审**：使用 `/superpowers:requesting-code-review` 在 PR 前自动评审
6. **Agent 开发**：使用 `/superpowers:dispatching-parallel-agents` 并行开发独立模块（如 ProfilingDataSource 和 InterpolatingDataSource 可以并行开发）

命名术语应匹配本文档：`EmpiricalPerformanceModel`、`DataSource`、`ProfilingDataSource`、`InterpolatingDataSource`、`op_mapping.yaml` 等。

## 附录 H：融合算子语义一致性详细分析

完整的 TensorCast op → Profiling kernel Type 映射和语义一致性分析：

| TensorCast Op | Profiling Kernel | Shape 一致? | 说明 |
|--------------|-----------------|------------|------|
| tensor_cast.static_quant_linear | QuantBatchMatmulV3 | ✓ M/K/N 对齐 | INT4 变体 w=[K/2,N]，需注意 K 恢复 |
| tensor_cast.attention | FusedInferAttentionScore | 需转换 | TC 用 (num_tokens, hidden_size)，Profiling 用 (batch, num_heads, q_len, head_dim)。通过 `attention_special` 查询模式处理 |
| tensor_cast.multihead_latent_attention | 1:N 映射 | MISMATCH | 一个 TC op 对应多个 kernel (TransposeBatchMatMul + FIA)。Q1 通过 `composite: true` + `_lookup_composite()` 分解查询子内核并求和；长期需 MLA decomposition pass |
| tensor_cast.mlapo | 无直接对应 | N/A | Qwen3 对应 split_qkv_rmsnorm_rope_kernel，DSV3 对应多个分立 kernel |
| tensor_cast.init_routing_v2 | MoeDistributeDispatchV2 | 部分 | TC 只含本地 permute，Profiling 含通信；TC 的通信由 all_to_all 分开计时 |
| tensor_cast.add_rms_norm | AddRmsNorm / InplaceAddRmsNorm | ✓ | |
| tensor_cast.swiglu | SwiGlu | ✓ | DequantSwigluQuant 是更大的融合，需单独 pass |
| 所有 comm ops | hcom_allReduce_ 等 | ✓ | message_bytes + num_devices 对齐 |

**已发现的 Bug**：
1. gitcode/develop 分支的 `grouped_matmul_*_swiglu` 输出 shape 为 (M, N) 但应为 (M, N//2)
2. `CommAnalyticModel` 中 `reduce_scatter` 缺少 dispatch 分支（已有方法但未被调用）

## 附录 I：理论导向 Shape Grid 详细设计

> 参考 [AI Configurator](https://github.com/ai-dynamo/aiconfigurator) 在 H100 上的实践：GEMM ~26K 点（21M × 19N × 19K × 3 dtype），Attention ~21K 点，MOE ~14K 点，总计 ~65K 点/设备。

### I.1 算子维度分类

每个算子的 shape 维度分为两类，决定了采样策略：

| 维度类型 | 含义 | 取值特征 | 插值策略 |
|---------|------|---------|---------|
| **固定维度** | 由 model_config + parallel_config 决定 | 离散有限集 | **精确匹配**（不插值） |
| **变化维度** | 由 runtime（batch/seq）决定 | 连续范围 | **插值**（linear/cubic/sqrt） |

各算子的维度分类：

| 算子 | 固定维度 | 变化维度 |
|------|---------|---------|
| MatMul (QKV/O/Gate/Up/Down) | N, K: hidden_size/tp, intermediate_size/tp | M (num_tokens) |
| GroupedMatmul (MoE expert) | N, K: expert_dim/tp, expert_intermediate/tp | M (tokens/expert) |
| QuantBatchMatmulV3 | N, K: 同 MatMul | M |
| Elementwise (Add, RmsNorm, SwiGlu) | D: hidden_size/tp 或 intermediate_size/tp | num_tokens |
| FusedInferAttentionScore | num_heads/tp, head_dim, num_kv_heads/tp | batch, avg_seq_len |
| RoPE (InterleaveRope) | num_heads/tp, head_dim | seq_len |
| Quantize (AscendQuant, DynamicQuant) | D: hidden_size/tp | num_tokens |
| MoE dispatch/combine | num_experts, expert_dim/ep | num_tokens |
| Communication (allReduce) | — | message_bytes = M × D × sizeof(dtype) |
| TransposeBatchMatMul (MLA) | num_heads/tp, head_dim | batch, seq |

### I.2 固定维度网格

覆盖常见 LLM 架构在所有 TP/EP 配置下的实际值。网格设计原则：
1. 包含主流模型的精确值（不依赖插值）
2. 值域覆盖 `[D_base/max_tp, D_base]`，即 TP=1 为上界
3. 对齐到 16（NPU Cube/Vector core 要求）

**GEMM N/K 网格**（~30 个值）：

```
[128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1728, 2048,
 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 10240, 12288, 13824, 16384, 27648, 55296]
```

覆盖主流模型验证：

| 模型 | hidden_size | intermediate_size | TP=1 | TP=4 | TP=8 | TP=16 |
|------|------------|-------------------|------|------|------|-------|
| Qwen3-32B | 5120 | 27648 | ✓ | 1280, 6912 | 640, 3456 | 320, 1728 |
| DSV3 | 7168 | 2048 (expert) | ✓ | 1792→❌ | 896 | 448 |
| LLaMA-70B | 8192 | 28672 | ✓ | 2048, 7168 | 1024, 3584 | 512, 1792→❌ |
| Kimi-K2 | 7168 | 2048 (expert) | ✓ | 同 DSV3 | — | — |

> ❌ 标记的值（1792, 28672）不在网格中，需补充。最终网格应在实际部署时根据目标模型列表校验并补齐。

**Attention heads 网格**（~15 个值）：

```
[1, 2, 3, 4, 5, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 128]
```

**head_dim**：离散集 `{64, 128, 256}`（几乎所有 LLM 只用这三个值）

**num_kv_heads**（GQA）：`{1, 2, 4, 8, 16, 32}`

### I.3 变化维度采样策略

#### GEMM M (num_tokens)

**性能特征**：阶梯函数 — Cube core tiling 策略在特定 M 值处切换，性能呈阶梯跳变。

**采样策略**：powers-of-2 骨架 + tiling 中间值（~23 个点）

```
[1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512, 768, 1024,
 2048, 4096, 8192, 16384, 32768]
```

**设计依据**：
- powers-of-2（1, 2, 4, ..., 32768）覆盖数量级跨度
- 中间值（48, 80, 96, 160, 192, 384, 768）填充 tiling boundary 附近
- 参考 AI Configurator 实践：21 个 M 值覆盖 [1, 8192]，本方案扩展至 32768

#### Elementwise num_tokens

**性能特征**：近似线性（memory-bound），latency ≈ α + β × numel。

**采样策略**：稀疏 powers-of-2（~15 个点）

```
[1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
```

**设计依据**：线性函数只需少量点即可精确插值。关键是覆盖 bandwidth 饱和点（通常在 numel > hidden_size 时）。

#### Attention seq_len / avg_seq_len

**性能特征**：O(seq²) 复杂度 → √seq 空间近似线性。

**采样策略**：sqrt 空间均匀（~20 个点）

```
[1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
```

**设计依据**：
- √seq 空间等距 → raw 空间在大 seq 处自然稀疏
- AI Configurator 的 context attention 使用 `sqrt_y_value=True` 处理 seq 维度
- 与 InterpolatingDataSource 的 `shape_transform: sqrt` 配合（§4.4）

#### MOE num_tokens_per_expert

**性能特征**：类似 GEMM（GroupedMatmul 本质是 batched GEMM）。

**采样策略**：powers-of-2（~20 个点）

```
[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
```

**注意**：MOE 的固定维度（hidden_size, intermediate_size, num_experts, topk）因模型而异，组合过多无法穷举。参考 AI Configurator 的做法——**维护目标模型列表**，为每个模型的 (hidden, inter, experts, topk) 组合 × TP/EP 配置采集数据。

#### 通信 message_bytes

**性能特征**：α-β model + protocol 切换点。

**采样策略**：powers-of-2 + 生产工作点（已由 HDY 实现）

```
标准网格: [1KB, 2KB, 4KB, ..., 512MB]  （20 个点）
生产工作点: Qwen3 (nd=16) + DSV3 (nd=8) 实际通信量
```

### I.4 Shape 总量预估

| 算子类型 | 固定维度组合 | 变化维度采样 | 总点数/算子 | 算子数 | 小计 |
|---------|------------|-----------|-----------|-------|------|
| GEMM | ~30 (N) × ~30 (K) | ~23 (M) | ~20,700 | 3-4 | ~60K |
| Elementwise | ~30 (D) | ~15 | ~450 | 5-6 | ~2.5K |
| Attention | ~15 (heads) × 3 (head_dim) × 6 (kv_heads) | ~20 (seq) × 9 (batch) | ~48,600 | 1-2 | ~50-100K |
| MOE | 模型列表 (~5) × TP/EP (~10) | ~20 (tokens) | ~1,000 | 2-3 | ~2-3K |
| 通信 | ~4 (topology) × ~4 (num_devices) | ~20 (msg_bytes) | ~320 | 4 | ~1.3K |
| 其他 (Quant, RoPE, ...) | ~30 (D 或 heads) | ~15-20 | ~500 | 5-6 | ~3K |
| **总计** | | | | | **~70-170K** |

> 注：Attention 点数较大（因固定维度组合多），可通过 head_dim 和 kv_heads 实际使用值裁剪。GEMM 可通过仅采集目标模型实际出现的 (N, K) 对来大幅减少。实际采集量取决于精度需求和设备可用时间。

### I.5 与当前 Template Mutation 方案的对比

| 维度 | 当前方案（template mutation） | 理论导向方案 |
|------|--------------------------|------------|
| 模板来源 | vLLM profiling 中出现的 shape | 模型无关的理论网格 |
| 采样范围 | `[template/2, template×2]` | `[1, max_value]`（全范围覆盖） |
| 采样策略 | 均匀随机 | 按性能曲线特征（powers-of-2, sqrt, tiling-aware） |
| 模型绑定 | 是（模板来自特定模型 profiling） | 否（GEMM/Attention/Elementwise 模型无关） |
| 覆盖保证 | 无（10K random samples 无法保证覆盖特定 shape） | 有（固定维度精确覆盖，变化维度有 bracket 保证） |
| 新模型适配 | 需重新 profiling 获取模板 | GEMM/Attn 自动覆盖；MOE 需加入模型列表 |
| 插值协同 | InterpolatingDataSource 可能无 bracket | bracket 覆盖有保证 |

### I.6 实现路径

`generate_shape_grid.py` 的改造方向：

1. **新增模式**：`--mode theory`（理论导向，替代当前的 template mutation 模式 `--mode template`）
2. **维度配置**：工具内部维护 NK_GRID / M_GRID / HEADS_GRID 等常量，按 kernel_type 分派
3. **MOE 模型列表**：`--moe-models` 参数接受模型列表（或读取配置文件）
4. **裁剪**：`--target-models` 可选参数，用于只采集指定模型实际出现的 (N, K) 对，减少采集量
5. **保持向后兼容**：`--mode template` 保留当前 template mutation 行为

---

## Change Log

### v1.4 → v1.5 (2026.3.24)

**重设计**:
- §4.8: FusedAttention 特殊处理全面重写
  - KV 维度问题：确认 CANN 8.3/8.5 均为 pool shape（by design），profiling FIA 数据不能用于查询
  - CSV 格式：结构化 CSV `(batch_size, avg_seq_len, ...)` → enriched CSV（raw 31 slots + `avg_seq_len` + `sparse_mode` 2 列）
  - 查询逻辑：`_lookup_attention()` 覆盖为 6 维匹配（N, D, num_kv_heads, dtype, sparse_mode 精确 + avg_seq_len sqrt 插值），忽略 KV slots
  - CANN 源码分析：FIA 4 条 kernel 路径（FAI/IFA/PFA/V3），sparse_mode 2/3 跳过 ~50% 计算 vs 0/1，num_kv_heads 影响 GQA dispatch
  - CANN 版本差异：新增 8.3 TND vs 8.5 BNSD layout 归一化（squeeze 到 3D）
- §6: 数据库构建方案全面更新
  - §6.2: 计算算子 microbench 从脚本生成模式（`generate_microbench.py`）改为 op-replay 框架（`op_replay/` 26 个算子脚本 + `start_microbench.py` 编排）
  - §6.6: `discover_operators.py` → `compute_m6.py` 端到端验证
  - §6.7: 新增工具链总览表
- §7.3: Shape 网格策略从"模型配置提取 + powers-of-2 补充"改为"理论导向的模型无关网格 + 按算子性能特征采样"

**新增**:
- §4.8.7: MLA composite prefill 路径修正 — DSV3 prefill 不走 FIA，走 RINGMLAPrefillBF16Kernel
- §4.8.7: RINGMLAPrefillBF16Kernel 和 FIA 同类问题分析 — `seqlen` 参数（CPU tensor）控制实际计算量，CSV 不记录值，同 shape 下 duration 差 3.5 倍，不能用 compute 路径匹配，需要 enriched CSV
- §4.8.5: CANN 8.3/8.5 Q layout 差异处理
- 附录 I: 理论导向 Shape Grid 详细设计（含各算子维度分类、网格建议、AI Configurator 参考、与 template mutation 对比）

**修正**:
- §9 Shape 对齐：MLA composite 描述更新为 decode/prefill 分别映射不同 kernel
- MLA decomposer prefill 分支：FIA SubKernelSpec → RINGMLAPrefillBF16Kernel（需 enriched CSV，不能用 compute 路径）
- §6: 删除对已移除工具的引用（`generate_microbench.py`、`discover_operators.py`、`build_database.py`）

**上下文**: (1) FIA 重设计基于 ZH 的 FIA KV 维度分析报告和 MLA 研究报告（3.21），HXW review 后补充 avg_seq_len 维度分析和方案修正。(2) §6 更新反映 TCX (@Secluded_Ocean) 的 op-replay pipeline 和 HDY (@Hudingyi) 的 5-mode 通信 bench 工具链。(3) §7.3 和附录 I 的理论导向 shape grid 策略参考 [AI Configurator](https://github.com/ai-dynamo/aiconfigurator) 实践。详见 `reports/FIA_TODO_20260322.md`。

### v1.3.1 → v1.4 (2026.3.15)

**新增**:
- §4.2: 通信 alpha-beta 模型插值（least-squares fit，默认开启）
- §4.2: 3D→2D Flatten Batch 匹配规则（quantize/norm 类 kernel）
- §4.9: FRACTAL_NZ 恢复后权重转置匹配（所有 matmul 变体）
- §7.5: M1-M5 五层评估指标体系 + 悲观规则 + 融合分组

**变更**:
- §4.2: 通信查询从"精确匹配"改为"精确匹配优先 + alpha-beta 插值 fallback"
- 版本: 1.3.1 → 1.4

**上下文**: Phase 1 E2E 集成验证（4 场景，Qwen3-32B + DSv3）完成后，基于验证发现的 shape 匹配规则和评估指标需求更新。详见 `reports/phase1-e2e-20260314/phase1_e2e_v2_verification_report_zh.md`

### v1.2 → v1.3 (2026.3.10)

#### 新增功能

- **[CLI]** `--performance-model` 支持多次指定，可同时运行多个性能模型
  - 旧用法: `--performance-model analytic` (单选)
  - 新用法: `--performance-model analytic --performance-model profiling` (多选)
  - 默认值: `["analytic"]`

- **[ModelRunnerMetrics]** 新增 `tps_per_model: Dict[str, float]` 字段
  - 存储每个性能模型独立计算的 TPS
  - `print_info()` 遍历输出每个模型的结果

#### 接口变更

| 组件 | 变更 | 影响 |
|------|------|------|
| `UserInputConfig.performance_model` | `str` → `Union[str, List[str]]` | 向后兼容，字符串自动包装为列表 |
| `ModelRunnerMetrics.execution_time_s` | `float` → `Dict[str, float]` | **Breaking**: 下游代码需适配字典类型 |
| `ModelRunner.perf_model` | `PerformanceModel` → `List[PerformanceModel]` | 内部变更，API 不变 |
| `ProfilingDataSource.__init__` | 移除 `comm_grid` 参数，改用 `device_profile` | **Breaking**: 调用方需更新参数 |

#### 代码质量

- **[model_runner.py]** `PerformanceModel` 导入移至 `TYPE_CHECKING` 块 (RUFF TC001)
- **[user_config.py]** 新增 `_normalize_performance_model()` 规范化逻辑
- **[user_config.py]** 新增 `word_embedding_tp_mode` 字段及 `_normalize_embedding_tp_mode()` 方法

#### 测试适配

- `test_text_generate.py`: `execution_time_s` 相关断言适配 `Dict[str, float]`
- `test_vl_compile.py`: 同上
- `test_text_generate.py`: `ModelRunnerMetrics` 构造新增 `tps_per_model` 参数

#### 文件变更清单

```
cli/inference/text_generate.py           | CLI 参数改为 action="append"
tensor_cast/core/user_config.py          | performance_model 类型变更 + WordEmbeddingTPMode
tensor_cast/core/model_runner.py         | 多模型支持 + ModelRunnerMetrics 字段变更
tests/test_tensor_cast/test_text_generate.py  | 测试适配
tests/test_tensor_cast/test_vl_compile.py      | 测试适配
```

### v1.1 → v1.2

**架构变更**：
1. `ProfilingPerformanceModel` → `EmpiricalPerformanceModel` + `DataSource` 模式
2. `PerfDatabase` → `DataSource` ABC，首要实现 `ProfilingDataSource`
3. `OperatorSchema` 类消除 → 职责分散到 `op_mapping.yaml`（名字映射）+ `InterpolatingDataSource`（插值配置）+ `ProfilingDataSource`（通用匹配逻辑）
4. `QueryEngine` → `InterpolatingDataSource`（Wrapper 模式）
5. `OperatorKey` 消除 → `ProfilingDataSource.lookup(OpInvokeInfo)` 直接查询

**数据格式变更**：
6. CSV 命名：蛇形 → Profiling Type 列原始大小写（`MatMulV2.csv`）
7. CSV 格式：自定义列（m,k,n）移除 → Profiling 原始格式（Input Shapes, Input Data Types, Input Formats 等）
8. `metadata.yaml` 合并入 `op_mapping.yaml`
9. 通信 CSV：独立格式，含 `topology_tier` 整数列
10. FusedAttention CSV：特殊 microbenchmark 格式（batch_size, avg_seq_len, num_heads, head_dim, dtype）

**op_mapping.yaml 设计变更**：
11. 纯名字映射，不含 per-op 维度提取逻辑
12. 新增 `interpolation_policy`（按类别配置精确匹配/插值维度）
13. 新增 `communication_data_ref`（指向 HCCL 数据目录的相对路径）
14. 新增 `communication_fallback: analytic`
15. `torch_npu_reference` 结构化为 `apis` 列表 + `microbench_api`
16. 通信拓扑描述移至 `hccl/{cann_version}/comm_config.yaml`

**查询逻辑变更**：
17. FRACTAL_NZ：通用 `fractal_nz_to_nd()` 恢复函数，无需转置（268 行验证零例外）
18. 匹配策略：匹配所有 input shape + dtype，output shape 作为验证
19. 通信查询：`rank_group → CommGrid._get_topology_idx_for_group() → topology_tier`
20. FusedAttention：`query_mode: attention_special`，使用 `args[6]`（seq_lens）
21. Composite 查询：`composite: true` 时主动分解为多个子内核查询并求和（MLA 分解函数复用 analytic model shape 推导），不再 fallback to analytic

**存储结构变更**：
22. 计算数据：`data/{device}/vllm_ascend/{version}/`
23. 通信数据：`data/{device}/hccl/{cann_version}/`（跨 vLLM 版本复用）
24. 工具：`tools/perf_data_collection/`

**内容变更**：
25. 2.1 节：修正"内核启动开销"为"tiling/利用率"，删除无来源的"实测44-95%"数据
26. 数据库构建：两级策略 → 三步走（Profiling → microbenchmark 网格 → 验证）
27. 通信 microbenchmark：Python 脚本统一计算+通信，HCCL Test 作为交叉验证
28. 9.1 节：更新融合 Gap 已关闭/仍开放状态（基于 gitcode/develop 分析）
29. 遗留问题 1：中期方案（融合 Pass 对齐）为正式方案，长期方案（FX graph + 侵入式 DispatchMode）移至 Future Work
30. 遗留问题 2：FusedAttention microbenchmark 为当前方案，两条长期路径在 Future Work

**新增示例文件**：
31. `docs/perf_database/examples/op_mapping_example.yaml`：完整 op_mapping.yaml 示例（~25 条算子映射 + torch_npu_reference）
32. `docs/perf_database/examples/comm_config_example.yaml`：完整 comm_config.yaml 示例（拓扑描述 + 通信算子映射）

**新增附录**：
33. 附录 A：小 Batch 场景 Roofline 偏差分析
34. 附录 B：FRACTAL_NZ 布局分析与验证数据
35. 附录 C：FusedAttention KV Cache 维度分析
36. 附录 D：vLLM 自定义算子覆盖分析
37. 附录 E：op-plugin Type → torch_npu 映射分析
38. 附录 F：AIConfigurator 通信方案对比
39. 附录 G：AI 辅助开发实践建议
40. 附录 H：融合算子语义一致性详细分析

**开发计划**：
41. 两团队结构保持，目标更新为 3.23
42. 三阶段：Phase 1 计算算子穿刺（→3.14）→ Phase 2 通信算子接入+插值（→3.20）→ Phase 3 集成验证（→3.23）

### v1.0 → v1.1

1. 从单一 `ProfilingPerformanceModel`（内含 Roofline 兜底）重构为三个独立 PerformanceModel 并列，Roofline 兜底上移至 Model 层 `fallback_model`
2. `PerfDatabase` 从 Profiling 专用包提升为共享数据层，同时为 Profiling（只读）和 Empirical（读写持久化缓存）服务
3. 新增 `OperatorKey` 共享抽象，封装 `OpInvokeInfo → 查询 key` 转换逻辑
4. `PerfDatabase` API 从 `(system, backend, version)` 三元组改为 `data_path` 路径直传，删除 `VersionManager`
5. Schema 粒度从按抽象算子类型分组（~6 类）改为按 kernel_details.csv 硬件内核一一对齐（~17 个）
6. YAML 映射格式从按 Schema 分组改为扁平的 `tensorcast_op_to_schema`（1:1）
7. QueryEngine 从 4 级降级（含 Roofline）精简为 3 级（精确→插值→外推）
8. 存储格式从 Parquet 为主改为 CSV 为主；数据采集扩展为三种可选方案
9. 基于 DeepSeekV3/Qwen3 实测 Profiling 数据修正了多项算子映射
10. 数据采集流水线从 TensorCast 包内移至仓库顶层独立子系统；开发计划从 5 阶段单团队改为 3 阶段双团队分工
