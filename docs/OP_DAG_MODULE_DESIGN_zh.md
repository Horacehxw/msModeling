# 算子 DAG 与 DES 多流仿真：技术设计文档

## 文档信息

| 项目 | 内容 |
|------|------|
| 版本 | 2.0 |
| 状态 | 草案 — 待评审 |
| 分支 | feat/op_dag_des |
| 最后更新 | 2026-02-05 |
| 作者 | — |

---

## 目录

1. [概述](#1-概述)
2. [背景与动机](#2-背景与动机)
3. [系统架构](#3-系统架构)
4. [模块一 — 算子 DAG](#4-模块一--算子-dag)
    - 4.1 [TensorIdentity](#41-tensoridentity)
    - 4.2 [OpGraphTracer](#42-opgraphtracer)
    - 4.3 [OperatorDAG 与 OpNode](#43-operatordag-与-opnode)
    - 4.4 [DAGAnalyzer](#44-daganalyzer)
5. [模块二 — DES 多流仿真](#5-模块二--des-多流仿真)
    - 5.1 [DES 架构概览](#51-des-架构概览)
    - 5.2 [DESConfig](#52-desconfig)
    - 5.3 [OpHandle](#53-ophandle)
    - 5.4 [StreamType 与 StreamAssigner](#54-streamtype-与-streamassigner)
    - 5.5 [DeviceStream](#55-devicestream)
    - 5.6 [CPUDispatcher](#56-cpudispatcher)
    - 5.7 [TraceCollector](#57-tracecollector)
    - 5.8 [DESSimulator](#58-dessimulator)
    - 5.9 [重叠机制与同步](#59-重叠机制与同步)
6. [代码集成分析](#6-代码集成分析)
    - 6.1 [Runtime 集成](#61-runtime-集成)
    - 6.2 [ModelRunner 集成](#62-modelrunner-集成)
    - 6.3 [UserInputConfig 集成](#63-userinputconfig-集成)
    - 6.4 [Chrome Trace 扩展](#64-chrome-trace-扩展)
7. [文件清单](#7-文件清单)
8. [开发计划](#8-开发计划)
9. [测试计划](#9-测试计划)
10. [风险评估](#10-风险评估)
11. [未来扩展](#11-未来扩展)

---

## 1. 概述

本文档定义了 TensorCast 新增的两个紧密耦合模块的技术设计：

1. **算子 DAG** — 在 `TorchDispatchMode` 追踪期间构建算子数据依赖的有向无环图（DAG）。通过版本化张量标识（versioned tensor identity）准确处理原地修改（inplace mutation）、视图别名（view aliasing）和张量 ID 复用问题。

2. **DES 多流仿真** — 基于 salabim 的离散事件仿真，在多条设备流（CPU 调度、计算流、通信流）上回放算子执行。利用 DAG 建模数据依赖关系，生成兼容 chrome://tracing 的多流 trace，展示计算/通信重叠。

两个模块协同实现：

- 模型执行的关键路径分析
- 计算/通信重叠度量化
- 多流 chrome trace 可视化（兼容 MindStudio Insight / chrome://tracing）
- 为后续双 batch 重叠、chunk prefill/decode 交替、多设备流水线仿真奠定基础

---

## 2. 背景与动机

### 2.1 现有局限

现有的 TensorCast Runtime（`tensor_cast/runtime.py`）通过 `__torch_dispatch__` 按顺序捕获算子，并使用单流 roofline 模型计算性能。当前的 `get_trace_events()`（runtime.py:414-488）将事件输出到每个性能模型进程的单一线程（`tid=0`），不支持多流执行或重叠分析。

现有的 `MemoryTracker`（`tensor_cast/performance_model/memory_tracker.py`）已经通过 `_TensorInfo.def_op_idx` 和 `use_op_indices` 进行隐式数据依赖追踪，并通过 `_handle_aliasing()` 进行别名追踪。然而，这些信息仅用于内存生命周期分析，未用于构建显式 DAG 或调度仿真。

### 2.2 目标

| 目标 | 描述 |
|------|------|
| G1 | 构建具有精确数据依赖边的显式算子 DAG |
| G2 | 建模 CPU 内核启动队列行为（提前下发，不等待设备完成） |
| G3 | 在独立设备流上仿真计算/通信重叠 |
| G4 | 生成多流 chrome trace 用于可视化 |
| G5 | 提供关键路径和并行度指标 |
| G6 | 可扩展设计（支持添加更多流、多设备、动态调度） |

### 2.3 关键设计决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| DES 框架 | **salabim**（非 simpy） | ServingCast 已使用（`stime.py`）；成熟的 `Task`、`wait()/notify()` 原语 |
| 仿真方式 | **完整 DES**（非基于 trace 的解析式） | 支持动态调度场景（双 batch 重叠、chunk prefill/decode 交替、流水线并行） |
| CPU 调度模型 | **CPU 提前下发** | 匹配真实 NPU 运行时的内核启动队列行为 — CPU 不阻塞等待设备执行 |
| V1 流分配 | 通信算子 → COMMUNICATION 流，其余 → COMPUTE 流 | 简单，可通过注解扩展；V1 重叠分析足够 |
| V1 通信流 | 单条 COMMUNICATION 流 | 未来可扩展为节点内/节点间、AIV/AIC 等 |

---

## 3. 系统架构

### 3.1 端到端流程

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            Runtime (TorchDispatchMode)                        │
│                                                                              │
│  __torch_dispatch__(func, args, kwargs)                                      │
│       │                                                                      │
│       ├──► OpInvokeInfo ──► op_invoke_infos[]                                │
│       ├──► MemoryTracker.record_op_invocation()                              │
│       └──► OpGraphTracer.record_op()  ◄── 新增                              │
│                                                                              │
│  __exit__()                                                                  │
│       │                                                                      │
│       ├──► repeat_op_invoke_infos()                                          │
│       ├──► replay_op_invoke_infos() ──► event_list[]                         │
│       ├──► MemoryTracker.analyze()                                           │
│       ├──► OpGraphTracer.build_dag() ──► OperatorDAG  ◄── 新增              │
│       └──► DESSimulator.run() ──► DES trace events    ◄── 新增              │
│                                                                              │
│  get_trace_events()                                                          │
│       └──► 现有事件 + DES 多流事件  ◄── 新增                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 模块依赖图

```
tensor_cast/
├── runtime.py                          # 修改：集成 OpGraphTracer + DES
├── performance_model/
│   ├── __init__.py                     # 修改：导出新类
│   ├── memory_tracker.py               # 不变
│   ├── analytic.py                     # 不变
│   ├── op_graph_tracer.py              # 新增：TensorIdentity, OpGraphTracer
│   ├── operator_dag.py                 # 新增：OpNode, OperatorDAG
│   ├── dag_analyzer.py                 # 新增：DAGAnalyzer
│   └── des/                            # 新增：DES 仿真包
│       ├── __init__.py
│       ├── config.py                   #   DESConfig
│       ├── op_handle.py                #   OpHandle
│       ├── stream.py                   #   StreamType, DeviceStream(stime.Task)
│       ├── cpu_dispatcher.py           #   CPUDispatcher(stime.Task)
│       ├── stream_assigner.py          #   StreamAssigner
│       ├── trace_collector.py          #   TraceCollector
│       └── des_simulator.py            #   DESSimulator（顶层编排器）
├── core/
│   ├── model_runner.py                 # 修改：传递 DES 标志，打印 DES 结果
│   └── user_config.py                  # 修改：新增 enable_des_simulation 字段
└── tests/
    ├── test_op_graph_tracer.py         # 新增
    └── test_des_simulator.py           # 新增
```

### 3.3 与现有组件的关系

```
                    ┌─────────────────┐
                    │  UserInputConfig │
                    │ (user_config.py) │
                    └───────┬─────────┘
                            │ enable_des_simulation
                            ▼
                    ┌─────────────────┐
                    │   ModelRunner   │
                    │(model_runner.py)│
                    └───────┬─────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────┐
│                     Runtime                            │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │OpInvoke  │  │  Memory      │  │  OpGraph     │    │
│  │Info      │  │  Tracker     │  │  Tracer      │    │
│  │(现有)    │  │  (现有)      │  │  (新增)      │    │
│  └─────┬────┘  └──────────────┘  └──────┬───────┘    │
│        │                                 │            │
│        │        ┌────────────────┐       │            │
│        └───────►│ PerformanceModel│       │            │
│                 │ (现有)         │       │            │
│                 └───────┬────────┘       │            │
│                         │                │            │
│                    event_list      OperatorDAG        │
│                         │                │            │
│                         ▼                ▼            │
│                 ┌────────────────────────────┐        │
│                 │      DESSimulator          │        │
│                 │      (新增)                │        │
│                 └────────────────────────────┘        │
└───────────────────────────────────────────────────────┘
```

---

## 4. 模块一 — 算子 DAG

### 4.1 TensorIdentity

#### 4.1.1 问题陈述

构建 DAG 需要通过 `id()` 匹配张量的生产者和消费者。这里存在两个问题：

**问题一 — 张量 ID 复用**：Python 的 `id()` 返回对象的内存地址，仅在对象生命周期内唯一。当中间张量被垃圾回收后，新张量可能复用其地址，导致 DAG 中产生错误的依赖边。

**问题二 — 原地操作**：原地操作（如 `add_()`）返回同一个 Python 对象。基于 id 的朴素方法无法区分张量修改前后的状态。

#### 4.1.2 解决方案

| 问题 | 解决方案 |
|------|----------|
| ID 复用 | **强引用池** — tracer 将所有张量保存在列表中，阻止垃圾回收。由于 TensorCast 使用 meta 张量（无存储），内存开销可忽略（每个张量对象约 128 字节）。 |
| 原地操作 | **版本化标识** `(uid, version)` — 每个张量分配单调递增的 `uid`。原地修改时 `version` 递增生成新标识。输入侧读取 `(uid, v)`，输出侧生成 `(uid, v+1)`。 |

#### 4.1.3 数据结构

```python
# 文件：tensor_cast/performance_model/op_graph_tracer.py

@dataclass(frozen=True, slots=True)
class TensorIdentity:
    """
    版本化张量标识，用于精确的 DAG 边追踪。

    (uid, version) 元组唯一标识一个张量状态。
    - uid：追踪期间单调分配
    - version：每次原地修改时递增
    """
    uid: int
    version: int = 0

    def __repr__(self) -> str:
        return f"T{self.uid}v{self.version}"
```

#### 4.1.4 别名组追踪

当张量 `A` 存在视图 `B`（通过 `view()`、切片、`transpose()` 等创建）时，两者共享底层存储。如果 `A` 被原地修改，`B` 的数据也会受到影响。tracer 维护**别名组**（alias group）— 共享存储的 UID 集合 — 并将版本递增传播到组内所有成员。

```python
# 别名组：uid -> 别名 uid 集合
_alias_groups: Dict[int, Set[int]] = {}

def _increment_version(self, uid: int) -> int:
    """递增 uid 及其所有别名的版本。"""
    new_version = self._version_map[uid] + 1
    for alias_uid in self._alias_groups.get(uid, {uid}):
        self._version_map[alias_uid] = new_version
    return new_version
```

视图操作的检测复用了 `MemoryTracker._handle_aliasing()`（memory_tracker.py:84-146）中相同的 schema 检查逻辑，通过比对 `func._schema.returns[i].alias_info.before_set` 与参数别名集合来判断。

### 4.2 OpGraphTracer

#### 4.2.1 职责

| 职责 | 机制 |
|------|------|
| 分配稳定的张量标识 | `_id_to_uid` 映射 + 单调计数器 |
| 防止追踪期间张量被 GC | `_tensor_pool: List[torch.Tensor]` |
| 检测原地修改 | 基于 Schema：`arg.alias_info.is_write`；回退：`id(input) == id(output)` |
| 检测视图别名 | 基于 Schema：`return.alias_info.before_set & arg.alias_info.before_set` |
| 追踪 kwargs 修改 | Schema 的 `mutates_args`（如 `reshape_and_cache` 修改 `kv_cache`） |
| 记录算子张量流 | 每个算子存储 `(OpInvokeInfo, input_identities, output_identities)` |
| 构建 DAG | 两遍扫描：创建节点，然后通过标识匹配连接边 |

#### 4.2.2 API

```python
class OpGraphTracer:
    def __init__(self, track_aliases: bool = True): ...

    def record_op(self, op_invoke_info: OpInvokeInfo):
        """每个算子在 __torch_dispatch__ 中调用。记录张量流。"""

    def build_dag(self) -> OperatorDAG:
        """追踪结束后调用。从记录的张量流构建 DAG。"""

    def clear(self):
        """DAG 构建完成后释放张量引用。"""

    @property
    def dag(self) -> Optional[OperatorDAG]: ...

    def get_statistics(self) -> Dict[str, Any]: ...
```

#### 4.2.3 `record_op` 算法

```
1. 从 args + kwargs 中提取输入张量
2. 对每个输入张量：
   a. 若为新张量则分配 UID（加入池中）
   b. 读取当前标识 (uid, version)
   c. 加入 input_identities 集合
3. 通过 schema 检测修改（alias_info.is_write）
4. 通过 schema 检测视图别名（alias_info.before_set 匹配）
5. 若检测到视图：register_alias(source, view)
6. 提取输出张量
7. 对每个输出张量：
   a. 若 id(output) 在 input_ids 中 → 原地操作：递增版本
   b. 否则 → 新张量：分配全新标识
8. 处理 kwargs 修改（如 kv_cache）：
   a. 遍历 schema 中标记为修改的参数索引
   b. 递增版本，加入 output_identities
9. 存储 (op_invoke_info, input_identities, output_identities)
```

#### 4.2.4 `build_dag` 算法

```
第一遍 — 创建节点：
  对每个记录的算子：
    创建带有 input/output identities 的 OpNode
    在 identity_producers 映射中记录每个输出标识的生产者 op_idx

第二遍 — 连接边：
  对每个节点的输入标识：
    在 identity_producers 中查找生产者
    若找到：添加边（生产者 → 当前节点）
    若未找到：标记为模型输入

第三遍 — 识别模型输出：
  对每个被生产的标识：
    若未被任何节点消费：标记为模型输出
```

### 4.3 OperatorDAG 与 OpNode

#### 4.3.1 OpNode

```python
# 文件：tensor_cast/performance_model/operator_dag.py

@dataclass
class OpNode:
    op_idx: int
    op_invoke_info: OpInvokeInfo
    input_identities: Set[TensorIdentity]
    output_identities: Set[TensorIdentity]
    predecessors: Set[int]    # 当前节点依赖的算子索引
    successors: Set[int]      # 依赖当前节点的算子索引

    @property
    def op_name(self) -> str: ...

    @property
    def in_degree(self) -> int: ...

    @property
    def out_degree(self) -> int: ...
```

#### 4.3.2 OperatorDAG

```python
@dataclass
class OperatorDAG:
    nodes: Dict[int, OpNode]
    model_inputs: Set[TensorIdentity]
    model_outputs: Set[TensorIdentity]

    # 反向索引：标识 -> 生产者 op_idx
    _identity_producers: Dict[TensorIdentity, int]

    @property
    def num_nodes(self) -> int: ...

    @property
    def num_edges(self) -> int: ...

    def get_roots(self) -> List[int]:
        """无前驱节点（仅消费模型输入的节点）。"""

    def get_leaves(self) -> List[int]:
        """无后继节点（仅生产模型输出的节点）。"""

    def topological_order(self) -> Iterator[int]:
        """基于 BFS 的 Kahn 算法。依赖在前，被依赖在后。"""

    def reverse_topological_order(self) -> Iterator[int]: ...

    def to_graphviz(self, show_tensors: bool = False) -> str:
        """导出为 DOT 格式，用于 Graphviz 可视化。"""

    def to_json(self) -> str:
        """导出为 JSON 格式，用于程序化分析。"""
```

### 4.4 DAGAnalyzer

```python
# 文件：tensor_cast/performance_model/dag_analyzer.py

@dataclass
class CriticalPathResult:
    path: List[int]                          # 关键路径上的算子索引序列
    total_time_s: float                      # 路径上执行时间总和
    bottleneck_ops: List[Tuple[int, float]]  # Top-5 (op_idx, time_s)

class DAGAnalyzer:
    def __init__(self, dag: OperatorDAG): ...

    def compute_critical_path(
        self, get_op_time: Optional[Callable[[int], float]] = None
    ) -> CriticalPathResult:
        """
        拓扑序上的最长路径动态规划。
        get_op_time 未提供时默认使用单位代价。
        """

    def compute_parallelism_profile(self) -> List[int]:
        """
        为每个节点分配层级（最大前驱层级 + 1）。
        返回列表，索引 i = 第 i 层的算子数量。
        """

    def get_summary(self) -> Dict:
        """
        返回：num_nodes, num_edges, num_roots, num_leaves,
              dag_depth, max_parallelism, avg_parallelism
        """
```

---

## 5. 模块二 — DES 多流仿真

### 5.1 DES 架构概览

DES 模块在具有多条硬件流的设备上仿真算子执行。它基于现有的 `stime.py` 封装的 salabim 框架，该框架提供 `Task`（扩展自 `salabim.Component`）、`wait()/notify()` 原语和 `elapse()` 逻辑时间推进。

```
┌──────────────────────────────────────────────────────────────────┐
│                        DESSimulator                               │
│  （创建 salabim 环境，设置流，运行仿真）                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐                                             │
│  │  CPUDispatcher   │  stime.Task：按拓扑序遍历 DAG，             │
│  │  (stime.Task)    │  每个算子花费 dispatch_latency，             │
│  │                  │  将 OpHandle 入队到目标 DeviceStream。        │
│  │                  │  不等待设备完成。                             │
│  └───────┬──────────┘                                             │
│          │ enqueue(op_handle)                                     │
│          ▼                                                        │
│  ┌──────────────────┐    ┌──────────────────┐                     │
│  │  DeviceStream    │    │  DeviceStream    │                     │
│  │  "计算流"        │    │  "通信流"        │                     │
│  │  (stime.Task)    │    │  (stime.Task)    │                     │
│  │                  │    │                  │                     │
│  │  FIFO 队列：     │    │  FIFO 队列：     │                     │
│  │  出队算子        │    │  出队算子        │                     │
│  │  等待依赖满足    │    │  等待依赖满足    │                     │
│  │  elapse(时间)    │    │  elapse(时间)    │                     │
│  │  通知后继        │    │  通知后继        │                     │
│  └──────────────────┘    └──────────────────┘                     │
│                                                                   │
│  ┌──────────────────┐                                             │
│  │  TraceCollector  │  记录 (算子名, 流, 开始, 结束)             │
│  │                  │  用于 chrome trace 导出。                    │
│  └──────────────────┘                                             │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 DESConfig

```python
# 文件：tensor_cast/performance_model/des/config.py

@dataclass
class DESConfig:
    """DES 仿真配置。"""

    # 每个算子的固定 CPU 调度延迟（秒）。
    # 建模 CPU 准备和启动内核的时间。
    cpu_dispatch_latency_s: float = 10e-6  # 默认 10 μs

    # 算子级别的流分配覆盖。
    # 键：算子名字符串（如 "tensor_cast::all_reduce"）
    # 值：StreamType 枚举值
    stream_overrides: Dict[str, StreamType] = field(default_factory=dict)

    # 是否收集 trace 事件用于 chrome trace 导出
    enable_trace_collection: bool = True
```

### 5.3 OpHandle

`OpHandle` 是单个算子在仿真时的表示。它持有 DES 执行期间所需的所有元数据：DAG 节点引用、性能结果、计时记录、依赖追踪状态和分配的流。

```python
# 文件：tensor_cast/performance_model/des/op_handle.py

@dataclass
class OpHandle:
    """DES 仿真中的算子表示。"""

    # 标识
    op_idx: int
    op_name: str
    dag_node: OpNode

    # PerformanceModel 给出的执行时间（秒）
    device_execution_time_s: float

    # 流分配（仿真前由 StreamAssigner 设置）
    target_stream_type: StreamType
    assigned_stream: Optional['DeviceStream'] = None

    # 计时记录（仿真期间填充）
    cpu_start_s: float = 0.0
    cpu_end_s: float = 0.0
    device_start_s: float = 0.0
    device_end_s: float = 0.0

    # DAG 依赖追踪
    successor_handles: List['OpHandle'] = field(default_factory=list)
    unsatisfied_dep_count: int = 0  # 随前驱完成而递减

    @property
    def deps_satisfied(self) -> bool:
        return self.unsatisfied_dep_count == 0

    def on_dep_satisfied(self, predecessor: 'OpHandle'):
        """前驱完成时调用。"""
        self.unsatisfied_dep_count -= 1

    def mark_completed(self):
        """通知所有后继本算子已完成。"""
        for succ in self.successor_handles:
            succ.on_dep_satisfied(self)
```

**生命周期**：

```
创建（由 DESSimulator） → CPU 调度（由 CPUDispatcher） → 入队 DeviceStream
→ 检查依赖 → 执行（elapse） → 完成 → 通知后继
```

### 5.4 StreamType 与 StreamAssigner

```python
# 文件：tensor_cast/performance_model/des/stream.py

class StreamType(Enum):
    CPU = "cpu"
    COMPUTE = "compute"
    COMMUNICATION = "communication"
```

```python
# 文件：tensor_cast/performance_model/des/stream_assigner.py

class StreamAssigner:
    """将算子映射到目标设备流。"""

    # 默认 V1 映射：通信算子发送到 COMMUNICATION 流，
    # 其余发送到 COMPUTE 流。
    COMM_OPS = {
        "tensor_cast::all_reduce",
        "tensor_cast::all_gather",
        "tensor_cast::reduce_scatter",
        "tensor_cast::all_to_all",
    }

    def __init__(self, config: DESConfig):
        self._overrides = config.stream_overrides

    def assign(self, op_invoke_info: OpInvokeInfo) -> StreamType:
        op_name = str(op_invoke_info.func)

        # 优先检查显式覆盖
        if op_name in self._overrides:
            return self._overrides[op_name]

        # 默认：通信算子 -> COMMUNICATION，其余 -> COMPUTE
        if op_name in self.COMM_OPS:
            return StreamType.COMMUNICATION

        return StreamType.COMPUTE
```

该设计具有可扩展性：未来版本可以添加 `StreamType.AIV`、`StreamType.AIC`、`StreamType.INTRA_NODE_COMM` 等，并通过 `DESConfig.stream_overrides` 或注解覆盖每个算子的分配。

### 5.5 DeviceStream

`DeviceStream` 扩展自 `stime.Task`，建模硬件执行流。运行 FIFO 循环：出队一个算子，等待数据依赖满足，执行（推进逻辑时间），通知后继。

```python
# 文件：tensor_cast/performance_model/des/stream.py

class DeviceStream(stime.Task):
    """
    建模为 salabim Task 的设备执行流。

    运行 FIFO 循环：
    1. 从队列中出队下一个算子
    2. 等待所有前驱算子完成（deps_satisfied）
    3. 执行：记录开始时间，elapse device_execution_time_s，记录结束时间
    4. 标记完成，通知后继算子
    5. 若后继的依赖全部满足，唤醒其所在的流
    """

    def __init__(self, stream_type: StreamType, trace_collector: 'TraceCollector'):
        super().__init__(name=stream_type.value)
        self._stream_type = stream_type
        self._trace_collector = trace_collector
        self._queue: Deque[OpHandle] = deque()
        self._shutdown = False

    def enqueue(self, op_handle: OpHandle):
        """由 CPUDispatcher 调用，将算子添加到此流的 FIFO。"""
        self._queue.append(op_handle)
        self.notify()  # 若 process() 正在等待工作则唤醒

    def shutdown(self):
        """通知此流不会再有新算子入队。"""
        self._shutdown = True
        self.notify()

    def process(self):
        """salabim Task 主循环。"""
        while True:
            # 等待工作
            while not self._queue and not self._shutdown:
                self.wait()
            if not self._queue and self._shutdown:
                break

            op = self._queue.popleft()

            # 等待所有数据依赖
            while not op.deps_satisfied:
                self.wait()

            # 执行
            op.device_start_s = stime.now()
            stime.elapse(op.device_execution_time_s)
            op.device_end_s = stime.now()

            # 记录 trace 事件
            self._trace_collector.record_device_event(op, self._stream_type)

            # 通知后继
            op.mark_completed()
            for succ in op.successor_handles:
                if succ.deps_satisfied and succ.assigned_stream is not None:
                    succ.assigned_stream.notify()
```

**关键行为**：`while not op.deps_satisfied: self.wait()` 循环使流进入被动状态（将控制权交给 salabim 调度器）。当前驱完成并调用 `succ.assigned_stream.notify()` 时，流被唤醒。

### 5.6 CPUDispatcher

`CPUDispatcher` 扩展自 `stime.Task`，建模 CPU 的内核启动行为。按拓扑序遍历 DAG，每个算子花费 `cpu_dispatch_latency_s`，并将算子入队到目标设备流。关键是 **CPU 不等待设备执行** — 提前下发，建模真实 NPU 运行时的内核启动队列。

```python
# 文件：tensor_cast/performance_model/des/cpu_dispatcher.py

class CPUDispatcher(stime.Task):
    """
    CPU 调度任务。按 DAG 拓扑序遍历，将算子下发到设备流。

    建模真实 NPU 运行时行为：CPU 将多个内核入队到设备流，
    不阻塞等待设备完成。
    """

    def __init__(
        self,
        op_handles: Dict[int, OpHandle],
        topo_order: List[int],
        device_streams: Dict[StreamType, DeviceStream],
        trace_collector: 'TraceCollector',
        config: DESConfig,
    ):
        super().__init__(name="cpu_dispatcher")
        self._op_handles = op_handles
        self._topo_order = topo_order
        self._device_streams = device_streams
        self._trace_collector = trace_collector
        self._config = config

    def process(self):
        for op_idx in self._topo_order:
            op = self._op_handles[op_idx]

            # CPU 调度延迟
            op.cpu_start_s = stime.now()
            stime.elapse(self._config.cpu_dispatch_latency_s)
            op.cpu_end_s = stime.now()

            # 记录 CPU trace 事件
            self._trace_collector.record_cpu_event(op)

            # 入队到目标设备流（非阻塞）
            target_stream = self._device_streams[op.target_stream_type]
            op.assigned_stream = target_stream
            target_stream.enqueue(op)

        # 通知所有流调度已完成
        for stream in self._device_streams.values():
            stream.shutdown()
```

### 5.7 TraceCollector

`TraceCollector` 在 DES 仿真期间记录计时事件，并将其转换为 Chrome Trace Event Format 用于可视化。

```python
# 文件：tensor_cast/performance_model/des/trace_collector.py

@dataclass
class TraceEvent:
    op_name: str
    stream_type: StreamType
    start_s: float
    end_s: float
    op_idx: int
    extra_args: Dict[str, Any] = field(default_factory=dict)

class TraceCollector:
    """收集 DES 事件并生成 chrome trace JSON。"""

    # chrome trace 的线程 ID 映射
    STREAM_TID = {
        StreamType.CPU: 0,
        StreamType.COMPUTE: 1,
        StreamType.COMMUNICATION: 2,
    }

    STREAM_NAMES = {
        StreamType.CPU: "CPU Dispatch",
        StreamType.COMPUTE: "Compute Stream",
        StreamType.COMMUNICATION: "Communication Stream",
    }

    def __init__(self):
        self.events: List[TraceEvent] = []

    def record_cpu_event(self, op: OpHandle):
        self.events.append(TraceEvent(
            op_name=op.op_name,
            stream_type=StreamType.CPU,
            start_s=op.cpu_start_s,
            end_s=op.cpu_end_s,
            op_idx=op.op_idx,
        ))

    def record_device_event(self, op: OpHandle, stream_type: StreamType):
        self.events.append(TraceEvent(
            op_name=op.op_name,
            stream_type=stream_type,
            start_s=op.device_start_s,
            end_s=op.device_end_s,
            op_idx=op.op_idx,
        ))

    def to_chrome_trace_events(self, pid: int = 0) -> List[Dict]:
        """
        转换为 Chrome Trace Event Format。

        Args:
            pid: 进程 ID 偏移，用于与现有 Runtime trace 集成。

        Returns:
            Chrome Trace JSON 事件字典列表。
        """
        trace_events = []

        # 元数据：命名进程和线程
        trace_events.append({
            "name": "process_name", "ph": "M", "pid": pid,
            "args": {"name": "DES Multi-Stream Simulation"},
        })
        for stream_type, tid in self.STREAM_TID.items():
            trace_events.append({
                "name": "thread_name", "ph": "M",
                "pid": pid, "tid": tid,
                "args": {"name": self.STREAM_NAMES[stream_type]},
            })

        # 完整事件（"X" 类型）
        for event in self.events:
            tid = self.STREAM_TID[event.stream_type]
            trace_events.append({
                "name": event.op_name,
                "cat": event.stream_type.value,
                "ph": "X",
                "ts": event.start_s * 1e6,   # 秒 → 微秒
                "dur": (event.end_s - event.start_s) * 1e6,
                "pid": pid,
                "tid": tid,
                "args": {"op_idx": event.op_idx, **event.extra_args},
            })

        return trace_events
```

**Chrome trace 线程布局**：

| TID | 流 | 描述 |
|-----|------|------|
| 0 | CPU Dispatch | 内核启动队列 — 每个算子调度一个事件 |
| 1 | Compute Stream | 矩阵运算、注意力、逐元素运算等 |
| 2 | Communication Stream | all_reduce、all_gather、all_to_all 等 |

可直接在 chrome://tracing 或 MindStudio Insight 中查看。

### 5.8 DESSimulator

`DESSimulator` 是顶层编排器。接收 DAG、事件列表（含性能结果）和配置，运行完整的 salabim 仿真。

```python
# 文件：tensor_cast/performance_model/des/des_simulator.py

@dataclass
class DESResult:
    """DES 仿真结果。"""
    total_time_s: float
    compute_time_s: float         # 计算流上的总时间
    communication_time_s: float   # 通信流上的总时间
    overlap_time_s: float         # 两条流同时活跃的时间
    trace_collector: TraceCollector

class DESSimulator:
    """
    多流算子执行的顶层 DES 仿真器。

    用法：
        simulator = DESSimulator(dag, event_list, config)
        result = simulator.run()
        chrome_events = result.trace_collector.to_chrome_trace_events()
    """

    def __init__(
        self,
        dag: OperatorDAG,
        event_list: List[RuntimeEvent],
        config: DESConfig = DESConfig(),
        perf_model_name: Optional[str] = None,
    ):
        self._dag = dag
        self._event_list = event_list
        self._config = config
        self._perf_model_name = perf_model_name

    def run(self) -> DESResult:
        # 1. 初始化 salabim 环境
        stime.init_simulation()

        # 2. 从 DAG + event_list 构建 OpHandle
        stream_assigner = StreamAssigner(self._config)
        op_handles = self._build_op_handles(stream_assigner)

        # 3. 根据 DAG 边连接后继关系
        self._wire_dependencies(op_handles)

        # 4. 创建 TraceCollector
        trace_collector = TraceCollector()

        # 5. 创建 DeviceStream
        device_streams = {
            StreamType.COMPUTE: DeviceStream(StreamType.COMPUTE, trace_collector),
            StreamType.COMMUNICATION: DeviceStream(StreamType.COMMUNICATION, trace_collector),
        }

        # 6. 创建 CPUDispatcher
        topo_order = list(self._dag.topological_order())
        CPUDispatcher(op_handles, topo_order, device_streams, trace_collector, self._config)

        # 7. 运行仿真
        stime.start_simulation()

        # 8. 收集结果
        result = self._collect_results(trace_collector)

        # 9. 清理 salabim 环境
        stime.SimulationEnv.clear()

        return result
```

### 5.9 重叠机制与同步

#### 5.9.1 CPU 提前下发模型

CPU 不等待设备执行完成就下发下一个算子。这建模了真实 NPU 运行时中 CPU 将多个内核入队到设备命令队列的行为：

```
CPU:      [下发 A][下发 B][下发 C][下发 D]...
计算流:           [====== A ======]       [=== D ===]
通信流:                  [=== B ===][=== C ===]
                         ↑  重叠  ↑
```

在此示例中，算子 B 和 C（通信）与算子 A（计算）无数据依赖，因此通信流在计算流仍在执行 A 时就开始执行 B。

#### 5.9.2 依赖同步

数据依赖通过 `OpHandle.unsatisfied_dep_count` 机制执行：

1. `DESSimulator._wire_dependencies()` 被调用时，每个算子的 `unsatisfied_dep_count` 设置为其 DAG 前驱数量。
2. 前驱完成时（`mark_completed()`），通过 `on_dep_satisfied()` 递减每个后继的 `unsatisfied_dep_count`。
3. 当 `unsatisfied_dep_count` 降为 0 时，`deps_satisfied` 变为 `True`。
4. 若后继的 `DeviceStream` 处于被动状态（等待中），通过 `notify()` 将其唤醒。

这确保了：
- 不同流上的无依赖算子并发执行（重叠）。
- 不同流上的有依赖算子遵守数据顺序（如 `all_reduce` 必须等待生产其输入的 `mm`）。
- 同一流上的算子按 FIFO 顺序执行（流内串行化）。

#### 5.9.3 salabim 执行模型

salabim 使用协作式多任务：

| 原语 | salabim API | stime 封装 | 行为 |
|------|-------------|------------|------|
| 创建任务 | `sim.Component()` | `stime.Task()` | 在环境中注册组件 |
| 推进时间 | `component.hold(t)` | `stime.elapse(t)` | 让出控制权，经过 `t` 逻辑时间后恢复 |
| 被动化 | `component.passivate()` | `task.wait()` | 让出控制权，等待显式激活 |
| 激活 | `component.activate()` | `task.notify()` | 唤醒被动任务（若已激活则无操作） |
| 运行 | `env.run()` | `stime.start_simulation()` | 运行直到所有任务终止 |

仿真在所有三个任务（CPUDispatcher、Compute DeviceStream、Communication DeviceStream）完成其 `process()` 方法时终止。

---

## 6. 代码集成分析

### 6.1 Runtime 集成

**文件**：`tensor_cast/runtime.py`

**`__init__` 的修改**（第 41-61 行）：

```python
def __init__(
    self,
    perf_models: Union[PerformanceModel, List[PerformanceModel]],
    device_profile: DeviceProfile,
    memory_tracker: Optional[MemoryTracker] = None,
    enable_dag_tracing: bool = False,          # 新增
    enable_des_simulation: bool = False,       # 新增
    des_config: Optional[DESConfig] = None,    # 新增
):
    # ... 现有初始化 ...

    # 新增：DAG 追踪
    self._enable_dag_tracing = enable_dag_tracing or enable_des_simulation
    self._op_graph_tracer: Optional[OpGraphTracer] = None
    if self._enable_dag_tracing:
        self._op_graph_tracer = OpGraphTracer(track_aliases=True)

    # 新增：DES 仿真
    self._enable_des_simulation = enable_des_simulation
    self._des_config = des_config or DESConfig()
    self._des_result: Optional[DESResult] = None
```

注：`enable_des_simulation=True` 隐含 `enable_dag_tracing=True`，因为 DES 需要 DAG。

**`__torch_dispatch__` 的修改**（第 69-77 行）：

```python
def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    kwargs = {} if kwargs is None else kwargs
    if not torch.compiler.is_compiling():
        out = func(*args, **kwargs)
        op_invoke_info = OpInvokeInfo(func, args, kwargs, out)
        self.op_invoke_infos.append(op_invoke_info)

        # 新增：记录用于 DAG 追踪
        if self._op_graph_tracer is not None:
            self._op_graph_tracer.record_op(op_invoke_info)

        return out
    else:
        return func(*args, **kwargs)
```

**`__exit__` 的修改**（约第 150-170 行）：

```python
def __exit__(self, exc_type, exc_val, exc_tb):
    super().__exit__(exc_type, exc_val, exc_tb)
    self.repeat_op_invoke_infos()
    self.replay_op_invoke_infos()
    if self.memory_tracker:
        self.memory_tracker.analyze()

    # 新增：构建 DAG
    if self._op_graph_tracer is not None:
        self._op_graph_tracer.build_dag()

    # 新增：运行 DES 仿真
    if self._enable_des_simulation and self._op_graph_tracer is not None:
        dag = self._op_graph_tracer.dag
        if dag is not None:
            simulator = DESSimulator(
                dag=dag,
                event_list=self.event_list,
                config=self._des_config,
                perf_model_name=self.perf_models[0].name,
            )
            self._des_result = simulator.run()

    # 释放张量引用（保留 DAG 结构）
    if self._op_graph_tracer is not None:
        self._op_graph_tracer.clear()

    _current_runtime.value = None
```

**`get_trace_events` 的修改**（第 414-488 行）：

```python
def get_trace_events(self):
    trace_events = []
    # ... 现有 trace 事件生成 ...

    # 新增：附加 DES 多流 trace 事件
    if self._des_result is not None:
        # 使用 PID 偏移将 DES 事件与性能模型事件分开
        des_pid = len(self.perf_models)
        des_events = self._des_result.trace_collector.to_chrome_trace_events(pid=des_pid)
        trace_events.extend(des_events)

    return trace_events
```

**新增属性**：

```python
@property
def operator_dag(self) -> Optional[OperatorDAG]:
    if self._op_graph_tracer is not None:
        return self._op_graph_tracer.dag
    return None

@property
def des_result(self) -> Optional[DESResult]:
    return self._des_result
```

### 6.2 ModelRunner 集成

**文件**：`tensor_cast/core/model_runner.py`（第 95-100 行）

Runtime 构造函数调用需要新增标志：

```python
with (
    Runtime(
        self.perf_model,
        self.device_profile,
        memory_tracker=MemoryTracker(self.device_profile),
        enable_des_simulation=self.user_input.enable_des_simulation,  # 新增
    ) as runtime,
    torch.no_grad(),
):
    logits = self.model.forward(**input_kwargs)
```

在 `with` 块之后，打印 DES 结果：

```python
# 新增：打印 DES 重叠结果
if runtime.des_result is not None:
    des = runtime.des_result
    print(f"\n--- DES 多流仿真 ---")
    print(f"总仿真时间:    {des.total_time_s * 1e3:.3f} ms")
    print(f"计算时间:      {des.compute_time_s * 1e3:.3f} ms")
    print(f"通信时间:      {des.communication_time_s * 1e3:.3f} ms")
    print(f"重叠时间:      {des.overlap_time_s * 1e3:.3f} ms")
    if des.communication_time_s > 0:
        overlap_pct = des.overlap_time_s / des.communication_time_s * 100
        print(f"重叠比例:      {overlap_pct:.1f}% 的通信被重叠")
```

现有的 `export_chrome_trace` 调用（model_runner.py:164-165）无需修改，因为 `get_trace_events()` 在启用 DES 时已自动包含 DES 事件。

### 6.3 UserInputConfig 集成

**文件**：`tensor_cast/core/user_config.py`

新增字段：

```python
@dataclass
class UserInputConfig:
    # ... 现有字段 ...
    enable_des_simulation: bool = False    # 新增
```

CLI 参数（在相关的 argparse 设置中）：

```python
parser.add_argument(
    "--enable-des-simulation",
    action="store_true",
    default=False,
    help="启用 DES 多流仿真，用于计算/通信重叠分析",
)
```

### 6.4 Chrome Trace 扩展

当前 chrome trace 输出结构（runtime.py:414-488）使用：
- 每个性能模型一个 PID
- TID=0（每个模型单流）

启用 DES 后，trace 文件将额外包含：
- PID = `len(perf_models)` 用于 DES 事件
- TID=0（CPU Dispatch）、TID=1（计算流）、TID=2（通信流）

**chrome://tracing 中的示例 trace 布局**：

```
PID 0: "AnalyticPerformanceModel"（现有单流）
  TID 0: Stream 0
    [op1][op2][op3]...

PID 1: "DES Multi-Stream Simulation"（新增）
  TID 0: CPU Dispatch
    [下发1][下发2][下发3]...
  TID 1: Compute Stream
    [======op1======][===op3===]...
  TID 2: Communication Stream
             [===op2===]...
```

---

## 7. 文件清单

### 7.1 新增文件

| 文件 | 模块 | 关键类/函数 |
|------|------|-------------|
| `tensor_cast/performance_model/op_graph_tracer.py` | DAG | `TensorIdentity`、`OpGraphTracer` |
| `tensor_cast/performance_model/operator_dag.py` | DAG | `OpNode`、`OperatorDAG` |
| `tensor_cast/performance_model/dag_analyzer.py` | DAG | `DAGAnalyzer`、`CriticalPathResult` |
| `tensor_cast/performance_model/des/__init__.py` | DES | 包导出 |
| `tensor_cast/performance_model/des/config.py` | DES | `DESConfig` |
| `tensor_cast/performance_model/des/op_handle.py` | DES | `OpHandle` |
| `tensor_cast/performance_model/des/stream.py` | DES | `StreamType`、`DeviceStream` |
| `tensor_cast/performance_model/des/cpu_dispatcher.py` | DES | `CPUDispatcher` |
| `tensor_cast/performance_model/des/stream_assigner.py` | DES | `StreamAssigner` |
| `tensor_cast/performance_model/des/trace_collector.py` | DES | `TraceCollector`、`TraceEvent` |
| `tensor_cast/performance_model/des/des_simulator.py` | DES | `DESSimulator`、`DESResult` |
| `tensor_cast/tests/test_op_graph_tracer.py` | 测试 | DAG 模块单元/集成测试 |
| `tensor_cast/tests/test_des_simulator.py` | 测试 | DES 模块单元/集成测试 |

### 7.2 修改文件

| 文件 | 修改摘要 |
|------|----------|
| `tensor_cast/runtime.py` | 新增 `enable_dag_tracing`、`enable_des_simulation`、`des_config` 参数；在 `__torch_dispatch__` 中钩入 `record_op`；在 `__exit__` 中构建 DAG + 运行 DES；扩展 `get_trace_events()`；新增 `operator_dag` 和 `des_result` 属性 |
| `tensor_cast/performance_model/__init__.py` | 导出 `OpGraphTracer`、`TensorIdentity`、`OperatorDAG`、`OpNode`、`DAGAnalyzer` |
| `tensor_cast/core/user_config.py` | 新增 `enable_des_simulation: bool = False` |
| `tensor_cast/core/model_runner.py` | 向 Runtime 传递 `enable_des_simulation`；推理后打印 DES 结果 |

---

## 8. 开发计划

### 8.1 阶段概览

| 阶段 | 范围 | 依赖 |
|------|------|------|
| **阶段 1** — 核心 DAG | TensorIdentity、OpGraphTracer（基础，无别名）、OpNode、OperatorDAG、单元测试 | 无 |
| **阶段 2** — 高级 DAG | 原地版本追踪、视图/别名组传播、基于 schema 的修改检测、与 MemoryTracker 别名逻辑集成 | 阶段 1 |
| **阶段 3** — DES 模块 | DESConfig、OpHandle、StreamType、StreamAssigner、DeviceStream、CPUDispatcher、TraceCollector、DESSimulator、单元测试 | 阶段 2 |
| **阶段 4** — 集成 | Runtime 集成、ModelRunner/UserInputConfig 修改、chrome trace 多流输出、DAGAnalyzer | 阶段 3 |
| **阶段 5** — 验证 | 使用真实模型的集成测试（Qwen3-32B、Kimi-K2）、端到端 chrome trace 验证、性能基准测试 | 阶段 4 |

### 8.2 阶段 1 — 核心 DAG

| 任务 | 描述 | 产出 |
|------|------|------|
| P1-01 | 在 `performance_model/` 下创建文件结构 | 目录 + `__init__.py` |
| P1-02 | 实现 `TensorIdentity`（frozen dataclass，uid，version） | `op_graph_tracer.py` |
| P1-03 | 实现 `OpNode`（带边的 dataclass） | `operator_dag.py` |
| P1-04 | 实现 `OperatorDAG`（容器，拓扑排序，graphviz/json 导出） | `operator_dag.py` |
| P1-05 | 实现基础 `OpGraphTracer`（record_op，build_dag）— 无别名追踪 | `op_graph_tracer.py` |
| P1-06 | 单元测试：张量追踪、DAG 构建、拓扑序、导出 | `tests/test_op_graph_tracer.py` |

### 8.3 阶段 2 — 高级 DAG

| 任务 | 描述 | 产出 |
|------|------|------|
| P2-01 | 基于 schema 的原地检测（`alias_info.is_write`） | `op_graph_tracer.py` |
| P2-02 | 原地递增的版本追踪 | `op_graph_tracer.py` |
| P2-03 | 视图/别名检测和 `register_alias()` | `op_graph_tracer.py` |
| P2-04 | 别名组版本传播 | `op_graph_tracer.py` |
| P2-05 | kwargs 修改处理（如 `reshape_and_cache` 修改 `kv_cache`） | `op_graph_tracer.py` |
| P2-06 | 测试：原地操作、视图、别名传播、kwargs 修改 | `tests/test_op_graph_tracer.py` |

### 8.4 阶段 3 — DES 模块

| 任务 | 描述 | 产出 |
|------|------|------|
| P3-01 | 创建 `des/` 包结构 | 目录 + `__init__.py` |
| P3-02 | 实现 `DESConfig`、`StreamType` | `des/config.py`、`des/stream.py` |
| P3-03 | 实现 `OpHandle` | `des/op_handle.py` |
| P3-04 | 实现 `StreamAssigner` | `des/stream_assigner.py` |
| P3-05 | 实现 `DeviceStream(stime.Task)`，含 FIFO + 依赖等待 | `des/stream.py` |
| P3-06 | 实现 `CPUDispatcher(stime.Task)`，含提前下发 | `des/cpu_dispatcher.py` |
| P3-07 | 实现 `TraceCollector`，含 chrome trace 导出 | `des/trace_collector.py` |
| P3-08 | 实现 `DESSimulator` 编排器 | `des/des_simulator.py` |
| P3-09 | 单元测试：mock DAG、验证重叠、验证 trace 格式 | `tests/test_des_simulator.py` |

### 8.5 阶段 4 — 集成

| 任务 | 描述 | 产出 |
|------|------|------|
| P4-01 | 修改 `Runtime.__init__` 新增参数 | `runtime.py` |
| P4-02 | 在 `__torch_dispatch__` 中钩入 `OpGraphTracer.record_op` | `runtime.py` |
| P4-03 | 在 `__exit__` 中构建 DAG + 运行 DES | `runtime.py` |
| P4-04 | 扩展 `get_trace_events()` 包含 DES 事件 | `runtime.py` |
| P4-05 | 新增 `operator_dag` 和 `des_result` 属性 | `runtime.py` |
| P4-06 | 实现 `DAGAnalyzer` | `dag_analyzer.py` |
| P4-07 | 修改 `UserInputConfig` + CLI 参数 | `user_config.py` |
| P4-08 | 修改 `ModelRunner` 传递标志并打印 DES 结果 | `model_runner.py` |

### 8.6 阶段 5 — 验证

| 任务 | 描述 | 产出 |
|------|------|------|
| P5-01 | 集成测试：`enable_dag_tracing=True` 的 Runtime 在合成算子上运行 | 测试文件 |
| P5-02 | 集成测试：`enable_des_simulation=True` 的 Runtime 在合成算子上运行 | 测试文件 |
| P5-03 | 端到端：`text_generate Qwen/Qwen3-32B --enable-des-simulation --chrome-trace` | Trace 文件 |
| P5-04 | 在 chrome://tracing 中验证 3 流 chrome trace | 手动验证 |
| P5-05 | 性能基准：测量 10K 算子上 DAG 追踪 + DES 的开销 | 基准报告 |
| P5-06 | 运行完整测试套件（`pytest tensor_cast/tests -n auto`）验证无回归 | CI 通过 |

---

## 9. 测试计划

### 9.1 单元测试 — DAG 模块

```python
# tensor_cast/tests/test_op_graph_tracer.py

class TestTensorIdentity:
    def test_equality_and_hashing(self): ...
    def test_different_versions_not_equal(self): ...
    def test_repr(self): ...

class TestOpGraphTracer:
    def test_basic_chain(self):
        """A -> B -> C：三个算子形成线性链。"""

    def test_diamond_dependency(self):
        """A 生产 T1；B 和 C 都消费 T1；D 消费 B 和 C 的输出。"""

    def test_tensor_id_reuse_prevented(self):
        """验证张量池防止 GC 导致的 ID 冲突。"""

    def test_inplace_operation_versioning(self):
        """A.add_(B) 应创建从 (A@v0, B@v0) 到 (A@v1) 的边。"""

    def test_view_alias_propagation(self):
        """A.view() 创建别名；修改 A 应同时递增两者的版本。"""

    def test_kwargs_mutation(self):
        """reshape_and_cache(key, value, kv_cache=...) 应追踪 kv_cache 修改。"""

    def test_model_input_detection(self):
        """无生产者的张量应被识别为模型输入。"""

    def test_model_output_detection(self):
        """无消费者的张量应被识别为模型输出。"""

class TestOperatorDAG:
    def test_topological_order(self): ...
    def test_get_roots_and_leaves(self): ...
    def test_graphviz_export(self): ...
    def test_json_export(self): ...

class TestDAGAnalyzer:
    def test_critical_path_linear(self): ...
    def test_critical_path_diamond(self): ...
    def test_parallelism_profile(self): ...
```

### 9.2 单元测试 — DES 模块

```python
# tensor_cast/tests/test_des_simulator.py

class TestStreamAssigner:
    def test_comm_ops_to_comm_stream(self): ...
    def test_compute_ops_to_compute_stream(self): ...
    def test_override(self): ...

class TestDESSimulator:
    def test_linear_chain_no_overlap(self):
        """三个连续计算算子：总时间 = 时间之和。"""

    def test_compute_comm_overlap(self):
        """无依赖的计算和通信算子应重叠。"""

    def test_dependency_blocks_execution(self):
        """依赖计算算子的通信算子必须等待。"""

    def test_cpu_dispatch_ahead(self):
        """验证 CPU 在设备完成之前就完成了调度。"""

    def test_chrome_trace_format(self):
        """验证输出具有正确的 pid/tid/ph/ts/dur 字段。"""

    def test_overlap_metrics(self):
        """验证 DESResult overlap_time_s 的计算。"""
```

### 9.3 集成测试

```python
class TestRuntimeDAGIntegration:
    def test_dag_tracing_enabled(self):
        """enable_dag_tracing=True 的 Runtime 构建 DAG。"""

    def test_des_simulation_enabled(self):
        """enable_des_simulation=True 的 Runtime 产生 DESResult。"""

    def test_chrome_trace_with_des(self):
        """export_chrome_trace 包含 DES 多流事件。"""

    def test_des_disabled_by_default(self):
        """默认 Runtime 行为不变。"""
```

---

## 10. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 张量池内存开销过高 | 低 | 中 | TensorCast 使用 meta 张量（每个约 128B）；在 100K 张量时添加阈值警告 |
| 基于 schema 的原地/别名检测对自定义算子不完整 | 中 | 高 | 回退到 ID 比较；文档化自定义算子要求 |
| 别名组传播 Bug | 中 | 高 | 对视图密集的模型进行充分测试；将 DAG 边与 MemoryTracker 对比 |
| salabim 环境状态在仿真之间泄漏 | 低 | 中 | 始终在 `finally` 块中调用 `stime.SimulationEnv.clear()` |
| 大模型上的 DES 开销 | 中 | 低 | DES 为可选功能；目标是 10K 算子的性能分析 |
| 现有测试套件回归 | 低 | 高 | 所有新功能均为可选（默认关闭）；运行完整测试套件 |
| salabim 中的 `unsatisfied_dep_count` 竞争 | 低 | 中 | salabim 是单线程协作式的，无真正竞争。通过断言验证 |

---

## 11. 未来扩展

| 扩展 | 描述 | 预估工作量 |
|------|------|------------|
| **双 batch 重叠** | 在独立计算流上仿真交替的 prefill/decode batch | 新增 DeviceStream 实例 + batch 感知的流分配 |
| **Chunk prefill/decode** | 建模分块 prefill 与交替 decode 迭代 | 扩展 CPUDispatcher 的调度策略 |
| **多设备流水线** | 跨设备仿真流水线并行，包含设备间通信 | 多个 DESSimulator 实例 + 设备间 Channel 任务 |
| **AIC/AIV 分流** | 将计算分为矩阵（AIC/Cube）和向量（AIV）流 | 新增 StreamType 值 + 更细粒度的算子分类 |
| **节点内/节点间通信** | 分离 NVLink 和网络的通信流 | 新增 StreamType 值 + CommGrid 感知的分配 |
| **DES 中的算子融合** | 将连续算子融合为单个 DES 执行单元 | FusionDetector + DES 前的 DAG 重写 |
| **动态调度** | 基于优先级或工作窃取的调度器 | 将 DeviceStream 中的 FIFO 队列替换为优先队列 |
| **DES 中的内存仿真** | 在 DES 执行期间追踪内存水位线 | 结合 MemoryTracker 分析与 DES 时间线 |

---

*文档版本：2.0*
*最后更新：2026-02-05*
