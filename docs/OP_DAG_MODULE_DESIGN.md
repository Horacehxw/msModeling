# Operator DAG & DES Multi-Stream Simulation: Technical Design

## Document Information

| Item | Value |
|------|-------|
| Version | 2.0 |
| Status | Draft — Pending Review |
| Branch | feat/op_dag_des |
| Last Updated | 2026-02-05 |
| Authors | — |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background & Motivation](#2-background--motivation)
3. [System Architecture](#3-system-architecture)
4. [Module 1 — Operator DAG](#4-module-1--operator-dag)
    - 4.1 [TensorIdentity](#41-tensoridentity)
    - 4.2 [OpGraphTracer](#42-opgraphtracer)
    - 4.3 [OperatorDAG & OpNode](#43-operatordag--opnode)
    - 4.4 [DAGAnalyzer](#44-daganalyzer)
5. [Module 2 — DES Multi-Stream Simulation](#5-module-2--des-multi-stream-simulation)
    - 5.1 [DES Architecture Overview](#51-des-architecture-overview)
    - 5.2 [DESConfig](#52-desconfig)
    - 5.3 [OpHandle](#53-ophandle)
    - 5.4 [StreamType & StreamAssigner](#54-streamtype--streamassigner)
    - 5.5 [DeviceStream](#55-devicestream)
    - 5.6 [CPUDispatcher](#56-cpudispatcher)
    - 5.7 [TraceCollector](#57-tracecollector)
    - 5.8 [DESSimulator](#58-dessimulator)
    - 5.9 [Overlap Mechanism & Synchronization](#59-overlap-mechanism--synchronization)
6. [Code Integration Analysis](#6-code-integration-analysis)
    - 6.1 [Runtime Integration](#61-runtime-integration)
    - 6.2 [ModelRunner Integration](#62-modelrunner-integration)
    - 6.3 [UserInputConfig Integration](#63-userinputconfig-integration)
    - 6.4 [Chrome Trace Extension](#64-chrome-trace-extension)
7. [File Inventory](#7-file-inventory)
8. [Development Plan](#8-development-plan)
9. [Test Plan](#9-test-plan)
10. [Risk Assessment](#10-risk-assessment)
11. [Future Extensions](#11-future-extensions)

---

## 1. Executive Summary

This document defines the technical design for two tightly-coupled modules added to TensorCast:

1. **Operator DAG** — Builds an explicit directed acyclic graph of operator data dependencies during `TorchDispatchMode` tracing. The DAG tracks tensor flow through versioned identities, correctly handling inplace mutations, view aliasing, and tensor ID reuse.

2. **DES Multi-Stream Simulation** — A salabim-based discrete event simulation that replays operator execution across multiple device streams (CPU dispatch, Compute, Communication). It uses the DAG to model data dependencies and produces chrome-compatible traces showing compute/communication overlap.

Together these modules enable:

- Critical path analysis of model execution
- Compute/communication overlap quantification
- Multi-stream chrome trace visualization (compatible with MindStudio Insight / chrome://tracing)
- Foundation for future dual-batch overlap, chunk prefill/decode interleaving, and multi-device pipeline simulation

---

## 2. Background & Motivation

### 2.1 Current Limitations

The existing TensorCast Runtime (`tensor_cast/runtime.py`) captures operations sequentially via `__torch_dispatch__` and computes performance using a single-stream roofline model. The current `get_trace_events()` (runtime.py:414-488) outputs events to a single thread (`tid=0`) per performance model process, with no support for multi-stream execution or overlap analysis.

The existing `MemoryTracker` (`tensor_cast/performance_model/memory_tracker.py`) already performs implicit data dependency tracking through `_TensorInfo.def_op_idx` and `use_op_indices`, as well as alias tracking via `_handle_aliasing()`. However, this information is used solely for memory lifecycle analysis, not for building an explicit DAG or scheduling simulation.

### 2.2 Goals

| Goal | Description |
|------|-------------|
| G1 | Build an explicit operator DAG with accurate data dependency edges |
| G2 | Model CPU kernel launch queue behavior (dispatch ahead of device) |
| G3 | Simulate compute/communication overlap on separate device streams |
| G4 | Produce multi-stream chrome traces for visualization |
| G5 | Provide critical path and parallelism metrics |
| G6 | Design for extensibility (additional streams, multi-device, dynamic scheduling) |

### 2.3 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| DES framework | **salabim** (not simpy) | Already used in ServingCast (`stime.py`); proven `Task`, `wait()/notify()` primitives |
| Simulation approach | **Full DES** (not trace-driven analytical) | Enables dynamic scheduling scenarios (dual-batch overlap, chunk prefill/decode interleaving, pipeline parallelism) |
| CPU dispatch model | **CPU dispatches ahead** | Matches real NPU runtime kernel launch queue behavior — CPU does not block on device execution |
| Stream assignment V1 | Communication ops → COMMUNICATION, all else → COMPUTE | Simple, extensible via annotation; sufficient for V1 overlap analysis |
| Communication streams V1 | Single COMMUNICATION stream | Can extend to intra/inter-node, AIV/AIC in future |

---

## 3. System Architecture

### 3.1 End-to-End Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            Runtime (TorchDispatchMode)                        │
│                                                                              │
│  __torch_dispatch__(func, args, kwargs)                                      │
│       │                                                                      │
│       ├──► OpInvokeInfo ──► op_invoke_infos[]                                │
│       ├──► MemoryTracker.record_op_invocation()                              │
│       └──► OpGraphTracer.record_op()  ◄── NEW                               │
│                                                                              │
│  __exit__()                                                                  │
│       │                                                                      │
│       ├──► repeat_op_invoke_infos()                                          │
│       ├──► replay_op_invoke_infos() ──► event_list[]                         │
│       ├──► MemoryTracker.analyze()                                           │
│       ├──► OpGraphTracer.build_dag() ──► OperatorDAG  ◄── NEW               │
│       └──► DESSimulator.run() ──► DES trace events    ◄── NEW               │
│                                                                              │
│  get_trace_events()                                                          │
│       └──► existing events + DES multi-stream events  ◄── NEW               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Module Dependency Graph

```
tensor_cast/
├── runtime.py                          # MODIFIED: integrate OpGraphTracer + DES
├── performance_model/
│   ├── __init__.py                     # MODIFIED: export new classes
│   ├── memory_tracker.py               # UNCHANGED
│   ├── analytic.py                     # UNCHANGED
│   ├── op_graph_tracer.py              # NEW: TensorIdentity, OpGraphTracer
│   ├── operator_dag.py                 # NEW: OpNode, OperatorDAG
│   ├── dag_analyzer.py                 # NEW: DAGAnalyzer
│   └── des/                            # NEW: DES simulation package
│       ├── __init__.py
│       ├── config.py                   #   DESConfig
│       ├── op_handle.py                #   OpHandle
│       ├── stream.py                   #   StreamType, DeviceStream(stime.Task)
│       ├── cpu_dispatcher.py           #   CPUDispatcher(stime.Task)
│       ├── stream_assigner.py          #   StreamAssigner
│       ├── trace_collector.py          #   TraceCollector
│       └── des_simulator.py            #   DESSimulator (top-level orchestrator)
├── core/
│   ├── model_runner.py                 # MODIFIED: pass DES flags, print DES results
│   └── user_config.py                  # MODIFIED: add enable_des_simulation flag
└── tests/
    ├── test_op_graph_tracer.py         # NEW
    └── test_des_simulator.py           # NEW
```

### 3.3 Relationship to Existing Components

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
│  │(existing)│  │  (existing)  │  │  (NEW)       │    │
│  └─────┬────┘  └──────────────┘  └──────┬───────┘    │
│        │                                 │            │
│        │        ┌────────────────┐       │            │
│        └───────►│ PerformanceModel│       │            │
│                 │ (existing)     │       │            │
│                 └───────┬────────┘       │            │
│                         │                │            │
│                    event_list      OperatorDAG        │
│                         │                │            │
│                         ▼                ▼            │
│                 ┌────────────────────────────┐        │
│                 │      DESSimulator          │        │
│                 │      (NEW)                 │        │
│                 └────────────────────────────┘        │
└───────────────────────────────────────────────────────┘
```

---

## 4. Module 1 — Operator DAG

### 4.1 TensorIdentity

#### 4.1.1 Problem Statement

Building a DAG requires matching tensor producers to consumers via `id()`. Two problems arise:

**Problem 1 — Tensor ID Reuse**: Python's `id()` returns an object's memory address, unique only during its lifetime. When an intermediate tensor is garbage-collected, a new tensor may reuse its address, creating false dependency edges.

**Problem 2 — Inplace Operations**: Inplace ops (e.g., `add_()`) return the same Python object. A naive id-based approach cannot distinguish the pre-mutation and post-mutation states of the tensor.

#### 4.1.2 Solution

| Problem | Solution |
|---------|----------|
| ID reuse | **Strong reference pool** — The tracer holds all tensors in a list, preventing GC. Since TensorCast uses meta tensors (no storage), memory overhead is negligible (~128 bytes per tensor object). |
| Inplace | **Versioned identity** `(uid, version)` — Each tensor gets a monotonic `uid`. On inplace mutation, `version` is incremented to create a new identity. The input side reads `(uid, v)`, the output side produces `(uid, v+1)`. |

#### 4.1.3 Data Structure

```python
# File: tensor_cast/performance_model/op_graph_tracer.py

@dataclass(frozen=True, slots=True)
class TensorIdentity:
    """
    Versioned tensor identity for accurate DAG edge tracking.

    The (uid, version) tuple uniquely identifies a tensor state.
    - uid: monotonically assigned during tracing
    - version: incremented on each inplace mutation
    """
    uid: int
    version: int = 0

    def __repr__(self) -> str:
        return f"T{self.uid}v{self.version}"
```

#### 4.1.4 Alias Group Tracking

When tensor `A` has a view `B` (via `view()`, `slice`, `transpose()`, etc.), both share the same underlying storage. If `A` is mutated inplace, `B`'s data is also affected. The tracer maintains **alias groups** — sets of UIDs that share storage — and propagates version increments to all members of the group.

```python
# Alias groups: uid -> set of aliased uids
_alias_groups: Dict[int, Set[int]] = {}

def _increment_version(self, uid: int) -> int:
    """Increment version for uid AND all aliases in its group."""
    new_version = self._version_map[uid] + 1
    for alias_uid in self._alias_groups.get(uid, {uid}):
        self._version_map[alias_uid] = new_version
    return new_version
```

Detection of view operations reuses the same schema inspection logic as `MemoryTracker._handle_aliasing()` (memory_tracker.py:84-146), checking `func._schema.returns[i].alias_info.before_set` against argument alias sets.

### 4.2 OpGraphTracer

#### 4.2.1 Responsibilities

| Responsibility | Mechanism |
|---------------|-----------|
| Assign stable tensor identities | `_id_to_uid` mapping + monotonic counter |
| Prevent tensor GC during tracing | `_tensor_pool: List[torch.Tensor]` |
| Detect inplace mutations | Schema-based: `arg.alias_info.is_write`; fallback: `id(input) == id(output)` |
| Detect view aliasing | Schema-based: `return.alias_info.before_set & arg.alias_info.before_set` |
| Track kwargs mutations | Schema `mutates_args` (e.g., `reshape_and_cache` mutates `kv_cache`) |
| Record op tensor flow | Store `(OpInvokeInfo, input_identities, output_identities)` per op |
| Build DAG | Two-pass: create nodes, then wire edges via identity matching |

#### 4.2.2 API

```python
class OpGraphTracer:
    def __init__(self, track_aliases: bool = True): ...

    def record_op(self, op_invoke_info: OpInvokeInfo):
        """Called per op during __torch_dispatch__. Records tensor flow."""

    def build_dag(self) -> OperatorDAG:
        """Called after tracing. Builds DAG from recorded tensor flow."""

    def clear(self):
        """Release tensor references after DAG is built."""

    @property
    def dag(self) -> Optional[OperatorDAG]: ...

    def get_statistics(self) -> Dict[str, Any]: ...
```

#### 4.2.3 `record_op` Algorithm

```
1. Extract input tensors from args + kwargs
2. For each input tensor:
   a. Assign UID if new (add to pool)
   b. Read current identity (uid, version)
   c. Add to input_identities set
3. Detect mutations via schema (alias_info.is_write)
4. Detect view aliasing via schema (alias_info.before_set matching)
5. If view detected: register_alias(source, view)
6. Extract output tensors
7. For each output tensor:
   a. If id(output) in input_ids → inplace: increment version
   b. Else → new tensor: assign fresh identity
8. Handle kwargs mutations (e.g., kv_cache):
   a. For each mutated arg index from schema
   b. Increment version, add to output_identities
9. Store (op_invoke_info, input_identities, output_identities)
```

#### 4.2.4 `build_dag` Algorithm

```
Pass 1 — Create nodes:
  For each recorded op:
    Create OpNode with input/output identities
    Register each output identity's producer op_idx in identity_producers map

Pass 2 — Wire edges:
  For each node's input identity:
    Look up producer in identity_producers
    If found: add edge (producer → this node)
    If not found: mark as model input

Pass 3 — Identify model outputs:
  For each produced identity:
    If never consumed by any node: mark as model output
```

### 4.3 OperatorDAG & OpNode

#### 4.3.1 OpNode

```python
# File: tensor_cast/performance_model/operator_dag.py

@dataclass
class OpNode:
    op_idx: int
    op_invoke_info: OpInvokeInfo
    input_identities: Set[TensorIdentity]
    output_identities: Set[TensorIdentity]
    predecessors: Set[int]    # op indices this node depends on
    successors: Set[int]      # op indices that depend on this node

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

    # Reverse index: identity -> producer op_idx
    _identity_producers: Dict[TensorIdentity, int]

    @property
    def num_nodes(self) -> int: ...

    @property
    def num_edges(self) -> int: ...

    def get_roots(self) -> List[int]:
        """Nodes with no predecessors (consume only model inputs)."""

    def get_leaves(self) -> List[int]:
        """Nodes with no successors (produce only model outputs)."""

    def topological_order(self) -> Iterator[int]:
        """BFS-based Kahn's algorithm. Dependencies before dependents."""

    def reverse_topological_order(self) -> Iterator[int]: ...

    def to_graphviz(self, show_tensors: bool = False) -> str:
        """Export to DOT format for Graphviz visualization."""

    def to_json(self) -> str:
        """Export to JSON for programmatic analysis."""
```

### 4.4 DAGAnalyzer

```python
# File: tensor_cast/performance_model/dag_analyzer.py

@dataclass
class CriticalPathResult:
    path: List[int]                          # Op indices in critical path order
    total_time_s: float                      # Sum of execution times on the path
    bottleneck_ops: List[Tuple[int, float]]  # Top-5 (op_idx, time_s)

class DAGAnalyzer:
    def __init__(self, dag: OperatorDAG): ...

    def compute_critical_path(
        self, get_op_time: Optional[Callable[[int], float]] = None
    ) -> CriticalPathResult:
        """
        Longest-path DP on topological order.
        get_op_time defaults to unit cost if not provided.
        """

    def compute_parallelism_profile(self) -> List[int]:
        """
        Assign each node to a level (max predecessor level + 1).
        Returns list where index i = number of ops at level i.
        """

    def get_summary(self) -> Dict:
        """
        Returns: num_nodes, num_edges, num_roots, num_leaves,
                 dag_depth, max_parallelism, avg_parallelism
        """
```

---

## 5. Module 2 — DES Multi-Stream Simulation

### 5.1 DES Architecture Overview

The DES module simulates operator execution on a device with multiple hardware streams. It builds on salabim via the existing `stime.py` wrapper, which provides `Task` (extends `salabim.Component`), `wait()`/`notify()` primitives, and `elapse()` for logical time advancement.

```
┌──────────────────────────────────────────────────────────────────┐
│                        DESSimulator                               │
│  (creates salabim env, sets up streams, runs simulation)          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐                                             │
│  │  CPUDispatcher   │  stime.Task: walks DAG in topo order,      │
│  │  (stime.Task)    │  spends dispatch_latency per op,            │
│  │                  │  enqueues OpHandle on target DeviceStream.   │
│  │                  │  Does NOT wait for device completion.        │
│  └───────┬──────────┘                                             │
│          │ enqueue(op_handle)                                     │
│          ▼                                                        │
│  ┌──────────────────┐    ┌──────────────────┐                     │
│  │  DeviceStream    │    │  DeviceStream    │                     │
│  │  "Compute"       │    │  "Communication" │                     │
│  │  (stime.Task)    │    │  (stime.Task)    │                     │
│  │                  │    │                  │                     │
│  │  FIFO queue:     │    │  FIFO queue:     │                     │
│  │  dequeue op      │    │  dequeue op      │                     │
│  │  wait for deps   │    │  wait for deps   │                     │
│  │  elapse(time)    │    │  elapse(time)    │                     │
│  │  notify succs    │    │  notify succs    │                     │
│  └──────────────────┘    └──────────────────┘                     │
│                                                                   │
│  ┌──────────────────┐                                             │
│  │  TraceCollector  │  Records (op_name, stream, start, end)     │
│  │                  │  per op for chrome trace export.             │
│  └──────────────────┘                                             │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 DESConfig

```python
# File: tensor_cast/performance_model/des/config.py

@dataclass
class DESConfig:
    """Configuration for DES simulation."""

    # Fixed CPU dispatch latency per op (seconds).
    # Models the time CPU spends preparing and launching a kernel.
    cpu_dispatch_latency_s: float = 10e-6  # 10 μs default

    # Per-op overrides for stream assignment.
    # Keys: op name string (e.g., "tensor_cast::all_reduce")
    # Values: StreamType enum value
    stream_overrides: Dict[str, StreamType] = field(default_factory=dict)

    # Whether to collect trace events for chrome trace export
    enable_trace_collection: bool = True
```

### 5.3 OpHandle

`OpHandle` is the simulation-time representation of a single op. It holds all metadata needed during DES execution: the DAG node reference, performance results, timing records, dependency tracking state, and assigned stream.

```python
# File: tensor_cast/performance_model/des/op_handle.py

@dataclass
class OpHandle:
    """Represents an operator during DES simulation."""

    # Identity
    op_idx: int
    op_name: str
    dag_node: OpNode

    # Execution time from PerformanceModel (seconds)
    device_execution_time_s: float

    # Stream assignment (set by StreamAssigner, before simulation)
    target_stream_type: StreamType
    assigned_stream: Optional['DeviceStream'] = None

    # Timing records (filled during simulation)
    cpu_start_s: float = 0.0
    cpu_end_s: float = 0.0
    device_start_s: float = 0.0
    device_end_s: float = 0.0

    # DAG dependency tracking
    successor_handles: List['OpHandle'] = field(default_factory=list)
    unsatisfied_dep_count: int = 0  # Decremented as predecessors complete

    @property
    def deps_satisfied(self) -> bool:
        return self.unsatisfied_dep_count == 0

    def on_dep_satisfied(self, predecessor: 'OpHandle'):
        """Called when a predecessor completes."""
        self.unsatisfied_dep_count -= 1

    def mark_completed(self):
        """Signal all successors that this op is done."""
        for succ in self.successor_handles:
            succ.on_dep_satisfied(self)
```

**Lifecycle**:

```
Created (by DESSimulator) → CPU dispatched (by CPUDispatcher) → Enqueued on DeviceStream
→ Deps checked → Executed (elapse) → Completed → Successors notified
```

### 5.4 StreamType & StreamAssigner

```python
# File: tensor_cast/performance_model/des/stream.py

class StreamType(Enum):
    CPU = "cpu"
    COMPUTE = "compute"
    COMMUNICATION = "communication"
```

```python
# File: tensor_cast/performance_model/des/stream_assigner.py

class StreamAssigner:
    """Maps operators to target device streams."""

    # Default V1 mapping: communication ops go to COMMUNICATION stream,
    # everything else to COMPUTE stream.
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

        # Check explicit overrides first
        if op_name in self._overrides:
            return self._overrides[op_name]

        # Default: comm ops -> COMMUNICATION, else -> COMPUTE
        if op_name in self.COMM_OPS:
            return StreamType.COMMUNICATION

        return StreamType.COMPUTE
```

This design is extensible: future versions can add `StreamType.AIV`, `StreamType.AIC`, `StreamType.INTRA_NODE_COMM`, etc., and override assignments per op via `DESConfig.stream_overrides` or annotation.

### 5.5 DeviceStream

`DeviceStream` extends `stime.Task` and models a hardware execution stream. It runs a FIFO loop: dequeue an op, wait until its data dependencies are satisfied, execute (advance logical time), and notify successors.

```python
# File: tensor_cast/performance_model/des/stream.py

class DeviceStream(stime.Task):
    """
    A device execution stream modeled as a salabim Task.

    Runs a FIFO loop:
    1. Dequeue next op from the queue
    2. Wait until all predecessor ops have completed (deps_satisfied)
    3. Execute: record start time, elapse device_execution_time_s, record end time
    4. Mark completed, notify successor ops
    5. If successor's deps are now all satisfied, wake its assigned stream
    """

    def __init__(self, stream_type: StreamType, trace_collector: 'TraceCollector'):
        super().__init__(name=stream_type.value)
        self._stream_type = stream_type
        self._trace_collector = trace_collector
        self._queue: Deque[OpHandle] = deque()
        self._shutdown = False

    def enqueue(self, op_handle: OpHandle):
        """Called by CPUDispatcher to add an op to this stream's FIFO."""
        self._queue.append(op_handle)
        self.notify()  # Wake process() if it's waiting for work

    def shutdown(self):
        """Signal that no more ops will be enqueued."""
        self._shutdown = True
        self.notify()

    def process(self):
        """salabim Task main loop."""
        while True:
            # Wait for work
            while not self._queue and not self._shutdown:
                self.wait()
            if not self._queue and self._shutdown:
                break

            op = self._queue.popleft()

            # Wait for all data dependencies
            while not op.deps_satisfied:
                self.wait()

            # Execute
            op.device_start_s = stime.now()
            stime.elapse(op.device_execution_time_s)
            op.device_end_s = stime.now()

            # Record trace event
            self._trace_collector.record_device_event(op, self._stream_type)

            # Notify successors
            op.mark_completed()
            for succ in op.successor_handles:
                if succ.deps_satisfied and succ.assigned_stream is not None:
                    succ.assigned_stream.notify()
```

**Key behavior**: The `while not op.deps_satisfied: self.wait()` loop causes the stream to passivate (yield control to the salabim scheduler). The stream is woken when a predecessor completes and calls `succ.assigned_stream.notify()`.

### 5.6 CPUDispatcher

`CPUDispatcher` extends `stime.Task` and models the CPU's kernel launch behavior. It walks the DAG in topological order, spending `cpu_dispatch_latency_s` per op, and enqueues each op onto its target device stream. Critically, **the CPU does not wait for device execution** — it dispatches ahead, modeling the real NPU runtime's kernel launch queue.

```python
# File: tensor_cast/performance_model/des/cpu_dispatcher.py

class CPUDispatcher(stime.Task):
    """
    CPU dispatch task. Walks DAG topo order, dispatches ops to device streams.

    Models real NPU runtime behavior where CPU queues multiple kernels
    onto device streams without blocking for device completion.
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

            # CPU dispatch latency
            op.cpu_start_s = stime.now()
            stime.elapse(self._config.cpu_dispatch_latency_s)
            op.cpu_end_s = stime.now()

            # Record CPU trace event
            self._trace_collector.record_cpu_event(op)

            # Enqueue on target device stream (non-blocking)
            target_stream = self._device_streams[op.target_stream_type]
            op.assigned_stream = target_stream
            target_stream.enqueue(op)

        # Signal all streams that dispatch is complete
        for stream in self._device_streams.values():
            stream.shutdown()
```

### 5.7 TraceCollector

`TraceCollector` records timing events during the DES simulation and converts them to the Chrome Trace Event Format for visualization.

```python
# File: tensor_cast/performance_model/des/trace_collector.py

@dataclass
class TraceEvent:
    op_name: str
    stream_type: StreamType
    start_s: float
    end_s: float
    op_idx: int
    extra_args: Dict[str, Any] = field(default_factory=dict)

class TraceCollector:
    """Collects DES events and produces chrome trace JSON."""

    # Thread ID mapping for chrome trace
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
        Convert to Chrome Trace Event Format.

        Args:
            pid: Process ID offset for integration with existing Runtime traces.

        Returns:
            List of Chrome Trace JSON event dicts.
        """
        trace_events = []

        # Metadata: name the process and threads
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

        # Complete events ("X" type)
        for event in self.events:
            tid = self.STREAM_TID[event.stream_type]
            trace_events.append({
                "name": event.op_name,
                "cat": event.stream_type.value,
                "ph": "X",
                "ts": event.start_s * 1e6,   # seconds → microseconds
                "dur": (event.end_s - event.start_s) * 1e6,
                "pid": pid,
                "tid": tid,
                "args": {"op_idx": event.op_idx, **event.extra_args},
            })

        return trace_events
```

**Chrome trace thread layout**:

| TID | Stream | Description |
|-----|--------|-------------|
| 0 | CPU Dispatch | Kernel launch queue — one event per op dispatch |
| 1 | Compute Stream | Matrix ops, attention, elementwise, etc. |
| 2 | Communication Stream | all_reduce, all_gather, all_to_all, etc. |

This is directly viewable in chrome://tracing or MindStudio Insight.

### 5.8 DESSimulator

`DESSimulator` is the top-level orchestrator. It takes a DAG, event list (with performance results), and config, and runs the full salabim simulation.

```python
# File: tensor_cast/performance_model/des/des_simulator.py

@dataclass
class DESResult:
    """Results from DES simulation."""
    total_time_s: float
    compute_time_s: float         # Total time on compute stream
    communication_time_s: float   # Total time on comm stream
    overlap_time_s: float         # Time where both streams active
    trace_collector: TraceCollector

class DESSimulator:
    """
    Top-level DES simulator for multi-stream operator execution.

    Usage:
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
        # 1. Initialize salabim environment
        stime.init_simulation()

        # 2. Build OpHandles from DAG + event_list
        stream_assigner = StreamAssigner(self._config)
        op_handles = self._build_op_handles(stream_assigner)

        # 3. Wire successor relationships from DAG edges
        self._wire_dependencies(op_handles)

        # 4. Create TraceCollector
        trace_collector = TraceCollector()

        # 5. Create DeviceStreams
        device_streams = {
            StreamType.COMPUTE: DeviceStream(StreamType.COMPUTE, trace_collector),
            StreamType.COMMUNICATION: DeviceStream(StreamType.COMMUNICATION, trace_collector),
        }

        # 6. Create CPUDispatcher
        topo_order = list(self._dag.topological_order())
        CPUDispatcher(op_handles, topo_order, device_streams, trace_collector, self._config)

        # 7. Run simulation
        stime.start_simulation()

        # 8. Collect results
        result = self._collect_results(trace_collector)

        # 9. Clean up salabim environment
        stime.SimulationEnv.clear()

        return result

    def _build_op_handles(self, stream_assigner: StreamAssigner) -> Dict[int, OpHandle]:
        """Create OpHandle for each DAG node, with execution time from event_list."""
        op_handles = {}
        for op_idx, node in self._dag.nodes.items():
            # Look up execution time from the event_list
            event = self._event_list[op_idx]
            exec_time = self._get_execution_time(event)

            op_handles[op_idx] = OpHandle(
                op_idx=op_idx,
                op_name=node.op_name,
                dag_node=node,
                device_execution_time_s=exec_time,
                target_stream_type=stream_assigner.assign(node.op_invoke_info),
            )
        return op_handles

    def _wire_dependencies(self, op_handles: Dict[int, OpHandle]):
        """Wire successor_handles and unsatisfied_dep_count from DAG edges."""
        for op_idx, handle in op_handles.items():
            node = self._dag.nodes[op_idx]
            # Set up successor handles
            handle.successor_handles = [op_handles[s] for s in node.successors]
            # Count how many predecessors this op must wait for
            # Only count cross-stream deps + same-stream non-adjacent deps
            # In practice: count all DAG predecessors
            handle.unsatisfied_dep_count = len(node.predecessors)

    def _get_execution_time(self, event: RuntimeEvent) -> float:
        """Extract execution time from RuntimeEvent."""
        if self._perf_model_name and self._perf_model_name in event.perf_results:
            return event.perf_results[self._perf_model_name].execution_time_s
        # Default: first model's result
        for result in event.perf_results.values():
            return result.execution_time_s
        return 0.0

    def _collect_results(self, trace_collector: TraceCollector) -> DESResult:
        """Compute overlap metrics from collected events."""
        compute_events = [e for e in trace_collector.events
                         if e.stream_type == StreamType.COMPUTE]
        comm_events = [e for e in trace_collector.events
                      if e.stream_type == StreamType.COMMUNICATION]

        compute_time = sum(e.end_s - e.start_s for e in compute_events)
        comm_time = sum(e.end_s - e.start_s for e in comm_events)
        total_time = stime.now() if trace_collector.events else 0.0
        overlap_time = compute_time + comm_time - total_time

        return DESResult(
            total_time_s=total_time,
            compute_time_s=compute_time,
            communication_time_s=comm_time,
            overlap_time_s=max(0.0, overlap_time),
            trace_collector=trace_collector,
        )
```

### 5.9 Overlap Mechanism & Synchronization

#### 5.9.1 CPU Dispatch Ahead Model

The CPU does not wait for device execution to complete before dispatching the next op. This models real NPU runtime behavior where the CPU enqueues multiple kernels into a device command queue:

```
CPU:      [disp A][disp B][disp C][disp D]...
Compute:          [====== A ======]       [=== D ===]
Comm:                    [=== B ===][=== C ===]
                         ↑ overlap ↑
```

In this example, ops B and C (communication) have no data dependency on A (compute), so the Communication stream starts executing B while the Compute stream is still executing A.

#### 5.9.2 Dependency Synchronization

Data dependencies are enforced through the `OpHandle.unsatisfied_dep_count` mechanism:

1. When `DESSimulator._wire_dependencies()` is called, each op's `unsatisfied_dep_count` is set to the number of its DAG predecessors.
2. When a predecessor completes (`mark_completed()`), it decrements each successor's `unsatisfied_dep_count` via `on_dep_satisfied()`.
3. When `unsatisfied_dep_count` reaches 0, `deps_satisfied` becomes `True`.
4. If the successor's `DeviceStream` is passivated (waiting), it is woken via `notify()`.

This ensures that:
- Independent ops on different streams execute concurrently (overlap).
- Dependent ops on different streams respect data ordering (e.g., `all_reduce` must wait for the `mm` that produces its input).
- Same-stream ops execute in FIFO order (stream serialization).

#### 5.9.3 salabim Execution Model

salabim uses cooperative multitasking:

| Primitive | salabim API | stime Wrapper | Behavior |
|-----------|-------------|---------------|----------|
| Create task | `sim.Component()` | `stime.Task()` | Registers component in environment |
| Advance time | `component.hold(t)` | `stime.elapse(t)` | Yields control, resumes after `t` logical time |
| Passivate | `component.passivate()` | `task.wait()` | Yields control, waits for explicit activation |
| Activate | `component.activate()` | `task.notify()` | Wakes a passivated task (no-op if already active) |
| Run | `env.run()` | `stime.start_simulation()` | Runs until all tasks terminate |

The simulation terminates when all three tasks (CPUDispatcher, Compute DeviceStream, Communication DeviceStream) have completed their `process()` methods.

---

## 6. Code Integration Analysis

### 6.1 Runtime Integration

**File**: `tensor_cast/runtime.py`

**Changes to `__init__`** (line 41-61):

```python
def __init__(
    self,
    perf_models: Union[PerformanceModel, List[PerformanceModel]],
    device_profile: DeviceProfile,
    memory_tracker: Optional[MemoryTracker] = None,
    enable_dag_tracing: bool = False,          # NEW
    enable_des_simulation: bool = False,       # NEW
    des_config: Optional[DESConfig] = None,    # NEW
):
    # ... existing init ...

    # NEW: DAG tracing
    self._enable_dag_tracing = enable_dag_tracing or enable_des_simulation
    self._op_graph_tracer: Optional[OpGraphTracer] = None
    if self._enable_dag_tracing:
        self._op_graph_tracer = OpGraphTracer(track_aliases=True)

    # NEW: DES simulation
    self._enable_des_simulation = enable_des_simulation
    self._des_config = des_config or DESConfig()
    self._des_result: Optional[DESResult] = None
```

Note: `enable_des_simulation=True` implies `enable_dag_tracing=True` since the DES requires the DAG.

**Changes to `__torch_dispatch__`** (line 69-77):

```python
def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    kwargs = {} if kwargs is None else kwargs
    if not torch.compiler.is_compiling():
        out = func(*args, **kwargs)
        op_invoke_info = OpInvokeInfo(func, args, kwargs, out)
        self.op_invoke_infos.append(op_invoke_info)

        # NEW: Record for DAG tracing
        if self._op_graph_tracer is not None:
            self._op_graph_tracer.record_op(op_invoke_info)

        return out
    else:
        return func(*args, **kwargs)
```

**Changes to `__exit__`** (currently around line 150-170):

```python
def __exit__(self, exc_type, exc_val, exc_tb):
    super().__exit__(exc_type, exc_val, exc_tb)
    self.repeat_op_invoke_infos()
    self.replay_op_invoke_infos()
    if self.memory_tracker:
        self.memory_tracker.analyze()

    # NEW: Build DAG
    if self._op_graph_tracer is not None:
        self._op_graph_tracer.build_dag()

    # NEW: Run DES simulation
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

    # Release tensor references (keep DAG structure)
    if self._op_graph_tracer is not None:
        self._op_graph_tracer.clear()

    _current_runtime.value = None
```

**Changes to `get_trace_events`** (line 414-488):

```python
def get_trace_events(self):
    trace_events = []
    # ... existing trace event generation ...

    # NEW: Append DES multi-stream trace events
    if self._des_result is not None:
        # Use a PID offset to separate DES events from performance model events
        des_pid = len(self.perf_models)
        des_events = self._des_result.trace_collector.to_chrome_trace_events(pid=des_pid)
        trace_events.extend(des_events)

    return trace_events
```

**New properties**:

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

### 6.2 ModelRunner Integration

**File**: `tensor_cast/core/model_runner.py` (line 95-100)

The `Runtime()` constructor call needs the new flags:

```python
with (
    Runtime(
        self.perf_model,
        self.device_profile,
        memory_tracker=MemoryTracker(self.device_profile),
        enable_des_simulation=self.user_input.enable_des_simulation,  # NEW
    ) as runtime,
    torch.no_grad(),
):
    logits = self.model.forward(**input_kwargs)
```

After the `with` block, print DES results:

```python
# NEW: Print DES overlap results
if runtime.des_result is not None:
    des = runtime.des_result
    print(f"\n--- DES Multi-Stream Simulation ---")
    print(f"Total simulated time: {des.total_time_s * 1e3:.3f} ms")
    print(f"Compute time:         {des.compute_time_s * 1e3:.3f} ms")
    print(f"Communication time:   {des.communication_time_s * 1e3:.3f} ms")
    print(f"Overlap time:         {des.overlap_time_s * 1e3:.3f} ms")
    if des.communication_time_s > 0:
        overlap_pct = des.overlap_time_s / des.communication_time_s * 100
        print(f"Overlap ratio:        {overlap_pct:.1f}% of communication overlapped")
```

The existing `export_chrome_trace` call (model_runner.py:164-165) requires no changes since `get_trace_events()` already includes DES events when enabled.

### 6.3 UserInputConfig Integration

**File**: `tensor_cast/core/user_config.py`

Add a new field:

```python
@dataclass
class UserInputConfig:
    # ... existing fields ...
    enable_des_simulation: bool = False    # NEW
```

CLI argument (in the relevant argparse setup):

```python
parser.add_argument(
    "--enable-des-simulation",
    action="store_true",
    default=False,
    help="Enable DES multi-stream simulation for compute/communication overlap analysis",
)
```

### 6.4 Chrome Trace Extension

The current chrome trace output structure (from runtime.py:414-488) uses:
- PID per performance model
- TID=0 (single stream per model)

With DES enabled, the trace file will additionally contain:
- PID = `len(perf_models)` for DES events
- TID=0 (CPU Dispatch), TID=1 (Compute), TID=2 (Communication)

**Example trace layout in chrome://tracing**:

```
PID 0: "AnalyticPerformanceModel" (existing single-stream)
  TID 0: Stream 0
    [op1][op2][op3]...

PID 1: "DES Multi-Stream Simulation" (new)
  TID 0: CPU Dispatch
    [disp1][disp2][disp3]...
  TID 1: Compute Stream
    [======op1======][===op3===]...
  TID 2: Communication Stream
             [===op2===]...
```

---

## 7. File Inventory

### 7.1 New Files

| File | Module | Key Classes/Functions |
|------|--------|----------------------|
| `tensor_cast/performance_model/op_graph_tracer.py` | DAG | `TensorIdentity`, `OpGraphTracer` |
| `tensor_cast/performance_model/operator_dag.py` | DAG | `OpNode`, `OperatorDAG` |
| `tensor_cast/performance_model/dag_analyzer.py` | DAG | `DAGAnalyzer`, `CriticalPathResult` |
| `tensor_cast/performance_model/des/__init__.py` | DES | Package exports |
| `tensor_cast/performance_model/des/config.py` | DES | `DESConfig` |
| `tensor_cast/performance_model/des/op_handle.py` | DES | `OpHandle` |
| `tensor_cast/performance_model/des/stream.py` | DES | `StreamType`, `DeviceStream` |
| `tensor_cast/performance_model/des/cpu_dispatcher.py` | DES | `CPUDispatcher` |
| `tensor_cast/performance_model/des/stream_assigner.py` | DES | `StreamAssigner` |
| `tensor_cast/performance_model/des/trace_collector.py` | DES | `TraceCollector`, `TraceEvent` |
| `tensor_cast/performance_model/des/des_simulator.py` | DES | `DESSimulator`, `DESResult` |
| `tensor_cast/tests/test_op_graph_tracer.py` | Test | DAG module unit/integration tests |
| `tensor_cast/tests/test_des_simulator.py` | Test | DES module unit/integration tests |

### 7.2 Modified Files

| File | Change Summary |
|------|----------------|
| `tensor_cast/runtime.py` | Add `enable_dag_tracing`, `enable_des_simulation`, `des_config` params; hook `record_op` in `__torch_dispatch__`; build DAG + run DES in `__exit__`; extend `get_trace_events()`; add `operator_dag` and `des_result` properties |
| `tensor_cast/performance_model/__init__.py` | Export `OpGraphTracer`, `TensorIdentity`, `OperatorDAG`, `OpNode`, `DAGAnalyzer` |
| `tensor_cast/core/user_config.py` | Add `enable_des_simulation: bool = False` |
| `tensor_cast/core/model_runner.py` | Pass `enable_des_simulation` to Runtime; print DES results after inference |

---

## 8. Development Plan

### 8.1 Phase Overview

| Phase | Scope | Dependencies |
|-------|-------|-------------|
| **Phase 1** — Core DAG | TensorIdentity, OpGraphTracer (basic, no aliases), OpNode, OperatorDAG, unit tests | None |
| **Phase 2** — Advanced DAG | Inplace version tracking, view/alias group propagation, schema-based mutation detection, integration with MemoryTracker alias logic | Phase 1 |
| **Phase 3** — DES Module | DESConfig, OpHandle, StreamType, StreamAssigner, DeviceStream, CPUDispatcher, TraceCollector, DESSimulator, unit tests | Phase 2 |
| **Phase 4** — Integration | Runtime integration, ModelRunner/UserInputConfig changes, chrome trace multi-stream output, DAGAnalyzer | Phase 3 |
| **Phase 5** — Validation | Integration tests with real models (Qwen3-32B, Kimi-K2), end-to-end chrome trace verification, performance benchmarking | Phase 4 |

### 8.2 Phase 1 — Core DAG

| Task | Description | Deliverable |
|------|-------------|-------------|
| P1-01 | Create file structure under `performance_model/` | Directory + `__init__.py` |
| P1-02 | Implement `TensorIdentity` (frozen dataclass, uid, version) | `op_graph_tracer.py` |
| P1-03 | Implement `OpNode` (dataclass with edges) | `operator_dag.py` |
| P1-04 | Implement `OperatorDAG` (container, topo sort, graphviz/json export) | `operator_dag.py` |
| P1-05 | Implement basic `OpGraphTracer` (record_op, build_dag) — no alias tracking | `op_graph_tracer.py` |
| P1-06 | Unit tests: tensor tracking, DAG building, topo order, export | `tests/test_op_graph_tracer.py` |

### 8.3 Phase 2 — Advanced DAG

| Task | Description | Deliverable |
|------|-------------|-------------|
| P2-01 | Schema-based inplace detection (`alias_info.is_write`) | `op_graph_tracer.py` |
| P2-02 | Version tracking with inplace increment | `op_graph_tracer.py` |
| P2-03 | View/alias detection and `register_alias()` | `op_graph_tracer.py` |
| P2-04 | Alias group version propagation | `op_graph_tracer.py` |
| P2-05 | kwargs mutation handling (e.g., `reshape_and_cache` mutates `kv_cache`) | `op_graph_tracer.py` |
| P2-06 | Tests: inplace ops, views, alias propagation, kwargs mutations | `tests/test_op_graph_tracer.py` |

### 8.4 Phase 3 — DES Module

| Task | Description | Deliverable |
|------|-------------|-------------|
| P3-01 | Create `des/` package structure | Directory + `__init__.py` |
| P3-02 | Implement `DESConfig`, `StreamType` | `des/config.py`, `des/stream.py` |
| P3-03 | Implement `OpHandle` | `des/op_handle.py` |
| P3-04 | Implement `StreamAssigner` | `des/stream_assigner.py` |
| P3-05 | Implement `DeviceStream(stime.Task)` with FIFO + dep wait | `des/stream.py` |
| P3-06 | Implement `CPUDispatcher(stime.Task)` with dispatch-ahead | `des/cpu_dispatcher.py` |
| P3-07 | Implement `TraceCollector` with chrome trace export | `des/trace_collector.py` |
| P3-08 | Implement `DESSimulator` orchestrator | `des/des_simulator.py` |
| P3-09 | Unit tests: mock DAG, verify overlap, verify trace format | `tests/test_des_simulator.py` |

### 8.5 Phase 4 — Integration

| Task | Description | Deliverable |
|------|-------------|-------------|
| P4-01 | Modify `Runtime.__init__` with new params | `runtime.py` |
| P4-02 | Hook `OpGraphTracer.record_op` in `__torch_dispatch__` | `runtime.py` |
| P4-03 | Build DAG + run DES in `__exit__` | `runtime.py` |
| P4-04 | Extend `get_trace_events()` with DES events | `runtime.py` |
| P4-05 | Add `operator_dag` and `des_result` properties | `runtime.py` |
| P4-06 | Implement `DAGAnalyzer` | `dag_analyzer.py` |
| P4-07 | Modify `UserInputConfig` + CLI args | `user_config.py` |
| P4-08 | Modify `ModelRunner` to pass flags and print DES results | `model_runner.py` |

### 8.6 Phase 5 — Validation

| Task | Description | Deliverable |
|------|-------------|-------------|
| P5-01 | Integration test: Runtime with `enable_dag_tracing=True` on synthetic ops | Test file |
| P5-02 | Integration test: Runtime with `enable_des_simulation=True` on synthetic ops | Test file |
| P5-03 | End-to-end: `text_generate Qwen/Qwen3-32B --enable-des-simulation --chrome-trace` | Trace file |
| P5-04 | Verify 3-stream chrome trace in chrome://tracing | Manual verification |
| P5-05 | Performance benchmark: measure overhead of DAG tracing + DES on 10K ops | Benchmark report |
| P5-06 | Run full test suite (`pytest tensor_cast/tests -n auto`) to verify no regressions | CI green |

---

## 9. Test Plan

### 9.1 Unit Tests — DAG Module

```python
# tensor_cast/tests/test_op_graph_tracer.py

class TestTensorIdentity:
    def test_equality_and_hashing(self): ...
    def test_different_versions_not_equal(self): ...
    def test_repr(self): ...

class TestOpGraphTracer:
    def test_basic_chain(self):
        """A -> B -> C: three ops forming a linear chain."""

    def test_diamond_dependency(self):
        """A produces T1; B and C both consume T1; D consumes outputs of B and C."""

    def test_tensor_id_reuse_prevented(self):
        """Verify tensor pool prevents GC-induced ID collisions."""

    def test_inplace_operation_versioning(self):
        """A.add_(B) should create edge from (A@v0, B@v0) to (A@v1)."""

    def test_view_alias_propagation(self):
        """A.view() creates alias; mutating A should increment both versions."""

    def test_kwargs_mutation(self):
        """reshape_and_cache(key, value, kv_cache=...) should track kv_cache mutation."""

    def test_model_input_detection(self):
        """Tensors with no producer should be identified as model inputs."""

    def test_model_output_detection(self):
        """Tensors with no consumer should be identified as model outputs."""

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

### 9.2 Unit Tests — DES Module

```python
# tensor_cast/tests/test_des_simulator.py

class TestStreamAssigner:
    def test_comm_ops_to_comm_stream(self): ...
    def test_compute_ops_to_compute_stream(self): ...
    def test_override(self): ...

class TestDESSimulator:
    def test_linear_chain_no_overlap(self):
        """Three sequential compute ops: total = sum of times."""

    def test_compute_comm_overlap(self):
        """Independent compute and comm ops should overlap."""

    def test_dependency_blocks_execution(self):
        """Comm op depending on compute op must wait."""

    def test_cpu_dispatch_ahead(self):
        """Verify CPU finishes dispatching before device finishes executing."""

    def test_chrome_trace_format(self):
        """Verify output has correct pid/tid/ph/ts/dur fields."""

    def test_overlap_metrics(self):
        """Verify DESResult overlap_time_s computation."""
```

### 9.3 Integration Tests

```python
class TestRuntimeDAGIntegration:
    def test_dag_tracing_enabled(self):
        """Runtime with enable_dag_tracing=True builds a DAG."""

    def test_des_simulation_enabled(self):
        """Runtime with enable_des_simulation=True produces DESResult."""

    def test_chrome_trace_with_des(self):
        """export_chrome_trace includes DES multi-stream events."""

    def test_des_disabled_by_default(self):
        """Default Runtime behavior is unchanged."""
```

---

## 10. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Tensor pool memory overhead too high | Low | Medium | TensorCast uses meta tensors (~128B each); add threshold warning at 100K tensors |
| Schema-based inplace/alias detection incomplete for custom ops | Medium | High | Fallback to ID comparison; document custom op requirements |
| Alias group propagation bugs | Medium | High | Extensive testing with view-heavy models; compare DAG edges with MemoryTracker |
| salabim env state leaks between simulations | Low | Medium | Always call `stime.SimulationEnv.clear()` in `finally` block |
| DES overhead for large models | Medium | Low | DES is opt-in; profiling on 10K ops target |
| Existing test suite regressions | Low | High | All new features are opt-in (default disabled); run full test suite |
| `unsatisfied_dep_count` race in salabim | Low | Medium | salabim is single-threaded cooperative; no true races. Validate with assertions |

---

## 11. Future Extensions

| Extension | Description | Estimated Effort |
|-----------|-------------|-----------------|
| **Dual-batch overlap** | Simulate interleaved prefill/decode batches on separate compute streams | Additional DeviceStream instances + batch-aware stream assignment |
| **Chunk prefill/decode** | Model chunked prefill with interleaved decode iterations | Extend CPUDispatcher with scheduling policies |
| **Multi-device pipeline** | Simulate pipeline parallelism across devices with inter-device communication | Multiple DESSimulator instances + inter-device Channel tasks |
| **AIC/AIV split streams** | Separate compute into Matrix (AIC/Cube) and Vector (AIV) streams | Additional StreamType values + finer op classification |
| **Intra/inter-node communication** | Separate communication streams for NVLink vs. network | Additional StreamType values + CommGrid-aware assignment |
| **Operator fusion in DES** | Fuse consecutive ops into single DES execution unit | FusionDetector + DAG rewriting before DES |
| **Dynamic scheduling** | Priority-based or work-stealing schedulers | Replace FIFO queue with priority queue in DeviceStream |
| **Memory simulation in DES** | Track memory watermark during DES execution | Combine MemoryTracker analysis with DES timeline |

---

*Document Version: 2.0*
*Last Updated: 2026-02-05*
