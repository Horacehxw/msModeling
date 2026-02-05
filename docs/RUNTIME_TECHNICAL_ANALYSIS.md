# TensorCast Runtime Module: Technical Analysis Report

## Executive Summary

This report provides a comprehensive technical analysis of the TensorCast Runtime module, focusing on:
1. **Operator DAG Generation** - Feasibility and implementation approaches for constructing data dependency graphs by tensor ID matching
2. **NPU Operator Mapping** - Strategies for matching captured operators to real NPU operations

The Runtime module uses PyTorch's `TorchDispatchMode` to intercept all PyTorch operations, capturing `OpInvokeInfo` objects that record operation metadata, input/output tensors, and performance characteristics.

---

## 1. Architecture Overview

### 1.1 Core Components

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Runtime (TorchDispatchMode)                      │
├──────────────────────────────────────────────────────────────────────────────┤
│  __torch_dispatch__()  ─────►  OpInvokeInfo  ─────►  RuntimeEvent            │
│         │                           │                      │                  │
│         ▼                           ▼                      ▼                  │
│  op_invoke_infos[]           PerformanceModel        event_list[]            │
│                                     │                                         │
│                                     ▼                                         │
│                              MemoryTracker                                    │
│                                     │                                         │
│                                     ▼                                         │
│                          Tensor Lifecycle Analysis                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Classes and Their Roles

| Class | File | Purpose |
|-------|------|---------|
| `Runtime` | `runtime.py:36-537` | TorchDispatchMode-based operation interceptor |
| `OpInvokeInfo` | `performance_model/__init__.py:17-167` | Operation metadata container |
| `MemoryTracker` | `performance_model/memory_tracker.py:28-302` | Tensor lifecycle tracking |
| `PerformanceModel` | `performance_model/__init__.py:169-227` | Abstract performance estimation |
| `AnalyticPerformanceModel` | `performance_model/analytic.py:96-114` | Roofline-based performance model |

### 1.3 Operation Flow

```python
# Entry point - tensor_cast/runtime.py:69-77
def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    kwargs = {} if kwargs is None else kwargs
    if not torch.compiler.is_compiling():
        out = func(*args, **kwargs)  # Execute on meta tensors
        op_invoke_info = OpInvokeInfo(func, args, kwargs, out)  # Capture metadata
        self.op_invoke_infos.append(op_invoke_info)
        return out
    else:
        return func(*args, **kwargs)  # Bypass during torch.compile
```

---

## 2. Operator DAG Generation Analysis

### 2.1 Current Data Dependency Tracking

The MemoryTracker already implements **implicit data dependency tracking** through tensor lifecycle analysis:

```python
# tensor_cast/performance_model/memory_tracker.py:19-25
@dataclasses.dataclass
class _TensorInfo:
    size_bytes: int
    def_op_idx: int = -1              # Producer operation index
    use_op_indices: List[int] = []     # Consumer operation indices
    use_op_indices_by_alias: List[int] = []  # Indirect uses via aliases
    last_use_op_idx: int = -1          # Final consumer index
```

**Key Insight**: The `def_op_idx` and `use_op_indices` fields already encode a producer-consumer relationship graph.

### 2.2 Feasibility Assessment

**Yes, generating an operator DAG by matching tensor IDs is feasible.** The existing infrastructure provides:

1. **Tensor ID Tracking**: Python's `id()` function uniquely identifies tensor objects
2. **Producer-Consumer Recording**: `_TensorInfo.def_op_idx` and `use_op_indices` already capture this
3. **Alias Handling**: The `alias_info` dictionary properly handles tensor views and slices

### 2.3 Implementation Approach for Explicit DAG

Here's how to construct an explicit operator DAG:

```python
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import torch

@dataclass
class OpNode:
    """Node in the operator DAG"""
    op_idx: int
    op_invoke_info: 'OpInvokeInfo'

    # Dependency edges
    predecessors: Set[int] = field(default_factory=set)  # Operations this depends on
    successors: Set[int] = field(default_factory=set)    # Operations depending on this

    # Tensor flow information
    input_tensor_ids: Set[int] = field(default_factory=set)
    output_tensor_ids: Set[int] = field(default_factory=set)

class OperatorDAG:
    """Explicit DAG representation of operator data dependencies"""

    def __init__(self):
        self.nodes: Dict[int, OpNode] = {}  # op_idx -> OpNode
        self.tensor_producers: Dict[int, int] = {}  # tensor_id -> producer op_idx
        self.tensor_consumers: Dict[int, List[int]] = {}  # tensor_id -> consumer op_indices
        self.model_inputs: Set[int] = set()  # tensor_ids with no producer
        self.model_outputs: Set[int] = set()  # tensor_ids with no consumer

    def build_from_runtime(self, runtime: 'Runtime'):
        """Construct DAG from Runtime's captured operations"""

        # Pass 1: Record all operations and their tensor I/O
        for op_idx, op_invoke_info in enumerate(runtime.op_invoke_infos):
            node = OpNode(op_idx=op_idx, op_invoke_info=op_invoke_info)

            # Extract input tensor IDs
            input_tensors = self._extract_tensors(op_invoke_info.args, op_invoke_info.kwargs)
            for tensor in input_tensors:
                tensor_id = id(tensor)
                node.input_tensor_ids.add(tensor_id)

                # Record consumer relationship
                if tensor_id not in self.tensor_consumers:
                    self.tensor_consumers[tensor_id] = []
                self.tensor_consumers[tensor_id].append(op_idx)

            # Extract output tensor IDs
            output_tensors = self._extract_tensors(op_invoke_info.out)
            for tensor in output_tensors:
                tensor_id = id(tensor)
                node.output_tensor_ids.add(tensor_id)

                # Record producer relationship (should be unique)
                if tensor_id not in self.tensor_producers:
                    self.tensor_producers[tensor_id] = op_idx

            self.nodes[op_idx] = node

        # Pass 2: Build dependency edges by matching tensor IDs
        for op_idx, node in self.nodes.items():
            for input_tensor_id in node.input_tensor_ids:
                if input_tensor_id in self.tensor_producers:
                    producer_idx = self.tensor_producers[input_tensor_id]
                    # Create edge: producer -> current
                    node.predecessors.add(producer_idx)
                    self.nodes[producer_idx].successors.add(op_idx)
                else:
                    # No producer = model input
                    self.model_inputs.add(input_tensor_id)

        # Pass 3: Identify model outputs (tensors with no consumers after all ops)
        for tensor_id, producer_idx in self.tensor_producers.items():
            if tensor_id not in self.tensor_consumers or not self.tensor_consumers[tensor_id]:
                self.model_outputs.add(tensor_id)

    def _extract_tensors(self, *data) -> List[torch.Tensor]:
        """Recursively extract all tensors from nested data structures"""
        tensors = []
        for item in data:
            if isinstance(item, torch.Tensor):
                tensors.append(item)
            elif isinstance(item, (list, tuple)):
                tensors.extend(self._extract_tensors(*item))
            elif isinstance(item, dict):
                tensors.extend(self._extract_tensors(*item.values()))
        return tensors

    def get_critical_path(self) -> List[int]:
        """Find the critical path (longest path) through the DAG"""
        # Topological sort + dynamic programming
        visited = set()
        topo_order = []

        def dfs(node_idx):
            if node_idx in visited:
                return
            visited.add(node_idx)
            for succ in self.nodes[node_idx].successors:
                dfs(succ)
            topo_order.append(node_idx)

        for idx in self.nodes:
            dfs(idx)
        topo_order.reverse()

        # DP for longest path
        dist = {idx: 0 for idx in self.nodes}
        parent = {idx: None for idx in self.nodes}

        for idx in topo_order:
            for succ in self.nodes[idx].successors:
                if dist[idx] + 1 > dist[succ]:
                    dist[succ] = dist[idx] + 1
                    parent[succ] = idx

        # Reconstruct path
        end_node = max(dist, key=dist.get)
        path = []
        while end_node is not None:
            path.append(end_node)
            end_node = parent[end_node]
        return list(reversed(path))

    def to_graphviz(self) -> str:
        """Export DAG to Graphviz DOT format"""
        lines = ["digraph OperatorDAG {", "  rankdir=TB;"]

        for idx, node in self.nodes.items():
            op_name = str(node.op_invoke_info.func).split('.')[-1]
            lines.append(f'  op_{idx} [label="{idx}: {op_name}"];')

        for idx, node in self.nodes.items():
            for succ in node.successors:
                lines.append(f'  op_{idx} -> op_{succ};')

        lines.append("}")
        return "\n".join(lines)
```

### 2.4 Tensor ID Matching Challenges (Original Assessment)

| Challenge | Description | Solution |
|-----------|-------------|----------|
| **Tensor Aliasing** | Views share storage with parent tensor | Use `alias_info` dict from MemoryTracker |
| **In-place Operations** | Modify tensor without changing ID | Track via schema's `mutates_args` |
| **Tensor Reuse** | Same ID may be reused after deallocation | Process operations sequentially, not globally |
| **Meta Tensors** | Different behavior than real tensors | Already handled by TensorCast design |

---

## 2.5 Proposed OpGraphTracer Design (Updated)

### 2.5.1 Problem Analysis

Two critical issues must be addressed for robust tensor identity tracking:

#### Problem 1: Tensor ID Reuse

Python `id()` returns the memory address of an object, which is only guaranteed unique during the object's **lifetime**. When a tensor's reference count drops to zero, CPython immediately frees it, and a subsequently created tensor may reuse the same memory address — resulting in the same `id()`. In our tracing scenario, intermediate tensors produced by one operator may be GC'd before the next operator executes, causing **false dependency edges** in the DAG.

#### Problem 2: Inplace Operations

Inplace operators (e.g., `add_()`, `relu_()`) modify a tensor in-place and return the **same object** — the `id()` does not change. A naive id-based DAG builder would see identical input and output ids, failing to model the "read-then-write" data dependency.

### 2.5.2 Proposed Solution

**Solution for Problem 1 (Tensor ID Reuse):**
The `OpGraphTracer` maintains a **strong reference pool** (`tensor_pool: List[Tensor]`) that holds every intermediate tensor encountered during `__torch_dispatch__`. This prevents any tensor from being garbage-collected throughout the entire tracing session, guaranteeing that `id()` remains globally unique. Each tensor is further assigned a monotonically increasing `uid` via a counter, mapped through `_id_to_uid: Dict[int, int]`.

**Solution for Problem 2 (Inplace Operations):**
Detect inplace ops by checking `input_id == output_id` after dispatch. When detected, use a **versioned tensor identity** `(uid, version)` to distinguish pre- and post-mutation states. For inplace ops, the input is recorded as `(uid, v)` and the output as `(uid, v+1)`, where `version` is tracked via an internal `_version_map: Dict[int, int]` (incremented on each inplace mutation). Non-inplace ops produce outputs with fresh `uid` and `version=0`. This ensures every DAG edge refers to a unique tensor state.

### 2.5.3 Proposed Implementation

```python
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import torch

@dataclass(frozen=True)
class TensorIdentity:
    """Versioned tensor identity for DAG edge tracking"""
    uid: int
    version: int = 0

    def __repr__(self):
        return f"T{self.uid}v{self.version}"


@dataclass
class OpNode:
    """Node in the operator DAG with versioned tensor tracking"""
    op_idx: int
    op_invoke_info: 'OpInvokeInfo'

    # Versioned tensor identities
    input_identities: Set[TensorIdentity] = field(default_factory=set)
    output_identities: Set[TensorIdentity] = field(default_factory=set)

    # Dependency edges (op indices)
    predecessors: Set[int] = field(default_factory=set)
    successors: Set[int] = field(default_factory=set)


class OpGraphTracer:
    """
    Operator graph tracer with robust tensor identity management.

    Solves two key problems:
    1. Tensor ID reuse after GC - via strong reference pool
    2. Inplace operations - via versioned tensor identities
    """

    def __init__(self):
        # Strong reference pool to prevent GC
        self.tensor_pool: List[torch.Tensor] = []

        # Tensor identity management
        self._uid_counter: int = 0
        self._id_to_uid: Dict[int, int] = {}
        self._version_map: Dict[int, int] = {}  # uid -> current version

        # DAG structure
        self.nodes: Dict[int, OpNode] = {}
        self.identity_producers: Dict[TensorIdentity, int] = {}  # identity -> producer op_idx

        # Model I/O
        self.model_inputs: Set[TensorIdentity] = set()
        self.model_outputs: Set[TensorIdentity] = set()

    def _get_or_assign_uid(self, tensor: torch.Tensor) -> int:
        """Get existing UID or assign new one, adding to reference pool"""
        tensor_id = id(tensor)
        if tensor_id not in self._id_to_uid:
            uid = self._uid_counter
            self._uid_counter += 1
            self._id_to_uid[tensor_id] = uid
            self._version_map[uid] = 0
            self.tensor_pool.append(tensor)  # Prevent GC
        return self._id_to_uid[tensor_id]

    def _get_current_identity(self, tensor: torch.Tensor) -> TensorIdentity:
        """Get current versioned identity for a tensor"""
        uid = self._get_or_assign_uid(tensor)
        version = self._version_map[uid]
        return TensorIdentity(uid=uid, version=version)

    def _increment_version(self, tensor: torch.Tensor) -> TensorIdentity:
        """Increment version for inplace mutation, return new identity"""
        uid = self._get_or_assign_uid(tensor)
        self._version_map[uid] += 1
        return TensorIdentity(uid=uid, version=self._version_map[uid])

    def record_op(self, op_invoke_info: 'OpInvokeInfo'):
        """Record an operation and build DAG edges"""
        op_idx = len(self.nodes)
        node = OpNode(op_idx=op_idx, op_invoke_info=op_invoke_info)

        # Extract input tensors and their identities
        input_tensors = self._extract_tensors(op_invoke_info.args, op_invoke_info.kwargs)
        for tensor in input_tensors:
            identity = self._get_current_identity(tensor)
            node.input_identities.add(identity)

            # Build predecessor edge
            if identity in self.identity_producers:
                pred_idx = self.identity_producers[identity]
                node.predecessors.add(pred_idx)
                self.nodes[pred_idx].successors.add(op_idx)
            else:
                self.model_inputs.add(identity)

        # Extract output tensors
        output_tensors = self._extract_tensors(op_invoke_info.out)
        for out_tensor in output_tensors:
            out_id = id(out_tensor)

            # Check for inplace operation
            is_inplace = any(id(t) == out_id for t in input_tensors)

            if is_inplace:
                # Inplace: increment version to create new identity
                identity = self._increment_version(out_tensor)
            else:
                # Non-inplace: fresh identity with version 0
                identity = self._get_current_identity(out_tensor)

            node.output_identities.add(identity)
            self.identity_producers[identity] = op_idx

        self.nodes[op_idx] = node

    def _extract_tensors(self, *data) -> List[torch.Tensor]:
        """Recursively extract tensors from nested structures"""
        tensors = []
        for item in data:
            if isinstance(item, torch.Tensor):
                tensors.append(item)
            elif isinstance(item, (list, tuple)):
                tensors.extend(self._extract_tensors(*item))
            elif isinstance(item, dict):
                tensors.extend(self._extract_tensors(*item.values()))
        return tensors

    def finalize(self):
        """Finalize DAG and identify model outputs"""
        all_consumed = set()
        for node in self.nodes.values():
            all_consumed.update(node.input_identities)

        for identity, _ in self.identity_producers.items():
            if identity not in all_consumed:
                self.model_outputs.add(identity)

    def clear(self):
        """Release all references (call after tracing complete)"""
        self.tensor_pool.clear()
        self._id_to_uid.clear()
        self._version_map.clear()
```

---

## 2.6 Design Evaluation

### 2.6.1 Strengths

| Aspect | Assessment |
|--------|------------|
| **GC Prevention** | Strong reference pool correctly prevents tensor ID reuse |
| **Inplace Handling** | Versioned identity elegantly models read-then-write semantics |
| **Monotonic UIDs** | Counter-based UIDs provide stable, debuggable identifiers |
| **Clean Separation** | Clear distinction between `id()` (Python) and `uid` (logical) |

### 2.6.2 Status: **Needs Clarification**

The design is fundamentally sound but has several gaps that need addressing before implementation.

---

## 2.7 Potential Problems and Concerns

### Problem 1: Memory Consumption for Large Models

**Issue**: The `tensor_pool` holds strong references to ALL tensors during tracing. For large models with many layers, this could consume significant memory.

**Mitigation**: TensorCast uses **meta tensors** which don't allocate storage — only shape/dtype metadata. This should keep memory overhead minimal. However, verify this assumption:

```python
# Meta tensor memory is negligible
>>> t = torch.empty(1024, 1024, device='meta')
>>> import sys; sys.getsizeof(t)
128  # Only metadata, not 4MB of data
```

**Recommendation**: Add a memory guard that warns if `len(tensor_pool)` exceeds a threshold (e.g., 100K tensors).

---

### Problem 2: View/Alias Propagation on Inplace Mutations

**Issue**: When tensor `A` has a view `B`, and `A` is mutated in-place:
- `A`'s version increments to `(uid_A, v+1)`
- But `B` (which shares storage) is **also affected** — its logical state has changed
- The current design doesn't propagate version increments to aliases

**Example**:
```python
A = torch.randn(4, 4)        # uid=0, v=0
B = A.view(2, 8)             # uid=1, v=0 (shares storage with A)
A.add_(1)                    # A becomes (0, v=1), but B's data also changed!
C = B + 1                    # Should depend on the post-mutation state
```

**Recommendation**: Integrate with existing `MemoryTracker._handle_aliasing()` to track alias relationships. When any tensor in an alias group is mutated, increment versions for the entire group:

```python
# Proposed extension
self._alias_groups: Dict[int, Set[int]] = {}  # uid -> set of aliased uids

def _increment_version_with_aliases(self, tensor: torch.Tensor) -> TensorIdentity:
    uid = self._get_or_assign_uid(tensor)
    self._version_map[uid] += 1

    # Also increment all aliases
    if uid in self._alias_groups:
        for alias_uid in self._alias_groups[uid]:
            self._version_map[alias_uid] += 1

    return TensorIdentity(uid=uid, version=self._version_map[uid])
```

---

### Problem 3: Inplace Detection Reliability

**Issue**: The design uses `input_id == output_id` to detect inplace ops. This works for most cases but may miss:

1. **Ops with `out=` parameter**: `torch.add(a, b, out=c)` — `c` is mutated but not an input
2. **Multi-output ops**: Some ops return tuples where only some outputs are inplace

**Recommendation**: Use PyTorch schema as the primary detection method, with ID comparison as fallback:

```python
def _is_inplace_output(self, op_invoke_info, output_tensor, input_tensors) -> bool:
    """Detect inplace using schema + ID comparison"""
    # Method 1: Schema-based (preferred)
    func = op_invoke_info.func
    if hasattr(func, '_schema'):
        for i, ret in enumerate(func._schema.returns):
            if ret.alias_info is not None:
                # Output aliases an input = inplace or view
                return True

    # Method 2: ID comparison (fallback)
    out_id = id(output_tensor)
    return any(id(t) == out_id for t in input_tensors)
```

---

### Problem 4: Kwargs Mutation Detection

**Issue**: The current `_extract_tensors` treats args and kwargs uniformly, but inplace detection only checks `args`:

```python
# Missing case: tensor passed via kwargs
torch.ops.aten.add_.Tensor(input, other)  # Detected
torch.ops.tensor_cast.reshape_and_cache(key, value, kv_cache=cache, ...)  # kv_cache mutated!
```

**Recommendation**: Use the `mutates_args` parameter from schema to identify which arguments (including kwargs) are mutated:

```python
def _get_mutated_arg_names(self, op_invoke_info) -> Set[str]:
    """Get names of arguments that are mutated by this op"""
    func = op_invoke_info.func
    mutated = set()
    if hasattr(func, '_schema'):
        for arg in func._schema.arguments:
            if arg.alias_info and arg.alias_info.is_write:
                mutated.add(arg.name)
    return mutated
```

---

### Problem 5: Thread Safety

**Issue**: If multiple threads call into `__torch_dispatch__` concurrently:
- `_uid_counter` increment is not atomic
- `tensor_pool.append()` is not thread-safe
- `_version_map` updates can race

**Assessment**: TensorCast currently appears to be single-threaded during tracing. However, if multi-threaded tracing is ever needed:

**Recommendation**: Add thread safety as an optional mode:

```python
import threading

class OpGraphTracer:
    def __init__(self, thread_safe: bool = False):
        self._lock = threading.Lock() if thread_safe else None

    def _get_or_assign_uid(self, tensor: torch.Tensor) -> int:
        if self._lock:
            with self._lock:
                return self._get_or_assign_uid_unsafe(tensor)
        return self._get_or_assign_uid_unsafe(tensor)
```

---

### Problem 6: Cleanup Timing

**Issue**: When should `tensor_pool` be cleared? The design shows a `clear()` method but doesn't specify when to call it.

**Recommendation**: Integrate with `Runtime.__exit__`:

```python
# In Runtime.__exit__
def __exit__(self, exc_type, exc_val, exc_tb):
    super().__exit__(exc_type, exc_val, exc_tb)
    self.repeat_op_invoke_infos()
    self.replay_op_invoke_infos()
    if self.op_graph_tracer:
        self.op_graph_tracer.finalize()
        # Keep graph data, but release tensor references
        self.op_graph_tracer.tensor_pool.clear()
```

---

### Problem 7: Version Overflow (Minor)

**Issue**: For extremely long traces with many inplace ops on the same tensor, `version` could theoretically overflow.

**Assessment**: Python integers have arbitrary precision, so this is not a practical concern. No action needed.

---

## 2.8 Summary: Implementation Readiness

| Aspect | Status | Action Required |
|--------|--------|-----------------|
| Core tensor pool mechanism | **Ready** | None |
| Versioned identity concept | **Ready** | None |
| Inplace detection | **Needs Enhancement** | Add schema-based detection |
| View/alias handling | **Needs Enhancement** | Integrate with alias tracking |
| Multi-tensor mutations (kwargs) | **Needs Enhancement** | Use `mutates_args` from schema |
| Memory management | **Ready** | Meta tensors keep it bounded |
| Thread safety | **Deferred** | Not needed for current use case |
| Integration with Runtime | **Needs Design** | Define integration points |

### Recommended Next Steps

1. **Enhance inplace detection** with schema-based approach
2. **Add alias group tracking** for view propagation
3. **Define Runtime integration** — where in the lifecycle to build/finalize DAG
4. **Add unit tests** for edge cases: inplace, views, kwargs mutations

### 2.9 Integration with Existing Code

The DAG builder can leverage `MemoryTracker._handle_aliasing()` for proper alias tracking:

```python
# tensor_cast/performance_model/memory_tracker.py:84-146
def _handle_aliasing(self, op_invoke_info: OpInvokeInfo):
    """Handles tensor aliasing by inspecting PyTorch schema metadata"""
    # Uses op_invoke_info.func._schema.returns[i].alias_info
    # to determine if output aliases input
```

---

## 3. NPU Operator Mapping Analysis

### 3.1 Current Operator Abstraction Levels

TensorCast operates at three abstraction levels:

```
┌─────────────────────────────────────────────────────────────────┐
│  Level 3: High-Level Composite Ops (TensorCast Custom Ops)      │
│  ─────────────────────────────────────────────────────────────  │
│  tensor_cast.attention, tensor_cast.multihead_latent_attention  │
│  tensor_cast.grouped_matmul, tensor_cast.static_quant_linear    │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│  Level 2: PyTorch ATen Ops (Standard PyTorch Operations)        │
│  ─────────────────────────────────────────────────────────────  │
│  aten.mm, aten.bmm, aten.addmm, aten.convolution                │
│  aten.embedding, aten.index_select, aten.softmax                │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│  Level 1: NPU/Hardware Primitives (Target Level)                │
│  ─────────────────────────────────────────────────────────────  │
│  Cube (MMA), Vector, Scalar operations on actual hardware       │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Custom Operation Registration

TensorCast uses `torch.library.custom_op` for registering custom operations:

```python
# tensor_cast/utils.py:18-32
def register_tensor_cast_op(name, mutates_args=(), **kwargs):
    def decorator(func):
        custom_op = torch.library.custom_op(
            f"tensor_cast::{name}", mutates_args=mutates_args, **kwargs
        )(func)
        custom_op.register_fake(func)  # Same impl for fake/meta tensors
        return func
    return decorator
```

### 3.3 Strategies for NPU Operator Matching

#### Strategy 1: Operation Decomposition

**Concept**: Break down high-level TensorCast ops into NPU-equivalent primitives.

```python
class NPUOperatorMapper:
    """Maps TensorCast operations to NPU operator sequences"""

    # Mapping table: TensorCast op -> NPU op sequence
    DECOMPOSITION_MAP = {
        torch.ops.tensor_cast.attention.default: [
            # Q @ K^T -> NPU Cube MatMul
            ("npu_matmul", {"transposed_b": True}),
            # Softmax -> NPU Vector Softmax
            ("npu_softmax", {}),
            # Scores @ V -> NPU Cube MatMul
            ("npu_matmul", {}),
        ],
        torch.ops.tensor_cast.static_quant_linear.default: [
            # Dequantize weight if needed
            ("npu_dequant", {"scheme": "per_channel"}),
            # Quantized matmul -> NPU INT8 Cube
            ("npu_quant_matmul", {"input_dtype": "int8", "weight_dtype": "int8"}),
            # Requantize output
            ("npu_requant", {}),
        ],
    }

    @classmethod
    def decompose(cls, op_invoke_info: OpInvokeInfo) -> List[Tuple[str, dict]]:
        """Decompose TensorCast op into NPU primitive sequence"""
        if op_invoke_info.func in cls.DECOMPOSITION_MAP:
            return cls.DECOMPOSITION_MAP[op_invoke_info.func]

        # Fallback: map ATen ops directly
        return cls._map_aten_op(op_invoke_info)

    @classmethod
    def _map_aten_op(cls, op_invoke_info: OpInvokeInfo) -> List[Tuple[str, dict]]:
        """Map standard ATen ops to NPU equivalents"""
        func_name = str(op_invoke_info.func)

        ATEN_TO_NPU = {
            "aten.mm.default": [("npu_matmul", {})],
            "aten.bmm.default": [("npu_batch_matmul", {})],
            "aten.addmm.default": [("npu_matmul", {}), ("npu_add", {})],
            "aten.convolution.default": [("npu_conv2d", {})],
            "aten.softmax.int": [("npu_softmax", {})],
            "aten.layer_norm.default": [("npu_layer_norm", {})],
        }

        return ATEN_TO_NPU.get(func_name, [("npu_generic", {})])
```

#### Strategy 2: Performance Property Matching

**Concept**: Use `PerformanceProperties` to match operations by computational characteristics.

```python
# Current property extraction - tensor_cast/performance_model/__init__.py:88-121
def get_memory_access_properties(self, ...) -> PerformanceProperties:
    """Get memory read/write properties"""
    # Uses schema metadata to determine access patterns
    for i, arg in enumerate(itertools.chain(self.args, self.kwargs.values())):
        if args_schema[i].is_out:
            memory_write_bytes += access_bytes
        elif args_schema[i].is_write:
            memory_readwrite_bytes += access_bytes
        else:
            memory_read_bytes += access_bytes
```

**Enhancement for NPU Matching**:

```python
@dataclass
class NPUOperatorCharacteristics:
    """Characteristics that map to NPU operator selection"""
    compute_pattern: str  # "matmul", "elementwise", "reduction", "attention"
    dtype_compute: torch.dtype
    dtype_io: torch.dtype
    is_quantized: bool
    tensor_shapes: Tuple[Tuple[int, ...], ...]  # (input_shapes, output_shapes)
    memory_pattern: str  # "dense", "sparse", "blocked", "paged"

    @classmethod
    def from_op_invoke_info(cls, op_info: OpInvokeInfo) -> 'NPUOperatorCharacteristics':
        """Extract NPU-relevant characteristics from OpInvokeInfo"""
        perf_props = op_info.get_perf_properties()

        # Determine compute pattern
        if perf_props.compute_ops:
            dtypes = list(perf_props.compute_ops.keys())
            compute_ops = perf_props.compute_ops[dtypes[0]]
            if compute_ops.mma_ops > 0:
                pattern = "matmul"
            else:
                pattern = "elementwise"
        else:
            pattern = "memory_only"

        # Extract shapes
        input_shapes = tuple(
            tuple(t.shape) for t in cls._get_tensors(op_info.args)
        )
        output_shapes = tuple(
            tuple(t.shape) for t in cls._get_tensors(op_info.out)
        )

        return cls(
            compute_pattern=pattern,
            dtype_compute=dtypes[0] if dtypes else torch.float32,
            dtype_io=cls._get_tensors(op_info.args)[0].dtype if op_info.args else torch.float32,
            is_quantized="quant" in str(op_info.func).lower(),
            tensor_shapes=(input_shapes, output_shapes),
            memory_pattern=cls._infer_memory_pattern(op_info),
        )
```

#### Strategy 3: Device Profile Integration

**Concept**: Leverage `DeviceProfile` to constrain operator selection to hardware-supported operations.

```python
# tensor_cast/device.py provides hardware specifications
class DeviceProfile:
    DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2, torch.int8, ...]

    mma_ops: Dict[torch.dtype, float]  # MMA ops/second by dtype
    gp_ops: Dict[torch.dtype, float]   # General purpose ops/second
    memory_bandwidth_bytes_ps: float
```

**NPU-Specific Device Profile Extension**:

```python
@dataclass
class NPUDeviceProfile(DeviceProfile):
    """Extended device profile with NPU-specific operator support"""

    # Supported NPU operator types
    supported_cube_ops: Set[str] = field(default_factory=lambda: {
        "matmul", "batch_matmul", "conv2d", "conv3d"
    })

    supported_vector_ops: Set[str] = field(default_factory=lambda: {
        "add", "mul", "softmax", "layernorm", "relu", "gelu"
    })

    # Quantization support
    quant_schemes: Set[str] = field(default_factory=lambda: {
        "W8A8_STATIC", "W8A8_DYNAMIC", "W4A8_STATIC", "FP8"
    })

    # Operator fusion rules
    fusion_patterns: List[Tuple[List[str], str]] = field(default_factory=lambda: [
        (["matmul", "add"], "matmul_bias"),
        (["matmul", "relu"], "matmul_relu"),
        (["matmul", "gelu"], "matmul_gelu"),
        (["layernorm", "linear"], "layernorm_linear"),
    ])

    def can_execute(self, npu_op: str, dtype: torch.dtype) -> bool:
        """Check if NPU can execute given operation with dtype"""
        all_ops = self.supported_cube_ops | self.supported_vector_ops
        dtype_supported = dtype in self.mma_ops or dtype in self.gp_ops
        return npu_op in all_ops and dtype_supported
```

### 3.4 Recommended Implementation Approach

For accurate NPU operator mapping, implement a **three-stage pipeline**:

```
┌──────────────────┐    ┌───────────────────┐    ┌─────────────────┐
│  Stage 1: Capture│───►│  Stage 2: Analyze │───►│  Stage 3: Map   │
│                  │    │                   │    │                 │
│  TorchDispatch   │    │  DAG Construction │    │  NPU Selection  │
│  OpInvokeInfo    │    │  Fusion Detection │    │  Cost Modeling  │
└──────────────────┘    └───────────────────┘    └─────────────────┘
```

**Stage 1**: Already implemented via `Runtime.__torch_dispatch__`

**Stage 2**: Extend with explicit DAG + fusion pattern detection

```python
class FusionDetector:
    """Detect operator fusion opportunities in the DAG"""

    FUSION_PATTERNS = {
        # Pattern: (sequence of ops) -> fused_op_name
        ("aten.mm.default", "aten.add.Tensor"): "mm_bias",
        ("aten.mm.default", "aten.relu.default"): "mm_relu",
        ("aten.layer_norm.default", "aten.mm.default"): "layernorm_mm",
    }

    def detect_fusions(self, dag: OperatorDAG) -> List[FusionCandidate]:
        """Find all fusible operator sequences"""
        candidates = []

        for node_idx, node in dag.nodes.items():
            # Check if node starts a fusion pattern
            for pattern, fused_name in self.FUSION_PATTERNS.items():
                if self._matches_pattern_start(node, pattern[0]):
                    sequence = self._try_extend_pattern(dag, node_idx, pattern)
                    if sequence:
                        candidates.append(FusionCandidate(
                            node_indices=sequence,
                            fused_op_name=fused_name,
                            pattern=pattern
                        ))

        return candidates
```

**Stage 3**: Map to NPU operators with cost-aware selection

```python
class NPUOperatorSelector:
    """Select optimal NPU operators based on hardware profile"""

    def __init__(self, device_profile: NPUDeviceProfile):
        self.profile = device_profile

    def select_operators(
        self,
        dag: OperatorDAG,
        fusions: List[FusionCandidate]
    ) -> List[NPUOperator]:
        """Select NPU operators for the execution plan"""
        npu_ops = []
        fused_nodes = set()

        # Apply fusions first
        for fusion in fusions:
            if self._is_fusion_profitable(fusion):
                npu_ops.append(self._create_fused_op(fusion))
                fused_nodes.update(fusion.node_indices)

        # Map remaining nodes
        for node_idx, node in dag.nodes.items():
            if node_idx not in fused_nodes:
                npu_op = self._select_single_op(node)
                npu_ops.append(npu_op)

        return npu_ops

    def _is_fusion_profitable(self, fusion: FusionCandidate) -> bool:
        """Determine if fusion reduces execution time"""
        # Estimate separate execution cost
        separate_cost = sum(
            self._estimate_op_cost(node_idx)
            for node_idx in fusion.node_indices
        )

        # Estimate fused execution cost
        fused_cost = self._estimate_fused_cost(fusion)

        return fused_cost < separate_cost * 0.9  # 10% improvement threshold
```

---

## 4. Key Implementation Recommendations

### 4.1 For Operator DAG Generation

1. **Extend `MemoryTracker`** with explicit DAG construction
2. **Add `OperatorDAG` class** as shown in Section 2.3
3. **Preserve tensor ID mapping** throughout the pipeline
4. **Handle aliasing** using existing `_handle_aliasing()` logic
5. **Support DAG serialization** for debugging (Graphviz, JSON)

### 4.2 For NPU Operator Matching

1. **Create operator decomposition tables** mapping TensorCast ops to NPU primitives
2. **Extend `DeviceProfile`** with NPU-specific operator support information
3. **Implement fusion detection** based on DAG structure
4. **Add cost modeling** for NPU operator selection
5. **Support multiple NPU backends** via pluggable mapper implementations

### 4.3 Code Changes Required

| File | Changes |
|------|---------|
| `runtime.py` | Add DAG construction hook in `__exit__` |
| `memory_tracker.py` | Export `_TensorInfo` as public API |
| `performance_model/__init__.py` | Add `OperatorDAG` class |
| `device.py` | Extend `DeviceProfile` for NPU specifics |
| New: `npu_mapper.py` | NPU operator mapping implementation |
| New: `dag_builder.py` | Explicit DAG construction logic |

---

## 5. References

### PyTorch Documentation
- [TorchDispatchMode Guide](https://dev-discuss.pytorch.org/t/torchdispatchmode-for-debugging-testing-and-more/717)
- [What is __torch_dispatch__?](https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557)
- [PyTorch Dispatcher Walkthrough](https://github.com/pytorch/pytorch/wiki/PyTorch-dispatcher-walkthrough)
- [torch.compile Tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [Edward Yang's Dispatcher Blog](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)

### Key Source Files
- `tensor_cast/runtime.py:36-537` - Runtime class implementation
- `tensor_cast/performance_model/__init__.py:17-167` - OpInvokeInfo class
- `tensor_cast/performance_model/memory_tracker.py:28-302` - MemoryTracker class
- `tensor_cast/performance_model/analytic.py:96-226` - AnalyticPerformanceModel
- `tensor_cast/ops/` - Custom operation definitions

---

## Appendix A: Existing Tensor Tracking Code Reference

### A.1 Tensor Recording (memory_tracker.py:148-186)

```python
def record_op_invocation(self, op_invoke_info: OpInvokeInfo):
    op_idx = len(self.op_invoke_infos)
    self.op_invoke_infos.append(op_invoke_info)

    # Identify input tensors and record usage
    input_tensors = self._extract_tensors(op_invoke_info.args) + \
                    self._extract_tensors(op_invoke_info.kwargs)
    for tensor in input_tensors:
        tensor_id = id(tensor)
        if tensor_id not in self.tensor_infos:
            self.tensor_infos[tensor_id] = _TensorInfo(size_bytes=bytes_of_tensor(tensor))
        self.tensor_infos[tensor_id].use_op_indices.append(op_idx)

    # Identify output tensors and record definition site
    output_tensors = self._extract_tensors(op_invoke_info.out)
    for tensor in output_tensors:
        tensor_id = id(tensor)
        if tensor_id not in self.tensor_infos:
            self.tensor_infos[tensor_id] = _TensorInfo(size_bytes=bytes_of_tensor(tensor))
            self.tensor_infos[tensor_id].def_op_idx = op_idx

    self._handle_aliasing(op_invoke_info)
```

### A.2 Alias Handling (memory_tracker.py:84-146)

```python
def _handle_aliasing(self, op_invoke_info: OpInvokeInfo):
    # Inspects PyTorch schema metadata to track tensor aliasing
    for i, output_schema in enumerate(op_invoke_info.func._schema.returns):
        if output_schema.alias_info is None:
            continue
        output_alias_set = output_schema.alias_info.before_set
        for j, input_schema in enumerate(op_invoke_info.func._schema.arguments):
            if input_schema.alias_info is None:
                continue
            if output_alias_set & input_schema.alias_info.before_set:
                # Found aliasing relationship
                self.alias_info[output_id] = input_id
                break
```

---

## Appendix B: Custom Operation Examples

### B.1 Attention Operation (ops/attention.py:18-45)

```python
@register_tensor_cast_op("attention")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    query_start_loc: Optional[torch.Tensor],
    seq_lens: Optional[torch.Tensor],
    query_lens: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.empty_like(query).contiguous()
```

### B.2 Communication Operation (ops/communication.py:20-22)

```python
@register_tensor_cast_op("all_reduce")
def _(x: torch.Tensor, rank: int, rank_group: List[int]) -> torch.Tensor:
    return torch.empty_like(x)
```

---

*Report generated: 2026-02-02*
*TensorCast Version: feat/op_dag_des branch*
