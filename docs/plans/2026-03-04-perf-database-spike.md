# Perf Database End-to-End Spike Implementation Plan (v3)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a minimal end-to-end spike: `--performance-model profiling` runs Qwen3-32B with pre-collected CSV data, outputs alignment report showing matched vs unmatched ops.

**Architecture:** Per design doc v1.2, `EmpiricalPerformanceModel` accepts a generic `DataSource` with `lookup(OpInvokeInfo) -> Optional[QueryResult]`. `ProfilingDataSource` **internally** handles all mapping (op_mapping.yaml), feature extraction (shapes/dtypes from meta tensors), FRACTAL_NZ shape restoration, and CSV lookup. Unmatched ops fall back to `AnalyticPerformanceModel`.

**Tech Stack:** Python 3.10+, pandas, PyYAML, torch (meta tensors only), pytest

**Design Doc:** `docs/perf_database/OPERATOR_PERF_DATABASE_DESIGN_zh_v1.2.md`

---

## Design Doc Alignment Checklist

| Design Doc Section | Requirement | Status in This Spike |
|---|---|---|
| §4.1 DataSource ABC | `lookup(OpInvokeInfo) -> Optional[QueryResult]` with `QuerySource` enum | **IMPLEMENT** — rewrite existing `query(kernel_type, features)` |
| §4.2 ProfilingDataSource | Internal op_mapping + FRACTAL_NZ + dtype match + 4 dispatch paths | **IMPLEMENT** `_lookup_compute` only; comm/attention/composite return None |
| §4.3 EmpiricalPerformanceModel | `(device_profile, data_source, fallback_model)` | **IMPLEMENT** |
| §4.4 InterpolatingDataSource | Wrapper with interpolation | **DEFER** — stub passthrough OK |
| §4.5 op_mapping.yaml | Config-driven mapping, no per-op Python code | **IMPLEMENT** — copy example to production |
| §4.6 Query Example | MatMulV2 with FRACTAL_NZ weight restoration | **IMPLEMENT** — this is the spike's core scenario |
| §4.7 Communication data | topology_tier, message_bytes | **DEFER** — return None, fallback to analytic |
| §4.8 FusedAttention special | batch_size, avg_seq_len index | **DEFER** — return None, fallback to analytic |
| §4.9 FRACTAL_NZ | `fractal_nz_to_nd()` generic restoration | **IMPLEMENT** |
| §5.1 Runtime integration | Runtime unchanged, accepts PerformanceModel | **VERIFY** — already works |
| §5.2 CLI interface | `--performance-model`, `--perf-database` | **IMPLEMENT** |
| §5.3 Data flow | profiling → EmpiricalPM(ProfilingDS) → Runtime | **IMPLEMENT** |

---

## Current State (已有代码)

| Component | Status | Key Gap vs Design Doc |
|-----------|--------|----------------------|
| `DataSource` ABC | EXISTS but wrong interface | `query(kernel_type, features)` → should be `lookup(OpInvokeInfo)` |
| `QueryResult` | EXISTS but wrong fields | `(kernel_type, latency_us, source: str)` → should be `(latency_us, confidence, source: QuerySource, details)` |
| `ProfilingDataSource` | EXISTS but minimal | Plain string match only — no op_mapping, no FRACTAL_NZ, no dtype |
| `InterpolatingDataSource` | STUB (passthrough) | OK for spike |
| CSV data (32 kernel types) | EXISTS at `v0.14.0/` | Missing `op_mapping.yaml` in production path |
| `op_mapping_example.yaml` | EXISTS (comprehensive) | In `docs/perf_database/examples/`, needs copy to data dir |
| `EmpiricalPerformanceModel` | STUB (uses OpBenchmark) | Needs full rewrite to use DataSource |
| CLI `--performance-model` | MISSING | |
| `ModelRunner` integration | HARDCODED analytic | |
| Tests | NONE | |

---

## Task Dependency Graph

```
Task 1 (DataSource + QueryResult rewrite) ───┐
                                               ├── Task 3 (ProfilingDataSource rewrite) ──┐
Task 2 (op_mapping.yaml → production path) ───┘                                           │
                                                                                           ├── Task 5 (CLI + ModelRunner)
Task 4 (EmpiricalPerformanceModel rewrite) ───────────────────────────────────────────────┘
                                                                                           │
                                                                                           └── Task 6 (E2E spike run)
```

Tasks 1 & 2 parallel. Task 3 depends on both. Task 4 depends on Task 1. Task 5 depends on 3+4. Task 6 is final integration.

---

### Task 1: Rewrite DataSource ABC + QueryResult to match design doc §4.1

**Files:**
- Modify: `tensor_cast/performance_model/perf_database/data_source.py`
- Modify: `tensor_cast/performance_model/perf_database/__init__.py`
- Create: `tests/perf_database/__init__.py`
- Create: `tests/perf_database/test_data_source.py`

**Reference:** Design doc §4.1

**Step 1: Write the failing test**

```python
# tests/perf_database/__init__.py
# (empty)
```

```python
# tests/perf_database/test_data_source.py
import pytest
from tensor_cast.performance_model.perf_database.data_source import (
    DataSource, QueryResult, QuerySource,
)


def test_query_source_enum():
    assert QuerySource.MEASURED.name == "MEASURED"
    assert QuerySource.INTERPOLATED.name == "INTERPOLATED"
    assert QuerySource.EXTRAPOLATED.name == "EXTRAPOLATED"


def test_query_result_creation():
    r = QueryResult(latency_us=45.3, confidence=1.0, source=QuerySource.MEASURED)
    assert r.latency_us == 45.3
    assert r.confidence == 1.0
    assert r.source == QuerySource.MEASURED
    assert r.details == {}


def test_query_result_with_details():
    r = QueryResult(
        latency_us=100.0, confidence=0.8,
        source=QuerySource.INTERPOLATED,
        details={"kernel_type": "MatMulV2", "csv_row": 42},
    )
    assert r.details["kernel_type"] == "MatMulV2"


def test_data_source_is_abstract():
    with pytest.raises(TypeError):
        DataSource()


def test_data_source_subclass_must_implement_lookup():
    class BadSource(DataSource):
        pass
    with pytest.raises(TypeError):
        BadSource()


def test_data_source_store_raises_by_default():
    class ReadOnlySource(DataSource):
        def lookup(self, op_invoke_info):
            return None

    source = ReadOnlySource()
    with pytest.raises(NotImplementedError, match="read-only"):
        source.store(None, None)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/perf_database/test_data_source.py -v`
Expected: FAIL — old interface doesn't have `QuerySource`, `lookup`, `confidence`, `details`

**Step 3: Rewrite data_source.py per design doc §4.1**

```python
# tensor_cast/performance_model/perf_database/data_source.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..op_invoke_info import OpInvokeInfo


class QuerySource(Enum):
    MEASURED = auto()       # Exact match (confidence: 1.0)
    INTERPOLATED = auto()   # Interpolated estimate (confidence: 0.7-0.95)
    EXTRAPOLATED = auto()   # Extrapolated estimate (confidence: 0.3-0.6)


@dataclass
class QueryResult:
    latency_us: float
    confidence: float
    source: QuerySource
    details: Dict[str, Any] = field(default_factory=dict)


class DataSource(ABC):
    """Abstract base class for performance data sources.
    TensorCast queries via OpInvokeInfo only, unaware of underlying data format.
    (Design doc §4.1)"""

    @abstractmethod
    def lookup(self, op_invoke_info: "OpInvokeInfo") -> Optional[QueryResult]:
        """Query operator performance from OpInvokeInfo.
        Internally handles mapping, shape extraction, format conversion.
        Returns None if not found."""
        ...

    def store(self, op_invoke_info: "OpInvokeInfo", result: QueryResult) -> None:
        """Store performance data (optional). Default: read-only."""
        raise NotImplementedError("This DataSource is read-only")
```

**Step 4: Update `__init__.py` exports**

```python
# tensor_cast/performance_model/perf_database/__init__.py
from .data_source import DataSource, QueryResult, QuerySource
from .interpolating_data_source import InterpolatingDataSource
from .profiling_data_source import ProfilingDataSource

__all__ = [
    "DataSource",
    "InterpolatingDataSource",
    "ProfilingDataSource",
    "QueryResult",
    "QuerySource",
]
```

**Step 5: Update InterpolatingDataSource to use new interface**

```python
# tensor_cast/performance_model/perf_database/interpolating_data_source.py
from typing import Optional, TYPE_CHECKING

from .data_source import DataSource, QueryResult

if TYPE_CHECKING:
    from ..op_invoke_info import OpInvokeInfo


class InterpolatingDataSource(DataSource):
    """Wrapper datasource for future interpolation fallback. (Design doc §4.4)"""

    def __init__(self, base: DataSource):
        self.base = base

    def lookup(self, op_invoke_info: "OpInvokeInfo") -> Optional[QueryResult]:
        result = self.base.lookup(op_invoke_info)
        if result is not None:
            return result
        # TODO: interpolation logic (Phase 2)
        return None
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/perf_database/test_data_source.py -v`
Expected: All 6 tests PASS

**Step 7: Commit**

```bash
git add tensor_cast/performance_model/perf_database/data_source.py \
        tensor_cast/performance_model/perf_database/__init__.py \
        tensor_cast/performance_model/perf_database/interpolating_data_source.py \
        tests/perf_database/__init__.py \
        tests/perf_database/test_data_source.py
git commit -m "refactor(perf-db): rewrite DataSource ABC to lookup(OpInvokeInfo) per design doc v1.2 §4.1"
```

---

### Task 2: Copy op_mapping.yaml to production data path

**Files:**
- Copy: `docs/perf_database/examples/op_mapping_example.yaml` → `tensor_cast/performance_model/perf_database/data/atlas_800_a3_752t_128g_die/vllm_ascend/v0.14.0/op_mapping.yaml`

**Step 1: Copy the file**

```bash
cp docs/perf_database/examples/op_mapping_example.yaml \
   tensor_cast/performance_model/perf_database/data/atlas_800_a3_752t_128g_die/vllm_ascend/v0.14.0/op_mapping.yaml
```

**Step 2: Update version field**

The example yaml has `version: "0.13.0"` but data dir is `v0.14.0/`. Edit the copied file: change `version: "0.13.0"` to `version: "0.14.0"`.

**Step 3: Verify kernel type CSV coverage**

Write a quick script to cross-check that kernel_types referenced in op_mapping.yaml have matching `.csv` files:

```bash
python3 -c "
import yaml
from pathlib import Path
data_dir = Path('tensor_cast/performance_model/perf_database/data/atlas_800_a3_752t_128g_die/vllm_ascend/v0.14.0')
with open(data_dir / 'op_mapping.yaml') as f:
    cfg = yaml.safe_load(f)
csvs = {p.stem for p in data_dir.glob('*.csv')}
for func, m in cfg.get('operator_mappings', {}).items():
    kt = m.get('kernel_type')
    if kt and kt not in csvs and not m.get('composite'):
        print(f'MISSING CSV: {kt} (from {func})')
    elif kt and kt in csvs:
        print(f'OK: {kt}.csv')
"
```

Expected: Some kernel types like `QuantBatchMatmulV3`, `GroupedMatmul`, `DequantSwigluQuant` will be MISSING (DSV3 decode-only, current data is Qwen3 prefill). That's expected for the spike.

**Step 4: Commit**

```bash
git add tensor_cast/performance_model/perf_database/data/atlas_800_a3_752t_128g_die/vllm_ascend/v0.14.0/op_mapping.yaml
git commit -m "data(perf-db): copy op_mapping.yaml to production data path for spike"
```

---

### Task 3: Rewrite ProfilingDataSource per design doc §4.2

**Files:**
- Modify: `tensor_cast/performance_model/perf_database/profiling_data_source.py`
- Create: `tests/perf_database/test_profiling_data_source.py`

**Depends on:** Task 1 (new DataSource interface), Task 2 (op_mapping.yaml)

**Reference:** Design doc §4.2, §4.6 (query example), §4.9 (FRACTAL_NZ)

This is the core spike work. ProfilingDataSource must:
1. Load `op_mapping.yaml` at init
2. In `lookup()`: resolve `OpInvokeInfo.func` → mapping entry
3. Dispatch: `_lookup_compute()` for standard ops, return None for composite/comm/attention_special
4. In `_lookup_compute()`: extract tensor shapes/dtypes from `OpInvokeInfo.args`, apply `fractal_nz_to_nd()` to FRACTAL_NZ weights, match against CSV rows

**Step 1: Write the failing test**

```python
# tests/perf_database/test_profiling_data_source.py
import torch
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from tensor_cast.performance_model.perf_database.profiling_data_source import (
    ProfilingDataSource, fractal_nz_to_nd, DTYPE_MAP,
)
from tensor_cast.performance_model.perf_database.data_source import QuerySource


# --- fractal_nz_to_nd tests (design doc §4.9, Appendix B) ---

def test_fractal_nz_to_nd_bf16():
    # BF16 MatMulV2: [320,48,16,16] -> K=320*16=5120, N=48*16=768
    assert fractal_nz_to_nd((320, 48, 16, 16)) == (5120, 768)


def test_fractal_nz_to_nd_int8():
    # INT8 QuantBatchMatmulV3: [48,448,16,32] -> K=448*32=14336, N=48*16=768
    assert fractal_nz_to_nd((48, 448, 16, 32)) == (14336, 768)


def test_fractal_nz_to_nd_batched():
    # GroupedMatmul INT8: [E, N/32, K/16, 16, 32]
    assert fractal_nz_to_nd((64, 48, 448, 16, 32)) == (64, 14336, 768)


def test_dtype_map():
    assert DTYPE_MAP[torch.bfloat16] == "DT_BF16"
    assert DTYPE_MAP[torch.float16] == "DT_BF16"
    assert DTYPE_MAP[torch.int8] == "INT8"
    assert DTYPE_MAP[torch.float32] == "FLOAT"


# --- ProfilingDataSource tests ---

SPIKE_OP_MAPPING_YAML = """
version: "0.14.0"
device: TEST_DEVICE

operator_mappings:
  "aten.mm.default":
    kernel_type: MatMulV2
  "aten.bmm.default":
    kernel_type: TransposeBatchMatMul
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
  "tensor_cast.multihead_latent_attention.default":
    composite: true
    sub_kernels: [TransposeBatchMatMul, FusedInferAttentionScore]
  "tensor_cast.all_reduce.default":
    kernel_type: hcom_allReduce_
    category: communication
"""

# CSV with FRACTAL_NZ weight, matching design doc §4.6 query example
SPIKE_MATMUL_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"136,5120;320,48,16,16","DT_BF16;DT_BF16","ND;FRACTAL_NZ","136,768","DT_BF16","ND",45.3
"1,5120;320,48,16,16","DT_BF16;DT_BF16","ND;FRACTAL_NZ","1,768","DT_BF16","ND",12.1
"""


@pytest.fixture
def spike_data_dir(tmp_path):
    data_dir = tmp_path / "spike"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(SPIKE_OP_MAPPING_YAML)
    (data_dir / "MatMulV2.csv").write_text(SPIKE_MATMUL_CSV.strip())
    return data_dir


def _make_op_info(func, input_tensors, output_tensors=None):
    """Create a mock OpInvokeInfo with real torch.ops func and meta tensors."""
    mock = MagicMock()
    mock.func = func
    mock.args = tuple(input_tensors)
    mock.kwargs = {}
    if output_tensors:
        mock.out = output_tensors[0] if len(output_tensors) == 1 else tuple(output_tensors)
    else:
        mock.out = None
    return mock


def test_exact_match_with_fractal_nz(spike_data_dir):
    """Design doc §4.6: aten.mm(A[136,5120], B[5120,768]) matches
    CSV row with FRACTAL_NZ weight [320,48,16,16] after restoration."""
    ds = ProfilingDataSource(spike_data_dir)
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(136, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, 768, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 45.3) < 0.01
    assert result.confidence == 1.0
    assert result.source == QuerySource.MEASURED
    assert result.details.get("kernel_type") == "MatMulV2"


def test_miss_wrong_shape(spike_data_dir):
    ds = ProfilingDataSource(spike_data_dir)
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(256, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, 768, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None


def test_miss_unmapped_op(spike_data_dir):
    ds = ProfilingDataSource(spike_data_dir)
    op = _make_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(136, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(136, 5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None


def test_composite_returns_none(spike_data_dir):
    """Composite ops (MLA) return None in spike, fallback to analytic."""
    ds = ProfilingDataSource(spike_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.multihead_latent_attention.default,
        [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16)],
    )
    result = ds.lookup(op)
    assert result is None


def test_communication_returns_none(spike_data_dir):
    """Communication ops return None in spike, fallback to analytic."""
    ds = ProfilingDataSource(spike_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16), 0, [0, 1]],
    )
    result = ds.lookup(op)
    assert result is None


def test_attention_special_returns_none(spike_data_dir):
    """attention_special ops return None in spike, fallback to analytic."""
    ds = ProfilingDataSource(spike_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.attention.default,
        [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16)],
    )
    result = ds.lookup(op)
    assert result is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/perf_database/test_profiling_data_source.py -v`
Expected: FAIL — old ProfilingDataSource doesn't have `lookup()`, `fractal_nz_to_nd`, `DTYPE_MAP`

**Step 3: Rewrite profiling_data_source.py per design doc §4.2**

```python
# tensor_cast/performance_model/perf_database/profiling_data_source.py
"""ProfilingDataSource: CSV-backed data source with op_mapping + FRACTAL_NZ.

Design doc reference: §4.2 (ProfilingDataSource), §4.9 (FRACTAL_NZ)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd
import torch
import yaml

from .data_source import DataSource, QueryResult, QuerySource

if TYPE_CHECKING:
    from ..op_invoke_info import OpInvokeInfo

logger = logging.getLogger(__name__)

# torch dtype → Profiling dtype string (design doc §4.2)
DTYPE_MAP = {
    torch.bfloat16: "DT_BF16",
    torch.float16: "DT_BF16",  # FP16 treated as BF16 on Ascend
    torch.int8: "INT8",
    torch.int32: "INT32",
    torch.int64: "INT64",
    torch.float32: "FLOAT",
    torch.bool: "BOOL",
}


def fractal_nz_to_nd(nz_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Restore FRACTAL_NZ tiled shape to ND shape.
    [..., H, W, block_h, block_w] → [..., H*block_w, W*block_h]

    Design doc §4.9, Appendix B:
    - BF16: [K/16, N/16, 16, 16] → (K, N)
    - INT8: [N/32, K/16, 16, 32] → (K, N) after H*block_w, W*block_h
    - Batched: [E, N/32, K/16, 16, 32] → (E, K, N)
    """
    *batch, H, W, block_h, block_w = nz_shape
    return (*batch, H * block_w, W * block_h)


def _normalize_func_name(func) -> str:
    """Convert torch op to string matching op_mapping.yaml keys.
    e.g. torch.ops.aten.mm.default -> 'aten.mm.default'
         torch.ops.tensor_cast.attention.default -> 'tensor_cast.attention.default'"""
    s = str(func)
    for prefix in ("torch.ops.",):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


def _parse_shape_str(s: str) -> List[Tuple[int, ...]]:
    """Parse CSV shape string -> list of tuples.
    e.g. '"136,5120;320,48,16,16"' -> [(136,5120), (320,48,16,16)]"""
    s = s.strip().strip('"')
    shapes = []
    for part in s.split(";"):
        part = part.strip()
        if part:
            shapes.append(tuple(int(x) for x in part.split(",")))
    return shapes


def _parse_str_list(s: str) -> List[str]:
    """Parse 'A;B;C' -> ['A', 'B', 'C']"""
    s = s.strip().strip('"')
    return [x.strip() for x in s.split(";") if x.strip()]


class ProfilingDataSource(DataSource):
    """CSV-backed data source with op_mapping.yaml + FRACTAL_NZ.

    Design doc §4.2: internally handles all mapping, shape extraction,
    format conversion. The caller (EmpiricalPerformanceModel) only calls
    lookup(OpInvokeInfo).

    Init args:
        data_dir: path containing op_mapping.yaml + {KernelType}.csv files
        comm_grid: optional CommGrid for topology_tier resolution (Phase 2)
    """

    def __init__(self, data_dir: str | Path, comm_grid=None):
        self.data_dir = Path(data_dir)
        self.comm_grid = comm_grid
        self._op_mapping = self._load_op_mapping()
        self._csv_cache: Dict[str, Optional[pd.DataFrame]] = {}

    def _load_op_mapping(self) -> dict:
        yaml_path = self.data_dir / "op_mapping.yaml"
        if not yaml_path.exists():
            logger.warning("op_mapping.yaml not found at %s", yaml_path)
            return {}
        with open(yaml_path) as f:
            return yaml.safe_load(f)

    def _load_csv(self, kernel_type: str) -> Optional[pd.DataFrame]:
        if kernel_type in self._csv_cache:
            return self._csv_cache[kernel_type]
        csv_path = self.data_dir / f"{kernel_type}.csv"
        if not csv_path.exists():
            logger.debug("CSV not found: %s", csv_path)
            self._csv_cache[kernel_type] = None
            return None
        df = pd.read_csv(csv_path)
        self._csv_cache[kernel_type] = df
        return df

    # ---- Main lookup (design doc §4.2 dispatch logic) ----

    def lookup(self, op_invoke_info: "OpInvokeInfo") -> Optional[QueryResult]:
        """Query perf data for an op.

        Dispatch logic (design doc §4.2):
          func_name → op_mapping.yaml
            ├─ not found → return None
            ├─ composite == true → return None (spike: fallback to analytic)
            ├─ category == "communication" → return None (spike: fallback)
            ├─ query_mode == "attention_special" → return None (spike: fallback)
            └─ default → _lookup_compute()
        """
        func_str = _normalize_func_name(op_invoke_info.func)
        mappings = self._op_mapping.get("operator_mappings", {})
        mapping = mappings.get(func_str)
        if mapping is None:
            return None

        # Spike: skip composite, communication, attention_special
        if mapping.get("composite"):
            return None
        if mapping.get("category") == "communication":
            return None
        if mapping.get("query_mode") == "attention_special":
            return None

        return self._lookup_compute(op_invoke_info, mapping)

    # ---- Compute op lookup (design doc §4.2 _lookup_compute) ----

    def _lookup_compute(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
    ) -> Optional[QueryResult]:
        kernel_type = mapping["kernel_type"]
        df = self._load_csv(kernel_type)
        if df is None:
            return None

        # Extract tensor shapes and dtypes from OpInvokeInfo.args
        tc_inputs = self._extract_tensor_inputs(op_invoke_info)

        # Match against CSV rows
        for _, row in df.iterrows():
            if self._inputs_match(tc_inputs, row):
                # Use "Average Duration(us)" if available, else "Duration(us)"
                latency_col = (
                    "Average Duration(us)"
                    if "Average Duration(us)" in df.columns
                    else "Duration(us)"
                )
                return QueryResult(
                    latency_us=float(row[latency_col]),
                    confidence=1.0,
                    source=QuerySource.MEASURED,
                    details={"kernel_type": kernel_type},
                )
        return None

    def _extract_tensor_inputs(
        self, op_invoke_info: "OpInvokeInfo"
    ) -> List[Tuple[Tuple[int, ...], torch.dtype]]:
        """Extract (shape, dtype) for each tensor arg."""
        inputs = []
        for arg in op_invoke_info.args:
            if isinstance(arg, torch.Tensor):
                inputs.append((tuple(arg.shape), arg.dtype))
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    if isinstance(item, torch.Tensor):
                        inputs.append((tuple(item.shape), item.dtype))
        return inputs

    def _inputs_match(
        self,
        tc_inputs: List[Tuple[Tuple[int, ...], torch.dtype]],
        csv_row: pd.Series,
    ) -> bool:
        """Match TensorCast input shapes/dtypes against a CSV row.
        Handles FRACTAL_NZ restoration (design doc §4.9)."""
        csv_shapes = _parse_shape_str(str(csv_row.get("Input Shapes", "")))
        csv_dtypes = _parse_str_list(str(csv_row.get("Input Data Types", "")))
        csv_formats = _parse_str_list(str(csv_row.get("Input Formats", "")))

        if len(tc_inputs) != len(csv_shapes):
            return False

        for i, (tc_shape, tc_dtype) in enumerate(tc_inputs):
            # Check dtype
            expected_dtype = DTYPE_MAP.get(tc_dtype)
            if expected_dtype is None or i >= len(csv_dtypes):
                return False
            if expected_dtype != csv_dtypes[i]:
                return False

            # Get CSV shape, restore FRACTAL_NZ if needed
            csv_shape = csv_shapes[i]
            if i < len(csv_formats) and csv_formats[i] == "FRACTAL_NZ":
                csv_shape = fractal_nz_to_nd(csv_shape)

            if tc_shape != csv_shape:
                return False

        return True
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/perf_database/test_profiling_data_source.py -v`
Expected: All 10 tests PASS (4 unit + 6 integration)

**Step 5: Commit**

```bash
git add tensor_cast/performance_model/perf_database/profiling_data_source.py \
        tests/perf_database/test_profiling_data_source.py
git commit -m "refactor(perf-db): rewrite ProfilingDataSource with lookup(OpInvokeInfo), FRACTAL_NZ, dtype matching per design doc §4.2"
```

---

### Task 4: Rewrite EmpiricalPerformanceModel per design doc §4.3

**Files:**
- Modify: `tensor_cast/performance_model/empirical.py`
- Create: `tests/perf_database/test_empirical.py`

**Depends on:** Task 1 (new DataSource interface)

**Reference:** Design doc §4.3

**Step 1: Write the failing test**

```python
# tests/perf_database/test_empirical.py
import torch
import pytest
from unittest.mock import MagicMock

from tensor_cast.performance_model.base import PerformanceModel
from tensor_cast.performance_model.empirical import EmpiricalPerformanceModel
from tensor_cast.performance_model.perf_database.data_source import (
    DataSource, QueryResult, QuerySource,
)


class HitDataSource(DataSource):
    def lookup(self, op_invoke_info):
        return QueryResult(
            latency_us=45.3, confidence=1.0, source=QuerySource.MEASURED,
            details={"kernel_type": "MatMulV2"},
        )


class MissDataSource(DataSource):
    def lookup(self, op_invoke_info):
        return None


def _make_mock_op_invoke_info():
    mock = MagicMock()
    mock.func = torch.ops.aten.mm.default
    mock.args = (
        torch.empty(136, 5120, device="meta"),
        torch.empty(5120, 768, device="meta"),
    )
    return mock


def _make_mock_device_profile():
    mock = MagicMock()
    mock.name = "TEST_DEVICE"
    return mock


def test_empirical_uses_datasource_when_hit():
    """Design doc §4.3: data_source.lookup() hit → use measured latency."""
    device = _make_mock_device_profile()
    model = EmpiricalPerformanceModel(device, data_source=HitDataSource())
    result = model.process_op(_make_mock_op_invoke_info())
    assert abs(result.execution_time_s - 45.3e-6) < 1e-12
    assert result.statistics.get("source") == "MEASURED"
    assert result.statistics.get("kernel_type") == "MatMulV2"


def test_empirical_falls_back_when_miss():
    """Design doc §4.3: data_source.lookup() miss → fallback_model.process_op()."""
    device = _make_mock_device_profile()
    fallback = MagicMock(spec=PerformanceModel)
    fallback.process_op.return_value = PerformanceModel.Result(execution_time_s=100e-6)

    model = EmpiricalPerformanceModel(
        device, data_source=MissDataSource(), fallback_model=fallback,
    )
    result = model.process_op(_make_mock_op_invoke_info())
    fallback.process_op.assert_called_once()
    assert abs(result.execution_time_s - 100e-6) < 1e-12


def test_empirical_model_name():
    device = _make_mock_device_profile()
    model = EmpiricalPerformanceModel(device, data_source=MissDataSource())
    assert model.name == "empirical"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/perf_database/test_empirical.py -v`
Expected: FAIL — old EmpiricalPerformanceModel doesn't accept `data_source`

**Step 3: Rewrite empirical.py per design doc §4.3**

```python
# tensor_cast/performance_model/empirical.py
"""EmpiricalPerformanceModel: measurement-based performance model.

Design doc reference: §4.3
"""

import logging
from typing import Optional

from overrides import override

from ..device import DeviceProfile
from .base import PerformanceModel
from .op_invoke_info import OpInvokeInfo
from .perf_database.data_source import DataSource

logger = logging.getLogger(__name__)


class EmpiricalPerformanceModel(PerformanceModel):
    """Performance model based on measured data from a DataSource.

    Design doc §4.3: accepts DataSource instance, process_op() queries
    data source first, falls back to fallback_model on miss.

    Usage (design doc §5.1):
        data_source = ProfilingDataSource(data_dir, comm_grid=...)
        pm = EmpiricalPerformanceModel(device_profile, data_source)
    """

    def __init__(
        self,
        device_profile: DeviceProfile,
        data_source: DataSource,
        fallback_model: Optional[PerformanceModel] = None,
    ):
        super().__init__("empirical", device_profile)
        self.data_source = data_source
        self._fallback_model = fallback_model
        self._stats = {"hit": 0, "miss": 0}

    @property
    def fallback_model(self) -> PerformanceModel:
        if self._fallback_model is None:
            from .analytic import AnalyticPerformanceModel
            self._fallback_model = AnalyticPerformanceModel(self.device_profile)
        return self._fallback_model

    @override
    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        result = self.data_source.lookup(op_invoke_info)
        if result is not None:
            self._stats["hit"] += 1
            return PerformanceModel.Result(
                execution_time_s=result.latency_us * 1e-6,
                statistics={
                    "source": result.source.name,
                    "confidence": result.confidence,
                    **result.details,
                },
            )
        self._stats["miss"] += 1
        return self.fallback_model.process_op(op_invoke_info)

    def get_stats(self) -> dict:
        total = self._stats["hit"] + self._stats["miss"]
        return {
            **self._stats,
            "total": total,
            "hit_rate": self._stats["hit"] / total if total > 0 else 0,
        }

    def log_stats(self):
        stats = self.get_stats()
        logger.info(
            "EmpiricalPerformanceModel: %d/%d ops matched (%.1f%%)",
            stats["hit"], stats["total"], stats["hit_rate"] * 100,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/perf_database/test_empirical.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add tensor_cast/performance_model/empirical.py \
        tests/perf_database/test_empirical.py
git commit -m "refactor(perf-model): rewrite EmpiricalPerformanceModel with DataSource per design doc §4.3"
```

---

### Task 5: CLI + ModelRunner integration per design doc §5.2

**Files:**
- Modify: `tensor_cast/scripts/text_generate.py:265` (add 2 argparse args before `args = parser.parse_args()`)
- Modify: `tensor_cast/core/user_config.py:68` (add 2 fields to dataclass)
- Modify: `tensor_cast/core/model_runner.py:59-61` (replace hardcoded analytic model)

**Depends on:** Task 3 (ProfilingDataSource), Task 4 (EmpiricalPerformanceModel)

**Reference:** Design doc §5.2 (CLI), §5.3 (data flow)

**Step 1: Add fields to UserInputConfig**

In `tensor_cast/core/user_config.py`, add after line 68 (`image_width: Optional[int] = None`):

```python
    performance_model: str = "analytic"
    perf_database: Optional[str] = None
```

**Step 2: Add CLI args to text_generate.py**

Add before `args = parser.parse_args()` (line 265):

```python
    # Performance model selection (design doc §5.2)
    parser.add_argument(
        "--performance-model",
        choices=["analytic", "profiling"],
        default="analytic",
        help="Performance model type: analytic (roofline) or profiling (CSV database)",
    )
    parser.add_argument(
        "--perf-database",
        type=str,
        default=None,
        help="Path to performance database directory containing op_mapping.yaml + CSV files "
             "(required for --performance-model profiling)",
    )
```

Note: Design doc §5.2 lists `"empirical"` as a third choice (for JIT benchmark mode). We omit it for the spike since `OpBenchmark` requires a real device.

**Step 3: Modify ModelRunner to use configurable performance model**

In `tensor_cast/core/model_runner.py`, replace lines 59-61 with:

```python
        logger.info("Initializing performance model")
        if getattr(user_input, "performance_model", "analytic") == "profiling":
            perf_db_path = getattr(user_input, "perf_database", None)
            if perf_db_path is None:
                raise ValueError(
                    "--perf-database is required when using --performance-model profiling"
                )
            from ..performance_model.empirical import EmpiricalPerformanceModel
            from ..performance_model.perf_database import ProfilingDataSource
            from pathlib import Path

            db_path = Path(perf_db_path)
            data_source = ProfilingDataSource(
                db_path, comm_grid=self.device_profile.comm_grid
            )
            self.perf_model = EmpiricalPerformanceModel(
                self.device_profile, data_source=data_source
            )
        else:
            self.perf_model = AnalyticPerformanceModel(self.device_profile)
        logger.debug("Performance model initialized: %s", self.perf_model)
```

**Step 4: Verify analytic mode still works (no regressions)**

```bash
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 2 --query-length 3500 --device TEST_DEVICE
```

Expected: Same output as before.

**Step 5: Commit**

```bash
git add tensor_cast/scripts/text_generate.py \
        tensor_cast/core/user_config.py \
        tensor_cast/core/model_runner.py
git commit -m "feat(cli): add --performance-model profiling and --perf-database per design doc §5.2"
```

---

### Task 6: End-to-end spike run + alignment report

**Depends on:** Tasks 1-5 all complete

**Step 1: Run all tests**

```bash
pytest tests/perf_database/ -v
```

Expected: All tests pass.

**Step 2: Run the spike with profiling mode**

```bash
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 2 --query-length 3500 \
  --device ATLAS_800_A3_752T_128G_DIE \
  --performance-model profiling \
  --perf-database tensor_cast/performance_model/perf_database/data/atlas_800_a3_752t_128g_die/vllm_ascend/v0.14.0/ \
  --log-level debug 2>&1 | tee /tmp/spike_output.log
```

**Step 3: Debug and fix issues iteratively**

Expect issues with:
- Shape format mismatches (meta tensor shapes vs CSV quoting)
- `_normalize_func_name()` edge cases for some torch.ops
- Some ops may have non-tensor args that confuse `_extract_tensor_inputs()`

Fix iteratively until the run completes.

**Step 4: Collect stats and run comparison with analytic mode**

```bash
# Add stats logging at end of model_runner run_inference
# Then compare profiling vs analytic:
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 2 --query-length 3500 \
  --device ATLAS_800_A3_752T_128G_DIE \
  2>&1 | tee /tmp/analytic_output.log
```

**Step 5: Write spike alignment report**

Create: `docs/perf_database/reports/spike_alignment_report.md`

```markdown
# Spike Alignment Report

## Summary
- Total unique op types dispatched: X
- Matched by ProfilingDataSource: Y (Z%)
- Fell back to AnalyticPerformanceModel: W
  - Unmapped ops (not in op_mapping.yaml): A
  - Mapped but no CSV data file: B
  - Mapped + CSV exists but shape mismatch: C
  - Composite/communication/attention_special (deferred): D

## Matched Ops
| Kernel Type | Match Count | Avg Latency (μs) |
|---|---|---|

## Unmatched Ops Analysis
| Op func | Reason | op_mapping entry? | CSV exists? |
|---|---|---|---|

## FRACTAL_NZ Matching Observations
- Formats seen:
- Restoration success rate:

## Findings & Next Steps
- ...
```

**Step 6: Commit**

```bash
git add docs/perf_database/reports/
git commit -m "docs: add spike alignment report"
```

---

## 需要团队协助的事项

以下事项咱俩不方便搞，需要有 NPU 机器的人帮忙：

1. **跑 DSV3 Decode Profiling 数据** — 当前 CSV 只有 Qwen3 Prefill 数据。DSV3 Decode 的高频算子（QuantBatchMatmulV3, GroupedMatmul, TransposeBatchMatMul 等）完全没有数据
2. **HCCL Test 通信数据** — Phase 2 需要在多卡环境跑 `hccl_test`，生成 `hccl/{cann_version}/` 下的通信 CSV
3. **FusedAttention Microbenchmark** — 需要真机调用 `torch_npu.npu_fused_infer_attention_score` 遍历 (batch_size, avg_seq_len) 网格
4. **验证 FRACTAL_NZ 还原** — 真机 Profiling 输出上验证还原逻辑对 INT8/batched shapes 是否正确

---

## Execution Estimate

| Task | Effort | Notes |
|------|--------|-------|
| Task 1: DataSource rewrite | 10 min | Claude Code writes, you review |
| Task 2: Copy op_mapping.yaml | 5 min | Claude Code |
| Task 3: ProfilingDataSource rewrite | 30-40 min | Core work, needs careful FRACTAL_NZ testing |
| Task 4: EmpiricalPerformanceModel | 15 min | Claude Code writes, you review |
| Task 5: CLI + ModelRunner | 15 min | Claude Code |
| Task 6: E2E spike + debug | 30-60 min | 一起 debug shape mismatches |
| **Total** | **~2-3 hours** | |
