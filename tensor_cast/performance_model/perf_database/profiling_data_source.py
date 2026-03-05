"""ProfilingDataSource: CSV-backed data source with op_mapping + FRACTAL_NZ.

Design doc reference: S4.2 (ProfilingDataSource), S4.9 (FRACTAL_NZ)
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

# torch dtype -> Profiling dtype string (design doc S4.2)
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
    [..., H, W, block_h, block_w] -> [..., H*block_w, W*block_h]

    Design doc S4.9, Appendix B:
    - BF16: [K/16, N/16, 16, 16] -> (K, N)
    - INT8: [N/32, K/16, 16, 32] -> (K, N) after H*block_w, W*block_h
    - Batched: [E, N/32, K/16, 16, 32] -> (E, K, N)
    """
    *batch, H, W, block_h, block_w = nz_shape
    return (*batch, H * block_w, W * block_h)


def _normalize_func_name(func) -> str:
    """Convert torch op to string matching op_mapping.yaml keys.
    e.g. torch.ops.aten.mm.default -> 'aten.mm.default'
         torch.ops.tensor_cast.attention.default -> 'tensor_cast.attention.default'
    """
    s = str(func)
    return s.removeprefix("torch.ops.")


def _parse_shape_str(s: str) -> List[Tuple[int, ...]]:
    """Parse CSV shape string -> list of tuples.
    e.g. '"136,5120;320,48,16,16"' -> [(136,5120), (320,48,16,16)]
    """
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


# Matmul kernel types where ND weight may be stored as (N,K)
# while TC's aten.mm receives (K,N) after F.linear transpose.
# FRACTAL_NZ weights restore to (K,N) directly — no transpose needed.
_MATMUL_KERNELS = frozenset({"MatMulV2", "MatMul", "TransposeBatchMatMul"})

# SwiGlu kernel types: TC dispatches 2 inputs (gate, up) as separate tensors,
# but profiling CSVs store 1 concatenated input along last dim.
_SWIGLU_KERNELS = frozenset({"SwiGlu"})

# Common NPU tile alignment sizes (Da Vinci Cube unit)
# BF16: 16x16, INT8: 16x32
_BLOCK_SIZES = (16, 32, 64)


def _strip_batch_dim(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Strip leading batch dim=1 from TC shapes.
    TC keeps explicit batch: (1, seq, dim). Profiling flattens: (seq, dim).
    Only strip if leading dim is exactly 1.
    """
    if len(shape) > 1 and shape[0] == 1:
        return shape[1:]
    return shape


def _is_block_padded(tc_dim: int, csv_dim: int) -> bool:
    """Check if tc_dim is a block-padded version of csv_dim.
    TC pads sequence dims to NPU tile alignment; profiling stores unpadded shapes.
    """
    if tc_dim <= csv_dim:
        return False
    for bs in _BLOCK_SIZES:
        padded = ((csv_dim + bs - 1) // bs) * bs
        if tc_dim == padded:
            return True
    return False


class ProfilingDataSource(DataSource):
    """CSV-backed data source with op_mapping.yaml + FRACTAL_NZ.

    Design doc S4.2: internally handles all mapping, shape extraction,
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

    # ---- Main lookup (design doc S4.2 dispatch logic) ----

    def lookup(self, op_invoke_info: "OpInvokeInfo") -> Optional[QueryResult]:
        """Query perf data for an op.

        Dispatch logic (design doc S4.2):
          func_name -> op_mapping.yaml
            - not found -> return None
            - composite == true -> return None (spike: fallback to analytic)
            - category == "communication" -> return None (spike: fallback)
            - query_mode == "attention_special" -> return None (spike: fallback)
            - default -> _lookup_compute()
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

        # Zero-cost ops: shape-only operations with no kernel execution
        if mapping.get("zero_cost"):
            return QueryResult(
                latency_us=0.0,
                confidence=1.0,
                source=QuerySource.MEASURED,
                details={"kernel_type": "zero_cost", "zero_cost": True},
            )

        return self._lookup_compute(op_invoke_info, mapping)

    # ---- Compute op lookup (design doc S4.2 _lookup_compute) ----

    def _lookup_compute(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
    ) -> Optional[QueryResult]:
        # Build list of kernel_types to try: primary + alternates
        kernel_types = [mapping["kernel_type"]]
        for alt in mapping.get("alternate_kernel_types", []):
            if alt not in kernel_types:
                kernel_types.append(alt)

        # Extract tensor shapes and dtypes from OpInvokeInfo.args
        tc_inputs = self._extract_tensor_inputs(op_invoke_info)

        # Try each kernel_type until one matches
        for kernel_type in kernel_types:
            df = self._load_csv(kernel_type)
            if df is None:
                continue

            for _, row in df.iterrows():
                if self._inputs_match(tc_inputs, row, kernel_type=kernel_type):
                    latency_col = (
                        "Average Duration(us)"
                        if "Average Duration(us)" in df.columns
                        else "Duration(us)"
                    )
                    logger.debug(
                        "HIT %s: tc_shapes=%s -> %s (%.1f us)",
                        kernel_type,
                        [s for s, _ in tc_inputs],
                        row.get("Input Shapes", ""),
                        float(row[latency_col]),
                    )
                    return QueryResult(
                        latency_us=float(row[latency_col]),
                        confidence=1.0,
                        source=QuerySource.MEASURED,
                        details={"kernel_type": kernel_type},
                    )

        # Log miss with shape details for debugging (use primary kernel_type)
        primary_kernel = kernel_types[0]
        df = self._load_csv(primary_kernel)
        csv_shapes_list = []
        if df is not None:
            for _, row in df.iterrows():
                csv_shapes_list.append(str(row.get("Input Shapes", "")))
        logger.debug(
            "MISS %s: tc_shapes=%s, csv_shapes=%s",
            primary_kernel,
            [s for s, _ in tc_inputs],
            csv_shapes_list,
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
        kernel_type: str = "",
    ) -> bool:
        """Match TensorCast input shapes/dtypes against a CSV row.

        Handles:
        - FRACTAL_NZ restoration (design doc S4.9)
        - ND weight transpose for matmul kernels (CSV stores (N,K), TC sees (K,N))
        - Block-padding tolerance (TC pads seq to NPU tile alignment)
        """
        csv_shapes = _parse_shape_str(str(csv_row.get("Input Shapes", "")))
        csv_dtypes = _parse_str_list(str(csv_row.get("Input Data Types", "")))
        csv_formats = _parse_str_list(str(csv_row.get("Input Formats", "")))

        # SwiGlu input normalization: TC sends 2 inputs (gate, up),
        # profiling CSV has 1 fused input concatenated along last dim.
        tc_inputs_normalized = tc_inputs
        if (
            kernel_type in _SWIGLU_KERNELS
            and len(tc_inputs) == 2
            and len(csv_shapes) == 1
        ):
            s1, dtype1 = tc_inputs[0]
            s2, dtype2 = tc_inputs[1]
            s1 = _strip_batch_dim(s1)
            s2 = _strip_batch_dim(s2)
            if (
                len(s1) == len(s2)
                and s1[:-1] == s2[:-1]
                and dtype1 == dtype2
            ):
                merged_shape = s1[:-1] + (s1[-1] + s2[-1],)
                tc_inputs_normalized = [(merged_shape, dtype1)]

        if len(tc_inputs_normalized) != len(csv_shapes):
            return False

        for i, (tc_shape, tc_dtype) in enumerate(tc_inputs_normalized):
            # Check dtype
            expected_dtype = DTYPE_MAP.get(tc_dtype)
            if expected_dtype is None or i >= len(csv_dtypes):
                return False
            if expected_dtype != csv_dtypes[i]:
                return False

            # Get CSV shape, restore FRACTAL_NZ if needed
            csv_shape = csv_shapes[i]
            fmt = csv_formats[i] if i < len(csv_formats) else "ND"
            if fmt == "FRACTAL_NZ":
                csv_shape = fractal_nz_to_nd(csv_shape)

            if tc_shape == csv_shape:
                continue

            # Strip leading batch dim=1: TC keeps (1, seq, dim), profiling has (seq, dim)
            tc_shape_stripped = _strip_batch_dim(tc_shape)
            if tc_shape_stripped == csv_shape:
                continue

            # ND weight transpose for matmul: CSV stores (N,K), TC sees (K,N)
            # Only for 2D ND weights (not activation, not FRACTAL_NZ)
            if (
                kernel_type in _MATMUL_KERNELS
                and i >= 1
                and fmt == "ND"
                and len(tc_shape_stripped) == 2
                and len(csv_shape) == 2
                and tc_shape_stripped == (csv_shape[1], csv_shape[0])
            ):
                continue

            # Block-padding tolerance: TC pads to NPU tile alignment
            if self._shapes_match_with_padding(tc_shape_stripped, csv_shape):
                continue

            return False

        return True

    @staticmethod
    def _shapes_match_with_padding(
        tc_shape: Tuple[int, ...], csv_shape: Tuple[int, ...]
    ) -> bool:
        """Check if shapes match allowing block-padding on any dimension."""
        if len(tc_shape) != len(csv_shape):
            return False
        has_padding = False
        for tc_dim, csv_dim in zip(tc_shape, csv_shape):
            if tc_dim == csv_dim:
                continue
            if _is_block_padded(tc_dim, csv_dim):
                has_padding = True
                continue
            return False
        return has_padding
