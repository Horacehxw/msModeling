"""ProfilingDataSource: CSV-backed data source with op_mapping + FRACTAL_NZ.

Design doc reference: S4.2 (ProfilingDataSource), S4.9 (FRACTAL_NZ)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd
import torch
import yaml

from ...device import DeviceProfile
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

# RoPE kernel types: TC dispatches (B,H,S,D) layout with [Q, K, cos, sin],
# but profiling CSVs store (B,S,H,D) layout with [K, Q, cos, sin] and
# cos/sin have an extra head dim (1).
_ROPE_KERNELS = frozenset({"ApplyRotaryPosEmb"})

# Common NPU tile alignment sizes (Da Vinci Cube unit)
# BF16: 16x16, INT8: 16x32
_BLOCK_SIZES = (16, 32, 64)


def _normalize_rope_inputs(
    tc_inputs: List[Tuple[Tuple[int, ...], torch.dtype]],
) -> List[Tuple[Tuple[int, ...], torch.dtype]]:
    """Normalize RoPE inputs from TC layout to profiling CSV layout.

    TC dispatches: [Q(B,Hq,S,D), K(B,Hk,S,D), cos(1,S,D), sin(1,S,D)]
    CSV expects:   [K(B,S,Hk,D), Q(B,S,Hq,D), cos(B,S,1,D), sin(B,S,1,D)]

    Transformations:
    1. Swap Q and K (TC: [Q,K,...] → CSV: [K,Q,...])
    2. Transpose H,S dims in Q and K: (B,H,S,D) → (B,S,H,D)
    3. Insert head dim=1 at position 2 for cos/sin: (1,S,D) → (1,S,1,D)
    """
    q_shape, q_dtype = tc_inputs[0]
    k_shape, k_dtype = tc_inputs[1]
    cos_shape, cos_dtype = tc_inputs[2]
    sin_shape, sin_dtype = tc_inputs[3]

    # Transpose Q and K: (B,H,S,D) → (B,S,H,D)
    if len(q_shape) == 4:
        q_shape = (q_shape[0], q_shape[2], q_shape[1], q_shape[3])
    if len(k_shape) == 4:
        k_shape = (k_shape[0], k_shape[2], k_shape[1], k_shape[3])

    # Insert head dim=1 for cos/sin: (1,S,D) → (1,S,1,D)
    if len(cos_shape) == 3:
        cos_shape = (cos_shape[0], cos_shape[1], 1, cos_shape[2])
    if len(sin_shape) == 3:
        sin_shape = (sin_shape[0], sin_shape[1], 1, sin_shape[2])

    # Reorder: [Q, K, cos, sin] → [K, Q, cos, sin]
    return [
        (k_shape, k_dtype),
        (q_shape, q_dtype),
        (cos_shape, cos_dtype),
        (sin_shape, sin_dtype),
    ]


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
        device_profile: DeviceProfile for comm_grid topology_tier resolution.
            Optional — when omitted, communication lookups skip tier filtering.
    """

    def __init__(
        self, data_dir: str | Path, device_profile: Optional[DeviceProfile] = None
    ):
        self.data_dir = Path(data_dir)
        self.comm_grid = device_profile.comm_grid if device_profile else None
        self._op_mapping = self._load_op_mapping()
        self._csv_cache: Dict[str, Optional[pd.DataFrame]] = {}
        # Set after each lookup() miss to explain why
        self.last_miss_reason: str = ""

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
            - composite == true -> _lookup_composite()
            - category == "communication" -> _lookup_comm()
            - query_mode == "attention_special" -> _lookup_attention()
            - zero_cost == true -> return QueryResult(0.0)
            - default -> _lookup_compute()
        """
        func_str = _normalize_func_name(op_invoke_info.func)
        mappings = self._op_mapping.get("operator_mappings", {})
        mapping = mappings.get(func_str)
        if mapping is None:
            self.last_miss_reason = "unmapped"
            return None

        # Composite ops: try decomposition via sub_kernels, else skip
        if mapping.get("composite"):
            return self._lookup_composite(op_invoke_info, mapping)
        if mapping.get("category") == "communication":
            return self._lookup_comm(op_invoke_info, mapping)
        if mapping.get("query_mode") == "attention_special":
            return self._lookup_attention(op_invoke_info, mapping)

        # Zero-cost ops: shape-only operations with no kernel execution
        if mapping.get("zero_cost"):
            return QueryResult(
                latency_us=0.0,
                confidence=1.0,
                source=QuerySource.MEASURED,
                details={"kernel_type": "zero_cost", "zero_cost": True},
            )

        return self._lookup_compute(op_invoke_info, mapping)

    # ---- Composite op lookup ----

    def _lookup_composite(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
    ) -> Optional[QueryResult]:
        """Decompose composite ops and look up compute sub-kernels.

        For matmul+comm composites (e.g., matmul_all_reduce), look up the
        compute sub-kernel (MatMulV2) with the op's tensor inputs.
        Communication sub-kernels are left to the analytic model.
        Returns None if no sub_kernels or no match found.
        """
        sub_kernels = mapping.get("sub_kernels", [])
        if not sub_kernels:
            self.last_miss_reason = "no_sub_kernels"
            return None

        # Extract tensor inputs from the composite op
        tc_inputs = self._extract_tensor_inputs(op_invoke_info)

        # Try each compute sub-kernel (skip communication kernels)
        for kernel_type in sub_kernels:
            if kernel_type.startswith("hcom_"):
                continue
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
                        "HIT (composite) %s: tc_shapes=%s -> %s (%.1f us)",
                        kernel_type,
                        [s for s, _ in tc_inputs],
                        row.get("Input Shapes", ""),
                        float(row[latency_col]),
                    )
                    return QueryResult(
                        latency_us=float(row[latency_col]),
                        confidence=0.8,  # Lower confidence: partial match
                        source=QuerySource.MEASURED,
                        details={
                            "kernel_type": kernel_type,
                            "composite": True,
                            "note": "compute sub-kernel only; comm handled by analytic",
                        },
                    )
        self.last_miss_reason = "shape_mismatch"
        return None

    # ---- Communication op lookup (design doc §4.7) ----

    def _resolve_topology_tier(self, group: list) -> Optional[int]:
        """Resolve topology_tier from group using CommGrid, mirroring
        CommAnalyticModel._get_topology_idx_for_group().

        Returns start_dim (topology_tier) or None if comm_grid is not set.
        """
        if self.comm_grid is None:
            return None
        coords = [
            self._rank_to_coord(r, self.comm_grid.grid.shape) for r in group
        ]
        diff_dim = -1
        for dim_idx in range(self.comm_grid.grid.dim()):
            first = coords[0][dim_idx]
            if any(c[dim_idx] != first for c in coords[1:]):
                diff_dim = dim_idx
                break
        if diff_dim == -1:
            return max(self.comm_grid.topologies.keys())
        for start_dim in sorted(self.comm_grid.topologies.keys(), reverse=True):
            if start_dim <= diff_dim:
                return start_dim
        return None

    @staticmethod
    def _rank_to_coord(rank: int, grid_shape) -> list:
        coord = []
        temp = rank
        for dim_size in reversed(grid_shape):
            coord.insert(0, temp % dim_size)
            temp //= dim_size
        return coord

    def _lookup_comm(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
    ) -> Optional[QueryResult]:
        """Look up communication op latency by message_bytes + num_devices + topology_tier.

        Communication CSV columns: message_bytes, num_devices, dtype,
        topology_tier, Duration(us).

        Args are expected as (tensor, ..., rank, rank_group) where rank is
        second-to-last and rank_group (list of device ranks) is always last.
        topology_tier is resolved from rank + rank_group via CommGrid when
        comm_grid is set; otherwise the CSV is queried without tier filtering.
        """
        kernel_type = mapping.get("kernel_type")
        if not kernel_type:
            self.last_miss_reason = "unmapped"
            return None

        df = self._load_csv(kernel_type)
        if df is None:
            self.last_miss_reason = "csv_not_found"
            return None

        # Check that CSV has the expected microbenchmark columns.
        # Raw profiling CSVs (from kernel_details.csv) have "Input Shapes" etc.
        # and cannot be queried by structured fields — fall back to analytic.
        required_cols = {"message_bytes", "num_devices"}
        if not required_cols.issubset(df.columns):
            logger.debug(
                "MISS (comm) %s: CSV missing columns %s, need microbenchmark format",
                kernel_type,
                required_cols - set(df.columns),
            )
            self.last_miss_reason = "csv_format_raw"
            return None

        # Extract the first tensor arg for message_bytes
        tensor = op_invoke_info.args[0]
        if not isinstance(tensor, torch.Tensor):
            self.last_miss_reason = "invalid_args"
            return None
        message_bytes = tensor.nelement() * tensor.element_size()

        # Extract rank (second-to-last) and rank_group (last)
        rank_group = op_invoke_info.args[-1]
        rank = op_invoke_info.args[-2]
        if not isinstance(rank_group, (list, tuple)):
            self.last_miss_reason = "invalid_args"
            return None
        num_devices = len(rank_group)

        # Resolve topology_tier from group via CommGrid
        topology_tier = self._resolve_topology_tier(list(rank_group))

        # Build match mask: always filter on message_bytes + num_devices;
        # add topology_tier filter when the CSV has the column and tier is known.
        mask = (df["message_bytes"] == message_bytes) & (df["num_devices"] == num_devices)
        if topology_tier is not None and "topology_tier" in df.columns:
            mask = mask & (df["topology_tier"] == topology_tier)

        matched = df[mask]
        if matched.empty:
            logger.debug(
                "MISS (comm) %s: message_bytes=%d, num_devices=%d, topology_tier=%s",
                kernel_type,
                message_bytes,
                num_devices,
                topology_tier,
            )
            self.last_miss_reason = "shape_mismatch"
            return None

        row = matched.iloc[0]
        latency_col = (
            "Average Duration(us)"
            if "Average Duration(us)" in df.columns
            else "Duration(us)"
        )
        latency = float(row[latency_col])
        logger.debug(
            "HIT (comm) %s: message_bytes=%d, num_devices=%d, topology_tier=%s -> %.2f us",
            kernel_type,
            message_bytes,
            num_devices,
            topology_tier,
            latency,
        )
        return QueryResult(
            latency_us=latency,
            confidence=0.9,
            source=QuerySource.MEASURED,
            details={"kernel_type": kernel_type, "topology_tier": topology_tier},
        )

    # ---- Attention special lookup (design doc §4.8) ----

    def _lookup_attention(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
    ) -> Optional[QueryResult]:
        """Look up attention op latency using FIA microbenchmark CSV.

        Attention CSV columns: batch_size, avg_seq_len, num_heads, head_dim,
        dtype, Duration(us).

        Attention op args layout (from tensor_cast/ops/attention.py):
          args[0]: query  (num_tokens, hidden_size)
          args[1]: key    (total_blocks, block_size, kv_heads, head_dim) or (*, kv_heads, head_dim)
          args[6]: seq_lens (batch_size,) — per-request KV lengths
        """
        kernel_type = mapping.get("kernel_type")
        if not kernel_type:
            self.last_miss_reason = "unmapped"
            return None

        df = self._load_csv(kernel_type)
        if df is None:
            self.last_miss_reason = "csv_not_found"
            return None

        # Check that CSV has the expected microbenchmark columns.
        # Raw profiling CSVs have "Input Shapes" etc. and cannot be queried
        # by structured attention fields — fall back to analytic.
        required_cols = {"batch_size", "avg_seq_len", "num_heads", "head_dim"}
        if not required_cols.issubset(df.columns):
            logger.debug(
                "MISS (attention) %s: CSV missing columns %s, "
                "need microbenchmark format",
                kernel_type,
                required_cols - set(df.columns),
            )
            self.last_miss_reason = "csv_format_raw"
            return None

        # Extract seq_lens from args[6]
        args = op_invoke_info.args
        if len(args) < 7:
            self.last_miss_reason = "invalid_args"
            return None
        seq_lens = args[6]
        if not isinstance(seq_lens, torch.Tensor):
            self.last_miss_reason = "invalid_args"
            return None

        # Compute batch_size and avg_seq_len from seq_lens tensor
        batch_size = seq_lens.shape[0]
        # Use .float().mean().item() — seq_lens must be on CPU (not meta)
        try:
            avg_seq_len = int(seq_lens.float().mean().item())
        except Exception:
            self.last_miss_reason = "invalid_args"
            return None

        # Extract head_dim and kv_heads from key tensor (args[1])
        key = args[1]
        if not isinstance(key, torch.Tensor) or key.ndim < 2:
            self.last_miss_reason = "invalid_args"
            return None
        head_dim = key.shape[-1]
        kv_heads = key.shape[-2]

        # Compute num_heads from query hidden_size / head_dim
        query = args[0]
        if not isinstance(query, torch.Tensor) or query.ndim < 2:
            self.last_miss_reason = "invalid_args"
            return None
        hidden_size = query.shape[-1]
        num_heads = hidden_size // head_dim

        # Get dtype string
        dtype_str = DTYPE_MAP.get(query.dtype)
        if dtype_str is None:
            self.last_miss_reason = "invalid_args"
            return None

        # Match CSV rows on all 5 dimensions
        mask = (
            (df["batch_size"] == batch_size)
            & (df["avg_seq_len"] == avg_seq_len)
            & (df["num_heads"] == num_heads)
            & (df["head_dim"] == head_dim)
            & (df["dtype"] == dtype_str)
        )
        matched = df[mask]
        if matched.empty:
            logger.debug(
                "MISS (attention) %s: batch=%d, avg_seq=%d, heads=%d, "
                "head_dim=%d, dtype=%s",
                kernel_type,
                batch_size,
                avg_seq_len,
                num_heads,
                head_dim,
                dtype_str,
            )
            self.last_miss_reason = "shape_mismatch"
            return None

        row = matched.iloc[0]
        latency_col = (
            "Average Duration(us)"
            if "Average Duration(us)" in df.columns
            else "Duration(us)"
        )
        latency = float(row[latency_col])
        logger.debug(
            "HIT (attention) %s: batch=%d, avg_seq=%d, heads=%d, "
            "head_dim=%d -> %.2f us",
            kernel_type,
            batch_size,
            avg_seq_len,
            num_heads,
            head_dim,
            latency,
        )
        return QueryResult(
            latency_us=latency,
            confidence=0.9,
            source=QuerySource.MEASURED,
            details={"kernel_type": kernel_type},
        )

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
        # Determine miss reason: input count mismatch vs shape mismatch
        if df is not None and len(df) > 0:
            csv_first_shapes = _parse_shape_str(str(df.iloc[0].get("Input Shapes", "")))
            if len(tc_inputs) != len(csv_first_shapes):
                self.last_miss_reason = "input_count_mismatch"
            else:
                self.last_miss_reason = "shape_mismatch"
        else:
            self.last_miss_reason = "csv_not_found"
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
        """Extract (shape, dtype) for each non-scalar tensor arg.

        Scalar tensors (ndim=0, shape=()) are filtered out because profiling
        CSVs never include scalar inputs in their shape strings.
        """
        inputs = []
        for arg in op_invoke_info.args:
            if isinstance(arg, torch.Tensor) and arg.ndim > 0:
                inputs.append((tuple(arg.shape), arg.dtype))
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    if isinstance(item, torch.Tensor) and item.ndim > 0:
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

        # RoPE input normalization: TC sends [Q(B,H,S,D), K(B,H,S,D), cos(1,S,D), sin(1,S,D)]
        # CSV expects [K(B,S,H,D), Q(B,S,H,D), cos(B,S,1,D), sin(B,S,1,D)]
        tc_inputs_normalized = tc_inputs
        if (
            kernel_type in _ROPE_KERNELS
            and len(tc_inputs) == 4
            and len(csv_shapes) == 4
        ):
            tc_inputs_normalized = _normalize_rope_inputs(tc_inputs)

        # SwiGlu input normalization: TC sends 2 inputs (gate, up),
        # profiling CSV has 1 fused input concatenated along last dim.
        if (
            kernel_type in _SWIGLU_KERNELS
            and len(tc_inputs) == 2
            and len(csv_shapes) == 1
        ):
            s1, dtype1 = tc_inputs[0]
            s2, dtype2 = tc_inputs[1]
            s1 = _strip_batch_dim(s1)
            s2 = _strip_batch_dim(s2)
            if len(s1) == len(s2) and s1[:-1] == s2[:-1] and dtype1 == dtype2:
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
            csv_shape_stripped = _strip_batch_dim(csv_shape)
            if tc_shape_stripped == csv_shape_stripped:
                continue
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
            # Also try with both batch dims stripped
            if self._shapes_match_with_padding(tc_shape_stripped, csv_shape_stripped):
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
