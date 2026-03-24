"""InterpolatingDataSource: wraps ProfilingDataSource with 1D linear interpolation.

When exact lookup misses, tries interpolation on the varying dimension:
- Compute ops: interpolate on first dim of first input (seq_len/num_tokens)
- Comm ops: interpolate on message_bytes
- Attention ops: interpolate on avg_seq_len (with optional sqrt transform)

Design doc reference: S4.4 (InterpolatingDataSource)
"""

import logging
import math
from typing import List, Optional, Tuple, TYPE_CHECKING

import torch

from .data_source import DataSource, QueryResult, QuerySource
from .profiling_data_source import (
    _dtype_byte_size,
    _normalize_func_name,
    _parse_shape_str,
    _parse_str_list,
    _strip_batch_dim,
    DTYPE_MAP,
    ProfilingDataSource,
)

if TYPE_CHECKING:
    from ..op_invoke_info import OpInvokeInfo

logger = logging.getLogger(__name__)


def _interp_1d(x0: float, y0: float, x1: float, y1: float, target_x: float) -> float:
    """Linear interpolation between two points."""
    if x1 == x0:
        return y0
    t = (target_x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def _find_bracket(values: List[float], target: float) -> Optional[Tuple[float, float]]:
    """Find (left, right) values that bracket target. Returns None if can't bracket."""
    below = [v for v in values if v <= target]
    above = [v for v in values if v >= target]
    if not below or not above:
        return None  # Can't interpolate, would need extrapolation
    return (max(below), min(above))


class InterpolatingDataSource(DataSource):
    """Wrapper datasource that adds 1D linear interpolation fallback.

    When the base ProfilingDataSource returns None (exact miss), this layer
    attempts interpolation by finding bracketing data points and linearly
    interpolating on the varying dimension.
    """

    def __init__(self, base: ProfilingDataSource):
        self.base = base
        # Read kernel_overrides from op_mapping for sqrt transform etc.
        ip = self.base._op_mapping.get("interpolation_policy", {})
        self._kernel_overrides = ip.get("kernel_overrides", {})

    def lookup(self, op_invoke_info: "OpInvokeInfo") -> Optional[QueryResult]:
        result = self.base.lookup(op_invoke_info)
        if result is not None:
            return result
        return self._interpolate(op_invoke_info)

    def _interpolate(self, op_invoke_info: "OpInvokeInfo") -> Optional[QueryResult]:
        """Determine which query path to use and dispatch to the right interpolator."""
        func_str = _normalize_func_name(op_invoke_info.func)
        mappings = self.base._op_mapping.get("operator_mappings", {})
        mapping = mappings.get(func_str)
        if mapping is None:
            return None

        # Don't interpolate zero_cost or composite ops
        if mapping.get("zero_cost") or mapping.get("composite"):
            return None

        if mapping.get("category") == "communication":
            return self._interpolate_comm(op_invoke_info, mapping)
        if mapping.get("query_mode") == "attention_special":
            return self._interpolate_attention(op_invoke_info, mapping)
        if mapping.get("query_mode") == "elementwise":
            return self._interpolate_elementwise(op_invoke_info, mapping)
        return self._interpolate_compute(op_invoke_info, mapping)

    # ---- Compute interpolation ----

    def _interpolate_compute(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
    ) -> Optional[QueryResult]:
        """Interpolate compute ops on the first dimension of the first input.

        Strategy: find all CSV rows where dtype matches and all dimensions
        match EXCEPT the first dim of the first input. Collect
        (first_dim_value, duration) pairs, bracket the target, and interpolate.
        """
        kernel_type = mapping.get("kernel_type")
        if not kernel_type:
            return None

        df = self.base._load_csv(kernel_type)
        if df is None:
            return None

        tc_inputs = self.base._extract_tensor_inputs(op_invoke_info)
        if not tc_inputs:
            return None

        # Target: first dim of first input (typically seq_len)
        target_dim = tc_inputs[0][0][0]

        latency_col = (
            "Average Duration(us)"
            if "Average Duration(us)" in df.columns
            else "Duration(us)"
        )

        # Find CSV rows where all dims/dtypes match except first dim of first input
        candidates: List[Tuple[float, float]] = []  # (first_dim_value, duration)

        for _, row in df.iterrows():
            csv_shapes = _parse_shape_str(str(row.get("Input Shapes", "")))
            csv_dtypes = _parse_str_list(str(row.get("Input Data Types", "")))

            if len(csv_shapes) != len(tc_inputs):
                continue

            # Check all dtypes match
            dtype_match = True
            for i, (_, tc_dtype) in enumerate(tc_inputs):
                expected = DTYPE_MAP.get(tc_dtype)
                if i >= len(csv_dtypes) or expected != csv_dtypes[i]:
                    dtype_match = False
                    break
            if not dtype_match:
                continue

            # Check all dims match except first dim of first input
            all_match = True
            for i, (tc_shape, _) in enumerate(tc_inputs):
                csv_shape = csv_shapes[i]
                if i == 0:
                    # First input: skip first dim, check rest match
                    if len(tc_shape) != len(csv_shape):
                        all_match = False
                        break
                    if tc_shape[1:] != csv_shape[1:]:
                        all_match = False
                        break
                else:
                    # Other inputs: all dims must match exactly
                    if tc_shape != csv_shape:
                        all_match = False
                        break
            if all_match:
                candidates.append((float(csv_shapes[0][0]), float(row[latency_col])))

        if len(candidates) < 2:
            return None

        candidates.sort(key=lambda x: x[0])
        return self._interpolate_from_candidates(
            candidates, float(target_dim), kernel_type
        )

    # ---- Communication interpolation ----

    def _interpolate_comm(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
    ) -> Optional[QueryResult]:
        """Interpolate communication ops on message_bytes."""
        kernel_type = mapping.get("kernel_type")
        if not kernel_type:
            return None

        df = self.base._load_csv(kernel_type)
        if df is None:
            return None

        # Extract message_bytes and num_devices (same logic as ProfilingDataSource)
        tensor = op_invoke_info.args[0]
        if not isinstance(tensor, torch.Tensor):
            return None
        message_bytes = tensor.nelement() * tensor.element_size()

        rank_group = op_invoke_info.args[-1]
        if not isinstance(rank_group, (list, tuple)):
            return None
        num_devices = len(rank_group)

        # reduce_scatter: TC args[0] is the full input tensor (sendBuf), but
        # bench CSV message_bytes follows HCCL API convention where recvCount
        # is the per-rank output size.  Divide by num_devices to align.
        func_str = _normalize_func_name(op_invoke_info.func)
        if func_str == "tensor_cast.reduce_scatter.default" and num_devices > 1:
            message_bytes = message_bytes // num_devices

        latency_col = (
            "Average Duration(us)"
            if "Average Duration(us)" in df.columns
            else "Duration(us)"
        )

        # Filter by num_devices, collect (message_bytes, duration)
        mask = df["num_devices"] == num_devices
        matched = df[mask]
        if matched.empty:
            return None

        candidates: List[Tuple[float, float]] = []
        for _, row in matched.iterrows():
            candidates.append((float(row["message_bytes"]), float(row[latency_col])))

        if len(candidates) < 2:
            return None

        candidates.sort(key=lambda x: x[0])
        return self._interpolate_from_candidates(
            candidates, float(message_bytes), kernel_type
        )

    # ---- Attention interpolation ----

    def _interpolate_attention(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
    ) -> Optional[QueryResult]:
        """Interpolate attention ops on avg_seq_len with optional sqrt transform."""
        kernel_type = mapping.get("kernel_type")
        if not kernel_type:
            return None

        df = self.base._load_csv(kernel_type)
        if df is None:
            return None

        # Extract attention parameters (same logic as ProfilingDataSource._lookup_attention)
        args = op_invoke_info.args
        if len(args) < 7:
            return None

        seq_lens = args[6]
        if not isinstance(seq_lens, torch.Tensor):
            return None

        batch_size = seq_lens.shape[0]
        try:
            avg_seq_len = int(seq_lens.float().mean().item())
        except Exception:
            return None

        key = args[1]
        if not isinstance(key, torch.Tensor) or key.ndim < 2:
            return None
        head_dim = key.shape[-1]
        kv_heads = key.shape[-2]

        query = args[0]
        if not isinstance(query, torch.Tensor) or query.ndim < 2:
            return None
        hidden_size = query.shape[-1]
        num_heads = hidden_size // head_dim

        dtype_str = DTYPE_MAP.get(query.dtype)
        if dtype_str is None:
            return None

        latency_col = (
            "Average Duration(us)"
            if "Average Duration(us)" in df.columns
            else "Duration(us)"
        )

        # Filter by batch_size + num_heads + head_dim + dtype
        mask = (
            (df["batch_size"] == batch_size)
            & (df["num_heads"] == num_heads)
            & (df["head_dim"] == head_dim)
            & (df["dtype"] == dtype_str)
        )
        matched = df[mask]
        if matched.empty:
            return None

        candidates: List[Tuple[float, float]] = []
        for _, row in matched.iterrows():
            candidates.append((float(row["avg_seq_len"]), float(row[latency_col])))

        if len(candidates) < 2:
            return None

        candidates.sort(key=lambda x: x[0])

        # Check for sqrt transform in kernel_overrides
        override = self._kernel_overrides.get(kernel_type, {})
        transform = override.get("shape_transform")

        if transform == "sqrt":
            return self._interpolate_from_candidates_sqrt(
                candidates, float(avg_seq_len), kernel_type
            )
        return self._interpolate_from_candidates(
            candidates, float(avg_seq_len), kernel_type
        )

    # ---- Elementwise interpolation ----

    def _interpolate_elementwise(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
    ) -> Optional[QueryResult]:
        """Interpolate elementwise ops on first dim of output shape, dtype-relaxed.

        Groups CSV rows by output_shape[1:] (hidden dims must match exactly).
        Collects (output_shape[0], latency_scaled) candidates and interpolates
        on the first dim (num_tokens). Byte-ratio scaling applied per-candidate
        before interpolation.
        """
        kernel_type = mapping.get("kernel_type")
        if not kernel_type:
            return None

        df = self.base._load_csv(kernel_type)
        if df is None:
            return None

        # NOTE: OpInvokeInfo uses .out (not .output); aten ops may return tuple.
        out = op_invoke_info.out
        if isinstance(out, (list, tuple)):
            out = out[0] if out else None
        if out is None or not isinstance(out, torch.Tensor) or out.ndim == 0:
            return None

        output_shape = _strip_batch_dim(tuple(out.shape))
        if len(output_shape) < 1:
            return None
        target_dim = float(output_shape[0])
        tc_dtype_str = DTYPE_MAP.get(out.dtype)

        latency_col = self.base._latency_col(df)
        has_dtype_scaling = False

        candidates: List[Tuple[float, float]] = []
        for _, row in df.iterrows():
            csv_out_shapes = _parse_shape_str(str(row.get("Output Shapes", "")))
            csv_out_dtypes = _parse_str_list(str(row.get("Output Data Types", "")))
            if not csv_out_shapes:
                continue

            csv_shape = _strip_batch_dim(tuple(csv_out_shapes[0]))

            # Hidden dims must match (everything except first dim)
            if len(csv_shape) != len(output_shape) or csv_shape[1:] != output_shape[1:]:
                continue

            # Compute byte-ratio scaled latency
            latency = float(row[latency_col])
            csv_dtype_str = csv_out_dtypes[0] if csv_out_dtypes else None
            if csv_dtype_str and tc_dtype_str and csv_dtype_str != tc_dtype_str:
                tc_bytes = _dtype_byte_size(tc_dtype_str)
                csv_bytes = _dtype_byte_size(csv_dtype_str)
                if tc_bytes > 0 and csv_bytes > 0:
                    latency *= tc_bytes / csv_bytes
                    has_dtype_scaling = True

            candidates.append((float(csv_shape[0]), latency))

        if len(candidates) < 2:
            return None

        candidates.sort(key=lambda x: x[0])
        # Confidence: 0.6 if dtype-scaled (combining dtype approximation + interpolation),
        # 0.7 if same dtype (standard interpolation confidence).
        result = self._interpolate_from_candidates(candidates, target_dim, kernel_type)
        if result is not None and has_dtype_scaling:
            result = QueryResult(
                latency_us=result.latency_us,
                confidence=0.6,
                source=result.source,
                details={**result.details, "dtype_scaled": True},
            )
        return result

    # ---- Shared interpolation helpers ----

    def _interpolate_from_candidates(
        self,
        candidates: List[Tuple[float, float]],
        target: float,
        kernel_type: str,
    ) -> Optional[QueryResult]:
        """Find bracket and linearly interpolate from sorted candidates."""
        xs = [c[0] for c in candidates]
        bracket = _find_bracket(xs, target)
        if bracket is None:
            return None

        x_lo, x_hi = bracket

        # Find the duration values for the bracket bounds
        y_lo = next(y for x, y in candidates if x == x_lo)
        y_hi = next(y for x, y in candidates if x == x_hi)

        # If exact match was found by bracket (x_lo == x_hi == target),
        # that should have been caught by base lookup. But handle gracefully.
        latency = _interp_1d(x_lo, y_lo, x_hi, y_hi, target)

        logger.debug(
            "INTERPOLATED %s: target=%.1f, bracket=(%.1f, %.1f), "
            "durations=(%.1f, %.1f) -> %.1f us",
            kernel_type,
            target,
            x_lo,
            x_hi,
            y_lo,
            y_hi,
            latency,
        )
        return QueryResult(
            latency_us=latency,
            confidence=0.7,
            source=QuerySource.INTERPOLATED,
            details={
                "kernel_type": kernel_type,
                "method": "linear_1d",
                "bracket": (x_lo, x_hi),
            },
        )

    def _interpolate_from_candidates_sqrt(
        self,
        candidates: List[Tuple[float, float]],
        target: float,
        kernel_type: str,
    ) -> Optional[QueryResult]:
        """Interpolate with sqrt transform on the x-axis.

        For O(n^2) attention ops, sqrt-transform the interpolation dimension
        before linear interpolation. This better captures the quadratic
        relationship between seq_len and latency.
        """
        xs = [c[0] for c in candidates]
        bracket = _find_bracket(xs, target)
        if bracket is None:
            return None

        x_lo, x_hi = bracket
        y_lo = next(y for x, y in candidates if x == x_lo)
        y_hi = next(y for x, y in candidates if x == x_hi)

        # Transform to sqrt space for interpolation
        sqrt_lo = math.sqrt(x_lo)
        sqrt_hi = math.sqrt(x_hi)
        sqrt_target = math.sqrt(target)

        latency = _interp_1d(sqrt_lo, y_lo, sqrt_hi, y_hi, sqrt_target)

        logger.debug(
            "INTERPOLATED (sqrt) %s: target=%.1f (sqrt=%.2f), "
            "bracket=(%.1f, %.1f), durations=(%.1f, %.1f) -> %.1f us",
            kernel_type,
            target,
            sqrt_target,
            x_lo,
            x_hi,
            y_lo,
            y_hi,
            latency,
        )
        return QueryResult(
            latency_us=latency,
            confidence=0.6,  # Lower confidence for transformed interpolation
            source=QuerySource.INTERPOLATED,
            details={
                "kernel_type": kernel_type,
                "method": "linear_1d_sqrt",
                "bracket": (x_lo, x_hi),
            },
        )
