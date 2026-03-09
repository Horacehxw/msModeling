"""EmpiricalPerformanceModel: measurement-based performance model.

Design doc reference: §4.3
"""

import logging
from collections import Counter
from typing import Optional

import torch

from overrides import override

from ..device import DeviceProfile
from .base import PerformanceModel
from .op_invoke_info import OpInvokeInfo
from .perf_database.data_source import DataSource

logger = logging.getLogger(__name__)

# Human-readable descriptions for miss reason codes
_MISS_REASON_LABELS = {
    "unmapped": "not in op_mapping.yaml",
    "shape_mismatch": "kernel found, no matching shape in CSV",
    "input_count_mismatch": "TC input count differs from CSV",
    "csv_format_raw": "CSV has raw profiling format (needs microbenchmark)",
    "csv_not_found": "kernel CSV file missing",
    "no_sub_kernels": "composite op has no sub_kernels defined",
    "invalid_args": "op args could not be parsed",
}


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
        self._hit_details: list[str] = []
        # Each miss: (func_name, reason, tc_shapes)
        self._miss_details: list[tuple[str, str, list[tuple]]] = []

    @property
    def fallback_model(self) -> PerformanceModel:
        if self._fallback_model is None:
            from .analytic import AnalyticPerformanceModel

            self._fallback_model = AnalyticPerformanceModel(self.device_profile)
        return self._fallback_model

    @override
    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        result = self.data_source.lookup(op_invoke_info)
        func_name = str(op_invoke_info.func).removeprefix("torch.ops.")
        if result is not None:
            self._stats["hit"] += 1
            self._hit_details.append(
                f"{func_name}->{result.details.get('kernel_type', '?')}"
            )
            return PerformanceModel.Result(
                execution_time_s=result.latency_us * 1e-6,
                statistics={
                    "source": result.source.name,
                    "confidence": result.confidence,
                    **result.details,
                },
            )
        self._stats["miss"] += 1
        tc_shapes = [
            tuple(a.shape) for a in op_invoke_info.args if isinstance(a, torch.Tensor)
        ]
        # Read miss reason from data source (if it supports it)
        reason = getattr(self.data_source, "last_miss_reason", "unknown")
        self._miss_details.append((func_name, reason, tc_shapes))
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
            stats["hit"],
            stats["total"],
            stats["hit_rate"] * 100,
        )

        # Deduplicated HITs: count occurrences of each mapping
        if self._hit_details:
            hit_counts = Counter(self._hit_details)
            hit_lines = [
                f"  {mapping} (x{count})" if count > 1 else f"  {mapping}"
                for mapping, count in hit_counts.most_common()
            ]
            logger.info(
                "  HITs (%d unique):\n%s", len(hit_counts), "\n".join(hit_lines)
            )

        # MISSes grouped by reason category
        if self._miss_details:
            by_reason: dict[str, list[tuple[str, list[tuple]]]] = {}
            for func_name, reason, tc_shapes in self._miss_details:
                by_reason.setdefault(reason, []).append((func_name, tc_shapes))

            miss_lines = []
            for reason, ops in sorted(by_reason.items()):
                label = _MISS_REASON_LABELS.get(reason, reason)
                # Deduplicate ops with same func_name
                op_counts = Counter(func_name for func_name, _ in ops)
                op_strs = [
                    f"{name} (x{count})" if count > 1 else name
                    for name, count in op_counts.most_common()
                ]
                miss_lines.append(f"  [{reason}] {label}: {', '.join(op_strs)}")
                # Log shape details at DEBUG level
                for func_name, tc_shapes in ops:
                    logger.debug("    %s shapes: %s", func_name, tc_shapes)

            logger.info(
                "  MISSes (%d unique reasons):\n%s",
                len(by_reason),
                "\n".join(miss_lines),
            )
