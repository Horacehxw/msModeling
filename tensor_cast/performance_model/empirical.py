"""EmpiricalPerformanceModel: measurement-based performance model.

Design doc reference: §4.3
"""

import logging
from typing import Optional

from overrides import override

import torch

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
        self._hit_details: list[str] = []
        self._miss_details: list[str] = []

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
        self._miss_details.append(f"{func_name} {tc_shapes}")
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
        if self._hit_details:
            logger.info("  HITs: %s", " | ".join(self._hit_details))
        if self._miss_details:
            logger.info("  MISSes: %s", " | ".join(self._miss_details))
