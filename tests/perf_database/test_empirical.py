import torch
import pytest
from unittest.mock import MagicMock

from tensor_cast.performance_model.base import PerformanceModel
from tensor_cast.performance_model.empirical import EmpiricalPerformanceModel
from tensor_cast.performance_model.perf_database.data_source import (
    DataSource,
    QueryResult,
    QuerySource,
)


class HitDataSource(DataSource):
    def lookup(self, op_invoke_info):
        return QueryResult(
            latency_us=45.3,
            confidence=1.0,
            source=QuerySource.MEASURED,
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
    fallback.process_op.return_value = PerformanceModel.Result(
        execution_time_s=100e-6
    )

    model = EmpiricalPerformanceModel(
        device,
        data_source=MissDataSource(),
        fallback_model=fallback,
    )
    result = model.process_op(_make_mock_op_invoke_info())
    fallback.process_op.assert_called_once()
    assert abs(result.execution_time_s - 100e-6) < 1e-12


def test_empirical_model_name():
    device = _make_mock_device_profile()
    model = EmpiricalPerformanceModel(device, data_source=MissDataSource())
    assert model.name == "empirical"
