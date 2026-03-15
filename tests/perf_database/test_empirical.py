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


# --- C5: Interpolation toggle tests ---


def test_interpolation_toggle_off_by_default(tmp_path):
    """TC_ENABLE_INTERPOLATION unset → ProfilingDataSource used directly."""
    import os
    import yaml
    from unittest.mock import patch
    from tensor_cast.core.model_runner import _create_data_source
    from tensor_cast.performance_model.perf_database import ProfilingDataSource
    from tensor_cast.performance_model.perf_database.interpolating_data_source import (
        InterpolatingDataSource,
    )

    op_mapping = {"version": "test", "device": "TEST", "operator_mappings": {}}
    (tmp_path / "op_mapping.yaml").write_text(yaml.dump(op_mapping))

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TC_ENABLE_INTERPOLATION", None)
        ds = _create_data_source(str(tmp_path), device_profile=MagicMock())
        assert isinstance(ds, ProfilingDataSource)
        assert not isinstance(ds, InterpolatingDataSource)


def test_interpolation_toggle_on(tmp_path):
    """TC_ENABLE_INTERPOLATION=1 → InterpolatingDataSource wraps ProfilingDataSource."""
    import os
    import yaml
    from unittest.mock import patch
    from tensor_cast.core.model_runner import _create_data_source
    from tensor_cast.performance_model.perf_database.interpolating_data_source import (
        InterpolatingDataSource,
    )

    op_mapping = {"version": "test", "device": "TEST", "operator_mappings": {}}
    (tmp_path / "op_mapping.yaml").write_text(yaml.dump(op_mapping))

    with patch.dict(os.environ, {"TC_ENABLE_INTERPOLATION": "1"}):
        ds = _create_data_source(str(tmp_path), device_profile=MagicMock())
        assert isinstance(ds, InterpolatingDataSource)


# --- C6: Fused Op HR metric tests ---


def test_fused_op_hr_groups_dfc_as_one():
    """DFC constituent ops should be counted as 1 fused op."""
    from tensor_cast.performance_model.empirical import compute_fused_op_stats

    hit_details = [
        "aten.mm.default->MatMulV2",
        "tensor_cast.swiglu.default->SwiGlu",
        "aten.mm.default->MatMulV2",  # duplicate
    ]
    miss_details = [
        ("tensor_cast.permute_tokens.default", "csv_not_found", []),
        ("tensor_cast.grouped_matmul_quant_swiglu.default", "csv_not_found", []),
        ("tensor_cast.unpermute_tokens.default", "csv_not_found", []),
        ("tensor_cast.all_to_all.default", "csv_not_found", []),
        ("aten.embedding.default", "shape_mismatch", []),
    ]

    fused_groups = {
        "DispatchFFNCombine": [
            "tensor_cast.permute_tokens",
            "tensor_cast.grouped_matmul",
            "tensor_cast.unpermute_tokens",
            "tensor_cast.all_to_all",
        ],
    }

    stats = compute_fused_op_stats(hit_details, miss_details, fused_groups)

    # 2 unique HITs (mm, swiglu) + 1 DFC group MISS + 1 embedding MISS = 4
    assert stats["fused_total"] == 4
    assert stats["fused_hit"] == 2
    assert stats["fused_miss"] == 2


def test_fused_op_hr_excludes_zero_cost():
    """Reference view should exclude zero_cost ops from count."""
    from tensor_cast.performance_model.empirical import compute_fused_op_stats

    hit_details = [
        "aten.mm.default->MatMulV2",
        "aten.view.default->zero_cost",
        "aten.permute.default->zero_cost",
    ]
    miss_details = [
        ("aten.embedding.default", "shape_mismatch", []),
    ]

    stats = compute_fused_op_stats(hit_details, miss_details, fused_groups={})

    # With zero_cost: 3 HITs + 1 MISS = 4 total
    assert stats["fused_total"] == 4
    assert stats["fused_hit"] == 3

    # Without zero_cost: 1 HIT + 1 MISS = 2 total
    assert stats["fused_total_no_zc"] == 2
    assert stats["fused_hit_no_zc"] == 1


def test_fused_op_hr_pessimistic_partial_shape():
    """Op that HITs for some shapes and MISSes for others → MISS (pessimistic).

    This prevents double-counting: an op in both hits and misses with different
    shapes should count as 1 MISS (not 1 HIT + 1 MISS inflating both).
    """
    from tensor_cast.performance_model.empirical import compute_fused_op_stats

    hit_details = [
        "tensor_cast.quantize.default->AscendQuantV2",  # shape A: HIT
        "aten.mm.default->MatMulV2",                     # shape X: HIT
        "aten.view.default->zero_cost",
    ]
    miss_details = [
        # quantize with shape B: MISS
        ("tensor_cast.quantize.default", "shape_mismatch", [(16, 128)]),
        # mm with shape Y: MISS
        ("aten.mm.default", "shape_mismatch", [(16, 7168)]),
        # embedding: pure MISS
        ("aten.embedding.default", "shape_mismatch", [(9496, 5120)]),
    ]

    stats = compute_fused_op_stats(hit_details, miss_details, fused_groups={})

    # Pessimistic: quantize has MISS → MISS, mm has MISS → MISS
    # Only view (zero_cost, no MISS) is a HIT
    # Total unique ops: quantize, mm, view, embedding = 4
    assert stats["fused_total"] == 4
    assert stats["fused_hit"] == 1   # only view (zero_cost)
    assert stats["fused_miss"] == 3  # quantize + mm + embedding

    # Without zero_cost: 0 HITs, 3 MISSes
    assert stats["fused_hit_no_zc"] == 0
    assert stats["fused_total_no_zc"] == 3
