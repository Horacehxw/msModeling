"""Unit tests for M4/M5/M6 evaluation metrics."""

from unittest.mock import MagicMock

import torch

from tensor_cast.performance_model.base import PerformanceModel
from tensor_cast.performance_model.empirical import (
    compute_per_shape_stats,
    EmpiricalPerformanceModel,
)
from tensor_cast.performance_model.profiling_database.data_source import (
    DataSource,
    QueryResult,
    QuerySource,
)


class TestM4PerShapeMatchRate:
    """M4: Per-Shape Match HR -- unique (func_name, shape) pairs, excl zero_cost."""

    def test_mixed_hit_miss(self):
        hit_details = [
            ("aten.mm.default", "MatMulV2", ((2048, 5120), (5120, 5120)), 45.3e-6),
            ("tensor_cast.swiglu.default", "SwiGlu", ((2048, 6912),), 12.1e-6),
        ]
        miss_details = [
            ("aten.mm.default", "shape_mismatch", [(4096, 5120), (5120, 5120)]),
            ("tensor_cast.swiglu.default", "shape_mismatch", [(4096, 6912)]),
        ]
        stats = compute_per_shape_stats(hit_details, miss_details)
        assert stats["m4_hit_shapes"] == 2
        assert stats["m4_total_shapes"] == 4
        assert abs(stats["m4_per_shape_hr"] - 0.5) < 1e-9

    def test_all_hit(self):
        hit_details = [
            ("aten.mm.default", "MatMulV2", ((2048, 5120), (5120, 5120)), 45.3e-6),
        ]
        stats = compute_per_shape_stats(hit_details, [])
        assert stats["m4_per_shape_hr"] == 1.0

    def test_all_miss(self):
        miss_details = [
            ("aten.mm.default", "shape_mismatch", [(2048, 5120), (5120, 5120)]),
        ]
        stats = compute_per_shape_stats([], miss_details)
        assert stats["m4_per_shape_hr"] == 0.0

    def test_zero_cost_excluded(self):
        hit_details = [
            ("aten.mm.default", "MatMulV2", ((2048, 5120), (5120, 5120)), 45.3e-6),
            ("aten.view.default", "zero_cost", ((2048, 5120),), 0.0),
            ("aten.permute.default", "zero_cost", ((2048, 5120),), 0.0),
        ]
        miss_details = [
            ("aten.mm.default", "shape_mismatch", [(4096, 5120), (5120, 5120)]),
        ]
        stats = compute_per_shape_stats(hit_details, miss_details)
        assert stats["m4_hit_shapes"] == 1
        assert stats["m4_total_shapes"] == 2
        assert abs(stats["m4_per_shape_hr"] - 0.5) < 1e-9

    def test_duplicate_shape_calls_unique(self):
        hit_details = [
            ("aten.mm.default", "MatMulV2", ((2048, 5120), (5120, 5120)), 45.3e-6),
            ("aten.mm.default", "MatMulV2", ((2048, 5120), (5120, 5120)), 45.3e-6),
            ("aten.mm.default", "MatMulV2", ((2048, 5120), (5120, 5120)), 45.3e-6),
        ]
        stats = compute_per_shape_stats(hit_details, [])
        assert stats["m4_hit_shapes"] == 1
        assert stats["m4_total_shapes"] == 1

    def test_empty_inputs(self):
        stats = compute_per_shape_stats([], [])
        assert stats["m4_per_shape_hr"] == 0.0
        assert stats["m4_hit_shapes"] == 0
        assert stats["m4_total_shapes"] == 0

    def test_miss_shape_list_sorted(self):
        miss_details = [
            ("z_op", "unmapped", [(10, 20)]),
            ("a_op", "unmapped", [(30, 40)]),
        ]
        stats = compute_per_shape_stats([], miss_details)
        assert stats["m4_miss_shape_list"][0][0] == "a_op"
        assert stats["m4_miss_shape_list"][1][0] == "z_op"


# --- M5: Simulated Latency Coverage ---


def _make_op(shape_pairs):
    """Create a mock OpInvokeInfo with given tensor shapes."""
    mock = MagicMock()
    mock.func = torch.ops.aten.mm.default
    mock.args = tuple(torch.empty(*s, device="meta") for s in shape_pairs)
    return mock


def _make_device():
    mock = MagicMock()
    mock.name = "TEST_DEVICE"
    return mock


class ControlledDataSource(DataSource):
    """DataSource that returns HIT for shapes in hit_set, MISS otherwise."""

    def __init__(self, hit_set: set):
        self.hit_set = hit_set
        self.last_miss_reason = "shape_mismatch"

    def lookup(self, op_invoke_info):
        shapes = tuple(
            tuple(a.shape) for a in op_invoke_info.args if isinstance(a, torch.Tensor)
        )
        if shapes in self.hit_set:
            return QueryResult(
                latency_us=100.0,
                confidence=1.0,
                source=QuerySource.MEASURED,
                details={"kernel_type": "MatMulV2"},
            )
        return None


class TestM5SimulatedLatencyCoverage:
    """M5: analytic-latency-weighted coverage of HIT ops."""

    def _make_model(self, hit_shapes, analytic_latency_s=50e-6):
        device = _make_device()
        ds = ControlledDataSource(hit_shapes)
        fallback = MagicMock(spec=PerformanceModel)
        fallback.process_op.return_value = PerformanceModel.Result(
            execution_time_s=analytic_latency_s,
        )
        return EmpiricalPerformanceModel(
            device, data_source=ds, fallback_model=fallback
        )

    def test_all_hit(self):
        shape_a = ((2048, 5120), (5120, 768))
        model = self._make_model(hit_shapes={shape_a})
        model.process_op(_make_op([(2048, 5120), (5120, 768)]))
        model.process_op(_make_op([(2048, 5120), (5120, 768)]))
        assert model._total_latency_sum > 0
        assert abs(model._hit_latency_sum / model._total_latency_sum - 1.0) < 1e-9

    def test_all_miss(self):
        model = self._make_model(hit_shapes=set())
        model.process_op(_make_op([(2048, 5120), (5120, 768)]))
        assert model._total_latency_sum > 0
        assert model._hit_latency_sum == 0.0

    def test_mixed_coverage(self):
        """2 HITs + 1 MISS, all same analytic weight -> M5 = 2/3."""
        shape_a = ((2048, 5120), (5120, 768))
        model = self._make_model(hit_shapes={shape_a})
        model.process_op(_make_op([(2048, 5120), (5120, 768)]))  # HIT
        model.process_op(_make_op([(2048, 5120), (5120, 768)]))  # HIT
        model.process_op(_make_op([(4096, 5120), (5120, 768)]))  # MISS
        m5 = model._hit_latency_sum / model._total_latency_sum
        assert abs(m5 - 2.0 / 3.0) < 1e-9

    def test_empty(self):
        model = self._make_model(hit_shapes=set())
        assert model._hit_latency_sum == 0.0
        assert model._total_latency_sum == 0.0


# --- export_hit_miss_report ---


class TestExportHitMissReport:
    """Tests for EmpiricalPerformanceModel.export_hit_miss_report()."""

    def _make_model(self, hit_shapes, analytic_latency_s=50e-6):
        device = _make_device()
        ds = ControlledDataSource(hit_shapes)
        fallback = MagicMock(spec=PerformanceModel)
        fallback.process_op.return_value = PerformanceModel.Result(
            execution_time_s=analytic_latency_s,
        )
        return EmpiricalPerformanceModel(
            device, data_source=ds, fallback_model=fallback
        )

    def test_report_structure(self):
        """Report contains all expected top-level keys."""
        shape_a = ((2048, 5120), (5120, 768))
        model = self._make_model(hit_shapes={shape_a})
        model.process_op(_make_op([(2048, 5120), (5120, 768)]))  # HIT
        model.process_op(_make_op([(4096, 5120), (5120, 768)]))  # MISS

        report = model.export_hit_miss_report()

        assert "m1" in report
        assert "m2" in report
        assert "m3" in report
        assert "m4" in report
        assert "m5" in report
        assert "hits" in report
        assert "misses" in report
        assert "m6_input" in report

    def test_m1_keys(self):
        shape_a = ((2048, 5120), (5120, 768))
        model = self._make_model(hit_shapes={shape_a})
        model.process_op(_make_op([(2048, 5120), (5120, 768)]))
        model.process_op(_make_op([(4096, 5120), (5120, 768)]))

        m1 = model.export_hit_miss_report()["m1"]
        assert m1["m1_hit"] == 1
        assert m1["m1_miss"] == 1
        assert m1["m1_total"] == 2
        assert abs(m1["m1_raw_op_count_hr"] - 0.5) < 1e-9

    def test_m6_input_tc_predicted_total(self):
        """m6_input contains tc_predicted_total_s when provided."""
        shape_a = ((2048, 5120), (5120, 768))
        model = self._make_model(hit_shapes={shape_a})
        model.process_op(_make_op([(2048, 5120), (5120, 768)]))

        report = model.export_hit_miss_report(tc_predicted_total_s=1.5)
        assert report["m6_input"]["tc_predicted_total_s"] == 1.5

    def test_m6_input_empirical_hit_total(self):
        """m6_input.empirical_hit_total_s accumulates across process_op calls."""
        shape_a = ((2048, 5120), (5120, 768))
        model = self._make_model(hit_shapes={shape_a})
        model.process_op(_make_op([(2048, 5120), (5120, 768)]))  # HIT
        model.process_op(_make_op([(2048, 5120), (5120, 768)]))  # HIT

        report = model.export_hit_miss_report()
        # ControlledDataSource returns 100.0 us = 100e-6 s per HIT
        expected = 100e-6 * 2
        assert abs(report["m6_input"]["empirical_hit_total_s"] - expected) < 1e-12

    def test_m6_input_default_none(self):
        """m6_input.tc_predicted_total_s is None when not provided."""
        shape_a = ((2048, 5120), (5120, 768))
        model = self._make_model(hit_shapes={shape_a})
        model.process_op(_make_op([(2048, 5120), (5120, 768)]))

        report = model.export_hit_miss_report()
        assert report["m6_input"]["tc_predicted_total_s"] is None

    def test_write_json(self, tmp_path):
        """export_hit_miss_report writes valid JSON when output_path given."""
        import json

        shape_a = ((2048, 5120), (5120, 768))
        model = self._make_model(hit_shapes={shape_a})
        model.process_op(_make_op([(2048, 5120), (5120, 768)]))

        out = tmp_path / "report.json"
        model.export_hit_miss_report(output_path=out)

        assert out.exists()
        data = json.loads(out.read_text())
        assert data["m1"]["m1_hit"] == 1
        assert "m6_input" in data

    def test_empty_report(self):
        """Report works with no ops processed."""
        model = self._make_model(hit_shapes=set())
        report = model.export_hit_miss_report()
        assert report["m1"]["m1_total"] == 0
        assert report["m5"]["m5_simulated_latency_coverage"] == 0.0
        assert report["m6_input"]["tc_predicted_total_s"] is None
        assert report["m6_input"]["empirical_hit_total_s"] == 0.0
