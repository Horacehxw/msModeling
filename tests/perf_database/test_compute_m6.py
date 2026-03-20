"""Tests for compute_m6.py (M6: Empirical E2E Prediction Ratio)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "tools" / "perf_data_collection")
)
from compute_m6 import (
    compute_m6,
    estimate_forward_passes,
    parse_kernel_details_diagnostics,
    parse_step_trace_time,
)


def _make_step_trace_time(tmp_path, computing=3000.0, comm_no=2000.0):
    """Create a step_trace_time.csv fixture."""
    csv_path = tmp_path / "step_trace_time.csv"
    csv_path.write_text(
        "Device_id,Step,Computing,Communication(Not Overlapped),Overlapped,"
        "Communication,Free,Stage,Bubble,"
        "Communication(Not Overlapped and Exclude Receive),Preparing\n"
        f"0,,{computing},{comm_no},0.0,{comm_no},100.0,"
        f"{computing + comm_no + 100.0},0,{comm_no},50.0\n"
    )
    return csv_path


def _make_kernel_details(tmp_path, rows=None):
    """Create a kernel_details.csv fixture."""
    if rows is None:
        rows = [
            ("MatMulV2", "100.0"),
            ("SwiGlu", "50.0"),
            ("FusedInferAttentionScore", "200.0"),
            ("allgatherAicpuKernel", "300.0"),
            ("hcom_allGather_", "150.0"),
            ("TensorMove", "10.0"),
        ]
    csv_path = tmp_path / "kernel_details.csv"
    lines = ["Type,Duration(us)"]
    for kt, dur in rows:
        lines.append(f"{kt},{dur}")
    csv_path.write_text("\n".join(lines))
    return csv_path


def _make_tc_report(hits=None, empirical_hit_total_s=0.005, tc_predicted_total_s=0.006):
    """Create a TC report dict fixture."""
    if hits is None:
        hits = [
            {
                "func_name": "aten.mm.default",
                "kernel_type": "MatMulV2",
                "tc_shapes": [[2048, 5120]],
                "empirical_duration_s": 0.0005,
            },
            {
                "func_name": "tensor_cast.swiglu.default",
                "kernel_type": "SwiGlu",
                "tc_shapes": [[2048, 6912]],
                "empirical_duration_s": 0.0005,
            },
        ]
    return {
        "hits": hits,
        "misses": [],
        "m6_input": {
            "empirical_hit_total_s": empirical_hit_total_s,
            "tc_predicted_total_s": tc_predicted_total_s,
        },
    }


class TestParseStepTraceTime:
    def test_basic(self, tmp_path):
        _make_step_trace_time(tmp_path, computing=3000.0, comm_no=2000.0)
        result = parse_step_trace_time(tmp_path)
        assert result["computing_us"] == 3000.0
        assert result["comm_not_overlapped_us"] == 2000.0
        assert result["step_total_us"] == 5000.0

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_step_trace_time(tmp_path)

    def test_empty_csv(self, tmp_path):
        csv_path = tmp_path / "step_trace_time.csv"
        csv_path.write_text(
            "Device_id,Step,Computing,Communication(Not Overlapped)\n"
        )
        with pytest.raises(ValueError, match="Empty"):
            parse_step_trace_time(tmp_path)


class TestEstimateForwardPasses:
    def test_auto_detect_by_delimiter(self, tmp_path):
        """ArgMaxV2 appears 5 times → 5 forward passes."""
        rows = (
            [("MatMulV2", "100.0")] * 50
            + [("ArgMaxV2", "1.0")] * 5
        )
        _make_kernel_details(tmp_path, rows)
        result = estimate_forward_passes(tmp_path, delimiter="ArgMaxV2")
        assert result["n_forward_passes"] == 5
        assert result["delimiter"] == "ArgMaxV2"
        assert result["avg_fwd_duration_us"] == pytest.approx(5005.0 / 5)

    def test_custom_delimiter(self, tmp_path):
        """Works with any delimiter kernel type."""
        rows = (
            [("MatMulV2", "100.0")] * 30
            + [("MyCustomSampling", "2.0")] * 3
        )
        _make_kernel_details(tmp_path, rows)
        result = estimate_forward_passes(tmp_path, delimiter="MyCustomSampling")
        assert result["n_forward_passes"] == 3

    def test_no_delimiter_found(self, tmp_path):
        """No delimiter kernel → defaults to 1 forward pass."""
        rows = [("MatMulV2", "100.0")] * 10
        _make_kernel_details(tmp_path, rows)
        result = estimate_forward_passes(tmp_path, delimiter="ArgMaxV2")
        assert result["n_forward_passes"] == 1

    def test_missing_file_returns_1(self, tmp_path):
        result = estimate_forward_passes(tmp_path)
        assert result["n_forward_passes"] == 1

    def test_excludes_aicpu_kernel(self, tmp_path):
        """AicpuKernel entries excluded from total duration."""
        rows = [
            ("MatMulV2", "100.0"),
            ("allgatherAicpuKernel", "999.0"),
            ("ArgMaxV2", "1.0"),
        ]
        _make_kernel_details(tmp_path, rows)
        result = estimate_forward_passes(tmp_path)
        assert result["total_kernel_duration_us"] == pytest.approx(101.0)


class TestKernelDetailsDiagnostics:
    def test_excludes_aicpu_kernel(self, tmp_path):
        _make_kernel_details(tmp_path)
        unmatched = parse_kernel_details_diagnostics(
            tmp_path, hit_kernel_types={"MatMulV2", "SwiGlu"}
        )
        kernel_types = [k["kernel_type"] for k in unmatched]
        assert "allgatherAicpuKernel" not in kernel_types
        assert "FusedInferAttentionScore" in kernel_types

    def test_sorted_by_duration(self, tmp_path):
        _make_kernel_details(tmp_path)
        unmatched = parse_kernel_details_diagnostics(
            tmp_path, hit_kernel_types=set()
        )
        durations = [k["duration_us"] for k in unmatched]
        assert durations == sorted(durations, reverse=True)

    def test_missing_file_returns_empty(self, tmp_path):
        result = parse_kernel_details_diagnostics(tmp_path, set())
        assert result == []


class TestComputeM6:
    def _setup(self, tmp_path, n_fwd=1, kernel_dur=5000.0, emp_hit=0.005):
        """Helper: create fixtures with n_fwd forward passes."""
        _make_step_trace_time(tmp_path)
        rows = (
            [("MatMulV2", str(kernel_dur / n_fwd))] * n_fwd
            + [("ArgMaxV2", "1.0")] * n_fwd
        )
        _make_kernel_details(tmp_path, rows)
        return _make_tc_report(empirical_hit_total_s=emp_hit)

    def test_perfect_prediction(self, tmp_path):
        """Empirical = real per-fwd → M6 = 1.0."""
        # 1 fwd pass, total kernel = 5001 us, emp = 5001 us
        tc_report = self._setup(tmp_path, n_fwd=1, kernel_dur=5000.0, emp_hit=5001e-6)
        result = compute_m6(tc_report, tmp_path)
        assert result["m6_ratio"] == pytest.approx(1.0, rel=1e-3)

    def test_overestimate(self, tmp_path):
        """Empirical > real → M6 > 1."""
        # 1 fwd pass, kernel = 5001 us, emp = 10000 us
        tc_report = self._setup(tmp_path, n_fwd=1, kernel_dur=5000.0, emp_hit=0.010)
        result = compute_m6(tc_report, tmp_path)
        assert result["m6_ratio"] > 1.0

    def test_underestimate(self, tmp_path):
        """Empirical < real → M6 < 1."""
        # 1 fwd pass, kernel = 5001 us, emp = 2500 us
        tc_report = self._setup(tmp_path, n_fwd=1, kernel_dur=5000.0, emp_hit=0.0025)
        result = compute_m6(tc_report, tmp_path)
        assert result["m6_ratio"] < 1.0

    def test_multiple_forward_passes(self, tmp_path):
        """5 fwd passes detected → normalize correctly."""
        # 5 fwd, total kernel = 5000 + 5 = 5005 us, per-fwd = 1001 us
        tc_report = self._setup(tmp_path, n_fwd=5, kernel_dur=5000.0, emp_hit=1001e-6)
        result = compute_m6(tc_report, tmp_path)
        assert result["n_forward_passes"] == 5
        assert result["m6_ratio"] == pytest.approx(1.0, rel=1e-3)

    def test_n_forward_passes_override(self, tmp_path):
        """User override takes precedence over auto-detect."""
        tc_report = self._setup(tmp_path, n_fwd=5, kernel_dur=5000.0, emp_hit=1001e-6)
        # Override to 10 → per-fwd halved → M6 doubled
        result = compute_m6(tc_report, tmp_path, n_forward_passes_override=10)
        assert result["n_forward_passes"] == 10
        assert result["delimiter"] == "user_override"

    def test_null_empirical_raises(self, tmp_path):
        _make_step_trace_time(tmp_path)
        rows = [("MatMulV2", "100.0"), ("ArgMaxV2", "1.0")]
        _make_kernel_details(tmp_path, rows)
        tc_report = {
            "hits": [],
            "misses": [],
            "m6_input": {"empirical_hit_total_s": None},
        }
        with pytest.raises(ValueError, match="empirical_hit_total_s is missing"):
            compute_m6(tc_report, tmp_path)

    def test_phase3_pass(self, tmp_path):
        """M6 within [0.85, 1.15] passes Phase 3."""
        # 1 fwd, kernel = 10001 us, emp = 9000 us → M6 ≈ 0.9
        tc_report = self._setup(tmp_path, n_fwd=1, kernel_dur=10000.0, emp_hit=0.009)
        result = compute_m6(tc_report, tmp_path)
        assert 0.85 <= result["m6_ratio"] <= 1.15

    def test_unmatched_kernels_in_result(self, tmp_path):
        _make_step_trace_time(tmp_path)
        _make_kernel_details(tmp_path)
        tc_report = _make_tc_report()
        result = compute_m6(tc_report, tmp_path)
        unmatched_types = [k["kernel_type"] for k in result["unmatched_kernels"]]
        assert "FusedInferAttentionScore" in unmatched_types
        assert "MatMulV2" not in unmatched_types  # HIT
