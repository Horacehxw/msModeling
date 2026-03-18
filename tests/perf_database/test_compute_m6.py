"""Tests for compute_m6.py (M6: Empirical Prediction Coverage)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "tools" / "perf_data_collection")
)
from compute_m6 import (
    compute_m6,
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
            ("allgatherAicpuKernel", "300.0"),  # Should be excluded
            ("hcom_allGather_", "150.0"),
            ("TensorMove", "10.0"),
        ]
    csv_path = tmp_path / "kernel_details.csv"
    lines = ["Type,Duration(us)"]
    for kt, dur in rows:
        lines.append(f"{kt},{dur}")
    csv_path.write_text("\n".join(lines))
    return csv_path


def _make_tc_report(hits=None, empirical_sum_s=0.001):
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
        "m6_input": {"empirical_hit_duration_sum_s": empirical_sum_s},
    }


class TestParseStepTraceTime:
    def test_basic(self, tmp_path):
        _make_step_trace_time(tmp_path, computing=3000.0, comm_no=2000.0)
        result = parse_step_trace_time(tmp_path)
        assert result["computing_us"] == 3000.0
        assert result["comm_not_overlapped_us"] == 2000.0
        assert result["denominator_us"] == 5000.0

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_step_trace_time(tmp_path)

    def test_empty_csv(self, tmp_path):
        csv_path = tmp_path / "step_trace_time.csv"
        csv_path.write_text("Device_id,Step,Computing,Communication(Not Overlapped)\n")
        with pytest.raises(ValueError, match="Empty"):
            parse_step_trace_time(tmp_path)

    def test_step_index_out_of_range(self, tmp_path):
        _make_step_trace_time(tmp_path)
        with pytest.raises(ValueError, match="out of range"):
            parse_step_trace_time(tmp_path, step_index=5)


class TestKernelDetailsDiagnostics:
    def test_excludes_aicpu_kernel(self, tmp_path):
        _make_kernel_details(tmp_path)
        unmatched = parse_kernel_details_diagnostics(
            tmp_path, hit_kernel_types={"MatMulV2", "SwiGlu"}
        )
        kernel_types = [k["kernel_type"] for k in unmatched]
        assert "allgatherAicpuKernel" not in kernel_types
        assert "FusedInferAttentionScore" in kernel_types
        assert "hcom_allGather_" in kernel_types

    def test_matched_excluded_from_unmatched(self, tmp_path):
        _make_kernel_details(tmp_path)
        unmatched = parse_kernel_details_diagnostics(
            tmp_path, hit_kernel_types={"MatMulV2", "SwiGlu", "hcom_allGather_"}
        )
        kernel_types = [k["kernel_type"] for k in unmatched]
        assert "MatMulV2" not in kernel_types
        assert "SwiGlu" not in kernel_types
        assert "hcom_allGather_" not in kernel_types

    def test_sorted_by_duration(self, tmp_path):
        _make_kernel_details(tmp_path)
        unmatched = parse_kernel_details_diagnostics(tmp_path, hit_kernel_types=set())
        durations = [k["duration_us"] for k in unmatched]
        assert durations == sorted(durations, reverse=True)

    def test_missing_file_returns_empty(self, tmp_path):
        result = parse_kernel_details_diagnostics(tmp_path, set())
        assert result == []


class TestComputeM6:
    def test_basic(self, tmp_path):
        _make_step_trace_time(tmp_path, computing=3000.0, comm_no=2000.0)
        _make_kernel_details(tmp_path)
        tc_report = _make_tc_report(empirical_sum_s=0.002)  # 2000 us

        result = compute_m6(tc_report, tmp_path)

        assert result["empirical_hit_sum_us"] == pytest.approx(2000.0)
        assert result["denominator_us"] == 5000.0
        # M6 = 2000 / 5000 = 0.4
        assert result["m6_empirical_prediction_coverage"] == pytest.approx(0.4)

    def test_zero_denominator(self, tmp_path):
        _make_step_trace_time(tmp_path, computing=0.0, comm_no=0.0)
        _make_kernel_details(tmp_path)
        tc_report = _make_tc_report(empirical_sum_s=0.001)

        result = compute_m6(tc_report, tmp_path)
        assert result["m6_empirical_prediction_coverage"] == 0.0

    def test_m6_can_exceed_100_percent(self, tmp_path):
        """M6 > 100% when microbench overestimates vs real-run."""
        _make_step_trace_time(tmp_path, computing=1000.0, comm_no=500.0)
        _make_kernel_details(tmp_path)
        # empirical sum = 2000 us > denominator 1500 us
        tc_report = _make_tc_report(empirical_sum_s=0.002)

        result = compute_m6(tc_report, tmp_path)
        assert result["m6_empirical_prediction_coverage"] > 1.0

    def test_unmatched_kernels_in_result(self, tmp_path):
        _make_step_trace_time(tmp_path)
        _make_kernel_details(tmp_path)
        tc_report = _make_tc_report()

        result = compute_m6(tc_report, tmp_path)
        unmatched_types = [k["kernel_type"] for k in result["unmatched_kernels"]]
        assert "FusedInferAttentionScore" in unmatched_types
        assert "MatMulV2" not in unmatched_types  # HIT
