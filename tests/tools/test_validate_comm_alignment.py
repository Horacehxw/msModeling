"""Tests for validate_comm_alignment.py (C9 comm data alignment)."""

import csv
import math
from pathlib import Path

import pytest

from tools.perf_data_collection.validate_comm_alignment import (
    _A3_TOPOLOGIES,
    _CSV_TO_OP,
    AlignmentReport,
    AlignmentRow,
    analytic_predict_us,
    validate_csv,
    validate_directory,
)


# ---------------------------------------------------------------------------
# analytic_predict_us unit tests
# ---------------------------------------------------------------------------


class TestAnalyticPredictUs:
    """Verify analytic formulas match CommAnalyticModel logic."""

    def _topo(self, tier: int):
        return _A3_TOPOLOGIES[tier]

    def test_all_reduce_ring_dominates_large_message(self):
        """Large message → ring algorithm selected (bandwidth-bound)."""
        # 128 MB, 16 devices, tier=1 (intra_pod)
        result = analytic_predict_us("all_reduce", 128 * 1024 * 1024, 16, 1)
        assert result > 0

        topo = self._topo(1)
        bw = topo.bandwidth_bytes_ps * topo.comm_efficiency
        lat = topo.latency_s
        n, m = 16, 128 * 1024 * 1024
        time_ring = (2 * (n - 1) * lat + 2 * (n - 1) * m / n / bw) * 1e6
        time_tree = (2 * math.log2(n) * lat + 2 * m / bw) * 1e6
        assert abs(result - min(time_ring, time_tree)) < 0.01

    def test_all_reduce_tree_dominates_small_message(self):
        """Small message → tree algorithm selected (latency-bound)."""
        # 1 KB, 16 devices, tier=1
        result = analytic_predict_us("all_reduce", 1024, 16, 1)
        assert result > 0

        topo = self._topo(1)
        bw = topo.bandwidth_bytes_ps * topo.comm_efficiency
        lat = topo.latency_s
        n, m = 16, 1024
        time_ring = (2 * (n - 1) * lat + 2 * (n - 1) * m / n / bw) * 1e6
        time_tree = (2 * math.log2(n) * lat + 2 * m / bw) * 1e6
        assert abs(result - min(time_ring, time_tree)) < 0.01

    def test_all_gather_formula(self):
        """all_gather: min(ring, recursive_doubling)."""
        m, n, tier = 655360, 16, 1
        result = analytic_predict_us("all_gather", m, n, tier)
        topo = self._topo(tier)
        bw = topo.bandwidth_bytes_ps * topo.comm_efficiency
        lat = topo.latency_s
        time_ring = ((n - 1) * lat + (n - 1) * m / bw) * 1e6
        time_rec = (math.log2(n) * lat + (n - 1) * m / bw) * 1e6
        assert abs(result - min(time_ring, time_rec)) < 0.01

    def test_reduce_scatter_formula(self):
        """reduce_scatter: min(ring, recursive_halving)."""
        m, n, tier = 1310720, 16, 1
        result = analytic_predict_us("reduce_scatter", m, n, tier)
        topo = self._topo(tier)
        bw = topo.bandwidth_bytes_ps * topo.comm_efficiency
        lat = topo.latency_s
        time_ring = ((n - 1) * lat + (n - 1) * m / n / bw) * 1e6
        time_rec = (math.log2(n) * lat + (n - 1) * m / n / bw) * 1e6
        assert abs(result - min(time_ring, time_rec)) < 0.01

    def test_all_to_all_formula(self):
        """all_to_all: min(pairwise, bruck)."""
        m, n, tier = 262144, 8, 1
        result = analytic_predict_us("all_to_all", m, n, tier)
        topo = self._topo(tier)
        bw = topo.bandwidth_bytes_ps * topo.comm_efficiency
        lat = topo.latency_s
        time_pairwise = ((n - 1) * lat + m / bw) * 1e6
        time_bruck = (math.log2(n) * lat + m / bw) * 1e6
        assert abs(result - min(time_pairwise, time_bruck)) < 0.01

    def test_single_device_returns_zero(self):
        assert analytic_predict_us("all_reduce", 1024, 1, 1) == 0.0

    def test_die_level_tier2_faster_than_intra_pod_tier1(self):
        """tier=2 (SIO, 0.2µs latency) should be faster than tier=1 (0.5µs)."""
        m, n = 65536, 2
        t1 = analytic_predict_us("all_reduce", m, n, 1)
        t2 = analytic_predict_us("all_reduce", m, n, 2)
        assert t2 < t1

    def test_unknown_op_raises(self):
        with pytest.raises(ValueError, match="Unknown op_type"):
            analytic_predict_us("unknown_op", 1024, 8, 1)


# ---------------------------------------------------------------------------
# AlignmentRow tests
# ---------------------------------------------------------------------------


class TestAlignmentRow:
    def _row(self, measured, predicted):
        return AlignmentRow(
            op_type="all_reduce",
            message_bytes=1024,
            num_devices=16,
            topology_tier=1,
            measured_us=measured,
            predicted_us=predicted,
        )

    def test_ratio_exact(self):
        assert self._row(100.0, 100.0).ratio == pytest.approx(1.0)

    def test_ratio_2x(self):
        assert self._row(200.0, 100.0).ratio == pytest.approx(2.0)

    def test_ratio_half(self):
        assert self._row(50.0, 100.0).ratio == pytest.approx(0.5)

    def test_status_pass_within_tolerance(self):
        assert self._row(150.0, 100.0).status(2.0) == "PASS"
        assert self._row(60.0, 100.0).status(2.0) == "PASS"

    def test_status_warn_outside_tolerance_within_4x(self):
        assert self._row(250.0, 100.0).status(2.0) == "WARN"
        assert self._row(30.0, 100.0).status(2.0) == "WARN"

    def test_status_fail_beyond_4x(self):
        assert self._row(500.0, 100.0).status(2.0) == "FAIL"
        assert self._row(10.0, 100.0).status(2.0) == "FAIL"

    def test_ratio_zero_predicted(self):
        assert self._row(100.0, 0.0).ratio == float("inf")


# ---------------------------------------------------------------------------
# AlignmentReport tests
# ---------------------------------------------------------------------------


class TestAlignmentReport:
    def _make_report(self, ratios, tolerance=2.0):
        rows = [
            AlignmentRow("all_reduce", 1024, 16, 1, r * 100.0, 100.0) for r in ratios
        ]
        return AlignmentReport(rows=rows, tolerance=tolerance)

    def test_all_pass(self):
        report = self._make_report([1.0, 1.5, 0.8])
        assert report.pass_count == 3
        assert report.warn_count == 0
        assert report.fail_count == 0
        assert report.ok()

    def test_mixed(self):
        report = self._make_report([1.0, 2.5, 6.0])
        assert report.pass_count == 1
        assert report.warn_count == 1
        assert report.fail_count == 1
        assert not report.ok()

    def test_mean_ratio(self):
        report = self._make_report([1.0, 2.0, 3.0])
        assert report.mean_ratio == pytest.approx(2.0)

    def test_empty_report(self):
        report = AlignmentReport(rows=[], tolerance=2.0)
        assert report.pass_count == 0
        assert report.ok()
        assert math.isnan(report.mean_ratio)


# ---------------------------------------------------------------------------
# validate_csv integration tests
# ---------------------------------------------------------------------------


def _write_comm_csv(path: Path, rows: list):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "message_bytes",
                "num_devices",
                "dtype",
                "topology_tier",
                "Duration(us)",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def comm_csv_dir(tmp_path):
    """Write synthetic HCCL CSVs with known measured values."""
    # Use analytic predictions as measured values → ratio should be ~1.0 (PASS)
    rows_allreduce = []
    for msg_bytes in [65536, 1310720]:
        predicted = analytic_predict_us("all_reduce", msg_bytes, 16, 1)
        rows_allreduce.append(
            {
                "message_bytes": msg_bytes,
                "num_devices": 16,
                "dtype": "DT_BF16",
                "topology_tier": 1,
                "Duration(us)": f"{predicted:.2f}",
            }
        )
    _write_comm_csv(tmp_path / "hcom_allReduce_.csv", rows_allreduce)

    rows_allgather = []
    for msg_bytes in [655360]:
        predicted = analytic_predict_us("all_gather", msg_bytes, 16, 1)
        rows_allgather.append(
            {
                "message_bytes": msg_bytes,
                "num_devices": 16,
                "dtype": "DT_BF16",
                "topology_tier": 1,
                "Duration(us)": f"{predicted:.2f}",
            }
        )
    _write_comm_csv(tmp_path / "hcom_allGather_.csv", rows_allgather)

    return tmp_path


def test_validate_csv_all_pass(comm_csv_dir):
    """When measured == predicted, all rows should PASS."""
    report = validate_csv(
        comm_csv_dir / "hcom_allReduce_.csv", "all_reduce", tolerance=2.0
    )
    assert report.fail_count == 0
    assert report.warn_count == 0
    assert report.pass_count == 2
    assert report.ok()


def test_validate_csv_ratio_near_one(comm_csv_dir):
    """Measured values equal to analytic predictions → ratio ≈ 1.0."""
    report = validate_csv(
        comm_csv_dir / "hcom_allReduce_.csv", "all_reduce", tolerance=2.0
    )
    for row in report.rows:
        assert abs(row.ratio - 1.0) < 0.01, f"Expected ratio≈1.0, got {row.ratio:.3f}"


def test_validate_csv_fail_on_large_discrepancy(tmp_path):
    """Measured 10x predicted → FAIL."""
    predicted = analytic_predict_us("all_reduce", 1310720, 16, 1)
    _write_comm_csv(
        tmp_path / "hcom_allReduce_.csv",
        [
            {
                "message_bytes": 1310720,
                "num_devices": 16,
                "dtype": "DT_BF16",
                "topology_tier": 1,
                "Duration(us)": f"{predicted * 10:.2f}",
            }
        ],
    )
    report = validate_csv(tmp_path / "hcom_allReduce_.csv", "all_reduce", tolerance=2.0)
    assert report.fail_count == 1
    assert not report.ok()


def test_validate_directory_skips_missing_csv(comm_csv_dir):
    """Missing CSVs are skipped (not an error)."""
    # comm_csv_dir only has allReduce and allGather
    reports, all_ok = validate_directory(comm_csv_dir, tolerance=2.0)
    assert "all_reduce" in reports
    assert "all_gather" in reports
    # reduce_scatter and all_to_all are missing → skipped, not in reports
    assert "reduce_scatter" not in reports
    assert "all_to_all" not in reports
    assert all_ok


def test_validate_directory_all_ok_when_all_pass(comm_csv_dir):
    reports, all_ok = validate_directory(comm_csv_dir, tolerance=2.0)
    assert all_ok


def test_validate_directory_not_ok_when_fail(tmp_path):
    """Directory with a failing CSV → all_ok=False."""
    predicted = analytic_predict_us("all_reduce", 1310720, 16, 1)
    _write_comm_csv(
        tmp_path / "hcom_allReduce_.csv",
        [
            {
                "message_bytes": 1310720,
                "num_devices": 16,
                "dtype": "DT_BF16",
                "topology_tier": 1,
                "Duration(us)": f"{predicted * 10:.2f}",
            }
        ],
    )
    _, all_ok = validate_directory(tmp_path, tolerance=2.0)
    assert not all_ok


def test_csv_to_op_mapping_covers_all_four_ops():
    """All four HCCL op types must be covered."""
    assert set(_CSV_TO_OP.values()) == {
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "all_to_all",
    }


def test_topology_tier2_die_level_params():
    """tier=2 should use 224 GB/s bandwidth and 0.2µs latency."""
    topo = _A3_TOPOLOGIES[2]
    assert topo.bandwidth_bytes_ps == pytest.approx(224e9)
    assert topo.latency_s == pytest.approx(0.2e-6)


def test_topology_tier1_intra_pod_params():
    """tier=1 should use 196 GB/s bandwidth and 0.5µs latency."""
    topo = _A3_TOPOLOGIES[1]
    assert topo.bandwidth_bytes_ps == pytest.approx(196e9)
    assert topo.latency_s == pytest.approx(0.5e-6)
