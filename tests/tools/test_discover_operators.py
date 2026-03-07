"""Tests for discover_operators.py."""

import csv
from pathlib import Path

import pytest
import yaml

from tools.perf_data_collection.discover_operators import discover_operators


# --- Fixtures ---

SAMPLE_OP_MAPPING = {
    "version": "test",
    "device": "TEST_DEVICE",
    "operator_mappings": {
        "aten.mm.default": {"kernel_type": "MatMulV2"},
        "aten.add.Tensor": {"kernel_type": "Add"},
        "tensor_cast.swiglu": {"kernel_type": "SwiGlu"},
        "tensor_cast.attention.default": {
            "kernel_type": "FusedInferAttentionScore",
            "query_mode": "attention_special",
        },
        "tensor_cast.all_reduce.default": {
            "kernel_type": "hcom_allReduce_",
            "category": "communication",
        },
        "tensor_cast.matmul_all_reduce": {
            "composite": True,
            "sub_kernels": ["MatMulV2", "hcom_allReduce_"],
        },
        "aten.view.default": {"zero_cost": True},
    },
}

KERNEL_DETAILS_HEADER = [
    "Type",
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
    "Output Formats",
    "Duration(us)",
    "OP State",
    "Accelerator Core",
]

KERNEL_DETAILS_ROWS = [
    # Known kernel types
    {"Type": "MatMulV2", "Duration(us)": "10.0"},
    {"Type": "MatMulV2", "Duration(us)": "20.0"},
    {"Type": "MatMulV2", "Duration(us)": "15.0"},
    {"Type": "Add", "Duration(us)": "1.0"},
    {"Type": "Add", "Duration(us)": "2.0"},
    {"Type": "SwiGlu", "Duration(us)": "5.0"},
    {"Type": "FusedInferAttentionScore", "Duration(us)": "100.0"},
    {"Type": "hcom_allReduce_", "Duration(us)": "50.0"},
    {"Type": "hcom_allReduce_", "Duration(us)": "60.0"},
    # Unknown kernel types
    {"Type": "QuantBatchMatmulV3", "Duration(us)": "30.0"},
    {"Type": "QuantBatchMatmulV3", "Duration(us)": "25.0"},
    {"Type": "UnknownKernel", "Duration(us)": "0.5"},
    # TensorMove (not in mapping)
    {"Type": "TensorMove", "Duration(us)": "0.1"},
    {"Type": "TensorMove", "Duration(us)": "0.2"},
]


def _write_kernel_csv(path: Path, rows: list[dict]):
    """Write a kernel_details.csv with minimal required columns."""
    defaults = dict.fromkeys(KERNEL_DETAILS_HEADER, "")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KERNEL_DETAILS_HEADER)
        writer.writeheader()
        for row in rows:
            merged = {**defaults, **row}
            writer.writerow(merged)


@pytest.fixture
def test_data(tmp_path):
    """Create test op_mapping.yaml and kernel_details.csv."""
    mapping_path = tmp_path / "op_mapping.yaml"
    with mapping_path.open("w") as f:
        yaml.dump(SAMPLE_OP_MAPPING, f)

    csv_path = tmp_path / "kernel_details.csv"
    _write_kernel_csv(csv_path, KERNEL_DETAILS_ROWS)

    return mapping_path, csv_path


# --- Tests ---


def test_discover_returns_known_and_unknown(test_data):
    """Basic discovery: should separate known from unknown kernel types."""
    mapping_path, csv_path = test_data
    result = discover_operators(csv_path, mapping_path)

    assert "known" in result
    assert "unknown" in result
    assert isinstance(result["known"], list)
    assert isinstance(result["unknown"], list)


def test_known_kernel_types(test_data):
    """Known types should include all mapped kernel_types found in profiling."""
    mapping_path, csv_path = test_data
    result = discover_operators(csv_path, mapping_path)

    known_types = {k["kernel_type"] for k in result["known"]}
    assert "MatMulV2" in known_types
    assert "Add" in known_types
    assert "SwiGlu" in known_types
    assert "FusedInferAttentionScore" in known_types
    assert "hcom_allReduce_" in known_types


def test_unknown_kernel_types(test_data):
    """Unknown types should include profiling types not in op_mapping."""
    mapping_path, csv_path = test_data
    result = discover_operators(csv_path, mapping_path)

    unknown_types = {u["kernel_type"] for u in result["unknown"]}
    assert "QuantBatchMatmulV3" in unknown_types
    assert "UnknownKernel" in unknown_types
    assert "TensorMove" in unknown_types


def test_call_counts(test_data):
    """Each entry should have correct call count."""
    mapping_path, csv_path = test_data
    result = discover_operators(csv_path, mapping_path)

    known_map = {k["kernel_type"]: k for k in result["known"]}
    assert known_map["MatMulV2"]["count"] == 3
    assert known_map["Add"]["count"] == 2
    assert known_map["hcom_allReduce_"]["count"] == 2

    unknown_map = {u["kernel_type"]: u for u in result["unknown"]}
    assert unknown_map["QuantBatchMatmulV3"]["count"] == 2
    assert unknown_map["TensorMove"]["count"] == 2


def test_total_duration(test_data):
    """Each entry should have total duration."""
    mapping_path, csv_path = test_data
    result = discover_operators(csv_path, mapping_path)

    known_map = {k["kernel_type"]: k for k in result["known"]}
    assert abs(known_map["MatMulV2"]["total_duration_us"] - 45.0) < 0.01
    assert abs(known_map["hcom_allReduce_"]["total_duration_us"] - 110.0) < 0.01


def test_coverage_stats(test_data):
    """Result should include coverage statistics."""
    mapping_path, csv_path = test_data
    result = discover_operators(csv_path, mapping_path)

    assert "coverage" in result
    cov = result["coverage"]
    # 5 known types out of 8 total types
    assert cov["known_types"] == 5
    assert cov["unknown_types"] == 3
    assert cov["total_types"] == 8
    # Coverage by call count: 9 known calls out of 14 total
    assert cov["known_calls"] == 9
    assert cov["total_calls"] == 14
    assert abs(cov["call_coverage_pct"] - 9 / 14 * 100) < 0.1


def test_duration_coverage(test_data):
    """Coverage should include duration-weighted coverage."""
    mapping_path, csv_path = test_data
    result = discover_operators(csv_path, mapping_path)

    cov = result["coverage"]
    total_dur = sum(float(r["Duration(us)"]) for r in KERNEL_DETAILS_ROWS)
    known_dur = 10 + 20 + 15 + 1 + 2 + 5 + 100 + 50 + 60  # 263.0
    assert abs(cov["known_duration_us"] - known_dur) < 0.01
    assert abs(cov["total_duration_us"] - total_dur) < 0.01
    assert abs(cov["duration_coverage_pct"] - known_dur / total_dur * 100) < 0.1


def test_sorted_by_duration(test_data):
    """Results should be sorted by total_duration_us descending."""
    mapping_path, csv_path = test_data
    result = discover_operators(csv_path, mapping_path)

    known_durations = [k["total_duration_us"] for k in result["known"]]
    assert known_durations == sorted(known_durations, reverse=True)

    unknown_durations = [u["total_duration_us"] for u in result["unknown"]]
    assert unknown_durations == sorted(unknown_durations, reverse=True)


def test_empty_profiling(tmp_path):
    """Empty profiling data should return empty results."""
    mapping_path = tmp_path / "op_mapping.yaml"
    with mapping_path.open("w") as f:
        yaml.dump(SAMPLE_OP_MAPPING, f)

    csv_path = tmp_path / "kernel_details.csv"
    _write_kernel_csv(csv_path, [])

    result = discover_operators(csv_path, mapping_path)
    assert result["known"] == []
    assert result["unknown"] == []
    assert result["coverage"]["total_types"] == 0


def test_sub_kernels_counted_as_known(test_data):
    """Kernel types referenced in sub_kernels should be treated as known."""
    mapping_path, csv_path = test_data
    result = discover_operators(csv_path, mapping_path)

    # MatMulV2 and hcom_allReduce_ are both in sub_kernels AND primary mappings
    known_types = {k["kernel_type"] for k in result["known"]}
    assert "MatMulV2" in known_types
    assert "hcom_allReduce_" in known_types
