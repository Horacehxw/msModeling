"""Tests for validate.py."""

import csv
from pathlib import Path

import pytest

from tools.perf_data_collection.validate import validate_database, ValidationReport


COMPUTE_FIELDS = [
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
    "Output Formats",
    "Average Duration(us)",
]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] = COMPUTE_FIELDS):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def valid_db(tmp_path):
    """Create a valid database directory."""
    db = tmp_path / "db"
    db.mkdir()
    _write_csv(
        db / "MatMulV2.csv",
        [
            {
                "Input Shapes": '"100,512;512,1024"',
                "Input Data Types": '"DT_BF16;DT_BF16"',
                "Input Formats": '"ND;ND"',
                "Output Shapes": '"100,1024"',
                "Output Data Types": '"DT_BF16"',
                "Output Formats": '"ND"',
                "Average Duration(us)": "10.5",
            },
            {
                "Input Shapes": '"200,512;512,1024"',
                "Input Data Types": '"DT_BF16;DT_BF16"',
                "Input Formats": '"ND;ND"',
                "Output Shapes": '"200,1024"',
                "Output Data Types": '"DT_BF16"',
                "Output Formats": '"ND"',
                "Average Duration(us)": "20.3",
            },
        ],
    )
    _write_csv(
        db / "Add.csv",
        [
            {
                "Input Shapes": '"100,512;100,512"',
                "Input Data Types": '"DT_BF16;DT_BF16"',
                "Input Formats": '"ND;ND"',
                "Output Shapes": '"100,512"',
                "Output Data Types": '"DT_BF16"',
                "Output Formats": '"ND"',
                "Average Duration(us)": "1.2",
            },
        ],
    )
    return db


@pytest.fixture
def db_with_issues(tmp_path):
    """Create a database with quality issues."""
    db = tmp_path / "bad_db"
    db.mkdir()
    _write_csv(
        db / "MatMulV2.csv",
        [
            {
                "Input Shapes": '"100,512;512,1024"',
                "Input Data Types": '"DT_BF16;DT_BF16"',
                "Input Formats": '"ND;ND"',
                "Output Shapes": '"100,1024"',
                "Output Data Types": '"DT_BF16"',
                "Output Formats": '"ND"',
                "Average Duration(us)": "0.0",
            },  # zero duration
            {
                "Input Shapes": '""',
                "Input Data Types": '""',
                "Input Formats": '""',
                "Output Shapes": '""',
                "Output Data Types": '""',
                "Output Formats": '""',
                "Average Duration(us)": "5.0",
            },  # empty shapes
        ],
    )
    return db


# --- Tests ---


def test_valid_db_passes(valid_db):
    """Valid database should pass with no errors."""
    report = validate_database(valid_db)
    assert isinstance(report, ValidationReport)
    assert report.total_kernel_types == 2
    assert report.total_shapes == 3
    assert len(report.errors) == 0


def test_reports_zero_duration(db_with_issues):
    """Should flag rows with zero duration."""
    report = validate_database(db_with_issues)
    assert any("zero" in e.lower() or "duration" in e.lower() for e in report.warnings)


def test_reports_empty_shapes(db_with_issues):
    """Should flag rows with empty shapes."""
    report = validate_database(db_with_issues)
    assert any("empty" in e.lower() or "shape" in e.lower() for e in report.warnings)


def test_reports_kernel_stats(valid_db):
    """Report should include per-kernel statistics."""
    report = validate_database(valid_db)
    assert "MatMulV2" in report.kernel_stats
    assert report.kernel_stats["MatMulV2"]["shapes"] == 2
    assert "Add" in report.kernel_stats
    assert report.kernel_stats["Add"]["shapes"] == 1


def test_empty_directory(tmp_path):
    """Empty directory should produce empty report."""
    db = tmp_path / "empty"
    db.mkdir()
    report = validate_database(db)
    assert report.total_kernel_types == 0
    assert report.total_shapes == 0
    assert len(report.errors) == 0
