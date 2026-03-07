"""Tests for build_database.py."""

import csv
from pathlib import Path

import pytest

from tools.perf_data_collection.build_database import build_database, DatabaseReport


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    """Write a CSV file."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


COMPUTE_FIELDS = [
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
    "Output Formats",
    "Average Duration(us)",
]


@pytest.fixture
def db_dirs(tmp_path):
    """Create source CSVs and target data directory."""
    # Source 1: profiling data
    src1 = tmp_path / "source1"
    src1.mkdir()
    _write_csv(
        src1 / "MatMulV2.csv",
        [
            {
                "Input Shapes": '"100,512;512,1024"',
                "Input Data Types": '"DT_BF16;DT_BF16"',
                "Input Formats": '"ND;ND"',
                "Output Shapes": '"100,1024"',
                "Output Data Types": '"DT_BF16"',
                "Output Formats": '"ND"',
                "Average Duration(us)": "10.0",
            },
        ],
        COMPUTE_FIELDS,
    )

    # Source 2: microbenchmark data (different shape for same kernel + overlapping shape)
    src2 = tmp_path / "source2"
    src2.mkdir()
    _write_csv(
        src2 / "MatMulV2.csv",
        [
            {
                "Input Shapes": '"200,512;512,1024"',
                "Input Data Types": '"DT_BF16;DT_BF16"',
                "Input Formats": '"ND;ND"',
                "Output Shapes": '"200,1024"',
                "Output Data Types": '"DT_BF16"',
                "Output Formats": '"ND"',
                "Average Duration(us)": "20.0",
            },
            {
                "Input Shapes": '"100,512;512,1024"',
                "Input Data Types": '"DT_BF16;DT_BF16"',
                "Input Formats": '"ND;ND"',
                "Output Shapes": '"100,1024"',
                "Output Data Types": '"DT_BF16"',
                "Output Formats": '"ND"',
                "Average Duration(us)": "11.0",
            },  # duplicate shape, should be skipped
        ],
        COMPUTE_FIELDS,
    )

    # Source 2 also has a new kernel type
    _write_csv(
        src2 / "Add.csv",
        [
            {
                "Input Shapes": '"100,512;100,512"',
                "Input Data Types": '"DT_BF16;DT_BF16"',
                "Input Formats": '"ND;ND"',
                "Output Shapes": '"100,512"',
                "Output Data Types": '"DT_BF16"',
                "Output Formats": '"ND"',
                "Average Duration(us)": "1.0",
            },
        ],
        COMPUTE_FIELDS,
    )

    # Target data directory (empty)
    target = tmp_path / "data"
    target.mkdir()

    return [src1, src2], target


# --- Tests ---


def test_build_creates_target_csvs(db_dirs):
    """Build should create CSVs in target directory."""
    sources, target = db_dirs
    report = build_database(sources, target)

    assert (target / "MatMulV2.csv").exists()
    assert (target / "Add.csv").exists()


def test_build_merges_shapes(db_dirs):
    """Build should merge shapes from multiple sources without duplicates."""
    sources, target = db_dirs
    build_database(sources, target)

    with (target / "MatMulV2.csv").open() as f:
        rows = list(csv.DictReader(f))
    # 2 unique shapes: 100x512 and 200x512
    assert len(rows) == 2


def test_build_first_source_wins(db_dirs):
    """When shapes overlap, first source's duration should be kept."""
    sources, target = db_dirs
    build_database(sources, target)

    with (target / "MatMulV2.csv").open() as f:
        rows = list(csv.DictReader(f))
    # The 100x512 shape should have duration 10.0 (from source1), not 11.0
    row_100 = next(r for r in rows if "100,512" in r["Input Shapes"])
    assert abs(float(row_100["Average Duration(us)"]) - 10.0) < 0.01


def test_build_report(db_dirs):
    """Build should return a report with statistics."""
    sources, target = db_dirs
    report = build_database(sources, target)

    assert isinstance(report, DatabaseReport)
    assert report.total_kernel_types == 2  # MatMulV2 + Add
    assert report.total_shapes >= 3  # 2 MatMulV2 + 1 Add


def test_build_preserves_existing_target(db_dirs):
    """If target already has data, build should merge into it."""
    sources, target = db_dirs

    # Pre-populate target with existing data
    _write_csv(
        target / "MatMulV2.csv",
        [
            {
                "Input Shapes": '"50,512;512,1024"',
                "Input Data Types": '"DT_BF16;DT_BF16"',
                "Input Formats": '"ND;ND"',
                "Output Shapes": '"50,1024"',
                "Output Data Types": '"DT_BF16"',
                "Output Formats": '"ND"',
                "Average Duration(us)": "5.0",
            },
        ],
        COMPUTE_FIELDS,
    )

    build_database(sources, target)

    with (target / "MatMulV2.csv").open() as f:
        rows = list(csv.DictReader(f))
    # Should have 3 shapes: 50x512 (existing) + 100x512 (src1) + 200x512 (src2)
    assert len(rows) == 3


def test_build_empty_sources(tmp_path):
    """Empty source list should produce empty report."""
    target = tmp_path / "data"
    target.mkdir()
    report = build_database([], target)
    assert report.total_kernel_types == 0
    assert report.total_shapes == 0
