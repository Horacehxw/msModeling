"""Build operator performance CSV database from multiple sources.

Merges profiling and microbenchmark CSV data into a unified database directory.
Sources are processed in order — first occurrence of a shape wins (profiling
data takes precedence over microbenchmark data).

Design doc reference: S6.1 (Three-Step Strategy)

Usage:
    python3.10 tools/perf_data_collection/build_database.py \
        --sources /path/to/profiling_csvs /path/to/microbench_csvs \
        --target tensor_cast/performance_model/perf_database/data/.../
"""

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple


@dataclass
class DatabaseReport:
    """Report from a database build operation."""

    total_kernel_types: int = 0
    total_shapes: int = 0
    new_shapes: int = 0
    kernel_details: Dict[str, int] = field(default_factory=dict)  # type -> shape count


def _shape_key(row: dict) -> Tuple[str, str]:
    """Extract shape key for deduplication."""
    return (
        row.get("Input Shapes", "").strip(),
        row.get("Output Shapes", "").strip(),
    )


def _read_csv(path: Path) -> Tuple[List[str], List[dict]]:
    """Read a CSV file and return (fieldnames, rows)."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return list(fieldnames), rows


def _write_csv(path: Path, fieldnames: List[str], rows: List[dict]):
    """Write rows to a CSV file."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in fieldnames})


def build_database(
    source_dirs: List[Path],
    target_dir: Path,
) -> DatabaseReport:
    """Merge CSV data from multiple source directories into target.

    Sources are processed in order. For each kernel type CSV:
    1. Read existing target data (if any)
    2. Read source CSVs in order
    3. Add new shapes (first occurrence wins, dedup by Input+Output Shapes)
    4. Write merged result to target

    Args:
        source_dirs: List of directories containing per-kernel CSVs
        target_dir: Output directory for merged database

    Returns:
        DatabaseReport with merge statistics
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    report = DatabaseReport()

    # Collect all kernel types across all sources
    all_kernel_csvs: Dict[str, List[Path]] = {}
    for src in source_dirs:
        if not src.exists():
            continue
        for csv_file in sorted(src.glob("*.csv")):
            kernel_type = csv_file.stem
            if kernel_type not in all_kernel_csvs:
                all_kernel_csvs[kernel_type] = []
            all_kernel_csvs[kernel_type].append(csv_file)

    for kernel_type, source_files in sorted(all_kernel_csvs.items()):
        # Start with existing target data
        target_file = target_dir / f"{kernel_type}.csv"
        existing_rows: List[dict] = []
        fieldnames: List[str] = []
        seen_shapes: Set[Tuple[str, str]] = set()

        if target_file.exists():
            fieldnames, existing_rows = _read_csv(target_file)
            for row in existing_rows:
                seen_shapes.add(_shape_key(row))

        # Merge from each source
        new_rows: List[dict] = []
        for src_file in source_files:
            src_fields, src_rows = _read_csv(src_file)
            if not fieldnames:
                fieldnames = src_fields
            for row in src_rows:
                key = _shape_key(row)
                if key not in seen_shapes:
                    seen_shapes.add(key)
                    new_rows.append(row)
                    report.new_shapes += 1

        merged = existing_rows + new_rows
        if merged:
            _write_csv(target_file, fieldnames, merged)

        total_shapes = len(merged)
        report.kernel_details[kernel_type] = total_shapes
        report.total_shapes += total_shapes

    report.total_kernel_types = len(report.kernel_details)
    return report


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build CSV database from multiple profiling/microbenchmark sources."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Source directories containing per-kernel CSVs (processed in order)",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target directory for merged database",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    source_dirs = [Path(s) for s in args.sources]
    target_dir = Path(args.target)

    for src in source_dirs:
        if not src.exists():
            print(f"Warning: source directory not found: {src}", file=sys.stderr)

    report = build_database(source_dirs, target_dir)

    print(f"Database built in {target_dir}")
    print(f"  Kernel types: {report.total_kernel_types}")
    print(f"  Total shapes: {report.total_shapes}")
    print(f"  New shapes added: {report.new_shapes}")
    if report.kernel_details:
        print("\n  Per-kernel shape counts:")
        for kt, count in sorted(report.kernel_details.items()):
            print(f"    {kt}: {count}")


if __name__ == "__main__":
    main()
