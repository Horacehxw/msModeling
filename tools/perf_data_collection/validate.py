"""Validate performance database quality and internal consistency.

Checks:
- All CSVs have required columns
- No zero or negative durations
- No empty shape strings
- No duplicate shapes within a kernel type
- Per-kernel statistics (shape count, duration range)

Design doc reference: S6.1 Step 3 (Validation)

Usage:
    python3.10 tools/perf_data_collection/validate.py \
        --database tensor_cast/performance_model/perf_database/data/.../
"""

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple


@dataclass
class ValidationReport:
    """Report from database validation."""

    total_kernel_types: int = 0
    total_shapes: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    kernel_stats: Dict[str, dict] = field(default_factory=dict)


def _shape_key(row: dict) -> Tuple[str, str]:
    return (
        row.get("Input Shapes", "").strip(),
        row.get("Output Shapes", "").strip(),
    )


def validate_database(db_dir: Path) -> ValidationReport:
    """Validate all CSV files in a database directory.

    Args:
        db_dir: Directory containing per-kernel CSV files

    Returns:
        ValidationReport with errors, warnings, and statistics
    """
    report = ValidationReport()

    csv_files = sorted(db_dir.glob("*.csv"))
    if not csv_files:
        return report

    for csv_file in csv_files:
        kernel_type = csv_file.stem
        _validate_kernel_csv(csv_file, kernel_type, report)

    report.total_kernel_types = len(report.kernel_stats)
    report.total_shapes = sum(
        s["shapes"] for s in report.kernel_stats.values()
    )
    return report


def _validate_kernel_csv(
    csv_file: Path, kernel_type: str, report: ValidationReport
):
    """Validate a single kernel CSV file."""
    try:
        with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    except Exception as e:
        report.errors.append(f"{kernel_type}: Failed to read CSV: {e}")
        return

    # Detect duration column
    dur_col = None
    for candidate in ["Average Duration(us)", "Duration(us)"]:
        if candidate in fieldnames:
            dur_col = candidate
            break

    if dur_col is None:
        report.errors.append(
            f"{kernel_type}: Missing duration column "
            "(need 'Average Duration(us)' or 'Duration(us)')"
        )
        return

    seen_shapes: Set[Tuple[str, str]] = set()
    valid_shapes = 0
    durations: List[float] = []
    duplicates = 0

    for i, row in enumerate(rows):
        # Check for empty shapes
        input_shapes = row.get("Input Shapes", "").strip()
        if not input_shapes or input_shapes == '""':
            report.warnings.append(
                f"{kernel_type} row {i + 1}: empty Input Shapes"
            )
            continue

        # Check for zero/negative duration
        try:
            duration = float(row.get(dur_col, "0").strip())
        except ValueError:
            report.warnings.append(
                f"{kernel_type} row {i + 1}: invalid duration value"
            )
            continue

        if duration <= 0:
            report.warnings.append(
                f"{kernel_type} row {i + 1}: zero or negative duration ({duration})"
            )

        # Check for duplicate shapes
        key = _shape_key(row)
        if key in seen_shapes:
            duplicates += 1
        else:
            seen_shapes.add(key)

        valid_shapes += 1
        durations.append(duration)

    if duplicates > 0:
        report.warnings.append(
            f"{kernel_type}: {duplicates} duplicate shape(s)"
        )

    report.kernel_stats[kernel_type] = {
        "shapes": valid_shapes,
        "unique_shapes": len(seen_shapes),
        "duplicates": duplicates,
        "min_duration_us": min(durations) if durations else 0,
        "max_duration_us": max(durations) if durations else 0,
        "avg_duration_us": sum(durations) / len(durations) if durations else 0,
    }


def _format_report(report: ValidationReport, db_dir: Path) -> str:
    """Format validation report as human-readable text."""
    lines = []
    lines.append("=" * 60)
    lines.append("Database Validation Report")
    lines.append("=" * 60)
    lines.append(f"Database: {db_dir}")
    lines.append(f"Kernel types: {report.total_kernel_types}")
    lines.append(f"Total shapes: {report.total_shapes}")
    lines.append("")

    if report.errors:
        lines.append(f"ERRORS ({len(report.errors)}):")
        for e in report.errors:
            lines.append(f"  [ERROR] {e}")
        lines.append("")

    if report.warnings:
        lines.append(f"WARNINGS ({len(report.warnings)}):")
        for w in report.warnings:
            lines.append(f"  [WARN] {w}")
        lines.append("")

    if report.kernel_stats:
        lines.append("Per-kernel Statistics:")
        lines.append(
            f"  {'Kernel':<35} {'Shapes':>8} {'Min(us)':>10} "
            f"{'Max(us)':>10} {'Avg(us)':>10}"
        )
        lines.append(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
        for kt, stats in sorted(report.kernel_stats.items()):
            lines.append(
                f"  {kt:<35} {stats['shapes']:>8} "
                f"{stats['min_duration_us']:>10.1f} "
                f"{stats['max_duration_us']:>10.1f} "
                f"{stats['avg_duration_us']:>10.1f}"
            )

    status = "PASS" if not report.errors else "FAIL"
    lines.append("")
    lines.append(f"Result: {status}")
    if report.warnings:
        lines.append(f"  ({len(report.warnings)} warning(s))")

    return "\n".join(lines)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate performance database quality and consistency."
    )
    parser.add_argument(
        "--database",
        required=True,
        help="Path to database directory containing per-kernel CSVs",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    db_dir = Path(args.database)
    if not db_dir.exists():
        print(f"Error: database directory not found: {db_dir}", file=sys.stderr)
        sys.exit(1)

    report = validate_database(db_dir)
    print(_format_report(report, db_dir))

    if report.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
