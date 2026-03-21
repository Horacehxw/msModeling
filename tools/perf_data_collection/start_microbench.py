"""
Profile replay scripts and write the results back to the database.

Purpose:
  Run op_replay/run_all_op.py with msprof, collect generated op_summary_*.csv
  files, aggregate profiling metrics, and update matching operator CSV files
  under perf_database/data/{device}/vllm_ascend/{version}.

Usage:
  py -3 tools/perf_data_collection/start_microbench.py ^
    --device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.15.0

  py -3 tools/perf_data_collection/start_microbench.py ^
    --device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.15.0 ^
    --op MatMulV2 PadV3
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
OP_REPLAY_DIR = CURRENT_DIR / "op_replay"
if str(OP_REPLAY_DIR) not in sys.path:
    sys.path.insert(0, str(OP_REPLAY_DIR))

from common import SUPPORTED_DEVICES, check_version, get_target_data_dir

if TYPE_CHECKING:
    import torch
    import torch_npu


RUN_ALL_SCRIPT = OP_REPLAY_DIR / "run_all_op.py"
torch = None
torch_npu = None

BASE_COLUMNS = [
    "OP State",
    "Accelerator Core",
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
    "Output Formats",
]
LEGACY_MICROBENCH_DURATION = "MicroBench Duration(us)"
MICROBENCH_DURATION = "Average Duration(us)"
PROFILING_AVERAGE_DURATION = "Profiling Average Duration(us)"
PROFILING_MEDIAN_DURATION = "Profiling Median Duration(us)"
PROFILING_STD_DURATION = "Profiling Std Duration(us)"
PROFILING_AVERAGE_EXTRA_COLUMNS = [
    "Profiling Average aicore_time(us)",
    "Profiling Average aic_total_cycles",
    "Profiling Average aic_mac_time(us)",
    "Profiling Average aic_mac_ratio",
    "Profiling Average aic_scalar_time(us)",
    "Profiling Average aic_scalar_ratio",
    "Profiling Average aic_mte1_time(us)",
    "Profiling Average aic_mte1_ratio",
    "Profiling Average aic_mte2_time(us)",
    "Profiling Average aic_mte2_ratio",
    "Profiling Average aic_fixpipe_time(us)",
    "Profiling Average aic_fixpipe_ratio",
    "Profiling Average aic_icache_miss_rate",
    "Profiling Average aiv_time(us)",
    "Profiling Average aiv_total_cycles",
    "Profiling Average aiv_vec_time(us)",
    "Profiling Average aiv_vec_ratio",
    "Profiling Average aiv_scalar_time(us)",
    "Profiling Average aiv_scalar_ratio",
    "Profiling Average aiv_mte2_time(us)",
    "Profiling Average aiv_mte2_ratio",
    "Profiling Average aiv_mte3_time(us)",
    "Profiling Average aiv_mte3_ratio",
    "Profiling Average aiv_icache_miss_rate",
    "Profiling Average cube_utilization(%)",
]
MATCH_COLUMNS = [
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
]
OP_SUMMARY_TO_DB_COLUMN = {
    "aicore_time(us)": "Profiling Average aicore_time(us)",
    "aic_total_cycles": "Profiling Average aic_total_cycles",
    "aic_mac_time(us)": "Profiling Average aic_mac_time(us)",
    "aic_mac_ratio": "Profiling Average aic_mac_ratio",
    "aic_scalar_time(us)": "Profiling Average aic_scalar_time(us)",
    "aic_scalar_ratio": "Profiling Average aic_scalar_ratio",
    "aic_mte1_time(us)": "Profiling Average aic_mte1_time(us)",
    "aic_mte1_ratio": "Profiling Average aic_mte1_ratio",
    "aic_mte2_time(us)": "Profiling Average aic_mte2_time(us)",
    "aic_mte2_ratio": "Profiling Average aic_mte2_ratio",
    "aic_fixpipe_time(us)": "Profiling Average aic_fixpipe_time(us)",
    "aic_fixpipe_ratio": "Profiling Average aic_fixpipe_ratio",
    "aic_icache_miss_rate": "Profiling Average aic_icache_miss_rate",
    "aiv_time(us)": "Profiling Average aiv_time(us)",
    "aiv_total_cycles": "Profiling Average aiv_total_cycles",
    "aiv_vec_time(us)": "Profiling Average aiv_vec_time(us)",
    "aiv_vec_ratio": "Profiling Average aiv_vec_ratio",
    "aiv_scalar_time(us)": "Profiling Average aiv_scalar_time(us)",
    "aiv_scalar_ratio": "Profiling Average aiv_scalar_ratio",
    "aiv_mte2_time(us)": "Profiling Average aiv_mte2_time(us)",
    "aiv_mte2_ratio": "Profiling Average aiv_mte2_ratio",
    "aiv_mte3_time(us)": "Profiling Average aiv_mte3_time(us)",
    "aiv_mte3_ratio": "Profiling Average aiv_mte3_ratio",
    "aiv_icache_miss_rate": "Profiling Average aiv_icache_miss_rate",
    "cube_utilization(%)": "Profiling Average cube_utilization(%)",
}
SUMMARY_SAMPLE_LIMIT = 3
DEFAULT_GAP_RATIO_LOWER_BOUND = 0.8
DEFAULT_GAP_RATIO_UPPER_BOUND = 1.2
TOP_GAP_REPORT_LIMIT = 20


def normalize_op_name(name: str) -> str:
    normalized = name.strip()
    if normalized.endswith("_run.py"):
        normalized = normalized.removesuffix("_run.py")
    elif normalized.endswith("_run"):
        normalized = normalized.removesuffix("_run")
    elif normalized.endswith(".csv"):
        normalized = normalized.removesuffix(".csv")
    return normalized


def list_available_ops() -> list[str]:
    return sorted(normalize_op_name(path.stem) for path in OP_REPLAY_DIR.glob("*_run.py"))


def format_available_ops_for_help() -> str:
    available_ops = list_available_ops()
    return ", ".join(available_ops)


def to_microbench_column(profiling_column: str) -> str:
    if profiling_column.startswith("Profiling Average "):
        return "MicroBench " + profiling_column.removeprefix("Profiling Average ")
    if profiling_column == PROFILING_AVERAGE_DURATION:
        return MICROBENCH_DURATION
    raise ValueError(f"Unsupported profiling column for microbench mapping: {profiling_column}")


MICROBENCH_EXTRA_COLUMN_MAP = {
    source_col: to_microbench_column(db_col)
    for source_col, db_col in OP_SUMMARY_TO_DB_COLUMN.items()
}


def init_runtime() -> None:
    global torch
    global torch_npu

    if torch is not None and torch_npu is not None:
        return

    try:
        import torch as torch_module
        import torch_npu as torch_npu_module
    except ImportError as exc:
        raise RuntimeError("NPU runtime is unavailable") from exc

    torch = torch_module
    torch_npu = torch_npu_module


def ensure_npu_available() -> None:
    init_runtime()
    has_npu = hasattr(torch, "npu") and torch.npu.is_available()
    if not has_npu:
        raise RuntimeError("NPU runtime is unavailable")


def build_argparser() -> argparse.ArgumentParser:
    available_ops_text = format_available_ops_for_help()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Profile operator replay scripts with msprof and update the\n"
            "performance database under perf_database/data."
        ),
        epilog=(
            "Usage examples:\n"
            "  py -3 tools/perf_data_collection/start_microbench.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.15.0\n"
            "  py -3 tools/perf_data_collection/start_microbench.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.15.0 --op MatMulV2 PadV3\n\n"
            f"Available operators for --op:\n  {available_ops_text}\n\n"
            "Workflow:\n"
            "  1. Run `msprof python tools/perf_data_collection/op_replay/run_all_op.py`.\n"
            "  2. Parse PROF_*/mindstudio_profiler_output/op_summary_*.csv.\n"
            "  3. Write profiling-derived microbench values back into matching operator CSV files.\n"
            "  4. Generate summary reports under the target data directory.\n"
        ),
    )
    parser.add_argument(
        "--device",
        required=True,
        choices=SUPPORTED_DEVICES,
        help=(
            "Target device folder under "
            "tensor_cast/performance_model/perf_database/data/{device}/"
        ),
    )
    parser.add_argument(
        "--vllm-ascend-version",
        required=True,
        type=check_version,
        help="vLLM-Ascend version, e.g. 0.15.0.",
    )
    parser.add_argument(
        "--prof-path",
        default=None,
        help=(
            "Optional existing PROF_* directory to parse directly. "
            "When provided, the script skips launching msprof."
        ),
    )
    parser.add_argument(
        "--op",
        nargs="+",
        default=None,
        help=(
            "Optional operator names to update, e.g. MatMulV2 PadV3. "
            f"Available: {available_ops_text}"
        ),
    )
    return parser


def validate_selected_ops(selected_ops: list[str] | None) -> list[str] | None:
    if not selected_ops:
        return None
    available = set(list_available_ops())
    normalized = [normalize_op_name(item) for item in selected_ops]
    invalid = sorted(item for item in normalized if item not in available)
    if invalid:
        available_text = ", ".join(sorted(available))
        invalid_text = ", ".join(invalid)
        raise ValueError(f"Unsupported --op value(s): {invalid_text}. Available operators: {available_text}")
    return normalized


def list_prof_dirs() -> set[Path]:
    return {path for path in REPO_ROOT.glob("PROF_*") if path.is_dir()}


def run_msprof(device: str, vllm_ascend_version: str, selected_ops: list[str] | None) -> set[Path]:
    before_prof_dirs = list_prof_dirs()
    command = (
        f"msprof python {shlex.quote(RUN_ALL_SCRIPT.as_posix())} "
        f"--device {shlex.quote(device)} "
        f"--vllm-ascend-version {shlex.quote(vllm_ascend_version)} "
        f"--execution-mode inprocess"
    )
    if selected_ops:
        command += " --op " + " ".join(shlex.quote(op_name) for op_name in selected_ops)
    subprocess.run(command, shell=True, check=True, cwd=REPO_ROOT)
    after_prof_dirs = list_prof_dirs()
    return after_prof_dirs - before_prof_dirs


def find_op_summary_files(prof_dirs: set[Path]) -> list[Path]:
    op_summary_files = []
    for prof_dir in sorted(prof_dirs):
        output_dir = prof_dir / "mindstudio_profiler_output"
        op_summary_files.extend(sorted(output_dir.glob("op_summary_*.csv")))
    if not op_summary_files:
        raise FileNotFoundError("No op_summary_*.csv found in generated PROF_* directories")
    return op_summary_files


def resolve_prof_dirs(prof_path: str | None) -> set[Path]:
    if not prof_path:
        return set()
    path = Path(prof_path)
    if not path.exists():
        raise FileNotFoundError(f"PROF path does not exist: {path}")
    if path.is_file():
        raise ValueError(f"--prof-path must point to a PROF_* directory, got file: {path}")
    return {path}


def parse_float(value: str) -> float:
    try:
        return float((value or "").strip())
    except ValueError:
        return 0.0


def format_float(value: float) -> str:
    return f"{value:.6f}"


def build_signature(row: dict[str, str]) -> tuple[str, ...]:
    return tuple((row.get(column, "") or "").strip() for column in MATCH_COLUMNS)


def summarize_signature(row: dict[str, str]) -> str:
    input_shapes = (row.get("Input Shapes", "") or "").strip() or "N/A"
    output_shapes = (row.get("Output Shapes", "") or "").strip() or "N/A"
    return f"{input_shapes} -> {output_shapes}"


def make_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_None_"
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(cells: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells)) + " |"

    separator = "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |"
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


@dataclass
class GapRecord:
    op_type: str
    csv_name: str
    signature: str
    microbench_us: float
    profiling_us: float
    abs_diff_us: float
    microbench_vs_profiling_ratio: float


@dataclass
class MissingRowRecord:
    signature: str


@dataclass
class UpdateResult:
    csv_path: Path
    updated_count: int = 0
    added_count: int = 0
    unchanged_count: int = 0
    missing_rows: list[MissingRowRecord] = field(default_factory=list)
    gap_records: list[GapRecord] = field(default_factory=list)


def aggregate_op_summary(op_summary_files: list[Path]) -> dict[str, list[dict[str, str]]]:
    grouped_rows: dict[tuple[str, tuple[str, ...]], dict[str, object]] = {}

    for csv_path in op_summary_files:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                op_type = (row.get("OP Type", "") or "").strip()
                if not op_type:
                    continue

                signature = build_signature(row)
                group_key = (op_type, signature)
                current = grouped_rows.setdefault(
                    group_key,
                    {
                        "count": 0,
                        "row": row,
                        "microbench_sum": 0.0,
                        "metric_sums": defaultdict(float),
                    },
                )
                current["count"] = int(current["count"]) + 1
                current["microbench_sum"] = float(current["microbench_sum"]) + parse_float(
                    row.get("Task Duration(us)", "")
                )
                for source_col in OP_SUMMARY_TO_DB_COLUMN:
                    current["metric_sums"][source_col] += parse_float(row.get(source_col, ""))

    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for (op_type, _), item in grouped_rows.items():
        count = int(item["count"])
        source_row = dict(item["row"])
        aggregated_row = {
            "OP State": (source_row.get("OP State", "") or "").strip(),
            "Accelerator Core": (source_row.get("Task Type", "") or "").strip(),
            "Input Shapes": (source_row.get("Input Shapes", "") or "").strip(),
            "Input Data Types": (source_row.get("Input Data Types", "") or "").strip(),
            "Input Formats": (source_row.get("Input Formats", "") or "").strip(),
            "Output Shapes": (source_row.get("Output Shapes", "") or "").strip(),
            "Output Data Types": (source_row.get("Output Data Types", "") or "").strip(),
            "Output Formats": (source_row.get("Output Formats", "") or "").strip(),
            MICROBENCH_DURATION: format_float(float(item["microbench_sum"]) / count),
        }
        for source_col, microbench_col in MICROBENCH_EXTRA_COLUMN_MAP.items():
            aggregated_row[microbench_col] = format_float(item["metric_sums"][source_col] / count)
        result[op_type].append(aggregated_row)

    return result


def filter_aggregated_rows(
    aggregated_rows: dict[str, list[dict[str, str]]],
    selected_ops: list[str] | None,
) -> dict[str, list[dict[str, str]]]:
    if not selected_ops:
        return aggregated_rows
    selected = set(selected_ops)
    return {
        op_type: rows
        for op_type, rows in aggregated_rows.items()
        if normalize_op_name(op_type) in selected
    }


def get_default_columns() -> list[str]:
    columns = list(BASE_COLUMNS)
    columns.append(MICROBENCH_DURATION)
    columns.append(PROFILING_AVERAGE_DURATION)
    columns.append(PROFILING_MEDIAN_DURATION)
    columns.append(PROFILING_STD_DURATION)
    for profiling_col in PROFILING_AVERAGE_EXTRA_COLUMNS:
        columns.append(to_microbench_column(profiling_col))
        columns.append(profiling_col)
    return columns


def ensure_microbench_column(fieldnames: list[str]) -> list[str]:
    columns = list(fieldnames)
    if LEGACY_MICROBENCH_DURATION in columns and MICROBENCH_DURATION not in columns:
        columns[columns.index(LEGACY_MICROBENCH_DURATION)] = MICROBENCH_DURATION

    if MICROBENCH_DURATION not in columns and PROFILING_AVERAGE_DURATION in columns:
        insert_index = columns.index(PROFILING_AVERAGE_DURATION)
        columns.insert(insert_index, MICROBENCH_DURATION)
    elif MICROBENCH_DURATION not in columns:
        columns = BASE_COLUMNS + [MICROBENCH_DURATION] + [col for col in columns if col not in BASE_COLUMNS]

    for profiling_col in PROFILING_AVERAGE_EXTRA_COLUMNS:
        microbench_col = to_microbench_column(profiling_col)
        if profiling_col in columns and microbench_col not in columns:
            insert_index = columns.index(profiling_col)
            columns.insert(insert_index, microbench_col)
    return columns


def normalize_row_for_columns(row: dict[str, str], columns: list[str]) -> dict[str, str]:
    normalized_row = dict(row)
    if LEGACY_MICROBENCH_DURATION in normalized_row and MICROBENCH_DURATION not in normalized_row:
        normalized_row[MICROBENCH_DURATION] = normalized_row.get(LEGACY_MICROBENCH_DURATION, "")
    return {column: normalized_row.get(column, "") for column in columns}


def build_gap_record(csv_path: Path, row: dict[str, str]) -> GapRecord | None:
    microbench_us = parse_float(row.get(MICROBENCH_DURATION, ""))
    profiling_us = parse_float(row.get(PROFILING_AVERAGE_DURATION, ""))
    if microbench_us <= 0.0 or profiling_us <= 0.0:
        return None

    ratio = microbench_us / profiling_us
    abs_diff_us = abs(microbench_us - profiling_us)
    return GapRecord(
        op_type=csv_path.stem,
        csv_name=csv_path.name,
        signature=summarize_signature(row),
        microbench_us=microbench_us,
        profiling_us=profiling_us,
        abs_diff_us=abs_diff_us,
        microbench_vs_profiling_ratio=ratio,
    )


def update_op_csv(csv_path: Path, rows_to_merge: list[dict[str, str]]) -> UpdateResult:
    existing_rows: list[dict[str, str]] = []
    existing_columns = get_default_columns()

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            existing_columns = ensure_microbench_column(list(reader.fieldnames or get_default_columns()))
            existing_rows = list(reader)
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    result = UpdateResult(csv_path=csv_path)

    for new_row in rows_to_merge:
        matched = False
        for existing_row in existing_rows:
            if build_signature(existing_row) == build_signature(new_row):
                if LEGACY_MICROBENCH_DURATION in existing_row and MICROBENCH_DURATION not in existing_row:
                    existing_row[MICROBENCH_DURATION] = existing_row.get(LEGACY_MICROBENCH_DURATION, "")
                old_microbench = existing_row.get(MICROBENCH_DURATION, "")
                existing_row[MICROBENCH_DURATION] = new_row.get(MICROBENCH_DURATION, "")
                for microbench_col in MICROBENCH_EXTRA_COLUMN_MAP.values():
                    if microbench_col in new_row:
                        existing_row[microbench_col] = new_row[microbench_col]
                matched = True
                if (old_microbench or "").strip() == (existing_row.get(MICROBENCH_DURATION, "") or "").strip():
                    result.unchanged_count += 1
                else:
                    result.updated_count += 1
                gap_record = build_gap_record(csv_path, existing_row)
                if gap_record is not None:
                    result.gap_records.append(gap_record)
                break

        if not matched:
            existing_rows.append(new_row)
            result.added_count += 1
            result.missing_rows.append(MissingRowRecord(signature=summarize_signature(new_row)))

    normalized_rows = [normalize_row_for_columns(row, existing_columns) for row in existing_rows]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=existing_columns)
        writer.writeheader()
        writer.writerows(normalized_rows)

    return result


def update_database(
    device: str,
    vllm_ascend_version: str,
    aggregated_rows: dict[str, list[dict[str, str]]],
) -> list[UpdateResult]:
    target_data_dir = get_target_data_dir(device, vllm_ascend_version)
    results: list[UpdateResult] = []
    for op_type, rows in sorted(aggregated_rows.items()):
        csv_path = target_data_dir / f"{op_type}.csv"
        results.append(update_op_csv(csv_path, rows))
    return results


def print_update_summary(results: list[UpdateResult]) -> None:
    headers = ["Operator", "Updated", "Added", "Unchanged", "Missing Samples"]
    rows: list[list[str]] = []
    for result in sorted(results, key=lambda item: (item.added_count, item.updated_count, item.csv_path.name), reverse=True):
        missing_samples = ", ".join(item.signature for item in result.missing_rows[:SUMMARY_SAMPLE_LIMIT]) or "-"
        rows.append([
            result.csv_path.stem,
            str(result.updated_count),
            str(result.added_count),
            str(result.unchanged_count),
            missing_samples,
        ])
    print("\n[SUMMARY] Update Summary")
    print(make_markdown_table(headers, rows))


def print_missing_summary(results: list[UpdateResult]) -> None:
    missing_results = [result for result in results if result.missing_rows]
    if not missing_results:
        print("\n[SUMMARY] Missing Shapes\n_None_")
        return

    headers = ["Operator", "Missing Rows", "Examples"]
    rows: list[list[str]] = []
    for result in sorted(missing_results, key=lambda item: (len(item.missing_rows), item.csv_path.name), reverse=True):
        rows.append([
            result.csv_path.stem,
            str(len(result.missing_rows)),
            "; ".join(item.signature for item in result.missing_rows[:SUMMARY_SAMPLE_LIMIT]),
        ])
    print("\n[SUMMARY] Missing Shapes")
    print(make_markdown_table(headers, rows))


def collect_hotspot_gaps(
    results: list[UpdateResult],
    *,
    ratio_lower_bound: float,
    ratio_upper_bound: float,
) -> list[GapRecord]:
    all_gaps = [gap for result in results for gap in result.gap_records]
    return sorted(
        [
            gap
            for gap in all_gaps
            if (
                gap.microbench_vs_profiling_ratio < ratio_lower_bound
                or gap.microbench_vs_profiling_ratio > ratio_upper_bound
            )
        ],
        key=lambda item: (
            max(item.microbench_vs_profiling_ratio, 1.0 / item.microbench_vs_profiling_ratio),
            item.abs_diff_us,
            item.op_type,
        ),
        reverse=True,
    )


def print_gap_summary(gaps: list[GapRecord]) -> None:
    if not gaps:
        print("\n[SUMMARY] Duration Gap Hotspots\n_None_")
        return

    headers = ["Operator", "MicroBench(us)", "Profiling(us)", "Abs Diff(us)", "MB/Profile", "Shape"]
    rows: list[list[str]] = []
    for gap in gaps[:TOP_GAP_REPORT_LIMIT]:
        rows.append([
            gap.op_type,
            format_float(gap.microbench_us),
            format_float(gap.profiling_us),
            format_float(gap.abs_diff_us),
            f"{gap.microbench_vs_profiling_ratio:.2f}x",
            gap.signature,
        ])
    print("\n[SUMMARY] Duration Gap Hotspots")
    print(make_markdown_table(headers, rows))


def write_full_gap_csv(report_dir: Path, gaps: list[GapRecord], timestamp: str) -> Path:
    csv_path = report_dir / f"duration_gap_hotspots_full_{timestamp}.csv"
    fieldnames = [
        "Operator",
        "CSV Name",
        "MicroBench(us)",
        "Profiling(us)",
        "Abs Diff(us)",
        "MB/Profile",
        "Shape",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for gap in gaps:
            writer.writerow(
                {
                    "Operator": gap.op_type,
                    "CSV Name": gap.csv_name,
                    "MicroBench(us)": format_float(gap.microbench_us),
                    "Profiling(us)": format_float(gap.profiling_us),
                    "Abs Diff(us)": format_float(gap.abs_diff_us),
                    "MB/Profile": f"{gap.microbench_vs_profiling_ratio:.6f}",
                    "Shape": gap.signature,
                }
            )
    return csv_path


def write_report(
    target_data_dir: Path,
    results: list[UpdateResult],
    gaps: list[GapRecord],
    *,
    ratio_lower_bound: float,
    ratio_upper_bound: float,
) -> tuple[Path, Path]:
    report_dir = target_data_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"profile_update_report_{timestamp}.md"
    full_gap_csv_path = write_full_gap_csv(report_dir, gaps, timestamp)

    summary_rows = [
        ["CSV files touched", str(len(results))],
        ["Rows updated", str(sum(result.updated_count for result in results))],
        ["Rows added", str(sum(result.added_count for result in results))],
        ["Rows unchanged", str(sum(result.unchanged_count for result in results))],
        ["Missing shapes", str(sum(len(result.missing_rows) for result in results))],
        ["Hotspots", str(len(gaps))],
        ["Gap threshold", f"microbench/profiling not in [{ratio_lower_bound:.2f}x, {ratio_upper_bound:.2f}x]"],
    ]

    update_rows = []
    for result in sorted(results, key=lambda item: (item.added_count, item.updated_count, item.csv_path.name), reverse=True):
        update_rows.append([
            result.csv_path.stem,
            str(result.updated_count),
            str(result.added_count),
            str(result.unchanged_count),
            "; ".join(item.signature for item in result.missing_rows[:SUMMARY_SAMPLE_LIMIT]) or "-",
        ])

    gap_rows = []
    for gap in gaps[:TOP_GAP_REPORT_LIMIT]:
        gap_rows.append([
            gap.op_type,
            format_float(gap.microbench_us),
            format_float(gap.profiling_us),
            format_float(gap.abs_diff_us),
            f"{gap.microbench_vs_profiling_ratio:.2f}x",
            gap.signature,
        ])

    with report_path.open("w", encoding="utf-8", newline="\n") as report_file:
        report_file.write("# Profile Update Report\n\n")
        report_file.write("## Overview\n")
        report_file.write(make_markdown_table(["Metric", "Value"], summary_rows))
        report_file.write("\n\n## Update Summary\n")
        report_file.write(make_markdown_table(["Operator", "Updated", "Added", "Unchanged", "Missing Samples"], update_rows))
        report_file.write("\n\n## Duration Gap Hotspots\n")
        report_file.write(make_markdown_table(["Operator", "MicroBench(us)", "Profiling(us)", "Abs Diff(us)", "MB/Profile", "Shape"], gap_rows))
        report_file.write("\n\nFull hotspot CSV: ")
        report_file.write(full_gap_csv_path.name)
        report_file.write("\n")

    return report_path, full_gap_csv_path


def cleanup_prof_dirs(prof_dirs: set[Path]) -> None:
    for prof_dir in sorted(prof_dirs):
        if prof_dir.exists():
            shutil.rmtree(prof_dir)
            print(f"[CLEAN] removed {prof_dir}")


def main() -> None:
    args = build_argparser().parse_args()
    selected_ops = validate_selected_ops(args.op)
    if not args.prof_path:
        ensure_npu_available()

    prof_dirs: set[Path] = set()
    target_data_dir = get_target_data_dir(args.device, args.vllm_ascend_version)
    try:
        if args.prof_path:
            prof_dirs = resolve_prof_dirs(args.prof_path)
        else:
            prof_dirs = run_msprof(
                device=args.device,
                vllm_ascend_version=args.vllm_ascend_version,
                selected_ops=selected_ops,
            )
        op_summary_files = find_op_summary_files(prof_dirs)
        aggregated_rows = aggregate_op_summary(op_summary_files)
        aggregated_rows = filter_aggregated_rows(aggregated_rows, selected_ops)
        results = update_database(
            device=args.device,
            vllm_ascend_version=args.vllm_ascend_version,
            aggregated_rows=aggregated_rows,
        )
        gaps = collect_hotspot_gaps(
            results,
            ratio_lower_bound=DEFAULT_GAP_RATIO_LOWER_BOUND,
            ratio_upper_bound=DEFAULT_GAP_RATIO_UPPER_BOUND,
        )
        print_update_summary(results)
        print_missing_summary(results)
        print_gap_summary(gaps)
        report_path, full_gap_csv_path = write_report(
            target_data_dir=target_data_dir,
            results=results,
            gaps=gaps,
            ratio_lower_bound=DEFAULT_GAP_RATIO_LOWER_BOUND,
            ratio_upper_bound=DEFAULT_GAP_RATIO_UPPER_BOUND,
        )
        print(f"\n[REPORT] {report_path}")
        print(f"[REPORT] {full_gap_csv_path}")
    finally:
        if not args.prof_path:
            cleanup_prof_dirs(prof_dirs)


if __name__ == "__main__":
    main()
