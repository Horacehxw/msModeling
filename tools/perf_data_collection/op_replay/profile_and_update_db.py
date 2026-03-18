"""
Profile replay scripts and write the results back to the database.

Purpose:
  Run run_all_op.py with msprof, collect generated op_summary_*.csv files,
  aggregate the profiling metrics, and update the matching operator CSV files
  under perf_database/data/{device}/vllm_ascend/{version}.

Usage:
  python tensor_cast/performance_model/perf_database/op_run/profile_and_update_db.py ^
    --device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.13.0

Arguments:
  --device                Selects the device directory under perf_database/data.
  --vllm-ascend-version   Selects the version directory under {device}/vllm_ascend.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import TYPE_CHECKING

from common import SUPPORTED_DEVICES, check_version, get_target_data_dir

if TYPE_CHECKING:
    import torch
    import torch_npu


RUN_ALL_SCRIPT = Path(__file__).resolve().parent / "run_all_op.py"
# profile_and_update_db.py → op_replay/ [0] → perf_data_collection/ [1] → tools/ [2] → repo_root [3]
REPO_ROOT = Path(__file__).resolve().parents[3]
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
MICROBENCH_DURATION = "MicroBench Duration(us)"
AVERAGE_DURATION = "Average Duration(us)"
STD_DURATION = "Std Duration(us)"
AVERAGE_EXTRA_COLUMNS = [
    "Average aicore_time(us)",
    "Average aic_total_cycles",
    "Average aic_mac_time(us)",
    "Average aic_mac_ratio",
    "Average aic_scalar_time(us)",
    "Average aic_scalar_ratio",
    "Average aic_mte1_time(us)",
    "Average aic_mte1_ratio",
    "Average aic_mte2_time(us)",
    "Average aic_mte2_ratio",
    "Average aic_fixpipe_time(us)",
    "Average aic_fixpipe_ratio",
    "Average aic_icache_miss_rate",
    "Average aiv_time(us)",
    "Average aiv_total_cycles",
    "Average aiv_vec_time(us)",
    "Average aiv_vec_ratio",
    "Average aiv_scalar_time(us)",
    "Average aiv_scalar_ratio",
    "Average aiv_mte2_time(us)",
    "Average aiv_mte2_ratio",
    "Average aiv_mte3_time(us)",
    "Average aiv_mte3_ratio",
    "Average aiv_icache_miss_rate",
    "Average cube_utilization(%)",
]
MATCH_COLUMNS = [
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
]
OP_SUMMARY_TO_DB_COLUMN = {
    "aicore_time(us)": "Average aicore_time(us)",
    "aic_total_cycles": "Average aic_total_cycles",
    "aic_mac_time(us)": "Average aic_mac_time(us)",
    "aic_mac_ratio": "Average aic_mac_ratio",
    "aic_scalar_time(us)": "Average aic_scalar_time(us)",
    "aic_scalar_ratio": "Average aic_scalar_ratio",
    "aic_mte1_time(us)": "Average aic_mte1_time(us)",
    "aic_mte1_ratio": "Average aic_mte1_ratio",
    "aic_mte2_time(us)": "Average aic_mte2_time(us)",
    "aic_mte2_ratio": "Average aic_mte2_ratio",
    "aic_fixpipe_time(us)": "Average aic_fixpipe_time(us)",
    "aic_fixpipe_ratio": "Average aic_fixpipe_ratio",
    "aic_icache_miss_rate": "Average aic_icache_miss_rate",
    "aiv_time(us)": "Average aiv_time(us)",
    "aiv_total_cycles": "Average aiv_total_cycles",
    "aiv_vec_time(us)": "Average aiv_vec_time(us)",
    "aiv_vec_ratio": "Average aiv_vec_ratio",
    "aiv_scalar_time(us)": "Average aiv_scalar_time(us)",
    "aiv_scalar_ratio": "Average aiv_scalar_ratio",
    "aiv_mte2_time(us)": "Average aiv_mte2_time(us)",
    "aiv_mte2_ratio": "Average aiv_mte2_ratio",
    "aiv_mte3_time(us)": "Average aiv_mte3_time(us)",
    "aiv_mte3_ratio": "Average aiv_mte3_ratio",
    "aiv_icache_miss_rate": "Average aiv_icache_miss_rate",
    "cube_utilization(%)": "Average cube_utilization(%)",
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
        raise RuntimeError("未找到NPU") from exc

    torch = torch_module
    torch_npu = torch_npu_module


def ensure_npu_available() -> None:
    init_runtime()
    has_npu = hasattr(torch, "npu") and torch.npu.is_available()
    if not has_npu:
        raise RuntimeError("未找到NPU")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Profile operator replay scripts with msprof and update the\n"
            "performance database under perf_database/data."
        ),
        epilog=(
            "Usage examples:\n"
            "  python tensor_cast/performance_model/perf_database/op_run/profile_and_update_db.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.13.0\n\n"
            "Workflow:\n"
            "  1. Run `msprof python .../run_all_op.py` with the same arguments.\n"
            "  2. Parse PROF_*/mindstudio_profiler_output/op_summary_*.csv.\n"
            "  3. Write Task Duration(us) into `MicroBench Duration(us)` in the\n"
            "     matching {OP Type}.csv under perf_database/data.\n"
            "  4. Remove the generated PROF_* directories."
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
        help="vLLM-Ascend version, e.g. 0.13.0.",
    )
    return parser


def list_prof_dirs() -> set[Path]:
    return {path for path in REPO_ROOT.glob("PROF_*") if path.is_dir()}


def run_msprof(device: str, vllm_ascend_version: str) -> set[Path]:
    before_prof_dirs = list_prof_dirs()
    command = (
        f"msprof python {shlex.quote(RUN_ALL_SCRIPT.as_posix())} "
        f"--device {shlex.quote(device)} "
        f"--vllm-ascend-version {shlex.quote(vllm_ascend_version)}"
    )
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


def parse_float(value: str) -> float:
    try:
        return float((value or "").strip())
    except ValueError:
        return 0.0


def format_float(value: float) -> str:
    return f"{value:.6f}"


def build_signature(row: dict[str, str]) -> tuple[str, ...]:
    return tuple((row.get(column, "") or "").strip() for column in MATCH_COLUMNS)


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
            AVERAGE_DURATION: format_float(float(item["microbench_sum"]) / count),
            STD_DURATION: format_float(0.0),
        }
        for source_col, db_col in OP_SUMMARY_TO_DB_COLUMN.items():
            aggregated_row[db_col] = format_float(item["metric_sums"][source_col] / count)
        result[op_type].append(aggregated_row)

    return result


def get_default_columns() -> list[str]:
    return BASE_COLUMNS + [MICROBENCH_DURATION, AVERAGE_DURATION, STD_DURATION] + AVERAGE_EXTRA_COLUMNS


def ensure_microbench_column(fieldnames: list[str]) -> list[str]:
    columns = list(fieldnames)
    if MICROBENCH_DURATION in columns:
        return columns

    if AVERAGE_DURATION in columns:
        insert_index = columns.index(AVERAGE_DURATION)
        columns.insert(insert_index, MICROBENCH_DURATION)
        return columns

    return BASE_COLUMNS + [MICROBENCH_DURATION] + [col for col in columns if col not in BASE_COLUMNS]


def normalize_row_for_columns(row: dict[str, str], columns: list[str]) -> dict[str, str]:
    return {column: row.get(column, "") for column in columns}


def update_op_csv(csv_path: Path, rows_to_merge: list[dict[str, str]]) -> tuple[int, int]:
    existing_rows: list[dict[str, str]] = []
    existing_columns = get_default_columns()

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            existing_columns = ensure_microbench_column(list(reader.fieldnames or get_default_columns()))
            existing_rows = list(reader)
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    updated_count = 0
    added_count = 0

    for new_row in rows_to_merge:
        matched = False
        for existing_row in existing_rows:
            if build_signature(existing_row) == build_signature(new_row):
                existing_row[MICROBENCH_DURATION] = new_row[MICROBENCH_DURATION]
                matched = True
                updated_count += 1
                break

        if not matched:
            existing_rows.append(new_row)
            added_count += 1
            print(
                f"[ADD] {csv_path.name} missing database row for "
                f"{new_row['Input Shapes']} -> {new_row['Output Shapes']}"
            )

    normalized_rows = [normalize_row_for_columns(row, existing_columns) for row in existing_rows]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=existing_columns)
        writer.writeheader()
        writer.writerows(normalized_rows)

    return updated_count, added_count


def update_database(device: str, vllm_ascend_version: str, aggregated_rows: dict[str, list[dict[str, str]]]) -> None:
    target_data_dir = get_target_data_dir(device, vllm_ascend_version)
    for op_type, rows in sorted(aggregated_rows.items()):
        csv_path = target_data_dir / f"{op_type}.csv"
        updated_count, added_count = update_op_csv(csv_path, rows)
        print(f"[DB] {csv_path.name}: updated={updated_count}, added={added_count}")


def cleanup_prof_dirs(prof_dirs: set[Path]) -> None:
    for prof_dir in sorted(prof_dirs):
        if prof_dir.exists():
            shutil.rmtree(prof_dir)
            print(f"[CLEAN] removed {prof_dir}")


def main() -> None:
    args = build_argparser().parse_args()
    ensure_npu_available()

    prof_dirs: set[Path] = set()
    try:
        prof_dirs = run_msprof(
            device=args.device,
            vllm_ascend_version=args.vllm_ascend_version,
        )
        op_summary_files = find_op_summary_files(prof_dirs)
        aggregated_rows = aggregate_op_summary(op_summary_files)
        update_database(
            device=args.device,
            vllm_ascend_version=args.vllm_ascend_version,
            aggregated_rows=aggregated_rows,
        )
    finally:
        cleanup_prof_dirs(prof_dirs)


if __name__ == "__main__":
    main()
