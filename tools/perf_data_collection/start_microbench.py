"""
Profile replay scripts and write the results back to the database.

Purpose:
  Run op_replay/run_all_op.py with msprof, collect generated op_summary_*.csv
  files, aggregate profiling metrics, and update matching operator CSV files
  under profiling_database/data/{device}/vllm_ascend/{version}.

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
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
OP_REPLAY_DIR = CURRENT_DIR / "op_replay"
if str(OP_REPLAY_DIR) not in sys.path:
    sys.path.insert(0, str(OP_REPLAY_DIR))

from common import SUPPORTED_DEVICES, check_version, get_target_data_dir, normalize_op_name

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
# EP Size 列名（仅对 DispatchFFNCombine 生效）
EP_SIZE_COL = "EP Size"
LEGACY_MICROBENCH_DURATION = "MicroBench Duration(us)"
MICROBENCH_DURATION = "Average Duration(us)"
MICROBENCH_TASK_DURATION = "MicroBench Task Duration(us)"
MICROBENCH_KERNEL_DURATION = "MicroBench Kernel Duration(us)"
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
# DFC 专用匹配列（额外包含 EP Size）
DFC_MATCH_COLUMNS = MATCH_COLUMNS + [EP_SIZE_COL]
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
            "performance database under profiling_database/data."
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
            "tensor_cast/performance_model/profiling_database/data/{device}/"
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
    parser.add_argument(
        "--ep-size",
        type=int,
        default=16,
        help=(
            "EP (Expert Parallel) size for DispatchFFNCombine. "
            "Only used when --op includes DispatchFFNCombine. "
            "Default: 16."
        ),
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=True,
        help="Use balanced expert distribution for DispatchFFNCombine. Default: True.",
    )
    parser.add_argument(
        "--no-balanced",
        action="store_false",
        dest="balanced",
        help="Use random expert distribution for DispatchFFNCombine instead of balanced.",
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


DISPATCH_FFN_COMBINE_OP_NAME = "DispatchFFNCombine"
DISPATCH_FFN_COMBINE_SCRIPT = OP_REPLAY_DIR / "DispatchFFNCombine_run.py"


def run_dispatch_ffn_combine_direct(
        device: str,
        vllm_ascend_version: str,
        ep_size: int = 16,
        balanced: bool = True,
) -> Path:
    """
    直接运行 DispatchFFNCombine_run.py，使用内置的 benchmark 计时逻辑。
    返回输出的 CSV 文件路径。
    """
    output_csv = REPO_ROOT / f"DispatchFFNCombine_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    command = [
        sys.executable,
        str(DISPATCH_FFN_COMBINE_SCRIPT),
        "--device",
        device,
        "--vllm-ascend-version",
        vllm_ascend_version,
        "--output-csv",
        str(output_csv),
        "--ep-size",
        str(ep_size),
    ]
    if not balanced:
        command.append("--no-balanced")
    print(f"[RUN] {' '.join(command)}")
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    return output_csv


# DispatchFFNCombine 输出列名到 MicroBench 列名的映射
DFC_COLUMN_TO_MICROBENCH = {
    "aicore_time(us)": "MicroBench aicore_time(us)",
    "aic_total_cycles": "MicroBench aic_total_cycles",
    "aic_mac_time(us)": "MicroBench aic_mac_time(us)",
    "aic_mac_ratio": "MicroBench aic_mac_ratio",
    "aic_scalar_time(us)": "MicroBench aic_scalar_time(us)",
    "aic_scalar_ratio": "MicroBench aic_scalar_ratio",
    "aic_mte1_time(us)": "MicroBench aic_mte1_time(us)",
    "aic_mte1_ratio": "MicroBench aic_mte1_ratio",
    "aic_mte2_time(us)": "MicroBench aic_mte2_time(us)",
    "aic_mte2_ratio": "MicroBench aic_mte2_ratio",
    "aic_fixpipe_time(us)": "MicroBench aic_fixpipe_time(us)",
    "aic_fixpipe_ratio": "MicroBench aic_fixpipe_ratio",
    "aic_icache_miss_rate": "MicroBench aic_icache_miss_rate",
    "aiv_time(us)": "MicroBench aiv_time(us)",
    "aiv_total_cycles": "MicroBench aiv_total_cycles",
    "aiv_vec_time(us)": "MicroBench aiv_vec_time(us)",
    "aiv_vec_ratio": "MicroBench aiv_vec_ratio",
    "aiv_scalar_time(us)": "MicroBench aiv_scalar_time(us)",
    "aiv_scalar_ratio": "MicroBench aiv_scalar_ratio",
    "aiv_mte2_time(us)": "MicroBench aiv_mte2_time(us)",
    "aiv_mte2_ratio": "MicroBench aiv_mte2_ratio",
    "aiv_mte3_time(us)": "MicroBench aiv_mte3_time(us)",
    "aiv_mte3_ratio": "MicroBench aiv_mte3_ratio",
    "aiv_icache_miss_rate": "MicroBench aiv_icache_miss_rate",
    "cube_utilization(%)": "MicroBench cube_utilization(%)",
}


def parse_dispatch_ffn_combine_csv(csv_path: Path) -> dict[str, list[dict[str, str]]]:
    """
    解析 DispatchFFNCombine_run.py 输出的 CSV 文件。
    返回格式与 aggregate_op_summary 一致，方便后续流程复用。
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"DispatchFFNCombine benchmark output not found: {csv_path}")

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            # 确保有 OP Type 字段
            normalized_row = dict(row)
            if "OP Type" not in normalized_row:
                normalized_row["OP Type"] = DISPATCH_FFN_COMBINE_OP_NAME
            # 映射硬件指标列到 MicroBench 列
            for src_col, microbench_col in DFC_COLUMN_TO_MICROBENCH.items():
                if src_col in normalized_row and microbench_col not in normalized_row:
                    normalized_row[microbench_col] = normalized_row[src_col]
            # 将 Average Duration(us) 同时映射到 MicroBench Kernel Duration(us)
            # 使 gap 报告可以正确对比 microbench 与 profiling 数据
            avg_duration = normalized_row.get(MICROBENCH_DURATION, "")
            if avg_duration and MICROBENCH_KERNEL_DURATION not in normalized_row:
                normalized_row[MICROBENCH_KERNEL_DURATION] = avg_duration
            # 保留 EP Size 列（从源 CSV 读取的重要字段）
            # ep_size_col 已在 reader 中自动读取，无需额外处理
            rows.append(normalized_row)

    return {DISPATCH_FFN_COMBINE_OP_NAME: rows}


def list_prof_dirs() -> set[Path]:
    return {path for path in REPO_ROOT.glob("PROF_*") if path.is_dir()}


def run_msprof(device: str, vllm_ascend_version: str, exclude_dfc: bool = True,
               selected_ops: list[str] | None = None) -> set[Path]:
    """
    运行 msprof 进行性能采集。

    Args:
        device: 设备名称
        vllm_ascend_version: vLLM-Ascend 版本
        exclude_dfc: 是否排除 DispatchFFNCombine（默认 True，因为 DFC 使用内置 benchmark）
        selected_ops: 要 profiling 的算子列表，如果为 None 则运行所有算子（除了被排除的）
    """
    before_prof_dirs = list_prof_dirs()
    command = [
        "msprof",
        "python",
        str(RUN_ALL_SCRIPT),
        "--device",
        device,
        "--vllm-ascend-version",
        vllm_ascend_version,
        "--execution-mode",
        "inprocess",
    ]
    # 构建要运行的算子列表：排除 DispatchFFNCombine
    ops_to_run = list(selected_ops) if selected_ops else []
    if exclude_dfc and ops_to_run:
        ops_to_run = [
            op for op in ops_to_run
            if normalize_op_name(op) != normalize_op_name(DISPATCH_FFN_COMBINE_OP_NAME)
        ]
    if ops_to_run:
        command += ["--op"] + ops_to_run
    subprocess.run(command, check=True, cwd=REPO_ROOT)
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


def find_kernel_details_files(prof_dirs: set[Path]) -> list[Path]:
    kernel_details_files = []
    for prof_dir in sorted(prof_dirs):
        output_dir = prof_dir / "mindstudio_profiler_output"
        current_files = sorted(output_dir.glob("kernel_details*.csv"))
        if not current_files:
            current_files = sorted(prof_dir.glob("kernel_details*.csv"))
        kernel_details_files.extend(current_files)
    return kernel_details_files


def find_task_time_files(prof_dirs: set[Path]) -> list[Path]:
    task_time_files = []
    for prof_dir in sorted(prof_dirs):
        output_dir = prof_dir / "mindstudio_profiler_output"
        task_time_files.extend(sorted(output_dir.glob("task_time_*.csv")))
    return task_time_files


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


def build_signature(row: dict[str, str], columns: list[str] | None = None) -> tuple[str, ...]:
    """构建行签名，用于匹配 CSV 行。

    Args:
        row: 行数据
        columns: 用于匹配的列名列表，默认使用 MATCH_COLUMNS
                 DFC 行应传入 DFC_MATCH_COLUMNS 以包含 EP Size
    """
    match_columns = columns if columns is not None else MATCH_COLUMNS
    return tuple((row.get(column, "") or "").strip() for column in match_columns)


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
                        "task_duration_sum": 0.0,
                        "task_duration_count": 0,
                        "metric_sums": defaultdict(float),
                    },
                )
                # Seed both duration columns from op_summary. MICROBENCH_DURATION is the
                # default value and may later be overridden by task_time/kernel_details,
                # while MICROBENCH_TASK_DURATION preserves the raw Task Duration(us).
                task_duration = parse_float(row.get("Task Duration(us)", ""))
                current["count"] = int(current["count"]) + 1
                current["microbench_sum"] = float(current["microbench_sum"]) + task_duration
                current["task_duration_sum"] = float(current["task_duration_sum"]) + task_duration
                current["task_duration_count"] = int(current["task_duration_count"]) + 1
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
            MICROBENCH_TASK_DURATION: format_float(float(item["task_duration_sum"]) / int(item["task_duration_count"])),
        }
        for source_col, microbench_col in MICROBENCH_EXTRA_COLUMN_MAP.items():
            aggregated_row[microbench_col] = format_float(item["metric_sums"][source_col] / count)
        result[op_type].append(aggregated_row)

    return result


def aggregate_task_time(
    op_summary_files: list[Path],
    task_time_files: list[Path],
) -> dict[tuple[str, tuple[str, ...]], float]:
    if not task_time_files:
        return {}

    task_time_by_parent: dict[Path, list[Path]] = defaultdict(list)
    for csv_path in task_time_files:
        task_time_by_parent[csv_path.parent].append(csv_path)

    grouped_rows: dict[tuple[str, tuple[str, ...]], dict[str, object]] = {}
    for op_summary_path in op_summary_files:
        output_dir = op_summary_path.parent
        candidate_task_files = task_time_by_parent.get(output_dir, [])
        if not candidate_task_files:
            continue

        op_index: dict[tuple[str, str], tuple[str, tuple[str, ...]]] = {}
        with op_summary_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                op_type = (row.get("OP Type", "") or "").strip()
                task_id = (row.get("Task ID", "") or "").strip()
                stream_id = (row.get("Stream ID", "") or "").strip()
                if not op_type or not task_id:
                    continue
                op_index[(task_id, stream_id)] = (op_type, build_signature(row))

        for task_time_path in candidate_task_files:
            with task_time_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    task_id = (row.get("task_id", "") or "").strip()
                    stream_id = (row.get("stream_id", "") or "").strip()
                    if not task_id:
                        continue
                    matched = op_index.get((task_id, stream_id))
                    if matched is None:
                        continue
                    group_key = matched
                    current = grouped_rows.setdefault(
                        group_key,
                        {
                            "count": 0,
                            "duration_sum": 0.0,
                        },
                    )
                    current["count"] = int(current["count"]) + 1
                    current["duration_sum"] = float(current["duration_sum"]) + parse_float(
                        row.get("task_time(us)", "")
                    )

    result: dict[tuple[str, tuple[str, ...]], float] = {}
    for group_key, item in grouped_rows.items():
        count = int(item["count"])
        if count <= 0:
            continue
        result[group_key] = float(item["duration_sum"]) / count
    return result


def aggregate_kernel_details(kernel_details_files: list[Path]) -> dict[tuple[str, tuple[str, ...]], float]:
    grouped_rows: dict[tuple[str, tuple[str, ...]], dict[str, object]] = {}

    for csv_path in kernel_details_files:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                op_type = (row.get("Type", "") or "").strip()
                if not op_type:
                    continue
                signature = build_signature(row)
                group_key = (op_type, signature)
                current = grouped_rows.setdefault(
                    group_key,
                    {
                        "count": 0,
                        "duration_sum": 0.0,
                    },
                )
                current["count"] = int(current["count"]) + 1
                current["duration_sum"] = float(current["duration_sum"]) + parse_float(
                    row.get("Duration(us)", "")
                )

    result: dict[tuple[str, tuple[str, ...]], float] = {}
    for group_key, item in grouped_rows.items():
        count = int(item["count"])
        if count <= 0:
            continue
        result[group_key] = float(item["duration_sum"]) / count
    return result


def attach_task_durations(
    aggregated_rows: dict[str, list[dict[str, str]]],
    task_duration_map: dict[tuple[str, tuple[str, ...]], float],
) -> dict[str, list[dict[str, str]]]:
    # task_time is the first override source for MICROBENCH_DURATION. It replaces
    # the op_summary-derived default when a per-signature task_time average exists.
    for op_type, rows in aggregated_rows.items():
        for row in rows:
            key = (op_type, build_signature(row))
            task_duration = task_duration_map.get(key)
            if task_duration is not None:
                row[MICROBENCH_DURATION] = format_float(task_duration)
    return aggregated_rows


def attach_kernel_durations(
    aggregated_rows: dict[str, list[dict[str, str]]],
    kernel_duration_map: dict[tuple[str, tuple[str, ...]], float],
) -> dict[str, list[dict[str, str]]]:
    # kernel_details has the highest priority. When present, it overwrites the
    # current MICROBENCH_DURATION value (possibly sourced from task_time) and also
    # records the same value in MICROBENCH_KERNEL_DURATION.
    for op_type, rows in aggregated_rows.items():
        for row in rows:
            key = (op_type, build_signature(row))
            kernel_duration = kernel_duration_map.get(key)
            if kernel_duration is not None:
                row[MICROBENCH_KERNEL_DURATION] = format_float(kernel_duration)
                row[MICROBENCH_DURATION] = format_float(kernel_duration)
    return aggregated_rows


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
    columns.append(MICROBENCH_TASK_DURATION)
    columns.append(MICROBENCH_KERNEL_DURATION)
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

    if MICROBENCH_TASK_DURATION not in columns and PROFILING_AVERAGE_DURATION in columns:
        insert_index = columns.index(PROFILING_AVERAGE_DURATION)
        columns.insert(insert_index, MICROBENCH_TASK_DURATION)
    elif MICROBENCH_TASK_DURATION not in columns:
        insert_index = columns.index(MICROBENCH_DURATION) + 1 if MICROBENCH_DURATION in columns else len(BASE_COLUMNS)
        columns.insert(insert_index, MICROBENCH_TASK_DURATION)

    if MICROBENCH_KERNEL_DURATION not in columns and PROFILING_AVERAGE_DURATION in columns:
        insert_index = columns.index(PROFILING_AVERAGE_DURATION)
        columns.insert(insert_index, MICROBENCH_KERNEL_DURATION)
    elif MICROBENCH_KERNEL_DURATION not in columns:
        insert_index = columns.index(MICROBENCH_DURATION) + 1 if MICROBENCH_DURATION in columns else len(BASE_COLUMNS)
        columns.insert(insert_index, MICROBENCH_KERNEL_DURATION)

    for profiling_col in PROFILING_AVERAGE_EXTRA_COLUMNS:
        microbench_col = to_microbench_column(profiling_col)
        if profiling_col in columns and microbench_col not in columns:
            insert_index = columns.index(profiling_col)
            columns.insert(insert_index, microbench_col)

    # 确保 EP Size 列存在于正确的位置（在 Output Formats 后、Duration 前）
    # 注意：EP Size 仅对 DispatchFFNCombine 有意义，但如果 CSV 中已有该列则保留
    if EP_SIZE_COL in columns:
        # 已存在，保持原位置
        pass
    # 注意：不要自动添加 EP Size 列，只有 DispatchFFNCombine.csv 才需要

    return columns


def normalize_row_for_columns(row: dict[str, str], columns: list[str]) -> dict[str, str]:
    normalized_row = dict(row)
    if LEGACY_MICROBENCH_DURATION in normalized_row and MICROBENCH_DURATION not in normalized_row:
        normalized_row[MICROBENCH_DURATION] = normalized_row.get(LEGACY_MICROBENCH_DURATION, "")
    return {column: normalized_row.get(column, "") for column in columns}


def build_gap_record(csv_path: Path, row: dict[str, str]) -> GapRecord | None:
    microbench_us = parse_float(row.get(MICROBENCH_KERNEL_DURATION, "")) or parse_float(row.get(MICROBENCH_DURATION, ""))
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

    # 判断是否是 DFC CSV，如果是则使用 DFC_MATCH_COLUMNS
    is_dfc_csv = csv_path.stem == DISPATCH_FFN_COMBINE_OP_NAME
    match_columns = DFC_MATCH_COLUMNS if is_dfc_csv else MATCH_COLUMNS

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
            if build_signature(existing_row, match_columns) == build_signature(new_row, match_columns):
                if LEGACY_MICROBENCH_DURATION in existing_row and MICROBENCH_DURATION not in existing_row:
                    existing_row[MICROBENCH_DURATION] = existing_row.get(LEGACY_MICROBENCH_DURATION, "")
                old_microbench = existing_row.get(MICROBENCH_DURATION, "")
                existing_row[MICROBENCH_DURATION] = new_row.get(MICROBENCH_DURATION, "")
                if MICROBENCH_TASK_DURATION in new_row:
                    existing_row[MICROBENCH_TASK_DURATION] = new_row.get(MICROBENCH_TASK_DURATION, "")
                if MICROBENCH_KERNEL_DURATION in new_row:
                    existing_row[MICROBENCH_KERNEL_DURATION] = new_row.get(MICROBENCH_KERNEL_DURATION, "")
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
    for result in sorted(results, key=lambda item: (item.added_count, item.updated_count, item.csv_path.name),
                         reverse=True):
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
    for result in sorted(results, key=lambda item: (item.added_count, item.updated_count, item.csv_path.name),
                         reverse=True):
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
        report_file.write(
            make_markdown_table(["Operator", "Updated", "Added", "Unchanged", "Missing Samples"], update_rows))
        report_file.write("\n\n## Duration Gap Hotspots\n")
        report_file.write(
            make_markdown_table(["Operator", "MicroBench(us)", "Profiling(us)", "Abs Diff(us)", "MB/Profile", "Shape"],
                                gap_rows))
        report_file.write("\n\nFull hotspot CSV: ")
        report_file.write(full_gap_csv_path.name)
        report_file.write("\n")

    return report_path, full_gap_csv_path


def cleanup_prof_dirs(prof_dirs: set[Path], keep_all: bool = False) -> None:
    """清理 profiler 数据目录。

    Args:
        prof_dirs: 要清理的 profiler 目录集合
        keep_all: 如果为 True，保留所有目录，只打印日志
    """
    if keep_all:
        for prof_dir in sorted(prof_dirs):
            if prof_dir.exists():
                print(f"[KEEP] preserved profiler data: {prof_dir}")
        return

    for prof_dir in sorted(prof_dirs):
        if prof_dir.exists():
            shutil.rmtree(prof_dir)
            print(f"[CLEAN] removed {prof_dir}")


def main() -> None:
    args = build_argparser().parse_args()
    selected_ops = validate_selected_ops(args.op)
    if not args.prof_path:
        ensure_npu_available()

    succeeded = False
    prof_dirs: set[Path] = set()
    target_data_dir = get_target_data_dir(args.device, args.vllm_ascend_version)
    aggregated_rows: dict[str, list[dict[str, str]]] = {}

    try:
        # 单独处理 DispatchFFNCombine（使用内置 benchmark，不使用 msprof）
        selected_ops_set = set(selected_ops) if selected_ops else None
        should_run_dispatch_ffn_combine = (
                selected_ops_set is None
                or normalize_op_name(DISPATCH_FFN_COMBINE_OP_NAME) in selected_ops_set
        )

        if should_run_dispatch_ffn_combine:
            print(f"\n[RUN] {DISPATCH_FFN_COMBINE_OP_NAME}: using built-in benchmark (not msprof)")
            dfc_csv_path = run_dispatch_ffn_combine_direct(
                device=args.device,
                vllm_ascend_version=args.vllm_ascend_version,
                ep_size=args.ep_size,
                balanced=args.balanced,
            )
            dfc_rows = parse_dispatch_ffn_combine_csv(dfc_csv_path)
            aggregated_rows.update(dfc_rows)
            print(
                f"[DONE] {DISPATCH_FFN_COMBINE_OP_NAME} results: {len(dfc_rows.get(DISPATCH_FFN_COMBINE_OP_NAME, []))} rows")

            # 清理临时 CSV 文件
            if dfc_csv_path.exists():
                dfc_csv_path.unlink()

        # 其他算子使用 msprof
        # 计算 other_ops：排除 DispatchFFNCombine
        if selected_ops:
            other_ops = [
                op for op in selected_ops
                if normalize_op_name(op) != normalize_op_name(DISPATCH_FFN_COMBINE_OP_NAME)
            ]
        else:
            # selected_ops 为 None 表示全量运行，此时 other_ops 也为 None
            # 但 run_msprof 会默认 exclude_dfc=True
            other_ops = None

        if args.prof_path:
            prof_dirs = resolve_prof_dirs(args.prof_path)
        elif other_ops is not None and len(other_ops) == 0:
            # 只有 DispatchFFNCombine 被选中，跳过 msprof
            print("[SKIP] No other operators selected, skipping msprof")
        else:
            prof_dirs = run_msprof(
                device=args.device,
                vllm_ascend_version=args.vllm_ascend_version,
                exclude_dfc=True,
                selected_ops=other_ops,
            )

        if prof_dirs:
            op_summary_files = find_op_summary_files(prof_dirs)
            task_time_files = find_task_time_files(prof_dirs)
            kernel_details_files = find_kernel_details_files(prof_dirs)
            msprof_rows = aggregate_op_summary(op_summary_files)
            task_time_duration_map = aggregate_task_time(op_summary_files, task_time_files)
            # Duration priority: op_summary default < task_time < kernel_details.
            msprof_rows = attach_task_durations(msprof_rows, task_time_duration_map)
            kernel_duration_map = aggregate_kernel_details(kernel_details_files)
            msprof_rows = attach_kernel_durations(msprof_rows, kernel_duration_map)
            # 移除 DFC 行，避免覆盖内置 benchmark 数据
            msprof_rows.pop(DISPATCH_FFN_COMBINE_OP_NAME, None)
            aggregated_rows.update(msprof_rows)

        aggregated_rows = filter_aggregated_rows(aggregated_rows, selected_ops)
        if not aggregated_rows:
            print("[WARN] No aggregated rows found, nothing to update")
            return

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
        succeeded = True
    finally:
        if not args.prof_path and succeeded:
            cleanup_prof_dirs(prof_dirs)
        elif not args.prof_path and prof_dirs:
            preserved = ", ".join(str(path) for path in sorted(prof_dirs))
            print(f"[PRESERVE] profiling data kept for debugging: {preserved}")


if __name__ == "__main__":
    main()
