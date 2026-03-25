import argparse
import csv
import math
import statistics
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


SUPPORTED_DEVICES = [
    "TEST_DEVICE",
    "ATLAS_800_A2_376T_64G",
    "ATLAS_800_A2_313T_64G",
    "ATLAS_800_A2_280T_64G",
    "ATLAS_800_A2_280T_64G_PCIE",
    "ATLAS_800_A2_280T_32G_PCIE",
    "ATLAS_800_A3_752T_128G_DIE",
    "ATLAS_800_A3_560T_128G_DIE",
]

INPUT_SHAPES = "Input Shapes"
INPUT_DTYPES = "Input Data Types"
INPUT_FORMATS = "Input Formats"
OUTPUT_SHAPES = "Output Shapes"
OUTPUT_DTYPES = "Output Data Types"
OUTPUT_FORMATS = "Output Formats"
TYPE_COL = "Type"
OP_STATE = "OP State"
ACCELERATOR_CORE = "Accelerator Core"
DURATION_US = "Duration(us)"
AVG_DURATION_US = "Profiling Average Duration(us)"
STD_DURATION_US = "Profiling Std Duration(us)"
MEDIAN_DURATION_US = "Profiling Median Duration(us)"
EXTRA_NUMERIC_COLUMNS = [
    "aicore_time(us)",
    "aic_total_cycles",
    "aic_mac_time(us)",
    "aic_mac_ratio",
    "aic_scalar_time(us)",
    "aic_scalar_ratio",
    "aic_mte1_time(us)",
    "aic_mte1_ratio",
    "aic_mte2_time(us)",
    "aic_mte2_ratio",
    "aic_fixpipe_time(us)",
    "aic_fixpipe_ratio",
    "aic_icache_miss_rate",
    "aiv_time(us)",
    "aiv_total_cycles",
    "aiv_vec_time(us)",
    "aiv_vec_ratio",
    "aiv_scalar_time(us)",
    "aiv_scalar_ratio",
    "aiv_mte2_time(us)",
    "aiv_mte2_ratio",
    "aiv_mte3_time(us)",
    "aiv_mte3_ratio",
    "aiv_icache_miss_rate",
    "cube_utilization(%)",
]


def _render_progress(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return f"[{'-' * width}]"
    ratio = min(max(current / total, 0.0), 1.0)
    filled = min(width, int(ratio * width))
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def print_progress(*, stage: str, current: int, total: int, detail: str = "") -> None:
    bar = _render_progress(current, total)
    message = f"\r{stage} {bar} {current}/{total}"
    if detail:
        message += f" | {detail}"
    print(message, end="", flush=True)


def clear_progress() -> None:
    print("\r" + " " * 160 + "\r", end="", flush=True)


def profiling_column_name(column: str) -> str:
    return f"Profiling {column}"


def check_version(value: str) -> str:
    version = value.strip()
    if not re.fullmatch(r"[0-9A-Za-z]+(?:[._-][0-9A-Za-z]+)*", version):
        raise argparse.ArgumentTypeError(
            f"Invalid --vllm-ascend-version: {value!r}. "
            "Expected value like 0.9.2 or vllm0.13.0_torch2.8.0_cann8.3"
        )
    return version


def normalize_device_name(device: str) -> str:
    return device.strip()


def normalize_vllm_ascend_version(version: str) -> str:
    normalized = version.strip()
    if not normalized.startswith("v"):
        normalized = f"v{normalized}"
    return normalized


class KernelDetailsParser:
    """Parse one or more kernel_details*.csv files and export aggregated op stats by op type."""

    def __init__(self, device: str, kernel_details_path: str, vllm_ascend_version: str):
        self.device = device
        self.kernel_details_path = Path(kernel_details_path)
        self.vllm_ascend_version = normalize_vllm_ascend_version(vllm_ascend_version)
        self.device_dir = normalize_device_name(device)
        self.repo_root = Path(__file__).resolve().parents[2]
        self.output_dir = (
            self.repo_root
            / "tensor_cast"
            / "performance_model"
            / "perf_database"
            / "data"
            / self.device_dir
            / "vllm_ascend"
            / self.vllm_ascend_version
        )

    def _resolve_kernel_details_files(self) -> List[Path]:
        if not self.kernel_details_path.exists():
            raise FileNotFoundError(
                f"kernel_details source path not found: {self.kernel_details_path}"
            )
        if self.kernel_details_path.is_file():
            return [self.kernel_details_path]
        files = sorted(
            path
            for path in self.kernel_details_path.rglob("*.csv")
            if "kernel_details" in path.stem.lower()
        )
        if not files:
            raise FileNotFoundError(
                f"No CSV files with 'kernel_details' in the filename found under: {self.kernel_details_path}"
            )
        return files

    @staticmethod
    def _parse_duration(value: str) -> float:
        try:
            return float((value or "").strip())
        except ValueError:
            return 0.0

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name.strip())
        return sanitized or "UNKNOWN_TYPE"

    @staticmethod
    def _safe_cell(row: Dict[str, str], key: str) -> str:
        return (row.get(key, "") or "").strip()

    @staticmethod
    def _is_na_shape(value: str) -> bool:
        normalized = value.strip().strip('"').upper()
        return normalized == "N/A"

    # Kernel name normalization: map variant names to canonical kernel type.
    # - split_qkv_rmsnorm_rope_kernel_0: Triton JIT grid-config variant of
    #   split_qkv_rmsnorm_rope_kernel (same CANN kernel, different launch config
    #   for decode vs prefill). Merge into one CSV.
    _KERNEL_NAME_NORMALIZE: Dict[str, str] = {
        "split_qkv_rmsnorm_rope_kernel_0": "split_qkv_rmsnorm_rope_kernel",
    }

    @classmethod
    def _normalize_kernel_type(cls, op_type: str) -> str:
        return cls._KERNEL_NAME_NORMALIZE.get(op_type, op_type)

    def _load_rows(self) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        kernel_details_files = self._resolve_kernel_details_files()
        required_columns = {
            TYPE_COL,
            OP_STATE,
            ACCELERATOR_CORE,
            INPUT_SHAPES,
            INPUT_DTYPES,
            INPUT_FORMATS,
            OUTPUT_SHAPES,
            OUTPUT_DTYPES,
            OUTPUT_FORMATS,
            DURATION_US,
        }
        required_columns.update(EXTRA_NUMERIC_COLUMNS)

        total_files = len(kernel_details_files)
        for file_index, kernel_details_file in enumerate(kernel_details_files, start=1):
            with kernel_details_file.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                missing = required_columns - set(reader.fieldnames or [])
                if missing:
                    missing_str = ", ".join(sorted(missing))
                    raise ValueError(
                        f"{kernel_details_file} is missing required columns: "
                        f"{missing_str}"
                    )
                rows.extend(reader)
            print_progress(
                stage="Load",
                current=file_index,
                total=total_files,
                detail=kernel_details_file.parent.name,
            )
        return rows

    @staticmethod
    def _shape_key(row: Dict[str, object]) -> Tuple[str, str]:
        return (str(row.get(INPUT_SHAPES, "")), str(row.get(OUTPUT_SHAPES, "")))

    def parse_and_export(self) -> List[Path]:
        rows = self._load_rows()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        grouped: Dict[
            Tuple[str, str, str], Dict[str, object]
        ] = defaultdict(  # (type, input_shapes, output_shapes)
            lambda: {
                "sum_duration": 0.0,
                "sum_duration_sq": 0.0,
                "durations": [],
                "count": 0,
                "op_state": "",
                "accelerator_core": "",
                "input_dtypes": "",
                "input_formats": "",
                "output_dtypes": "",
                "output_formats": "",
                "sum_extra": {col: 0.0 for col in EXTRA_NUMERIC_COLUMNS},
            }
        )

        total_rows = len(rows)
        progress_interval = max(1, total_rows // 100) if total_rows else 1
        for row_index, row in enumerate(rows, start=1):
            op_type = self._normalize_kernel_type(self._safe_cell(row, TYPE_COL))
            input_shapes = self._safe_cell(row, INPUT_SHAPES)
            output_shapes = self._safe_cell(row, OUTPUT_SHAPES)
            if self._is_na_shape(input_shapes) or self._is_na_shape(output_shapes):
                continue
            key = (op_type, input_shapes, output_shapes)
            item = grouped[key]

            duration = self._parse_duration(self._safe_cell(row, DURATION_US))
            item["sum_duration"] = float(item["sum_duration"]) + duration
            item["sum_duration_sq"] = float(item["sum_duration_sq"]) + duration * duration
            item["durations"].append(duration)
            item["count"] = int(item["count"]) + 1
            for col in EXTRA_NUMERIC_COLUMNS:
                item["sum_extra"][col] = float(item["sum_extra"][col]) + self._parse_duration(
                    self._safe_cell(row, col)
                )

            # Keep the first non-empty meta fields for this shape pair.
            if not item["op_state"]:
                item["op_state"] = self._safe_cell(row, OP_STATE)
            if not item["accelerator_core"]:
                item["accelerator_core"] = self._safe_cell(row, ACCELERATOR_CORE)
            if not item["input_dtypes"]:
                item["input_dtypes"] = self._safe_cell(row, INPUT_DTYPES)
            if not item["input_formats"]:
                item["input_formats"] = self._safe_cell(row, INPUT_FORMATS)
            if not item["output_dtypes"]:
                item["output_dtypes"] = self._safe_cell(row, OUTPUT_DTYPES)
            if not item["output_formats"]:
                item["output_formats"] = self._safe_cell(row, OUTPUT_FORMATS)
            if row_index == total_rows or row_index % progress_interval == 0:
                print_progress(
                    stage="Aggregate",
                    current=row_index,
                    total=total_rows,
                    detail=op_type,
                )

        rows_by_type: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for (op_type, input_shapes, output_shapes), item in grouped.items():
            if not op_type:
                continue
            count = int(item["count"])
            avg_duration = float(item["sum_duration"]) / count
            avg_duration_sq = float(item["sum_duration_sq"]) / count
            variance = max(0.0, avg_duration_sq - avg_duration * avg_duration)
            std_duration = math.sqrt(variance)
            median_duration = statistics.median(item["durations"])
            avg_extra = {
                profiling_column_name(f"Average {col}"): (
                    float(item["sum_extra"][col]) / count
                )
                for col in EXTRA_NUMERIC_COLUMNS
            }
            rows_by_type[op_type].append(
                {
                    OP_STATE: item["op_state"],
                    ACCELERATOR_CORE: item["accelerator_core"],
                    INPUT_SHAPES: input_shapes,
                    INPUT_DTYPES: item["input_dtypes"],
                    INPUT_FORMATS: item["input_formats"],
                    OUTPUT_SHAPES: output_shapes,
                    OUTPUT_DTYPES: item["output_dtypes"],
                    OUTPUT_FORMATS: item["output_formats"],
                    AVG_DURATION_US: f"{avg_duration:.6f}",
                    MEDIAN_DURATION_US: f"{median_duration:.6f}",
                    STD_DURATION_US: f"{std_duration:.6f}",
                    **{k: f"{v:.6f}" for k, v in avg_extra.items()},
                }
            )

        output_files: List[Path] = []
        ordered_columns = [
            OP_STATE,
            ACCELERATOR_CORE,
            INPUT_SHAPES,
            INPUT_DTYPES,
            INPUT_FORMATS,
            OUTPUT_SHAPES,
            OUTPUT_DTYPES,
            OUTPUT_FORMATS,
            AVG_DURATION_US,
            MEDIAN_DURATION_US,
            STD_DURATION_US,
        ]
        ordered_columns.extend([profiling_column_name(f"Average {col}") for col in EXTRA_NUMERIC_COLUMNS])
        total_output_files = len(rows_by_type)
        for file_index, (op_type, type_rows) in enumerate(rows_by_type.items(), start=1):
            output_path = self.output_dir / f"{self._sanitize_filename(op_type)}.csv"
            normalized_rows = []
            for row in type_rows:
                normalized_rows.append({col: row.get(col, "") for col in ordered_columns})

            with output_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ordered_columns)
                writer.writeheader()
                writer.writerows(normalized_rows)
            output_files.append(output_path)
            print_progress(
                stage="Write",
                current=file_index,
                total=total_output_files,
                detail=output_path.name,
            )

        clear_progress()
        return sorted(output_files)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse kernel_details.csv and split by operator type "
            "with averaged duration grouped by input/output shapes."
        )
    )
    parser.add_argument(
        "--device",
        required=True,
        choices=SUPPORTED_DEVICES,
        help=(
            "Target device name used as output folder: "
            "tensor_cast/performance_model/perf_database/data/{device}/vllm_ascend/{version}/"
        ),
    )
    parser.add_argument(
        "--vllm-ascend-version",
        required=True,
        type=check_version,
        help="vLLM-Ascend version, e.g. 0.9.2.",
    )
    parser.add_argument(
        "--kernel-details-path",
        required=True,
        help=(
            "Path to a kernel_details*.csv file or a directory. "
            "If a directory is provided, the script recursively scans all CSV files whose filename contains "
            "'kernel_details'."
        ),
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    parser = KernelDetailsParser(
        device=args.device,
        kernel_details_path=args.kernel_details_path,
        vllm_ascend_version=args.vllm_ascend_version,
    )
    output_files = parser.parse_and_export()
    print(
        f"Generated {len(output_files)} csv file(s) under "
        f"{parser.output_dir.as_posix()} "
        f"from {args.kernel_details_path}"
    )


if __name__ == "__main__":
    main()


# Backward-compatible alias for external imports from older naming.
AscendProfilerParser = KernelDetailsParser
