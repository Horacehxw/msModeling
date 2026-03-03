import argparse
import csv
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
DURATION_US = "Duration(us)"
AVG_DURATION_US = "Average Duration(us)"


class AscendProfilerParser:
    """Parse Ascend kernel_details.csv and export averaged op duration by op type."""

    def __init__(self, device: str, kernel_details_path: str):
        self.device = device
        self.kernel_details_path = Path(kernel_details_path)
        self.base_dir = Path(__file__).resolve().parents[1]
        self.output_dir = self.base_dir / "data" / device

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

    def _load_rows(self) -> List[Dict[str, str]]:
        if not self.kernel_details_path.exists():
            raise FileNotFoundError(
                f"kernel_details.csv not found: {self.kernel_details_path}"
            )

        with self.kernel_details_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required_columns = {
                TYPE_COL,
                INPUT_SHAPES,
                INPUT_DTYPES,
                INPUT_FORMATS,
                OUTPUT_SHAPES,
                OUTPUT_DTYPES,
                OUTPUT_FORMATS,
                DURATION_US,
            }
            missing = required_columns - set(reader.fieldnames or [])
            if missing:
                missing_str = ", ".join(sorted(missing))
                raise ValueError(
                    "kernel_details.csv is missing required columns: "
                    f"{missing_str}"
                )

            return list(reader)

    def parse_and_export(self) -> List[Path]:
        rows = self._load_rows()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        grouped: Dict[
            Tuple[str, str, str], Dict[str, object]
        ] = defaultdict(  # (type, input_shapes, output_shapes)
            lambda: {
                "sum_duration": 0.0,
                "count": 0,
                "input_dtypes": "",
                "input_formats": "",
                "output_dtypes": "",
                "output_formats": "",
            }
        )

        for row in rows:
            op_type = self._safe_cell(row, TYPE_COL)
            input_shapes = self._safe_cell(row, INPUT_SHAPES)
            output_shapes = self._safe_cell(row, OUTPUT_SHAPES)
            key = (op_type, input_shapes, output_shapes)
            item = grouped[key]

            item["sum_duration"] = float(item["sum_duration"]) + self._parse_duration(
                self._safe_cell(row, DURATION_US)
            )
            item["count"] = int(item["count"]) + 1

            # Keep the first non-empty meta fields for this shape pair.
            if not item["input_dtypes"]:
                item["input_dtypes"] = self._safe_cell(row, INPUT_DTYPES)
            if not item["input_formats"]:
                item["input_formats"] = self._safe_cell(row, INPUT_FORMATS)
            if not item["output_dtypes"]:
                item["output_dtypes"] = self._safe_cell(row, OUTPUT_DTYPES)
            if not item["output_formats"]:
                item["output_formats"] = self._safe_cell(row, OUTPUT_FORMATS)

        rows_by_type: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for (op_type, input_shapes, output_shapes), item in grouped.items():
            if not op_type:
                continue
            avg_duration = float(item["sum_duration"]) / int(item["count"])
            rows_by_type[op_type].append(
                {
                    INPUT_SHAPES: input_shapes,
                    INPUT_DTYPES: item["input_dtypes"],
                    INPUT_FORMATS: item["input_formats"],
                    OUTPUT_SHAPES: output_shapes,
                    OUTPUT_DTYPES: item["output_dtypes"],
                    OUTPUT_FORMATS: item["output_formats"],
                    AVG_DURATION_US: f"{avg_duration:.6f}",
                }
            )

        output_files: List[Path] = []
        ordered_columns = [
            INPUT_SHAPES,
            INPUT_DTYPES,
            INPUT_FORMATS,
            OUTPUT_SHAPES,
            OUTPUT_DTYPES,
            OUTPUT_FORMATS,
            AVG_DURATION_US,
        ]
        for op_type, type_rows in rows_by_type.items():
            output_path = self.output_dir / f"{self._sanitize_filename(op_type)}.csv"
            # Stable ordering for reproducible output
            type_rows.sort(key=lambda r: (str(r[INPUT_SHAPES]), str(r[OUTPUT_SHAPES])))

            with output_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ordered_columns)
                writer.writeheader()
                writer.writerows(type_rows)
            output_files.append(output_path)

        return sorted(output_files)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse Ascend profiler kernel_details.csv and split by operator type "
            "with averaged duration grouped by input/output shapes."
        )
    )
    parser.add_argument(
        "--device",
        required=True,
        choices=SUPPORTED_DEVICES,
        help="Target device name used as output folder: perf_database/data/{device}/",
    )
    parser.add_argument(
        "--kernel-details-path",
        required=True,
        help="Path to Ascend profiler kernel_details.csv file.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    parser = AscendProfilerParser(
        device=args.device, kernel_details_path=args.kernel_details_path
    )
    output_files = parser.parse_and_export()
    print(
        f"Generated {len(output_files)} csv file(s) under "
        f"{parser.output_dir.as_posix()}"
    )


if __name__ == "__main__":
    main()
