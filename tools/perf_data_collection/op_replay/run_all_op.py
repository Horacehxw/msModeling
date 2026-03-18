"""
Run all operator replay scripts in the current op_run directory.

Purpose:
  Discover every *_run.py script next to this file and execute each one with
  the same --device and --vllm-ascend-version arguments.

Usage:
  python tensor_cast/performance_model/perf_database/op_run/run_all_op.py ^
    --device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.13.0

Arguments:
  --device                Passed through to every operator replay script.
  --vllm-ascend-version   Passed through to every operator replay script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from common import SUPPORTED_DEVICES, check_version, get_target_data_dir


SCRIPT_DIR = Path(__file__).resolve().parent
SELF_NAME = Path(__file__).name


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Run all operator replay scripts under perf_database/op_run.\n"
            "Each operator script is executed once with the same device and\n"
            "vllm_ascend version arguments."
        ),
        epilog=(
            "Usage examples:\n"
            "  py -3 tensor_cast/performance_model/perf_database/op_run/run_all_op.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.13.0\n"
            "  python tensor_cast/performance_model/perf_database/op_run/run_all_op.py "
            "--device TEST_DEVICE --vllm-ascend-version 0.9.2\n\n"
            "Parameter notes:\n"
            "  --device                Passed through to every *_run.py script.\n"
            "  --vllm-ascend-version   Passed through to every *_run.py script.\n"
            "  -h, --help              Show this help message and exit."
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
        help="vLLM-Ascend version, e.g. 0.9.2.",
    )
    return parser


def discover_run_scripts() -> list[Path]:
    scripts = []
    for script_path in sorted(SCRIPT_DIR.glob("*_run.py")):
        if script_path.name == SELF_NAME:
            continue
        scripts.append(script_path)
    return scripts


def get_csv_name(script_path: Path) -> str:
    return f"{script_path.stem.removesuffix('_run')}.csv"


def has_operator_csv(target_data_dir: Path, csv_name: str) -> bool:
    return any(target_data_dir.rglob(csv_name))


def run_script(script_path: Path, device: str, vllm_ascend_version: str) -> None:
    command = [
        sys.executable,
        str(script_path),
        "--device",
        device,
        "--vllm-ascend-version",
        vllm_ascend_version,
    ]
    print(f"[RUN] {script_path.name}")
    subprocess.run(command, check=True, cwd=SCRIPT_DIR)
    print(f"[DONE] {script_path.name}")


def main() -> None:
    args = build_argparser().parse_args()
    scripts = discover_run_scripts()
    if not scripts:
        raise FileNotFoundError(f"No operator run scripts found under {SCRIPT_DIR}")

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_ascend_version,
    )
    executed_count = 0
    skipped_count = 0

    for script_path in scripts:
        csv_name = get_csv_name(script_path)
        if not has_operator_csv(target_data_dir, csv_name):
            print(f"No {csv_name} operator file found in this database. Skipping.")
            skipped_count += 1
            continue

        try:
            run_script(
                script_path=script_path,
                device=args.device,
                vllm_ascend_version=args.vllm_ascend_version,
            )
            executed_count += 1
        except subprocess.CalledProcessError as exc:
            if exc.returncode != 0:
                print(f"[FAIL] {script_path.name} exited with code {exc.returncode}")
                raise
        except FileNotFoundError:
            print(f"No {csv_name} operator file found in this database. Skipping.")
            skipped_count += 1

    print(
        f"Executed {executed_count} operator run script(s), skipped {skipped_count} "
        f"script(s) under {SCRIPT_DIR}."
    )


if __name__ == "__main__":
    main()
