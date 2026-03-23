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
import runpy
import subprocess
import sys

from common import SUPPORTED_DEVICES, check_version, get_target_data_dir, normalize_op_name


SCRIPT_DIR = Path(__file__).resolve().parent
SELF_NAME = Path(__file__).name

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Run all operator replay scripts under perf_database/op_run.\n"
            "Each operator script is executed once with the same device and\n"
            "vllm_ascend version arguments.\n"
            "By default, scripts run in-process so a single outer `msprof`\n"
            "session can capture all operators into one PROF_* directory."
        ),
        epilog=(
            "Usage examples:\n"
            "  py -3 tensor_cast/performance_model/perf_database/op_run/run_all_op.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.13.0\n"
            "  python tensor_cast/performance_model/perf_database/op_run/run_all_op.py "
            "--device TEST_DEVICE --vllm-ascend-version 0.9.2\n"
            "  msprof python tools/perf_data_collection/op_replay/run_all_op.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.15.0\n\n"
            "Parameter notes:\n"
            "  --device                Passed through to every *_run.py script.\n"
            "  --vllm-ascend-version   Passed through to every *_run.py script.\n"
            "  --execution-mode        `inprocess` keeps all operators in one Python process;\n"
            "                          `subprocess` preserves the old per-script child-process behavior.\n"
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
    parser.add_argument(
        "--execution-mode",
        choices=["inprocess", "subprocess"],
        default="inprocess",
        help=(
            "How to invoke each *_run.py script. Default: inprocess."
        ),
    )
    parser.add_argument(
        "--op",
        nargs="+",
        default=None,
        help=(
            "Optional operator names to run, e.g. MatMulV2 PadV3. "
            "Names may be given as OP, OP_run, or OP_run.py."
        ),
    )
    return parser


def discover_run_scripts() -> list[Path]:
    scripts = []
    for script_path in sorted(SCRIPT_DIR.glob("*_run.py")):
        if script_path.name == SELF_NAME:
            continue
        scripts.append(script_path)
    return scripts


def filter_run_scripts(scripts: list[Path], selected_ops: set[str] | None) -> list[Path]:
    if not selected_ops:
        return scripts
    return [script_path for script_path in scripts if normalize_op_name(script_path.stem) in selected_ops]


def get_csv_name(script_path: Path) -> str:
    return f"{script_path.stem.removesuffix('_run')}.csv"


def has_operator_csv(target_data_dir: Path, csv_name: str) -> bool:
    return any(target_data_dir.rglob(csv_name))


def run_script_subprocess(script_path: Path, device: str, vllm_ascend_version: str) -> None:
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


def run_script_inprocess(script_path: Path, device: str, vllm_ascend_version: str) -> None:
    original_argv = sys.argv[:]
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    sys.argv = [
        str(script_path),
        "--device",
        device,
        "--vllm-ascend-version",
        vllm_ascend_version,
    ]
    print(f"[RUN] {script_path.name}")
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = original_argv
    print(f"[DONE] {script_path.name}")


def run_script(
    script_path: Path,
    device: str,
    vllm_ascend_version: str,
    execution_mode: str,
) -> None:
    if execution_mode == "subprocess":
        run_script_subprocess(script_path, device, vllm_ascend_version)
        return
    run_script_inprocess(script_path, device, vllm_ascend_version)


def main() -> None:
    args = build_argparser().parse_args()
    selected_ops = None
    if args.op:
        selected_ops = {normalize_op_name(item) for item in args.op}
    scripts = discover_run_scripts()
    scripts = filter_run_scripts(scripts, selected_ops)
    if not scripts:
        if selected_ops:
            requested = ", ".join(sorted(selected_ops))
            raise FileNotFoundError(f"No matching operator run scripts found for: {requested}")
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
                execution_mode=args.execution_mode,
            )
            executed_count += 1
        except subprocess.CalledProcessError as exc:
            print(f"[FAIL] {script_path.name} exited with code {exc.returncode}")
            raise
        except SystemExit as exc:
            if exc.code not in (0, None):
                print(f"[FAIL] {script_path.name} exited with code {exc.code}")
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
