"""
Replay MatMulCommon cases from the performance database on Ascend NPU.

Purpose:
  Read MatMulCommon rows from
  profiling_database/data/{device}/vllm_ascend/{version}/MatMulCommon.csv,
  rebuild input tensors from the recorded shapes, formats, and dtypes,
  then execute torch.mm() matching the specific dimensions.

Usage:
  python tools/perf_data_collection/op_replay/MatMulCommon_run.py ^
    --device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.15.0

Arguments:
  --device                Selects the device directory under profiling_database/data.
  --vllm-ascend-version   Selects the version directory under {device}/vllm_ascend.
"""

from __future__ import annotations

from common import (
    build_input_tensor,
    build_standard_argparser,
    ensure_npu_available,
    get_replay_repeat_count,
    get_runtime_modules,
    get_target_data_dir,
    iter_repeated_csv_rows,
    parse_list_field,
    parse_shape,
)


def build_argparser():
    return build_standard_argparser(
        description=(
            "Run MatMulCommon workload replay on Ascend NPU/CUDA/CPU.\n"
            "The script reads MatMulCommon.csv under the selected device and\n"
            "vllm_ascend version directory, reconstructs input tensors from\n"
            "Input Shapes / Input Formats / Input Data Types, then runs torch.mm()."
        ),
        usage_examples=[
            "py -3 tools/perf_data_collection/op_replay/MatMulCommon_run.py --device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.15.0",
        ],
        version_help="vLLM-Ascend version, e.g. 0.15.0.",
    )


def run_row(csv_path, row_index: int, row: dict[str, str]) -> None:
    runtime_torch, _ = get_runtime_modules()
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = [
        dt if dt.startswith("DT_") else f"DT_{dt}"
        for dt in parse_list_field(row["Input Data Types"])
    ]

    if len(input_shapes) < 2 or len(input_formats) < 2 or len(input_dtypes) < 2:
        raise ValueError("MatMulCommon expects at least two inputs")

    a_tensor = build_input_tensor(
        shape=input_shapes[0],
        input_format=input_formats[0],
        dtype_name=input_dtypes[0],
        transpose=False,
    )
    b_tensor = build_input_tensor(
        shape=input_shapes[1],
        input_format=input_formats[1],
        dtype_name=input_dtypes[1],
        transpose=True,
    )

    result = runtime_torch.mm(a_tensor, b_tensor)
    if hasattr(runtime_torch, "npu") and runtime_torch.npu.is_available():
        runtime_torch.npu.synchronize()
    elif hasattr(runtime_torch, "cuda") and runtime_torch.cuda.is_available():
        runtime_torch.cuda.synchronize()

    print(
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={tuple(result.shape)}"
    )


def main() -> None:
    args = build_argparser().parse_args()
    repeat_count = get_replay_repeat_count(args.repeat_count)
    ensure_npu_available()

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_ascend_version,
    )
    csv_paths = sorted(target_data_dir.rglob("MatMulCommon.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No MatMulCommon.csv found under {target_data_dir}")

    total_rows = 0
    for csv_path, row_index, row in iter_repeated_csv_rows(target_data_dir, "MatMulCommon.csv", repeat_count):
        run_row(csv_path, row_index, row)
        total_rows += 1

    print(
        f"Processed {total_rows} MatMulCommon rows from {len(csv_paths)} csv file(s) "
        f"under {target_data_dir}."
    )


if __name__ == "__main__":
    main()
