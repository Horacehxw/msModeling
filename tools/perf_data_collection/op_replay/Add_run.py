"""
Replay Add cases from the performance database on Ascend NPU, CUDA, or CPU.

Purpose:
  Read Add rows from
  perf_database/data/{device}/vllm_ascend/{version}/Add.csv,
  rebuild input tensors from the recorded shapes, formats, and dtypes,
  then execute torch.add() workload.

Usage:
  python tools/perf_data_collection/op_replay/Add_run.py ^
    --device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.13.0
"""

from __future__ import annotations

from common import (
    build_input_tensor,
    build_standard_argparser,
    ensure_npu_available,
    get_runtime_modules,
    get_target_data_dir,
    iter_csv_rows,
    parse_list_field,
    parse_shape,
)


def build_argparser():
    return build_standard_argparser(
        description=(
            "Run Add workload replay on Ascend NPU/CUDA/CPU.\n"
            "The script reads Add.csv under the selected device and\n"
            "vllm_ascend version directory, reconstructs input tensors from\n"
            "Input Shapes / Input Formats / Input Data Types, then runs torch.add()."
        ),
        usage_examples=[
            "python tools/perf_data_collection/op_replay/Add_run.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.13.0",
        ],
        version_help="vLLM-Ascend version, e.g. 0.9.2.",
    )


def run_row(csv_path, row_index: int, row: dict[str, str]) -> None:
    runtime_torch, _ = get_runtime_modules()
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = parse_list_field(row["Input Data Types"])

    if not input_shapes:
        print(f"[SKIP] {csv_path}:{row_index} No input shapes found")
        return

    # Create first tensor
    dtype_name_0 = input_dtypes[0] if len(input_dtypes) > 0 else "DT_FLOAT"
    if not dtype_name_0.startswith("DT_"):
        dtype_name_0 = f"DT_{dtype_name_0}"

    a_tensor = build_input_tensor(
        shape=input_shapes[0],
        input_format=input_formats[0] if len(input_formats) > 0 else "ND",
        dtype_name=dtype_name_0,
    )

    # Handle second input
    if len(input_shapes) >= 2:
        dtype_name_1 = input_dtypes[1] if len(input_dtypes) > 1 else "DT_FLOAT"
        if not dtype_name_1.startswith("DT_"):
            
            dtype_name_1 = f"DT_{dtype_name_1}"

        b_tensor = build_input_tensor(
            shape=input_shapes[1],
            input_format=input_formats[1] if len(input_formats) > 1 else "ND",
            dtype_name=dtype_name_1,
        )
        result = runtime_torch.add(a_tensor, b_tensor)
    else:
        # If second shape is missing, it might be a scalar add. 
        # Using a scalar 1.0 as a default fallback to ensure execution.
        result = runtime_torch.add(a_tensor, 1.0)

    # Synchronization
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
    ensure_npu_available()

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_ascend_version,
    )
    csv_paths = sorted(target_data_dir.rglob("Add.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No Add.csv found under {target_data_dir}")

    total_rows = 0
    for csv_path, row_index, row in iter_csv_rows(target_data_dir, "Add.csv"):
        run_row(csv_path, row_index, row)
        total_rows += 1

    print(
        f"Processed {total_rows} Add rows from {len(csv_paths)} csv file(s) "
        f"under {target_data_dir}."
    )


if __name__ == "__main__":
    main()
