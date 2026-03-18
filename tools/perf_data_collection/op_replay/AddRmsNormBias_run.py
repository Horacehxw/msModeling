"""
Replay AddRmsNormBias cases from the performance database on Ascend NPU.

Purpose:
  Read AddRmsNormBias rows from
  perf_database/data/{device}/vllm_ascend/{version}/AddRmsNormBias.csv,
  rebuild input tensors from the recorded shapes, formats, and dtypes,
  then execute torch.ops._C_ascend.npu_add_rms_norm_bias().

Usage:
  python tools/perf_data_collection/op_replay/AddRmsNormBias_run.py ^
    --device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.15.0

Arguments:
  --device                Selects the device directory under perf_database/data.
  --vllm-ascend-version   Selects the version directory under {device}/vllm_ascend.
"""

from __future__ import annotations

import collections

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
            "Run AddRmsNormBias workload replay on Ascend NPU/CUDA/CPU.\n"
            "The script reads AddRmsNormBias.csv under the selected device and\n"
            "vllm_ascend version directory, reconstructs input tensors from\n"
            "Input Shapes / Input Formats / Input Data Types, then runs\n"
            "torch.ops._C_ascend.npu_add_rms_norm_bias()."
        ),
        usage_examples=[
            "py -3 tools/perf_data_collection/op_replay/AddRmsNormBias_run.py --device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.15.0",
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

    if len(input_shapes) < 3:
        raise ValueError("AddRmsNormBias expects at least three inputs (x, residual, gamma)")

    x_tensor = build_input_tensor(
        shape=input_shapes[0],
        input_format=input_formats[0],
        dtype_name=input_dtypes[0],
    )
    residual_tensor = build_input_tensor(
        shape=input_shapes[1],
        input_format=input_formats[1],
        dtype_name=input_dtypes[1],
    )
    gamma_tensor = build_input_tensor(
        shape=input_shapes[2],
        input_format=input_formats[2],
        dtype_name=input_dtypes[2],
    )
    epsilon = 1e-6

    # Optional Beta tensor logic can be here if it arrives as the 4th input
    args = [x_tensor, residual_tensor, gamma_tensor]
    if len(input_shapes) >= 4 and len(input_shapes[3]) > 0:
        beta_tensor = build_input_tensor(
            shape=input_shapes[3],
            input_format=input_formats[3],
            dtype_name=input_dtypes[3],
        )
        args.append(beta_tensor)
        args.append(epsilon)
    else:
        # Default behavior for just x, residual, gamma, epsilon
        if len(input_shapes) == 3:
             args.append(runtime_torch.zeros_like(gamma_tensor))
             args.append(epsilon)

    # API interface from op_mapping.yaml: torch.ops._C_ascend.npu_add_rms_norm_bias
    result_tuple = runtime_torch.ops._C_ascend.npu_add_rms_norm_bias(*args)
    if hasattr(runtime_torch, "npu") and runtime_torch.npu.is_available():
        runtime_torch.npu.synchronize()
    elif hasattr(runtime_torch, "cuda") and runtime_torch.cuda.is_available():
        runtime_torch.cuda.synchronize()

    # result_tuple contains: (y, rstd, x)
    # y: resulting tensor after applying Add + RmsNorm
    output_shapes_str = tuple(tuple(t.shape) for t in result_tuple if isinstance(t, runtime_torch.Tensor))
    
    print(
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} outputs={output_shapes_str}"
    )


def main() -> None:
    args = build_argparser().parse_args()
    ensure_npu_available()

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_ascend_version,
    )
    csv_paths = sorted(target_data_dir.rglob("AddRmsNormBias.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No AddRmsNormBias.csv found under {target_data_dir}")

    total_rows = 0
    for csv_path, row_index, row in iter_csv_rows(target_data_dir, "AddRmsNormBias.csv"):
        run_row(csv_path, row_index, row)
        total_rows += 1

    print(
        f"Processed {total_rows} AddRmsNormBias rows from {len(csv_paths)} csv file(s) "
        f"under {target_data_dir}."
    )


if __name__ == "__main__":
    main()
