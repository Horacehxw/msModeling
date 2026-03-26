"""
Replay MaskedFill cases from the performance database on Ascend NPU.

Purpose:
  Read MaskedFill rows from
  profiling_database/data/{device}/vllm_ascend/{version}/MaskedFill.csv,
  rebuild input tensors from the recorded shapes, formats, and dtypes,
  then execute torch.Tensor.masked_fill_().

Usage:
  python tools/perf_data_collection/op_replay/MaskedFill_run.py ^
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
            "Run MaskedFill workload replay on Ascend NPU/CUDA/CPU.\n"
            "The script reads MaskedFill.csv under the selected device and\n"
            "vllm_ascend version directory, reconstructs input tensors from\n"
            "Input Shapes / Input Formats / Input Data Types, then runs\n"
            "tensor.masked_fill_(mask, value)."
        ),
        usage_examples=[
            "py -3 tools/perf_data_collection/op_replay/MaskedFill_run.py --device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.15.0",
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

    if len(input_shapes) < 2:
        raise ValueError("MaskedFill expects at least two inputs (tensor, mask)")

    tensor = build_input_tensor(
        shape=input_shapes[0],
        input_format=input_formats[0],
        dtype_name=input_dtypes[0],
    )
    
    # mask to bool
    mask_tensor = build_input_tensor(
        shape=input_shapes[1],
        input_format=input_formats[1],
        dtype_name="DT_BOOL",
    )

    # Inplace masked_fill with 0 usually, but can be configured per use
    # Using 0 as a default scalar fill value
    tensor.masked_fill_(mask_tensor, 0)
    if hasattr(runtime_torch, "npu") and runtime_torch.npu.is_available():
        runtime_torch.npu.synchronize()
    elif hasattr(runtime_torch, "cuda") and runtime_torch.cuda.is_available():
        runtime_torch.cuda.synchronize()

    print(
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={tuple(tensor.shape)}"
    )


def main() -> None:
    args = build_argparser().parse_args()
    repeat_count = get_replay_repeat_count(args.repeat_count)
    ensure_npu_available()

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_ascend_version,
    )
    csv_paths = sorted(target_data_dir.rglob("MaskedFill.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No MaskedFill.csv found under {target_data_dir}")

    total_rows = 0
    for csv_path, row_index, row in iter_repeated_csv_rows(target_data_dir, "MaskedFill.csv", repeat_count):
        run_row(csv_path, row_index, row)
        total_rows += 1

    print(
        f"Processed {total_rows} MaskedFill rows from {len(csv_paths)} csv file(s) "
        f"under {target_data_dir}."
    )


if __name__ == "__main__":
    main()
