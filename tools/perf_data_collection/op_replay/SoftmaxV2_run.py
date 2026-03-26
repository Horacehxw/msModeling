"""
Replay SoftmaxV2 cases from the performance database on Ascend NPU.

Purpose:
  Read SoftmaxV2 rows from
  profiling_database/data/{device}/vllm_ascend/{version}/SoftmaxV2.csv,
  rebuild the recorded tensor inputs, then execute
  torch.nn.functional.softmax() along the last dimension.
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
            "Run SoftmaxV2 workload replay on Ascend NPU.\n"
            "The script reads SoftmaxV2.csv under the selected device and\n"
            "vllm_ascend version directory, reconstructs input tensors from\n"
            "Input Shapes / Input Formats / Input Data Types, then runs\n"
            "torch.nn.functional.softmax(input, dim=-1)."
        ),
        usage_examples=[
            "py -3 tools/perf_data_collection/op_replay/SoftmaxV2_run.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.13.0",
        ],
        version_help="vLLM-Ascend version, e.g. 0.13.0.",
    )


def run_row(csv_path, row_index: int, row: dict[str, str]) -> None:
    runtime_torch, _ = get_runtime_modules()
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = parse_list_field(row["Input Data Types"])

    if len(input_shapes) != 1:
        raise ValueError(f"SoftmaxV2 expects exactly one input, got {len(input_shapes)}")

    dtype_name = input_dtypes[0]
    if not dtype_name.startswith("DT_"):
        dtype_name = f"DT_{dtype_name}"

    input_tensor = build_input_tensor(
        shape=input_shapes[0],
        input_format=input_formats[0],
        dtype_name=dtype_name,
    )
    output = runtime_torch.nn.functional.softmax(input_tensor, dim=-1)
    runtime_torch.npu.synchronize()

    print(
        f"[OK] {csv_path}:{row_index} "
        f"shape={tuple(input_tensor.shape)} dtype={row['Input Data Types']} "
        f"format={row['Input Formats']} output={tuple(output.shape)} dim=-1"
    )


def main() -> None:
    args = build_argparser().parse_args()
    repeat_count = get_replay_repeat_count(args.repeat_count)
    ensure_npu_available()

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_ascend_version,
    )
    csv_paths = sorted(target_data_dir.rglob("SoftmaxV2.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No SoftmaxV2.csv found under {target_data_dir}")

    total_rows = 0
    for csv_path, row_index, row in iter_repeated_csv_rows(target_data_dir, "SoftmaxV2.csv", repeat_count):
        run_row(csv_path, row_index, row)
        total_rows += 1

    print(
        f"Processed {total_rows} SoftmaxV2 rows from {len(csv_paths)} csv file(s) "
        f"under {target_data_dir}."
    )


if __name__ == "__main__":
    main()
