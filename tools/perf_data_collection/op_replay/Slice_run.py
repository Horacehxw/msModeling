"""
Replay Slice cases from the performance database on Ascend NPU.

Purpose:
  Read Slice rows from
  profiling_database/data/{device}/vllm_ascend/{version}/Slice.csv,
  rebuild the recorded tensor inputs, then execute torch_npu.npu_slice().

Notes:
  - The current dataset records the source tensor plus two INT64 vector inputs
    whose shapes equal the tensor rank. The CSV does not record the actual
    offset values.
  - For the current profiled rows, the output shapes match a front slice with
    zero offsets, so the replay reconstructs:
      * offsets = [0, 0, ..., 0]
      * sizes = output_shape
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


def normalize_dtype_name(dtype_name: str) -> str:
    return dtype_name if dtype_name.startswith("DT_") else f"DT_{dtype_name}"


def build_argparser():
    return build_standard_argparser(
        description=(
            "Run Slice workload replay on Ascend NPU.\n"
            "The script reads Slice.csv under the selected device and\n"
            "vllm_ascend version directory, reconstructs the source tensor,\n"
            "infers zero-based offsets from the current dataset, then runs\n"
            "torch_npu.npu_slice(input, offsets, sizes)."
        ),
        usage_examples=[
            "py -3 tools/perf_data_collection/op_replay/Slice_run.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.13.0",
        ],
        version_help="vLLM-Ascend version, e.g. 0.13.0.",
    )


def run_row(csv_path, row_index: int, row: dict[str, str]) -> None:
    runtime_torch, runtime_torch_npu = get_runtime_modules()
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = parse_list_field(row["Input Data Types"])
    output_shapes = [parse_shape(item) for item in parse_list_field(row["Output Shapes"])]

    if len(input_shapes) < 3:
        raise ValueError(f"Slice expects three inputs, got {len(input_shapes)}")
    if not output_shapes:
        raise ValueError("Slice requires at least one recorded output shape")

    source = build_input_tensor(
        shape=input_shapes[0],
        input_format=input_formats[0],
        dtype_name=normalize_dtype_name(input_dtypes[0]),
    )
    output_shape = output_shapes[0]
    offsets = [0] * len(output_shape)
    sizes = list(output_shape)

    output = runtime_torch_npu.npu_slice(source, offsets, sizes)
    runtime_torch.npu.synchronize()

    print(
        f"[OK] {csv_path}:{row_index} "
        f"source={tuple(source.shape)} output={tuple(output.shape)} "
        f"offsets={offsets} sizes={sizes}"
    )


def main() -> None:
    args = build_argparser().parse_args()
    repeat_count = get_replay_repeat_count(args.repeat_count)
    ensure_npu_available()

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_ascend_version,
    )
    csv_paths = sorted(target_data_dir.rglob("Slice.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No Slice.csv found under {target_data_dir}")

    total_rows = 0
    for csv_path, row_index, row in iter_repeated_csv_rows(target_data_dir, "Slice.csv", repeat_count):
        run_row(csv_path, row_index, row)
        total_rows += 1

    print(
        f"Processed {total_rows} Slice rows from {len(csv_paths)} csv file(s) "
        f"under {target_data_dir}."
    )


if __name__ == "__main__":
    main()
