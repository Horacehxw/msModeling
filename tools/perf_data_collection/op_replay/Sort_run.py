"""
Replay Sort cases from the performance database on Ascend NPU.

Purpose:
  Read Sort rows from
  profiling_database/data/{device}/vllm_ascend/{version}/Sort.csv,
  rebuild the recorded tensor inputs, then execute torch.sort()
  along the last dimension with descending order.

Notes:
  - The current v0.13.0 dataset is generated from sampling-path sort kernels.
  - The CSV does not record scalar flags, so the replay follows the existing
    top-k style path and uses dim=-1, descending=True.
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
            "Run Sort workload replay on Ascend NPU.\n"
            "The script reads Sort.csv under the selected device and\n"
            "vllm_ascend version directory, reconstructs input tensors from\n"
            "Input Shapes / Input Formats / Input Data Types, then runs\n"
            "torch.sort(input, dim=-1, descending=True)."
        ),
        usage_examples=[
            "py -3 tools/perf_data_collection/op_replay/Sort_run.py "
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
        raise ValueError(f"Sort expects exactly one input, got {len(input_shapes)}")

    dtype_name = input_dtypes[0]
    if not dtype_name.startswith("DT_"):
        dtype_name = f"DT_{dtype_name}"

    input_tensor = build_input_tensor(
        shape=input_shapes[0],
        input_format=input_formats[0],
        dtype_name=dtype_name,
    )
    values, indices = runtime_torch.sort(input_tensor, dim=-1, descending=True)
    runtime_torch.npu.synchronize()

    print(
        f"[OK] {csv_path}:{row_index} "
        f"shape={tuple(input_tensor.shape)} dtype={row['Input Data Types']} "
        f"values={tuple(values.shape)} indices={tuple(indices.shape)} "
        f"descending=True dim=-1"
    )


def main() -> None:
    args = build_argparser().parse_args()
    ensure_npu_available()

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_ascend_version,
    )
    csv_paths = sorted(target_data_dir.rglob("Sort.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No Sort.csv found under {target_data_dir}")

    total_rows = 0
    for csv_path, row_index, row in iter_csv_rows(target_data_dir, "Sort.csv"):
        run_row(csv_path, row_index, row)
        total_rows += 1

    print(
        f"Processed {total_rows} Sort rows from {len(csv_paths)} csv file(s) "
        f"under {target_data_dir}."
    )


if __name__ == "__main__":
    main()
