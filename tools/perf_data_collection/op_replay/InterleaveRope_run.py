"""
Replay InterleaveRope cases from the performance database on Ascend NPU.

Purpose:
  Read InterleaveRope rows from
  profiling_database/data/{device}/vllm_ascend/{version}/InterleaveRope.csv,
  rebuild input tensors from the recorded shapes, formats, and dtypes,
  then execute torch_npu.npu_interleave_rope().

Notes:
  - The current ATLAS_800_A3_752T_128G_DIE/v0.13.0 dataset contains a single
    BF16 ND case with three inputs:
      * x: (B, N, S, D)
      * cos: (B, 1, 1, D)
      * sin: (B, 1, 1, D)
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
            "Run InterleaveRope workload replay on Ascend NPU.\n"
            "The script reads InterleaveRope.csv under the selected device and\n"
            "vllm_ascend version directory, reconstructs input tensors from\n"
            "Input Shapes / Input Formats / Input Data Types, then runs\n"
            "torch_npu.npu_interleave_rope()."
        ),
        usage_examples=[
            "py -3 tools/perf_data_collection/op_replay/InterleaveRope_run.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.13.0",
            "python tools/perf_data_collection/op_replay/InterleaveRope_run.py "
            "--device TEST_DEVICE --vllm-ascend-version 0.9.2",
        ],
        version_help="vLLM-Ascend version, e.g. 0.13.0.",
    )


def build_row_case(row: dict[str, str]):
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = parse_list_field(row["Input Data Types"])

    if len(input_shapes) != 3 or len(input_formats) != 3 or len(input_dtypes) != 3:
        raise ValueError("InterleaveRope expects exactly three inputs")

    x_tensor = build_input_tensor(
        shape=input_shapes[0],
        input_format=input_formats[0],
        dtype_name=input_dtypes[0],
    )
    cos_tensor = build_input_tensor(
        shape=input_shapes[1],
        input_format=input_formats[1],
        dtype_name=input_dtypes[1],
    )
    sin_tensor = build_input_tensor(
        shape=input_shapes[2],
        input_format=input_formats[2],
        dtype_name=input_dtypes[2],
    )
    return {
        "x_tensor": x_tensor,
        "cos_tensor": cos_tensor,
        "sin_tensor": sin_tensor,
    }


def run_row(csv_path, row_index: int, row: dict[str, str]) -> None:
    runtime_torch, runtime_torch_npu = get_runtime_modules()
    case = build_row_case(row)

    output = runtime_torch_npu.npu_interleave_rope(
        case["x_tensor"],
        case["cos_tensor"],
        case["sin_tensor"],
    )
    runtime_torch.npu.synchronize()

    print(
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={tuple(output.shape)}"
    )


def main() -> None:
    args = build_argparser().parse_args()
    repeat_count = get_replay_repeat_count(args.repeat_count)
    ensure_npu_available()

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_ascend_version,
    )
    csv_paths = sorted(target_data_dir.rglob("InterleaveRope.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No InterleaveRope.csv found under {target_data_dir}")

    total_rows = 0
    for csv_path, row_index, row in iter_repeated_csv_rows(
        target_data_dir,
        "InterleaveRope.csv",
        repeat_count,
    ):
        run_row(csv_path, row_index, row)
        total_rows += 1

    print(
        f"Processed {total_rows} InterleaveRope rows from {len(csv_paths)} csv file(s) "
        f"under {target_data_dir}."
    )


if __name__ == "__main__":
    main()
