"""
Replay TensorMove cases from the performance database on Ascend NPU.

Purpose:
  Read TensorMove rows from
  profiling_database/data/{device}/vllm_ascend/{version}/TensorMove.csv,
  rebuild the recorded source tensor, create a matching destination tensor,
  then execute torch.Tensor.copy_().

Notes:
  - TensorMove corresponds to data movement only. The profiling CSV stores
    the source tensor metadata; destination tensor metadata is reconstructed
    from the same shape / dtype / format.
  - The current ATLAS_800_A3_752T_128G_DIE/v0.13.0 dataset contains BF16 rows
    in ND and NCL layouts.
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
            "Run TensorMove workload replay on Ascend NPU.\n"
            "The script reads TensorMove.csv under the selected device and\n"
            "vllm_ascend version directory, reconstructs the recorded source\n"
            "tensor from Input Shapes / Input Formats / Input Data Types,\n"
            "creates a matching destination tensor, then runs\n"
            "torch.Tensor.copy_()."
        ),
        usage_examples=[
            "py -3 tools/perf_data_collection/op_replay/TensorMove_run.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.13.0",
            "python tools/perf_data_collection/op_replay/TensorMove_run.py "
            "--device TEST_DEVICE --vllm-ascend-version 0.9.2",
        ],
        version_help="vLLM-Ascend version, e.g. 0.13.0.",
    )


def build_row_case(row: dict[str, str]):
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = parse_list_field(row["Input Data Types"])

    if len(input_shapes) != 1 or len(input_formats) != 1 or len(input_dtypes) != 1:
        raise ValueError("TensorMove expects exactly one recorded input")

    src = build_input_tensor(
        shape=input_shapes[0],
        input_format=input_formats[0],
        dtype_name=input_dtypes[0],
    )
    dst = build_input_tensor(
        shape=input_shapes[0],
        input_format=input_formats[0],
        dtype_name=input_dtypes[0],
    )
    return {
        "src": src,
        "dst": dst,
    }


def run_row(csv_path, row_index: int, row: dict[str, str]) -> None:
    runtime_torch, _ = get_runtime_modules()
    case = build_row_case(row)

    case["dst"].copy_(case["src"])
    runtime_torch.npu.synchronize()

    print(
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={tuple(case['dst'].shape)}"
    )


def main() -> None:
    args = build_argparser().parse_args()
    ensure_npu_available()

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_ascend_version,
    )
    csv_paths = sorted(target_data_dir.rglob("TensorMove.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No TensorMove.csv found under {target_data_dir}")

    total_rows = 0
    for csv_path, row_index, row in iter_csv_rows(target_data_dir, "TensorMove.csv"):
        run_row(csv_path, row_index, row)
        total_rows += 1

    print(
        f"Processed {total_rows} TensorMove rows from {len(csv_paths)} csv file(s) "
        f"under {target_data_dir}."
    )


if __name__ == "__main__":
    main()
