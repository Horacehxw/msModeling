"""
Replay AscendQuantV2 cases from the performance database on Ascend NPU.

Purpose:
  Read AscendQuantV2 rows from
  perf_database/data/{device}/vllm_ascend/{version}/AscendQuantV2.csv,
  rebuild the recorded tensor inputs, then execute
  torch_npu.npu_quantize() with the same tensor metadata layout as the
  profiled row.

Notes:
  - The current ATLAS_800_A3_752T_128G_DIE/v0.13.0 dataset contains
    BF16 input / BF16 scale / BF16 zero_points rows only.
  - The repo's microbench path already uses axis=-1 and div_mode=False
    for AscendQuantV2 replay, so this script follows the same convention.
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


DTYPE_ALIASES = {
    "FLOAT": "DT_FLOAT",
    "FLOAT16": "DT_FLOAT16",
    "BF16": "DT_BF16",
    "DOUBLE": "DT_DOUBLE",
    "INT8": "DT_INT8",
    "UINT8": "DT_UINT8",
    "INT16": "DT_INT16",
    "INT32": "DT_INT32",
    "INT64": "DT_INT64",
    "BOOL": "DT_BOOL",
}


def normalize_dtype_name(dtype_name: str) -> str:
    normalized = dtype_name.strip()
    if normalized.startswith("DT_"):
        return normalized
    return DTYPE_ALIASES.get(normalized, normalized)


def to_quantized_dtype(dtype_name: str):
    runtime_torch, _ = get_runtime_modules()
    normalized = normalize_dtype_name(dtype_name)
    dtype = {
        "DT_INT8": runtime_torch.qint8,
        "DT_UINT8": runtime_torch.quint8,
        "DT_INT32": runtime_torch.int32,
    }.get(normalized)
    if dtype is None and hasattr(runtime_torch, "quint4x2"):
        dtype = {"DT_INT4": runtime_torch.quint4x2}.get(normalized)
    if dtype is None:
        raise ValueError(f"Unsupported AscendQuantV2 output dtype: {dtype_name}")
    return dtype


def build_row_case(row: dict[str, str]):
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = [normalize_dtype_name(item) for item in parse_list_field(row["Input Data Types"])]
    output_dtypes = [normalize_dtype_name(item) for item in parse_list_field(row["Output Data Types"])]

    if len(input_shapes) != len(input_formats) or len(input_shapes) != len(input_dtypes):
        raise ValueError("AscendQuantV2 input metadata length mismatch")
    if len(input_shapes) < 2:
        raise ValueError("AscendQuantV2 expects at least input and scales tensors")
    if len(output_dtypes) != 1:
        raise ValueError("AscendQuantV2 expects exactly one output dtype")

    x_tensor = build_input_tensor(
        shape=input_shapes[0],
        input_format=input_formats[0],
        dtype_name=input_dtypes[0],
    )
    scale_tensor = build_input_tensor(
        shape=input_shapes[1],
        input_format=input_formats[1],
        dtype_name=input_dtypes[1],
    )

    zero_points_tensor = None
    if len(input_shapes) >= 3:
        zero_points_tensor = build_input_tensor(
            shape=input_shapes[2],
            input_format=input_formats[2],
            dtype_name=input_dtypes[2],
        )

    return {
        "x_tensor": x_tensor,
        "scale_tensor": scale_tensor,
        "zero_points_tensor": zero_points_tensor,
        "quant_dtype": to_quantized_dtype(output_dtypes[0]),
    }


def build_argparser():
    return build_standard_argparser(
        description=(
            "Run AscendQuantV2 workload replay on Ascend NPU.\n"
            "The script reads AscendQuantV2.csv under the selected device and\n"
            "vllm_ascend version directory, reconstructs input tensors from\n"
            "Input Shapes / Input Formats / Input Data Types, then runs\n"
            "torch_npu.npu_quantize(axis=-1, div_mode=False)."
        ),
        usage_examples=[
            "py -3 tools/perf_data_collection/op_replay/AscendQuantV2_run.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.13.0",
            "python tools/perf_data_collection/op_replay/AscendQuantV2_run.py "
            "--device TEST_DEVICE --vllm-ascend-version 0.9.2",
        ],
        version_help="vLLM-Ascend version, e.g. 0.13.0.",
    )


def run_row(csv_path, row_index: int, row: dict[str, str]) -> None:
    runtime_torch, runtime_torch_npu = get_runtime_modules()
    case = build_row_case(row)

    result = runtime_torch_npu.npu_quantize(
        case["x_tensor"],
        case["scale_tensor"],
        case["zero_points_tensor"],
        case["quant_dtype"],
        -1,
        False,
    )
    runtime_torch.npu.synchronize()

    print(
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={tuple(result.shape)} "
        f"output_dtype={row['Output Data Types']}"
    )


def main() -> None:
    args = build_argparser().parse_args()
    ensure_npu_available()

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_ascend_version,
    )
    csv_paths = sorted(target_data_dir.rglob("AscendQuantV2.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No AscendQuantV2.csv found under {target_data_dir}")

    total_rows = 0
    for csv_path, row_index, row in iter_csv_rows(target_data_dir, "AscendQuantV2.csv"):
        run_row(csv_path, row_index, row)
        total_rows += 1

    print(
        f"Processed {total_rows} AscendQuantV2 rows from {len(csv_paths)} csv file(s) "
        f"under {target_data_dir}."
    )


if __name__ == "__main__":
    main()
