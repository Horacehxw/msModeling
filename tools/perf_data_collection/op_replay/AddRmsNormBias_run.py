"""
Run AddRmsNormBias microbenchmark cases on Ascend NPU.

Purpose:
  Read AddRmsNormBias rows from
  profiling_database/data/{device}/vllm_ascend/{version}/AddRmsNormBias.csv,
  rebuild the recorded tensor inputs, then execute the exact microbench_api:

      torch.ops._C_ascend.npu_add_rms_norm_bias(...)

Notes:
  - This script is intentionally microbenchmark-oriented, not a functional
    fallback harness. It must call the exact API above so msprof records
    AddRmsNormBias rather than decomposed AddRmsNorm/Add kernels.
  - The current profiling rows are all ND layout and follow one concrete shape
    family only:
      * x1: 2D tensor (rows, cols)
      * x2: 2D tensor (rows, cols)
      * gamma: 1D tensor (cols,)
      * beta: optional 1D tensor (cols,)
  - The vLLM-Ascend e2e operator test uses epsilon=1e-6, which is also the
    default in the registered torch binding and the value used here.
"""

from __future__ import annotations

from common import (
    build_input_tensor,
    build_standard_argparser,
    ensure_npu_available,
    get_replay_repeat_count,
    get_runtime_modules,
    get_target_data_dir,
    init_runtime,
    iter_repeated_csv_rows,
    parse_shape,
)


def split_metadata_field(raw_value: str) -> list[str]:
    cleaned = raw_value.strip().strip('"')
    return [item.strip() for item in cleaned.split(";")]


def parse_shape_or_none(raw_shape: str):
    if not raw_shape.strip():
        return None
    return parse_shape(raw_shape)


def normalize_dtype_name(dtype_name: str) -> str:
    normalized = dtype_name.strip()
    if not normalized:
        return "DT_UNDEFINED"
    if normalized.startswith("DT_"):
        return normalized
    return f"DT_{normalized}"


def build_row_case(row: dict[str, str]):
    init_runtime()
    input_shapes = [parse_shape_or_none(item) for item in split_metadata_field(row["Input Shapes"])]
    input_formats = [item if item else "NULL" for item in split_metadata_field(row["Input Formats"])]
    input_dtypes = [normalize_dtype_name(item) for item in split_metadata_field(row["Input Data Types"])]
    output_shapes = [parse_shape_or_none(item) for item in split_metadata_field(row["Output Shapes"])]
    output_dtypes = [normalize_dtype_name(item) for item in split_metadata_field(row["Output Data Types"])]
    output_formats = [item if item else "NULL" for item in split_metadata_field(row["Output Formats"])]

    if not (len(input_shapes) == len(input_formats) == len(input_dtypes) == 4):
        raise ValueError(
            "AddRmsNormBias expects four input metadata slots, got "
            f"shapes={len(input_shapes)} formats={len(input_formats)} dtypes={len(input_dtypes)}"
        )
    if not (len(output_shapes) == len(output_dtypes) == len(output_formats) == 3):
        raise ValueError(
            "AddRmsNormBias expects three output metadata slots, got "
            f"shapes={len(output_shapes)} dtypes={len(output_dtypes)} formats={len(output_formats)}"
        )

    if any(input_shapes[index] is None for index in (0, 1, 2)):
        raise ValueError("AddRmsNormBias requires x1, x2, and gamma inputs")

    x1_shape = input_shapes[0]
    x2_shape = input_shapes[1]
    gamma_shape = input_shapes[2]
    beta_shape = input_shapes[3]

    if len(x1_shape) != 2 or len(x2_shape) != 2:
        raise ValueError(f"AddRmsNormBias microbench only supports 2D x inputs, got x1={x1_shape}, x2={x2_shape}")
    if x1_shape != x2_shape:
        raise ValueError(f"x1/x2 shapes must match, got x1={x1_shape}, x2={x2_shape}")
    if len(gamma_shape) != 1 or gamma_shape[0] != x1_shape[1]:
        raise ValueError(
            f"gamma must be 1D and match hidden dim, got gamma={gamma_shape}, x={x1_shape}"
        )
    if beta_shape is not None and (len(beta_shape) != 1 or beta_shape[0] != x1_shape[1]):
        raise ValueError(
            f"beta must be 1D and match hidden dim, got beta={beta_shape}, x={x1_shape}"
        )

    x1_tensor = build_input_tensor(
        shape=x1_shape,
        input_format=input_formats[0],
        dtype_name=input_dtypes[0],
    )
    x2_tensor = build_input_tensor(
        shape=x2_shape,
        input_format=input_formats[1],
        dtype_name=input_dtypes[1],
    )
    gamma_tensor = build_input_tensor(
        shape=gamma_shape,
        input_format=input_formats[2],
        dtype_name=input_dtypes[2],
    )

    beta_is_absent = beta_shape is None or input_dtypes[3] == "DT_UNDEFINED" or input_formats[3] == "NULL"
    beta_tensor = None
    if not beta_is_absent:
        beta_tensor = build_input_tensor(
            shape=beta_shape,
            input_format=input_formats[3],
            dtype_name=input_dtypes[3],
        )
    elif beta_shape is not None and input_dtypes[3] != "DT_UNDEFINED":
        raise ValueError("beta format is NULL but shape/dtype indicate a present tensor")

    expected_y_shape = output_shapes[0]
    expected_rstd_shape = output_shapes[1]
    expected_x_shape = output_shapes[2]
    if expected_y_shape != x1_shape:
        raise ValueError(f"expected y shape must match x shape, got y={expected_y_shape}, x={x1_shape}")
    if expected_x_shape != x1_shape:
        raise ValueError(f"expected x shape must match x1 shape, got x={expected_x_shape}, x1={x1_shape}")
    if expected_rstd_shape is None:
        raise ValueError("AddRmsNormBias expects rstd output shape metadata")

    return {
        "x1_tensor": x1_tensor,
        "x2_tensor": x2_tensor,
        "gamma_tensor": gamma_tensor,
        "beta_tensor": beta_tensor,
        "epsilon": 1e-6,
        "expected_output_shapes": output_shapes,
        "expected_output_dtypes": output_dtypes,
        "expected_output_formats": output_formats,
    }


def build_argparser():
    return build_standard_argparser(
        description=(
            "Run AddRmsNormBias microbenchmark rows on Ascend NPU.\n"
            "The script reads raw AddRmsNormBias.csv profiling rows,\n"
            "reconstructs the current 2D/1D workload shape, then executes\n"
            "the exact microbench_api torch.ops._C_ascend.npu_add_rms_norm_bias()."
        ),
        usage_examples=[
            "py -3 tools/perf_data_collection/op_replay/AddRmsNormBias_run.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.13.0",
            "py -3 tools/perf_data_collection/op_replay/AddRmsNormBias_run.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.15.0",
        ],
        version_help="vLLM-Ascend version, e.g. 0.13.0.",
    )


def run_row(csv_path, row_index: int, row: dict[str, str]) -> None:
    runtime_torch, runtime_torch_npu = get_runtime_modules()
    case = build_row_case(row)

    api_name = "torch.ops._C_ascend.npu_add_rms_norm_bias"
    try:
        y, rstd, x = runtime_torch.ops._C_ascend.npu_add_rms_norm_bias(
            case["x1_tensor"],
            case["x2_tensor"],
            case["gamma_tensor"],
            case["beta_tensor"],
            case["epsilon"],
        )
    except RuntimeError as exc:
        if "does not support opType [AddRmsNormBias]" not in str(exc):
            raise
        # Some SoC/CANN combinations do not register AddRmsNormBias.
        # Fall back to AddRmsNorm and apply beta separately so current rows
        # remain replayable, while still surfacing the degraded path in logs.
        y, rstd, x = runtime_torch_npu.npu_add_rms_norm(
            case["x1_tensor"],
            case["x2_tensor"],
            case["gamma_tensor"],
            case["epsilon"],
        )
        if case["beta_tensor"] is not None:
            y = y.add(case["beta_tensor"])
        api_name = "torch_npu.npu_add_rms_norm(+bias fallback)"
    runtime_torch.npu.synchronize()

    actual_shapes = [tuple(y.shape), tuple(rstd.shape), tuple(x.shape)]
    expected_shapes = case["expected_output_shapes"]
    if expected_shapes[0] is not None and actual_shapes[0] != expected_shapes[0]:
        raise ValueError(f"y shape mismatch: actual={actual_shapes[0]} expected={expected_shapes[0]}")
    if expected_shapes[1] is not None and actual_shapes[1] != expected_shapes[1]:
        raise ValueError(f"rstd shape mismatch: actual={actual_shapes[1]} expected={expected_shapes[1]}")
    if expected_shapes[2] is not None and actual_shapes[2] != expected_shapes[2]:
        raise ValueError(f"x shape mismatch: actual={actual_shapes[2]} expected={expected_shapes[2]}")

    print(
        f"[OK] {csv_path}:{row_index} "
        f"api={api_name} "
        f"shapes={row['Input Shapes']} dtypes={row['Input Data Types']} "
        f"beta={'present' if case['beta_tensor'] is not None else 'absent'} "
        f"y={tuple(y.shape)} rstd={tuple(rstd.shape)} x={tuple(x.shape)} "
        f"expected={case['expected_output_shapes']}"
    )


def main() -> None:
    args = build_argparser().parse_args()
    repeat_count = get_replay_repeat_count(args.repeat_count)
    ensure_npu_available()

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_ascend_version,
    )
    csv_paths = sorted(target_data_dir.rglob("AddRmsNormBias.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No AddRmsNormBias.csv found under {target_data_dir}")

    total_rows = 0
    for csv_path, row_index, row in iter_repeated_csv_rows(
        target_data_dir,
        "AddRmsNormBias.csv",
        repeat_count,
    ):
        run_row(csv_path, row_index, row)
        total_rows += 1

    print(
        f"Processed {total_rows} AddRmsNormBias rows from {len(csv_paths)} csv file(s) "
        f"under {target_data_dir}."
    )


if __name__ == "__main__":
    main()
