"""
Run PadV3 microbenchmark cases on Ascend NPU, CUDA, or CPU.

Purpose:
  Read PadV3 rows, rebuild input tensor, and execute padding operation.
  Maps to aclnnPadV3 on NPU.
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
        description="Run PadV3 workload replay on Ascend NPU/CUDA/CPU.",
        usage_examples=[
            "py -3 tools/perf_data_collection/op_replay/PadV3_run.py --device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.16.0",
        ],
        version_help="vLLM-Ascend version.",
    )


def run_row(csv_path, row_index: int, row: dict[str, str]) -> None:
    runtime_torch, runtime_torch_npu = get_runtime_modules()
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = parse_list_field(row["Input Data Types"])
    output_shapes = [parse_shape(item) for item in parse_list_field(row["Output Shapes"])]

    if not input_shapes or not output_shapes:
        print(f"[SKIP] {csv_path}:{row_index} Missing shapes")
        return

    # Resolve device
    device_type = "cpu"
    if hasattr(runtime_torch, "npu") and runtime_torch.npu.is_available():
        device_type = "npu"
    elif hasattr(runtime_torch, "cuda") and runtime_torch.cuda.is_available():
        device_type = "cuda"

    # Reconstruction
    x = build_input_tensor(input_shapes[0], input_formats[0], input_dtypes[0])
    if device_type == "cuda":
        x = x.cuda()
    elif device_type == "cpu":
        x = x.cpu()

    # Infer paddings from shape difference
    paddings = []
    for in_dim, out_dim in zip(reversed(input_shapes[0]), reversed(output_shapes[0])):
        diff = out_dim - in_dim
        paddings.extend([0, diff])

        # microbench_api
    if device_type == "npu":
        result = runtime_torch.nn.functional.pad(x, paddings, mode='constant', value=0.0)
        runtime_torch.npu.synchronize()
    else:
        # Fallback to standard torch pad for CUDA/CPU
        import torch.nn.functional as F
        result = F.pad(x, paddings)
        if device_type == "cuda":
            runtime_torch.cuda.synchronize()

    print(
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={tuple(result.shape)}"
    )


def main() -> None:
    args = build_argparser().parse_args()
    try:
        ensure_npu_available()
    except RuntimeError:
        print("Warning: NPU not found, attempting to run on alternative device.")

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_ascend_version,
    )
    csv_paths = sorted(target_data_dir.rglob("PadV3.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No PadV3.csv found under {target_data_dir}")

    total_rows = 0
    for csv_path, row_index, row in iter_csv_rows(target_data_dir, "PadV3.csv"):
        run_row(csv_path, row_index, row)
        total_rows += 1

    print(
        f"Processed {total_rows} PadV3 rows from {len(csv_paths)} csv file(s) "
        f"under {target_data_dir}."
    )


if __name__ == "__main__":
    main()
