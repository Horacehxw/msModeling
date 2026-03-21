"""
Run Transpose microbenchmark cases on Ascend NPU, CUDA, or CPU.

Purpose:
  Read Transpose rows, execute torch.transpose().contiguous() to trigger aclnnTranspose on NPU.
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
        description="Run Transpose workload replay on Ascend NPU/CUDA/CPU.",
        usage_examples=[
            "py -3 tools/perf_data_collection/op_replay/Transpose_run.py --device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.16.0",
        ],
        version_help="vLLM-Ascend version.",
    )

def run_row(csv_path, row_index: int, row: dict[str, str]) -> None:
    runtime_torch, _ = get_runtime_modules()
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

    x = build_input_tensor(input_shapes[0], input_formats[0], input_dtypes[0])
    if device_type == "cuda":
        x = x.cuda()
    elif device_type == "cpu":
        x = x.cpu()
    
    # Infer transposed dims
    in_shape = input_shapes[0]
    out_shape = output_shapes[0]
    dim0, dim1 = 0, 1
    diff_dims = [i for i, (d1, d2) in enumerate(zip(in_shape, out_shape)) if d1 != d2]
    if len(diff_dims) >= 2:
        dim0, dim1 = diff_dims[0], diff_dims[1]
    elif len(in_shape) >= 2:
        dim0, dim1 = 0, 1

    # microbench_api: torch.transpose + contiguous
    result = runtime_torch.transpose(x, dim0, dim1).contiguous()
    
    # Synchronization
    if device_type == "npu":
        runtime_torch.npu.synchronize()
    elif device_type == "cuda":
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
    csv_paths = sorted(target_data_dir.rglob("Transpose.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No Transpose.csv found under {target_data_dir}")

    total_rows = 0
    for csv_path, row_index, row in iter_csv_rows(target_data_dir, "Transpose.csv"):
        run_row(csv_path, row_index, row)
        total_rows += 1

    print(
        f"Processed {total_rows} Transpose rows from {len(csv_paths)} csv file(s) "
        f"under {target_data_dir}."
    )

if __name__ == "__main__":
    main()
