"""
Replay ReshapeAndCacheNdKernel cases from the performance database on Ascend NPU.

Purpose:
  Read ReshapeAndCacheNdKernel rows from
  perf_database/data/{device}/vllm_ascend/{version}/ReshapeAndCacheNdKernel.csv,
  rebuild the recorded tensor inputs, then execute
  torch_npu._npu_reshape_and_cache().

Notes:
  - The current ATLAS_800_A3_752T_128G_DIE/v0.13.0 dataset contains ND BF16
    rows only:
      * key: (tokens, kv_heads, head_dim)
      * value: (tokens, kv_heads, head_dim)
      * key_cache: (num_blocks, block_size, kv_heads, head_dim)
      * value_cache: (num_blocks, block_size, kv_heads, head_dim)
      * slot_mapping: (tokens,)
  - slot_mapping is rebuilt as a valid in-range int32 permutation over the
    available cache slots so the kernel can legally write all token rows.
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


def build_slot_mapping_tensor(
    slot_mapping_shape: tuple[int, ...],
    key_cache_shape: tuple[int, ...],
):
    runtime_torch, _ = get_runtime_modules()
    if len(slot_mapping_shape) != 1:
        raise ValueError(f"slot_mapping must be 1D, got shape={slot_mapping_shape}")
    if len(key_cache_shape) < 2:
        raise ValueError(f"key_cache rank must be >= 2, got shape={key_cache_shape}")

    token_count = slot_mapping_shape[0]
    total_slots = key_cache_shape[0] * key_cache_shape[1]
    if token_count > total_slots:
        raise ValueError(
            "slot_mapping token count exceeds cache capacity: "
            f"tokens={token_count}, total_slots={total_slots}, key_cache_shape={key_cache_shape}"
        )

    permutation = runtime_torch.randperm(total_slots, dtype=runtime_torch.int64)[:token_count]
    return permutation.to(dtype=runtime_torch.int32, device="npu")


def build_argparser():
    return build_standard_argparser(
        description=(
            "Run ReshapeAndCacheNdKernel workload replay on Ascend NPU.\n"
            "The script reads ReshapeAndCacheNdKernel.csv under the selected\n"
            "device and vllm_ascend version directory, reconstructs input\n"
            "tensors from Input Shapes / Input Formats / Input Data Types,\n"
            "builds a legal slot_mapping tensor, then runs\n"
            "torch_npu._npu_reshape_and_cache()."
        ),
        usage_examples=[
            "py -3 tools/perf_data_collection/op_replay/ReshapeAndCacheNdKernel_run.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.13.0",
            "python tools/perf_data_collection/op_replay/ReshapeAndCacheNdKernel_run.py "
            "--device TEST_DEVICE --vllm-ascend-version 0.9.2",
        ],
        version_help="vLLM-Ascend version, e.g. 0.13.0.",
    )


def build_row_case(row: dict[str, str]):
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = parse_list_field(row["Input Data Types"])

    if len(input_shapes) != 5 or len(input_formats) != 5 or len(input_dtypes) != 5:
        raise ValueError("ReshapeAndCacheNdKernel expects exactly five inputs")

    key = build_input_tensor(
        shape=input_shapes[0],
        input_format=input_formats[0],
        dtype_name=input_dtypes[0],
    )
    value = build_input_tensor(
        shape=input_shapes[1],
        input_format=input_formats[1],
        dtype_name=input_dtypes[1],
    )
    key_cache = build_input_tensor(
        shape=input_shapes[2],
        input_format=input_formats[2],
        dtype_name=input_dtypes[2],
    )
    value_cache = build_input_tensor(
        shape=input_shapes[3],
        input_format=input_formats[3],
        dtype_name=input_dtypes[3],
    )
    slot_mapping = build_slot_mapping_tensor(
        slot_mapping_shape=input_shapes[4],
        key_cache_shape=input_shapes[2],
    )

    return {
        "key": key,
        "value": value,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "slot_mapping": slot_mapping,
    }


def run_row(csv_path, row_index: int, row: dict[str, str]) -> None:
    runtime_torch, runtime_torch_npu = get_runtime_modules()
    case = build_row_case(row)

    runtime_torch_npu._npu_reshape_and_cache(
        case["key"],
        case["value"],
        case["key_cache"],
        case["value_cache"],
        case["slot_mapping"],
    )
    runtime_torch.npu.synchronize()

    print(
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} "
        f"key_cache={tuple(case['key_cache'].shape)} "
        f"value_cache={tuple(case['value_cache'].shape)}"
    )


def main() -> None:
    args = build_argparser().parse_args()
    ensure_npu_available()

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_ascend_version,
    )
    csv_paths = sorted(target_data_dir.rglob("ReshapeAndCacheNdKernel.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No ReshapeAndCacheNdKernel.csv found under {target_data_dir}"
        )

    total_rows = 0
    for csv_path, row_index, row in iter_csv_rows(
        target_data_dir,
        "ReshapeAndCacheNdKernel.csv",
    ):
        run_row(csv_path, row_index, row)
        total_rows += 1

    print(
        f"Processed {total_rows} ReshapeAndCacheNdKernel rows from {len(csv_paths)} "
        f"csv file(s) under {target_data_dir}."
    )


if __name__ == "__main__":
    main()
