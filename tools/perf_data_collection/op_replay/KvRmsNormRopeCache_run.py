"""
Replay KvRmsNormRopeCache cases from the performance database on Ascend NPU.

Purpose:
  Read KvRmsNormRopeCache rows from
  profiling_database/data/{device}/vllm_ascend/{version}/KvRmsNormRopeCache.csv,
  rebuild the recorded tensor inputs, infer the required scalar arguments,
  then execute torch_npu.npu_kv_rmsnorm_rope_cache().

Notes:
  - The current ATLAS_800_A3_752T_128G_DIE/v0.13.0 dataset contains one BF16
    row with 12 input slots. Only the first 7 slots are populated:
      * kv, gamma, cos, sin, index, k_cache, ckv_cache
    and the remaining optional quant / offset / v slots are empty.
  - The recorded outputs include both updated caches and the output KV tensors,
    so this replay enables `is_output_kv=True`.
"""

from __future__ import annotations

from common import (
    build_input_tensor,
    build_standard_argparser,
    ensure_npu_available,
    get_runtime_modules,
    get_target_data_dir,
    iter_csv_rows,
    parse_shape,
)


def split_metadata_field(raw_value: str) -> list[str]:
    cleaned = raw_value.strip().strip('"')
    return [item.strip() for item in cleaned.split(";")]


def parse_shape_or_none(raw_shape: str):
    if not raw_shape.strip():
        return None
    return parse_shape(raw_shape)


def build_index_tensor(index_shape: tuple[int, ...], cache_shape: tuple[int, ...]):
    runtime_torch, _ = get_runtime_modules()
    if len(index_shape) != 1:
        raise ValueError(f"index must be 1D, got shape={index_shape}")
    if len(cache_shape) < 2:
        raise ValueError(f"cache rank must be >= 2, got shape={cache_shape}")

    token_count = index_shape[0]
    total_slots = cache_shape[0] * cache_shape[1]
    if token_count > total_slots:
        raise ValueError(
            "index token count exceeds cache capacity: "
            f"tokens={token_count}, total_slots={total_slots}, cache_shape={cache_shape}"
        )

    return runtime_torch.arange(token_count, dtype=runtime_torch.int64, device="npu")


def infer_cache_mode(k_cache_shape: tuple[int, ...], index_shape: tuple[int, ...]) -> str:
    if len(k_cache_shape) != 4:
        raise ValueError(f"Unsupported k_cache shape for cache_mode inference: {k_cache_shape}")
    if len(index_shape) != 1:
        raise ValueError(f"Unsupported index shape for cache_mode inference: {index_shape}")
    # Current profiling row uses 4D BNSD caches and per-token flat indices.
    return "PA_BNSD"


def build_row_case(row: dict[str, str]):
    input_shapes = [parse_shape_or_none(item) for item in split_metadata_field(row["Input Shapes"])]
    input_dtypes = [item.strip() for item in split_metadata_field(row["Input Data Types"])]
    input_formats = [item.strip() if item.strip() else "NULL" for item in split_metadata_field(row["Input Formats"])]
    output_shapes = [parse_shape_or_none(item) for item in split_metadata_field(row["Output Shapes"])]

    if not (len(input_shapes) == len(input_dtypes) == len(input_formats) == 12):
        raise ValueError(
            "KvRmsNormRopeCache expects 12 input metadata slots, got "
            f"shapes={len(input_shapes)} dtypes={len(input_dtypes)} formats={len(input_formats)}"
        )
    if len(output_shapes) != 4:
        raise ValueError("KvRmsNormRopeCache expects four recorded outputs")

    kv = build_input_tensor(
        shape=input_shapes[0],
        input_format=input_formats[0],
        dtype_name=input_dtypes[0],
    )
    gamma = build_input_tensor(
        shape=input_shapes[1],
        input_format=input_formats[1],
        dtype_name=input_dtypes[1],
    )
    cos = build_input_tensor(
        shape=input_shapes[2],
        input_format=input_formats[2],
        dtype_name=input_dtypes[2],
    )
    sin = build_input_tensor(
        shape=input_shapes[3],
        input_format=input_formats[3],
        dtype_name=input_dtypes[3],
    )
    k_cache = build_input_tensor(
        shape=input_shapes[5],
        input_format=input_formats[5],
        dtype_name=input_dtypes[5],
    )
    ckv_cache = build_input_tensor(
        shape=input_shapes[6],
        input_format=input_formats[6],
        dtype_name=input_dtypes[6],
    )
    index = build_index_tensor(
        index_shape=input_shapes[4],
        cache_shape=input_shapes[5],
    )

    return {
        "kv": kv,
        "gamma": gamma,
        "cos": cos,
        "sin": sin,
        "index": index,
        "k_cache": k_cache,
        "ckv_cache": ckv_cache,
        "k_rope_scale": None,
        "c_kv_scale": None,
        "k_rope_offset": None,
        "c_kv_offset": None,
        "v": None,
        "epsilon": 1e-5,
        "cache_mode": infer_cache_mode(input_shapes[5], input_shapes[4]),
        "is_output_kv": True,
    }


def build_argparser():
    return build_standard_argparser(
        description=(
            "Run KvRmsNormRopeCache workload replay on Ascend NPU.\n"
            "The script reads KvRmsNormRopeCache.csv under the selected device\n"
            "and vllm_ascend version directory, reconstructs the recorded tensor\n"
            "inputs, infers cache_mode from the cache layout, then runs\n"
            "torch_npu.npu_kv_rmsnorm_rope_cache()."
        ),
        usage_examples=[
            "py -3 tools/perf_data_collection/op_replay/KvRmsNormRopeCache_run.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.13.0",
            "python tools/perf_data_collection/op_replay/KvRmsNormRopeCache_run.py "
            "--device TEST_DEVICE --vllm-ascend-version 0.9.2",
        ],
        version_help="vLLM-Ascend version, e.g. 0.13.0.",
    )


def run_row(csv_path, row_index: int, row: dict[str, str]) -> None:
    runtime_torch, runtime_torch_npu = get_runtime_modules()
    case = build_row_case(row)

    k_cache, ckv_cache, k_rope, c_kv = runtime_torch_npu.npu_kv_rmsnorm_rope_cache(
        case["kv"],
        case["gamma"],
        case["cos"],
        case["sin"],
        case["index"],
        case["k_cache"],
        case["ckv_cache"],
        k_rope_scale=case["k_rope_scale"],
        c_kv_scale=case["c_kv_scale"],
        k_rope_offset=case["k_rope_offset"],
        c_kv_offset=case["c_kv_offset"],
        v=case["v"],
        epsilon=case["epsilon"],
        cache_mode=case["cache_mode"],
        is_output_kv=case["is_output_kv"],
    )
    runtime_torch.npu.synchronize()

    print(
        f"[OK] {csv_path}:{row_index} "
        f"cache_mode={case['cache_mode']} is_output_kv={case['is_output_kv']} "
        f"k_cache={tuple(k_cache.shape)} ckv_cache={tuple(ckv_cache.shape)} "
        f"k_rope={tuple(k_rope.shape)} c_kv={tuple(c_kv.shape)}"
    )


def main() -> None:
    args = build_argparser().parse_args()
    ensure_npu_available()

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_ascend_version,
    )
    csv_paths = sorted(target_data_dir.rglob("KvRmsNormRopeCache.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No KvRmsNormRopeCache.csv found under {target_data_dir}")

    total_rows = 0
    for csv_path, row_index, row in iter_csv_rows(target_data_dir, "KvRmsNormRopeCache.csv"):
        run_row(csv_path, row_index, row)
        total_rows += 1

    print(
        f"Processed {total_rows} KvRmsNormRopeCache rows from {len(csv_paths)} csv file(s) "
        f"under {target_data_dir}."
    )


if __name__ == "__main__":
    main()
