"""
Replay FusedInferAttentionScore cases from the performance database on Ascend NPU.

Purpose:
  Read FusedInferAttentionScore rows from
  perf_database/data/{device}/vllm_ascend/{version}/FusedInferAttentionScore.csv,
  rebuild the recorded tensor inputs, infer the minimal scalar/list arguments
  required by torch_npu.npu_fused_infer_attention_score(), then execute the op
  on NPU.

Notes:
  - The CSV keeps 31 input slots and many of them are intentionally empty. This
    script preserves those holes so the populated tensors map back to the same
    API arguments as the profiled row.
  - The current ATLAS_800_A3_752T_128G_DIE/v0.13.0 database contains three
    practical shapes:
      1) paged TND attention with block_table and mask;
      2) paged MLA-style attention with query_rope/key_rope;
      3) non-paged TND attention without block_table.
"""

from __future__ import annotations

import math

from common import (
    build_input_tensor,
    build_standard_argparser,
    ensure_npu_available,
    get_runtime_modules,
    get_target_data_dir,
    iter_csv_rows,
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
    "DT_UNDEFINED": "DT_UNDEFINED",
}

QUERY_INDEX = 0
KEY_INDEX = 1
VALUE_INDEX = 2
ATTEN_MASK_INDEX = 4
ACTUAL_SEQ_LENGTHS_INDEX = 5
ACTUAL_SEQ_LENGTHS_KV_INDEX = 6
BLOCK_TABLE_INDEX = 14
QUERY_ROPE_INDEX = 24
KEY_ROPE_INDEX = 25


def normalize_dtype_name(dtype_name: str) -> str:
    normalized = dtype_name.strip()
    if normalized.startswith("DT_"):
        return normalized
    return DTYPE_ALIASES.get(normalized, normalized)


def split_metadata_field(raw_value: str) -> list[str]:
    cleaned = raw_value.strip().strip('"')
    return [item.strip() for item in cleaned.split(";")]


def parse_shape_or_none(raw_shape: str):
    if not raw_shape.strip():
        return None
    return parse_shape(raw_shape)


def shape_numel(shape: tuple[int, ...] | None) -> int:
    if not shape:
        return 0
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def distribute_total(total: int, bucket_count: int, *, min_value: int) -> list[int]:
    if bucket_count <= 0:
        return []
    if total < bucket_count * min_value:
        raise ValueError(
            f"Cannot distribute total={total} into {bucket_count} buckets with "
            f"min_value={min_value}"
        )

    values = [min_value] * bucket_count
    remaining = total - bucket_count * min_value
    base, extra = divmod(remaining, bucket_count)
    for index in range(bucket_count):
        values[index] += base
        if index < extra:
            values[index] += 1
    return values


def cumulative_lengths(lengths: list[int]) -> list[int]:
    total = 0
    cumulative = []
    for length in lengths:
        total += length
        cumulative.append(total)
    return cumulative


def build_zero_tensor(shape: tuple[int, ...], dtype_name: str):
    runtime_torch, _ = get_runtime_modules()
    dtype = {
        "DT_BOOL": runtime_torch.bool,
        "DT_INT8": runtime_torch.int8,
        "DT_UINT8": runtime_torch.uint8,
        "DT_INT16": runtime_torch.int16,
        "DT_INT32": runtime_torch.int32,
        "DT_INT64": runtime_torch.int64,
        "DT_FLOAT16": runtime_torch.float16,
        "DT_BF16": runtime_torch.bfloat16,
        "DT_FLOAT": runtime_torch.float32,
        "DT_DOUBLE": runtime_torch.float64,
    }.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Unsupported zero tensor dtype: {dtype_name}")
    return runtime_torch.zeros(shape, dtype=dtype).npu()


def build_causal_mask(shape: tuple[int, ...], dtype_name: str):
    runtime_torch, _ = get_runtime_modules()
    rows = shape[-2]
    cols = shape[-1]
    base_mask = runtime_torch.triu(
        runtime_torch.ones((rows, cols), dtype=runtime_torch.bool),
        diagonal=1,
    )

    if len(shape) == 2:
        mask = base_mask
    elif len(shape) == 3:
        mask = base_mask.unsqueeze(0).expand(shape[0], rows, cols)
    elif len(shape) == 4:
        mask = base_mask.unsqueeze(0).unsqueeze(0).expand(shape[0], shape[1], rows, cols)
    else:
        raise ValueError(f"Unsupported atten_mask shape: {shape}")

    if dtype_name == "DT_BOOL":
        return mask.npu()
    return mask.to({
        "DT_INT8": runtime_torch.int8,
        "DT_UINT8": runtime_torch.uint8,
    }[dtype_name]).npu()


def build_scalar_length_list(length_shape: tuple[int, ...] | None) -> int | None:
    if not length_shape:
        return None
    return shape_numel(length_shape)


def infer_case_args(shapes: list[tuple[int, ...] | None]):
    query_shape = shapes[QUERY_INDEX]
    key_shape = shapes[KEY_INDEX]
    block_table_shape = shapes[BLOCK_TABLE_INDEX]
    query_rope_shape = shapes[QUERY_ROPE_INDEX]

    if query_shape is None or key_shape is None:
        raise ValueError("FusedInferAttentionScore requires query/key/value shapes")

    if len(query_shape) == 3:
        input_layout = "TND"
        num_heads = query_shape[1]
        if block_table_shape is None:
            num_key_value_heads = key_shape[1]
            scale_dim = query_shape[-1]
        else:
            num_key_value_heads = 1
            scale_dim = query_shape[-1]
        return {
            "input_layout": input_layout,
            "num_heads": num_heads,
            "num_key_value_heads": num_key_value_heads,
            "scale": 1.0 / math.sqrt(scale_dim),
        }

    if len(query_shape) == 4 and query_rope_shape is not None:
        input_layout = "BNSD_NBSD"
        num_heads = query_shape[1]
        num_key_value_heads = 1
        scale_dim = query_shape[-1] + query_rope_shape[-1]
        return {
            "input_layout": input_layout,
            "num_heads": num_heads,
            "num_key_value_heads": num_key_value_heads,
            "scale": 1.0 / math.sqrt(scale_dim),
        }

    raise ValueError(
        "Unsupported FusedInferAttentionScore shape pattern: "
        f"query={query_shape}, key={key_shape}, block_table={block_table_shape}, "
        f"query_rope={query_rope_shape}"
    )


def infer_sparse_mode(atten_mask_shape: tuple[int, ...] | None) -> int:
    if atten_mask_shape is None:
        return 0
    if len(atten_mask_shape) >= 2 and atten_mask_shape[-2:] == (2048, 2048):
        # The profiled database uses the optimized 2048 causal mask path.
        return 3
    return 0


def infer_block_size(
    key_shape: tuple[int, ...],
    block_table_shape: tuple[int, ...] | None,
) -> int:
    if block_table_shape is None:
        return 0
    if len(key_shape) == 3:
        return key_shape[1]
    if len(key_shape) == 4:
        return key_shape[2]
    raise ValueError(f"Unsupported key rank for block_size inference: {key_shape}")


def infer_query_lens(
    query_shape: tuple[int, ...],
    batch_size: int | None,
):
    if batch_size is None:
        return None
    if len(query_shape) == 3:
        return cumulative_lengths(
            distribute_total(query_shape[0], batch_size, min_value=1)
        )
    if len(query_shape) == 4:
        if query_shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch between query={query_shape} and actual_seq_lengths={batch_size}"
            )
        return [query_shape[2]] * batch_size
    raise ValueError(f"Unsupported query rank for actual_seq_lengths: {query_shape}")


def infer_kv_block_shape(key_shape: tuple[int, ...], block_table_shape):
    if block_table_shape is None:
        return None, None
    if len(key_shape) == 3:
        return key_shape[0], key_shape[1]
    if len(key_shape) == 4:
        return key_shape[0], key_shape[2]
    raise ValueError(f"Unsupported key rank for paged attention: {key_shape}")


def infer_seq_lens_kv(
    key_shape: tuple[int, ...],
    batch_size: int | None,
    block_table_shape: tuple[int, ...] | None,
):
    if batch_size is None:
        return None

    total_blocks, block_size = infer_kv_block_shape(key_shape, block_table_shape)
    if total_blocks is not None and block_size is not None:
        # For paged attention the first key/value dimension is the size of the
        # whole KV cache pool, not the effective context blocks consumed by the
        # current batch. The CSV does not record per-request KV lengths, only the
        # shape of `actual_seq_lengths_kv` and `block_table`. Use a minimal valid
        # one-block context for each request so the op receives legal metadata
        # strictly matching the recorded tensor shapes.
        _, max_blocks_per_seq = block_table_shape
        if max_blocks_per_seq < 1:
            raise ValueError(f"Invalid block_table shape: {block_table_shape}")
        return [block_size] * batch_size

    if len(key_shape) == 3:
        return cumulative_lengths(
            distribute_total(key_shape[0], batch_size, min_value=1)
        )
    if len(key_shape) == 4:
        if key_shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch between key={key_shape} and actual_seq_lengths_kv={batch_size}"
            )
        return [key_shape[2]] * batch_size
    raise ValueError(f"Unsupported key rank for actual_seq_lengths_kv: {key_shape}")


def build_block_table_tensor(
    block_table_shape: tuple[int, ...] | None,
    seq_lens_kv: list[int] | None,
    key_shape: tuple[int, ...],
):
    if block_table_shape is None:
        return None
    runtime_torch, _ = get_runtime_modules()
    total_blocks, block_size = infer_kv_block_shape(key_shape, block_table_shape)
    if total_blocks is None or block_size is None:
        raise ValueError("block_table exists but key shape is not paged")

    batch_size, max_blocks_per_seq = block_table_shape
    if seq_lens_kv is None or len(seq_lens_kv) != batch_size:
        raise ValueError("block_table requires actual_seq_lengths_kv with matching batch size")

    block_table = runtime_torch.zeros(block_table_shape, dtype=runtime_torch.int32)
    cursor = 0
    for row_index, seq_len in enumerate(seq_lens_kv):
        needed_blocks = max(1, math.ceil(seq_len / block_size))
        if needed_blocks > max_blocks_per_seq:
            raise ValueError(
                f"Sequence length {seq_len} needs {needed_blocks} blocks, "
                f"but block_table width is {max_blocks_per_seq}"
            )
        values = [
            (cursor + offset) % total_blocks
            for offset in range(needed_blocks)
        ]
        block_table[row_index, :needed_blocks] = runtime_torch.tensor(
            values,
            dtype=runtime_torch.int32,
        )
        cursor += needed_blocks
    return block_table.npu()


def build_row_case(row: dict[str, str]):
    input_shapes = [parse_shape_or_none(item) for item in split_metadata_field(row["Input Shapes"])]
    input_dtypes = [normalize_dtype_name(item) for item in split_metadata_field(row["Input Data Types"])]
    input_formats = [item if item else "NULL" for item in split_metadata_field(row["Input Formats"])]
    output_shapes = [parse_shape_or_none(item) for item in split_metadata_field(row["Output Shapes"])]
    output_dtypes = [normalize_dtype_name(item) for item in split_metadata_field(row["Output Data Types"]) if item.strip()]

    if not (
        len(input_shapes) == len(input_dtypes) == len(input_formats) == 31
    ):
        raise ValueError(
            "Expected 31 input metadata slots for FusedInferAttentionScore, got "
            f"shapes={len(input_shapes)} dtypes={len(input_dtypes)} formats={len(input_formats)}"
        )

    query = build_input_tensor(
        shape=input_shapes[QUERY_INDEX],
        input_format=input_formats[QUERY_INDEX],
        dtype_name=input_dtypes[QUERY_INDEX],
    )
    key = build_input_tensor(
        shape=input_shapes[KEY_INDEX],
        input_format=input_formats[KEY_INDEX],
        dtype_name=input_dtypes[KEY_INDEX],
    )
    value = build_input_tensor(
        shape=input_shapes[VALUE_INDEX],
        input_format=input_formats[VALUE_INDEX],
        dtype_name=input_dtypes[VALUE_INDEX],
    )

    atten_mask = None
    if input_shapes[ATTEN_MASK_INDEX] is not None:
        if input_shapes[ATTEN_MASK_INDEX][-2:] == (2048, 2048):
            atten_mask = build_causal_mask(
                input_shapes[ATTEN_MASK_INDEX],
                input_dtypes[ATTEN_MASK_INDEX],
            )
        else:
            atten_mask = build_zero_tensor(
                input_shapes[ATTEN_MASK_INDEX],
                input_dtypes[ATTEN_MASK_INDEX],
            )

    query_lens_batch = build_scalar_length_list(input_shapes[ACTUAL_SEQ_LENGTHS_INDEX])
    seq_lens_batch = build_scalar_length_list(input_shapes[ACTUAL_SEQ_LENGTHS_KV_INDEX])
    query_lens = infer_query_lens(input_shapes[QUERY_INDEX], query_lens_batch)
    seq_lens_kv = infer_seq_lens_kv(
        input_shapes[KEY_INDEX],
        seq_lens_batch,
        input_shapes[BLOCK_TABLE_INDEX],
    )

    block_table = build_block_table_tensor(
        input_shapes[BLOCK_TABLE_INDEX],
        seq_lens_kv,
        input_shapes[KEY_INDEX],
    )

    query_rope = None
    if input_shapes[QUERY_ROPE_INDEX] is not None:
        query_rope = build_input_tensor(
            shape=input_shapes[QUERY_ROPE_INDEX],
            input_format=input_formats[QUERY_ROPE_INDEX],
            dtype_name=input_dtypes[QUERY_ROPE_INDEX],
        )

    key_rope = None
    if input_shapes[KEY_ROPE_INDEX] is not None:
        key_rope = build_input_tensor(
            shape=input_shapes[KEY_ROPE_INDEX],
            input_format=input_formats[KEY_ROPE_INDEX],
            dtype_name=input_dtypes[KEY_ROPE_INDEX],
        )

    inferred = infer_case_args(input_shapes)
    return {
        "query": query,
        "key": key,
        "value": value,
        "atten_mask": atten_mask,
        "actual_seq_lengths": query_lens,
        "actual_seq_lengths_kv": seq_lens_kv,
        "block_table": block_table,
        "query_rope": query_rope,
        "key_rope": key_rope,
        "input_layout": inferred["input_layout"],
        "num_heads": inferred["num_heads"],
        "num_key_value_heads": inferred["num_key_value_heads"],
        "scale": inferred["scale"],
        "sparse_mode": infer_sparse_mode(input_shapes[ATTEN_MASK_INDEX]),
        "block_size": infer_block_size(
            input_shapes[KEY_INDEX],
            input_shapes[BLOCK_TABLE_INDEX],
        ),
        "softmax_lse_flag": len(output_shapes) >= 2 and output_shapes[1] is not None,
        "expected_output_shapes": output_shapes,
        "expected_output_dtypes": output_dtypes,
    }


def build_argparser():
    return build_standard_argparser(
        description=(
            "Run FusedInferAttentionScore workload replay on Ascend NPU.\n"
            "The script reads FusedInferAttentionScore.csv under the selected\n"
            "device and vllm_ascend version directory, reconstructs the tensor\n"
            "inputs recorded in the database, infers the matching list/scalar\n"
            "arguments for the profiled shape pattern, then runs\n"
            "torch_npu.npu_fused_infer_attention_score()."
        ),
        usage_examples=[
            "py -3 tools/perf_data_collection/op_replay/FusedInferAttentionScore_run.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.13.0",
            "python tools/perf_data_collection/op_replay/FusedInferAttentionScore_run.py "
            "--device TEST_DEVICE --vllm-ascend-version 0.9.2",
        ],
        version_help="vLLM-Ascend version, e.g. 0.13.0.",
    )


def run_row(csv_path, row_index: int, row: dict[str, str]) -> None:
    runtime_torch, runtime_torch_npu = get_runtime_modules()
    case = build_row_case(row)

    output, softmax_lse = runtime_torch_npu.npu_fused_infer_attention_score(
        case["query"],
        case["key"],
        case["value"],
        atten_mask=case["atten_mask"],
        actual_seq_lengths=case["actual_seq_lengths"],
        actual_seq_lengths_kv=case["actual_seq_lengths_kv"],
        block_table=case["block_table"],
        query_rope=case["query_rope"],
        key_rope=case["key_rope"],
        num_heads=case["num_heads"],
        scale=case["scale"],
        pre_tokens=65535,
        next_tokens=65535,
        input_layout=case["input_layout"],
        num_key_value_heads=case["num_key_value_heads"],
        sparse_mode=case["sparse_mode"],
        block_size=case["block_size"],
        softmax_lse_flag=case["softmax_lse_flag"],
    )
    runtime_torch.npu.synchronize()

    softmax_shape = None if softmax_lse is None else tuple(softmax_lse.shape)
    print(
        f"[OK] {csv_path}:{row_index} "
        f"layout={case['input_layout']} heads={case['num_heads']} "
        f"kv_heads={case['num_key_value_heads']} sparse_mode={case['sparse_mode']} "
        f"block_size={case['block_size']} output={tuple(output.shape)} "
        f"softmax_lse={softmax_shape} "
        f"actual_seq_lengths={case['actual_seq_lengths']} "
        f"actual_seq_lengths_kv={case['actual_seq_lengths_kv']}"
    )
    print(
        f"[ARGS] {csv_path}:{row_index} "
        f"softmax_lse_flag={case['softmax_lse_flag']} "
        f"expected_output_shapes={case['expected_output_shapes']} "
        f"expected_output_dtypes={case['expected_output_dtypes']} "
        f"block_table_shape={None if case['block_table'] is None else tuple(case['block_table'].shape)} "
        f"atten_mask_shape={None if case['atten_mask'] is None else tuple(case['atten_mask'].shape)}"
    )


def main() -> None:
    args = build_argparser().parse_args()
    ensure_npu_available()

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_ascend_version,
    )
    total_rows = 0
    for csv_path, row_index, row in iter_csv_rows(
        target_data_dir,
        "FusedInferAttentionScore.csv",
    ):
        run_row(csv_path, row_index, row)
        total_rows += 1

    if total_rows == 0:
        raise FileNotFoundError(
            f"No FusedInferAttentionScore.csv rows found under {target_data_dir}"
        )

    print(
        f"Processed {total_rows} FusedInferAttentionScore rows under {target_data_dir}."
    )


if __name__ == "__main__":
    main()
