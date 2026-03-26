"""
Run DispatchFFNCombine microbenchmark cases on Ascend NPU.

Purpose:
  Read DispatchFFNCombine rows from
  profiling_database/data/{device}/vllm_ascend/{version}/DispatchFFNCombine.csv,
  rebuild the recorded tensor inputs, then execute the exact microbench_api:

      torch.ops._C_ascend.dispatch_ffn_combine(...)

Notes:
  - The operator binding takes Tensor[] for weight1/weight2/scale1/scale2.
    Current perf database rows store the packed expert tensors as single
    entries, so this script wraps each packed tensor in a singleton list.
  - The upstream runtime path in vllm-ascend passes max_output_size=65536 for
    the fused MC2 dispatch+FFN+combine path. This script follows that value.
  - The custom op needs an HCCL communication group name. For standalone
    replay, the script initializes a single-process HCCL default group when
    no distributed process group is active yet.
"""

from __future__ import annotations

import socket
from typing import Any

from common import (
    FRACTAL_NZ_FORMAT_ID,
    build_host_tensor,
    build_standard_argparser,
    ensure_npu_available,
    get_replay_repeat_count,
    get_runtime_modules,
    get_target_data_dir,
    init_runtime,
    iter_repeated_csv_rows,
    parse_shape,
)

HCOMM_INFO: str | None = None
MAX_OUTPUT_SIZE = 65536


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


def resolve_runtime_dtype(dtype_name: str):
    runtime_torch, _ = get_runtime_modules()
    normalized = normalize_dtype_name(dtype_name)
    dtype_map = {
        "DT_FLOAT": runtime_torch.float32,
        "DT_FLOAT16": runtime_torch.float16,
        "DT_BF16": runtime_torch.bfloat16,
        "DT_DOUBLE": runtime_torch.float64,
        "DT_INT8": runtime_torch.int8,
        "DT_UINT8": runtime_torch.uint8,
        "DT_INT16": runtime_torch.int16,
        "DT_INT32": runtime_torch.int32,
        "DT_INT64": runtime_torch.int64,
        "DT_BOOL": runtime_torch.bool,
    }
    if normalized not in dtype_map:
        raise ValueError(f"Unsupported dtype for DispatchFFNCombine: {dtype_name}")
    return dtype_map[normalized]


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def get_default_hccl_group_name() -> str:
    global HCOMM_INFO
    if HCOMM_INFO is not None:
        return HCOMM_INFO

    runtime_torch, runtime_torch_npu = get_runtime_modules()
    import torch.distributed as dist
    from torch.distributed.distributed_c10d import _get_default_group

    device_index = runtime_torch.npu.current_device()
    runtime_torch_npu.npu.set_device(device_index)

    if not dist.is_initialized():
        port = find_free_port()
        dist.init_process_group(
            backend="hccl",
            rank=0,
            world_size=1,
            init_method=f"tcp://127.0.0.1:{port}",
        )

    default_pg = _get_default_group()
    if runtime_torch.__version__ > "2.0.1":
        backend = default_pg._get_backend(runtime_torch.device("npu"))
        HCOMM_INFO = backend.get_hccl_comm_name(0)
    else:
        HCOMM_INFO = default_pg.get_hccl_comm_name(0)
    return HCOMM_INFO


def maybe_cast_internal_format(tensor, tensor_format: str):
    _, runtime_torch_npu = get_runtime_modules()
    if tensor_format == "FRACTAL_NZ":
        return runtime_torch_npu.npu_format_cast(tensor, FRACTAL_NZ_FORMAT_ID)
    return tensor


def build_npu_tensor(shape: tuple[int, ...], dtype_name: str, tensor_format: str):
    dtype = resolve_runtime_dtype(dtype_name)
    tensor = build_host_tensor(shape, dtype).npu()
    return maybe_cast_internal_format(tensor, tensor_format)


def build_expert_idx_tensor(shape: tuple[int, ...], num_experts: int):
    runtime_torch, _ = get_runtime_modules()
    return runtime_torch.randint(0, num_experts, shape, dtype=runtime_torch.int32).npu()


def build_scale_tensor(flattened_shape: tuple[int, ...], expected_shape: tuple[int, int], dtype_name: str):
    runtime_torch, _ = get_runtime_modules()
    dtype = resolve_runtime_dtype(dtype_name)
    if len(flattened_shape) == 2:
        if flattened_shape != expected_shape:
            raise ValueError(
                f"scale shape mismatch: actual={flattened_shape} expected={expected_shape}"
            )
        return runtime_torch.zeros(flattened_shape, dtype=dtype).npu()
    if len(flattened_shape) != 1:
        raise ValueError(f"scale tensor must be 1D or 2D, got {flattened_shape}")
    flat_size = flattened_shape[0]
    if flat_size != expected_shape[0] * expected_shape[1]:
        raise ValueError(
            f"flattened scale size mismatch: actual={flat_size} expected={expected_shape[0] * expected_shape[1]}"
        )
    return runtime_torch.zeros(expected_shape, dtype=dtype).reshape(-1).npu()


def build_output_tensor(shape: tuple[int, ...], dtype_name: str, tensor_format: str):
    dtype = resolve_runtime_dtype(dtype_name)
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"invalid output shape: {shape}")
    tensor = build_host_tensor(shape, dtype).npu()
    return maybe_cast_internal_format(tensor, tensor_format)


def build_row_case(row: dict[str, str]) -> dict[str, Any]:
    init_runtime()
    input_shapes = [parse_shape_or_none(item) for item in split_metadata_field(row["Input Shapes"])]
    input_dtypes = [normalize_dtype_name(item) for item in split_metadata_field(row["Input Data Types"])]
    input_formats = [item if item else "NULL" for item in split_metadata_field(row["Input Formats"])]
    output_shapes = [parse_shape_or_none(item) for item in split_metadata_field(row["Output Shapes"])]
    output_dtypes = [normalize_dtype_name(item) for item in split_metadata_field(row["Output Data Types"])]
    output_formats = [item if item else "NULL" for item in split_metadata_field(row["Output Formats"])]

    if not (len(input_shapes) == len(input_dtypes) == len(input_formats) == 7):
        raise ValueError(
            "DispatchFFNCombine expects seven input metadata slots, got "
            f"shapes={len(input_shapes)} dtypes={len(input_dtypes)} formats={len(input_formats)}"
        )
    if not (len(output_shapes) == len(output_dtypes) == len(output_formats) == 2):
        raise ValueError(
            "DispatchFFNCombine expects two output metadata slots, got "
            f"shapes={len(output_shapes)} dtypes={len(output_dtypes)} formats={len(output_formats)}"
        )
    if any(item is None for item in input_shapes + output_shapes):
        raise ValueError("DispatchFFNCombine metadata contains unexpected empty shape slots")

    x_shape, weight1_shape, weight2_shape, expert_idx_shape, scale1_shape, scale2_shape, probs_shape = input_shapes
    out_shape, expert_token_nums_shape = output_shapes

    if len(x_shape) != 2:
        raise ValueError(f"x must be 2D, got {x_shape}")
    if len(weight1_shape) != 3 or len(weight2_shape) != 3:
        raise ValueError(f"weight tensors must be 3D, got w1={weight1_shape} w2={weight2_shape}")
    if len(expert_idx_shape) != 2 or len(probs_shape) != 2:
        raise ValueError(f"expert_idx/probs must be 2D, got idx={expert_idx_shape} probs={probs_shape}")

    num_experts, hidden_size, inter_size = weight1_shape
    if weight2_shape[0] != num_experts:
        raise ValueError(f"expert count mismatch between weight1 and weight2: {weight1_shape} vs {weight2_shape}")
    if weight2_shape[2] != hidden_size:
        raise ValueError(f"hidden size mismatch between x/weight2: x={x_shape} w2={weight2_shape}")
    if x_shape[1] != hidden_size:
        raise ValueError(f"x hidden size mismatch: x={x_shape} weight1={weight1_shape}")
    if expert_idx_shape != probs_shape:
        raise ValueError(f"expert_idx/probs shape mismatch: idx={expert_idx_shape} probs={probs_shape}")
    if expert_idx_shape[0] != x_shape[0]:
        raise ValueError(f"token count mismatch between x and expert_idx: x={x_shape} idx={expert_idx_shape}")
    if out_shape != x_shape:
        raise ValueError(f"output shape must match x shape, got out={out_shape} x={x_shape}")
    if expert_token_nums_shape not in {(num_experts,), (1, num_experts)}:
        raise ValueError(
            f"expert_token_nums shape must be ({num_experts},) or (1, {num_experts}), got {expert_token_nums_shape}"
        )

    scale1_expected_shape = (num_experts, inter_size)
    scale2_expected_shape = (num_experts, hidden_size)

    x = build_npu_tensor(x_shape, input_dtypes[0], input_formats[0])
    weight1 = build_npu_tensor(weight1_shape, input_dtypes[1], input_formats[1])
    weight2 = build_npu_tensor(weight2_shape, input_dtypes[2], input_formats[2])
    expert_idx = build_expert_idx_tensor(expert_idx_shape, num_experts)
    scale1 = build_scale_tensor(scale1_shape, scale1_expected_shape, input_dtypes[4])
    scale2 = build_scale_tensor(scale2_shape, scale2_expected_shape, input_dtypes[5])
    probs = build_npu_tensor(probs_shape, input_dtypes[6], input_formats[6])
    out = build_output_tensor(out_shape, output_dtypes[0], output_formats[0])
    expert_token_nums = build_output_tensor(
        expert_token_nums_shape,
        output_dtypes[1],
        output_formats[1],
    )

    return {
        "x": x,
        "weight1_list": [weight1],
        "weight2_list": [weight2],
        "expert_idx": expert_idx,
        "scale1_list": [scale1],
        "scale2_list": [scale2],
        "probs": probs,
        "group": get_default_hccl_group_name(),
        "max_output_size": MAX_OUTPUT_SIZE,
        "out": out,
        "expert_token_nums": expert_token_nums,
        "expected_output_shapes": output_shapes,
        "weight_kind": input_dtypes[1],
        "num_experts": num_experts,
        "topk": expert_idx_shape[1],
    }


def build_argparser():
    return build_standard_argparser(
        description=(
            "Run DispatchFFNCombine microbenchmark rows on Ascend NPU.\n"
            "The script reads DispatchFFNCombine.csv profiling rows,\n"
            "reconstructs the packed MoE expert tensors, then executes\n"
            "torch.ops._C_ascend.dispatch_ffn_combine()."
        ),
        usage_examples=[
            "py -3 tools/perf_data_collection/op_replay/DispatchFFNCombine_run.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.13.0",
            "py -3 tools/perf_data_collection/op_replay/DispatchFFNCombine_run.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.15.0",
        ],
        version_help="vLLM-Ascend version, e.g. 0.13.0.",
    )


def run_row(csv_path, row_index: int, row: dict[str, str]) -> None:
    runtime_torch, _ = get_runtime_modules()
    case = build_row_case(row)

    api_name = "torch.ops._C_ascend.dispatch_ffn_combine"
    try:
        out, expert_token_nums = runtime_torch.ops._C_ascend.dispatch_ffn_combine(
            x=case["x"],
            weight1=case["weight1_list"],
            weight2=case["weight2_list"],
            expert_idx=case["expert_idx"],
            scale1=case["scale1_list"],
            scale2=case["scale2_list"],
            probs=case["probs"],
            group=case["group"],
            max_output_size=case["max_output_size"],
            out=case["out"],
            expert_token_nums=case["expert_token_nums"],
        )
    except RuntimeError as exc:
        if "does not support opType [DispatchFFNCombine]" not in str(exc):
            raise

        out, expert_token_nums = runtime_torch.ops._C_ascend.dispatch_gmm_combine_decode(
            x=case["x"],
            expert_ids=case["expert_idx"],
            gmm1_permuted_weight=case["weight1_list"],
            gmm1_permuted_weight_scale=[tensor.to(runtime_torch.float32) for tensor in case["scale1_list"]],
            gmm2_weight=case["weight2_list"],
            gmm2_weight_scale=[tensor.to(runtime_torch.float32) for tensor in case["scale2_list"]],
            expert_scales=case["probs"],
            expert_smooth_scales=None,
            x_active_mask=None,
            group_ep=case["group"],
            ep_rank_size=1,
            ep_rank_id=0,
            moe_expert_num=case["num_experts"],
            shared_expert_num=1,
            shared_expert_rank_num=0,
            quant_mode=0,
            global_bs=case["x"].shape[0],
        )
        api_name = "torch.ops._C_ascend.dispatch_gmm_combine_decode(fallback)"
        if case["expert_token_nums"].ndim == 2 and expert_token_nums.ndim == 1:
            expert_token_nums = expert_token_nums.unsqueeze(0)
        expert_token_nums = expert_token_nums.to(case["expert_token_nums"].dtype)
    runtime_torch.npu.synchronize()

    actual_shapes = [tuple(out.shape), tuple(expert_token_nums.shape)]
    expected_shapes = case["expected_output_shapes"]
    if actual_shapes[0] != expected_shapes[0]:
        raise ValueError(f"out shape mismatch: actual={actual_shapes[0]} expected={expected_shapes[0]}")
    if actual_shapes[1] != expected_shapes[1]:
        raise ValueError(
            "expert_token_nums shape mismatch: "
            f"actual={actual_shapes[1]} expected={expected_shapes[1]}"
        )

    print(
        f"[OK] {csv_path}:{row_index} "
        f"api={api_name} "
        f"x={tuple(case['x'].shape)} "
        f"w1={tuple(case['weight1_list'][0].shape)} "
        f"w2={tuple(case['weight2_list'][0].shape)} "
        f"topk={case['topk']} experts={case['num_experts']} "
        f"weight_kind={case['weight_kind']} "
        f"out={tuple(out.shape)} expert_token_nums={tuple(expert_token_nums.shape)}"
    )


def main() -> None:
    args = build_argparser().parse_args()
    repeat_count = get_replay_repeat_count(args.repeat_count)
    ensure_npu_available()

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_ascend_version,
    )
    csv_paths = sorted(target_data_dir.rglob("DispatchFFNCombine.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No DispatchFFNCombine.csv found under {target_data_dir}")

    total_rows = 0
    for csv_path, row_index, row in iter_repeated_csv_rows(
        target_data_dir,
        "DispatchFFNCombine.csv",
        repeat_count,
    ):
        run_row(csv_path, row_index, row)
        total_rows += 1

    print(
        f"Processed {total_rows} DispatchFFNCombine rows from {len(csv_paths)} csv file(s) "
        f"under {target_data_dir}."
    )


if __name__ == "__main__":
    main()
