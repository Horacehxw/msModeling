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
  - EP (Expert Parallel) support: Modify EP_SIZE below to control EP size.
    EP_SIZE equals to world_size / rank count.
    When EP_SIZE > 1, the script will automatically launch EP_SIZE processes
    via `torchrun` to simulate EP distributed environment.
  - Profiler mode: Uses torch_npu.profiler to capture Duration data and
    outputs CSV with performance metrics (aligns with ProfilingDataSource).
"""

from __future__ import annotations

import csv
import os
import socket
import subprocess
import time
import sys
from pathlib import Path
from typing import Any

from common import (
    FRACTAL_NZ_FORMAT_ID,
    build_host_tensor,
    build_standard_argparser,
    ensure_npu_available,
    get_runtime_modules,
    get_target_data_dir,
    init_runtime,
    iter_csv_rows,
    parse_shape,
)

# ============================================================================
# Benchmark configuration
# ============================================================================
# 连续调用次数
BENCHMARK_TOTAL_ITERS = 100
# 预热次数（不计入统计）
WARMUP_ITERS = 10

# 默认 EP 规模（可通过 --ep-size 参数覆盖）
DEFAULT_EP_SIZE = 16

# EP_RANK: 当前 rank ID（多进程模式下由 torchrun 自动设置）
EP_RANK: int = 0
EP_GROUP = None
HCOMM_INFO: str | None = None
MAX_OUTPUT_SIZE = 65536

# 全局 EP_SIZE，在 main() 中根据参数设置
EP_SIZE: int = DEFAULT_EP_SIZE

# 是否启用均衡 expert 分布（可通过 --balanced 参数覆盖）
ENABLE_BALANCED: bool = True


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
    """找一个空闲端口。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def launch_torchrun_and_wait(ep_size: int, args: list[str]) -> int:
    """
    使用 torchrun 启动多进程 EP 环境，等待所有进程完成。

    Args:
        ep_size: 需要启动的 rank 数量
        args: 传递给脚本的命令行参数

    Returns:
        子进程的退出码
    """
    port = find_free_port()

    # 构建 torchrun 命令
    torchrun_cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={ep_size}",
        f"--master_port={port}",
        __file__,  # 当前脚本路径
        *args,
    ]

    print(f"[Auto EP] Launching torchrun with {ep_size} ranks on port {port}...")

    # 设置环境变量，标记这是由脚本自动启动的
    env = os.environ.copy()
    env["_DFC_AUTO_TORCHRUN"] = "1"

    result = subprocess.run(torchrun_cmd, env=env, check=True)
    return result.returncode


def init_ep_process_group(ep_size: int, ep_rank: int, master_addr: str, master_port: int):
    """初始化 EP（专家并行）分布式进程组。"""
    global EP_SIZE, EP_RANK, EP_GROUP, HCOMM_INFO

    runtime_torch, runtime_torch_npu = get_runtime_modules()
    import torch.distributed as dist
    from torch.distributed.distributed_c10d import _get_default_group

    EP_SIZE = ep_size
    EP_RANK = ep_rank

    # 设置当前设备 (每个 rank 对应一个 DIE)
    device_index = ep_rank % runtime_torch.npu.device_count()
    runtime_torch_npu.npu.set_device(device_index)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="hccl",
            rank=ep_rank,
            world_size=ep_size,
            init_method=f"tcp://{master_addr}:{master_port}",
        )

    default_pg = _get_default_group()
    EP_GROUP = default_pg

    if runtime_torch.__version__ > "2.0.1":
        backend = default_pg._get_backend(runtime_torch.device("npu"))
        HCOMM_INFO = backend.get_hccl_comm_name(ep_rank)
    else:
        HCOMM_INFO = default_pg.get_hccl_comm_name(ep_rank)

    return HCOMM_INFO


def get_default_hccl_group_name() -> str:
    """获取 HCCL 通信组名称，支持单卡和多卡 EP 模式。"""
    global HCOMM_INFO

    if HCOMM_INFO is not None:
        return HCOMM_INFO

    # EP 模式：多卡分布式
    if EP_SIZE > 1:
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = int(os.environ.get("MASTER_PORT", "29500"))
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        return init_ep_process_group(world_size, rank, master_addr, master_port)

    # 单卡模式
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


def build_balanced_expert_idx_tensor(shape: tuple[int, ...], num_experts: int):
    """Round-robin 均衡 expert 分配，保证每个 expert 的 token 数相同 ±1。"""
    runtime_torch, _ = get_runtime_modules()
    num_tokens, topk = shape
    total_slots = num_tokens * topk
    flat_ids = runtime_torch.arange(total_slots, dtype=runtime_torch.int32) % num_experts
    return flat_ids.reshape(num_tokens, topk).npu()


def build_uniform_probs_tensor(shape: tuple[int, ...], topk: int):
    """构造均匀权重的 probs 张量，避免随机权重引入 combine 阶段的数值差异。"""
    runtime_torch, _ = get_runtime_modules()
    return runtime_torch.full(shape, 1.0 / topk, dtype=runtime_torch.float32).npu()


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


def build_row_case(row: dict[str, str], balanced: bool = True) -> dict[str, Any]:
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

    topk = expert_idx_shape[1]
    # 从 CSV 的 EP Size 列获取 ep_world_size，总专家数 = num_experts_per_rank * ep_size
    ep_size_str = row.get("EP Size", "") or ""
    if ep_size_str.strip():
        try:
            ep_size_from_csv = int(ep_size_str.strip())
            expert_idx_num_experts = num_experts * ep_size_from_csv
        except ValueError:
            expert_idx_num_experts = num_experts * EP_SIZE
    else:
        expert_idx_num_experts = num_experts * EP_SIZE
    if balanced:
        expert_idx = build_balanced_expert_idx_tensor(expert_idx_shape, expert_idx_num_experts)
        probs = build_uniform_probs_tensor(probs_shape, topk)
    else:
        expert_idx = build_expert_idx_tensor(expert_idx_shape, expert_idx_num_experts)
        probs = build_npu_tensor(probs_shape, input_dtypes[6], input_formats[6])

    scale1 = build_scale_tensor(scale1_shape, scale1_expected_shape, input_dtypes[4])
    scale2 = build_scale_tensor(scale2_shape, scale2_expected_shape, input_dtypes[5])
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
    parser = build_standard_argparser(
        description=(
            "Run DispatchFFNCombine microbenchmark rows on Ascend NPU.\n"
            "EP Mode: Use --ep-size to control EP size (default: 16).\n"
            "         EP_SIZE=1: single-process, no EP.\n"
            "         EP_SIZE>1: auto-launch EP_SIZE processes via torchrun.\n"
            "Benchmark: 连续调用100次，取 Duration(us) 最小的一次数据。"
        ),
        usage_examples=[
            "# Single-process mode (EP=1):",
            "python tools/perf_data_collection/op_replay/DispatchFFNCombine_run.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.20.0 "
            "--ep-size 1 --output-csv ./results.csv",
            "# EP=8 mode:",
            "python tools/perf_data_collection/op_replay/DispatchFFNCombine_run.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.20.0 "
            "--ep-size 8 --output-csv ./results.csv",
        ],
        version_help="vLLM-Ascend version, e.g. 0.20.0.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path to output CSV file with benchmark results.",
    )
    parser.add_argument(
        "--ep-size",
        type=int,
        default=DEFAULT_EP_SIZE,
        help=(
            f"EP (Expert Parallel) size, equals to world_size/rank count. "
            f"EP_SIZE=1: single-process, no EP. "
            f"EP_SIZE>1: auto-launch EP_SIZE processes via torchrun. "
            f"Default: {DEFAULT_EP_SIZE}."
        ),
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=True,
        help="Use balanced expert distribution (round-robin). Default: True.",
    )
    parser.add_argument(
        "--no-balanced",
        action="store_false",
        dest="balanced",
        help="Use random expert distribution instead of balanced.",
    )
    return parser


# ============================================================================
# Benchmark utilities
# ============================================================================


def execute_dfc_op(case: dict[str, Any], use_fallback: bool = False) -> tuple:
    """执行一次 DFC 算子调用，返回 (out, expert_token_nums)。"""
    runtime_torch, _ = get_runtime_modules()

    if not use_fallback:
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
            return out, expert_token_nums, False
        except RuntimeError as exc:
            if "does not support opType [DispatchFFNCombine]" not in str(exc):
                raise
            use_fallback = True

    # Fallback mode
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
        ep_rank_size=EP_SIZE,
        ep_rank_id=EP_RANK,
        moe_expert_num=case["num_experts"],
        shared_expert_num=1,
        shared_expert_rank_num=0,
        quant_mode=0,
        global_bs=case["x"].shape[0],
    )
    if case["expert_token_nums"].ndim == 2 and expert_token_nums.ndim == 1:
        expert_token_nums = expert_token_nums.unsqueeze(0)
    expert_token_nums = expert_token_nums.to(case["expert_token_nums"].dtype)
    return out, expert_token_nums, True


def run_benchmark_with_profiler(
        case: dict[str, Any],
        prof_dir: str,
) -> dict[str, float]:
    """
    使用 profiler 执行 benchmark，返回各项性能指标（微秒）。
    同时保留 profiler 数据到 prof_dir。

    策略：
      1. 预热 WARMUP_ITERS 次
      2. 使用 profiler 采集 BENCHMARK_TOTAL_ITERS 次调用
      3. 从 kernel_details.csv 中读取每次调用的 Duration，找到最小的一次
      4. 返回 Duration 最小那次的所有硬件指标

    Returns:
        dict 包含各项硬件指标（Duration 最小那次的数据）
    """
    runtime_torch, runtime_torch_npu = get_runtime_modules()

    # 检测是否需要使用 fallback 模式
    use_fallback = False
    try:
        _, _, _ = execute_dfc_op(case, use_fallback=False)
    except RuntimeError as exc:
        if "does not support opType [DispatchFFNCombine]" in str(exc):
            use_fallback = True
        else:
            raise

    # Warmup
    for _ in range(WARMUP_ITERS):
        execute_dfc_op(case, use_fallback)
        runtime_torch.npu.synchronize()

    # 使用 profiler 采集所有调用
    experimental_config = runtime_torch_npu.profiler._ExperimentalConfig(
        profiler_level=runtime_torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=runtime_torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=True,
        op_attr=True,
        data_simplification=True,
    )

    with runtime_torch_npu.profiler.profile(
            activities=[
                runtime_torch_npu.profiler.ProfilerActivity.CPU,
                runtime_torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=runtime_torch_npu.profiler.schedule(
                wait=0,
                warmup=0,
                active=BENCHMARK_TOTAL_ITERS,
                repeat=1,
            ),
            on_trace_ready=runtime_torch_npu.profiler.tensorboard_trace_handler(prof_dir),
            experimental_config=experimental_config,
            record_shapes=True,
            with_stack=True,
    ) as prof:
        for _ in range(BENCHMARK_TOTAL_ITERS):
            execute_dfc_op(case, use_fallback)
            runtime_torch.npu.synchronize()
            prof.step()

    # 等待 profiler 数据写入完成
    runtime_torch.npu.synchronize()
    time.sleep(1)

    # 从 kernel_details.csv 解析硬件指标
    result = extract_metrics_from_kernel_details(prof_dir)

    result["use_fallback"] = use_fallback
    return result


def extract_metrics_from_kernel_details(prof_dir: str) -> dict[str, float]:
    """
    从 profiler 输出的 kernel_details.csv 中提取硬件指标。
    取 100 次调用中 Duration(us) 最小的一次数据。

    Args:
        prof_dir: profiler 输出目录

    Returns:
        dict 包含各项硬件指标（Duration 最小那次的数据）
    """
    prof_path = Path(prof_dir)

    # 查找 ASCEND_PROFILER_OUTPUT/kernel_details.csv
    # EP 模式下仅处理当前 rank 的数据（prof_dir 已包含 rank 后缀）
    kernel_details_files = list(prof_path.glob("*_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))

    if not kernel_details_files:
        print(f"[WARN] No kernel_details.csv found in {prof_dir}")
        return get_empty_metrics()

    # 读取所有 kernel_details.csv 中的 DispatchFFNCombine 行
    all_rows: list[dict[str, float]] = []

    for csv_path in kernel_details_files:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Name", "") or row.get("name", "")
                # 匹配 DFC 算子名称，包括 fallback 模式下的 dispatch_gmm_combine_decode
                if ("DispatchFFNCombine" in name or "dispatch_ffn_combine" in name
                        or "DispatchGmmCombineDecode" in name or "dispatch_gmm_combine_decode" in name):
                    try:
                        # 提取需要的指标
                        parsed_row = {
                            "Duration(us)": float(row.get("Duration(us)", 0) or 0),
                            "aicore_time(us)": float(row.get("aicore_time(us)", 0) or 0),
                            "aic_total_cycles": float(row.get("aic_total_cycles", 0) or 0),
                            "aic_mac_time(us)": float(row.get("aic_mac_time(us)", 0) or 0),
                            "aic_mac_ratio": float(row.get("aic_mac_ratio", 0) or 0),
                            "aic_scalar_time(us)": float(row.get("aic_scalar_time(us)", 0) or 0),
                            "aic_scalar_ratio": float(row.get("aic_scalar_ratio", 0) or 0),
                            "aic_mte1_time(us)": float(row.get("aic_mte1_time(us)", 0) or 0),
                            "aic_mte1_ratio": float(row.get("aic_mte1_ratio", 0) or 0),
                            "aic_mte2_time(us)": float(row.get("aic_mte2_time(us)", 0) or 0),
                            "aic_mte2_ratio": float(row.get("aic_mte2_ratio", 0) or 0),
                            "aic_fixpipe_time(us)": float(row.get("aic_fixpipe_time(us)", 0) or 0),
                            "aic_fixpipe_ratio": float(row.get("aic_fixpipe_ratio", 0) or 0),
                            "aic_icache_miss_rate": float(row.get("aic_icache_miss_rate", 0) or 0),
                            "aiv_time(us)": float(row.get("aiv_time(us)", 0) or 0),
                            "aiv_total_cycles": float(row.get("aiv_total_cycles", 0) or 0),
                            "aiv_vec_time(us)": float(row.get("aiv_vec_time(us)", 0) or 0),
                            "aiv_vec_ratio": float(row.get("aiv_vec_ratio", 0) or 0),
                            "aiv_scalar_time(us)": float(row.get("aiv_scalar_time(us)", 0) or 0),
                            "aiv_scalar_ratio": float(row.get("aiv_scalar_ratio", 0) or 0),
                            "aiv_mte2_time(us)": float(row.get("aiv_mte2_time(us)", 0) or 0),
                            "aiv_mte2_ratio": float(row.get("aiv_mte2_ratio", 0) or 0),
                            "aiv_mte3_time(us)": float(row.get("aiv_mte3_time(us)", 0) or 0),
                            "aiv_mte3_ratio": float(row.get("aiv_mte3_ratio", 0) or 0),
                            "aiv_icache_miss_rate": float(row.get("aiv_icache_miss_rate", 0) or 0),
                            "cube_utilization(%)": float(row.get("cube_utilization(%)", 0) or 0),
                        }
                        all_rows.append(parsed_row)
                    except (ValueError, TypeError):
                        continue

    if not all_rows:
        print(f"[WARN] No DispatchFFNCombine rows found in kernel_details.csv")
        return get_empty_metrics()

    # 找到 Duration 最小的那一行
    best_row = min(all_rows, key=lambda r: r["Duration(us)"])

    # 直接返回该行的数据，不做平均
    result = dict(best_row)
    result["sample_count"] = len(all_rows)

    return result


def get_empty_metrics() -> dict[str, float]:
    """返回空的指标字典。"""
    return {
        "Duration(us)": 0.0,
        "aicore_time(us)": 0.0,
        "aic_total_cycles": 0.0,
        "aic_mac_time(us)": 0.0,
        "aic_mac_ratio": 0.0,
        "aic_scalar_time(us)": 0.0,
        "aic_scalar_ratio": 0.0,
        "aic_mte1_time(us)": 0.0,
        "aic_mte1_ratio": 0.0,
        "aic_mte2_time(us)": 0.0,
        "aic_mte2_ratio": 0.0,
        "aic_fixpipe_time(us)": 0.0,
        "aic_fixpipe_ratio": 0.0,
        "aic_icache_miss_rate": 0.0,
        "aiv_time(us)": 0.0,
        "aiv_total_cycles": 0.0,
        "aiv_vec_time(us)": 0.0,
        "aiv_vec_ratio": 0.0,
        "aiv_scalar_time(us)": 0.0,
        "aiv_scalar_ratio": 0.0,
        "aiv_mte2_time(us)": 0.0,
        "aiv_mte2_ratio": 0.0,
        "aiv_mte3_time(us)": 0.0,
        "aiv_mte3_ratio": 0.0,
        "aiv_icache_miss_rate": 0.0,
        "cube_utilization(%)": 0.0,
        "sample_count": 0,
    }


def append_result_to_csv(
        output_csv: str,
        row: dict[str, Any],
        metrics: dict[str, float],
        ep_size: int,
) -> None:
    """将 benchmark 结果追加到 CSV 文件。"""
    p = Path(output_csv)
    write_header = not p.exists()

    result = {
        "OP State": row.get("OP State", ""),
        "Accelerator Core": row.get("Accelerator Core", ""),
        "Input Shapes": row.get("Input Shapes", ""),
        "Input Data Types": row.get("Input Data Types", ""),
        "Input Formats": row.get("Input Formats", ""),
        "Output Shapes": row.get("Output Shapes", ""),
        "Output Data Types": row.get("Output Data Types", ""),
        "Output Formats": row.get("Output Formats", ""),
        "EP Size": ep_size,
        "Average Duration(us)": f"{metrics.get('Duration(us)', 0):.6f}",
        "aicore_time(us)": f"{metrics.get('aicore_time(us)', 0):.6f}",
        "aic_total_cycles": f"{metrics.get('aic_total_cycles', 0):.0f}",
        "aic_mac_time(us)": f"{metrics.get('aic_mac_time(us)', 0):.6f}",
        "aic_mac_ratio": f"{metrics.get('aic_mac_ratio', 0):.6f}",
        "aic_scalar_time(us)": f"{metrics.get('aic_scalar_time(us)', 0):.6f}",
        "aic_scalar_ratio": f"{metrics.get('aic_scalar_ratio', 0):.6f}",
        "aic_mte1_time(us)": f"{metrics.get('aic_mte1_time(us)', 0):.6f}",
        "aic_mte1_ratio": f"{metrics.get('aic_mte1_ratio', 0):.6f}",
        "aic_mte2_time(us)": f"{metrics.get('aic_mte2_time(us)', 0):.6f}",
        "aic_mte2_ratio": f"{metrics.get('aic_mte2_ratio', 0):.6f}",
        "aic_fixpipe_time(us)": f"{metrics.get('aic_fixpipe_time(us)', 0):.6f}",
        "aic_fixpipe_ratio": f"{metrics.get('aic_fixpipe_ratio', 0):.6f}",
        "aic_icache_miss_rate": f"{metrics.get('aic_icache_miss_rate', 0):.6f}",
        "aiv_time(us)": f"{metrics.get('aiv_time(us)', 0):.6f}",
        "aiv_total_cycles": f"{metrics.get('aiv_total_cycles', 0):.0f}",
        "aiv_vec_time(us)": f"{metrics.get('aiv_vec_time(us)', 0):.6f}",
        "aiv_vec_ratio": f"{metrics.get('aiv_vec_ratio', 0):.6f}",
        "aiv_scalar_time(us)": f"{metrics.get('aiv_scalar_time(us)', 0):.6f}",
        "aiv_scalar_ratio": f"{metrics.get('aiv_scalar_ratio', 0):.6f}",
        "aiv_mte2_time(us)": f"{metrics.get('aiv_mte2_time(us)', 0):.6f}",
        "aiv_mte2_ratio": f"{metrics.get('aiv_mte2_ratio', 0):.6f}",
        "aiv_mte3_time(us)": f"{metrics.get('aiv_mte3_time(us)', 0):.6f}",
        "aiv_mte3_ratio": f"{metrics.get('aiv_mte3_ratio', 0):.6f}",
        "aiv_icache_miss_rate": f"{metrics.get('aiv_icache_miss_rate', 0):.6f}",
        "cube_utilization(%)": f"{metrics.get('cube_utilization(%)', 0):.6f}",
        "sample_count": int(metrics.get("sample_count", 0)),
        "Source": "benchmark",
    }

    fieldnames = list(result.keys())

    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(result)


def run_row(
        csv_path,
        row_index: int,
        row: dict[str, str],
        output_csv: str,
        prof_base_dir: str,
        balanced: bool = True,
) -> float:
    """执行单个 DFC 行的 benchmark 并写入 CSV。

    Args:
        csv_path: 源 CSV 路径
        row_index: 行索引
        row: 行数据
        output_csv: 输出 CSV 路径
        prof_base_dir: profiler 数据保存的基础目录
        balanced: 是否使用均衡 expert 分布

    Returns:
        平均耗时（微秒）
    """
    case = build_row_case(row, balanced=balanced)

    # 为每个 case 创建独立的 profiler 目录
    # 使用 row_index、token 数和 rank 作为标识，避免 EP 模式下目录冲突
    token_count = case['x'].shape[0]
    prof_dir = os.path.join(prof_base_dir, f"prof_row{row_index}_tokens{token_count}_rank{EP_RANK}")

    # 使用 profiler 执行 benchmark
    metrics = run_benchmark_with_profiler(case, prof_dir)
    duration_us = metrics.get("Duration(us)", 0)
    use_fallback = metrics.get("use_fallback", False)

    api_name = "dispatch_gmm_combine_decode" if use_fallback else "dispatch_ffn_combine"

    # 验证输出形状
    out, expert_token_nums, _ = execute_dfc_op(case, use_fallback)
    actual_shapes = [tuple(out.shape), tuple(expert_token_nums.shape)]
    expected_shapes = case["expected_output_shapes"]
    if actual_shapes[0] != expected_shapes[0]:
        raise ValueError(f"out shape mismatch: actual={actual_shapes[0]} expected={expected_shapes[0]}")
    if actual_shapes[1] != expected_shapes[1]:
        raise ValueError(
            f"expert_token_nums shape mismatch: actual={actual_shapes[1]} expected={expected_shapes[1]}"
        )

    # 仅 rank 0 写入 CSV，避免多 rank 并发写入竞态
    if EP_RANK == 0:
        append_result_to_csv(output_csv, row, metrics, EP_SIZE)

    # 打印日志（仅 rank 0）
    if EP_RANK == 0:
        balance_tag = " balanced" if balanced else ""
        ep_tag = f" EP={EP_SIZE}" if EP_SIZE > 1 else ""
        print(
            f"[OK]{balance_tag}{ep_tag} {csv_path}:{row_index} "
            f"api={api_name} "
            f"x={tuple(case['x'].shape)} "
            f"w1={tuple(case['weight1_list'][0].shape)} "
            f"w2={tuple(case['weight2_list'][0].shape)} "
            f"topk={case['topk']} experts={case['num_experts']} "
            f"weight_kind={case['weight_kind']} "
            f"Duration={duration_us:.2f}us "
            f"aicore={metrics.get('aicore_time(us)', 0):.2f}us "
            f"prof_dir={prof_dir}"
        )

    return duration_us


def main() -> None:
    global EP_SIZE, EP_RANK, ENABLE_BALANCED

    args = build_argparser().parse_args()

    # 从 CLI 参数获取 EP_SIZE
    EP_SIZE = args.ep_size

    # 从 CLI 参数获取 ENABLE_BALANCED
    ENABLE_BALANCED = args.balanced

    # 检测是否在 torchrun 环境中运行（子进程）
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    is_auto_torchrun = os.environ.get("_DFC_AUTO_TORCHRUN", "0") == "1"

    # 如果 EP_SIZE > 1 且不是在 torchrun 子进程中，则自动启动 torchrun
    if EP_SIZE > 1 and env_world_size == 1 and not is_auto_torchrun:
        # 收集命令行参数
        cli_args = []
        if args.device:
            cli_args.extend(["--device", args.device])
        if args.vllm_ascend_version:
            cli_args.extend(["--vllm-ascend-version", args.vllm_ascend_version])
        cli_args.extend(["--output-csv", args.output_csv])
        cli_args.extend(["--ep-size", str(EP_SIZE)])
        if not ENABLE_BALANCED:
            cli_args.append("--no-balanced")

        # 启动 torchrun
        exit_code = launch_torchrun_and_wait(EP_SIZE, cli_args)
        sys.exit(exit_code)

    # 以下是实际的执行逻辑（单进程或 torchrun 子进程）
    ensure_npu_available()

    # 设置 EP 配置
    if env_world_size > 1:
        # torchrun 子进程环境
        EP_SIZE = env_world_size
        EP_RANK = env_rank
        get_default_hccl_group_name()
        if EP_RANK == 0:
            print(f"[EP Mode] EP_SIZE={EP_SIZE}, EP_RANK={EP_RANK}")
    else:
        # 单进程模式
        print(f"[Single-process Mode] EP_SIZE={EP_SIZE} (no EP communication)")

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_ascend_version,
    )
    csv_paths = sorted(target_data_dir.rglob("DispatchFFNCombine.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No DispatchFFNCombine.csv found under {target_data_dir}")

    # 创建 profiler 数据保存目录（在输出 CSV 同级目录下）
    output_csv_path = Path(args.output_csv)
    prof_base_dir = output_csv_path.parent / f"PROF_{output_csv_path.stem}"
    prof_base_dir.mkdir(parents=True, exist_ok=True)
    if EP_RANK == 0:
        print(f"[PROF] Profiler data will be saved to: {prof_base_dir}")

    total_rows = 0
    for csv_path, row_index, row in iter_csv_rows(target_data_dir, "DispatchFFNCombine.csv"):
        run_row(csv_path, row_index, row, output_csv=args.output_csv, prof_base_dir=str(prof_base_dir),
                balanced=ENABLE_BALANCED)
        total_rows += 1
        # EP 模式下逐行 barrier，避免某 rank 异常导致其他 rank 永久阻塞
        if EP_SIZE > 1:
            import torch.distributed as dist
            dist.barrier()

    # 分布式模式下，只有 rank 0 打印总结
    if EP_RANK == 0:
        print(
            f"Processed {total_rows} DispatchFFNCombine rows "
            f"from {len(csv_paths)} csv file(s) under {target_data_dir}."
        )
        print(f"Results written to: {args.output_csv}")
        print(f"Profiler data saved to: {prof_base_dir}")

    # 同步所有 rank
    if EP_SIZE > 1:
        import torch.distributed as dist
        dist.barrier()


if __name__ == "__main__":
    main()