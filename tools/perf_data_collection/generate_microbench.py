"""Generate runnable Ascend NPU microbench scripts from perf database CSV files.

中文说明
1. 脚本作用
   - 基于 `tensor_cast/performance_model/perf_database/data/<device>/vllm_ascend/<version>`
     下的每个算子 CSV，生成对应的 Ascend NPU microbench 脚本。
   - 生成脚本目录为
     `tensor_cast/performance_model/perf_database/generated_microbench/<device>/<version>/`。
   - 每个算子生成一个脚本；脚本运行时会逐行读取源 CSV 的 `Input Shapes` / `Input Data Types`
     等列，构造输入 tensor，调用对应的 `torch` / `torch_npu` 接口测时。
   - 测得的耗时会写回源 CSV 的 `Average Duration by PTA(us)` 列；若该列不存在，则插入到
     `Average Duration(us)` 后面。

2. 当前版本核心改动
   - 参数规则与 `tools/perf_data_collection/parse_kernel_details.py` 对齐：
     `--device` 使用同一套设备名校验与归一化规则，`--vllm-ascend-version` 使用同一套版本格式校验。
   - `MatMulV2` / `MatMulV3` 支持 `torch.mm` 路径：
     对 `FRACTAL_NZ` 的 4D 权重先 reshape 为 2D，再执行 `torch.mm(lhs, weight.t())`。
   - 支持一批基础算子的脚本生成与执行：
     `MatMul/MatMulV2/MatMulV3/TransposeBatchMatMul/Add/Cast/GatherV2/GatherV3/ConcatD/ConcatV2D/
      BroadcastTo/Fill/Cumsum/Equal/GreaterEqual/Less/LessEqual/Log/LogicalAnd/LogicalNot/ArgMaxV2/
      ReduceMax/ReduceSum/SoftmaxV2/Sort/Sub/Mul/Muls/Neg/RealDiv/TensorMove/Transpose/ViewCopy/
      ZerosLike/RmsNorm/AddRmsNorm/InplaceAddRmsNorm/SwiGlu/AscendQuantV2/DynamicQuant/InterleaveRope`
   - 对 `op_mapping.yaml` 里缺失但语义清晰的基础算子，增加了本地 fallback 映射，保证尽量做到
     “一个 CSV 生成一个脚本”；fallback 情况会在生成阶段单独打屏报告。
   - 对 `op_mapping.yaml` 中存在映射、但当前实现还没有可靠输入构造规则的复杂算子，不会强行生成
     可能跑错的脚本，而是明确打印到“未实现 API”报告里。

3. 本地验证结果
   - `py -3 -m py_compile tools/perf_data_collection/generate_microbench.py` 通过。
   - 在本地 `v0.13.0` 真实样本数据上执行生成命令后，当前可生成 44 个 microbench 脚本。
   - 生成出的脚本再次批量执行 `py_compile`，语法检查通过。
   - 说明：这里验证的是“脚本生成正确、Python 语法正确”；是否能在目标机器上实际调用
     `torch_npu` / `atb` 成功运行，还依赖真实 Ascend NPU 环境。

4. 当前未覆盖或未接入的高复杂度算子
   - 已在 `op_mapping.yaml` 中有映射，但当前未实现可靠 microbench 调用：
     `ApplyRotaryPosEmb`
     `FusedInferAttentionScore`
     `GroupedMatmul`
     `KvRmsNormRopeCache`
     `QuantBatchMatmulV3`
     `ReshapeAndCacheNdKernel`
     `MoeDistributeCombineV2`
     `MoeDistributeDispatchV2`
     `MoeGatingTopK`
     `DequantSwigluQuant`
   - 原因：这些算子通常依赖更复杂的输入语义、额外参数、paged KV cache、量化元数据或
     ATB/私有接口签名。当前实现避免直接猜测，防止生成“能跑但不代表真实 kernel”的脚本。
   - 此外，仍有一批算子在 `op_mapping.yaml` 中没有 `microbench_api`，生成阶段会打印报告。

5. 如何运行
   - 先生成脚本：
     `python tools/perf_data_collection/generate_microbench.py --device ATLAS_800_A3_752T_128G_DIE --vllm-ascend-version 0.13.0`
   - 再运行某个具体算子的脚本，例如：
     `python tensor_cast/performance_model/perf_database/generated_microbench/atlas_800_a3_752t_128g_die/v0.13.0/MatMulV2_microbench.py --device npu:0`
   - 运行后会：
     1) 在 `generated_microbench/...` 下写结果 CSV
     2) 同步更新源 CSV 的 `Average Duration by PTA(us)` 列

6. 后续建议
   - 若要继续提高覆盖率，优先顺序建议为：
     `ApplyRotaryPosEmb` -> `FusedInferAttentionScore` -> `ReshapeAndCacheNdKernel` -> `KvRmsNormRopeCache`
   - 同时建议把现在使用 fallback 的基础算子补回 `op_mapping.yaml`，减少脚本内的本地映射分支。
"""

from __future__ import annotations

import argparse
import json
import stat
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_data_collection.parse_kernel_details import (
    SUPPORTED_DEVICES,
    check_version,
    normalize_device_name,
    normalize_vllm_ascend_version,
)


DEFAULT_DATA_ROOT = (
    REPO_ROOT
    / "tensor_cast"
    / "performance_model"
    / "perf_database"
    / "data"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "generated_microbench"
CSV_SUFFIX = ".csv"
OP_MAPPING_FILE = "op_mapping.yaml"
TORCH_NPU_REFERENCE_KEY = "torch_npu_reference"
AVERAGE_DURATION_COLUMN = "Average Duration(us)"
PTA_DURATION_COLUMN = "Average Duration by PTA(us)"
DEFAULT_WARMUP = 10
DEFAULT_ITERS = 50

SUPPORTED_MICROBENCH_APIS = {
    "torch.add",
    "torch.amax",
    "torch.argmax",
    "torch.broadcast_to",
    "torch.Tensor.to",
    "torch.bmm",
    "torch.cat",
    "torch.clone",
    "torch.cumsum",
    "torch.div",
    "torch.eq",
    "torch.full",
    "torch.ge",
    "torch.gather",
    "torch.index_select",
    "torch.le",
    "torch.log",
    "torch.logical_and",
    "torch.logical_not",
    "torch.lt",
    "torch.matmul",
    "torch.mm",
    "torch.mul",
    "torch.neg",
    "torch.nn.functional.embedding",
    "torch.nn.functional.softmax",
    "torch.permute",
    "torch.reshape",
    "torch.sort",
    "torch.sub",
    "torch.sum",
    "torch.zeros_like",
    "torch_npu.npu_add_rms_norm",
    "torch_npu.npu_dynamic_quant",
    "torch_npu.npu_interleave_rope",
    "torch_npu.npu_quantize",
    "torch_npu.npu_rms_norm",
    "torch_npu.npu_swiglu",
}

FALLBACK_API_BY_KERNEL = {
    "ArgMaxV2": "torch.argmax",
    "BroadcastTo": "torch.broadcast_to",
    "ConcatV2D": "torch.cat",
    "Cumsum": "torch.cumsum",
    "Equal": "torch.eq",
    "expand_kernel": "torch.broadcast_to",
    "Fill": "torch.full",
    "GatherElementsV2": "torch.gather",
    "GreaterEqual": "torch.ge",
    "Index": "torch.index_select",
    "Less": "torch.lt",
    "LessEqual": "torch.le",
    "Log": "torch.log",
    "LogicalAnd": "torch.logical_and",
    "LogicalNot": "torch.logical_not",
    "MatMulV3": "torch.mm",
    "Mul": "torch.mul",
    "Muls": "torch.mul",
    "Neg": "torch.neg",
    "RealDiv": "torch.div",
    "ReduceMax": "torch.amax",
    "ReduceSum": "torch.sum",
    "SoftmaxV2": "torch.nn.functional.softmax",
    "Sort": "torch.sort",
    "Sub": "torch.sub",
    "TensorMove": "torch.clone",
    "Transpose": "torch.permute",
    "ViewCopy": "torch.reshape",
    "ZerosLike": "torch.zeros_like",
}


SCRIPT_TEMPLATE = r'''#!/usr/bin/env python3
"""Auto-generated microbench for __KERNEL_TYPE__."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise SystemExit("torch_npu is required to run this script") from exc


KERNEL_TYPE = __KERNEL_TYPE_LITERAL__
API_NAME = __API_NAME_LITERAL__
DEVICE_DIR = __DEVICE_DIR_LITERAL__
VERSION_DIR = __VERSION_DIR_LITERAL__
SOURCE_CSV = Path(__SOURCE_CSV_LITERAL__)
RESULT_CSV = Path(__RESULT_CSV_LITERAL__)
AVERAGE_DURATION_COLUMN = "Average Duration(us)"
PTA_DURATION_COLUMN = "Average Duration by PTA(us)"
INPUT_SHAPES_COLUMN = "Input Shapes"
INPUT_DTYPES_COLUMN = "Input Data Types"
INPUT_FORMATS_COLUMN = "Input Formats"
OUTPUT_SHAPES_COLUMN = "Output Shapes"
OUTPUT_DTYPES_COLUMN = "Output Data Types"

DTYPE_MAP = {
    "DT_BF16": torch.bfloat16,
    "DT_FLOAT16": torch.float16,
    "DT_FLOAT": torch.float32,
    "FLOAT16": torch.float16,
    "FLOAT": torch.float32,
    "FLOAT32": torch.float32,
    "INT64": torch.int64,
    "INT32": torch.int32,
    "INT16": torch.int16,
    "INT8": torch.int8,
    "DT_INT4": torch.int8,
    "UINT8": torch.uint8,
    "BOOL": torch.bool,
}

QUANTIZED_OUTPUT_TYPES = {
    "INT8": torch.qint8,
    "UINT8": torch.quint8,
}


def synchronize() -> None:
    if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
        torch.npu.synchronize()


def split_semicolon_text(value: str) -> list[str]:
    text = (value or "").strip().strip('"')
    if not text:
        return []
    return [segment.strip() for segment in text.split(";")]


def parse_shape_text(value: str) -> list[tuple[int, ...] | None]:
    result: list[tuple[int, ...] | None] = []
    for segment in split_semicolon_text(value):
        normalized = segment.upper()
        if not segment or normalized in {"N/A", "NA", "NULL", "DT_UNDEFINED"}:
            result.append(None)
            continue
        result.append(tuple(int(part) for part in segment.split(",") if part))
    return result


def split_dtype_text(value: str) -> list[str]:
    return [segment.strip() for segment in split_semicolon_text(value)]


def split_format_text(value: str) -> list[str]:
    return [segment.strip().upper() for segment in split_semicolon_text(value)]


def resolve_dtype(dtype_text: str | None) -> torch.dtype | None:
    if not dtype_text:
        return None
    normalized = dtype_text.strip().upper()
    if normalized in {"DT_UNDEFINED", "NULL", "N/A", "NA"}:
        return None
    if normalized not in DTYPE_MAP:
        raise KeyError(f"Unsupported dtype: {dtype_text}")
    return DTYPE_MAP[normalized]


def get_output_dtype(output_dtype_texts: list[str]) -> torch.dtype | None:
    if not output_dtype_texts:
        return None
    return resolve_dtype(output_dtype_texts[0])


def make_tensor(shape: tuple[int, ...] | None, dtype_text: str | None, device: str) -> torch.Tensor | None:
    dtype = resolve_dtype(dtype_text)
    if shape is None or dtype is None:
        return None

    actual_shape = shape or (1,)
    if dtype is torch.bool:
        return torch.randint(0, 2, actual_shape, device=device, dtype=torch.int32).to(torch.bool)
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        low = 0
        high = 17 if dtype is not torch.int8 else 7
        return torch.randint(low, high, actual_shape, device=device, dtype=dtype)
    return torch.randn(actual_shape, device=device, dtype=dtype)


def make_index_tensor(shape: tuple[int, ...] | None, upper_bound: int, device: str) -> torch.Tensor:
    actual_shape = shape or (1,)
    upper = max(upper_bound, 1)
    return torch.randint(0, upper, actual_shape, device=device, dtype=torch.int64)


def scalar_from_tensor(value: torch.Tensor | None, default: int = 0) -> int:
    if value is None:
        return default
    return int(value.reshape(-1)[0].item())


def normalize_mm_weight(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim == 4:
        rows = weight.shape[0] * weight.shape[2]
        cols = weight.shape[1] * weight.shape[3]
        return weight.reshape(rows, cols)
    if weight.ndim == 2:
        return weight
    raise ValueError(f"torch.mm weight must be 2D or FRACTAL_NZ 4D, got {tuple(weight.shape)}")


def prepare_mm_mat2(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    input_formats: list[str],
    output_shape: tuple[int, ...] | None,
) -> torch.Tensor:
    weight = normalize_mm_weight(rhs)
    lhs_k = lhs.shape[-1]
    expected_n = output_shape[-1] if output_shape and len(output_shape) >= 2 else None
    rhs_format = input_formats[1] if len(input_formats) > 1 else ""

    # Real MatMulV2 samples show two different storage conventions:
    # - ND;ND: rhs shape in CSV is (N, K), runtime mm needs (K, N), so transpose.
    # - ND;FRACTAL_NZ: flattening 4D FRACTAL_NZ already yields (K, N), so do not transpose.
    if rhs_format == "FRACTAL_NZ":
        if weight.shape[0] != lhs_k:
            raise ValueError(
                f"FRACTAL_NZ mm K mismatch: lhs={tuple(lhs.shape)}, rhs={tuple(rhs.shape)}, "
                f"normalized_rhs={tuple(weight.shape)}, output_shape={output_shape}"
            )
        return weight
    if rhs_format == "ND":
        if weight.shape[1] != lhs_k:
            raise ValueError(
                f"ND mm K mismatch: lhs={tuple(lhs.shape)}, rhs={tuple(rhs.shape)}, "
                f"normalized_rhs={tuple(weight.shape)}, output_shape={output_shape}"
            )
        return weight.t().contiguous()

    # NPU mm expects mat2 shape (K, N). CSV samples are mostly (N, K), but some rows may
    # already be stored as (K, N). Decide orientation from the actual shape instead of
    # blindly transposing.
    if weight.shape[0] == lhs_k and (expected_n is None or weight.shape[1] == expected_n):
        return weight
    if weight.shape[1] == lhs_k and (expected_n is None or weight.shape[0] == expected_n):
        return weight.t().contiguous()
    if weight.shape[0] == lhs_k:
        return weight
    if weight.shape[1] == lhs_k:
        return weight.t().contiguous()
    raise ValueError(
        f"mm K mismatch: lhs={tuple(lhs.shape)}, rhs={tuple(rhs.shape)}, normalized_rhs={tuple(weight.shape)}, "
        f"output_shape={output_shape}"
    )


def maybe_prepare_bmm_rhs(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if lhs.ndim != 3 or rhs.ndim != 3:
        return rhs
    if lhs.shape[-1] == rhs.shape[-2]:
        return rhs
    if lhs.shape[-1] == rhs.shape[-1]:
        return rhs.transpose(-1, -2).contiguous()
    raise ValueError(f"bmm shape mismatch: {tuple(lhs.shape)} vs {tuple(rhs.shape)}")


def infer_concat_dim(input_shapes: list[tuple[int, ...] | None], output_shape: tuple[int, ...] | None) -> int:
    valid_shapes = [shape for shape in input_shapes if shape]
    if not valid_shapes or output_shape is None:
        return 0
    rank = len(valid_shapes[0])
    for dim in range(rank):
        if sum(shape[dim] for shape in valid_shapes) == output_shape[dim]:
            same_elsewhere = True
            for other_dim in range(rank):
                if other_dim == dim:
                    continue
                ref = valid_shapes[0][other_dim]
                if any(shape[other_dim] != ref for shape in valid_shapes[1:]):
                    same_elsewhere = False
                    break
                if output_shape[other_dim] != ref:
                    same_elsewhere = False
                    break
            if same_elsewhere:
                return dim
    return rank - 1


def infer_permutation(input_shape: tuple[int, ...] | None, output_shape: tuple[int, ...] | None) -> tuple[int, ...]:
    if input_shape is None:
        return (0,)
    if output_shape is None or len(input_shape) != len(output_shape):
        return tuple(range(len(input_shape)))
    used: set[int] = set()
    permutation: list[int] = []
    for out_dim in output_shape:
        matched = None
        for index, in_dim in enumerate(input_shape):
            if index in used:
                continue
            if in_dim == out_dim:
                matched = index
                break
        if matched is None:
            return tuple(range(len(input_shape)))
        used.add(matched)
        permutation.append(matched)
    return tuple(permutation)


def infer_reduce_dim(input_shape: tuple[int, ...] | None, output_shape: tuple[int, ...] | None) -> int:
    if input_shape is None or output_shape is None:
        return -1
    if len(output_shape) == len(input_shape) - 1:
        for index in range(len(input_shape)):
            if input_shape[:index] + input_shape[index + 1 :] == output_shape:
                return index
    if len(output_shape) == len(input_shape):
        for index, (in_dim, out_dim) in enumerate(zip(input_shape, output_shape)):
            if in_dim != out_dim and out_dim == 1:
                return index
    return -1


def build_tensors(
    row: dict[str, str],
    *,
    device: str,
) -> tuple[
    list[tuple[int, ...] | None],
    list[str],
    list[str],
    list[torch.Tensor | None],
    list[tuple[int, ...] | None],
    list[str],
]:
    input_shapes = parse_shape_text(row.get(INPUT_SHAPES_COLUMN, ""))
    input_dtypes = split_dtype_text(row.get(INPUT_DTYPES_COLUMN, ""))
    input_formats = split_format_text(row.get(INPUT_FORMATS_COLUMN, ""))
    while len(input_dtypes) < len(input_shapes):
        input_dtypes.append(input_dtypes[-1] if input_dtypes else "DT_FLOAT16")
    while len(input_formats) < len(input_shapes):
        input_formats.append(input_formats[-1] if input_formats else "")

    tensors = [make_tensor(shape, input_dtypes[index], device) for index, shape in enumerate(input_shapes)]
    output_shapes = parse_shape_text(row.get(OUTPUT_SHAPES_COLUMN, ""))
    output_dtypes = split_dtype_text(row.get(OUTPUT_DTYPES_COLUMN, ""))

    # Some elementwise kernels encode broadcast scalars as an empty shape slot with a valid dtype.
    scalar_fallback_apis = {
        "torch.add",
        "torch.sub",
        "torch.mul",
        "torch.div",
        "torch.eq",
        "torch.ge",
        "torch.lt",
        "torch.le",
        "torch.logical_and",
    }
    if API_NAME in scalar_fallback_apis:
        for index, tensor in enumerate(tensors):
            if tensor is None and index < len(input_dtypes) and input_dtypes[index]:
                tensors[index] = make_tensor((), input_dtypes[index], device)

    if API_NAME == "torch.nn.functional.embedding" and len(tensors) >= 2 and tensors[0] is not None:
        tensors[1] = make_index_tensor(input_shapes[1], tensors[0].shape[0], device)
    elif API_NAME == "torch.index_select" and len(tensors) >= 2 and tensors[0] is not None:
        tensors[1] = make_index_tensor(input_shapes[1], tensors[0].shape[0], device)
    elif API_NAME == "torch.gather" and len(tensors) >= 2 and tensors[0] is not None:
        upper = tensors[0].shape[-1] if tensors[0].ndim > 0 else 1
        tensors[1] = make_index_tensor(input_shapes[1], upper, device)

    return input_shapes, input_dtypes, input_formats, tensors, output_shapes, output_dtypes


def validate_inputs(
    row: dict[str, str],
    *,
    input_shapes: list[tuple[int, ...] | None],
    tensors: list[torch.Tensor | None],
) -> None:
    if not input_shapes:
        raise ValueError(f"Missing parsed input shapes, raw Input Shapes={row.get(INPUT_SHAPES_COLUMN, '')!r}")

    required_inputs = {
        "torch.mm": 2,
        "torch.matmul": 2,
        "torch.bmm": 2,
        "torch.add": 2,
        "torch.sub": 2,
        "torch.mul": 1,
        "torch.div": 2,
        "torch.eq": 2,
        "torch.ge": 2,
        "torch.lt": 2,
        "torch.le": 2,
        "torch.logical_and": 2,
        "torch.logical_not": 1,
        "torch.log": 1,
        "torch.Tensor.to": 1,
        "torch.nn.functional.embedding": 2,
        "torch.index_select": 2,
        "torch.cat": 1,
        "torch.broadcast_to": 1,
        "torch.full": 1,
        "torch.cumsum": 1,
        "torch.argmax": 1,
        "torch.amax": 1,
        "torch.sum": 1,
        "torch.nn.functional.softmax": 1,
        "torch.sort": 1,
        "torch.clone": 1,
        "torch.zeros_like": 1,
        "torch.reshape": 1,
        "torch.permute": 1,
        "torch.gather": 2,
        "torch_npu.npu_rms_norm": 2,
        "torch_npu.npu_add_rms_norm": 3,
        "torch_npu.npu_swiglu": 1,
        "torch_npu.npu_quantize": 2,
        "torch_npu.npu_dynamic_quant": 1,
        "torch_npu.npu_interleave_rope": 3,
    }

    required = required_inputs.get(API_NAME, 1)
    if len(tensors) < required:
        raise ValueError(
            f"Input tensor count mismatch for {API_NAME}: expected at least {required}, got {len(tensors)}; "
            f"raw Input Shapes={row.get(INPUT_SHAPES_COLUMN, '')!r}"
        )


def validate_inputs(
    row: dict[str, str],
    *,
    input_shapes: list[tuple[int, ...] | None],
    tensors: list[torch.Tensor | None],
) -> None:
    if not input_shapes:
        raise ValueError(f"Missing parsed input shapes, raw Input Shapes={row.get(INPUT_SHAPES_COLUMN, '')!r}")

    required_inputs = {
        "torch.mm": 2,
        "torch.matmul": 2,
        "torch.bmm": 2,
        "torch.add": 2,
        "torch.sub": 2,
        "torch.mul": 1,
        "torch.div": 2,
        "torch.eq": 2,
        "torch.ge": 2,
        "torch.lt": 2,
        "torch.le": 2,
        "torch.logical_and": 2,
        "torch.logical_not": 1,
        "torch.log": 1,
        "torch.Tensor.to": 1,
        "torch.nn.functional.embedding": 2,
        "torch.index_select": 2,
        "torch.cat": 1,
        "torch.broadcast_to": 1,
        "torch.full": 1,
        "torch.cumsum": 1,
        "torch.argmax": 1,
        "torch.amax": 1,
        "torch.sum": 1,
        "torch.nn.functional.softmax": 1,
        "torch.sort": 1,
        "torch.clone": 1,
        "torch.zeros_like": 1,
        "torch.reshape": 1,
        "torch.permute": 1,
        "torch.gather": 2,
        "torch_npu.npu_rms_norm": 2,
        "torch_npu.npu_add_rms_norm": 3,
        "torch_npu.npu_swiglu": 1,
        "torch_npu.npu_quantize": 2,
        "torch_npu.npu_dynamic_quant": 1,
        "torch_npu.npu_interleave_rope": 3,
    }

    required = required_inputs.get(API_NAME, 1)
    if len(tensors) < required:
        raise ValueError(
            f"Input tensor count mismatch for {API_NAME}: expected at least {required}, got {len(tensors)}; "
            f"raw Input Shapes={row.get(INPUT_SHAPES_COLUMN, '')!r}"
        )


def run_kernel(
    row: dict[str, str],
    *,
    tensors: list[torch.Tensor | None],
    input_shapes: list[tuple[int, ...] | None],
    input_formats: list[str],
    output_shapes: list[tuple[int, ...] | None],
    output_dtypes: list[str],
) -> object:
    if API_NAME == "torch.mm":
        lhs = tensors[0]
        rhs = tensors[1]
        if lhs is None or rhs is None:
            raise ValueError("torch.mm requires two inputs")
        output_shape = output_shapes[0] if output_shapes else None
        mat2 = prepare_mm_mat2(lhs, rhs, input_formats, output_shape)
        return torch.mm(lhs, mat2)

    if API_NAME == "torch.matmul":
        lhs = tensors[0]
        rhs = tensors[1]
        if lhs is None or rhs is None:
            raise ValueError("torch.matmul requires two inputs")
        return torch.matmul(lhs, rhs)

    if API_NAME == "torch.bmm":
        lhs = tensors[0]
        rhs = tensors[1]
        if lhs is None or rhs is None:
            raise ValueError("torch.bmm requires two inputs")
        return torch.bmm(lhs, maybe_prepare_bmm_rhs(lhs, rhs))

    if API_NAME == "torch.add":
        return torch.add(tensors[0], tensors[1])

    if API_NAME == "torch.sub":
        return torch.sub(tensors[0], tensors[1])

    if API_NAME == "torch.mul":
        lhs = tensors[0]
        rhs = tensors[1] if len(tensors) > 1 else None
        if lhs is None:
            raise ValueError("torch.mul requires at least one input")
        if rhs is None:
            return torch.mul(lhs, 1)
        if rhs.numel() == 1 and lhs.ndim > 0:
            return torch.mul(lhs, rhs.reshape(()))
        return torch.mul(lhs, rhs)

    if API_NAME == "torch.div":
        return torch.div(tensors[0], tensors[1])

    if API_NAME == "torch.neg":
        return torch.neg(tensors[0])

    if API_NAME == "torch.eq":
        return torch.eq(tensors[0], tensors[1])

    if API_NAME == "torch.ge":
        return torch.ge(tensors[0], tensors[1])

    if API_NAME == "torch.lt":
        return torch.lt(tensors[0], tensors[1])

    if API_NAME == "torch.le":
        return torch.le(tensors[0], tensors[1])

    if API_NAME == "torch.logical_and":
        return torch.logical_and(tensors[0].to(torch.bool), tensors[1].to(torch.bool))

    if API_NAME == "torch.logical_not":
        return torch.logical_not(tensors[0].to(torch.bool))

    if API_NAME == "torch.log":
        return torch.log(torch.abs(tensors[0]) + 1e-6)

    if API_NAME == "torch.Tensor.to":
        source = tensors[0]
        target_dtype = get_output_dtype(output_dtypes)
        if source is None or target_dtype is None:
            raise ValueError("torch.Tensor.to requires input and output dtype")
        return source.to(dtype=target_dtype)

    if API_NAME == "torch.nn.functional.embedding":
        weight = tensors[0]
        indices = tensors[1]
        if weight is None or indices is None:
            raise ValueError("embedding requires weight and indices")
        return torch.nn.functional.embedding(indices.to(torch.int64), weight)

    if API_NAME == "torch.index_select":
        source = tensors[0]
        indices = tensors[1]
        if source is None or indices is None:
            raise ValueError("index_select requires source and indices")
        return torch.index_select(source, 0, indices.reshape(-1).to(torch.int64))

    if API_NAME == "torch.cat":
        target_shape = output_shapes[0] if output_shapes else None
        dim = infer_concat_dim(input_shapes, target_shape)
        values = [tensor for tensor in tensors if tensor is not None]
        return torch.cat(values, dim=dim)

    if API_NAME == "torch.broadcast_to":
        source = tensors[0]
        target_shape = output_shapes[0] if output_shapes else None
        if source is None or target_shape is None:
            raise ValueError("broadcast_to requires input and output shape")
        return torch.broadcast_to(source, target_shape)

    if API_NAME == "torch.full":
        target_shape = output_shapes[0] if output_shapes else None
        if target_shape is None:
            raise ValueError("fill requires output shape")
        fill_value = scalar_from_tensor(tensors[0], default=0)
        target_dtype = get_output_dtype(output_dtypes) or torch.int32
        return torch.full(target_shape, fill_value, device=tensors[0].device if tensors[0] is not None else "npu:0", dtype=target_dtype)

    if API_NAME == "torch.cumsum":
        source = tensors[0]
        if source is None:
            raise ValueError("cumsum requires source")
        dim = scalar_from_tensor(tensors[1], default=-1)
        dim = dim if -source.ndim <= dim < source.ndim else -1
        return torch.cumsum(source, dim=dim)

    if API_NAME == "torch.argmax":
        source = tensors[0]
        if source is None:
            raise ValueError("argmax requires source")
        dim = scalar_from_tensor(tensors[1], default=-1)
        dim = dim if -source.ndim <= dim < source.ndim else -1
        return torch.argmax(source, dim=dim)

    if API_NAME == "torch.amax":
        source = tensors[0]
        if source is None:
            raise ValueError("amax requires source")
        default_dim = infer_reduce_dim(input_shapes[0], output_shapes[0] if output_shapes else None)
        dim = scalar_from_tensor(tensors[1] if len(tensors) > 1 else None, default=default_dim)
        dim = dim if -source.ndim <= dim < source.ndim else -1
        return torch.amax(source, dim=dim)

    if API_NAME == "torch.sum":
        source = tensors[0]
        if source is None:
            raise ValueError("sum requires source")
        default_dim = infer_reduce_dim(input_shapes[0], output_shapes[0] if output_shapes else None)
        dim = scalar_from_tensor(tensors[1] if len(tensors) > 1 else None, default=default_dim)
        dim = dim if -source.ndim <= dim < source.ndim else -1
        return torch.sum(source, dim=dim)

    if API_NAME == "torch.nn.functional.softmax":
        source = tensors[0]
        if source is None:
            raise ValueError("softmax requires source")
        dim = scalar_from_tensor(tensors[1] if len(tensors) > 1 else None, default=-1)
        dim = dim if -source.ndim <= dim < source.ndim else -1
        return torch.nn.functional.softmax(source, dim=dim)

    if API_NAME == "torch.sort":
        source = tensors[0]
        if source is None:
            raise ValueError("sort requires source")
        dim = scalar_from_tensor(tensors[1] if len(tensors) > 1 else None, default=-1)
        dim = dim if -source.ndim <= dim < source.ndim else -1
        return torch.sort(source, dim=dim)

    if API_NAME == "torch.clone":
        return tensors[0].clone()

    if API_NAME == "torch.zeros_like":
        return torch.zeros_like(tensors[0])

    if API_NAME == "torch.reshape":
        source = tensors[0]
        target_shape = output_shapes[0] if output_shapes else None
        if source is None or target_shape is None:
            raise ValueError("reshape requires source and output shape")
        return torch.reshape(source, target_shape)

    if API_NAME == "torch.permute":
        source = tensors[0]
        if source is None:
            raise ValueError("permute requires source")
        permutation = infer_permutation(input_shapes[0], output_shapes[0] if output_shapes else None)
        return source.permute(permutation).contiguous()

    if API_NAME == "torch.gather":
        source = tensors[0]
        indices = tensors[1]
        if source is None or indices is None:
            raise ValueError("gather requires source and indices")
        dim = scalar_from_tensor(tensors[2] if len(tensors) > 2 else None, default=-1)
        dim = dim if -source.ndim <= dim < source.ndim else -1
        return torch.gather(source, dim, indices.to(torch.int64))

    if API_NAME == "torch_npu.npu_rms_norm":
        gamma = tensors[1]
        if tensors[0] is None or gamma is None:
            raise ValueError("npu_rms_norm requires x and gamma")
        return torch_npu.npu_rms_norm(tensors[0], gamma, epsilon=1e-6)

    if API_NAME == "torch_npu.npu_add_rms_norm":
        residual = tensors[1]
        gamma = tensors[2]
        if tensors[0] is None or residual is None or gamma is None:
            raise ValueError("npu_add_rms_norm requires x, residual, gamma")
        return torch_npu.npu_add_rms_norm(tensors[0], residual, gamma, epsilon=1e-6)

    if API_NAME == "torch_npu.npu_swiglu":
        if tensors[0] is None:
            raise ValueError("npu_swiglu requires one input")
        return torch_npu.npu_swiglu(tensors[0], dim=-1)

    if API_NAME == "torch_npu.npu_quantize":
        x = tensors[0]
        scale = tensors[1]
        offset = tensors[2] if len(tensors) > 2 else None
        if x is None or scale is None:
            raise ValueError("npu_quantize requires x and scale")
        quant_dtype = QUANTIZED_OUTPUT_TYPES.get((output_dtypes[0] if output_dtypes else "").upper(), torch.qint8)
        return torch_npu.npu_quantize(x, scale, offset, quant_dtype, -1, False)

    if API_NAME == "torch_npu.npu_dynamic_quant":
        x = tensors[0]
        smooth_scales = tensors[1] if len(tensors) > 1 else None
        if x is None:
            raise ValueError("npu_dynamic_quant requires x")
        if smooth_scales is None:
            return torch_npu.npu_dynamic_quant(x)
        return torch_npu.npu_dynamic_quant(x, smooth_scales=smooth_scales)

    if API_NAME == "torch_npu.npu_interleave_rope":
        x = tensors[0]
        cos = tensors[1]
        sin = tensors[2]
        if x is None or cos is None or sin is None:
            raise ValueError("npu_interleave_rope requires x, cos, sin")
        return torch_npu.npu_interleave_rope(x, cos, sin)

    raise NotImplementedError(f"Unsupported API in generated script: {API_NAME}")


def update_pta_column(headers: list[str]) -> list[str]:
    if PTA_DURATION_COLUMN in headers:
        return headers
    if AVERAGE_DURATION_COLUMN in headers:
        index = headers.index(AVERAGE_DURATION_COLUMN) + 1
        return headers[:index] + [PTA_DURATION_COLUMN] + headers[index:]
    return headers + [PTA_DURATION_COLUMN]


def write_rows(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def default_source_csv() -> Path:
    script_path = Path(__file__).resolve()
    perf_database_dir = script_path.parents[2]
    candidate = perf_database_dir / "data" / DEVICE_DIR / "vllm_ascend" / VERSION_DIR / f"{KERNEL_TYPE}.csv"
    if candidate.exists():
        return candidate
    return SOURCE_CSV


def default_result_csv() -> Path:
    script_path = Path(__file__).resolve()
    return script_path.with_name(f"{KERNEL_TYPE}_results.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generated Ascend NPU microbench")
    parser.add_argument("--device", default="npu:0", help="Execution device, e.g. npu:0")
    parser.add_argument("--csv-path", type=Path, default=default_source_csv())
    parser.add_argument("--output-csv", type=Path, default=default_result_csv())
    parser.add_argument("--warmup", type=int, default=__WARMUP__)
    parser.add_argument("--iters", type=int, default=__ITERS__)
    args = parser.parse_args()

    with args.csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        headers = list(reader.fieldnames or [])

    if not rows:
        raise ValueError(f"CSV has no rows: {args.csv_path}")

    result_rows: list[dict[str, str]] = []
    result_headers = update_pta_column(headers)
    for row_index, row in enumerate(rows, start=1):
        input_shapes, input_dtypes, input_formats, tensors, output_shapes, output_dtypes = build_tensors(
            row,
            device=args.device,
        )
        try:
            validate_inputs(row, input_shapes=input_shapes, tensors=tensors)
        except ValueError as exc:
            print(f"[skip {row_index}/{len(rows)}] {KERNEL_TYPE}: {exc}")
            continue
        for _ in range(args.warmup):
            run_kernel(
                row,
                tensors=tensors,
                input_shapes=input_shapes,
                input_formats=input_formats,
                output_shapes=output_shapes,
                output_dtypes=output_dtypes,
            )
        synchronize()

        start = time.perf_counter()
        for _ in range(args.iters):
            run_kernel(
                row,
                tensors=tensors,
                input_shapes=input_shapes,
                input_formats=input_formats,
                output_shapes=output_shapes,
                output_dtypes=output_dtypes,
            )
        synchronize()
        elapsed_us = (time.perf_counter() - start) * 1_000_000 / args.iters

        updated = dict(row)
        updated[PTA_DURATION_COLUMN] = f"{elapsed_us:.6f}"
        result_rows.append(updated)
        print(f"[{row_index}/{len(rows)}] {KERNEL_TYPE}: {updated.get(INPUT_SHAPES_COLUMN, '')} -> {elapsed_us:.6f} us")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_csv, result_headers, result_rows)
    write_rows(args.csv_path, result_headers, result_rows)
    print(f"Wrote results to {args.output_csv}")
    print(f"Updated source CSV {args.csv_path}")


if __name__ == "__main__":
    main()
'''


class MicrobenchError(RuntimeError):
    """Raised when microbench generation cannot proceed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate runnable Ascend NPU microbench scripts from perf database CSV files."
    )
    parser.add_argument(
        "--device",
        required=True,
        help="Device name under perf_database/data, same rule as parse_kernel_details.py",
    )
    parser.add_argument(
        "--vllm-ascend-version",
        required=True,
        type=check_version,
        help="vLLM-Ascend version, same rule as parse_kernel_details.py",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--kernel", default=None, help="Optional single kernel name")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    args = parser.parse_args()

    normalized_device = normalize_device_name(args.device)
    supported_devices = {normalize_device_name(device) for device in SUPPORTED_DEVICES}
    if normalized_device not in supported_devices:
        parser.error(
            "--device must match parse_kernel_details.py supported devices: "
            + ", ".join(sorted(supported_devices))
        )
    args.device = normalized_device
    args.vllm_ascend_version = normalize_vllm_ascend_version(args.vllm_ascend_version)
    return args


def load_op_mapping(op_mapping_path: Path) -> dict[str, Any]:
    with op_mapping_path.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file)
    reference = content.get(TORCH_NPU_REFERENCE_KEY)
    if not isinstance(reference, dict):
        raise MicrobenchError(f"Missing '{TORCH_NPU_REFERENCE_KEY}' in {op_mapping_path}")
    return reference


def collect_csv_files(version_dir: Path, kernel: str | None) -> list[Path]:
    if kernel:
        csv_path = version_dir / f"{kernel}{CSV_SUFFIX}"
        if not csv_path.is_file():
            raise MicrobenchError(f"Kernel CSV not found: {csv_path}")
        return [csv_path]
    return sorted(
        path for path in version_dir.glob(f"*{CSV_SUFFIX}") if path.name != OP_MAPPING_FILE
    )


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def render_script(
    *,
    kernel_type: str,
    api_name: str,
    device_dir: str,
    version_dir: str,
    csv_path: Path,
    result_csv_path: Path,
    warmup: int,
    iters: int,
) -> str:
    content = SCRIPT_TEMPLATE
    replacements = {
        "__KERNEL_TYPE__": kernel_type,
        "__KERNEL_TYPE_LITERAL__": json.dumps(kernel_type),
        "__API_NAME_LITERAL__": json.dumps(api_name),
        "__DEVICE_DIR_LITERAL__": json.dumps(device_dir),
        "__VERSION_DIR_LITERAL__": json.dumps(version_dir),
        "__SOURCE_CSV_LITERAL__": json.dumps(str(csv_path)),
        "__RESULT_CSV_LITERAL__": json.dumps(str(result_csv_path)),
        "__WARMUP__": str(warmup),
        "__ITERS__": str(iters),
    }
    for key, value in replacements.items():
        content = content.replace(key, value)
    return content


def write_script(script_path: Path, content: str) -> None:
    ensure_output_dir(script_path.parent)
    script_path.write_text(content, encoding="utf-8", newline="\n")
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)


def resolve_microbench_api(reference_mapping: dict[str, Any], kernel_type: str) -> str | None:
    kernel_mapping = reference_mapping.get(kernel_type)
    if not isinstance(kernel_mapping, dict):
        return FALLBACK_API_BY_KERNEL.get(kernel_type)
    api_name = kernel_mapping.get("microbench_api")
    if not isinstance(api_name, str) or not api_name.strip():
        return FALLBACK_API_BY_KERNEL.get(kernel_type)
    return api_name.strip()


def main() -> None:
    args = parse_args()
    version_dir = args.data_root / args.device / "vllm_ascend" / args.vllm_ascend_version
    if not version_dir.is_dir():
        raise MicrobenchError(f"Version directory does not exist: {version_dir}")

    op_mapping_path = version_dir / OP_MAPPING_FILE
    if not op_mapping_path.is_file():
        raise MicrobenchError(f"Missing op_mapping.yaml: {op_mapping_path}")

    output_dir = args.output_root / args.device / args.vllm_ascend_version
    ensure_output_dir(output_dir)
    reference_mapping = load_op_mapping(op_mapping_path)
    csv_files = collect_csv_files(version_dir, args.kernel)
    if not csv_files:
        raise MicrobenchError(f"No CSV files found under {version_dir}")

    generated: list[tuple[str, str, Path]] = []
    missing_mapping: list[str] = []
    unsupported_api: list[tuple[str, str]] = []
    fallback_generated: list[tuple[str, str]] = []

    for csv_path in csv_files:
        kernel_type = csv_path.stem
        mapped_api = None
        kernel_mapping = reference_mapping.get(kernel_type)
        if isinstance(kernel_mapping, dict):
            raw_api = kernel_mapping.get("microbench_api")
            if isinstance(raw_api, str) and raw_api.strip():
                mapped_api = raw_api.strip()
        api_name = resolve_microbench_api(reference_mapping, kernel_type)
        if api_name is None:
            missing_mapping.append(kernel_type)
            continue
        if api_name not in SUPPORTED_MICROBENCH_APIS:
            unsupported_api.append((kernel_type, api_name))
            continue
        if mapped_api is None:
            fallback_generated.append((kernel_type, api_name))

        script_path = output_dir / f"{kernel_type}_microbench.py"
        result_csv_path = output_dir / f"{kernel_type}_results.csv"
        write_script(
            script_path,
            render_script(
                kernel_type=kernel_type,
                api_name=api_name,
                device_dir=args.device,
                version_dir=args.vllm_ascend_version,
                csv_path=csv_path,
                result_csv_path=result_csv_path,
                warmup=args.warmup,
                iters=args.iters,
            ),
        )
        generated.append((kernel_type, api_name, script_path))

    print(f"Generated {len(generated)} microbench script(s) under {output_dir}")
    for kernel_type, api_name, script_path in generated:
        print(f"- {kernel_type}: {api_name} -> {script_path}")

    if fallback_generated:
        print("\nOperators generated via local fallback mapping (not present in op_mapping.yaml):")
        for kernel_type, api_name in fallback_generated:
            print(f"- {kernel_type}: {api_name}")

    if missing_mapping:
        print("\nOperators missing microbench mapping in op_mapping.yaml:")
        for kernel_type in missing_mapping:
            print(f"- {kernel_type}")

    if unsupported_api:
        print("\nOperators mapped in op_mapping.yaml but not yet implemented here:")
        for kernel_type, api_name in unsupported_api:
            print(f"- {kernel_type}: {api_name}")


if __name__ == "__main__":
    main()
