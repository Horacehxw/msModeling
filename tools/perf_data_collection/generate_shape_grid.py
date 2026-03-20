"""Append randomized, constraint-aware shape-grid rows to perf database CSV files."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import stat
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "tensor_cast"
    / "performance_model"
    / "perf_database"
    / "data"
)
DEFAULT_ROWS = 10_000
DEFAULT_MIN = 1
DEFAULT_MAX = 20_000
SUPPORTED_DEVICES = [
    "TEST_DEVICE",
    "ATLAS_800_A2_376T_64G",
    "ATLAS_800_A2_313T_64G",
    "ATLAS_800_A2_280T_64G",
    "ATLAS_800_A2_280T_64G_PCIE",
    "ATLAS_800_A2_280T_32G_PCIE",
    "ATLAS_800_A3_752T_128G_DIE",
    "ATLAS_800_A3_560T_128G_DIE",
]
KEEP_COLUMNS = {
    "OP State",
    "Accelerator Core",
    "Input Data Types",
    "Input Formats",
    "Output Data Types",
    "Output Formats",
}
INPUT_SHAPES_COLUMN = "Input Shapes"
OUTPUT_SHAPES_COLUMN = "Output Shapes"
ZERO_VALUE = "0"
ELEMENTWISE_BINARY_KERNELS = {
    "Add",
    "Equal",
    "FloorDiv",
    "FloorMod",
    "GreaterEqual",
    "Less",
    "LessAiCore",
    "LogicalAndAiCore",
    "LogicalAnd",
    "MaskedFill",
    "MaskedFillAiCore",
    "Mul",
    "MulAiCore",
    "NotEqual",
    "RealDiv",
    "Sub",
    "SubAiCore",
}
ELEMENTWISE_UNARY_KERNELS = {
    "Cast",
    "CastAiCore",
    "ZerosLike",
    "Fill",
    "Log",
    "LogicalNot",
    "LogicalNotAiCore",
    "Muls",
    "Neg",
    "SoftmaxV2",
    "TensorMove",
}
MATMUL_KERNELS = {
    "BatchMatMulV2",
    "MatMul",
    "MatMulCommon",
    "MatMulV2",
    "MatMulV3",
    "MatmulReduceScatterV2",
}
LATENCY_HEADER_KEYWORDS = ("duration", "latency", "time", "cycles", "ratio", "miss", "utilization")
MISSING_SHAPE_TOKENS = {"", "N/A", "NA", "NULL", "NONE", "UNDEFINED"}
OUTPUT_TEMPLATE_KERNELS = {"Range"}
SPLIT_QKV_ROPE_HEAD_DIM = 64


def _render_progress(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return f"[{'-' * width}]"
    ratio = min(max(current / total, 0.0), 1.0)
    filled = min(width, int(ratio * width))
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def print_progress(
    *,
    file_index: int,
    total_files: int,
    csv_path: Path,
    row_index: int,
    total_rows: int,
    appended_rows: int,
) -> None:
    file_bar = _render_progress(file_index, total_files)
    row_bar = _render_progress(row_index, total_rows)
    message = (
        f"\rFiles {file_bar} {file_index}/{total_files} | "
        f"Rows {row_bar} {row_index}/{total_rows} | "
        f"Appended {appended_rows} | {csv_path.name}"
    )
    print(message, end="", file=sys.stderr, flush=True)


def clear_progress() -> None:
    print("\r" + " " * 160 + "\r", end="", file=sys.stderr, flush=True)


def check_version(value: str) -> str:
    version = value.strip()
    if not re.fullmatch(r"[0-9A-Za-z]+(?:[._-][0-9A-Za-z]+)*", version):
        raise argparse.ArgumentTypeError(
            f"Invalid --vllm-ascend-version: {value!r}. "
            "Expected value like 0.9.2 or vllm0.13.0_torch2.8.0_cann8.3"
        )
    return version


def normalize_device_name(device: str) -> str:
    return device.strip()


def normalize_vllm_ascend_version(version: str) -> str:
    normalized = version.strip()
    if not normalized.startswith("v"):
        normalized = f"v{normalized}"
    return normalized


def resolve_data_dir(
    data_dir: Path | None,
    device: str | None,
    vllm_ascend_version: str | None,
) -> Path:
    if data_dir is not None:
        return data_dir
    if device and vllm_ascend_version:
        return (
            DEFAULT_DATA_DIR
            / normalize_device_name(device)
            / "vllm_ascend"
            / normalize_vllm_ascend_version(vllm_ascend_version)
        )
    return DEFAULT_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append randomized rows to all perf database CSV files while preserving "
            "existing real data."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "CSV root directory. If omitted, the script uses either "
            "{repo}/tensor_cast/performance_model/perf_database/data or "
            "{repo}/.../data/{device}/vllm_ascend/{version}/ when --device and "
            "--vllm-ascend-version are provided."
        ),
    )
    parser.add_argument(
        "--device",
        choices=SUPPORTED_DEVICES,
        help=(
            "Target device name used as input folder: "
            "tensor_cast/performance_model/perf_database/data/{device}/vllm_ascend/{version}/"
        ),
    )
    parser.add_argument(
        "--vllm-ascend-version",
        type=check_version,
        help="vLLM-Ascend version, e.g. 0.9.2.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
        help=f"Number of rows to append for each CSV. Default: {DEFAULT_ROWS}",
    )
    parser.add_argument(
        "--min-value",
        type=int,
        default=DEFAULT_MIN,
        help=f"Minimum random value for each generated dimension. Default: {DEFAULT_MIN}",
    )
    parser.add_argument(
        "--max-value",
        type=int,
        default=DEFAULT_MAX,
        help=f"Maximum random value for each generated dimension. Default: {DEFAULT_MAX}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible output.",
    )
    return parser.parse_args()


def parse_shape_text(shape_text: str) -> list[tuple[int, ...]]:
    text = (shape_text or "").strip().strip('"')
    if text.upper() in MISSING_SHAPE_TOKENS:
        return []

    shapes: list[tuple[int, ...]] = []
    for segment in text.split(";"):
        segment = segment.strip().strip('"')
        if not segment or segment.upper() in MISSING_SHAPE_TOKENS:
            shapes.append(())
            continue
        parts = [part.strip() for part in segment.split(",")]
        if any(part.upper() in MISSING_SHAPE_TOKENS for part in parts):
            shapes.append(())
            continue
        try:
            shapes.append(tuple(int(part) for part in parts if part))
        except ValueError:
            shapes.append(())
    return shapes


def build_shape_text(shapes: list[tuple[int, ...]]) -> str:
    return ";".join(",".join(str(value) for value in shape) if shape else "" for shape in shapes)


def build_shape_cell(shapes: list[tuple[int, ...]]) -> str:
    return f'"{build_shape_text(shapes)}"'


def infer_scalar_shape_count(shape_count: int, template_shapes: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    if len(template_shapes) >= shape_count:
        return list(template_shapes[:shape_count])
    return list(template_shapes) + [()] * (shape_count - len(template_shapes))


def align_shape_slot_count(
    template_shapes: list[tuple[int, ...]],
    generated_shapes: list[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    target_count = len(template_shapes) if template_shapes else len(generated_shapes)
    if len(generated_shapes) >= target_count:
        return list(generated_shapes[:target_count])
    return list(generated_shapes) + [()] * (target_count - len(generated_shapes))


def random_dim(
    rng: random.Random,
    min_value: int,
    max_value: int,
    *,
    template_dim: int | None = None,
    alignment: int | None = None,
) -> int:
    if template_dim == 1:
        return 1
    base_lower = min_value
    base_upper = max_value
    lower = base_lower
    upper = base_upper
    if template_dim and template_dim > 1:
        lower = max(min_value, max(2, template_dim // 2))
        upper = min(max_value, max(lower, template_dim * 2))
        if lower > upper:
            lower = base_lower
            upper = base_upper
    if alignment and alignment > 1:
        aligned_lower = max(alignment, ((lower + alignment - 1) // alignment) * alignment)
        aligned_upper = (upper // alignment) * alignment
        if aligned_lower <= aligned_upper:
            steps = ((aligned_upper - aligned_lower) // alignment) + 1
            return aligned_lower + alignment * rng.randrange(steps)
    return rng.randint(lower, upper)


def mutate_shape(
    shape: tuple[int, ...],
    rng: random.Random,
    min_value: int,
    max_value: int,
    shared_dims: dict[int, int] | None = None,
) -> tuple[int, ...]:
    shared_dims = shared_dims or {}
    mutated: list[int] = []
    for dim in shape:
        if dim <= 1:
            mutated.append(dim)
            continue
        if dim not in shared_dims:
            shared_dims[dim] = random_dim(rng, min_value, max_value, template_dim=dim)
        mutated.append(shared_dims[dim])
    return tuple(mutated)


def zero_fill_column(header: str) -> bool:
    normalized = header.lower()
    return any(keyword in normalized for keyword in LATENCY_HEADER_KEYWORDS)


def generate_generic_shapes(
    template_inputs: list[tuple[int, ...]],
    template_outputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    shared_dims: dict[int, int] = {}
    inputs = [mutate_shape(shape, rng, min_value, max_value, shared_dims) for shape in template_inputs]
    outputs = [mutate_shape(shape, rng, min_value, max_value, shared_dims) for shape in template_outputs]
    return inputs, outputs


def generate_elementwise_binary_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(2, template_inputs)
    base = mutate_shape(inputs[0], rng, min_value, max_value)
    rhs_template = inputs[1]
    if not rhs_template:
        rhs = ()
    elif rhs_template == inputs[0]:
        rhs = base
    else:
        base_offset = max(0, len(base) - len(rhs_template))
        rhs_values: list[int] = []
        for index, dim in enumerate(rhs_template):
            if dim == 1:
                rhs_values.append(1)
                continue
            base_index = base_offset + index
            rhs_values.append(base[base_index] if base else dim)
        rhs = tuple(rhs_values)
    return [base, rhs], [base]


def generate_elementwise_unary_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    input_shape = mutate_shape(template_inputs[0], rng, min_value, max_value) if template_inputs else ()
    outputs = [input_shape] if input_shape else []
    return ([input_shape] if input_shape else []), outputs


def generate_matmul_shapes(
    template_inputs: list[tuple[int, ...]],
    template_formats: list[str],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    lhs_template, rhs_template = infer_scalar_shape_count(2, template_inputs)
    lhs_rank = len(lhs_template) or 2
    rhs_format = template_formats[1] if len(template_formats) > 1 else "ND"

    if lhs_rank == 2:
        m = random_dim(
            rng,
            min_value,
            max_value,
            template_dim=lhs_template[0] if len(lhs_template) >= 1 else None,
            alignment=8,
        )
        k = random_dim(
            rng,
            min_value,
            max_value,
            template_dim=lhs_template[1] if len(lhs_template) >= 2 else None,
            alignment=16,
        )
        if rhs_format == "FRACTAL_NZ" and len(rhs_template) >= 4:
            block_h = rhs_template[-2]
            block_w = rhs_template[-1]
            k_tiles = max(1, k // block_w)
            n_template = rhs_template[-3] * block_h if len(rhs_template) >= 3 else None
            n = random_dim(
                rng,
                min_value,
                max_value,
                template_dim=n_template,
                alignment=block_h,
            )
            n_tiles = max(1, n // block_h)
            rhs = (k_tiles, n_tiles, block_h, block_w)
        else:
            n = random_dim(
                rng,
                min_value,
                max_value,
                template_dim=rhs_template[0] if len(rhs_template) >= 1 else None,
                alignment=16,
            )
            rhs = (n, k)
        lhs = (m, k)
        return [lhs, rhs], [(m, n)]

    batch_dims = tuple(
        1 if lhs_template[index] == 1 else random_dim(rng, min_value, max_value, template_dim=lhs_template[index])
        for index in range(max(lhs_rank - 2, 0))
    )
    m = random_dim(rng, min_value, max_value, template_dim=lhs_template[-2], alignment=8)
    k = random_dim(rng, min_value, max_value, template_dim=lhs_template[-1], alignment=16)
    n = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=rhs_template[-1] if rhs_template else None,
        alignment=16,
    )
    lhs = batch_dims + (m, k)
    rhs = batch_dims + (k, n)
    return [lhs, rhs], [batch_dims + (m, n)]


def generate_matmul_reduce_scatter_shapes(
    template_inputs: list[tuple[int, ...]],
    template_outputs: list[tuple[int, ...]],
    template_formats: list[str],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs, _ = generate_matmul_shapes(template_inputs[:2], template_formats[:2], rng, min_value, max_value)
    scatter_factor = 8
    if template_outputs and template_outputs[0] and len(template_outputs[0]) >= 1 and template_inputs and template_inputs[0]:
        in_m = max(1, template_inputs[0][0])
        out_m = max(1, template_outputs[0][0])
        ratio = max(1, round(in_m / out_m))
        scatter_factor = ratio
    reduced_m = max(1, math.ceil(inputs[0][-2] / scatter_factor))
    outputs = [(reduced_m, inputs[1][0] if len(inputs[1]) == 2 else inputs[1][-1]), ()]
    padding = infer_scalar_shape_count(len(template_inputs), template_inputs)[2:]
    return inputs + padding, outputs


def generate_quant_batch_matmul_shapes(
    template_inputs: list[tuple[int, ...]],
    template_outputs: list[tuple[int, ...]],
    template_formats: list[str],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    lhs_template, rhs_template = infer_scalar_shape_count(2, template_inputs)
    rhs_format = template_formats[1] if len(template_formats) > 1 else "ND"
    if rhs_format == "FRACTAL_NZ" and len(lhs_template) == 2 and len(rhs_template) >= 4:
        m = random_dim(
            rng,
            min_value,
            max_value,
            template_dim=lhs_template[0] if len(lhs_template) >= 1 else None,
            alignment=8,
        )
        k = random_dim(
            rng,
            min_value,
            max_value,
            template_dim=lhs_template[1] if len(lhs_template) >= 2 else None,
            alignment=16,
        )
        block_h = rhs_template[-2]
        block_w = rhs_template[-1]
        k_tiles = max(1, k // block_h)
        n_template = rhs_template[0] * block_w if rhs_template else None
        n = random_dim(
            rng,
            min_value,
            max_value,
            template_dim=n_template,
            alignment=block_w,
        )
        n_tiles = max(1, n // block_w)
        inputs = [(m, k), (n_tiles, k_tiles, block_h, block_w)]
        outputs = [(m, n)]
    else:
        inputs, outputs = generate_matmul_shapes(
            template_inputs[:2],
            template_formats[:2],
            rng,
            min_value,
            max_value,
        )
    output_shape = outputs[0] if outputs else ()
    output_n = output_shape[-1] if output_shape else 1
    output_m = output_shape[0] if output_shape else 1

    quant_inputs = infer_scalar_shape_count(4, template_inputs)
    input2 = (output_n,) if quant_inputs[2] else ()

    input3 = ()
    if quant_inputs[3]:
        template_dim = quant_inputs[3][0]
        if template_dim == 1:
            input3 = (1,)
        elif template_inputs and template_inputs[0] and template_dim == template_inputs[0][0]:
            input3 = (output_m,)
        else:
            input3 = (output_n,)

    return [inputs[0], inputs[1], input2, input3], outputs


def generate_add_rmsnorm_bias_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(4, template_inputs)
    seq = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=inputs[0][0] if inputs[0] else None,
    )
    hidden = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=inputs[0][-1] if inputs[0] else None,
        alignment=16,
    )
    norm = (seq, hidden)
    weights = (hidden,)
    bias = weights if inputs[3] else ()
    return [norm, norm, weights, bias], [norm, (seq, 1), norm]


def generate_ascend_quant_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(3, template_inputs)
    seq = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=inputs[0][0] if inputs[0] else None,
    )
    hidden = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=inputs[0][-1] if inputs[0] else None,
        alignment=16,
    )
    main = (seq, hidden)
    scale = (hidden,)
    return [main, scale, scale], [main]


def generate_dynamic_quant_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    source = template_inputs[0] if template_inputs else ()
    seq = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=source[0] if source else None,
    )
    hidden = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=source[-1] if source else None,
        alignment=16,
    )
    main = (seq, hidden)
    return [main], [main, (seq,)]


def generate_moe_gating_topk_shapes(
    template_inputs: list[tuple[int, ...]],
    template_outputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(2, template_inputs)
    outputs = infer_scalar_shape_count(3, template_outputs)
    tokens = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=inputs[0][0] if inputs[0] else None,
    )
    experts = random_dim(
        rng,
        2,
        min(max_value, 512),
        template_dim=inputs[0][-1] if inputs[0] else None,
        alignment=8,
    )
    topk = outputs[0][-1] if outputs[0] and len(outputs[0]) >= 2 else 8
    logits = (tokens, experts)
    scores = (experts,)
    return [logits, scores], [(tokens, topk), (tokens, topk), logits]


def generate_dispatch_ffn_combine_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(7, template_inputs)
    tokens = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=inputs[0][0] if inputs[0] else None,
    )
    hidden = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=inputs[0][-1] if inputs[0] else None,
        alignment=16,
    )
    experts = random_dim(
        rng,
        2,
        min(max_value, 256),
        template_dim=inputs[1][0] if inputs[1] else None,
        alignment=8,
    )
    inter = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=inputs[1][-1] if inputs[1] else None,
        alignment=16,
    )
    routed = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=inputs[2][1] if len(inputs[2]) >= 2 else None,
        alignment=16,
    )
    topk = inputs[3][-1] if inputs[3] else 8
    token_hidden = (tokens, hidden)
    route = (tokens, topk)
    return [
        token_hidden,
        (experts, hidden, inter),
        (experts, routed, hidden),
        route,
        (experts * inter,),
        (experts * hidden,),
        route,
    ], [token_hidden, (experts,)]


def generate_interleave_rope_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(3, template_inputs)
    seq = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=inputs[0][0] if inputs[0] else None,
        alignment=8,
    )
    heads = random_dim(
        rng,
        1,
        min(max_value, 64),
        template_dim=inputs[0][1] if len(inputs[0]) >= 2 else None,
    )
    dim = 64
    rope = (seq, 1, 1, dim)
    main = (seq, heads, 1, dim)
    return [main, rope, rope], [main]


def generate_kv_rmsnorm_rope_cache_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(12, template_inputs)
    tokens = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=inputs[0][0] if inputs[0] else None,
        alignment=8,
    )
    rope_dim = 64
    allowed_hidden_dims = (192, 512)
    hidden_template = inputs[1][0] if inputs[1] else None
    if hidden_template:
        hidden = min(allowed_hidden_dims, key=lambda candidate: abs(candidate - hidden_template))
    else:
        hidden = rng.choice(allowed_hidden_dims)
    slots = inputs[5][0] if inputs[5] else 892
    block = inputs[5][1] if len(inputs[5]) >= 2 else 128
    cache_k = (slots, block, 1, rope_dim)
    cache_v = (slots, block, 1, hidden)
    token_k = (tokens, 1, 1, rope_dim)
    token_v = (tokens, 1, 1, hidden)
    combined = (tokens, 1, 1, hidden + rope_dim)
    generated_inputs = [combined, (hidden,), token_k, token_k, (tokens,), cache_k, cache_v]
    generated_inputs.extend([()] * max(0, len(inputs) - len(generated_inputs)))
    return generated_inputs, [cache_k, cache_v, token_k, token_v]


def generate_slice_shapes(
    template_inputs: list[tuple[int, ...]],
    template_outputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(3, template_inputs)
    if inputs[0]:
        source = list(inputs[0])
        for index, dim in enumerate(source):
            alignment = 16 if index == len(source) - 1 else None
            source[index] = 1 if dim == 1 else random_dim(
                rng,
                min_value,
                max_value,
                template_dim=dim,
                alignment=alignment,
            )
        output = list(source)
        if template_outputs and template_outputs[0]:
            template_last = template_outputs[0][-1]
            input_last = max(1, inputs[0][-1])
            ratio = max(1 / input_last, min(1.0, template_last / input_last))
            output[-1] = max(1, int(round(source[-1] * ratio)))
            if source[-1] > 1:
                output[-1] = min(source[-1], output[-1])
        return [tuple(source), (len(source),), (len(source),)], [tuple(output)]
    return [(), (), ()], [()]


def generate_broadcast_to_shapes(
    template_inputs: list[tuple[int, ...]],
    template_outputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    output_template = template_outputs[0] if template_outputs else ()
    source_template = template_inputs[0] if template_inputs else ()
    if not output_template:
        input0 = mutate_shape(source_template, rng, min_value, max_value)
        return [input0, (len(input0),)], [input0]

    output = list(output_template)
    for index, dim in enumerate(output):
        alignment = 16 if index == len(output) - 1 else None
        output[index] = 1 if dim == 1 else random_dim(
            rng,
            min_value,
            max_value,
            template_dim=dim,
            alignment=alignment,
        )
    source = list(output)
    if len(source) >= 2:
        source[1] = 1
    elif source:
        source[-1] = 1
    return [tuple(source), (len(output),)], [tuple(output)]


def generate_transpose_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    source = template_inputs[0] if template_inputs else ()
    if len(source) < 2:
        input0 = mutate_shape(source, rng, min_value, max_value)
        return [input0, (len(input0),)], [input0]

    dims = list(source)
    for index, dim in enumerate(dims):
        alignment = 16 if index == len(dims) - 1 else None
        dims[index] = 1 if dim == 1 else random_dim(
            rng,
            min_value,
            max_value,
            template_dim=dim,
            alignment=alignment,
        )
    output = list(dims)
    output[0], output[1] = output[1], output[0]
    return [tuple(dims), (len(dims),)], [tuple(output)]


def generate_as_strided_shapes(
    template_outputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    output_template = template_outputs[0] if template_outputs else (16, 4, 128)
    output = list(output_template)
    for index, dim in enumerate(output):
        alignment = 16 if index == len(output) - 1 else None
        output[index] = 1 if dim == 1 else random_dim(
            rng,
            min_value,
            max_value,
            template_dim=dim,
            alignment=alignment,
        )
    flat = max(1, math.ceil(math.prod(output) * 1.5))
    return [(flat,), (len(output),), (len(output),), (1,)], [tuple(output)]


def generate_transdata_shapes(
    template_outputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    output_template = template_outputs[0] if template_outputs else (16, 128, 512)
    batch = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=output_template[0] if len(output_template) >= 1 else None,
        alignment=16,
    )
    m = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=output_template[1] if len(output_template) >= 2 else None,
        alignment=16,
    )
    n = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=output_template[2] if len(output_template) >= 3 else None,
        alignment=16,
    )
    return [(batch, n // 16, m // 16, 16, 16)], [(batch, m, n)]


def generate_transpose_batch_matmul_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(2, template_inputs)
    batch = random_dim(
        rng,
        1,
        min(max_value, 64),
        template_dim=inputs[0][0] if inputs[0] else None,
    )
    m = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=inputs[0][1] if len(inputs[0]) >= 2 else None,
        alignment=8,
    )
    k = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=inputs[0][2] if len(inputs[0]) >= 3 else None,
        alignment=16,
    )
    n = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=inputs[1][2] if len(inputs[1]) >= 3 else None,
        alignment=16,
    )
    return [(batch, m, k), (batch, k, n)], [(m, batch, n)]


def generate_ringmla_prefill_shapes(
    template_inputs: list[tuple[int, ...]],
    template_outputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(len(template_inputs), template_inputs)
    outputs = infer_scalar_shape_count(len(template_outputs), template_outputs)
    seq = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=inputs[0][0] if inputs and inputs[0] else None,
        alignment=8,
    )
    heads = random_dim(
        rng,
        1,
        min(max_value, 64),
        template_dim=inputs[0][1] if inputs and len(inputs[0]) >= 2 else None,
    )
    dim_main = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=inputs[0][-1] if inputs and inputs[0] else None,
        alignment=16,
    )
    dim_rope = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=inputs[1][-1] if len(inputs) > 1 and inputs[1] else None,
        alignment=16,
    )
    generated_inputs = [
        (seq, heads, dim_main),
        (seq, heads, dim_rope),
        (seq, heads, dim_main),
        (seq, heads, dim_rope),
        (seq, heads, dim_main),
    ]
    if len(inputs) > 5:
        generated_inputs.append((512, 512) if inputs[5] else ())
    generated_inputs.extend([()] * max(0, len(inputs) - len(generated_inputs)))

    generated_outputs = [(seq, heads, dim_main)] if outputs else []
    if len(outputs) > 1:
        generated_outputs.append((heads, seq) if outputs[1] else ())
    return generated_inputs, generated_outputs


def generate_add_rmsnorm_dynamic_quant_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(4, template_inputs)
    tokens = random_dim(rng, min_value, max_value, template_dim=inputs[0][0] if inputs[0] else None)
    hidden = random_dim(
        rng, min_value, max_value, template_dim=inputs[0][-1] if inputs[0] else None, alignment=16
    )
    norm = (tokens, hidden)
    weight = (hidden,)
    return [norm, norm, weight, weight], [norm, norm, norm, (tokens,), (tokens,)]


def generate_apply_topk_top_p_custom_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    logits, outputs = generate_apply_topk_top_p_shapes(template_inputs, rng, min_value, max_value)
    batch = logits[0][0] if logits and logits[0] else 1
    return [logits[0], logits[0], (batch,), (batch,)], outputs


def generate_atb_rope_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(5, template_inputs)
    tokens = random_dim(rng, min_value, max_value, template_dim=inputs[0][0] if inputs[0] else None, alignment=8)
    hidden = random_dim(rng, min_value, max_value, template_dim=inputs[0][-1] if inputs[0] else None, alignment=16)
    rope_dim = random_dim(rng, min_value, max_value, template_dim=inputs[1][-1] if inputs[1] else None, alignment=16)
    side = (tokens, rope_dim)
    main = (tokens, hidden)
    return [main, side, side, side, (1,)], [main, side]


def generate_concat_shapes(
    template_inputs: list[tuple[int, ...]],
    template_outputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = [shape for shape in template_inputs if shape]
    if not inputs:
        return template_inputs, template_outputs
    rank = len(inputs[0])
    output_template = template_outputs[0] if template_outputs else ()
    axis = rank - 1
    if output_template and len(output_template) == rank:
        for dim_index in range(rank):
            dim_sum = sum(shape[dim_index] for shape in inputs if len(shape) > dim_index)
            if output_template[dim_index] == dim_sum:
                axis = dim_index
                break
    base = []
    for dim_index in range(rank):
        template_dim = inputs[0][dim_index]
        alignment = 16 if dim_index == rank - 1 else None
        base.append(1 if template_dim == 1 else random_dim(rng, min_value, max_value, template_dim=template_dim, alignment=alignment))
    generated_inputs = []
    concat_total = 0
    for shape in inputs:
        dims = list(base)
        template_dim = shape[axis]
        alignment = 16 if axis == rank - 1 else None
        dims[axis] = 1 if template_dim == 1 else random_dim(rng, min_value, max_value, template_dim=template_dim, alignment=alignment)
        concat_total += dims[axis]
        generated_inputs.append(tuple(dims))
    output = list(base)
    output[axis] = concat_total
    return generated_inputs, [tuple(output)]


def generate_same_shape_with_scalar_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    input0 = mutate_shape(template_inputs[0], rng, min_value, max_value) if template_inputs else ()
    scalar = (1,) if len(template_inputs) > 1 and template_inputs[1] else ()
    return [input0, scalar], [input0]


def generate_gather_elements_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(2, template_inputs)
    source = list(inputs[0]) if inputs[0] else [random_dim(rng, min_value, max_value), random_dim(rng, min_value, max_value)]
    source = [1 if dim == 1 else random_dim(rng, min_value, max_value, template_dim=dim, alignment=16 if i == len(source) - 1 else None) for i, dim in enumerate(source)]
    index = list(inputs[1]) if inputs[1] else [source[0], 1]
    for i, dim in enumerate(index):
        if i < len(source):
            index[i] = 1 if dim == 1 else min(source[i], random_dim(rng, 1, source[i], template_dim=dim))
    return [tuple(source), tuple(index)], [tuple(index)]


def generate_grouped_matmul_shapes(
    template_inputs: list[tuple[int, ...]],
    template_outputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(9, template_inputs)
    tokens = random_dim(rng, min_value, max_value, template_dim=inputs[0][0] if inputs[0] else None)
    k = random_dim(rng, min_value, max_value, template_dim=inputs[0][-1] if inputs[0] else None, alignment=16)
    experts = random_dim(rng, 2, min(max_value, 256), template_dim=inputs[1][0] if inputs[1] else None, alignment=8)
    n = random_dim(rng, min_value, max_value, template_dim=template_outputs[0][-1] if template_outputs and template_outputs[0] else None, alignment=16)
    expert_rows = max(1, n // 16)
    generated = [(tokens, k), (experts, k, expert_rows, 16), (), (experts, n), (), (), (), (experts,), (tokens,)]
    return generated[: len(inputs)], [(tokens, n)]


def generate_grouped_matmul_swiglu_quant_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(5, template_inputs)
    tokens = random_dim(rng, min_value, max_value, template_dim=inputs[0][0] if inputs[0] else None)
    hidden = random_dim(rng, min_value, max_value, template_dim=inputs[0][-1] if inputs[0] else None, alignment=16)
    experts = random_dim(rng, 2, min(max_value, 256), template_dim=inputs[1][0] if inputs[1] else None, alignment=8)
    inter = random_dim(rng, min_value, max_value, template_dim=inputs[2][-1] if inputs[2] else None, alignment=16)
    out_hidden = max(16, inter // 2)
    return [
        (tokens, hidden),
        (experts, max(1, out_hidden // 16), max(1, hidden // 16), 16, 32),
        (experts, inter),
        (tokens,),
        (experts,),
    ], [(tokens, out_hidden), (tokens,)]


def generate_index_put_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(5, template_inputs)
    source = mutate_shape(inputs[0], rng, min_value, max_value) if inputs[0] else (random_dim(rng, min_value, max_value),)
    index_rows = random_dim(rng, 1, max_value, template_dim=inputs[1][0] if inputs[1] else None)
    index_rank = len(source) if len(source) > 1 else 1
    indices = (index_rows, index_rank) if len(source) > 1 else (index_rows,)
    return [source, indices, (1,), (1,), (index_rows,)], [source]


def generate_moe_dispatch_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(6, template_inputs)
    tokens = random_dim(rng, min_value, max_value, template_dim=inputs[0][0] if inputs[0] else None)
    hidden = random_dim(rng, min_value, max_value, template_dim=inputs[0][-1] if inputs[0] else None, alignment=16)
    groups = random_dim(rng, 1, min(max_value, 32), template_dim=inputs[0][0] if inputs[0] else None)
    topk = inputs[1][-1] if inputs[1] else 8
    capacity = tokens * topk
    experts = random_dim(rng, 2, min(max_value, 64), template_dim=inputs[4][0] if inputs[4] else None)
    return [
        (groups, hidden),
        (groups, topk),
        (),
        (groups,),
        (groups, topk),
        (),
    ], [(capacity, hidden), (capacity,), (capacity * hidden,), (experts,), (max(1, capacity // max(experts, 1)),), (1,), (capacity,)]


def generate_moe_combine_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(17, template_inputs)
    capacity = random_dim(rng, min_value, max_value, template_dim=inputs[0][0] if inputs[0] else None)
    hidden = random_dim(rng, min_value, max_value, template_dim=inputs[0][-1] if inputs[0] else None, alignment=16)
    groups = random_dim(rng, 1, min(max_value, 32), template_dim=inputs[1][0] if inputs[1] else None)
    topk = inputs[1][-1] if inputs[1] else 8
    generated = [
        (capacity, hidden),
        (groups, topk),
        (capacity * hidden,),
        (max(1, capacity // max(groups, 1)),),
        (groups, topk),
        (1,),
        (groups,),
        (),
        (),
        (),
        (capacity,),
    ]
    generated.extend([()] * max(0, len(inputs) - len(generated)))
    return generated, [(groups, hidden)]


def generate_moe_token_permute_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(2, template_inputs)
    tokens = random_dim(rng, min_value, max_value, template_dim=inputs[0][0] if inputs[0] else None)
    hidden = random_dim(rng, min_value, max_value, template_dim=inputs[0][-1] if inputs[0] else None, alignment=16)
    topk = inputs[1][-1] if inputs[1] and len(inputs[1]) >= 2 else 1
    return [(tokens, hidden), (tokens, topk) if inputs[1] and len(inputs[1]) >= 2 else (tokens,)], [(tokens * topk, hidden), (tokens * topk,)]


def generate_moe_token_unpermute_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(3, template_inputs)
    routed = random_dim(rng, min_value, max_value, template_dim=inputs[0][0] if inputs[0] else None)
    hidden = random_dim(rng, min_value, max_value, template_dim=inputs[0][-1] if inputs[0] else None, alignment=16)
    topk = inputs[2][-1] if inputs[2] and len(inputs[2]) >= 2 else 1
    base_tokens = max(1, routed // max(topk, 1))
    route = (base_tokens, topk) if inputs[2] else ()
    return [(routed, hidden), (routed,), route], [(base_tokens, hidden)]


def generate_pad_shapes(
    template_inputs: list[tuple[int, ...]],
    template_outputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    source = list(template_inputs[0]) if template_inputs and template_inputs[0] else [random_dim(rng, min_value, max_value)]
    source = [1 if dim == 1 else random_dim(rng, min_value, max_value, template_dim=dim, alignment=16 if i == len(source) - 1 else None) for i, dim in enumerate(source)]
    output = list(source)
    if template_outputs and template_outputs[0]:
        output[0] = max(source[0], random_dim(rng, source[0], max_value, template_dim=template_outputs[0][0]))
    else:
        output[0] = source[0] + 4
    pads = (len(source) * 2,)
    return [tuple(source), pads, ()], [tuple(output)]


def generate_paged_cache_load_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(7, template_inputs)
    slots = inputs[0][0] if inputs[0] else 892
    block = inputs[0][1] if len(inputs[0]) >= 2 else 128
    hidden = random_dim(rng, min_value, max_value, template_dim=inputs[0][-1] if inputs[0] else None, alignment=16)
    rope_dim = random_dim(rng, min_value, max_value, template_dim=inputs[1][-1] if inputs[1] else None, alignment=16)
    batch = random_dim(rng, 1, min(max_value, 8), template_dim=inputs[2][0] if inputs[2] else None)
    tokens = random_dim(rng, min_value, max_value, template_dim=inputs[4][0] if inputs[4] else None, alignment=8)
    cache_v = (slots, block, 1, hidden)
    cache_k = (slots, block, 1, rope_dim)
    slots_view = (batch, hidden)
    return [cache_v, cache_k, slots_view, (batch,), (tokens, 1, hidden), (tokens, 1, rope_dim), (batch,)], [(tokens, 1, hidden), (tokens, 1, rope_dim)]


def generate_range_shapes(
    template_outputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    out_dim = random_dim(rng, min_value, max_value, template_dim=template_outputs[0][0] if template_outputs and template_outputs[0] else None)
    return [(), (), ()], [(out_dim,)]


def generate_repeat_interleave_shapes(
    template_inputs: list[tuple[int, ...]],
    template_outputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    base = random_dim(rng, 1, min(max_value, 1024), template_dim=template_inputs[0][0] if template_inputs and template_inputs[0] else None)
    out_dim = random_dim(rng, 1, max_value, template_dim=template_outputs[0][0] if template_outputs and template_outputs[0] else None)
    return [(base,), (base,)], [(out_dim,)]


def generate_scatter_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    source = mutate_shape(template_inputs[0], rng, min_value, max_value) if template_inputs and template_inputs[0] else (random_dim(rng, min_value, max_value),)
    index_dim = random_dim(rng, 1, source[0] if source else max_value, template_dim=template_inputs[1][0] if len(template_inputs) > 1 and template_inputs[1] else None)
    updates = (index_dim,)
    return [source, updates, updates], [source]


def generate_select_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(3, template_inputs)
    data = mutate_shape(inputs[0], rng, min_value, max_value) if inputs[0] else (random_dim(rng, min_value, max_value),)
    cond = data if inputs[1] else ()
    other = data if inputs[2] else ()
    return [data, cond, other], [data]


def generate_tile_shapes(
    template_inputs: list[tuple[int, ...]],
    template_outputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    source = list(template_inputs[0]) if template_inputs and template_inputs[0] else [1, 1]
    source = [1 if dim == 1 else random_dim(rng, min_value, max_value, template_dim=dim, alignment=16 if i == len(source) - 1 else None) for i, dim in enumerate(source)]
    output = list(source)
    if template_outputs and template_outputs[0]:
        for i in range(min(len(output), len(template_outputs[0]))):
            output[i] = max(output[i], template_outputs[0][i])
    elif output:
        output[-1] *= 2
    return [tuple(source), (len(source),)], [tuple(output)]


def generate_triton_rope_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(3, template_inputs)
    tokens = random_dim(rng, min_value, max_value, template_dim=inputs[0][0] if inputs[0] else None, alignment=8)
    heads = random_dim(rng, 1, min(max_value, 64), template_dim=inputs[0][1] if inputs[0] else None)
    dim = random_dim(rng, min_value, max_value, template_dim=inputs[0][-1] if inputs[0] else None, alignment=16)
    main = (tokens, heads, dim)
    side = (tokens, 1, dim)
    table = (random_dim(rng, tokens, max_value, template_dim=inputs[2][0] if inputs[2] else None, alignment=16), dim)
    return [main, side, table], [main, side]


def generate_expand_shapes(
    template_outputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    out_dim = random_dim(rng, min_value, max_value, template_dim=template_outputs[0][0] if template_outputs and template_outputs[0] else None)
    return [(1,), (1,)], [(out_dim,)]


def generate_swiglu_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    template = template_inputs[0] if template_inputs else (1, 2048)
    prefix = tuple(
        1 if dim == 1 else random_dim(rng, min_value, max_value, template_dim=dim)
        for dim in template[:-1]
    )
    hidden = random_dim(rng, min_value, max_value, template_dim=max(2, template[-1] // 2), alignment=16)
    return [prefix + (hidden * 2,)], [prefix + (hidden,)]


def generate_rope_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    base = infer_scalar_shape_count(4, template_inputs)
    batch = 1 if len(base[0]) < 1 or base[0][0] == 1 else random_dim(rng, min_value, max_value, template_dim=base[0][0])
    seq = random_dim(rng, min_value, max_value, template_dim=base[0][1] if len(base[0]) >= 2 else None, alignment=8)
    heads = random_dim(rng, 1, min(max_value, 64), template_dim=base[0][2] if len(base[0]) >= 3 else None)
    dim = random_dim(rng, min_value, max_value, template_dim=base[0][3] if len(base[0]) >= 4 else None, alignment=16)
    input0 = (batch, seq, heads, dim)
    side = (batch, seq, 1, dim)
    return [input0, side, side, side], [input0, side]


def generate_rmsnorm_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(2, template_inputs)
    seq = random_dim(rng, min_value, max_value, template_dim=inputs[0][0] if inputs[0] else None)
    hidden = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=inputs[0][-1] if inputs[0] else None,
        alignment=16,
    )
    return [(seq, hidden), (hidden,)], [(seq, hidden), (seq, 1)]


def generate_add_rmsnorm_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    seq = random_dim(rng, min_value, max_value, template_dim=template_inputs[0][0] if template_inputs and template_inputs[0] else None)
    hidden = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=template_inputs[0][-1] if template_inputs and template_inputs[0] else None,
        alignment=16,
    )
    norm = (seq, hidden)
    return [norm, norm, (hidden,)], [norm, (seq, 1), norm]


def generate_gather_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(3, template_inputs)
    vocab = random_dim(rng, min_value, max_value, template_dim=inputs[0][0] if inputs[0] else None, alignment=16)
    hidden = random_dim(rng, min_value, max_value, template_dim=inputs[0][-1] if inputs[0] else None, alignment=16)
    index_count = random_dim(rng, min_value, max_value, template_dim=inputs[1][0] if inputs[1] else None)
    return [(vocab, hidden), (index_count,), (1,)], [(index_count, hidden)]


def generate_gather_v3_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(3, template_inputs)
    source = inputs[0]
    index_count = random_dim(rng, min_value, max_value, template_dim=inputs[1][0] if inputs[1] else None, alignment=8)
    trailing = tuple(
        1 if dim == 1 else random_dim(rng, min_value, max_value, template_dim=dim, alignment=16 if idx == len(source[1:]) - 1 else None)
        for idx, dim in enumerate(source[1:])
    )
    source_shape = (
        random_dim(rng, min_value, max_value, template_dim=source[0] if source else None, alignment=16),
        *trailing,
    )
    return [source_shape, (index_count,), (1,)], [(index_count, *trailing)]


def generate_index_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    inputs = infer_scalar_shape_count(4, template_inputs)
    seq = random_dim(rng, min_value, max_value, template_dim=inputs[0][0] if inputs[0] else None, alignment=8)
    hidden = random_dim(rng, min_value, max_value, template_dim=inputs[0][-1] if inputs[0] else None, alignment=16)
    out_rows = random_dim(rng, 1, max_value, template_dim=inputs[2][0] if inputs[2] else None)
    return [(seq, hidden), (1,), (out_rows,), (1,)], [(out_rows, hidden)]


def generate_argmax_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    shapes = infer_scalar_shape_count(2, template_inputs)
    input0 = mutate_shape(shapes[0], rng, min_value, max_value)
    input1 = shapes[1]
    output = (1,) if input1 == () else input1
    return [input0, input1], [output]


def generate_sort_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    input0 = mutate_shape(template_inputs[0], rng, min_value, max_value) if template_inputs else ()
    return [input0], [input0, input0]


def generate_split_qkv_rmsnorm_rope_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    seq = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=template_inputs[0][0] if template_inputs and template_inputs[0] else None,
        alignment=8,
    )
    max_position_embeddings = random_dim(
        rng,
        max(seq, min_value),
        max_value,
        template_dim=template_inputs[1][0] if len(template_inputs) > 1 and template_inputs[1] else None,
        alignment=16,
    )
    rope_dim = SPLIT_QKV_ROPE_HEAD_DIM
    hidden_template = (
        template_inputs[0][-1] - 2 * template_inputs[1][-1]
        if template_inputs and template_inputs[0] and len(template_inputs) > 1 and template_inputs[1]
        else 1024
    )
    hidden_lower = max(rope_dim, min_value)
    hidden_upper = max(hidden_lower, max_value)
    if hidden_template > 0:
        approx_multiplier = max(1, round(hidden_template / rope_dim))
        lower_multiplier = max(1, approx_multiplier // 2)
        upper_multiplier = max(lower_multiplier, approx_multiplier * 2)
        min_multiplier = max(1, math.ceil(hidden_lower / rope_dim))
        max_multiplier = max(min_multiplier, hidden_upper // rope_dim)
        lower_multiplier = max(lower_multiplier, min_multiplier)
        upper_multiplier = min(upper_multiplier, max_multiplier)
        if lower_multiplier <= upper_multiplier:
            hidden = rope_dim * rng.randint(lower_multiplier, upper_multiplier)
        else:
            hidden = rope_dim * min_multiplier
    else:
        hidden = rope_dim * max(1, math.ceil(hidden_lower / rope_dim))
    combined = hidden + rope_dim * 2
    return [(seq, combined), (max_position_embeddings, rope_dim), (seq,)], [(seq, hidden), (seq, rope_dim)]


def generate_reshape_and_cache_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    seq = random_dim(rng, min_value, max_value, template_dim=template_inputs[0][0] if template_inputs and template_inputs[0] else None, alignment=8)
    kv_heads = 1 if not template_inputs or not template_inputs[0] or template_inputs[0][1] == 1 else random_dim(rng, 1, min(max_value, 32), template_dim=template_inputs[0][1])
    head_dim = random_dim(rng, min_value, max_value, template_dim=template_inputs[0][-1] if template_inputs and template_inputs[0] else None, alignment=16)
    block_size = template_inputs[2][1] if len(template_inputs) > 2 and len(template_inputs[2]) > 1 else 128
    slots = random_dim(rng, min_value, max_value, template_dim=template_inputs[2][0] if len(template_inputs) > 2 and template_inputs[2] else None, alignment=8)
    cache = (slots, block_size, kv_heads, head_dim)
    token_ids = (seq,)
    token = (seq, kv_heads, head_dim)
    return [token, token, cache, cache, token_ids], [cache, cache]


def generate_fused_attention_shapes(
    template_inputs: list[tuple[int, ...]],
    template_outputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    input_shapes = infer_scalar_shape_count(len(template_inputs), template_inputs)
    output_shapes = infer_scalar_shape_count(len(template_outputs), template_outputs)

    query_template = input_shapes[0] if input_shapes else ()
    is_mla_layout = len(query_template) == 4 and len(input_shapes) > 24 and bool(input_shapes[24])

    if is_mla_layout:
        batch = random_dim(
            rng,
            1,
            min(max_value, 64),
            template_dim=query_template[0] if len(query_template) >= 1 else None,
        )
        heads = random_dim(
            rng,
            1,
            min(max_value, 64),
            template_dim=query_template[1] if len(query_template) >= 2 else None,
        )
        seq = random_dim(
            rng,
            min_value,
            max_value,
            template_dim=query_template[2] if len(query_template) >= 3 else None,
            alignment=8,
        )
        dim = random_dim(
            rng,
            min_value,
            max_value,
            template_dim=query_template[3] if len(query_template) >= 4 else None,
            alignment=16,
        )
        kv_seq = random_dim(
            rng,
            max(seq, min_value),
            max_value,
            template_dim=input_shapes[1][0] if len(input_shapes) > 1 and input_shapes[1] else None,
            alignment=8,
        )
        kv_heads = 1
        kv_dim = dim
        if len(input_shapes) > 14 and input_shapes[14]:
            max_blocks_per_seq = random_dim(
                rng,
                1,
                max_value,
                template_dim=input_shapes[14][1] if len(input_shapes[14]) >= 2 else None,
                alignment=16,
            )
            input_shapes[14] = (batch, max_blocks_per_seq)
        input_shapes[0] = (batch, heads, seq, dim)
        input_shapes[1] = (kv_seq, kv_heads, 128, kv_dim)
        input_shapes[2] = (kv_seq, kv_heads, 128, kv_dim)
        if len(input_shapes) > 5 and input_shapes[5]:
            input_shapes[5] = (batch,)
        if len(input_shapes) > 6 and input_shapes[6]:
            input_shapes[6] = (batch,)
        if len(input_shapes) > 24 and input_shapes[24]:
            rope_dim = 64
            input_shapes[24] = (batch, heads, 1, rope_dim)
        if len(input_shapes) > 25 and input_shapes[25]:
            rope_dim = input_shapes[24][-1] if len(input_shapes) > 24 and input_shapes[24] else kv_dim
            input_shapes[25] = (kv_seq, kv_heads, 128, rope_dim)
        if len(input_shapes) > 4 and input_shapes[4]:
            input_shapes[4] = (2048, 2048)
        if output_shapes:
            output_shapes[0] = (heads, batch, 1, dim)
    else:
        seq = random_dim(
            rng,
            min_value,
            max_value,
            template_dim=query_template[0] if query_template else None,
            alignment=8,
        )
        heads = random_dim(
            rng,
            1,
            min(max_value, 64),
            template_dim=query_template[1] if len(query_template) > 1 else None,
        )
        kv_heads = (
            1
            if len(input_shapes) < 2 or not input_shapes[1] or input_shapes[1][1] == 1
            else random_dim(
                rng,
                1,
                min(max_value, heads),
                template_dim=input_shapes[1][1],
            )
        )
        dim = random_dim(
            rng,
            min_value,
            max_value,
            template_dim=query_template[-1] if query_template else None,
            alignment=16,
        )
        has_block_table = len(input_shapes) > 14 and bool(input_shapes[14])
        batch_size = None
        if has_block_table:
            batch_size = random_dim(
                rng,
                1,
                min(max_value, 64, max(1, seq)),
                template_dim=input_shapes[14][0] if input_shapes[14] else None,
            )
            max_blocks_per_seq = random_dim(
                rng,
                1,
                max_value,
                template_dim=input_shapes[14][1] if len(input_shapes[14]) >= 2 else None,
                alignment=16,
            )
            total_blocks = random_dim(
                rng,
                max(batch_size, 1),
                max_value,
                template_dim=input_shapes[1][0] if len(input_shapes) > 1 and input_shapes[1] else None,
                alignment=8,
            )
            block_size = random_dim(
                rng,
                1,
                max_value,
                template_dim=input_shapes[1][1] if len(input_shapes) > 1 and len(input_shapes[1]) >= 2 else None,
                alignment=16,
            )
        else:
            batch_size = random_dim(
                rng,
                1,
                min(max_value, 64, max(1, seq)),
                template_dim=input_shapes[5][0] if len(input_shapes) > 5 and input_shapes[5] else (
                    input_shapes[6][0] if len(input_shapes) > 6 and input_shapes[6] else 1
                ),
            )
        input_shapes[0] = (seq, heads, dim)
        if has_block_table:
            input_shapes[1] = (total_blocks, block_size, dim)
            input_shapes[2] = (total_blocks, block_size, dim)
            input_shapes[14] = (batch_size, max_blocks_per_seq)
        else:
            kv_seq = random_dim(
                rng,
                min_value,
                max_value,
                template_dim=input_shapes[1][0] if len(input_shapes) > 1 and input_shapes[1] else None,
                alignment=8,
            )
            input_shapes[1] = (kv_seq, kv_heads, dim)
            input_shapes[2] = (kv_seq, kv_heads, dim)
        if len(input_shapes) > 4:
            input_shapes[4] = (2048, 2048) if input_shapes[4] else ()
        for scalar_index in (5, 6):
            if scalar_index < len(input_shapes) and input_shapes[scalar_index]:
                input_shapes[scalar_index] = (batch_size,)
        if output_shapes:
            output_shapes[0] = (seq, heads, dim)
    return input_shapes, output_shapes


def generate_apply_topk_top_p_shapes(
    template_inputs: list[tuple[int, ...]],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    batch = 1 if not template_inputs or not template_inputs[0] or template_inputs[0][0] == 1 else random_dim(rng, min_value, max_value, template_dim=template_inputs[0][0])
    vocab = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=template_inputs[0][-1] if template_inputs and template_inputs[0] else None,
        alignment=16,
    )
    logits = (batch, vocab)
    return [logits, logits, (1,), (1,)], [logits]


def generate_shapes_for_kernel(
    kernel_type: str,
    template_inputs: list[tuple[int, ...]],
    template_outputs: list[tuple[int, ...]],
    template_formats: list[str],
    rng: random.Random,
    min_value: int,
    max_value: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    if kernel_type in {"split_qkv_rmsnorm_rope_kernel", "split_qkv_rmsnorm_rope_kernel_0"}:
        return generate_split_qkv_rmsnorm_rope_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type in {"ReshapeAndCacheNdKernel", "reshape_and_cache_200000000"}:
        return generate_reshape_and_cache_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "AddRmsNormDynamicQuant":
        return generate_add_rmsnorm_dynamic_quant_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "ApplyRotaryPosEmb":
        return generate_rope_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type in {"AtbRopeKernel", "_triton_rope"}:
        return generate_atb_rope_shapes(template_inputs, rng, min_value, max_value) if kernel_type == "AtbRopeKernel" else generate_triton_rope_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "SwiGlu":
        return generate_swiglu_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "AddRmsNorm":
        return generate_add_rmsnorm_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "AddRmsNormBias":
        return generate_add_rmsnorm_bias_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "RmsNorm":
        return generate_rmsnorm_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "AscendQuantV2":
        return generate_ascend_quant_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "DynamicQuant":
        return generate_dynamic_quant_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type in {"GatherV2"}:
        return generate_gather_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "GatherV2AiCore":
        return generate_gather_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "GatherElementsV2":
        return generate_gather_elements_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "GatherV3":
        return generate_gather_v3_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "Index":
        return generate_index_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type in {"IndexFill", "ClipByValueV2"}:
        return generate_generic_shapes(template_inputs, template_outputs, rng, min_value, max_value)
    if kernel_type == "IndexPutV2":
        return generate_index_put_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type in {"Cumsum", "LinearIndex", "ReduceSum"}:
        return generate_same_shape_with_scalar_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "BroadcastTo":
        return generate_broadcast_to_shapes(template_inputs, template_outputs, rng, min_value, max_value)
    if kernel_type == "ArgMaxV2":
        return generate_argmax_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type in {"Slice", "SliceAiCore"}:
        return generate_slice_shapes(template_inputs, template_outputs, rng, min_value, max_value)
    if kernel_type == "Sort":
        return generate_sort_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "Transpose":
        return generate_transpose_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "AsStrided":
        return generate_as_strided_shapes(template_outputs, rng, min_value, max_value)
    if kernel_type == "TransData":
        return generate_transdata_shapes(template_outputs, rng, min_value, max_value)
    if kernel_type == "TransposeBatchMatMul":
        return generate_transpose_batch_matmul_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type in {"ApplyTopKTopPWithSorted", "ApplyTopKTopPCustom"}:
        if kernel_type == "ApplyTopKTopPCustom":
            return generate_apply_topk_top_p_custom_shapes(template_inputs, rng, min_value, max_value)
        return generate_apply_topk_top_p_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "MoeGatingTopK":
        return generate_moe_gating_topk_shapes(
            template_inputs,
            template_outputs,
            rng,
            min_value,
            max_value,
        )
    if kernel_type == "DispatchFFNCombine":
        return generate_dispatch_ffn_combine_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "MoeDistributeDispatchV2":
        return generate_moe_dispatch_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "MoeDistributeCombineV2":
        return generate_moe_combine_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "MoeTokenPermute":
        return generate_moe_token_permute_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "MoeTokenUnpermute":
        return generate_moe_token_unpermute_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "InterleaveRope":
        return generate_interleave_rope_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "KvRmsNormRopeCache":
        return generate_kv_rmsnorm_rope_cache_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "PagedCacheLoadNdKernel":
        return generate_paged_cache_load_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "PadV3":
        return generate_pad_shapes(template_inputs, template_outputs, rng, min_value, max_value)
    if kernel_type == "ConcatD":
        return generate_concat_shapes(template_inputs, template_outputs, rng, min_value, max_value)
    if kernel_type == "Range":
        return generate_range_shapes(template_outputs, rng, min_value, max_value)
    if kernel_type == "RepeatInterleave":
        return generate_repeat_interleave_shapes(template_inputs, template_outputs, rng, min_value, max_value)
    if kernel_type == "ScatterElementsV2":
        return generate_scatter_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "SelectV2":
        return generate_select_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "Tile":
        return generate_tile_shapes(template_inputs, template_outputs, rng, min_value, max_value)
    if kernel_type == "expand_kernel":
        return generate_expand_shapes(template_outputs, rng, min_value, max_value)
    if kernel_type == "RINGMLAPrefillBF16Kernel":
        return generate_ringmla_prefill_shapes(
            template_inputs,
            template_outputs,
            rng,
            min_value,
            max_value,
        )
    if kernel_type == "FusedInferAttentionScore":
        return generate_fused_attention_shapes(template_inputs, template_outputs, rng, min_value, max_value)
    if kernel_type == "GroupedMatmul":
        return generate_grouped_matmul_shapes(template_inputs, template_outputs, rng, min_value, max_value)
    if kernel_type == "GroupedMatmulSwigluQuant":
        return generate_grouped_matmul_swiglu_quant_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "QuantBatchMatmulV3":
        return generate_quant_batch_matmul_shapes(
            template_inputs,
            template_outputs,
            template_formats,
            rng,
            min_value,
            max_value,
        )
    if kernel_type == "MatmulReduceScatterV2":
        return generate_matmul_reduce_scatter_shapes(
            template_inputs,
            template_outputs,
            template_formats,
            rng,
            min_value,
            max_value,
        )
    if kernel_type in MATMUL_KERNELS:
        return generate_matmul_shapes(template_inputs, template_formats, rng, min_value, max_value)
    if kernel_type in ELEMENTWISE_BINARY_KERNELS:
        return generate_elementwise_binary_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type in ELEMENTWISE_UNARY_KERNELS:
        return generate_elementwise_unary_shapes(template_inputs, rng, min_value, max_value)
    return generate_generic_shapes(template_inputs, template_outputs, rng, min_value, max_value)


def build_row_template(headers: list[str], source_row: dict[str, str]) -> dict[str, str]:
    row_template: dict[str, str] = {}
    for header in headers:
        if header in KEEP_COLUMNS or header == OUTPUT_SHAPES_COLUMN:
            row_template[header] = source_row.get(header, "")
        elif header == INPUT_SHAPES_COLUMN:
            row_template[header] = ""
        elif zero_fill_column(header):
            row_template[header] = ZERO_VALUE
        else:
            row_template[header] = source_row.get(header, "")
    return row_template


def generate_rows(
    writer: csv.DictWriter,
    headers: list[str],
    source_rows: list[dict[str, str]],
    kernel_type: str,
    row_count: int,
    min_value: int,
    max_value: int,
    rng: random.Random,
    *,
    file_index: int,
    total_files: int,
    csv_path: Path,
) -> int:
    if not source_rows:
        raise ValueError("CSV does not contain any data rows.")

    valid_source_rows = []
    for row in source_rows:
        input_shapes = parse_shape_text(row.get(INPUT_SHAPES_COLUMN, ""))
        output_shapes = parse_shape_text(row.get(OUTPUT_SHAPES_COLUMN, ""))
        if any(input_shapes):
            valid_source_rows.append(row)
            continue
        if kernel_type in OUTPUT_TEMPLATE_KERNELS and any(output_shapes):
            valid_source_rows.append(row)
    if not valid_source_rows:
        return 0

    appended_rows = 0
    progress_interval = max(1, row_count // 100)
    print_progress(
        file_index=file_index,
        total_files=total_files,
        csv_path=csv_path,
        row_index=0,
        total_rows=row_count,
        appended_rows=appended_rows,
    )
    for _ in range(row_count):
        source_row = rng.choice(valid_source_rows)
        template_inputs = parse_shape_text(source_row.get(INPUT_SHAPES_COLUMN, ""))
        template_outputs = parse_shape_text(source_row.get(OUTPUT_SHAPES_COLUMN, ""))
        template_formats = [
            part.strip()
            for part in str(source_row.get("Input Formats", "")).strip().strip('"').split(";")
            if part.strip()
        ]
        generated_inputs, generated_outputs = generate_shapes_for_kernel(
            kernel_type,
            template_inputs,
            template_outputs,
            template_formats,
            rng,
            min_value,
            max_value,
        )
        generated_inputs = align_shape_slot_count(template_inputs, generated_inputs)
        generated_outputs = align_shape_slot_count(template_outputs, generated_outputs)
        row = build_row_template(headers, source_row)
        row[INPUT_SHAPES_COLUMN] = build_shape_cell(generated_inputs)
        if OUTPUT_SHAPES_COLUMN in headers:
            row[OUTPUT_SHAPES_COLUMN] = build_shape_cell(generated_outputs)
        writer.writerow(row)
        appended_rows += 1
        if appended_rows == row_count or appended_rows % progress_interval == 0:
            print_progress(
                file_index=file_index,
                total_files=total_files,
                csv_path=csv_path,
                row_index=appended_rows,
                total_rows=row_count,
                appended_rows=appended_rows,
            )
    return appended_rows


def process_csv_file(
    csv_path: Path,
    row_count: int,
    min_value: int,
    max_value: int,
    rng: random.Random,
    *,
    file_index: int,
    total_files: int,
) -> int | None:
    with csv_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        headers = reader.fieldnames
        if not headers:
            raise ValueError(f"{csv_path} is missing a header row.")
        if INPUT_SHAPES_COLUMN not in headers:
            return None
        source_rows = list(reader)
        if not source_rows:
            raise ValueError(f"{csv_path} does not contain a data row.")

    temp_path = csv_path.with_name(f"{csv_path.stem}.tmp{csv_path.suffix}")
    with temp_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=headers)
        writer.writeheader()
        for source_row in source_rows:
            writer.writerow(source_row)
        appended_rows = generate_rows(
            writer=writer,
            headers=headers,
            source_rows=source_rows,
            kernel_type=csv_path.stem,
            row_count=row_count,
            min_value=min_value,
            max_value=max_value,
            rng=rng,
            file_index=file_index,
            total_files=total_files,
            csv_path=csv_path,
        )

    if csv_path.exists():
        os.chmod(csv_path, stat.S_IWRITE | stat.S_IREAD)
    os.replace(temp_path, csv_path)
    return appended_rows


def iter_csv_files(data_dir: Path) -> Iterable[Path]:
    return sorted(
        path for path in data_dir.rglob("*.csv") if f".tmp{path.suffix}" not in path.name
    )


def main() -> None:
    args = parse_args()
    if (args.device is None) != (args.vllm_ascend_version is None):
        raise ValueError("--device and --vllm-ascend-version must be provided together.")
    if args.rows <= 0:
        raise ValueError("--rows must be greater than 0.")
    if args.min_value > args.max_value:
        raise ValueError("--min-value must be less than or equal to --max-value.")
    data_dir = resolve_data_dir(args.data_dir, args.device, args.vllm_ascend_version)
    if not data_dir.is_dir():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    rng = random.Random(args.seed)
    csv_files = list(iter_csv_files(data_dir))
    if not csv_files:
        raise ValueError(f"No CSV files found under: {data_dir}")

    total_appended_rows = 0
    skipped_files: list[Path] = []
    total_files = len(csv_files)
    for file_index, csv_path in enumerate(csv_files, start=1):
        appended_rows = process_csv_file(
            csv_path=csv_path,
            row_count=args.rows,
            min_value=args.min_value,
            max_value=args.max_value,
            rng=rng,
            file_index=file_index,
            total_files=total_files,
        )
        if appended_rows is None or appended_rows == 0:
            skipped_files.append(csv_path)
            continue
        total_appended_rows += appended_rows

    clear_progress()
    print(f"Appended {total_appended_rows} rows across {len(csv_files)} CSV files under {data_dir}.")
    if skipped_files:
        print("Skipped files with no 'Input Shapes' column or no usable Input Shapes templates:")
        for csv_path in skipped_files:
            print(f"  - {csv_path}")


if __name__ == "__main__":
    main()
