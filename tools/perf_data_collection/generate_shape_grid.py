"""Append randomized, constraint-aware shape-grid rows to perf database CSV files."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import stat
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
    "GreaterEqual",
    "Less",
    "LogicalAnd",
    "MaskedFill",
    "Mul",
    "RealDiv",
    "Sub",
}
ELEMENTWISE_UNARY_KERNELS = {
    "Cast",
    "Fill",
    "Log",
    "LogicalNot",
    "Neg",
    "SoftmaxV2",
    "TensorMove",
}
MATMUL_KERNELS = {
    "MatMul",
    "MatMulV2",
    "MatMulV3",
    "MatmulReduceScatterV2",
}
LATENCY_HEADER_KEYWORDS = ("duration", "latency", "time", "cycles", "ratio", "miss", "utilization")
MISSING_SHAPE_TOKENS = {"", "N/A", "NA", "NULL", "NONE", "UNDEFINED"}


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
        default=DEFAULT_DATA_DIR,
        help=f"CSV root directory. Default: {DEFAULT_DATA_DIR}",
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


def infer_scalar_shape_count(shape_count: int, template_shapes: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    if len(template_shapes) >= shape_count:
        return list(template_shapes[:shape_count])
    return list(template_shapes) + [()] * (shape_count - len(template_shapes))


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
        rhs = tuple(1 if dim == 1 else base[min(index, len(base) - 1)] for index, dim in enumerate(rhs_template))
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
    seq = random_dim(rng, min_value, max_value, template_dim=template_inputs[0][0] if template_inputs and template_inputs[0] else None, alignment=8)
    rope_dim = random_dim(rng, min_value, max_value, template_dim=template_inputs[1][-1] if len(template_inputs) > 1 and template_inputs[1] else None, alignment=16)
    hidden_template = (
        template_inputs[0][-1] - 2 * template_inputs[1][-1]
        if template_inputs and template_inputs[0] and len(template_inputs) > 1 and template_inputs[1]
        else 1024
    )
    hidden = random_dim(rng, min_value, max_value, template_dim=hidden_template, alignment=16)
    combined = hidden + rope_dim * 2
    rope = (1, seq, 1, rope_dim)
    return [(seq, combined), rope, rope], [(seq, hidden), (seq, rope_dim)]


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
    seq = random_dim(rng, min_value, max_value, template_dim=template_inputs[0][0] if template_inputs and template_inputs[0] else None, alignment=8)
    heads = random_dim(rng, 1, min(max_value, 64), template_dim=template_inputs[0][1] if template_inputs and len(template_inputs[0]) > 1 else None)
    kv_heads = 1 if len(template_inputs) < 2 or not template_inputs[1] or template_inputs[1][1] == 1 else random_dim(rng, 1, min(max_value, heads), template_dim=template_inputs[1][1])
    dim = random_dim(rng, min_value, max_value, template_dim=template_inputs[0][-1] if template_inputs and template_inputs[0] else None, alignment=16)
    mask_size = random_dim(
        rng,
        min_value,
        max_value,
        template_dim=template_inputs[4][0] if len(template_inputs) > 4 and template_inputs[4] else None,
        alignment=16,
    )
    input_shapes = infer_scalar_shape_count(len(template_inputs), template_inputs)
    input_shapes[0] = (seq, heads, dim)
    input_shapes[1] = (seq, kv_heads, dim)
    input_shapes[2] = (seq, kv_heads, dim)
    if len(input_shapes) > 4:
        input_shapes[4] = (mask_size, mask_size)
    for scalar_index in (5, 6):
        if scalar_index < len(input_shapes) and input_shapes[scalar_index]:
            input_shapes[scalar_index] = (1,)
    output_shapes = infer_scalar_shape_count(len(template_outputs), template_outputs)
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
    if kernel_type == "ApplyRotaryPosEmb":
        return generate_rope_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "SwiGlu":
        return generate_swiglu_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "AddRmsNorm":
        return generate_add_rmsnorm_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "RmsNorm":
        return generate_rmsnorm_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type in {"GatherV2"}:
        return generate_gather_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "GatherV3":
        return generate_gather_v3_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "Index":
        return generate_index_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "ArgMaxV2":
        return generate_argmax_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "Sort":
        return generate_sort_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "ApplyTopKTopPWithSorted":
        return generate_apply_topk_top_p_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "split_qkv_rmsnorm_rope_kernel":
        return generate_split_qkv_rmsnorm_rope_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "ReshapeAndCacheNdKernel":
        return generate_reshape_and_cache_shapes(template_inputs, rng, min_value, max_value)
    if kernel_type == "FusedInferAttentionScore":
        return generate_fused_attention_shapes(template_inputs, template_outputs, rng, min_value, max_value)
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
) -> int:
    if not source_rows:
        raise ValueError("CSV does not contain any data rows.")

    valid_source_rows = [
        row for row in source_rows if any(parse_shape_text(row.get(INPUT_SHAPES_COLUMN, "")))
    ]
    if not valid_source_rows:
        return 0

    appended_rows = 0
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
        row = build_row_template(headers, source_row)
        row[INPUT_SHAPES_COLUMN] = build_shape_text(generated_inputs)
        if OUTPUT_SHAPES_COLUMN in headers:
            row[OUTPUT_SHAPES_COLUMN] = build_shape_text(generated_outputs)
        writer.writerow(row)
        appended_rows += 1
    return appended_rows


def process_csv_file(
    csv_path: Path,
    row_count: int,
    min_value: int,
    max_value: int,
    rng: random.Random,
) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        headers = reader.fieldnames
        if not headers:
            raise ValueError(f"{csv_path} is missing a header row.")
        if INPUT_SHAPES_COLUMN not in headers:
            raise ValueError(f"{csv_path} is missing the '{INPUT_SHAPES_COLUMN}' column.")
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
    if args.rows <= 0:
        raise ValueError("--rows must be greater than 0.")
    if args.min_value > args.max_value:
        raise ValueError("--min-value must be less than or equal to --max-value.")
    if not args.data_dir.is_dir():
        raise ValueError(f"Data directory does not exist: {args.data_dir}")

    rng = random.Random(args.seed)
    csv_files = list(iter_csv_files(args.data_dir))
    if not csv_files:
        raise ValueError(f"No CSV files found under: {args.data_dir}")

    total_appended_rows = 0
    skipped_files: list[Path] = []
    for csv_path in csv_files:
        appended_rows = process_csv_file(
            csv_path=csv_path,
            row_count=args.rows,
            min_value=args.min_value,
            max_value=args.max_value,
            rng=rng,
        )
        total_appended_rows += appended_rows
        if appended_rows == 0:
            skipped_files.append(csv_path)

    print(f"Appended {total_appended_rows} rows across {len(csv_files)} CSV files under {args.data_dir}.")
    if skipped_files:
        print("Skipped files with no usable Input Shapes templates:")
        for csv_path in skipped_files:
            print(f"  - {csv_path}")


if __name__ == "__main__":
    main()
