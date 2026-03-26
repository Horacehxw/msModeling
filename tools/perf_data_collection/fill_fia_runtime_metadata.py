"""
Backfill FIA runtime metadata from a JSONL dump into FusedInferAttentionScore.csv.

The JSONL is expected to be produced by the vllm-ascend runtime instrumentation
added around torch_npu.npu_fused_infer_attention_score calls.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from fia_common import parse_runtime_int, parse_shape_or_none, split_metadata_field


RUNTIME_ACTUAL_SEQ_LENGTHS_SHAPE = "Runtime actual_seq_lengths_shape"
RUNTIME_ACTUAL_SEQ_LENGTHS_VALUES = "Runtime actual_seq_lengths_values"
RUNTIME_ACTUAL_SEQ_LENGTHS_KV_SHAPE = "Runtime actual_seq_lengths_kv_shape"
RUNTIME_ACTUAL_SEQ_LENGTHS_KV_VALUES = "Runtime actual_seq_lengths_kv_values"
RUNTIME_AVG_SEQ_LEN = "Runtime avg_seq_len"
RUNTIME_OPERATOR_INPUT_SHAPES_RAW = "Runtime operator_input_shapes_raw"
RUNTIME_BLOCK_TABLE_SHAPE = "Runtime block_table_shape"
RUNTIME_BLOCK_TABLE_VALID_BLOCKS = "Runtime block_table_valid_blocks"
RUNTIME_NUM_HEADS = "Runtime num_heads"
RUNTIME_NUM_KEY_VALUE_HEADS = "Runtime num_key_value_heads"
RUNTIME_SPARSE_MODE = "Runtime sparse_mode"
RUNTIME_INPUT_LAYOUT = "Runtime input_layout"
RUNTIME_BLOCK_SIZE = "Runtime block_size"
RUNTIME_METADATA_COMPLETENESS = "Runtime metadata_completeness"
FIA_RUNTIME_COLUMNS = [
    RUNTIME_ACTUAL_SEQ_LENGTHS_SHAPE,
    RUNTIME_ACTUAL_SEQ_LENGTHS_VALUES,
    RUNTIME_ACTUAL_SEQ_LENGTHS_KV_SHAPE,
    RUNTIME_ACTUAL_SEQ_LENGTHS_KV_VALUES,
    RUNTIME_AVG_SEQ_LEN,
    RUNTIME_BLOCK_TABLE_VALID_BLOCKS,
    RUNTIME_METADATA_COMPLETENESS,
]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill FIA runtime metadata from JSONL into CSV.",
    )
    parser.add_argument("--csv-path", required=True, help="Path to FusedInferAttentionScore.csv")
    parser.add_argument("--jsonl-path", required=True, help="Path to fia_runtime_metadata.jsonl")
    parser.add_argument(
        "--output-path",
        help="Optional output CSV path. Defaults to overwriting --csv-path.",
    )
    parser.add_argument(
        "--metadata-tag",
        default="runtime_values_dumped",
        help="Value to write into Runtime metadata_completeness for matched rows.",
    )
    return parser


def parse_shape_token(raw_value: str | None) -> tuple[int, ...] | None:
    return parse_shape_or_none((raw_value or "").strip().strip('"'))


def split_input_shapes(raw_value: str) -> list[str]:
    return split_metadata_field(raw_value)


def parse_input_shape_slot(raw_value: str, index: int) -> tuple[int, ...] | None:
    slots = split_input_shapes(raw_value)
    if index >= len(slots):
        return None
    return parse_shape_token(slots[index])


def format_int_list(values: list[int] | None) -> str:
    if not values:
        return ""
    return ",".join(str(int(value)) for value in values)


def format_shape_len(values: list[int] | None) -> str:
    if not values:
        return ""
    return str(len(values))


def build_csv_signature(row: dict[str, str]) -> tuple:
    input_shapes_source = row.get(RUNTIME_OPERATOR_INPUT_SHAPES_RAW) or row.get("Input Shapes", "")
    return (
        parse_input_shape_slot(input_shapes_source, 0),
        parse_input_shape_slot(input_shapes_source, 1),
        parse_input_shape_slot(input_shapes_source, 2),
        parse_shape_token(row.get(RUNTIME_BLOCK_TABLE_SHAPE, "")),
        parse_runtime_int(row.get(RUNTIME_NUM_HEADS, "")),
        parse_runtime_int(row.get(RUNTIME_NUM_KEY_VALUE_HEADS, "")),
        (row.get(RUNTIME_INPUT_LAYOUT, "") or "").strip() or None,
        parse_runtime_int(row.get(RUNTIME_SPARSE_MODE, "")),
        parse_runtime_int(row.get(RUNTIME_BLOCK_SIZE, "")),
    )


def normalize_shape_list(value) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(item) for item in value)


def normalize_int_list(value) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(item) for item in value)


def build_json_signature(record: dict) -> tuple:
    return (
        normalize_shape_list(record.get("query_shape")),
        normalize_shape_list(record.get("key_shape")),
        normalize_shape_list(record.get("value_shape")),
        normalize_shape_list(record.get("block_table_shape")),
        int(record["num_heads"]) if record.get("num_heads") is not None else None,
        int(record["num_key_value_heads"]) if record.get("num_key_value_heads") is not None else None,
        record.get("input_layout"),
        int(record["sparse_mode"]) if record.get("sparse_mode") is not None else None,
        int(record["block_size"]) if record.get("block_size") is not None else None,
    )


def choose_most_common(counter: Counter) -> list[int] | None:
    if not counter:
        return None
    values, _ = counter.most_common(1)[0]
    return list(values)


def find_runtime_values(
    signature: tuple,
    runtime_summary: dict[tuple, dict[str, list[int] | float | int | None]],
) -> dict[str, list[int] | float | int | None] | None:
    exact = runtime_summary.get(signature)
    if exact is not None:
        return exact

    if signature[-1] is not None:
        return None

    prefix = signature[:-1]
    candidates = [
        values
        for candidate_signature, values in runtime_summary.items()
        if candidate_signature[:-1] == prefix
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def load_jsonl_summary(jsonl_path: Path) -> dict[tuple, dict[str, list[int] | float | int | None]]:
    grouped: dict[tuple, dict[str, Counter]] = defaultdict(
        lambda: {
            "actual_seq_lengths": Counter(),
            "actual_seq_lengths_kv": Counter(),
            "block_table_valid_blocks": Counter(),
        }
    )

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            signature = build_json_signature(record)
            if record.get("actual_seq_lengths") is not None:
                grouped[signature]["actual_seq_lengths"][
                    normalize_int_list(record.get("actual_seq_lengths"))
                ] += 1
            if record.get("actual_seq_lengths_kv") is not None:
                grouped[signature]["actual_seq_lengths_kv"][
                    normalize_int_list(record.get("actual_seq_lengths_kv"))
                ] += 1
            if record.get("block_table_valid_blocks") is not None:
                grouped[signature]["block_table_valid_blocks"][
                    normalize_int_list(record.get("block_table_valid_blocks"))
                ] += 1

    summary: dict[tuple, dict[str, list[int] | float | int | None]] = {}
    for signature, counters in grouped.items():
        actual_seq_lengths = choose_most_common(counters["actual_seq_lengths"])
        actual_seq_lengths_kv = choose_most_common(counters["actual_seq_lengths_kv"])
        block_table_valid_blocks = choose_most_common(counters["block_table_valid_blocks"])
        summary[signature] = {
            "actual_seq_lengths": actual_seq_lengths,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "block_table_valid_blocks": block_table_valid_blocks,
            "avg_seq_len": float(mean(actual_seq_lengths_kv)) if actual_seq_lengths_kv else None,
        }
    return summary


def backfill_rows(
    rows: list[dict[str, str]],
    runtime_summary: dict[tuple, dict[str, list[int] | float | int | None]],
    metadata_tag: str,
) -> tuple[list[dict[str, str]], int]:
    matched = 0
    for row in rows:
        signature = build_csv_signature(row)
        runtime_values = find_runtime_values(signature, runtime_summary)
        if runtime_values is None:
            continue

        actual_seq_lengths = runtime_values["actual_seq_lengths"]
        actual_seq_lengths_kv = runtime_values["actual_seq_lengths_kv"]
        block_table_valid_blocks = runtime_values["block_table_valid_blocks"]
        avg_seq_len = runtime_values["avg_seq_len"]

        row[RUNTIME_ACTUAL_SEQ_LENGTHS_SHAPE] = format_shape_len(actual_seq_lengths)
        row[RUNTIME_ACTUAL_SEQ_LENGTHS_VALUES] = format_int_list(actual_seq_lengths)
        row[RUNTIME_ACTUAL_SEQ_LENGTHS_KV_SHAPE] = format_shape_len(actual_seq_lengths_kv)
        row[RUNTIME_ACTUAL_SEQ_LENGTHS_KV_VALUES] = format_int_list(actual_seq_lengths_kv)
        row[RUNTIME_BLOCK_TABLE_VALID_BLOCKS] = format_int_list(block_table_valid_blocks)
        if avg_seq_len is not None:
            row[RUNTIME_AVG_SEQ_LEN] = f"{avg_seq_len:.6f}"
        row[RUNTIME_METADATA_COMPLETENESS] = metadata_tag
        matched += 1
    return rows, matched


def ensure_runtime_fieldnames(fieldnames: list[str]) -> list[str]:
    merged = list(fieldnames)
    for column in FIA_RUNTIME_COLUMNS:
        if column not in merged:
            merged.append(column)
    return merged


def main() -> None:
    args = build_argparser().parse_args()
    csv_path = Path(args.csv_path)
    jsonl_path = Path(args.jsonl_path)
    output_path = Path(args.output_path) if args.output_path else csv_path

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"CSV header is empty: {csv_path}")
        rows = list(reader)
        fieldnames = ensure_runtime_fieldnames(list(fieldnames))

    runtime_summary = load_jsonl_summary(jsonl_path)
    rows, matched = backfill_rows(rows, runtime_summary, args.metadata_tag)

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Backfilled {matched}/{len(rows)} FIA rows from {jsonl_path} into {output_path}"
    )


if __name__ == "__main__":
    main()
