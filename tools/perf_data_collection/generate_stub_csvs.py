#!/usr/bin/env python3
"""Generate stub CSVs from TensorCast trace data for end-to-end op_mapping verification.

This script:
1. Reads TC chrome trace extraction JSONs (from extract_tc_ops.py)
2. Loads op_mapping.yaml to determine kernel_type for each op
3. Applies TC→NPU shape transforms (reverse of profiling_data_source._inputs_match)
4. Writes stub {kernel_type}.csv files with fake durations for shape matching tests

The generated CSVs let ProfilingDataSource.lookup() find shape matches for all 49
dispatched ops, verifying the entire op_mapping + shape_matching pipeline end-to-end.

Usage:
    python tools/perf_data_collection/generate_stub_csvs.py \
        --traces docs/perf_database/reports/op_mapping/traces/ \
        --op-mapping tensor_cast/performance_model/perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/v0.13.0/op_mapping.yaml \
        --output-dir /tmp/stub_csvs
"""

import argparse
import csv
import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# NPU tile alignment sizes (Da Vinci Cube unit)
BLOCK_SIZES = (16, 32, 64)

# torch dtype string -> Profiling dtype string
DTYPE_MAP = {
    "bfloat16": "DT_BF16",
    "float16": "DT_BF16",  # FP16 treated as BF16 on Ascend
    "int8": "INT8",
    "int32": "INT32",
    "int64": "INT64",
    "float32": "FLOAT",
    "bool": "BOOL",
}

# Kernel types where ND weight is stored transposed
MATMUL_KERNELS = frozenset({"MatMulV2", "MatMul", "TransposeBatchMatMul"})

# SwiGlu: TC 2 inputs -> profiling 1 concatenated input
SWIGLU_KERNELS = frozenset({"SwiGlu"})

# RoPE: TC [Q,K,cos,sin] (B,H,S,D) -> profiling [K,Q,cos,sin] (B,S,H,D)
ROPE_KERNELS = frozenset({"ApplyRotaryPosEmb", "InterleaveRope"})

CSV_HEADER = [
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
    "Output Formats",
    "Average Duration(us)",
]


def parse_tc_shape(shape_str: str) -> Tuple[int, ...]:
    """Parse TC shape string like '(7008, 512)' -> (7008, 512)."""
    s = shape_str.strip().strip("()")
    if not s:
        return ()
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def shape_to_csv(shape: Tuple[int, ...]) -> str:
    """Convert shape tuple to CSV format: '7008,512'."""
    return ",".join(str(d) for d in shape)


def strip_batch_dim(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Strip leading batch dim=1 (TC keeps explicit batch, NPU doesn't)."""
    if len(shape) > 1 and shape[0] == 1:
        return shape[1:]
    return shape


def pad_to_block(dim: int, block: int = 16) -> int:
    """Pad dimension to NPU block alignment."""
    return ((dim + block - 1) // block) * block


def nd_to_fractal_nz_bf16(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Convert ND (K, N) -> FRACTAL_NZ [K/16, N/16, 16, 16] for BF16 weights."""
    if len(shape) != 2:
        return shape
    K, N = shape
    return (K // 16, N // 16, 16, 16)


def transform_shape_for_csv(
    tc_shapes: List[str],
    tc_dtypes: List[str],
    kernel_type: str,
    op_name: str,
) -> Optional[Tuple[List[Tuple[int, ...]], List[str], List[str]]]:
    """Transform TC shapes to NPU profiling CSV shapes.

    Returns: (csv_shapes, csv_dtypes, csv_formats) or None if skip.
    """
    shapes = [parse_tc_shape(s) for s in tc_shapes]
    # Ensure dtypes list matches shapes list length; default to DT_BF16
    dtypes = []
    for i, s in enumerate(tc_shapes):
        if i < len(tc_dtypes) and tc_dtypes[i]:
            dtypes.append(DTYPE_MAP.get(tc_dtypes[i], "DT_BF16"))
        else:
            # Infer: scalar () -> FLOAT, bool ops -> BOOL, else DT_BF16
            if s.strip() == "()" or s.strip() == "":
                dtypes.append("FLOAT")
            elif "bool" in op_name.lower() or (
                "where" in op_name and i == 0
            ):
                dtypes.append("BOOL")
            else:
                dtypes.append("DT_BF16")

    # Filter out scalar shapes (profiling _parse_shape_str drops empty shapes)
    non_scalar = [(s, d) for s, d in zip(shapes, dtypes) if len(s) > 0]
    if non_scalar:
        shapes = [s for s, _ in non_scalar]
        dtypes = [d for _, d in non_scalar]

    # Default: all ND format
    formats = ["ND"] * len(shapes)

    # Strip batch dim=1 from all shapes
    shapes = [strip_batch_dim(s) for s in shapes]

    # SwiGlu: merge 2 inputs into 1 concatenated on last dim
    if kernel_type in SWIGLU_KERNELS and len(shapes) == 2:
        s1, s2 = shapes
        if len(s1) == len(s2) and s1[:-1] == s2[:-1]:
            merged = s1[:-1] + (s1[-1] + s2[-1],)
            shapes = [merged]
            dtypes = [dtypes[0]]
            formats = ["ND"]

    # RoPE: TC [Q(B,H,S,D), K(B,H,S,D), cos(1,S,D), sin(1,S,D)]
    #     -> CSV [K(B,S,H,D), Q(B,S,H,D), cos(B,S,1,D), sin(B,S,1,D)]
    if kernel_type in ROPE_KERNELS and len(shapes) >= 4:
        q, k, cos, sin = shapes[:4]
        # Transpose Q/K: (B,H,S,D) -> (B,S,H,D) (after batch strip, (H,S,D)->(S,H,D))
        if len(q) == 3:
            q = (q[1], q[0], q[2])  # (H,S,D) -> (S,H,D)
        elif len(q) == 4:
            q = (q[0], q[2], q[1], q[3])  # (B,H,S,D) -> (B,S,H,D)
        if len(k) == 3:
            k = (k[1], k[0], k[2])
        elif len(k) == 4:
            k = (k[0], k[2], k[1], k[3])
        # cos/sin: insert head dim=1
        if len(cos) == 2:  # after batch strip: (S,D)
            cos = (cos[0], 1, cos[1])
        elif len(cos) == 3:
            cos = (cos[0], cos[1], 1, cos[2])
        if len(sin) == 2:
            sin = (sin[0], 1, sin[1])
        elif len(sin) == 3:
            sin = (sin[0], sin[1], 1, sin[2])
        # Reorder: [Q,K,cos,sin] -> [K,Q,cos,sin]
        shapes = [k, q, cos, sin] + shapes[4:]
        dtypes = [dtypes[1], dtypes[0]] + dtypes[2:]

    # MatMul: make weight FRACTAL_NZ for realism (optional - ND also works)
    # Keep ND for simplicity but transpose weight for matmul kernels
    if kernel_type in MATMUL_KERNELS and len(shapes) >= 2:
        w = shapes[1]
        if len(w) == 2:
            # CSV stores weight as (N, K) while TC sees (K, N)
            shapes[1] = (w[1], w[0])

    return shapes, dtypes, formats


def generate_stub_csvs(
    trace_dir: Path,
    op_mapping_path: Path,
    output_dir: Path,
) -> Dict[str, List[dict]]:
    """Generate stub CSVs and return generation report."""

    # Load op_mapping
    with open(op_mapping_path) as f:
        mapping = yaml.safe_load(f)
    entries = mapping.get("operator_mappings", {})

    # Load all traces
    all_ops: Dict[str, List[dict]] = {}  # op_name -> [variants]
    for trace_file in sorted(trace_dir.glob("*.json")):
        with open(trace_file) as f:
            data = json.load(f)
        for op in data["ops"]:
            op_name = op["op_name"]
            if op_name not in all_ops:
                all_ops[op_name] = []
            for v in op.get("shape_variants", []):
                # Tag with source trace
                v["_trace"] = trace_file.stem
                all_ops[op_name].append(v)

    # Collect CSV rows per kernel_type
    kernel_rows: Dict[str, List[dict]] = {}  # kernel_type -> [row_dicts]
    report = {
        "generated": [],
        "skipped_zero_cost": [],
        "skipped_communication": [],
        "skipped_attention": [],
        "skipped_unmapped": [],
        "errors": [],
    }

    for op_name, variants in sorted(all_ops.items()):
        config = entries.get(op_name)
        if not config:
            report["skipped_unmapped"].append(op_name)
            continue
        if config.get("zero_cost"):
            report["skipped_zero_cost"].append(op_name)
            continue
        if config.get("category") == "communication":
            report["skipped_communication"].append(op_name)
            continue
        if config.get("query_mode") == "attention_special":
            report["skipped_attention"].append(op_name)
            continue

        # Determine which kernel_types to generate for
        if config.get("composite"):
            sub_kernels = [
                sk for sk in config.get("sub_kernels", [])
                if not sk.startswith("hcom_")
            ]
        else:
            # Use csv_file if present (e.g., MoE ops where CSV name != kernel_type)
            primary = config.get("csv_file", config["kernel_type"])
            sub_kernels = [primary]
            for alt in config.get("alternate_kernel_types", []):
                if alt not in sub_kernels:
                    sub_kernels.append(alt)

        # Deduplicate shape variants
        seen = set()
        for variant in variants:
            input_shapes = variant.get("input_shapes", [])
            input_dtypes = variant.get("input_dtypes", [])

            variant_key = (op_name, tuple(input_shapes), tuple(input_dtypes))
            if variant_key in seen:
                continue
            seen.add(variant_key)

            for kernel_type in sub_kernels:
                result = transform_shape_for_csv(
                    input_shapes, input_dtypes, kernel_type, op_name
                )
                if result is None:
                    report["errors"].append(
                        f"{op_name} -> {kernel_type}: transform failed"
                    )
                    continue

                csv_shapes, csv_dtypes, csv_formats = result

                # Build output shapes (simplified: use first output from variant)
                out_shapes = variant.get("output_shapes", [])
                out_dtypes = variant.get("output_dtypes", [])
                out_csv_shapes = [
                    shape_to_csv(strip_batch_dim(parse_tc_shape(s)))
                    for s in out_shapes
                ]
                out_csv_dtypes = [DTYPE_MAP.get(d, "DT_BF16") for d in out_dtypes]

                row = {
                    "Input Shapes": ";".join(shape_to_csv(s) for s in csv_shapes),
                    "Input Data Types": ";".join(csv_dtypes),
                    "Input Formats": ";".join(csv_formats),
                    "Output Shapes": ";".join(out_csv_shapes),
                    "Output Data Types": ";".join(out_csv_dtypes),
                    "Output Formats": ";".join(["ND"] * len(out_csv_shapes)),
                    "Average Duration(us)": round(1.0 + hash(variant_key) % 100, 3),
                    "_op_name": op_name,
                    "_trace": variant.get("_trace", "unknown"),
                }

                if kernel_type not in kernel_rows:
                    kernel_rows[kernel_type] = []
                kernel_rows[kernel_type].append(row)

                # For elementwise ops with missing dtypes, also generate
                # FLOAT variant (MoE gating uses float32 for numerical stability)
                has_missing_dtype = any(
                    i >= len(input_dtypes) or not input_dtypes[i]
                    for i in range(len(input_shapes))
                    if input_shapes[i].strip() not in ("()", "")
                )
                if has_missing_dtype:
                    float_dtypes = [
                        "FLOAT" if d == "DT_BF16" else d for d in csv_dtypes
                    ]
                    float_row = dict(row)
                    float_row["Input Data Types"] = ";".join(float_dtypes)
                    float_row["Output Data Types"] = ";".join(
                        "FLOAT" if d == "DT_BF16" else d for d in out_csv_dtypes
                    )
                    kernel_rows[kernel_type].append(float_row)

                # For gather/scatter/index, generate all reasonable dtype combos.
                # These ops have mixed dtypes: data tensor + index tensor (INT64).
                if kernel_type in ("GatherV2", "ScatterElements"):
                    n_inputs = len(csv_shapes)
                    for data_dt in ("DT_BF16", "FLOAT", "INT64", "INT32"):
                        for idx_dt in ("INT64", "INT32"):
                            # gather/index: first=data, rest=index
                            # scatter: first=data, second=index
                            combo = [data_dt] * n_inputs
                            if n_inputs >= 2:
                                combo[-1] = idx_dt  # last is index
                            mixed_row = dict(row)
                            mixed_row["Input Data Types"] = ";".join(combo)
                            kernel_rows[kernel_type].append(mixed_row)

    # Write CSV files
    output_dir.mkdir(parents=True, exist_ok=True)

    for kernel_type, rows in sorted(kernel_rows.items()):
        # Deduplicate by Input Shapes + Input Data Types
        seen_shapes = set()
        unique_rows = []
        for row in rows:
            key = (row["Input Shapes"], row["Input Data Types"])
            if key not in seen_shapes:
                seen_shapes.add(key)
                unique_rows.append(row)

        csv_path = output_dir / f"{kernel_type}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=CSV_HEADER,
                quoting=csv.QUOTE_ALL,
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in unique_rows:
                writer.writerow(row)

        report["generated"].append({
            "kernel_type": kernel_type,
            "rows": len(unique_rows),
            "ops": list(set(r["_op_name"] for r in unique_rows)),
        })
        logger.info("Wrote %s: %d rows", csv_path, len(unique_rows))

    # Also copy op_mapping.yaml to output dir
    import shutil
    shutil.copy2(op_mapping_path, output_dir / "op_mapping.yaml")

    return report


def main():
    parser = argparse.ArgumentParser(description="Generate stub CSVs from TC traces")
    parser.add_argument(
        "--traces",
        type=Path,
        default=Path("docs/perf_database/reports/op_mapping/traces"),
        help="Directory containing trace JSON files",
    )
    parser.add_argument(
        "--op-mapping",
        type=Path,
        default=Path(
            "tensor_cast/performance_model/perf_database/data/"
            "ATLAS_800_A3_752T_128G_DIE/vllm_ascend/v0.13.0/op_mapping.yaml"
        ),
        help="Path to op_mapping.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/stub_csvs"),
        help="Output directory for stub CSVs",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    report = generate_stub_csvs(args.traces, args.op_mapping, args.output_dir)

    # Print summary
    print("\n=== Stub CSV Generation Report ===\n")
    print(f"Generated {len(report['generated'])} kernel CSVs:")
    for entry in report["generated"]:
        print(f"  {entry['kernel_type']}.csv: {entry['rows']} rows ({entry['ops']})")

    print(f"\nSkipped: {len(report['skipped_zero_cost'])} zero_cost, "
          f"{len(report['skipped_communication'])} communication, "
          f"{len(report['skipped_attention'])} attention_special, "
          f"{len(report['skipped_unmapped'])} unmapped")

    if report["errors"]:
        print(f"\nErrors ({len(report['errors'])}):")
        for e in report["errors"]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
