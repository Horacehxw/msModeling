#!/usr/bin/env python3.10
"""Extract dispatched ops and shapes from TensorCast chrome trace JSON files.

Parses the chrome trace output from TensorCast simulations and produces a
structured JSON report of all dispatched ops, their input shapes, invocation
counts, and mapping status against op_mapping.yaml.

Usage:
    python3.10 tools/perf_data_collection/extract_tc_ops.py \
        --chrome-trace /tmp/qwen3_32b_prefill_trace.json \
        --output docs/perf_database/reports/op_mapping/traces/qwen3_32b_prefill.json \
        --model Qwen/Qwen3-32B --scenario prefill
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

# Default op_mapping.yaml path (relative to repo root)
DEFAULT_OP_MAPPING = (
    "tensor_cast/performance_model/perf_database/data/"
    "ATLAS_800_A3_752T_128G_DIE/vllm_ascend/v0.13.0/op_mapping.yaml"
)


def parse_tensor_shapes(inputs_str: str) -> list[str]:
    """Extract tensor shape tuples from the Inputs string.

    The Inputs string looks like:
        "(tensor(..., size=(1, 7008, 5120), dtype=torch.float16), ...) kwargs: {}"

    Returns a list of shape strings like ["(1, 7008, 5120)", "(5120, 320)"].
    """
    # Match size=(...) patterns
    shapes = re.findall(r"size=\(([^)]*)\)", inputs_str)
    result = []
    for s in shapes:
        # Clean up and normalize
        dims = [d.strip() for d in s.split(",") if d.strip()]
        if dims:
            result.append("(" + ", ".join(dims) + ")")
        else:
            result.append("()")
    return result


def parse_tensor_dtypes(inputs_str: str) -> list[str]:
    """Extract tensor dtypes from the Inputs string."""
    return re.findall(r"dtype=torch\.(\w+)", inputs_str)


def normalize_op_name(name: str) -> str:
    """Normalize op name from trace format to op_mapping key format.

    Trace format: 'aten.mm.default' or 'tensor_cast.swiglu.default'
    op_mapping key: 'aten.mm.default' or 'tensor_cast.swiglu.default'

    They should match directly.
    """
    return name


def load_op_mapping(op_mapping_path: str) -> dict:
    """Load op_mapping.yaml and return the operator_mappings dict."""
    with open(op_mapping_path) as f:
        data = yaml.safe_load(f)
    return data.get("operator_mappings", {})


def extract_ops(trace_path: str) -> list[dict]:
    """Extract all ops from a chrome trace file.

    Returns list of dicts with keys: name, input_shapes, dtypes, duration_us
    """
    with open(trace_path) as f:
        trace_data = json.load(f)

    events = trace_data.get("traceEvents", [])
    ops = []
    for event in events:
        if event.get("ph") != "X":
            continue

        name = event.get("name", "")
        dur = event.get("dur", 0.0)
        inputs_str = event.get("args", {}).get("Inputs", "")
        output_str = event.get("args", {}).get("Output", "")

        input_shapes = parse_tensor_shapes(inputs_str)
        input_dtypes = parse_tensor_dtypes(inputs_str)
        output_shapes = parse_tensor_shapes(output_str)
        output_dtypes = parse_tensor_dtypes(output_str)

        ops.append(
            {
                "name": name,
                "input_shapes": input_shapes,
                "input_dtypes": input_dtypes,
                "output_shapes": output_shapes,
                "output_dtypes": output_dtypes,
                "duration_us": dur,
            }
        )

    return ops


def build_report(
    ops: list[dict],
    op_mapping: dict,
    model: str,
    scenario: str,
) -> dict:
    """Build the analysis report from extracted ops.

    Returns a dict with:
      - model, scenario
      - total_ops: total number of op invocations
      - unique_ops: number of unique op names
      - ops: per-op details (sorted by total_duration desc)
      - unmapped_ops: list of ops not found in op_mapping
      - mapped_ops: list of ops found in op_mapping
    """
    # Aggregate by op name
    op_stats = defaultdict(
        lambda: {
            "count": 0,
            "total_duration_us": 0.0,
            "shape_variants": [],
        }
    )

    for op in ops:
        name = op["name"]
        stats = op_stats[name]
        stats["count"] += 1
        stats["total_duration_us"] += op["duration_us"]

        # Track unique shape combinations
        shape_key = {
            "input_shapes": op["input_shapes"],
            "input_dtypes": op["input_dtypes"],
            "output_shapes": op["output_shapes"],
            "output_dtypes": op["output_dtypes"],
        }
        # Use JSON string for dedup
        shape_str = json.dumps(shape_key, sort_keys=True)
        if not any(
            json.dumps(sv, sort_keys=True) == shape_str
            for sv in stats["shape_variants"]
        ):
            stats["shape_variants"].append(shape_key)

    # Classify mapped vs unmapped
    mapped_ops = []
    unmapped_ops = []
    op_details = []

    for name, stats in sorted(
        op_stats.items(), key=lambda x: -x[1]["total_duration_us"]
    ):
        mapping_key = normalize_op_name(name)
        mapping_entry = op_mapping.get(mapping_key)
        is_mapped = mapping_entry is not None

        if is_mapped:
            # Extract key mapping info
            kernel_type = mapping_entry.get("kernel_type", None)
            is_zero_cost = mapping_entry.get("zero_cost", False)
            is_composite = mapping_entry.get("composite", False)
            category = mapping_entry.get("category", None)
            sub_kernels = mapping_entry.get("sub_kernels", None)
        else:
            kernel_type = None
            is_zero_cost = False
            is_composite = False
            category = None
            sub_kernels = None

        detail = {
            "op_name": name,
            "count": stats["count"],
            "total_duration_us": round(stats["total_duration_us"], 3),
            "unique_shape_variants": len(stats["shape_variants"]),
            "shape_variants": stats["shape_variants"],
            "is_mapped": is_mapped,
            "kernel_type": kernel_type,
            "zero_cost": is_zero_cost,
            "composite": is_composite,
            "category": category,
            "sub_kernels": sub_kernels,
        }
        op_details.append(detail)

        if is_mapped:
            mapped_ops.append(name)
        else:
            unmapped_ops.append(name)

    return {
        "model": model,
        "scenario": scenario,
        "total_ops": len(ops),
        "unique_ops": len(op_stats),
        "mapped_count": len(mapped_ops),
        "unmapped_count": len(unmapped_ops),
        "unmapped_ops": sorted(unmapped_ops),
        "mapped_ops": sorted(mapped_ops),
        "ops": op_details,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract dispatched ops from TensorCast chrome trace"
    )
    parser.add_argument(
        "--chrome-trace", required=True, help="Path to chrome trace JSON file"
    )
    parser.add_argument(
        "--output", required=True, help="Path for output JSON report"
    )
    parser.add_argument(
        "--model", default="unknown", help="Model name for report metadata"
    )
    parser.add_argument(
        "--scenario", default="unknown", help="Scenario (prefill/decode)"
    )
    parser.add_argument(
        "--op-mapping",
        default=None,
        help="Path to op_mapping.yaml (auto-detected if not specified)",
    )
    args = parser.parse_args()

    # Find op_mapping.yaml
    if args.op_mapping:
        op_mapping_path = args.op_mapping
    else:
        # Try to find relative to script location
        repo_root = Path(__file__).resolve().parent.parent.parent
        op_mapping_path = str(repo_root / DEFAULT_OP_MAPPING)

    if not os.path.exists(op_mapping_path):
        print(f"WARNING: op_mapping.yaml not found at {op_mapping_path}")
        op_mapping = {}
    else:
        op_mapping = load_op_mapping(op_mapping_path)
        print(f"Loaded op_mapping with {len(op_mapping)} entries from {op_mapping_path}")

    # Extract ops from trace
    print(f"Parsing trace: {args.chrome_trace}")
    ops = extract_ops(args.chrome_trace)
    print(f"  Found {len(ops)} op invocations")

    # Build report
    report = build_report(ops, op_mapping, args.model, args.scenario)
    print(f"  Unique ops: {report['unique_ops']}")
    print(f"  Mapped: {report['mapped_count']}, Unmapped: {report['unmapped_count']}")

    if report["unmapped_ops"]:
        print(f"  Unmapped ops: {report['unmapped_ops']}")

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report written to: {args.output}")

    return report


if __name__ == "__main__":
    main()
