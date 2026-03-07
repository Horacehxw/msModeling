"""Discover operators in profiling data and compare against op_mapping.yaml coverage.

Reads a kernel_details.csv (Ascend profiler output) and an op_mapping.yaml,
then reports which profiling kernel types are mapped (known) vs unmapped (unknown).

Usage:
    python3.10 tools/perf_data_collection/discover_operators.py \
        --kernel-details /path/to/kernel_details.csv \
        --op-mapping tensor_cast/performance_model/perf_database/data/.../op_mapping.yaml

Design doc reference: S6.6 (New Model Operator Discovery)
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

import yaml


def _collect_mapped_kernel_types(op_mapping: dict) -> Set[str]:
    """Extract all kernel_types referenced in op_mapping (primary + alternates + sub_kernels)."""
    kernel_types: Set[str] = set()
    for _op_name, mapping in op_mapping.get("operator_mappings", {}).items():
        if mapping.get("kernel_type"):
            kernel_types.add(mapping["kernel_type"])
        for alt in mapping.get("alternate_kernel_types", []):
            kernel_types.add(alt)
        for sk in mapping.get("sub_kernels", []):
            kernel_types.add(sk)
    return kernel_types


def _parse_profiling_types(kernel_details_path: Path) -> List[Dict]:
    """Parse kernel_details.csv and aggregate by Type."""
    type_stats: Dict[str, Dict] = {}

    with kernel_details_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernel_type = (row.get("Type") or "").strip()
            if not kernel_type:
                continue
            try:
                duration = float((row.get("Duration(us)") or "0").strip())
            except ValueError:
                duration = 0.0

            if kernel_type not in type_stats:
                type_stats[kernel_type] = {
                    "kernel_type": kernel_type,
                    "count": 0,
                    "total_duration_us": 0.0,
                }
            type_stats[kernel_type]["count"] += 1
            type_stats[kernel_type]["total_duration_us"] += duration

    return list(type_stats.values())


def discover_operators(
    kernel_details_path: Path, op_mapping_path: Path
) -> Dict:
    """Discover profiling kernel types and classify as known/unknown vs op_mapping.

    Args:
        kernel_details_path: Path to kernel_details.csv (Ascend profiler output)
        op_mapping_path: Path to op_mapping.yaml

    Returns:
        Dict with keys: known, unknown (lists sorted by duration desc), coverage (stats)
    """
    with op_mapping_path.open("r", encoding="utf-8") as f:
        op_mapping = yaml.safe_load(f)

    mapped_types = _collect_mapped_kernel_types(op_mapping)
    profiling_types = _parse_profiling_types(kernel_details_path)

    known = []
    unknown = []
    for entry in profiling_types:
        if entry["kernel_type"] in mapped_types:
            known.append(entry)
        else:
            unknown.append(entry)

    known.sort(key=lambda x: x["total_duration_us"], reverse=True)
    unknown.sort(key=lambda x: x["total_duration_us"], reverse=True)

    total_calls = sum(e["count"] for e in profiling_types)
    known_calls = sum(e["count"] for e in known)
    total_duration = sum(e["total_duration_us"] for e in profiling_types)
    known_duration = sum(e["total_duration_us"] for e in known)

    coverage = {
        "known_types": len(known),
        "unknown_types": len(unknown),
        "total_types": len(profiling_types),
        "known_calls": known_calls,
        "total_calls": total_calls,
        "call_coverage_pct": (known_calls / total_calls * 100) if total_calls else 0.0,
        "known_duration_us": known_duration,
        "total_duration_us": total_duration,
        "duration_coverage_pct": (
            known_duration / total_duration * 100
        )
        if total_duration
        else 0.0,
    }

    return {"known": known, "unknown": unknown, "coverage": coverage}


def _format_report(result: Dict, kernel_details_path: Path, op_mapping_path: Path) -> str:
    """Format discovery result as human-readable report."""
    lines = []
    cov = result["coverage"]

    lines.append("=" * 70)
    lines.append("Operator Discovery Report")
    lines.append("=" * 70)
    lines.append(f"Profiling:  {kernel_details_path}")
    lines.append(f"op_mapping: {op_mapping_path}")
    lines.append("")

    lines.append("Coverage Summary:")
    lines.append(
        f"  Types:    {cov['known_types']}/{cov['total_types']} "
        f"({cov['known_types']/cov['total_types']*100:.1f}%)"
        if cov["total_types"]
        else "  Types:    0/0"
    )
    lines.append(
        f"  Calls:    {cov['known_calls']}/{cov['total_calls']} "
        f"({cov['call_coverage_pct']:.1f}%)"
    )
    lines.append(
        f"  Duration: {cov['known_duration_us']:.1f}/{cov['total_duration_us']:.1f} us "
        f"({cov['duration_coverage_pct']:.1f}%)"
    )
    lines.append("")

    if result["unknown"]:
        lines.append(f"Unknown Kernel Types ({len(result['unknown'])}):")
        lines.append(f"  {'Type':<40} {'Count':>8} {'Duration(us)':>14} {'%Total':>8}")
        lines.append(f"  {'-'*40} {'-'*8} {'-'*14} {'-'*8}")
        for u in result["unknown"]:
            pct = (
                u["total_duration_us"] / cov["total_duration_us"] * 100
                if cov["total_duration_us"]
                else 0
            )
            lines.append(
                f"  {u['kernel_type']:<40} {u['count']:>8} "
                f"{u['total_duration_us']:>14.1f} {pct:>7.1f}%"
            )
        lines.append("")

    if result["known"]:
        lines.append(f"Known Kernel Types ({len(result['known'])}):")
        lines.append(f"  {'Type':<40} {'Count':>8} {'Duration(us)':>14} {'%Total':>8}")
        lines.append(f"  {'-'*40} {'-'*8} {'-'*14} {'-'*8}")
        for k in result["known"]:
            pct = (
                k["total_duration_us"] / cov["total_duration_us"] * 100
                if cov["total_duration_us"]
                else 0
            )
            lines.append(
                f"  {k['kernel_type']:<40} {k['count']:>8} "
                f"{k['total_duration_us']:>14.1f} {pct:>7.1f}%"
            )

    return "\n".join(lines)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover profiling kernel types and compare against op_mapping.yaml coverage."
    )
    parser.add_argument(
        "--kernel-details",
        required=True,
        help="Path to Ascend profiler kernel_details.csv",
    )
    parser.add_argument(
        "--op-mapping",
        default=None,
        help="Path to op_mapping.yaml (default: auto-detect from repo)",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Path to write JSON result (optional)",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    kernel_details_path = Path(args.kernel_details)
    if not kernel_details_path.exists():
        print(f"Error: kernel_details.csv not found: {kernel_details_path}", file=sys.stderr)
        sys.exit(1)

    if args.op_mapping:
        op_mapping_path = Path(args.op_mapping)
    else:
        repo_root = Path(__file__).resolve().parents[2]
        op_mapping_path = (
            repo_root
            / "tensor_cast/performance_model/perf_database/data"
            / "ATLAS_800_A3_752T_128G_DIE/vllm_ascend/v0.13.0/op_mapping.yaml"
        )

    if not op_mapping_path.exists():
        print(f"Error: op_mapping.yaml not found: {op_mapping_path}", file=sys.stderr)
        sys.exit(1)

    result = discover_operators(kernel_details_path, op_mapping_path)

    print(_format_report(result, kernel_details_path, op_mapping_path))

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nJSON output written to: {output_path}")


if __name__ == "__main__":
    main()
