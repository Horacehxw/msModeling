"""Compute M6: Empirical Prediction Coverage (offline metric).

M6 = Σ(empirical HIT duration) / (Computing + Communication_NotOverlapped)

Inputs:
  1. TC metrics report JSON (from --export-metrics)
  2. ASCEND_PROFILER_OUTPUT directory (contains step_trace_time.csv + kernel_details.csv)

Usage:
    python3.10 tools/perf_data_collection/compute_m6.py \
        --tc-report results/qwen3_prefill_metrics.json \
        --profiler-output /path/to/ASCEND_PROFILER_OUTPUT

Design doc reference: §7.5 M6
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def parse_step_trace_time(
    profiler_output_dir: Path, step_index: int = 0
) -> dict:
    """Parse step_trace_time.csv and return Computing + Comm(Not Overlapped).

    Args:
        profiler_output_dir: ASCEND_PROFILER_OUTPUT directory
        step_index: Which step row to use (default: 0, first step)

    Returns:
        dict with computing_us, comm_not_overlapped_us, denominator_us
    """
    step_csv = profiler_output_dir / "step_trace_time.csv"
    if not step_csv.exists():
        raise FileNotFoundError(f"step_trace_time.csv not found: {step_csv}")

    with step_csv.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Empty step_trace_time.csv: {step_csv}")
    if step_index >= len(rows):
        raise ValueError(
            f"step_index {step_index} out of range (file has {len(rows)} rows)"
        )

    row = rows[step_index]
    computing = float(row["Computing"])
    comm_not_overlapped = float(row["Communication(Not Overlapped)"])

    return {
        "computing_us": computing,
        "comm_not_overlapped_us": comm_not_overlapped,
        "denominator_us": computing + comm_not_overlapped,
    }


def parse_kernel_details_diagnostics(
    profiler_output_dir: Path, hit_kernel_types: set
) -> list:
    """Parse kernel_details.csv for diagnostic per-kernel breakdown.

    Excludes *AicpuKernel entries (AICPU dispatch wrappers that duplicate
    hcom_* communication kernels).

    Args:
        profiler_output_dir: ASCEND_PROFILER_OUTPUT directory
        hit_kernel_types: set of kernel_type strings that were HIT

    Returns:
        List of unmatched kernel dicts sorted by duration desc
    """
    kd_csv = profiler_output_dir / "kernel_details.csv"
    if not kd_csv.exists():
        return []

    type_durations: dict[str, float] = defaultdict(float)
    with kd_csv.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            kernel_type = (row.get("Type") or "").strip()
            if not kernel_type or kernel_type.endswith("AicpuKernel"):
                continue
            duration = float((row.get("Duration(us)") or "0").strip())
            type_durations[kernel_type] += duration

    total_kd = sum(type_durations.values())
    unmatched = []
    for kt, dur in sorted(type_durations.items(), key=lambda x: -x[1]):
        if kt not in hit_kernel_types:
            unmatched.append(
                {
                    "kernel_type": kt,
                    "duration_us": dur,
                    "pct_of_kd_total": dur / total_kd * 100 if total_kd else 0,
                }
            )

    return unmatched


def compute_m6(
    tc_report: dict, profiler_output_dir: Path, step_index: int = 0
) -> dict:
    """Compute M6: Empirical Prediction Coverage.

    M6 = Σ(empirical HIT duration) / (Computing + Comm_NotOverlapped)

    Args:
        tc_report: Parsed TC metrics JSON (from export_hit_miss_report)
        profiler_output_dir: ASCEND_PROFILER_OUTPUT directory
        step_index: Which step to use from step_trace_time.csv

    Returns:
        dict with m6 value, components, and diagnostics
    """
    # 1. Numerator: from TC report (scaled by replay_multiplier for full model)
    m6_input = tc_report["m6_input"]
    if "empirical_hit_duration_scaled_s" in m6_input:
        empirical_hit_sum_s = m6_input["empirical_hit_duration_scaled_s"]
    else:
        # Fallback for reports without multiplier
        empirical_hit_sum_s = m6_input["empirical_hit_duration_sum_s"]
    empirical_hit_sum_us = empirical_hit_sum_s * 1e6

    # 2. Denominator: from step_trace_time.csv
    step_info = parse_step_trace_time(profiler_output_dir, step_index)
    denominator_us = step_info["denominator_us"]

    # 3. M6
    m6 = empirical_hit_sum_us / denominator_us if denominator_us > 0 else 0.0

    # 4. Diagnostics: unmatched kernels from kernel_details.csv
    hit_kernel_types = {h["kernel_type"] for h in tc_report.get("hits", [])}
    unmatched = parse_kernel_details_diagnostics(
        profiler_output_dir, hit_kernel_types
    )

    return {
        "m6_empirical_prediction_coverage": m6,
        "empirical_hit_sum_us": empirical_hit_sum_us,
        **step_info,
        "unmatched_kernels": unmatched,
    }


def _format_report(result: dict) -> str:
    """Format M6 result as human-readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("M6: Empirical Prediction Coverage")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        f"Empirical HIT duration: {result['empirical_hit_sum_us']:,.1f} us"
    )
    lines.append(
        f"Step duration:          {result['denominator_us']:,.1f} us "
        f"(Computing: {result['computing_us']:,.1f} "
        f"+ Comm: {result['comm_not_overlapped_us']:,.1f})"
    )
    lines.append("")
    m6_pct = result["m6_empirical_prediction_coverage"] * 100
    lines.append(f"M6 = {m6_pct:.1f}%")
    lines.append("")

    if result["unmatched_kernels"]:
        lines.append(
            f"Top unmatched kernels "
            f"(from kernel_details.csv, excl AicpuKernel):"
        )
        lines.append(
            f"  {'Type':<40} {'Duration(us)':>14} {'%KD Total':>10}"
        )
        lines.append(f"  {'-'*40} {'-'*14} {'-'*10}")
        for k in result["unmatched_kernels"][:15]:
            lines.append(
                f"  {k['kernel_type']:<40} "
                f"{k['duration_us']:>14,.1f} "
                f"{k['pct_of_kd_total']:>9.1f}%"
            )
        remaining = len(result["unmatched_kernels"]) - 15
        if remaining > 0:
            lines.append(f"  ... and {remaining} more")

    return "\n".join(lines)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute M6: Empirical Prediction Coverage (offline metric)."
    )
    parser.add_argument(
        "--tc-report",
        required=True,
        help="Path to TC metrics JSON (from --export-metrics)",
    )
    parser.add_argument(
        "--profiler-output",
        required=True,
        help="Path to ASCEND_PROFILER_OUTPUT directory "
        "(contains step_trace_time.csv + kernel_details.csv)",
    )
    parser.add_argument(
        "--step-index",
        type=int,
        default=0,
        help="Step index in step_trace_time.csv (default: 0)",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Path to write JSON result (optional)",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    tc_report_path = Path(args.tc_report)
    if not tc_report_path.exists():
        print(f"Error: TC report not found: {tc_report_path}", file=sys.stderr)
        sys.exit(1)

    profiler_output_dir = Path(args.profiler_output)
    if not profiler_output_dir.exists():
        print(
            f"Error: Profiler output dir not found: {profiler_output_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    with tc_report_path.open() as f:
        tc_report = json.load(f)

    result = compute_m6(tc_report, profiler_output_dir, args.step_index)
    print(_format_report(result))

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nJSON output written to: {output_path}")


if __name__ == "__main__":
    main()
