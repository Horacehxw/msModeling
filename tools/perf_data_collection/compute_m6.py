"""Compute M6: Empirical E2E Prediction Ratio (offline metric).

M6 = Empirical_hit_total / Real_per_forward_pass

Where:
  - Empirical_hit_total = sum of empirical (microbench CSV) latencies for all
    HIT ops across full model replay (excludes analytic fallback for MISS ops)
  - Real_per_forward_pass = avg kernel duration per forward pass, derived from
    kernel_details.csv by splitting on a sampling delimiter (ArgMaxV2).

M6 = 1.0 means perfect prediction. M6 > 1 = overestimate, M6 < 1 = underestimate.
Phase 3 target: 0.85 ≤ M6 ≤ 1.15 (i.e., within ±15%).

Usage:
    # Auto-detect forward passes via sampling delimiter (default: ArgMaxV2)
    python3.10 tools/perf_data_collection/compute_m6.py \
        --tc-report results/qwen3_prefill_metrics.json \
        --profiler-output /path/to/ASCEND_PROFILER_OUTPUT

    # Override forward pass count
    python3.10 tools/perf_data_collection/compute_m6.py \
        --tc-report results/qwen3_prefill_metrics.json \
        --profiler-output /path/to/ASCEND_PROFILER_OUTPUT \
        --n-forward-passes 5

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

    Note: step_trace_time.csv may aggregate the entire profiling window
    (Step column empty), covering multiple forward passes.
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
        "step_total_us": computing + comm_not_overlapped,
    }


def estimate_forward_passes(
    profiler_output_dir: Path,
    delimiter: str = "ArgMaxV2",
) -> dict:
    """Estimate forward pass count and per-fwd duration from kernel_details.csv.

    Model-agnostic: splits the kernel trace on a sampling delimiter kernel
    that appears exactly once per forward pass. Default: ArgMaxV2 (final
    token selection in autoregressive decoding/sampling).

    The total kernel duration divided by N gives avg per-forward-pass time.

    Args:
        profiler_output_dir: ASCEND_PROFILER_OUTPUT directory
        delimiter: Kernel type that appears once per fwd pass (default: ArgMaxV2)

    Returns:
        dict with n_forward_passes, delimiter, total_kernel_duration_us,
        avg_fwd_duration_us
    """
    kd_csv = profiler_output_dir / "kernel_details.csv"
    if not kd_csv.exists():
        return {
            "n_forward_passes": 1,
            "delimiter": delimiter,
            "total_kernel_duration_us": 0.0,
            "avg_fwd_duration_us": 0.0,
        }

    total_duration = 0.0
    delimiter_count = 0
    with kd_csv.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            t = (row.get("Type") or "").strip()
            dur = float((row.get("Duration(us)") or "0").strip())
            if not t or t.endswith("AicpuKernel"):
                continue
            total_duration += dur
            if t == delimiter:
                delimiter_count += 1

    n_fwd = max(1, delimiter_count)
    avg_fwd = total_duration / n_fwd if n_fwd > 0 else total_duration

    return {
        "n_forward_passes": n_fwd,
        "delimiter": delimiter,
        "total_kernel_duration_us": total_duration,
        "avg_fwd_duration_us": avg_fwd,
    }


def parse_kernel_details_diagnostics(
    profiler_output_dir: Path, hit_kernel_types: set
) -> list:
    """Parse kernel_details.csv for diagnostic per-kernel breakdown.

    Excludes *AicpuKernel entries (AICPU dispatch wrappers that duplicate
    hcom_* communication kernels).
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
    tc_report: dict,
    profiler_output_dir: Path,
    step_index: int = 0,
    n_forward_passes_override: int = 0,
    delimiter: str = "ArgMaxV2",
) -> dict:
    """Compute M6: Empirical E2E Prediction Ratio.

    M6 = Empirical_hit_total / Real_per_fwd

    M6 = 1.0 is perfect. >1 = overestimate, <1 = underestimate.
    Only counts HIT ops with empirical (CSV) data; excludes analytic fallback.

    Args:
        tc_report: Parsed TC metrics JSON (from export_hit_miss_report)
        profiler_output_dir: ASCEND_PROFILER_OUTPUT directory
        step_index: Which step to use from step_trace_time.csv
        n_forward_passes_override: Directly specify N forward passes (0 = auto)
        delimiter: Kernel type for auto-detecting fwd pass boundaries

    Returns:
        dict with m6 ratio, components, and diagnostics
    """
    # 1. Empirical-only prediction (excludes analytic fallback for MISSes)
    m6_input = tc_report["m6_input"]
    empirical_hit_s = m6_input.get("empirical_hit_total_s")
    if empirical_hit_s is None:
        raise ValueError(
            "empirical_hit_total_s is missing in report. "
            "Re-run TC with updated --export-metrics to include it."
        )
    empirical_hit_us = empirical_hit_s * 1e6

    # 2. step_trace for reference
    step_info = parse_step_trace_time(profiler_output_dir, step_index)

    # 3. Estimate forward passes and per-fwd duration from kernel_details
    fwd_info = estimate_forward_passes(profiler_output_dir, delimiter)

    if n_forward_passes_override > 0:
        fwd_info["n_forward_passes"] = n_forward_passes_override
        fwd_info["delimiter"] = "user_override"
        fwd_info["avg_fwd_duration_us"] = (
            fwd_info["total_kernel_duration_us"] / n_forward_passes_override
        )

    real_per_fwd_us = fwd_info["avg_fwd_duration_us"]

    # Also store full TC prediction for reference
    tc_predicted_s = m6_input.get("tc_predicted_total_s")
    tc_predicted_us = tc_predicted_s * 1e6 if tc_predicted_s else None

    # 4. M6: ratio (1.0 = perfect, >1 overestimate, <1 underestimate)
    m6_ratio = empirical_hit_us / real_per_fwd_us if real_per_fwd_us > 0 else 0.0

    # 5. Diagnostics: unmatched kernels from kernel_details.csv
    hit_kernel_types = {h["kernel_type"] for h in tc_report.get("hits", [])}
    unmatched = parse_kernel_details_diagnostics(
        profiler_output_dir, hit_kernel_types
    )

    return {
        "m6_ratio": m6_ratio,
        "empirical_hit_us": empirical_hit_us,
        "tc_predicted_us": tc_predicted_us,
        "real_per_fwd_us": real_per_fwd_us,
        **step_info,
        **fwd_info,
        "unmatched_kernels": unmatched,
    }


def _format_report(result: dict) -> str:
    """Format M6 result as human-readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("M6: Empirical E2E Prediction Ratio")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        f"Empirical HIT total: {result['empirical_hit_us']:>12,.1f} us "
        f"({result['empirical_hit_us'] / 1e3:,.1f} ms)"
    )
    lines.append(
        f"Real per-fwd:        {result['real_per_fwd_us']:>12,.1f} us "
        f"({result['real_per_fwd_us'] / 1e3:,.1f} ms)"
    )
    if result.get("tc_predicted_us"):
        lines.append(
            f"TC full prediction:  {result['tc_predicted_us']:>12,.1f} us "
            f"({result['tc_predicted_us'] / 1e3:,.1f} ms)  [for reference]"
        )
    lines.append(
        f"  Step total:        {result['step_total_us']:>12,.1f} us "
        f"(Computing: {result['computing_us']:,.1f} "
        f"+ Comm: {result['comm_not_overlapped_us']:,.1f})"
    )
    lines.append(
        f"  Forward passes:    {result['n_forward_passes']:>12d}   "
        f"(delimiter: {result['delimiter']})"
    )
    lines.append("")
    m6 = result["m6_ratio"]
    lines.append(f"M6 = {m6:.3f}  (Empirical / Real)")

    if m6 > 1.15:
        verdict = f"overestimate by {(m6 - 1) * 100:.0f}%"
    elif m6 < 0.85:
        verdict = f"underestimate by {(1 - m6) * 100:.0f}%"
    else:
        verdict = "within ±15%"
    lines.append(f"     {verdict}")

    goal_status = "PASS" if 0.85 <= m6 <= 1.15 else "FAIL"
    lines.append(f"Phase 3 target: 0.85 ≤ M6 ≤ 1.15 [{goal_status}]")
    lines.append("")

    if result["unmatched_kernels"]:
        lines.append(
            "Top unmatched kernels "
            "(from kernel_details.csv, excl AicpuKernel):"
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
        description="Compute M6: Empirical E2E Prediction Ratio (offline). "
        "M6=1.0 is perfect; Phase 3 target 0.85-1.15. "
        "Empirical only, no analytic fallback."
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
        "--n-forward-passes",
        type=int,
        default=0,
        help="Directly specify N forward passes (0 = auto-detect via delimiter)",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default="ArgMaxV2",
        help="Kernel type that appears once per forward pass for auto-detection "
        "(default: ArgMaxV2, the sampling kernel)",
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

    result = compute_m6(
        tc_report,
        profiler_output_dir,
        step_index=args.step_index,
        n_forward_passes_override=args.n_forward_passes,
        delimiter=args.delimiter,
    )
    print(_format_report(result))

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nJSON output written to: {output_path}")


if __name__ == "__main__":
    main()
