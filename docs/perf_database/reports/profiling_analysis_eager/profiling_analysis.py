#!/usr/bin/env python3
"""Profiling data analysis for DSv3 & Qwen3-32B (CANN 8.5 / vLLM 0.15.0 / torch 2.9.0).

Analyzes kernel_details.csv for:
- Q2: Top ops by duration, 99% cumulative time
- Q3: kernel sum vs step_trace e2e gap
- Q4: Op consistency (CV, avg, std) for same input shapes
- Q5: Outlier deep-dive with aiv/aic/mte breakdown
"""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path("/mnt/d/Data/Profiling/dsv3_qwen3_full_torch2.9.0_vllm0.15.0_cann8.5_eager")
DSV3_DIR = BASE / "profilier_pd_dsv3" / "profilier_pd_dsv3"
QWEN3_DIR = BASE / "profilier_pd_qwen32b" / "profilier_pd_qwen32b"

# Known comm ops
COMM_OPS = {
    "hcom_allReduce_", "hcom_allGather_", "hcom_alltoallv_",
    "hcom_reduceScatter_", "HcomReduceScatter",
    "allgatherAicpuKernel", "reduce_scatterAicpuKernel",
    "RINGMLAPrefillBF16Kernel",
}

# Known comm+compute fused ops
COMM_COMPUTE_OPS = {
    "DispatchFFNCombine",
}


def read_kernel_details(csv_path):
    """Read kernel_details.csv and return list of dicts."""
    def safe_float(val, default=0.0):
        if val is None or val == "" or val == "N/A":
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dur = safe_float(row.get("Duration(us)"))
            if dur <= 0:
                continue
            rows.append({
                "name": row.get("Name", ""),
                "type": row.get("Type", ""),
                "core": row.get("Accelerator Core", ""),
                "duration": dur,
                "input_shapes": row.get("Input Shapes", ""),
                "input_dtypes": row.get("Input Data Types", ""),
                "input_formats": row.get("Input Formats", ""),
                "output_shapes": row.get("Output Shapes", ""),
                # AIC breakdown
                "aic_mac_time": safe_float(row.get("aic_mac_time(us)")),
                "aic_scalar_time": safe_float(row.get("aic_scalar_time(us)")),
                "aic_mte1_time": safe_float(row.get("aic_mte1_time(us)")),
                "aic_mte2_time": safe_float(row.get("aic_mte2_time(us)")),
                "aic_fixpipe_time": safe_float(row.get("aic_fixpipe_time(us)")),
                "aicore_time": safe_float(row.get("aicore_time(us)")),
                # AIV breakdown
                "aiv_time": safe_float(row.get("aiv_time(us)")),
                "aiv_vec_time": safe_float(row.get("aiv_vec_time(us)")),
                "aiv_scalar_time": safe_float(row.get("aiv_scalar_time(us)")),
                "aiv_mte2_time": safe_float(row.get("aiv_mte2_time(us)")),
                "aiv_mte3_time": safe_float(row.get("aiv_mte3_time(us)")),
                "cube_utilization": safe_float(row.get("cube_utilization(%)")),
                "wait_time": safe_float(row.get("Wait Time(us)")),
            })
    return rows


def read_step_trace(csv_path):
    """Read step_trace_time.csv."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return {
                "computing": float(row.get("Computing", 0)),
                "comm_not_overlapped": float(row.get("Communication(Not Overlapped)", 0)),
                "overlapped": float(row.get("Overlapped", 0)),
                "communication": float(row.get("Communication", 0)),
                "free": float(row.get("Free", 0)),
                "stage": float(row.get("Stage", 0)),
            }
    return {}


def classify_op(op_type):
    """Classify op as comm, comm_compute, or compute."""
    if op_type in COMM_OPS:
        return "comm"
    if op_type in COMM_COMPUTE_OPS:
        return "comm_compute"
    return "compute"


def analyze_scenario(scenario_dir, scenario_name):
    """Full analysis for one scenario."""
    kernel_csv = scenario_dir / "kernel_details.csv"
    step_csv = scenario_dir / "step_trace_time.csv"

    if not kernel_csv.exists():
        print(f"  SKIP: {kernel_csv} not found")
        return None

    rows = read_kernel_details(kernel_csv)
    step = read_step_trace(step_csv) if step_csv.exists() else {}

    # Group by Type (op type)
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    total_kernel_time = sum(r["duration"] for r in rows)

    # Q2: Top ops by duration
    type_summary = []
    for op_type, op_rows in by_type.items():
        total = sum(r["duration"] for r in op_rows)
        cat = classify_op(op_type)
        type_summary.append({
            "type": op_type,
            "category": cat,
            "count": len(op_rows),
            "total_us": total,
            "pct": total / total_kernel_time * 100 if total_kernel_time > 0 else 0,
            "avg_us": total / len(op_rows),
            "min_us": min(r["duration"] for r in op_rows),
            "max_us": max(r["duration"] for r in op_rows),
        })
    type_summary.sort(key=lambda x: x["total_us"], reverse=True)

    # Q3: Gap analysis
    step_e2e = step.get("stage", 0) if step else 0

    # Q4: Consistency analysis for important ops (top ops by %)
    consistency = []
    important_types = [s["type"] for s in type_summary if s["pct"] >= 0.05]  # >0.05% of time

    for op_type in important_types:
        op_rows = by_type[op_type]
        # Group by (input_shapes, input_dtypes) to find same-shape instances
        shape_groups = defaultdict(list)
        for r in op_rows:
            key = (r["input_shapes"], r["input_dtypes"])
            shape_groups[key].append(r)

        for (shapes, dtypes), group in shape_groups.items():
            if len(group) < 3:  # Need at least 3 for meaningful stats
                continue
            durations = [r["duration"] for r in group]
            arr = np.array(durations)
            mean = np.mean(arr)
            std = np.std(arr)
            cv = std / mean if mean > 0 else 0
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            outlier_count = np.sum((arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr))

            consistency.append({
                "type": op_type,
                "category": classify_op(op_type),
                "shapes": shapes,
                "dtypes": dtypes,
                "count": len(group),
                "mean_us": mean,
                "std_us": std,
                "cv": cv,
                "min_us": np.min(arr),
                "max_us": np.max(arr),
                "p50_us": np.median(arr),
                "outlier_count": int(outlier_count),
                "pct_of_total": sum(durations) / total_kernel_time * 100,
            })

    consistency.sort(key=lambda x: x["pct_of_total"], reverse=True)

    return {
        "scenario": scenario_name,
        "total_kernels": len(rows),
        "total_kernel_time_us": total_kernel_time,
        "step_e2e_us": step_e2e,
        "step": step,
        "type_summary": type_summary,
        "consistency": consistency,
        "by_type": by_type,  # Keep raw data for Q5
    }


def print_top_ops(result, top_n=15):
    """Print top ops table."""
    print(f"\n{'='*90}")
    print(f"  {result['scenario']}")
    print(f"  Total kernels: {result['total_kernels']}, Total kernel time: {result['total_kernel_time_us']:.0f} us")
    if result['step']:
        s = result['step']
        e2e = s.get('stage', 0)
        print(f"  Step trace e2e: {e2e:.0f} us (computing={s['computing']:.0f}, "
              f"comm_no_overlap={s['comm_not_overlapped']:.0f}, free={s['free']:.0f})")
        gap = e2e - result['total_kernel_time_us']
        gap_pct = gap / e2e * 100 if e2e > 0 else 0
        print(f"  GAP (step_e2e - kernel_sum): {gap:.0f} us ({gap_pct:.1f}%)")
    print(f"{'='*90}")

    print(f"\n  Top {top_n} ops by duration:")
    print(f"  {'Type':<35s} {'Cat':<12s} {'Count':>6s} {'Total(us)':>12s} {'Pct':>7s} {'Avg(us)':>10s} {'Max(us)':>10s}")
    print(f"  {'-'*35} {'-'*12} {'-'*6} {'-'*12} {'-'*7} {'-'*10} {'-'*10}")

    cum_pct = 0
    for i, s in enumerate(result['type_summary'][:top_n]):
        cum_pct += s['pct']
        marker = " *" if cum_pct <= 99 or i == 0 else ""
        print(f"  {s['type']:<35s} {s['category']:<12s} {s['count']:>6d} {s['total_us']:>12.1f} {s['pct']:>6.2f}% {s['avg_us']:>10.1f} {s['max_us']:>10.1f}{marker}")

    # Find where 99% threshold is (excluding comm and comm+compute)
    compute_ops = [s for s in result['type_summary'] if s['category'] == 'compute']
    compute_total = sum(s['total_us'] for s in compute_ops)
    cum = 0
    print(f"\n  99% compute-only ops (excl. comm & comm+compute, compute_total={compute_total:.0f} us):")
    for s in compute_ops:
        cum += s['total_us']
        pct_of_compute = s['total_us'] / compute_total * 100 if compute_total > 0 else 0
        if pct_of_compute >= 0.1:  # Show ops > 0.1% of compute time
            print(f"    {s['type']:<35s} {s['count']:>6d} {s['total_us']:>12.1f} us ({pct_of_compute:.1f}% compute, cum {cum/compute_total*100:.1f}%)")
        if cum / compute_total >= 0.99:
            break


def print_consistency(result, top_n=20):
    """Print consistency analysis."""
    print(f"\n  Top {top_n} shape groups by time contribution (CV analysis):")
    print(f"  {'Type':<30s} {'Cat':<10s} {'N':>5s} {'Mean(us)':>10s} {'Std':>8s} {'CV':>7s} {'Min':>8s} {'Max':>8s} {'Out':>4s} {'%Tot':>6s}")
    print(f"  {'-'*30} {'-'*10} {'-'*5} {'-'*10} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*4} {'-'*6}")

    high_cv_ops = []
    for c in result['consistency'][:top_n]:
        flag = " !!" if c['cv'] > 0.1 else ""
        print(f"  {c['type']:<30s} {c['category']:<10s} {c['count']:>5d} {c['mean_us']:>10.2f} {c['std_us']:>8.2f} {c['cv']:>7.3f} {c['min_us']:>8.2f} {c['max_us']:>8.2f} {c['outlier_count']:>4d} {c['pct_of_total']:>5.2f}%{flag}")
        if c['cv'] > 0.1 and c['pct_of_total'] > 0.05:
            high_cv_ops.append(c)

    return high_cv_ops


def deep_dive_outliers(result, high_cv_ops):
    """Q5: Deep-dive into ops with high CV, analyzing aiv/aic/mte breakdown."""
    if not high_cv_ops:
        print("\n  No high-CV ops found (CV > 0.1 and >0.05% of total time).")
        return

    print(f"\n  {'='*80}")
    print(f"  OUTLIER DEEP-DIVE (ops with CV > 0.1)")
    print(f"  {'='*80}")

    by_type = result['by_type']
    for hc in high_cv_ops[:10]:
        op_type = hc['type']
        shapes = hc['shapes']
        dtypes = hc['dtypes']

        # Get matching rows
        matching = [r for r in by_type[op_type]
                    if r['input_shapes'] == shapes and r['input_dtypes'] == dtypes]

        durations = np.array([r['duration'] for r in matching])
        mean_d = np.mean(durations)

        print(f"\n  --- {op_type} | shapes={shapes[:60]} | N={len(matching)} ---")
        print(f"  Duration: mean={mean_d:.2f}, std={np.std(durations):.2f}, "
              f"min={np.min(durations):.2f}, max={np.max(durations):.2f}, CV={hc['cv']:.3f}")

        # Analyze timing breakdown
        # For AIV ops
        if matching[0]['core'] in ('AI_VECTOR_CORE', 'MIX_AIV'):
            aiv_times = np.array([r['aiv_time'] for r in matching])
            vec_times = np.array([r['aiv_vec_time'] for r in matching])
            scalar_times = np.array([r['aiv_scalar_time'] for r in matching])
            mte2_times = np.array([r['aiv_mte2_time'] for r in matching])
            mte3_times = np.array([r['aiv_mte3_time'] for r in matching])
            wait_times = np.array([r['wait_time'] for r in matching])

            print(f"  Core: {matching[0]['core']}")
            print(f"  AIV time:    mean={np.mean(aiv_times):.2f}, std={np.std(aiv_times):.2f}, CV={np.std(aiv_times)/np.mean(aiv_times):.3f}" if np.mean(aiv_times) > 0 else "  AIV time: 0")
            print(f"    vec:       mean={np.mean(vec_times):.2f}, std={np.std(vec_times):.2f}")
            print(f"    scalar:    mean={np.mean(scalar_times):.2f}, std={np.std(scalar_times):.2f}")
            print(f"    mte2:      mean={np.mean(mte2_times):.2f}, std={np.std(mte2_times):.2f}")
            print(f"    mte3:      mean={np.mean(mte3_times):.2f}, std={np.std(mte3_times):.2f}")
            print(f"  Wait time:   mean={np.mean(wait_times):.2f}, std={np.std(wait_times):.2f}, CV={np.std(wait_times)/np.mean(wait_times):.3f}" if np.mean(wait_times) > 0 else "  Wait time: 0")

        # For AIC/MIX_AIC ops
        elif matching[0]['core'] in ('AI_CORE', 'MIX_AIC'):
            aic_times = np.array([r['aicore_time'] for r in matching])
            mac_times = np.array([r['aic_mac_time'] for r in matching])
            scalar_times = np.array([r['aic_scalar_time'] for r in matching])
            mte1_times = np.array([r['aic_mte1_time'] for r in matching])
            mte2_times = np.array([r['aic_mte2_time'] for r in matching])
            wait_times = np.array([r['wait_time'] for r in matching])
            # Also check AIV part for MIX ops
            aiv_times = np.array([r['aiv_time'] for r in matching])

            print(f"  Core: {matching[0]['core']}")
            print(f"  AIC time:    mean={np.mean(aic_times):.2f}, std={np.std(aic_times):.2f}, CV={np.std(aic_times)/np.mean(aic_times):.3f}" if np.mean(aic_times) > 0 else "  AIC time: 0")
            print(f"    mac:       mean={np.mean(mac_times):.2f}, std={np.std(mac_times):.2f}")
            print(f"    scalar:    mean={np.mean(scalar_times):.2f}, std={np.std(scalar_times):.2f}")
            print(f"    mte1:      mean={np.mean(mte1_times):.2f}, std={np.std(mte1_times):.2f}")
            print(f"    mte2:      mean={np.mean(mte2_times):.2f}, std={np.std(mte2_times):.2f}")
            if np.mean(aiv_times) > 0:
                print(f"  AIV time:    mean={np.mean(aiv_times):.2f}, std={np.std(aiv_times):.2f}")
            print(f"  Wait time:   mean={np.mean(wait_times):.2f}, std={np.std(wait_times):.2f}, CV={np.std(wait_times)/np.mean(wait_times):.3f}" if np.mean(wait_times) > 0 else "  Wait time: 0")

        # Show a few outlier examples
        sorted_idx = np.argsort(durations)
        print(f"  Fastest 3: {durations[sorted_idx[:3]]}")
        print(f"  Slowest 3: {durations[sorted_idx[-3:]]}")

        # Check if wait time correlates with outliers
        if len(matching) > 5:
            wait_times = np.array([r['wait_time'] for r in matching])
            corr = np.corrcoef(durations, wait_times)[0, 1] if np.std(wait_times) > 0 else 0
            print(f"  Correlation(duration, wait_time): {corr:.3f}")


def main():
    models = [
        ("DSv3", DSV3_DIR),
        ("Qwen3-32B", QWEN3_DIR),
    ]

    for model_name, model_dir in models:
        print(f"\n{'#'*100}")
        print(f"  MODEL: {model_name}")
        print(f"{'#'*100}")

        scenarios = sorted(model_dir.iterdir())
        for scenario_dir in scenarios:
            if not scenario_dir.is_dir():
                continue
            scenario_name = f"{model_name}/{scenario_dir.name}"
            result = analyze_scenario(scenario_dir, scenario_name)
            if result is None:
                continue

            print_top_ops(result)
            high_cv_ops = print_consistency(result)
            deep_dive_outliers(result, high_cv_ops)


if __name__ == "__main__":
    main()

