#!/usr/bin/env python3
"""Analyze NPU profiling kernel_details.csv files for DSv3 and Qwen3-32B.

Produces summary tables with comm/compute classification, cumulative coverage,
CV analysis, and raw data export.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

BASE_DIR = Path(
    "/mnt/d/Data/Profiling/dsv3_qwen3_full_torch2.9.0_vllm0.15.0_cann8.5_eager"
)

MODELS = {
    "DSv3": BASE_DIR / "profilier_pd_dsv3" / "profilier_pd_dsv3",
    "Qwen3-32B": BASE_DIR / "profilier_pd_qwen32b" / "profilier_pd_qwen32b",
}

# Classification
COMM_OPS = {
    "hcom_allReduce_",
    "hcom_reduceScatter_",
    "hcom_allGather_",
    "hcom_alltoAll_",
    "reduce_scatterAicpuKernel",
    "allgatherAicpuKernel",
    "RINGMLAPrefillBF16Kernel",
}

FUSED_COMM_COMPUTE_OPS = {
    "DispatchFFNCombine",
}


def classify_op(op_type: str) -> str:
    if op_type in COMM_OPS:
        return "comm"
    if op_type in FUSED_COMM_COMPUTE_OPS:
        return "comm+compute fused"
    return "compute"


def load_model_data(model_dir: Path) -> pd.DataFrame:
    """Load all kernel_details.csv for a model, adding scenario column."""
    frames = []
    for scenario_dir in sorted(model_dir.iterdir()):
        csv_path = scenario_dir / "kernel_details.csv"
        if not csv_path.exists():
            continue
        scenario = scenario_dir.name  # e.g. profiler_decode_batch1
        df = pd.read_csv(csv_path)
        df["scenario"] = scenario
        frames.append(df)
    if not frames:
        raise ValueError(f"No kernel_details.csv found in {model_dir}")
    return pd.concat(frames, ignore_index=True)


def analyze_model(model_name: str, df: pd.DataFrame) -> dict:
    """Run full analysis for one model. Returns dict of result tables."""

    # Clean duration column
    df["Duration(us)"] = pd.to_numeric(df["Duration(us)"], errors="coerce")
    df = df.dropna(subset=["Duration(us)"])

    # Classify
    df["category"] = df["Type"].apply(classify_op)

    total_time = df["Duration(us)"].sum()
    compute_time = df.loc[df["category"] == "compute", "Duration(us)"].sum()
    comm_time = df.loc[df["category"] == "comm", "Duration(us)"].sum()
    fused_time = df.loc[df["category"] == "comm+compute fused", "Duration(us)"].sum()

    print(f"\n{'='*100}")
    print(f"  MODEL: {model_name}")
    print(f"{'='*100}")
    print(f"  Scenarios: {sorted(df['scenario'].unique())}")
    print(f"  Total kernel rows: {len(df):,}")
    print(f"  Total duration: {total_time:,.1f} us ({total_time/1e6:.3f} s)")
    print(f"  Compute: {compute_time:,.1f} us ({compute_time/total_time*100:.1f}%)")
    print(f"  Comm: {comm_time:,.1f} us ({comm_time/total_time*100:.1f}%)")
    print(f"  Fused: {fused_time:,.1f} us ({fused_time/total_time*100:.1f}%)")

    results = {}

    # ── Per-scenario breakdown ──
    scenario_rows = []
    for scenario in sorted(df["scenario"].unique()):
        sdf = df[df["scenario"] == scenario]
        s_total = sdf["Duration(us)"].sum()
        s_compute = sdf.loc[sdf["category"] == "compute", "Duration(us)"].sum()
        s_comm = sdf.loc[sdf["category"] == "comm", "Duration(us)"].sum()
        s_fused = sdf.loc[sdf["category"] == "comm+compute fused", "Duration(us)"].sum()
        scenario_rows.append({
            "scenario": scenario,
            "total_us": s_total,
            "compute_us": s_compute,
            "compute_pct": s_compute / s_total * 100 if s_total > 0 else 0,
            "comm_us": s_comm,
            "comm_pct": s_comm / s_total * 100 if s_total > 0 else 0,
            "fused_us": s_fused,
            "fused_pct": s_fused / s_total * 100 if s_total > 0 else 0,
            "n_kernels": len(sdf),
        })
    scenario_df = pd.DataFrame(scenario_rows)
    results["scenario_breakdown"] = scenario_df
    print(f"\n── Per-Scenario Breakdown ──")
    print(scenario_df.to_string(index=False))

    # ── COMM ops table ──
    comm_df = df[df["category"] == "comm"]
    comm_summary = []
    for op_type in sorted(comm_df["Type"].unique()):
        odf = comm_df[comm_df["Type"] == op_type]
        dur_sum = odf["Duration(us)"].sum()
        comm_summary.append({
            "OP Type": op_type,
            "category": "comm",
            "total_duration_us": dur_sum,
            "pct_of_total": dur_sum / total_time * 100,
            "n_invocations": len(odf),
            "mean_duration_us": odf["Duration(us)"].mean(),
            "std_duration_us": odf["Duration(us)"].std(),
        })

    # ── FUSED ops table ──
    fused_df = df[df["category"] == "comm+compute fused"]
    fused_summary = []
    for op_type in sorted(fused_df["Type"].unique()):
        odf = fused_df[fused_df["Type"] == op_type]
        dur_sum = odf["Duration(us)"].sum()
        fused_summary.append({
            "OP Type": op_type,
            "category": "comm+compute fused",
            "total_duration_us": dur_sum,
            "pct_of_total": dur_sum / total_time * 100,
            "n_invocations": len(odf),
            "mean_duration_us": odf["Duration(us)"].mean(),
            "std_duration_us": odf["Duration(us)"].std(),
        })

    # ── COMPUTE ops: top ops covering >99% ──
    compute_df = df[df["category"] == "compute"]
    op_durations = compute_df.groupby("Type")["Duration(us)"].sum().sort_values(ascending=False)
    cumsum = op_durations.cumsum()
    cumsum_pct = cumsum / compute_time * 100

    # Find cutoff: first index where cumsum_pct >= 99
    cutoff_idx = (cumsum_pct >= 99.0).idxmax()
    cutoff_pos = list(op_durations.index).index(cutoff_idx)
    top_ops = op_durations.iloc[: cutoff_pos + 1]

    compute_summary = []
    for op_type in top_ops.index:
        odf = compute_df[compute_df["Type"] == op_type]
        dur_sum = odf["Duration(us)"].sum()

        # CV analysis: group by (Type, Input Shapes, Input Data Types)
        # Each group is a unique "shape config"
        groups = odf.groupby(["Type", "Input Shapes", "Input Data Types"])["Duration(us)"]
        cvs = []
        for gname, gvals in groups:
            if len(gvals) >= 2:
                mean_g = gvals.mean()
                if mean_g > 0:
                    cv = gvals.std() / mean_g
                    cvs.append(cv)

        cv_min = min(cvs) if cvs else float("nan")
        cv_max = max(cvs) if cvs else float("nan")
        cv_median = np.median(cvs) if cvs else float("nan")
        n_shape_groups = len(list(groups))

        compute_summary.append({
            "OP Type": op_type,
            "category": "compute",
            "total_duration_us": dur_sum,
            "pct_of_total": dur_sum / total_time * 100,
            "pct_of_compute": dur_sum / compute_time * 100,
            "cumul_pct_compute": cumsum_pct[op_type],
            "n_invocations": len(odf),
            "mean_duration_us": odf["Duration(us)"].mean(),
            "std_duration_us": odf["Duration(us)"].std(),
            "n_shape_groups": n_shape_groups,
            "n_groups_with_cv": len(cvs),
            "cv_min": cv_min,
            "cv_median": cv_median,
            "cv_max": cv_max,
        })

    # Remaining compute ops
    remaining_ops = op_durations.iloc[cutoff_pos + 1 :]
    remaining_dur = remaining_ops.sum()
    remaining_pct = remaining_dur / compute_time * 100

    # ── Print tables ──
    print(f"\n── Communication Ops ──")
    comm_tbl = pd.DataFrame(comm_summary)
    if not comm_tbl.empty:
        comm_tbl = comm_tbl.sort_values("total_duration_us", ascending=False)
        print(comm_tbl.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    else:
        print("  (none)")

    print(f"\n── Fused Comm+Compute Ops ──")
    fused_tbl = pd.DataFrame(fused_summary)
    if not fused_tbl.empty:
        print(fused_tbl.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    else:
        print("  (none)")

    print(f"\n── Compute Ops (top ops covering >=99% of compute time) ──")
    compute_tbl = pd.DataFrame(compute_summary)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    print(compute_tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n  Remaining {len(remaining_ops)} compute op types: "
          f"{remaining_dur:,.1f} us ({remaining_pct:.2f}% of compute)")
    if len(remaining_ops) > 0:
        print(f"  Remaining ops: {', '.join(remaining_ops.index.tolist())}")

    results["comm"] = comm_tbl
    results["fused"] = fused_tbl
    results["compute"] = compute_tbl
    results["remaining_compute_ops"] = remaining_ops
    results["totals"] = {
        "total_time_us": total_time,
        "compute_time_us": compute_time,
        "comm_time_us": comm_time,
        "fused_time_us": fused_time,
    }

    # ── Per-scenario per-op breakdown (raw data) ──
    # For each scenario, show top ops
    print(f"\n── Per-Scenario Top Op Breakdown ──")
    for scenario in sorted(df["scenario"].unique()):
        sdf = df[df["scenario"] == scenario]
        s_total = sdf["Duration(us)"].sum()
        s_by_type = sdf.groupby(["Type", "category"])["Duration(us)"].agg(["sum", "count", "mean"]).reset_index()
        s_by_type.columns = ["Type", "category", "total_us", "count", "mean_us"]
        s_by_type["pct"] = s_by_type["total_us"] / s_total * 100
        s_by_type = s_by_type.sort_values("total_us", ascending=False)
        print(f"\n  {scenario} (total: {s_total:,.1f} us)")
        print(s_by_type.head(15).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # ── CV detail for compute ops ──
    print(f"\n── CV Detail for Compute Ops ──")
    for row in compute_summary:
        op = row["OP Type"]
        odf = compute_df[compute_df["Type"] == op]
        groups = odf.groupby(["Input Shapes", "Input Data Types"])["Duration(us)"]
        print(f"\n  {op}: {row['n_shape_groups']} shape groups, "
              f"{row['n_groups_with_cv']} with CV")
        group_stats = []
        for (shapes, dtypes), gvals in groups:
            n = len(gvals)
            mean_v = gvals.mean()
            std_v = gvals.std() if n >= 2 else 0
            cv = std_v / mean_v if mean_v > 0 and n >= 2 else float("nan")
            total_v = gvals.sum()
            group_stats.append({
                "Input Shapes": shapes[:60] if len(str(shapes)) > 60 else shapes,
                "Input Data Types": dtypes[:40] if len(str(dtypes)) > 40 else dtypes,
                "n": n,
                "mean_us": mean_v,
                "std_us": std_v,
                "cv": cv,
                "total_us": total_v,
            })
        gdf = pd.DataFrame(group_stats)
        if gdf.empty:
            print("    (no shape groups with data)")
            continue
        gdf = gdf.sort_values("total_us", ascending=False)
        print(gdf.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return results


def main():
    all_results = {}
    for model_name, model_dir in MODELS.items():
        if not model_dir.exists():
            print(f"SKIP {model_name}: {model_dir} not found")
            continue
        df = load_model_data(model_dir)
        all_results[model_name] = analyze_model(model_name, df)

    # ── Export raw CSV ──
    out_dir = Path("/home/horacehxw/Projects/msmodeling-perf-db-profiling-analysis/profiling_analysis_output")
    out_dir.mkdir(exist_ok=True)

    for model_name, res in all_results.items():
        prefix = model_name.replace("-", "").replace(" ", "_").lower()
        res["comm"].to_csv(out_dir / f"{prefix}_comm_ops.csv", index=False)
        res["fused"].to_csv(out_dir / f"{prefix}_fused_ops.csv", index=False)
        res["compute"].to_csv(out_dir / f"{prefix}_compute_ops.csv", index=False)
        res["scenario_breakdown"].to_csv(out_dir / f"{prefix}_scenario_breakdown.csv", index=False)
        print(f"\nExported {model_name} CSVs to {out_dir}")

    # ── Combined summary table (for report) ──
    print(f"\n\n{'='*120}")
    print("  COMBINED SUMMARY FOR REPORT")
    print(f"{'='*120}")
    for model_name, res in all_results.items():
        totals = res["totals"]
        print(f"\n{'─'*80}")
        print(f"  {model_name}")
        print(f"  Total: {totals['total_time_us']/1e6:.3f}s | "
              f"Compute: {totals['compute_time_us']/totals['total_time_us']*100:.1f}% | "
              f"Comm: {totals['comm_time_us']/totals['total_time_us']*100:.1f}% | "
              f"Fused: {totals['fused_time_us']/totals['total_time_us']*100:.1f}%")
        print(f"{'─'*80}")

        # Unified table
        rows = []
        for _, r in res["comm"].iterrows():
            rows.append({
                "OP Type": r["OP Type"],
                "Category": "comm",
                "Total(us)": r["total_duration_us"],
                "% Total": r["pct_of_total"],
                "% Compute": "",
                "Invocations": int(r["n_invocations"]),
                "Mean(us)": r["mean_duration_us"],
                "CV Range": "",
            })
        for _, r in res["fused"].iterrows():
            rows.append({
                "OP Type": r["OP Type"],
                "Category": "fused",
                "Total(us)": r["total_duration_us"],
                "% Total": r["pct_of_total"],
                "% Compute": "",
                "Invocations": int(r["n_invocations"]),
                "Mean(us)": r["mean_duration_us"],
                "CV Range": "",
            })
        for _, r in res["compute"].iterrows():
            cv_range = (f"{r['cv_min']:.3f}-{r['cv_max']:.3f}"
                        if not np.isnan(r["cv_min"]) else "N/A")
            rows.append({
                "OP Type": r["OP Type"],
                "Category": "compute",
                "Total(us)": r["total_duration_us"],
                "% Total": r["pct_of_total"],
                "% Compute": f"{r['pct_of_compute']:.2f}",
                "Invocations": int(r["n_invocations"]),
                "Mean(us)": r["mean_duration_us"],
                "CV Range": cv_range,
            })

        summary = pd.DataFrame(rows)
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
