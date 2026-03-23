"""Per-shape stability analysis on Phase 1 profiling data.

For each scenario (DSv3 prefill/decode, Qwen3 prefill/decode):
1. Filter to compute kernels only
2. Group by (Name, Input Shapes) for per-shape analysis
3. Compute count, mean, std, CV for groups with count >= 5
4. Show top 15 operators by total duration with all shape groups
5. Summary table with min/max/median CV per operator
"""

import sys
import io
import pandas as pd
import numpy as np

# ── Data paths ──────────────────────────────────────────────────────────────
SCENARIOS = {
    "DSv3 Prefill (input4096, output1)": {
        "path": "/mnt/d/Data/Profiling/Profiling-0313-phase1-e2e-test/profilier_prefill_dsv3-input4096-output1/e57aa6f6d21d_746902_20260313152307903_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv",
        "comm_streams": {},  # no special stream-based comm exclusion
    },
    "DSv3 Decode (input4096, output1536)": {
        "path": "/mnt/d/Data/Profiling/Profiling-0313-phase1-e2e-test/profilier_decode_dsv3-input4096-output1536/e57aa6f6d21d_746902_20260313153338705_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv",
        "comm_streams": {37: "AivKernel"},  # Stream 37 AivKernel = comm
    },
    "Qwen3-32B Prefill (input4096, output1)": {
        "path": "/mnt/d/Data/Profiling/Profiling-0313-phase1-e2e-test/profilier_prefill_qwen32b-input4096-output1/e57aa6f6d21d_729199_20260313134512603_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv",
        "comm_streams": {},
    },
    "Qwen3-32B Decode (input4k)": {
        "path": "/mnt/d/Data/Profiling/Profiling-0313-phase1-e2e-test/profilier_decode_qwen32b-input4k/e57aa6f6d21d_631368_20260313101831169_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv",
        "comm_streams": {149: "AivKernel"},  # Stream 149 AivKernel = comm
    },
}

# Kernel name prefixes/patterns to exclude as communication
COMM_KERNEL_PREFIXES = [
    "hcom_",
    "reduce_scatterAicpuKernel",
    "allgatherAicpuKernel",
    "DispatchFFNCombine",
    "RINGMLAPrefillBF16Kernel",
]


def load_and_filter(path: str, comm_streams: dict) -> pd.DataFrame:
    """Load kernel_details.csv and filter to compute kernels only."""
    df = pd.read_csv(path)

    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # 1. Exclude rows with NaN Stream ID
    df = df.dropna(subset=["Stream ID"])

    # 2. Exclude communication kernels by name
    for prefix in COMM_KERNEL_PREFIXES:
        df = df[~df["Name"].str.startswith(prefix, na=False)]

    # 3. Exclude stream-based comm kernels
    for stream_id, kernel_type in comm_streams.items():
        mask = (df["Stream ID"] == stream_id) & df["Name"].str.contains(
            kernel_type, na=False
        )
        df = df[~mask]

    return df


def analyze_scenario(name: str, cfg: dict, out: io.StringIO):
    """Run per-shape stability analysis for one scenario."""
    out.write(f"\n{'='*100}\n")
    out.write(f"  {name}\n")
    out.write(f"{'='*100}\n\n")

    df = load_and_filter(cfg["path"], cfg["comm_streams"])
    out.write(f"Total compute kernels after filtering: {len(df)}\n\n")

    # Clean Duration column
    df["Duration(us)"] = pd.to_numeric(df["Duration(us)"], errors="coerce")

    # Group by (Name, Input Shapes)
    grouped = df.groupby(["Name", "Input Shapes"])

    # Compute stats for groups with count >= 5
    records = []
    for (op_name, shape), grp in grouped:
        dur = grp["Duration(us)"].dropna()
        if len(dur) < 5:
            continue
        mean_dur = dur.mean()
        std_dur = dur.std()
        cv = std_dur / mean_dur if mean_dur > 0 else np.nan
        total_dur = dur.sum()
        records.append(
            {
                "Name": op_name,
                "Input Shapes": shape,
                "Count": len(dur),
                "Mean(us)": mean_dur,
                "Std(us)": std_dur,
                "CV": cv,
                "Total(us)": total_dur,
            }
        )

    stats_df = pd.DataFrame(records)
    if stats_df.empty:
        out.write("No groups with count >= 5 found.\n")
        return

    # ── Top 15 operators by total duration ──────────────────────────────
    op_total = stats_df.groupby("Name")["Total(us)"].sum().sort_values(ascending=False)
    top15_ops = op_total.head(15).index.tolist()

    out.write(f"{'─'*100}\n")
    out.write(f"  TOP 15 OPERATORS BY TOTAL DURATION — Per-Shape CV Breakdown\n")
    out.write(f"{'─'*100}\n\n")

    for rank, op in enumerate(top15_ops, 1):
        op_rows = stats_df[stats_df["Name"] == op].sort_values(
            "Total(us)", ascending=False
        )
        total = op_rows["Total(us)"].sum()
        out.write(
            f"  #{rank}  {op}  (total={total:,.1f} us, {len(op_rows)} shape groups)\n"
        )
        out.write(
            f"  {'Count':>7s}  {'Mean(us)':>12s}  {'Std(us)':>12s}  {'CV':>8s}  {'Total(us)':>14s}  {'Input Shapes'}\n"
        )
        for _, row in op_rows.iterrows():
            flag = " *** HIGH CV" if row["CV"] > 0.1 else ""
            shape_str = str(row["Input Shapes"])
            if len(shape_str) > 100:
                shape_str = shape_str[:97] + "..."
            out.write(
                f"  {row['Count']:>7d}  {row['Mean(us)']:>12.3f}  {row['Std(us)']:>12.3f}  {row['CV']:>8.4f}  {row['Total(us)']:>14.1f}  {shape_str}{flag}\n"
            )
        out.write("\n")

    # ── Summary table: per-operator CV statistics ───────────────────────
    out.write(f"{'─'*100}\n")
    out.write(f"  OPERATOR CV SUMMARY (all operators with count>=5 shape groups)\n")
    out.write(f"{'─'*100}\n\n")

    op_cv_summary = (
        stats_df.groupby("Name")
        .agg(
            num_shape_groups=("CV", "count"),
            min_CV=("CV", "min"),
            median_CV=("CV", "median"),
            max_CV=("CV", "max"),
            total_dur=("Total(us)", "sum"),
        )
        .sort_values("total_dur", ascending=False)
    )

    out.write(
        f"  {'Operator':<60s} {'#Shapes':>7s} {'MinCV':>8s} {'MedCV':>8s} {'MaxCV':>8s} {'Total(us)':>14s}\n"
    )
    out.write(f"  {'─'*60} {'─'*7} {'─'*8} {'─'*8} {'─'*8} {'─'*14}\n")
    for op, row in op_cv_summary.iterrows():
        op_display = op if len(op) <= 60 else op[:57] + "..."
        flag = " ***" if row["max_CV"] > 0.1 else ""
        out.write(
            f"  {op_display:<60s} {row['num_shape_groups']:>7.0f} {row['min_CV']:>8.4f} {row['median_CV']:>8.4f} {row['max_CV']:>8.4f} {row['total_dur']:>14.1f}{flag}\n"
        )

    out.write("\n")

    # ── High CV flagged groups ──────────────────────────────────────────
    high_cv = stats_df[stats_df["CV"] > 0.1].sort_values("Total(us)", ascending=False)
    out.write(f"{'─'*100}\n")
    out.write(
        f"  HIGH CV GROUPS (CV > 0.1): {len(high_cv)} of {len(stats_df)} total shape groups\n"
    )
    out.write(f"{'─'*100}\n\n")

    if not high_cv.empty:
        out.write(
            f"  {'Name':<50s} {'Count':>6s} {'Mean(us)':>12s} {'CV':>8s} {'Total(us)':>14s}  Input Shapes\n"
        )
        out.write(f"  {'─'*50} {'─'*6} {'─'*12} {'─'*8} {'─'*14}  {'─'*40}\n")
        for _, row in high_cv.head(30).iterrows():
            op_display = (
                row["Name"] if len(row["Name"]) <= 50 else row["Name"][:47] + "..."
            )
            shape_str = str(row["Input Shapes"])
            if len(shape_str) > 80:
                shape_str = shape_str[:77] + "..."
            out.write(
                f"  {op_display:<50s} {row['Count']:>6d} {row['Mean(us)']:>12.3f} {row['CV']:>8.4f} {row['Total(us)']:>14.1f}  {shape_str}\n"
            )
    out.write("\n")


def main():
    buf = io.StringIO()
    buf.write("=" * 100 + "\n")
    buf.write("  PER-SHAPE STABILITY ANALYSIS — Phase 1 Profiling Data\n")
    buf.write("  Groups: (Name, Input Shapes), min count = 5\n")
    buf.write("=" * 100 + "\n")

    for scenario_name, cfg in SCENARIOS.items():
        analyze_scenario(scenario_name, cfg, buf)

    output = buf.getvalue()
    print(output)

    out_path = "/home/horacehxw/Projects/msmodeling-perf-db-profiling-analysis/docs/perf_database/reports/profiling_analysis_phase1/per_shape_cv_output.txt"
    with open(out_path, "w") as f:
        f.write(output)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
