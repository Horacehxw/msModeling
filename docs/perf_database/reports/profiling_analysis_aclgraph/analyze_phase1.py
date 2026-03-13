#!/usr/bin/env python3
"""Analyze NPU profiling kernel_details.csv for Phase 1 e2e test scenarios."""

import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
from io import StringIO

BASE = Path("/mnt/d/Data/Profiling/Profiling-0313-phase1-e2e-test")

SCENARIOS = {
    "DSv3 Prefill (input4096, output1)": (
        "profilier_prefill_dsv3-input4096-output1/"
        "e57aa6f6d21d_746902_20260313152307903_ascend_pt/ASCEND_PROFILER_OUTPUT"
    ),
    "DSv3 Decode (input4096, output1536)": (
        "profilier_decode_dsv3-input4096-output1536/"
        "e57aa6f6d21d_746902_20260313153338705_ascend_pt/ASCEND_PROFILER_OUTPUT"
    ),
    "Qwen3-32B Prefill (input4096, output1)": (
        "profilier_prefill_qwen32b-input4096-output1/"
        "e57aa6f6d21d_729199_20260313134512603_ascend_pt/ASCEND_PROFILER_OUTPUT"
    ),
    "Qwen3-32B Decode (input4k)": (
        "profilier_decode_qwen32b-input4k/"
        "e57aa6f6d21d_631368_20260313101831169_ascend_pt/ASCEND_PROFILER_OUTPUT"
    ),
}

COMM_PATTERNS = re.compile(
    r"hcom_|allReduce|reduceScatter|allGather|alltoall|RING|reduce_scatter|allgather",
    re.IGNORECASE,
)
FUSED_COMPUTE_COMM_PATTERNS = re.compile(r"DispatchFFNCombine", re.IGNORECASE)

SEP = "=" * 90


def is_comm_kernel(name: str, stream_id) -> bool:
    """Communication kernel: matches pattern OR has NaN Stream ID."""
    if pd.isna(stream_id):
        return True
    if isinstance(name, str) and COMM_PATTERNS.search(name):
        return True
    return False


def is_fused_compute_comm(name: str) -> bool:
    if isinstance(name, str) and FUSED_COMPUTE_COMM_PATTERNS.search(name):
        return True
    return False


def analyze_scenario(name: str, profiler_dir: Path, out):
    def p(s=""):
        out.write(str(s) + "\n")

    p(f"\n{SEP}")
    p(f"  SCENARIO: {name}")
    p(SEP)

    kernel_csv = profiler_dir / "kernel_details.csv"
    step_csv = profiler_dir / "step_trace_time.csv"

    if not kernel_csv.exists():
        p(f"  [ERROR] kernel_details.csv not found: {kernel_csv}")
        return

    df = pd.read_csv(kernel_csv)
    # Clean up column: Start Time may have trailing tab
    if "Start Time(us)" in df.columns:
        df["Start Time(us)"] = pd.to_numeric(
            df["Start Time(us)"].astype(str).str.strip(), errors="coerce"
        )
    # Stream ID: keep as-is (may have NaN)
    df["Stream ID"] = pd.to_numeric(df["Stream ID"], errors="coerce")

    total_rows = len(df)
    p(f"\n  Total kernel rows: {total_rows}")
    p(f"  Columns: {list(df.columns[:10])}...")

    # ── Step 2: Stream ID analysis ───────────────────────────────────────
    p(f"\n  {'─'*60}")
    p("  STREAM ID ANALYSIS")
    p(f"  {'─'*60}")

    nan_stream = df["Stream ID"].isna()
    p(f"  Rows with NaN Stream ID: {nan_stream.sum()}")

    streams = df.loc[~nan_stream, "Stream ID"].unique()
    streams = sorted(streams)
    p(f"  Unique valid Stream IDs: {streams}")

    for sid in streams:
        mask = df["Stream ID"] == sid
        count = mask.sum()
        sample_names = df.loc[mask, "Name"].unique()[:5]
        p(f"\n    Stream {int(sid)}: {count} kernels")
        for n in sample_names:
            p(f"      - {n}")

    # NaN stream samples
    if nan_stream.sum() > 0:
        sample_names = df.loc[nan_stream, "Name"].unique()[:5]
        p(f"\n    Stream NaN: {nan_stream.sum()} kernels")
        for n in sample_names:
            p(f"      - {n}")

    # ── Step 3: Classify kernels ─────────────────────────────────────────
    p(f"\n  {'─'*60}")
    p("  KERNEL CLASSIFICATION")
    p(f"  {'─'*60}")

    fused_mask = df["Name"].apply(lambda x: is_fused_compute_comm(x) if isinstance(x, str) else False)
    comm_mask = df.apply(lambda r: is_comm_kernel(r["Name"], r["Stream ID"]), axis=1) & ~fused_mask
    compute_mask = ~comm_mask & ~fused_mask

    p(f"  Compute kernels:          {compute_mask.sum()}")
    p(f"  Communication kernels:    {comm_mask.sum()}")
    p(f"  Fused compute+comm:       {fused_mask.sum()}")

    # ── Step 4: Compute kernel analysis (top 20) ─────────────────────────
    p(f"\n  {'─'*60}")
    p("  TOP 20 COMPUTE OPERATORS (by total duration)")
    p(f"  {'─'*60}")

    df_compute = df[compute_mask].copy()
    if len(df_compute) > 0:
        total_compute_dur = df_compute["Duration(us)"].sum()
        p(f"  Total compute duration: {total_compute_dur:,.1f} us ({total_compute_dur/1e6:.3f} s)")

        grp = df_compute.groupby("Name")["Duration(us)"].agg(
            ["sum", "count", "mean", "std"]
        )
        grp = grp.rename(columns={"sum": "Total(us)", "count": "Count", "mean": "Mean(us)", "std": "Std(us)"})
        grp["CV"] = grp["Std(us)"] / grp["Mean(us)"]
        grp = grp.sort_values("Total(us)", ascending=False)
        grp["Cum%"] = (grp["Total(us)"].cumsum() / total_compute_dur * 100)

        top20 = grp.head(20)
        p(f"\n  {'Rank':<5} {'Name':<55} {'Total(us)':>12} {'Count':>7} {'Mean(us)':>10} {'CV':>6} {'Cum%':>7}")
        p(f"  {'─'*5} {'─'*55} {'─'*12} {'─'*7} {'─'*10} {'─'*6} {'─'*7}")
        for i, (op_name, row) in enumerate(top20.iterrows(), 1):
            cv_str = f"{row['CV']:.2f}" if not pd.isna(row["CV"]) else "N/A"
            p(
                f"  {i:<5} {op_name[:55]:<55} {row['Total(us)']:>12,.1f} "
                f"{int(row['Count']):>7} {row['Mean(us)']:>10,.1f} {cv_str:>6} {row['Cum%']:>6.1f}%"
            )

        p(f"\n  Total unique compute op types: {len(grp)}")

    # ── Step 5: Communication kernel analysis (deduped) ──────────────────
    p(f"\n  {'─'*60}")
    p("  COMMUNICATION KERNELS (deduped: exclude NaN Stream ID rows)")
    p(f"  {'─'*60}")

    df_comm = df[comm_mask].copy()
    # Dedup: remove NaN stream ID rows (they are duplicates of valid stream rows for hcom_*)
    df_comm_deduped = df_comm[df_comm["Stream ID"].notna()].copy()
    df_comm_nan = df_comm[df_comm["Stream ID"].isna()].copy()

    p(f"  Total comm rows: {len(df_comm)} (valid stream: {len(df_comm_deduped)}, NaN stream: {len(df_comm_nan)})")

    if len(df_comm_deduped) > 0:
        total_comm_dur = df_comm_deduped["Duration(us)"].sum()
        p(f"  Total comm duration (deduped): {total_comm_dur:,.1f} us ({total_comm_dur/1e6:.3f} s)")

        grp_comm = df_comm_deduped.groupby("Name")["Duration(us)"].agg(["sum", "count", "mean"])
        grp_comm = grp_comm.rename(columns={"sum": "Total(us)", "count": "Count", "mean": "Mean(us)"})
        grp_comm = grp_comm.sort_values("Total(us)", ascending=False)

        p(f"\n  {'Name':<55} {'Total(us)':>12} {'Count':>7} {'Mean(us)':>10}")
        p(f"  {'─'*55} {'─'*12} {'─'*7} {'─'*10}")
        for op_name, row in grp_comm.iterrows():
            p(f"  {op_name[:55]:<55} {row['Total(us)']:>12,.1f} {int(row['Count']):>7} {row['Mean(us)']:>10,.1f}")
    else:
        p("  No valid-stream comm kernels found.")

    # Also show NaN-stream comm kernels for comparison
    if len(df_comm_nan) > 0:
        total_comm_nan_dur = df_comm_nan["Duration(us)"].sum()
        p(f"\n  NaN-stream comm duration: {total_comm_nan_dur:,.1f} us ({total_comm_nan_dur/1e6:.3f} s)")

    # ── Step 5b: Fused compute+comm kernels ──────────────────────────────
    if fused_mask.sum() > 0:
        p(f"\n  {'─'*60}")
        p("  FUSED COMPUTE+COMM KERNELS")
        p(f"  {'─'*60}")
        df_fused = df[fused_mask].copy()
        total_fused_dur = df_fused["Duration(us)"].sum()
        p(f"  Total fused duration: {total_fused_dur:,.1f} us ({total_fused_dur/1e6:.3f} s)")
        grp_fused = df_fused.groupby("Name")["Duration(us)"].agg(["sum", "count", "mean"])
        grp_fused = grp_fused.rename(columns={"sum": "Total(us)", "count": "Count", "mean": "Mean(us)"})
        grp_fused = grp_fused.sort_values("Total(us)", ascending=False)
        for op_name, row in grp_fused.iterrows():
            p(f"  {op_name[:55]:<55} {row['Total(us)']:>12,.1f} {int(row['Count']):>7} {row['Mean(us)']:>10,.1f}")

    # ── Step 6: Per-stream duration sums ─────────────────────────────────
    p(f"\n  {'─'*60}")
    p("  PER-STREAM DURATION SUMS (deduped: valid Stream ID only)")
    p(f"  {'─'*60}")

    # Only use rows with valid stream IDs
    df_valid = df[df["Stream ID"].notna()].copy()

    # Classify valid-stream rows
    for sid in sorted(df_valid["Stream ID"].unique()):
        mask_s = df_valid["Stream ID"] == sid
        dur = df_valid.loc[mask_s, "Duration(us)"].sum()
        cnt = mask_s.sum()

        # Check what kind of kernels are on this stream
        names = df_valid.loc[mask_s, "Name"]
        n_comm = names.apply(lambda x: bool(COMM_PATTERNS.search(x)) if isinstance(x, str) else False).sum()
        n_fused = names.apply(lambda x: bool(FUSED_COMPUTE_COMM_PATTERNS.search(x)) if isinstance(x, str) else False).sum()
        n_compute = cnt - n_comm - n_fused

        label = "COMPUTE" if n_compute > n_comm else "COMM"
        if n_fused > 0:
            label += "+FUSED"
        p(f"  Stream {int(sid):>3}: {dur:>14,.1f} us  ({cnt:>6} kernels)  [{label}: {n_compute} compute, {n_comm} comm, {n_fused} fused]")

    # Aggregate
    compute_stream_ids = set()
    comm_stream_ids = set()
    for sid in sorted(df_valid["Stream ID"].unique()):
        mask_s = df_valid["Stream ID"] == sid
        names = df_valid.loc[mask_s, "Name"]
        n_comm = names.apply(lambda x: bool(COMM_PATTERNS.search(x)) if isinstance(x, str) else False).sum()
        n_compute = mask_s.sum() - n_comm
        if n_compute > n_comm:
            compute_stream_ids.add(sid)
        else:
            comm_stream_ids.add(sid)

    t1_compute = df_valid[df_valid["Stream ID"].isin(compute_stream_ids)]["Duration(us)"].sum()
    t1_comm = df_valid[df_valid["Stream ID"].isin(comm_stream_ids)]["Duration(us)"].sum()

    p(f"\n  Compute streams {sorted(int(s) for s in compute_stream_ids)}: total = {t1_compute:,.1f} us ({t1_compute/1e6:.3f} s)")
    p(f"  Comm streams {sorted(int(s) for s in comm_stream_ids)}: total = {t1_comm:,.1f} us ({t1_comm/1e6:.3f} s)")
    p(f"  Sum (compute + comm): {t1_compute + t1_comm:,.1f} us ({(t1_compute + t1_comm)/1e6:.3f} s)")

    # Compare with step_trace
    if step_csv.exists():
        df_step = pd.read_csv(step_csv)
        p(f"\n  step_trace_time.csv:")
        p(f"    Columns: {list(df_step.columns)}")
        for _, row in df_step.iterrows():
            computing = row.get("Computing", 0)
            comm_not_overlap = row.get("Communication(Not Overlapped)", 0)
            overlapped = row.get("Overlapped", 0)
            comm_total = row.get("Communication", 0)
            free = row.get("Free", 0)
            stage = row.get("Stage", 0)
            p(f"    Computing:                   {computing:>14,.1f} us")
            p(f"    Communication(Not Overlapped):{comm_not_overlap:>14,.1f} us")
            p(f"    Overlapped:                  {overlapped:>14,.1f} us")
            p(f"    Communication (total):       {comm_total:>14,.1f} us")
            p(f"    Free:                        {free:>14,.1f} us")
            p(f"    Stage (e2e):                 {stage:>14,.1f} us ({stage/1e6:.3f} s)")
            p(f"    Derived e2e = Computing + Comm(NotOverlap) + Free = {computing + comm_not_overlap + free:,.1f} us")

    # ── Step 7: HCCL double-counting check ───────────────────────────────
    p(f"\n  {'─'*60}")
    p("  HCCL DOUBLE-COUNTING CHECK")
    p(f"  {'─'*60}")

    hcom_mask = df["Name"].str.startswith("hcom_", na=False)
    hcom_df = df[hcom_mask].copy()
    hcom_nan = hcom_df["Stream ID"].isna().sum()
    hcom_valid = hcom_df["Stream ID"].notna().sum()
    p(f"  hcom_* rows total: {len(hcom_df)}")
    p(f"    - NaN Stream ID:   {hcom_nan}")
    p(f"    - Valid Stream ID: {hcom_valid}")

    if hcom_nan > 0 and hcom_valid > 0:
        dur_nan = hcom_df[hcom_df["Stream ID"].isna()]["Duration(us)"].sum()
        dur_valid = hcom_df[hcom_df["Stream ID"].notna()]["Duration(us)"].sum()
        p(f"    - NaN Stream duration:   {dur_nan:>14,.1f} us")
        p(f"    - Valid Stream duration:  {dur_valid:>14,.1f} us")
        p(f"    ⚠ DOUBLE-COUNTED if both are summed! Ratio NaN/Valid = {dur_nan/dur_valid:.2f}")
    elif hcom_nan > 0 and hcom_valid == 0:
        p("    → hcom_* only on NaN streams (no valid stream rows)")
    elif hcom_valid > 0 and hcom_nan == 0:
        p("    → hcom_* only on valid streams (no NaN stream duplicates)")
    else:
        p("    → No hcom_* kernels found")

    p("")


def main():
    out_dir = Path(
        "/home/horacehxw/Projects/msmodeling-perf-db-profiling-analysis/"
        "docs/perf_database/reports/profiling_analysis_phase1"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "analysis_output.txt"

    buf = StringIO()
    buf.write("Phase 1 E2E Profiling Analysis\n")
    buf.write(f"Data: {BASE}\n")
    buf.write(f"{'=' * 90}\n")

    for scenario_name, rel_path in SCENARIOS.items():
        profiler_dir = BASE / rel_path
        analyze_scenario(scenario_name, profiler_dir, buf)

    text = buf.getvalue()
    print(text)
    out_file.write_text(text)
    print(f"\n[Saved to {out_file}]")


if __name__ == "__main__":
    main()
