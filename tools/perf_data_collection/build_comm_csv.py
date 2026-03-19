#!/usr/bin/env python3
"""Build final hcom CSV files by merging bench data from multiple sources.

Data source selection per bench_vs_profiler_comm_20260318.md §4/§5:
  - allReduce all sizes:           alternating (profiler fallback)
  - allGather/reduceScatter >=1MB: alternating (peer flow eliminates warmup bias)
  - allGather/reduceScatter <1MB:  kernel/profiler (event floor ~60us drowns real value)
  - DSV3 allGather 768KB nd=8:    profiler trace P50 (HCCL protocol switch anomaly)

Supports merging production msg_bytes + standard grid supplementary data.
Applies fixed-overhead corrections for decode small messages.
"""

import argparse
import csv
import os
import statistics
import sys
from pathlib import Path

# Fixed overhead corrections (us) from bench_vs_profiler report Section 1.
# These represent vLLM production dispatch overhead (scheduler dispatch →
# c10d wrapper → HCCL group lookup → stream sync) that exists in production
# but not in bench's isolated single-op execution. Applied to all bench modes
# (both kernel and profiler-batch) since neither includes this overhead.
OVERHEAD = {
    "allReduce": {16: 7.7},
    "allGather": {16: 14.6, 8: 1.2},
    "reduceScatter": {16: 14.6, 8: 2.0},
}

ONE_MB = 1_048_576
DSV3_768KB = 786_432

CSV_COLUMNS = ["message_bytes", "num_devices", "dtype", "topology_tier", "Duration(us)", "bandwidth_gbps"]


def read_csv(path: str) -> list[dict]:
    """Read a bench CSV and return list of row dicts with typed values."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "message_bytes": int(row["message_bytes"]),
                "num_devices": int(row["num_devices"]),
                "dtype": row["dtype"],
                "topology_tier": int(row["topology_tier"]),
                "Duration(us)": float(row["Duration(us)"]),
                "bandwidth_gbps": float(row["bandwidth_gbps"]),
            })
    return rows


def calc_bandwidth(message_bytes: int, duration_us: float) -> float:
    """Calculate bandwidth in GB/s."""
    if duration_us <= 0:
        return 0.0
    return round(message_bytes / (duration_us * 1e-6) / 1e9, 2)


def parse_profiler_allgather_p50(profiler_trace_dir: str) -> float:
    """Parse all kernel_details.csv under profiler_trace_dir to get allGather P50 duration."""
    durations = []
    trace_dir = Path(profiler_trace_dir)
    for subdir in sorted(trace_dir.iterdir()):
        kd_path = subdir / "kernel_details.csv"
        if not kd_path.exists():
            continue
        with open(kd_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                op_type = row.get("Type", "")
                name = row.get("Name", "")
                if op_type.startswith("hcom_allGather_") and "AivKernel" not in name:
                    try:
                        durations.append(float(row["Duration(us)"]))
                    except (ValueError, KeyError):
                        pass
    if not durations:
        print(f"WARNING: No allGather entries found in {profiler_trace_dir}", file=sys.stderr)
        return 0.0
    p50 = round(statistics.median(durations), 2)
    print(f"  Profiler allGather P50: {p50} us (from {len(durations)} samples)")
    return p50


def build_index(rows: list[dict]) -> dict[tuple[int, int], dict]:
    """Index rows by (message_bytes, num_devices)."""
    idx = {}
    for r in rows:
        key = (r["message_bytes"], r["num_devices"])
        idx[key] = r
    return idx


def build_allreduce(alt_rows: list[dict]) -> list[dict]:
    """Build allReduce output: alternating data + fixed dispatch overhead.

    allReduce has no peer in alternating mode, so it falls back to
    profiler-batch (operator_details Device Total Duration). The bench value
    does NOT include vLLM production dispatch overhead (scheduler → c10d →
    HCCL group lookup → stream sync), so we add the fixed overhead here.
    """
    result = []
    for row in alt_rows:
        r = dict(row)
        nd = r["num_devices"]
        overhead = OVERHEAD["allReduce"].get(nd, 0.0)
        if overhead > 0:
            r["Duration(us)"] = round(r["Duration(us)"] + overhead, 2)
            r["bandwidth_gbps"] = calc_bandwidth(r["message_bytes"], r["Duration(us)"])
        result.append(r)
    return result


def build_allgather(
    alt_rows: list[dict],
    kern_rows: list[dict],
    profiler_p50: float,
) -> list[dict]:
    """Build allGather output with source selection and overhead correction."""
    alt_idx = build_index(alt_rows)
    kern_idx = build_index(kern_rows)
    result = []

    # Collect all unique (msg, nd) keys from both sources
    all_keys = set(alt_idx.keys()) | set(kern_idx.keys())

    for msg, nd in sorted(all_keys):
        # DSV3 768KB nd=8: use profiler P50
        if nd == 8 and msg == DSV3_768KB and profiler_p50 > 0:
            dur = profiler_p50
            result.append({
                "message_bytes": msg,
                "num_devices": nd,
                "dtype": "DT_BF16",
                "topology_tier": 1,
                "Duration(us)": dur,
                "bandwidth_gbps": calc_bandwidth(msg, dur),
            })
            continue

        # nd=4, nd=2: alternating for large, kernel (no overhead) for small
        if nd in (2, 4):
            row = alt_idx.get((msg, nd)) or kern_idx.get((msg, nd))
            if row:
                result.append(dict(row))
            continue

        # nd=16 or nd=8
        overhead = OVERHEAD["allGather"].get(nd, 0.0)
        if msg < ONE_MB:
            # Small messages: use kernel data + overhead
            row = kern_idx.get((msg, nd))
            if row:
                r = dict(row)
                r["Duration(us)"] = round(r["Duration(us)"] + overhead, 2)
                r["bandwidth_gbps"] = calc_bandwidth(r["message_bytes"], r["Duration(us)"])
                result.append(r)
        else:
            # Large messages: use alternating directly
            row = alt_idx.get((msg, nd))
            if row:
                result.append(dict(row))

    return result


def build_reducescatter(
    alt_rows: list[dict],
    kern_rows: list[dict],
) -> list[dict]:
    """Build reduceScatter output with source selection and overhead correction."""
    alt_idx = build_index(alt_rows)
    kern_idx = build_index(kern_rows)
    result = []

    all_keys = set(alt_idx.keys()) | set(kern_idx.keys())

    for msg, nd in sorted(all_keys):
        # nd=4, nd=2: alternating for large, kernel (no overhead) for small
        if nd in (2, 4):
            row = alt_idx.get((msg, nd)) or kern_idx.get((msg, nd))
            if row:
                result.append(dict(row))
            continue

        # nd=16 or nd=8
        overhead = OVERHEAD["reduceScatter"].get(nd, 0.0)
        if msg < ONE_MB:
            row = kern_idx.get((msg, nd))
            if row:
                r = dict(row)
                r["Duration(us)"] = round(r["Duration(us)"] + overhead, 2)
                r["bandwidth_gbps"] = calc_bandwidth(r["message_bytes"], r["Duration(us)"])
                result.append(r)
        else:
            row = alt_idx.get((msg, nd))
            if row:
                result.append(dict(row))

    return result


def write_csv(rows: list[dict], path: str) -> None:
    """Write rows to CSV."""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  Wrote {len(rows)} rows -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Build final hcom CSV files")
    parser.add_argument("--alternating-dir", required=True,
                        help="Alternating bench data directory (allReduce all + allGather/reduceScatter >=1MB)")
    parser.add_argument("--kernel-dir", required=True,
                        help="Kernel (profiler) bench data directory (allGather/reduceScatter <1MB)")
    parser.add_argument("--profiler-trace-dir", required=True, help="DSV3 profiler trace directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for final CSVs")
    args = parser.parse_args()

    alt_dir = args.alternating_dir
    kern_dir = args.kernel_dir

    # Read bench data
    print("Reading alternating bench data...")
    alt_ar = read_csv(os.path.join(alt_dir, "hcom_allReduce_.csv"))
    alt_ag = read_csv(os.path.join(alt_dir, "hcom_allGather_.csv"))
    alt_rs = read_csv(os.path.join(alt_dir, "hcom_reduceScatter_.csv"))
    print(f"  allReduce: {len(alt_ar)}, allGather: {len(alt_ag)}, reduceScatter: {len(alt_rs)}")

    print("Reading kernel bench data...")
    kern_ag = read_csv(os.path.join(kern_dir, "hcom_allGather_.csv"))
    kern_rs = read_csv(os.path.join(kern_dir, "hcom_reduceScatter_.csv"))
    print(f"  allGather: {len(kern_ag)}, reduceScatter: {len(kern_rs)}")

    print("Parsing profiler trace for DSV3 allGather 768KB P50...")
    profiler_p50 = parse_profiler_allgather_p50(args.profiler_trace_dir)

    # Build outputs
    print("\nBuilding allReduce...")
    ar_out = build_allreduce(alt_ar)

    print("Building allGather...")
    ag_out = build_allgather(alt_ag, kern_ag, profiler_p50)

    print("Building reduceScatter...")
    rs_out = build_reducescatter(alt_rs, kern_rs)

    # Write outputs
    print("\nWriting output CSVs...")
    write_csv(ar_out, os.path.join(args.output_dir, "hcom_allReduce_.csv"))
    write_csv(ag_out, os.path.join(args.output_dir, "hcom_allGather_.csv"))
    write_csv(rs_out, os.path.join(args.output_dir, "hcom_reduceScatter_.csv"))

    # alltoallv: no bench data, write empty CSV with header only
    alltoall_path = os.path.join(args.output_dir, "hcom_alltoallv_.csv")
    os.makedirs(os.path.dirname(alltoall_path), exist_ok=True)
    with open(alltoall_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
    print(f"  Wrote 0 rows -> {alltoall_path} (no bench data)")

    # Summary
    print(f"\nDone. Output: {args.output_dir}")
    print(f"  allReduce:     {len(ar_out)} rows")
    print(f"  allGather:     {len(ag_out)} rows")
    print(f"  reduceScatter: {len(rs_out)} rows")
    print(f"  alltoallv:     0 rows (header only)")


if __name__ == "__main__":
    main()