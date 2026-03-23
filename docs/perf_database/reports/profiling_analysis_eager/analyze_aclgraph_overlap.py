"""
Analyze aclgraph profiling data for compute/comm overlap.
Compare with eager mode results.
"""

import csv
import sys
from collections import defaultdict


def parse_kernel_details(csv_path):
    """Parse kernel_details.csv, handling trailing tab in Start Time."""
    kernels = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Start Time has trailing tab - clean it
            start_str = row["Start Time(us)"].strip()
            start = float(start_str)
            duration = float(row["Duration(us)"].strip())
            stream_id_str = row["Stream ID"].strip()
            if stream_id_str == "N/A" or stream_id_str == "":
                continue
            stream_id = int(stream_id_str)
            name = row["Name"].strip()
            kernels.append({
                "stream_id": stream_id,
                "name": name,
                "start": start,
                "end": start + duration,
                "duration": duration,
            })
    return kernels


def is_comm_kernel(name):
    """Check if kernel is a communication kernel."""
    comm_patterns = ["hcom_", "Hcom", "HCCL", "hccl", "AivKernel"]
    return any(p in name for p in comm_patterns)


def analyze_model(csv_path, model_name):
    print(f"\n{'='*80}")
    print(f"  {model_name} - aclgraph profiling analysis")
    print(f"{'='*80}")
    print(f"  File: {csv_path}")

    kernels = parse_kernel_details(csv_path)
    print(f"  Total kernels parsed: {len(kernels)}")

    # Group by stream
    streams = defaultdict(list)
    for k in kernels:
        streams[k["stream_id"]].append(k)

    # Per-stream stats
    print(f"\n--- Per-Stream Breakdown ---")
    print(f"{'Stream':>8} {'Count':>7} {'Sum(us)':>12} {'Span(us)':>12} {'Gap(us)':>12} {'Gap%':>7}  {'Sample Kernels'}")

    stream_stats = {}
    for sid in sorted(streams.keys()):
        sk = streams[sid]
        count = len(sk)
        kernel_sum = sum(k["duration"] for k in sk)
        min_start = min(k["start"] for k in sk)
        max_end = max(k["end"] for k in sk)
        span = max_end - min_start
        gap = span - kernel_sum
        gap_pct = (gap / span * 100) if span > 0 else 0

        # Classify stream
        comm_count = sum(1 for k in sk if is_comm_kernel(k["name"]))
        comm_ratio = comm_count / count if count > 0 else 0

        # Get unique kernel name samples
        unique_names = list(set(k["name"] for k in sk))
        sample = ", ".join(unique_names[:3])
        if len(unique_names) > 3:
            sample += f", ... ({len(unique_names)} unique)"

        stream_type = "COMM" if comm_ratio > 0.5 else "COMPUTE" if comm_ratio < 0.1 else "MIXED"

        print(f"{sid:>8} {count:>7} {kernel_sum:>12.1f} {span:>12.1f} {gap:>12.1f} {gap_pct:>6.1f}%  [{stream_type}] {sample}")

        stream_stats[sid] = {
            "kernels": sk,
            "count": count,
            "sum": kernel_sum,
            "span": span,
            "gap": gap,
            "gap_pct": gap_pct,
            "type": stream_type,
            "comm_ratio": comm_ratio,
        }

    # Identify compute and comm streams
    compute_streams = [sid for sid, s in stream_stats.items() if s["type"] == "COMPUTE"]
    comm_streams = [sid for sid, s in stream_stats.items() if s["type"] == "COMM"]
    mixed_streams = [sid for sid, s in stream_stats.items() if s["type"] == "MIXED"]

    print(f"\n  Compute streams: {compute_streams}")
    print(f"  Comm streams: {comm_streams}")
    print(f"  Mixed streams: {mixed_streams}")

    # Show comm kernel names for comm streams
    for sid in comm_streams:
        comm_names = set(k["name"] for k in streams[sid] if is_comm_kernel(k["name"]))
        print(f"  Stream {sid} comm kernels: {comm_names}")

    # Also show what's in mixed streams
    for sid in mixed_streams:
        comm_names = set(k["name"] for k in streams[sid] if is_comm_kernel(k["name"]))
        non_comm_names = set(k["name"] for k in streams[sid] if not is_comm_kernel(k["name"]))
        print(f"  Stream {sid} mixed - comm: {list(comm_names)[:5]}, non-comm: {list(non_comm_names)[:5]}")

    # Aggregate compute and comm kernel durations
    all_compute_kernels = []
    all_comm_kernels = []
    for k in kernels:
        if is_comm_kernel(k["name"]):
            all_comm_kernels.append(k)
        else:
            all_compute_kernels.append(k)

    t1_comp = sum(k["duration"] for k in all_compute_kernels)
    t1_comm = sum(k["duration"] for k in all_comm_kernels)

    # E2E span
    global_min = min(k["start"] for k in kernels)
    global_max = max(k["end"] for k in kernels)
    e2e = global_max - global_min

    print(f"\n--- Aggregate Timing ---")
    print(f"  t1_comp (sum of compute kernels): {t1_comp:>12.1f} us ({t1_comp/1000:.1f} ms)")
    print(f"  t1_comm (sum of comm kernels):    {t1_comm:>12.1f} us ({t1_comm/1000:.1f} ms)")
    print(f"  t1_comp + t1_comm:                {t1_comp+t1_comm:>12.1f} us ({(t1_comp+t1_comm)/1000:.1f} ms)")
    print(f"  E2E span (wall clock):            {e2e:>12.1f} us ({e2e/1000:.1f} ms)")

    # Overlap estimation: if comp+comm > e2e, the excess is overlap
    # But we also need to account for CPU gaps
    total_kernel_sum = sum(s["sum"] for s in stream_stats.values())
    total_cpu_gap = e2e - total_kernel_sum  # This is wrong for multi-stream; need merge intervals

    # Better: merge all kernel intervals across all streams and compute actual busy time
    all_intervals = [(k["start"], k["end"]) for k in kernels]
    all_intervals.sort()
    merged = []
    for s, e in all_intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    total_busy = sum(e - s for s, e in merged)
    total_free = e2e - total_busy

    print(f"\n--- Overlap Analysis (interval merging) ---")
    print(f"  Total busy time (merged):   {total_busy:>12.1f} us ({total_busy/1000:.1f} ms)")
    print(f"  Total free/CPU gap:         {total_free:>12.1f} us ({total_free/1000:.1f} ms)")
    print(f"  Free %:                     {total_free/e2e*100:>6.1f}%")

    # Compute actual overlap between compute and comm kernels
    # Method: for each comm kernel, find overlap with compute kernels
    # Sort compute kernels by start time for efficient search
    comp_intervals = sorted([(k["start"], k["end"]) for k in all_compute_kernels])
    comm_intervals = sorted([(k["start"], k["end"]) for k in all_comm_kernels])

    # Merge compute intervals
    merged_comp = []
    for s, e in comp_intervals:
        if merged_comp and s <= merged_comp[-1][1]:
            merged_comp[-1] = (merged_comp[-1][0], max(merged_comp[-1][1], e))
        else:
            merged_comp.append((s, e))

    # Merge comm intervals
    merged_comm = []
    for s, e in comm_intervals:
        if merged_comm and s <= merged_comm[-1][1]:
            merged_comm[-1] = (merged_comm[-1][0], max(merged_comm[-1][1], e))
        else:
            merged_comm.append((s, e))

    # Compute overlap between merged compute and merged comm intervals
    overlap_total = 0.0
    ci = 0
    for cs, ce in merged_comm:
        while ci < len(merged_comp) and merged_comp[ci][1] <= cs:
            ci += 1
        j = ci
        while j < len(merged_comp) and merged_comp[j][0] < ce:
            ov_start = max(cs, merged_comp[j][0])
            ov_end = min(ce, merged_comp[j][1])
            if ov_end > ov_start:
                overlap_total += ov_end - ov_start
            j += 1

    merged_comp_total = sum(e - s for s, e in merged_comp)
    merged_comm_total = sum(e - s for s, e in merged_comm)

    print(f"\n  Merged compute time:        {merged_comp_total:>12.1f} us ({merged_comp_total/1000:.1f} ms)")
    print(f"  Merged comm time:           {merged_comm_total:>12.1f} us ({merged_comm_total/1000:.1f} ms)")
    print(f"  Compute-Comm OVERLAP:       {overlap_total:>12.1f} us ({overlap_total/1000:.1f} ms)")
    if merged_comm_total > 0:
        print(f"  Overlap / Comm:             {overlap_total/merged_comm_total*100:>6.1f}%")
    if e2e > 0:
        print(f"  Overlap / E2E:              {overlap_total/e2e*100:>6.1f}%")

    return {
        "t1_comp": t1_comp,
        "t1_comm": t1_comm,
        "e2e": e2e,
        "overlap": overlap_total,
        "free": total_free,
        "merged_comp": merged_comp_total,
        "merged_comm": merged_comm_total,
        "n_kernels": len(kernels),
    }


def analyze_step_trace(csv_path):
    """Parse step_trace_time.csv for official breakdown."""
    print(f"\n--- Qwen3 step_trace_time.csv (official CANN breakdown) ---")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            computing = float(row["Computing"])
            comm_not_ov = float(row["Communication(Not Overlapped)"])
            overlapped = float(row["Overlapped"])
            comm_total = float(row["Communication"])
            free = float(row["Free"])
            stage = float(row["Stage"]) if row.get("Stage") else 0

            total = computing + comm_not_ov + free
            print(f"  Computing:                  {computing:>12.1f} us ({computing/1000:.1f} ms)")
            print(f"  Comm (Not Overlapped):      {comm_not_ov:>12.1f} us ({comm_not_ov/1000:.1f} ms)")
            print(f"  Overlapped:                 {overlapped:>12.1f} us ({overlapped/1000:.1f} ms)")
            print(f"  Communication (total):      {comm_total:>12.1f} us ({comm_total/1000:.1f} ms)")
            print(f"  Free:                       {free:>12.1f} us ({free/1000:.1f} ms)")
            print(f"  Stage (total):              {stage:>12.1f} us ({stage/1000:.1f} ms)")
            print(f"  Overlap / Comm total:       {overlapped/comm_total*100:.1f}%")
            print(f"  Overlap / Stage:            {overlapped/stage*100:.1f}%")


def comparison_table(dsv3_res, qwen3_res):
    print(f"\n{'='*80}")
    print(f"  COMPARISON: eager vs aclgraph")
    print(f"{'='*80}")

    # Eager baselines (from user)
    # DSv3 decode_b8: t1_comp=4420ms, t1_comm=935ms, e2e=5425ms, CPU gap=70ms
    # Qwen3 decode_b8: t1_comp=88ms, t1_comm=3211ms, e2e=3361ms, CPU gap=62ms

    print(f"\n{'':>30} {'DSv3 eager':>14} {'DSv3 aclgraph':>14} {'Qwen3 eager':>14} {'Qwen3 aclgraph':>14}")
    print(f"{'':>30} {'(decode b8)':>14} {'':>14} {'(decode b8)':>14} {'':>14}")
    print("-" * 100)

    rows = [
        ("t1_comp (ms)", 4420, dsv3_res["t1_comp"]/1000, 88, qwen3_res["t1_comp"]/1000),
        ("t1_comm (ms)", 935, dsv3_res["t1_comm"]/1000, 3211, qwen3_res["t1_comm"]/1000),
        ("E2E span (ms)", 5425, dsv3_res["e2e"]/1000, 3361, qwen3_res["e2e"]/1000),
        ("Overlap (ms)", 0, dsv3_res["overlap"]/1000, 0, qwen3_res["overlap"]/1000),
        ("Free/CPU gap (ms)", 70, dsv3_res["free"]/1000, 62, qwen3_res["free"]/1000),
    ]

    for label, eager_dsv3, acl_dsv3, eager_qwen3, acl_qwen3 in rows:
        print(f"{label:>30} {eager_dsv3:>14.1f} {acl_dsv3:>14.1f} {eager_qwen3:>14.1f} {acl_qwen3:>14.1f}")

    # Overlap percentages
    print()
    dsv3_ov_pct = dsv3_res["overlap"] / dsv3_res["e2e"] * 100 if dsv3_res["e2e"] > 0 else 0
    qwen3_ov_pct = qwen3_res["overlap"] / qwen3_res["e2e"] * 100 if qwen3_res["e2e"] > 0 else 0
    print(f"{'Overlap/E2E (%)':>30} {'~0%':>14} {dsv3_ov_pct:>13.1f}% {'~0%':>14} {qwen3_ov_pct:>13.1f}%")

    dsv3_ov_comm_pct = dsv3_res["overlap"] / dsv3_res["merged_comm"] * 100 if dsv3_res["merged_comm"] > 0 else 0
    qwen3_ov_comm_pct = qwen3_res["overlap"] / qwen3_res["merged_comm"] * 100 if qwen3_res["merged_comm"] > 0 else 0
    print(f"{'Overlap/Comm (%)':>30} {'~0%':>14} {dsv3_ov_comm_pct:>13.1f}% {'~0%':>14} {qwen3_ov_comm_pct:>13.1f}%")


def main():
    dsv3_path = "/mnt/d/Data/Profiling/deepseekv3_torch2.9.0_vllm0.15.0_cann8.5_aclgraph_PandD/kernel_details_deepseekv3-cann85.csv"
    qwen3_path = "/mnt/d/Data/Profiling/qwen3-32b_torch2.9.0_vllm0.15.0_cann8.5_aclgraph_PandD/e57aa6f6d21d_33066_20260310073540522_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv"
    qwen3_step_trace = "/mnt/d/Data/Profiling/qwen3-32b_torch2.9.0_vllm0.15.0_cann8.5_aclgraph_PandD/e57aa6f6d21d_33066_20260310073540522_ascend_pt/ASCEND_PROFILER_OUTPUT/step_trace_time.csv"

    # Scenario info
    print("="*80)
    print("  SCENARIO SUMMARY")
    print("="*80)
    print()
    print("  DSv3 (aclgraph):")
    print("    - Model: DeepSeekV3-AQ-w8a8")
    print("    - TP=8, DP=2, EP enabled")
    print("    - max-num-seqs=4, max_num_batched_tokens=2048")
    print("    - quantization=ascend (W8A8)")
    print("    - aclgraph: cudagraph_mode=FULL_DECODE_ONLY, sizes=[5,10,15,20,25,30,35,40]")
    print("    - CANN 8.5, torch 2.9.0, vllm 0.15.0")
    print("    - FUSED_MC2=1, FLASHCOMM1=1, HCCL_OP_EXPANSION_MODE=AIV")
    print()
    print("  Qwen3-32B (aclgraph):")
    print("    - Model: Qwen3-32B (BF16)")
    print("    - TP=16, no EP")
    print("    - max_num_batched_tokens=65536")
    print("    - aclgraph: cudagraph_mode=FULL_DECODE_ONLY, sizes=[4,8,16,32,64]")
    print("    - CANN 8.5, torch 2.9.0, vllm 0.15.0")
    print("    - FUSED_MC2=1, FLASHCOMM1=1, HCCL_OP_EXPANSION_MODE=AIV")
    print()

    dsv3_res = analyze_model(dsv3_path, "DeepSeek-V3 (aclgraph, W8A8)")
    qwen3_res = analyze_model(qwen3_path, "Qwen3-32B (aclgraph, BF16)")

    # Step trace for Qwen3
    analyze_step_trace(qwen3_step_trace)

    # Comparison
    comparison_table(dsv3_res, qwen3_res)

    # Conclusions
    print(f"\n{'='*80}")
    print(f"  CONCLUSIONS")
    print(f"{'='*80}")
    dsv3_ov_pct = dsv3_res["overlap"] / dsv3_res["e2e"] * 100 if dsv3_res["e2e"] > 0 else 0
    qwen3_ov_pct = qwen3_res["overlap"] / qwen3_res["e2e"] * 100 if qwen3_res["e2e"] > 0 else 0
    dsv3_ov_comm = dsv3_res["overlap"] / dsv3_res["merged_comm"] * 100 if dsv3_res["merged_comm"] > 0 else 0
    qwen3_ov_comm = qwen3_res["overlap"] / qwen3_res["merged_comm"] * 100 if qwen3_res["merged_comm"] > 0 else 0

    if dsv3_ov_pct > 5 or qwen3_ov_pct > 5:
        print("  aclgraph DOES enable meaningful compute/comm overlap.")
    elif dsv3_ov_pct > 1 or qwen3_ov_pct > 1:
        print("  aclgraph enables SOME compute/comm overlap, but it is modest.")
    else:
        print("  aclgraph does NOT enable meaningful compute/comm overlap (similar to eager).")

    print(f"  DSv3: overlap = {dsv3_res['overlap']/1000:.1f} ms ({dsv3_ov_pct:.1f}% of E2E, {dsv3_ov_comm:.1f}% of comm)")
    print(f"  Qwen3: overlap = {qwen3_res['overlap']/1000:.1f} ms ({qwen3_ov_pct:.1f}% of E2E, {qwen3_ov_comm:.1f}% of comm)")
    print()


if __name__ == "__main__":
    main()

