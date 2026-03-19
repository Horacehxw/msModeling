"""Generate and optionally run communication-operator microbenchmark scripts.

Generates torch.distributed scripts for HCCL communication benchmarking:
- all_reduce, all_gather, reduce_scatter, all_to_all
- topology_tier is derived from rank + group via CommGrid logic (not manually specified)
- Outputs CSV in the format required by ProfilingDataSource (§4.7)

Design doc reference: §6.3 (Communication Microbenchmark)

CSV output format (§4.7):
    message_bytes,num_devices,dtype,topology_tier,Duration(us),bandwidth_gbps

topology_tier semantics (mirrors CommAnalyticModel._get_topology_idx_for_group):
    Determined by the outermost grid dimension where ranks in the group differ.
    For ATLAS_800_A3 with grid shape [48, 8, 2] (48 pods × 8 nodes × 2 dies):
        tier 0 = inter_pod  (ranks span multiple pods, stride=16)
        tier 1 = intra_pod  (ranks within one pod, span multiple nodes, stride=2)
        tier 2 = die_level  (ranks within one node, 2 dies, stride=1)
        e.g. TP=16 uses ranks 0..15 (pod0, all 8 nodes × 2 dies) → tier=1

    The group_ranks argument controls which ranks participate, which determines
    the tier automatically. Use --grid-shape to match your hardware topology.

Usage examples:
    # Generate scripts for all ops, tier-2 (die-level, 16 devices)
    python generate_comm_microbench.py --output-dir ./comm_scripts \\
        --ops all_reduce all_gather reduce_scatter \\
        --grid-shape 48 8 2 --num-devices 16 --topology-tier 2

    # Generate for all 3 tiers
    python generate_comm_microbench.py --output-dir ./comm_scripts \\
        --ops all_reduce --grid-shape 48 8 2 \\
        --num-devices 16 64 128 --topology-tier 0 1 2

    # Run directly with profiler mode (default, aligns Comm_NO)
    torchrun --nproc_per_node=16 generate_comm_microbench.py \\
        --do-run --output-csv ./hccl_v8.5/hcom_allReduce_.csv \\
        --ops all_reduce --grid-shape 48 8 2

    # Run with event mode (fast, hcom_kernel only)
    torchrun --nproc_per_node=16 generate_comm_microbench.py \\
        --do-run --bench-mode event --output-dir ./hccl_data \\
        --ops all_reduce all_gather --grid-shape 48 8 2

    # Run with pipeline mode (original, backward compat)
    torchrun --nproc_per_node=16 generate_comm_microbench.py \\
        --do-run --bench-mode pipeline --output-dir ./hccl_data \\
        --ops all_reduce --grid-shape 48 8 2
"""

import argparse
import csv
import glob
import math
import os
import shutil
import statistics
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from textwrap import dedent
from typing import Callable, Dict, List, Optional, Tuple

WARMUP_ITERS = 20
BENCH_ITERS = 100

# Profiler bench mode constants
PROFILER_WARMUP_ITERS = 5   # profiler-internal warmup steps (separate from op warmup)
PROFILER_ACTIVE_ITERS = 10  # active profiling steps → 10 Duration samples, take median
PROFILER_WAIT_ITERS = 0

_BENCH_MODES = ["profiler", "event", "pipeline", "kernel", "alternating"]

_COMM_OPS = ["all_reduce", "all_gather", "reduce_scatter", "all_to_all"]

# Maps bench op_type → c10d operator name prefix(es) in operator_details.csv.
# Uses startswith matching to handle suffix variations across PyTorch/CANN versions.
_OP_TO_C10D_NAMES = {
    "all_reduce": ["c10d::allreduce_"],
    "all_gather": ["c10d::_allgather_base_"],
    "reduce_scatter": ["c10d::_reduce_scatter_base_"],
    "all_to_all": ["c10d::alltoall_", "c10d::all_to_all"],
}

# Maps op_type → kernel_details.csv Type prefix for hcom_* kernels
_OP_TO_KERNEL_TYPE = {
    "all_reduce": "hcom_allReduce_",
    "all_gather": "hcom_allGather_",
    "reduce_scatter": "hcom_reduceScatter_",
    "all_to_all": "hcom_alltoallv_",
}

# Maps op_type → canonical CSV filename expected by ProfilingDataSource / op_mapping.yaml
_OP_TO_CSV_FILENAME = {
    "all_reduce": "hcom_allReduce_.csv",
    "all_gather": "hcom_allGather_.csv",
    "reduce_scatter": "hcom_reduceScatter_.csv",
    "all_to_all": "hcom_alltoallv_.csv",  # kernel_type in op_mapping.yaml is hcom_alltoallv_
}

# message_bytes grid: 1KB ~ 512MB, powers of 2 (dense grid for interpolation)
_DEFAULT_BYTES_GRID = [
    1024,        # 1 KB
    2048,        # 2 KB
    4096,        # 4 KB
    8192,        # 8 KB
    16384,       # 16 KB
    32768,       # 32 KB
    65536,       # 64 KB
    131072,      # 128 KB
    262144,      # 256 KB
    524288,      # 512 KB
    1048576,     # 1 MB
    2097152,     # 2 MB
    4194304,     # 4 MB
    8388608,     # 8 MB
    16777216,    # 16 MB
    33554432,    # 32 MB
    67108864,    # 64 MB
    134217728,   # 128 MB
    268435456,   # 256 MB
    536870912,   # 512 MB
]

_DTYPE_ELEM_SIZE = {
    "torch.bfloat16": 2,
    "torch.float16": 2,
    "torch.float32": 4,
    "torch.int8": 1,
}

_DTYPE_TO_CSV = {
    "torch.bfloat16": "DT_BF16",
    "torch.float16": "DT_FP16",
    "torch.float32": "DT_FLOAT",
    "torch.int8": "DT_INT8",
}

# CSV columns per §4.7
_CSV_COLUMNS = ["message_bytes", "num_devices", "dtype", "topology_tier", "Duration(us)", "bandwidth_gbps"]


# ============================================================================
# Topology tier resolution (mirrors CommAnalyticModel._get_topology_idx_for_group)
# ============================================================================

def _rank_to_coord(rank: int, grid_shape: List[int]) -> List[int]:
    coord = []
    temp = rank
    for dim_size in reversed(grid_shape):
        coord.insert(0, temp % dim_size)
        temp //= dim_size
    return coord


def resolve_topology_tier(group_ranks: List[int], grid_shape: List[int]) -> int:
    """Determine topology_tier for a group, matching CommAnalyticModel logic.

    Finds the outermost grid dimension where ranks differ, then returns the
    largest start_dim <= diff_dim (most specific topology that covers the span).

    For ATLAS_800_A3 grid_shape=[pods, nodes, dies]:
        All ranks same node  → diff_dim=2 → tier=2 (SIO / die-level)
        Ranks span nodes     → diff_dim=1 → tier=1 (1-level CLOS / intra-pod)
        Ranks span pods      → diff_dim=0 → tier=0 (2-level CLOS / inter-pod)
    """
    ndim = len(grid_shape)
    coords = [_rank_to_coord(r, grid_shape) for r in group_ranks]

    diff_dim = -1
    for dim_idx in range(ndim):
        first = coords[0][dim_idx]
        if any(c[dim_idx] != first for c in coords[1:]):
            diff_dim = dim_idx
            break

    if diff_dim == -1:
        return ndim - 1  # all same rank (shouldn't happen), use fastest

    # Most specific topology: largest start_dim <= diff_dim
    for start_dim in range(ndim - 1, -1, -1):
        if start_dim <= diff_dim:
            return start_dim

    return 0


def build_group_for_tier(
    rank: int, num_devices: int, topology_tier: int, grid_shape: List[int]
) -> List[int]:
    """Build a contiguous group of num_devices ranks at the given topology_tier.

    The group is anchored to rank's position in the grid: all ranks in the
    group share the same coordinates in dimensions > topology_tier, and span
    contiguously within the tier dimension.

    Example (grid_shape=[3,8,2], rank=5, num_devices=16, tier=1):
        rank 5 coord = [0, 2, 1]
        tier=1 means we span dims [1,2] → group size per pod = 8*2=16
        group = ranks 0..15 (pod 0, all nodes, all dies)
    """
    ndim = len(grid_shape)
    coord = _rank_to_coord(rank, grid_shape)

    # Compute the stride and size for each dimension
    strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * grid_shape[i + 1]

    # The group spans dims [topology_tier .. ndim-1]
    # Fix the prefix (dims 0 .. topology_tier-1) to rank's own coordinates
    # and enumerate all combinations within the span
    span_dims = list(range(topology_tier, ndim))
    span_sizes = [grid_shape[d] for d in span_dims]
    total_in_span = math.prod(span_sizes)

    if num_devices > total_in_span:
        raise ValueError(
            f"num_devices={num_devices} exceeds span size {total_in_span} "
            f"for tier={topology_tier}, grid_shape={grid_shape}"
        )

    # Base rank: fix prefix dims, set span dims to 0
    base_rank = sum(coord[d] * strides[d] for d in range(topology_tier))

    # Enumerate num_devices consecutive ranks within the span
    group = [base_rank + i for i in range(num_devices)]
    return group


# ============================================================================
# Script generation (offline mode)
# ============================================================================

def _script_header(
    op_type: str,
    message_bytes: int,
    num_devices: int,
    topology_tier: int,
    group_ranks: List[int],
    dtype: str,
    grid_shape: List[int],
    bench_mode: str = "kernel",
) -> str:
    elem_size = _DTYPE_ELEM_SIZE.get(dtype, 2)
    num_elements = message_bytes // elem_size
    return dedent(f"""\
        #!/usr/bin/env python3
        \"\"\"Auto-generated communication microbenchmark script.

        Op: {op_type}
        Message bytes: {message_bytes}
        Num devices: {num_devices}
        Topology tier: {topology_tier}  (grid_shape={grid_shape})
        Group ranks: {group_ranks}
        Bench mode: {bench_mode}
        Generated by: tools/perf_data_collection/generate_comm_microbench.py

        Run with:
            torchrun --nproc_per_node={num_devices} <this_script>.py [--output-csv results.csv]
        \"\"\"

        import argparse
        import csv
        import os
        import statistics
        import time
        from pathlib import Path
        import torch
        import torch.distributed as dist

        try:
            import torch_npu  # noqa: F401
            BACKEND = "hccl"
            DEVICE = "npu"
        except ImportError:
            BACKEND = "gloo"
            DEVICE = "cpu"
            print("WARNING: torch_npu not available, using gloo/CPU (results not meaningful)")

        WARMUP_ITERS = {WARMUP_ITERS}
        BENCH_ITERS = {BENCH_ITERS}
        MESSAGE_BYTES = {message_bytes}
        NUM_ELEMENTS = {num_elements}
        NUM_DEVICES = {num_devices}
        TOPOLOGY_TIER = {topology_tier}
        GROUP_RANKS = {group_ranks}
        DTYPE = {dtype}
        DTYPE_CSV = "{_DTYPE_TO_CSV.get(dtype, 'DT_BF16')}"
        BENCH_MODE = "{bench_mode}"


        def setup():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            if DEVICE == "npu":
                torch.npu.set_device(rank % torch.npu.device_count())
            dist.init_process_group(backend=BACKEND, rank=rank, world_size=world_size)
            return rank, world_size


        def bench(rank, world_size, output_csv=None):
            group = dist.new_group(ranks=GROUP_RANKS)

            if rank not in GROUP_RANKS:
                return

    """)


def _op_body(op_type: str) -> str:
    if op_type == "all_reduce":
        return dedent("""\
            tensor = torch.randn(NUM_ELEMENTS, dtype=DTYPE, device=DEVICE)

            def run_op():
                dist.all_reduce(tensor, group=group)
        """)
    elif op_type == "all_gather":
        return dedent("""\
            local_tensor = torch.randn(NUM_ELEMENTS, dtype=DTYPE, device=DEVICE)
            gather_list = [torch.empty_like(local_tensor) for _ in GROUP_RANKS]

            def run_op():
                dist.all_gather(gather_list, local_tensor, group=group)
        """)
    elif op_type == "reduce_scatter":
        return dedent("""\
            input_list = [torch.randn(NUM_ELEMENTS, dtype=DTYPE, device=DEVICE) for _ in GROUP_RANKS]
            output_tensor = torch.empty(NUM_ELEMENTS, dtype=DTYPE, device=DEVICE)

            def run_op():
                dist.reduce_scatter(output_tensor, input_list, group=group)
        """)
    elif op_type == "all_to_all":
        return dedent("""\
            per_rank = max(1, NUM_ELEMENTS // len(GROUP_RANKS))
            input_list = [torch.randn(per_rank, dtype=DTYPE, device=DEVICE) for _ in GROUP_RANKS]
            output_list = [torch.empty(per_rank, dtype=DTYPE, device=DEVICE) for _ in GROUP_RANKS]

            def run_op():
                dist.all_to_all(output_list, input_list, group=group)
        """)
    else:
        raise ValueError(f"Unknown op_type: {op_type}")


def _bench_tail(op_type: str, bench_mode: str = "kernel") -> str:
    # Common CSV-writing and main() footer shared by all modes (top-level, 0 indent)
    csv_and_main = dedent(f"""\

        def _write_result(rank, duration_us, output_csv):
            bandwidth_gbps = MESSAGE_BYTES / (duration_us * 1e-6) / 1e9
            if rank == GROUP_RANKS[0]:
                print(f"op={op_type}  bytes={{MESSAGE_BYTES}}  devices={{len(GROUP_RANKS)}}"
                      f"  tier={{TOPOLOGY_TIER}}  rank={{rank}}  group={{GROUP_RANKS}}"
                      f"  duration={{duration_us:.2f}}us  bw={{bandwidth_gbps:.2f}}GB/s")
                if output_csv:
                    write_header = not Path(output_csv).exists()
                    with open(output_csv, "a", newline="") as f:
                        w = csv.writer(f)
                        if write_header:
                            w.writerow(["message_bytes", "num_devices", "dtype",
                                        "topology_tier", "Duration(us)", "bandwidth_gbps"])
                        w.writerow([MESSAGE_BYTES, len(GROUP_RANKS), DTYPE_CSV,
                                    TOPOLOGY_TIER, f"{{duration_us:.2f}}", f"{{bandwidth_gbps:.2f}}"])


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--output-csv", default=None)
            args = parser.parse_args()
            rank, world_size = setup()
            try:
                bench(rank, world_size, output_csv=args.output_csv)
            finally:
                dist.destroy_process_group()


        if __name__ == "__main__":
            main()
    """)

    if bench_mode == "kernel":
        body = dedent(f"""\
        # --- Peer op mapping for kernel (alternating) mode ---
        _PEER_OP = {{
            "all_gather": "reduce_scatter",
            "reduce_scatter": "all_gather",
        }}

        def _create_peer_op(group):
            peer_type = _PEER_OP.get("{op_type}")
            if peer_type is None:
                return None
            if peer_type == "all_gather":
                local_t = torch.randn(NUM_ELEMENTS, dtype=DTYPE, device=DEVICE)
                gather_list = [torch.empty_like(local_t) for _ in GROUP_RANKS]
                def peer_op():
                    dist.all_gather(gather_list, local_t, group=group)
            elif peer_type == "reduce_scatter":
                input_list = [torch.randn(NUM_ELEMENTS, dtype=DTYPE, device=DEVICE) for _ in GROUP_RANKS]
                output_t = torch.empty(NUM_ELEMENTS, dtype=DTYPE, device=DEVICE)
                def peer_op():
                    dist.reduce_scatter(output_t, input_list, group=group)
            else:
                return None
            return peer_op

        peer_run_op = _create_peer_op(group)

        # Warmup: alternating if peer exists
        for _ in range(WARMUP_ITERS):
            if peer_run_op is not None:
                peer_run_op()
            run_op()
        if DEVICE == "npu":
            torch.npu.synchronize()

        # Benchmark: per-iteration event timing with alternating execution
        durations_us = []
        for _ in range(BENCH_ITERS):
            if peer_run_op is not None:
                peer_run_op()
                if DEVICE == "npu":
                    torch.npu.synchronize()

            if DEVICE == "npu":
                start_evt = torch.npu.Event(enable_timing=True)
                end_evt = torch.npu.Event(enable_timing=True)
                start_evt.record()
                run_op()
                end_evt.record()
                torch.npu.synchronize()
                durations_us.append(start_evt.elapsed_time(end_evt) * 1000)  # ms -> us
            else:
                t0 = time.perf_counter()
                run_op()
                durations_us.append((time.perf_counter() - t0) * 1e6)

        duration_us = statistics.median(durations_us)
        _write_result(rank, duration_us, output_csv)
        """)
        return _indent(body, 4) + "\n" + csv_and_main

    elif bench_mode == "event":
        body = dedent(f"""\
        # Warmup
        for _ in range(WARMUP_ITERS):
            run_op()
        if DEVICE == "npu":
            torch.npu.synchronize()

        # Benchmark: per-iteration event timing, take median
        durations_us = []
        for _ in range(BENCH_ITERS):
            if DEVICE == "npu":
                torch.npu.synchronize()
                start_evt = torch.npu.Event(enable_timing=True)
                end_evt = torch.npu.Event(enable_timing=True)
                start_evt.record()
                run_op()
                end_evt.record()
                torch.npu.synchronize()
                durations_us.append(start_evt.elapsed_time(end_evt) * 1000)  # ms -> us
            else:
                t0 = time.perf_counter()
                run_op()
                durations_us.append((time.perf_counter() - t0) * 1e6)

        duration_us = statistics.median(durations_us)
        _write_result(rank, duration_us, output_csv)
        """)
        return _indent(body, 4) + "\n" + csv_and_main

    else:
        # pipeline mode: original logic (backward compat)
        body = dedent(f"""\
        # Warmup
        for _ in range(WARMUP_ITERS):
            run_op()
        if DEVICE == "npu":
            torch.npu.synchronize()

        # Benchmark
        if DEVICE == "npu":
            torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(BENCH_ITERS):
            run_op()
        if DEVICE == "npu":
            torch.npu.synchronize()
        elapsed = time.perf_counter() - start

        duration_us = elapsed / BENCH_ITERS * 1e6
        _write_result(rank, duration_us, output_csv)
        """)
        return _indent(body, 4) + "\n" + csv_and_main


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line.strip() else line for line in text.splitlines()) + "\n"


def generate_comm_script(
    op_type: str,
    message_bytes: int,
    num_devices: int,
    topology_tier: int,
    group_ranks: List[int],
    dtype: str = "torch.bfloat16",
    grid_shape: Optional[List[int]] = None,
    bench_mode: str = "kernel",
) -> str:
    """Generate a self-contained communication benchmark script.

    Args:
        op_type: One of all_reduce, all_gather, reduce_scatter, all_to_all
        message_bytes: Total message size in bytes
        num_devices: Number of devices in the communicator group
        topology_tier: Resolved tier (0=inter_pod, 1=intra_pod, 2=die_level)
        group_ranks: Explicit list of ranks in the group
        dtype: Tensor dtype string
        grid_shape: Hardware grid shape for documentation
        bench_mode: "kernel" (alternating), "event" (per-iter), or "pipeline" (original)
    """
    header = _script_header(
        op_type, message_bytes, num_devices, topology_tier, group_ranks, dtype,
        grid_shape or [], bench_mode=bench_mode,
    )
    op_body = _indent(_op_body(op_type), 4)
    tail = _bench_tail(op_type, bench_mode=bench_mode)
    return header + op_body + "\n" + tail


# ============================================================================
# Direct run mode (--do-run)
# ============================================================================

def _build_run_op(
    op_type: str, message_bytes: int, dtype_str: str, device: str, group, group_ranks: List[int],
) -> Callable:
    """Build a run_op closure for the given comm op and message size."""
    import torch
    import torch.distributed as dist

    dtype = getattr(torch, dtype_str.replace("torch.", ""))
    elem_size = _DTYPE_ELEM_SIZE.get(dtype_str, 2)
    num_elements = message_bytes // elem_size
    num_devices = len(group_ranks)

    if op_type == "all_reduce":
        tensor = torch.randn(num_elements, dtype=dtype, device=device)
        def run_op(): dist.all_reduce(tensor, group=group)
    elif op_type == "all_gather":
        # Use tensor-based API so operator_details records c10d::_allgather_base_
        # (matching _OP_TO_C10D_NAMES and inference profiling).
        local_tensor = torch.randn(num_elements, dtype=dtype, device=device)
        output_tensor = torch.empty(num_elements * num_devices, dtype=dtype, device=device)
        def run_op(): dist.all_gather_into_tensor(output_tensor, local_tensor, group=group)
    elif op_type == "reduce_scatter":
        # Use tensor-based API so operator_details records c10d::_reduce_scatter_base_
        input_tensor = torch.randn(num_elements * num_devices, dtype=dtype, device=device)
        output_tensor = torch.empty(num_elements, dtype=dtype, device=device)
        def run_op(): dist.reduce_scatter_tensor(output_tensor, input_tensor, group=group)
    elif op_type == "all_to_all":
        per_rank = max(1, num_elements // num_devices)
        input_list = [torch.randn(per_rank, dtype=dtype, device=device) for _ in group_ranks]
        output_list = [torch.empty(per_rank, dtype=dtype, device=device) for _ in group_ranks]
        def run_op(): dist.all_to_all(output_list, input_list, group=group)
    else:
        raise ValueError(f"Unknown op_type: {op_type}")
    return run_op


# Peer op mapping for alternating mode.
# In production, allGather follows reduceScatter and vice versa (MC2 pattern).
_PEER_OP = {
    "all_gather": "reduce_scatter",
    "reduce_scatter": "all_gather",
}


def _build_peer_run_op(
    op_type: str, message_bytes: int, dtype_str: str, device: str, group, group_ranks: List[int],
) -> Optional[Callable]:
    """Build a peer run_op for alternating execution, or None if no peer defined."""
    peer_type = _PEER_OP.get(op_type)
    if peer_type is None:
        return None
    return _build_run_op(peer_type, message_bytes, dtype_str, device, group, group_ranks)


def run_benchmark(
    op_type: str,
    message_bytes: int,
    group_ranks: List[int],
    topology_tier: int,
    dtype_str: str,
    output_csv: Optional[str],
    group=None,
    bench_mode: str = "profiler",
) -> Optional[dict]:
    """Run a single benchmark directly in the current process.

    Args:
        group: pre-created dist.ProcessGroup. If None, creates one internally
               (only safe when called once per group_ranks combination).
        bench_mode: "profiler" (方案 B, aligns Comm_NO), "kernel" (hcom_* Duration, aligns Communication),
                    "event" (方案 A, hcom_kernel only), or "pipeline" (original, backward compat).

    Note:
        In profiler/kernel mode, only the first rank in group_ranks runs the profiler.
        Other ranks use pipeline mode to participate in collective communication
        without starting their own profiler (avoids resource contention and /tmp bloat).
    """
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        print("ERROR: torch not available", file=sys.stderr)
        return None

    try:
        import torch_npu  # noqa: F401
        is_npu = True
    except ImportError:
        is_npu = False

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    # Bind each rank to its own NPU device (critical for HCCL init)
    if is_npu:
        torch.npu.set_device(local_rank)
        device = f"npu:{local_rank}"
    else:
        device = "cpu"

    if group is None:
        group = dist.new_group(ranks=group_ranks)
    if rank not in group_ranks:
        return None

    dtype = getattr(torch, dtype_str.replace("torch.", ""))
    elem_size = _DTYPE_ELEM_SIZE.get(dtype_str, 2)
    num_elements = message_bytes // elem_size
    num_devices = len(group_ranks)

    if op_type == "all_reduce":
        tensor = torch.randn(num_elements, dtype=dtype, device=device)
        def run_op(): dist.all_reduce(tensor, group=group)
    elif op_type == "all_gather":
        local_tensor = torch.randn(num_elements, dtype=dtype, device=device)
        output_tensor = torch.empty(num_elements * num_devices, dtype=dtype, device=device)
        def run_op(): dist.all_gather_into_tensor(output_tensor, local_tensor, group=group)
    elif op_type == "reduce_scatter":
        input_tensor = torch.randn(num_elements * num_devices, dtype=dtype, device=device)
        output_tensor = torch.empty(num_elements, dtype=dtype, device=device)
        def run_op(): dist.reduce_scatter_tensor(output_tensor, input_tensor, group=group)
    elif op_type == "all_to_all":
        per_rank = max(1, num_elements // num_devices)
        input_list = [torch.randn(per_rank, dtype=dtype, device=device) for _ in group_ranks]
        output_list = [torch.empty(per_rank, dtype=dtype, device=device) for _ in group_ranks]
        def run_op(): dist.all_to_all(output_list, input_list, group=group)
    else:
        raise ValueError(f"Unknown op_type: {op_type}")

    # ---- Measurement: dispatch by bench_mode ----
    # In profiler mode, only the first rank in the group runs the profiler to
    # avoid resource contention (multiple profilers writing /tmp simultaneously).
    # All ranks must execute the same number of run_op() calls to avoid hangs.
    if bench_mode == "profiler" and is_npu:
        is_leader = (rank == group_ranks[0])
        duration_us = _run_bench_profiler(run_op, op_type, is_npu, is_leader=is_leader)
        if duration_us is None:
            return None  # follower ranks don't report results
    elif bench_mode == "kernel" and is_npu:
        is_leader = (rank == group_ranks[0])
        duration_us = _run_bench_kernel(run_op, op_type, is_npu, is_leader=is_leader)
        if duration_us is None:
            return None  # follower ranks don't report results
    elif bench_mode == "alternating":
        peer_run_op = _build_peer_run_op(
            op_type, message_bytes, dtype_str, device, group, group_ranks,
        )
        duration_us = _run_bench_alternating(run_op, is_npu, peer_run_op=peer_run_op)
    elif bench_mode == "event":
        duration_us = _run_bench_event(run_op, is_npu)
    else:
        duration_us = _run_bench_pipeline(run_op, is_npu)

    bandwidth_gbps = message_bytes / (duration_us * 1e-6) / 1e9

    result = {
        "message_bytes": message_bytes,
        "num_devices": num_devices,
        "dtype": _DTYPE_TO_CSV.get(dtype_str, "DT_BF16"),
        "topology_tier": topology_tier,
        "Duration(us)": round(duration_us, 2),
        "bandwidth_gbps": round(bandwidth_gbps, 2),
    }

    if rank == group_ranks[0]:
        print(
            f"op={op_type}  bytes={message_bytes}  devices={num_devices}"
            f"  tier={topology_tier}  rank={rank}  group={group_ranks}"
            f"  duration={duration_us:.2f}us  bw={bandwidth_gbps:.2f}GB/s"
        )
        if output_csv:
            _append_csv(output_csv, result)

    return result


def _append_csv(path: str, row: dict) -> None:
    p = Path(path)
    write_header = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ============================================================================
# Bench mode: profiler (方案 B — aligns with operator_details = Comm_NO)
# ============================================================================

def _parse_comm_duration(prof_dir: str, op_type: str) -> List[float]:
    """Parse operator_details.csv from profiler output, extract Device Total Duration.

    Profiler output structure (Ascend CANN 8.5):
        prof_dir/
            ASCEND_PROFILER_OUTPUT_{timestamp}/
                {rank_id}/
                    operator_details.csv

    The c10d::* entry is the top-level operator whose Device Total Duration
    includes both hcom_kernel and AicpuKernel, matching Comm_NO exactly.
    """
    target_prefixes = _OP_TO_C10D_NAMES[op_type]
    durations: List[float] = []

    pattern = os.path.join(prof_dir, "**", "operator_details.csv")
    csv_files = glob.glob(pattern, recursive=True)

    if not csv_files:
        print(f"WARNING: No operator_details.csv found in {prof_dir}", file=sys.stderr)
        return durations

    csv_path = sorted(csv_files)[0]  # rank 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "")
            if any(name.startswith(p) for p in target_prefixes):
                dur = float(row.get("Device Total Duration(us)", "0"))
                if dur > 0:
                    durations.append(dur)

    return durations


def _warmup_and_sync(run_op, is_npu: bool) -> None:
    """Shared warmup: run WARMUP_ITERS iterations then sync."""
    import torch

    for _ in range(WARMUP_ITERS):
        run_op()
    if is_npu:
        torch.npu.synchronize()


def _profiler_loop_steps(run_op, is_npu: bool) -> None:
    """Execute the profiler-mode iteration pattern (total_steps with per-step sync).

    Used by both the profiler leader (inside profiler context) and followers
    (without profiler) to keep collective communication in lockstep.
    """
    import torch

    total_steps = PROFILER_WAIT_ITERS + PROFILER_WARMUP_ITERS + PROFILER_ACTIVE_ITERS
    for _ in range(total_steps):
        run_op()
        if is_npu:
            torch.npu.synchronize()


def _run_bench_profiler(run_op, op_type: str, is_npu: bool, *, is_leader: bool = True) -> Optional[float]:
    """Run bench via torch_npu.profiler, return median Device Total Duration (us).

    This measures the same physical quantity as operator_details in inference
    profiling, which equals Comm_NO per-call (including AicpuKernel overhead).

    When is_leader=False, executes the same call pattern without starting a
    profiler (follower mode for non-leader ranks in collective communication).
    """
    _warmup_and_sync(run_op, is_npu)

    if not is_leader:
        _profiler_loop_steps(run_op, is_npu)
        return None

    prof_dir = tempfile.mkdtemp(prefix="comm_bench_prof_")

    try:
        import torch
        import torch_npu  # noqa: F811

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
            data_simplification=True,
        )

        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(
                wait=PROFILER_WAIT_ITERS,
                warmup=PROFILER_WARMUP_ITERS,
                active=PROFILER_ACTIVE_ITERS,
                repeat=1,
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
            experimental_config=experimental_config,
        ) as prof:
            total_steps = PROFILER_WAIT_ITERS + PROFILER_WARMUP_ITERS + PROFILER_ACTIVE_ITERS
            for _step in range(total_steps):
                run_op()
                if is_npu:
                    torch.npu.synchronize()
                prof.step()

        durations = _parse_comm_duration(prof_dir, op_type)
        if not durations:
            raise RuntimeError(
                f"No {_OP_TO_C10D_NAMES[op_type]} entries found in "
                f"operator_details.csv under {prof_dir}"
            )

        return statistics.median(durations)

    finally:
        try:
            shutil.rmtree(prof_dir)
        except OSError as e:
            print(f"WARNING: Failed to clean up profiler dir {prof_dir}: {e}", file=sys.stderr)


def _run_bench_profiler_batch(
    op_type: str,
    msg_bytes_list: List[int],
    dtype_str: str,
    device: str,
    group,
    group_ranks: List[int],
    is_npu: bool,
    is_leader: bool,
) -> Optional[Dict[int, float]]:
    """Run ONE profiler session for all msg_sizes, return {msg_bytes: median_us}.

    CANN profiler cannot be started/stopped repeatedly in the same process.
    This function batches all msg_sizes into a single profiler session to avoid
    the crash. Each msg_size gets PROFILER_ACTIVE_ITERS iterations in the active
    phase; durations are split by position in operator_details.csv.

    All ranks in group_ranks must call this function together (collective ops).
    Only is_leader=True rank runs the profiler; others run the same call pattern.
    Returns None for non-leader ranks.
    """
    import torch

    # Build run_op for each msg_size
    run_ops: List[Tuple[int, Callable]] = [
        (mb, _build_run_op(op_type, mb, dtype_str, device, group, group_ranks))
        for mb in msg_bytes_list
    ]

    # Warmup all msg_sizes (outside profiler)
    for _, run_op in run_ops:
        for _ in range(WARMUP_ITERS):
            run_op()
    if is_npu:
        torch.npu.synchronize()

    n_sizes = len(run_ops)
    total_active = n_sizes * PROFILER_ACTIVE_ITERS
    total_steps = PROFILER_WAIT_ITERS + PROFILER_WARMUP_ITERS + total_active

    if not is_leader:
        # Follower: match exact call pattern without profiler
        # Wait + warmup phase: use first op
        first_run_op = run_ops[0][1]
        for _ in range(PROFILER_WAIT_ITERS + PROFILER_WARMUP_ITERS):
            first_run_op()
            if is_npu:
                torch.npu.synchronize()
        # Active phase: each msg_size for PROFILER_ACTIVE_ITERS
        for _, run_op in run_ops:
            for _ in range(PROFILER_ACTIVE_ITERS):
                run_op()
                if is_npu:
                    torch.npu.synchronize()
        return None

    prof_dir = tempfile.mkdtemp(prefix="comm_bench_prof_")

    try:
        import torch_npu  # noqa: F811

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
            data_simplification=True,
        )

        first_run_op = run_ops[0][1]

        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(
                wait=PROFILER_WAIT_ITERS,
                warmup=PROFILER_WARMUP_ITERS,
                active=total_active,
                repeat=1,
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
            experimental_config=experimental_config,
        ) as prof:
            # Wait + warmup phase: use first op (data discarded by profiler)
            for _ in range(PROFILER_WAIT_ITERS + PROFILER_WARMUP_ITERS):
                first_run_op()
                if is_npu:
                    torch.npu.synchronize()
                prof.step()
            # Active phase: each msg_size for PROFILER_ACTIVE_ITERS
            for _, run_op in run_ops:
                for _ in range(PROFILER_ACTIVE_ITERS):
                    run_op()
                    if is_npu:
                        torch.npu.synchronize()
                    prof.step()

        # Parse all durations (chronological order in operator_details.csv)
        durations = _parse_comm_duration(prof_dir, op_type)

        expected = total_active
        if len(durations) < expected:
            print(
                f"WARNING: Expected {expected} entries but got {len(durations)} "
                f"for {op_type} in {prof_dir}",
                file=sys.stderr,
            )

        # Split durations by msg_size: entries [i*N : (i+1)*N] → msg_bytes_list[i]
        results: Dict[int, float] = {}
        for i, (msg_bytes, _) in enumerate(run_ops):
            start_idx = i * PROFILER_ACTIVE_ITERS
            end_idx = start_idx + PROFILER_ACTIVE_ITERS
            chunk = durations[start_idx:end_idx]
            if chunk:
                results[msg_bytes] = statistics.median(chunk)
            else:
                print(f"WARNING: No duration data for {op_type} msg_bytes={msg_bytes}", file=sys.stderr)

        return results

    finally:
        try:
            shutil.rmtree(prof_dir)
        except OSError as e:
            print(f"WARNING: Failed to clean up profiler dir {prof_dir}: {e}", file=sys.stderr)


# ============================================================================
# Bench mode: kernel (直接采集 hcom_* kernel Duration, 去 AivKernel)
# ============================================================================

def _parse_kernel_comm_duration(prof_dir: str, op_type: str) -> List[float]:
    """Parse kernel_details.csv, extract hcom_* Duration excluding AivKernel.

    This measures the same physical quantity as Communication in step_trace:
        Communication = Σ kernel_details hcom_* Duration (去 AivKernel)

    Deduplication: kernel_details.csv records each HCCL op twice —
    once as hcom_* (Stream ID = NaN) and once as AivKernel (on HCCL stream).
    We filter by: Type starts with 'hcom_' AND Name does NOT contain 'AivKernel'.
    """
    target_type = _OP_TO_KERNEL_TYPE[op_type]
    durations: List[float] = []

    pattern = os.path.join(prof_dir, "**", "kernel_details.csv")
    csv_files = glob.glob(pattern, recursive=True)

    if not csv_files:
        print(f"WARNING: No kernel_details.csv found in {prof_dir}", file=sys.stderr)
        return durations

    csv_path = sorted(csv_files)[0]  # rank 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_type = row.get("Type", "")
            row_name = row.get("Name", "")
            if row_type.startswith(target_type) and "AivKernel" not in row_name:
                dur = float(row.get("Duration(us)", "0"))
                if dur > 0:
                    durations.append(dur)

    return durations


def _run_bench_kernel(run_op, op_type: str, is_npu: bool, *, is_leader: bool = True) -> Optional[float]:
    """Run bench via torch_npu.profiler, return median hcom_* kernel Duration (us).

    Uses the same profiler collection as _run_bench_profiler() but parses
    kernel_details.csv instead of operator_details.csv, extracting only
    hcom_* Duration (excluding AivKernel duplicates).

    This aligns exactly with:
        Communication (step_trace) = Σ kernel_details hcom_* Duration (去 AivKernel)

    When is_leader=False, executes the same call pattern without starting a
    profiler (follower mode for non-leader ranks in collective communication).
    """
    _warmup_and_sync(run_op, is_npu)

    if not is_leader:
        _profiler_loop_steps(run_op, is_npu)
        return None

    prof_dir = tempfile.mkdtemp(prefix="comm_bench_kernel_")

    try:
        import torch
        import torch_npu  # noqa: F811

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
            data_simplification=True,
        )

        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(
                wait=PROFILER_WAIT_ITERS,
                warmup=PROFILER_WARMUP_ITERS,
                active=PROFILER_ACTIVE_ITERS,
                repeat=1,
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
            experimental_config=experimental_config,
        ) as prof:
            total_steps = PROFILER_WAIT_ITERS + PROFILER_WARMUP_ITERS + PROFILER_ACTIVE_ITERS
            for _step in range(total_steps):
                run_op()
                if is_npu:
                    torch.npu.synchronize()
                prof.step()

        durations = _parse_kernel_comm_duration(prof_dir, op_type)
        if not durations:
            raise RuntimeError(
                f"No {_OP_TO_KERNEL_TYPE[op_type]} entries found in "
                f"kernel_details.csv under {prof_dir}"
            )

        return statistics.median(durations)

    finally:
        try:
            shutil.rmtree(prof_dir)
        except OSError as e:
            print(f"WARNING: Failed to clean up profiler dir {prof_dir}: {e}", file=sys.stderr)


# ============================================================================
# Bench mode: event (方案 A — measures hcom_kernel only, no AicpuKernel)
# ============================================================================

def _run_bench_event(run_op, is_npu: bool) -> float:
    """Run bench via per-iteration NPU Event timing, return median Duration (us).

    NPU Events measure Device-side hcom_kernel execution time. This does NOT
    include AicpuKernel overhead, so it underestimates Comm_NO on models like
    Qwen3 where AicpuKernel > 0. Use as a fast sanity check alongside profiler mode.
    """
    import torch

    _warmup_and_sync(run_op, is_npu)

    durations_us: List[float] = []
    if is_npu:
        # Reuse a single pair of Event objects across iterations
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        for _ in range(BENCH_ITERS):
            torch.npu.synchronize()
            start_event.record()
            run_op()
            end_event.record()
            torch.npu.synchronize()
            durations_us.append(start_event.elapsed_time(end_event) * 1000)  # ms → us
    else:
        for _ in range(BENCH_ITERS):
            t0 = time.perf_counter()
            run_op()
            durations_us.append((time.perf_counter() - t0) * 1e6)

    return statistics.median(durations_us)


# ============================================================================
# Bench mode: alternating (peer→target pipelined, no sync between them)
# ============================================================================

def _run_bench_alternating(run_op, is_npu: bool, peer_run_op=None) -> float:
    """Run bench with alternating peer→target execution, NO sync between them.

    Simulates production pattern where allGather↔reduceScatter alternate
    every layer on the same HCCL stream. The peer op keeps HCCL internal
    buffers and RDMA links warm so the target op hits the hot path.

    Key difference from event mode: no torch.npu.synchronize() between
    peer and target — they execute back-to-back on the device stream.
    NPU Event timing only captures the target op's device execution.

    If peer_run_op is None, falls back to standard event-mode measurement.
    """
    import torch

    if peer_run_op is None:
        return _run_bench_event(run_op, is_npu)

    # Warmup: alternating without sync (match production pattern)
    for _ in range(WARMUP_ITERS):
        peer_run_op()
        run_op()
    if is_npu:
        torch.npu.synchronize()

    durations_us: List[float] = []
    if is_npu:
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        for _ in range(BENCH_ITERS):
            # Peer op: pipelined on same stream, NO sync after it
            peer_run_op()
            # Measure target op only
            start_event.record()
            run_op()
            end_event.record()
            torch.npu.synchronize()
            durations_us.append(start_event.elapsed_time(end_event) * 1000)  # ms → us
    else:
        for _ in range(BENCH_ITERS):
            peer_run_op()
            t0 = time.perf_counter()
            run_op()
            durations_us.append((time.perf_counter() - t0) * 1e6)

    return statistics.median(durations_us)


# ============================================================================
# Bench mode: pipeline (original — kept for backward compatibility)
# ============================================================================

def _run_bench_pipeline(run_op, is_npu: bool) -> float:
    """Original pipeline measurement: 100 iterations without per-iteration sync.

    WARNING: This measures pipeline steady-state throughput, NOT single-call
    latency. The result does NOT align with operator_details / Comm_NO.
    Kept only for backward compatibility and A/B comparison.
    """
    import torch

    _warmup_and_sync(run_op, is_npu)

    if is_npu:
        torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(BENCH_ITERS):
        run_op()
    if is_npu:
        torch.npu.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / BENCH_ITERS * 1e6


# ============================================================================
# CLI
# ============================================================================

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or run HCCL communication microbenchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""\
            Examples:
              # Generate scripts for tier-2 (die-level), 16 devices
              python generate_comm_microbench.py --output-dir ./scripts \\
                  --ops all_reduce --grid-shape 48 8 2 --num-devices 16 --topology-tier 2

              # Run all ops + all tiers in ONE torchrun session (recommended)
              # tier=1 (intra_pod): 16 devices; tier=2 (die_level): 2 devices
              # Writes hcom_allReduce_.csv / hcom_allGather_.csv / etc. to --output-dir
              torchrun --nproc_per_node=16 generate_comm_microbench.py \\
                  --do-run --output-dir ./hccl_data \\
                  --ops all_reduce all_gather reduce_scatter all_to_all \\
                  --grid-shape 48 8 2 --num-devices 16 2

              # Single op, single CSV (legacy)
              torchrun --nproc_per_node=16 generate_comm_microbench.py \\
                  --do-run --output-csv ./hccl_v8.5/hcom_allReduce_.csv \\
                  --ops all_reduce --grid-shape 48 8 2
        """),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Script generation mode: directory to write generated benchmark scripts "
            "(default: ./comm_scripts). "
            "Run mode (--do-run): directory to write per-op CSV files "
            "(hcom_allReduce_.csv, hcom_allGather_.csv, etc.). "
            "Ignored in run mode when --output-csv is given."
        ),
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        default=_COMM_OPS,
        choices=_COMM_OPS,
        help="Communication ops to benchmark (default: all 4)",
    )
    parser.add_argument(
        "--num-devices",
        type=int,
        nargs="+",
        default=[16],
        help="Number of devices per communicator group (default: 16)",
    )
    parser.add_argument(
        "--topology-tier",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Topology tier(s) to benchmark: 0=inter_pod 1=intra_pod 2=die_level. "
            "Default: auto-resolve all tiers from --grid-shape and --num-devices."
        ),
    )
    parser.add_argument(
        "--grid-shape",
        type=int,
        nargs="+",
        default=[48, 8, 2],
        help=(
            "Hardware grid shape (outermost to innermost), e.g. '48 8 2' for "
            "ATLAS_800_A3 (48 pods × 8 nodes × 2 dies, stride=[16,2,1]). "
            "Used to resolve topology_tier from group composition. (default: 48 8 2)"
        ),
    )
    parser.add_argument(
        "--dtype",
        default="torch.bfloat16",
        choices=list(_DTYPE_ELEM_SIZE.keys()),
        help="Tensor dtype (default: torch.bfloat16)",
    )
    parser.add_argument(
        "--bytes-grid",
        type=int,
        nargs="+",
        default=None,
        help="Custom message_bytes grid (default: 1KB~512MB powers-of-4)",
    )
    parser.add_argument(
        "--do-run",
        action="store_true",
        dest="run",
        help="Run benchmarks directly instead of generating scripts (requires torchrun)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="CSV file to append results to (used with --do-run, format per §4.7)",
    )
    parser.add_argument(
        "--bench-mode",
        default="profiler",
        choices=_BENCH_MODES,
        help=(
            "Measurement mode (default: profiler). "
            "'profiler': torch_npu.profiler → operator_details Device Total Duration "
            "(aligns Comm_NO, includes AicpuKernel). "
            "'kernel': torch_npu.profiler → kernel_details hcom_* Duration "
            "(excludes AivKernel, aligns Communication in step_trace). "
            "'alternating': event timing with peer→target pipelined execution "
            "(no sync between peer and target, simulates production allGather↔reduceScatter). "
            "'event': per-iteration NPU Event timing (hcom_kernel only, fast). "
            "'pipeline': original 100-iteration pipeline (backward compat, does NOT align Comm_NO)."
        ),
    )
    return parser


def _iter_configs(
    ops: List[str],
    num_devices_list: List[int],
    topology_tiers: Optional[List[int]],
    grid_shape: List[int],
    bytes_grid: List[int],
    dtype: str,
) -> List[Tuple]:
    """Yield (op_type, message_bytes, num_devices, topology_tier, group_ranks) tuples.

    If topology_tiers is None, auto-resolve tier from group composition.
    Uses rank=0 as the anchor rank for group construction.
    """
    configs = []
    anchor_rank = 0
    for op_type in ops:
        for num_devices in num_devices_list:
            tiers_to_run = topology_tiers
            if tiers_to_run is None:
                # Build group anchored at rank 0 spanning the full num_devices,
                # then resolve tier from the group composition.
                try:
                    group = list(range(num_devices))
                    tier = resolve_topology_tier(group, grid_shape)
                    tiers_to_run = [tier]
                except ValueError:
                    tiers_to_run = [len(grid_shape) - 1]

            for tier in tiers_to_run:
                try:
                    group_ranks = build_group_for_tier(anchor_rank, num_devices, tier, grid_shape)
                except ValueError as e:
                    print(f"WARNING: skipping tier={tier}, num_devices={num_devices}: {e}", file=sys.stderr)
                    continue
                for msg_bytes in bytes_grid:
                    configs.append((op_type, msg_bytes, num_devices, tier, group_ranks))
    return configs


def main() -> None:
    args = build_argparser().parse_args()
    bytes_grid = args.bytes_grid or _DEFAULT_BYTES_GRID
    grid_shape = args.grid_shape

    # Validate: --output-csv only supports a single op to avoid silent data mixing
    if args.output_csv and len(args.ops) > 1:
        print(
            "ERROR: --output-csv only supports a single --ops value. "
            "Use --output-dir for multiple ops.",
            file=sys.stderr,
        )
        sys.exit(1)

    configs = _iter_configs(
        args.ops, args.num_devices, args.topology_tier,
        grid_shape, bytes_grid, args.dtype,
    )

    if args.run:
        try:
            import torch.distributed as dist
            if not dist.is_initialized():
                dist.init_process_group(backend="hccl" if _has_torch_npu() else "gloo")
        except Exception as e:
            print(f"ERROR: Failed to initialize distributed: {e}", file=sys.stderr)
            print("Hint: run with `torchrun --nproc_per_node=N generate_comm_microbench.py --do-run ...`",
                  file=sys.stderr)
            sys.exit(1)

        import torch
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if _has_torch_npu():
            import torch_npu  # noqa: F401
            torch.npu.set_device(local_rank)

        # Resolve per-op output CSV paths.
        # Priority: --output-csv (single file, legacy) > --output-dir (per-op files) > None
        if args.output_csv:
            def _csv_for_op(op_type: str) -> Optional[str]:
                return args.output_csv
        elif args.output_dir:
            run_out_dir = Path(args.output_dir)
            run_out_dir.mkdir(parents=True, exist_ok=True)
            def _csv_for_op(op_type: str) -> Optional[str]:
                return str(run_out_dir / _OP_TO_CSV_FILENAME[op_type])
        else:
            def _csv_for_op(op_type: str) -> Optional[str]:
                return None

        # Pre-create one process group per unique group_ranks to avoid
        # repeated hcclCommInitRootInfoConfig calls (HCCL error code 1).
        # dist.new_group() must be called by ALL ranks in the world even if
        # they are not in the group — so we call it unconditionally here.
        group_cache: dict = {}
        unique_groups = []
        seen = set()
        for _, _, _, _, group_ranks in configs:
            key = tuple(group_ranks)
            if key not in seen:
                seen.add(key)
                unique_groups.append(group_ranks)
        for group_ranks in unique_groups:
            key = tuple(group_ranks)
            group_cache[key] = dist.new_group(ranks=list(group_ranks))

        # Global warmup: run each (op, group, msg_bytes) combination once to
        # trigger HCCL JIT compilation for ALL message sizes before benchmarking.
        # HCCL compiles different internal kernels per message size; warming up
        # only the smallest size leaves mid-range sizes (1~5MB) un-compiled,
        # causing the first real measurement to include JIT overhead.
        # Always use pipeline mode for warmup (fast, no profiler overhead).
        if rank == 0:
            print("Running global warmup to trigger HCCL JIT compilation "
                  f"({len(configs)} configs)...")
        warmed = set()
        for op_type, msg_bytes, _, _, group_ranks in configs:
            wkey = (op_type, msg_bytes, tuple(group_ranks))
            if wkey not in warmed:
                warmed.add(wkey)
                run_benchmark(
                    op_type, msg_bytes, group_ranks,
                    resolve_topology_tier(list(group_ranks), grid_shape),
                    args.dtype, output_csv=None,
                    group=group_cache[tuple(group_ranks)],
                    bench_mode="pipeline",
                )

        if rank == 0:
            print(f"Global warmup done. Starting benchmarks (mode={args.bench_mode})...\n")

        if args.bench_mode == "profiler" and _has_torch_npu():
            # Profiler mode: batch all msg_sizes per (op, group) into ONE
            # profiler session to avoid CANN profiler repeated start/stop crash.
            batched: OrderedDict = OrderedDict()
            for op_type, msg_bytes, num_devices, tier, group_ranks in configs:
                key = (op_type, tuple(group_ranks))
                if key not in batched:
                    batched[key] = []
                batched[key].append((msg_bytes, num_devices, tier))

            is_npu = True
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            device = f"npu:{local_rank}"

            for (op_type, gr_tuple), items in batched.items():
                group_ranks = list(gr_tuple)
                group = group_cache[gr_tuple]
                is_member = rank in group_ranks

                if is_member:
                    is_leader = rank == group_ranks[0]
                    msg_bytes_list = [mb for mb, _, _ in items]

                    if is_leader:
                        print(f"[profiler-batch] op={op_type}  group={group_ranks}  "
                              f"msg_sizes={len(msg_bytes_list)}  active_iters={PROFILER_ACTIVE_ITERS}")

                    results = _run_bench_profiler_batch(
                        op_type, msg_bytes_list, args.dtype, device,
                        group, group_ranks, is_npu, is_leader,
                    )

                    if results and is_leader:
                        for msg_bytes, nd, tier in items:
                            if msg_bytes not in results:
                                continue
                            duration_us = results[msg_bytes]
                            bandwidth_gbps = msg_bytes / (duration_us * 1e-6) / 1e9
                            row = {
                                "message_bytes": msg_bytes,
                                "num_devices": nd,
                                "dtype": _DTYPE_TO_CSV.get(args.dtype, "DT_BF16"),
                                "topology_tier": tier,
                                "Duration(us)": round(duration_us, 2),
                                "bandwidth_gbps": round(bandwidth_gbps, 2),
                            }
                            print(
                                f"  op={op_type}  bytes={msg_bytes}  devices={nd}"
                                f"  tier={tier}  duration={duration_us:.2f}us"
                                f"  bw={bandwidth_gbps:.2f}GB/s"
                            )
                            csv_path = _csv_for_op(op_type)
                            if csv_path:
                                _append_csv(csv_path, row)

                # World barrier: keep ALL ranks in sync across batches.
                # Without this, non-member ranks skip ahead and enter the
                # next batch's warmup collective ops before members finish,
                # causing hangs when the next batch requires more ranks.
                dist.barrier()
        elif args.bench_mode == "alternating" and _has_torch_npu():
            # Alternating mode: ops with a peer (allGather↔reduceScatter) use
            # pipelined event timing; ops without a peer (allReduce) use
            # profiler-batch (kernel mode) to avoid NPU Event sync overhead
            # dominating short-duration kernels.
            has_peer_configs = [c for c in configs if _PEER_OP.get(c[0]) is not None]
            no_peer_configs = [c for c in configs if _PEER_OP.get(c[0]) is None]

            # 1) Ops with peer: per-point alternating measurement
            for op_type, msg_bytes, num_devices, tier, group_ranks in has_peer_configs:
                run_benchmark(
                    op_type, msg_bytes, group_ranks, tier, args.dtype,
                    _csv_for_op(op_type),
                    group=group_cache[tuple(group_ranks)],
                    bench_mode="alternating",
                )

            # 2) Ops without peer: profiler-batch (kernel mode) for accuracy
            if no_peer_configs:
                batched: OrderedDict = OrderedDict()
                for op_type, msg_bytes, num_devices, tier, group_ranks in no_peer_configs:
                    key = (op_type, tuple(group_ranks))
                    if key not in batched:
                        batched[key] = []
                    batched[key].append((msg_bytes, num_devices, tier))

                is_npu = True
                local_rank = int(os.environ.get("LOCAL_RANK", rank))
                device = f"npu:{local_rank}"

                for (op_type, gr_tuple), items in batched.items():
                    group_ranks = list(gr_tuple)
                    group = group_cache[gr_tuple]
                    is_member = rank in group_ranks

                    if is_member:
                        is_leader = rank == group_ranks[0]
                        msg_bytes_list = [mb for mb, _, _ in items]

                        if is_leader:
                            print(f"[alternating/profiler-fallback] op={op_type}  "
                                  f"group={group_ranks}  msg_sizes={len(msg_bytes_list)}")

                        results = _run_bench_profiler_batch(
                            op_type, msg_bytes_list, args.dtype, device,
                            group, group_ranks, is_npu, is_leader,
                        )

                        if results and is_leader:
                            for msg_bytes, nd, tier in items:
                                if msg_bytes not in results:
                                    continue
                                duration_us = results[msg_bytes]
                                bandwidth_gbps = msg_bytes / (duration_us * 1e-6) / 1e9
                                row = {
                                    "message_bytes": msg_bytes,
                                    "num_devices": nd,
                                    "dtype": _DTYPE_TO_CSV.get(args.dtype, "DT_BF16"),
                                    "topology_tier": tier,
                                    "Duration(us)": round(duration_us, 2),
                                    "bandwidth_gbps": round(bandwidth_gbps, 2),
                                }
                                print(
                                    f"  op={op_type}  bytes={msg_bytes}  devices={nd}"
                                    f"  tier={tier}  duration={duration_us:.2f}us"
                                    f"  bw={bandwidth_gbps:.2f}GB/s"
                                )
                                csv_path = _csv_for_op(op_type)
                                if csv_path:
                                    _append_csv(csv_path, row)

                    dist.barrier()
        else:
            # Event / pipeline mode: per-point measurement (no profiler session issue)
            for op_type, msg_bytes, num_devices, tier, group_ranks in configs:
                run_benchmark(
                    op_type, msg_bytes, group_ranks, tier, args.dtype,
                    _csv_for_op(op_type),
                    group=group_cache[tuple(group_ranks)],
                    bench_mode=args.bench_mode,
                )

        dist.destroy_process_group()
        if rank == 0:
            n = len(configs)
            out_info = args.output_csv or args.output_dir or "(no output)"
            print(f"\nCompleted {n} benchmarks. Results saved to {out_info}")
    else:
        output_dir = Path(args.output_dir or "./comm_scripts")
        output_dir.mkdir(parents=True, exist_ok=True)

        for op_type, msg_bytes, num_devices, tier, group_ranks in configs:
            script = generate_comm_script(
                op_type, msg_bytes, num_devices, tier, group_ranks, args.dtype, grid_shape,
                bench_mode=args.bench_mode,
            )
            fname = f"comm_{op_type}_B{msg_bytes}_D{num_devices}_T{tier}.py"
            (output_dir / fname).write_text(script)

        print(f"Generated {len(configs)} communication benchmark scripts in {output_dir}")
        print(f"  ops: {args.ops}")
        print(f"  num_devices: {args.num_devices}")
        print(f"  grid_shape: {grid_shape}")
        print(f"  topology_tiers: {args.topology_tier or 'auto'}")
        print(f"  message_bytes: {len(bytes_grid)} sizes ({bytes_grid[0]}~{bytes_grid[-1]} bytes)")
        print()
        print("To run a script:")
        print(f"  torchrun --nproc_per_node=<N> {output_dir}/comm_<op>_B<bytes>_D<N>_T<tier>.py --output-csv results.csv")


def _has_torch_npu() -> bool:
    try:
        import torch_npu  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
