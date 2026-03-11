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

    # Run directly (requires torchrun + torch_npu)
    torchrun --nproc_per_node=16 generate_comm_microbench.py \\
        --run --output-csv ./hccl_v8.5/hcom_allReduce_.csv \\
        --ops all_reduce --grid-shape 48 8 2
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Tuple

WARMUP_ITERS = 10
BENCH_ITERS = 100

_COMM_OPS = ["all_reduce", "all_gather", "reduce_scatter", "all_to_all"]

# message_bytes grid: 1KB ~ 512MB, powers of 4 (covers typical LLM TP/EP sizes)
_DEFAULT_BYTES_GRID = [
    1024,        # 1 KB
    4096,        # 4 KB
    16384,       # 16 KB
    65536,       # 64 KB
    262144,      # 256 KB
    1048576,     # 1 MB
    4194304,     # 4 MB
    16777216,    # 16 MB
    67108864,    # 64 MB
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
        Generated by: tools/perf_data_collection/generate_comm_microbench.py

        Run with:
            torchrun --nproc_per_node={num_devices} <this_script>.py [--output-csv results.csv]
        \"\"\"

        import argparse
        import csv
        import os
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


def _bench_tail(op_type: str) -> str:
    return dedent(f"""\
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
            bandwidth_gbps = MESSAGE_BYTES / (duration_us * 1e-6) / 1e9

            if rank == GROUP_RANKS[0]:
                print(f"op={'{op_type}'}  bytes={{MESSAGE_BYTES}}  devices={{len(GROUP_RANKS)}}"
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
    """
    header = _script_header(
        op_type, message_bytes, num_devices, topology_tier, group_ranks, dtype,
        grid_shape or []
    )
    op_body = _indent(_op_body(op_type), 4)
    tail = _bench_tail(op_type)
    return header + op_body + "\n" + tail


# ============================================================================
# Direct run mode (--run)
# ============================================================================

def run_benchmark(
    op_type: str,
    message_bytes: int,
    group_ranks: List[int],
    topology_tier: int,
    dtype_str: str,
    output_csv: Optional[str],
) -> Optional[dict]:
    """Run a single benchmark directly in the current process."""
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        print("ERROR: torch not available", file=sys.stderr)
        return None

    try:
        import torch_npu  # noqa: F401
        device = "npu"
    except ImportError:
        device = "cpu"

    rank = dist.get_rank()
    world_size = dist.get_world_size()

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
        gather_list = [torch.empty_like(local_tensor) for _ in group_ranks]
        def run_op(): dist.all_gather(gather_list, local_tensor, group=group)
    elif op_type == "reduce_scatter":
        input_list = [torch.randn(num_elements, dtype=dtype, device=device) for _ in group_ranks]
        output_tensor = torch.empty(num_elements, dtype=dtype, device=device)
        def run_op(): dist.reduce_scatter(output_tensor, input_list, group=group)
    elif op_type == "all_to_all":
        per_rank = max(1, num_elements // num_devices)
        input_list = [torch.randn(per_rank, dtype=dtype, device=device) for _ in group_ranks]
        output_list = [torch.empty(per_rank, dtype=dtype, device=device) for _ in group_ranks]
        def run_op(): dist.all_to_all(output_list, input_list, group=group)
    else:
        raise ValueError(f"Unknown op_type: {op_type}")

    for _ in range(WARMUP_ITERS):
        run_op()
    if device == "npu":
        torch.npu.synchronize()

    if device == "npu":
        torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(BENCH_ITERS):
        run_op()
    if device == "npu":
        torch.npu.synchronize()
    elapsed = time.perf_counter() - start

    duration_us = elapsed / BENCH_ITERS * 1e6
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

              # Generate for all 3 tiers
              python generate_comm_microbench.py --output-dir ./scripts \\
                  --ops all_reduce all_gather reduce_scatter \\
                  --grid-shape 48 8 2 --num-devices 16 64 128 --topology-tier 0 1 2

              # Run directly (requires torchrun + torch_npu)
              torchrun --nproc_per_node=16 generate_comm_microbench.py \\
                  --run --output-csv ./hccl_v8.5/hcom_allReduce_.csv \\
                  --ops all_reduce --grid-shape 48 8 2
        """),
    )
    parser.add_argument(
        "--output-dir",
        default="./comm_scripts",
        help="Directory to write generated benchmark scripts (default: ./comm_scripts)",
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
        "--run",
        action="store_true",
        help="Run benchmarks directly instead of generating scripts (requires torchrun)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="CSV file to append results to (used with --run, format per §4.7)",
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
                except Exception:
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
            print("Hint: run with `torchrun --nproc_per_node=N generate_comm_microbench.py --run ...`",
                  file=sys.stderr)
            sys.exit(1)

        rank = dist.get_rank()
        for op_type, msg_bytes, num_devices, tier, group_ranks in configs:
            run_benchmark(op_type, msg_bytes, group_ranks, tier, args.dtype, args.output_csv)

        dist.destroy_process_group()
        if rank == 0:
            print(f"\nCompleted {len(configs)} benchmarks."
                  + (f" Results saved to {args.output_csv}" if args.output_csv else ""))
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for op_type, msg_bytes, num_devices, tier, group_ranks in configs:
            script = generate_comm_script(
                op_type, msg_bytes, num_devices, tier, group_ranks, args.dtype, grid_shape
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
