#!/bin/bash
# ============================================================
# HCCL Communication Microbenchmark -- kernel mode full collection
# ============================================================
#
# Uses CANN profiler -> kernel_details.csv to collect hcom_* Duration,
# excluding AivKernel, aligning with Communication in step_trace.
# Skips torch.npu.synchronize() between iterations for HCCL pipeline overlap.
#
# Profiler interference optimization:
#   - Small msgs (<512KB): batched into one profiler session, 10 active iters
#   - Large msgs (>=512KB): per msg_bytes separate session, active=1,
#     repeated 10 sessions taking median, eliminates ring buffer pressure
#
# Hardware: ATLAS_800_A3, grid_shape=[48,8,2]
# Device groups: nd=16 (tier=1), nd=8 (tier=1), nd=4 (tier=1), nd=2 (tier=2)
#
# Organization: per-operator collection, each op iterates all nd
#   Round 1: allReduce       (nd=16/8/4/2)
#   Round 2: allGather + reduceScatter (nd=16/8/4/2)
#   Round 3: alltoallv       (nd=16/8/4/2)
#
# Data points:
#   Standard grid (powers of 2, 1KB~512MB, 20 points) +
#   Production msg_bytes (Qwen3 nd=16 + DSV3 nd=8 actual comm sizes)
#
# Fault tolerant: single session failure does not abort collection.
#
# Usage:
#   bash run_comm_bench.sh [OUTPUT_DIR]
#   # Default output: ./hccl_bench_data/
# ============================================================

set +e  # SIGSEGV at shutdown is a known torch_npu driver issue

BASEDIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$BASEDIR/generate_comm_microbench.py"
OUTPUT_DIR="${1:-./hccl_bench_data}"

mkdir -p "$OUTPUT_DIR"

echo "=== HCCL Communication Microbenchmark ==="
echo "Script: $SCRIPT"
echo "Output: $OUTPUT_DIR"
echo "Mode: kernel (profiler -> kernel_details.csv hcom_* Duration, no inter-iter sync)"
echo ""
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ============================================================
# Message bytes grids
# ============================================================

# Standard grid (powers of 2, 1KB~512MB)
STD_SMALL="1024 2048 4096 8192 16384 32768 65536 131072 262144 524288"
STD_LARGE="1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456 536870912"
STD_ALL="$STD_SMALL $STD_LARGE"

# Production msg_bytes -- Qwen3 (nd=16, TP=16)
QWEN3_BYTES="19008 75968 163840 284880 303872 327680 588752 607744 1320960 2641920 2652160 3952640 5263360 9185280 22435840 41943040"

# Production msg_bytes -- DSV3 (nd=8, TP=8)
DSV3_BYTES="272 528 1152 3072 7168 14336 32320 64640 74880 129280 149760 223488 294912 465920 595968 786432 946176 1863680 2781184 3670016"

# nd=4, nd=2: production msg_bytes (subset of DSV3)
ND4_BYTES="1152 3072 7168 14336 64640 129280 294912 786432 1863680 3670016"

# Pre-compute merged grids
ALL_ND16="$(echo $STD_ALL $QWEN3_BYTES | tr ' ' '\n' | sort -un | tr '\n' ' ')"
ALL_ND8="$(echo $STD_ALL $DSV3_BYTES | tr ' ' '\n' | sort -un | tr '\n' ' ')"
ALL_ND4="$(echo $STD_ALL $ND4_BYTES | tr ' ' '\n' | sort -un | tr '\n' ' ')"
ALL_ND2="$ALL_ND4"

# Helper: run a torchrun session
run_session() {
    local port=$1 ndev=$2 ops=$3 bytes=$4 outdir=$5
    local desc=$6
    echo ""
    echo "--- [$desc] $(date '+%H:%M:%S') ---"
    echo "    port=$port  ndev=$ndev  ops=$ops"
    echo "    bytes_count=$(echo $bytes | wc -w | tr -d ' ')"

    MASTER_PORT=$port torchrun --nproc_per_node=$ndev "$SCRIPT" \
        --do-run \
        --bench-mode kernel \
        --ops $ops \
        --grid-shape 48 8 2 \
        --num-devices $ndev \
        --bytes-grid $bytes \
        --output-dir "$outdir"
    local rc=$?

    if [ $rc -eq 0 ]; then
        echo "    OK ($(date '+%H:%M:%S'))"
    elif [ $rc -eq 139 ]; then
        echo "    SIGSEGV at shutdown (known torch_npu issue, data is safe)"
    else
        echo "    WARNING: exit code $rc, continuing..." >&2
    fi
}

PORT=29700

# ============================================================
# Round 1: allReduce -- nd=16, 8, 4, 2
# ============================================================

echo "=========================================="
echo "Round 1: allReduce (all nd)"
echo "=========================================="

run_session $((PORT++)) 16 "all_reduce" "$ALL_ND16" "$OUTPUT_DIR" \
    "allReduce nd=16 (std+qwen3)"

run_session $((PORT++)) 8 "all_reduce" "$ALL_ND8" "$OUTPUT_DIR" \
    "allReduce nd=8 (std+dsv3)"

run_session $((PORT++)) 4 "all_reduce" "$ALL_ND4" "$OUTPUT_DIR" \
    "allReduce nd=4"

run_session $((PORT++)) 2 "all_reduce" "$ALL_ND2" "$OUTPUT_DIR" \
    "allReduce nd=2"

# ============================================================
# Round 2: allGather + reduceScatter -- nd=16, 8, 4, 2
# ============================================================

echo ""
echo "=========================================="
echo "Round 2: allGather + reduceScatter (all nd)"
echo "=========================================="

run_session $((PORT++)) 16 "all_gather reduce_scatter" "$ALL_ND16" "$OUTPUT_DIR" \
    "allGather+reduceScatter nd=16 (std+qwen3)"

run_session $((PORT++)) 8 "all_gather reduce_scatter" "$ALL_ND8" "$OUTPUT_DIR" \
    "allGather+reduceScatter nd=8 (std+dsv3)"

run_session $((PORT++)) 4 "all_gather reduce_scatter" "$ALL_ND4" "$OUTPUT_DIR" \
    "allGather+reduceScatter nd=4"

run_session $((PORT++)) 2 "all_gather reduce_scatter" "$ALL_ND2" "$OUTPUT_DIR" \
    "allGather+reduceScatter nd=2"

# ============================================================
# Round 3: alltoallv -- nd=16, 8, 4, 2
# ============================================================

echo ""
echo "=========================================="
echo "Round 3: alltoallv (all nd)"
echo "=========================================="

run_session $((PORT++)) 16 "all_to_all" "$STD_ALL" "$OUTPUT_DIR" \
    "alltoallv nd=16"

run_session $((PORT++)) 8 "all_to_all" "$STD_ALL" "$OUTPUT_DIR" \
    "alltoallv nd=8"

run_session $((PORT++)) 4 "all_to_all" "$STD_ALL" "$OUTPUT_DIR" \
    "alltoallv nd=4"

run_session $((PORT++)) 2 "all_to_all" "$STD_ALL" "$OUTPUT_DIR" \
    "alltoallv nd=2"

# ============================================================
# Summary
# ============================================================

echo ""
echo "=========================================="
echo "Collection complete"
echo "=========================================="
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Output directory:"
ls -lh "$OUTPUT_DIR/"
echo ""
echo "CSV files:"
wc -l "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "(no CSV files found)"
echo ""
echo "Mode: kernel (profiler -> kernel_details hcom_* Duration, excl. AivKernel, no inter-iter sync)"
echo "Optimization: >=512KB per-msg session (active=1, 10 sessions), <512KB batch (active=10)"
