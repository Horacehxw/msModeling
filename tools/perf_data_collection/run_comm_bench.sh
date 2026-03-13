#!/bin/bash
# C9/C10-4: HCCL communication microbenchmark runner
# Hardware: ATLAS_800_A3, 16-card single node, grid_shape=[48,8,2]
# Usage: bash run_comm_bench.sh [OUTPUT_DIR]
#
# Runs all 4 ops across 3 group sizes in separate torchrun sessions:
#   Session 1 — n=16, tier=1 (intra_pod):  16 devices, group=[0..15]
#   Session 2 — n=4,  tier=1 (intra_pod):   4 devices, group=[0..3]  (DSV3 TP=4)
#   Session 3 — n=2,  tier=2 (die_level):   2 devices, group=[0,1]
#
# Each session appends rows to the same CSV files, so the final output
# contains all 3 group sizes (33 rows per file: 11 sizes × 3 n_dev).
#
# Output files (named to match op_mapping.yaml kernel_type):
#   hcom_allReduce_.csv / hcom_allGather_.csv / hcom_reduceScatter_.csv / hcom_alltoallv_.csv
#
# SIGSEGV at shutdown is a known torch_npu driver issue; CSV is written before crash.

OUTPUT_DIR="${1:-./hccl_data}"
SCRIPT="$(dirname "$0")/generate_comm_microbench.py"

mkdir -p "$OUTPUT_DIR"

echo "=== C9/C10-4 HCCL Microbenchmark ==="
echo "Output dir : $OUTPUT_DIR"
echo "Sessions   : n=16 tier=1 (intra_pod) | n=4 tier=1 (intra_pod) | n=2 tier=2 (die_level)"
echo "Ops        : all_reduce all_gather reduce_scatter all_to_all"
echo ""

# --- Session 1: n=16, tier=1 ---
echo "--- Session 1/3: n=16, tier=1 (intra_pod) ---"
MASTER_PORT=29600 torchrun --nproc_per_node=16 "$SCRIPT" \
    --do-run \
    --ops all_reduce all_gather reduce_scatter all_to_all \
    --grid-shape 48 8 2 \
    --num-devices 16 \
    --output-dir "$OUTPUT_DIR"
rc=$?
[ $rc -ne 0 ] && [ $rc -ne 139 ] && exit $rc

# --- Session 2: n=4, tier=1 (C10-4: DSV3 TP=4) ---
echo ""
echo "--- Session 2/3: n=4, tier=1 (intra_pod, DSV3 TP=4) ---"
MASTER_PORT=29601 torchrun --nproc_per_node=4 "$SCRIPT" \
    --do-run \
    --ops all_reduce all_gather reduce_scatter all_to_all \
    --grid-shape 48 8 2 \
    --num-devices 4 \
    --output-dir "$OUTPUT_DIR"
rc=$?
[ $rc -ne 0 ] && [ $rc -ne 139 ] && exit $rc

# --- Session 3: n=2, tier=2 ---
echo ""
echo "--- Session 3/3: n=2, tier=2 (die_level) ---"
MASTER_PORT=29602 torchrun --nproc_per_node=2 "$SCRIPT" \
    --do-run \
    --ops all_reduce all_gather reduce_scatter all_to_all \
    --grid-shape 48 8 2 \
    --num-devices 2 \
    --output-dir "$OUTPUT_DIR"
rc=$?
[ $rc -ne 0 ] && [ $rc -ne 139 ] && exit $rc

echo ""
echo "=== Done. CSV files in $OUTPUT_DIR ==="
ls -lh "$OUTPUT_DIR"
