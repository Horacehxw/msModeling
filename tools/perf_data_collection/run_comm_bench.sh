#!/bin/bash
# C9: HCCL communication microbenchmark runner
# Hardware: ATLAS_800_A3, 16-card single node, grid_shape=[48,8,2]
# Usage: bash run_comm_bench.sh [OUTPUT_DIR]
#
# Runs all 4 ops + 2 topology tiers in ONE torchrun session to avoid HCCL re-init overhead.
#   tier=1 (intra_pod):  16 devices, group=[0..15], spans 8 nodes x 2 dies within pod
#   tier=2 (die_level):   2 devices, group=[0,1],   2 dies within one node
#   tier=0 (inter_pod): requires multi-node (>16 cards), not covered here
#
# Output files (named to match op_mapping.yaml kernel_type):
#   hcom_allReduce_.csv / hcom_allGather_.csv / hcom_reduceScatter_.csv / hcom_alltoallv_.csv
#
# SIGSEGV at shutdown is a known torch_npu driver issue; CSV is written before crash.

OUTPUT_DIR="${1:-./hccl_data}"
SCRIPT="$(dirname "$0")/generate_comm_microbench.py"

mkdir -p "$OUTPUT_DIR"

echo "=== C9 HCCL Microbenchmark ==="
echo "Output dir : $OUTPUT_DIR"
echo "Tiers      : tier=1 (16 devices, intra_pod) + tier=2 (2 devices, die_level)"
echo "Ops        : all_reduce all_gather reduce_scatter all_to_all"
echo ""

MASTER_PORT=29600 torchrun --nproc_per_node=16 "$SCRIPT" \
    --do-run \
    --ops all_reduce all_gather reduce_scatter all_to_all \
    --grid-shape 48 8 2 \
    --num-devices 16 2 \
    --output-dir "$OUTPUT_DIR"
rc=$?
# SIGSEGV (exitcode 139) at shutdown is a known torch_npu driver issue;
# CSV is written before crash. All other non-zero exit codes are real errors.
[ $rc -ne 0 ] && [ $rc -ne 139 ] && exit $rc

echo ""
echo "=== Done. CSV files in $OUTPUT_DIR ==="
ls -lh "$OUTPUT_DIR"
