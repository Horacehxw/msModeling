#!/bin/bash
# ============================================================
# HCCL Communication Microbenchmark — Unified Runner
# ============================================================
#
# 统一的通信算子 bench 采集脚本，合并了所有历史脚本的功能。
#
# 硬件：ATLAS_800_A3, grid_shape=[48,8,2]
# 设备组：nd=16 (tier=1), nd=8 (tier=1), nd=4 (tier=1), nd=2 (tier=2)
#
# 采集模式选择（依据 bench_vs_profiler_comm_20260318.md §4/§5）：
#
#   --bench-mode 参数说明：
#   alternating: peer→target 无 sync 流水执行，NPU Event 仅测 target op
#                allReduce 无 peer 时自动 fallback 到 event 模式
#   kernel:      profiler 采集 kernel_details.csv 中 hcom_* Duration（去 AivKernel）
#
#   allReduce 所有 msg_bytes → alternating（无 peer → event fallback，底噪可接受）
#   allGather/reduceScatter ≥1MB → alternating（peer 流水消除 1~5MB 预热偏高）
#   allGather/reduceScatter <1MB → kernel（profiler kernel_details，避免 event 底噪）
#
# 数据点：
#   标准网格 (powers of 2, 1KB~512MB, 20 个点) +
#   生产 msg_bytes (Qwen3 nd=16 + DSV3 nd=8 实际通信量)
#
# 用法：
#   bash run_comm_bench.sh [OUTPUT_BASE]
#   # 输出: OUTPUT_BASE/alternating/ 和 OUTPUT_BASE/kernel/
#
# 采集完成后本地运行 build_comm_csv.py 合并生成最终 CSV。
# ============================================================

set +e  # SIGSEGV at shutdown is a known torch_npu driver issue

OUTPUT_BASE="${1:-./hccl_bench_data}"
SCRIPT="$(dirname "$0")/generate_comm_microbench.py"

mkdir -p "$OUTPUT_BASE/alternating" "$OUTPUT_BASE/kernel"

echo "=== HCCL Communication Microbenchmark ==="
echo "Output: $OUTPUT_BASE"
echo ""

# --- PLACEHOLDER_GRID_DEFINITIONS ---

# ============================================================
# Message bytes grids
# ============================================================

# Standard grid (powers of 2, 1KB~512MB)
STD_SMALL="1024 2048 4096 8192 16384 32768 65536 131072 262144 524288"
STD_LARGE="1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456 536870912"
STD_ALL="$STD_SMALL $STD_LARGE"

# Production msg_bytes — Qwen3 (nd=16, TP=16)
# From trace_view.json: ISL=4096, OSL=1536, max_concurrency=64
QWEN3_BYTES="19008 75968 163840 284880 303872 327680 588752 607744 1320960 2641920 2652160 3952640 5263360 9185280 22435840 41943040"

# Production msg_bytes — DSV3 (nd=8, TP=8)
DSV3_BYTES="272 528 1152 3072 7168 14336 32320 64640 74880 129280 149760 223488 294912 465920 595968 786432 946176 1863680 2781184 3670016"

# nd=4, nd=2: production msg_bytes (subset of DSV3)
ND4_BYTES="1152 3072 7168 14336 64640 129280 294912 786432 1863680 3670016"

# Helper: split bytes into <1MB and >=1MB
split_bytes() {
    local small="" large=""
    for b in $1; do
        if [ "$b" -lt 1048576 ]; then
            small="$small $b"
        else
            large="$large $b"
        fi
    done
    SPLIT_SMALL="${small# }"
    SPLIT_LARGE="${large# }"
}

# Helper: run a torchrun session
run_session() {
    local port=$1 ndev=$2 mode=$3 ops=$4 bytes=$5 outdir=$6
    local desc=$7
    echo ""
    echo "--- $desc ---"
    MASTER_PORT=$port torchrun --nproc_per_node=$ndev "$SCRIPT" \
        --do-run \
        --bench-mode $mode \
        --ops $ops \
        --grid-shape 48 8 2 \
        --num-devices $ndev \
        --bytes-grid $bytes \
        --output-dir "$outdir"
    local rc=$?
    [ $rc -ne 0 ] && [ $rc -ne 139 ] && echo "FAILED (rc=$rc)" && exit $rc
}

# --- PLACEHOLDER_ROUND1 ---

# ============================================================
# Round 1: alternating 模式
#   allReduce: 所有 msg_bytes（无 peer → event fallback）
#   allGather/reduceScatter: 仅 ≥1MB（有 peer → no-sync 流水执行）
# ============================================================

echo "=========================================="
echo "Round 1: alternating 模式"
echo "=========================================="

PORT=29650

# --- nd=16: standard grid + Qwen3 production ---
ALL_ND16="$(echo $STD_ALL $QWEN3_BYTES | tr ' ' '\n' | sort -un | tr '\n' ' ')"
split_bytes "$ALL_ND16"
LARGE_ND16="$SPLIT_LARGE"

run_session $((PORT++)) 16 alternating "all_reduce" "$ALL_ND16" "$OUTPUT_BASE/alternating" \
    "nd=16 alternating: allReduce (std+qwen3, $(echo $ALL_ND16 | wc -w) points)"
run_session $((PORT++)) 16 alternating "all_gather reduce_scatter" "$LARGE_ND16" "$OUTPUT_BASE/alternating" \
    "nd=16 alternating: allGather/reduceScatter >=1MB ($(echo $LARGE_ND16 | wc -w) points)"

# --- nd=8: standard grid + DSV3 production ---
ALL_ND8="$(echo $STD_ALL $DSV3_BYTES | tr ' ' '\n' | sort -un | tr '\n' ' ')"
split_bytes "$ALL_ND8"
LARGE_ND8="$SPLIT_LARGE"

run_session $((PORT++)) 8 alternating "all_reduce" "$ALL_ND8" "$OUTPUT_BASE/alternating" \
    "nd=8 alternating: allReduce (std+dsv3, $(echo $ALL_ND8 | wc -w) points)"
run_session $((PORT++)) 8 alternating "all_gather reduce_scatter" "$LARGE_ND8" "$OUTPUT_BASE/alternating" \
    "nd=8 alternating: allGather/reduceScatter >=1MB ($(echo $LARGE_ND8 | wc -w) points)"

# --- nd=4: standard grid + production ---
ALL_ND4="$(echo $STD_ALL $ND4_BYTES | tr ' ' '\n' | sort -un | tr '\n' ' ')"
split_bytes "$ALL_ND4"
LARGE_ND4="$SPLIT_LARGE"

run_session $((PORT++)) 4 alternating "all_reduce" "$ALL_ND4" "$OUTPUT_BASE/alternating" \
    "nd=4 alternating: allReduce ($(echo $ALL_ND4 | wc -w) points)"
run_session $((PORT++)) 4 alternating "all_gather reduce_scatter" "$LARGE_ND4" "$OUTPUT_BASE/alternating" \
    "nd=4 alternating: allGather/reduceScatter >=1MB ($(echo $LARGE_ND4 | wc -w) points)"

# --- nd=2: standard grid + production ---
ALL_ND2="$ALL_ND4"  # same bytes as nd=4
split_bytes "$ALL_ND2"
LARGE_ND2="$SPLIT_LARGE"

run_session $((PORT++)) 2 alternating "all_reduce" "$ALL_ND2" "$OUTPUT_BASE/alternating" \
    "nd=2 alternating: allReduce ($(echo $ALL_ND2 | wc -w) points)"
run_session $((PORT++)) 2 alternating "all_gather reduce_scatter" "$LARGE_ND2" "$OUTPUT_BASE/alternating" \
    "nd=2 alternating: allGather/reduceScatter >=1MB ($(echo $LARGE_ND2 | wc -w) points)"

# ============================================================
# Round 2: kernel 模式（profiler kernel_details）— 小消息
#   allGather/reduceScatter <1MB
# ============================================================

echo ""
echo "=========================================="
echo "Round 2: kernel 模式 — 小消息"
echo "=========================================="

# --- nd=16 ---
split_bytes "$ALL_ND16"
SMALL_ND16="$SPLIT_SMALL"
run_session $((PORT++)) 16 kernel "all_gather reduce_scatter" "$SMALL_ND16" "$OUTPUT_BASE/kernel" \
    "nd=16 kernel: allGather/reduceScatter <1MB ($(echo $SMALL_ND16 | wc -w) points)"

# --- nd=8 ---
split_bytes "$ALL_ND8"
SMALL_ND8="$SPLIT_SMALL"
run_session $((PORT++)) 8 kernel "all_gather reduce_scatter" "$SMALL_ND8" "$OUTPUT_BASE/kernel" \
    "nd=8 kernel: allGather/reduceScatter <1MB ($(echo $SMALL_ND8 | wc -w) points)"

# --- nd=4 ---
split_bytes "$ALL_ND4"
SMALL_ND4="$SPLIT_SMALL"
run_session $((PORT++)) 4 kernel "all_gather reduce_scatter" "$SMALL_ND4" "$OUTPUT_BASE/kernel" \
    "nd=4 kernel: allGather/reduceScatter <1MB ($(echo $SMALL_ND4 | wc -w) points)"

# --- nd=2 ---
split_bytes "$ALL_ND2"
SMALL_ND2="$SPLIT_SMALL"
run_session $((PORT++)) 2 kernel "all_gather reduce_scatter" "$SMALL_ND2" "$OUTPUT_BASE/kernel" \
    "nd=2 kernel: allGather/reduceScatter <1MB ($(echo $SMALL_ND2 | wc -w) points)"

# ============================================================
# Summary
# ============================================================

echo ""
echo "=== 采集完成 ==="
echo ""
echo "alternating 模式:"
ls -lh "$OUTPUT_BASE/alternating/"
echo ""
echo "kernel 模式:"
ls -lh "$OUTPUT_BASE/kernel/"
echo ""
echo "下一步：运行 build_comm_csv.py 生成最终 CSV"
echo ""
echo "  python tools/perf_data_collection/build_comm_csv.py \\"
echo "    --alternating-dir $OUTPUT_BASE/alternating \\"
echo "    --kernel-dir $OUTPUT_BASE/kernel \\"
echo "    --profiler-trace-dir /path/to/dsv3-profiler \\"
echo "    --output-dir tensor_cast/.../hccl/v8.5/"
