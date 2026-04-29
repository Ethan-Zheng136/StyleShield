#!/bin/bash
# 多 GPU 并行生成数据集
#
# 用法：
#   bash scripts/run_build_parallel.sh --domain news --num 100000 --gpus 8
#   bash scripts/run_build_parallel.sh --domain academic --num 20000 --gpus 8

set -e

DOMAIN=""
NUM=0
GPUS=8

while [[ $# -gt 0 ]]; do
    case $1 in
        --domain) DOMAIN="$2"; shift 2 ;;
        --num)    NUM="$2";    shift 2 ;;
        --gpus)   GPUS="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$DOMAIN" ] || [ "$NUM" -eq 0 ]; then
    echo "Usage: bash $0 --domain news|academic --num N --gpus G"
    exit 1
fi

cd /root/workspace/AIGC_FUCK_7

PER_WORKER=$((NUM / GPUS))

echo "============================================"
echo "  Parallel Dataset Builder"
echo "  Domain: $DOMAIN"
echo "  Total:  $NUM pairs ($PER_WORKER per GPU)"
echo "  GPUs:   $GPUS"
echo "  Started: $(date)"
echo "============================================"

LOG_DIR="/tmp/build_${DOMAIN}_logs"
mkdir -p "$LOG_DIR"

PIDS=()

# Launch workers (PYTHONUNBUFFERED ensures tqdm writes to log immediately)
for ((i=0; i<GPUS; i++)); do
    echo "Launching worker $i on GPU $i..."
    CUDA_VISIBLE_DEVICES=$i PYTHONUNBUFFERED=1 python scripts/build_dataset_parallel.py \
        --domain "$DOMAIN" \
        --num "$NUM" \
        --total_workers "$GPUS" \
        --worker_id "$i" \
        > "${LOG_DIR}/worker_${i}.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All $GPUS workers launched (PIDs: ${PIDS[*]}). Monitoring..."
echo ""

# Monitor progress every 60 seconds
while true; do
    ALL_DONE=true
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            ALL_DONE=false
            break
        fi
    done

    if $ALL_DONE; then
        break
    fi

    echo "--- Progress at $(date '+%H:%M:%S') ---"
    for ((i=0; i<GPUS; i++)); do
        WFILE="/root/workspace/AIGC_FUCK/dataset_${DOMAIN}_pairs_w${i}.jsonl"
        if [ -f "$WFILE" ]; then
            COUNT=$(wc -l < "$WFILE")
        else
            COUNT=0
        fi
        # Get latest tqdm line from log
        TQDM=$(tail -c 500 "${LOG_DIR}/worker_${i}.log" 2>/dev/null | tr '\r' '\n' | grep -oE "W${i}:.*\|" | tail -1)
        if kill -0 "${PIDS[$i]}" 2>/dev/null; then
            STATUS="running"
        else
            STATUS="done"
        fi
        printf "  W%d [%s]: %5d pairs  %s\n" "$i" "$STATUS" "$COUNT" "$TQDM"
    done
    echo ""
    sleep 60
done

echo ""
echo "All workers finished at $(date)"

# Check for errors
ERRORS=0
for ((i=0; i<GPUS; i++)); do
    if grep -q "Traceback\|Error\|Exception" "${LOG_DIR}/worker_${i}.log" 2>/dev/null; then
        echo "  WARNING: Worker $i had errors! Check ${LOG_DIR}/worker_${i}.log"
        ERRORS=$((ERRORS + 1))
    fi
done
if [ $ERRORS -gt 0 ]; then
    echo "  $ERRORS workers had errors."
fi

# Merge worker outputs (only if at least one worker produced data)
echo ""
echo "Merging worker outputs..."
FINAL="/root/workspace/AIGC_FUCK/dataset_${DOMAIN}_pairs.jsonl"
MERGE_TMP="${FINAL}.tmp"
> "$MERGE_TMP"
TOTAL=0
MISSING=0
for ((i=0; i<GPUS; i++)); do
    WFILE="/root/workspace/AIGC_FUCK/dataset_${DOMAIN}_pairs_w${i}.jsonl"
    if [ -f "$WFILE" ] && [ -s "$WFILE" ]; then
        COUNT=$(wc -l < "$WFILE")
        echo "  Worker $i: $COUNT pairs"
        cat "$WFILE" >> "$MERGE_TMP"
        TOTAL=$((TOTAL + COUNT))
        rm "$WFILE"
    else
        echo "  Worker $i: MISSING or empty output!"
        MISSING=$((MISSING + 1))
    fi
done

if [ $TOTAL -gt 0 ]; then
    mv "$MERGE_TMP" "$FINAL"
else
    rm -f "$MERGE_TMP"
    echo "  ERROR: No data produced! Keeping existing $FINAL if any."
fi

echo ""
echo "============================================"
echo "  DONE: $TOTAL pairs → $FINAL"
echo "  Finished: $(date)"
echo "============================================"
