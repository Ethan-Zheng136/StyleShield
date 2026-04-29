#!/bin/bash
# 完整流水线：torchrun 多 GPU 并行生成 + 打分过滤 + 合并
#
# 集群上提交命令：
#   PIP_REQUIRE_VIRTUALENV=false
#   pip install torch transformers tqdm sentencepiece protobuf
#   cd /root/workspace/AIGC_FUCK_7
#   bash scripts/run_full_build.sh 2>&1 | tee /tmp/full_build.log

set -e
cd /root/workspace/AIGC_FUCK_7

DATA_DIR="/root/workspace/AIGC_FUCK"

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected!"
    exit 1
fi

echo "============================================"
echo "  Full Multi-Domain Build Pipeline"
echo "  GPUs: $NUM_GPUS"
echo "  Started: $(date)"
echo "============================================"

# ── Step 1: Generate News (100k pairs) ──
echo ""
echo "[Step 1/5] Building NEWS: 100000 pairs on $NUM_GPUS GPUs (batch=32)..."
torchrun --nproc_per_node "$NUM_GPUS" --nnodes 1 --node_rank 0 \
    --master_addr localhost --master_port 29501 \
    scripts/build_dataset_distributed.py \
    --domain news --num 100000 --batch_size 32

echo ""
echo "  Merging news worker outputs..."
NEWS_FINAL="${DATA_DIR}/dataset_news_pairs.jsonl"
NEWS_TMP="${NEWS_FINAL}.tmp"
> "$NEWS_TMP"
NEWS_TOTAL=0
for ((i=0; i<NUM_GPUS; i++)); do
    WFILE="${DATA_DIR}/dataset_news_pairs_w${i}.jsonl"
    if [ -f "$WFILE" ] && [ -s "$WFILE" ]; then
        COUNT=$(wc -l < "$WFILE")
        echo "    Worker $i: $COUNT pairs"
        cat "$WFILE" >> "$NEWS_TMP"
        NEWS_TOTAL=$((NEWS_TOTAL + COUNT))
        rm "$WFILE"
    else
        echo "    Worker $i: MISSING or empty!"
    fi
done
if [ $NEWS_TOTAL -gt 0 ]; then
    mv "$NEWS_TMP" "$NEWS_FINAL"
    echo "  News total: $NEWS_TOTAL pairs"
else
    rm -f "$NEWS_TMP"
    echo "  ERROR: No news data produced!"
fi
echo "[Step 1/5] News DONE at $(date)"

# ── Step 2: Generate Academic (25k pairs) ──
echo ""
echo "[Step 2/5] Building ACADEMIC: 25000 pairs on $NUM_GPUS GPUs (batch=32)..."
torchrun --nproc_per_node "$NUM_GPUS" --nnodes 1 --node_rank 0 \
    --master_addr localhost --master_port 29502 \
    scripts/build_dataset_distributed.py \
    --domain academic --num 25000 --batch_size 32

echo ""
echo "  Merging academic worker outputs..."
ACAD_FINAL="${DATA_DIR}/dataset_academic_pairs.jsonl"
ACAD_TMP="${ACAD_FINAL}.tmp"
> "$ACAD_TMP"
ACAD_TOTAL=0
for ((i=0; i<NUM_GPUS; i++)); do
    WFILE="${DATA_DIR}/dataset_academic_pairs_w${i}.jsonl"
    if [ -f "$WFILE" ] && [ -s "$WFILE" ]; then
        COUNT=$(wc -l < "$WFILE")
        echo "    Worker $i: $COUNT pairs"
        cat "$WFILE" >> "$ACAD_TMP"
        ACAD_TOTAL=$((ACAD_TOTAL + COUNT))
        rm "$WFILE"
    else
        echo "    Worker $i: MISSING or empty!"
    fi
done
if [ $ACAD_TOTAL -gt 0 ]; then
    mv "$ACAD_TMP" "$ACAD_FINAL"
    echo "  Academic total: $ACAD_TOTAL pairs"
else
    rm -f "$ACAD_TMP"
    echo "  ERROR: No academic data produced!"
fi
echo "[Step 2/5] Academic DONE at $(date)"

# ── Step 3: MPU scoring + filtering ──
echo ""
echo "[Step 3/5] Scoring + filtering with AIGC detector..."
for DOMAIN in news academic; do
    RAW="${DATA_DIR}/dataset_${DOMAIN}_pairs.jsonl"
    FILTERED="${DATA_DIR}/dataset_${DOMAIN}_pairs_filtered.jsonl"
    if [ -f "$RAW" ] && [ -s "$RAW" ]; then
        echo "  Processing ${DOMAIN}..."
        python scripts/score_and_filter.py \
            --input "$RAW" \
            --output "$FILTERED" \
            --batch_size 64
    else
        echo "  Skipping ${DOMAIN} (no data)"
    fi
done
echo "[Step 3/5] Filtering DONE at $(date)"

# ── Step 4: Merge all domains ──
echo ""
echo "[Step 4/5] Merging all datasets..."
MERGE_INPUTS="${DATA_DIR}/dataset_zhihu_pairs_full.jsonl"
for DOMAIN in news academic; do
    FILTERED="${DATA_DIR}/dataset_${DOMAIN}_pairs_filtered.jsonl"
    if [ -f "$FILTERED" ] && [ -s "$FILTERED" ]; then
        MERGE_INPUTS="$MERGE_INPUTS $FILTERED"
    fi
done
python scripts/merge_datasets.py \
    --inputs $MERGE_INPUTS \
    --output "${DATA_DIR}/dataset_multidomain.jsonl" \
    --merge_only
echo "[Step 4/5] Merge DONE at $(date)"

# ── Step 5: Summary ──
echo ""
echo "============================================"
echo "  ALL DONE at $(date)"
echo "============================================"
echo ""
echo "Final dataset:"
wc -l "${DATA_DIR}/dataset_multidomain.jsonl"
echo ""
echo "Per-domain counts:"
python3 -c "
import json
from collections import Counter
counts = Counter()
with open('${DATA_DIR}/dataset_multidomain.jsonl') as f:
    for line in f:
        r = json.loads(line)
        counts[r.get('source', 'unknown')] += 1
for s, c in sorted(counts.items()):
    pct = c / sum(counts.values()) * 100
    print(f'  {s}: {c} ({pct:.1f}%)')
print(f'  TOTAL: {sum(counts.values())}')
"
echo ""
echo "Filter stats:"
for DOMAIN in news academic; do
    RAW="${DATA_DIR}/dataset_${DOMAIN}_pairs.jsonl"
    FILTERED="${DATA_DIR}/dataset_${DOMAIN}_pairs_filtered.jsonl"
    if [ -f "$RAW" ] && [ -f "$FILTERED" ]; then
        RAW_N=$(wc -l < "$RAW")
        FILT_N=$(wc -l < "$FILTERED")
        echo "  ${DOMAIN}: ${RAW_N} raw → ${FILT_N} kept"
    fi
done
echo ""
echo "Train with: data_path=${DATA_DIR}/dataset_multidomain.jsonl"
