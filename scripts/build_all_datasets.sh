#!/bin/bash
set -e

echo "============================================"
echo "  Multi-domain Dataset Builder"
echo "============================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" && cd "$SCRIPT_DIR/.."

# Step 1: Build academic (if not done)
ACAD_FILE="data/dataset_academic_pairs.jsonl"
if [ -f "$ACAD_FILE" ]; then
    ACAD_COUNT=$(wc -l < "$ACAD_FILE")
    echo "[Academic] Found $ACAD_COUNT existing pairs"
    if [ "$ACAD_COUNT" -lt 2000 ]; then
        echo "[Academic] Continuing generation..."
        python scripts/build_dataset.py --domain academic --num 2000
    else
        echo "[Academic] Already complete."
    fi
else
    echo "[Academic] Starting generation..."
    python scripts/build_dataset.py --domain academic --num 2000
fi

# Step 2: Build news
NEWS_FILE="data/dataset_news_pairs.jsonl"
if [ -f "$NEWS_FILE" ]; then
    NEWS_COUNT=$(wc -l < "$NEWS_FILE")
    echo "[News] Found $NEWS_COUNT existing pairs"
    if [ "$NEWS_COUNT" -lt 2000 ]; then
        echo "[News] Continuing generation..."
        python scripts/build_dataset.py --domain news --num 2000
    else
        echo "[News] Already complete."
    fi
else
    echo "[News] Starting generation..."
    python scripts/build_dataset.py --domain news --num 2000
fi

# Step 3: Merge + score
echo ""
echo "[Merge] Merging all datasets..."
python scripts/merge_datasets.py \
    --inputs \
    data/dataset_zhihu_pairs_full.jsonl \
    data/dataset_academic_pairs.jsonl \
    data/dataset_news_pairs.jsonl \
    --output data/dataset_multidomain.jsonl

echo ""
echo "============================================"
echo "  ALL DONE"
echo "============================================"
echo "Output: data/dataset_multidomain.jsonl"
wc -l data/dataset_multidomain.jsonl
