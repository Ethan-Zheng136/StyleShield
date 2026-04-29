"""用 AIGC 检测器给数据集打分并过滤。

对每条记录：
  1. 计算 human_text 的 P(AI) 和 ai_text 的 P(AI)
  2. 过滤掉：
     - human P(AI) > 0.5  （检测器认为 human 也像 AI，噪声样本）
     - ai P(AI) < 0.5     （检测器认为 AI 不像 AI，改写失败）
     - human P(AI) 和 ai P(AI) 差距 < 0.2 （区分度不够）

用法：
  python scripts/score_and_filter.py \
      --input dataset_news_pairs.jsonl \
      --output dataset_news_pairs_filtered.jsonl \
      --batch_size 64
"""

import argparse
import json
import sys
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DETECTOR_PATH = "/root/workspace/AIGC_FUCK/models/AIGC_detector_zhv3"


def score_batch(texts: list[str], model, tokenizer, device,
                max_length: int = 512) -> list[float]:
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--human_threshold", type=float, default=0.5,
                        help="Remove if human P(AI) > this")
    parser.add_argument("--ai_threshold", type=float, default=0.5,
                        help="Remove if ai P(AI) < this")
    parser.add_argument("--min_gap", type=float, default=0.2,
                        help="Remove if ai_P(AI) - human_P(AI) < this")
    args = parser.parse_args()

    print(f"Loading detector from {DETECTOR_PATH}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(DETECTOR_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        DETECTOR_PATH).to(device)
    model.eval()

    print(f"Loading data from {args.input}...")
    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Total records: {len(records)}")

    print(f"Scoring (batch_size={args.batch_size})...")
    for i in tqdm(range(0, len(records), args.batch_size), desc="Scoring"):
        batch = records[i:i + args.batch_size]

        hu_texts = [r["human_text"] for r in batch]
        ai_texts = [r["ai_text"] for r in batch]

        hu_scores = score_batch(hu_texts, model, tokenizer, device)
        ai_scores = score_batch(ai_texts, model, tokenizer, device)

        for r, hs, ais in zip(batch, hu_scores, ai_scores):
            r["human_mpu_score"] = round(hs, 4)
            r["ai_mpu_score"] = round(ais, 4)

    kept = []
    reasons = {"human_too_high": 0, "ai_too_low": 0, "gap_too_small": 0}
    for r in records:
        hu = r["human_mpu_score"]
        ai = r["ai_mpu_score"]
        if hu > args.human_threshold:
            reasons["human_too_high"] += 1
            continue
        if ai < args.ai_threshold:
            reasons["ai_too_low"] += 1
            continue
        if ai - hu < args.min_gap:
            reasons["gap_too_small"] += 1
            continue
        kept.append(r)

    print(f"\nFiltering results:")
    print(f"  Input:  {len(records)}")
    if len(records) == 0:
        print(f"  ERROR: No records to filter!")
        sys.exit(1)
    print(f"  Kept:   {len(kept)} ({len(kept)/len(records)*100:.1f}%)")
    print(f"  Removed: {len(records) - len(kept)}")
    for reason, count in sorted(reasons.items()):
        print(f"    {reason}: {count}")

    hu_kept = [r["human_mpu_score"] for r in kept]
    ai_kept = [r["ai_mpu_score"] for r in kept]
    if hu_kept:
        print(f"\n  Kept stats:")
        print(f"    Human P(AI): mean={sum(hu_kept)/len(hu_kept):.4f}")
        print(f"    AI    P(AI): mean={sum(ai_kept)/len(ai_kept):.4f}")

    with open(args.output, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(kept)} records to {args.output}")


if __name__ == "__main__":
    main()
