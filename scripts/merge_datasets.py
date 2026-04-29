"""合并多 domain 数据集并打分。

用法：
  # 仅合并
  python scripts/merge_datasets.py --merge_only \
      --inputs dataset_zhihu.jsonl dataset_news.jsonl dataset_academic.jsonl \
      --output dataset_multidomain.jsonl

  # 合并 + 用检测器打分 (补齐 mpu_score)
  python scripts/merge_datasets.py \
      --inputs dataset_zhihu.jsonl dataset_news.jsonl dataset_academic.jsonl \
      --output dataset_multidomain.jsonl
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from tqdm import tqdm

DETECTOR_PATH = "/root/workspace/AIGC_FUCK/models/AIGC_detector_zhv3"
DATA_DIR = "/root/workspace/AIGC_FUCK"


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def score_texts(texts: list[str], model, tokenizer, device, batch_size: int = 64) -> list[float]:
    import torch
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=512,
                        return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            ai_probs = probs[:, 1].cpu().tolist()
        scores.extend(ai_probs)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Input JSONL files (absolute paths or relative to DATA_DIR)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output merged JSONL (absolute path or relative to DATA_DIR)")
    parser.add_argument("--merge_only", action="store_true",
                        help="Skip scoring, just merge")
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_records = []
    for inp in args.inputs:
        path = inp if os.path.isabs(inp) else os.path.join(DATA_DIR, inp)
        records = load_jsonl(path)
        if not records:
            print(f"  WARNING: {path} is empty, skipping")
            continue
        print(f"  Loaded {len(records)} from {path} (source={records[0].get('source', '?')})")
        all_records.extend(records)

    print(f"\nTotal records: {len(all_records)}")

    domain_counts = {}
    for r in all_records:
        s = r.get("source", "unknown")
        domain_counts[s] = domain_counts.get(s, 0) + 1
    for s, c in sorted(domain_counts.items()):
        print(f"  {s}: {c}")

    if not args.merge_only:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        need_score = [r for r in all_records
                      if r.get("human_mpu_score", -1) < 0 or r.get("ai_mpu_score", -1) < 0]
        if need_score:
            print(f"\nScoring {len(need_score)} records with detector...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(DETECTOR_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(DETECTOR_PATH).to(device)
            model.eval()

            human_texts = [r["human_text"] for r in need_score]
            ai_texts = [r["ai_text"] for r in need_score]

            print("  Scoring human texts...")
            human_scores = score_texts(human_texts, model, tokenizer, device)
            print("  Scoring AI texts...")
            ai_scores = score_texts(ai_texts, model, tokenizer, device)

            for r, hs, ais in zip(need_score, human_scores, ai_scores):
                r["human_mpu_score"] = round(hs, 4)
                r["ai_mpu_score"] = round(ais, 4)

            scored_humans = [s for s in human_scores]
            scored_ais = [s for s in ai_scores]
            print(f"  Human mean P(AI) = {sum(scored_humans)/len(scored_humans):.4f}")
            print(f"  AI mean P(AI) = {sum(scored_ais)/len(scored_ais):.4f}")

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(all_records)

    out_path = args.output if os.path.isabs(args.output) else os.path.join(DATA_DIR, args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_records)} records to {out_path}")


if __name__ == "__main__":
    main()
