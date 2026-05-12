"""多 GPU 并行构建 AI/Human 配对数据集（离线版）。

读取本地预下载的 raw_*.jsonl，用 Qwen 改写为 AI 版本。
不依赖 datasets 库，不需要联网。

用法：
  CUDA_VISIBLE_DEVICES=0 python scripts/build_dataset_parallel.py \
      --domain news --num 100000 --total_workers 8 --worker_id 0
"""

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


QWEN_PATH = "models/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "data/"

RAW_DATA = {
    "news": "data/raw_news_human.jsonl",
    "academic": "data/raw_academic_human.jsonl",
}

REWRITE_NEWS_PROMPT = """请用规范、正式的语言重写以下新闻报道，保持核心内容和事实不变，但使用标准的书面语体。

原文：
{text}

请直接输出重写后的文本，不要添加任何说明或标题："""

REWRITE_ACADEMIC_PROMPT = """请用标准的学术论文摘要格式重写以下研究摘要，保持研究内容和结论不变，使语言更加规范和专业。

原文：
{text}

请直接输出重写后的摘要，不要添加任何说明："""


def load_raw_data(domain: str) -> list[str]:
    path = RAW_DATA[domain]
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            texts.append(obj["text"])
    return texts


def generate_rewrite(model, tokenizer, text: str, domain: str,
                     max_new_tokens: int = 800) -> str:
    if domain == "news":
        prompt = REWRITE_NEWS_PROMPT.format(text=text[:1200])
    else:
        prompt = REWRITE_ACADEMIC_PROMPT.format(text=text[:800])

    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.3, top_p=0.85, do_sample=True,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["news", "academic"], required=True)
    parser.add_argument("--num", type=int, required=True)
    parser.add_argument("--total_workers", type=int, default=1)
    parser.add_argument("--worker_id", type=int, default=0)
    args = parser.parse_args()

    per_worker = args.num // args.total_workers
    start_idx = args.worker_id * per_worker
    if args.worker_id == args.total_workers - 1:
        per_worker = args.num - start_idx

    output_path = os.path.join(
        OUTPUT_DIR, f"dataset_{args.domain}_pairs_w{args.worker_id}.jsonl")

    existing = 0
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = sum(1 for _ in f)
    remaining = per_worker - existing
    if remaining <= 0:
        print(f"Worker {args.worker_id}: already done ({existing}/{per_worker})")
        return

    print(f"Worker {args.worker_id}/{args.total_workers}: "
          f"range [{start_idx}..{start_idx+per_worker}), "
          f"existing={existing}, todo={remaining}")

    print(f"Loading raw data from {RAW_DATA[args.domain]}...")
    all_texts = load_raw_data(args.domain)
    print(f"  Total raw texts: {len(all_texts)}")

    my_texts = all_texts[start_idx + existing: start_idx + per_worker]
    if not my_texts:
        print(f"Worker {args.worker_id}: no data in range, done.")
        return

    print(f"Loading Qwen from {QWEN_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    print(f"Generating {len(my_texts)} pairs...")
    t_start = time.time()
    written = 0
    skipped = 0

    pbar = tqdm(my_texts, desc=f"W{args.worker_id}",
                miniters=1, mininterval=5,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}] written={postfix}")
    pbar.set_postfix_str(f"{written}")

    with open(output_path, "a", encoding="utf-8") as fout:
        for i, human_text in enumerate(pbar):
            ai_text = generate_rewrite(model, tokenizer, human_text, args.domain)

            if len(ai_text) < 80:
                skipped += 1
                pbar.set_postfix_str(f"{written} (skip={skipped})")
                continue
            length_ratio = len(ai_text) / len(human_text) if len(human_text) > 0 else 0
            if length_ratio < 0.3 or length_ratio > 3.0:
                skipped += 1
                pbar.set_postfix_str(f"{written} (skip={skipped})")
                continue

            record = {
                "id": f"{args.domain}_{start_idx + existing + i:06d}",
                "source": args.domain,
                "human_text": human_text,
                "ai_text": ai_text,
                "human_mpu_score": -1.0,
                "ai_mpu_score": -1.0,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
            pbar.set_postfix_str(f"{written} (skip={skipped})")

    pbar.close()
    elapsed = time.time() - t_start
    rate = written / elapsed if elapsed > 0 else 0
    print(f"\nWorker {args.worker_id} done: {written} pairs, {skipped} skipped, "
          f"{elapsed:.0f}s ({rate:.2f} pairs/s) → {output_path}")


if __name__ == "__main__":
    main()
