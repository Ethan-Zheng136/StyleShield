"""构建多 domain 中文 AI/Human 配对数据集。

策略：
  - 新闻：从 THUCNews（清华新闻数据集，83万条真实新闻）取 human_text，Qwen 改写为 ai_text
  - 学术：从 CLUE-CSL（2万条真实论文摘要）取 human_text，Qwen 改写为 ai_text
  - 保证 human_text 是真实人类写的，ai_text 是 LLM 标准输出风格

输出格式与现有 zhihu 数据集完全一致：
  {"id": "news_000000", "source": "news", "human_text": "...", "ai_text": "...",
   "human_mpu_score": -1, "ai_mpu_score": -1}

用法：
  python scripts/build_dataset.py --domain news --num 2000
  python scripts/build_dataset.py --domain academic --num 2000
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


QWEN_PATH = "models/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "data/"

PREFERRED_NEWS_CATEGORIES = ["社会", "教育", "科技", "财经", "时政", "家居", "娱乐"]

REWRITE_NEWS_PROMPT = """请用规范、正式的语言重写以下新闻报道，保持核心内容和事实不变，但使用标准的书面语体。

原文：
{text}

请直接输出重写后的文本，不要添加任何说明或标题："""

REWRITE_ACADEMIC_PROMPT = """请用标准的学术论文摘要格式重写以下研究摘要，保持研究内容和结论不变，使语言更加规范和专业。

原文：
{text}

请直接输出重写后的摘要，不要添加任何说明："""


def load_news_data(num_samples: int, seed: int = 42) -> list[dict]:
    """从 THUCNews 加载真实新闻文本，优选社会/教育/科技等类别。"""
    print("Loading THUCNews from HuggingFace (full dataset)...")
    ds = load_dataset("SirlyDreamer/THUCNews", split="train")
    print(f"  Total: {len(ds)} articles")

    random.seed(seed)

    preferred = []
    others = []
    for item in ds:
        text = item.get("text", "")
        label = item.get("label", "")
        if not text or len(text) < 200 or len(text) > 1500:
            continue
        if label in PREFERRED_NEWS_CATEGORIES:
            preferred.append(text)
        else:
            others.append(text)

    print(f"  Preferred categories: {len(preferred)} articles")
    print(f"  Others: {len(others)} articles")

    random.shuffle(preferred)
    random.shuffle(others)

    selected = preferred[:num_samples]
    if len(selected) < num_samples:
        selected.extend(others[:num_samples - len(selected)])

    random.shuffle(selected)
    print(f"  Selected {len(selected)} news articles")
    return [{"text": t} for t in selected]


def load_academic_data(num_samples: int, seed: int = 42) -> list[dict]:
    """从 CLUE-CSL 加载真实学术摘要。"""
    print("Loading CLUE-CSL from HuggingFace...")
    ds = load_dataset("clue", "csl", split="train")
    print(f"  Total: {len(ds)} abstracts")

    random.seed(seed)
    valid = []
    for item in ds:
        abstract = item.get("abst", "")
        if not abstract or len(abstract) < 100 or len(abstract) > 1000:
            continue
        valid.append({"text": abstract})

    ds_test = load_dataset("clue", "csl", split="test")
    for item in ds_test:
        abstract = item.get("abst", "")
        if not abstract or len(abstract) < 100 or len(abstract) > 1000:
            continue
        valid.append({"text": abstract})

    ds_val = load_dataset("clue", "csl", split="validation")
    for item in ds_val:
        abstract = item.get("abst", "")
        if not abstract or len(abstract) < 100 or len(abstract) > 1000:
            continue
        valid.append({"text": abstract})

    random.shuffle(valid)
    selected = valid[:num_samples]
    print(f"  Selected {len(selected)} academic abstracts (100-1000 chars)")
    return selected


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


def build_pairs(model, tokenizer, raw_data: list[dict], domain: str,
                start_id: int = 0) -> list[dict]:
    pairs = []
    for i, item in enumerate(tqdm(raw_data, desc=f"Rewriting {domain}")):
        human_text = item["text"]
        ai_text = generate_rewrite(model, tokenizer, human_text, domain)

        if len(ai_text) < 80:
            continue

        length_ratio = len(ai_text) / len(human_text) if len(human_text) > 0 else 0
        if length_ratio < 0.3 or length_ratio > 3.0:
            continue

        pairs.append({
            "id": f"{domain}_{start_id + i:06d}",
            "source": domain,
            "human_text": human_text,
            "ai_text": ai_text,
            "human_mpu_score": -1.0,
            "ai_mpu_score": -1.0,
        })
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["news", "academic"], required=True)
    parser.add_argument("--num", type=int, default=2000,
                        help="Number of pairs to generate")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=200,
                        help="Save checkpoint every N pairs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(OUTPUT_DIR,
                                   f"dataset_{args.domain}_pairs.jsonl")

    print(f"=== Building {args.domain} dataset ({args.num} pairs) ===\n")

    extra = int(args.num * 1.3)
    if args.domain == "news":
        raw_data = load_news_data(extra, seed=args.seed)
    else:
        raw_data = load_academic_data(extra, seed=args.seed)

    print(f"\nLoading Qwen from {QWEN_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    existing = 0
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            existing = sum(1 for _ in f)
        print(f"Found {existing} existing pairs, continuing from {existing}")

    remaining = args.num - existing
    if remaining <= 0:
        print(f"Already have {existing} pairs, target is {args.num}. Done.")
        return

    raw_data = raw_data[existing:]

    print(f"\nGenerating {remaining} {args.domain} pairs via Qwen rewrite...")
    t_start = time.time()

    total_written = existing
    for batch_start in range(0, remaining, args.batch_size):
        batch_end = min(batch_start + args.batch_size, remaining)
        batch_data = raw_data[batch_start:batch_end]
        pairs = build_pairs(
            model, tokenizer, batch_data, args.domain,
            start_id=existing + batch_start,
        )

        with open(args.output, "a", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        total_written += len(pairs)
        elapsed = time.time() - t_start
        rate = (batch_start + len(batch_data)) / elapsed if elapsed > 0 else 0
        print(f"  Saved {total_written}/{args.num} pairs "
              f"({elapsed:.0f}s, {rate:.1f} items/s)")

        if total_written >= args.num:
            break

    print(f"\nDone! Total: {total_written} pairs in {args.output}")


if __name__ == "__main__":
    main()
