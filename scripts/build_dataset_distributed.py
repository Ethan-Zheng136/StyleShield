"""分布式 batch 数据集构建（torchrun 启动）。

每张 GPU 加载一个 Qwen，batch 推理改写。A800 80G 可跑 batch_size=16。

用法：
  torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 \
      --master_addr localhost --master_port 29501 \
      scripts/build_dataset_distributed.py \
      --domain news --num 100000 --batch_size 16
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


def batch_rewrite(model, tokenizer, texts: list[str], domain: str,
                  max_new_tokens: int = 800) -> list[str]:
    """Batch 推理：一次改写多条文本。"""
    if domain == "news":
        template = REWRITE_NEWS_PROMPT
        trim = 1200
    else:
        template = REWRITE_ACADEMIC_PROMPT
        trim = 800

    prompts = []
    for text in texts:
        messages = [{"role": "user", "content": template.format(text=text[:trim])}]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))

    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.3, top_p=0.85, do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    results = []
    input_len = inputs["input_ids"].shape[1]
    for i in range(len(texts)):
        # With left padding, each prompt's real tokens start after its padding.
        # But generate() output aligns to the padded input, so we can just skip input_len.
        new_tokens = outputs[i][input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append(text)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["news", "academic"], required=True)
    parser.add_argument("--num", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.distributed.init_process_group(backend="gloo")
    else:
        local_rank = 0
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    per_worker = args.num // world_size
    start_idx = local_rank * per_worker
    if local_rank == world_size - 1:
        per_worker = args.num - start_idx

    output_path = os.path.join(
        OUTPUT_DIR, f"dataset_{args.domain}_pairs_w{local_rank}.jsonl")

    existing = 0
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = sum(1 for _ in f)
    remaining = per_worker - existing
    if remaining <= 0:
        print(f"[W{local_rank}] Already done ({existing}/{per_worker})")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return

    print(f"[W{local_rank}/{world_size}] GPU={local_rank}, "
          f"range=[{start_idx}..{start_idx+per_worker}), "
          f"existing={existing}, todo={remaining}, batch_size={args.batch_size}")

    all_texts = load_raw_data(args.domain)
    print(f"[W{local_rank}] Raw texts: {len(all_texts)}")

    my_texts = all_texts[start_idx + existing: start_idx + per_worker]
    if not my_texts:
        print(f"[W{local_rank}] No data, done.")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return

    print(f"[W{local_rank}] Loading Qwen on GPU {local_rank}...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()

    print(f"[W{local_rank}] Generating {len(my_texts)} pairs (batch={args.batch_size})...")
    t_start = time.time()
    written = 0
    skipped = 0

    total_batches = (len(my_texts) + args.batch_size - 1) // args.batch_size
    pbar = tqdm(range(0, len(my_texts), args.batch_size),
                desc=f"W{local_rank}", total=total_batches,
                disable=(local_rank != 0),
                miniters=1, mininterval=10)

    with open(output_path, "a", encoding="utf-8") as fout:
        for batch_start in pbar:
            batch_end = min(batch_start + args.batch_size, len(my_texts))
            batch_human = my_texts[batch_start:batch_end]

            batch_ai = batch_rewrite(model, tokenizer, batch_human, args.domain)

            for j, (human_text, ai_text) in enumerate(zip(batch_human, batch_ai)):
                if len(ai_text) < 80:
                    skipped += 1
                    continue
                length_ratio = len(ai_text) / len(human_text) if len(human_text) > 0 else 0
                if length_ratio < 0.3 or length_ratio > 3.0:
                    skipped += 1
                    continue

                idx = start_idx + existing + batch_start + j
                record = {
                    "id": f"{args.domain}_{idx:06d}",
                    "source": args.domain,
                    "human_text": human_text,
                    "ai_text": ai_text,
                    "human_mpu_score": -1.0,
                    "ai_mpu_score": -1.0,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

            fout.flush()

            if local_rank == 0:
                pbar.set_postfix(written=written, skip=skipped)

            items_done = batch_end
            if items_done % 500 < args.batch_size:
                elapsed = time.time() - t_start
                rate = items_done / elapsed
                eta_h = (len(my_texts) - items_done) / rate / 3600 if rate > 0 else 0
                print(f"[W{local_rank}] {items_done}/{len(my_texts)} "
                      f"written={written} skip={skipped} "
                      f"({rate:.1f} it/s, ETA {eta_h:.1f}h)")
                sys.stdout.flush()

    elapsed = time.time() - t_start
    rate = written / elapsed if elapsed > 0 else 0
    print(f"[W{local_rank}] DONE: {written} pairs, {skipped} skipped, "
          f"{elapsed:.0f}s ({rate:.2f} pairs/s) → {output_path}")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
