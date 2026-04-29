"""Quick sweep for ablation_split21 checkpoints.
Find best checkpoint using 50 samples, gamma=6.5, zhv3 detector.

Usage (single GPU):
    python scripts/sweep_split21.py

Results saved to eval_results/sweep_split21.json
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import StyleFlowConfig
from src.model import StyleFlowZh
from src.qwen_encoder import QwenHiddenExtractor

TEST_SET = "/root/workspace/AIGC_FUCK/test_set_1000.jsonl"
QWEN_PATH = "/root/workspace/AIGC_FUCK/models/Qwen2.5-7B-Instruct"
DETECTOR_PATH = "/root/workspace/AIGC_FUCK/models/AIGC_detector_zhv3"
CKPT_DIR = "experiments/ablation_split21/checkpoints"
SPLIT_LAYER = 21
GAMMA = 6.5
N_SAMPLES = 50
NUM_STEPS = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_test_data(n):
    import random
    random.seed(42)
    samples = []
    with open(TEST_SET) as f:
        for line in f:
            samples.append(json.loads(line))
    return random.sample(samples, min(n, len(samples)))


def main():
    os.makedirs("eval_results", exist_ok=True)
    samples = load_test_data(N_SAMPLES)
    print(f"Loaded {len(samples)} test samples")

    bert_tok_path = str(Path(__file__).resolve().parent.parent / "tokenizer" / "bert-base-chinese")
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_tok_path)
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)

    print(f"Loading Qwen encoder (split_layer={SPLIT_LAYER})...")
    qwen_raw = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(DEVICE)
    qwen_encoder = QwenHiddenExtractor(qwen_raw, split_layer=SPLIT_LAYER)
    qwen_encoder.eval()

    print("Loading detector...")
    det_tokenizer = AutoTokenizer.from_pretrained(DETECTOR_PATH)
    det_model = AutoModelForSequenceClassification.from_pretrained(DETECTOR_PATH).to(DEVICE).eval()
    for p in det_model.parameters():
        p.requires_grad = False

    labels = det_model.config.id2label
    ai_idx = 1
    for idx, name in labels.items():
        if "ai" in name.lower() or "machine" in name.lower():
            ai_idx = int(idx)
            break

    ckpt_files = sorted(Path(CKPT_DIR).glob("step_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]))
    print(f"Found {len(ckpt_files)} checkpoints in {CKPT_DIR}\n")

    model = None
    results = []

    for ckpt in ckpt_files:
        step = int(ckpt.stem.split("_")[1])
        print(f"Evaluating {ckpt.name}...", end=" ", flush=True)

        if model is not None:
            del model
            torch.cuda.empty_cache()

        cfg = StyleFlowConfig()
        model, vocab_size, cfg = StyleFlowZh.from_langflow_ckpt(str(ckpt), cfg, DEVICE)
        model = model.to(DEVICE).eval()

        p_ais = []
        sims = []

        with torch.no_grad():
            for s in samples:
                text = s["ai_text"]
                max_length = cfg.model_length

                bert_enc = bert_tokenizer(text, max_length=max_length, padding="max_length",
                                          truncation=True, return_tensors="pt").to(DEVICE)
                qwen_enc = qwen_tokenizer(text, max_length=max_length, padding="max_length",
                                          truncation=True, return_tensors="pt").to(DEVICE)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    h = qwen_encoder.encode(qwen_enc["input_ids"], qwen_enc["attention_mask"])

                cond_kv = model.project_qwen_cond(h)
                x_ai = model.embed_tokens(bert_enc["input_ids"])
                output_ids = model.transfer(x_ai_embed=x_ai, cond_kv=cond_kv,
                                            gamma_start=GAMMA, num_steps=NUM_STEPS)
                output_text = bert_tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(" ", "")

                det_enc = det_tokenizer(output_text, max_length=512, padding=True,
                                        truncation=True, return_tensors="pt").to(DEVICE)
                logits = det_model(**det_enc).logits
                probs = F.softmax(logits.float(), dim=-1)
                p_ai = probs[0, ai_idx].item()
                p_ais.append(p_ai)

                enc1 = qwen_tokenizer(text, max_length=max_length, padding="max_length",
                                      truncation=True, return_tensors="pt").to(DEVICE)
                enc2 = qwen_tokenizer(output_text, max_length=max_length, padding="max_length",
                                      truncation=True, return_tensors="pt").to(DEVICE)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    h1 = qwen_encoder.encode(enc1["input_ids"], enc1["attention_mask"])
                    h2 = qwen_encoder.encode(enc2["input_ids"], enc2["attention_mask"])
                v1 = h1.float().mean(dim=1)
                v2 = h2.float().mean(dim=1)
                sim = F.cosine_similarity(v1, v2, dim=-1).item()
                sims.append(sim)

        mean_pai = np.mean(p_ais)
        mean_sim = np.mean(sims)
        score = (1.0 - mean_pai) * 0.6 + mean_sim * 0.4

        results.append({
            "step": step,
            "ckpt": str(ckpt),
            "mean_pai": round(mean_pai, 4),
            "mean_sim": round(mean_sim, 4),
            "score": round(score, 4),
        })
        print(f"P(AI)={mean_pai:.4f}  sim={mean_sim:.4f}  score={score:.4f}")

    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]

    print(f"\n{'='*60}")
    print(f"  BEST: step_{best['step']} (score={best['score']:.4f})")
    print(f"  P(AI)={best['mean_pai']:.4f}  Sim={best['mean_sim']:.4f}")
    print(f"{'='*60}")

    save_path = "eval_results/sweep_split21.json"
    with open(save_path, "w") as f:
        json.dump({"best": best, "all": results}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
