"""Evaluate all ablation checkpoints: quick sweep to find best, then full ACL eval.

Phase 1: Quick screening (50 samples, gamma=6.5, zhv3 only) on all checkpoints
Phase 2: Full ACL eval on best checkpoint per ablation

Usage:
    python scripts/eval_ablations.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
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

ABLATIONS = {
    "no_detector": {
        "ckpt_dir": "experiments/ablation_no_detector/checkpoints",
        "label": "w/o Detector Reward",
    },
    "split7": {
        "ckpt_dir": "experiments/ablation_split7/checkpoints",
        "label": "Split Layer 7",
    },
    "split21": {
        "ckpt_dir": "experiments/ablation_split21/checkpoints",
        "label": "Split Layer 21",
    },
    "zhihu_only": {
        "ckpt_dir": "experiments/styleflow/checkpoints",
        "label": "w/o Multi-domain (v1)",
        "single_ckpt": "step_30000.pt",
    },
}

GAMMA_QUICK = 6.5
GAMMA_FULL = [5.0, 5.5, 6.0, 6.5, 7.0]
N_QUICK = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_test_data(n=None):
    samples = []
    with open(TEST_SET) as f:
        for line in f:
            samples.append(json.loads(line))
    if n:
        import random
        random.seed(42)
        samples = random.sample(samples, min(n, len(samples)))
    return samples


class QuickPipeline:
    def __init__(self):
        print("[QuickPipeline] Loading shared models...")

        self.bert_tok_path = str(Path(__file__).resolve().parent.parent / "tokenizer" / "bert-base-chinese")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_tok_path)
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)

        print("[QuickPipeline] Loading Qwen encoder...")
        qwen_raw = AutoModelForCausalLM.from_pretrained(
            QWEN_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(DEVICE)
        self.qwen_encoder = QwenHiddenExtractor(qwen_raw, split_layer=14)
        self.qwen_encoder.eval()

        print("[QuickPipeline] Loading detector...")
        self.det_tokenizer = AutoTokenizer.from_pretrained(DETECTOR_PATH)
        self.det_model = AutoModelForSequenceClassification.from_pretrained(DETECTOR_PATH)
        self.det_model.to(DEVICE).eval()
        for p in self.det_model.parameters():
            p.requires_grad = False

        labels = self.det_model.config.id2label
        self.ai_idx = 1
        for idx, name in labels.items():
            if "ai" in name.lower() or "machine" in name.lower():
                self.ai_idx = int(idx)
                break

        self.model = None
        self.cfg = None

    def load_checkpoint(self, ckpt_path):
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()

        cfg = StyleFlowConfig()
        model, vocab_size, cfg = StyleFlowZh.from_langflow_ckpt(ckpt_path, cfg, DEVICE)
        self.model = model.to(DEVICE).eval()
        self.cfg = cfg

        if cfg.qwen_split_layer != 14:
            del self.qwen_encoder
            torch.cuda.empty_cache()
            qwen_raw = AutoModelForCausalLM.from_pretrained(
                QWEN_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True
            ).to(DEVICE)
            self.qwen_encoder = QwenHiddenExtractor(qwen_raw, split_layer=cfg.qwen_split_layer)
            self.qwen_encoder.eval()

    @torch.no_grad()
    def score_text(self, text):
        enc = self.det_tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        logits = self.det_model(**enc).logits
        probs = F.softmax(logits.float(), dim=-1)
        return probs[0, self.ai_idx].item()

    @torch.no_grad()
    def transfer(self, text, gamma=6.5, num_steps=64):
        max_length = self.cfg.model_length
        bert_enc = self.bert_tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)
        qwen_enc = self.qwen_tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h = self.qwen_encoder.encode(qwen_enc["input_ids"], qwen_enc["attention_mask"])

        cond_kv = self.model.project_qwen_cond(h)
        x_ai = self.model.embed_tokens(bert_enc["input_ids"])
        output_ids = self.model.transfer(x_ai_embed=x_ai, cond_kv=cond_kv, gamma_start=gamma, num_steps=num_steps)
        output_text = self.bert_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text.replace(" ", "")

    @torch.no_grad()
    def semantic_sim(self, text1, text2):
        max_length = self.cfg.model_length
        enc1 = self.qwen_tokenizer(text1, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)
        enc2 = self.qwen_tokenizer(text2, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h1 = self.qwen_encoder.encode(enc1["input_ids"], enc1["attention_mask"])
            h2 = self.qwen_encoder.encode(enc2["input_ids"], enc2["attention_mask"])

        v1 = h1.float().mean(dim=1)
        v2 = h2.float().mean(dim=1)
        return F.cosine_similarity(v1, v2, dim=-1).item()


def phase1_quick_sweep(pipeline, ckpt_dir, samples):
    ckpt_files = sorted(Path(ckpt_dir).glob("step_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]))
    if not ckpt_files:
        print(f"  No checkpoints found in {ckpt_dir}")
        return None, []

    results = []
    for ckpt in ckpt_files:
        step = int(ckpt.stem.split("_")[1])
        print(f"  Evaluating {ckpt.name}...", end=" ", flush=True)

        pipeline.load_checkpoint(str(ckpt))

        p_ais = []
        sims = []
        for s in samples:
            out = pipeline.transfer(s["ai_text"], gamma=GAMMA_QUICK, num_steps=64)
            p_ai = pipeline.score_text(out)
            sim = pipeline.semantic_sim(s["ai_text"], out)
            p_ais.append(p_ai)
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
    print(f"\n  Best: step_{best['step']} (score={best['score']:.4f})")
    return best, results


def main():
    os.makedirs("eval_results", exist_ok=True)

    samples_quick = load_test_data(N_QUICK)
    print(f"Loaded {len(samples_quick)} samples for quick sweep\n")

    pipeline = QuickPipeline()

    all_results = {}

    for name, info in ABLATIONS.items():
        ckpt_dir = info["ckpt_dir"]
        label = info["label"]

        if not Path(ckpt_dir).exists():
            print(f"\n{'='*60}")
            print(f"  SKIP: {label} — {ckpt_dir} not found")
            print(f"{'='*60}")
            continue

        print(f"\n{'='*60}")
        print(f"  ABLATION: {label}")
        print(f"  Checkpoint dir: {ckpt_dir}")
        print(f"{'='*60}")

        if "single_ckpt" in info:
            ckpt_path = str(Path(ckpt_dir) / info["single_ckpt"])
            print(f"  Single checkpoint: {ckpt_path}")
            pipeline.load_checkpoint(ckpt_path)

            p_ais = []
            sims = []
            for s in samples_quick:
                out = pipeline.transfer(s["ai_text"], gamma=GAMMA_QUICK, num_steps=64)
                p_ai = pipeline.score_text(out)
                sim = pipeline.semantic_sim(s["ai_text"], out)
                p_ais.append(p_ai)
                sims.append(sim)

            mean_pai = np.mean(p_ais)
            mean_sim = np.mean(sims)
            score = (1.0 - mean_pai) * 0.6 + mean_sim * 0.4
            best = {
                "step": 30000,
                "ckpt": ckpt_path,
                "mean_pai": round(mean_pai, 4),
                "mean_sim": round(mean_sim, 4),
                "score": round(score, 4),
            }
            sweep_results = [best]
            print(f"  P(AI)={mean_pai:.4f}  sim={mean_sim:.4f}  score={score:.4f}")
        else:
            best, sweep_results = phase1_quick_sweep(pipeline, ckpt_dir, samples_quick)

        if best is None:
            continue

        all_results[name] = {
            "label": label,
            "best_step": best["step"],
            "best_ckpt": best["ckpt"],
            "sweep": sweep_results,
        }

    summary_path = "eval_results/ablation_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSweep summary saved to {summary_path}")

    print(f"\n\n{'='*60}")
    print("  PHASE 2: Full ACL eval on best checkpoints")
    print(f"{'='*60}")

    for name, info in all_results.items():
        best_ckpt = info["best_ckpt"]
        label = info["label"]
        print(f"\n  Running full eval: {label} ({Path(best_ckpt).name})")
        tag = f"ablation_{name}"
        cmd = (f"python scripts/eval_acl.py --ckpt {best_ckpt} "
               f"--tag {tag} --gammas 5.0 5.5 6.0 6.5 7.0")
        print(f"  CMD: {cmd}")
        os.system(cmd)

    print(f"\n\n{'='*60}")
    print("  ALL DONE")
    print(f"{'='*60}")

    print("\nFinal summary of best checkpoints:")
    for name, info in all_results.items():
        print(f"  {info['label']:30s} → step_{info['best_step']} "
              f"(P(AI)={info['sweep'][0]['mean_pai']:.4f}, "
              f"sim={info['sweep'][0]['mean_sim']:.4f})")


if __name__ == "__main__":
    main()
