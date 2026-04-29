"""SDEdit-style transfer with Qwen conditioning and detector evaluation.

Uses the same denoising procedure as the original LangFlow SDEdit
(AIGC_FUCK_6/scripts/generate.py), with Qwen cross-attention conditioning.

Usage:
    python scripts/transfer.py \
        --ckpt experiments/styleflow/checkpoints/final.pt \
        --input "AI生成的文本内容..." \
        --gamma 5.0 \
        --num_steps 64

    python scripts/transfer.py \
        --ckpt experiments/styleflow/checkpoints/final.pt \
        --input_file input.txt \
        --sweep   # sweep gamma from 3.0 to 8.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import StyleFlowConfig
from src.model import StyleFlowZh
from src.qwen_encoder import QwenHiddenExtractor


# ═══════════════════════════════════════════════════════════════════
#  Detector
# ═══════════════════════════════════════════════════════════════════

class AIGCDetector:
    def __init__(self, model_path: str, device: torch.device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        labels = self.model.config.id2label
        self.ai_idx = None
        for idx, name in labels.items():
            if "ai" in name.lower() or "machine" in name.lower():
                self.ai_idx = int(idx)
                break
        if self.ai_idx is None:
            self.ai_idx = 1

    @torch.no_grad()
    def score_text(self, text: str, max_length: int = 512) -> float:
        enc = self.tokenizer(
            text, max_length=max_length, padding=True,
            truncation=True, return_tensors="pt",
        ).to(self.model.device)
        logits = self.model(**enc).logits
        probs = F.softmax(logits.float(), dim=-1)
        return probs[0, self.ai_idx].item()


# ═══════════════════════════════════════════════════════════════════
#  Transfer Pipeline
# ═══════════════════════════════════════════════════════════════════

def load_pipeline(ckpt_path: str, device: torch.device):
    """Load StyleFlow model + Qwen encoder from checkpoint."""
    cfg = StyleFlowConfig()
    model, vocab_size, cfg = StyleFlowZh.from_langflow_ckpt(ckpt_path, cfg, device)
    model = model.to(device).eval()

    print(f"Loading Qwen encoder from {cfg.qwen_model_path}")
    qwen_raw = AutoModelForCausalLM.from_pretrained(
        cfg.qwen_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    qwen_encoder = QwenHiddenExtractor(qwen_raw, split_layer=cfg.qwen_split_layer)
    qwen_encoder.eval()

    bert_tok_path = str(Path(__file__).resolve().parent.parent / "tokenizer" / "bert-base-chinese")
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_tok_path)
    qwen_tok_path = cfg.qwen_tokenizer_name or cfg.qwen_model_path
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_tok_path, trust_remote_code=True)

    return model, qwen_encoder, bert_tokenizer, qwen_tokenizer, cfg


@torch.no_grad()
def transfer_text(
    text: str,
    model: StyleFlowZh,
    qwen_encoder: QwenHiddenExtractor,
    bert_tokenizer,
    qwen_tokenizer,
    gamma_start: float = 5.0,
    num_steps: int = 64,
    max_length: int = 512,
    device: torch.device = None,
) -> str:
    """Transfer a single AI text toward human style."""
    if device is None:
        device = next(model.parameters()).device

    bert_enc = bert_tokenizer(
        text, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    ).to(device)
    ai_ids_bert = bert_enc["input_ids"]

    qwen_enc = qwen_tokenizer(
        text, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    ).to(device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        h_ai = qwen_encoder.encode(qwen_enc["input_ids"], qwen_enc["attention_mask"])

    cond_kv = model.project_qwen_cond(h_ai)
    x_ai = model.embed_tokens(ai_ids_bert)

    output_ids = model.transfer(
        x_ai_embed=x_ai,
        cond_kv=cond_kv,
        gamma_start=gamma_start,
        num_steps=num_steps,
    )

    output_text = bert_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    output_text = output_text.replace(" ", "")
    return output_text


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="StyleFlow Transfer")
    parser.add_argument("--ckpt", type=str, required=True, help="StyleFlow checkpoint")
    parser.add_argument("--input", type=str, default=None, help="Input AI text")
    parser.add_argument("--input_file", type=str, default=None, help="Input file (one text)")
    parser.add_argument("--gamma", type=float, default=5.0,
                        help="Noise level for SDEdit (higher=more change)")
    parser.add_argument("--num_steps", type=int, default=64, help="Denoising steps")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep gamma from 3.0 to 8.0")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    if args.input is None and args.input_file is None:
        parser.error("Provide --input or --input_file")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading pipeline...")
    model, qwen_encoder, bert_tok, qwen_tok, cfg = load_pipeline(args.ckpt, device)

    print("Loading detector...")
    detector = AIGCDetector(cfg.detector_path, device)

    text = args.input
    if text is None:
        with open(args.input_file, "r", encoding="utf-8") as f:
            text = f.read().strip()

    original_p_ai = detector.score_text(text)
    print(f"\n{'='*60}")
    print(f"Input text ({len(text)} chars):")
    print(f"  {text[:200]}...")
    print(f"  P(AI) = {original_p_ai:.4f}")
    print(f"{'='*60}\n")

    if args.sweep:
        gamma_values = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0]
    else:
        gamma_values = [args.gamma]

    results = []
    for g_val in gamma_values:
        print(f"\n--- gamma = {g_val:.1f} ---")
        output = transfer_text(
            text, model, qwen_encoder, bert_tok, qwen_tok,
            gamma_start=g_val, num_steps=args.num_steps,
            max_length=cfg.model_length, device=device,
        )
        p_ai = detector.score_text(output)
        drop = original_p_ai - p_ai

        print(f"  Output ({len(output)} chars): {output[:200]}...")
        print(f"  P(AI) = {p_ai:.4f} (drop = {drop:+.4f})")

        results.append({
            "gamma": g_val,
            "output": output,
            "p_ai": p_ai,
            "p_ai_drop": drop,
        })

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "input": text,
                "original_p_ai": original_p_ai,
                "results": results,
            }, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Original P(AI) = {original_p_ai:.4f}")
    for r in results:
        print(f"  γ={r['gamma']:.1f} → P(AI)={r['p_ai']:.4f} (Δ={r['p_ai_drop']:+.4f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
