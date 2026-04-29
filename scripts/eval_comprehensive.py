"""Comprehensive evaluation of StyleFlow step_30000.

Metrics:
  1. Detector Evasion Rate: % of samples where P(AI) < threshold after transfer
  2. Mean P(AI) Drop: average reduction in P(AI)
  3. Semantic Similarity: cosine similarity between Qwen embeddings of input/output
  4. Text Quality: character-level edit distance ratio (how much was changed)
  5. Controllability: monotonicity of P(AI) w.r.t. gamma
  6. Consistency: variance across multiple runs at same gamma

Evaluated on 50 random samples from the test set.
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import StyleFlowConfig
from src.model import StyleFlowZh
from src.qwen_encoder import QwenHiddenExtractor


CKPT = os.environ.get("EVAL_CKPT", "experiments/styleflow_v2/checkpoints/step_5000.pt")
DATA = os.environ.get("EVAL_DATA", "/root/workspace/AIGC_FUCK/dataset_multidomain.jsonl")
DETECTOR = "/root/workspace/AIGC_FUCK/models/AIGC_detector_zhv3"
QWEN_PATH = "/root/workspace/AIGC_FUCK/models/Qwen2.5-7B-Instruct"
N_SAMPLES = 50
GAMMAS = [5.0, 5.5, 6.0, 6.5, 7.0]
NUM_STEPS = 64
SEED = 42


def edit_distance_ratio(s1: str, s2: str) -> float:
    """Normalized edit distance (0=identical, 1=completely different)."""
    if not s1 and not s2:
        return 0.0
    n, m = len(s1), len(s2)
    if n > 500 or m > 500:
        s1, s2 = s1[:500], s2[:500]
        n, m = len(s1), len(s2)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if s1[i-1] == s2[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[m] / max(n, m)


def char_overlap(s1: str, s2: str) -> float:
    """Character-level overlap ratio."""
    if not s1 or not s2:
        return 0.0
    s1_set = set(enumerate(s1[:500]))
    common = sum(1 for i, c in enumerate(s2[:500]) if i < len(s1) and s1[i] == c)
    return common / max(len(s1[:500]), len(s2[:500]))


def main():
    random.seed(SEED)
    device = torch.device("cuda")

    print("Loading dataset...")
    with open(DATA, "r") as f:
        all_lines = f.readlines()
    indices = random.sample(range(len(all_lines)), min(N_SAMPLES * 3, len(all_lines)))
    samples = []
    for idx in indices:
        d = json.loads(all_lines[idx])
        ai_text = d.get("ai_text", "")
        hu_text = d.get("human_text", "")
        if len(ai_text) > 50 and len(hu_text) > 50:
            samples.append({"ai": ai_text, "human": hu_text, "id": d.get("id", str(idx))})
        if len(samples) >= N_SAMPLES:
            break
    print(f"Selected {len(samples)} samples for evaluation")

    print("Loading models...")
    cfg = StyleFlowConfig()
    model, vocab_size, cfg = StyleFlowZh.from_langflow_ckpt(CKPT, cfg, device)
    model = model.to(device).eval()

    qwen_raw = AutoModelForCausalLM.from_pretrained(
        cfg.qwen_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    qwen_encoder = QwenHiddenExtractor(qwen_raw, split_layer=cfg.qwen_split_layer)
    qwen_encoder.eval()

    bert_tok_path = str(Path(__file__).resolve().parent.parent / "tokenizer" / "bert-base-chinese")
    bert_tok = AutoTokenizer.from_pretrained(bert_tok_path)
    qwen_tok = AutoTokenizer.from_pretrained(cfg.qwen_model_path, trust_remote_code=True)

    det_tok = AutoTokenizer.from_pretrained(DETECTOR)
    det_model = AutoModelForSequenceClassification.from_pretrained(DETECTOR).to(device).eval()
    labels = det_model.config.id2label
    ai_idx = 1
    for idx, name in labels.items():
        if "ai" in name.lower():
            ai_idx = int(idx)
            break

    @torch.no_grad()
    def score(text):
        enc = det_tok(text, max_length=512, truncation=True, padding=True, return_tensors="pt").to(device)
        logits = det_model(**enc).logits
        return F.softmax(logits.float(), dim=-1)[0, ai_idx].item()

    @torch.no_grad()
    def transfer(text, gamma):
        ml = cfg.model_length
        b_enc = bert_tok(text, max_length=ml, padding="max_length", truncation=True, return_tensors="pt").to(device)
        q_enc = qwen_tok(text, max_length=ml, padding="max_length", truncation=True, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h = qwen_encoder.encode(q_enc["input_ids"], q_enc["attention_mask"])
        cond_kv = model.project_qwen_cond(h)
        x_ai = model.embed_tokens(b_enc["input_ids"])
        out_ids = model.transfer(x_ai_embed=x_ai, cond_kv=cond_kv, gamma_start=gamma, num_steps=NUM_STEPS)
        return bert_tok.decode(out_ids[0], skip_special_tokens=True).replace(" ", "")

    @torch.no_grad()
    def semantic_sim(text1, text2):
        """Cosine similarity of Qwen hidden states as semantic similarity proxy."""
        ml = cfg.model_length
        enc1 = qwen_tok(text1, max_length=ml, padding="max_length", truncation=True, return_tensors="pt").to(device)
        enc2 = qwen_tok(text2, max_length=ml, padding="max_length", truncation=True, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h1 = qwen_encoder.encode(enc1["input_ids"], enc1["attention_mask"])
            h2 = qwen_encoder.encode(enc2["input_ids"], enc2["attention_mask"])
        h1_mean = h1.float().mean(dim=1)
        h2_mean = h2.float().mean(dim=1)
        return F.cosine_similarity(h1_mean, h2_mean).item()

    print("Loading Qwen full model for PPL computation...")
    qwen_full = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    ppl_tok = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)

    @torch.no_grad()
    def compute_ppl(text):
        """Perplexity of text under Qwen language model."""
        enc = ppl_tok(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        input_ids = enc["input_ids"]
        if input_ids.shape[1] < 2:
            return float("nan")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = qwen_full(input_ids=input_ids, labels=input_ids)
        return torch.exp(outputs.loss).item()

    # ═══════════════════════════════════════════════════════════════
    #  Evaluation
    # ═══════════════════════════════════════════════════════════════

    results_by_gamma = {g: [] for g in GAMMAS}

    print(f"\nRunning evaluation: {len(samples)} samples x {len(GAMMAS)} gammas = {len(samples)*len(GAMMAS)} transfers")
    print("="*70)

    for si, sample in enumerate(tqdm(samples, desc="Samples")):
        ai_text = sample["ai"]
        hu_text = sample["human"]
        p_ai_orig = score(ai_text)
        p_ai_human = score(hu_text)

        for gamma in GAMMAS:
            output = transfer(ai_text, gamma)
            p_ai_out = score(output)
            sem_sim = semantic_sim(ai_text, output)
            edit_ratio = edit_distance_ratio(ai_text, output)
            char_ovlp = char_overlap(ai_text, output)

            ppl_out = compute_ppl(output)

            results_by_gamma[gamma].append({
                "id": sample["id"],
                "p_ai_orig": p_ai_orig,
                "p_ai_human": p_ai_human,
                "p_ai_out": p_ai_out,
                "p_ai_drop": p_ai_orig - p_ai_out,
                "semantic_sim": sem_sim,
                "edit_ratio": edit_ratio,
                "char_overlap": char_ovlp,
                "ppl": ppl_out,
                "output_len": len(output),
                "input_len": len(ai_text),
            })

    # ═══════════════════════════════════════════════════════════════
    #  Report
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION REPORT")
    print(f"Model: {Path(CKPT).name} | Samples: {len(samples)} | Steps: {NUM_STEPS}")
    print("="*70)

    # PPL baselines: original AI text and human text
    print("\nComputing PPL baselines...")
    ppl_ai_baseline = [compute_ppl(s["ai"]) for s in samples[:20]]
    ppl_hu_baseline = [compute_ppl(s["human"]) for s in samples[:20]]
    ppl_ai_mean = np.nanmean(ppl_ai_baseline)
    ppl_hu_mean = np.nanmean(ppl_hu_baseline)

    print(f"\n{'γ':>5s} | {'mean P(AI)':>10s} | {'P(AI)<0.5':>9s} | {'P(AI)<0.3':>9s} | "
          f"{'Δ P(AI)':>8s} | {'Sem.Sim':>7s} | {'EditDist':>8s} | {'CharOvlp':>8s} | {'PPL':>8s}")
    print("-"*100)

    for gamma in GAMMAS:
        rs = results_by_gamma[gamma]
        mean_pai = np.mean([r["p_ai_out"] for r in rs])
        rate_50 = np.mean([r["p_ai_out"] < 0.5 for r in rs]) * 100
        rate_30 = np.mean([r["p_ai_out"] < 0.3 for r in rs]) * 100
        mean_drop = np.mean([r["p_ai_drop"] for r in rs])
        mean_sim = np.mean([r["semantic_sim"] for r in rs])
        mean_edit = np.mean([r["edit_ratio"] for r in rs])
        mean_ovlp = np.mean([r["char_overlap"] for r in rs])
        mean_ppl = np.nanmean([r["ppl"] for r in rs])

        print(f"{gamma:>5.1f} | {mean_pai:>10.4f} | {rate_50:>8.1f}% | {rate_30:>8.1f}% | "
              f"{mean_drop:>+8.4f} | {mean_sim:>7.4f} | {mean_edit:>8.4f} | {mean_ovlp:>8.4f} | {mean_ppl:>8.1f}")

    print(f"\n[PPL Baseline] AI text: {ppl_ai_mean:.1f} | Human text: {ppl_hu_mean:.1f}")

    # Baseline: human text P(AI)
    human_scores = [score(s["human"]) for s in samples[:20]]
    print(f"\n[Baseline] Human text mean P(AI): {np.mean(human_scores):.4f} "
          f"(std={np.std(human_scores):.4f})")

    # Controllability: is P(AI) monotonically decreasing with gamma?
    print("\n[Controllability] Per-sample monotonicity check:")
    mono_count = 0
    for si in range(len(samples)):
        pai_sequence = [results_by_gamma[g][si]["p_ai_out"] for g in GAMMAS]
        is_mono = all(pai_sequence[i] >= pai_sequence[i+1] - 0.05 for i in range(len(pai_sequence)-1))
        if is_mono:
            mono_count += 1
    print(f"  Monotonic (P(AI) decreases with γ, ε=0.05): {mono_count}/{len(samples)} "
          f"({mono_count/len(samples)*100:.1f}%)")

    # Quality at best gamma (6.0)
    print(f"\n[Text Quality @ γ=6.0] Sample outputs:")
    for r in results_by_gamma[6.0][:5]:
        si = results_by_gamma[6.0].index(r)
        ai_text = samples[si]["ai"][:80]
        out_text = [x for x in results_by_gamma[6.0] if x["id"] == samples[si]["id"]]
        print(f"  [{r['id']}] P(AI): {r['p_ai_orig']:.4f}→{r['p_ai_out']:.4f}  "
              f"sim={r['semantic_sim']:.4f}  edit={r['edit_ratio']:.4f}")

    os.makedirs("eval_results", exist_ok=True)
    ckpt_name = Path(CKPT).stem
    save_path = f"eval_results/comprehensive_eval_{ckpt_name}.json"
    with open(save_path, "w") as f:
        json.dump({
            "config": {"ckpt": CKPT, "n_samples": len(samples), "gammas": GAMMAS, "num_steps": NUM_STEPS},
            "results_by_gamma": {str(g): rs for g, rs in results_by_gamma.items()},
            "human_baseline_p_ai": float(np.mean(human_scores)),
            "ppl_baseline_ai": float(ppl_ai_mean),
            "ppl_baseline_human": float(ppl_hu_mean),
        }, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to {save_path}")


if __name__ == "__main__":
    main()
