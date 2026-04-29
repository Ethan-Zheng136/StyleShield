"""Checkpoint sweep: find the best checkpoint across all saved steps.

Phase 1 (quick): 50 samples, gamma=5.0, zhv3 only → rank all checkpoints
Phase 2 (fine):  200 samples, multi-gamma, multi-detector, PPL, semantic → top-K detail

Usage:
    python scripts/sweep_checkpoints.py
    python scripts/sweep_checkpoints.py --phase1_n 100 --top_k 5
"""

from __future__ import annotations

import argparse
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

CKPT_DIR = "experiments/styleflow_v2/checkpoints"
TEST_SET = "/root/workspace/AIGC_FUCK/test_set_1000.jsonl"
QWEN_PATH = "/root/workspace/AIGC_FUCK/models/Qwen2.5-7B-Instruct"
GPT2_PATH = "/root/workspace/AIGC_FUCK/models/gpt2-chinese-cluecorpussmall"
DETECTOR_PATH_ZHV3 = "/root/workspace/AIGC_FUCK/models/AIGC_detector_zhv3"
DETECTOR_CONFIGS = {
    "zhv3": {"path": "/root/workspace/AIGC_FUCK/models/AIGC_detector_zhv3", "ai_idx": 1},
    "zhv2": {"path": "/root/workspace/AIGC_FUCK/models/AIGC_detector_zhv2", "ai_idx": 1},
    "anx-bert": {"path": "/root/workspace/AIGC_FUCK/models/chinese-ai-detector-bert", "ai_idx": 1},
    "gpt2-det": {"path": "/root/workspace/AIGC_FUCK/models/AITextDetector", "ai_idx": 0},
}

SEED = 42


def load_test_set(path: str, n: int | None = None) -> list[dict]:
    import random
    samples = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())
            ai = obj.get("ai_text", "").strip()
            hu = obj.get("human_text", "").strip()
            if ai and hu and len(ai) > 30:
                sid = obj.get("id", "")
                domain = sid.split("_")[0] if "_" in sid else "unknown"
                samples.append({"id": sid, "ai": ai, "human": hu, "domain": domain})
    if n and n < len(samples):
        random.seed(SEED)
        samples = random.sample(samples, n)
    return samples


class SingleDetector:
    def __init__(self, name: str, device: torch.device):
        cfg = DETECTOR_CONFIGS[name]
        self.tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg["path"], trust_remote_code=True
        ).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.ai_idx = cfg["ai_idx"]

    @torch.no_grad()
    def score(self, text: str) -> float:
        enc = self.tok(text, max_length=512, truncation=True, padding=True, return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        probs = F.softmax(logits.float(), dim=-1)
        return probs[0, self.ai_idx].item()


class MultiDetector:
    def __init__(self, names: list[str], device: torch.device):
        self.detectors = {}
        for name in names:
            self.detectors[name] = SingleDetector(name, device)

    @torch.no_grad()
    def score(self, text: str) -> dict[str, float]:
        return {name: d.score(text) for name, d in self.detectors.items()}


class SharedPipeline:
    """Shared Qwen encoder + BERT tokenizer, reused across checkpoint loads."""

    def __init__(self, device: torch.device):
        self.device = device
        self.cfg = StyleFlowConfig()

        print("[Shared] Loading Qwen encoder...")
        qwen_raw = AutoModelForCausalLM.from_pretrained(
            QWEN_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
        self.qwen_encoder = QwenHiddenExtractor(qwen_raw, split_layer=self.cfg.qwen_split_layer)
        self.qwen_encoder.eval()

        bert_tok_path = str(Path(__file__).resolve().parent.parent / "tokenizer" / "bert-base-chinese")
        self.bert_tok = AutoTokenizer.from_pretrained(bert_tok_path)
        self.qwen_tok = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)

    def load_model(self, ckpt_path: str) -> StyleFlowZh:
        cfg = StyleFlowConfig()
        model, _, cfg = StyleFlowZh.from_langflow_ckpt(ckpt_path, cfg, self.device)
        model = model.to(self.device).eval()
        return model

    @torch.no_grad()
    def transfer(self, model: StyleFlowZh, text: str, gamma: float, num_steps: int = 64) -> str:
        ml = self.cfg.model_length
        b_enc = self.bert_tok(
            text, max_length=ml, padding="max_length", truncation=True, return_tensors="pt"
        ).to(self.device)
        q_enc = self.qwen_tok(
            text, max_length=ml, padding="max_length", truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h = self.qwen_encoder.encode(q_enc["input_ids"], q_enc["attention_mask"])
        cond_kv = model.project_qwen_cond(h)
        x_ai = model.embed_tokens(b_enc["input_ids"])
        out_ids = model.transfer(
            x_ai_embed=x_ai, cond_kv=cond_kv,
            gamma_start=gamma, num_steps=num_steps,
        )
        return self.bert_tok.decode(out_ids[0], skip_special_tokens=True).replace(" ", "")


class PPLScorer:
    def __init__(self, device: torch.device):
        print("[PPL] Loading GPT-2 Chinese...")
        self.tok = AutoTokenizer.from_pretrained(GPT2_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            GPT2_PATH, trust_remote_code=True
        ).to(device).eval()

    @torch.no_grad()
    def score(self, text: str) -> float:
        enc = self.tok(text, return_tensors="pt", max_length=512, truncation=True).to(self.model.device)
        ids = enc["input_ids"]
        if ids.shape[1] < 2:
            return float("nan")
        out = self.model(input_ids=ids, labels=ids)
        return torch.exp(out.loss).item()


def edit_distance_ratio(s1: str, s2: str) -> float:
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
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m] / max(n, m)


def semantic_sim(pipeline: SharedPipeline, text1: str, text2: str) -> float:
    ml = pipeline.cfg.model_length
    dev = pipeline.device
    enc1 = pipeline.qwen_tok(text1, max_length=ml, padding="max_length",
                              truncation=True, return_tensors="pt").to(dev)
    enc2 = pipeline.qwen_tok(text2, max_length=ml, padding="max_length",
                              truncation=True, return_tensors="pt").to(dev)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        h1 = pipeline.qwen_encoder.encode(enc1["input_ids"], enc1["attention_mask"])
        h2 = pipeline.qwen_encoder.encode(enc2["input_ids"], enc2["attention_mask"])
    h1_mean = h1.float().mean(dim=1)
    h2_mean = h2.float().mean(dim=1)
    return F.cosine_similarity(h1_mean, h2_mean).item()


# ═══════════════════════════════════════════════════════════════════
#  Phase 1: Quick screening
# ═══════════════════════════════════════════════════════════════════

def phase1(pipeline: SharedPipeline, detector: SingleDetector,
           samples: list[dict], ckpt_paths: list[str],
           gamma: float = 5.0, num_steps: int = 64) -> list[dict]:
    """Quick screen all checkpoints: mean P(AI) on small sample set."""
    print("\n" + "=" * 80)
    print(f"  PHASE 1: Quick Screening ({len(samples)} samples, gamma={gamma}, zhv3)")
    print(f"  Checkpoints: {len(ckpt_paths)}")
    print("=" * 80)

    rankings = []
    for ci, ckpt_path in enumerate(ckpt_paths):
        ckpt_name = Path(ckpt_path).stem
        print(f"\n[{ci+1}/{len(ckpt_paths)}] Loading {ckpt_name}...")
        model = pipeline.load_model(ckpt_path)

        pais = []
        for s in tqdm(samples, desc=f"  {ckpt_name}", leave=False):
            output = pipeline.transfer(model, s["ai"], gamma=gamma, num_steps=num_steps)
            p_ai = detector.score(output)
            pais.append(p_ai)

        mean_pai = float(np.mean(pais))
        rate_50 = float(np.mean([p < 0.5 for p in pais]) * 100)
        rate_30 = float(np.mean([p < 0.3 for p in pais]) * 100)

        rankings.append({
            "ckpt": ckpt_name,
            "path": ckpt_path,
            "mean_pai": mean_pai,
            "rate_below_50": rate_50,
            "rate_below_30": rate_30,
            "std_pai": float(np.std(pais)),
        })

        print(f"  → P(AI)={mean_pai:.4f}±{np.std(pais):.4f}  "
              f"<0.5: {rate_50:.1f}%  <0.3: {rate_30:.1f}%")

        del model
        torch.cuda.empty_cache()

    rankings.sort(key=lambda x: x["mean_pai"])

    print("\n" + "=" * 80)
    print("  PHASE 1 RANKING (by mean P(AI), lower is better):")
    print(f"  {'Rank':>4s} | {'Checkpoint':>20s} | {'mean P(AI)':>12s} | {'<0.5':>6s} | {'<0.3':>6s}")
    print(f"  {'-'*60}")
    for i, r in enumerate(rankings):
        marker = " ★" if i < 5 else ""
        print(f"  {i+1:>4d} | {r['ckpt']:>20s} | "
              f"{r['mean_pai']:.4f}±{r['std_pai']:.4f} | "
              f"{r['rate_below_50']:>5.1f}% | {r['rate_below_30']:>5.1f}%{marker}")
    print("=" * 80)

    return rankings


# ═══════════════════════════════════════════════════════════════════
#  Phase 2: Detailed evaluation of top-K
# ═══════════════════════════════════════════════════════════════════

def phase2(pipeline: SharedPipeline, multi_det: MultiDetector,
           ppl_scorer: PPLScorer, samples: list[dict],
           top_ckpts: list[dict], gammas: list[float],
           num_steps: int = 64) -> list[dict]:
    """Detailed evaluation of top-K checkpoints."""
    det_names = list(multi_det.detectors.keys())

    print("\n" + "=" * 80)
    print(f"  PHASE 2: Detailed Evaluation")
    print(f"  Checkpoints: {[c['ckpt'] for c in top_ckpts]}")
    print(f"  Samples: {len(samples)} | Gammas: {gammas} | Detectors: {det_names}")
    print("=" * 80)

    all_results = []

    for ci, ckpt_info in enumerate(top_ckpts):
        ckpt_name = ckpt_info["ckpt"]
        ckpt_path = ckpt_info["path"]
        print(f"\n[{ci+1}/{len(top_ckpts)}] Loading {ckpt_name}...")
        model = pipeline.load_model(ckpt_path)

        ckpt_results = {"ckpt": ckpt_name, "path": ckpt_path, "gamma_results": {}}

        for gamma in gammas:
            print(f"  gamma={gamma:.1f}:")
            gamma_data = []

            for s in tqdm(samples, desc=f"    γ={gamma:.1f}", leave=False):
                output = pipeline.transfer(model, s["ai"], gamma=gamma, num_steps=num_steps)
                det_scores = multi_det.score(output)
                sim = semantic_sim(pipeline, s["ai"], output)
                ppl = ppl_scorer.score(output)
                edit = edit_distance_ratio(s["ai"], output)

                gamma_data.append({
                    "id": s["id"],
                    "domain": s["domain"],
                    "det_scores": det_scores,
                    "semantic_sim": sim,
                    "ppl": ppl,
                    "edit_ratio": edit,
                })

            pais_zhv3 = [d["det_scores"]["zhv3"] for d in gamma_data]
            sims = [d["semantic_sim"] for d in gamma_data]
            ppls = [d["ppl"] for d in gamma_data if not np.isnan(d["ppl"])]

            summary = {
                "mean_pai_zhv3": float(np.mean(pais_zhv3)),
                "rate_below_50_zhv3": float(np.mean([p < 0.5 for p in pais_zhv3]) * 100),
                "rate_below_30_zhv3": float(np.mean([p < 0.3 for p in pais_zhv3]) * 100),
                "mean_sem_sim": float(np.mean(sims)),
                "median_ppl": float(np.nanmedian(ppls)) if ppls else float("nan"),
                "mean_edit_ratio": float(np.mean([d["edit_ratio"] for d in gamma_data])),
            }

            for det_name in det_names:
                pais = [d["det_scores"][det_name] for d in gamma_data]
                summary[f"mean_pai_{det_name}"] = float(np.mean(pais))
                summary[f"rate50_{det_name}"] = float(np.mean([p < 0.5 for p in pais]) * 100)

            print(f"    P(AI)_zhv3={summary['mean_pai_zhv3']:.4f}  "
                  f"<0.5:{summary['rate_below_50_zhv3']:.1f}%  "
                  f"Sim={summary['mean_sem_sim']:.4f}  "
                  f"PPL={summary['median_ppl']:.1f}  "
                  f"Edit={summary['mean_edit_ratio']:.4f}")

            ckpt_results["gamma_results"][str(gamma)] = {
                "summary": summary,
                "raw": gamma_data,
            }

        all_results.append(ckpt_results)
        del model
        torch.cuda.empty_cache()

    return all_results


def compute_composite_score(summary: dict) -> float:
    """Composite score: lower P(AI) is better, higher semantic sim is better,
    lower PPL is better. Weighted combination."""
    pai = summary.get("mean_pai_zhv3", 1.0)
    sim = summary.get("mean_sem_sim", 0.0)
    ppl = summary.get("median_ppl", 1000.0)

    score = (1.0 - pai) * 0.5 + sim * 0.3 + max(0, 1 - ppl / 500) * 0.2
    return score


def main():
    parser = argparse.ArgumentParser(description="Checkpoint Sweep")
    parser.add_argument("--phase1_n", type=int, default=50,
                        help="Number of samples for Phase 1 quick screening")
    parser.add_argument("--phase2_n", type=int, default=200,
                        help="Number of samples for Phase 2 detailed eval")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top checkpoints to evaluate in Phase 2")
    parser.add_argument("--gamma", type=float, default=5.0,
                        help="Gamma for Phase 1 screening")
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--skip_phase1", action="store_true",
                        help="Skip Phase 1, go directly to Phase 2 on all ckpts")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(CKPT_DIR)
    ckpt_paths = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for special in ["best_det.pt", "final.pt"]:
        sp = ckpt_dir / special
        if sp.exists():
            ckpt_paths.append(sp)

    print(f"Found {len(ckpt_paths)} checkpoints:")
    for p in ckpt_paths:
        print(f"  {p.name}")

    samples_p1 = load_test_set(TEST_SET, args.phase1_n)
    samples_p2 = load_test_set(TEST_SET, args.phase2_n)
    print(f"\nPhase 1 samples: {len(samples_p1)}")
    print(f"Phase 2 samples: {len(samples_p2)}")

    print("\nInitializing shared pipeline (Qwen + BERT tokenizer)...")
    pipeline = SharedPipeline(device)

    print("Loading zhv3 detector...")
    detector_zhv3 = SingleDetector("zhv3", device)

    # ── Phase 1 ──
    t0 = time.time()
    if not args.skip_phase1:
        rankings = phase1(
            pipeline, detector_zhv3, samples_p1,
            [str(p) for p in ckpt_paths],
            gamma=args.gamma, num_steps=args.num_steps,
        )
    else:
        rankings = [{"ckpt": p.stem, "path": str(p), "mean_pai": 0.0,
                      "rate_below_50": 0.0, "rate_below_30": 0.0, "std_pai": 0.0}
                     for p in ckpt_paths]

    phase1_time = time.time() - t0
    print(f"\nPhase 1 completed in {phase1_time / 60:.1f} minutes")

    os.makedirs("eval_results", exist_ok=True)
    with open("eval_results/sweep_phase1.json", "w") as f:
        json.dump({"rankings": rankings, "config": {
            "n_samples": len(samples_p1), "gamma": args.gamma,
            "num_steps": args.num_steps,
        }}, f, ensure_ascii=False, indent=2)
    print("Phase 1 results saved to eval_results/sweep_phase1.json")

    # ── Phase 2 ──
    top_ckpts = rankings[:args.top_k]
    print(f"\nTop-{args.top_k} for Phase 2: {[c['ckpt'] for c in top_ckpts]}")

    print("\nLoading additional detectors for Phase 2...")
    multi_det = MultiDetector(["zhv3", "zhv2", "anx-bert", "gpt2-det"], device)

    ppl_scorer = PPLScorer(device)

    gammas_p2 = [4.0, 5.0, 5.5, 6.0, 6.5, 7.0]

    t1 = time.time()
    phase2_results = phase2(
        pipeline, multi_det, ppl_scorer, samples_p2,
        top_ckpts, gammas_p2, num_steps=args.num_steps,
    )
    phase2_time = time.time() - t1
    print(f"\nPhase 2 completed in {phase2_time / 60:.1f} minutes")

    # ── Final ranking ──
    print("\n" + "=" * 100)
    print("  FINAL RANKING")
    print("=" * 100)

    final_ranking = []
    for cr in phase2_results:
        best_gamma = None
        best_composite = -1
        best_summary = None
        for g_str, gdata in cr["gamma_results"].items():
            comp = compute_composite_score(gdata["summary"])
            if comp > best_composite:
                best_composite = comp
                best_gamma = float(g_str)
                best_summary = gdata["summary"]

        final_ranking.append({
            "ckpt": cr["ckpt"],
            "best_gamma": best_gamma,
            "composite_score": best_composite,
            "summary": best_summary,
        })

    final_ranking.sort(key=lambda x: x["composite_score"], reverse=True)

    print(f"\n  {'Rank':>4s} | {'Checkpoint':>15s} | {'γ':>5s} | {'Composite':>10s} | "
          f"{'P(AI)_zhv3':>12s} | {'<0.5%':>6s} | {'Sem.Sim':>8s} | {'PPL(med)':>8s} | {'EditDist':>8s}")
    print(f"  {'-'*100}")

    for i, r in enumerate(final_ranking):
        s = r["summary"]
        marker = " ← BEST" if i == 0 else ""
        print(f"  {i+1:>4d} | {r['ckpt']:>15s} | {r['best_gamma']:>5.1f} | "
              f"{r['composite_score']:>10.4f} | "
              f"{s['mean_pai_zhv3']:.4f}          | {s['rate_below_50_zhv3']:>5.1f}% | "
              f"{s['mean_sem_sim']:>8.4f} | {s['median_ppl']:>8.1f} | "
              f"{s['mean_edit_ratio']:>8.4f}{marker}")

    print("=" * 100)

    winner = final_ranking[0]
    print(f"\n  ★ BEST CHECKPOINT: {winner['ckpt']}")
    print(f"    Best gamma: {winner['best_gamma']}")
    print(f"    Composite score: {winner['composite_score']:.4f}")
    print(f"    P(AI) zhv3: {winner['summary']['mean_pai_zhv3']:.4f}")
    print(f"    Evasion rate (<0.5): {winner['summary']['rate_below_50_zhv3']:.1f}%")
    print(f"    Semantic similarity: {winner['summary']['mean_sem_sim']:.4f}")
    print(f"    PPL (median): {winner['summary']['median_ppl']:.1f}")

    save_data = {
        "final_ranking": final_ranking,
        "phase2_detail": [{
            "ckpt": cr["ckpt"],
            "gamma_results": {
                g: {"summary": gd["summary"]}
                for g, gd in cr["gamma_results"].items()
            },
        } for cr in phase2_results],
        "config": {
            "phase1_n": len(samples_p1),
            "phase2_n": len(samples_p2),
            "top_k": args.top_k,
            "gammas": gammas_p2,
            "num_steps": args.num_steps,
        },
        "timing": {
            "phase1_minutes": phase1_time / 60,
            "phase2_minutes": phase2_time / 60,
            "total_minutes": (phase1_time + phase2_time) / 60,
        },
    }
    with open("eval_results/sweep_final.json", "w") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to eval_results/sweep_final.json")
    print(f"Total time: {(phase1_time + phase2_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
