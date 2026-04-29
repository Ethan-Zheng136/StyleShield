"""ACL-grade comprehensive evaluation for StyleShield.

Supports:
  - Multiple AIGC detectors (zhv3, zhv2, anx-bert, gpt2-det)
  - Per-domain and overall reporting
  - Semantic similarity (Qwen cosine)
  - PPL (GPT-2 Chinese, neutral evaluator)
  - Edit distance, char overlap
  - Statistical significance (mean ± std)
  - Baseline comparison mode (--baseline)

Usage:
    # StyleShield evaluation
    python scripts/eval_acl.py --ckpt experiments/styleflow_v2/checkpoints/step_5000.pt

    # Baseline: LLM rewrite
    python scripts/eval_acl.py --baseline llm_rewrite

    # Baseline: backtranslation
    python scripts/eval_acl.py --baseline backtrans

    # Baseline: synonym substitution
    python scripts/eval_acl.py --baseline synonym
"""

from __future__ import annotations

import argparse
import json
import os
import random
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

TEST_SET = "/root/workspace/AIGC_FUCK/test_set_1000.jsonl"
QWEN_PATH = "/root/workspace/AIGC_FUCK/models/Qwen2.5-7B-Instruct"
GPT2_PATH = "/root/workspace/AIGC_FUCK/models/gpt2-chinese-cluecorpussmall"

DETECTOR_CONFIGS = {
    "zhv3": {"path": "/root/workspace/AIGC_FUCK/models/AIGC_detector_zhv3", "ai_idx": 1},
    "zhv2": {"path": "/root/workspace/AIGC_FUCK/models/AIGC_detector_zhv2", "ai_idx": 1},
    "anx-bert": {"path": "/root/workspace/AIGC_FUCK/models/chinese-ai-detector-bert", "ai_idx": 1},
    "gpt2-det": {"path": "/root/workspace/AIGC_FUCK/models/AITextDetector", "ai_idx": 0},
}

GAMMAS = [5.0, 5.5, 6.0, 6.5, 7.0]
NUM_STEPS = 64
SEED = 42


# ═══════════════════════════════════════════════════════════════════
#  Metrics helpers
# ═══════════════════════════════════════════════════════════════════

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


def char_overlap(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    common = sum(1 for i, c in enumerate(s2[:500]) if i < len(s1) and s1[i] == c)
    return common / max(len(s1[:500]), len(s2[:500]))


# ═══════════════════════════════════════════════════════════════════
#  Detector wrapper
# ═══════════════════════════════════════════════════════════════════

class MultiDetector:
    def __init__(self, names: list[str], device: torch.device):
        self.detectors = {}
        for name in names:
            cfg = DETECTOR_CONFIGS[name]
            tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForSequenceClassification.from_pretrained(
                cfg["path"], trust_remote_code=True
            ).to(device).eval()
            for p in model.parameters():
                p.requires_grad = False
            self.detectors[name] = {
                "tok": tok, "model": model, "ai_idx": cfg["ai_idx"]
            }

    @torch.no_grad()
    def score(self, text: str) -> dict[str, float]:
        results = {}
        for name, d in self.detectors.items():
            enc = d["tok"](
                text, max_length=512, truncation=True, padding=True, return_tensors="pt"
            ).to(d["model"].device)
            logits = d["model"](**enc).logits
            probs = F.softmax(logits.float(), dim=-1)
            results[name] = probs[0, d["ai_idx"]].item()
        return results


# ═══════════════════════════════════════════════════════════════════
#  Baselines
# ═══════════════════════════════════════════════════════════════════

class LLMRewriteBaseline:
    def __init__(self, device: torch.device):
        print("[Baseline] Loading Qwen-7B for LLM rewrite...")
        self.tok = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            QWEN_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device).eval()
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    @torch.no_grad()
    def rewrite(self, text: str) -> str:
        prompt = f"请用更自然、更像人类写作的风格重写以下文本，保持原意不变，不要添加任何额外内容：\n\n{text}"
        messages = [{"role": "user", "content": prompt}]
        chat_text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(chat_text, return_tensors="pt", max_length=1024, truncation=True).to(self.model.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.model.generate(
                **enc, max_new_tokens=512, do_sample=True, temperature=0.7,
                top_p=0.9, pad_token_id=self.tok.pad_token_id,
            )
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        return self.tok.decode(new_tokens, skip_special_tokens=True).strip()


class BacktransBaseline:
    def __init__(self, device: torch.device):
        print("[Baseline] Loading NLLB for backtranslation...")
        from transformers import AutoModelForSeq2SeqLM
        nllb_name = "facebook/nllb-200-distilled-600M"
        self.tok = AutoTokenizer.from_pretrained(nllb_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(nllb_name).to(device).eval()

    @torch.no_grad()
    def rewrite(self, text: str) -> str:
        self.tok.src_lang = "zho_Hans"
        enc = self.tok(text, return_tensors="pt", max_length=512, truncation=True).to(self.model.device)
        en_ids = self.model.generate(
            **enc, forced_bos_token_id=self.tok.convert_tokens_to_ids("eng_Latn"),
            max_new_tokens=512,
        )
        en_text = self.tok.decode(en_ids[0], skip_special_tokens=True)

        self.tok.src_lang = "eng_Latn"
        enc2 = self.tok(en_text, return_tensors="pt", max_length=512, truncation=True).to(self.model.device)
        zh_ids = self.model.generate(
            **enc2, forced_bos_token_id=self.tok.convert_tokens_to_ids("zho_Hans"),
            max_new_tokens=512,
        )
        return self.tok.decode(zh_ids[0], skip_special_tokens=True).strip()


class SynonymBaseline:
    def __init__(self):
        print("[Baseline] Loading jieba + synonyms for synonym substitution...")
        import jieba
        self.jieba = jieba

    def rewrite(self, text: str, replace_ratio: float = 0.2) -> str:
        import jieba.posseg as pseg
        words = list(pseg.cut(text))
        result = []
        for word, flag in words:
            if flag.startswith(("n", "v", "a")) and len(word) >= 2 and random.random() < replace_ratio:
                try:
                    import synonyms
                    syns, scores = synonyms.nearby(word, size=5)
                    candidates = [s for s, sc in zip(syns, scores) if s != word and sc > 0.7]
                    if candidates:
                        result.append(random.choice(candidates))
                        continue
                except Exception:
                    pass
            result.append(word)
        return "".join(result)


# ═══════════════════════════════════════════════════════════════════
#  StyleShield transfer
# ═══════════════════════════════════════════════════════════════════

class StyleShieldTransfer:
    def __init__(self, ckpt_path: str, device: torch.device):
        from src.config import StyleFlowConfig
        from src.model import StyleFlowZh
        from src.qwen_encoder import QwenHiddenExtractor

        print(f"[StyleShield] Loading checkpoint: {ckpt_path}")
        cfg = StyleFlowConfig()
        model, vocab_size, cfg = StyleFlowZh.from_langflow_ckpt(ckpt_path, cfg, device)
        self.model = model.to(device).eval()
        self.cfg = cfg

        qwen_raw = AutoModelForCausalLM.from_pretrained(
            cfg.qwen_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
        self.qwen_encoder = QwenHiddenExtractor(qwen_raw, split_layer=cfg.qwen_split_layer)
        self.qwen_encoder.eval()

        bert_tok_path = str(Path(__file__).resolve().parent.parent / "tokenizer" / "bert-base-chinese")
        self.bert_tok = AutoTokenizer.from_pretrained(bert_tok_path)
        self.qwen_tok = AutoTokenizer.from_pretrained(
            cfg.qwen_tokenizer_name or cfg.qwen_model_path, trust_remote_code=True
        )
        self.device = device

    @torch.no_grad()
    def transfer(self, text: str, gamma: float, num_steps: int = 64) -> str:
        ml = self.cfg.model_length
        b_enc = self.bert_tok(
            text, max_length=ml, padding="max_length", truncation=True, return_tensors="pt"
        ).to(self.device)
        q_enc = self.qwen_tok(
            text, max_length=ml, padding="max_length", truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h = self.qwen_encoder.encode(q_enc["input_ids"], q_enc["attention_mask"])
        cond_kv = self.model.project_qwen_cond(h)
        x_ai = self.model.embed_tokens(b_enc["input_ids"])
        out_ids = self.model.transfer(
            x_ai_embed=x_ai, cond_kv=cond_kv,
            gamma_start=gamma, num_steps=num_steps,
        )
        return self.bert_tok.decode(out_ids[0], skip_special_tokens=True).replace(" ", "")


# ═══════════════════════════════════════════════════════════════════
#  PPL & Semantic Similarity
# ═══════════════════════════════════════════════════════════════════

class PPLScorer:
    def __init__(self, device: torch.device):
        print("[PPL] Loading GPT-2 Chinese (neutral evaluator)...")
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


class SemanticScorer:
    def __init__(self, qwen_encoder, qwen_tok, device: torch.device, max_length: int = 512):
        self.encoder = qwen_encoder
        self.tok = qwen_tok
        self.device = device
        self.ml = max_length

    @torch.no_grad()
    def score(self, text1: str, text2: str) -> float:
        enc1 = self.tok(text1, max_length=self.ml, padding="max_length",
                        truncation=True, return_tensors="pt").to(self.device)
        enc2 = self.tok(text2, max_length=self.ml, padding="max_length",
                        truncation=True, return_tensors="pt").to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h1 = self.encoder.encode(enc1["input_ids"], enc1["attention_mask"])
            h2 = self.encoder.encode(enc2["input_ids"], enc2["attention_mask"])
        h1_mean = h1.float().mean(dim=1)
        h2_mean = h2.float().mean(dim=1)
        return F.cosine_similarity(h1_mean, h2_mean).item()


# ═══════════════════════════════════════════════════════════════════
#  Main evaluation loop
# ═══════════════════════════════════════════════════════════════════

def load_test_set(path: str, n: int | None = None) -> list[dict]:
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
    if n:
        random.seed(SEED)
        samples = random.sample(samples, min(n, len(samples)))
    return samples


def run_eval(args):
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test set
    samples = load_test_set(args.test_set, args.n_samples)
    print(f"Test set: {len(samples)} samples")
    domain_counts = {}
    for s in samples:
        domain_counts[s["domain"]] = domain_counts.get(s["domain"], 0) + 1
    print(f"  Domains: {domain_counts}")

    # Load detectors
    det_names = args.detectors.split(",")
    print(f"\nLoading {len(det_names)} detectors: {det_names}")
    multi_det = MultiDetector(det_names, device)

    # Load PPL scorer
    ppl_scorer = PPLScorer(device)

    # Determine method
    if args.baseline:
        method_name = args.baseline
        if args.baseline == "llm_rewrite":
            baseline = LLMRewriteBaseline(device)
            rewrite_fn = baseline.rewrite
        elif args.baseline == "backtrans":
            baseline = BacktransBaseline(device)
            rewrite_fn = baseline.rewrite
        elif args.baseline == "synonym":
            baseline = SynonymBaseline()
            rewrite_fn = baseline.rewrite
        else:
            raise ValueError(f"Unknown baseline: {args.baseline}")
        gammas = [0.0]
        sem_scorer = None
    else:
        method_name = f"StyleShield ({Path(args.ckpt).stem})"
        ss = StyleShieldTransfer(args.ckpt, device)
        sem_scorer = SemanticScorer(ss.qwen_encoder, ss.qwen_tok, device)
        gammas = args.gammas if args.gammas else GAMMAS

        def rewrite_fn(text, gamma=5.0, **kw):
            return ss.transfer(text, gamma=gamma, num_steps=NUM_STEPS)

    # If no semantic scorer yet (baseline mode), load Qwen encoder
    if sem_scorer is None:
        print("[Semantic] Loading Qwen encoder for similarity...")
        from src.qwen_encoder import QwenHiddenExtractor
        qwen_raw = AutoModelForCausalLM.from_pretrained(
            QWEN_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
        from src.config import StyleFlowConfig
        cfg = StyleFlowConfig()
        qwen_enc = QwenHiddenExtractor(qwen_raw, split_layer=cfg.qwen_split_layer)
        qwen_enc.eval()
        qwen_tok = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
        sem_scorer = SemanticScorer(qwen_enc, qwen_tok, device)

    # Run evaluation
    results = {g: [] for g in gammas}
    total_tasks = len(samples) * len(gammas)
    print(f"\nRunning: {len(samples)} samples x {len(gammas)} settings = {total_tasks} rewrites")
    print("=" * 80)

    t0 = time.time()
    for si, sample in enumerate(tqdm(samples, desc="Evaluating")):
        ai_text = sample["ai"]
        domain = sample["domain"]

        for gamma in gammas:
            if args.baseline:
                output = rewrite_fn(ai_text)
            else:
                output = rewrite_fn(ai_text, gamma=gamma)

            det_scores = multi_det.score(output)
            sem_sim = sem_scorer.score(ai_text, output)
            ppl_val = ppl_scorer.score(output)
            edit_ratio = edit_distance_ratio(ai_text, output)
            char_ovlp = char_overlap(ai_text, output)

            results[gamma].append({
                "id": sample["id"],
                "domain": domain,
                "det_scores": det_scores,
                "semantic_sim": sem_sim,
                "ppl": ppl_val,
                "edit_ratio": edit_ratio,
                "char_overlap": char_ovlp,
                "output_len": len(output),
                "input_len": len(ai_text),
            })

    elapsed = time.time() - t0
    print(f"\nEvaluation completed in {elapsed / 60:.1f} minutes")

    # ═══════════════════════════════════════════════════════════════
    #  Report
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 100)
    print(f"EVALUATION REPORT: {method_name}")
    print(f"Test set: {len(samples)} samples | Detectors: {det_names}")
    print("=" * 100)

    domains_to_report = ["overall"] + sorted(set(s["domain"] for s in samples))

    for domain_filter in domains_to_report:
        if domain_filter == "overall":
            domain_label = "OVERALL"
            filter_fn = lambda r: True
        else:
            domain_label = domain_filter.upper()
            filter_fn = lambda r, d=domain_filter: r["domain"] == d

        n_domain = sum(1 for s in samples if filter_fn({"domain": s["domain"]}))
        if n_domain == 0:
            continue

        print(f"\n{'─'*80}")
        print(f"  {domain_label} (n={n_domain})")
        print(f"{'─'*80}")

        for det_name in det_names:
            print(f"\n  Detector: {det_name}")
            header = f"  {'γ':>5s} | {'mean±std P(AI)':>18s} | {'<0.5':>6s} | {'<0.3':>6s} | "
            header += f"{'Sem.Sim':>10s} | {'PPL(med)':>10s} | {'EditDist':>10s}"
            print(header)
            print(f"  {'-'*85}")

            for gamma in gammas:
                rs = [r for r in results[gamma] if filter_fn(r)]
                if not rs:
                    continue
                pais = [r["det_scores"][det_name] for r in rs]
                sims = [r["semantic_sim"] for r in rs]
                ppls = [r["ppl"] for r in rs if not np.isnan(r["ppl"])]
                edits = [r["edit_ratio"] for r in rs]

                g_label = f"{gamma:.1f}" if gamma > 0 else "-"
                mean_pai = np.mean(pais)
                std_pai = np.std(pais)
                rate_50 = np.mean([p < 0.5 for p in pais]) * 100
                rate_30 = np.mean([p < 0.3 for p in pais]) * 100
                mean_sim = np.mean(sims)
                std_sim = np.std(sims)
                med_ppl = np.nanmedian(ppls) if ppls else float("nan")
                mean_edit = np.mean(edits)

                print(f"  {g_label:>5s} | {mean_pai:.4f}±{std_pai:.4f}   | "
                      f"{rate_50:>5.1f}% | {rate_30:>5.1f}% | "
                      f"{mean_sim:.4f}±{std_sim:.4f} | {med_ppl:>10.1f} | "
                      f"{mean_edit:.4f}±{np.std(edits):.4f}")

    # PPL baselines
    print(f"\n{'─'*80}")
    print("  PPL BASELINES (GPT-2 Chinese)")
    print(f"{'─'*80}")
    ai_ppls = [ppl_scorer.score(s["ai"]) for s in samples[:100]]
    hu_ppls = [ppl_scorer.score(s["human"]) for s in samples[:100]]
    print(f"  AI original:  median={np.nanmedian(ai_ppls):.1f}  mean={np.nanmean(ai_ppls):.1f}")
    print(f"  Human text:   median={np.nanmedian(hu_ppls):.1f}  mean={np.nanmean(hu_ppls):.1f}")

    # Save results
    os.makedirs("eval_results", exist_ok=True)
    tag = args.tag if args.tag else (args.baseline if args.baseline else Path(args.ckpt).stem)
    save_path = f"eval_results/acl_eval_{tag}.json"
    with open(save_path, "w") as f:
        json.dump({
            "method": method_name,
            "config": {
                "test_set": args.test_set,
                "n_samples": len(samples),
                "detectors": det_names,
                "gammas": gammas,
                "num_steps": NUM_STEPS,
            },
            "domain_counts": domain_counts,
            "results_by_gamma": {str(g): rs for g, rs in results.items()},
            "ppl_baseline_ai": float(np.nanmean(ai_ppls)),
            "ppl_baseline_human": float(np.nanmean(hu_ppls)),
        }, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="ACL Evaluation for StyleShield")
    parser.add_argument("--ckpt", type=str, default=None, help="StyleShield checkpoint")
    parser.add_argument("--baseline", type=str, default=None,
                        choices=["llm_rewrite", "backtrans", "synonym"],
                        help="Run baseline method instead of StyleShield")
    parser.add_argument("--test_set", type=str, default=TEST_SET)
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Subsample N from test set (default: use all)")
    parser.add_argument("--detectors", type=str, default="zhv3,zhv2,anx-bert,gpt2-det",
                        help="Comma-separated detector names")
    parser.add_argument("--tag", type=str, default=None,
                        help="Custom tag for output filename (default: auto from ckpt/baseline)")
    parser.add_argument("--gammas", type=float, nargs="+", default=None,
                        help="Override gamma values")
    args = parser.parse_args()

    if args.ckpt is None and args.baseline is None:
        parser.error("Provide --ckpt or --baseline")

    run_eval(args)


if __name__ == "__main__":
    main()
