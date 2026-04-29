"""StyleFlow Allocator: 可控 AIGC 检测率的长文本改写系统

将 StyleFlow (SDEdit + Qwen conditioning) 集成到 allocator 框架中。
与旧版 LLaMA-SFT allocator 不同，StyleFlow 支持：
  - gamma 控制改写强度（连续可调）
  - 多次重试选最优（SDEdit 有随机性，每次结果不同）
  - 自适应 gamma 搜索（先小 gamma 尝试，不够再加大）

核心流程:
  1. 按句子/段落将长文本分成 chunks
  2. 用 AIGC 检测器给每个 chunk 打分
  3. 按 P(AI) 从高到低排序，优先改写"最像 AI"的 chunk
  4. 对每个选中的 chunk：用 StyleFlow 多次尝试不同 gamma，
     选择 P(AI) 最低且文本质量可接受的版本
  5. 贪心累加，直到整体加权 P(AI) 逼近目标 rate
  6. 拼接输出

用法:
  python scripts/allocator_sf.py --text_file input.txt --target_rate 60
  python scripts/allocator_sf.py --text "长文本..." --target_rate 30
  python scripts/allocator_sf.py --text_file input.txt --target_rate 50 --retries 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

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


DETECTOR_PATH = "/root/workspace/AIGC_FUCK/models/AIGC_detector_zhv3"
CKPT_PATH = "/root/workspace/AIGC_FUCK_7/experiments/styleflow_v2/checkpoints/step_30000.pt"
OUTPUT_DIR = "/root/workspace/AIGC_FUCK_7/allocator_results"


# ═══════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    """一个文本块及其元信息"""
    index: int
    text: str
    char_count: int = 0
    p_ai_original: float = -1.0
    rewritten_text: Optional[str] = None
    p_ai_rewritten: float = -1.0
    selected_for_rewrite: bool = False
    gamma_used: float = -1.0
    attempts: int = 0

    def __post_init__(self):
        self.char_count = len(self.text)

    @property
    def final_text(self) -> str:
        if self.selected_for_rewrite and self.rewritten_text is not None:
            return self.rewritten_text
        return self.text


# ═══════════════════════════════════════════════════════════════════
#  Text Chunking (复用原版 allocator 逻辑)
# ═══════════════════════════════════════════════════════════════════

def chunk_by_sentence(text: str, target_size: int = 500) -> list[str]:
    sentence_endings = re.compile(r'([。！？；\n]+)')
    parts = sentence_endings.split(text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentences.append(parts[i] + parts[i + 1])
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1])

    chunks = []
    buffer = ""
    for sent in sentences:
        if len(buffer) + len(sent) > target_size and buffer:
            chunks.append(buffer.strip())
            buffer = sent
        else:
            buffer += sent
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks


def chunk_by_paragraph(text: str, min_chars: int = 100) -> list[str]:
    raw_paragraphs = re.split(r'\n\s*\n|\n', text)
    raw_paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]
    chunks = []
    buffer = ""
    for para in raw_paragraphs:
        buffer = (buffer + "\n" + para).strip() if buffer else para
        if len(buffer) >= min_chars:
            chunks.append(buffer)
            buffer = ""
    if buffer:
        if chunks:
            chunks[-1] = chunks[-1] + "\n" + buffer
        else:
            chunks.append(buffer)
    return chunks


def chunk_text(text: str, method: str = "sentence",
               chunk_size: int = 500) -> list[str]:
    if method == "paragraph":
        return chunk_by_paragraph(text)
    elif method == "sentence":
        return chunk_by_sentence(text, target_size=chunk_size)
    else:
        raise ValueError(f"Unknown chunk method: {method}")


# ═══════════════════════════════════════════════════════════════════
#  Model Loading
# ═══════════════════════════════════════════════════════════════════

class StyleFlowPipeline:
    """Encapsulates StyleFlow model + Qwen encoder + detector."""

    def __init__(self, ckpt_path: str, detector_path: str,
                 device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[Pipeline] Loading StyleFlow from {ckpt_path}")
        cfg = StyleFlowConfig()
        model, vocab_size, cfg = StyleFlowZh.from_langflow_ckpt(ckpt_path, cfg, self.device)
        self.model = model.to(self.device).eval()
        self.cfg = cfg

        print(f"[Pipeline] Loading Qwen encoder from {cfg.qwen_model_path}")
        qwen_raw = AutoModelForCausalLM.from_pretrained(
            cfg.qwen_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.qwen_encoder = QwenHiddenExtractor(qwen_raw, split_layer=cfg.qwen_split_layer)
        self.qwen_encoder.eval()

        bert_tok_path = str(Path(__file__).resolve().parent.parent / "tokenizer" / "bert-base-chinese")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_tok_path)
        qwen_tok_path = cfg.qwen_tokenizer_name or cfg.qwen_model_path
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_tok_path, trust_remote_code=True)

        print(f"[Pipeline] Loading detector from {detector_path}")
        self.det_tokenizer = AutoTokenizer.from_pretrained(detector_path)
        self.det_model = AutoModelForSequenceClassification.from_pretrained(detector_path)
        self.det_model.to(self.device).eval()
        for p in self.det_model.parameters():
            p.requires_grad = False

        labels = self.det_model.config.id2label
        self.ai_idx = 1
        for idx, name in labels.items():
            if "ai" in name.lower() or "machine" in name.lower():
                self.ai_idx = int(idx)
                break

    @torch.no_grad()
    def score_text(self, text: str, max_length: int = 512) -> float:
        enc = self.det_tokenizer(
            text, max_length=max_length, padding=True,
            truncation=True, return_tensors="pt",
        ).to(self.device)
        logits = self.det_model(**enc).logits
        probs = F.softmax(logits.float(), dim=-1)
        return probs[0, self.ai_idx].item()

    @torch.no_grad()
    def score_texts(self, texts: list[str], max_length: int = 512,
                    batch_size: int = 16) -> list[float]:
        all_scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.det_tokenizer(
                batch, max_length=max_length, truncation=True,
                padding=True, return_tensors="pt",
            ).to(self.device)
            logits = self.det_model(**enc).logits
            probs = F.softmax(logits.float(), dim=-1).cpu()
            all_scores.extend(probs[:, self.ai_idx].tolist())
        return all_scores

    @torch.no_grad()
    def transfer(self, text: str, gamma_start: float = 6.0,
                 num_steps: int = 32) -> str:
        max_length = self.cfg.model_length

        bert_enc = self.bert_tokenizer(
            text, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        ).to(self.device)
        ai_ids_bert = bert_enc["input_ids"]

        qwen_enc = self.qwen_tokenizer(
            text, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        ).to(self.device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h_ai = self.qwen_encoder.encode(qwen_enc["input_ids"], qwen_enc["attention_mask"])

        cond_kv = self.model.project_qwen_cond(h_ai)
        x_ai = self.model.embed_tokens(ai_ids_bert)

        output_ids = self.model.transfer(
            x_ai_embed=x_ai, cond_kv=cond_kv,
            gamma_start=gamma_start, num_steps=num_steps,
        )

        output_text = self.bert_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text.replace(" ", "")

    def transfer_with_retry(self, text: str, gamma_start: float = 6.0,
                            num_steps: int = 32, retries: int = 3) -> tuple[str, float]:
        """多次尝试 transfer，选 P(AI) 最低的结果。"""
        best_text = text
        best_p_ai = 1.0

        for _ in range(retries):
            output = self.transfer(text, gamma_start=gamma_start, num_steps=num_steps)
            p_ai = self.score_text(output)
            if p_ai < best_p_ai:
                best_p_ai = p_ai
                best_text = output

        return best_text, best_p_ai

    def adaptive_transfer(self, text: str, target_p_ai: float = 0.5,
                          num_steps: int = 32, retries: int = 3,
                          gamma_candidates: list[float] = None) -> tuple[str, float, float]:
        """自适应 gamma + 多次重试。

        从小 gamma 开始尝试，逐渐加大，直到 P(AI) 达到目标。
        每个 gamma 多次重试取最佳。

        Returns: (best_text, best_p_ai, gamma_used)
        """
        if gamma_candidates is None:
            gamma_candidates = [6.0, 6.5, 7.0, 7.5]

        best_text = text
        best_p_ai = self.score_text(text)
        best_gamma = 0.0

        for gamma in gamma_candidates:
            text_out, p_ai = self.transfer_with_retry(
                text, gamma_start=gamma, num_steps=num_steps, retries=retries,
            )

            if abs(p_ai - target_p_ai) < abs(best_p_ai - target_p_ai):
                best_text = text_out
                best_p_ai = p_ai
                best_gamma = gamma

            if p_ai <= target_p_ai:
                break

        return best_text, best_p_ai, best_gamma


# ═══════════════════════════════════════════════════════════════════
#  Allocation Strategy
# ═══════════════════════════════════════════════════════════════════

def weighted_chunk_rate(chunks: list[Chunk]) -> float:
    total_chars = sum(c.char_count for c in chunks)
    if total_chars == 0:
        return 0.0
    weighted_sum = 0.0
    for c in chunks:
        p_ai = c.p_ai_rewritten if c.selected_for_rewrite else c.p_ai_original
        weighted_sum += p_ai * c.char_count
    return weighted_sum / total_chars


def allocate_and_rewrite(
    text: str,
    target_rate: float,
    pipeline: StyleFlowPipeline,
    chunk_method: str = "sentence",
    chunk_size: int = 500,
    num_steps: int = 32,
    retries: int = 3,
    gamma_candidates: list[float] = None,
    tolerance: float = 0.02,
    verbose: bool = True,
) -> dict:
    """核心函数：分块 → 打分 → 贪心逐块改写 → 动态停止

    策略 (greedy-adaptive):
      1. 分块 + 打分
      2. 按 P(AI) 降序排列 chunk
      3. 从最高 P(AI) 的 chunk 开始，用 adaptive_transfer 改写
      4. 每改写一个 chunk，重新估算整体加权 P(AI)
      5. 当整体 P(AI) <= target 时停止
      6. 改写后如果低于目标太多，跳过当前块（不改写），
         让后续更高 P(AI) 的块来填补
    """
    target_p_ai = target_rate / 100.0
    t_start = time.time()

    if gamma_candidates is None:
        gamma_candidates = [6.0, 6.5, 7.0, 7.5]

    raw_chunks = chunk_text(text, method=chunk_method, chunk_size=chunk_size)
    chunks = [Chunk(index=i, text=t) for i, t in enumerate(raw_chunks)]
    n_chunks = len(chunks)
    total_chars = sum(c.char_count for c in chunks)

    if verbose:
        print(f"\n[Allocator] 输入: {total_chars} 字, 分成 {n_chunks} 块")
        print(f"[Allocator] 目标 AIGC 疑似率: {target_rate}% (P(AI)={target_p_ai:.2f})")

    if verbose:
        print(f"[Allocator] 对 {n_chunks} 个 chunk 打分...")
    scores = pipeline.score_texts([c.text for c in chunks])
    for i, score in enumerate(scores):
        chunks[i].p_ai_original = score
    if verbose:
        for c in chunks:
            print(f"  chunk[{c.index}]: P(AI)={c.p_ai_original:.4f} ({c.char_count}字)")

    original_rate = weighted_chunk_rate(chunks)
    if verbose:
        print(f"[Allocator] 原始加权 P(AI): {original_rate:.4f}")

    if target_p_ai >= original_rate - tolerance:
        if verbose:
            print(f"[Allocator] 目标 rate >= 原始 rate, 无需改写")
        return _build_result(chunks, target_rate, text, t_start, pipeline, verbose)

    ranked = sorted(range(n_chunks), key=lambda i: chunks[i].p_ai_original, reverse=True)

    if verbose:
        print(f"\n[Allocator] 开始贪心改写 (retries={retries}, gammas={gamma_candidates})")

    for rank, chunk_idx in enumerate(ranked):
        c = chunks[chunk_idx]
        current_rate = weighted_chunk_rate(chunks)

        if current_rate <= target_p_ai + tolerance:
            if verbose:
                print(f"\n[Allocator] 已达目标! 当前加权 P(AI)={current_rate:.4f}, "
                      f"改写了 {rank}/{n_chunks} 块")
            break

        remaining_drop = current_rate - target_p_ai
        chunk_weight = c.char_count / total_chars
        max_drop_this_chunk = (c.p_ai_original - 0.0) * chunk_weight

        if verbose:
            print(f"\n  改写 chunk[{c.index}] (P(AI)={c.p_ai_original:.4f}, {c.char_count}字)...")

        ideal_new_p = c.p_ai_original - remaining_drop / chunk_weight
        ideal_new_p = max(ideal_new_p, 0.0)

        best_text, best_p_ai, gamma_used = pipeline.adaptive_transfer(
            c.text,
            target_p_ai=max(ideal_new_p, 0.0),
            num_steps=num_steps,
            retries=retries,
            gamma_candidates=gamma_candidates,
        )

        c.rewritten_text = best_text
        c.p_ai_rewritten = best_p_ai
        c.selected_for_rewrite = True
        c.gamma_used = gamma_used
        c.attempts = retries * len(gamma_candidates)

        new_rate = weighted_chunk_rate(chunks)

        if new_rate < target_p_ai - tolerance and best_p_ai < c.p_ai_original * 0.3:
            c.selected_for_rewrite = False
            c.rewritten_text = None
            c.p_ai_rewritten = -1.0
            if verbose:
                print(f"    γ={gamma_used:.1f}, P(AI): {c.p_ai_original:.4f} → {best_p_ai:.4f}")
                print(f"    ⚠ 改写后过低 ({new_rate:.4f} < {target_p_ai:.4f}), 跳过此块")
            continue

        if verbose:
            print(f"    γ={gamma_used:.1f}, P(AI): {c.p_ai_original:.4f} → {best_p_ai:.4f}")
            print(f"    加权 P(AI): {current_rate:.4f} → {new_rate:.4f}")

    return _build_result(chunks, target_rate, text, t_start, pipeline, verbose)


def _build_result(chunks: list[Chunk], target_rate: float,
                  original_text: str, t_start: float,
                  pipeline: StyleFlowPipeline = None,
                  verbose: bool = False) -> dict:
    final_text = "\n".join(c.final_text for c in chunks)
    achieved_rate = weighted_chunk_rate(chunks)

    n_rewritten = sum(1 for c in chunks if c.selected_for_rewrite)
    total_chars = sum(c.char_count for c in chunks)
    rewritten_chars = sum(c.char_count for c in chunks if c.selected_for_rewrite)

    final_p_ais = [c.p_ai_rewritten if c.selected_for_rewrite else c.p_ai_original
                   for c in chunks]
    n_below_50 = sum(1 for p in final_p_ais if p < 0.5)
    n_below_30 = sum(1 for p in final_p_ais if p < 0.3)
    orig_scores = [c.p_ai_original for c in chunks]
    orig_weighted = sum(p * c.char_count for p, c in zip(orig_scores, chunks)) / max(total_chars, 1)

    stats = {
        "n_chunks": len(chunks),
        "n_rewritten": n_rewritten,
        "n_kept": len(chunks) - n_rewritten,
        "total_chars_input": len(original_text),
        "total_chars_output": len(final_text),
        "rewrite_char_ratio": round(rewritten_chars / max(total_chars, 1), 4),
        "target_rate": target_rate,
        "achieved_rate_weighted": round(achieved_rate * 100, 2),
        "original_rate_weighted": round(orig_weighted * 100, 2),
        "chunks_below_50pct": n_below_50,
        "chunks_below_30pct": n_below_30,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }

    chunk_details = []
    for c in chunks:
        detail = {
            "index": c.index,
            "char_count": c.char_count,
            "p_ai_original": round(c.p_ai_original, 4),
            "rewritten": c.selected_for_rewrite,
        }
        if c.selected_for_rewrite:
            detail["p_ai_rewritten"] = round(c.p_ai_rewritten, 4)
            detail["gamma_used"] = c.gamma_used
            detail["attempts"] = c.attempts
        chunk_details.append(detail)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  StyleFlow Allocator 结果 (逐段检测)")
        print(f"{'='*60}")
        print(f"  目标 AIGC 疑似率:     {target_rate}%")
        print(f"  达成 AIGC 疑似率:     {stats['achieved_rate_weighted']}%  "
              f"(原始: {stats['original_rate_weighted']}%)")
        print(f"  改写块数:             {n_rewritten}/{len(chunks)}")
        print(f"  改写字符比例:         {stats['rewrite_char_ratio']*100:.1f}%")
        print(f"  P(AI)<0.5 的块数:     {n_below_50}/{len(chunks)}")
        print(f"  P(AI)<0.3 的块数:     {n_below_30}/{len(chunks)}")
        print(f"  耗时:                 {stats['elapsed_seconds']}s")
        print(f"{'='*60}\n")

        for d in chunk_details:
            status = "✎ 改写" if d["rewritten"] else "  保留"
            p_orig = d["p_ai_original"]
            p_new = d.get("p_ai_rewritten", p_orig)
            gamma = d.get("gamma_used", "")
            gamma_str = f" γ={gamma:.1f}" if isinstance(gamma, float) and gamma > 0 else ""
            print(f"  [{status}] chunk[{d['index']}] "
                  f"P(AI): {p_orig:.4f} → {p_new:.4f}{gamma_str}  ({d['char_count']}字)")

    return {
        "final_text": final_text,
        "original_text": original_text,
        "target_rate": target_rate,
        "achieved_rate_weighted": stats["achieved_rate_weighted"],
        "original_rate_weighted": stats["original_rate_weighted"],
        "stats": stats,
        "chunk_details": chunk_details,
    }


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="StyleFlow Allocator: 可控 AIGC rate 的长文本改写",
    )
    p.add_argument("--text", type=str, default=None, help="直接传入文本")
    p.add_argument("--text_file", type=str, default=None, help="从文件读取文本")
    p.add_argument("--target_rate", type=float, required=True,
                   help="目标 AIGC rate (0-100)")
    p.add_argument("--chunk_method", choices=["paragraph", "sentence"],
                   default="sentence")
    p.add_argument("--chunk_size", type=int, default=500)

    p.add_argument("--ckpt", default=CKPT_PATH)
    p.add_argument("--detector_path", default=DETECTOR_PATH)

    p.add_argument("--num_steps", type=int, default=64)
    p.add_argument("--retries", type=int, default=3,
                   help="每个 gamma 重试次数 (default 3)")
    p.add_argument("--gammas", type=float, nargs="+",
                   default=[6.0, 6.5, 7.0, 7.5],
                   help="候选 gamma 值列表 (从小到大)")
    p.add_argument("--tolerance", type=float, default=0.02)

    p.add_argument("--output_dir", default=OUTPUT_DIR)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.text:
        text = args.text
    elif args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print("错误: 请提供 --text 或 --text_file")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  StyleFlow Allocator")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  输入文本: {len(text)} 字")
    print(f"  目标 AIGC rate: {args.target_rate}%")
    print(f"{'='*60}")

    pipeline = StyleFlowPipeline(
        ckpt_path=args.ckpt,
        detector_path=args.detector_path,
    )

    result = allocate_and_rewrite(
        text=text,
        target_rate=args.target_rate,
        pipeline=pipeline,
        chunk_method=args.chunk_method,
        chunk_size=args.chunk_size,
        num_steps=args.num_steps,
        retries=args.retries,
        gamma_candidates=sorted(args.gammas),
        tolerance=args.tolerance,
        verbose=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = args.output or f"sf_alloc_rate{int(args.target_rate)}_{timestamp}.json"
    out_path = os.path.join(args.output_dir, out_name)

    save_data = {
        "target_rate": result["target_rate"],
        "achieved_rate_weighted": result["achieved_rate_weighted"],
        "full_text_p_ai": result.get("full_text_p_ai"),
        "original_p_ai": result.get("original_p_ai"),
        "stats": result["stats"],
        "chunk_details": result["chunk_details"],
        "final_text": result["final_text"],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
