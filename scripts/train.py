"""Training script for StyleShield: LangFlow denoising + Qwen cross-attention.

Same training objective as original LangFlow (CE loss on noisy→clean),
but with Qwen encoder providing cross-attention conditioning.
At inference, SDEdit-style transfer is used (add noise → denoise with condition).

Usage (single-node multi-GPU):
    torchrun --nproc_per_node=8 scripts/train.py --config configs/styleshield.yaml

Usage (single GPU):
    python scripts/train.py --config configs/styleshield.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import StyleShieldConfig
from src.dataset import PairDataset
from src.model import StyleShieldModel
from src.qwen_encoder import QwenHiddenExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("StyleShield")


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def is_dist():
    return dist.is_initialized()


def rank():
    return dist.get_rank() if is_dist() else 0


def world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main():
    return rank() == 0


def load_config(yaml_path: str) -> StyleShieldConfig:
    cfg = StyleShieldConfig()
    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            overrides = yaml.safe_load(f)
        if overrides:
            for k, v in overrides.items():
                if hasattr(cfg, k):
                    expected_type = type(getattr(cfg, k))
                    if expected_type in (int, float, bool, str) and v is not None:
                        v = expected_type(v)
                    setattr(cfg, k, v)
    return cfg



def get_scheduler(optimizer, cfg: StyleShieldConfig):
    """Cosine-annealing with linear warmup."""
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0,
        total_iters=max(cfg.warmup_steps, 1),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.total_steps - cfg.warmup_steps, 1),
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine],
        milestones=[cfg.warmup_steps],
    )


# ═══════════════════════════════════════════════════════════════════
#  Detector
# ═══════════════════════════════════════════════════════════════════

class AIGCDetector:
    """Wraps the AIGC detector for computing detector loss."""

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
    def score_texts(self, texts: list[str], max_length: int = 512) -> torch.Tensor:
        """Return P(AI) for each text."""
        enc = self.tokenizer(
            texts, max_length=max_length, padding=True,
            truncation=True, return_tensors="pt",
        ).to(self.model.device)
        logits = self.model(**enc).logits
        probs = F.softmax(logits.float(), dim=-1)
        return probs[:, self.ai_idx]

    def detector_reward(self, texts: list[str]) -> float:
        """Return mean (1 - P_ai) as reward."""
        p_ai = self.score_texts(texts)
        return (1.0 - p_ai).mean().item()


# ═══════════════════════════════════════════════════════════════════
#  Training Loop
# ═══════════════════════════════════════════════════════════════════

def train(cfg: StyleShieldConfig):
    # ── DDP setup ──
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if is_main():
        log.info("=" * 60)
        log.info("  StyleShield: Conditional Flow Matching Training")
        log.info("=" * 60)

    # ── Load pretrained LangFlow → StyleShieldModel ──
    if is_main():
        log.info(f"Loading LangFlow from {cfg.langflow_ckpt}")
    model, vocab_size, cfg = StyleShieldModel.from_langflow_ckpt(
        cfg.langflow_ckpt, cfg, device=device,
    )
    model = model.to(device)

    for p in model.proposal.parameters():
        p.requires_grad = False

    if is_main():
        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Model params: {total_p:,} total, {train_p:,} trainable")

    # ── Load frozen Qwen encoder ──
    if is_main():
        log.info(f"Loading Qwen encoder from {cfg.qwen_model_path}")
    qwen_raw = AutoModelForCausalLM.from_pretrained(
        cfg.qwen_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    qwen_encoder = QwenHiddenExtractor(qwen_raw, split_layer=cfg.qwen_split_layer)
    qwen_encoder.eval()
    if is_main():
        log.info(f"Qwen encoder: split_layer={cfg.qwen_split_layer}, "
                 f"hidden_size={qwen_encoder.hidden_size}")

    # ── Detector (all ranks need it for REINFORCE loss in DDP) ──
    detector = None
    if cfg.det_weight > 0:
        if is_main():
            log.info(f"Loading AIGC detector from {cfg.detector_path}")
        detector = AIGCDetector(cfg.detector_path, device)
        if is_main():
            log.info(f"Detector ai_idx={detector.ai_idx}")

    # ── Dataset ──
    bert_tok_path = "bert-base-chinese"
    qwen_tok_path = cfg.qwen_tokenizer_name or cfg.qwen_model_path
    if is_main():
        log.info(f"Loading dataset from {cfg.data_path}")

    dataset = PairDataset(
        data_path=cfg.data_path,
        bert_tokenizer_path=bert_tok_path,
        qwen_tokenizer_path=qwen_tok_path,
        max_length=cfg.model_length,
        qwen_max_length=cfg.model_length,
        cache_dir=str(Path(cfg.save_dir).parent / "cache"),
    )
    bert_tokenizer = dataset.bert_tokenizer

    sampler = DistributedSampler(dataset, shuffle=True) if is_dist() else None
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Optimizer (two param groups: backbone vs new) ──
    backbone_params = []
    new_params = []
    new_param_names = {"cross_attn_adapters", "cond_proj"}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(np in name for np in new_param_names):
            new_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": float(cfg.lr)},
        {"params": new_params, "lr": float(cfg.lr_cond)},
    ], weight_decay=float(cfg.weight_decay))

    if is_main():
        log.info(f"Optimizer: backbone_lr={float(cfg.lr)}, new_lr={float(cfg.lr_cond)}")

    scheduler = get_scheduler(optimizer, cfg)

    # ── DDP wrap ──
    if is_dist():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    raw_model = model.module if is_dist() else model

    # ── Scaler for mixed precision ──
    scaler = torch.amp.GradScaler("cuda")

    # ── Training state ──
    global_step = 0
    best_det_reward = 0.0
    os.makedirs(cfg.save_dir, exist_ok=True)

    if is_main():
        log.info(f"Effective batch = {cfg.batch_size * world_size() * cfg.grad_accum_steps}")
        log.info(f"Total steps = {cfg.total_steps}")
        log.info(f"Starting training...")

    epoch = 0
    optimizer.zero_grad()
    train_start_time = time.time()

    while global_step < cfg.total_steps:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch in loader:
            if global_step >= cfg.total_steps:
                break

            ai_ids_bert = batch["ai_ids_bert"].to(device)
            hu_ids_bert = batch["hu_ids_bert"].to(device)
            ai_ids_qwen = batch["ai_ids_qwen"].to(device)
            ai_mask_qwen = batch["ai_mask_qwen"].to(device)

            B = ai_ids_bert.shape[0]

            # ── Step 1: Qwen encode (frozen, no grad) ──
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h_ai = qwen_encoder.encode(ai_ids_qwen, ai_mask_qwen)  # (B, L, 3584)

            # ── Step 2: Embed Human tokens (target) ──
            with torch.no_grad():
                x_hu = raw_model.embed_tokens(hu_ids_bert)

            # ── Step 3: Sample gamma and add noise to human embedding ──
            # Same as original LangFlow training: forward_diffusion on target
            gamma = raw_model.proposal.sample((B,), device=device).detach()
            with torch.no_grad():
                z_gamma = raw_model.forward_diffusion(x_hu, gamma)

            # ── Step 5: Self-conditioning (probabilistic) ──
            # Same logic as original LangFlow: deterministic based on step
            x_self_cond = None
            do_self_cond = (
                cfg.self_conditioning
                and ((global_step * 1000 + epoch) % 4 == 0)
            )
            if do_self_cond:
                with torch.no_grad():
                    sc_logits = raw_model.forward(
                        noisy_embeds=z_gamma, gamma=gamma,
                        cond_raw=h_ai, x_self_cond=None,
                        use_bias=(global_step >= cfg.bias_warmup_steps),
                    )
                    sc_probs = F.softmax(sc_logits.float(), dim=-1)
                    x_self_cond = raw_model.embed_tokens(sc_probs).detach()

            # ── Step 6: Forward pass + loss ──
            # Same as original LangFlow: CE loss on hu_ids given noisy input
            use_bias = global_step >= cfg.bias_warmup_steps
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(
                    noisy_embeds=z_gamma, gamma=gamma,
                    cond_raw=h_ai, x_self_cond=x_self_cond,
                    use_bias=use_bias,
                )

                # CE loss: predict human tokens from noisy human embedding + AI condition
                loss_ce = F.cross_entropy(
                    logits.float().view(-1, logits.size(-1)),
                    hu_ids_bert.view(-1),
                    ignore_index=0,  # [PAD]
                )

                loss = loss_ce

            # ── Detector reward loss (Eq. 5: L = L_CE + λ_det · P_AI) ──
            # REINFORCE estimator: argmax decoding is non-differentiable,
            # so we weight the log-probabilities of predicted tokens by
            # the detector's P(AI) score as a policy-gradient signal.
            det_loss_val = 0.0
            if detector is not None and global_step >= cfg.det_warmup_steps:
                with torch.no_grad():
                    pred_ids = logits.argmax(dim=-1)
                    pred_texts = bert_tokenizer.batch_decode(
                        pred_ids, skip_special_tokens=True
                    )
                    pred_texts = [t.replace(" ", "") for t in pred_texts]
                    p_ai = detector.score_texts(pred_texts)  # (B,)

                log_probs = F.log_softmax(logits, dim=-1)
                sel_log_probs = log_probs.gather(
                    -1, pred_ids.unsqueeze(-1)
                ).squeeze(-1)
                pad_mask = (pred_ids != 0).float()
                per_sample_lp = (
                    (sel_log_probs * pad_mask).sum(dim=-1)
                    / pad_mask.sum(dim=-1).clamp(min=1)
                )

                # REINFORCE with batch-mean baseline for variance reduction
                advantage = (p_ai - p_ai.mean()).detach()
                det_loss = (advantage * per_sample_lp).mean()
                loss = loss + cfg.det_weight * det_loss
                det_loss_val = p_ai.mean().item()

            scaled_loss = loss / cfg.grad_accum_steps
            scaler.scale(scaled_loss).backward()

            # ── Gradient accumulation ──
            if (global_step + 1) % cfg.grad_accum_steps == 0 or True:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            global_step += 1

            # ── Logging ──
            if is_main() and global_step % cfg.log_every_steps == 0:
                lr_cur = optimizer.param_groups[0]["lr"]
                lr_new = optimizer.param_groups[1]["lr"]
                elapsed = time.time() - train_start_time
                steps_per_sec = global_step / max(elapsed, 1)
                remaining = (cfg.total_steps - global_step) / max(steps_per_sec, 1e-9)
                pct = 100.0 * global_step / cfg.total_steps
                eta_h, eta_m = divmod(int(remaining), 3600)
                eta_m = eta_m // 60
                det_str = f" det_p_ai={det_loss_val:.4f}" if det_loss_val > 0 else ""
                log.info(
                    f"[{pct:5.1f}%] Step {global_step}/{cfg.total_steps} | "
                    f"ce={loss_ce.item():.4f}{det_str} | "
                    f"lr_bb={lr_cur:.2e} lr_new={lr_new:.2e} | "
                    f"gnorm={grad_norm:.3f} | "
                    f"ETA {eta_h}h{eta_m:02d}m"
                )

            # ── Detector evaluation (main only, after warmup) ──
            if (
                is_main()
                and detector is not None
                and global_step >= cfg.det_warmup_steps
                and global_step % cfg.eval_every_steps == 0
            ):
                model.eval()
                with torch.no_grad():
                    x_ai = raw_model.embed_tokens(ai_ids_bert)
                    cond_kv = raw_model.project_qwen_cond(h_ai)
                    eval_ids = raw_model.transfer(
                        x_ai_embed=x_ai,
                        cond_kv=cond_kv,
                        gamma_start=5.0,
                        num_steps=32,
                    )
                    texts = bert_tokenizer.batch_decode(eval_ids, skip_special_tokens=True)
                    texts = [t.replace(" ", "") for t in texts]

                det_reward = detector.detector_reward(texts)
                p_ai_scores = detector.score_texts(texts)
                mean_p_ai = p_ai_scores.mean().item()
                log.info(
                    f"[Detector] Step {global_step} | "
                    f"mean_P(AI)={mean_p_ai:.4f} reward={det_reward:.4f}"
                )

                if det_reward > best_det_reward:
                    best_det_reward = det_reward
                    _save_checkpoint(raw_model, optimizer, scheduler, global_step,
                                     cfg, vocab_size, cfg.save_dir, "best_det.pt")
                    log.info(f"  -> New best detector reward: {det_reward:.4f}")
                model.train()

            # ── Save checkpoint ──
            if is_main() and global_step % cfg.save_every_steps == 0:
                _save_checkpoint(
                    raw_model, optimizer, scheduler, global_step,
                    cfg, vocab_size, cfg.save_dir, f"step_{global_step}.pt",
                )

    # ── Final save ──
    if is_main():
        _save_checkpoint(
            raw_model, optimizer, scheduler, global_step,
            cfg, vocab_size, cfg.save_dir, "final.pt",
        )
        log.info(f"Training complete at step {global_step}")

    if is_dist():
        dist.destroy_process_group()


def _save_checkpoint(model, optimizer, scheduler, step, cfg, vocab_size, save_dir, name):
    path = os.path.join(save_dir, name)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "step": step,
        "vocab_size": vocab_size,
        "cfg": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
    }, path)
    log.info(f"Saved checkpoint → {path}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train StyleShield")
    parser.add_argument("--config", type=str, default="configs/styleshield.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
