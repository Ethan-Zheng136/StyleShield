"""Configuration for StyleShield.

Extends LangFlow config with Qwen conditioning and conditional flow matching params.
"""

from dataclasses import dataclass


@dataclass
class StyleShieldConfig:
    # ── LangFlow backbone (pretrained) ──
    tokenizer_name: str = "bert-base-chinese"
    hidden_size: int = 768
    cond_dim: int = 128
    n_blocks: int = 12
    n_heads: int = 12
    dropout: float = 0.1
    model_length: int = 512
    mlp_ratio: int = 4
    use_normalized_embedding: bool = True
    self_conditioning: bool = True
    self_cond_prob: float = 0.25
    use_bias: bool = True
    bias_warmup_steps: int = 0

    gumbel_loc: float = 4.723
    gumbel_scale: float = 0.852
    gumbel_cutoff: float = 1e-5
    gumbel_entropy: float = 7.02

    # ── Qwen conditioning ──
    qwen_model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    qwen_hidden_size: int = 3584
    qwen_split_layer: int = 14
    qwen_tokenizer_name: str = ""  # defaults to qwen_model_path

    # ── Conditional flow matching ──
    # t sampling distribution: "uniform" or "logit_normal"
    t_sampling: str = "logit_normal"
    t_logit_mean: float = 0.0
    t_logit_std: float = 1.0

    # ── Loss weights ──
    fm_weight: float = 1.0
    ce_weight: float = 0.5
    det_weight: float = 0.1
    det_warmup_steps: int = 5000

    # ── Detector ──
    detector_path: str = "models/AIGC_detector_zhv3"

    # ── Training ──
    lr: float = 1e-4
    lr_cond: float = 5e-4  # higher LR for new cross-attn/cond_proj params
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    total_steps: int = 100_000
    batch_size: int = 8
    grad_accum_steps: int = 2
    max_grad_norm: float = 1.0

    # ── Data ──
    data_path: str = "data/train_pairs.jsonl"
    num_workers: int = 4

    # ── Checkpointing ──
    save_dir: str = "./experiments/styleflow/checkpoints"
    save_every_steps: int = 5_000
    eval_every_steps: int = 2_500
    log_every_steps: int = 50

    # ── Pretrained LangFlow checkpoint ──
    langflow_ckpt: str = ""
