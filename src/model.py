"""StyleFlow: Conditional Flow Matching for AI-to-Human style transfer.

Builds on top of the pretrained LangFlow Chinese DiT backbone.
Key additions:
  - Cross-attention in each DiT block for Qwen semantic conditioning
  - cond_proj to project Qwen hidden (3584d) → LangFlow dim (768d)
  - flow_interpolate: linear interpolation between AI and Human embeddings
  - Pretrained weights are loaded from LangFlow checkpoint

The cross-attention layers are zero-initialized so the model starts
equivalent to the pretrained LangFlow, then gradually learns to
incorporate Qwen conditioning during fine-tuning.
"""

from __future__ import annotations

import math
import typing

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
#  JIT helpers (from LangFlow)
# ═══════════════════════════════════════════════════════════════════

@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=True)
    else:
        out = scale * F.dropout(x, p=prob, training=True)
    if residual is not None:
        out = residual + out
    return out


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    if bias is not None:
        out = scale * (x + bias)
    else:
        out = scale * x
    if residual is not None:
        out = residual + out
    return out


@torch.jit.script
def modulate_fused(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1 + scale) + shift


# ═══════════════════════════════════════════════════════════════════
#  Rotary Embedding
# ═══════════════════════════════════════════════════════════════════

class Rotary(nn.Module):
    def __init__(self, dim: int, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor, seq_dim: int = 1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            cos = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            sin = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            cos[:, :, 2, :, :].fill_(1.0)
            sin[:, :, 2, :, :].fill_(0.0)
            self.cos_cached = cos
            self.sin_cached = sin
        return self.cos_cached, self.sin_cached


def _apply_rotary_emb(x, cos, sin):
    ro_dim = cos.shape[-1] * 2
    cos = torch.cat([cos, cos], dim=-1)[None, :, None, :]
    sin = torch.cat([sin, sin], dim=-1)[None, :, None, :]
    x_rot = x[..., :ro_dim]
    x1, x2 = x_rot.chunk(2, dim=-1)
    x_rotated = torch.cat([-x2, x1], dim=-1)
    return torch.cat([x_rot * cos + x_rotated * sin, x[..., ro_dim:]], dim=-1)


def split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin):
    with torch.autocast(device_type="cuda", enabled=False):
        cos, sin = rotary_cos_sin
        cos = cos.to(qkv.dtype)
        sin = sin.to(qkv.dtype)
        cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
        sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]
        q, k, v = qkv.chunk(3, dim=2)
        q = _apply_rotary_emb(q.squeeze(dim=2), cos, sin)
        k = _apply_rotary_emb(k.squeeze(dim=2), cos, sin)
        v = v.squeeze(dim=2)
    return q, k, v


# ═══════════════════════════════════════════════════════════════════
#  LayerNorm / TimestepEmbedder
# ═══════════════════════════════════════════════════════════════════

class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_dim = freq_dim

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.freq_dim))


# ═══════════════════════════════════════════════════════════════════
#  DDiT Block with Cross-Attention
# ═══════════════════════════════════════════════════════════════════

class DDiTBlock(nn.Module):
    """Original DiT block: self-attn + FFN with adaLN-Zero."""

    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim),
        )
        self.dropout = dropout
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, rotary_cos_sin, c, cond_kv=None):
        if self.training:
            bda = bias_dropout_add_scale_fused_train
        else:
            bda = bias_dropout_add_scale_fused_inference

        x_skip = x
        x = self.norm1(x)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )
        x = modulate_fused(x, shift_msa, scale_msa)

        qkv = einops.rearrange(
            self.attn_qkv(x), "b s (three h d) -> b s three h d",
            three=3, h=self.n_heads,
        )
        q, k, v = split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin)
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=False,
        ).transpose(1, 2)
        x = einops.rearrange(x, "b s h d -> b s (h d)")

        x = bda(self.attn_out(x), None, gate_msa, x_skip, self.dropout)
        x = bda(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None, gate_mlp, x, self.dropout,
        )
        return x


class CrossAttentionAdapter(nn.Module):
    """Cross-attention from flow tokens (query) to Qwen condition (key/value).

    Zero-initialized gate so it starts as identity (no effect).
    """

    def __init__(self, dim: int, n_heads: int, cond_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.norm = LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.gate = nn.Linear(cond_dim, dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor, cond_kv: torch.Tensor,
                t_cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) flow token features
            cond_kv: (B, L_cond, D) projected Qwen hidden states
            t_cond: (B, D_cond) timestep conditioning for gate
        """
        residual = x
        x_normed = self.norm(x)

        q = self.q_proj(x_normed)
        kv = self.kv_proj(cond_kv)
        k, v = kv.chunk(2, dim=-1)

        q = q.view(q.shape[0], q.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.n_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_out = attn_out.transpose(1, 2).reshape(
            attn_out.shape[0], -1, self.n_heads * self.head_dim
        )
        attn_out = self.out_proj(attn_out)

        gate = self.gate(t_cond)[:, None, :]  # (B, 1, D)
        return residual + gate * attn_out


# ═══════════════════════════════════════════════════════════════════
#  Final Layer / Embedding (from LangFlow)
# ═══════════════════════════════════════════════════════════════════

class DDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        x = self.norm_final(x)
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(x, shift, scale)
        return self.linear(x)


def _normalize_embedding_layernorm(weight: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(weight.float(), dim=-1)
    return (normalized * math.sqrt(weight.shape[-1])).to(weight.dtype)


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim, use_normalized=True):
        super().__init__()
        self.dim = dim
        self.vocab_dim = vocab_dim
        self.use_normalized = use_normalized
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def get_weight(self):
        if self.use_normalized:
            return _normalize_embedding_layernorm(self.embedding)
        return self.embedding

    def forward(self, x):
        w = self.get_weight()
        if x.ndim == 2:
            return w[x]
        return torch.einsum("blv,ve->ble", x.float(), w.float()).to(x.dtype)


# ═══════════════════════════════════════════════════════════════════
#  Gumbel Proposal (from LangFlow)
# ═══════════════════════════════════════════════════════════════════

class GumbelProposal(nn.Module):
    def __init__(self, loc=4.723, scale=0.852, cutoff=1e-5, entropy=7.02):
        super().__init__()
        self.loc = nn.Parameter(torch.tensor(loc))
        self.scale = nn.Parameter(torch.tensor(scale))
        self.cutoff = cutoff
        self.entropy = nn.Parameter(torch.tensor(entropy))

    def _dist(self):
        return torch.distributions.Gumbel(self.loc, self.scale)

    @property
    def gamma_min(self):
        return float(self.loc - math.log(-math.log(self.cutoff)) * self.scale)

    @property
    def gamma_max(self):
        return float(self.loc - math.log(self.cutoff) * self.scale)

    def forward(self, q):
        return self._dist().icdf(q).clamp(min=self.gamma_min, max=self.gamma_max)

    def sample(self, shape, device):
        return self._dist().sample(shape).to(device).clamp(
            min=self.gamma_min, max=self.gamma_max
        )

    def cdf_value(self, gamma):
        z = (gamma - self.loc) / self.scale
        return self.entropy * torch.exp(-torch.exp(-z))


# ═══════════════════════════════════════════════════════════════════
#  StyleFlowZh: Main Model
# ═══════════════════════════════════════════════════════════════════

class StyleFlowZh(nn.Module):
    """Conditional Flow Matching for AI→Human style transfer.

    Extends pretrained LangFlow with:
      - Cross-attention adapters in each DiT block (Qwen condition)
      - cond_proj: Linear(qwen_hidden, langflow_dim)
      - flow_interpolate: x_t = (1-t)*x_ai + t*x_hu
    """

    def __init__(self, vocab_size: int, cfg):
        super().__init__()
        self.cfg = cfg
        dim = cfg.hidden_size

        self.vocab_embed = EmbeddingLayer(dim, vocab_size, cfg.use_normalized_embedding)
        self.sigma_map = TimestepEmbedder(cfg.cond_dim)
        self.rotary_emb = Rotary(dim // cfg.n_heads)

        self.blocks = nn.ModuleList([
            DDiTBlock(dim, cfg.n_heads, cfg.cond_dim, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.n_blocks)
        ])

        self.cross_attn_adapters = nn.ModuleList([
            CrossAttentionAdapter(dim, cfg.n_heads, cfg.cond_dim)
            for _ in range(cfg.n_blocks)
        ])

        self.output_layer = DDiTFinalLayer(dim, vocab_size, cfg.cond_dim)

        if cfg.self_conditioning:
            self.self_cond_proj = nn.Linear(dim * 2, dim, bias=False)
            nn.init.zeros_(self.self_cond_proj.weight)

        self.proposal = GumbelProposal(
            cfg.gumbel_loc, cfg.gumbel_scale,
            cfg.gumbel_cutoff, cfg.gumbel_entropy,
        )

        self.cond_proj = nn.Linear(cfg.qwen_hidden_size, dim)

    def get_embedding_matrix(self):
        return self.vocab_embed.get_weight()

    def embed_tokens(self, x):
        return self.vocab_embed(x)

    def flow_interpolate(
        self, x_ai: torch.Tensor, x_hu: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditional OT interpolation: x_t = (1-t)*x_ai + t*x_hu.

        Returns:
            x_t: interpolated state
            v_target: target velocity (x_hu - x_ai)
        """
        t_expand = t[:, None, None]
        x_t = (1.0 - t_expand) * x_ai + t_expand * x_hu
        v_target = x_hu - x_ai
        return x_t, v_target

    def project_qwen_cond(self, h_qwen: torch.Tensor) -> torch.Tensor:
        """Project Qwen hidden states to LangFlow dimension."""
        return self.cond_proj(h_qwen.float())

    def backbone_forward(
        self,
        z: torch.Tensor,
        sigma: torch.Tensor,
        cond_kv: torch.Tensor | None = None,
        x_self_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """DiT backbone with cross-attention: noisy embeddings + γ + cond → logits."""
        x = z

        if self.cfg.self_conditioning:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = x + self.self_cond_proj(torch.cat([x, x_self_cond], dim=-1))

        t_cond = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb(x)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for block, cross_attn in zip(self.blocks, self.cross_attn_adapters):
                x = block(x, rotary_cos_sin, c=t_cond)
                if cond_kv is not None:
                    x = cross_attn(x, cond_kv, t_cond)
            logits = self.output_layer(x, c=t_cond)

        return logits.float()

    def forward(
        self,
        noisy_embeds: torch.Tensor,
        gamma: torch.Tensor,
        cond_kv: torch.Tensor | None = None,
        cond_raw: torch.Tensor | None = None,
        x_self_cond: torch.Tensor | None = None,
        use_bias: bool = True,
    ) -> torch.Tensor:
        """Full forward: embeddings + γ + condition → logits (+ bias skip).

        Args:
            cond_kv: pre-projected Qwen hidden (B, L, D). Used if provided.
            cond_raw: raw Qwen hidden (B, L, 3584). Projected inside forward
                      so that cond_proj participates in DDP's forward graph.
        """
        if cond_kv is None and cond_raw is not None:
            cond_kv = self.project_qwen_cond(cond_raw)

        sigma = gamma
        if sigma.ndim == 2:
            sigma = sigma.mean(-1)

        logits = self.backbone_forward(noisy_embeds, sigma, cond_kv, x_self_cond)

        if use_bias:
            c_skip = ((F.softplus(-sigma) - sigma) / 2).exp()
            emb_mat = self.get_embedding_matrix()
            skip_logits = torch.matmul(noisy_embeds.float(), emb_mat.t().float())
            logits = logits + c_skip[:, None, None] * skip_logits.to(logits.dtype)

        return logits

    def forward_diffusion(
        self, x_embed: torch.Tensor, gamma: torch.Tensor
    ) -> torch.Tensor:
        """Add noise at level gamma: z_γ = α·x + σ·ε.

        Copied from AIGC_FUCK_6/src/model.py (LangFlowZh.forward_diffusion).
        """
        gamma = gamma.float()
        alpha = torch.sigmoid(-gamma).sqrt()[:, None, None]
        sigma = torch.sigmoid(gamma).sqrt()[:, None, None]
        noise = torch.randn_like(x_embed)
        return (x_embed * alpha + noise * sigma).to(x_embed.dtype)

    def euler_edm_step(
        self, z: torch.Tensor, x_pred: torch.Tensor,
        t: torch.Tensor, s: torch.Tensor,
    ) -> torch.Tensor:
        """EDM Euler step from gamma=t to gamma=s.

        Copied from AIGC_FUCK_6/src/model.py (LangFlowZh.euler_edm_step).
        """
        t_ = t.double()
        s_ = s.double()
        cur = z.double() * ((F.softplus(t_) - F.softplus(s_)) / 2).exp()
        end = torch.sigmoid(-s_).sqrt() * x_pred.double()
        return end.lerp(cur, ((s_ - t_) / 2).exp()).to(z.dtype)

    @torch.no_grad()
    def transfer(
        self,
        x_ai_embed: torch.Tensor,
        cond_kv: torch.Tensor,
        gamma_start: float = 5.0,
        num_steps: int = 64,
    ) -> torch.LongTensor:
        """SDEdit-style transfer: AI embedding → add noise → denoise with Qwen condition.

        Directly follows AIGC_FUCK_6/scripts/generate.py sdedit_transfer logic.
        Only difference: forward() receives cond_kv for cross-attention.

        Args:
            x_ai_embed: (B, L, D) embedded AI tokens
            cond_kv: (B, L_cond, D) projected Qwen condition
            gamma_start: noise level (higher=more change). Default 5.0.
            num_steps: number of denoising steps
        """
        B = x_ai_embed.shape[0]
        device = x_ai_embed.device

        gamma_s = torch.tensor(gamma_start, device=device)
        z = self.forward_diffusion(x_ai_embed, gamma_s.unsqueeze(0).expand(B))

        gamma_end = self.proposal.gamma_min
        gamma_schedule = torch.linspace(
            gamma_start, gamma_end, num_steps, device=device,
        )

        x_self_cond = None

        for i in range(len(gamma_schedule) - 1):
            g_t = gamma_schedule[i]
            g_s = gamma_schedule[i + 1]
            g_exp = g_t.unsqueeze(0).expand(B)

            logits = self.forward(
                noisy_embeds=z, gamma=g_exp, cond_kv=cond_kv,
                x_self_cond=x_self_cond, use_bias=True,
            )
            probs = F.softmax(logits.float(), dim=-1)
            x_pred = self.embed_tokens(probs)

            if self.cfg.self_conditioning:
                x_self_cond = x_pred

            z = self.euler_edm_step(z, x_pred, g_t, g_s)

        g_final = gamma_schedule[-1].unsqueeze(0).expand(B)
        final_logits = self.forward(
            noisy_embeds=z, gamma=g_final, cond_kv=cond_kv,
            x_self_cond=x_self_cond, use_bias=True,
        )
        return final_logits.argmax(dim=-1)

    @classmethod
    def from_langflow_ckpt(cls, ckpt_path: str, cfg, device: torch.device = None):
        """Load pretrained LangFlow weights into StyleFlowZh.

        Cross-attention adapters and cond_proj are initialized fresh
        (zero-init for adapters, random for cond_proj).
        """
        if device is None:
            device = torch.device("cpu")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        vocab_size = ckpt["vocab_size"]

        for k, v in ckpt.get("cfg", {}).items():
            if hasattr(cfg, k) and k not in (
                "qwen_model_path", "qwen_hidden_size", "qwen_split_layer",
                "detector_path", "data_path", "save_dir", "langflow_ckpt",
                "lr", "lr_cond", "total_steps", "batch_size",
            ):
                setattr(cfg, k, v)

        model = cls(vocab_size, cfg).to(device)

        langflow_state = ckpt["model_state"]
        model_state = model.state_dict()

        loaded, skipped = 0, 0
        for k, v in langflow_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
            else:
                skipped += 1

        model.load_state_dict(model_state)
        print(f"[StyleFlow] Loaded {loaded} params from LangFlow, "
              f"skipped {skipped}, new params: "
              f"{len(model_state) - loaded}")

        return model, vocab_size, cfg
