"""Tests for StyleShield model, dataset, and loss computation.

Run:
    pytest tests/test_styleshield.py -v
"""

import json
import math
import os
import sys
import tempfile

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import StyleShieldConfig
from src.model import (
    CrossAttentionAdapter,
    DDiTBlock,
    DDiTFinalLayer,
    EmbeddingLayer,
    GumbelProposal,
    StyleShieldModel,
    TimestepEmbedder,
)


# ═══════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def cfg():
    c = StyleShieldConfig()
    c.hidden_size = 64
    c.cond_dim = 32
    c.n_blocks = 2
    c.n_heads = 4
    c.dropout = 0.0
    c.model_length = 16
    c.mlp_ratio = 2
    c.use_normalized_embedding = True
    c.self_conditioning = True
    c.self_cond_prob = 0.5
    c.use_bias = True
    c.qwen_hidden_size = 128
    return c


@pytest.fixture
def vocab_size():
    return 100


@pytest.fixture
def model(cfg, vocab_size):
    return StyleShieldModel(vocab_size, cfg)


@pytest.fixture
def device():
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════
#  Building Blocks Tests
# ═══════════════════════════════════════════════════════════════════

class TestEmbeddingLayer:
    def test_forward_ids(self, cfg, vocab_size):
        emb = EmbeddingLayer(cfg.hidden_size, vocab_size, use_normalized=True)
        ids = torch.randint(0, vocab_size, (2, 8))
        out = emb(ids)
        assert out.shape == (2, 8, cfg.hidden_size)

    def test_forward_probs(self, cfg, vocab_size):
        emb = EmbeddingLayer(cfg.hidden_size, vocab_size, use_normalized=True)
        probs = torch.randn(2, 8, vocab_size).softmax(dim=-1)
        out = emb(probs)
        assert out.shape == (2, 8, cfg.hidden_size)

    def test_normalized_unit_norm(self, cfg, vocab_size):
        emb = EmbeddingLayer(cfg.hidden_size, vocab_size, use_normalized=True)
        w = emb.get_weight()
        norms = w.norm(dim=-1)
        expected_norm = math.sqrt(cfg.hidden_size)
        assert torch.allclose(norms, torch.full_like(norms, expected_norm), atol=1e-4)


class TestTimestepEmbedder:
    def test_output_shape(self, cfg):
        te = TimestepEmbedder(cfg.cond_dim)
        t = torch.rand(4)
        out = te(t)
        assert out.shape == (4, cfg.cond_dim)


class TestGumbelProposal:
    def test_sample(self):
        gp = GumbelProposal()
        samples = gp.sample((10,), torch.device("cpu"))
        assert samples.shape == (10,)
        assert (samples >= gp.gamma_min).all()
        assert (samples <= gp.gamma_max).all()

    def test_forward_monotone(self):
        gp = GumbelProposal()
        q = torch.linspace(0.01, 0.99, 50)
        gamma = gp(q)
        assert (gamma[1:] >= gamma[:-1]).all(), "icdf should be monotonically non-decreasing"


class TestDDiTBlock:
    def test_forward(self, cfg):
        block = DDiTBlock(cfg.hidden_size, cfg.n_heads, cfg.cond_dim,
                          cfg.mlp_ratio, cfg.dropout)
        from src.model import Rotary
        rot = Rotary(cfg.hidden_size // cfg.n_heads)
        x = torch.randn(2, 8, cfg.hidden_size)
        cos_sin = rot(x)
        c = torch.randn(2, cfg.cond_dim)
        out = block(x, cos_sin, c)
        assert out.shape == x.shape


class TestCrossAttentionAdapter:
    def test_zero_init_identity(self, cfg):
        """At initialization, cross-attention should be identity (gate=0)."""
        ca = CrossAttentionAdapter(cfg.hidden_size, cfg.n_heads, cfg.cond_dim)
        x = torch.randn(2, 8, cfg.hidden_size)
        cond = torch.randn(2, 12, cfg.hidden_size)
        t_cond = torch.randn(2, cfg.cond_dim)
        out = ca(x, cond, t_cond)
        assert torch.allclose(out, x, atol=1e-5), \
            "Zero-init gate should make cross-attn a no-op"

    def test_output_shape(self, cfg):
        ca = CrossAttentionAdapter(cfg.hidden_size, cfg.n_heads, cfg.cond_dim)
        x = torch.randn(2, 8, cfg.hidden_size)
        cond = torch.randn(2, 12, cfg.hidden_size)
        t_cond = torch.randn(2, cfg.cond_dim)
        out = ca(x, cond, t_cond)
        assert out.shape == x.shape

    def test_different_cond_lengths(self, cfg):
        """Cross-attention should handle different condition sequence lengths."""
        ca = CrossAttentionAdapter(cfg.hidden_size, cfg.n_heads, cfg.cond_dim)
        nn = torch.nn
        nn.init.normal_(ca.gate.weight)
        nn.init.normal_(ca.gate.bias)

        x = torch.randn(2, 8, cfg.hidden_size)
        t_cond = torch.randn(2, cfg.cond_dim)

        for cond_len in [4, 8, 16, 32]:
            cond = torch.randn(2, cond_len, cfg.hidden_size)
            out = ca(x, cond, t_cond)
            assert out.shape == x.shape


# ═══════════════════════════════════════════════════════════════════
#  StyleShieldModel Tests
# ═══════════════════════════════════════════════════════════════════

class TestStyleShieldModel:
    def test_init(self, model, cfg, vocab_size):
        assert len(model.blocks) == cfg.n_blocks
        assert len(model.cross_attn_adapters) == cfg.n_blocks
        assert model.cond_proj.in_features == cfg.qwen_hidden_size
        assert model.cond_proj.out_features == cfg.hidden_size

    def test_embed_tokens(self, model, vocab_size, cfg):
        ids = torch.randint(0, vocab_size, (2, 8))
        emb = model.embed_tokens(ids)
        assert emb.shape == (2, 8, cfg.hidden_size)

    def test_flow_interpolate(self, model, cfg):
        B, L, D = 3, 8, cfg.hidden_size
        x_ai = torch.randn(B, L, D)
        x_hu = torch.randn(B, L, D)
        t = torch.tensor([0.0, 0.5, 1.0])

        x_t, v_target = model.flow_interpolate(x_ai, x_hu, t)
        assert x_t.shape == (B, L, D)
        assert v_target.shape == (B, L, D)

        assert torch.allclose(x_t[0], x_ai[0], atol=1e-5), "t=0 should give x_ai"
        assert torch.allclose(x_t[2], x_hu[2], atol=1e-5), "t=1 should give x_hu"
        mid = 0.5 * x_ai[1] + 0.5 * x_hu[1]
        assert torch.allclose(x_t[1], mid, atol=1e-5), "t=0.5 should be midpoint"

        expected_v = x_hu - x_ai
        assert torch.allclose(v_target, expected_v, atol=1e-5)

    def test_project_qwen_cond(self, model, cfg):
        h_qwen = torch.randn(2, 10, cfg.qwen_hidden_size)
        cond = model.project_qwen_cond(h_qwen)
        assert cond.shape == (2, 10, cfg.hidden_size)

    def test_backbone_forward_no_cond(self, model, cfg, vocab_size):
        """Forward without conditioning (should work like vanilla LangFlow)."""
        z = torch.randn(2, 8, cfg.hidden_size)
        sigma = torch.rand(2)
        logits = model.backbone_forward(z, sigma, cond_kv=None)
        assert logits.shape == (2, 8, vocab_size)

    def test_backbone_forward_with_cond(self, model, cfg, vocab_size):
        """Forward with Qwen conditioning."""
        z = torch.randn(2, 8, cfg.hidden_size)
        sigma = torch.rand(2)
        cond = torch.randn(2, 10, cfg.hidden_size)
        logits = model.backbone_forward(z, sigma, cond_kv=cond)
        assert logits.shape == (2, 8, vocab_size)

    def test_forward_full(self, model, cfg, vocab_size):
        """Full forward with bias skip."""
        z = torch.randn(2, 8, cfg.hidden_size)
        gamma = torch.rand(2)
        cond = torch.randn(2, 10, cfg.hidden_size)
        logits = model.forward(noisy_embeds=z, gamma=gamma, cond_kv=cond, use_bias=True)
        assert logits.shape == (2, 8, vocab_size)

    def test_zero_init_cross_attn_equals_no_cond(self, cfg, vocab_size):
        """At init, model with cond should produce same output as without cond
        (because cross-attn gates are zero-initialized)."""
        model = StyleShieldModel(vocab_size, cfg)
        model.eval()

        z = torch.randn(1, 8, cfg.hidden_size)
        sigma = torch.tensor([0.5])

        with torch.no_grad():
            logits_no_cond = model.backbone_forward(z, sigma, cond_kv=None)
            cond = torch.randn(1, 10, cfg.hidden_size)
            logits_with_cond = model.backbone_forward(z, sigma, cond_kv=cond)

        assert torch.allclose(logits_no_cond, logits_with_cond, atol=1e-4), \
            "Zero-init cross-attn should not change output"

    def test_backward(self, model, cfg, vocab_size):
        """Loss backward should flow gradients to cross-attn and cond_proj."""
        z = torch.randn(2, 8, cfg.hidden_size, requires_grad=True)
        gamma = torch.rand(2)
        h_qwen = torch.randn(2, 8, cfg.qwen_hidden_size)
        cond = model.project_qwen_cond(h_qwen)

        logits = model.forward(noisy_embeds=z, gamma=gamma, cond_kv=cond, use_bias=True)
        loss = logits.sum()
        loss.backward()

        assert model.cond_proj.weight.grad is not None
        for i, ca in enumerate(model.cross_attn_adapters):
            assert ca.q_proj.weight.grad is not None, \
                f"cross_attn_adapters[{i}].q_proj should have gradient"


# ═══════════════════════════════════════════════════════════════════
#  Loss Computation Tests
# ═══════════════════════════════════════════════════════════════════

class TestLossComputation:
    def test_fm_loss(self, model, cfg, vocab_size):
        """FM velocity MSE loss should be computable and finite."""
        B, L, D = 2, 8, cfg.hidden_size
        ai_ids = torch.randint(0, vocab_size, (B, L))
        hu_ids = torch.randint(0, vocab_size, (B, L))

        x_ai = model.embed_tokens(ai_ids)
        x_hu = model.embed_tokens(hu_ids)
        t = torch.tensor([0.3, 0.7])
        x_t, v_target = model.flow_interpolate(x_ai, x_hu, t)
        gamma = model.proposal(t)

        cond = torch.randn(B, L, cfg.hidden_size)
        logits = model.forward(noisy_embeds=x_t, gamma=gamma, cond_kv=cond, use_bias=True)
        probs = F.softmax(logits.float(), dim=-1)
        x_pred = model.embed_tokens(probs)
        v_pred = x_pred - x_t

        loss_fm = F.mse_loss(v_pred, v_target.detach())
        assert loss_fm.isfinite(), "FM loss should be finite"
        assert loss_fm.item() > 0, "FM loss should be positive"

    def test_ce_loss(self, model, cfg, vocab_size):
        """CE token prediction loss."""
        B, L = 2, 8
        hu_ids = torch.randint(1, vocab_size, (B, L))
        logits = torch.randn(B, L, vocab_size)
        loss_ce = F.cross_entropy(logits.view(-1, vocab_size), hu_ids.view(-1))
        assert loss_ce.isfinite()

    def test_combined_loss_backward(self, model, cfg, vocab_size):
        """Combined FM + CE loss should backprop cleanly."""
        B, L, D = 2, 8, cfg.hidden_size
        ai_ids = torch.randint(0, vocab_size, (B, L))
        hu_ids = torch.randint(1, vocab_size, (B, L))

        x_ai = model.embed_tokens(ai_ids)
        x_hu = model.embed_tokens(hu_ids)
        t = torch.rand(B).clamp(0.01, 0.99)
        x_t, v_target = model.flow_interpolate(x_ai, x_hu, t)
        gamma = model.proposal(t)

        cond = torch.randn(B, L, cfg.hidden_size)
        logits = model.forward(noisy_embeds=x_t, gamma=gamma, cond_kv=cond, use_bias=True)

        probs = F.softmax(logits.float(), dim=-1)
        x_pred = model.embed_tokens(probs)
        v_pred = x_pred - x_t
        loss_fm = F.mse_loss(v_pred, v_target.detach())

        loss_ce = F.cross_entropy(
            logits.float().view(-1, vocab_size), hu_ids.view(-1)
        )

        loss = cfg.fm_weight * loss_fm + cfg.ce_weight * loss_ce
        loss.backward()

        grad_count = sum(
            1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0, "Should have nonzero gradients"


# ═══════════════════════════════════════════════════════════════════
#  Transfer (inference) Test
# ═══════════════════════════════════════════════════════════════════

class TestForwardDiffusion:
    def test_adds_noise(self, model, cfg):
        x = torch.randn(2, 8, cfg.hidden_size)
        gamma = torch.tensor([4.0, 6.0])
        z = model.forward_diffusion(x, gamma)
        assert z.shape == x.shape
        assert (z - x).abs().mean() > 0.01

    def test_more_noise_at_higher_gamma(self, model, cfg):
        x = torch.randn(2, 8, cfg.hidden_size)
        z_low = model.forward_diffusion(x, torch.tensor([3.0, 3.0]))
        z_high = model.forward_diffusion(x, torch.tensor([7.0, 7.0]))
        diff_low = (z_low - x).abs().mean()
        diff_high = (z_high - x).abs().mean()
        assert diff_high > diff_low


class TestEulerEdmStep:
    def test_output_shape(self, model, cfg):
        z = torch.randn(2, 8, cfg.hidden_size)
        x_pred = torch.randn(2, 8, cfg.hidden_size)
        g_t = torch.tensor(5.0)
        g_s = torch.tensor(4.5)
        z_next = model.euler_edm_step(z, x_pred, g_t, g_s)
        assert z_next.shape == z.shape

    def test_moves_toward_pred(self, model, cfg):
        z = torch.randn(2, 8, cfg.hidden_size)
        x_pred = torch.randn(2, 8, cfg.hidden_size)
        g_t = torch.tensor(5.0)
        g_s = torch.tensor(4.5)
        z_next = model.euler_edm_step(z, x_pred, g_t, g_s)
        assert (z_next - z).abs().mean() > 1e-6


class TestTransfer:
    def test_transfer_produces_token_ids(self, model, cfg, vocab_size):
        """Transfer should output valid token IDs."""
        model.eval()
        B, L, D = 1, 8, cfg.hidden_size
        x_ai = torch.randn(B, L, D)
        cond = torch.randn(B, 10, D)

        output_ids = model.transfer(x_ai, cond, gamma_start=5.0, num_steps=4)
        assert output_ids.shape == (B, L)
        assert output_ids.dtype == torch.int64
        assert (output_ids >= 0).all()
        assert (output_ids < vocab_size).all()

    def test_transfer_different_gamma(self, model, cfg, vocab_size):
        """Different gamma should generally produce different outputs."""
        model.eval()
        B, L, D = 1, 8, cfg.hidden_size
        x_ai = torch.randn(B, L, D)
        cond = torch.randn(B, 10, D)

        out_low = model.transfer(x_ai, cond, gamma_start=3.0, num_steps=4)
        out_high = model.transfer(x_ai, cond, gamma_start=7.0, num_steps=4)
        assert out_low.shape == out_high.shape


# ═══════════════════════════════════════════════════════════════════
#  PairDataset Test (with synthetic data)
# ═══════════════════════════════════════════════════════════════════

class TestPairDataset:
    @pytest.fixture
    def synthetic_data(self, tmp_path):
        data_path = str(tmp_path / "pairs.jsonl")
        records = [
            {"ai_text": f"这是人工智能生成的第{i}条文本。", "human_text": f"这是人类写的第{i}篇文章。"}
            for i in range(10)
        ]
        with open(data_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return data_path

    def test_load(self, synthetic_data):
        bert_tok = "bert-base-chinese"

        qwen_tok = "models/Qwen2.5-7B-Instruct"
        if not os.path.exists(qwen_tok):
            pytest.skip("Qwen tokenizer not available")

        from src.dataset import PairDataset
        ds = PairDataset(
            data_path=synthetic_data,
            bert_tokenizer_path=bert_tok,
            qwen_tokenizer_path=qwen_tok,
            max_length=32,
            qwen_max_length=32,
        )
        assert len(ds) == 10

        sample = ds[0]
        assert "ai_ids_bert" in sample
        assert "hu_ids_bert" in sample
        assert "ai_ids_qwen" in sample
        assert sample["ai_ids_bert"].shape == (32,)
        assert sample["hu_ids_bert"].shape == (32,)
        assert sample["ai_ids_qwen"].shape == (32,)


# ═══════════════════════════════════════════════════════════════════
#  Config Test
# ═══════════════════════════════════════════════════════════════════

class TestConfig:
    def test_defaults(self):
        cfg = StyleShieldConfig()
        assert cfg.hidden_size == 768
        assert cfg.qwen_hidden_size == 3584
        assert cfg.n_blocks == 12
        assert cfg.fm_weight == 1.0
        assert cfg.ce_weight == 0.5
        assert cfg.det_weight == 0.1
        assert cfg.det_warmup_steps == 5000
