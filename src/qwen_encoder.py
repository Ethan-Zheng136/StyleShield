"""Qwen Encoder Wrapper for StyleShield.

Extracts hidden states at a configurable split layer from a frozen Qwen model.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class QwenHiddenExtractor(nn.Module):
    """Wraps a frozen Qwen model to extract hidden states at layer L.

    Architecture:
      Qwen layers [0..L-1]  = encoder (extract H at layer L)
    """

    def __init__(self, qwen_model: nn.Module, split_layer: int = 14):
        super().__init__()
        self.qwen = qwen_model
        self.split_layer = split_layer

        for p in self.qwen.parameters():
            p.requires_grad = False

        self.embed_tokens = qwen_model.model.embed_tokens
        self.layers = qwen_model.model.layers
        self.rotary_emb = qwen_model.model.rotary_emb
        self.n_layers = len(self.layers)

        assert 0 < split_layer < self.n_layers, \
            f"split_layer must be in (0, {self.n_layers}), got {split_layer}"

    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run layers [0..L-1] and return hidden states at layer L.

        Args:
            input_ids:      (B, seq_len)
            attention_mask:  (B, seq_len) 1=real, 0=pad

        Returns:
            hidden: (B, seq_len, d_model)
        """
        hidden = self.embed_tokens(input_ids)

        B, S = input_ids.shape
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        if attention_mask is not None:
            position_ids = position_ids * attention_mask.long()

        position_embeddings = self.rotary_emb(hidden, position_ids)

        causal_mask = self._make_causal_mask(B, S, hidden.device, hidden.dtype)
        if attention_mask is not None:
            pad_mask = attention_mask[:, None, None, :].to(hidden.dtype)
            pad_mask = (1.0 - pad_mask) * torch.finfo(hidden.dtype).min
            causal_mask = causal_mask + pad_mask

        for i in range(self.split_layer):
            layer_out = self.layers[i](
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
            hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        return hidden

    @staticmethod
    def _make_causal_mask(B: int, S: int, device: torch.device,
                          dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((S, S), torch.finfo(dtype).min, device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S)

    @property
    def hidden_size(self) -> int:
        return self.embed_tokens.embedding_dim
