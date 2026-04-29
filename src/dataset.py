"""PairDataset: loads AI/Human paired text for conditional flow matching.

Each sample returns:
  - ai_ids_bert, ai_mask_bert   (LangFlow tokenizer, for embedding)
  - hu_ids_bert, hu_mask_bert   (LangFlow tokenizer, for embedding)
  - ai_ids_qwen, ai_mask_qwen   (Qwen tokenizer, for condition)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PairDataset(Dataset):
    """Paired AI/Human text dataset with dual tokenization."""

    def __init__(
        self,
        data_path: str,
        bert_tokenizer_path: str,
        qwen_tokenizer_path: str,
        max_length: int = 512,
        qwen_max_length: int = 512,
        cache_dir: str | None = None,
    ):
        super().__init__()
        self.max_length = max_length
        self.qwen_max_length = qwen_max_length

        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            bert_tokenizer_path, trust_remote_code=True,
        )
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            qwen_tokenizer_path, trust_remote_code=True,
        )
        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token

        cache_path = self._resolve_cache(data_path, cache_dir)
        if cache_path and cache_path.exists():
            self._load_cache(cache_path)
        else:
            self._load_raw(data_path)
            if cache_path:
                self._save_cache(cache_path)

    def _resolve_cache(
        self, data_path: str, cache_dir: str | None
    ) -> Path | None:
        if cache_dir is None:
            return None
        os.makedirs(cache_dir, exist_ok=True)
        base = Path(data_path).stem
        return Path(cache_dir) / f"{base}_L{self.max_length}_QL{self.qwen_max_length}.pt"

    def _load_raw(self, data_path: str):
        records = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                ai_text = obj.get("ai_text", "").strip()
                hu_text = obj.get("human_text", "").strip()
                if not ai_text or not hu_text:
                    continue
                records.append((ai_text, hu_text))

        self.ai_texts = [r[0] for r in records]
        self.hu_texts = [r[1] for r in records]

        print(f"[PairDataset] Loaded {len(records)} pairs from {data_path}")
        self._tokenize_all()

    def _tokenize_all(self):
        print("[PairDataset] Tokenizing AI texts (BERT)...")
        ai_bert = self.bert_tokenizer(
            self.ai_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        print("[PairDataset] Tokenizing Human texts (BERT)...")
        hu_bert = self.bert_tokenizer(
            self.hu_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        print("[PairDataset] Tokenizing AI texts (Qwen)...")
        ai_qwen = self.qwen_tokenizer(
            self.ai_texts,
            max_length=self.qwen_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        self.ai_ids_bert = ai_bert["input_ids"]
        self.ai_mask_bert = ai_bert["attention_mask"]
        self.hu_ids_bert = hu_bert["input_ids"]
        self.hu_mask_bert = hu_bert["attention_mask"]
        self.ai_ids_qwen = ai_qwen["input_ids"]
        self.ai_mask_qwen = ai_qwen["attention_mask"]

    def _save_cache(self, path: Path):
        print(f"[PairDataset] Saving cache to {path}")
        torch.save({
            "ai_ids_bert": self.ai_ids_bert,
            "ai_mask_bert": self.ai_mask_bert,
            "hu_ids_bert": self.hu_ids_bert,
            "hu_mask_bert": self.hu_mask_bert,
            "ai_ids_qwen": self.ai_ids_qwen,
            "ai_mask_qwen": self.ai_mask_qwen,
        }, path)

    def _load_cache(self, path: Path):
        print(f"[PairDataset] Loading from cache {path}")
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.ai_ids_bert = data["ai_ids_bert"]
        self.ai_mask_bert = data["ai_mask_bert"]
        self.hu_ids_bert = data["hu_ids_bert"]
        self.hu_mask_bert = data["hu_mask_bert"]
        self.ai_ids_qwen = data["ai_ids_qwen"]
        self.ai_mask_qwen = data["ai_mask_qwen"]
        print(f"[PairDataset] Cache loaded: {len(self)} pairs")

    def __len__(self):
        return len(self.ai_ids_bert)

    def __getitem__(self, idx):
        return {
            "ai_ids_bert": self.ai_ids_bert[idx],
            "ai_mask_bert": self.ai_mask_bert[idx],
            "hu_ids_bert": self.hu_ids_bert[idx],
            "hu_mask_bert": self.hu_mask_bert[idx],
            "ai_ids_qwen": self.ai_ids_qwen[idx],
            "ai_mask_qwen": self.ai_mask_qwen[idx],
        }

    @property
    def bert_vocab_size(self) -> int:
        return self.bert_tokenizer.vocab_size
