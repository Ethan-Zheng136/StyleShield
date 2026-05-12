"""Verify all 4 AIGC detectors load and produce valid output."""
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_text = "人工智能技术的快速发展对社会产生了深远的影响。"

detectors = [
    ("zhv3", "models/AIGC_detector_zhv3"),
    ("zhv2", "models/AIGC_detector_zhv2"),
    ("anx-bert", "models/chinese-ai-detector-bert"),
    ("gpt2-det", "models/AITextDetector"),
]

for name, path in detectors:
    print(f"--- {name} ---", flush=True)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        path, trust_remote_code=True
    ).to(device).eval()
    labels = model.config.id2label
    print(f"  Labels: {labels}", flush=True)
    enc = tok(test_text, max_length=512, truncation=True,
              padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = F.softmax(logits.float(), dim=-1)
    print(f"  Probs: {[round(p, 4) for p in probs[0].tolist()]}", flush=True)

    ai_idx = None
    for idx, lbl in labels.items():
        lbl_lower = str(lbl).lower()
        if lbl_lower in ("ai", "ai-generated") or "machine" in lbl_lower:
            ai_idx = int(idx)
    if ai_idx is None:
        ai_idx = 1
    print(f"  AI idx={ai_idx}, P(AI)={probs[0, ai_idx].item():.4f}\n", flush=True)

print("ALL DONE", flush=True)
