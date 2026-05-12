"""Upload StyleShield checkpoints and datasets to HuggingFace Hub.

Usage:
    # Login first:
    python -c "from huggingface_hub import login; login()"
    
    # Then upload:
    python scripts/upload_to_hf.py --mode checkpoints
    python scripts/upload_to_hf.py --mode dataset
    python scripts/upload_to_hf.py --mode all
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_USERNAME = "Ethan-Zheng136"

MODEL_REPO = f"{HF_USERNAME}/StyleShield"
DATASET_REPO = f"{HF_USERNAME}/StyleShield-data"

CHECKPOINTS = {
    "checkpoints/styleflow_v2_step30000.pt": 
        "./experiments/styleflow_v2/checkpoints/step_30000.pt",
    "checkpoints/styleflow_v1_step30000.pt": 
        "./experiments/styleflow/checkpoints/step_30000.pt",
    "checkpoints/ablation_no_detector_step5000.pt": 
        "./experiments/ablation_no_detector/checkpoints/step_5000.pt",
    "checkpoints/ablation_split7_step15000.pt": 
        "./experiments/ablation_split7/checkpoints/step_15000.pt",
    "checkpoints/ablation_split21_step5000.pt": 
        "./experiments/ablation_split21/checkpoints/step_5000.pt",
    "checkpoints/langflow_pretrain_step460000.pt": 
        "./experiments/v6/checkpoints/step460000_ckpt.pt",
}

DATASETS = {
    "dataset_multidomain.jsonl": 
        "data/dataset_multidomain.jsonl",
    "dataset_zhihu_pairs_full.jsonl": 
        "data/dataset_zhihu_pairs_full.jsonl",
    "test_set_1000.jsonl": 
        "data/test_set_1000.jsonl",
}


def upload_checkpoints(api: HfApi):
    print(f"\n{'='*60}")
    print(f"  Uploading checkpoints to {MODEL_REPO}")
    print(f"{'='*60}")
    
    create_repo(MODEL_REPO, repo_type="model", private=True, exist_ok=True)
    
    # Upload README as model card
    model_card = """---
license: mit
tags:
  - text-style-transfer
  - aigc-detection
  - flow-matching
  - diffusion-transformer
language:
  - zh
---

# StyleShield Checkpoints

Model checkpoints for **StyleShield: Continuous and Controllable Style Transfer for Evading AI-Generated Content Detectors**.

## Files

| File | Description | Best γ |
|------|-------------|--------|
| `styleflow_v2_step30000.pt` | **Main model** (multi-domain, layer 14) | 6.5-7.0 |
| `styleflow_v1_step30000.pt` | Single-domain model (social media only) | 6.5-7.0 |
| `ablation_no_detector_step5000.pt` | A1: w/o detector reward | 6.5-7.0 |
| `ablation_split7_step15000.pt` | A3: Qwen split layer 7 | 5.5-6.0 |
| `ablation_split21_step5000.pt` | A4: Qwen split layer 21 | 6.5-7.0 |
| `langflow_pretrain_step460000.pt` | LangFlow pretrained backbone | -- |

## Usage

```python
from src.model import StyleShieldModel
from src.config import StyleShieldConfig

cfg = StyleShieldConfig()
model, vocab_size, cfg = StyleShieldModel.from_langflow_ckpt("langflow_pretrain_step460000.pt", cfg)

# Load fine-tuned weights
ckpt = torch.load("styleflow_v2_step30000.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state"])
```
"""
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=MODEL_REPO,
        repo_type="model",
    )
    print("  Uploaded model card")
    
    for hf_path, local_path in CHECKPOINTS.items():
        if not os.path.exists(local_path):
            print(f"  SKIP (not found): {local_path}")
            continue
        size_gb = os.path.getsize(local_path) / (1024**3)
        print(f"  Uploading {hf_path} ({size_gb:.1f} GB)...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=hf_path,
            repo_id=MODEL_REPO,
            repo_type="model",
        )
        print(f"    Done: {hf_path}")
    
    print(f"\n  Model repo: https://huggingface.co/{MODEL_REPO}")


def upload_dataset(api: HfApi):
    print(f"\n{'='*60}")
    print(f"  Uploading datasets to {DATASET_REPO}")
    print(f"{'='*60}")
    
    create_repo(DATASET_REPO, repo_type="dataset", private=True, exist_ok=True)
    
    dataset_card = """---
license: mit
task_categories:
  - text-generation
  - text-classification
language:
  - zh
tags:
  - aigc-detection
  - style-transfer
  - parallel-corpus
size_categories:
  - 10K<n<100K
---

# StyleShield Training Data

Parallel AI-human text pairs for training StyleShield style transfer models.

## Files

| File | Samples | Domains | Description |
|------|---------|---------|-------------|
| `dataset_multidomain.jsonl` | ~436K | Social Media + News + Academic | Full multi-domain training set |
| `dataset_zhihu_pairs_full.jsonl` | ~346K | Social Media only | Single-domain subset |
| `test_set_1000.jsonl` | 1,000 | All domains | Held-out evaluation set |

## Format

Each line is a JSON object:
```json
{
    "id": "zhihu_043650",
    "source": "zhihu",
    "human_text": "...",
    "ai_text": "..."
}
```

## Construction

Human texts were collected from three domains:
- **Social Media**: Chinese Q&A platform discussions
- **News**: Journalistic articles
- **Academic**: Research paper abstracts

AI counterparts were generated by prompting Qwen-2.5-7B-Instruct to rewrite each human text,
then filtered by an AIGC detector (P(AI) > 0.7 threshold).
"""
    api.upload_file(
        path_or_fileobj=dataset_card.encode(),
        path_in_repo="README.md",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print("  Uploaded dataset card")
    
    for hf_path, local_path in DATASETS.items():
        if not os.path.exists(local_path):
            print(f"  SKIP (not found): {local_path}")
            continue
        size_gb = os.path.getsize(local_path) / (1024**3)
        print(f"  Uploading {hf_path} ({size_gb:.1f} GB)...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=hf_path,
            repo_id=DATASET_REPO,
            repo_type="dataset",
        )
        print(f"    Done: {hf_path}")
    
    print(f"\n  Dataset repo: https://huggingface.co/datasets/{DATASET_REPO}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["checkpoints", "dataset", "all"], default="all")
    args = parser.parse_args()
    
    api = HfApi()
    
    # Verify login
    try:
        user_info = api.whoami()
        print(f"Logged in as: {user_info['name']}")
    except Exception:
        print("ERROR: Not logged in. Run this first:")
        print('  python3 -c "from huggingface_hub import login; login()"')
        return
    
    if args.mode in ("checkpoints", "all"):
        upload_checkpoints(api)
    
    if args.mode in ("dataset", "all"):
        upload_dataset(api)
    
    print("\n" + "="*60)
    print("  All uploads complete!")
    print("="*60)


if __name__ == "__main__":
    main()
