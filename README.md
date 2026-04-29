# StyleShield

**Continuous and Controllable Style Transfer for Evading AI-Generated Content Detectors**

> EMNLP 2026 Submission

## Overview

StyleShield is a flow-matching-based framework for continuous, controllable AI-to-human text style transfer. It builds on a Diffusion Transformer (DiT) backbone pretrained with flow matching (LangFlow) and introduces cross-attention adapters conditioned on frozen Qwen-7B representations.

Key features:
- **Continuous control**: A single parameter γ smoothly controls the evasion-preservation trade-off
- **Strong evasion**: ≥90.6% on training detector, ≥99% on unseen detectors
- **Semantic preservation**: 0.928 cosine similarity at γ=7.0
- **Cross-detector generalization**: Trained with one detector, transfers to all others
- **Long-document support**: Allocator algorithm for precise target detection rate control

## Project Structure

```
├── src/                  # Model architecture
│   ├── model.py          # StyleFlowZh: DiT + cross-attention adapters
│   ├── qwen_encoder.py   # QwenHiddenExtractor: frozen Qwen feature extraction
│   ├── dataset.py        # PairDataset: AI-human text pairs
│   └── config.py         # Configuration dataclass
├── scripts/
│   ├── train_styleflow.py    # Training script (DDP)
│   ├── eval_acl.py           # Comprehensive evaluation
│   ├── transfer.py           # Single-sample inference
│   ├── allocator_sf.py       # Long-document allocator
│   └── ...
├── configs/              # Training YAML configs
├── paper/                # LaTeX source for the paper
├── eval_results/         # Evaluation result JSONs
└── tokenizer/            # LangFlow tokenizer
```

## Checkpoints

Model checkpoints are available on Hugging Face Hub (link TBD).

| Checkpoint | Description |
|------------|-------------|
| `step_30000.pt` (v2) | Best full model (multi-domain, layer 14) |
| `ablation_no_detector/step_5000.pt` | A1: w/o detector reward |
| `ablation_split7/step_15000.pt` | A3: split layer 7 |
| `ablation_split21/step_5000.pt` | A4: split layer 21 |

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers
- einops

```bash
pip install -r requirements.txt
```

## Quick Start

```python
# Inference example
python scripts/transfer.py \
    --ckpt experiments/styleflow_v2/checkpoints/step_30000.pt \
    --gamma 6.5 \
    --text "你要转换的AI生成文本"
```

## Citation

```bibtex
@inproceedings{styleshield2026,
  title={StyleShield: Continuous and Controllable Style Transfer for Evading AI-Generated Content Detectors},
  author={Anonymous},
  booktitle={EMNLP},
  year={2026}
}
```

## License

MIT
