#!/bin/bash
# Full evaluation for ablation_split21 best checkpoint (step_5000)
# Run on cluster with single GPU:
#   bash scripts/eval_split21_full.sh

cd /root/workspace/AIGC_FUCK_7

python scripts/eval_acl.py \
    --ckpt experiments/ablation_split21/checkpoints/step_5000.pt \
    --tag ablation_split21 \
    --gammas 5.0 5.5 6.0 6.5 7.0

echo "Done! Results saved to eval_results/acl_eval_ablation_split21.json"
