#!/bin/bash
set -e

INPUT="人工智能技术的快速发展对社会产生了深远的影响。从医疗健康到金融服务，从教育领域到交通运输，人工智能正在重塑各行各业的运作方式。机器学习算法能够从海量数据中发现隐藏的模式和规律，帮助人类做出更加精准的决策。深度学习模型在图像识别、自然语言处理和语音合成等领域取得了突破性的进展。然而，人工智能的发展也带来了一些值得关注的问题，包括数据隐私保护、算法偏见以及人工智能对就业市场的潜在冲击。如何在推动技术创新的同时确保人工智能的安全和公平使用，是当前社会面临的重要挑战。"

CKPT_DIR="experiments/styleflow/checkpoints"

for STEP in 5000 10000 15000 20000 25000 30000 40000 50000 75000 100000; do
    echo ""
    echo "=========================================="
    echo "CHECKPOINT: step_${STEP}"
    echo "=========================================="
    python scripts/transfer.py \
        --ckpt "${CKPT_DIR}/step_${STEP}.pt" \
        --input "$INPUT" \
        --sweep 2>&1 | grep -E "(γ=|Summary|Original)"
    echo ""
done

echo "ALL DONE"
