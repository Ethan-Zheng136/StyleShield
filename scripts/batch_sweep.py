"""Batch sweep across multiple checkpoints to find optimal one."""
import subprocess, re, sys, json

INPUT_TEXT = (
    "人工智能技术的快速发展对社会产生了深远的影响。从医疗健康到金融服务，"
    "从教育领域到交通运输，人工智能正在重塑各行各业的运作方式。机器学习算法"
    "能够从海量数据中发现隐藏的模式和规律，帮助人类做出更加精准的决策。深度"
    "学习模型在图像识别、自然语言处理和语音合成等领域取得了突破性的进展。然而，"
    "人工智能的发展也带来了一些值得关注的问题，包括数据隐私保护、算法偏见以及"
    "人工智能对就业市场的潜在冲击。如何在推动技术创新的同时确保人工智能的安全"
    "和公平使用，是当前社会面临的重要挑战。"
)

STEPS = [5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 75000, 100000]
GAMMAS = [5.0, 5.5, 6.0, 6.5, 7.0, 8.0]

CKPT_DIR = "experiments/styleflow/checkpoints"

results = {}

for step in STEPS:
    ckpt = f"{CKPT_DIR}/step_{step}.pt"
    print(f"\n{'='*60}")
    print(f"Testing step_{step}")
    print(f"{'='*60}")
    step_results = {}
    for gamma in GAMMAS:
        cmd = [
            sys.executable, "scripts/transfer.py",
            "--ckpt", ckpt,
            "--input", INPUT_TEXT,
            "--gamma", str(gamma),
            "--num_steps", "32",
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            combined = out.stdout + out.stderr
            pai_match = re.search(r"P\(AI\)\s*=\s*([\d.]+)\s*\(drop", combined)
            out_match = re.search(r"Output \(\d+ chars\): (.+)", combined)
            if pai_match:
                pai = float(pai_match.group(1))
                text_preview = out_match.group(1)[:80] if out_match else ""
                step_results[gamma] = {"p_ai": pai, "text": text_preview}
                print(f"  γ={gamma:.1f} → P(AI)={pai:.4f}  {text_preview}")
            else:
                print(f"  γ={gamma:.1f} → PARSE ERROR")
                step_results[gamma] = {"p_ai": -1, "text": "PARSE ERROR"}
        except subprocess.TimeoutExpired:
            print(f"  γ={gamma:.1f} → TIMEOUT")
            step_results[gamma] = {"p_ai": -1, "text": "TIMEOUT"}
    results[step] = step_results

print(f"\n\n{'='*80}")
print("SUMMARY TABLE")
print(f"{'='*80}")
header = f"{'Step':>8s}"
for g in GAMMAS:
    header += f" | γ={g:<4.1f}"
print(header)
print("-" * len(header))
for step in STEPS:
    row = f"{step:>8d}"
    for g in GAMMAS:
        pai = results[step].get(g, {}).get("p_ai", -1)
        if pai >= 0:
            row += f" | {pai:.4f}"
        else:
            row += f" |   ERR "
    print(row)

print(f"\n\n{'='*80}")
print("BEST CHECKPOINT ANALYSIS")
print(f"{'='*80}")
best_step = None
best_score = -1
for step in STEPS:
    sr = results[step]
    score = 0
    for g in GAMMAS:
        pai = sr.get(g, {}).get("p_ai", 1.0)
        if pai < 0:
            continue
        if g <= 5.5:
            score += (1.0 - pai) * 0.5
        elif g <= 6.5:
            if 0.1 < pai < 0.7:
                score += 3.0
            elif pai <= 0.1:
                score += 1.5
            else:
                score += (1.0 - pai) * 0.5
        else:
            score += (1.0 - pai) * 1.0
    print(f"  step_{step:>6d}: composite_score = {score:.2f}")
    if score > best_score:
        best_score = score
        best_step = step

print(f"\n  >>> BEST CHECKPOINT: step_{best_step} (score={best_score:.2f})")

print(f"\n\n{'='*80}")
print("TEXT QUALITY SAMPLES (γ=6.0 for each checkpoint)")
print(f"{'='*80}")
for step in STEPS:
    sr = results[step]
    info = sr.get(6.0, {})
    pai = info.get("p_ai", -1)
    text = info.get("text", "N/A")
    print(f"\n  step_{step}: P(AI)={pai:.4f}")
    print(f"    {text}")
