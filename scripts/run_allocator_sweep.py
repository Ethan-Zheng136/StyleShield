"""批量测试 Allocator 在不同 target_rate 下的表现。

用法:
  python scripts/run_allocator_sweep.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.allocator_sf import StyleShieldPipeline, allocate_and_rewrite

TEXT_FILE = "data/test_ai_10k_zhihu.txt"
TARGET_RATES = [10, 20, 30, 40, 50, 60]
OUTPUT_DIR = "allocator_results/sweep"
GAMMAS = [6.0, 6.5, 7.0, 7.5]
NUM_STEPS = 64
RETRIES = 3


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"输入文本: {len(text)} 字")
    print(f"目标 rate 列表: {TARGET_RATES}")
    print(f"gammas: {GAMMAS}, num_steps: {NUM_STEPS}, retries: {RETRIES}")
    print("=" * 70)

    pipeline = StyleShieldPipeline(
        ckpt_path="experiments/styleflow_v2/checkpoints/step_30000.pt",
        detector_path="models/AIGC_detector_zhv3",
    )

    summary_rows = []

    for rate in TARGET_RATES:
        print(f"\n{'='*70}")
        print(f"  TARGET RATE = {rate}%")
        print(f"{'='*70}")

        t0 = time.time()
        result = allocate_and_rewrite(
            text=text,
            target_rate=rate,
            pipeline=pipeline,
            chunk_method="sentence",
            chunk_size=500,
            num_steps=NUM_STEPS,
            retries=RETRIES,
            gamma_candidates=GAMMAS,
            tolerance=0.05,
            verbose=True,
        )
        elapsed = time.time() - t0

        out_path = os.path.join(OUTPUT_DIR, f"rate_{rate}.json")
        save_data = {
            "target_rate": result["target_rate"],
            "achieved_rate_weighted": result["achieved_rate_weighted"],
            "original_rate_weighted": result.get("original_rate_weighted"),
            "stats": result["stats"],
            "chunk_details": result["chunk_details"],
            "final_text": result["final_text"],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        row = {
            "target_rate": rate,
            "achieved_weighted": result["achieved_rate_weighted"],
            "original_weighted": result.get("original_rate_weighted"),
            "n_rewritten": result["stats"]["n_rewritten"],
            "n_chunks": result["stats"]["n_chunks"],
            "below_50": result["stats"]["chunks_below_50pct"],
            "below_30": result["stats"]["chunks_below_30pct"],
            "rewrite_ratio": result["stats"]["rewrite_char_ratio"],
            "elapsed": round(elapsed, 1),
        }
        summary_rows.append(row)
        print(f"\n  >>> rate={rate}% 完成, 疑似率={row['achieved_weighted']}%, "
              f"改写{row['n_rewritten']}/{row['n_chunks']}块, "
              f"P<0.5: {row['below_50']}/{row['n_chunks']}, 耗时{row['elapsed']}s")

    print(f"\n\n{'='*70}")
    print("  汇总表 (逐段检测，类似知网)")
    print(f"{'='*70}")
    header = (f"{'Target':>8} {'Achieved':>10} {'Rewritten':>10} "
              f"{'P<0.5':>7} {'P<0.3':>7} {'Ratio':>8} {'Time':>8}")
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        print(f"{r['target_rate']:>7}% {r['achieved_weighted']:>9.1f}% "
              f"{r['n_rewritten']:>4}/{r['n_chunks']:<4} "
              f"{r['below_50']:>3}/{r['n_chunks']:<3} "
              f"{r['below_30']:>3}/{r['n_chunks']:<3} "
              f"{r['rewrite_ratio']*100:>6.1f}% "
              f"{r['elapsed']:>7.1f}s")

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)
    print(f"\n汇总保存: {summary_path}")


if __name__ == "__main__":
    main()
