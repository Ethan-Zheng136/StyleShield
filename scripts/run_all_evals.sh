#!/bin/bash
# ACL 完整评估流水线 — 在 tmux 中运行
# 包含：StyleShield 评估 + 3 个 baseline + 消融配置生成 + 跨域数据准备 + 人工评估问卷
set -e
cd /root/workspace/AIGC_FUCK_7

LOG_DIR="eval_results"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  ACL Evaluation Pipeline"
echo "  Started: $(date)"
echo "============================================"

# ── Step 1: Synonym baseline (fastest, ~30min for 1000 samples) ──
echo ""
echo "[Step 1/7] Running SYNONYM baseline..."
echo "  Start: $(date)"
python3 scripts/eval_acl.py \
  --baseline synonym \
  --detectors zhv3,zhv2,anx-bert,gpt2-det \
  2>&1 | tee "$LOG_DIR/log_baseline_synonym.txt"
echo "[Step 1/7] DONE at $(date)"

# ── Step 2: LLM Rewrite baseline (~2hr for 1000 samples) ──
echo ""
echo "[Step 2/7] Running LLM REWRITE baseline..."
echo "  Start: $(date)"
python3 scripts/eval_acl.py \
  --baseline llm_rewrite \
  --detectors zhv3,zhv2,anx-bert,gpt2-det \
  2>&1 | tee "$LOG_DIR/log_baseline_llm_rewrite.txt"
echo "[Step 2/7] DONE at $(date)"

# ── Step 3: Backtranslation baseline (~1hr for 1000 samples) ──
echo ""
echo "[Step 3/7] Running BACKTRANSLATION baseline..."
echo "  Start: $(date)"
python3 scripts/eval_acl.py \
  --baseline backtrans \
  --detectors zhv3,zhv2,anx-bert,gpt2-det \
  2>&1 | tee "$LOG_DIR/log_baseline_backtrans.txt"
echo "[Step 3/7] DONE at $(date)"

# ── Step 4: StyleShield full eval on 1000 samples ──
echo ""
echo "[Step 4/7] Running STYLESHIELD full eval (step_5000, 1000 samples)..."
echo "  Start: $(date)"
python3 scripts/eval_acl.py \
  --ckpt experiments/styleflow_v2/checkpoints/step_5000.pt \
  --detectors zhv3,zhv2,anx-bert,gpt2-det \
  2>&1 | tee "$LOG_DIR/log_styleshield_step5000.txt"
echo "[Step 4/7] DONE at $(date)"

# ── Step 5: Generate ablation configs ──
echo ""
echo "[Step 5/7] Generating ablation training configs..."
python3 << 'PYEOF'
import yaml, os, copy

base_path = "configs/styleflow.yaml"
with open(base_path) as f:
    base = yaml.safe_load(f)

ablations = {
    "ablation_no_detector": {
        "det_weight": 0.0,
        "det_warmup_steps": 999999,
        "save_dir": "./experiments/ablation_no_detector/checkpoints",
    },
    "ablation_zhihu_only": {
        "data_path": "/root/workspace/AIGC_FUCK/dataset_zhihu_pairs_full.jsonl",
        "save_dir": "./experiments/ablation_zhihu_only/checkpoints",
    },
    "ablation_split_layer_7": {
        "qwen_split_layer": 7,
        "save_dir": "./experiments/ablation_split7/checkpoints",
    },
    "ablation_split_layer_21": {
        "qwen_split_layer": 21,
        "save_dir": "./experiments/ablation_split21/checkpoints",
    },
}

os.makedirs("configs", exist_ok=True)
for name, overrides in ablations.items():
    cfg = copy.deepcopy(base)
    cfg.update(overrides)
    out_path = f"configs/{name}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"  Created {out_path}")

print("  Note: w/o Qwen conditioning requires code change (disable cross-attn in model)")
print("  All ablation configs saved.")
PYEOF
echo "[Step 5/7] DONE at $(date)"

# ── Step 6: Prepare cross-domain test data ──
echo ""
echo "[Step 6/7] Preparing cross-domain test data (novel/poetry/legal)..."
python3 << 'PYEOF'
import json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN = "/root/workspace/AIGC_FUCK/models/Qwen2.5-7B-Instruct"
OUT_DIR = "/root/workspace/AIGC_FUCK"
OUT_FILE = os.path.join(OUT_DIR, "test_cross_domain.jsonl")

domains = {
    "novel": [
        "月光洒在青石板路上，她独自走在回家的路上。远处的桂花香随风飘来，让她想起了小时候在外婆家的那些夏天。那时候日子过得慢，一个下午可以在院子里看蚂蚁搬家看好久。",
        "老张蹲在村口的大槐树下，手里捏着一根旱烟，眯着眼看远处的麦田。今年的麦子长得不错，比去年强多了。他心里盘算着，等收了麦子，给孙子买辆自行车。",
        "雨下了整整三天，巷子里的积水都没过了脚踝。李阿婆坐在门槛上，一边择菜一边跟邻居聊天。她说以前这条巷子可热闹了，卖豆腐的、磨剪刀的，天天都有人吆喝。",
        "火车缓缓驶出站台，窗外的景色从城市变成了田野。他靠在座椅上，翻开那本已经看了一半的小说。对面的大叔鼾声如雷，旁边的小姑娘在安静地听歌。",
        "厨房里飘出红烧肉的香味，奶奶在灶台前忙活着。她的手上全是面粉，脸上带着满足的笑。每年过年，她都要做一桌子菜，虽然现在吃的人越来越少了。",
    ],
    "poetry": [
        "春水初生，春林初盛，春风十里不如你。白云堆里，青山缺处，恰是故人来时路。落花无意，流水有情，天涯何处觅知音。",
        "竹影摇曳窗前月，茶烟袅袅案上书。半生浮沉终不悔，只因少年志未酬。秋来黄叶满阶砌，人去空楼独倚栏。",
        "江南好，风景旧曾谙。日出江花红胜火，春来江水绿如蓝。能不忆江南？山寺月中寻桂子，郡亭枕上看潮头。何日更重游？",
        "夜深人静时，窗外雨声淅沥。独坐灯前，翻开泛黄的信笺。字迹已经模糊，但那些年的故事却愈发清晰。时光如水，带走了太多，也沉淀了太多。",
        "一壶浊酒喜相逢，古今多少事，都付笑谈中。举杯邀明月，对影成三人。人生得意须尽欢，莫使金樽空对月。",
    ],
    "legal": [
        "根据《中华人民共和国民法典》第一百四十三条规定，具备下列条件的民事法律行为有效：行为人具有相应的民事行为能力；意思表示真实；不违反法律、行政法规的强制性规定，不违背公序良俗。",
        "当事人对合同的效力可以约定附条件。附生效条件的合同，自条件成就时生效。附解除条件的合同，自条件成就时失效。当事人为自己的利益不正当地阻止条件成就的，视为条件已成就。",
        "劳动者有下列情形之一的，用人单位不得依照本法第四十条、第四十一条的规定解除劳动合同：从事接触职业病危害作业的劳动者未进行离岗前职业健康检查，或者疑似职业病病人在诊断或者医学观察期间的。",
        "人民法院审理民事案件，应当根据自愿和合法的原则进行调解；调解不成的，应当及时判决。人民法院审理民事案件，除涉及国家秘密、个人隐私或者法律另有规定的以外，应当公开进行。",
        "公司合并可以采取吸收合并或者新设合并。一个公司吸收其他公司为吸收合并，被吸收的公司解散。两个以上公司合并设立一个新的公司为新设合并，合并各方解散。",
    ],
}

print("Loading Qwen for AI-style generation...")
tok = AutoTokenizer.from_pretrained(QWEN, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    QWEN, torch_dtype=torch.bfloat16, trust_remote_code=True
).to("cuda").eval()
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

records = []
for domain, texts in domains.items():
    for i, human_text in enumerate(texts):
        prompt = f"请用AI助手的标准风格重写以下文本，保持原意但使用更规范、更专业的表达：\n\n{human_text}"
        messages = [{"role": "user", "content": prompt}]
        chat_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(chat_text, return_tensors="pt", max_length=1024, truncation=True).to("cuda")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.generate(
                **enc, max_new_tokens=512, do_sample=True,
                temperature=0.7, top_p=0.9, pad_token_id=tok.pad_token_id,
            )
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        ai_text = tok.decode(new_tokens, skip_special_tokens=True).strip()

        records.append({
            "id": f"cross_{domain}_{i:03d}",
            "domain": domain,
            "ai_text": ai_text,
            "human_text": human_text,
        })
        print(f"  [{domain}_{i}] human={len(human_text)}ch -> ai={len(ai_text)}ch")

with open(OUT_FILE, "w") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nSaved {len(records)} cross-domain samples to {OUT_FILE}")
for domain in domains:
    n = sum(1 for r in records if r["domain"] == domain)
    print(f"  {domain}: {n}")
PYEOF
echo "[Step 6/7] DONE at $(date)"

# ── Step 7: Create human evaluation questionnaire ──
echo ""
echo "[Step 7/7] Creating human evaluation design..."
python3 << 'PYEOF'
import json, os

design = {
    "task": "StyleShield Human Evaluation",
    "description": "评估 AI 文本改写后的质量",
    "annotators": "3-5 名标注者",
    "samples": 30,
    "scoring": {
        "fluency": {
            "name": "流畅度",
            "scale": "1-5",
            "description": "文本是否通顺、自然、无语法错误",
            "rubric": {
                1: "完全不通顺，无法理解",
                2: "有多处语法错误，阅读困难",
                3: "基本通顺，有少量不自然之处",
                4: "通顺自然，偶有小瑕疵",
                5: "完全通顺自然，与人类写作无异",
            },
        },
        "semantic_preservation": {
            "name": "语义保留",
            "scale": "1-5",
            "description": "改写后是否保留了原文的核心意思",
            "rubric": {
                1: "完全偏离原意",
                2: "保留了少部分原意",
                3: "保留了核心意思，但有遗漏或偏差",
                4: "很好地保留了原意，仅有微小差异",
                5: "完美保留原意，无任何信息丢失",
            },
        },
        "naturalness": {
            "name": "自然度（像人写的程度）",
            "scale": "1-5",
            "description": "改写后的文本是否像人类写的",
            "rubric": {
                1: "明显是机器生成的",
                2: "大部分像机器生成的",
                3: "难以判断是人还是机器",
                4: "大部分像人写的",
                5: "完全像人写的",
            },
        },
    },
    "procedure": [
        "1. 随机从测试集中选取 30 个样本",
        "2. 对每个样本，展示原文（AI生成）和改写后的文本（不告知方法）",
        "3. 标注者对改写文本的三个维度分别打 1-5 分",
        "4. 不同方法（StyleShield + 3 baselines）的样本随机混合，标注者不知道哪个是哪个方法",
        "5. 计算 inter-annotator agreement (Fleiss' kappa)",
        "6. 对每个方法报告三个维度的 mean ± std",
    ],
    "blinding": "Double-blind: 标注者不知道方法名称，样本随机排列",
    "output_format": "每个标注者一份 CSV，列: sample_id, method (hidden), fluency, semantic, naturalness",
}

os.makedirs("eval_results", exist_ok=True)
with open("eval_results/human_eval_design.json", "w") as f:
    json.dump(design, f, ensure_ascii=False, indent=2)
print("Human evaluation design saved to eval_results/human_eval_design.json")
print("\nScoring rubric:")
for dim, info in design["scoring"].items():
    print(f"\n  {info['name']} ({info['scale']}):")
    for score, desc in info["rubric"].items():
        print(f"    {score}: {desc}")
PYEOF
echo "[Step 7/7] DONE at $(date)"

echo ""
echo "============================================"
echo "  ALL EVALUATIONS COMPLETE"
echo "  Finished: $(date)"
echo "============================================"
echo ""
echo "Results saved in eval_results/:"
ls -la eval_results/
