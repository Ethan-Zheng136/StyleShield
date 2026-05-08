"""Qualitative examples figure: 3 domains × 3 gammas — academic style v6.
   Row-based colors, gaps between boxes, arrows penetrate into boxes."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap

plt.rcParams["font.family"] = ["Noto Sans SC", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "dejavusans"

with open("/tmp/qualitative_results.json") as f:
    data = json.load(f)

CHAR_LIMIT = 50

domains = [
    ("news_020475", "News"),
    ("zhihu_044508", "Social Media"),
    ("academic_008509", "Academic"),
]
gammas = ["6.0", "6.5", "7.0"]

EN_GLOSS = {
    "news_020475": {
        "input": "Furniture inspection: 30 samples from 4 cities, only 11 passed (37%). Main issue: formaldehyde.",
        "6.0": "Furniture inspection drew 30 samples from dealers across 4 cities. 11 passed (~33%). Formaldehyde was the key failure.",
        "6.5": "Housing furniture spot-check: 40 dealers, 30 samples tested. 11-14 passed (37%). Main issue: formaldehyde.",
        "7.0": "Furniture test: 30 samples from 30 dealers in 4 cities, 11 qualified (37%). Failures mainly on formaldehyde.",
    },
    "zhihu_044508": {
        "input": "Given Beethoven's unique composition and harmony skills, his creativity is unmatched. For the same chords, Beethoven uses them masterfully, while pop songs chase short-term traffic...",
        "6.0": "Beethoven showed unique composition and harmony in his music, with unmatched creativity. For the same chords, he used them masterfully, while pop songs chase short-term traffic...",
        "6.5": "Beethoven's compositions were rich in imagination and harmony. For the same chords, he used them brilliantly; but some pop songs just want a catchy melody for quick traffic...",
        "7.0": "Because Beethoven had unique composition and harmony skills, his creativity was unmatched. He used the same chords masterfully, while pop songs chase traffic with catchy melodies...",
    },
    "academic_008509": {
        "input": "This study explores value-sharing based on constitutional consensus for China's social transformation under globalization...",
        "6.0": "This study explores value-sharing via constitutional consensus for China's transformation. Current phenomena stem from lacking it...",
        "6.5": "Under globalization, constitutional consensus is crucial for China's transformation. Phenomena root in lacking value consensus...",
        "7.0": "Under globalization, constitutional value consensus matters for China's transformation. Social phenomena stem from lacking it...",
    },
}

ROW_STYLES = {
    "input": {"bg": "#fef2f0", "border": "#c62828", "score_color": "#c62828"},
    "6.0":   {"bg": "#fff8e1", "border": "#e65100", "score_color": "#d84315"},
    "6.5":   {"bg": "#f1f8e9", "border": "#689f38", "score_color": "#33691e"},
    "7.0":   {"bg": "#e0f2f1", "border": "#00695c", "score_color": "#004d40"},
}

def truncate(text, limit=CHAR_LIMIT):
    text = text.replace("\n", " ").strip()
    return text[:limit] + "…" if len(text) > limit else text

def wrap_cjk(text, width=26):
    lines = []
    while len(text) > width:
        lines.append(text[:width])
        text = text[width:]
    if text:
        lines.append(text)
    return "\n".join(lines)

def wrap_en(text, width=50):
    return "\n".join(textwrap.wrap(text, width=width))

nrows = 4
ncols = 3
row_labels = ["Input\n(AI text)", "γ = 6.0", "γ = 6.5", "γ = 7.0"]
row_keys = ["input", "6.0", "6.5", "7.0"]

cell_w = 5.2
cell_h = 1.45
gap_x = 0.18
gap_y = 0.25
margin_l = 0.78
margin_t = 0.35
margin_b = 0.1
margin_r = 0.1

fig_w = margin_l + ncols * cell_w + (ncols - 1) * gap_x + margin_r
fig_h = margin_t + nrows * cell_h + (nrows - 1) * gap_y + margin_b

fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")
fig.patch.set_facecolor("white")

for col_idx, (sid, eng_name) in enumerate(domains):
    cx = margin_l + col_idx * (cell_w + gap_x) + cell_w / 2
    ax.text(cx, fig_h - 0.05, eng_name,
            ha="center", va="top", fontsize=13, fontweight="bold",
            color="#0d47a1")

for col_idx, (sid, eng_name) in enumerate(domains):
    d = data[sid]

    for row_idx, rkey in enumerate(row_keys):
        style = ROW_STYLES[rkey]

        x0 = margin_l + col_idx * (cell_w + gap_x)
        y_top = fig_h - margin_t - row_idx * (cell_h + gap_y)
        y0 = y_top - cell_h

        rect = mpatches.FancyBboxPatch(
            (x0, y0), cell_w, cell_h,
            boxstyle="round,pad=0.05",
            facecolor=style["bg"], edgecolor=style["border"],
            linewidth=1.2, clip_on=False, zorder=2,
        )
        ax.add_patch(rect)

        if rkey == "input":
            cn_text = truncate(d["ai_text"])
            p_ai = d["original_p_ai"]
            en_text = EN_GLOSS[sid]["input"]
        else:
            cn_text = truncate(d["transfers"][rkey]["output"])
            p_ai = d["transfers"][rkey]["p_ai"]
            en_text = EN_GLOSS[sid][rkey]

        cn_wrapped = wrap_cjk(cn_text, width=26)
        en_wrapped = wrap_en(en_text, width=50)
        cx = x0 + cell_w / 2
        cy = y0 + cell_h / 2

        cn_lines = cn_wrapped.count("\n") + 1
        en_lines = en_wrapped.count("\n") + 1
        cn_h = cn_lines * 0.175
        en_h = en_lines * 0.15
        score_h = 0.22
        sp1 = 0.04
        sp2 = 0.08
        total = cn_h + sp1 + en_h + sp2 + score_h
        top_y = cy + total / 2

        ax.text(cx, top_y, cn_wrapped,
                ha="center", va="top", fontsize=8.3,
                color="#111111", linespacing=1.3, zorder=3)

        en_top = top_y - cn_h - sp1
        ax.text(cx, en_top, en_wrapped,
                ha="center", va="top", fontsize=7.2,
                color="#424242", style="italic", linespacing=1.25, zorder=3)

        score_y = en_top - en_h - sp2
        score_str = f"P(AI) = {p_ai:.3f}"
        ax.text(cx, score_y, score_str,
                ha="center", va="top", fontsize=9.5, fontweight="bold",
                color=style["score_color"], zorder=3,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor=style["score_color"],
                          linewidth=0.9, alpha=0.95))

        if col_idx == 0:
            label_x = margin_l - 0.1
            label_y = y0 + cell_h / 2
            ax.text(label_x, label_y, row_labels[row_idx],
                    ha="right", va="center", fontsize=9,
                    fontweight="bold", color="#424242")

        if row_idx < nrows - 1:
            arrow_x = cx
            arr_start = y0 + 0.06
            arr_end = y0 - gap_y - 0.06
            ax.annotate(
                "", xy=(arrow_x, arr_end),
                xytext=(arrow_x, arr_start),
                arrowprops=dict(arrowstyle="-|>", color="#9e9e9e",
                                lw=1.5, mutation_scale=14),
                zorder=5, clip_on=False,
            )

out_path = "/root/workspace/AIGC_FUCK_7/paper/figures/fig6_qualitative.pdf"
plt.savefig(out_path, bbox_inches="tight", dpi=300, facecolor="white",
            pad_inches=0.08)
plt.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200,
            facecolor="white", pad_inches=0.08)
print(f"Saved: {out_path}  (aspect ~{fig_w/fig_h:.1f}:1)")
plt.close()
