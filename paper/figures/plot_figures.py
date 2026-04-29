"""Generate publication-quality figures for the StyleShield paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 8.5,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ── Data ──
gammas = [5.0, 5.5, 6.0, 6.5, 7.0]

# StyleShield on Det-v3
ss_pai    = [0.703, 0.635, 0.481, 0.249, 0.130]
ss_sim    = [0.948, 0.947, 0.942, 0.935, 0.928]
ss_evade  = [30.1, 36.8, 52.6, 79.7, 90.6]

# Baselines (single points)
baselines = {
    'Synonym':        {'pai': 0.985, 'sim': 1.000, 'evade': 0.0},
    'Backtranslation': {'pai': 0.203, 'sim': 0.852, 'evade': 82.6},
    'LLM Rewrite':    {'pai': 0.993, 'sim': 0.964, 'evade': 0.3},
}

# Cross-detector data at gamma=7.0
cross_detectors = ['Det-v3*', 'Det-v2', 'ANX-BERT', 'GPT2-Det']
cross_ss     = [0.130, 0.002, 0.035, 0.010]
cross_syn    = [0.985, 0.424, 0.856, 0.153]
cross_bt     = [0.203, 0.140, 0.270, 0.003]
cross_llm    = [0.993, 0.803, 0.904, 0.245]

# ── Figure 2: Gamma Curve (dual-axis) ──
fig, ax1 = plt.subplots(figsize=(4.5, 3.0))

color_pai = '#2171B5'
color_sim = '#CB181D'

ax1.set_xlabel(r'Style intensity $\gamma$')
ax1.set_ylabel(r'$P_{\mathrm{AI}}$ (Det-v3)', color=color_pai)
l1, = ax1.plot(gammas, ss_pai, 'o-', color=color_pai, linewidth=2, markersize=6, label=r'$P_{\mathrm{AI}}$')
ax1.tick_params(axis='y', labelcolor=color_pai)
ax1.set_ylim(0, 0.85)

# Backtranslation reference line
ax1.axhline(y=0.203, color='#756BB1', linestyle='--', linewidth=1, alpha=0.7, label='Backtrans. $P_{\\mathrm{AI}}$')
ax1.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax1.text(7.05, 0.51, '$P_{\\mathrm{AI}}$=0.5', fontsize=7, color='gray', va='bottom')

ax2 = ax1.twinx()
ax2.set_ylabel('Semantic Similarity', color=color_sim)
l2, = ax2.plot(gammas, ss_sim, 's--', color=color_sim, linewidth=2, markersize=6, label='Sem. Sim.')
ax2.tick_params(axis='y', labelcolor=color_sim)
ax2.set_ylim(0.90, 0.96)

# Backtranslation sim reference
ax2.axhline(y=0.852, color='#756BB1', linestyle=':', linewidth=1, alpha=0.5)

lines = [l1, l2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center left', framealpha=0.9)

ax1.set_xticks(gammas)
plt.title(r'\textbf{StyleShield}: $\gamma$ controls evasion--preservation trade-off', fontsize=10)
fig.tight_layout()
plt.savefig('fig2_gamma_curve.pdf')
plt.savefig('fig2_gamma_curve.png')
plt.close()
print("Saved fig2_gamma_curve.pdf")

# ── Figure 3: Pareto Plot (Evasion vs Similarity) ──
fig, ax = plt.subplots(figsize=(4.5, 3.2))

ax.plot(ss_sim, ss_evade, 'o-', color='#2171B5', linewidth=2.5, markersize=8, 
        label='StyleShield', zorder=5)

for i, g in enumerate(gammas):
    offset_x = 0.001 if i < 3 else -0.002
    offset_y = 2 if i < 4 else -5
    ax.annotate(f'$\\gamma$={g}', (ss_sim[i], ss_evade[i]),
                textcoords="offset points", xytext=(8, offset_y), fontsize=7,
                color='#2171B5')

markers = {'Synonym': ('^', '#E6550D'), 'Backtranslation': ('D', '#756BB1'), 'LLM Rewrite': ('v', '#31A354')}
for name, data in baselines.items():
    m, c = markers[name]
    ax.scatter(data['sim'], data['evade'], marker=m, color=c, s=100, 
              label=name, zorder=4, edgecolors='black', linewidth=0.5)

ax.set_xlabel('Semantic Similarity')
ax.set_ylabel('Evasion Rate@0.5 (%)')
ax.set_xlim(0.83, 0.97)
ax.set_ylim(-5, 100)
ax.axhline(y=50, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax.legend(loc='lower left', framealpha=0.9)
plt.title(r'\textbf{Pareto frontier}: StyleShield dominates all baselines', fontsize=10)
fig.tight_layout()
plt.savefig('fig3_pareto.pdf')
plt.savefig('fig3_pareto.png')
plt.close()
print("Saved fig3_pareto.pdf")

# ── Figure 4: Cross-Detector Bar Chart ──
fig, ax = plt.subplots(figsize=(5.5, 3.0))

x = np.arange(len(cross_detectors))
width = 0.2

bars1 = ax.bar(x - 1.5*width, cross_syn, width, label='Synonym', color='#E6550D', alpha=0.85)
bars2 = ax.bar(x - 0.5*width, cross_bt, width, label='Backtrans.', color='#756BB1', alpha=0.85)
bars3 = ax.bar(x + 0.5*width, cross_llm, width, label='LLM Rewrite', color='#31A354', alpha=0.85)
bars4 = ax.bar(x + 1.5*width, cross_ss, width, label='StyleShield', color='#2171B5', alpha=0.85)

ax.set_xlabel('AIGC Detector')
ax.set_ylabel(r'$P_{\mathrm{AI}}$ $\downarrow$')
ax.set_xticks(x)
ax.set_xticklabels(cross_detectors)
ax.legend(loc='upper right', ncol=2, fontsize=8)
ax.set_ylim(0, 1.15)
ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

plt.title(r'\textbf{Cross-detector generalization} ($\gamma$=7.0)', fontsize=10)
fig.tight_layout()
plt.savefig('fig4_cross_detector.pdf')
plt.savefig('fig4_cross_detector.png')
plt.close()
print("Saved fig4_cross_detector.pdf")

# ── Figure 5: Ablation PPL vs Similarity ──
abl_labels = ['Full\n($L$=14)', 'A1: w/o\nDet. Reward', 'A3: Split\nLayer 7']
abl_sim_65 = [0.935, 0.928, 0.904]
abl_ppl_65 = [21.1, 26.4, 99.9]
abl_evade_65 = [79.7, 89.4, 100.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

colors = ['#2171B5', '#E6550D', '#31A354']

# Left: Similarity vs Evasion
for i, (lab, sim, evd) in enumerate(zip(abl_labels, abl_sim_65, abl_evade_65)):
    ax1.scatter(sim, evd, color=colors[i], s=120, zorder=5, edgecolors='black', linewidth=0.5)
    ax1.annotate(lab.replace('\n', ' '), (sim, evd), 
                textcoords="offset points", xytext=(10, -5), fontsize=7.5)

ax1.set_xlabel('Semantic Similarity')
ax1.set_ylabel('Evasion Rate@0.5 (%)')
ax1.set_title(r'(a) Similarity vs Evasion ($\gamma$=6.5)', fontsize=9)

# Right: PPL comparison
bar_x = np.arange(len(abl_labels))
ax2.bar(bar_x, abl_ppl_65, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
ax2.set_xticks(bar_x)
ax2.set_xticklabels(abl_labels, fontsize=8)
ax2.set_ylabel('Perplexity (PPL)')
ax2.axhline(y=16.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax2.text(2.5, 17.5, 'Human PPL', fontsize=7, color='gray')
ax2.axhline(y=10.7, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax2.text(2.5, 11.7, 'AI PPL', fontsize=7, color='gray')
ax2.set_title(r'(b) Perplexity ($\gamma$=6.5)', fontsize=9)

fig.tight_layout()
plt.savefig('fig5_ablation.pdf')
plt.savefig('fig5_ablation.png')
plt.close()
print("Saved fig5_ablation.pdf")

print("\nAll figures generated successfully!")
