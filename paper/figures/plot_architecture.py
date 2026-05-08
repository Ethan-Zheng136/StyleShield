"""
High-quality compact architecture diagram for Figure 1.
Designed to fit single-column width (~3.3in) in ACL/EMNLP format.
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType fonts in PDF
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.colors as mcolors
import numpy as np

fig = plt.figure(figsize=(7.5, 4.8))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-3.5, 11.5)
ax.set_ylim(-10.0, 1.0)
ax.set_aspect('equal')
ax.axis('off')

C = dict(embed='#B8D4E8', flow='#E2E2E2', sattn='#F0C27A', cross='#B5D8A3',
         ffn='#C8C8C8', qwen='#DCDCDC', proj='#DCDCDC',
         euler='#D4E4F0', ditinf='#7FB0C8', gate='#FFF8DC')

def rbox(x, y, w, h, fc='white', ec='black', lw=0.7, rad=0.08):
    r = FancyBboxPatch((x-w/2, y-h/2), w, h,
                       boxstyle=f"round,pad={rad}", fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(r)
    return r

def txt(x, y, s, fs=7, bold=False, color='black', va='center', ha='center'):
    w = 'bold' if bold else 'normal'
    return ax.text(x, y, s, fontsize=fs, fontweight=w, color=color,
                   ha=ha, va=va, zorder=3)

def ar(x1, y1, x2, y2, lw=0.6, c='black', hw=0.12, hl=0.15):
    dx, dy = x2-x1, y2-y1
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                                mutation_scale=8), zorder=1)

def dar(x1, y1, x2, y2, lw=0.5, c='black'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                                linestyle='--', mutation_scale=7), zorder=1)

def ln(pts, lw=0.6, c='black'):
    ax.plot([p[0] for p in pts], [p[1] for p in pts],
            color=c, lw=lw, zorder=1, solid_capstyle='round')

S = 0.85  # vertical spacing scale

# ═══════════ LEFT: TRAINING ═══════════
tx = 0.0

txt(tx, 0.5, 'Training', fs=10, bold=True)

# x_AI, x_human
txt(tx-1.2, 0.0, r'$\mathbf{x}_{\mathrm{AI}}$', fs=8, bold=True)
txt(tx+1.2, 0.0, r'$\mathbf{x}_{\mathrm{human}}$', fs=8, bold=True)

# Embed
ey = -0.7
rbox(tx-1.2, ey, 1.3, 0.42, fc=C['embed']); txt(tx-1.2, ey, 'Embed', fs=7)
rbox(tx+1.2, ey, 1.3, 0.42, fc=C['embed']); txt(tx+1.2, ey, 'Embed', fs=7)
ar(tx-1.2, -0.15, tx-1.2, ey+0.24)
ar(tx+1.2, -0.15, tx+1.2, ey+0.24)

# e labels
ely = -1.25
txt(tx-1.2, ely, r'$\mathbf{e}_{\mathrm{AI}}$', fs=7, bold=True)
txt(tx+1.2, ely, r'$\mathbf{e}_{\mathrm{human}}$', fs=7, bold=True)
ar(tx-1.2, ey-0.24, tx-1.2, ely+0.12)
ar(tx+1.2, ey-0.24, tx+1.2, ely+0.12)

# Flow Interpolation
fy = -2.05
rbox(tx, fy, 4.2, 0.55, fc=C['flow'])
txt(tx, fy+0.1, 'Flow Interpolation', fs=7)
txt(tx, fy-0.13, r'$\mathbf{z}_t = (1\!-\!t)\!\cdot\!\mathbf{e}_{\mathrm{AI}} + t\!\cdot\!\mathbf{e}_{\mathrm{human}} + \mathrm{noise}(\gamma)$', fs=5.5)
ar(tx-1.2, ely-0.12, tx-0.4, fy+0.3)
ar(tx+1.2, ely-0.12, tx+0.4, fy+0.3)

# DiT Backbone outer box
dit_top = -2.95
dit_bot = -5.15
dit_w = 4.2
rbox(tx, (dit_top+dit_bot)/2, dit_w, dit_bot-dit_top, fc='#FAFAFA', ec='black', lw=1.0, rad=0.12)
txt(tx, dit_top+0.2, 'DiT Backbone (×12 blocks)', fs=7.5, bold=True)

ar(tx, fy-0.3, tx, dit_top+0.38)

# Self-Attention
say = -3.4
rbox(tx, say, 3.5, 0.38, fc=C['sattn']); txt(tx, say, 'Self-Attention + adaLN', fs=7)
dar(tx+2.6, say, tx+1.8, say)
txt(tx+3.2, say, r'$\gamma$ embedding', fs=5.5, color='#555')

# zero-init gate
txt(tx+1.0, say-0.32, 'zero-init gate', fs=4.5)
rbox(tx+1.0, say-0.32, 1.05, 0.22, fc=C['gate'], ec='#999', lw=0.4, rad=0.05)
txt(tx+1.0, say-0.32, 'zero-init gate', fs=4.5)

# Cross-Attention
cy = -4.02
rbox(tx, cy, 3.5, 0.38, fc=C['cross']); txt(tx, cy, 'Cross-Attention Adapter', fs=7)
ar(tx, say-0.22, tx, cy+0.22)

# FFN
ffy = -4.65
rbox(tx, ffy, 3.5, 0.38, fc=C['ffn']); txt(tx, ffy, 'FFN', fs=7)
ar(tx, cy-0.22, tx, ffy+0.22)

# Loss
ly = -5.6
ar(tx, ffy-0.22, tx, ly+0.12)
txt(tx, ly, r'$\mathcal{L}_{\mathrm{CE}} \to \mathbf{x}_{\mathrm{human}} \;+\; \mathcal{L}_{\mathrm{det}}$ (detector reward)', fs=5.5)

# ── Qwen Encoder ──
qx, qy = -0.5, -7.0
rbox(qx, qy, 3.0, 0.95, fc=C['qwen'], ec='black', lw=0.8, rad=0.1)
txt(qx, qy+0.25, 'layers 0→L', fs=4.5, color='#666')
txt(qx, qy, 'Qwen-7B Encoder', fs=7.5, bold=True)
txt(qx, qy-0.25, '(Frozen)', fs=5.5, color='#666')
txt(qx+1.1, qy-0.25, '❄', fs=9, color='#5BA3D9')

# x_AI -> Qwen
txt(-2.7, qy, r'$\mathbf{x}_{\mathrm{AI}}$', fs=8, bold=True)
ar(-2.4, qy, qx-1.52, qy)

# H_qwen
hx = 1.6
ar(qx+1.52, qy, hx-0.25, qy)
txt(hx, qy+0.13, r'$\mathbf{H}_{\mathrm{qwen}}$', fs=6, bold=True)
txt(hx, qy-0.1, '(3584d)', fs=5)

# cond_proj
px = 2.8
rbox(px, qy, 1.1, 0.55, fc=C['proj'], lw=0.7)
txt(px, qy+0.08, 'cond_proj', fs=5.5)
txt(px, qy-0.12, '(→768d)', fs=5)
ar(hx+0.35, qy, px-0.58, qy)

# K V
txt(3.65, qy+0.3, 'K  V', fs=7.5, bold=True)
ar(px+0.58, qy, 3.65, qy)
ln([(3.65, qy), (3.65, -6.15)])

# left branch -> cross-attn
ln([(3.65, -6.15), (2.0, -6.15)])
ar(2.0, -6.15, 2.0, cy+0.05)

# right branch -> inference
ln([(3.65, -6.15), (5.2, -6.15)])
ln([(5.2, -6.15), (5.2, -4.5)])


# ═══════════ RIGHT: INFERENCE ═══════════
ix = 7.8

txt(ix, 0.5, 'Inference (SDEdit)', fs=10, bold=True)

txt(ix, 0.0, r'$\mathbf{x}_{\mathrm{AI}}$', fs=8, bold=True)
rbox(ix, -0.7, 1.5, 0.42, fc=C['embed']); txt(ix, -0.7, 'Embed', fs=7)
ar(ix, -0.15, ix, -0.46)

txt(ix, -1.25, r'$\mathbf{e}_{\mathrm{AI}}$', fs=7, bold=True)
ar(ix, -0.94, ix, -1.13)

# Add Noise
ny = -1.9
rbox(ix, ny, 2.2, 0.42, fc='white'); txt(ix, ny, r'Add Noise($\gamma_{\mathrm{start}}$)', fs=7)
ar(ix, -1.37, ix, ny+0.24)

# Euler Denoise
euler_top = -2.95
euler_bot = -5.15
rbox(ix, (euler_top+euler_bot)/2, 2.8, euler_bot-euler_top,
     fc=C['euler'], ec='black', lw=0.8, rad=0.1)
txt(ix, -3.2, 'Euler Denoise', fs=8, bold=True)
txt(ix, -3.55, '(T=64 steps)', fs=6, color='#444')
txt(ix, -3.85, r'$\gamma_{\mathrm{start}} \to \gamma_{\mathrm{min}}$', fs=6.5)

# DiT + Qwen Cond sub-box
diy = -4.45
rbox(ix, diy, 2.3, 0.38, fc=C['ditinf'], lw=0.7)
txt(ix, diy, 'DiT + Qwen Cond', fs=7, bold=True)

ar(ix, ny-0.24, ix, euler_top+0.18)

# Arrow from Qwen
ar(5.2, -4.5, ix-1.42, diy)

# Output
txt(ix, -5.65, r'argmax  $\to$  $\mathbf{x}_{\mathrm{output}}$', fs=8)
ar(ix, euler_bot+0.18, ix, -5.45)

# ── Colorbar ──
cb_x, cb_y = 10.2, -0.5
txt(cb_x, cb_y+0.35, 'Continuous Control', fs=6, bold=True)
cmap = mcolors.LinearSegmentedColormap.from_list('g', ['#EEF2F7', '#2B5B8A'])
grad = np.linspace(0, 1, 256).reshape(1, -1)
cb_w, cb_h = 1.6, 0.18
ext = [cb_x-cb_w/2, cb_x+cb_w/2, cb_y-cb_h/2, cb_y+cb_h/2]
ax.imshow(grad, aspect='auto', cmap=cmap, extent=ext, zorder=3, interpolation='bilinear')
ax.plot([ext[0], ext[1], ext[1], ext[0], ext[0]],
        [ext[2], ext[2], ext[3], ext[3], ext[2]], 'k-', lw=0.3, zorder=4)
txt(cb_x-cb_w/2, cb_y-0.22, '5.0', fs=5, va='top')
txt(cb_x,        cb_y-0.22, '6.0', fs=5, va='top')
txt(cb_x+cb_w/2, cb_y-0.22, '7.0', fs=5, va='top')
ax.text(cb_x-cb_w/2-0.05, cb_y-0.38, 'mild', fontsize=4.5, ha='center', va='top', style='italic')
ax.text(cb_x+cb_w/2+0.1, cb_y-0.38, 'strong', fontsize=4.5, ha='center', va='top', style='italic')


fig.savefig('/root/workspace/AIGC_FUCK_7/paper/figures/fig1_architecture.pdf',
            format='pdf', bbox_inches='tight', pad_inches=0.03)
fig.savefig('/root/workspace/AIGC_FUCK_7/paper/figures/fig1_architecture.png',
            format='png', dpi=400, bbox_inches='tight', pad_inches=0.03)
print('Done')
