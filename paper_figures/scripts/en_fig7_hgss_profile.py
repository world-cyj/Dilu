#!/usr/bin/env python3
"""
en_fig7_hgss_profile.py
English version of Fig7: HGSS-NPU Profiling Results (Real NPU data)
Output: ../output/en_fig7_hgss_profile.pdf
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'output', 'en_fig7_hgss_profile.pdf')
plt.rcParams.update({'font.family': 'DejaVu Sans',
                     'axes.unicode_minus': False, 'font.size': 10})

models = ['ResNet-152', 'VGG-19', 'BERT-base']
v_req  = [15, 15, 30]; c_req = [3,  2,  7]
v_lim  = [17, 18, 33]; c_lim = [5,  4,  9]
eff    = [1.139, 1.056, 0.573]
colors = ['#E05252', '#F4A460', '#5B8DB8']
vutil  = [70, 75, 45]; cutil = [30, 25, 55]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Fig.7  HGSS-NPU Profiling Results  (Ascend 910B3, Real Hardware)',
             fontsize=12, fontweight='bold', y=1.02)

# (A) Request -> Limit anchor scatter
ax = axes[0]
for i, (m, vr, cr, vl, cl, c) in enumerate(
        zip(models, v_req, c_req, v_lim, c_lim, colors)):
    ax.scatter(vr, cr, s=180, color=c, zorder=5, marker='o')
    ax.scatter(vl, cl, s=220, color=c, zorder=5, marker='*')
    ax.annotate('', xy=(vl, cl), xytext=(vr, cr),
                arrowprops=dict(arrowstyle='->', color=c, lw=1.6))
    ax.text(vr-1.5, cr+0.3, f'Req({vr},{cr})', fontsize=7.5,
            color=c, ha='right')
    ax.text(vl+0.5, cl+0.3, f'Lim({vl},{cl})', fontsize=7.5,
            color=c, ha='left')
leg = [plt.Line2D([0],[0], marker='o', color='w',
                  markerfacecolor=c, markersize=9, label=m)
       for m, c in zip(models, colors)]
ax.legend(handles=leg, fontsize=8, loc='upper left')
ax.set_xlabel('Vector Core Quota  V'); ax.set_ylabel('Cube Core Quota  C')
ax.set_title('(A) Request -> Limit Anchor Trajectory\n(circle=Request, star=Limit)')
ax.set_xlim(0, 42); ax.set_ylim(0, 13)
ax.grid(alpha=0.3); ax.spines[['top','right']].set_visible(False)

# (B) Core utilization breakdown
ax = axes[1]
y = np.arange(len(models)); h = 0.4
ax.barh(y+h/2, vutil, h, label='Vector Core Util %', color='#5B8DB8', zorder=3)
ax.barh(y-h/2, cutil, h, label='Cube Core Util %',   color='#E07B54', zorder=3)
for i, (v, c) in enumerate(zip(vutil, cutil)):
    ax.text(v+1, i+h/2, f'{v}%', va='center', fontsize=9,
            color='#5B8DB8', fontweight='bold')
    ax.text(c+1, i-h/2, f'{c}%', va='center', fontsize=9,
            color='#E07B54', fontweight='bold')
ax.set_yticks(y); ax.set_yticklabels(models, fontsize=9)
ax.set_xlabel('Utilization (%)')
ax.set_title('(B) Vector / Cube Core Utilization\n(heterogeneity drives complementary scheduling)')
ax.axvline(50, color='gray', ls=':', alpha=.5)
ax.set_xlim(0, 95); ax.legend(fontsize=8)
ax.grid(axis='x', alpha=0.3, zorder=0)
ax.spines[['top','right']].set_visible(False)

# (C) Efficiency at Limit anchor
ax = axes[2]
bars = ax.bar(models, eff, color=colors, edgecolor='white', zorder=3, width=0.5)
for b, e in zip(bars, eff):
    ax.text(b.get_x()+b.get_width()/2, e+0.02, f'{e:.3f}',
            ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Efficiency  eff = throughput / (V + C)')
ax.set_title('(C) Limit Anchor Efficiency\neff = thr / (V+C)')
ax.set_ylim(0, 1.4); ax.grid(axis='y', alpha=0.3, zorder=0)
ax.tick_params(axis='x', labelsize=9)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
