#!/usr/bin/env python3
"""
en_fig4_colocation_comparison.py
English version of Fig4: Colocation Benefit (Real NPU data)
Output: ../output/en_fig4_colocation_comparison.pdf
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'output', 'en_fig4_colocation_comparison.pdf')
plt.rcParams.update({'font.family': 'DejaVu Sans',
                     'axes.unicode_minus': False, 'font.size': 10})

svcs  = ['ResNet-152', 'VGG-19', 'BERT-base']
avg_c = [24.5, 24.8, 24.9]; avg_e = [25.8, 26.1, 27.0]
p99_c = [36.8, 37.2, 37.6]; p99_e = [39.1, 40.3, 41.2]
vutil = [70, 75, 45];        cutil = [30, 25, 55]
x = np.arange(len(svcs)); w = 0.35

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle('Fig.4  Colocation: Complementary vs. Exclusive  (Ascend 910B3, Scenario S3)',
             fontsize=11, fontweight='bold', y=1.02)

for ax, mc, me, title, yl in [
        (axes[0], avg_c, avg_e, '(A) Mean Latency', 'Mean Latency (ms)'),
        (axes[1], p99_c, p99_e, '(B) P99 Latency',  'P99 Latency (ms)')]:
    b1 = ax.bar(x-w/2, mc, w, label='Complementary (Ours)',
                color='#E05252', edgecolor='white', zorder=3)
    b2 = ax.bar(x+w/2, me, w, label='Exclusive (Baseline)',
                color='#B0C4DE', edgecolor='white', zorder=3)
    for b, v in zip(b1, mc):
        ax.text(b.get_x()+b.get_width()/2, v+.2, f'{v}',
                ha='center', fontsize=8.5, fontweight='bold', color='#C0392B')
    for b, v in zip(b2, me):
        ax.text(b.get_x()+b.get_width()/2, v+.2, f'{v}',
                ha='center', fontsize=8.5, color='#555')
    for i, (vc, ve) in enumerate(zip(mc, me)):
        ax.annotate(f'-{ve-vc:.1f}ms', xy=(i, vc), xytext=(i, vc-4.5),
                    ha='center', fontsize=8, color='green', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(svcs, fontsize=9)
    ax.axhline(50, color='red', ls='--', lw=1, alpha=.6, label='SLO = 50ms')
    ax.set_ylabel(yl); ax.set_title(title)
    ax.set_ylim(0, 53); ax.legend(fontsize=7.5)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

ax = axes[2]
y = np.arange(len(svcs)); h = 0.4
ax.barh(y+h/2, vutil, h, label='Vector Core Util %', color='#5B8DB8', zorder=3)
ax.barh(y-h/2, cutil, h, label='Cube Core Util %',   color='#E07B54', zorder=3)
for i, (v, c) in enumerate(zip(vutil, cutil)):
    ax.text(v+1, i+h/2, f'{v}%', va='center', fontsize=9,
            color='#5B8DB8', fontweight='bold')
    ax.text(c+1, i-h/2, f'{c}%', va='center', fontsize=9,
            color='#E07B54', fontweight='bold')
ax.set_yticks(y); ax.set_yticklabels(svcs, fontsize=9)
ax.set_xlabel('Utilization (%)')
ax.set_title('(C) Vector/Cube Core Utilization\n(basis for complementary scheduling)')
ax.axvline(50, color='gray', ls=':', alpha=.5)
ax.set_xlim(0, 95); ax.legend(fontsize=8)
ax.grid(axis='x', alpha=0.3, zorder=0)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
