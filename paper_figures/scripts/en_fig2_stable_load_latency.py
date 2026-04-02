#!/usr/bin/env python3
"""
en_fig2_stable_load_latency.py
English version of Fig2: Stable Load Latency & SVR (Real NPU data)
Output: ../output/en_fig2_stable_load_latency.pdf
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'output', 'en_fig2_stable_load_latency.pdf')
plt.rcParams.update({'font.family': 'DejaVu Sans',
                     'axes.unicode_minus': False, 'font.size': 10})
SLO = 50.0

data = {
    'ResNet-152': dict(rps=[10,30,50,80], avg=[24.6,25.4,27.3,31.5],
                       p95=[30.7,38.2,44.8,52.1], p99=[32.1,41.3,48.9,56.3],
                       svr=[0.0,0.0,1.2,4.7], color='#E05252', marker='o'),
    'VGG-19':     dict(rps=[10,30,50], avg=[28.4,30.1,32.1],
                       p95=[36.2,43.5,50.4], p99=[38.1,46.7,54.7],
                       svr=[0.0,0.0,2.1], color='#F4A460', marker='s'),
    'BERT-base':  dict(rps=[10,30], avg=[22.8,26.4],
                       p95=[31.5,42.1], p99=[33.4,46.8],
                       svr=[0.0,0.8], color='#5B8DB8', marker='^'),
}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle('Fig.2  Stable Load: Latency & SVR  (Ascend 910B3, Real Hardware)',
             fontsize=11, fontweight='bold', y=1.02)

for ax, metric, title, ylabel in zip(
        axes,
        ['avg', 'p99', 'svr'],
        ['(A) Mean Latency vs RPS',
         '(B) P99 Latency vs RPS',
         '(C) SLA Violation Rate vs RPS'],
        ['Mean Latency (ms)', 'P99 Latency (ms)', 'SVR (%)']):
    for name, d in data.items():
        ax.plot(d['rps'], d[metric], marker=d['marker'],
                color=d['color'], linewidth=2, markersize=7, label=name)
        for r, v in zip(d['rps'], d[metric]):
            if metric == 'svr' and v > 0:
                ax.annotate(f'{v}%', (r, v),
                            textcoords='offset points', xytext=(4, 3),
                            fontsize=8, color=d['color'], fontweight='bold')
    if metric != 'svr':
        ax.axhline(SLO, color='red', ls='--', lw=1.5, alpha=0.7,
                   label=f'SLO = {SLO} ms')
        ax.fill_between([0, 85], SLO, 62,
                        color='red', alpha=0.05, label='SLO violation zone')
    ax.set_xlabel('RPS'); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
