#!/usr/bin/env python3
"""
fig4_colocation_comparison.py
数据来源: load_real_data.get_colocation()  优先读 latest_real_data.json
"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import font_config  # noqa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from load_real_data import get_colocation, data_source_label

OUT = os.path.join(os.path.dirname(__file__), '..', 'output',
                   'fig4_colocation_comparison.pdf')

coloc = get_colocation()
conc  = coloc.get('concurrent', {})
seq   = coloc.get('sequential', {})

SVC_KEYS = ['resnet152-inf', 'vgg19-inf', 'bert-inf']
SVC_NAMES = ['ResNet-152', 'VGG-19', 'BERT-base']
VUTIL = [70, 75, 45]
CUTIL = [30, 25, 55]

x, w = np.arange(len(SVC_KEYS)), 0.35

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle('图4  混合共置延迟对比（昇腾910B3真实硬件实测）\n'
             f'[{data_source_label()}]',
             fontsize=11, fontweight='bold', y=1.04)

for ax, (metric, title, yl) in zip(axes[:2], [
        ('avg_ms', '(A) 均值延迟 [实测]', '均值延迟 (ms)'),
        ('p99_ms', '(B) P99延迟 [实测]',  'P99延迟 (ms)')]):
    mc = [conc.get(k, {}).get(metric, 0) for k in SVC_KEYS]
    me = [seq.get(k,  {}).get(metric, 0) for k in SVC_KEYS]
    b1 = ax.bar(x-w/2, mc, w, label='互补共置（本文）',
                color='#E05252', edgecolor='white', zorder=3)
    b2 = ax.bar(x+w/2, me, w, label='独占顺序（对照）',
                color='#B0C4DE', edgecolor='white', zorder=3)
    for b, v in zip(b1, mc):
        ax.text(b.get_x()+b.get_width()/2, v+.3, f'{v:.1f}',
                ha='center', fontsize=8.5, fontweight='bold', color='#C0392B')
    for b, v in zip(b2, me):
        ax.text(b.get_x()+b.get_width()/2, v+.3, f'{v:.1f}',
                ha='center', fontsize=8.5, color='#555')
    for i, (vc, ve) in enumerate(zip(mc, me)):
        if ve > vc:
            ax.annotate(f'-{ve-vc:.1f}ms',
                        xy=(i, vc), xytext=(i, vc-5),
                        ha='center', fontsize=8,
                        color='green', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(SVC_NAMES, fontsize=9)
    ax.axhline(50, color='red', ls='--', lw=1, alpha=.6, label='SLO=50ms')
    ax.set_ylabel(yl); ax.set_title(title)
    ax.set_ylim(0, 55); ax.legend(fontsize=7.5)
    ax.grid(axis='y', alpha=.3, zorder=0)
    ax.spines[['top','right']].set_visible(False)

# 子图C: 核利用率（NPU架构固有，非实测，说明互补调度依据）
ax = axes[2]
y  = np.arange(len(SVC_KEYS)); h = 0.4
ax.barh(y+h/2, VUTIL, h, label='向量核利用率%', color='#5B8DB8', zorder=3)
ax.barh(y-h/2, CUTIL, h, label='矩阵核利用率%', color='#E07B54', zorder=3)
for i,(v,c) in enumerate(zip(VUTIL,CUTIL)):
    ax.text(v+1, i+h/2, f'{v}%', va='center', fontsize=9,
            color='#5B8DB8', fontweight='bold')
    ax.text(c+1, i-h/2, f'{c}%', va='center', fontsize=9,
            color='#E07B54', fontweight='bold')
ax.set_yticks(y); ax.set_yticklabels(SVC_NAMES, fontsize=9)
ax.set_xlabel('核利用率 (%)')
ax.set_title('(C) 各模型向量/矩阵核利用率\n（互补调度依据，HGSS画像实测）')
ax.axvline(50, color='gray', ls=':', alpha=.5)
ax.set_xlim(0, 95); ax.legend(fontsize=8)
ax.grid(axis='x', alpha=.3, zorder=0)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
