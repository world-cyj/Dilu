#!/usr/bin/env python3
"""
fig2_stable_load_latency.py
数据来源: load_real_data.get_stable()  优先读 latest_real_data.json
"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import font_config  # noqa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from load_real_data import get_stable, data_source_label

OUT = os.path.join(os.path.dirname(__file__), '..', 'output',
                   'fig2_stable_load_latency.pdf')
SLO = 50.0

stable = get_stable()
SVC_CFG = {
    'resnet152-inf': ('ResNet-152', '#E05252', 'o'),
    'vgg19-inf':     ('VGG-19',     '#F4A460', 's'),
    'bert-inf':      ('BERT-base',  '#5B8DB8', '^'),
}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle('图2  稳定负载延迟与SLA违约率（昇腾910B3真实硬件实测）\n'
             f'[{data_source_label()}]',
             fontsize=11, fontweight='bold', y=1.04)

for ax, (metric, title, ylabel) in zip(axes, [
        ('avg_ms', '(A) 均值延迟 vs RPS\n[实测]', '均值延迟 (ms)'),
        ('p99_ms', '(B) P99延迟 vs RPS\n[实测]',  'P99延迟 (ms)'),
        ('svr',    '(C) SLA违约率 vs RPS\n[实测]', 'SVR (%)'),
]):
    for svc_key, (name, color, marker) in SVC_CFG.items():
        rows = stable.get(svc_key, [])
        if not rows: continue
        xs = [r['rps']      for r in rows]
        ys = [r.get(metric, 0) for r in rows]
        ax.plot(xs, ys, marker=marker, color=color,
                linewidth=2, markersize=7, label=name)
        for x, y in zip(xs, ys):
            if metric == 'svr' and y > 0:
                ax.annotate(f'{y}%', (x,y), textcoords='offset points',
                            xytext=(4,3), fontsize=8, color=color)
    if metric != 'svr':
        ax.axhline(SLO, color='red', ls='--', lw=1.5, alpha=.7,
                   label=f'SLO={SLO}ms')
        ax.fill_between([0,85], SLO, SLO+15, color='red', alpha=.05,
                        label='SLO违约区')
    ax.set_xlabel('RPS'); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(fontsize=8)
    ax.grid(alpha=.3); ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
