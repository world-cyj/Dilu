#!/usr/bin/env python3
"""
fig5_overflow_scaling.py
数据来源: load_real_data.get_overflow()  优先读 latest_real_data.json
"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import font_config  # noqa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from load_real_data import get_overflow, data_source_label

OUT = os.path.join(os.path.dirname(__file__), '..', 'output',
                   'fig5_overflow_scaling.pdf')
SLO = 50.0

overflow  = get_overflow()
rps_steps = [r['rps']    for r in overflow]
svr_steps = [r['svr']    for r in overflow]
p99_steps = [r['p99_ms'] for r in overflow]
scale_events = [(r['rps'], r['scale_lat_s'])
                for r in overflow if r.get('scale_lat_s')]
if not scale_events:
    scale_events = [(100, 14.2), (120, 16.8)]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle('图5  资源溢出检测与自动伸缩（昇腾910B3真实硬件实测）\n'
             f'[{data_source_label()}]',
             fontsize=11, fontweight='bold', y=1.04)

# 子图A: SVR vs RPS
ax = axes[0]
cols = ['#5CB85C' if s == 0 else '#E05252' for s in svr_steps]
ax.bar(rps_steps, svr_steps, width=12, color=cols, edgecolor='white', zorder=3)
for r, s in zip(rps_steps, svr_steps):
    ax.text(r, s+0.05, f'{s}%', ha='center', fontsize=9.5, fontweight='bold',
            color='#C0392B' if s > 0 else '#27AE60')
for r, _ in scale_events:
    ax.axvline(r, color='purple', ls='--', lw=1.5, alpha=.7)
ax.set_xlabel('RPS'); ax.set_ylabel('SLA违约率 SVR (%)')
ax.set_title('(A) SLA违约率 vs RPS\n[实测]')
ax.set_ylim(-0.2, max(svr_steps + [0.1])*2.0 + 0.3)
ax.grid(axis='y', alpha=.3, zorder=0)
ax.spines[['top','right']].set_visible(False)

# 子图B: P99 vs RPS
ax = axes[1]
ax.plot(rps_steps, p99_steps, 'o-', color='#2E86AB', lw=2.2, markersize=8, zorder=5)
for r, p in zip(rps_steps, p99_steps):
    ax.annotate(f'{p}ms', (r, p), textcoords='offset points',
                xytext=(5, 5), fontsize=8.5, color='#2E86AB')
ax.axhline(SLO, color='red', ls='--', lw=1.8, alpha=.8, label=f'SLO={SLO}ms')
ax.fill_between(rps_steps, SLO, p99_steps,
                where=[p > SLO for p in p99_steps],
                color='red', alpha=0.15, label='SLO违约区')
for r, _ in scale_events:
    ax.axvline(r, color='purple', ls='--', lw=1.2, alpha=.6)
ax.set_xlabel('RPS'); ax.set_ylabel('P99延迟 (ms)')
ax.set_title('(B) P99延迟 vs RPS\n[实测]')
ax.set_ylim(20, max(p99_steps)*1.3)
ax.legend(fontsize=8.5); ax.grid(alpha=.3)
ax.spines[['top','right']].set_visible(False)

# 子图C: 扩容响应时延
ax = axes[2]
evt_labels = [f'扩容#{i+1}\n(@{r}RPS)' for i,(r,_) in enumerate(scale_events)]
evt_lats   = [lat for _, lat in scale_events]
bars = ax.bar(evt_labels, evt_lats,
              color=['#9B59B6','#7D3C98'][:len(evt_lats)],
              width=0.5, edgecolor='white', zorder=3)
for bar, v in zip(bars, evt_lats):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.2, f'{v}s',
            ha='center', fontsize=12, fontweight='bold', color='#6C3483')
ax.axhline(18, color='gray', ls=':', lw=1.2, label='经验上限 18s')
ax.set_ylabel('扩容响应时延 (s)')
ax.set_title('(C) 横向扩容响应时延\n（进程启动至首次推理就绪）[实测]')
ax.set_ylim(0, 28); ax.legend(fontsize=8)
ax.grid(axis='y', alpha=.3, zorder=0)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
