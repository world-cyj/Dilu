#!/usr/bin/env python3
"""
en_fig5_overflow_scaling.py
English version of Fig5: Overflow Auto-scaling (Real NPU data)
Output: ../output/en_fig5_overflow_scaling.pdf
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'output', 'en_fig5_overflow_scaling.pdf')
plt.rcParams.update({'font.family': 'DejaVu Sans',
                     'axes.unicode_minus': False, 'font.size': 10})
SLO = 50.0

rps_steps = [20,   60,   100,  120]
svr_steps = [0.0,  0.0,  0.3,  1.8]
p99_steps = [37.2, 38.8, 51.4, 58.2]
scale_lat = [14.2, 16.8]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle('Fig.5  Overflow Detection & Auto-scaling  (Ascend 910B3, Scenario S4)',
             fontsize=11, fontweight='bold', y=1.02)

# (A) SVR vs RPS
ax = axes[0]
cols = ['#5CB85C','#5CB85C','#E05252','#C0392B']
ax.bar(rps_steps, svr_steps, width=12, color=cols, edgecolor='white', zorder=3)
for r, s in zip(rps_steps, svr_steps):
    ax.text(r, s+0.05, f'{s}%', ha='center', fontsize=9.5, fontweight='bold',
            color='#C0392B' if s > 0 else '#27AE60')
for r, lbl in zip([100, 120], ['Scale-out #1', 'Scale-out #2']):
    ax.axvline(r, color='purple', ls='--', lw=1.5, alpha=.7)
    ax.text(r+1.5, 1.5, lbl, fontsize=8, color='purple', rotation=90)
ax.set_xlabel('RPS'); ax.set_ylabel('SLA Violation Rate SVR (%)')
ax.set_title('(A) SVR vs RPS')
ax.set_ylim(-0.2, 3.0); ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines[['top', 'right']].set_visible(False)

# (B) P99 vs RPS
ax = axes[1]
ax.plot(rps_steps, p99_steps, 'o-', color='#2E86AB', lw=2.2, markersize=8, zorder=5)
for r, p in zip(rps_steps, p99_steps):
    ax.annotate(f'{p}ms', (r, p), textcoords='offset points',
                xytext=(5, 5), fontsize=8.5, color='#2E86AB')
ax.axhline(SLO, color='red', ls='--', lw=1.8, alpha=.8, label=f'SLO={SLO}ms')
ax.fill_between(rps_steps, SLO, p99_steps,
                where=[p > SLO for p in p99_steps],
                color='red', alpha=0.15, label='SLO violation zone')
for r, lbl in zip([100, 120], ['Scale-out #1', 'Scale-out #2']):
    ax.axvline(r, color='purple', ls='--', lw=1.2, alpha=.6)
ax.set_xlabel('RPS'); ax.set_ylabel('P99 Latency (ms)')
ax.set_title('(B) P99 Latency vs RPS')
ax.set_ylim(25, 68); ax.legend(fontsize=8.5)
ax.grid(alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

# (C) Scale-out response latency
ax = axes[2]
events = ['Scale-out #1\n(@100 RPS)', 'Scale-out #2\n(@120 RPS)']
bars = ax.bar(events, scale_lat, color=['#9B59B6', '#7D3C98'],
              width=0.5, edgecolor='white', zorder=3)
for bar, v in zip(bars, scale_lat):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.2, f'{v}s',
            ha='center', fontsize=13, fontweight='bold', color='#6C3483')
ax.axhline(18, color='gray', ls=':', lw=1.2, label='Upper bound 18s')
ax.set_ylabel('Scale-out Response Latency (s)')
ax.set_title('(C) Scale-out Response Latency\n(process start to first healthy inference)')
ax.set_ylim(0, 25); ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
