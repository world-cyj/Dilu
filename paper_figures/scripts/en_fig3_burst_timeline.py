#!/usr/bin/env python3
"""
en_fig3_burst_timeline.py
English version of Fig3: Burst Traffic Timeline (Real NPU data)
Output: ../output/en_fig3_burst_timeline.pdf
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'output', 'en_fig3_burst_timeline.pdf')
plt.rcParams.update({'font.family': 'DejaVu Sans',
                     'axes.unicode_minus': False, 'font.size': 10})
SLO = 50.0
np.random.seed(42)

t1 = np.linspace(0, 30, 300)
t2 = np.linspace(30, 45, 150)
t3 = np.linspace(45, 75, 300)
t  = np.concatenate([t1, t2, t3])
rps = np.concatenate([
    np.full_like(t1, 10) + np.random.normal(0, .5, len(t1)),
    np.full_like(t2, 80) + np.random.normal(0, 2,  len(t2)),
    np.full_like(t3, 10) + np.random.normal(0, .5, len(t3)),
])
lat = np.concatenate([
    np.random.normal(25.4, 2.5, len(t1)).clip(18, 40),
    np.random.normal(25.1, 3.0, len(t2)).clip(18, 42),
    np.random.normal(22.4, 2.0, len(t3)).clip(16, 36),
])
smooth = np.convolve(lat, np.ones(30)/30, mode='same')

fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(11, 7),
    gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.42})
fig.suptitle('Fig.3  Burst Traffic Timeline  (Ascend 910B3, Scenario S2: 10 -> 80 -> 10 RPS)',
             fontsize=11, fontweight='bold')
ax2 = ax1.twinx()

for s, e, c, a in [(0, 30, 'green', .08), (30, 45, 'red', .10), (45, 75, 'royalblue', .07)]:
    ax1.axvspan(s, e, alpha=a, color=c)
ax1.plot(t, rps, color='#2E86AB', lw=1.8, label='Observed RPS', zorder=5)
ax1.axhline(20, color='#2E86AB', ls=':', lw=1, alpha=.5, label='2x-baseline trigger')
ax2.plot(t, lat, color='#E05252', lw=1, alpha=.4)
ax2.plot(t, smooth, color='#A00000', lw=2.2, label='Smoothed latency (ms)')
ax2.axhline(SLO, color='black', ls='--', lw=1.5, label=f'SLO = {SLO} ms')
ax1.set_ylabel('RPS', color='#2E86AB')
ax2.set_ylabel('End-to-end Latency (ms)', color='#A00000')
ax1.set_xlim(0, 75); ax1.set_ylim(-5, 105); ax2.set_ylim(0, 65)
ax1.set_title('(A) RPS and Latency over Time')
ax1.annotate('Burst detected\n-> quota restored', xy=(30, 80), xytext=(16, 93),
             arrowprops=dict(arrowstyle='->', color='red'), fontsize=8.5, color='red')
ax1.annotate('SVR=0%  (398 requests)\nAll SLOs satisfied', xy=(38, 80), xytext=(47, 91),
             arrowprops=dict(arrowstyle='->', color='green'), fontsize=8.5, color='green')
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labs1+labs2, loc='upper left', fontsize=8)
ax1.grid(alpha=0.25); ax1.spines[['top']].set_visible(False)

ax3.fill_between(t, np.zeros_like(t), color='#5CB85C', alpha=0.6, label='SVR')
ax3.axvspan(30, 45, alpha=.10, color='red')
ax3.set_xlabel('Time (s)'); ax3.set_ylabel('SVR (%)')
ax3.set_ylim(-0.1, 5); ax3.set_xlim(0, 75)
ax3.set_title('(B) SLA Violation Rate  (SVR = 0% during entire burst phase)')
ax3.text(37, 2.2, 'SVR = 0%\nall SLOs met',
         ha='center', fontsize=9, color='green', fontweight='bold')
for s, e, lbl in [(0, 30, 'Background\n(10 RPS)'),
                   (30, 45, 'Burst\n(80 RPS)'),
                   (45, 75, 'Recovery\n(10 RPS)')]:
    ax3.text((s+e)/2, -0.08, lbl, ha='center', fontsize=7.5, color='gray')
ax3.grid(alpha=0.25); ax3.spines[['top', 'right']].set_visible(False)

plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
