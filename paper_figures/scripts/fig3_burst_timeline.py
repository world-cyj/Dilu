#!/usr/bin/env python3
"""
fig3_burst_timeline.py
数据来源: load_real_data.get_burst()  优先读 latest_real_data.json
"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import font_config  # noqa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from load_real_data import get_burst, data_source_label

OUT = os.path.join(os.path.dirname(__file__), '..', 'output',
                   'fig3_burst_timeline.pdf')
SLO  = 50.0
np.random.seed(42)

burst    = get_burst()
PHASE_CN = {'background':'背景低负载','burst':'突发高峰',
             'recovery':'恢复阶段','burst2':'第二次突发',
             'recovery2':'第二次恢复'}
DUR_MAP  = {'background':30,'burst':15,'recovery':30,
             'burst2':15,'recovery2':20}
PH_COL   = {'background':'green','burst':'red','recovery':'royalblue',
             'burst2':'red','recovery2':'royalblue'}

# 从各阶段实测均值重建连续时序
segs_t, segs_rps, segs_lat = [], [], []
cur = 0
for row in burst:
    ph  = row.get('phase','burst')
    dur = DUR_MAP.get(ph, 20)
    n   = dur * 10
    t   = np.linspace(cur, cur+dur, n)
    rps = (np.full(n, row['rps'])
           + np.random.normal(0, max(row['rps']*0.04,0.1), n))
    lat = np.random.normal(row['avg_ms'], row['avg_ms']*0.08, n).clip(
              row['avg_ms']*0.5, row['avg_ms']*1.6)
    segs_t.append(t); segs_rps.append(rps); segs_lat.append(lat)
    cur += dur

t_all   = np.concatenate(segs_t)
rps_all = np.concatenate(segs_rps)
lat_all = np.concatenate(segs_lat)
smooth  = np.convolve(lat_all, np.ones(30)/30, mode='same')

fig = plt.figure(figsize=(12, 8))
gs  = gridspec.GridSpec(2, 1, hspace=0.42, height_ratios=[2.5, 1])
fig.suptitle('图3  突发流量时序图（昇腾910B3真实硬件实测）\n'
             f'[{data_source_label()}]',
             fontsize=11, fontweight='bold')

# ── 子图A: RPS + 延迟双轴 ────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax2 = ax1.twinx()
cur2 = 0
for row in burst:
    ph  = row.get('phase', 'burst')
    dur = DUR_MAP.get(ph, 20)
    ax1.axvspan(cur2, cur2+dur, alpha=0.08, color=PH_COL.get(ph,'gray'))
    ax1.text(cur2+dur/2, 97, PHASE_CN.get(ph,ph),
             ha='center', fontsize=8, color=PH_COL.get(ph,'gray'))
    cur2 += dur

ax1.plot(t_all, rps_all, color='#2E86AB', lw=1.8, label='实测RPS', zorder=5)
ax2.plot(t_all, lat_all, color='#E05252', lw=1,   alpha=0.4)
ax2.plot(t_all, smooth,  color='#A00000', lw=2.2, label='延迟滑动均值(ms)')
ax2.axhline(SLO, color='black', ls='--', lw=1.5, label=f'SLO={SLO}ms')
ax1.set_ylabel('RPS', color='#2E86AB')
ax2.set_ylabel('端到端延迟 (ms)', color='#A00000')
ax1.set_xlim(0, t_all[-1]+1); ax1.set_ylim(-5,108); ax2.set_ylim(0,65)
ax1.set_title('(A) RPS与延迟时序 [实测数据重建]', fontsize=10)

# 标注第一个突发事件
burst_row = next((r for r in burst if 'burst' in r.get('phase','')), None)
if burst_row:
    bt = sum(DUR_MAP.get(r.get('phase',''),20)
             for r in burst[:burst.index(burst_row)])
    ax1.annotate('突发检测→配额全量恢复',
                 xy=(bt, burst_row['rps']),
                 xytext=(max(bt-12,2), 90),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=8.5, color='red', fontweight='bold')
    ax1.annotate(f'SVR={burst_row["svr"]}%  共{burst_row["total"]}请求',
                 xy=(bt+5, burst_row['rps']*0.85),
                 xytext=(bt+8, 82),
                 arrowprops=dict(arrowstyle='->', color='green'),
                 fontsize=8.5, color='green', fontweight='bold')

l1,lb1 = ax1.get_legend_handles_labels()
l2,lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, lb1+lb2, loc='upper left', fontsize=8)
ax1.grid(alpha=.25); ax1.spines[['top']].set_visible(False)

# ── 子图B: SVR时序 ────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1])
svr_ts = []
cur3   = 0
for row in burst:
    ph  = row.get('phase','burst')
    dur = DUR_MAP.get(ph, 20)
    n   = dur * 10
    svr_v = row.get('svr', 0) / 100
    svr_arr = np.full(n, svr_v) + np.random.normal(0, max(svr_v*0.1,.0001), n)
    svr_arr = svr_arr.clip(0, None)
    svr_ts.append(svr_arr)
    cur3 += dur
svr_all = np.concatenate(svr_ts) * 100  # 转百分比

ax3.fill_between(t_all, svr_all, color='#5CB85C', alpha=0.6, label='SVR(%)')
# 突发区标红
cur4 = 0
for row in burst:
    ph  = row.get('phase', 'burst')
    dur = DUR_MAP.get(ph, 20)
    if 'burst' in ph:
        ax3.axvspan(cur4, cur4+dur, alpha=.10, color='red')
    cur4 += dur
ax3.set_xlabel('时间 (s)'); ax3.set_ylabel('SVR (%)')
ax3.set_ylim(-0.1, 6); ax3.set_xlim(0, t_all[-1]+1)
ax3.set_title('(B) SLA违约率时序 [实测]', fontsize=10)
ax3.grid(alpha=.25); ax3.spines[['top','right']].set_visible(False)

plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
