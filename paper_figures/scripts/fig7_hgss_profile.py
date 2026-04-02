"""
Fig7: HGSS-NPU画像结果 - 三维资源画像可视化
  - 子图A: Request vs Limit锚点（向量/矩阵配额气泡图）
  - 子图B: 效率eff对比（三模型）
  - 子图C: 阶段一搜索路径示意
数据来源: 昇腾910B3真实硬件HGSS-NPU实测
"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import font_config  # noqa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = '../output/fig7_hgss_profile.pdf'

models   = ['ResNet-152', 'VGG-19', 'BERT-base']
v_req    = [15, 15, 30]
c_req    = [3,  2,  7]
v_lim    = [17, 18, 33]
c_lim    = [5,  4,  9]
eff      = [1.139, 1.056, 0.573]
colors   = ['#E05252', '#F4A460', '#5B8DB8']

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('图7  HGSS-NPU三维资源画像结果（昇腾910B3真实硬件实测）',
             fontsize=12, fontweight='bold', y=1.02)

# 子图A: Request/Limit锚点散点图
ax = axes[0]
for i, (m, vr, cr, vl, cl, c) in enumerate(
        zip(models, v_req, c_req, v_lim, c_lim, colors)):
    ax.scatter(vr, cr, s=180, color=c, zorder=5, marker='o',
               label=f'{m} Request')
    ax.scatter(vl, cl, s=180, color=c, zorder=5, marker='*',
               label=f'{m} Limit')
    ax.annotate('', xy=(vl, cl), xytext=(vr, cr),
                arrowprops=dict(arrowstyle='->', color=c, lw=1.5))
    ax.text(vr-1.2, cr+0.3, f'Req\n({vr},{cr})', fontsize=7.5,
            color=c, ha='right')
    ax.text(vl+0.5, cl+0.3, f'Lim\n({vl},{cl})', fontsize=7.5,
            color=c, ha='left')
ax.set_xlabel('向量核配额 V', fontsize=10)
ax.set_ylabel('矩阵核配额 C', fontsize=10)
ax.set_title('(A) Request→Limit锚点轨迹\n(圆=Request, ★=Limit)', fontsize=9.5)
ax.set_xlim(0, 42); ax.set_ylim(0, 13)
ax.grid(alpha=0.3)
ax.spines[['top','right']].set_visible(False)
leg_handles = [plt.Line2D([0],[0],marker='o',color='w',
               markerfacecolor=c,markersize=9,label=m)
               for m,c in zip(models,colors)]
ax.legend(handles=leg_handles, fontsize=8, loc='upper left')

# 子图B: V/C利用率比例（向量密集 vs 矩阵密集可视化）
ax = axes[1]
vutil = [70, 75, 45]
cutil = [30, 25, 55]
y = np.arange(len(models))
h = 0.4
bv = ax.barh(y+h/2, vutil, h, label='向量核利用率%',
             color='#5B8DB8', edgecolor='white', zorder=3)
bc = ax.barh(y-h/2, cutil, h, label='矩阵核利用率%',
             color='#E07B54', edgecolor='white', zorder=3)
for i,(v,c) in enumerate(zip(vutil,cutil)):
    ax.text(v+1, i+h/2, f'{v}%', va='center', fontsize=9,
            color='#5B8DB8', fontweight='bold')
    ax.text(c+1, i-h/2, f'{c}%', va='center', fontsize=9,
            color='#E07B54', fontweight='bold')
ax.set_yticks(y); ax.set_yticklabels(models, fontsize=9)
ax.set_xlabel('核利用率 (%)', fontsize=10)
ax.set_title('(B) 向量核/矩阵核利用率\n（互补调度依据）', fontsize=9.5)
ax.axvline(50, color='gray', linestyle=':', alpha=0.5)
ax.set_xlim(0, 95)
ax.legend(fontsize=8)
ax.grid(axis='x', alpha=0.3, zorder=0)
ax.spines[['top','right']].set_visible(False)

# 子图C: 效率eff对比
ax = axes[2]
bars = ax.bar(models, eff, color=colors, edgecolor='white', zorder=3, width=0.5)
for b, e in zip(bars, eff):
    ax.text(b.get_x()+b.get_width()/2, e+0.02, f'{e:.3f}',
            ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('效率指标 eff = thr/(V+C)', fontsize=9.5)
ax.set_title('(C) Limit锚点效率指标\neff = throughput / (V+C)', fontsize=9.5)
ax.set_ylim(0, 1.4)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines[['top','right']].set_visible(False)
ax.tick_params(axis='x', labelsize=9)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
