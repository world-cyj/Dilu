#!/usr/bin/env python3
"""
run_all_figures.py - 一键生成论文所有图表

用法:
  cd /mnt/caoyujia/Dilu/paper_figures/scripts
  python3 run_all_figures.py

输出: ../output/fig{1-7}_*.pdf
"""
import subprocess
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 1. 直接指定 fc-list 返回的字体文件绝对路径
FONT_PATH = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
zh_font = FontProperties(fname=FONT_PATH)

# 2. 修正负号显示问题
plt.rcParams['axes.unicode_minus'] = False
# ──── 添加以下两行解决中文乱码 ────
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 使用微软雅黑，Windows自带
plt.rcParams['axes.unicode_minus'] = False           # 解决坐标轴负号显示为方块的问题
# ────────────────────────────────
scripts = [
    ('fig1_scheduler_comparison.py',  '图1: 五种调度器资源效率对比'),
    ('fig2_stable_load_latency.py',   '图2: 稳定负载延迟与SVR（真实NPU）'),
    ('fig3_burst_timeline.py',        '图3: 突发流量时序图（真实NPU）'),
    ('fig4_colocation_comparison.py', '图4: 混合共置延迟对比（真实NPU）'),
    ('fig5_overflow_scaling.py',      '图5: 资源溢出自动伸缩（真实NPU）'),
    ('fig6_ablation.py',              '图6: 消融实验各组件贡献'),
    ('fig7_hgss_profile.py',          '图7: HGSS-NPU画像结果（真实NPU）'),
]

here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(here, '..', 'output'), exist_ok=True)

ok, fail = [], []
for script, desc in scripts:
    print(f'\n[{script}] {desc}')
    path = os.path.join(here, script)
    r = subprocess.run([sys.executable, path], capture_output=True,
                       text=True, cwd=here)
    if r.returncode == 0:
        print(f'  OK  {r.stdout.strip()}')
        ok.append(script)
    else:
        print(f'  FAIL\n{r.stderr[-400:]}')
        fail.append(script)

print(f'\n=== 完成 {len(ok)}/{len(ok)+len(fail)} 张图 ===')
for s in ok:
    print(f'  [OK]   {s}')
for s in fail:
    print(f'  [FAIL] {s}')
