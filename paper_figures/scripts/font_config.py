"""font_config.py - 公共字体配置，所有绘图脚本import此模块"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 注册 WenQuanYi 字体
_wqy = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
if os.path.exists(_wqy):
    fm.fontManager.addfont(_wqy)
    plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'
else:
    # 回退：使用系统 sans-serif
    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
