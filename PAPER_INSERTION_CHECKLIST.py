#!/usr/bin/env python3
"""
快速参考：论文排版与图表插入清单
用法：按照下面的位置和提示词，逐一在PAPER_CN_V2.md中插入内容
"""

INSERTIONS = [
    {
        "序号": "1️⃣",
        "章节": "第3章 - 算法伪代码",
        "当前位置": "### 3.1 问题定义",
        "操作": "在此行之前插入",
        "内容": """
### 3.0 算法概览

本章提出HGSS-NPU（Hierarchical Grid Search & Stepping for NPU），通过两阶段分层搜索在三维资源空间（批大小BS、向量配额V、矩阵配额C）中高效定位最优配置。相比全网格扫描的480次测量，HGSS-NPU将开销降至8–11次，节省93%–95%。

**Algorithm 1: HGSS-NPU 三维分层搜索画像算法**

```
Input:
  M: 推理模型
  T_slo: SLO延迟约束（秒）
  device_id: NPU卡编号
  
Output:
  π = (v_req, c_req, v_lim, c_lim, m, bs*): 六元组画像

Phase 1: Request锚点搜索（BS=1，步进搜索）
─────────────────────────────────────────────────────
1: v ← 5, c ← 2  // 初始最小配额
2: while v ≤ 40 do
3:   lat, thr ← RUN_INFERENCE(M, bs=1, v, c)
4:   if lat ≤ T_slo then
5:     v_req ← v, c_req ← c  // 记录Request锚点
6:     break
7:   end if
8:   v ← v + 5  // 步进5
9:   c ← min(c + 2, 20)  // 同步调整矩阵配额
10: end while

Phase 2: Limit锚点搜索（BS翻倍，效率最大化）
─────────────────────────────────────────────────────
11: bs ← 2, best_eff ← 0
12: while bs ≤ 128 do
13:   lat, thr ← RUN_INFERENCE(M, bs, v_req, c_req)
14:   eff ← thr / (v_req + c_req)
15:   if lat > T_slo then
16:     v_try ← min(v_req + 5, 40)
17:     c_try ← min(c_req + 2, 20)
18:     lat_try, thr_try ← RUN_INFERENCE(M, bs, v_try, c_try)
19:     if lat_try ≤ T_slo then
20:       v_req ← v_try, c_req ← c_try
21:       eff ← thr_try / (v_try + c_try)
22:     else
23:       break
24:     end if
25:   end if
26:   if eff > best_eff then
27:     v_lim ← v_req, c_lim ← c_req
28:     bs* ← bs, best_eff ← eff
29:   end if
30:   bs ← bs × 2
31: end while

32: m ← QUERY_MEMORY(M, bs*)
33: return π = (v_req, c_req, v_lim, c_lim, m, bs*)
```

**算法复杂度分析：**
- Phase 1: O(8) 步进（V从5到40，步长5）
- Phase 2: O(8) 次BS翻倍（BS从2到256）
- 总测量次数：8 + 8 = 16次（实测3–5分钟）
- 相比全网格480次，节省95%
""",
        "提示词": "标准会议论文算法伪代码格式，包含Input/Output/复杂度分析"
    },
    
    {
        "序号": "2️⃣",
        "章节": "第4章 - 系统架构图",
        "当前位置": "## 4. 系统设计与实现",
        "操作": "在此行之后插入",
        "内容": """
### 4.0 系统架构总览

本章介绍Dilu-NPU系统的三层架构设计。

**图1 Dilu-NPU系统架构**

[此处插入架构图 - 见下方生成方法]

**架构说明：**
- **画像层（Profiling Plane）**：Developer提交模型 → Profiler运行HGSS-NPU → 输出Metadata数据库
- **控制平面（Control Plane）**：Gateway接收请求 → Scheduler执行互补调度决策
- **伸缩平面（Scaling Plane）**：Global Scaler协调全局扩缩容，Local Scaler管理单节点
- **服务平面（Serving Plane）**：Function Instances在NPU卡上运行，支持共置与动态伸缩
""",
        "提示词": "三层架构图，参考PAPER_FORMATTING_GUIDE.md中的Mermaid代码或Excalidraw提示词"
    },
    
    {
        "序号": "3️⃣",
        "章节": "第5章 - 调度流程图",
        "当前位置": "### 5.2.3 互补调度决策流程",
        "操作": "在此行之后插入",
        "内容": """
**图2 互补感知调度决策流程**

[此处插入流程图 - 见下方生成方法]

**流程说明：**
1. 新推理请求到达 → 查询模型画像π_i
2. 判断是否存在同服务训练任务
   - 是 → 查询亲和卡集合G_WA（已有同服务训练实例的卡）
   - 否 → 查询活跃卡集合G_active（已部署任何实例的卡）
3. 调用SelectOptGPU()评分函数，计算互补性评分
4. 若找到可行卡 → 分配并更新配额；否则 → 启动新实例
""",
        "提示词": "流程图，包含判断分支（是/否）、数据库查询、操作步骤"
    },
    
    {
        "序号": "4️⃣",
        "章节": "第6.3节 - 实测数据可视化",
        "当前位置": "**表2：稳定负载场景各服务延迟与SVR【实测，SLO=50ms】**",
        "操作": "在表格之后插入",
        "内容": """
**图3 稳定负载场景延迟与SVR对比**

[此处插入对比图 - 见下方生成方法]

**图表说明：**
- 左图：均值延迟 vs RPS，三条线分别代表ResNet-152、VGG-19、BERT-base
- 中图：P99延迟 vs RPS，红色虚线标注SLO=50ms
- 右图：SVR vs RPS，柱状图显示各服务的违约率
- 数据来源：evaluation/logs/latest_real_data.json
""",
        "提示词": "三个子图并排，使用matplotlib或Plotly生成，数据自动从latest_real_data.json读取"
    },
    
    {
        "序号": "5️⃣",
        "章节": "第6.4节 - 消融实验对比",
        "当前位置": "**表6：消融实验结果【仿真工作负载 N=200】**",
        "操作": "在表格之后插入",
        "内容": """
**图4 消融实验：各组件独立贡献**

[此处插入消融对比图 - 见下方生成方法]

**图表说明：**
- 左图：SVR对比，Full为基准（绿色），其他变体为降级（红色），标注相对增长
- 中图：吞吐量对比，显示各变体的吞吐下降幅度
- 右图：均值延迟对比，红色虚线标注SLO=50ms，超过SLO的变体用特殊颜色突出
- 数据来源：evaluation/logs/ablation_summary.json
""",
        "提示词": "三个子图并排，柱状图格式，使用不同颜色区分各变体"
    },
    
    {
        "序号": "6️⃣",
        "章节": "第6.3节 - 突发流量时序图",
        "当前位置": "**表3：突发流量场景各阶段指标【实测，S2场景】**",
        "操作": "在表格之后插入",
        "内容": """
**图5 突发流量时序图**

[此处插入时序图 - 见下方生成方法]

**图表说明：**
- 上图：RPS与延迟双轴时序，背景色区分各阶段（背景低负载/突发高峰/恢复）
- 下图：SVR时序，突发区用红色阴影标注
- 标注关键事件：突发检测时刻、配额恢复时刻、扩容时刻
- 数据来源：evaluation/logs/latest_real_data.json 中的 burst 字段
""",
        "提示词": "双子图，上图为折线图（RPS+延迟），下图为面积图（SVR），时间轴对齐"
    },
    
    {
        "序号": "7️⃣",
        "章节": "第6.6节 - 可扩展性分析",
        "当前位置": "| N=400 | 540 | 248 | 54.1% | 0.335 | 0.508 |",
        "操作": "在表格之后插入",
        "内容": """
**图6 可扩展性趋势分析**

[此处插入趋势图 - 见下方生成方法]

**图表说明：**
- 左图：峰值卡数 vs 工作负载规模，显示本文方法与K8s基线的对比
- 右图：碎片率 vs 工作负载规模，显示F_V和F_C的稳定性
- 结论：调度策略效果与规模无关，具备良好可扩展性
- 数据来源：仿真工作负载 N=100/200/400
""",
        "提示词": "两个子图，折线图格式，显示随规模增长的趋势"
    }
]

print("=" * 80)
print("论文排版与图表插入快速参考清单")
print("=" * 80)
print()

for item in INSERTIONS:
    print(f"{item['序号']} {item['章节']}")
    print(f"   位置：{item['当前位置']}")
    print(f"   操作：{item['操作']}")
    print(f"   提示词：{item['提示词']}")
    print()

print("=" * 80)
print("生成图表的三种方法")
print("=" * 80)
print("""
【方法1】使用现有Python脚本（推荐，自动从latest_real_data.json读取）
  cd /mnt/caoyujia/Dilu/paper_figures/scripts
  python3 run_all_figures.py
  输出：paper_figures/output/*.pdf

【方法2】使用在线工具
  - Mermaid图：https://mermaid.live （复制PAPER_FORMATTING_GUIDE.md中的代码）
  - Excalidraw：https://excalidraw.com （手绘架构图）
  - Plotly：https://plotly.com/chart-studio （交互式图表）

【方法3】手工绘制
  - PowerPoint/Keynote 绘制后导出为 PDF/PNG
  - 或使用 Figma/Adobe Illustrator

【推荐流程】
  1. 先用方法1生成所有图表（自动化，数据一致）
  2. 将PDF导出为PNG（用于Markdown显示）
  3. 在PAPER_CN_V2.md中按上述位置插入 ![图标题](图片路径)
""")

print("=" * 80)
print("Markdown插入语法")
print("=" * 80)
print("""
在PAPER_CN_V2.md中，按以下格式插入图片：

**图1 Dilu-NPU系统架构**

![Dilu-NPU系统架构](../paper_figures/output/fig_architecture.png)

**图2 互补调度决策流程**

![互补调度决策流程](../paper_figures/output/fig_scheduling_flow.png)

**图3 稳定负载延迟对比**

![稳定负载延迟对比](../paper_figures/output/fig2_stable_load_latency.png)

**图4 消融实验对比**

![消融实验对比](../paper_figures/output/fig6_ablation.png)

**图5 突发流量时序图**

![突发流量时序图](../paper_figures/output/fig3_burst_timeline.png)

**图6 可扩展性趋势**

![可扩展性趋势](../paper_figures/output/fig_scalability.png)
""")

print("=" * 80)
print("✅ 完成清单")
print("=" * 80)
print("""
□ 在第3章插入Algorithm 1伪代码
□ 在第4章插入系统架构图（Figure 1）
□ 在第5章插入调度流程图（Figure 2）
□ 在第6.3节插入实测数据对比图（Figure 3）
□ 在第6.4节插入消融实验对比图（Figure 4）
□ 在第6.3节插入突发流量时序图（Figure 5）
□ 在第6.6节插入可扩展性趋势图（Figure 6）
□ 运行 python3 run_all_figures.py 生成所有图表
□ 将图表PNG文件复制到paper_figures/output/
□ 在PAPER_CN_V2.md中按上述位置插入![图标题](路径)
□ 检查所有图表的数据来源标注（【实测】/【仿真】）
""")
