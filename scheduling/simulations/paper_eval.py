"""
paper_eval.py  --  论文级综合评估与可视化
==================================================
生成论文所需的完整图表集。

Usage:
  cd /mnt/caoyujia/Dilu
  python3 scheduling/simulations/paper_eval.py --workload scheduling/simulations/workload/instances-npu-100.txt
"""

import ast
import csv
import importlib
import json
import os
import sys
from datetime import datetime
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

ROOT     = os.path.dirname(os.path.dirname(
               os.path.dirname(os.path.abspath(__file__))))
SIM_DIR  = os.path.join(ROOT, 'scheduling', 'simulations')
PROF_DIR = os.path.join(ROOT, 'profiling', 'inference', 'observations')
OUT_DIR  = os.path.join(ROOT, 'paper_figures')
sys.path.insert(0, SIM_DIR)


# ── helpers ───────────────────────────────────────────────────────────────
def _load_profile(model):
    """Load npu_profile_*.csv or HGSS _hgss_npu.csv."""
    for path in [
        os.path.join(PROF_DIR, 'results', f'npu_profile_{model}.csv'),
        os.path.join(PROF_DIR, 'profiling_results_npu', f'{model}_hgss_npu.csv'),
    ]:
        if os.path.exists(path):
            rows = []
            with open(path, newline='') as f:
                for row in csv.DictReader(f):
                    rows.append({k: (float(v) if k != 'model' else v)
                                 for k, v in row.items()})
            return rows
    return []


def _run_scheduler(mod_name, wl_path):
    for k in list(sys.modules):
        if mod_name in k:
            del sys.modules[k]
    mod = importlib.import_module(mod_name)
    from baseline.scheduler_dilu_npu import PortManager
    mod.port_manager = PortManager()
    mod.new_npus     = [mod.NPU(i, n['ip'], n['index'])
                        for i, n in enumerate(mod.nodes_info)]
    mod.active_npus  = []
    with open(wl_path) as f:
        events = [ast.literal_eval(l.strip()) for l in f]
    base = datetime.strptime(events[0]['Time'], '%Y-%m-%d %H:%M:%S')
    ts, counts = [], []
    for ev in events:
        t  = datetime.strptime(ev['Time'], '%Y-%m-%d %H:%M:%S')
        td = (t - base).total_seconds() / 60.0
        if ev['Action'] == 'start':
            mod.schedule_instance(ev['Instance'])
        else:
            mod.delete_instance(ev['Instance'])
        ts.append(td)
        counts.append(len(mod.active_npus))
    vf, cf, mf = mod.calc_fragmentation()
    return ts, counts, vf, cf, mf


# ── Fig 1: 3D response surface ────────────────────────────────────────────
def fig_response_surface(model='resnet152'):
    rows = _load_profile(model)
    if not rows or not HAS_MPL:
        print(f'[Fig1] skipped (no data for {model})')
        return
    vkey  = 'vector' if 'vector' in rows[0] else 'vector_req'
    ckey  = 'cube'   if 'cube'   in rows[0] else 'cube_req'
    bkey  = 'batch_size' if 'batch_size' in rows[0] else 'bs'
    bs1   = [r for r in rows if int(r.get(bkey, 1)) == 1] or rows[:30]
    vs    = [r[vkey] for r in bs1]
    cs    = [r[ckey] for r in bs1]
    thrs  = [r.get('throughput', 100) for r in bs1]
    lats  = [r.get('latency_s',  0.05) for r in bs1]

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(f'NPU Resource Response Surface ({model.upper()}, BS=1)',
                 fontsize=12, fontweight='bold')
    ax1 = fig.add_subplot(121, projection='3d')
    sc  = ax1.scatter(vs, cs, thrs, c=thrs, cmap='RdYlGn', s=50)
    ax1.set_xlabel('Vector Cores'); ax1.set_ylabel('Cube Cores')
    ax1.set_zlabel('Throughput'); ax1.set_title('Throughput')
    plt.colorbar(sc, ax=ax1, shrink=0.5)

    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(vs, cs, lats, c=lats, cmap='RdYlGn_r', s=50)
    ax2.set_xlabel('Vector Cores'); ax2.set_ylabel('Cube Cores')
    ax2.set_zlabel('Latency (s)'); ax2.set_title('Latency')
    plt.colorbar(sc2, ax=ax2, shrink=0.5)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'fig1_surface_{model}.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[Fig1] {out}')


# ── Fig 2: marginal utility ───────────────────────────────────────────────
def fig_marginal(model='resnet152'):
    rows = _load_profile(model)
    if not rows or not HAS_MPL:
        return
    vkey = 'vector' if 'vector' in rows[0] else 'vector_req'
    ckey = 'cube'   if 'cube'   in rows[0] else 'cube_req'
    bkey = 'batch_size' if 'batch_size' in rows[0] else 'bs'

    v_pts = sorted(set(r[vkey] for r in rows if int(r.get(bkey,1))==1))
    c_pts = sorted(set(r[ckey] for r in rows if int(r.get(bkey,1))==1))
    mid_c = c_pts[len(c_pts)//2]
    mid_v = v_pts[len(v_pts)//2]

    v_data = sorted([(r[vkey], r.get('throughput',100))
                     for r in rows
                     if int(r.get(bkey,1))==1
                     and abs(r[ckey]-mid_c)<=2], key=lambda x: x[0])
    c_data = sorted([(r[ckey], r.get('throughput',100))
                     for r in rows
                     if int(r.get(bkey,1))==1
                     and abs(r[vkey]-mid_v)<=5], key=lambda x: x[0])

    if not v_data or not c_data:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Marginal Utility ({model.upper()})', fontsize=12, fontweight='bold')
    for ax, data, xlabel, title, color in [
        (axes[0], v_data, 'Vector Cores', f'Vector (Cube={mid_c})', 'blue'),
        (axes[1], c_data, 'Cube Cores',   f'Cube (Vector={mid_v})', 'red'),
    ]:
        xs, ys = zip(*data)
        ax.plot(xs, ys, f'{color[0]}-o', linewidth=2, markersize=7)
        ax.fill_between(xs, ys, alpha=0.12, color=color)
        if len(ys) > 2:
            diffs = [ys[i+1]-ys[i] for i in range(len(ys)-1)]
            knee  = xs[diffs.index(min(diffs))]
            ax.axvline(knee, color='orange', linestyle='--',
                       label=f'Diminishing return @ {knee}')
        ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel('Throughput (req/s)')
        ax.set_title(title); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'fig2_marginal_{model}.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[Fig2] {out}')


# ── Fig 3: baseline comparison ────────────────────────────────────────────
def fig_baseline(wl_path):
    if not HAS_MPL:
        return
    specs = [
        ('K8s',       'baseline.scheduler_k8s_npu',      'red',   ':'),
        ('INFless-L', 'baseline.scheduler_infless_l_npu', 'gold',  '--'),
        ('INFless-R', 'baseline.scheduler_infless_r_npu', 'blue',  '-.'),
        ('Dilu-NPU',  'baseline.scheduler_dilu_npu',      'green', '-'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Dilu-NPU vs Baselines', fontsize=12, fontweight='bold')
    peaks, frags = {}, {}
    for lbl, mod, col, ls in specs:
        try:
            ts, cnt, vf, cf, mf = _run_scheduler(mod, wl_path)
            axes[0].plot(ts, cnt, color=col, linestyle=ls,
                         linewidth=1.8, label=lbl)
            peaks[lbl] = max(cnt)
            frags[lbl] = (vf, cf, mf)
        except Exception as e:
            print(f'[Fig3] {lbl}: {e}')
    axes[0].set_xlabel('Time (min)'); axes[0].set_ylabel('Active NPU Cards')
    axes[0].set_title('Active Card Count'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    lbls = list(frags)
    x, w = np.arange(len(lbls)), 0.25
    axes[1].bar(x-w, [frags[l][0] for l in lbls], w, label='Vector', color='#3498db')
    axes[1].bar(x,   [frags[l][1] for l in lbls], w, label='Cube',   color='#e74c3c')
    axes[1].bar(x+w, [frags[l][2] for l in lbls], w, label='Memory', color='#2ecc71')
    axes[1].set_xticks(x); axes[1].set_xticklabels(lbls, fontsize=8)
    axes[1].set_ylabel('Fragmentation Rate'); axes[1].set_title('Fragmentation')
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig3_baseline.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[Fig3] {out}')
    if 'K8s' in peaks and 'Dilu-NPU' in peaks:
        print(f'[Fig3] Dilu-NPU saves '
              f'{(peaks["K8s"]-peaks["Dilu-NPU"])/peaks["K8s"]*100:.1f}% peak NPUs')


# ── Fig 4: ablation ───────────────────────────────────────────────────────
def fig_ablation(wl_path):
    from ablation_npu import run_variant, plot_ablation, save_summary
    variants = [
        ('Full',     dict(disable_vs=False, disable_wa=False, disable_rc=False)),
        ('-VS',      dict(disable_vs=True,  disable_wa=False, disable_rc=False)),
        ('-WA',      dict(disable_vs=False, disable_wa=True,  disable_rc=False)),
        ('-RC',      dict(disable_vs=False, disable_wa=False, disable_rc=True)),
        ('-VS-WA-RC',dict(disable_vs=True,  disable_wa=True,  disable_rc=True)),
    ]
    results = {}
    for lbl, kw in variants:
        m = run_variant(f'Dilu-NPU {lbl}', wl_path, **kw)
        results[f'Dilu-NPU {lbl}'] = m
    plot_ablation(results, OUT_DIR)
    save_summary(results, OUT_DIR)
    src = os.path.join(OUT_DIR, 'ablation_comparison.png')
    dst = os.path.join(OUT_DIR, 'fig4_ablation.png')
    if os.path.exists(src) and src != dst:
        os.rename(src, dst)
    print(f'[Fig4] {dst}')


# ── main ──────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload',
        default=os.path.join(SIM_DIR, 'workload', 'instances-npu-100.txt'))
    parser.add_argument('--model', default='resnet152')
    parser.add_argument('--figs', nargs='+', type=int,
        default=[1, 2, 3, 4],
        help='Which figures to generate (1=surface 2=marginal 3=baseline 4=ablation)')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'[PaperEval] Output dir: {OUT_DIR}')
    print(f'[PaperEval] Workload  : {args.workload}')

    if 1 in args.figs:
        fig_response_surface(args.model)
    if 2 in args.figs:
        fig_marginal(args.model)
    if 3 in args.figs:
        fig_baseline(args.workload)
    if 4 in args.figs:
        fig_ablation(args.workload)

    print(f'\n[PaperEval] All figures saved to {OUT_DIR}/')
    figs = [f for f in os.listdir(OUT_DIR) if f.endswith('.png')]
    for f in sorted(figs):
        print(f'  {f}')


if __name__ == '__main__':
    main()
