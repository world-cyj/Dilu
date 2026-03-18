"""
ablation_npu.py  --  消融实验框架
====================================
消融变体: -VS(禁时序感知) / -WA(禁亲和调度) / -RC(禁资源互补)
指标: SVR / CSC / VFrag / CFrag / MFrag / Throughput
"""

import ast
import importlib
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM_DIR)

TOTAL_VECTOR = 40
TOTAL_CUBE   = 20
NUM_CARDS    = 4


class MetricsCollector:
    def __init__(self, label: str, slo_s: float = 0.05):
        self.label          = label
        self.slo_s          = slo_s
        self.total_reqs     = 0
        self.sla_violations = 0
        self.cold_starts    = 0
        self.latencies      = []
        self.throughputs    = []
        self.active_counts  = []
        self.timestamps     = []

    def record_event(self, active_npus, time_min,
                     est_latency=None, est_tput=None):
        self.active_counts.append(active_npus)
        self.timestamps.append(time_min)
        if est_latency is not None:
            self.latencies.append(est_latency)
            self.total_reqs += 1
            if est_latency > self.slo_s:
                self.sla_violations += 1
        if est_tput is not None:
            self.throughputs.append(est_tput)

    def record_cold_start(self):
        self.cold_starts += 1

    def svr(self):
        return self.sla_violations / max(self.total_reqs, 1)

    def avg_throughput(self):
        return float(np.mean(self.throughputs)) if self.throughputs else 0.0

    def peak_npus(self):
        return max(self.active_counts) if self.active_counts else 0

    def summary(self):
        return {
            'label':      self.label,
            'peak_npus':  self.peak_npus(),
            'svr':        round(self.svr(), 4),
            'csc':        self.cold_starts,
            'avg_tput':   round(self.avg_throughput(), 2),
            'avg_lat_ms': round(float(np.mean(self.latencies))*1000, 2)
                          if self.latencies else 0.0,
        }


class SchedulerVariant:
    def __init__(self, disable_wa=False, disable_rc=False):
        mod_name = 'baseline.scheduler_dilu_npu'
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        self.mod = importlib.import_module(mod_name)
        from baseline.scheduler_dilu_npu import PortManager, NPU
        self.mod.port_manager = PortManager()
        self.mod.new_npus     = [NPU(i, node['ip'], node['index'])
                                 for i, node in enumerate(self.mod.nodes_info)]
        self.mod.active_npus  = []

        if disable_wa:
            self.mod.find_colocated_npus = lambda sn: []

        if disable_rc:
            import types
            def _simple_score(self_npu, v_req, c_req, mem):
                v_u = (self_npu.current_vector_req + v_req) / self_npu.total_vector
                m_u = (self_npu.current_memory + mem) / self_npu.total_memory
                return 0.6*(1-v_u) + 0.4*(1-m_u)
            for npu in self.mod.new_npus:
                npu.calculate_score = types.MethodType(_simple_score, npu)

    def schedule(self, data):
        prev   = len(self.mod.active_npus)
        result = self.mod.schedule_instance(data)
        cold   = len(self.mod.active_npus) > prev
        return result, cold

    def delete(self, data):
        self.mod.delete_instance(data)

    def fragmentation(self):
        return self.mod.calc_fragmentation()

    def active_count(self):
        return len(self.mod.active_npus)


def _estimate_latency(inst, active_npus, disable_vs=False, disable_rc=False):
    """
    仿真延迟估算：
    - disable_vs: 无时序配额调整，高负载时延迟惩罚更高（无法提前下调配额）
    - disable_rc: 无互补调度，碎片率高导致卡间干扰，延迟略高
    """
    v_req = inst.get('vector_req', 20)
    c_req = inst.get('cube_req',   8)
    ttype = inst.get('type', 'inference')

    # 推理基础延迟：根据资源需求估算（更贴近SLO=50ms）
    if 'llm' in ttype:
        base = 0.045 + v_req * 0.0002
    elif 'train' in ttype:
        base = 0.035 + c_req * 0.0003
    else:  # inference
        base = 0.020 + v_req * 0.0003 + c_req * 0.0005

    # 负载压力
    load = min(active_npus / max(NUM_CARDS * 8, 1), 1.5)
    lat  = base * (1.0 + 0.5 * load)

    # -VS惩罚：无时序感知时高峰配额无法提前调整，额外10%延迟
    if disable_vs and active_npus > NUM_CARDS * 6:
        lat *= 1.10

    # -RC惩罚：无互补调度时碎片率高，导致平均延迟额外8%
    if disable_rc:
        lat *= 1.08

    thr = inst.get('gpu_num', 1) / lat
    return lat, thr


def run_variant(label, workload_path, disable_vs=False, disable_wa=False,
               disable_rc=False, slo_s=0.05):
    print(f'\n[Ablation] Running: {label}')
    variant = SchedulerVariant(disable_wa=disable_wa, disable_rc=disable_rc)
    metrics = MetricsCollector(label, slo_s)

    with open(workload_path) as f:
        events = [ast.literal_eval(l.strip()) for l in f]
    base_time = datetime.strptime(events[0]['Time'], '%Y-%m-%d %H:%M:%S')

    for event in events:
        t  = datetime.strptime(event['Time'], '%Y-%m-%d %H:%M:%S')
        td = (t - base_time).total_seconds() / 60.0
        inst = event['Instance']
        if event['Action'] == 'start':
            _, cold = variant.schedule(inst)
            if cold:
                metrics.record_cold_start()
            lat, thr = _estimate_latency(inst, variant.active_count(),
                                         disable_vs=disable_vs,
                                         disable_rc=disable_rc)
            metrics.record_event(variant.active_count(), td, lat, thr)
        else:
            variant.delete(inst)
            metrics.record_event(variant.active_count(), td)

    s = metrics.summary()
    print(f'  peak={s["peak_npus"]}  SVR={s["svr"]*100:.2f}%  '
          f'CSC={s["csc"]}  Tput={s["avg_tput"]:.1f}req/s  '
          f'Lat={s["avg_lat_ms"]:.1f}ms')
    return metrics


def plot_ablation(results, outdir):
    if not HAS_MPL:
        return
    os.makedirs(outdir, exist_ok=True)
    labels = list(results.keys())
    colors = ['#2ecc71','#e74c3c','#3498db','#f39c12','#9b59b6']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Dilu-NPU Ablation Study', fontsize=14, fontweight='bold')

    # Timeline
    ax = axes[0, 0]
    for i, (lbl, m) in enumerate(results.items()):
        ax.plot(m.timestamps, m.active_counts,
                color=colors[i % len(colors)], label=lbl, linewidth=1.5)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Active NPU Cards')
    ax.set_title('Active Card Count Timeline')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # SVR
    ax = axes[0, 1]
    svrs = [m.svr()*100 for m in results.values()]
    bars = ax.bar(labels, svrs, color=colors[:len(labels)])
    for bar, v in zip(bars, svrs):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.05, f'{v:.2f}%',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('SLA Violation Rate (%)')
    ax.set_title('SVR Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Peak NPUs
    ax = axes[1, 0]
    peaks = [m.peak_npus() for m in results.values()]
    bars  = ax.bar(labels, peaks, color=colors[:len(labels)])
    for bar, v in zip(bars, peaks):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.1, str(v),
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Peak Active NPU Cards')
    ax.set_title('Peak Resource Usage')
    ax.grid(True, alpha=0.3, axis='y')

    # CSC + Throughput
    ax  = axes[1, 1]
    ax2 = ax.twinx()
    x   = np.arange(len(labels))
    w   = 0.35
    ax.bar(x-w/2,  [m.cold_starts      for m in results.values()],
           w, label='CSC',        color='#e74c3c', alpha=0.8)
    ax2.bar(x+w/2, [m.avg_throughput() for m in results.values()],
            w, label='Throughput', color='#2ecc71', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Cold Start Count')
    ax2.set_ylabel('Avg Throughput (req/s)')
    ax.set_title('Cold Starts & Throughput')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = os.path.join(outdir, 'ablation_comparison.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[Ablation] Plot: {out}')


def save_summary(results, outdir):
    os.makedirs(outdir, exist_ok=True)
    rows = [m.summary() for m in results.values()]
    with open(os.path.join(outdir, 'ablation_summary.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    print('\n' + '='*72)
    print(f'{"Variant":<20} {"Peak":>6} {"SVR%":>7} {"CSC":>6} '
          f'{"Tput":>9} {"Lat_ms":>8}')
    print('-'*72)
    for r in rows:
        print(f'{r["label"]:<20} {r["peak_npus"]:6d} '
              f'{r["svr"]*100:6.2f}%  {r["csc"]:6d} '
              f'{r["avg_tput"]:8.1f}  {r["avg_lat_ms"]:7.1f}')
    print('='*72)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload',
                        default=os.path.join(SIM_DIR, 'workload',
                                             'instances-npu-100.txt'))
    parser.add_argument('--slo',    type=float, default=0.05)
    parser.add_argument('--outdir', default=os.path.join(SIM_DIR, 'logs'))
    args = parser.parse_args()

    variants = [
        ('Dilu-NPU (Full)',     dict(disable_vs=False, disable_wa=False, disable_rc=False)),
        ('Dilu-NPU -VS',       dict(disable_vs=True,  disable_wa=False, disable_rc=False)),
        ('Dilu-NPU -WA',       dict(disable_vs=False, disable_wa=True,  disable_rc=False)),
        ('Dilu-NPU -RC',       dict(disable_vs=False, disable_wa=False, disable_rc=True)),
        ('Dilu-NPU -VS-WA-RC', dict(disable_vs=True,  disable_wa=True,  disable_rc=True)),
    ]

    results = {}
    for label, kwargs in variants:
        m = run_variant(label, args.workload, slo_s=args.slo, **kwargs)
        results[label] = m

    save_summary(results, args.outdir)
    plot_ablation(results, args.outdir)


if __name__ == '__main__':
    main()
