"""
comp_baselines_npu.py  --  NPU Baseline Comparison Runner
==========================================================
Runs all four schedulers on the same NPU workload trace and
produces a side-by-side active-card timeline plot.

Schedulers compared:
  Exclusive (K8s-NPU)    --  scheduler_k8s_npu.py
  INFless-L-NPU          --  scheduler_infless_l_npu.py
  INFless-R-NPU          --  scheduler_infless_r_npu.py
  Dilu-NPU               --  scheduler_dilu_npu.py

Usage:
  cd scheduling/simulations
  python comp_baselines_npu.py --workload workload/instances-npu-3200.txt
  python comp_baselines_npu.py --workload workload/instances-npu-100.txt

Outputs:
  logs/npu-baselines-timeline.png
  logs/npu-baselines-summary.txt
"""

import ast
import importlib
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASELINE_DIR)

BASELINES = [
    ('Exclusive',   'baseline.scheduler_k8s_npu',      'red',    ':'),
    ('INFless-L',   'baseline.scheduler_infless_l_npu', 'gold',   '--'),
    ('INFless-R',   'baseline.scheduler_infless_r_npu', 'blue',   '-.'),
    ('Dilu-NPU',    'baseline.scheduler_dilu_npu',      'green',  '-'),
]


def _reset_module(mod):
    """Reset global cluster state inside a scheduler module between runs."""
    from baseline import scheduler_dilu_npu
    total = 1000 * 4
    NPU_cls  = mod.NPU
    PM_cls   = mod.PortManager

    nodes_info = []
    for ip in range(1000):
        for i in range(mod.NUM_CARDS):
            nodes_info.append({'ip': str(ip), 'index': i})

    mod.nodes_info    = nodes_info
    mod.new_npus      = [NPU_cls(i, node['ip'], node['index'])
                         for i, node in enumerate(nodes_info)]
    mod.active_npus   = []
    mod.port_manager  = PM_cls()


def run_baseline(label, module_name, workload_path):
    """
    Load the scheduler module fresh, replay the workload,
    return (timestamps, active_npu_counts, max_npus, frag_metrics).
    """
    # Fresh import each time
    if module_name in sys.modules:
        del sys.modules[module_name]
    # Also remove sub-parts
    for key in list(sys.modules.keys()):
        if module_name in key:
            del sys.modules[key]

    mod = importlib.import_module(module_name)
    _reset_module(mod)

    with open(workload_path) as f:
        events = [ast.literal_eval(line.strip()) for line in f]

    timestamps, counts = [], []
    max_npus = 0
    frag_records = []
    base_time = datetime.strptime(events[0]['Time'], '%Y-%m-%d %H:%M:%S')

    for event in events:
        t  = datetime.strptime(event['Time'], '%Y-%m-%d %H:%M:%S')
        td = (t - base_time).total_seconds() / 60.0
        if event['Action'] == 'start':
            mod.schedule_instance(event['Instance'])
        else:
            mod.delete_instance(event['Instance'])
        cnt = len(mod.active_npus)
        timestamps.append(td)
        counts.append(cnt)
        if cnt > max_npus:
            max_npus = cnt
        # Collect fragmentation at peak
        if hasattr(mod, 'calc_fragmentation') and 660 <= max_npus <= 664:
            frag_records.append(mod.calc_fragmentation())

    avg_frag = None
    if frag_records:
        avg_frag = tuple(sum(x[i] for x in frag_records) / len(frag_records)
                         for i in range(3))

    print(f'  [{label:12s}] max_npus={max_npus:4d}  '
          + (f'VFrag={avg_frag[0]:.3f} CFrag={avg_frag[1]:.3f} '
             f'MFrag={avg_frag[2]:.3f}' if avg_frag else 'no-frag-data'))
    return timestamps, counts, max_npus, avg_frag


def main():
    import argparse
    parser = argparse.ArgumentParser(description='NPU baseline comparison')
    parser.add_argument('--workload',
                        default=os.path.join(
                            BASELINE_DIR, 'workload', 'instances-npu-3200.txt'),
                        help='Path to instances-npu-*.txt workload file')
    parser.add_argument('--outdir', default=os.path.join(BASELINE_DIR, '..', 'logs'))
    args = parser.parse_args()

    if not os.path.exists(args.workload):
        # Auto-generate workload if missing
        print(f'[INFO] Workload not found: {args.workload}')
        print('[INFO] Generating NPU workload...')
        gen_path = os.path.join(BASELINE_DIR, 'workload', 'service_generator_npu.py')
        spec = importlib.util.spec_from_file_location('gen', gen_path)
        gen  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        n    = int(Path(args.workload).stem.split('-')[-1])
        gen.generate_and_save(n, output_path=args.workload)

    os.makedirs(args.outdir, exist_ok=True)

    print(f'\n[NPU Baselines] Workload: {args.workload}')
    print('=' * 60)

    results = {}
    for label, module_name, color, ls in BASELINES:
        print(f'Running {label}...')
        t0 = time.time()
        ts, counts, max_n, frag = run_baseline(label, module_name, args.workload)
        elapsed = time.time() - t0
        results[label] = dict(timestamps=ts, counts=counts,
                              max_npus=max_n, frag=frag,
                              color=color, ls=ls, elapsed=elapsed)
        print(f'  [{label:12s}] done in {elapsed:.1f}s')

    # ---- Plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, r in results.items():
        ax.plot(r['timestamps'], r['counts'],
                linestyle=r['ls'], color=r['color'],
                linewidth=1.8, label=label)

    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Active NPU cards', fontsize=12)
    ax.set_title('Dilu-NPU vs Baselines: Active NPU Card Count', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(args.outdir, 'npu-baselines-timeline.png')
    plt.savefig(out_png, dpi=300)
    print(f'\n[NPU Baselines] Plot saved: {out_png}')

    # ---- Summary text -------------------------------------------------------
    out_txt = os.path.join(args.outdir, 'npu-baselines-summary.txt')
    with open(out_txt, 'w') as f:
        f.write('NPU Baseline Comparison Summary\n')
        f.write('=' * 50 + '\n')
        f.write(f'Workload : {args.workload}\n')
        f.write(f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        for label, r in results.items():
            f.write(f'{label:15s}  max_npus={r["max_npus"]:4d}')
            if r['frag']:
                vf, cf, mf = r['frag']
                f.write(f'  VFrag={vf:.3f}  CFrag={cf:.3f}  MFrag={mf:.3f}')
            f.write(f'  runtime={r["elapsed"]:.1f}s\n')
    print(f'[NPU Baselines] Summary: {out_txt}')

    # Print reduction vs K8s
    k8s_max = results.get('Exclusive', {}).get('max_npus', None)
    dilu_max = results.get('Dilu-NPU', {}).get('max_npus', None)
    if k8s_max and dilu_max and k8s_max > 0:
        reduction = (k8s_max - dilu_max) / k8s_max * 100
        print(f'\n[NPU Baselines] Dilu-NPU reduces peak NPU usage by '
              f'{reduction:.1f}% vs Exclusive (K8s)')


if __name__ == '__main__':
    main()
