"""
paper_experiments.py  --  论文场景完整实验套件
===============================================
场景1: 稳定负载基线对比
场景2: 突发流量伸缩验证
场景3: 多模型混合共置对比
场景4: 资源溢出监控与伸缩
场景5: Day/Night 时序弹性
场景6: 消融实验

用法:
  python paper_experiments.py --scene 1
  python paper_experiments.py --scene all --simulate
"""

import argparse
import json
import os
import threading
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests

ROOT   = '/mnt/caoyujia/Dilu'
LOGDIR = os.path.join(ROOT, 'evaluation', 'logs')
os.makedirs(LOGDIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--scene',     default='all')
parser.add_argument('--gateway',   default='http://127.0.0.1:14999')
parser.add_argument('--scheduler', default='http://127.0.0.1:5000')
parser.add_argument('--slo_ms',    type=float, default=50.0)
parser.add_argument('--simulate',  action='store_true', default=False)
args = parser.parse_args()
SLO_S = args.slo_ms / 1000.0

SERVICES = {
    'resnet152-inf': {'url': f'{args.gateway}/resnet152-inf',
                      'payload': {'data': [0.5]*3}},
    'vgg19-inf':     {'url': f'{args.gateway}/vgg19-inf',
                      'payload': {'data': [0.5]*3}},
    'bert-inf':      {'url': f'{args.gateway}/bert-inf',
                      'payload': {'text': 'The NPU scheduler improves resource utilization. ' * 8}},
}


def _req(svc_name):
    svc = SERVICES[svc_name]
    if args.simulate:
        import random
        lat = max(0.003, 0.025 + random.gauss(0, 0.008))
        return lat, True
    try:
        t0 = time.time()
        r  = requests.post(svc['url'], json=svc['payload'], timeout=5)
        return time.time()-t0, r.status_code == 200
    except Exception:
        return 9.999, False


def _flood(svc, rps, duration):
    interval = 1.0 / max(rps, 0.1)
    end_t    = time.time() + duration
    lats, viols, total, errors = [], 0, 0, 0
    lock = threading.Lock()

    def _do():
        nonlocal viols, total, errors
        lat, ok = _req(svc)
        with lock:
            total += 1
            if not ok:
                errors += 1
            else:
                lats.append(lat)
                if lat > SLO_S:
                    viols += 1

    if args.simulate:
        # 仿真模式：串行执行，避免线程池开销导致超时
        while time.time() < end_t:
            t0 = time.time()
            _do()
            sl = interval - (time.time() - t0)
            if sl > 0:
                time.sleep(sl)
        return lats, viols, total, errors

    with ThreadPoolExecutor(max_workers=max(16, int(rps*3))) as pool:
        futs = []
        while time.time() < end_t:
            t0 = time.time()
            futs.append(pool.submit(_do))
            sl = interval - (time.time()-t0)
            if sl > 0: time.sleep(sl)
        for f in as_completed(futs):
            try: f.result()
            except Exception: pass
    return lats, viols, total, errors


def _metrics(lats, viols, total, errors, label=''):
    n   = len(lats) or 1
    s   = sorted(lats)
    svr = viols / total if total > 0 else 0
    m   = {
        'label':      label, 'total': total, 'errors': errors,
        'svr':        round(svr, 4),
        'avg_lat_ms': round(statistics.mean(s)*1000, 2) if s else 0,
        'p50_ms':     round(s[int(n*0.50)]*1000, 2) if s else 0,
        'p95_ms':     round(s[min(int(n*0.95),n-1)]*1000, 2) if s else 0,
        'p99_ms':     round(s[min(int(n*0.99),n-1)]*1000, 2) if s else 0,
    }
    print(f'  [{label}] total={total} err={errors} SVR={svr*100:.1f}% '
          f'avg={m["avg_lat_ms"]}ms P95={m["p95_ms"]}ms')
    return m


def _snap():
    try:
        r = requests.get(f'{args.scheduler}/status', timeout=3)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def _save(name, data):
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(LOGDIR, f'exp_{name}_{ts}.json')
    with open(out, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'  [Saved] {out}')
    return out


# ── 场景1: 稳定负载基线对比 ───────────────────────────────────────────────
def scene1_stable_baseline():
    print('\n' + '='*60)
    print('Scene 1: Stable Load Baseline (resnet152/vgg19/bert)')
    print('='*60)
    rps_levels = [10, 30, 50, 80] if not args.simulate else [10, 30]
    dur        = 60 if not args.simulate else 5
    results    = {'scene': 1, 'services': {}}
    for svc in ['resnet152-inf', 'vgg19-inf', 'bert-inf']:
        svc_res = []
        for rps in rps_levels:
            print(f'  {svc} @ {rps} RPS for {dur}s')
            lats, viols, total, errors = _flood(svc, rps, dur)
            m = _metrics(lats, viols, total, errors, f'{svc}@{rps}')
            m.update({'rps': rps, 'npu_snap': _snap()})
            svc_res.append(m)
        results['services'][svc] = svc_res
    _save('scene1_stable', results)
    return results


# ── 场景2: 突发流量伸缩验证 ───────────────────────────────────────────────
def scene2_burst_scaling():
    print('\n' + '='*60)
    print('Scene 2: Burst Traffic Auto Scale-Out')
    print('='*60)
    results = {'scene': 2, 'phases': [], 'overflow_events': []}
    phases  = ([('background',10,30),('burst',80,15),
                ('recovery',10,30),('burst2',100,15),('recovery2',10,20)]
               if not args.simulate else
               [('bg',10,5),('burst',80,5),('recovery',10,5)])
    for name, rps, dur in phases:
        print(f'  [{name}] {rps} RPS for {dur}s')
        s_pre  = _snap()
        inst_b = len(s_pre.get('active_npus', []))
        lats, viols, total, errors = _flood('resnet152-inf', rps, dur)
        s_post = _snap()
        inst_a = len(s_post.get('active_npus', []))
        m = _metrics(lats, viols, total, errors, name)
        m.update({'rps': rps, 'inst_before': inst_b, 'inst_after': inst_a,
                  'scaled_out': inst_a > inst_b})
        results['phases'].append(m)
        if inst_a > inst_b:
            results['overflow_events'].append(
                {'phase': name, 'rps': rps,
                 'new_inst': inst_a - inst_b})
            print(f'  *** SCALE-OUT: {inst_b} → {inst_a} instances ***')
    _save('scene2_burst', results)
    return results


# ── 场景3: 多模型混合共置 ─────────────────────────────────────────────────
def scene3_colocation():
    print('\n' + '='*60)
    print('Scene 3: Multi-Model Colocation (Vector+Cube Complementarity)')
    print('='*60)
    dur     = 60 if not args.simulate else 6
    results = {'scene': 3, 'concurrent': {}, 'sequential': []}
    rps_map = {'resnet152-inf': 40, 'vgg19-inf': 30, 'bert-inf': 20}

    # 并发：触发互补调度
    print('  [Concurrent] resnet152+vgg19+bert simultaneously')
    all_stats = {}
    def _run(svc, rps, d):
        lats, viols, total, errors = _flood(svc, rps, d)
        all_stats[svc] = _metrics(lats, viols, total, errors, f'conc/{svc}')
    threads = [threading.Thread(target=_run, args=(s,r,dur))
               for s,r in rps_map.items()]
    for t in threads: t.start()
    for t in threads: t.join()
    results['concurrent'] = {'services': all_stats, 'npu_snap': _snap()}

    # 顺序：对照组
    print('  [Sequential] each model separately')
    for svc, rps in rps_map.items():
        lats, viols, total, errors = _flood(svc, rps, dur//3)
        results['sequential'].append(
            _metrics(lats, viols, total, errors, f'seq/{svc}'))
    _save('scene3_colocation', results)
    return results


# ── 场景4: 资源溢出监控与伸缩 ────────────────────────────────────────────
def scene4_overflow_scaling():
    print('\n' + '='*60)
    print('Scene 4: Resource Overflow Detection & Auto Scaling')
    print('='*60)
    results   = {'scene': 4, 'timeline': [], 'overflow_events': []}
    rps_steps = [20,40,60,80,100,120] if not args.simulate else [20,60,100]
    step_dur  = 20 if not args.simulate else 8
    for rps in rps_steps:
        s_pre  = _snap()
        inst_b = len(s_pre.get('active_npus', []))
        lats, viols, total, errors = _flood('resnet152-inf', rps, step_dur)
        s_post = _snap()
        inst_a = len(s_post.get('active_npus', []))
        m = _metrics(lats, viols, total, errors, f'overflow@{rps}')
        m.update({'rps': rps, 'inst_b': inst_b, 'inst_a': inst_a,
                  'scaled_out': inst_a > inst_b})
        results['timeline'].append(m)
        if inst_a > inst_b:
            results['overflow_events'].append(
                {'rps': rps, 'delta': inst_a-inst_b})
            print(f'  *** SCALE-OUT: {inst_b}→{inst_a} @ {rps}RPS ***')
        if m['svr'] > 0.3:
            print(f'  SVR high ({m["svr"]*100:.0f}%), waiting stabilize...')
            time.sleep(10 if not args.simulate else 3)
    _save('scene4_overflow', results)
    return results


# ── 场景5: Day/Night 时序弹性 ─────────────────────────────────────────────
def scene5_temporal():
    print('\n' + '='*60)
    print('Scene 5: Day/Night Temporal Elasticity')
    print('='*60)
    results = {'scene': 5, 'phases': []}
    phases  = [('day_peak', 50, 30), ('night_valley', 5, 30),
               ('dawn_ramp', 30, 20)]
    if args.simulate:
        phases = [('day_peak',50,10),('night_valley',5,10)]
    for name, rps, dur in phases:
        print(f'  [{name}] {rps} RPS for {dur}s')
        lats, viols, total, errors = _flood('resnet152-inf', rps, dur)
        m = _metrics(lats, viols, total, errors, name)
        m.update({'rps': rps, 'npu_snap': _snap()})
        results['phases'].append(m)
    _save('scene5_temporal', results)
    return results


# ── 场景6: 消融实验 ───────────────────────────────────────────────────────
def scene6_ablation():
    print('\n' + '='*60)
    print('Scene 6: Ablation Study (via simulation)')
    print('='*60)
    ablation_script = os.path.join(
        ROOT, 'scheduling', 'simulations', 'ablation_npu.py')
    workload = os.path.join(
        ROOT, 'scheduling', 'simulations', 'workload',
        'instances-npu-200.txt')
    if os.path.exists(ablation_script) and os.path.exists(workload):
        import subprocess, sys
        r = subprocess.run(
            [sys.executable, ablation_script,
             '--workload', workload, '--outdir', LOGDIR],
            capture_output=True, text=True, timeout=120)
        print(r.stdout[-1000:] if r.stdout else '')
        return {'scene': 6, 'status': 'ok'}
    return {'scene': 6, 'status': 'skipped (files not found)'}


# ── 主入口 ────────────────────────────────────────────────────────────────
SCENES = {
    '1': scene1_stable_baseline,
    '2': scene2_burst_scaling,
    '3': scene3_colocation,
    '4': scene4_overflow_scaling,
    '5': scene5_temporal,
    '6': scene6_ablation,
}

if __name__ == '__main__':
    print(f'\nDilu-NPU Paper Experiments')
    print(f'  Gateway:  {args.gateway}')
    print(f'  SLO:      {args.slo_ms}ms')
    print(f'  Simulate: {args.simulate}')

    to_run = list(SCENES.keys()) if args.scene == 'all' else [args.scene]
    all_results = {}
    for s in to_run:
        if s in SCENES:
            try:
                all_results[f'scene{s}'] = SCENES[s]()
            except Exception as e:
                print(f'[ERROR] Scene {s}: {e}')
                all_results[f'scene{s}'] = {'error': str(e)}
        else:
            print(f'[WARN] Unknown scene: {s}')

    # 汇总报告
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(LOGDIR, f'paper_experiments_{ts}.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n[Done] All results: {out}')
