"""
heavy_load_generator.py  --  高并发负载生成器
==============================================
专为论文实验设计的负载生成器，支持以下场景:
  - steady:  稳定负载（目标 RPS 持续发送）
  - burst:   突发负载（背景低负载 + 周期性突发）
  - mixed:   混合负载（resnet152 + vgg19 + bert 同时压测）
  - rampup:  线性增压（从低 RPS 逐渐增加到高 RPS）

用法:
  python heavy_load_generator.py --scenario steady --rps 50 --duration 120
  python heavy_load_generator.py --scenario burst  --base_rps 10 --burst_rps 80
  python heavy_load_generator.py --scenario mixed  --duration 120
"""

import argparse
import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests

parser = argparse.ArgumentParser()
parser.add_argument('--gateway',    default='http://127.0.0.1:14999')
parser.add_argument('--scenario',   default='steady',
                    choices=['steady','burst','mixed','rampup'])
parser.add_argument('--duration',   type=int,   default=120)
parser.add_argument('--rps',        type=float, default=30.0,
                    help='Target RPS for steady/mixed')
parser.add_argument('--base_rps',   type=float, default=10.0)
parser.add_argument('--burst_rps',  type=float, default=80.0)
parser.add_argument('--burst_dur',  type=float, default=15.0,
                    help='Burst duration in seconds')
parser.add_argument('--burst_interval', type=float, default=30.0)
parser.add_argument('--slo_ms',     type=float, default=50.0)
parser.add_argument('--outdir',     default='/mnt/caoyujia/Dilu/evaluation/logs')
parser.add_argument('--simulate',   action='store_true', default=False)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
SLO_S = args.slo_ms / 1000.0

# ── 服务配置 ──────────────────────────────────────────────────────────────
SERVICES = {
    'resnet152-inf': {'endpoint': '/resnet152-inf',
                      'payload':  {'data': [0.5]*3}},
    'vgg19-inf':     {'endpoint': '/vgg19-inf',
                      'payload':  {'data': [0.5]*3}},
    'bert-inf':      {'endpoint': '/bert-inf',
                      'payload':  {'text': 'The quick brown fox jumps over the lazy dog. ' * 5}},
}

# ── 统计 ──────────────────────────────────────────────────────────────────
class Stats:
    def __init__(self):
        self.lock     = threading.Lock()
        self.lats     = []
        self.total    = 0
        self.errors   = 0
        self.viols    = 0
        self.timeline = []   # (ts, tput) 每秒快照
        self._sec_cnt = 0
        self._sec_t   = time.time()

    def record(self, lat, ok):
        with self.lock:
            self.total += 1
            self._sec_cnt += 1
            if not ok:
                self.errors += 1
                return
            self.lats.append(lat)
            if lat > SLO_S:
                self.viols += 1
            if time.time() - self._sec_t >= 1.0:
                self.timeline.append({
                    'ts':   datetime.now().isoformat(),
                    'tput': self._sec_cnt})
                self._sec_cnt = 0
                self._sec_t   = time.time()

    def summary(self, label=''):
        n   = len(self.lats) or 1
        svr = self.viols / self.total if self.total > 0 else 0
        s   = sorted(self.lats)
        print(f'\n[{label}] total={self.total} err={self.errors} '
              f'SVR={svr*100:.1f}%')
        if s:
            print(f'  lat avg={statistics.mean(s)*1000:.1f}ms '
                  f'P50={s[int(n*0.5)]*1000:.1f}ms '
                  f'P95={s[int(n*0.95)]*1000:.1f}ms '
                  f'P99={s[min(int(n*0.99),n-1)]*1000:.1f}ms')
        return {'label': label, 'total': self.total,
                'svr': round(svr,4),
                'avg_lat_ms': round(statistics.mean(s)*1000,2) if s else 0,
                'p95_ms': round(s[int(n*0.95)]*1000,2) if s else 0,
                'timeline': self.timeline}


def _send(service_name, stats):
    svc = SERVICES[service_name]
    url = args.gateway + svc['endpoint']
    if args.simulate:
        import random
        lat = max(0.005, 0.025 + random.gauss(0, 0.005))
        stats.record(lat, True)
        return
    try:
        t0 = time.time()
        r  = requests.post(url, json=svc['payload'], timeout=5)
        stats.record(time.time()-t0, r.status_code == 200)
    except Exception:
        stats.record(9999, False)


def _run_steady(service, rps, duration, stats):
    interval = 1.0 / rps
    end_t    = time.time() + duration
    workers  = max(8, int(rps * 2))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = []
        while time.time() < end_t:
            t0 = time.time()
            futs.append(pool.submit(_send, service, stats))
            sleep = interval - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
        for f in as_completed(futs):
            try: f.result()
            except Exception: pass


# ── 场景 1: 稳定负载 ──────────────────────────────────────────────────────
def scenario_steady():
    print(f'\n[Scenario: STEADY] RPS={args.rps} Duration={args.duration}s')
    results = {}
    threads = []
    for svc in ['resnet152-inf']:
        st = Stats()
        t  = threading.Thread(
                target=_run_steady,
                args=(svc, args.rps, args.duration, st))
        t.start(); threads.append((svc, t, st))
    for svc, t, st in threads:
        t.join()
        results[svc] = st.summary(f'STEADY/{svc}')
    return results


# ── 场景 2: 突发负载 ──────────────────────────────────────────────────────
def scenario_burst():
    print(f'\n[Scenario: BURST] base={args.base_rps} '
          f'burst={args.burst_rps} dur={args.duration}s')
    st  = Stats()
    end = time.time() + args.duration
    while time.time() < end:
        remaining = end - time.time()
        # 背景低负载
        bg_dur = min(args.burst_interval - args.burst_dur, remaining)
        if bg_dur > 0:
            print(f'  [BG]    RPS={args.base_rps} for {bg_dur:.0f}s')
            _run_steady('resnet152-inf', args.base_rps, bg_dur, st)
        remaining = end - time.time()
        if remaining <= 0:
            break
        # 突发
        b_dur = min(args.burst_dur, remaining)
        print(f'  [BURST] RPS={args.burst_rps} for {b_dur:.0f}s')
        _run_steady('resnet152-inf', args.burst_rps, b_dur, st)
    return {'burst': st.summary('BURST/resnet152-inf')}


# ── 场景 3: 混合共置负载 ──────────────────────────────────────────────────
def scenario_mixed():
    print(f'\n[Scenario: MIXED] RPS={args.rps} '
          f'3 services concurrent Duration={args.duration}s')
    results = {}
    threads = []
    rps_map = {
        'resnet152-inf': args.rps,
        'vgg19-inf':     args.rps * 0.6,
        'bert-inf':      args.rps * 0.4,
    }
    for svc, rps in rps_map.items():
        st = Stats()
        t  = threading.Thread(
                target=_run_steady,
                args=(svc, rps, args.duration, st))
        t.start(); threads.append((svc, t, st))
    for svc, t, st in threads:
        t.join()
        results[svc] = st.summary(f'MIXED/{svc}')
    return results


# ── 场景 4: 线性增压 ──────────────────────────────────────────────────────
def scenario_rampup():
    max_rps  = args.burst_rps
    step_dur = 15
    steps    = int(args.duration / step_dur)
    print(f'\n[Scenario: RAMPUP] {args.base_rps}→{max_rps} '
          f'steps={steps} step_dur={step_dur}s')
    results = []
    for i in range(steps):
        rps = args.base_rps + (max_rps - args.base_rps) * i / max(steps-1, 1)
        st  = Stats()
        print(f'  Step {i+1}/{steps}: RPS={rps:.1f}')
        _run_steady('resnet152-inf', rps, step_dur, st)
        r = st.summary(f'RAMPUP step{i+1} rps={rps:.0f}')
        r['rps'] = round(rps, 1)
        results.append(r)
    return {'rampup': results}


# ── 主函数 ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    fn = {'steady': scenario_steady,
          'burst':  scenario_burst,
          'mixed':  scenario_mixed,
          'rampup': scenario_rampup}[args.scenario]
    result = fn()

    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(args.outdir,
                       f'load_{args.scenario}_{ts}.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\n[Load] Results: {out}')
