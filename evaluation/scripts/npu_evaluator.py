"""
npu_evaluator.py  --  NPU 推理评估脚本
==============================================
采集指标:
  - SLA 违约率 (SVR)
  - 吞吐量 (req/s)
  - 显存占用 (acl.rt.get_mem_info)
  - 内存利用率 (psutil)
  - 端到端延迟分布 (P50/P95/P99)

用法:
  python npu_evaluator.py --service resnet152-inf --duration 60 --target_rps 20
"""

import argparse
import csv
import json
import os
import statistics
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import acl  # type: ignore
    ACL_AVAILABLE = True
except ImportError:
    ACL_AVAILABLE = False

parser = argparse.ArgumentParser()
parser.add_argument('--service',     default='resnet152-inf')
parser.add_argument('--gateway',     default='http://127.0.0.1:14999')
parser.add_argument('--scheduler',   default='http://127.0.0.1:5000')
parser.add_argument('--slo_ms',      type=float, default=50.0)
parser.add_argument('--duration',    type=int,   default=60,
                    help='Test duration in seconds')
parser.add_argument('--target_rps',  type=float, default=10.0)
parser.add_argument('--num_devices', type=int,   default=4)
parser.add_argument('--outdir',      default='/mnt/caoyujia/Dilu/evaluation/logs')
parser.add_argument('--simulate',    action='store_true', default=False)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
SLO_S = args.slo_ms / 1000.0

# ── 辅助：显存查询 ────────────────────────────────────────────────────────
def get_npu_mem_info(device_id):
    """返回 (used_gb, total_gb) 对应指定 NPU 卡。"""
    if args.simulate or not ACL_AVAILABLE:
        import random
        return random.uniform(2, 10), 64.0
    try:
        acl.rt.set_device(device_id)
        free, total, ret = acl.rt.get_mem_info(0)
        if ret == 0:
            used = total - free
            return used / 1024**3, total / 1024**3
    except Exception:
        pass
    return 0.0, 64.0


def get_host_mem_util():
    """返回主机内存利用率 (0-1)。"""
    if HAS_PSUTIL:
        return psutil.virtual_memory().percent / 100.0
    return 0.0


def get_scheduler_status():
    """从 scheduler_npu /status 获取 NPU 使用情况。"""
    try:
        r = requests.get(f'{args.scheduler}/status', timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


# ── 单次请求 ──────────────────────────────────────────────────────────────
def send_request(url):
    """发送一次推理请求，返回 (latency_s, success)。"""
    if args.simulate:
        import random, math
        base = 0.025 + random.gauss(0, 0.005)
        return max(0.005, base), True
    payload = {'data': [0.5] * 224}
    try:
        t0 = time.time()
        r  = requests.post(url, json=payload, timeout=5)
        lat = time.time() - t0
        return lat, r.status_code == 200
    except Exception:
        return 9999.0, False


# ── 主评估循环 ────────────────────────────────────────────────────────────
def run_evaluation():
    url = f'{args.gateway}/{args.service}'
    print(f'\n[Eval] Service: {args.service}')
    print(f'[Eval] URL:     {url}')
    print(f'[Eval] SLO:     {args.slo_ms}ms  Duration: {args.duration}s  '
          f'Target RPS: {args.target_rps}')

    interval   = 1.0 / args.target_rps
    end_time   = time.time() + args.duration

    latencies  = []
    sla_viols  = 0
    total_reqs = 0
    errors     = 0
    mem_samples= []   # (used_gb, total_gb) per second
    host_mems  = []
    sched_snaps= []
    timeline   = []   # per-second throughput

    sec_start  = time.time()
    sec_reqs   = 0
    lock       = threading.Lock()

    def _do_req():
        nonlocal sla_viols, total_reqs, errors, sec_reqs
        lat, ok = send_request(url)
        with lock:
            total_reqs += 1
            sec_reqs   += 1
            if not ok:
                errors += 1
            else:
                latencies.append(lat)
                if lat > SLO_S:
                    sla_viols += 1

    with ThreadPoolExecutor(max_workers=max(8, int(args.target_rps * 2))) as pool:
        futs = []
        while time.time() < end_time:
            t0 = time.time()
            futs.append(pool.submit(_do_req))

            # 每秒采样一次资源
            if time.time() - sec_start >= 1.0:
                with lock:
                    tput = sec_reqs
                    sec_reqs = 0
                timeline.append({'ts': datetime.now().isoformat(),
                                 'tput': tput})
                sec_start = time.time()

                # 采集所有 NPU 卡显存
                card_mems = []
                for d in range(args.num_devices):
                    used, total = get_npu_mem_info(d)
                    card_mems.append({'device': d,
                                      'used_gb':  round(used, 2),
                                      'total_gb': round(total, 2),
                                      'util':     round(used/max(total,1), 3)})
                mem_samples.append(card_mems)
                host_mems.append(get_host_mem_util())
                sched_snaps.append(get_scheduler_status())

            sleep = interval - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

        for f in as_completed(futs):
            try: f.result()
            except Exception: pass

    return latencies, sla_viols, total_reqs, errors, \
           timeline, mem_samples, host_mems, sched_snaps


# ── 结果分析与输出 ────────────────────────────────────────────────────────
def analyze(latencies, sla_viols, total_reqs, errors,
            timeline, mem_samples, host_mems, sched_snaps):
    n = len(latencies) or 1
    svr = sla_viols / total_reqs if total_reqs > 0 else 0
    avg_lat = statistics.mean(latencies) if latencies else 0
    p50 = sorted(latencies)[int(n*0.50)] if latencies else 0
    p95 = sorted(latencies)[int(n*0.95)] if latencies else 0
    p99 = sorted(latencies)[int(n*0.99)] if latencies else 0
    avg_tput = sum(t['tput'] for t in timeline) / max(len(timeline), 1)

    # 显存
    avg_mem_util = 0.0
    if mem_samples:
        flat = [c['util'] for snap in mem_samples for c in snap]
        avg_mem_util = statistics.mean(flat) if flat else 0
    avg_host_mem = statistics.mean(host_mems) if host_mems else 0

    print('\n' + '='*60)
    print(f'[Result] Service:     {args.service}')
    print(f'[Result] Total Reqs:  {total_reqs}  Errors: {errors}')
    print(f'[Result] SVR:         {svr*100:.2f}%  '
          f'({sla_viols}/{total_reqs} violations)')
    print(f'[Result] Avg Tput:    {avg_tput:.1f} req/s')
    print(f'[Result] Latency:     avg={avg_lat*1000:.1f}ms '
          f'P50={p50*1000:.1f}ms P95={p95*1000:.1f}ms '
          f'P99={p99*1000:.1f}ms')
    print(f'[Result] NPU MemUtil: {avg_mem_util*100:.1f}%')
    print(f'[Result] Host MemUtil:{avg_host_mem*100:.1f}%')
    print('='*60)

    summary = {
        'service':       args.service,
        'timestamp':     datetime.now().isoformat(),
        'slo_ms':        args.slo_ms,
        'total_reqs':    total_reqs,
        'errors':        errors,
        'svr':           round(svr, 4),
        'avg_tput':      round(avg_tput, 2),
        'avg_lat_ms':    round(avg_lat*1000, 2),
        'p50_ms':        round(p50*1000, 2),
        'p95_ms':        round(p95*1000, 2),
        'p99_ms':        round(p99*1000, 2),
        'npu_mem_util':  round(avg_mem_util, 4),
        'host_mem_util': round(avg_host_mem, 4),
        'timeline':      timeline,
        'mem_samples':   mem_samples[-10:] if mem_samples else [],
    }

    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(args.outdir, f'eval_{args.service}_{ts}.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[Eval] Results saved: {out}')

    # CSV for paper
    csv_out = os.path.join(args.outdir, f'eval_{args.service}_{ts}.csv')
    with open(csv_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'service','svr','avg_tput','avg_lat_ms',
            'p50_ms','p95_ms','p99_ms','npu_mem_util','host_mem_util'])
        w.writeheader()
        w.writerow({k: summary[k] for k in w.fieldnames})
    print(f'[Eval] CSV: {csv_out}')
    return summary


if __name__ == '__main__':
    result = run_evaluation()
    analyze(*result)
