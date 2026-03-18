"""
npu_adaptive_scaler.py  --  自适应二维协同伸缩控制器
======================================================
纵向弹性: acl.rt.set_device_res_limit 动态调节 Vector/Cube 配额
横向弹性: 纵向调满后触发新实例启动 (scale-out)
时序整合: 低谷期将资源集中到少数卡
突发应对: 流量激增时紧急横向扩容
"""

import logging
import threading
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Callable
from collections import deque

logging.basicConfig(
    format='%(asctime)s [AdaptiveScaler] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
log = logging.getLogger(__name__)

try:
    import acl  # type: ignore
    ACL_AVAILABLE = True
except ImportError:
    ACL_AVAILABLE = False

MAX_VECTOR = 40
MAX_CUBE   = 20
ACL_RT_DEV_RES_CUBE_CORE   = 0
ACL_RT_DEV_RES_VECTOR_CORE = 1


def _acl_set(device_id, res_type, value, simulate):
    name = 'VECTOR' if res_type == ACL_RT_DEV_RES_VECTOR_CORE else 'CUBE'
    log.info(f'  acl.rt.set_device_res_limit(device={device_id}, {name}, {value})')
    if not simulate:
        if ACL_AVAILABLE:
            acl.rt.set_device_res_limit(device_id, res_type, value)
        else:
            log.warning('ACL not available')


class CardState:
    def __init__(self, device_id):
        self.device_id    = device_id
        self.vector       = MAX_VECTOR
        self.cube         = MAX_CUBE
        self.instances    = 0
        self.load         = 0.0
        self.rps_window   = deque(maxlen=12)  # 1-min sliding window at 5s interval
        self.lat_window   = deque(maxlen=12)

    def avg_rps(self):
        w = list(self.rps_window)
        return sum(w)/len(w) if w else 0.0

    def avg_lat(self):
        w = list(self.lat_window)
        return sum(w)/len(w) if w else 0.0

    def is_idle(self):
        return self.instances == 0


class NPUAdaptiveScaler:
    """
    自适应二维协同伸缩控制器。
    整合纵向(ACL配额调整)和横向(实例增减)两种弹性机制。
    新特性:
      - 时序整合: 低谷期把空闲卡资源集中到忙碌卡
      - 突发应对: 突发检测 -> 快速纵向满配 -> 横向扩容信号
    """

    def __init__(self, device_ids=None, check_interval=5,
                 slo_s=0.05, simulate=True,
                 scale_out_cb: Optional[Callable] = None,
                 scale_in_cb:  Optional[Callable] = None):
        self.device_ids    = device_ids or list(range(4))
        self.check_interval= check_interval
        self.slo_s         = slo_s
        self.simulate      = simulate
        self.scale_out_cb  = scale_out_cb
        self.scale_in_cb   = scale_in_cb
        self.cards: Dict[int, CardState] = {
            d: CardState(d) for d in self.device_ids}
        self._running      = False
        self._baseline_rps = 0.0
        # 突发阈值: baseline 的 2 倍
        self.burst_mult    = 2.0
        # 时间段
        self.peak_start    = 8
        self.peak_end      = 22
        # 配额步长
        self.vstep         = 5
        self.cstep         = 2
        # 论文数据记录
        self._log: List[dict] = []
        log.info(f'NPUAdaptiveScaler init devices={self.device_ids} simulate={simulate}')

    # ── 外部数据注入 ────────────────────────────────────────────────────────
    def update_metrics(self, device_id, rps, latency, instances, load):
        if device_id not in self.cards:
            return
        c = self.cards[device_id]
        c.rps_window.append(rps)
        c.lat_window.append(latency)
        c.instances = instances
        c.load      = load

    # ── 主控制循环 ──────────────────────────────────────────────────────────
    def start(self):
        self._running = True
        log.info('NPUAdaptiveScaler started')
        while self._running:
            self._cycle()
            time.sleep(self.check_interval)

    def stop(self):
        self._running = False

    def _is_peak(self):
        h = datetime.now().hour
        return self.peak_start <= h < self.peak_end

    def _total_rps(self):
        return sum(c.avg_rps() for c in self.cards.values())

    def _is_burst(self):
        if self._baseline_rps < 0.1:
            return False
        return self._total_rps() > self._baseline_rps * self.burst_mult

    def _cycle(self):
        ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        rps  = self._total_rps()
        burst = self._is_burst()
        peak  = self._is_peak()

        if burst:
            mode = 'BURST'
            self._handle_burst()
        elif peak:
            mode = 'PEAK'
            self._handle_peak()
        else:
            mode = 'VALLEY'
            self._handle_valley()
            # 更新基线
            alpha = 0.1
            self._baseline_rps = alpha*rps + (1-alpha)*self._baseline_rps

        self._log.append({
            'ts': ts, 'mode': mode, 'total_rps': round(rps, 2),
            'cards': [{'id': d, 'v': c.vector, 'c': c.cube,
                       'rps': round(c.avg_rps(), 2), 'lat': round(c.avg_lat(), 4)}
                      for d, c in self.cards.items()]
        })
        log.info(f'[{mode}] ts={ts} total_rps={rps:.1f} baseline={self._baseline_rps:.1f}')

    # ── 高峰期：按延迟动态调整配额 ─────────────────────────────────────────
    def _handle_peak(self):
        for did, c in self.cards.items():
            lat = c.avg_lat()
            if lat > self.slo_s * 0.8 and c.vector < MAX_VECTOR:
                nv = min(c.vector + self.vstep, MAX_VECTOR)
                nc = min(c.cube   + self.cstep, MAX_CUBE)
                log.info(f'[PEAK] card={did} lat={lat:.3f}s -> scale UP V:{c.vector}->{nv} C:{c.cube}->{nc}')
                _acl_set(did, ACL_RT_DEV_RES_VECTOR_CORE, nv, self.simulate)
                _acl_set(did, ACL_RT_DEV_RES_CUBE_CORE,   nc, self.simulate)
                c.vector, c.cube = nv, nc
            elif lat < self.slo_s * 0.4 and c.vector > 10:
                nv = max(c.vector - self.vstep, 10)
                nc = max(c.cube   - self.cstep,  4)
                log.info(f'[PEAK] card={did} lat={lat:.3f}s -> scale DN V:{c.vector}->{nv} C:{c.cube}->{nc}')
                _acl_set(did, ACL_RT_DEV_RES_VECTOR_CORE, nv, self.simulate)
                _acl_set(did, ACL_RT_DEV_RES_CUBE_CORE,   nc, self.simulate)
                c.vector, c.cube = nv, nc

    # ── 低谷期：整合空闲卡资源 ─────────────────────────────────────────────
    def _handle_valley(self):
        idle   = [c for c in self.cards.values() if c.is_idle()]
        active = [c for c in self.cards.values() if not c.is_idle()]
        if not idle:
            return
        freed_v = sum(c.vector for c in idle)
        freed_c = sum(c.cube   for c in idle)
        if active:
            target = max(active, key=lambda c: c.avg_rps())
            nv = min(target.vector + freed_v, MAX_VECTOR)
            nc = min(target.cube   + freed_c, MAX_CUBE)
            log.info(f'[VALLEY] Consolidate onto card={target.device_id} V->{nv} C->{nc}')
            _acl_set(target.device_id, ACL_RT_DEV_RES_VECTOR_CORE, nv, self.simulate)
            _acl_set(target.device_id, ACL_RT_DEV_RES_CUBE_CORE,   nc, self.simulate)
            target.vector, target.cube = nv, nc
        for c in idle:
            log.info(f'[VALLEY] Throttle idle card={c.device_id} to V=10 C=4')
            _acl_set(c.device_id, ACL_RT_DEV_RES_VECTOR_CORE, 10, self.simulate)
            _acl_set(c.device_id, ACL_RT_DEV_RES_CUBE_CORE,    4, self.simulate)
            c.vector, c.cube = 10, 4

    # ── 突发期：满配 + 横向扩容信号 ────────────────────────────────────────
    def _handle_burst(self):
        log.info('[BURST] Burst detected! Maxing all quotas')
        for did, c in self.cards.items():
            if c.vector < MAX_VECTOR or c.cube < MAX_CUBE:
                _acl_set(did, ACL_RT_DEV_RES_VECTOR_CORE, MAX_VECTOR, self.simulate)
                _acl_set(did, ACL_RT_DEV_RES_CUBE_CORE,   MAX_CUBE,   self.simulate)
                c.vector, c.cube = MAX_VECTOR, MAX_CUBE
        # 纵向已满 -> 触发横向扩容
        info = {'timestamp': datetime.now().isoformat(),
                'total_rps': self._total_rps(),
                'baseline':  self._baseline_rps}
        log.info(f'[BURST] Scale-out signal: {json.dumps(info)}')
        if self.scale_out_cb:
            self.scale_out_cb(info)

    def save_log(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._log, f, indent=2)
        log.info(f'Log saved: {path}')

    def print_status(self):
        h = datetime.now().hour
        mode = 'BURST' if self._is_burst() else ('PEAK' if self._is_peak() else 'VALLEY')
        print(f'\n=== NPUAdaptiveScaler [{mode}] ===')
        for did, c in self.cards.items():
            print(f'  Card {did}: V={c.vector:2d}/{MAX_VECTOR}  '
                  f'C={c.cube:2d}/{MAX_CUBE}  '
                  f'RPS={c.avg_rps():.1f}  '
                  f'Lat={c.avg_lat()*1000:.1f}ms  '
                  f'inst={c.instances}')


# ── CLI 演示 ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulate', action='store_true', default=True)
    parser.add_argument('--scenario',
                        choices=['peak', 'valley', 'burst'], default='burst')
    args = parser.parse_args()

    def on_scaleout(info):
        print(f'  [CALLBACK] Scale-out! rps={info["total_rps"]:.1f}')

    scaler = NPUAdaptiveScaler(simulate=args.simulate,
                               scale_out_cb=on_scaleout)
    # 设置基线
    scaler._baseline_rps = 5.0

    scenarios = {
        'peak':   [(3, 0.03, 1, 0.3), (6, 0.04, 2, 0.6), (9, 0.045, 2, 0.8)],
        'valley': [(2, 0.02, 1, 0.2), (0, 0.00, 0, 0.0), (1, 0.01, 0, 0.0)],
        'burst':  [(3, 0.02, 1, 0.3), (25, 0.08, 4, 0.95), (20, 0.07, 4, 0.9)],
    }

    print(f'[DEMO] Scenario: {args.scenario}')
    for i, (rps, lat, inst, load) in enumerate(scenarios[args.scenario]):
        print(f'\n--- Step {i+1}: RPS={rps} Lat={lat*1000:.0f}ms inst={inst} ---')
        for did in scaler.device_ids:
            scaler.update_metrics(did, rps, lat, inst, load)
        scaler._cycle()
        scaler.print_status()

    scaler.save_log('/tmp/dilu_npu/adaptive_scaler_log.json')
    print('\n[DEMO] Done.')

