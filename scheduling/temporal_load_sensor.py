"""
temporal_load_sensor.py  --  时序负载感知模块
================================================
实现「黑白夜模式」时序负载感知，核心功能：
1. 实时监控每张NPU卡的请求速率、延迟、并发量
2. 构建滑动窗口统计模型，识别「高峰/低谷」模式
3. 趋势预测：感知即将到来的流量波动（主动防御）
4. 触发 acl.rt.set_device_res_limit 进行稳态纵向弹性
5. 空闲时段：将分散在多卡上的资源汇集到少数卡（夜间整合）
6. 突发流量：检测到激增后触发紧急横向扩容信号

时间段定义（可配置）：
  峰值期  (PEAK):    08:00-22:00  全量配额
  低谷期  (VALLEY):  22:00-08:00  整合配额
  突发期  (BURST):   任意时段流量超阈值
"""

import threading
import time
import logging
import json
import os
from collections import deque
from datetime import datetime
from typing import Dict, List, Callable, Optional, Tuple
from enum import Enum

logging.basicConfig(
    format='%(asctime)s [TemporalSensor] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ── 物理常量 ──────────────────────────────────────────────────────────────
MAX_VECTOR = 40
MAX_CUBE   = 20
NUM_CARDS  = 4

ACL_RT_DEV_RES_CUBE_CORE   = 0
ACL_RT_DEV_RES_VECTOR_CORE = 1

try:
    import acl  # type: ignore
    ACL_AVAILABLE = True
except ImportError:
    ACL_AVAILABLE = False


# ── 负载模式枚举 ──────────────────────────────────────────────────────────
class LoadMode(Enum):
    PEAK   = 'PEAK'    # 高峰期：全量配额
    VALLEY = 'VALLEY'  # 低谷期：整合配额
    BURST  = 'BURST'   # 突发期：紧急扩容
    NORMAL = 'NORMAL'  # 正常期：标准配额


# ── 配置 ─────────────────────────────────────────────────────────────────
class SensorConfig:
    def __init__(self):
        # 时间段
        self.peak_start_hour   = 8
        self.peak_end_hour     = 22
        # 滑动窗口
        self.window_size       = 60    # 秒
        self.sample_interval   = 5     # 秒
        # 突发检测
        self.burst_multiplier  = 2.0   # 相对baseline的倍数
        self.burst_window      = 30    # 秒内的突发检测窗口
        # 缩放触发阈值
        self.scale_up_lat_pct  = 0.80  # 延迟超过SLO的80%触发上调
        self.scale_dn_lat_pct  = 0.40  # 延迟低于SLO的40%触发下调
        # SLO（默认50ms）
        self.slo_latency_s     = 0.05
        # 配额步长
        self.vector_step       = 5
        self.cube_step         = 2
        # 低谷期最小配额（保留基础服务能力）
        self.valley_min_vector = 10
        self.valley_min_cube   = 4
        # 模拟模式
        self.simulate          = True


# ── 卡级运行时状态 ────────────────────────────────────────────────────────
class CardMetrics:
    def __init__(self, device_id: int, config: SensorConfig):
        self.device_id      = device_id
        self.config         = config
        # 当前配额
        self.vector_quota   = MAX_VECTOR
        self.cube_quota     = MAX_CUBE
        # 统计滑动窗口
        win = int(config.window_size / config.sample_interval) + 1
        self.req_rate_window  = deque(maxlen=win)  # 请求速率 req/s
        self.latency_window   = deque(maxlen=win)  # 平均延迟 s
        self.concurrency_win  = deque(maxlen=win)  # 并发实例数
        # 历史基线（用于突发检测）
        self.baseline_rps     = 0.0
        self.peak_rps         = 0.0
        # 累计计数
        self.total_requests   = 0
        self.sla_violations   = 0
        self.lock             = threading.Lock()

    def record_sample(self, req_rate: float, avg_latency: float, concurrency: int):
        with self.lock:
            self.req_rate_window.append(req_rate)
            self.latency_window.append(avg_latency)
            self.concurrency_win.append(concurrency)
            if avg_latency > self.config.slo_latency_s:
                self.sla_violations += 1
            if req_rate > self.peak_rps:
                self.peak_rps = req_rate

    def avg_rps(self) -> float:
        w = list(self.req_rate_window)
        return sum(w) / len(w) if w else 0.0

    def avg_latency(self) -> float:
        w = list(self.latency_window)
        return sum(w) / len(w) if w else 0.0

    def is_burst(self) -> bool:
        """检测突发：当前RPS > baseline * burst_multiplier"""
        rps = self.avg_rps()
        if self.baseline_rps < 0.1:
            return False
        return rps > self.baseline_rps * self.config.burst_multiplier

    def update_baseline(self):
        """更新基线：使用低谷期的平均RPS作为基线"""
        rps = self.avg_rps()
        if rps > 0:
            # 指数移动平均
            alpha = 0.1
            self.baseline_rps = alpha * rps + (1 - alpha) * self.baseline_rps


# ── ACL 配额下发 ──────────────────────────────────────────────────────────
def _set_quota(device_id: int, vector: int, cube: int, simulate: bool):
    v = max(1, min(vector, MAX_VECTOR))
    c = max(1, min(cube,   MAX_CUBE))
    log.info(f'  [ACL] device={device_id}  '
             f'acl.rt.set_device_res_limit(VECTOR={v})')
    log.info(f'  [ACL] device={device_id}  '
             f'acl.rt.set_device_res_limit(CUBE={c})')
    if not simulate:
        if ACL_AVAILABLE:
            acl.rt.set_device_res_limit(device_id, ACL_RT_DEV_RES_VECTOR_CORE, v)
            acl.rt.set_device_res_limit(device_id, ACL_RT_DEV_RES_CUBE_CORE,   c)
        else:
            log.warning('ACL not available, skipping real call')


# ── 主传感器类 ────────────────────────────────────────────────────────────
class TemporalLoadSensor:
    """
    核心时序负载感知控制器。
    三大功能：
      1. 时间段感知（黑白夜模式）
      2. 突发流量检测 + 紧急横向扩容信号
      3. 低谷期资源整合（分散→集中）
    """

    def __init__(self,
                 device_ids: List[int] = None,
                 config: SensorConfig = None,
                 scale_out_callback: Optional[Callable] = None,
                 scale_in_callback:  Optional[Callable] = None):
        self.device_ids  = device_ids or list(range(NUM_CARDS))
        self.config      = config or SensorConfig()
        self.scale_out_cb = scale_out_callback  # 触发横向扩容
        self.scale_in_cb  = scale_in_callback   # 触发横向缩容
        self.cards: Dict[int, CardMetrics] = {
            did: CardMetrics(did, self.config)
            for did in self.device_ids
        }
        self._running    = False
        self._mode       = LoadMode.NORMAL
        self._last_mode  = LoadMode.NORMAL
        self._history    = []  # 用于论文数据记录
        log.info(f'TemporalLoadSensor init: devices={self.device_ids} '
                 f'simulate={self.config.simulate}')

    # ── 外部数据注入（由调度器/scaler调用）─────────────────────────────────
    def report_metrics(self, device_id: int,
                       req_rate: float, avg_latency: float,
                       concurrency: int):
        if device_id in self.cards:
            self.cards[device_id].record_sample(req_rate, avg_latency, concurrency)

    # ── 控制循环 ────────────────────────────────────────────────────────────
    def start(self):
        self._running = True
        log.info('TemporalLoadSensor started')
        while self._running:
            self._control_cycle()
            time.sleep(self.config.sample_interval)

    def stop(self):
        self._running = False
        log.info('TemporalLoadSensor stopped')

    def _detect_mode(self) -> LoadMode:
        h = datetime.now().hour
        # 检查突发：任一卡触发
        if any(c.is_burst() for c in self.cards.values()):
            return LoadMode.BURST
        if self.config.peak_start_hour <= h < self.config.peak_end_hour:
            return LoadMode.PEAK
        return LoadMode.VALLEY

    def _control_cycle(self):
        mode = self._detect_mode()
        if mode != self._last_mode:
            log.info(f'=== Mode transition: {self._last_mode.value} -> {mode.value} ===')
            self._last_mode = mode
        self._mode = mode

        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log.info(f'--- Control cycle [{mode.value}] {ts} ---')

        if mode == LoadMode.BURST:
            self._handle_burst()
        elif mode == LoadMode.PEAK:
            self._handle_peak()
        elif mode == LoadMode.VALLEY:
            self._handle_valley()
            for c in self.cards.values():
                c.update_baseline()

        self._record_history(mode)

    def _handle_peak(self):
        """高峰期：根据延迟动态调整配额（在Request~Limit区间内）"""
        for did, card in self.cards.items():
            lat = card.avg_latency()
            slo = self.config.slo_latency_s
            cfg = self.config

            if lat > slo * cfg.scale_up_lat_pct and card.vector_quota < MAX_VECTOR:
                # 延迟过高，上调配额
                nv = min(card.vector_quota + cfg.vector_step, MAX_VECTOR)
                nc = min(card.cube_quota   + cfg.cube_step,   MAX_CUBE)
                log.info(f'[PEAK] card={did} lat={lat:.3f}s > {slo*cfg.scale_up_lat_pct:.3f}s '
                         f'-> scale UP  V:{card.vector_quota}->{nv}  C:{card.cube_quota}->{nc}')
                _set_quota(did, nv, nc, cfg.simulate)
                card.vector_quota, card.cube_quota = nv, nc

            elif lat < slo * cfg.scale_dn_lat_pct and card.vector_quota > cfg.valley_min_vector:
                # 延迟很低，可下调配额释放资源
                nv = max(card.vector_quota - cfg.vector_step, cfg.valley_min_vector)
                nc = max(card.cube_quota   - cfg.cube_step,   cfg.valley_min_cube)
                log.info(f'[PEAK] card={did} lat={lat:.3f}s < {slo*cfg.scale_dn_lat_pct:.3f}s '
                         f'-> scale DOWN V:{card.vector_quota}->{nv}  C:{card.cube_quota}->{nc}')
                _set_quota(did, nv, nc, cfg.simulate)
                card.vector_quota, card.cube_quota = nv, nc
            else:
                log.info(f'[PEAK] card={did} lat={lat:.3f}s  quota stable '
                         f'V={card.vector_quota} C={card.cube_quota}')

    def _handle_valley(self):
        """
        低谷期资源整合：
        - 找出空闲卡（concurrency==0）
        - 将空闲卡的算力配额集中到负载最重的卡
        - 空闲卡降至最小配额
        """
        idle   = [c for c in self.cards.values()
                  if (list(c.concurrency_win) or [0])[-1] == 0]
        active = [c for c in self.cards.values()
                  if (list(c.concurrency_win) or [0])[-1] > 0]

        if not idle:
            log.info('[VALLEY] No idle cards, maintaining current quotas')
            return

        # 汇集空闲配额到负载最重的卡
        freed_v = sum(c.vector_quota for c in idle)
        freed_c = sum(c.cube_quota   for c in idle)

        if active:
            target = max(active, key=lambda c: c.avg_rps())
            nv = min(target.vector_quota + freed_v, MAX_VECTOR)
            nc = min(target.cube_quota   + freed_c, MAX_CUBE)
            log.info(f'[VALLEY] Consolidating {len(idle)} idle cards onto '
                     f'card={target.device_id}  '
                     f'V:{target.vector_quota}->{nv}  C:{target.cube_quota}->{nc}')
            _set_quota(target.device_id, nv, nc, self.config.simulate)
            target.vector_quota = nv
            target.cube_quota   = nc

        # 空闲卡降至最小
        for card in idle:
            cfg = self.config
            log.info(f'[VALLEY] Throttling idle card={card.device_id} '
                     f'to V={cfg.valley_min_vector} C={cfg.valley_min_cube}')
            _set_quota(card.device_id,
                       cfg.valley_min_vector, cfg.valley_min_cube,
                       cfg.simulate)
            card.vector_quota = cfg.valley_min_vector
            card.cube_quota   = cfg.valley_min_cube

    def _handle_burst(self):
        """
        突发流量处理：
        1. 立即将所有卡配额调满
        2. 触发横向扩容回调（由外部scaler处理新实例启动）
        """
        log.info('[BURST] Burst detected! Maximizing all card quotas')
        for did, card in self.cards.items():
            if card.vector_quota < MAX_VECTOR or card.cube_quota < MAX_CUBE:
                _set_quota(did, MAX_VECTOR, MAX_CUBE, self.config.simulate)
                card.vector_quota = MAX_VECTOR
                card.cube_quota   = MAX_CUBE
                log.info(f'[BURST] card={did} quota maxed: V={MAX_VECTOR} C={MAX_CUBE}')

        # 触发横向扩容
        burst_info = {
            'timestamp': datetime.now().isoformat(),
            'trigger':   'burst',
            'avg_rps':   {did: c.avg_rps() for did, c in self.cards.items()},
            'baseline':  {did: c.baseline_rps for did, c in self.cards.items()},
        }
        log.info(f'[BURST] Scale-out signal: {json.dumps(burst_info)}')
        if self.scale_out_cb:
            self.scale_out_cb(burst_info)

    def _record_history(self, mode: LoadMode):
        record = {
            'ts':   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'mode': mode.value,
            'cards': [
                {
                    'device_id':    did,
                    'vector_quota': c.vector_quota,
                    'cube_quota':   c.cube_quota,
                    'avg_rps':      round(c.avg_rps(), 2),
                    'avg_lat':      round(c.avg_latency(), 4),
                    'sla_viol':     c.sla_violations,
                }
                for did, c in self.cards.items()
            ]
        }
        self._history.append(record)

    def save_history(self, path: str):
        with open(path, 'w') as f:
            json.dump(self._history, f, indent=2)
        log.info(f'History saved: {path}')

    def print_status(self):
        print('\n=== Temporal Load Sensor Status ===')
        print(f'  Mode: {self._mode.value}  Time: {datetime.now().strftime("%H:%M:%S")}')
        for did, card in self.cards.items():
            print(f'  Card {did}: V={card.vector_quota:2d}/{MAX_VECTOR}  '
                  f'C={card.cube_quota:2d}/{MAX_CUBE}  '
                  f'RPS={card.avg_rps():.1f}  '
                  f'Lat={card.avg_latency()*1000:.1f}ms  '
                  f'SLA_viol={card.sla_violations}')
        print()

    def get_metrics_summary(self) -> dict:
        """供论文数据采集使用"""
        total_viol = sum(c.sla_violations for c in self.cards.values())
        total_reqs = sum(c.total_requests for c in self.cards.values())
        svr = total_viol / max(total_reqs, 1)
        return {
            'mode':          self._mode.value,
            'sla_viol_rate': round(svr, 4),
            'avg_rps_total': round(sum(c.avg_rps() for c in self.cards.values()), 2),
            'cards':         [
                {'device_id': did, 'vector': c.vector_quota,
                 'cube': c.cube_quota, 'rps': round(c.avg_rps(), 2)}
                for did, c in self.cards.items()
            ]
        }


# ── 演示/验证入口 ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulate', action='store_true', default=True)
    parser.add_argument('--scenario',
                        choices=['peak', 'valley', 'burst', 'cycle'],
                        default='cycle')
    parser.add_argument('--steps', type=int, default=6)
    args = parser.parse_args()

    cfg            = SensorConfig()
    cfg.simulate   = args.simulate
    cfg.sample_interval = 1  # 演示时缩短

    def on_scale_out(info):
        print(f'[CALLBACK] Scale-out triggered! avg_rps={info["avg_rps"]}')

    sensor = TemporalLoadSensor(config=cfg, scale_out_callback=on_scale_out)

    scenarios = {
        'peak':   [(3.0, 0.03, 2), (5.0, 0.04, 2), (8.0, 0.04, 3)],
        'valley': [(1.0, 0.01, 1), (0.5, 0.01, 0), (0.0, 0.00, 0)],
        'burst':  [(2.0, 0.02, 1), (20.0, 0.08, 4), (15.0, 0.07, 4)],
        'cycle':  [(2.0, 0.02, 1), (8.0, 0.045, 3),
                   (20.0, 0.09, 4), (3.0, 0.03, 2),
                   (0.5, 0.01, 0), (1.0, 0.01, 1)],
    }

    steps = scenarios.get(args.scenario, scenarios['cycle'])
    print(f'[DEMO] Scenario: {args.scenario}')

    for i, (rps, lat, conc) in enumerate(steps[:args.steps]):
        print(f'\n--- Step {i+1}: RPS={rps} Lat={lat*1000:.0f}ms Conc={conc} ---')
        for did in sensor.device_ids:
            sensor.report_metrics(did, rps, lat, conc)
        sensor._control_cycle()
        sensor.print_status()
        time.sleep(0.5)

    os.makedirs('/tmp/dilu_npu', exist_ok=True)
    sensor.save_history('/tmp/dilu_npu/temporal_history.json')
    print('\n[DEMO] Done. History saved to /tmp/dilu_npu/temporal_history.json')
