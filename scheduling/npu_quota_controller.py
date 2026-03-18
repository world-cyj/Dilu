"""
npu_quota_controller.py  --  NPU Quota Controller with ACL Memory Tracking
==========================================================================
Day/Night 时序弹性 + acl.rt.malloc 显存追踪 + HGSS 画像预测
"""

import argparse
import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Dict, List

logging.basicConfig(
    format='%(asctime)s [NPUQuotaCtrl] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
log = logging.getLogger(__name__)

MAX_VECTOR = 40
MAX_CUBE   = 20
MAX_MEMORY = 64
NUM_CARDS  = 4
ACL_RT_DEV_RES_CUBE_CORE   = 0
ACL_RT_DEV_RES_VECTOR_CORE = 1
DAY_START   = 8
NIGHT_START = 22

try:
    import acl  # type: ignore
    ACL_AVAILABLE = True
except ImportError:
    ACL_AVAILABLE = False

_PROFILE_DB: Dict[str, dict] = {}
PROFILE_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'profiling', 'inference',
    'observations', 'profiling_results_npu')


def _load_profiles():
    import csv, glob
    for fpath in glob.glob(os.path.join(PROFILE_DIR, '*_hgss_npu.csv')):
        model = os.path.basename(fpath).replace('_hgss_npu.csv', '')
        best  = None
        try:
            with open(fpath) as f:
                for row in csv.DictReader(f):
                    if row.get('qos_ok', 'True') in ('True', '1', 'true'):
                        eff = float(row.get('efficiency', 0))
                        if best is None or eff > best['efficiency']:
                            best = {
                                'vector_req': float(row['vector_req']),
                                'cube_req':   float(row['cube_req']),
                                'vector_lim': float(row.get('vector_lim', 23)),
                                'cube_lim':   float(row.get('cube_lim', 10)),
                                'memory_gb':  float(row.get('memory_gb', 4)),
                                'efficiency': eff,
                            }
        except Exception as e:
            log.warning(f'Profile load failed {fpath}: {e}')
        if best:
            _PROFILE_DB[model] = best
            log.info(f'  Profile: {model} V={best["vector_req"]} '
                     f'C={best["cube_req"]} Mem={best["memory_gb"]}GB')


def predict_resource(model: str) -> dict:
    if model in _PROFILE_DB:
        return _PROFILE_DB[model]
    return {'vector_req': 20, 'cube_req': 8, 'vector_lim': 23,
            'cube_lim': 10, 'memory_gb': 4, 'efficiency': 0}


def _acl_set(device_id, res_type, value, simulate):
    name = 'VECTOR' if res_type == ACL_RT_DEV_RES_VECTOR_CORE else 'CUBE'
    log.info(f'  acl.rt.set_device_res_limit(device={device_id}, {name}, {value})')
    if not simulate and ACL_AVAILABLE:
        ret = acl.rt.set_device_res_limit(device_id, res_type, int(value))
        if ret != 0:
            log.error(f'set_device_res_limit ret={ret}')


def _acl_malloc(device_id, size_bytes, simulate):
    log.info(f'  acl.rt.malloc(device={device_id}, {size_bytes//1024//1024}MB)')
    if simulate or not ACL_AVAILABLE:
        return None
    try:
        acl.rt.set_device(device_id)
        buf, ret = acl.rt.malloc(size_bytes, 0)
        if ret == 0:
            return buf
        log.error(f'acl.rt.malloc ret={ret}')
    except Exception as e:
        log.warning(f'acl_malloc: {e}')
    return None


def _acl_free(buf, simulate):
    if simulate or not ACL_AVAILABLE or buf is None:
        return
    try:
        acl.rt.free(buf)
    except Exception:
        pass


def _acl_get_mem_info(device_id, simulate):
    if simulate or not ACL_AVAILABLE:
        return MAX_MEMORY * 1024**3, MAX_MEMORY * 1024**3
    try:
        acl.rt.set_device(device_id)
        free, total, ret = acl.rt.get_mem_info(0)
        if ret == 0:
            return free, total
    except Exception:
        pass
    return 0, MAX_MEMORY * 1024**3


class CardState:
    def __init__(self, device_id):
        self.device_id    = device_id
        self.vector       = MAX_VECTOR
        self.cube         = MAX_CUBE
        self.load         = 0.0
        self.instances    = 0
        self.alloc_memory = 0.0
        self._malloc_bufs = []

    def is_idle(self):
        return self.instances == 0

    def to_dict(self):
        return {
            'device_id':    self.device_id,
            'vector':       self.vector,
            'cube':         self.cube,
            'instances':    self.instances,
            'load':         round(self.load, 3),
            'alloc_mem_gb': round(self.alloc_memory, 2),
            'mem_util':     round(self.alloc_memory / MAX_MEMORY, 3),
        }


class NPUQuotaController:
    def __init__(self, device_ids=None, check_interval=60,
                 simulate=False, log_path=None):
        self.device_ids     = device_ids or list(range(NUM_CARDS))
        self.check_interval = check_interval
        self.simulate       = simulate
        self.cards: Dict[int, CardState] = {
            d: CardState(d) for d in self.device_ids}
        self._running = False
        self._history = []
        _load_profiles()
        log.info(f'NPUQuotaController init: devices={self.device_ids} '
                 f'simulate={simulate} interval={check_interval}s')
        if log_path:
            os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter(
                '%(asctime)s [NPUQuotaCtrl] %(levelname)s %(message)s'))
            log.addHandler(fh)

    def update_card_load(self, device_id, instances, load_fraction):
        if device_id in self.cards:
            self.cards[device_id].instances = instances
            self.cards[device_id].load      = load_fraction

    def allocate_instance(self, device_id, model='unknown', memory_gb=None):
        """为新实例分配显存并记录 AI Core 预测量。"""
        pred = predict_resource(model)
        if memory_gb is None:
            memory_gb = pred['memory_gb']
        buf = _acl_malloc(device_id, int(memory_gb * 1024**3), self.simulate)
        if buf and device_id in self.cards:
            self.cards[device_id]._malloc_bufs.append(buf)
            self.cards[device_id].alloc_memory += memory_gb
        log.info(f'[Alloc] device={device_id} model={model} '
                 f'V={pred["vector_req"]} C={pred["cube_req"]} '
                 f'Mem={memory_gb:.1f}GB')
        return pred['vector_req'], pred['cube_req'], memory_gb

    def free_instance(self, device_id, memory_gb=0):
        if device_id in self.cards:
            c = self.cards[device_id]
            if c._malloc_bufs:
                _acl_free(c._malloc_bufs.pop(), self.simulate)
            c.alloc_memory = max(0.0, c.alloc_memory - memory_gb)

    def get_status(self):
        return [c.to_dict() for c in self.cards.values()]

    def start(self):
        self._running = True
        log.info('NPUQuotaController started')
        while self._running:
            self._control_cycle()
            time.sleep(self.check_interval)

    def stop(self):
        self._running = False

    def _is_day(self):
        return DAY_START <= datetime.now().hour < NIGHT_START

    def _control_cycle(self):
        mode = 'DAY' if self._is_day() else 'NIGHT'
        log.info(f'--- Control cycle [{mode}] {datetime.now().strftime("%H:%M:%S")} ---')
        if self._is_day():
            self._apply_day_mode()
        else:
            self._apply_night_mode()
        self._history.append({'ts': datetime.now().isoformat(),
                              'mode': mode, 'cards': self.get_status()})
        if len(self._history) > 1440:
            self._history.pop(0)

    def _apply_day_mode(self):
        for did, c in self.cards.items():
            if c.vector != MAX_VECTOR or c.cube != MAX_CUBE:
                log.info(f'[DAY] card={did} restore V={MAX_VECTOR} C={MAX_CUBE}')
                _acl_set(did, ACL_RT_DEV_RES_VECTOR_CORE, MAX_VECTOR, self.simulate)
                _acl_set(did, ACL_RT_DEV_RES_CUBE_CORE,   MAX_CUBE,   self.simulate)
                c.vector, c.cube = MAX_VECTOR, MAX_CUBE

    def _apply_night_mode(self):
        idle = [c for c in self.cards.values() if c.is_idle()]
        busy = [c for c in self.cards.values() if not c.is_idle()]
        if not idle:
            log.info('[NIGHT] No idle cards')
            return
        fv = sum(c.vector for c in idle)
        fc = sum(c.cube   for c in idle)
        if busy:
            tgt = max(busy, key=lambda c: c.load)
            nv  = min(tgt.vector + fv, MAX_VECTOR)
            nc  = min(tgt.cube   + fc, MAX_CUBE)
            log.info(f'[NIGHT] Consolidate card={tgt.device_id} '
                     f'V:{tgt.vector}->{nv} C:{tgt.cube}->{nc}')
            _acl_set(tgt.device_id, ACL_RT_DEV_RES_VECTOR_CORE, nv, self.simulate)
            _acl_set(tgt.device_id, ACL_RT_DEV_RES_CUBE_CORE,   nc, self.simulate)
            tgt.vector, tgt.cube = nv, nc
        for c in idle:
            log.info(f'[NIGHT] Throttle card={c.device_id} to V=1 C=1')
            _acl_set(c.device_id, ACL_RT_DEV_RES_VECTOR_CORE, 1, self.simulate)
            _acl_set(c.device_id, ACL_RT_DEV_RES_CUBE_CORE,   1, self.simulate)
            c.vector, c.cube = 1, 1

    def print_status(self):
        mode = 'DAY' if self._is_day() else 'NIGHT'
        print(f'\n=== NPU Quota Controller [{mode}] ===')
        for did, c in self.cards.items():
            free, _ = _acl_get_mem_info(did, self.simulate)
            print(f'  Card {did}: V={c.vector:2d}/{MAX_VECTOR} '
                  f'C={c.cube:2d}/{MAX_CUBE} '
                  f'AllocMem={c.alloc_memory:.1f}GB '
                  f'FreeMem={free//1024**3:.1f}GB '
                  f'inst={c.instances} load={c.load:.2f}')

    def save_history(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._history, f, indent=2)
        log.info(f'History: {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulate',  action='store_true', default=True)
    parser.add_argument('--night',     action='store_true')
    parser.add_argument('--interval',  type=int, default=5)
    parser.add_argument('--cards',     nargs='+', type=int, default=[0,1,2,3])
    parser.add_argument('--log_path',  default='logs/npu_quota_ctrl.log')
    args = parser.parse_args()

    ctrl = NPUQuotaController(
        device_ids=args.cards, check_interval=args.interval,
        simulate=args.simulate, log_path=args.log_path)
    ctrl.update_card_load(0, instances=3, load_fraction=0.75)
    ctrl.update_card_load(1, instances=1, load_fraction=0.20)
    ctrl.update_card_load(2, instances=0, load_fraction=0.00)
    ctrl.update_card_load(3, instances=0, load_fraction=0.00)

    # Demo: allocate_instance with profile prediction
    v, c, m = ctrl.allocate_instance(0, model='resnet152')
    print(f'Predicted for resnet152: V={v} C={c} Mem={m}GB')

    if args.night:
        ctrl._apply_night_mode()
    else:
        ctrl._apply_day_mode()
    ctrl.print_status()
    ctrl.save_history('/tmp/dilu_npu/quota_history.json')
    print('[DEMO] Done.')
