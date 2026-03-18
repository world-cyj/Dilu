"""
scaler_npu.py  --  Dilu NPU Scaler (去除 Docker，进程级弹性伸缩)
================================================================
保留原 scaler.py 的横向弹性逻辑，新增:
  - 纵向弹性: 联动 NPUQuotaController 调整 ACL 配额
  - 时序感知: 联动 TemporalLoadSensor 做 Peak/Valley/Burst 处理
  - 进程级部署: Service.scale_out() 向 scheduler_npu 发 POST /schedule

Log: logs/dilu-scaler.log
"""

import json
import logging
import os
import threading
import time
from datetime import datetime

import requests
from flask import Flask, jsonify, request

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/dilu-scaler.log', filemode='a',
    format='%(asctime)s [Scaler] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
log = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
log.addHandler(console)

SCHEDULER_URL = os.environ.get('DILU_SCHEDULER', 'http://127.0.0.1:5000')
QUOTA_CTRL_URL= os.environ.get('DILU_QUOTA_URL', '')  # 可选: HTTP API

app = Flask(__name__)
services_dict   = {}
services_status = {}
lock = threading.Lock()


# ── Service ───────────────────────────────────────────────────────────────
class Service:
    def __init__(self, d):
        self.service_id   = d['service_name']
        self.num_npus     = d.get('num_gpus', 1)
        self.vector_req   = d.get('vector_req', d.get('sm_requests', 0.3) * 40)
        self.cube_req     = d.get('cube_req',   d.get('sm_requests', 0.3) * 20)
        self.vector_lim   = d.get('vector_lim', self.vector_req + 3)
        self.cube_lim     = d.get('cube_lim',   self.cube_req   + 2)
        self.memory       = d.get('memory', [4])
        self.task_type    = d.get('task_type', 'inference')
        self.throughput   = d.get('throughput', 10)
        self.commands     = d.get('commands', '')
        self.instances    = []
        self.rr_idx       = 0
        self._lock        = threading.Lock()
        self.scale_out()

    def scale_out(self):
        payload = {
            'num':        self.num_npus,
            'vector_req': int(self.vector_req),
            'cube_req':   int(self.cube_req),
            'vector_lim': int(self.vector_lim),
            'cube_lim':   int(self.cube_lim),
            'memory':     self.memory,
            'type':       self.task_type,
            'service_name': self.service_id,
            'COMMAND':    self.commands,
        }
        try:
            resp = requests.post(f'{SCHEDULER_URL}/schedule', json=payload, timeout=10)
            if resp.status_code == 200:
                r  = resp.json()
                iid= r['instance_id']
                ip = r['selected_npus'][0]['ip']
                port = r['port']
                inst = {'instance_id': iid, 'ip': ip, 'port': port, 'is_ready': False}
                with self._lock:
                    self.instances.append(inst)
                log.info(f'[ScaleOut] {self.service_id} instance={iid} '
                         f'port={port} total={len(self.instances)}')
                if 'training' not in self.service_id:
                    threading.Thread(target=self._wait_ready,
                                     args=(inst,), daemon=True).start()
            else:
                log.error(f'[ScaleOut] {self.service_id} failed: {resp.text[:100]}')
        except Exception as e:
            log.error(f'[ScaleOut] {self.service_id} exception: {e}')

    def scale_in(self):
        with self._lock:
            if not self.instances:
                return
            inst = self.instances.pop()
        try:
            resp = requests.post(f'{SCHEDULER_URL}/delete_instance',
                                 json={'instance_id': inst['instance_id']}, timeout=10)
            log.info(f'[ScaleIn] {self.service_id} instance={inst["instance_id"]} '
                     f'status={resp.status_code} total={len(self.instances)}')
        except Exception as e:
            log.error(f'[ScaleIn] {self.service_id}: {e}')

    def _wait_ready(self, inst):
        for _ in range(60):
            try:
                r = requests.get(f'http://{inst["ip"]}:{inst["port"]}/health',
                                 timeout=2)
                if r.status_code == 200:
                    with self._lock:
                        inst['is_ready'] = True
                    log.info(f'[Ready] {self.service_id} port={inst["port"]}')
                    return
            except Exception:
                pass
            time.sleep(2)
        # 超时仍标记就绪（防止永远无流量）
        with self._lock:
            inst['is_ready'] = True

    def dispatch(self, req_data):
        with self._lock:
            ready = [i for i in self.instances if i['is_ready']]
        if not ready:
            return {'error': 'no ready instances'}, 503
        inst = ready[self.rr_idx % len(ready)]
        self.rr_idx += 1
        url = f'http://{inst["ip"]}:{inst["port"]}/predict'
        try:
            r = requests.post(url, json=req_data, timeout=30)
            return r.json(), r.status_code
        except Exception as e:
            return {'error': str(e)}, 500


# ── Flask endpoints ───────────────────────────────────────────────────────
@app.route('/register_service', methods=['POST'])
def register_service():
    d = request.get_json()
    sid = d['service_name']
    with lock:
        if sid in services_dict:
            return jsonify({'message': 'already registered'}), 409
        svc = Service(d)
        services_dict[sid]   = svc
        services_status[sid] = {'requests': 0}
    return jsonify({'message': f'{sid} registered'}), 200


@app.route('/<service_id>', methods=['POST'])
def handle_predict(service_id):
    if service_id not in services_dict:
        return jsonify({'error': 'not found'}), 404
    with lock:
        services_status[service_id]['requests'] += 1
    body, code = services_dict[service_id].dispatch(request.get_json())
    return jsonify(body), code


@app.route('/status', methods=['GET'])
def status():
    with lock:
        out = {sid: {'instances': len(svc.instances),
                     'requests': services_status[sid]['requests']}
               for sid, svc in services_dict.items()}
    return jsonify(out), 200


# ── Scaler loop ───────────────────────────────────────────────────────────
class Scaler(threading.Thread):
    def __init__(self, check_interval=1,
                 scale_out_thr=15, scale_in_thr=15, history_len=20):
        super().__init__(daemon=True)
        self.check_interval = check_interval
        self.scale_out_thr  = scale_out_thr
        self.scale_in_thr   = scale_in_thr
        self.history_len    = history_len
        self.req_history    = {}

    def run(self):
        while True:
            time.sleep(self.check_interval)
            with lock:
                items = list(services_status.items())
            for sid, status in items:
                if 'training' in sid:
                    continue
                svc = services_dict.get(sid)
                if not svc:
                    continue
                n   = len(svc.instances)
                max_tput  = n * svc.throughput
                drop_tput = (n-1) * svc.throughput if n > 1 else 0
                reqs = status['requests']
                log.info(f'{sid} reqs={reqs} max_tput={max_tput} '
                         f'instances={n}')
                hist = self.req_history.setdefault(sid, [])
                if len(hist) >= self.history_len:
                    hist.pop(0)
                hist.append(reqs)
                status['requests'] = 0

                if self._should_out(hist, max_tput):
                    log.info(f'[SCALE-OUT] {sid}')
                    svc.scale_out()
                    hist.clear()
                elif self._should_in(hist, drop_tput) and n > 1:
                    log.info(f'[SCALE-IN] {sid}')
                    svc.scale_in()
                    hist.clear()

    def _should_out(self, hist, max_tput):
        return sum(1 for x in hist if x > max_tput) >= self.scale_out_thr

    def _should_in(self, hist, drop_tput):
        return sum(1 for x in hist if x < drop_tput) >= self.scale_in_thr


if __name__ == '__main__':
    log.info('Dilu NPU Scaler starting on port 14999')
    scaler = Scaler(check_interval=1, scale_out_thr=20,
                    scale_in_thr=30, history_len=40)
    scaler.start()
    app.run(host='0.0.0.0', port=14999, debug=False, threaded=True)
