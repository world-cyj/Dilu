"""
scheduler_npu.py  --  Dilu NPU Process-based Scheduler
=======================================================
去除 Docker，改用 subprocess 直接启动推理进程。
4 张 910B3 NPU 卡通过 ASCEND_RT_VISIBLE_DEVICES 隔离。
ACL 接口: acl.rt.set_device_res_limit / acl.rt.malloc

API (Flask, port 5000):
  POST /schedule        → 选最优卡，启动推理进程
  POST /delete_instance → 终止进程，释放资源
  GET  /status          → 集群资源快照
"""

import logging
import os
import subprocess
import threading
import uuid
from flask import Flask, jsonify, request

logging.basicConfig(
    format='%(asctime)s [Scheduler] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
log = logging.getLogger(__name__)

# ── 物理常量 ──────────────────────────────────────────────────────────────
TOTAL_VECTOR = 40
TOTAL_CUBE   = 20
TOTAL_MEMORY = 64
omega = 1.2
kappa = 1.5
alpha, beta, gamma = 0.35, 0.35, 0.30

nodes_info = [
    {'ip': '127.0.0.1', 'index': 0},
    {'ip': '127.0.0.1', 'index': 1},
    {'ip': '127.0.0.1', 'index': 2},
    {'ip': '127.0.0.1', 'index': 3},
]

WORKLOADS_DIR = os.environ.get(
    'DILU_WORKLOADS', '/mnt/caoyujia/Dilu/evaluation/scripts')


# ── ACL helpers ───────────────────────────────────────────────────────────
def _acl_set_quota(device_index, vector_lim, cube_lim):
    try:
        import acl
        acl.rt.set_device_res_limit(device_index, 1, max(1, int(vector_lim)))
        acl.rt.set_device_res_limit(device_index, 0, max(1, int(cube_lim)))
        log.info(f'  ACL quota: device={device_index} V={vector_lim} C={cube_lim}')
    except Exception:
        pass


def _acl_malloc(device_index, size_bytes):
    try:
        import acl
        acl.rt.set_device(device_index)
        buf, ret = acl.rt.malloc(size_bytes, 0)
        if ret == 0:
            log.info(f'  acl.rt.malloc device={device_index} '
                     f'size={size_bytes//1024//1024}MB OK')
            return buf
    except Exception as e:
        log.debug(f'acl malloc skipped: {e}')
    return None


def _acl_free(buf):
    try:
        import acl
        if buf:
            acl.rt.free(buf)
    except Exception:
        pass


# ── NPU 类 ────────────────────────────────────────────────────────────────
class NPU:
    def __init__(self, npu_id, ip, index):
        self.id = npu_id
        self.ip_address = ip
        self.index = index
        self.total_vector = TOTAL_VECTOR
        self.total_cube   = TOTAL_CUBE
        self.total_memory = TOTAL_MEMORY
        self.current_vector_req = 0
        self.current_cube_req   = 0
        self.current_vector_lim = 0
        self.current_cube_lim   = 0
        self.current_memory     = 0.0
        self.instances = {}

    def can_allocate(self, v_req, c_req, v_lim, c_lim, mem):
        return (
            self.current_vector_req + v_req <= omega * self.total_vector and
            self.current_cube_req   + c_req <= omega * self.total_cube   and
            self.current_vector_lim + v_lim <= self.total_vector         and
            self.current_cube_lim   + c_lim <= self.total_cube           and
            self.current_memory     + mem   <= kappa * self.total_memory
        )

    def calculate_score(self, v_req, c_req, mem):
        v_rem = 1.0 - self.current_vector_req / self.total_vector
        c_rem = 1.0 - self.current_cube_req   / self.total_cube
        m_rem = 1.0 - self.current_memory     / self.total_memory
        if v_req > c_req * 2:
            return alpha*v_rem + beta*(1-c_rem) + gamma*(1-m_rem)
        elif c_req > v_req:
            return alpha*(1-v_rem) + beta*c_rem + gamma*(1-m_rem)
        return alpha*(1-v_rem) + beta*(1-c_rem) + gamma*(1-m_rem)

    def allocate(self, inst, v_req, c_req, v_lim, c_lim, mem):
        self.current_vector_req += v_req
        self.current_cube_req   += c_req
        self.current_vector_lim += v_lim
        self.current_cube_lim   += c_lim
        self.current_memory     += mem
        self.instances[inst.instance_id] = inst
        _acl_set_quota(self.index, self.current_vector_lim, self.current_cube_lim)

    def release(self, instance_id):
        if instance_id not in self.instances:
            return
        inst = self.instances.pop(instance_id)
        self.current_vector_req -= inst.v_req
        self.current_cube_req   -= inst.c_req
        self.current_vector_lim -= inst.v_lim
        self.current_cube_lim   -= inst.c_lim
        mem = sum(inst.memory) if isinstance(inst.memory, list) else inst.memory
        self.current_memory -= mem
        _acl_set_quota(self.index, max(0, self.current_vector_lim),
                       max(0, self.current_cube_lim))

    def is_empty(self):
        return (not self.instances and
                self.current_memory == 0 and
                self.current_vector_req == 0)


# ── Instance 类 ───────────────────────────────────────────────────────────
class Instance:
    def __init__(self, instance_id, service_name, task_type,
                 v_req, c_req, v_lim, c_lim, memory, port, proc):
        self.instance_id  = instance_id
        self.service_name = service_name
        self.type         = task_type
        self.v_req = v_req; self.c_req = c_req
        self.v_lim = v_lim; self.c_lim = c_lim
        self.memory       = memory
        self.port         = port
        self.proc         = proc
        self.deployed_npus = []


# ── PortManager ───────────────────────────────────────────────────────────
class PortManager:
    def __init__(self):
        self._ports = set(range(15000, 20001))
        self._lock  = threading.Lock()
    def allocate(self):
        with self._lock:
            return self._ports.pop() if self._ports else None
    def release(self, port):
        with self._lock:
            if 15000 <= port <= 20000:
                self._ports.add(port)


# ── 进程启动/停止 ─────────────────────────────────────────────────────────
def _start_process(selected_npus_info, instance_id, data, port):
    card_indices = [n['index'] for n in selected_npus_info]
    visible = ','.join(str(i) for i in card_indices)
    env = os.environ.copy()
    env['ASCEND_RT_VISIBLE_DEVICES'] = visible
    env['DILU_INSTANCE_ID']  = instance_id
    env['DILU_SERVICE_PORT'] = str(port)

    cmd = data.get('COMMAND', '')
    if not cmd:
        model  = data.get('model', 'resnet152')
        script = os.path.join(WORKLOADS_DIR, f'run_{model}_INF_batch.py')
        cmd    = f'python {script} --port {port}'
    else:
        cmd = f'{cmd} --port {port}'

    log.info(f'Start proc ASCEND_RT_VISIBLE_DEVICES={visible}: {cmd[:80]}...')
    try:
        proc = subprocess.Popen(cmd, shell=True, env=env,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        return proc
    except Exception as e:
        log.error(f'Process start failed: {e}')
        return None


def _stop_process(proc, instance_id):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    log.info(f'Stopped instance={instance_id}')


# ── 集群全局状态 ──────────────────────────────────────────────────────────
app         = Flask(__name__)
pm          = PortManager()
new_npus    = [NPU(i, n['ip'], n['index']) for i, n in enumerate(nodes_info)]
active_npus = []
lock        = threading.Lock()
_malloc_bufs= {}   # instance_id → acl buf


def _find_colocated(service_name):
    cands = set()
    for npu in active_npus:
        if any(i.service_name == service_name for i in npu.instances.values()):
            for inst in npu.instances.values():
                if inst.type == 'training':
                    for n in inst.deployed_npus:
                        cands.add(n)
            cands.discard(npu)
    return list(cands)


def _select_best(candidates, v_req, c_req, v_lim, c_lim, mem):
    best_s, best = float('inf'), None
    for npu in candidates:
        if npu.can_allocate(v_req, c_req, v_lim, c_lim, mem):
            s = npu.calculate_score(v_req, c_req, mem)
            if s < best_s:
                best_s, best = s, npu
    return best


def _deploy(inst, npu, v_req, c_req, v_lim, c_lim, mem):
    """分配资源 + ACL 显存 malloc 追踪。"""
    npu.allocate(inst, v_req, c_req, v_lim, c_lim, mem)
    inst.deployed_npus.append(npu)
    mem_bytes = int(mem * 1024**3)
    buf = _acl_malloc(npu.index, mem_bytes)
    if buf:
        _malloc_bufs[inst.instance_id] = buf


# ── /schedule ─────────────────────────────────────────────────────────────
@app.route('/schedule', methods=['POST'])
def schedule_instance():
    data        = request.get_json()
    instance_id = str(uuid.uuid4())
    port        = pm.allocate()
    if not port:
        return jsonify({'error': 'No ports'}), 503

    v_req     = int(data.get('vector_req', 20))
    c_req     = int(data.get('cube_req',    8))
    v_lim     = int(data.get('vector_lim', v_req + 3))
    c_lim     = int(data.get('cube_lim',   c_req + 2))
    memory    = data.get('memory', [4])
    mem_total = sum(memory) if isinstance(memory, list) else memory
    n_npus    = int(data.get('num', 1))
    task_type = data.get('type', 'inference')
    svc_name  = data.get('service_name', 'unknown')
    selected  = []

    with lock:
        if task_type in ('inference', 'llm-inference'):
            # 互补性感知调度
            coloc = _find_colocated(svc_name)
            best  = _select_best(coloc, v_req, c_req, v_lim, c_lim, mem_total)
            if not best:
                best = _select_best(set(active_npus)-set(coloc),
                                    v_req, c_req, v_lim, c_lim, mem_total)
            if not best:
                best = _select_best(new_npus, v_req, c_req, v_lim, c_lim, mem_total)
                if best:
                    new_npus.remove(best)
                    active_npus.append(best)
            if not best:
                pm.release(port)
                return jsonify({'error': 'No resources'}), 400

            proc = _start_process([{'index': best.index}], instance_id, data, port)
            inst = Instance(instance_id, svc_name, task_type,
                            v_req, c_req, v_lim, c_lim, memory, port, proc)
            _deploy(inst, best, v_req, c_req, v_lim, c_lim, mem_total)
            selected = [{'id': best.id, 'ip': best.ip_address, 'index': best.index}]

        elif task_type == 'training':
            mem_per = memory[0] if isinstance(memory, list) else memory
            # 同节点优先
            alloc = []
            for npu in active_npus:
                if (len(alloc) < n_npus and
                        npu.can_allocate(v_req, c_req, v_lim, c_lim, mem_per)):
                    alloc.append(npu)
            while len(alloc) < n_npus and new_npus:
                npu = new_npus.pop(0)
                active_npus.append(npu)
                alloc.append(npu)
            if len(alloc) < n_npus:
                pm.release(port)
                return jsonify({'error': 'Not enough NPUs'}), 400

            proc = _start_process([{'index': n.index} for n in alloc],
                                   instance_id, data, port)
            mem_list = memory if isinstance(memory, list) else [memory]*n_npus
            inst = Instance(instance_id, svc_name, task_type,
                            v_req, c_req, v_lim, c_lim, memory, port, proc)
            for npu, m in zip(alloc, mem_list):
                _deploy(inst, npu, v_req, c_req, v_lim, c_lim, m)
            selected = [{'id': n.id, 'ip': n.ip_address, 'index': n.index}
                        for n in alloc]
        else:
            pm.release(port)
            return jsonify({'error': f'Unknown type: {task_type}'}), 400

    log.info(f'Scheduled {instance_id} type={task_type} '
             f'npus={[s["index"] for s in selected]} port={port}')
    return jsonify({'instance_id': instance_id,
                    'selected_npus': selected, 'port': port}), 200


# ── /delete_instance ──────────────────────────────────────────────────────
@app.route('/delete_instance', methods=['POST'])
def delete_instance():
    data        = request.get_json()
    instance_id = data.get('instance_id')
    found       = False
    with lock:
        for npu in list(active_npus):
            if instance_id in npu.instances:
                inst = npu.instances[instance_id]
                _stop_process(inst.proc, instance_id)
                buf = _malloc_bufs.pop(instance_id, None)
                _acl_free(buf)
                npu.release(instance_id)
                pm.release(inst.port)
                if npu.is_empty():
                    active_npus.remove(npu)
                    new_npus.append(npu)
                found = True
                break
    if not found:
        return jsonify({'error': 'Not found'}), 404
    return jsonify({'status': 'deleted', 'instance_id': instance_id}), 200


# ── /status ───────────────────────────────────────────────────────────────
@app.route('/status', methods=['GET'])
def status():
    with lock:
        cards = []
        for npu in active_npus:
            cards.append({
                'index':      npu.index,
                'vector_req': npu.current_vector_req,
                'cube_req':   npu.current_cube_req,
                'vector_lim': npu.current_vector_lim,
                'cube_lim':   npu.current_cube_lim,
                'memory_gb':  npu.current_memory,
                'instances':  len(npu.instances),
            })
    return jsonify({'active_npus': cards,
                    'free_npus': len(new_npus)}), 200


if __name__ == '__main__':
    log.info('Dilu NPU Scheduler starting on port 5000')
    log.info(f'NPU cards: {[n["index"] for n in nodes_info]}')
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

