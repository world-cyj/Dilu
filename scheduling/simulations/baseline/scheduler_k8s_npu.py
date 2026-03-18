"""
K8s-NPU Baseline Scheduler (Exclusive allocation)
===================================================
Simulates Kubernetes-style exclusive allocation on Ascend NPU:
- Each instance gets its own NPU card (no sharing).
- No resource complementarity awareness.
- 2-D resource fields (vector/cube) accepted but not used for scoring.
"""

import ast
import threading
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TOTAL_VECTOR = 40
TOTAL_CUBE   = 20
TOTAL_MEMORY = 64
NUM_CARDS    = 4
omega = 1.0
omega_c = 1.0
kappa = 2.0   # K8s: generous over-commit on memory


class PortManager:
    def __init__(self, start=15000, end=20000):
        self.available_ports = set(range(start, end + 1))
        self.lock = threading.Lock()

    def allocate_port(self):
        with self.lock:
            return self.available_ports.pop() if self.available_ports else None


class NPU:
    def __init__(self, npu_id, ip_address, index):
        self.id            = npu_id
        self.ip_address    = ip_address
        self.index         = index
        self.total_vector  = TOTAL_VECTOR
        self.total_cube    = TOTAL_CUBE
        self.total_memory  = TOTAL_MEMORY
        self.current_vector_req = 0
        self.current_cube_req   = 0
        self.current_vector_lim = 0
        self.current_cube_lim   = 0
        self.current_memory     = 0.0
        self.instances          = {}

    def can_allocate(self, v_req, c_req, v_lim, c_lim, mem):
        return (
            self.current_vector_req + v_req <= omega   * self.total_vector and
            self.current_cube_req   + c_req <= omega_c * self.total_cube   and
            self.current_vector_lim + v_lim <= self.total_vector           and
            self.current_cube_lim   + c_lim <= self.total_cube             and
            self.current_memory     + mem   <= kappa  * self.total_memory
        )

    def allocate(self, instance, v_req, c_req, v_lim, c_lim, mem):
        self.current_vector_req += v_req
        self.current_cube_req   += c_req
        self.current_vector_lim += v_lim
        self.current_cube_lim   += c_lim
        self.current_memory     += mem
        self.instances[instance.instance_id] = instance

    def release(self, instance_id, v_req, c_req, v_lim, c_lim, mem):
        self.current_vector_req -= v_req
        self.current_cube_req   -= c_req
        self.current_vector_lim -= v_lim
        self.current_cube_lim   -= c_lim
        self.current_memory     -= mem
        self.instances.pop(instance_id, None)


class Instance:
    def __init__(self, instance_id, service_name, task_type,
                 vector_req, vector_lim, cube_req, cube_lim, memory, gpu_num, port):
        self.instance_id   = instance_id
        self.service_name  = service_name
        self.type          = task_type
        self.vector_req    = vector_req
        self.vector_lim    = vector_lim
        self.cube_req      = cube_req
        self.cube_lim      = cube_lim
        self.memory        = memory
        self.gpu_num       = gpu_num
        self.port          = port
        self.deployed_npus = []

    def assign_to_npu(self, npu):
        self.deployed_npus.append(npu)


nodes_info = []
for _ip in range(3000):
    for _i in range(NUM_CARDS):
        nodes_info.append({'ip': str(_ip), 'index': _i})

new_npus    = [NPU(i, node['ip'], node['index']) for i, node in enumerate(nodes_info)]
active_npus = []
lock        = threading.Lock()


def _extract_resources(data):
    if 'vector_req' in data:
        return data['vector_req'], data['vector_lim'], data['cube_req'], data['cube_lim']
    v_req = max(1, int(data['sm_requests'] * TOTAL_VECTOR))
    v_lim = max(1, int(data['sm_limits']   * TOTAL_VECTOR))
    c_req = max(1, int(data['sm_requests'] * TOTAL_CUBE))
    c_lim = max(1, int(data['sm_limits']   * TOTAL_CUBE))
    return v_req, v_lim, c_req, c_lim


def schedule_instance(data):
    instance_id = data['service_name']
    port        = port_manager.allocate_port()
    task_type   = data['type']
    gpu_num     = data.get('gpu_num', 1)
    memory      = data['memory']
    selected    = []
    v_req, v_lim, c_req, c_lim = _extract_resources(data)

    instance = Instance(instance_id, data['service_name'], task_type,
                        v_req, v_lim, c_req, c_lim, memory, gpu_num, port)

    if task_type in ('inference', 'llm-inference'):
        mem = memory[0]
        # K8s: just take the next free NPU card (exclusive)
        best = new_npus[0] if new_npus else None
        if best and best.can_allocate(v_req, c_req, v_lim, c_lim, mem):
            best.allocate(instance, v_req, c_req, v_lim, c_lim, mem)
            instance.assign_to_npu(best)
            selected.append({'id': best.id, 'ip': best.ip_address, 'index': best.index})
            new_npus.remove(best)
            active_npus.append(best)
        else:
            return None

    elif task_type == 'training':
        mem = memory[0]
        for _ in range(gpu_num):
            best = new_npus[0] if new_npus else None
            if best and best.can_allocate(v_req, c_req, v_lim, c_lim, mem):
                best.allocate(instance, v_req, c_req, v_lim, c_lim, mem)
                instance.assign_to_npu(best)
                selected.append({'id': best.id, 'ip': best.ip_address, 'index': best.index})
                new_npus.remove(best)
                active_npus.append(best)
            else:
                return None

    return {'selected_npus': selected, 'instance_id': instance_id, 'port': port}


def delete_instance(event_data):
    instance_id = event_data['service_name']
    v_req, v_lim, c_req, c_lim = _extract_resources(event_data)
    freed = []
    with lock:
        for npu in active_npus:
            if instance_id in npu.instances:
                inst = npu.instances[instance_id]
                mem  = inst.memory[0]
                npu.release(instance_id, v_req, c_req, v_lim, c_lim, mem)
                if not npu.instances:
                    freed.append(npu)
        for npu in freed:
            active_npus.remove(npu)
            new_npus.append(npu)


def calc_fragmentation():
    if not active_npus:
        return 0.0, 0.0, 0.0
    n  = len(active_npus)
    vf = sum(1 - x.current_vector_req / x.total_vector for x in active_npus) / n
    cf = sum(1 - x.current_cube_req   / x.total_cube   for x in active_npus) / n
    mf = sum(1 - x.current_memory     / x.total_memory for x in active_npus) / n
    return vf, cf, mf


if __name__ == '__main__':
    import sys, os
    os.makedirs('logs', exist_ok=True)
    wf = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(__file__), '..', 'workload', 'instances-npu-100.txt')
    port_manager = PortManager()
    with open(wf) as f:
        events = [ast.literal_eval(line.strip()) for line in f]
    timestamps, npu_counts = [], []
    max_npus  = 0
    base_time = datetime.strptime(events[0]['Time'], '%Y-%m-%d %H:%M:%S')
    for event in events:
        t  = datetime.strptime(event['Time'], '%Y-%m-%d %H:%M:%S')
        td = (t - base_time).total_seconds() / 60.0
        if event['Action'] == 'start':
            schedule_instance(event['Instance'])
        else:
            delete_instance(event['Instance'])
        timestamps.append(td)
        npu_counts.append(len(active_npus))
        if len(active_npus) > max_npus:
            max_npus = len(active_npus)
        if 30 <= max_npus <= 34:
            vf, cf, mf = calc_fragmentation()
            print(f'VectorFrag={vf:.3f} CubeFrag={cf:.3f} MemFrag={mf:.3f}')
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, npu_counts, ':', color='red', label='K8s-NPU')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Active NPU cards')
    plt.title('K8s-NPU (Exclusive): Active NPU Count Over Time')
    plt.tight_layout()
    plt.savefig('logs/k8s-npu-timeline.png', dpi=300)
    print(f'Max active NPU cards: {max_npus}')
