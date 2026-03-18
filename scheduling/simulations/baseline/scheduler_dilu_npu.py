"""
Dilu-NPU Scheduler Simulation
==============================
Replaces GPU SM% with Ascend NPU 2-D resource model:
  - Vector Cores  (0-40 per card, ACL_RT_DEV_RES_VECTOR_CORE)
  - Cube   Cores  (0-20 per card, ACL_RT_DEV_RES_CUBE_CORE)
  - Memory        (0-64 GB per card)

Scoring (resource complementarity / best-fit):
  Score = alpha*(1-V_usage) + beta*(1-C_usage) + gamma*(1-M_usage)
  Lower score => denser packing => preferred.

Key idea: placing a Cube-heavy job beside a Vector-heavy job minimises
fragmentation in both dimensions simultaneously (Introspective Elasticity).
"""

import ast
import threading
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Physical NPU constants
TOTAL_VECTOR = 40
TOTAL_CUBE   = 20
TOTAL_MEMORY = 64
NUM_CARDS    = 4

# Hyper-parameters
alpha  = 0.35
beta   = 0.35
gamma  = 0.30
omega  = 1.2   # req超卖系数: req可超物理上限的1.2倍（Dilu原论文设定）
omega_c= 1.2   # cube req超卖系数
kappa  = 1.5   # 内存超卖系数


class PortManager:
    def __init__(self, start=15000, end=20000):
        self.available_ports = set(range(start, end + 1))
        self.lock = threading.Lock()

    def allocate_port(self):
        with self.lock:
            return self.available_ports.pop() if self.available_ports else None

    def release_port(self, port):
        with self.lock:
            if 15000 <= port <= 20000:
                self.available_ports.add(port)


class NPU:
    def __init__(self, npu_id, ip_address, index,
                 total_vector=TOTAL_VECTOR,
                 total_cube=TOTAL_CUBE,
                 total_memory=TOTAL_MEMORY):
        self.id            = npu_id
        self.ip_address    = ip_address
        self.index         = index
        self.total_vector  = total_vector
        self.total_cube    = total_cube
        self.total_memory  = total_memory
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

    def calculate_score(self, v_req, c_req, mem):
        """
        互补性评分 (Complementarity-aware scoring):
        - 对 Vector-heavy 任务 (v_req > c_req*2): 优先选 Vector 剩余少的卡
          (说明该卡已有 Cube-heavy 任务，形成互补)
        - 对 Cube-heavy 任务 (c_req > v_req): 优先选 Cube 剩余少的卡
        - 其余: 标准 best-fit
        分越低越优先。
        """
        v_remain = 1.0 - self.current_vector_req / self.total_vector
        c_remain = 1.0 - self.current_cube_req   / self.total_cube
        m_remain = 1.0 - self.current_memory     / self.total_memory

        is_vector_heavy = v_req > c_req * 2
        is_cube_heavy   = c_req > v_req

        if is_vector_heavy:
            # 优先放到 Cube 剩余多、Vector 剩余少的卡（互补）
            score = alpha * v_remain + beta * (1 - c_remain) + gamma * (1 - m_remain)
        elif is_cube_heavy:
            # 优先放到 Vector 剩余多、Cube 剩余少的卡（互补）
            score = alpha * (1 - v_remain) + beta * c_remain + gamma * (1 - m_remain)
        else:
            # 均衡任务：标准 best-fit
            score = alpha * (1 - v_remain) + beta * (1 - c_remain) + gamma * (1 - m_remain)
        return score

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
                 vector_req, vector_lim, cube_req, cube_lim, memory,
                 gpu_num, port):
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
        self.memory_partitions = []

    def assign_to_npu(self, npu):
        self.deployed_npus.append(npu)

    def record_mem_partition(self, mem):
        self.memory_partitions.append(mem)


# Global cluster state
nodes_info = []
for _node_ip in range(1000):
    for _i in range(NUM_CARDS):
        nodes_info.append({'ip': str(_node_ip), 'index': _i})

new_npus    = [NPU(i, node['ip'], node['index']) for i, node in enumerate(nodes_info)]
active_npus = []
lock        = threading.Lock()


def _extract_resources(data):
    """Extract 2-D NPU resource fields, with legacy sm_* fallback."""
    if 'vector_req' in data:
        return (data['vector_req'], data['vector_lim'],
                data['cube_req'],   data['cube_lim'])
    v_req = max(1, int(data['sm_requests'] * TOTAL_VECTOR))
    v_lim = max(1, int(data['sm_limits']   * TOTAL_VECTOR))
    c_req = max(1, int(data['sm_requests'] * TOTAL_CUBE))
    c_lim = max(1, int(data['sm_limits']   * TOTAL_CUBE))
    return v_req, v_lim, c_req, c_lim


def find_colocated_npus(service_name):
    candidates = set()
    for npu in active_npus:
        has_instance = any(inst.service_name == service_name
                           for inst in npu.instances.values())
        if has_instance:
            for inst in npu.instances.values():
                if inst.type == 'training':
                    for n in inst.deployed_npus:
                        candidates.add(n)
                    candidates.discard(npu)
    return list(candidates)


def select_optimal_npu(candidate_npus, v_req, c_req, v_lim, c_lim, mem):
    best_score = float('inf')
    best_npu   = None
    for npu in candidate_npus:
        if npu.can_allocate(v_req, c_req, v_lim, c_lim, mem):
            score = npu.calculate_score(v_req, c_req, mem)
            if score < best_score:
                best_score = score
                best_npu   = npu
    return best_npu


def _do_place(instance, v_req, c_req, v_lim, c_lim, mem, npu, selected):
    npu.allocate(instance, v_req, c_req, v_lim, c_lim, mem)
    instance.assign_to_npu(npu)
    selected.append({'id': npu.id, 'ip': npu.ip_address, 'index': npu.index})


def schedule_instance(data):
    instance_id  = data['service_name']
    port         = port_manager.allocate_port()
    task_type    = data['type']
    service_name = data['service_name']
    gpu_num      = data.get('gpu_num', 1)
    memory       = data['memory']
    selected     = []
    v_req, v_lim, c_req, c_lim = _extract_resources(data)

    instance = Instance(instance_id, service_name, task_type,
                        v_req, v_lim, c_req, c_lim, memory, gpu_num, port)

    if task_type in ('inference', 'llm-inference'):
        mem = memory[0]
        lb_npus = find_colocated_npus(service_name)
        best = select_optimal_npu(lb_npus, v_req, c_req, v_lim, c_lim, mem)
        if best is None:
            remaining = set(active_npus) - set(lb_npus)
            best = select_optimal_npu(list(remaining), v_req, c_req, v_lim, c_lim, mem)
        if best is None:
            best = select_optimal_npu(new_npus, v_req, c_req, v_lim, c_lim, mem)
            if best:
                new_npus.remove(best)
                active_npus.append(best)
        if best is None:
            return None
        _do_place(instance, v_req, c_req, v_lim, c_lim, mem, best, selected)

    elif task_type == 'training':
        node_map = {}
        for npu in active_npus:
            if all(npu.can_allocate(v_req, c_req, v_lim, c_lim, m) for m in memory):
                node_map.setdefault(npu.ip_address, []).append(npu)
        allocated = None
        for _, npus_on_node in node_map.items():
            if len(npus_on_node) >= gpu_num:
                allocated = npus_on_node[:gpu_num]
                break
        if allocated is None:
            for node in nodes_info:
                new_on_node    = [n for n in new_npus    if n.ip_address == node['ip']]
                active_on_node = node_map.get(node['ip'], [])
                if len(active_on_node) + len(new_on_node) >= gpu_num:
                    needed   = gpu_num - len(active_on_node)
                    promoted = new_on_node[:needed]
                    for n in promoted:
                        new_npus.remove(n)
                        active_npus.append(n)
                    allocated = active_on_node + promoted
                    break
        if not allocated or len(allocated) < gpu_num:
            return None
        for npu, mem in zip(allocated, memory):
            _do_place(instance, v_req, c_req, v_lim, c_lim, mem, npu, selected)

    return {'selected_npus': selected, 'instance_id': instance_id, 'port': port}


def delete_instance(event_data):
    instance_id = event_data['service_name']
    task_type   = event_data['type']
    v_req, v_lim, c_req, c_lim = _extract_resources(event_data)
    freed = []
    with lock:
        for npu in active_npus:
            if instance_id in npu.instances:
                inst = npu.instances[instance_id]
                if task_type == 'llm-inference' and inst.memory_partitions:
                    idx = inst.deployed_npus.index(npu)
                    mem = inst.memory_partitions[idx]
                else:
                    mem = inst.memory[0]
                npu.release(instance_id, v_req, c_req, v_lim, c_lim, mem)
                if not npu.instances:
                    freed.append(npu)
        for npu in freed:
            active_npus.remove(npu)
            new_npus.append(npu)


def calc_fragmentation():
    if not active_npus:
        return 0.0, 0.0, 0.0
    n = len(active_npus)
    vf = sum(1 - x.current_vector_req / x.total_vector for x in active_npus) / n
    cf = sum(1 - x.current_cube_req   / x.total_cube   for x in active_npus) / n
    mf = sum(1 - x.current_memory     / x.total_memory for x in active_npus) / n
    return vf, cf, mf


if __name__ == '__main__':
    import sys, os
    os.makedirs('logs', exist_ok=True)
    wf = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(__file__), '..', 'workload', 'instances-npu-3200.txt')
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
        if 660 <= max_npus <= 664:
            vf, cf, mf = calc_fragmentation()
            print(f'VectorFrag={vf:.3f} CubeFrag={cf:.3f} MemFrag={mf:.3f}')
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, npu_counts, '-', color='green', label='Dilu-NPU')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Active NPU cards')
    plt.title('Dilu-NPU: Active NPU Count Over Time')
    plt.tight_layout()
    plt.savefig('logs/dilu-npu-timeline.png', dpi=300)
    print(f'Max active NPU cards: {max_npus}')
