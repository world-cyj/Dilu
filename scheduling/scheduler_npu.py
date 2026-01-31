# -*- coding: utf-8 -*-
"""
Dilu NPU 调度器：单机 4 张 910B3 NPU，Cube/Vector/Memory 资源模型。
Best-Fit（inference）、Worst-Fit（llm-inference）、多卡训练（training）。
实例启动使用 utils_npu（本机进程 + ACL 设限），无 Docker。
"""
import os
import sys
_sched_dir = os.path.dirname(os.path.abspath(__file__))
if _sched_dir not in sys.path:
    sys.path.insert(0, _sched_dir)

from flask import Flask, request, jsonify
import threading
import uuid
import time
import requests

from npu_resource import NPU, build_npu_nodes_info, NPU_CUBE_CORES_PER_DEVICE, NPU_VECTOR_CORES_PER_DEVICE, NPU_MEMORY_GB_PER_DEVICE
import utils_npu

# 导入资源推荐引擎
try:
    from resource_recommender import get_recommender, ResourceRecommender
    _recommender_available = True
except ImportError as e:
    print(f"[scheduler_npu] Resource recommender not available: {e}")
    _recommender_available = False


class PortManager:
    def __init__(self, start=15000, end=20000):
        self.available_ports = set(range(start, end + 1))
        self.lock = threading.Lock()

    def allocate_port(self):
        with self.lock:
            if not self.available_ports:
                return None
            return self.available_ports.pop()

    def release_port(self, port):
        with self.lock:
            if 15000 <= port <= 20000:
                self.available_ports.add(port)


class Instance:
    def __init__(self, id, memory, cube_requests, cube_limits, vector_requests, vector_limits, image, type, service_name, allocated_port):
        self.instance_id = id
        self.memory = memory
        self.cube_req = cube_requests
        self.cube_lim = cube_limits
        self.vector_req = vector_requests
        self.vector_lim = vector_limits
        self.image = image
        self.type = type
        self.service_name = service_name
        self.deployed_npus = []
        self.port = allocated_port
        self.status = "starting"
        self.created_at = time.time()
        self.last_health_at = None

    def assign_to_npu(self, deployed_npu):
        self.deployed_npus.append(deployed_npu)


app = Flask(__name__)
nodes_info = build_npu_nodes_info()

alpha = 0.5
beta = 0.4
delta = 0.1
omega = 1
gamma = 1.5
new_npus = [
    NPU(i, NPU_MEMORY_GB_PER_DEVICE, NPU_CUBE_CORES_PER_DEVICE, NPU_VECTOR_CORES_PER_DEVICE, n["ip"], n["index"])
    for i, n in enumerate(nodes_info)
]
active_npus = []
lock = threading.Lock()
# 运行态实例与任务状态（内存态，demo 级）
instance_registry = {}
task_registry = {}


def find_colocated_NPUs(service_name):
    candidate = set()
    for npu in active_npus:
        has_early = any(inst.service_name == service_name for inst in npu.instances.values())
        if not has_early:
            continue
        for inst in npu.instances.values():
            if inst.type == "training":
                for n in inst.deployed_npus:
                    candidate.add(n)
                candidate.discard(npu)
                break
    return list(candidate)


def select_optimal_NPU(candidate_npus, cube_req, cube_lim, vector_req, vector_lim, memory, omega, gamma, alpha, beta, delta, strategy="best_fit"):
    """
    选择最优NPU，支持Best-Fit和Worst-Fit策略

    Args:
        strategy: "best_fit" 或 "worst_fit"
            - best_fit: 选择资源最紧缺的NPU（最小化碎片）
            - worst_fit: 选择资源最充足的NPU（避免资源碎片化）
    """
    if strategy == "worst_fit":
        # Worst-Fit: 选择资源最充足的NPU（分数最大）
        best_score = -1.0
        best_npu = None
        for npu in candidate_npus:
            if npu.can_allocate(cube_req, cube_lim, vector_req, vector_lim, memory, omega, gamma):
                score = npu.calculate_worst_fit_score(cube_req, vector_req, memory, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_npu = npu
        return best_npu
    else:
        # Best-Fit: 选择资源最紧缺的NPU（分数最小）
        best_score = float("inf")
        best_npu = None
        for npu in candidate_npus:
            if npu.can_allocate(cube_req, cube_lim, vector_req, vector_lim, memory, omega, gamma):
                score = npu.calculate_score(cube_req, vector_req, memory, alpha, beta, delta)
                if score < best_score:
                    best_score = score
                    best_npu = npu
        return best_npu


def select_npus_for_training(candidate_npus, n_npus_needed, cube_req, cube_lim, vector_req, vector_lim, memory_list, omega, gamma):
    """
    为训练任务选择NPU，使用Worst-Fit策略

    Training任务特点：
    1. 需要多卡协同
    2. 需要预留足够资源空间
    3. 优先选择同一节点的NPU

    Returns:
        选中的NPU列表，如果无法满足则返回空列表
    """
    if len(memory_list) < n_npus_needed:
        return []

    # 按节点分组
    node_npus = {}
    for npu in candidate_npus:
        if npu.ip_address not in node_npus:
            node_npus[npu.ip_address] = []
        node_npus[npu.ip_address].append(npu)

    # 对每个节点内的NPU按Worst-Fit排序（资源越充足越优先）
    for node_ip in node_npus:
        node_npus[node_ip].sort(
            key=lambda n: n.calculate_worst_fit_score(cube_req, vector_req, memory_list[0] if memory_list else 8),
            reverse=True
        )

    # 寻找满足条件的节点
    for node_ip, npus_on_node in node_npus.items():
        if len(npus_on_node) < n_npus_needed:
            continue

        # 检查是否所有NPU都能满足资源需求
        can_allocate_all = True
        allocated_npus = []

        for i in range(n_npus_needed):
            npu = npus_on_node[i]
            mem_req = memory_list[i] if i < len(memory_list) else memory_list[-1]
            if not npu.can_allocate(cube_req, cube_lim, vector_req, vector_lim, mem_req, omega, gamma):
                can_allocate_all = False
                break
            allocated_npus.append(npu)

        if can_allocate_all:
            return allocated_npus

    return []


def update_deploy_info(instance, cube_req, cube_lim, vector_req, vector_lim, memory, best_npu, selected_npus):
    best_npu.update_resources(instance, cube_req, cube_lim, vector_req, vector_lim, memory)
    instance.assign_to_npu(best_npu)
    selected_npus.append({"id": best_npu.id, "ip": best_npu.ip_address, "index": best_npu.index})


def print_cluster_resources():
    print("NPU Cluster Resource Usage:")
    print("============== active_npus ==================")
    for npu in active_npus:
        print(f"NPU {npu.id} (index={npu.index}): Cube {npu.current_cube_req}/{npu.total_cube}, Vector {npu.current_vector_req}/{npu.total_vector}, Mem {npu.current_memory}/{npu.total_memory} GB, instances={len(npu.instances)}")
    print("============== new_npus ==================")
    for npu in new_npus:
        print(f"NPU {npu.id} (index={npu.index}): Cube {npu.current_cube_req}/{npu.total_cube}, Vector {npu.current_vector_req}/{npu.total_vector}, Mem {npu.current_memory}/{npu.total_memory} GB, instances={len(npu.instances)}")
    print("")


def start_instance(selected_npus, instance_id, args, allocated_port):
    ip_address = selected_npus[0]["ip"]
    utils_npu.start_instance(selected_npus, instance_id, args.get("image", ""), args["service_name"], args, allocated_port, ip_address)


def stop_instance(service_name, instance_id, ip_address):
    utils_npu.stop_instance(service_name, instance_id, ip_address)


def _memory_list(memory):
    if isinstance(memory, list):
        return memory
    return [int(memory)]


def _now_ts():
    return time.time()


def _instance_snapshot(instance):
    mem_used = instance.memory[0] if isinstance(instance.memory, list) and instance.memory else instance.memory
    return {
        "instance_id": instance.instance_id,
        "service_name": instance.service_name,
        "type": instance.type,
        "status": instance.status,
        "port": instance.port,
        "cube_req": instance.cube_req,
        "cube_lim": instance.cube_lim,
        "vector_req": instance.vector_req,
        "vector_lim": instance.vector_lim,
        "memory": mem_used,
        "deployed_npus": [{"id": n.id, "index": n.index, "ip": n.ip_address} for n in instance.deployed_npus],
        "created_at": instance.created_at,
        "last_health_at": instance.last_health_at,
    }


def _cluster_snapshot():
    def _npu_state(npu):
        return {
            "id": npu.id,
            "index": npu.index,
            "ip": npu.ip_address,
            "cube_total": npu.total_cube,
            "cube_used": npu.current_cube_req,
            "vector_total": npu.total_vector,
            "vector_used": npu.current_vector_req,
            "memory_total": npu.total_memory,
            "memory_used": npu.current_memory,
            "instances": len(npu.instances),
        }
    return {
        "active": [_npu_state(n) for n in active_npus],
        "new": [_npu_state(n) for n in new_npus],
    }


def _wait_for_health(instance, timeout=30):
    if not instance.port:
        return
    url = f"http://127.0.0.1:{instance.port}/health"
    start = _now_ts()
    while _now_ts() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                instance.status = "running"
                instance.last_health_at = _now_ts()
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    instance.status = "unhealthy"


def allocate_instance(data):
    instance_id = str(uuid.uuid4())
    allocated_port = port_manager.allocate_port()
    if not allocated_port:
        return None, None, None, None, ({"error": "No available ports"}, 503)

    n_npus_needed = data.get("num", 1)
    cube_req = float(data.get("cube_requests", data.get("sm_requests", 0.25)) * NPU_CUBE_CORES_PER_DEVICE)
    cube_lim = float(data.get("cube_limits", data.get("sm_limits", 0.75)) * NPU_CUBE_CORES_PER_DEVICE)
    vector_req = float(data.get("vector_requests", data.get("sm_requests", 0.25)) * NPU_VECTOR_CORES_PER_DEVICE)
    vector_lim = float(data.get("vector_limits", data.get("sm_limits", 0.75)) * NPU_VECTOR_CORES_PER_DEVICE)
    memory = _memory_list(data.get("memory", [8]))
    image = data.get("image", "")
    type_ = data.get("type", "inference")
    service_name = data.get("service_name", "default")

    instance = Instance(
        instance_id, memory, cube_req, cube_lim, vector_req, vector_lim,
        image, type_, service_name, allocated_port
    )
    selected_npus = []

    with lock:
        if type_ == "inference":
            LB_npus = find_colocated_NPUs(service_name)
            best_npu = select_optimal_NPU(LB_npus, cube_req, cube_lim, vector_req, vector_lim, memory[0], omega, gamma, alpha, beta, delta)
            if best_npu:
                update_deploy_info(instance, cube_req, cube_lim, vector_req, vector_lim, memory[0], best_npu, selected_npus)
            else:
                left = set(active_npus) - set(LB_npus)
                best_npu = select_optimal_NPU(left, cube_req, cube_lim, vector_req, vector_lim, memory[0], omega, gamma, alpha, beta, delta)
                if best_npu:
                    update_deploy_info(instance, cube_req, cube_lim, vector_req, vector_lim, memory[0], best_npu, selected_npus)
                else:
                    best_npu = select_optimal_NPU(new_npus, cube_req, cube_lim, vector_req, vector_lim, memory[0], omega, gamma, alpha, beta, delta)
                    if best_npu:
                        update_deploy_info(instance, cube_req, cube_lim, vector_req, vector_lim, memory[0], best_npu, selected_npus)
                        new_npus.remove(best_npu)
                        active_npus.append(best_npu)
                    else:
                        return None, None, None, None, ({"error": "Not enough resources"}, 400)

        elif type_ == "llm-inference":
            best_npu = None
            best_fit_score = float("inf")
            for npu in active_npus:
                mem = memory[0] if memory else 0
                if npu.can_allocate(cube_req, cube_lim, vector_req, vector_lim, mem, omega, gamma):
                    if npu.current_memory < best_fit_score:
                        best_fit_score = npu.current_memory
                        best_npu = npu
            if best_npu:
                mem = memory[0] if memory else 0
                best_npu.update_resources(instance, cube_req, cube_lim, vector_req, vector_lim, mem)
                instance.assign_to_npu(best_npu)
                selected_npus.append({"id": best_npu.id, "ip": best_npu.ip_address, "index": best_npu.index})
            else:
                node_groups = {}
                for npu in active_npus:
                    if npu.can_allocate(cube_req, cube_lim, vector_req, vector_lim, 0, omega, gamma):
                        node_groups.setdefault(npu.ip_address, []).append(npu)
                found = False
                req_mem = memory[0] if memory else 0
                for node, npus_on_node in node_groups.items():
                    npus_on_node.sort(key=lambda x: x.current_memory, reverse=True)
                    total_avail = sum(n.total_memory - n.current_memory for n in npus_on_node)
                    if total_avail >= req_mem:
                        remaining = req_mem
                        for npu in npus_on_node:
                            if remaining <= 0:
                                break
                            alloc_mem = min(npu.total_memory - npu.current_memory, remaining)
                            npu.update_resources(instance, cube_req, cube_lim, vector_req, vector_lim, alloc_mem)
                            instance.assign_to_npu(npu)
                            selected_npus.append({"id": npu.id, "ip": npu.ip_address, "index": npu.index})
                            remaining -= alloc_mem
                        found = True
                        break
                if not found and new_npus:
                    new_npu = new_npus.pop(0)
                    mem = memory[0] if memory else 0
                    new_npu.update_resources(instance, cube_req, cube_lim, vector_req, vector_lim, mem)
                    instance.assign_to_npu(new_npu)
                    active_npus.append(new_npu)
                    selected_npus.append({"id": new_npu.id, "ip": new_npu.ip_address, "index": new_npu.index})
                elif not found:
                    return None, None, None, None, ({"error": "Unable to allocate resources, no new NPUs available"}, 400)

        elif type_ == "training":
            # Training任务使用Worst-Fit策略，优先选择资源最充足的NPU
            # 1. 首先在active_npus中查找
            allocated = select_npus_for_training(
                active_npus, n_npus_needed, cube_req, cube_lim, vector_req, vector_lim, memory, omega, gamma
            )

            # 2. 如果active_npus不够，尝试从new_npus补充
            if not allocated and len(memory) >= n_npus_needed:
                all_npus = active_npus + new_npus
                allocated = select_npus_for_training(
                    all_npus, n_npus_needed, cube_req, cube_lim, vector_req, vector_lim, memory, omega, gamma
                )
                # 将新使用的NPU从new_npus移到active_npus
                if allocated:
                    for npu in allocated:
                        if npu in new_npus:
                            new_npus.remove(npu)
                        if npu not in active_npus:
                            active_npus.append(npu)

            if allocated and len(allocated) == n_npus_needed:
                for npu, mem in zip(allocated, memory):
                    npu.update_resources(instance, cube_req, cube_lim, vector_req, vector_lim, mem)
                    instance.assign_to_npu(npu)
                    selected_npus.append({"id": npu.id, "ip": npu.ip_address, "index": npu.index})
            else:
                return None, None, None, None, ({"error": "Not enough resources for training"}, 400)
        else:
            return None, None, None, None, ({"error": "Unknown type"}, 400)

    return instance, selected_npus, allocated_port, memory, None


@app.route("/schedule", methods=["POST"])
def schedule_instances():
    data = request.get_json()
    instance, selected_npus, allocated_port, memory, err = allocate_instance(data)
    if err:
        payload, status = err
        return jsonify(payload), status

    # 将解析后的核心数传入 worker，供 utils_npu 设置 ACL
    args_for_worker = dict(data)
    args_for_worker["cube_limits"] = int(instance.cube_lim)
    args_for_worker["vector_limits"] = int(instance.vector_lim)
    args_for_worker["memory"] = memory[0] if isinstance(memory, list) and memory else memory
    thrd = threading.Thread(target=start_instance, args=(selected_npus, instance.instance_id, args_for_worker, allocated_port))
    thrd.start()
    instance_registry[instance.instance_id] = instance
    threading.Thread(target=_wait_for_health, args=(instance,), daemon=True).start()
    print_cluster_resources()
    return jsonify({"selected_npus": selected_npus, "instance_id": instance.instance_id, "port": allocated_port}), 200


@app.route("/delete_instance", methods=["POST"])
def delete_instance():
    data = request.get_json()
    instance_id = data.get("instance_id")
    if not instance_id:
        return jsonify({"error": "instance_id required"}), 400

    allocated_port = None
    service_name = None
    ip_address = None
    found = False
    with lock:
        for npu in active_npus:
            if instance_id in npu.instances:
                instance = npu.instances.pop(instance_id)
                npu.current_cube_req -= instance.cube_req
                npu.current_cube_lim -= instance.cube_lim
                npu.current_vector_req -= instance.vector_req
                npu.current_vector_lim -= instance.vector_lim
                mem_used = instance.memory[0] if isinstance(instance.memory, list) and instance.memory else (instance.memory if isinstance(instance.memory, (int, float)) else 0)
                npu.current_memory -= mem_used
                found = True
                ip_address = instance.deployed_npus[0].ip_address
                allocated_port = instance.port
                service_name = instance.service_name
                if not npu.instances and npu.current_memory == 0 and npu.current_cube_req == 0 and npu.current_cube_lim == 0 and npu.current_vector_req == 0 and npu.current_vector_lim == 0:
                    active_npus.remove(npu)
                    new_npus.append(npu)
                break

    if not found:
        return jsonify({"error": "Instance not found"}), 404

    port_manager.release_port(allocated_port)
    thrd = threading.Thread(target=stop_instance, args=(service_name, instance_id, ip_address))
    thrd.start()
    if instance_id in instance_registry:
        instance_registry[instance_id].status = "stopped"
    print_cluster_resources()
    return jsonify({"status": "deleted", "instance_id": instance_id}), 200


@app.route("/reschedule_with_limits", methods=["POST"])
def reschedule_with_limits():
    """IE 垂直伸缩：按新 request/limit 替换实例（先删后建），实现 2D co-scaling 的垂直维。"""
    data = request.get_json()
    instance_id = data.get("instance_id")
    new_cube_limits = data.get("new_cube_limits")
    new_vector_limits = data.get("new_vector_limits")
    new_cube_requests = data.get("new_cube_requests")
    new_vector_requests = data.get("new_vector_requests")
    if not instance_id or new_cube_limits is None or new_vector_limits is None:
        return jsonify({"error": "instance_id, new_cube_limits, new_vector_limits required"}), 400
    # 从 body 取服务模板（与 /schedule 一致），用于重建
    template = {
        "num": data.get("num", 1),
        "cube_limits": float(new_cube_limits),
        "vector_limits": float(new_vector_limits),
        "cube_requests": float(new_cube_requests) if new_cube_requests is not None else float(new_cube_limits) * 0.8,
        "vector_requests": float(new_vector_requests) if new_vector_requests is not None else float(new_vector_limits) * 0.8,
        "memory": data.get("memory", [8]),
        "type": data.get("type", "inference"),
        "service_name": data.get("service_name", "default"),
        "image": data.get("image", ""),
        "COMMAND": data.get("COMMAND", ""),
        "MODEL_PATH": data.get("MODEL_PATH", ""),
    }
    # 先删
    with lock:
        found = False
        for npu in active_npus:
            if instance_id in npu.instances:
                instance = npu.instances.pop(instance_id)
                npu.current_cube_req -= instance.cube_req
                npu.current_cube_lim -= instance.cube_lim
                npu.current_vector_req -= instance.vector_req
                npu.current_vector_lim -= instance.vector_lim
                mem_used = instance.memory[0] if isinstance(instance.memory, list) and instance.memory else (instance.memory if isinstance(instance.memory, (int, float)) else 0)
                npu.current_memory -= mem_used
                found = True
                if not npu.instances and npu.current_memory == 0 and npu.current_cube_req == 0 and npu.current_cube_lim == 0 and npu.current_vector_req == 0 and npu.current_vector_lim == 0:
                    active_npus.remove(npu)
                    new_npus.append(npu)
                break
    if not found:
        return jsonify({"error": "Instance not found"}), 404
    port_manager.release_port(instance.port)
    threading.Thread(target=stop_instance, args=(instance.service_name, instance_id, instance.deployed_npus[0].ip_address)).start()
    if instance_id in instance_registry:
        instance_registry[instance_id].status = "stopped"
    # 等待旧进程完全退出，释放端口
    time.sleep(3)
    # 再建
    instance_new, selected_npus, allocated_port, memory, err = allocate_instance(template)
    if err:
        payload, status = err
        return jsonify(payload), status
    args_for_worker = dict(template)
    args_for_worker["cube_limits"] = int(instance_new.cube_lim)
    args_for_worker["vector_limits"] = int(instance_new.vector_lim)
    args_for_worker["memory"] = memory[0] if isinstance(memory, list) and memory else memory
    threading.Thread(target=start_instance, args=(selected_npus, instance_new.instance_id, args_for_worker, allocated_port)).start()
    instance_registry[instance_new.instance_id] = instance_new
    threading.Thread(target=_wait_for_health, args=(instance_new,), daemon=True).start()
    print_cluster_resources()
    return jsonify({
        "old_instance_id": instance_id,
        "instance_id": instance_new.instance_id,
        "port": allocated_port,
        "selected_npus": selected_npus,
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return "ok", 200


@app.route("/instances", methods=["GET"])
def list_instances():
    return jsonify({"instances": [_instance_snapshot(i) for i in instance_registry.values()]}), 200


@app.route("/instance/<instance_id>", methods=["GET"])
def get_instance(instance_id):
    inst = instance_registry.get(instance_id)
    if not inst:
        return jsonify({"error": "Instance not found"}), 404
    return jsonify(_instance_snapshot(inst)), 200


@app.route("/cluster", methods=["GET"])
def get_cluster():
    return jsonify(_cluster_snapshot()), 200


@app.route("/metrics", methods=["GET"])
def get_metrics():
    snapshot = _cluster_snapshot()
    totals = {
        "cube_total": sum(n["cube_total"] for n in snapshot["active"]) + sum(n["cube_total"] for n in snapshot["new"]),
        "cube_used": sum(n["cube_used"] for n in snapshot["active"]) + sum(n["cube_used"] for n in snapshot["new"]),
        "vector_total": sum(n["vector_total"] for n in snapshot["active"]) + sum(n["vector_total"] for n in snapshot["new"]),
        "vector_used": sum(n["vector_used"] for n in snapshot["active"]) + sum(n["vector_used"] for n in snapshot["new"]),
        "memory_total": sum(n["memory_total"] for n in snapshot["active"]) + sum(n["memory_total"] for n in snapshot["new"]),
        "memory_used": sum(n["memory_used"] for n in snapshot["active"]) + sum(n["memory_used"] for n in snapshot["new"]),
    }
    def _ratio(used, total):
        return float(used) / float(total) if total else 0.0
    totals["cube_utilization"] = _ratio(totals["cube_used"], totals["cube_total"])
    totals["vector_utilization"] = _ratio(totals["vector_used"], totals["vector_total"])
    totals["memory_utilization"] = _ratio(totals["memory_used"], totals["memory_total"])
    return jsonify({"cluster": snapshot, "totals": totals}), 200


@app.route("/submit_task", methods=["POST"])
def submit_task():
    data = request.get_json()
    payload = data.pop("payload", {}) if isinstance(data, dict) else {}
    task_id = str(uuid.uuid4())

    instance, selected_npus, allocated_port, memory, err = allocate_instance(data)
    if err:
        payload, status = err
        return jsonify(payload), status

    args_for_worker = dict(data)
    args_for_worker["cube_limits"] = int(instance.cube_lim)
    args_for_worker["vector_limits"] = int(instance.vector_lim)
    args_for_worker["memory"] = memory[0] if isinstance(memory, list) and memory else memory
    threading.Thread(target=start_instance, args=(selected_npus, instance.instance_id, args_for_worker, allocated_port)).start()
    instance_registry[instance.instance_id] = instance
    _wait_for_health(instance, timeout=30)

    task_registry[task_id] = {
        "task_id": task_id,
        "instance_id": instance.instance_id,
        "status": "running",
        "created_at": _now_ts(),
        "result": None,
    }
    result = None
    try:
        r = requests.post(f"http://127.0.0.1:{allocated_port}/predict", json=payload, timeout=60)
        if r.status_code == 200:
            result = r.json()
            task_registry[task_id]["status"] = "finished"
        else:
            result = {"error": r.text, "status_code": r.status_code}
            task_registry[task_id]["status"] = "failed"
    except requests.RequestException as e:
        result = {"error": str(e)}
        task_registry[task_id]["status"] = "failed"
    task_registry[task_id]["result"] = result

    return jsonify({
        "task_id": task_id,
        "instance_id": instance.instance_id,
        "selected_npus": selected_npus,
        "result": result,
    }), 200


@app.route("/task/<task_id>", methods=["GET"])
def get_task(task_id):
    task = task_registry.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task), 200


@app.route("/schedule_with_sla", methods=["POST"])
def schedule_with_sla():
    """
    基于SLA的智能调度端点
    
    请求体示例：
    {
        "service_name": "my-service",
        "model_path": "/vllm-workspace/models/Qwen3-4B",
        "sla_latency": 0.2,  # 目标延迟（秒）
        "sla_throughput": 5.0,  # 目标吞吐（可选）
        "batch_size": 1,
        "type": "inference"
    }
    
    返回：包含推荐资源配置的调度结果
    """
    if not _recommender_available:
        return jsonify({"error": "Resource recommender not available"}), 503
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    service_name = data.get("service_name")
    model_path = data.get("model_path")
    sla_latency = data.get("sla_latency")
    sla_throughput = data.get("sla_throughput")
    batch_size = data.get("batch_size", 1)
    task_type = data.get("type", "inference")
    
    if not service_name:
        return jsonify({"error": "service_name required"}), 400
    
    # 获取资源推荐
    try:
        recommender = get_recommender()
        schedule_request = recommender.generate_schedule_request(
            service_name=service_name,
            model_path=model_path or "",
            sla_latency=sla_latency,
            sla_throughput=sla_throughput,
            batch_size=batch_size,
            task_type=task_type
        )
        
        # 合并用户提供的其他参数
        for key in ["image", "COMMAND", "priority"]:
            if key in data:
                schedule_request[key] = data[key]
        
        # 执行调度
        instance, selected_npus, allocated_port, memory, err = allocate_instance(schedule_request)
        if err:
            payload, status = err
            return jsonify(payload), status
        
        # 启动实例
        args_for_worker = dict(schedule_request)
        args_for_worker["cube_limits"] = int(instance.cube_lim)
        args_for_worker["vector_limits"] = int(instance.vector_lim)
        args_for_worker["memory"] = memory[0] if isinstance(memory, list) and memory else memory
        
        thrd = threading.Thread(target=start_instance, args=(selected_npus, instance.instance_id, args_for_worker, allocated_port))
        thrd.start()
        instance_registry[instance.instance_id] = instance
        threading.Thread(target=_wait_for_health, args=(instance,), daemon=True).start()
        
        print_cluster_resources()
        
        # 返回包含推荐信息的响应
        response = {
            "selected_npus": selected_npus,
            "instance_id": instance.instance_id,
            "port": allocated_port,
            "recommendation": {
                "cube_requests": schedule_request["cube_requests"],
                "cube_limits": schedule_request["cube_limits"],
                "vector_requests": schedule_request["vector_requests"],
                "vector_limits": schedule_request["vector_limits"],
                "memory": schedule_request["memory"],
                "expected_latency": schedule_request.get("expected_latency"),
                "expected_throughput": schedule_request.get("expected_throughput"),
                "confidence": schedule_request.get("recommendation_confidence")
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to generate recommendation: {str(e)}"}), 500


@app.route("/recommend_resources", methods=["POST"])
def recommend_resources():
    """
    获取资源配置建议（不实际调度）
    
    请求体示例：
    {
        "model_path": "/vllm-workspace/models/Qwen3-4B",
        "sla_latency": 0.2,
        "batch_size": 1
    }
    
    返回：推荐的资源配置
    """
    if not _recommender_available:
        return jsonify({"error": "Resource recommender not available"}), 503
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    sla_latency = data.get("sla_latency")
    sla_throughput = data.get("sla_throughput")
    batch_size = data.get("batch_size", 1)
    
    try:
        recommender = get_recommender()
        
        if sla_latency is not None:
            config = recommender.recommend_for_latency(sla_latency, batch_size)
            recommendation_type = "latency_based"
        elif sla_throughput is not None:
            config = recommender.recommend_for_throughput(sla_throughput, batch_size)
            recommendation_type = "throughput_based"
        else:
            config = recommender.recommend_cost_effective(batch_size)
            recommendation_type = "cost_effective"
        
        if config is None:
            return jsonify({"error": "No suitable configuration found"}), 404
        
        return jsonify({
            "recommendation_type": recommendation_type,
            "cube": config.cube,
            "vector": config.vector,
            "memory": config.memory,
            "expected_latency": config.expected_latency,
            "expected_throughput": config.expected_throughput,
            "confidence": config.confidence
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to generate recommendation: {str(e)}"}), 500


@app.route("/scaling_advice", methods=["POST"])
def scaling_advice():
    """
    获取扩容/缩容建议
    
    请求体示例：
    {
        "current_cube": 8,
        "current_vector": 20,
        "current_latency": 0.5,
        "target_latency": 0.2,
        "batch_size": 1
    }
    
    返回：扩容/缩容建议
    """
    if not _recommender_available:
        return jsonify({"error": "Resource recommender not available"}), 503
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    current_cube = data.get("current_cube")
    current_vector = data.get("current_vector")
    current_latency = data.get("current_latency")
    target_latency = data.get("target_latency")
    batch_size = data.get("batch_size", 1)
    
    if None in [current_cube, current_vector, current_latency, target_latency]:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        recommender = get_recommender()
        advice = recommender.get_scaling_advice(
            current_cube=current_cube,
            current_vector=current_vector,
            current_latency=current_latency,
            target_latency=target_latency,
            batch_size=batch_size
        )
        
        # 转换ResourceConfig为可序列化的字典
        if "recommendation" in advice and advice["recommendation"]:
            rec = advice["recommendation"]
            advice["recommendation"] = {
                "cube": rec.cube,
                "vector": rec.vector,
                "memory": rec.memory,
                "expected_latency": rec.expected_latency,
                "expected_throughput": rec.expected_throughput,
                "confidence": rec.confidence
            }
        
        return jsonify(advice), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to generate scaling advice: {str(e)}"}), 500


# 模块级 PortManager，便于被其他脚本导入时直接使用
port_manager = PortManager()

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
