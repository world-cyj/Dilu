# -*- coding: utf-8 -*-
"""
NPU 版 Scaler：服务注册与扩缩容，请求字段为 cube/vector/memory，对接 NPU 调度器。
"""
import threading
import time
import requests
import json
from flask import Flask, request, jsonify
from datetime import datetime
import logging
import os

log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "dilu-scaler-npu.log"),
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.logger.disabled = True
services_status = {}
services_dict = {}
lock = threading.Lock()
scheduler_url = os.environ.get("SCHEDULER_URL", "http://localhost:5000")


class ServiceManager:
    def __init__(self):
        self.services = {}

    def register_service(self, service_data):
        global services_dict, services_status
        service_id = service_data["service_name"]
        if service_id in services_dict:
            return "Service already registered", 409
        new_service = Service(
            service_id,
            service_data.get("image", ""),
            service_data.get("num_npus", service_data.get("num_gpus", 1)),
            service_data.get("cube_requests", 0.25),
            service_data.get("cube_limits", 0.75),
            service_data.get("vector_requests", 0.25),
            service_data.get("vector_limits", 0.75),
            service_data.get("memory", [8]),
            service_data.get("is_llm", False),
            service_data.get("priority", "low"),
            service_data.get("task_type", "inference"),
            service_data.get("throughput", 10),
            service_data.get("commands", ""),
        )
        services_dict[service_id] = new_service
        services_status[service_id] = {"requests": 0}
        return f"Service {service_id} registered successfully", 200

    def delete_service(self, service_data):
        global services_dict, services_status
        service_id = service_data["service_name"]
        if service_id in services_dict:
            del services_dict[service_id]
            del services_status[service_id]
            return f"Service {service_id} deleted successfully", 200
        return "Service does not exist", 404


@app.route("/register_service", methods=["POST"])
def register_service():
    service_data = request.get_json()
    message, status = service_manager.register_service(service_data)
    return jsonify({"message": message}), status


@app.route("/<service_id>", methods=["POST"])
def handle_predict(service_id):
    if service_id in services_dict:
        services_status[service_id]["requests"] += 1
        service = services_dict[service_id]
        return service.dispatch(request)
    return jsonify({"error": "Service not found"}), 404


class Service:
    def __init__(
        self,
        service_id,
        image,
        num_npus,
        cube_requests,
        cube_limits,
        vector_requests,
        vector_limits,
        memory,
        is_llm,
        priority,
        task_type,
        throughput,
        commands,
    ):
        self.service_id = service_id
        self.image = image
        self.num_npus = num_npus
        self.cube_requests = cube_requests
        self.cube_limits = cube_limits
        self.vector_requests = vector_requests
        self.vector_limits = vector_limits
        self.memory = memory if isinstance(memory, list) else [memory]
        self.is_llm = is_llm
        self.priority = priority
        self.task_type = task_type
        self.commands = commands
        self.throughput = throughput
        self.instances = []
        self.round_robin_index = 0
        self.lock = threading.Lock()
        self.scale_out()

    def scale_out(self):
        url = f"{scheduler_url}/schedule"
        data = {
            "num": self.num_npus,
            "cube_requests": self.cube_requests,
            "cube_limits": self.cube_limits,
            "vector_requests": self.vector_requests,
            "vector_limits": self.vector_limits,
            "memory": self.memory,
            "is_llm": self.is_llm,
            "priority": self.priority,
            "type": self.task_type,
            "service_name": self.service_id,
            "image": self.image,
            "COMMAND": self.commands,
        }
        try:
            response = requests.post(url, json=data, timeout=30)
            if response.status_code == 200:
                resp_content = response.json()
                instance_id = resp_content.get("instance_id")
                selected = resp_content.get("selected_npus", resp_content.get("selected_gpus", []))
                ip_address = selected[0]["ip"] if selected else "127.0.0.1"
                port = resp_content.get("port")
                instance = {"instance_id": instance_id, "ip_address": ip_address, "port": port, "is_ready": False}
                self.instances.append(instance)
                print(f"[NPU] Instance deployed: {instance_id}, port={port}, instances={len(self.instances)}")
                if "training" not in self.service_id:
                    threading.Thread(target=self.check_instance_readiness, args=(instance,), daemon=True).start()
            else:
                print(f"[NPU] Instance deploy failed: {response.status_code} {response.content}")
        except requests.RequestException as e:
            print(f"[NPU] scale_out request error: {e}")

    def check_instance_readiness(self, instance):
        while True:
            try:
                r = requests.get(f"http://{instance['ip_address']}:{instance['port']}/health", timeout=2)
                if r.status_code == 200:
                    with self.lock:
                        instance["is_ready"] = True
                    break
            except requests.RequestException:
                pass
            time.sleep(1)

    def scale_in(self):
        with self.lock:
            if not self.instances:
                return
            instance_to_remove = self.instances.pop()
        url = f"{scheduler_url}/delete_instance"
        try:
            response = requests.post(url, json={"instance_id": instance_to_remove["instance_id"]}, timeout=30)
            if response.status_code == 200:
                print(f"[NPU] Instance deleted: {instance_to_remove['instance_id']}, remaining={len(self.instances)}")
            else:
                print(f"[NPU] Instance delete failed: {response.content}")
        except requests.RequestException as e:
            print(f"[NPU] scale_in request error: {e}")

    def scale_vertical_up(self):
        """IE 垂直扩：单实例时提高 request/limit，通过 reschedule 替换实例。"""
        with self.lock:
            if len(self.instances) != 1:
                return False
            inst = self.instances[0]
        new_cube = min(round(self.cube_limits * 1.2, 2), 1.0)
        new_vector = min(round(self.vector_limits * 1.2, 2), 1.0)
        url = f"{scheduler_url}/reschedule_with_limits"
        data = {
            "instance_id": inst["instance_id"],
            "new_cube_limits": new_cube,
            "new_vector_limits": new_vector,
            "service_name": self.service_id,
            "type": self.task_type,
            "memory": self.memory,
            "num": self.num_npus,
            "image": self.image,
            "COMMAND": self.commands,
        }
        try:
            r = requests.post(url, json=data, timeout=30)
            if r.status_code == 200:
                body = r.json()
                with self.lock:
                    self.instances.clear()
                    self.instances.append({
                        "instance_id": body["instance_id"],
                        "ip_address": body["selected_npus"][0]["ip"] if body.get("selected_npus") else "127.0.0.1",
                        "port": body["port"],
                        "is_ready": False,
                    })
                self.cube_limits = new_cube
                self.vector_limits = new_vector
                if "training" not in self.service_id:
                    threading.Thread(target=self.check_instance_readiness, args=(self.instances[0],), daemon=True).start()
                print(f"[NPU] scale_vertical_up: new limits cube={new_cube} vector={new_vector}")
                return True
        except requests.RequestException as e:
            print(f"[NPU] scale_vertical_up error: {e}")
        return False

    def scale_vertical_down(self):
        """IE 垂直缩：单实例时降低 request/limit。"""
        with self.lock:
            if len(self.instances) != 1:
                return False
            inst = self.instances[0]
        new_cube = max(round(self.cube_limits * 0.8, 2), 0.25)
        new_vector = max(round(self.vector_limits * 0.8, 2), 0.25)
        url = f"{scheduler_url}/reschedule_with_limits"
        data = {
            "instance_id": inst["instance_id"],
            "new_cube_limits": new_cube,
            "new_vector_limits": new_vector,
            "service_name": self.service_id,
            "type": self.task_type,
            "memory": self.memory,
            "num": self.num_npus,
            "image": self.image,
            "COMMAND": self.commands,
        }
        try:
            r = requests.post(url, json=data, timeout=30)
            if r.status_code == 200:
                body = r.json()
                with self.lock:
                    self.instances.clear()
                    self.instances.append({
                        "instance_id": body["instance_id"],
                        "ip_address": body["selected_npus"][0]["ip"] if body.get("selected_npus") else "127.0.0.1",
                        "port": body["port"],
                        "is_ready": False,
                    })
                self.cube_limits = new_cube
                self.vector_limits = new_vector
                if "training" not in self.service_id:
                    threading.Thread(target=self.check_instance_readiness, args=(self.instances[0],), daemon=True).start()
                print(f"[NPU] scale_vertical_down: new limits cube={new_cube} vector={new_vector}")
                return True
        except requests.RequestException as e:
            print(f"[NPU] scale_vertical_down error: {e}")
        return False

    def dispatch(self, request):
        post_data = request.get_json()
        if not post_data:
            return jsonify({"error": "No data provided"}), 400
        with self.lock:
            ready = [i for i in self.instances if i.get("is_ready")]
        if not ready:
            return jsonify({"error": "No ready instances"}), 503
        self.round_robin_index = (self.round_robin_index + 1) % len(ready)
        inst = ready[self.round_robin_index]
        url = f"http://{inst['ip_address']}:{inst['port']}/predict"
        try:
            r = requests.post(url, json=post_data, timeout=60)
            if r.status_code == 200:
                return jsonify(r.json()), 200
            return jsonify({"error": "Prediction failed", "details": r.text}), r.status_code
        except requests.RequestException as e:
            return jsonify({"error": "Network error", "details": str(e)}), 500


class Scaler(threading.Thread):
    """2D co-scaling：先尝试垂直伸缩（单实例时改 request/limit），再水平伸缩（增删实例）。"""
    def __init__(self, check_interval=1, scale_out_threshold=15, scale_in_threshold=15, history_length=20,
                 vertical_up_threshold=12, vertical_down_threshold=25):
        super().__init__()
        self.check_interval = check_interval
        self.scale_out_threshold = scale_out_threshold
        self.scale_in_threshold = scale_in_threshold
        self.history_length = history_length
        self.vertical_up_threshold = vertical_up_threshold
        self.vertical_down_threshold = vertical_down_threshold
        self.request_history = {}
        self.daemon = True

    def run(self):
        global services_status, services_dict
        while True:
            time.sleep(self.check_interval)
            for service_id, status in list(services_status.items()):
                if "training" in service_id:
                    continue
                if service_id not in services_dict:
                    continue
                svc = services_dict[service_id]
                n_inst = len(svc.instances)
                max_throughput = n_inst * svc.throughput
                scale_in_throughput = (n_inst - 1) * svc.throughput if n_inst > 1 else 0
                if service_id not in self.request_history:
                    self.request_history[service_id] = []
                if len(self.request_history[service_id]) >= self.history_length:
                    self.request_history[service_id].pop(0)
                self.request_history[service_id].append(status["requests"])
                status["requests"] = 0
                # 扩容：先垂直（单实例时提 limit），再水平
                if self.should_scale_out(service_id, max_throughput):
                    if n_inst == 1 and self._over_vertical_up(service_id):
                        svc.scale_vertical_up()
                    else:
                        svc.scale_out()
                    self.request_history[service_id].clear()
                # 缩容：先水平（多实例时减实例），再垂直（单实例时降 limit）
                if self.should_scale_in(service_id, scale_in_throughput):
                    svc.scale_in()
                    self.request_history[service_id].clear()
                elif n_inst == 1 and self._under_vertical_down(service_id):
                    svc.scale_vertical_down()
                    self.request_history[service_id].clear()

    def _over_vertical_up(self, service_id):
        hist = self.request_history.get(service_id, [])
        return sum(1 for x in hist if x > 0) >= self.vertical_up_threshold

    def _under_vertical_down(self, service_id):
        hist = self.request_history.get(service_id, [])
        return sum(1 for x in hist if x == 0) >= self.vertical_down_threshold

    def should_scale_out(self, service_id, max_throughput):
        hist = self.request_history.get(service_id, [])
        over = sum(1 for x in hist if x > max_throughput)
        return over >= self.scale_out_threshold

    def should_scale_in(self, service_id, scale_in_throughput):
        if len(services_dict[service_id].instances) <= 1:
            return False
        hist = self.request_history.get(service_id, [])
        under = sum(1 for x in hist if x < scale_in_throughput)
        return under >= self.scale_in_threshold


if __name__ == "__main__":
    service_manager = ServiceManager()
    scaler = Scaler(check_interval=1, scale_out_threshold=20, scale_in_threshold=30, history_length=40)
    scaler.start()
    app.run(debug=False, host="0.0.0.0", port=14999, threaded=True)
