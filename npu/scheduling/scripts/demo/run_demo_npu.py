# -*- coding: utf-8 -*-
"""
NPU Demo 纯 Python 版：启动调度器线程 → /schedule → 等 /health → /predict → /delete_instance。
"""
import os
import sys
import time
import threading
import subprocess
import requests

# 确保 scheduling 在 path 中
sched_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sched_dir not in sys.path:
    sys.path.insert(0, sched_dir)

SCHED_PORT = int(os.environ.get("SCHED_PORT", "5000"))
DEFAULT_WORKER_PORT = 15000
BASE = f"http://127.0.0.1:{SCHED_PORT}"


def _free_port_if_in_use(port):
    """若端口被占用则尝试释放，避免旧 worker 导致新实例绑定失败。"""
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))
    except OSError:
        try:
            subprocess.run(["fuser", "-k", "%s/tcp" % port], capture_output=True, timeout=5)
            time.sleep(1)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass


def run_scheduler():
    from scheduler_npu import app, port_manager
    app.run(debug=False, host="0.0.0.0", port=SCHED_PORT, use_reloader=False)


def main():
    print("=== 1. 启动 NPU 调度器 (port %s) ===" % SCHED_PORT)
    _free_port_if_in_use(DEFAULT_WORKER_PORT)
    t = threading.Thread(target=run_scheduler, daemon=True)
    t.start()
    time.sleep(2)

    model_path = os.environ.get("MODEL_PATH", "/vllm-workspace/models/Qwen3-4B")
    print("=== 2. 提交推理实例（cube/vector 0.25~0.75, 8GB, MODEL_PATH=%s）===" % model_path)
    r = requests.post(
        f"{BASE}/schedule",
        json={
            "num": 1,
            "cube_requests": 0.25,
            "cube_limits": 0.75,
            "vector_requests": 0.25,
            "vector_limits": 0.75,
            "memory": [8],
            "type": "inference",
            "service_name": "demo-npu-inference",
            "image": "",
            "COMMAND": "",
            "MODEL_PATH": model_path,
        },
        timeout=10,
    )
    if r.status_code != 200:
        print("Schedule failed:", r.status_code, r.text)
        return 1
    data = r.json()
    instance_id = data.get("instance_id")
    port = data.get("port")
    selected = data.get("selected_npus", [])
    print("  instance_id:", instance_id, "port:", port, "selected_npus:", selected)

    print("=== 3. 等待 worker /health（含模型加载，最多约 120s）===")
    for i in range(60):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
            if r.status_code == 200:
                j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                print("  Worker ready. model_loaded=%s" % j.get("model_loaded", "?"))
                break
        except requests.RequestException:
            pass
        if (i + 1) % 10 == 0:
            print("  ... still waiting (%ds)" % (i + 1))
        time.sleep(2)
    else:
        print("  Worker not ready in time.")
        return 1

    print("=== 4. 调用 /predict（Qwen3-4B 推理）===")
    r = requests.post(f"http://127.0.0.1:{port}/predict", json={"input": "你好，请用一句话介绍你自己。"}, timeout=120)
    print("  Response:", r.status_code, r.json() if r.ok else r.text)

    print("=== 5. 删除实例 ===")
    r = requests.post(f"{BASE}/delete_instance", json={"instance_id": instance_id}, timeout=5)
    print("  Delete:", r.status_code, r.json() if r.ok else r.text)

    print("Demo done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
