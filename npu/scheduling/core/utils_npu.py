# -*- coding: utf-8 -*-
"""
NPU 实例生命周期：本机进程 + ACL 核心/显存设限，替代 Docker 启动。
单容器内 4 张 NPU，通过子进程 + 环境变量传入设备与配额，子进程内先 ACL 设限再执行业务。
"""
import os
import sys
import subprocess
import shlex
import time

# 默认 worker 入口：当前包下的 demo worker（可被覆盖）
def _worker_script():
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "scripts_demo", "npu_worker_entry.py")

# 当指定 MODEL_PATH 且未指定 COMMAND 时，直接跑 LLM 推理服务
def _llm_worker_script():
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "scripts_demo", "llm_inference_npu.py")


def start_instance(selected_npus, instance_id, image_name, service_name, args, allocated_port, ip_address):
    """
    在本机启动一个 NPU 任务子进程。
    - selected_npus: [{"id", "ip", "index"}, ...]，单卡推理仅一个元素。
    - args: 含 cube_requests, cube_limits, vector_requests, vector_limits, memory, COMMAND 等。
    子进程通过环境变量接收：NPU_DEVICE_ID, CUBE_LIMIT, VECTOR_LIMIT, MEMORY_GB, PORT, INSTANCE_ID, SERVICE_NAME。
    """
    if not selected_npus:
        raise ValueError("selected_npus is empty")
    first = selected_npus[0]
    device_id = first["index"]
    # 单卡推理：配额来自 args（调度器已转为核心数：cube_limits/vector_limits 为整数）
    cube_lim = int(args.get("cube_limits", 15))
    vector_lim = int(args.get("vector_limits", 30))
    memory_gb = args.get("memory", [8])
    if isinstance(memory_gb, list):
        memory_gb = memory_gb[0] if memory_gb else 8
    memory_gb = int(memory_gb)

    env = os.environ.copy()
    env["NPU_DEVICE_ID"] = str(device_id)
    env["CUBE_LIMIT"] = str(cube_lim)
    env["VECTOR_LIMIT"] = str(vector_lim)
    env["MEMORY_GB"] = str(memory_gb)
    env["PORT"] = str(allocated_port)
    env["INSTANCE_ID"] = str(instance_id)
    env["SERVICE_NAME"] = str(service_name)
    env["NPU_CUBE_REQUESTS"] = str(args.get("cube_requests", cube_lim))
    env["NPU_VECTOR_REQUESTS"] = str(args.get("vector_requests", vector_lim))
    if args.get("COMMAND"):
        env["COMMAND"] = str(args.get("COMMAND"))
    if args.get("MODEL_PATH"):
        env["MODEL_PATH"] = str(args.get("MODEL_PATH"))

    if args.get("MODEL_PATH") and not args.get("COMMAND") and not args.get("WORKER_SCRIPT"):
        worker = _llm_worker_script()
        print(f"[NPU] Using LLM worker (MODEL_PATH={args.get('MODEL_PATH')})")
    else:
        worker = args.get("WORKER_SCRIPT") or _worker_script()
        print(f"[NPU] Using entry worker: {worker}")
    
    log_dir = args.get("LOG_DIR") or os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{service_name}-{instance_id}.log")
    
    if not os.path.isfile(worker):
        # 若无 demo worker，用 python -c 起一个最小 HTTP 服务（仅做端口占用与 /health /predict）
        print(f"[NPU] Warning: Worker script not found: {worker}, using minimal stub")
        cmd = [
            sys.executable, "-c",
            _minimal_worker_code(allocated_port, device_id, cube_lim, vector_lim),
        ]
    else:
        cmd = [sys.executable, worker]
    
    # 使用项目根目录作为工作目录，确保模块导入正常
    project_root = os.path.join(os.path.dirname(__file__), "..")
    project_root = os.path.abspath(project_root)
    
    # 将项目根目录和scheduling目录添加到PYTHONPATH
    python_path = env.get("PYTHONPATH", "")
    scheduling_dir = os.path.dirname(__file__)
    new_python_path = f"{project_root}:{scheduling_dir}"
    if python_path:
        new_python_path = f"{new_python_path}:{python_path}"
    env["PYTHONPATH"] = new_python_path
    
    print(f"[NPU] Starting worker with cwd={project_root}, PYTHONPATH={new_python_path}")
    
    # 先写入启动信息到日志
    with open(log_file, "w") as f:
        f.write(f"[NPU] Starting instance {instance_id}\n")
        f.write(f"[NPU] Command: {' '.join(cmd)}\n")
        f.write(f"[NPU] Working directory: {project_root}\n")
        f.write(f"[NPU] Environment: DEVICE_ID={device_id}, CUBE_LIMIT={cube_lim}, VECTOR_LIMIT={vector_lim}\n")
        f.flush()
        
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=project_root,
        )
    
    # 等待一小段时间检查进程是否还在运行
    time.sleep(2)
    if proc.poll() is not None:
        # 进程已经退出
        print(f"[NPU] ERROR: Worker process exited immediately with code {proc.returncode}")
        print(f"[NPU] Check log file for details: {log_file}")
        # 读取日志内容
        try:
            with open(log_file, "r") as f:
                log_content = f.read()
                if log_content:
                    print(f"[NPU] Log content:\n{log_content}")
        except Exception as e:
            print(f"[NPU] Could not read log file: {e}")
        return
    
    # 不 wait：让任务常驻；调用方通过 /delete_instance 时再 kill
    print(f"[NPU] Started instance {instance_id} on device {device_id}, port {allocated_port}, pid={proc.pid}, log={log_file}")
    # 将 pid 写入约定文件，便于 stop 时按 instance 杀进程
    pid_file = os.path.join(log_dir, f"{service_name}-{instance_id}.pid")
    with open(pid_file, "w") as f:
        f.write(str(proc.pid))


def _minimal_worker_code(port, device_id, cube_lim, vector_lim):
    """无 npu_worker_entry.py 时的最小 HTTP 占位，不依赖 ACL。"""
    return r"""
import os, sys
port = int(os.environ.get('PORT', %d))
device_id = int(os.environ.get('NPU_DEVICE_ID', %d))
try:
    from http.server import HTTPServer, BaseHTTPRequestHandler
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/health':
                self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
            else: self.send_response(404); self.end_headers()
        def do_POST(self):
            if self.path == '/predict':
                self.send_response(200); self.send_header('Content-type','application/json'); self.end_headers()
                self.wfile.write(b'{"status":"ok","npu_device":%d}')
            else: self.send_response(404); self.end_headers()
        def log_message(self, *a): pass
    HTTPServer(('0.0.0.0', port), H).serve_forever()
except Exception as e:
    sys.stderr.write(str(e)); sys.exit(1)
""" % (port, device_id, device_id)


def stop_instance(service_name, instance_id, ip_address):
    """根据 pid 文件结束子进程；若无 pid 文件则尝试按名称 kill（兼容）。"""
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    pid_file = os.path.join(log_dir, f"{service_name}-{instance_id}.pid")
    if os.path.isfile(pid_file):
        try:
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())
            os.kill(pid, 9)
            print(f"[NPU] Stopped instance {instance_id} (pid={pid})")
        except (ValueError, ProcessLookupError, OSError) as e:
            print(f"[NPU] stop_instance kill pid failed: {e}")
        try:
            os.remove(pid_file)
        except OSError:
            pass
    else:
        print(f"[NPU] stop_instance: no pid file {pid_file}, skip kill")
