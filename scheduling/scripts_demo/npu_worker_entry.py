# -*- coding: utf-8 -*-
"""
NPU 任务子进程入口：读取环境变量后设置 ACL 核心/显存限制，再启动 HTTP 服务（/health、/predict）。
调度器通过 utils_npu 启动本脚本，传入 NPU_DEVICE_ID、CUBE_LIMIT、VECTOR_LIMIT、PORT 等。
"""
import os
import sys

# 在 import 业务库之前先设置 ACL 限制
def apply_acl_limits():
    device_id = int(os.environ.get("NPU_DEVICE_ID", "0"))
    cube_limit = int(os.environ.get("CUBE_LIMIT", "20"))
    vector_limit = int(os.environ.get("VECTOR_LIMIT", "40"))
    try:
        # 将 scheduling/npu 加入路径（若从 scheduling/ 运行则已包含）
        parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from npu.acl_rt_wrapper import (
            init_device,
            init_context,
            set_device_res_limit,
            ACL_RT_DEV_RES_CUBE_CORE,
            ACL_RT_DEV_RES_VECTOR_CORE,
        )
        ok, err = init_device(device_id)
        if not ok:
            print("[npu_worker] init_device failed: {}".format(err), file=sys.stderr)
            return
        ok, err = set_device_res_limit(device_id, ACL_RT_DEV_RES_CUBE_CORE, cube_limit)
        if not ok:
            print("[npu_worker] set_device_res_limit CUBE failed: {}".format(err), file=sys.stderr)
        ok, err = set_device_res_limit(device_id, ACL_RT_DEV_RES_VECTOR_CORE, vector_limit)
        if not ok:
            print("[npu_worker] set_device_res_limit VECTOR failed: {}".format(err), file=sys.stderr)
        ok, err = init_context(device_id)
        if not ok:
            print("[npu_worker] init_context failed: {}".format(err), file=sys.stderr)
    except ImportError as e:
        print(f"[npu_worker] ACL not available (no NPU env?): {e}", file=sys.stderr)


apply_acl_limits()

# 可选：执行自定义命令（例如模型推理服务）
command = os.environ.get("COMMAND", "").strip()
if command:
    try:
        import subprocess
        subprocess.Popen(command, shell=True)
        print(f"[npu_worker] COMMAND started: {command}", file=sys.stderr)
    except Exception as e:
        print(f"[npu_worker] COMMAND failed: {e}", file=sys.stderr)

# 启动最小 HTTP 服务
port = int(os.environ.get("PORT", "15000"))
device_id = int(os.environ.get("NPU_DEVICE_ID", "0"))
instance_id = os.environ.get("INSTANCE_ID", "")
service_name = os.environ.get("SERVICE_NAME", "")


if __name__ == "__main__":
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class NpuHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/predict":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                body = '{"status":"ok","npu_device":%d,"instance_id":"%s","service":"%s"}' % (
                    device_id, instance_id, service_name
                )
                self.wfile.write(body.encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("0.0.0.0", port), NpuHandler)
    print(f"[npu_worker] NPU device={device_id} port={port} instance={instance_id} service={service_name}", file=sys.stderr)
    server.serve_forever()
