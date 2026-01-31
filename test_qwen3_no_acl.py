#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Qwen3-4B 推理服务 - 不使用 ACL 限制
"""
import os
import sys
import time
import subprocess
import requests

# 设置环境变量 - 不设置 CUBE_LIMIT 和 VECTOR_LIMIT，这样不会应用 ACL 限制
os.environ["MODEL_PATH"] = "/vllm-workspace/models/Qwen3-4B"
os.environ["NPU_DEVICE_ID"] = "0"
os.environ["PORT"] = "15003"
# 不设置 CUBE_LIMIT 和 VECTOR_LIMIT

# 启动推理服务
print("启动推理服务（无ACL限制）...")
proc = subprocess.Popen(
    ["python3", "/vllm-workspace/Dilu/scheduling/scripts_demo/llm_inference_npu.py"],
    env=os.environ.copy(),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# 等待服务就绪
time.sleep(30)

try:
    # 检查健康状态
    r = requests.get("http://127.0.0.1:15003/health", timeout=5)
    print(f"健康检查: {r.status_code}")
    print(f"健康数据: {r.json()}")
    
    # 发送推理请求
    print("\n发送推理请求...")
    r = requests.post(
        "http://127.0.0.1:15003/predict",
        json={"text": "你好，请用一句话介绍你自己。"},
        timeout=60
    )
    print(f"推理状态: {r.status_code}")
    data = r.json()
    print(f"\n推理结果:\n{data.get('prediction', '')}")
    print(f"\n延迟: {data.get('latency', 0):.3f}s")
    
finally:
    # 停止服务
    proc.terminate()
    proc.wait()
    print("\n服务已停止")
