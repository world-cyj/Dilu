#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接测试 Qwen3-4B 推理服务
"""
import os
import sys
import time
import threading
import subprocess
import requests

# 设置环境变量
os.environ["MODEL_PATH"] = "/vllm-workspace/models/Qwen3-4B"
os.environ["NPU_DEVICE_ID"] = "0"
os.environ["PORT"] = "15002"

# 启动推理服务
print("启动推理服务...")
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
    r = requests.get("http://127.0.0.1:15002/health", timeout=5)
    print(f"健康检查: {r.status_code}")
    print(f"健康数据: {r.json()}")
    
    # 发送推理请求
    print("\n发送推理请求...")
    r = requests.post(
        "http://127.0.0.1:15002/predict",
        json={"text": "你好"},
        timeout=60
    )
    print(f"推理状态: {r.status_code}")
    data = r.json()
    print(f"推理结果: {data.get('prediction', '')[:500]}")
    print(f"延迟: {data.get('latency', 0):.3f}s")
    
finally:
    # 停止服务
    proc.terminate()
    proc.wait()
    print("\n服务已停止")
