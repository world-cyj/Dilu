#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
展示Qwen3-4B推理结果
"""
import os
import sys
import time
import threading
import requests

# 确保 scheduling 在 path 中
sched_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sched_dir not in sys.path:
    sys.path.insert(0, sched_dir)

SCHED_PORT = int(os.environ.get("SCHED_PORT", "5000"))
BASE = f"http://127.0.0.1:{SCHED_PORT}"

# 测试配置
TEST_CONFIG = {
    "model_path": os.environ.get("MODEL_PATH", "/vllm-workspace/models/Qwen3-4B"),
    "service_name": "qwen3-4b-demo",
    "test_prompts": [
        "你好，请用一句话介绍你自己。",
        "什么是人工智能？",
        "请解释深度学习的基本原理。",
    ]
}


def run_scheduler():
    """启动调度器"""
    from scheduler_npu import app
    app.run(debug=False, host="0.0.0.0", port=SCHED_PORT, use_reloader=False)


def main():
    print("=" * 60)
    print("Qwen3-4B 推理结果展示")
    print("=" * 60)
    
    # 启动调度器
    print("\n[1] 启动调度器...")
    t = threading.Thread(target=run_scheduler, daemon=True)
    t.start()
    time.sleep(3)
    
    # 部署服务
    print("\n[2] 部署Qwen3-4B服务...")
    payload = {
        "service_name": TEST_CONFIG["service_name"],
        "model_path": TEST_CONFIG["model_path"],
        "sla_latency": 1.0,
        "batch_size": 1,
        "type": "inference"
    }
    
    r = requests.post(f"{BASE}/schedule_with_sla", json=payload, timeout=30)
    if r.status_code != 200:
        print(f"部署失败: {r.status_code} - {r.text}")
        return 1
    
    data = r.json()
    instance_id = data.get("instance_id")
    port = data.get("port")
    print(f"服务部署成功: {instance_id}, 端口: {port}")
    
    # 等待服务就绪
    print("\n[3] 等待服务就绪...")
    start_time = time.time()
    while time.time() - start_time < 120:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
            if r.status_code == 200:
                health_data = r.json()
                if health_data.get("model_loaded"):
                    print("服务就绪！")
                    print(f"  设备: {health_data.get('device', 'N/A')}")
                    print(f"  显存: {health_data.get('memory_gb', 'N/A')} GB")
                    break
        except:
            pass
        time.sleep(2)
    else:
        print("服务启动超时")
        return 1
    
    # 发送推理请求并显示结果
    print("\n" + "=" * 60)
    print("推理结果")
    print("=" * 60)
    
    for i, prompt in enumerate(TEST_CONFIG["test_prompts"], 1):
        print(f"\n{'='*60}")
        print(f"[问题 {i}] {prompt}")
        print('='*60)
        
        try:
            start = time.time()
            r = requests.post(
                f"http://127.0.0.1:{port}/predict",
                json={"text": prompt},
                timeout=60
            )
            elapsed = time.time() - start
            
            if r.status_code == 200:
                data = r.json()
                answer = data.get("prediction", "")
                latency = data.get("latency", elapsed)
                
                print(f"\n[回答] {answer}")
                print(f"\n[延迟] {latency:.3f}s")
            else:
                print(f"请求失败: {r.status_code}")
        except Exception as e:
            print(f"请求异常: {e}")
    
    # 清理
    print("\n" + "=" * 60)
    print("清理资源...")
    requests.post(f"{BASE}/delete_instance", json={"instance_id": instance_id}, timeout=10)
    print("完成！")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
