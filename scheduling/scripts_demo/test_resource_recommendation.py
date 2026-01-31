# -*- coding: utf-8 -*-
"""
测试资源推荐引擎API的脚本
验证 /recommend_resources、/schedule_with_sla、/scaling_advice 端点
"""
import os
import sys
import time
import threading
import requests
import json

# 确保 scheduling 在 path 中
sched_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sched_dir not in sys.path:
    sys.path.insert(0, sched_dir)

SCHED_PORT = int(os.environ.get("SCHED_PORT", "5000"))
BASE = f"http://127.0.0.1:{SCHED_PORT}"


def run_scheduler():
    """在后台线程中启动调度器"""
    from scheduler_npu import app
    app.run(debug=False, host="0.0.0.0", port=SCHED_PORT, use_reloader=False)


def test_recommend_resources():
    """测试资源推荐API"""
    print("\n=== Test 1: /recommend_resources ===")
    
    # 测试基于延迟的推荐
    print("\n1.1 基于延迟推荐 (sla_latency=0.2s)")
    r = requests.post(f"{BASE}/recommend_resources", json={
        "model_path": "/vllm-workspace/models/Qwen3-4B",
        "sla_latency": 0.2,
        "batch_size": 1
    })
    if r.status_code == 200:
        data = r.json()
        print(f"  ✓ Recommendation type: {data['recommendation_type']}")
        print(f"  ✓ Cube: {data['cube']}, Vector: {data['vector']}")
        print(f"  ✓ Expected latency: {data['expected_latency']}s")
        print(f"  ✓ Expected throughput: {data['expected_throughput']}")
        print(f"  ✓ Confidence: {data['confidence']}")
    else:
        print(f"  ✗ Failed: {r.status_code} - {r.text}")
    
    # 测试基于吞吐的推荐
    print("\n1.2 基于吞吐推荐 (sla_throughput=5.0)")
    r = requests.post(f"{BASE}/recommend_resources", json={
        "model_path": "/vllm-workspace/models/Qwen3-4B",
        "sla_throughput": 5.0,
        "batch_size": 1
    })
    if r.status_code == 200:
        data = r.json()
        print(f"  ✓ Recommendation type: {data['recommendation_type']}")
        print(f"  ✓ Cube: {data['cube']}, Vector: {data['vector']}")
        print(f"  ✓ Expected latency: {data['expected_latency']}s")
        print(f"  ✓ Expected throughput: {data['expected_throughput']}")
    else:
        print(f"  ✗ Failed: {r.status_code} - {r.text}")
    
    # 测试性价比推荐
    print("\n1.3 性价比推荐 (无SLA)")
    r = requests.post(f"{BASE}/recommend_resources", json={
        "model_path": "/vllm-workspace/models/Qwen3-4B",
        "batch_size": 1
    })
    if r.status_code == 200:
        data = r.json()
        print(f"  ✓ Recommendation type: {data['recommendation_type']}")
        print(f"  ✓ Cube: {data['cube']}, Vector: {data['vector']}")
        print(f"  ✓ Expected latency: {data['expected_latency']}s")
        print(f"  ✓ Expected throughput: {data['expected_throughput']}")
    else:
        print(f"  ✗ Failed: {r.status_code} - {r.text}")


def test_scaling_advice():
    """测试扩容/缩容建议API"""
    print("\n=== Test 2: /scaling_advice ===")
    
    # 测试扩容建议
    print("\n2.1 扩容建议 (当前延迟0.5s > 目标0.2s)")
    r = requests.post(f"{BASE}/scaling_advice", json={
        "current_cube": 8,
        "current_vector": 20,
        "current_latency": 0.5,
        "target_latency": 0.2,
        "batch_size": 1
    })
    if r.status_code == 200:
        data = r.json()
        print(f"  ✓ Action: {data['action']}")
        print(f"  ✓ Reason: {data['reason']}")
        if 'recommendation' in data:
            rec = data['recommendation']
            print(f"  ✓ Recommended: Cube={rec['cube']}, Vector={rec['vector']}")
            print(f"  ✓ Expected: latency={rec['expected_latency']}s, throughput={rec['expected_throughput']}")
    else:
        print(f"  ✗ Failed: {r.status_code} - {r.text}")
    
    # 测试缩容建议
    print("\n2.2 缩容建议 (当前延迟0.1s < 目标0.2s)")
    r = requests.post(f"{BASE}/scaling_advice", json={
        "current_cube": 16,
        "current_vector": 40,
        "current_latency": 0.1,
        "target_latency": 0.2,
        "batch_size": 1
    })
    if r.status_code == 200:
        data = r.json()
        print(f"  ✓ Action: {data['action']}")
        print(f"  ✓ Reason: {data['reason']}")
        if 'recommendation' in data:
            rec = data['recommendation']
            print(f"  ✓ Recommended: Cube={rec['cube']}, Vector={rec['vector']}")
    else:
        print(f"  ✗ Failed: {r.status_code} - {r.text}")


def test_schedule_with_sla():
    """测试基于SLA的智能调度API"""
    print("\n=== Test 3: /schedule_with_sla ===")
    
    print("\n3.1 基于SLA调度 (sla_latency=0.2s)")
    r = requests.post(f"{BASE}/schedule_with_sla", json={
        "service_name": "test-sla-service",
        "model_path": "/vllm-workspace/models/Qwen3-4B",
        "sla_latency": 0.2,
        "batch_size": 1,
        "type": "inference"
    })
    if r.status_code == 200:
        data = r.json()
        print(f"  ✓ Instance ID: {data['instance_id']}")
        print(f"  ✓ Port: {data['port']}")
        print(f"  ✓ Selected NPUs: {data['selected_npus']}")
        print(f"  ✓ Recommendation:")
        rec = data['recommendation']
        print(f"    - Cube: requests={rec['cube_requests']}, limits={rec['cube_limits']}")
        print(f"    - Vector: requests={rec['vector_requests']}, limits={rec['vector_limits']}")
        print(f"    - Memory: {rec['memory']} GB")
        print(f"    - Expected latency: {rec['expected_latency']}s")
        print(f"    - Expected throughput: {rec['expected_throughput']}")
        print(f"    - Confidence: {rec['confidence']}")
        
        # 保存instance_id用于后续删除
        return data['instance_id']
    else:
        print(f"  ✗ Failed: {r.status_code} - {r.text}")
        return None


def test_traditional_schedule():
    """测试传统调度API（对比）"""
    print("\n=== Test 4: /schedule (传统方式) ===")
    
    print("\n4.1 传统调度 (手动指定资源)")
    r = requests.post(f"{BASE}/schedule", json={
        "num": 1,
        "cube_requests": 0.5,
        "cube_limits": 0.75,
        "vector_requests": 0.5,
        "vector_limits": 0.75,
        "memory": [8],
        "type": "inference",
        "service_name": "test-traditional-service",
        "image": "",
        "COMMAND": "",
        "MODEL_PATH": "/vllm-workspace/models/Qwen3-4B"
    })
    if r.status_code == 200:
        data = r.json()
        print(f"  ✓ Instance ID: {data['instance_id']}")
        print(f"  ✓ Port: {data['port']}")
        print(f"  ✓ Selected NPUs: {data['selected_npus']}")
        return data['instance_id']
    else:
        print(f"  ✗ Failed: {r.status_code} - {r.text}")
        return None


def delete_instance(instance_id):
    """删除实例"""
    if instance_id:
        r = requests.post(f"{BASE}/delete_instance", json={"instance_id": instance_id})
        if r.status_code == 200:
            print(f"  ✓ Deleted instance: {instance_id}")
        else:
            print(f"  ✗ Failed to delete: {r.status_code} - {r.text}")


def main():
    print("=" * 60)
    print("资源推荐引擎API测试")
    print("=" * 60)
    
    # 启动调度器
    print("\n[1] 启动调度器...")
    t = threading.Thread(target=run_scheduler, daemon=True)
    t.start()
    time.sleep(3)  # 等待调度器启动
    
    try:
        # 测试资源推荐API
        test_recommend_resources()
        
        # 测试扩容/缩容建议API
        test_scaling_advice()
        
        # 测试基于SLA的智能调度
        print("\n[2] 测试智能调度...")
        instance_id_sla = test_schedule_with_sla()
        
        if instance_id_sla:
            # 等待实例启动
            print("\n[3] 等待实例启动 (10s)...")
            time.sleep(10)
            
            # 测试推理
            print("\n[4] 测试推理...")
            port = None
            r = requests.get(f"{BASE}/instances")
            if r.status_code == 200:
                instances = r.json().get('instances', [])
                for inst in instances:
                    if inst['instance_id'] == instance_id_sla:
                        port = inst['port']
                        break
            
            if port:
                r = requests.post(f"http://127.0.0.1:{port}/predict", 
                                json={"text": "Hello, how are you?"},
                                timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    print(f"  ✓ Inference successful")
                    print(f"  ✓ Latency: {data['latency']}s")
                    print(f"  ✓ Prediction: {data['prediction'][:100]}...")
                else:
                    print(f"  ✗ Inference failed: {r.status_code}")
            
            # 删除实例
            print("\n[5] 清理实例...")
            delete_instance(instance_id_sla)
        
        # 测试传统调度（对比）
        print("\n[6] 测试传统调度...")
        instance_id_traditional = test_traditional_schedule()
        if instance_id_traditional:
            time.sleep(5)
            delete_instance(instance_id_traditional)
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
