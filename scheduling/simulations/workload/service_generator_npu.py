# -*- coding: utf-8 -*-
"""
NPU Service Generator
为NPU场景生成仿真服务工作负载
支持Cube/Vector资源规格和LLM推理服务
"""
import random
import string
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# NPU资源常量
NPU_CUBE_CORES_PER_DEVICE = 20
NPU_VECTOR_CORES_PER_DEVICE = 40
NPU_MEMORY_GB_PER_DEVICE = 64


def generate_service_name(prefix):
    """生成服务名称"""
    return f"{prefix}-{''.join(random.choices(string.ascii_lowercase + string.digits, k=6))}"


def generate_npu_instances(total_instances, train_ratio=0.2, llm_ratio=0.2):
    """
    生成NPU实例配置
    
    Args:
        total_instances: 总实例数
        train_ratio: 训练任务比例
        llm_ratio: LLM推理任务比例
        
    Returns:
        实例配置列表
    """
    instances = []
    
    llm_inference_count = int(total_instances * llm_ratio)
    train_count = int(total_instances * train_ratio)
    inference_count = total_instances - train_count - llm_inference_count
    
    # 生成训练任务
    for _ in range(train_count):
        service_name = generate_service_name("training")
        npu_num = random.choice([2, 4])  # 训练任务使用多个NPU
        
        # 训练任务通常需要更多资源
        memory_per_npu = random.choice([32, 40, 48, 56, 64])
        memory_list = [memory_per_npu] * npu_num
        
        # Cube/Vector核心配置（比例形式）
        cube_requests = round(random.uniform(0.6, 0.9), 2)
        cube_limits = min(cube_requests + 0.1, 1.0)
        vector_requests = round(random.uniform(0.6, 0.9), 2)
        vector_limits = min(vector_requests + 0.1, 1.0)
        
        instance = {
            "service_name": service_name,
            "type": "training",
            "npu_num": npu_num,
            "cube_requests": cube_requests,
            "cube_limits": round(cube_limits, 2),
            "vector_requests": vector_requests,
            "vector_limits": round(vector_limits, 2),
            "memory": memory_list,
            "is_llm": 0,
            "priority": random.randint(1, 5),
            "throughput": random.randint(5, 15),
            "commands": "",
            "model_path": "",
            "sla_latency": None,
            "sla_throughput": None
        }
        instances.append(instance)
    
    # 生成普通推理任务
    for _ in range(inference_count):
        service_name = generate_service_name("inference")
        npu_num = 1
        
        # 普通推理任务资源需求较低
        memory = random.choice([4, 6, 8, 10, 12, 16])
        
        # Cube/Vector核心配置
        cube_requests = round(random.uniform(0.2, 0.5), 2)
        cube_limits = min(cube_requests + 0.15, 0.8)
        vector_requests = round(random.uniform(0.2, 0.5), 2)
        vector_limits = min(vector_requests + 0.15, 0.8)
        
        instance = {
            "service_name": service_name,
            "type": "inference",
            "npu_num": npu_num,
            "cube_requests": cube_requests,
            "cube_limits": round(cube_limits, 2),
            "vector_requests": vector_requests,
            "vector_limits": round(vector_limits, 2),
            "memory": [memory],
            "is_llm": 0,
            "priority": random.randint(1, 3),
            "throughput": random.randint(10, 30),
            "commands": "",
            "model_path": "",
            "sla_latency": round(random.uniform(0.1, 0.5), 2),
            "sla_throughput": None
        }
        instances.append(instance)
    
    # 生成LLM推理任务
    for _ in range(llm_inference_count):
        service_name = generate_service_name("llm-inference")
        npu_num = 1
        
        # LLM任务需要更多内存
        memory = random.choice([20, 24, 28, 32, 40, 48])
        
        # LLM任务需要更多Cube核心
        cube_requests = round(random.uniform(0.5, 0.8), 2)
        cube_limits = min(cube_requests + 0.15, 1.0)
        vector_requests = round(random.uniform(0.4, 0.7), 2)
        vector_limits = min(vector_requests + 0.15, 1.0)
        
        instance = {
            "service_name": service_name,
            "type": "llm-inference",
            "npu_num": npu_num,
            "cube_requests": cube_requests,
            "cube_limits": round(cube_limits, 2),
            "vector_requests": vector_requests,
            "vector_limits": round(vector_limits, 2),
            "memory": [memory],
            "is_llm": 1,
            "priority": random.randint(2, 4),
            "throughput": random.randint(5, 15),
            "commands": "",
            "model_path": f"/vllm-workspace/models/model-{random.randint(1, 10)}",
            "sla_latency": round(random.uniform(0.2, 1.0), 2),
            "sla_throughput": random.randint(3, 10)
        }
        instances.append(instance)
    
    return instances


def generate_npu_events(instances, total_instances, 
                        delete_ratio_inference=0.6,
                        delete_ratio_training=0.1,
                        inference_lifetime_min=1, inference_lifetime_max=3,
                        training_lifetime_min=10, training_lifetime_max=15):
    """
    生成NPU服务生命周期事件
    
    Args:
        instances: 实例列表
        total_instances: 总实例数
        delete_ratio_inference: 推理任务删除比例
        delete_ratio_training: 训练任务删除比例
        inference_lifetime_min/max: 推理任务生命周期（分钟）
        training_lifetime_min/max: 训练任务生命周期（分钟）
        
    Returns:
        事件列表
    """
    # 打乱实例顺序
    random.shuffle(instances)
    
    # 生成启动事件
    start_time = datetime.now()
    events = []
    for i, instance in enumerate(instances):
        events.append({
            "time": start_time + timedelta(seconds=i * 2),  # 每2秒启动一个
            "action": "start",
            "instance": instance
        })
    
    # 为推理和LLM推理任务生成删除事件
    deletable_inference = [e for e in events 
                          if e["instance"]["type"] in ["llm-inference", "inference"]]
    num_deletes_inference = int(total_instances * delete_ratio_inference)
    delete_events_inference = random.sample(deletable_inference, 
                                           min(num_deletes_inference, len(deletable_inference)))
    
    for event in delete_events_inference:
        lifetime = random.randint(inference_lifetime_min, inference_lifetime_max)
        delete_time = event["time"] + timedelta(minutes=lifetime)
        events.append({
            "time": delete_time,
            "action": "delete",
            "instance": event["instance"]
        })
    
    # 为训练任务生成删除事件
    deletable_training = [e for e in events if e["instance"]["type"] == "training"]
    num_deletes_training = int(total_instances * delete_ratio_training)
    delete_events_training = random.sample(deletable_training,
                                          min(num_deletes_training, len(deletable_training)))
    
    for event in delete_events_training:
        lifetime = random.randint(training_lifetime_min, training_lifetime_max)
        delete_time = event["time"] + timedelta(minutes=lifetime)
        events.append({
            "time": delete_time,
            "action": "delete",
            "instance": event["instance"]
        })
    
    # 按时间排序
    events.sort(key=lambda x: x["time"])
    
    return events


def convert_to_scheduler_format(instance):
    """
    将实例配置转换为调度器请求格式
    
    Args:
        instance: 实例配置
        
    Returns:
        调度器请求字典
    """
    return {
        "num": instance["npu_num"],
        "cube_requests": instance["cube_requests"],
        "cube_limits": instance["cube_limits"],
        "vector_requests": instance["vector_requests"],
        "vector_limits": instance["vector_limits"],
        "memory": instance["memory"],
        "type": instance["type"],
        "service_name": instance["service_name"],
        "image": "",
        "COMMAND": instance["commands"],
        "MODEL_PATH": instance["model_path"],
        "is_llm": instance["is_llm"],
        "priority": instance["priority"],
        "throughput": instance["throughput"],
        "sla_latency": instance.get("sla_latency"),
        "sla_throughput": instance.get("sla_throughput")
    }


def generate_npu_workload(total_instances=100, train_ratio=0.2, llm_ratio=0.2):
    """
    生成完整的NPU工作负载
    
    Args:
        total_instances: 总实例数
        train_ratio: 训练任务比例
        llm_ratio: LLM推理任务比例
        
    Returns:
        (instances, events) 元组
    """
    instances = generate_npu_instances(total_instances, train_ratio, llm_ratio)
    events = generate_npu_events(instances, total_instances)
    return instances, events


if __name__ == "__main__":
    # 测试代码
    print("=" * 80)
    print("NPU Service Generator Test")
    print("=" * 80)
    
    total_instances = 20
    train_ratio = 0.2
    llm_ratio = 0.2
    
    print(f"\nGenerating {total_instances} instances:")
    print(f"  - Training ratio: {train_ratio}")
    print(f"  - LLM inference ratio: {llm_ratio}")
    print(f"  - Regular inference ratio: {1 - train_ratio - llm_ratio}")
    
    instances, events = generate_npu_workload(total_instances, train_ratio, llm_ratio)
    
    print(f"\nGenerated {len(instances)} instances and {len(events)} events")
    
    # 统计信息
    training_count = sum(1 for i in instances if i["type"] == "training")
    inference_count = sum(1 for i in instances if i["type"] == "inference")
    llm_count = sum(1 for i in instances if i["type"] == "llm-inference")
    
    print(f"\nInstance breakdown:")
    print(f"  - Training: {training_count}")
    print(f"  - Regular inference: {inference_count}")
    print(f"  - LLM inference: {llm_count}")
    
    # 资源统计
    total_cube = sum(i["cube_limits"] * i["npu_num"] for i in instances)
    total_vector = sum(i["vector_limits"] * i["npu_num"] for i in instances)
    total_memory = sum(sum(i["memory"]) for i in instances)
    
    print(f"\nTotal resource requirements:")
    print(f"  - Cube cores: {total_cube:.1f} (ratio)")
    print(f"  - Vector cores: {total_vector:.1f} (ratio)")
    print(f"  - Memory: {total_memory} GB")
    
    # 打印前5个事件
    print(f"\nFirst 5 events:")
    for event in events[:5]:
        print(f"  {event['time'].strftime('%Y-%m-%d %H:%M:%S')} - {event['action']}")
        inst = event['instance']
        print(f"    Service: {inst['service_name']}, Type: {inst['type']}")
        print(f"    Resources: Cube={inst['cube_limits']}, Vector={inst['vector_limits']}, Memory={inst['memory']}")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)
