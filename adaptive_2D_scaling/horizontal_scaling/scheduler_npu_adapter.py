# -*- coding: utf-8 -*-
"""
NPU Horizontal Scaling Scheduler Adapter
适配器模式：复用 scheduling/scheduler_npu.py 的核心逻辑
为 horizontal_scaling 组件提供兼容的接口
"""
import os
import sys
import json
import requests
import threading
import time

# 添加 scheduling 目录到 path
scheduling_dir = os.path.join(os.path.dirname(__file__), "..", "..", "scheduling")
if scheduling_dir not in sys.path:
    sys.path.insert(0, scheduling_dir)

# 复用 scheduling 中的组件
from scheduler_npu import (
    app, npu_list, active_npus, new_npus, instance_registry,
    allocate_instance, start_instance, stop_instance,
    NPU_CUBE_CORES_PER_DEVICE, NPU_VECTOR_CORES_PER_DEVICE, NPU_MEMORY_GB_PER_DEVICE
)
from npu_resource import NPU


class HorizontalScalingSchedulerNPU:
    """
    Horizontal Scaling NPU 调度器适配器
    
    提供与原有 horizontal_scaling/scheduler.py 兼容的接口，
    但底层使用 scheduling/scheduler_npu.py 的实现。
    """
    
    def __init__(self, port=5000):
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self._server_thread = None
        self._running = False
        
    def start(self):
        """启动调度器服务"""
        if self._running:
            return
        
        def run_server():
            app.run(debug=False, host="0.0.0.0", port=self.port, use_reloader=False)
        
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        self._running = True
        
        # 等待服务启动
        time.sleep(2)
        print(f"[HorizontalScalingSchedulerNPU] Started on port {self.port}")
    
    def stop(self):
        """停止调度器服务"""
        self._running = False
        # Flask 服务无法优雅停止，依赖进程结束
    
    def schedule(self, service_config):
        """
        调度服务实例
        
        Args:
            service_config: 服务配置字典，包含：
                - name: 服务名称
                - model_path: 模型路径
                - sla_latency: 延迟SLA（可选）
                - sla_throughput: 吞吐SLA（可选）
                - cube_requests: Cube核心请求（比例）
                - cube_limits: Cube核心限制（比例）
                - vector_requests: Vector核心请求（比例）
                - vector_limits: Vector核心限制（比例）
                - memory: 内存需求（GB）
                
        Returns:
            调度结果字典
        """
        # 构建调度请求
        schedule_request = {
            "num": 1,
            "service_name": service_config.get("name", "unnamed-service"),
            "type": service_config.get("type", "inference"),
            "image": "",
            "COMMAND": "",
            "MODEL_PATH": service_config.get("model_path", ""),
        }
        
        # 添加资源需求
        if "cube_requests" in service_config:
            schedule_request["cube_requests"] = service_config["cube_requests"]
        if "cube_limits" in service_config:
            schedule_request["cube_limits"] = service_config["cube_limits"]
        if "vector_requests" in service_config:
            schedule_request["vector_requests"] = service_config["vector_requests"]
        if "vector_limits" in service_config:
            schedule_request["vector_limits"] = service_config["vector_limits"]
        if "memory" in service_config:
            schedule_request["memory"] = [service_config["memory"]]
        
        # 如果有SLA，使用智能调度
        if "sla_latency" in service_config or "sla_throughput" in service_config:
            return self._schedule_with_sla(service_config)
        
        # 否则使用普通调度
        try:
            r = requests.post(f"{self.base_url}/schedule", json=schedule_request, timeout=30)
            if r.status_code == 200:
                return r.json()
            else:
                return {"error": f"Schedule failed: {r.status_code}", "details": r.text}
        except Exception as e:
            return {"error": f"Schedule request failed: {str(e)}"}
    
    def _schedule_with_sla(self, service_config):
        """使用SLA智能调度"""
        sla_request = {
            "service_name": service_config.get("name", "unnamed-service"),
            "model_path": service_config.get("model_path", ""),
            "batch_size": service_config.get("batch_size", 1),
            "type": service_config.get("type", "inference"),
        }
        
        if "sla_latency" in service_config:
            sla_request["sla_latency"] = service_config["sla_latency"]
        if "sla_throughput" in service_config:
            sla_request["sla_throughput"] = service_config["sla_throughput"]
        
        try:
            r = requests.post(f"{self.base_url}/schedule_with_sla", json=sla_request, timeout=30)
            if r.status_code == 200:
                result = r.json()
                # 添加推荐信息
                result["recommended_config"] = result.get("recommendation", {})
                return result
            else:
                return {"error": f"SLA schedule failed: {r.status_code}", "details": r.text}
        except Exception as e:
            return {"error": f"SLA schedule request failed: {str(e)}"}
    
    def delete_instance(self, instance_id):
        """删除实例"""
        try:
            r = requests.post(f"{self.base_url}/delete_instance", json={"instance_id": instance_id}, timeout=30)
            return r.status_code == 200
        except Exception as e:
            print(f"[HorizontalScalingSchedulerNPU] Failed to delete instance: {e}")
            return False
    
    def get_instances(self):
        """获取所有实例"""
        try:
            r = requests.get(f"{self.base_url}/instances", timeout=10)
            if r.status_code == 200:
                return r.json().get("instances", [])
            return []
        except Exception as e:
            print(f"[HorizontalScalingSchedulerNPU] Failed to get instances: {e}")
            return []
    
    def get_cluster_resources(self):
        """获取集群资源使用情况"""
        try:
            r = requests.get(f"{self.base_url}/cluster_resources", timeout=10)
            if r.status_code == 200:
                return r.json()
            return {}
        except Exception as e:
            print(f"[HorizontalScalingSchedulerNPU] Failed to get cluster resources: {e}")
            return {}
    
    def get_scaling_advice(self, current_cube, current_vector, current_latency, target_latency, batch_size=1):
        """获取扩容/缩容建议"""
        try:
            r = requests.post(f"{self.base_url}/scaling_advice", json={
                "current_cube": current_cube,
                "current_vector": current_vector,
                "current_latency": current_latency,
                "target_latency": target_latency,
                "batch_size": batch_size
            }, timeout=10)
            if r.status_code == 200:
                return r.json()
            return {"error": f"Failed to get scaling advice: {r.status_code}"}
        except Exception as e:
            return {"error": f"Failed to get scaling advice: {str(e)}"}
    
    def recommend_resources(self, model_path, sla_latency=None, sla_throughput=None, batch_size=1):
        """获取资源配置建议"""
        try:
            request_data = {
                "model_path": model_path,
                "batch_size": batch_size
            }
            if sla_latency is not None:
                request_data["sla_latency"] = sla_latency
            if sla_throughput is not None:
                request_data["sla_throughput"] = sla_throughput
            
            r = requests.post(f"{self.base_url}/recommend_resources", json=request_data, timeout=10)
            if r.status_code == 200:
                return r.json()
            return {"error": f"Failed to get recommendation: {r.status_code}"}
        except Exception as e:
            return {"error": f"Failed to get recommendation: {str(e)}"}


# 兼容原有接口的函数

def create_scheduler(port=5000):
    """创建调度器实例（兼容原有接口）"""
    scheduler = HorizontalScalingSchedulerNPU(port)
    scheduler.start()
    return scheduler


def schedule_service(scheduler, service_config):
    """调度服务（兼容原有接口）"""
    return scheduler.schedule(service_config)


def delete_service_instance(scheduler, instance_id):
    """删除服务实例（兼容原有接口）"""
    return scheduler.delete_instance(instance_id)


if __name__ == "__main__":
    # 测试代码
    print("Testing HorizontalScalingSchedulerNPU Adapter...")
    
    scheduler = create_scheduler(port=5001)
    
    # 测试1：普通调度
    print("\n=== Test 1: Normal Schedule ===")
    result = schedule_service(scheduler, {
        "name": "test-service-1",
        "model_path": "/vllm-workspace/models/Qwen3-4B",
        "cube_requests": 0.5,
        "cube_limits": 0.75,
        "vector_requests": 0.5,
        "vector_limits": 0.75,
        "memory": 8
    })
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # 测试2：SLA智能调度
    print("\n=== Test 2: SLA-based Schedule ===")
    result = schedule_service(scheduler, {
        "name": "test-service-2",
        "model_path": "/vllm-workspace/models/Qwen3-4B",
        "sla_latency": 0.2,
        "batch_size": 1
    })
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # 测试3：获取资源配置建议
    print("\n=== Test 3: Resource Recommendation ===")
    result = scheduler.recommend_resources(
        model_path="/vllm-workspace/models/Qwen3-4B",
        sla_latency=0.2,
        batch_size=1
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # 测试4：获取扩容建议
    print("\n=== Test 4: Scaling Advice ===")
    result = scheduler.get_scaling_advice(
        current_cube=8,
        current_vector=20,
        current_latency=0.5,
        target_latency=0.2,
        batch_size=1
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # 清理
    print("\n=== Cleanup ===")
    instances = scheduler.get_instances()
    for inst in instances:
        scheduler.delete_instance(inst["instance_id"])
        print(f"Deleted instance: {inst['instance_id']}")
    
    print("\nAll tests completed!")
