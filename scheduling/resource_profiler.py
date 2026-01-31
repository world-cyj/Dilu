# -*- coding: utf-8 -*-
"""
NPU 资源画像收集器：自动化收集模型在不同资源配置下的性能数据
用于构建资源画像数据库，支持预测性扩缩容
"""
import os
import sys
import json
import time
import uuid
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
import threading
import requests

# 确保 scheduling 在 path 中
sched_dir = os.path.dirname(os.path.abspath(__file__))
if sched_dir not in sys.path:
    sys.path.insert(0, sched_dir)


@dataclass
class ProfilingResult:
    """单次画像结果"""
    # 配置信息
    config_id: str
    model_path: str
    model_name: str
    batch_size: int
    cube: int
    vector: int
    memory: int
    
    # 性能指标
    mean_latency: float
    p50_latency: float
    p90_latency: float
    p99_latency: float
    throughput: float
    
    # 资源使用
    npu_utilization: float
    memory_usage_gb: float
    
    # 元数据
    timestamp: str
    device_id: int
    test_iterations: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ResourceProfiler:
    """
    NPU 资源画像收集器
    
    功能：
    1. 自动化测试不同资源配置下的模型性能
    2. 收集延迟、吞吐、资源利用率等指标
    3. 构建资源画像数据库
    4. 支持增量更新和持续优化
    """
    
    def __init__(
        self,
        output_path: str = None,
        scheduler_url: str = "http://127.0.0.1:5000"
    ):
        """
        初始化画像收集器
        
        Args:
            output_path: 画像数据输出路径
            scheduler_url: 调度器API地址
        """
        self.output_path = output_path or os.path.join(
            os.path.dirname(__file__), "..", "profiling", "npu", "profiling_npu_result.json"
        )
        self.scheduler_url = scheduler_url
        self.results: List[ProfilingResult] = []
        self._lock = threading.Lock()
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # 加载已有数据
        self._load_existing()
    
    def _load_existing(self):
        """加载已有的画像数据"""
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for obs in data.get('observations', []):
                        self.results.append(ProfilingResult(**obs))
                print(f"[Profiler] Loaded {len(self.results)} existing results")
            except Exception as e:
                print(f"[Profiler] Error loading existing data: {e}")
    
    def _save_results(self):
        """保存画像数据到文件"""
        try:
            data = {
                'version': '2.0',
                'last_updated': datetime.now().isoformat(),
                'total_observations': len(self.results),
                'observations': [r.to_dict() for r in self.results]
            }
            
            # 先写入临时文件，再重命名，保证原子性
            temp_path = self.output_path + '.tmp'
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, self.output_path)
            
            print(f"[Profiler] Saved {len(self.results)} results to {self.output_path}")
        except Exception as e:
            print(f"[Profiler] Error saving results: {e}")
    
    def profile_model(
        self,
        model_path: str,
        model_name: str = None,
        test_prompts: List[str] = None,
        batch_sizes: List[int] = None,
        cube_configs: List[int] = None,
        vector_configs: List[int] = None,
        test_iterations: int = 10,
        warmup_iterations: int = 3,
        service_name: str = None
    ) -> List[ProfilingResult]:
        """
        对模型进行完整的资源画像
        
        Args:
            model_path: 模型路径
            model_name: 模型名称（用于标识）
            test_prompts: 测试用prompt列表
            batch_sizes: 测试的批处理大小列表
            cube_configs: 测试的Cube核心数列表
            vector_configs: 测试的Vector核心数列表
            test_iterations: 每个配置的测试次数
            warmup_iterations: 预热次数
            service_name: 服务名称
            
        Returns:
            所有测试结果列表
        """
        model_name = model_name or os.path.basename(model_path)
        service_name = service_name or f"profile-{model_name}-{uuid.uuid4().hex[:8]}"
        
        # 默认测试配置
        test_prompts = test_prompts or [
            "你好，请介绍一下自己。",
            "什么是机器学习？",
            "解释深度学习的基本原理。",
        ]
        batch_sizes = batch_sizes or [1, 2, 4]
        cube_configs = cube_configs or [4, 8, 12, 16, 20]
        vector_configs = vector_configs or [10, 20, 30, 40]
        
        new_results = []
        total_tests = len(batch_sizes) * len(cube_configs) * len(vector_configs)
        test_count = 0
        
        print(f"\n[Profiler] Starting profiling for {model_name}")
        print(f"[Profiler] Total configurations to test: {total_tests}")
        print(f"[Profiler] Test prompts: {len(test_prompts)}")
        print(f"[Profiler] Iterations per config: {test_iterations}")
        
        for batch_size in batch_sizes:
            for cube in cube_configs:
                for vector in vector_configs:
                    test_count += 1
                    print(f"\n[Profiler] [{test_count}/{total_tests}] Testing: "
                          f"batch={batch_size}, cube={cube}, vector={vector}")
                    
                    try:
                        result = self._test_configuration(
                            model_path=model_path,
                            model_name=model_name,
                            service_name=service_name,
                            batch_size=batch_size,
                            cube=cube,
                            vector=vector,
                            test_prompts=test_prompts,
                            test_iterations=test_iterations,
                            warmup_iterations=warmup_iterations
                        )
                        
                        if result:
                            new_results.append(result)
                            with self._lock:
                                self.results.append(result)
                            print(f"[Profiler] ✓ Latency: {result.mean_latency:.3f}s, "
                                  f"Throughput: {result.throughput:.2f}")
                        
                    except Exception as e:
                        print(f"[Profiler] ✗ Error: {e}")
                        traceback.print_exc()
                    
                    # 每完成一个配置就保存一次，防止数据丢失
                    if test_count % 5 == 0:
                        self._save_results()
        
        # 最终保存
        self._save_results()
        
        print(f"\n[Profiler] Profiling completed!")
        print(f"[Profiler] New results: {len(new_results)}")
        print(f"[Profiler] Total results: {len(self.results)}")
        
        return new_results
    
    def _test_configuration(
        self,
        model_path: str,
        model_name: str,
        service_name: str,
        batch_size: int,
        cube: int,
        vector: int,
        test_prompts: List[str],
        test_iterations: int,
        warmup_iterations: int
    ) -> Optional[ProfilingResult]:
        """测试单个配置"""
        
        # 1. 部署服务
        instance_id, port = self._deploy_service(
            service_name=service_name,
            model_path=model_path,
            cube=cube,
            vector=vector
        )
        
        if not instance_id or not port:
            print("[Profiler] Failed to deploy service")
            return None
        
        try:
            # 2. 等待服务就绪
            if not self._wait_for_ready(port, timeout=120):
                print("[Profiler] Service failed to become ready")
                return None
            
            # 3. 预热
            print(f"[Profiler] Warming up ({warmup_iterations} iterations)...")
            for _ in range(warmup_iterations):
                self._send_request(port, test_prompts[0])
            
            # 4. 正式测试
            print(f"[Profiler] Testing ({test_iterations} iterations)...")
            latencies = []
            
            for i in range(test_iterations):
                prompt = test_prompts[i % len(test_prompts)]
                latency = self._send_request(port, prompt)
                if latency > 0:
                    latencies.append(latency)
            
            if not latencies:
                print("[Profiler] No successful requests")
                return None
            
            # 5. 计算统计指标
            latencies.sort()
            mean_latency = sum(latencies) / len(latencies)
            p50 = latencies[len(latencies) // 2]
            p90 = latencies[int(len(latencies) * 0.9)]
            p99 = latencies[int(len(latencies) * 0.99)] if len(latencies) >= 100 else p90
            throughput = batch_size / mean_latency if mean_latency > 0 else 0
            
            # 6. 获取资源使用信息
            npu_util, memory_usage = self._get_resource_usage(port)
            
            result = ProfilingResult(
                config_id=f"{model_name}_b{batch_size}_c{cube}_v{vector}",
                model_path=model_path,
                model_name=model_name,
                batch_size=batch_size,
                cube=cube,
                vector=vector,
                memory=64,  # 简化处理
                mean_latency=mean_latency,
                p50_latency=p50,
                p90_latency=p90,
                p99_latency=p99,
                throughput=throughput,
                npu_utilization=npu_util,
                memory_usage_gb=memory_usage,
                timestamp=datetime.now().isoformat(),
                device_id=0,  # 简化处理
                test_iterations=len(latencies)
            )
            
            return result
            
        finally:
            # 7. 清理服务
            self._delete_instance(instance_id)
    
    def _deploy_service(
        self,
        service_name: str,
        model_path: str,
        cube: int,
        vector: int
    ) -> tuple:
        """部署测试服务"""
        try:
            payload = {
                "service_name": service_name,
                "model_path": model_path,
                "type": "inference",
                "cube_requests": cube / 20,
                "cube_limits": min(cube * 1.2 / 20, 1.0),
                "vector_requests": vector / 40,
                "vector_limits": min(vector * 1.2 / 40, 1.0),
                "memory": [64],
            }
            
            r = requests.post(
                f"{self.scheduler_url}/schedule",
                json=payload,
                timeout=30
            )
            
            if r.status_code == 200:
                data = r.json()
                return data.get("instance_id"), data.get("port")
            else:
                print(f"[Profiler] Deploy failed: {r.status_code} - {r.text}")
                return None, None
                
        except Exception as e:
            print(f"[Profiler] Deploy error: {e}")
            return None, None
    
    def _wait_for_ready(self, port: int, timeout: int = 120) -> bool:
        """等待服务就绪"""
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("model_loaded"):
                        return True
            except:
                pass
            time.sleep(2)
        return False
    
    def _send_request(self, port: int, prompt: str) -> float:
        """发送推理请求，返回延迟"""
        try:
            start = time.time()
            r = requests.post(
                f"http://127.0.0.1:{port}/predict",
                json={"text": prompt},
                timeout=60
            )
            elapsed = time.time() - start
            
            if r.status_code == 200:
                return elapsed
            return -1
        except:
            return -1
    
    def _get_resource_usage(self, port: int) -> tuple:
        """获取资源使用情况"""
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
            if r.status_code == 200:
                data = r.json()
                memory_gb = data.get("memory_gb", 0)
                # 简化处理，实际应该从NPU驱动获取利用率
                return 0.0, memory_gb
        except:
            pass
        return 0.0, 0.0
    
    def _delete_instance(self, instance_id: str):
        """删除实例"""
        try:
            requests.post(
                f"{self.scheduler_url}/delete_instance",
                json={"instance_id": instance_id},
                timeout=10
            )
        except:
            pass
    
    def get_profile_summary(self, model_name: str = None) -> Dict:
        """获取画像摘要"""
        with self._lock:
            results = self.results
            if model_name:
                results = [r for r in results if r.model_name == model_name]
        
        if not results:
            return {"error": "No profiling data available"}
        
        # 计算统计信息
        latencies = [r.mean_latency for r in results]
        throughputs = [r.throughput for r in results]
        
        summary = {
            "total_configs": len(results),
            "models": list(set(r.model_name for r in results)),
            "latency": {
                "min": min(latencies),
                "max": max(latencies),
                "mean": sum(latencies) / len(latencies),
            },
            "throughput": {
                "min": min(throughputs),
                "max": max(throughputs),
                "mean": sum(throughputs) / len(throughputs),
            },
            "best_latency_config": min(results, key=lambda x: x.mean_latency).to_dict(),
            "best_throughput_config": max(results, key=lambda x: x.throughput).to_dict(),
        }
        
        return summary


def run_profiling_workflow(
    model_path: str,
    model_name: str = None,
    scheduler_url: str = "http://127.0.0.1:5000"
):
    """
    运行完整的资源画像流程
    
    Args:
        model_path: 模型路径
        model_name: 模型名称
        scheduler_url: 调度器URL
    """
    print("=" * 60)
    print("NPU Resource Profiling Workflow")
    print("=" * 60)
    
    # 创建画像收集器
    profiler = ResourceProfiler(scheduler_url=scheduler_url)
    
    # 运行画像
    results = profiler.profile_model(
        model_path=model_path,
        model_name=model_name,
        test_iterations=5,  # 减少测试次数以加快流程
        warmup_iterations=2,
    )
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("Profiling Summary")
    print("=" * 60)
    
    summary = profiler.get_profile_summary(model_name)
    print(f"Total configurations tested: {summary['total_configs']}")
    print(f"\nLatency range: {summary['latency']['min']:.3f}s - {summary['latency']['max']:.3f}s")
    print(f"Throughput range: {summary['throughput']['min']:.2f} - {summary['throughput']['max']:.2f}")
    
    best_lat = summary['best_latency_config']
    print(f"\nBest latency config: Cube={best_lat['cube']}, Vector={best_lat['vector']}")
    print(f"  Latency: {best_lat['mean_latency']:.3f}s")
    
    best_tput = summary['best_throughput_config']
    print(f"\nBest throughput config: Cube={best_tput['cube']}, Vector={best_tput['vector']}")
    print(f"  Throughput: {best_tput['throughput']:.2f}")
    
    print("\n" + "=" * 60)
    print("Profiling workflow completed!")
    print(f"Results saved to: {profiler.output_path}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NPU Resource Profiler")
    parser.add_argument("--model_path", required=True, help="Path to the model")
    parser.add_argument("--model_name", help="Model name (optional)")
    parser.add_argument("--scheduler", default="http://127.0.0.1:5000", help="Scheduler URL")
    
    args = parser.parse_args()
    
    run_profiling_workflow(
        model_path=args.model_path,
        model_name=args.model_name,
        scheduler_url=args.scheduler
    )
