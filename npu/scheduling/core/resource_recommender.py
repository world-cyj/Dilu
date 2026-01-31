# -*- coding: utf-8 -*-
"""
NPU 资源推荐引擎：基于资源画像数据，为调度器提供最优的 Cube/Vector/Memory 配置建议。
连接 profiling_npu_result.json 和 scheduler_npu.py 的桥梁。
"""
import os
import sys
import json
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# 默认资源画像文件路径
DEFAULT_PROFILING_PATH = os.path.join(
    os.path.dirname(__file__), "..", "profiling", "npu", "profiling_npu_result.json"
)


@dataclass
class ResourceConfig:
    """资源配置"""
    cube: int
    vector: int
    memory: int
    expected_latency: float
    expected_throughput: float
    confidence: float  # 置信度（0-1）


@dataclass
class SLAResult:
    """SLA满足情况"""
    config: ResourceConfig
    meets_sla: bool
    headroom: float  # SLA余量（百分比）


class ResourceRecommender:
    """
    NPU资源推荐引擎
    
    基于资源画像数据，为不同模型和负载特征推荐最优资源配置。
    支持以下功能：
    1. 根据目标延迟推荐资源
    2. 根据目标吞吐推荐资源
    3. 根据成本预算推荐资源（性价比最优）
    4. 动态调整建议（基于实时性能反馈）
    """
    
    def __init__(self, profiling_path: str = None):
        """
        初始化推荐引擎
        
        Args:
            profiling_path: 资源画像JSON文件路径，默认使用标准路径
        """
        self.profiling_path = profiling_path or DEFAULT_PROFILING_PATH
        self.observations = []
        self.model_path = ""
        self.device_id = 0
        self.load_profiling_data()
        
    def load_profiling_data(self):
        """加载资源画像数据"""
        if not os.path.exists(self.profiling_path):
            print(f"[Recommender] Warning: Profiling data not found at {self.profiling_path}")
            print(f"[Recommender] Using default configurations")
            self._init_default_data()
            return
            
        try:
            with open(self.profiling_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.observations = data.get('observations', [])
                self.model_path = data.get('model_path', '')
                self.device_id = data.get('device_id', 0)
                print(f"[Recommender] Loaded {len(self.observations)} profiling observations")
                print(f"[Recommender] Model: {self.model_path}, Device: {self.device_id}")
        except Exception as e:
            print(f"[Recommender] Error loading profiling data: {e}")
            self._init_default_data()
    
    def _init_default_data(self):
        """初始化默认数据（当画像数据不存在时）"""
        # 基于910B3的典型配置
        self.observations = [
            {"batch_size": 1, "cube": 4, "vector": 10, "mean_latency": 0.5, "throughput": 2.0},
            {"batch_size": 1, "cube": 8, "vector": 20, "mean_latency": 0.3, "throughput": 3.3},
            {"batch_size": 1, "cube": 12, "vector": 30, "mean_latency": 0.2, "throughput": 5.0},
            {"batch_size": 1, "cube": 16, "vector": 40, "mean_latency": 0.15, "throughput": 6.7},
            {"batch_size": 1, "cube": 20, "vector": 40, "mean_latency": 0.12, "throughput": 8.3},
            {"batch_size": 2, "cube": 8, "vector": 20, "mean_latency": 0.5, "throughput": 4.0},
            {"batch_size": 2, "cube": 12, "vector": 30, "mean_latency": 0.35, "throughput": 5.7},
            {"batch_size": 2, "cube": 16, "vector": 40, "mean_latency": 0.25, "throughput": 8.0},
            {"batch_size": 4, "cube": 12, "vector": 30, "mean_latency": 0.6, "throughput": 6.7},
            {"batch_size": 4, "cube": 16, "vector": 40, "mean_latency": 0.45, "throughput": 8.9},
            {"batch_size": 4, "cube": 20, "vector": 40, "mean_latency": 0.35, "throughput": 11.4},
        ]
        print(f"[Recommender] Initialized with {len(self.observations)} default configurations")
    
    def recommend_for_latency(
        self, 
        target_latency: float, 
        batch_size: int = 1,
        max_cube: int = 20,
        max_vector: int = 40,
        max_memory: int = 64,
        confidence_threshold: float = 0.8
    ) -> Optional[ResourceConfig]:
        """
        根据目标延迟推荐资源配置
        
        Args:
            target_latency: 目标延迟（秒）
            batch_size: 批处理大小
            max_cube: 最大Cube核心数
            max_vector: 最大Vector核心数
            max_memory: 最大内存（GB）
            confidence_threshold: 置信度阈值
            
        Returns:
            最优资源配置，如果无法满足则返回None
        """
        candidates = []
        
        for obs in self.observations:
            if obs['batch_size'] != batch_size:
                continue
            if obs['cube'] > max_cube or obs['vector'] > max_vector:
                continue
            if obs['mean_latency'] <= 0:  # 无效数据
                continue
                
            # 计算是否满足延迟要求
            meets_sla = obs['mean_latency'] <= target_latency
            
            # 计算资源效率（延迟满足情况下的最小资源）
            resource_usage = (obs['cube'] / max_cube + obs['vector'] / max_vector) / 2
            
            # 计算置信度（基于测试迭代次数，这里简化为固定值）
            confidence = 0.9
            
            if meets_sla and confidence >= confidence_threshold:
                candidates.append({
                    'cube': obs['cube'],
                    'vector': obs['vector'],
                    'memory': max_memory,  # 简化处理，实际应根据模型计算
                    'latency': obs['mean_latency'],
                    'throughput': obs['throughput'],
                    'confidence': confidence,
                    'resource_usage': resource_usage
                })
        
        if not candidates:
            print(f"[Recommender] No configuration meets latency target {target_latency}s")
            return None
        
        # 选择资源使用最少且满足SLA的配置
        best = min(candidates, key=lambda x: x['resource_usage'])
        
        return ResourceConfig(
            cube=best['cube'],
            vector=best['vector'],
            memory=best['memory'],
            expected_latency=best['latency'],
            expected_throughput=best['throughput'],
            confidence=best['confidence']
        )
    
    def recommend_for_throughput(
        self,
        target_throughput: float,
        batch_size: int = 1,
        max_cube: int = 20,
        max_vector: int = 40,
        max_memory: int = 64
    ) -> Optional[ResourceConfig]:
        """
        根据目标吞吐推荐资源配置
        
        Args:
            target_throughput: 目标吞吐（请求/秒）
            batch_size: 批处理大小
            max_cube: 最大Cube核心数
            max_vector: 最大Vector核心数
            max_memory: 最大内存（GB）
            
        Returns:
            最优资源配置
        """
        candidates = []
        
        for obs in self.observations:
            if obs['batch_size'] != batch_size:
                continue
            if obs['cube'] > max_cube or obs['vector'] > max_vector:
                continue
            if obs['throughput'] <= 0:
                continue
                
            meets_sla = obs['throughput'] >= target_throughput
            
            if meets_sla:
                candidates.append({
                    'cube': obs['cube'],
                    'vector': obs['vector'],
                    'memory': max_memory,
                    'latency': obs['mean_latency'],
                    'throughput': obs['throughput'],
                    'resource_usage': (obs['cube'] / max_cube + obs['vector'] / max_vector) / 2
                })
        
        if not candidates:
            print(f"[Recommender] No configuration meets throughput target {target_throughput}")
            return None
        
        # 选择资源使用最少的配置
        best = min(candidates, key=lambda x: x['resource_usage'])
        
        return ResourceConfig(
            cube=best['cube'],
            vector=best['vector'],
            memory=best['memory'],
            expected_latency=best['latency'],
            expected_throughput=best['throughput'],
            confidence=0.9
        )
    
    def recommend_cost_effective(
        self,
        batch_size: int = 1,
        max_cube: int = 20,
        max_vector: int = 40,
        max_memory: int = 64
    ) -> ResourceConfig:
        """
        推荐性价比最高的资源配置（吞吐/资源比最高）
        
        Returns:
            性价比最优的资源配置
        """
        best_score = 0
        best_config = None
        
        for obs in self.observations:
            if obs['batch_size'] != batch_size:
                continue
            if obs['cube'] > max_cube or obs['vector'] > max_vector:
                continue
                
            # 计算资源消耗
            resource_cost = obs['cube'] + obs['vector'] / 2  # Vector核心权重较低
            
            # 计算性价比（吞吐/资源消耗）
            efficiency = obs['throughput'] / resource_cost if resource_cost > 0 else 0
            
            if efficiency > best_score:
                best_score = efficiency
                best_config = obs
        
        if best_config is None:
            # 返回默认配置
            return ResourceConfig(
                cube=8, vector=20, memory=8,
                expected_latency=0.3, expected_throughput=3.3, confidence=0.8
            )
        
        return ResourceConfig(
            cube=best_config['cube'],
            vector=best_config['vector'],
            memory=max_memory,
            expected_latency=best_config['mean_latency'],
            expected_throughput=best_config['throughput'],
            confidence=0.9
        )
    
    def get_scaling_advice(
        self,
        current_cube: int,
        current_vector: int,
        current_latency: float,
        target_latency: float,
        batch_size: int = 1
    ) -> Dict:
        """
        获取扩容/缩容建议
        
        Args:
            current_cube: 当前Cube核心数
            current_vector: 当前Vector核心数
            current_latency: 当前延迟
            target_latency: 目标延迟
            batch_size: 批处理大小
            
        Returns:
            包含建议动作和预期效果的字典
        """
        # 查找当前配置附近的性能数据
        current_perf = None
        better_configs = []
        
        for obs in self.observations:
            if obs['batch_size'] != batch_size:
                continue
                
            if obs['cube'] == current_cube and obs['vector'] == current_vector:
                current_perf = obs
            elif obs['mean_latency'] < current_latency:
                better_configs.append(obs)
        
        if current_perf is None:
            return {
                'action': 'unknown',
                'reason': 'Current configuration not in profiling data',
                'recommendation': self.recommend_for_latency(target_latency, batch_size)
            }
        
        latency_gap = current_latency - target_latency
        
        if latency_gap > 0:
            # 需要扩容
            if not better_configs:
                return {
                    'action': 'scale_up_max',
                    'reason': f'Current latency {current_latency}s exceeds target {target_latency}s, no better config available',
                    'current': {'cube': current_cube, 'vector': current_vector},
                    'recommendation': ResourceConfig(
                        cube=20, vector=40, memory=64,
                        expected_latency=0.12, expected_throughput=8.3, confidence=0.7
                    )
                }
            
            # 找到能满足SLA的最小配置
            best_better = min(better_configs, 
                            key=lambda x: (x['cube'] + x['vector']/2))
            
            return {
                'action': 'scale_up',
                'reason': f'Current latency {current_latency}s exceeds target {target_latency}s',
                'current': {'cube': current_cube, 'vector': current_vector, 'latency': current_latency},
                'target': {'latency': target_latency},
                'recommendation': ResourceConfig(
                    cube=best_better['cube'],
                    vector=best_better['vector'],
                    memory=64,
                    expected_latency=best_better['mean_latency'],
                    expected_throughput=best_better['throughput'],
                    confidence=0.85
                )
            }
        else:
            # 可以缩容
            # 找到仍能满足SLA的最小配置
            valid_configs = [obs for obs in self.observations 
                           if obs['batch_size'] == batch_size 
                           and obs['mean_latency'] <= target_latency
                           and obs['cube'] <= current_cube
                           and obs['vector'] <= current_vector]
            
            if valid_configs:
                best_smaller = min(valid_configs,
                                 key=lambda x: (x['cube'] + x['vector']/2))
                
                if best_smaller['cube'] < current_cube or best_smaller['vector'] < current_vector:
                    return {
                        'action': 'scale_down',
                        'reason': f'Current config over-provisioned for target latency {target_latency}s',
                        'current': {'cube': current_cube, 'vector': current_vector, 'latency': current_latency},
                        'recommendation': ResourceConfig(
                            cube=best_smaller['cube'],
                            vector=best_smaller['vector'],
                            memory=64,
                            expected_latency=best_smaller['mean_latency'],
                            expected_throughput=best_smaller['throughput'],
                            confidence=0.85
                        )
                    }
            
            return {
                'action': 'maintain',
                'reason': f'Current configuration is optimal for target latency {target_latency}s',
                'current': {'cube': current_cube, 'vector': current_vector, 'latency': current_latency}
            }
    
    def generate_schedule_request(
        self,
        service_name: str,
        model_path: str,
        sla_latency: float = None,
        sla_throughput: float = None,
        batch_size: int = 1,
        task_type: str = 'inference'
    ) -> Dict:
        """
        生成调度请求（供scheduler_npu.py使用）
        
        Args:
            service_name: 服务名称
            model_path: 模型路径
            sla_latency: 延迟SLA（秒）
            sla_throughput: 吞吐SLA（请求/秒）
            batch_size: 批处理大小
            task_type: 任务类型
            
        Returns:
            完整的调度请求字典
        """
        # 根据SLA选择推荐策略
        if sla_latency is not None:
            config = self.recommend_for_latency(sla_latency, batch_size)
        elif sla_throughput is not None:
            config = self.recommend_for_throughput(sla_throughput, batch_size)
        else:
            config = self.recommend_cost_effective(batch_size)
        
        if config is None:
            # 使用默认配置
            config = ResourceConfig(
                cube=10, vector=20, memory=8,
                expected_latency=0.3, expected_throughput=3.0, confidence=0.7
            )
        
        # 转换为调度器请求格式（比例形式）
        cube_req_ratio = config.cube / 20  # 910B3有20个Cube核心
        cube_lim_ratio = min(config.cube * 1.2 / 20, 1.0)  # limit比request稍高
        vector_req_ratio = config.vector / 40  # 910B3有40个Vector核心
        vector_lim_ratio = min(config.vector * 1.2 / 40, 1.0)
        
        return {
            'num': 1,
            'cube_requests': round(cube_req_ratio, 2),
            'cube_limits': round(cube_lim_ratio, 2),
            'vector_requests': round(vector_req_ratio, 2),
            'vector_limits': round(vector_lim_ratio, 2),
            'memory': [config.memory],
            'type': task_type,
            'service_name': service_name,
            'image': '',
            'COMMAND': '',
            'MODEL_PATH': model_path,
            'expected_latency': config.expected_latency,
            'expected_throughput': config.expected_throughput,
            'recommendation_confidence': config.confidence
        }


# 全局推荐引擎实例（单例模式）
_recommender_instance = None

def get_recommender(profiling_path: str = None) -> ResourceRecommender:
    """获取全局推荐引擎实例"""
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = ResourceRecommender(profiling_path)
    return _recommender_instance


def reset_recommender():
    """重置推荐引擎（用于重新加载数据）"""
    global _recommender_instance
    _recommender_instance = None


if __name__ == '__main__':
    # 测试代码
    print("Testing Resource Recommender...")
    
    recommender = ResourceRecommender()
    
    # 测试1：根据延迟推荐
    print("\n=== Test 1: Recommend for latency target 0.2s ===")
    config = recommender.recommend_for_latency(0.2, batch_size=1)
    if config:
        print(f"Recommended: Cube={config.cube}, Vector={config.vector}")
        print(f"Expected latency: {config.expected_latency}s, throughput: {config.expected_throughput}")
    
    # 测试2：根据吞吐推荐
    print("\n=== Test 2: Recommend for throughput target 5.0 ===")
    config = recommender.recommend_for_throughput(5.0, batch_size=1)
    if config:
        print(f"Recommended: Cube={config.cube}, Vector={config.vector}")
        print(f"Expected latency: {config.expected_latency}s, throughput: {config.expected_throughput}")
    
    # 测试3：性价比推荐
    print("\n=== Test 3: Cost-effective recommendation ===")
    config = recommender.recommend_cost_effective(batch_size=1)
    print(f"Recommended: Cube={config.cube}, Vector={config.vector}")
    print(f"Expected latency: {config.expected_latency}s, throughput: {config.expected_throughput}")
    
    # 测试4：扩容建议
    print("\n=== Test 4: Scaling advice ===")
    advice = recommender.get_scaling_advice(
        current_cube=8, current_vector=20,
        current_latency=0.5, target_latency=0.2,
        batch_size=1
    )
    print(f"Action: {advice['action']}")
    print(f"Reason: {advice['reason']}")
    if 'recommendation' in advice and advice['recommendation']:
        rec = advice['recommendation']
        print(f"Recommendation: Cube={rec.cube}, Vector={rec.vector}")
    
    # 测试5：生成调度请求
    print("\n=== Test 5: Generate schedule request ===")
    request = recommender.generate_schedule_request(
        service_name='test-service',
        model_path='/vllm-workspace/models/Qwen3-4B',
        sla_latency=0.2,
        batch_size=1
    )
    print(f"Schedule request: {json.dumps(request, indent=2)}")
    
    print("\nAll tests completed!")
