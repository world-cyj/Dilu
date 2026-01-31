# -*- coding: utf-8 -*-
"""
预测性扩缩容模块：基于负载预测和性能趋势自动调整资源配置
实现智能的提前扩缩容，避免性能抖动
"""
import os
import sys
import json
import time
import math
import threading
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from collections import deque
import requests

# 确保 scheduling 在 path 中
sched_dir = os.path.dirname(os.path.abspath(__file__))
if sched_dir not in sys.path:
    sys.path.insert(0, sched_dir)


@dataclass
class MetricPoint:
    """单个指标数据点"""
    timestamp: float
    value: float
    
    def to_dict(self):
        return {"timestamp": self.timestamp, "value": self.value}


@dataclass
class ScalingDecision:
    """扩缩容决策"""
    action: str  # 'scale_up', 'scale_down', 'maintain'
    reason: str
    confidence: float
    predicted_load: float
    current_config: Dict
    recommended_config: Optional[Dict]
    timestamp: float


class LoadPredictor:
    """
    负载预测器
    
    使用多种算法预测未来负载：
    1. 移动平均（简单/指数）
    2. 线性回归
    3. 趋势分析
    """
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.metrics: Dict[str, deque] = {}  # metric_name -> deque of MetricPoint
        self._lock = threading.Lock()
    
    def add_metric(self, metric_name: str, value: float, timestamp: float = None):
        """添加指标数据点"""
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = deque(maxlen=self.history_size)
            self.metrics[metric_name].append(MetricPoint(timestamp, value))
    
    def get_history(self, metric_name: str) -> List[MetricPoint]:
        """获取指标历史数据"""
        with self._lock:
            if metric_name not in self.metrics:
                return []
            return list(self.metrics[metric_name])
    
    def predict_simple_ma(self, metric_name: str, window: int = 10) -> Optional[float]:
        """简单移动平均预测"""
        history = self.get_history(metric_name)
        if len(history) < window:
            return None
        
        recent = history[-window:]
        return sum(p.value for p in recent) / len(recent)
    
    def predict_ema(self, metric_name: str, alpha: float = 0.3) -> Optional[float]:
        """指数移动平均预测"""
        history = self.get_history(metric_name)
        if not history:
            return None
        
        ema = history[0].value
        for point in history[1:]:
            ema = alpha * point.value + (1 - alpha) * ema
        return ema
    
    def predict_linear_trend(self, metric_name: str) -> Optional[Tuple[float, float]]:
        """
        线性趋势预测
        
        Returns:
            (predicted_value, trend_slope)
        """
        history = self.get_history(metric_name)
        if len(history) < 5:
            return None
        
        # 使用最近的数据点
        n = min(len(history), 20)
        recent = history[-n:]
        
        # 计算线性回归
        x_vals = [i for i in range(n)]
        y_vals = [p.value for p in recent]
        
        x_mean = sum(x_vals) / n
        y_mean = sum(y_vals) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)
        
        if denominator == 0:
            return y_mean, 0
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # 预测下一个值
        next_x = n
        prediction = slope * next_x + intercept
        
        return prediction, slope
    
    def predict_with_confidence(self, metric_name: str) -> Optional[Dict]:
        """
        综合预测，返回带置信度的结果
        """
        history = self.get_history(metric_name)
        if len(history) < 10:
            return None
        
        # 多种预测方法
        sma = self.predict_simple_ma(metric_name, window=10)
        ema = self.predict_ema(metric_name, alpha=0.3)
        trend_result = self.predict_linear_trend(metric_name)
        
        if trend_result:
            trend_pred, slope = trend_result
        else:
            trend_pred = sma
            slope = 0
        
        # 综合预测（加权平均）
        weights = {"sma": 0.3, "ema": 0.4, "trend": 0.3}
        prediction = (
            weights["sma"] * sma +
            weights["ema"] * ema +
            weights["trend"] * trend_pred
        )
        
        # 计算置信度（基于历史数据的方差）
        values = [p.value for p in history[-20:]]
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        std_dev = math.sqrt(variance)
        
        # 方差越小，置信度越高
        confidence = max(0.5, 1.0 - (std_dev / mean_val if mean_val > 0 else 0))
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "trend": slope,
            "sma": sma,
            "ema": ema,
            "trend_pred": trend_pred,
        }


class PredictiveScaler:
    """
    预测性扩缩容控制器
    
    功能：
    1. 持续监控服务性能指标
    2. 预测未来负载趋势
    3. 提前执行扩缩容决策
    4. 避免性能抖动和过度反应
    """
    
    def __init__(
        self,
        scheduler_url: str = "http://127.0.0.1:5000",
        check_interval: int = 30,
        prediction_window: int = 60,
    ):
        """
        初始化预测性扩缩容器
        
        Args:
            scheduler_url: 调度器API地址
            check_interval: 检查间隔（秒）
            prediction_window: 预测时间窗口（秒）
        """
        self.scheduler_url = scheduler_url
        self.check_interval = check_interval
        self.prediction_window = prediction_window
        
        self.predictor = LoadPredictor(history_size=200)
        self.monitored_services: Dict[str, Dict] = {}  # service_name -> config
        self.decision_history: deque = deque(maxlen=100)
        self._lock = threading.Lock()
        self._running = False
        self._monitor_thread = None
        
        # 扩缩容配置
        self.scale_up_threshold = 0.8  # 预测负载超过80%时扩容
        self.scale_down_threshold = 0.3  # 预测负载低于30%时缩容
        self.confidence_threshold = 0.7  # 置信度阈值
        self.cooldown_period = 120  # 扩缩容冷却期（秒）
        self.last_scale_time: Dict[str, float] = {}
    
    def register_service(
        self,
        service_name: str,
        instance_id: str,
        current_config: Dict,
        sla_latency: float = 1.0,
        sla_throughput: float = None,
    ):
        """
        注册服务进行监控
        
        Args:
            service_name: 服务名称
            instance_id: 实例ID
            current_config: 当前资源配置
            sla_latency: 延迟SLA
            sla_throughput: 吞吐SLA
        """
        with self._lock:
            self.monitored_services[service_name] = {
                "instance_id": instance_id,
                "current_config": current_config,
                "sla_latency": sla_latency,
                "sla_throughput": sla_throughput,
                "registered_at": time.time(),
            }
        print(f"[PredictiveScaler] Registered service: {service_name}")
    
    def unregister_service(self, service_name: str):
        """取消服务监控"""
        with self._lock:
            if service_name in self.monitored_services:
                del self.monitored_services[service_name]
        print(f"[PredictiveScaler] Unregistered service: {service_name}")
    
    def start(self):
        """启动监控线程"""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        print("[PredictiveScaler] Started monitoring loop")
    
    def stop(self):
        """停止监控线程"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        print("[PredictiveScaler] Stopped monitoring loop")
    
    def _monitor_loop(self):
        """监控主循环"""
        while self._running:
            try:
                self._check_all_services()
            except Exception as e:
                print(f"[PredictiveScaler] Error in monitor loop: {e}")
                traceback.print_exc()
            
            time.sleep(self.check_interval)
    
    def _check_all_services(self):
        """检查所有监控的服务"""
        with self._lock:
            services = dict(self.monitored_services)
        
        for service_name, config in services.items():
            try:
                self._check_service(service_name, config)
            except Exception as e:
                print(f"[PredictiveScaler] Error checking {service_name}: {e}")
    
    def _check_service(self, service_name: str, config: Dict):
        """检查单个服务"""
        instance_id = config["instance_id"]
        
        # 获取性能指标
        metrics = self._collect_metrics(instance_id)
        if not metrics:
            return
        
        # 添加到预测器
        for metric_name, value in metrics.items():
            self.predictor.add_metric(f"{service_name}/{metric_name}", value)
        
        # 预测未来负载
        latency_pred = self.predictor.predict_with_confidence(
            f"{service_name}/latency"
        )
        throughput_pred = self.predictor.predict_with_confidence(
            f"{service_name}/throughput"
        )
        
        if not latency_pred:
            return
        
        # 做出扩缩容决策
        decision = self._make_decision(
            service_name=service_name,
            config=config,
            current_metrics=metrics,
            predictions={
                "latency": latency_pred,
                "throughput": throughput_pred,
            }
        )
        
        if decision:
            self._execute_decision(service_name, decision)
    
    def _collect_metrics(self, instance_id: str) -> Optional[Dict]:
        """收集服务性能指标"""
        try:
            # 获取实例信息
            r = requests.get(
                f"{self.scheduler_url}/instance/{instance_id}",
                timeout=5
            )
            if r.status_code != 200:
                return None
            
            instance = r.json()
            
            # 从健康检查获取性能数据
            port = instance.get("port")
            if not port:
                return None
            
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
            if r.status_code != 200:
                return None
            
            health = r.json()
            
            # 构造指标数据
            metrics = {
                "latency": health.get("avg_latency", 0.5),
                "throughput": health.get("rps", 0),
                "memory_usage": health.get("memory_gb", 0),
            }
            
            return metrics
            
        except Exception as e:
            return None
    
    def _make_decision(
        self,
        service_name: str,
        config: Dict,
        current_metrics: Dict,
        predictions: Dict,
    ) -> Optional[ScalingDecision]:
        """
        做出扩缩容决策
        """
        current_config = config["current_config"]
        sla_latency = config["sla_latency"]
        
        latency_pred = predictions.get("latency", {})
        predicted_latency = latency_pred.get("prediction", current_metrics.get("latency", 0.5))
        confidence = latency_pred.get("confidence", 0.5)
        trend = latency_pred.get("trend", 0)
        
        # 检查冷却期
        last_scale = self.last_scale_time.get(service_name, 0)
        if time.time() - last_scale < self.cooldown_period:
            return None
        
        # 置信度不足时不做决策
        if confidence < self.confidence_threshold:
            return None
        
        # 计算负载比例
        load_ratio = predicted_latency / sla_latency if sla_latency > 0 else 0.5
        
        action = "maintain"
        reason = ""
        recommended_config = None
        
        # 判断是否需要扩容
        if load_ratio > self.scale_up_threshold or trend > 0.1:
            action = "scale_up"
            reason = f"Predicted latency {predicted_latency:.3f}s exceeds SLA {sla_latency}s (load ratio: {load_ratio:.2f})"
            recommended_config = self._calculate_scale_up_config(current_config)
        
        # 判断是否需要缩容
        elif load_ratio < self.scale_down_threshold and trend < -0.05:
            action = "scale_down"
            reason = f"Predicted load ratio {load_ratio:.2f} below threshold {self.scale_down_threshold}"
            recommended_config = self._calculate_scale_down_config(current_config)
        
        else:
            reason = f"Current load ratio {load_ratio:.2f} within normal range"
        
        return ScalingDecision(
            action=action,
            reason=reason,
            confidence=confidence,
            predicted_load=load_ratio,
            current_config=current_config,
            recommended_config=recommended_config,
            timestamp=time.time(),
        )
    
    def _calculate_scale_up_config(self, current_config: Dict) -> Dict:
        """计算扩容后的配置"""
        new_config = dict(current_config)
        
        # 增加资源分配
        current_cube = current_config.get("cube", 8)
        current_vector = current_config.get("vector", 20)
        
        new_config["cube"] = min(current_cube + 4, 20)
        new_config["vector"] = min(current_vector + 10, 40)
        new_config["scale_reason"] = "predictive_scale_up"
        
        return new_config
    
    def _calculate_scale_down_config(self, current_config: Dict) -> Dict:
        """计算缩容后的配置"""
        new_config = dict(current_config)
        
        # 减少资源分配
        current_cube = current_config.get("cube", 8)
        current_vector = current_config.get("vector", 20)
        
        new_config["cube"] = max(current_cube - 4, 4)
        new_config["vector"] = max(current_vector - 10, 10)
        new_config["scale_reason"] = "predictive_scale_down"
        
        return new_config
    
    def _execute_decision(self, service_name: str, decision: ScalingDecision):
        """执行扩缩容决策"""
        print(f"[PredictiveScaler] Executing {decision.action} for {service_name}")
        print(f"  Reason: {decision.reason}")
        print(f"  Confidence: {decision.confidence:.2f}")
        
        if decision.action == "maintain":
            return
        
        with self._lock:
            config = self.monitored_services.get(service_name)
        
        if not config:
            return
        
        instance_id = config["instance_id"]
        
        try:
            if decision.action == "scale_up":
                self._do_reschedule(instance_id, decision.recommended_config)
            elif decision.action == "scale_down":
                self._do_reschedule(instance_id, decision.recommended_config)
            
            # 更新最后扩缩容时间
            self.last_scale_time[service_name] = time.time()
            
            # 记录决策
            self.decision_history.append({
                "service_name": service_name,
                "decision": decision,
                "timestamp": time.time(),
            })
            
            # 更新服务配置
            with self._lock:
                if service_name in self.monitored_services:
                    self.monitored_services[service_name]["current_config"] = decision.recommended_config
            
        except Exception as e:
            print(f"[PredictiveScaler] Error executing decision: {e}")
    
    def _do_reschedule(self, instance_id: str, new_config: Dict):
        """执行重新调度（垂直扩缩容）"""
        try:
            payload = {
                "instance_id": instance_id,
                "new_cube_requests": new_config.get("cube", 8) / 20,
                "new_cube_limits": min(new_config.get("cube", 8) * 1.2 / 20, 1.0),
                "new_vector_requests": new_config.get("vector", 20) / 40,
                "new_vector_limits": min(new_config.get("vector", 20) * 1.2 / 40, 1.0),
            }
            
            r = requests.post(
                f"{self.scheduler_url}/reschedule_with_limits",
                json=payload,
                timeout=30
            )
            
            if r.status_code == 200:
                print(f"[PredictiveScaler] Reschedule successful: {r.json()}")
            else:
                print(f"[PredictiveScaler] Reschedule failed: {r.status_code} - {r.text}")
                
        except Exception as e:
            print(f"[PredictiveScaler] Error rescheduling: {e}")
    
    def get_status(self) -> Dict:
        """获取扩缩容控制器状态"""
        with self._lock:
            return {
                "running": self._running,
                "monitored_services": len(self.monitored_services),
                "service_list": list(self.monitored_services.keys()),
                "decision_count": len(self.decision_history),
                "recent_decisions": list(self.decision_history)[-10:],
            }
    
    def get_prediction(self, service_name: str, metric_name: str) -> Optional[Dict]:
        """获取特定服务的预测结果"""
        return self.predictor.predict_with_confidence(f"{service_name}/{metric_name}")


# 全局扩缩容控制器实例
_scaler_instance: Optional[PredictiveScaler] = None


def get_predictive_scaler(scheduler_url: str = "http://127.0.0.1:5000") -> PredictiveScaler:
    """获取全局预测性扩缩容控制器"""
    global _scaler_instance
    if _scaler_instance is None:
        _scaler_instance = PredictiveScaler(scheduler_url=scheduler_url)
    return _scaler_instance


if __name__ == "__main__":
    # 测试代码
    print("Testing Predictive Scaler...")
    
    scaler = PredictiveScaler()
    
    # 模拟添加一些指标数据
    for i in range(50):
        # 模拟逐渐增加的延迟
        latency = 0.3 + i * 0.01 + (i % 5) * 0.02
        scaler.predictor.add_metric("test/latency", latency)
    
    # 获取预测
    pred = scaler.predictor.predict_with_confidence("test/latency")
    print(f"\nPrediction: {pred}")
    
    # 测试趋势预测
    trend = scaler.predictor.predict_linear_trend("test/latency")
    print(f"Trend: {trend}")
    
    print("\nTest completed!")
