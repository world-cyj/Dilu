# -*- coding: utf-8 -*-
"""
NPU Horizontal Scaling Scaler
基于NPU资源模型的自动扩缩容组件
支持2D协同缩放（垂直+水平）和基于资源画像的智能决策
"""
import threading
import time
import requests
import json
import sys
import os
from flask import Flask, request, jsonify
from datetime import datetime
import logging

# 添加 scheduling 目录到 path
scheduling_dir = os.path.join(os.path.dirname(__file__), "..", "..", "scheduling")
if scheduling_dir not in sys.path:
    sys.path.insert(0, scheduling_dir)

# 尝试导入资源推荐引擎
try:
    from resource_recommender import get_recommender, ResourceRecommender
    _recommender_available = True
except ImportError as e:
    print(f"[ScalerNPU] Warning: Resource recommender not available: {e}")
    _recommender_available = False

# 配置日志
logging.basicConfig(
    filename='logs/dilu-scaler-npu.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d/%b/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.logger.disabled = True

# 全局状态
services_status = {}
services_dict = {}
lock = threading.Lock()
scheduler_url = 'http://localhost:5000'

# NPU资源常量
NPU_CUBE_CORES_PER_DEVICE = 20
NPU_VECTOR_CORES_PER_DEVICE = 40
NPU_MEMORY_GB_PER_DEVICE = 64


class ServiceManager:
    """服务管理器"""
    
    def register_service(self, service_data):
        """注册服务"""
        global services_dict, services_status
        service_id = service_data['service_name']
        
        with lock:
            if service_id in services_dict:
                return "Service already registered", 409
            
            # 创建NPU服务实例
            new_service = NPUService(
                service_id=service_id,
                image=service_data.get('image', ''),
                num_npus=service_data.get('num_npus', 1),
                cube_requests=service_data.get('cube_requests', 0.25),
                cube_limits=service_data.get('cube_limits', 0.75),
                vector_requests=service_data.get('vector_requests', 0.25),
                vector_limits=service_data.get('vector_limits', 0.75),
                memory=service_data.get('memory', [8]),
                is_llm=service_data.get('is_llm', 0),
                priority=service_data.get('priority', 0),
                task_type=service_data.get('task_type', 'inference'),
                throughput=service_data.get('throughput', 10),
                commands=service_data.get('commands', ''),
                model_path=service_data.get('model_path', ''),
                sla_latency=service_data.get('sla_latency'),
                sla_throughput=service_data.get('sla_throughput')
            )
            
            services_dict[service_id] = new_service
            services_status[service_id] = {
                "requests": 0,
                "total_latency": 0,
                "request_count": 0
            }
        
        logger.info(f"Service {service_id} registered successfully")
        return f"Service {service_id} registered successfully", 200

    def delete_service(self, service_data):
        """删除服务"""
        global services_dict, services_status
        service_id = service_data['service_name']
        
        with lock:
            if service_id in services_dict:
                # 清理所有实例
                service = services_dict[service_id]
                for instance in service.instances:
                    service._delete_instance(instance['instance_id'])
                
                del services_dict[service_id]
                del services_status[service_id]
                logger.info(f"Service {service_id} deleted")
                return f"Service {service_id} deleted successfully", 200
            else:
                return "Service does not exist", 404


class NPUService:
    """NPU服务类"""
    
    def __init__(self, service_id, image, num_npus, cube_requests, cube_limits,
                 vector_requests, vector_limits, memory, is_llm, priority,
                 task_type, throughput, commands, model_path='',
                 sla_latency=None, sla_throughput=None):
        self.service_id = service_id
        self.image = image
        self.num_npus = num_npus
        
        # NPU资源配额
        self.cube_requests = cube_requests
        self.cube_limits = cube_limits
        self.vector_requests = vector_requests
        self.vector_limits = vector_limits
        self.memory = memory if isinstance(memory, list) else [memory]
        
        self.is_llm = is_llm
        self.priority = priority
        self.task_type = task_type
        self.commands = commands
        self.model_path = model_path
        self.throughput = throughput
        
        # SLA要求
        self.sla_latency = sla_latency
        self.sla_throughput = sla_throughput
        
        # 实例管理
        self.instances = []
        self.round_robin_index = 0
        self.lock = threading.Lock()
        
        # 性能监控
        self.performance_history = []
        self.max_history_length = 100
        
        # 初始扩容
        self.scale_out()
    
    def scale_out(self, use_sla=True):
        """水平扩容（添加实例）"""
        # 如果使用SLA且推荐引擎可用，获取智能推荐
        if use_sla and _recommender_available and (self.sla_latency or self.sla_throughput):
            try:
                recommender = get_recommender()
                config = recommender.recommend_for_latency(
                    target_latency=self.sla_latency,
                    batch_size=1
                ) if self.sla_latency else recommender.recommend_for_throughput(
                    target_throughput=self.sla_throughput,
                    batch_size=1
                )
                
                if config:
                    # 使用推荐配置
                    cube_req_ratio = config.cube / NPU_CUBE_CORES_PER_DEVICE
                    cube_lim_ratio = min(config.cube * 1.2 / NPU_CUBE_CORES_PER_DEVICE, 1.0)
                    vector_req_ratio = config.vector / NPU_VECTOR_CORES_PER_DEVICE
                    vector_lim_ratio = min(config.vector * 1.2 / NPU_VECTOR_CORES_PER_DEVICE, 1.0)
                    
                    logger.info(f"Using recommended config for {self.service_id}: "
                              f"Cube={config.cube}/{NPU_CUBE_CORES_PER_DEVICE}, "
                              f"Vector={config.vector}/{NPU_VECTOR_CORES_PER_DEVICE}")
                else:
                    # 使用默认配置
                    cube_req_ratio = self.cube_requests
                    cube_lim_ratio = self.cube_limits
                    vector_req_ratio = self.vector_requests
                    vector_lim_ratio = self.vector_limits
            except Exception as e:
                logger.warning(f"Failed to get recommendation: {e}, using default config")
                cube_req_ratio = self.cube_requests
                cube_lim_ratio = self.cube_limits
                vector_req_ratio = self.vector_requests
                vector_lim_ratio = self.vector_limits
        else:
            # 使用配置值
            cube_req_ratio = self.cube_requests
            cube_lim_ratio = self.cube_limits
            vector_req_ratio = self.vector_requests
            vector_lim_ratio = self.vector_limits
        
        # 调用调度器
        url = f"{scheduler_url}/schedule"
        data = {
            'num': self.num_npus,
            'cube_requests': cube_req_ratio,
            'cube_limits': cube_lim_ratio,
            'vector_requests': vector_req_ratio,
            'vector_limits': vector_lim_ratio,
            'memory': self.memory,
            'is_llm': self.is_llm,
            'priority': self.priority,
            'type': self.task_type,
            'service_name': self.service_id,
            'image': self.image,
            'COMMAND': self.commands,
            'MODEL_PATH': self.model_path
        }
        
        try:
            response = requests.post(url, json=data, timeout=30)
            if response.status_code == 200:
                resp_content = response.json()
                instance_id = resp_content.get('instance_id')
                selected_npus = resp_content.get('selected_npus', [])
                ip_address = selected_npus[0]['ip'] if selected_npus else '127.0.0.1'
                port = resp_content.get('port')
                
                instance = {
                    "instance_id": instance_id,
                    "ip_address": ip_address,
                    "port": port,
                    "is_ready": False,
                    "cube_limits": cube_lim_ratio * NPU_CUBE_CORES_PER_DEVICE,
                    "vector_limits": vector_lim_ratio * NPU_VECTOR_CORES_PER_DEVICE
                }
                
                with self.lock:
                    self.instances.append(instance)
                
                current_time = datetime.now().strftime("%d/%b/%Y %H:%M:%S")
                msg = (f"Instance deployed: {instance_id}, "
                       f"Cube={cube_lim_ratio * NPU_CUBE_CORES_PER_DEVICE:.1f}, "
                       f"Vector={vector_lim_ratio * NPU_VECTOR_CORES_PER_DEVICE:.1f}, "
                       f"current_time: {current_time}, "
                       f"current_instance_counts: {len(self.instances)}")
                print(msg)
                logger.info(msg)
                
                # 启动健康检查线程
                if "training" not in self.service_id:
                    threading.Thread(target=self.check_instance_readiness, 
                                   args=(instance,), daemon=True).start()
                
                return True
            else:
                logger.error(f'Instance deploy failed: {response.content}')
                return False
        except Exception as e:
            logger.error(f'Scale out failed: {e}')
            return False
    
    def scale_in(self):
        """水平缩容（删除实例）"""
        with self.lock:
            if len(self.instances) <= 1:
                logger.warning(f"Cannot scale in {self.service_id}: only 1 instance left")
                return False
            
            instance_to_remove = self.instances.pop()
        
        return self._delete_instance(instance_to_remove['instance_id'])
    
    def _delete_instance(self, instance_id):
        """删除指定实例"""
        url = f"{scheduler_url}/delete_instance"
        data = {'instance_id': instance_id}
        
        try:
            response = requests.post(url, json=data, timeout=30)
            if response.status_code == 200:
                current_time = datetime.now().strftime("%d/%b/%Y %H:%M:%S")
                msg = (f"Instance deleted: {instance_id}, "
                       f"current_time: {current_time}, "
                       f"current_instance_counts: {len(self.instances)}")
                print(msg)
                logger.info(msg)
                return True
            else:
                logger.error(f"Instance delete failed: {response.content}")
                return False
        except Exception as e:
            logger.error(f"Delete instance failed: {e}")
            return False
    
    def vertical_scale(self, new_cube_limits, new_vector_limits):
        """
        垂直缩放（调整资源限制）
        
        Args:
            new_cube_limits: 新的Cube核心限制（比例 0-1）
            new_vector_limits: 新的Vector核心限制（比例 0-1）
        """
        url = f"{scheduler_url}/reschedule_with_limits"
        
        with self.lock:
            for instance in self.instances:
                data = {
                    'instance_id': instance['instance_id'],
                    'new_cube_limits': new_cube_limits,
                    'new_vector_limits': new_vector_limits
                }
                
                try:
                    response = requests.post(url, json=data, timeout=30)
                    if response.status_code == 200:
                        instance['cube_limits'] = new_cube_limits * NPU_CUBE_CORES_PER_DEVICE
                        instance['vector_limits'] = new_vector_limits * NPU_VECTOR_CORES_PER_DEVICE
                        logger.info(f"Vertically scaled {instance['instance_id']}: "
                                  f"Cube={new_cube_limits}, Vector={new_vector_limits}")
                    else:
                        logger.error(f"Vertical scale failed: {response.content}")
                except Exception as e:
                    logger.error(f"Vertical scale request failed: {e}")
    
    def check_instance_readiness(self, instance):
        """检查实例是否就绪"""
        max_retries = 120  # 最多等待120秒
        retry_interval = 1
        
        for _ in range(max_retries):
            try:
                response = requests.get(
                    f"http://{instance['ip_address']}:{instance['port']}/health",
                    timeout=5
                )
                
                if response.status_code == 200:
                    with self.lock:
                        instance['is_ready'] = True
                    logger.info(f"Instance {instance['instance_id']} is ready")
                    return True
            except requests.RequestException:
                pass
            
            time.sleep(retry_interval)
        
        logger.warning(f"Instance {instance['instance_id']} readiness check timeout")
        return False
    
    def dispatch(self, request):
        """分发请求到实例"""
        post_data = request.get_json()
        if not post_data:
            return jsonify({'error': 'No data provided'}), 400
        
        with self.lock:
            ready_instances = [inst for inst in self.instances if inst.get('is_ready')]
        
        if not ready_instances:
            return jsonify({'error': 'No ready instances available'}), 503
        
        # 轮询选择实例
        self.round_robin_index = (self.round_robin_index + 1) % len(ready_instances)
        instance = ready_instances[self.round_robin_index]
        
        ip = instance['ip_address']
        port = instance['port']
        url = f'http://{ip}:{port}/predict'
        
        try:
            start_time = time.time()
            response = requests.post(url, json=post_data, timeout=60)
            latency = time.time() - start_time
            
            # 记录性能数据
            self._record_performance(latency, response.status_code == 200)
            
            if response.status_code == 200:
                return jsonify(response.json()), 200
            else:
                return jsonify({'error': 'Failed to process prediction', 
                              'details': response.text}), response.status_code
        except requests.exceptions.RequestException as e:
            return jsonify({'error': 'Network error', 'details': str(e)}), 500
    
    def _record_performance(self, latency, success):
        """记录性能数据"""
        with self.lock:
            self.performance_history.append({
                'timestamp': datetime.now().isoformat(),
                'latency': latency,
                'success': success
            })
            
            # 限制历史记录长度
            if len(self.performance_history) > self.max_history_length:
                self.performance_history = self.performance_history[-self.max_history_length:]
    
    def get_average_latency(self, window=10):
        """获取最近窗口的平均延迟"""
        with self.lock:
            if not self.performance_history:
                return None
            
            recent = self.performance_history[-window:]
            latencies = [p['latency'] for p in recent if p['success']]
            
            if not latencies:
                return None
            
            return sum(latencies) / len(latencies)
    
    def get_scaling_advice(self):
        """获取扩容/缩容建议"""
        if not _recommender_available or not self.sla_latency:
            return None
        
        avg_latency = self.get_average_latency()
        if avg_latency is None:
            return None
        
        try:
            recommender = get_recommender()
            
            # 获取当前配置
            current_cube = self.instances[0]['cube_limits'] if self.instances else 10
            current_vector = self.instances[0]['vector_limits'] if self.instances else 20
            
            advice = recommender.get_scaling_advice(
                current_cube=int(current_cube),
                current_vector=int(current_vector),
                current_latency=avg_latency,
                target_latency=self.sla_latency,
                batch_size=1
            )
            
            return advice
        except Exception as e:
            logger.error(f"Failed to get scaling advice: {e}")
            return None


class NPUScaler(threading.Thread):
    """NPU自动扩缩容器"""
    
    def __init__(self, check_interval=1, scale_out_threshold=15, 
                 scale_in_threshold=15, history_length=20,
                 enable_vertical_scaling=True, enable_intelligent_scaling=True):
        super().__init__()
        self.check_interval = check_interval
        self.scale_out_threshold = scale_out_threshold
        self.scale_in_threshold = scale_in_threshold
        self.history_length = history_length
        self.enable_vertical_scaling = enable_vertical_scaling
        self.enable_intelligent_scaling = enable_intelligent_scaling
        
        self.request_history = {}
        self.daemon = True
    
    def run(self):
        """主循环"""
        global services_status, services_dict
        
        while True:
            time.sleep(self.check_interval)
            
            for service_id, status in services_status.items():
                if service_id not in services_dict:
                    continue
                
                service = services_dict[service_id]
                
                # 跳过训练任务
                if "training" in service_id:
                    continue
                
                # 计算当前吞吐能力
                num_instances = len(service.instances)
                max_throughput = num_instances * service.throughput
                scale_in_throughput = (num_instances - 1) * service.throughput if num_instances > 1 else 0
                
                current_time = datetime.now().strftime("%d/%b/%Y %H:%M:%S")
                avg_latency = service.get_average_latency()
                latency_str = f"{avg_latency:.3f}s" if avg_latency else "N/A"
                
                print(f"{service_id}, time: {current_time}, "
                      f"Requests: {status['requests']}, "
                      f"Max Throughput: {max_throughput}, "
                      f"Avg Latency: {latency_str}")
                
                # 更新请求历史
                if service_id not in self.request_history:
                    self.request_history[service_id] = []
                
                if len(self.request_history[service_id]) >= self.history_length:
                    self.request_history[service_id].pop(0)
                
                self.request_history[service_id].append(status['requests'])
                status['requests'] = 0
                
                # 智能缩放决策
                if self.enable_intelligent_scaling and _recommender_available:
                    self._intelligent_scaling_decision(service, service_id)
                else:
                    # 基础阈值缩放
                    self._threshold_based_scaling(service, service_id, 
                                                 max_throughput, scale_in_throughput)
    
    def _intelligent_scaling_decision(self, service, service_id):
        """基于资源画像的智能缩放决策"""
        advice = service.get_scaling_advice()
        
        if not advice:
            return
        
        action = advice.get('action')
        
        if action == 'scale_up':
            # 先尝试垂直扩容
            if self.enable_vertical_scaling and 'recommendation' in advice:
                rec = advice['recommendation']
                new_cube = rec['cube'] / NPU_CUBE_CORES_PER_DEVICE
                new_vector = rec['vector'] / NPU_VECTOR_CORES_PER_DEVICE
                
                # 如果当前配置已经低于推荐，先垂直扩容
                current_cube_ratio = service.instances[0]['cube_limits'] / NPU_CUBE_CORES_PER_DEVICE if service.instances else 0
                if new_cube > current_cube_ratio * 1.2:  # 如果推荐配置明显更高
                    service.vertical_scale(new_cube, new_vector)
                    logger.info(f"Vertically scaled {service_id} based on recommendation")
                    return
            
            # 否则水平扩容
            if self._should_scale_out(service_id):
                service.scale_out()
                self.request_history[service_id].clear()
        
        elif action == 'scale_down' and len(service.instances) > 1:
            # 先尝试垂直缩容
            if self.enable_vertical_scaling and 'recommendation' in advice:
                rec = advice['recommendation']
                new_cube = rec['cube'] / NPU_CUBE_CORES_PER_DEVICE
                new_vector = rec['vector'] / NPU_VECTOR_CORES_PER_DEVICE
                
                current_cube_ratio = service.instances[0]['cube_limits'] / NPU_CUBE_CORES_PER_DEVICE if service.instances else 1
                if new_cube < current_cube_ratio * 0.8:  # 如果推荐配置明显更低
                    service.vertical_scale(new_cube, new_vector)
                    logger.info(f"Vertically scaled down {service_id} based on recommendation")
                    return
            
            # 否则水平缩容
            if self._should_scale_in(service_id):
                service.scale_in()
                self.request_history[service_id].clear()
    
    def _threshold_based_scaling(self, service, service_id, max_throughput, scale_in_throughput):
        """基于阈值的缩放决策"""
        if self._should_scale_out(service_id, max_throughput):
            service.scale_out()
            self.request_history[service_id].clear()
        
        if self._should_scale_in(service_id, scale_in_throughput) and len(service.instances) > 1:
            service.scale_in()
            self.request_history[service_id].clear()
    
    def _should_scale_out(self, service_id, max_throughput):
        """判断是否应该扩容"""
        if service_id not in self.request_history:
            return False
        
        over_threshold_times = sum(1 for x in self.request_history[service_id] 
                                  if x > max_throughput)
        return over_threshold_times >= self.scale_out_threshold
    
    def _should_scale_in(self, service_id, scale_in_throughput):
        """判断是否应该缩容"""
        if service_id not in self.request_history:
            return False
        
        under_threshold_times = sum(1 for x in self.request_history[service_id] 
                                   if x < scale_in_throughput)
        return under_threshold_times >= self.scale_in_threshold


# Flask路由
service_manager = ServiceManager()

@app.route('/register_service', methods=['POST'])
def register_service():
    """注册服务"""
    service_data = request.get_json()
    message, status = service_manager.register_service(service_data)
    return jsonify({'message': message}), status


@app.route('/delete_service', methods=['POST'])
def delete_service():
    """删除服务"""
    service_data = request.get_json()
    message, status = service_manager.delete_service(service_data)
    return jsonify({'message': message}), status


@app.route('/<service_id>', methods=['POST'])
def handle_predict(service_id):
    """处理预测请求"""
    if service_id in services_dict:
        with lock:
            services_status[service_id]['requests'] += 1
        service = services_dict[service_id]
        return service.dispatch(request)
    else:
        return jsonify({'error': 'Service not found'}), 404


@app.route('/service/<service_id>/status', methods=['GET'])
def get_service_status(service_id):
    """获取服务状态"""
    if service_id not in services_dict:
        return jsonify({'error': 'Service not found'}), 404
    
    service = services_dict[service_id]
    
    return jsonify({
        'service_id': service_id,
        'instance_count': len(service.instances),
        'ready_instance_count': sum(1 for inst in service.instances if inst.get('is_ready')),
        'average_latency': service.get_average_latency(),
        'scaling_advice': service.get_scaling_advice()
    }), 200


@app.route('/service/<service_id>/scale', methods=['POST'])
def manual_scale(service_id):
    """手动扩缩容"""
    if service_id not in services_dict:
        return jsonify({'error': 'Service not found'}), 404
    
    data = request.get_json()
    action = data.get('action')
    
    service = services_dict[service_id]
    
    if action == 'out':
        success = service.scale_out(use_sla=data.get('use_sla', True))
        return jsonify({'success': success, 'instance_count': len(service.instances)}), 200
    elif action == 'in':
        success = service.scale_in()
        return jsonify({'success': success, 'instance_count': len(service.instances)}), 200
    elif action == 'vertical':
        cube = data.get('cube_limits')
        vector = data.get('vector_limits')
        if cube is not None and vector is not None:
            service.vertical_scale(cube, vector)
            return jsonify({'success': True}), 200
        else:
            return jsonify({'error': 'cube_limits and vector_limits required'}), 400
    else:
        return jsonify({'error': 'Invalid action'}), 400


if __name__ == "__main__":
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    
    # 启动自动扩缩容器
    scaler = NPUScaler(
        check_interval=1,
        scale_out_threshold=20,
        scale_in_threshold=30,
        history_length=40,
        enable_vertical_scaling=True,
        enable_intelligent_scaling=_recommender_available
    )
    scaler.start()
    
    print(f"[ScalerNPU] Started with intelligent_scaling={_recommender_available}")
    print(f"[ScalerNPU] Resource recommender available: {_recommender_available}")
    
    # 启动Flask服务
    app.run(debug=False, host='0.0.0.0', port=14999, threaded=True)
