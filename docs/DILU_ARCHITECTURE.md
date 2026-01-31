# Dilu 架构分层设计文档

本文档系统梳理Dilu的架构设计，为NPU侧完善提供清晰的路线图。

---

## 一、Dilu 核心设计哲学

### 1.1 核心概念：Introspective Elasticity (IE)

**内省弹性**：细粒度的二维协同扩缩容机制
- **垂直缩放（Vertical Scaling）**：调整单个实例的资源配额（SM/Cube/Vector核心）
- **水平缩放（Horizontal Scaling）**：增减实例数量
- **协同决策**：根据实时性能反馈，智能选择缩放方向

### 1.2 三大核心特性

1. **多因素画像（Multi-factor Profiling）**
   - 网格搜索不同资源配置下的性能表现
   - 高效剪枝搜索空间
   - 输出：资源配置 → 延迟/吞吐的映射表

2. **资源互补调度（Resourcing-complementary Scheduling）**
   - Best-Fit/Worst-Fit算法
   - 在QoS约束下最大化GPU利用率
   - 支持资源超售和动态调整

3. **自适应2D协同缩放（Adaptive 2D Co-scaling）**
   - 实时监控服务性能
   - 基于资源画像做智能决策
   - 垂直+水平协同执行

---

## 二、架构分层总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Layer 5: 用户接口层                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │   Demo脚本      │  │   部署脚本      │  │   仿真脚本      │              │
│  │                 │  │                 │  │                 │              │
│  │ run_demo_npu.py │  │ deploy_*.py     │  │ comp_baselines  │              │
│  │ test_resource_  │  │ submiter_*.py   │  │ service_gen_*.py│              │
│  │   recommendation│  │                 │  │                 │              │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │
└───────────┼────────────────────┼────────────────────┼───────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Layer 4: 控制平面层                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Scaler (自动扩缩容器)                            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │ 性能监控     │  │ 决策引擎     │  │ 垂直缩放     │  │ 水平缩放    │ │   │
│  │  │             │  │             │  │             │  │            │ │   │
│  │  │ 收集延迟/吞吐│  │ 基于画像决策 │  │ 调整资源限制 │  │ 增减实例    │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Scheduler (调度器)                              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │ 资源分配     │  │ 实例生命周期 │  │ 资源推荐     │  │ 集群管理    │ │   │
│  │  │             │  │             │  │             │  │            │ │   │
│  │  │ Best/Worst  │  │ 启动/停止    │  │ 基于SLA推荐  │  │ 节点管理    │ │   │
│  │  │ Fit算法     │  │ 健康检查    │  │ 最优配置    │  │ 资源统计    │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Layer 3: 资源管理层                              │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │   资源模型       │  │   资源推荐引擎   │  │   资源画像       │              │
│  │                 │  │                 │  │                 │              │
│  │ npu_resource.py │  │ resource_       │  │ profile_npu.py  │              │
│  │  - NPU类        │  │   recommender   │  │  - 网格测试      │              │
│  │  - 分配算法     │  │   .py           │  │  - 性能数据      │              │
│  │  - 资源统计     │  │  - SLA推荐      │  │  - 画像结果      │              │
│  └────────┬────────┘  └─────────────────┘  └─────────────────┘              │
│           │                                                                 │
│  ┌────────┴─────────────────────────────────────────────────────────────┐   │
│  │                      资源隔离层 (ACL/CUDA/MPS)                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │ ACL Wrapper │  │ CUDA MPS    │  │ RCKM        │  │ Docker     │  │   │
│  │  │ (NPU)       │  │ (GPU)       │  │ (GPU)       │  │ (通用)     │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Layer 2: 运行时层                                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      实例管理器 (utils_*.py)                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │ 进程管理     │  │ 环境变量设置 │  │ 日志管理     │  │ 健康检查    │ │   │
│  │  │             │  │             │  │             │  │            │ │   │
│  │  │ start/stop  │  │ NPU_DEVICE  │  │ 输出重定向   │  │ /health    │ │   │
│  │  │ _instance   │  │ _ID等       │  │ PID文件     │  │ 轮询       │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      推理服务 (run_*_INF*.py)                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │ 模型加载     │  │ 批量推理     │  │ HTTP服务     │  │ 性能上报    │ │   │
│  │  │             │  │             │  │             │  │            │ │   │
│  │  │ Transformers│  │ Batch Queue │  │ Flask/Fast  │  │ 延迟统计    │ │   │
│  │  │ DeepSpeed   │  │ 动态批处理   │  │ API         │  │ 吞吐计算    │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Layer 1: 基础设施层                              │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   NPU/GPU   │  │   CANN/     │  │   PyTorch/  │  │   其他依赖           │ │
│  │   硬件      │  │   CUDA      │  │   Transformers│  │   (Flask等)        │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、各层详细设计

### 3.1 Layer 1: 基础设施层

**职责**：硬件和底层软件栈

**组件**：
| 组件 | GPU版本 | NPU版本 | 说明 |
|------|---------|---------|------|
| 硬件 | NVIDIA GPU | 华为910B3 | 计算设备 |
| 驱动 | CUDA | CANN | 计算架构 |
| 运行时 | CUDA Runtime | ACL Runtime | 运行时库 |
| 深度学习框架 | PyTorch + CUDA | PyTorch + torch_npu | 训练推理框架 |
| 服务框架 | Flask/FastAPI | Flask/FastAPI | HTTP服务 |

**NPU侧完善点**：
- ✅ CANN/ACL环境配置
- ✅ torch_npu安装和验证
- ⏳ 性能对比测试（CUDA vs CANN）

---

### 3.2 Layer 2: 运行时层

**职责**：管理推理服务的生命周期

#### 3.2.1 实例管理器

**核心文件**：
- `scheduling/utils_npu.py` (NPU)
- `scheduling/utils_docker.py` (GPU)

**关键功能**：
```python
# 启动实例
def start_instance(selected_npus, instance_id, image_name, 
                   service_name, args, allocated_port, ip_address):
    # 1. 设置环境变量
    env["NPU_DEVICE_ID"] = device_id
    env["CUBE_LIMIT"] = cube_lim
    env["VECTOR_LIMIT"] = vector_lim
    env["PORT"] = allocated_port
    
    # 2. 启动子进程
    proc = subprocess.Popen(cmd, env=env, ...)
    
    # 3. 记录PID
    write_pid_file(pid_file, proc.pid)

# 停止实例
def stop_instance(service_name, instance_id, ip_address):
    # 1. 读取PID
    pid = read_pid_file(pid_file)
    
    # 2. 终止进程
    os.kill(pid, 9)
    
    # 3. 清理资源
    cleanup(instance_id)
```

**NPU侧完善点**：
- ✅ 子进程管理
- ✅ 环境变量传递
- ✅ PID文件管理
- ⏳ 优雅关闭（graceful shutdown）
- ⏳ 异常恢复机制

#### 3.2.2 推理服务

**核心文件**：
- `scheduling/scripts_demo/llm_inference_npu.py` (NPU LLM)
- `scheduling/scripts_tasks/run_*_INF*.py` (GPU各类模型)

**架构模式**：
```
┌─────────────────────────────────────┐
│           HTTP Server               │
│  ┌─────────┐      ┌─────────────┐  │
│  │ /health │      │  /predict   │  │
│  └────┬────┘      └──────┬──────┘  │
│       │                  │         │
│       ▼                  ▼         │
│  ┌─────────┐      ┌─────────────┐  │
│  │ 健康检查 │      │  推理处理    │  │
│  │         │      │             │  │
│  │ 返回状态 │      │ 1. 加入Batch │  │
│  │ 模型加载 │      │ 2. 批量推理  │  │
│  └─────────┘      │ 3. 返回结果  │  │
│                   └─────────────┘  │
└─────────────────────────────────────┘
```

**NPU侧完善点**：
- ✅ LLM推理服务（llm_inference_npu.py）
- ⏳ BERT/GPT2/VGG/ResNet等CV/NLP模型NPU化
- ⏳ 动态批处理优化
- ⏳ 性能监控和上报

---

### 3.3 Layer 3: 资源管理层

**职责**：资源模型定义、资源推荐、资源隔离

#### 3.3.1 资源模型

**核心文件**：`scheduling/npu_resource.py`

**关键类**：
```python
class NPU:
    """NPU资源模型"""
    def __init__(self, id, index, cube, vector, memory):
        self.id = id          # 设备ID
        self.index = index    # 索引
        self.cube = cube      # Cube核心（20个）
        self.vector = vector  # Vector核心（40个）
        self.memory = memory  # 显存（64GB）
    
    def allocate(self, cube_req, vector_req, memory_req):
        """分配资源"""
        # Best-Fit/Worst-Fit算法
        pass
    
    def deallocate(self, instance_id):
        """释放资源"""
        pass
```

**NPU资源常量**：
```python
NPU_CUBE_CORES_PER_DEVICE = 20
NPU_VECTOR_CORES_PER_DEVICE = 40
NPU_MEMORY_GB_PER_DEVICE = 64
```

**NPU侧完善点**：
- ✅ NPU类定义
- ✅ 资源分配算法
- ✅ 资源统计
- ⏳ 资源碎片整理
- ⏳ 资源预测（未来需求预估）

#### 3.3.2 资源推荐引擎

**核心文件**：`scheduling/resource_recommender.py`

**核心功能**：
```python
class ResourceRecommender:
    """基于资源画像的推荐引擎"""
    
    def recommend_for_latency(self, target_latency, batch_size):
        """根据延迟SLA推荐资源配置"""
        # 1. 筛选满足延迟要求的配置
        # 2. 选择资源使用最少的
        pass
    
    def recommend_for_throughput(self, target_throughput, batch_size):
        """根据吞吐SLA推荐资源配置"""
        pass
    
    def get_scaling_advice(self, current_cube, current_vector, 
                          current_latency, target_latency):
        """获取扩缩容建议"""
        # 1. 计算性能差距
        # 2. 推荐新配置
        pass
```

**NPU侧完善点**：
- ✅ 基础推荐算法
- ✅ SLA感知推荐
- ⏳ 在线学习（根据实际运行数据优化推荐）
- ⏳ 多目标优化（延迟+成本+能耗）

#### 3.3.3 资源画像

**核心文件**：
- `profiling/npu/profile_npu.py`
- `profiling/npu/run_one_observation_npu.py`

**画像流程**：
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  网格配置    │ -> │  运行测试    │ -> │  收集指标    │ -> │  生成画像    │
│             │    │             │    │             │    │             │
│ Cube: 4-20  │    │ 加载模型    │    │ 延迟        │    │ 配置->性能   │
│ Vector:     │    │ 推理N次     │    │ 吞吐        │    │ 映射表      │
│   10-40     │    │             │    │ 内存使用    │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**NPU侧完善点**：
- ✅ 基础画像脚本
- ⏳ 自动化画像流程
- ⏳ 增量画像（新模型快速画像）
- ⏳ 画像数据可视化

#### 3.3.4 资源隔离层

**NPU版本**：`scheduling/npu/acl_rt_wrapper.py`

**核心功能**：
```python
def init_device(device_id):
    """初始化NPU设备"""
    acl.rt.set_device(device_id)

def set_device_res_limit(device_id, res_type, limit):
    """设置资源限制"""
    # res_type: ACL_RT_DEV_RES_CUBE_CORE / ACL_RT_DEV_RES_VECTOR_CORE
    acl.rt.set_device_res_limit(device_id, res_type, limit)
```

**NPU侧完善点**：
- ✅ ACL基础封装
- ⏳ 显存限制（当前ACL可能不支持细粒度显存限制）
- ⏳ 资源使用监控
- ⏳ 多租户隔离验证

---

### 3.4 Layer 4: 控制平面层

**职责**：调度决策、自动扩缩容

#### 3.4.1 Scheduler（调度器）

**核心文件**：`scheduling/scheduler_npu.py`

**架构**：
```
┌─────────────────────────────────────────────────────────────┐
│                    Flask HTTP Server                        │
├─────────────────────────────────────────────────────────────┤
│  API层                                                       │
│  ├─ /schedule          - 普通调度                           │
│  ├─ /schedule_with_sla - 基于SLA的智能调度                   │
│  ├─ /delete_instance   - 删除实例                           │
│  ├─ /reschedule_with_limits - 垂直缩放                     │
│  ├─ /recommend_resources - 资源推荐                         │
│  ├─ /scaling_advice    - 扩缩容建议                         │
│  └─ /health, /metrics  - 监控接口                           │
├─────────────────────────────────────────────────────────────┤
│  核心逻辑层                                                  │
│  ├─ allocate_instance()  - 资源分配算法                      │
│  ├─ start_instance()     - 启动实例（后台线程）               │
│  ├─ stop_instance()      - 停止实例                         │
│  └─ reschedule_limits()  - 调整资源限制                     │
├─────────────────────────────────────────────────────────────┤
│  数据层                                                      │
│  ├─ npu_list[]          - NPU设备列表                       │
│  ├─ instance_registry{} - 实例注册表                        │
│  └─ task_registry{}     - 任务注册表                        │
└─────────────────────────────────────────────────────────────┘
```

**NPU侧完善点**：
- ✅ 基础调度API
- ✅ 智能调度（/schedule_with_sla）
- ✅ 资源推荐集成
- ⏳ 多节点调度
- ⏳ 调度策略插件化
- ⏳ 调度性能优化

#### 3.4.2 Scaler（自动扩缩容器）

**核心文件**：`adaptive_2D_scaling/horizontal_scaling/scaler_npu.py`

**架构**：
```
┌─────────────────────────────────────────────────────────────┐
│                    NPUScaler (后台线程)                      │
├─────────────────────────────────────────────────────────────┤
│  监控层                                                      │
│  ├─ 收集延迟指标                                            │
│  ├─ 收集吞吐指标                                            │
│  └─ 维护历史数据                                            │
├─────────────────────────────────────────────────────────────┤
│  决策层                                                      │
│  ├─ _intelligent_scaling_decision() - 智能决策              │
│  │   ├─ get_scaling_advice() - 获取推荐                     │
│  │   ├─ 优先垂直缩放                                        │
│  │   └─ 必要时水平缩放                                      │
│  └─ _threshold_based_scaling() - 基于阈值决策               │
├─────────────────────────────────────────────────────────────┤
│  执行层                                                      │
│  ├─ vertical_scale()  - 垂直缩放（调用/reschedule_with_limits）│
│  ├─ scale_out()       - 水平扩容（调用/schedule）            │
│  └─ scale_in()        - 水平缩容（调用/delete_instance）     │
└─────────────────────────────────────────────────────────────┘
```

**NPU侧完善点**：
- ✅ 基础扩缩容逻辑
- ✅ 智能决策（基于资源画像）
- ✅ 2D协同缩放
- ⏳ 预测性扩缩容（基于负载预测）
- ⏳ 扩缩容策略配置化

---

### 3.5 Layer 5: 用户接口层

**职责**：提供用户友好的接口和工具

#### 3.5.1 Demo脚本

**核心文件**：
- `scheduling/scripts_demo/run_demo_npu.py`
- `scheduling/scripts_demo/test_resource_recommendation.py`

**功能**：
- 快速验证系统功能
- 演示完整调用链
- 测试新特性

#### 3.5.2 部署脚本

**核心文件**：
- `scheduling/scripts_deploy/deploy_inference_funcs.py`
- `scheduling/scripts_deploy/deploy_train_funcs.py`

**NPU侧完善点**：
- ⏳ NPU部署脚本
- ⏳ 自动化部署流程
- ⏳ 配置管理

#### 3.5.3 仿真脚本

**核心文件**：
- `scheduling/simulations/workload/service_generator_npu.py`
- `scheduling/simulations/comp_baselines.py`

**功能**：
- 生成测试工作负载
- 对比不同调度策略
- 评估系统性能

---

## 四、NPU侧完善路线图

### 4.1 已完成的P0/P1核心功能

✅ **资源模型**：NPU类、Cube/Vector/Memory资源定义
✅ **调度器**：scheduler_npu.py、智能调度API
✅ **资源推荐**：resource_recommender.py、SLA感知推荐
✅ **扩缩容**：scaler_npu.py、2D协同缩放
✅ **实例管理**：utils_npu.py、进程管理
✅ **LLM推理**：llm_inference_npu.py、模型服务
✅ **资源画像**：profile_npu.py、性能测试
✅ **服务生成**：service_generator_npu.py、工作负载生成

### 4.2 待完善的P1功能

⏳ **更多模型支持**：
- BERT/RoBERTa/GPT2的NPU化
- ResNet/VGG等CV模型的NPU化
- 统一模型加载和推理框架

⏳ **部署脚本**：
- deploy_inference_npu.py
- deploy_train_npu.py
- 自动化部署流程

⏳ **仿真基线**：
- scheduler_dilu_npu.py（NPU版本的Dilu调度器仿真）
- 性能对比测试

### 4.3 待完善的P2功能

⏳ **高级特性**：
- 多节点调度
- 预测性扩缩容
- 能耗优化
- 在线学习

⏳ **可观测性**：
- 监控Dashboard
- 日志聚合
- 性能分析工具

⏳ **易用性**：
- 配置管理工具
- 调试工具
- 文档和示例

---

## 五、核心设计模式总结

### 5.1 分层架构模式

```
用户接口层 -> 控制平面层 -> 资源管理层 -> 运行时层 -> 基础设施层
```

**优势**：
- 职责清晰
- 易于测试
- 便于扩展

### 5.2 插件化资源隔离

```python
# 资源隔离接口
class ResourceIsolation:
    def init_device(self, device_id): pass
    def set_resource_limit(self, device_id, res_type, limit): pass
    def cleanup(self, device_id): pass

# GPU实现
class CudaMPSIsolation(ResourceIsolation): ...

# NPU实现
class ACLIsolation(ResourceIsolation): ...
```

### 5.3 策略模式调度

```python
# 调度策略接口
class SchedulingStrategy:
    def select_npu(self, npus, requirements): pass

# 具体策略
class BestFitStrategy(SchedulingStrategy): ...
class WorstFitStrategy(SchedulingStrategy): ...
class SLAwareStrategy(SchedulingStrategy): ...
```

### 5.4 观察者模式扩缩容

```python
# 性能监控 -> 决策引擎 -> 执行器
class PerformanceMonitor:
    def register_observer(self, observer): ...
    def notify(self, metrics): ...

class ScalingDecisionEngine(Observer):
    def on_performance_update(self, metrics): ...
```

---

## 六、下一步行动建议

### 6.1 短期（1-2周）

1. **完善模型支持**
   - 将scripts_tasks中的关键模型NPU化
   - 统一模型服务接口

2. **部署脚本**
   - 创建deploy_inference_npu.py
   - 实现自动化部署流程

3. **测试验证**
   - 端到端测试
   - 性能基准测试

### 6.2 中期（1个月）

1. **仿真基线**
   - 完成scheduler_dilu_npu.py
   - 进行对比实验

2. **可观测性**
   - 添加监控指标
   - 创建Dashboard

3. **文档完善**
   - API文档
   - 部署指南
   - 最佳实践

### 6.3 长期（2-3个月）

1. **高级特性**
   - 多节点调度
   - 预测性扩缩容

2. **生产就绪**
   - 高可用性
   - 容错机制
   - 性能优化

---

**文档版本**：2026-01-30
