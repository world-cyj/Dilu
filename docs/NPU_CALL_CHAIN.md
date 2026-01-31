# NPU 运行流程完整调用链

本文档详细描述了Dilu NPU版本的完整运行流程，包括各个文件的调用关系和执行顺序。

---

## 一、整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                    用户层                                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                  │
│  │   run_demo_npu  │  │   test_resource │  │   scaler_npu    │                  │
│  │      .py        │  │  _recommendation│  │      .py        │                  │
│  │                 │  │      .py        │  │                 │                  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘                  │
│           │                    │                    │                           │
│           └────────────────────┼────────────────────┘                           │
│                                ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         调度器层 (Flask HTTP API)                        │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │                    scheduler_npu.py (Port 5000)                  │   │   │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │   │   │
│  │  │  │  /schedule  │ │/schedule_with│ │/recommend_  │ │/scaling_   │ │   │   │
│  │  │  │             │ │    _sla      │ │  resources  │ │  advice    │ │   │   │
│  │  │  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘ │   │   │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │   │   │
│  │  │  │/delete_inst │ │/reschedule_ │ │  /health    │ │  /metrics  │ │   │   │
│  │  │  │   ance      │ │ with_limits │ │             │ │            │ │   │   │
│  │  │  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘ │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                │                                                │
│                                ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         资源推荐引擎层                                   │   │
│  │                    resource_recommender.py                              │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │   │
│  │  │recommend_for_   │  │recommend_for_   │  │get_scaling_     │         │   │
│  │  │    latency()    │  │   throughput()  │  │    advice()     │         │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                              │   │
│  │  │recommend_cost_  │  │generate_schedule│                              │   │
│  │  │   effective()   │  │   _request()    │                              │   │
│  │  └─────────────────┘  └─────────────────┘                              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                │                                                │
│                                ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         资源管理层                                       │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │   │
│  │  │  npu_resource.py│  │   utils_npu.py  │  │  npu/acl_rt_    │         │   │
│  │  │                 │  │                 │  │   wrapper.py    │         │   │
│  │  │  - NPU类定义    │  │  - start_inst   │  │                 │         │   │
│  │  │  - 资源分配     │  │    ance()       │  │  - ACL初始化    │         │   │
│  │  │  - Best/Worst   │  │  - stop_inst    │  │  - 资源限制     │         │   │
│  │  │    Fit算法      │  │    ance()       │  │  - 设备管理     │         │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                │                                                │
│                                ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         推理服务层                                       │   │
│  │                llm_inference_npu.py (Port 15000+)                       │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │   │
│  │  │  _apply_acl()   │  │    inference    │  │  Flask Server   │         │   │
│  │  │                 │  │    (batch)      │  │   /health       │         │   │
│  │  │  - ACL初始化    │  │                 │  │   /predict      │         │   │
│  │  │  - 资源限制     │  │  - 模型推理     │  │                 │         │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、详细调用链

### 2.1 Demo运行流程

#### 入口：`scheduling/scripts_demo/run_demo_npu.py`

```python
# 第1步：启动调度器
threading.Thread(target=run_scheduler, daemon=True).start()
    └── scheduler_npu.app.run(host="0.0.0.0", port=5000)
        └── Flask服务器启动，监听HTTP请求

# 第2步：发送调度请求
requests.post("http://127.0.0.1:5000/schedule", json={...})
    └── scheduler_npu.schedule() [Flask路由]
        └── scheduler_npu.allocate_instance(data)
            └── npu_resource.NPU.allocate()
                └── Best-Fit/Worst-Fit算法选择NPU
        └── threading.Thread(target=scheduler_npu.start_instance)
            └── utils_npu.start_instance()
                └── 设置环境变量：NPU_DEVICE_ID, CUBE_LIMIT, VECTOR_LIMIT
                └── subprocess.Popen([python3, llm_inference_npu.py])
                    └── llm_inference_npu.py启动
                        └── _apply_acl()  # ACL初始化
                            └── npu.acl_rt_wrapper.init_device()
                            └── npu.acl_rt_wrapper.set_device_res_limit()
                        └── 加载模型：AutoModelForCausalLM.from_pretrained()
                        └── Flask服务器启动：app.run(port=15000)

# 第3步：健康检查
requests.get(f"http://127.0.0.1:{port}/health")
    └── llm_inference_npu.health() [Flask路由]
        └── 返回：{"status": "healthy", "model_loaded": true}

# 第4步：推理请求
requests.post(f"http://127.0.0.1:{port}/predict", json={"text": "..."})
    └── llm_inference_npu.predict() [Flask路由]
        └── 加入batch_queue
        └── batch_inference_loop线程处理
            └── inference()  # 实际推理
                └── model.generate()  # 模型推理

# 第5步：删除实例
requests.post("http://127.0.0.1:5000/delete_instance", json={"instance_id": ...})
    └── scheduler_npu.delete_instance() [Flask路由]
        └── npu_resource.NPU.deallocate()
        └── utils_npu.stop_instance()
            └── os.kill(pid, 9)  # 终止进程
```

---

### 2.2 资源推荐流程

#### 入口：`scheduling/resource_recommender.py`

```python
# 初始化
recommender = ResourceRecommender(profiling_path)
    └── load_profiling_data()
        └── 读取：profiling/npu/profiling_npu_result.json
        └── 解析observations列表

# 根据延迟推荐
config = recommender.recommend_for_latency(target_latency=0.2, batch_size=1)
    └── 遍历observations
    └── 筛选：batch_size匹配、资源不超限制、延迟满足SLA
    └── 选择：资源使用最少的配置
    └── 返回：ResourceConfig(cube, vector, memory, expected_latency, ...)

# 生成调度请求
request = recommender.generate_schedule_request(
    service_name='my-service',
    model_path='/vllm-workspace/models/Qwen3-4B',
    sla_latency=0.2
)
    └── 调用recommend_for_latency()
    └── 转换为比例形式：
        cube_requests = config.cube / 20  # 20是910B3的Cube核心数
        cube_limits = min(config.cube * 1.2 / 20, 1.0)
    └── 返回调度请求字典
```

---

### 2.3 智能调度流程（基于SLA）

#### 入口：`scheduling/scheduler_npu.py /schedule_with_sla`

```python
# HTTP请求
POST /schedule_with_sla
    └── schedule_with_sla() [Flask路由]
        └── from resource_recommender import get_recommender
        └── recommender = get_recommender()
        └── schedule_request = recommender.generate_schedule_request(
               service_name=..., model_path=..., sla_latency=...
           )
            └── 调用ResourceRecommender.recommend_for_latency()
        └── instance, selected_npus, allocated_port, memory, err = allocate_instance(schedule_request)
            └── npu_resource.NPU.allocate()
        └── threading.Thread(target=start_instance, args=(selected_npus, instance.instance_id, ...))
            └── utils_npu.start_instance()
        └── 返回JSON响应（包含推荐配置）
```

---

### 2.4 扩容/缩容建议流程

#### 入口：`scheduling/scheduler_npu.py /scaling_advice`

```python
# HTTP请求
POST /scaling_advice
    └── scaling_advice() [Flask路由]
        └── from resource_recommender import get_recommender
        └── recommender = get_recommender()
        └── advice = recommender.get_scaling_advice(
               current_cube=8, current_vector=20,
               current_latency=0.5, target_latency=0.2
           )
            └── 查找当前配置附近的性能数据
            └── 计算latency_gap
            └── 如果latency_gap > 0: 返回scale_up建议
            └── 如果latency_gap < 0: 返回scale_down建议
        └── 返回JSON响应（包含action、reason、recommendation）
```

---

### 2.5 Horizontal Scaling流程

#### 入口：`adaptive_2D_scaling/horizontal_scaling/scaler_npu.py`

```python
# 启动Scaler
scaler = NPUScaler(enable_intelligent_scaling=True)
scaler.start()
    └── NPUScaler.run() [线程]
        └── 循环监控每个服务
            └── service.get_average_latency()
            └── 如果enable_intelligent_scaling:
                └── NPUScaler._intelligent_scaling_decision(service, service_id)
                    └── service.get_scaling_advice()
                        └── resource_recommender.get_recommender()
                        └── recommender.get_scaling_advice()
                    └── 如果action == 'scale_up':
                        └── 如果enable_vertical_scaling:
                            └── service.vertical_scale(new_cube, new_vector)
                                └── POST /reschedule_with_limits
                            └── 否则：service.scale_out()
                                └── POST /schedule
                    └── 如果action == 'scale_down':
                        └── 类似处理

# 注册服务
POST /register_service
    └── service_manager.register_service(service_data)
        └── NPUService.__init__()
            └── self.scale_out()
                └── 调用scheduler_npu的/schedule端点

# 处理请求
POST /<service_id>
    └── handle_predict(service_id)
        └── service.dispatch(request)
            └── 轮询选择ready实例
            └── requests.post(f'http://{ip}:{port}/predict', ...)
```

---

### 2.6 仿真工作负载生成流程

#### 入口：`scheduling/simulations/workload/service_generator_npu.py`

```python
# 生成工作负载
instances, events = generate_npu_workload(total_instances=100, train_ratio=0.2, llm_ratio=0.2)
    └── generate_npu_instances()
        ├── 生成训练任务（20%）
        │   └── cube_requests=0.6-0.9, vector_requests=0.6-0.9
        ├── 生成普通推理任务（60%）
        │   └── cube_requests=0.2-0.5, vector_requests=0.2-0.5
        └── 生成LLM推理任务（20%）
            └── cube_requests=0.5-0.8, vector_requests=0.4-0.7
    └── generate_npu_events(instances)
        ├── 生成start事件（每2秒一个）
        ├── 生成inference的delete事件（60%比例，1-3分钟后）
        └── 生成training的delete事件（10%比例，10-15分钟后）

# 转换为调度器格式
request = convert_to_scheduler_format(instance)
    └── 返回：{"cube_requests": ..., "vector_requests": ..., "memory": ..., ...}
```

---

## 三、文件依赖关系图

```
scheduler_npu.py
    ├── 导入: npu_resource.py
    │       └── NPU类（资源分配算法）
    ├── 导入: utils_npu.py
    │       └── start_instance(), stop_instance()
    ├── 导入: resource_recommender.py [可选]
    │       └── get_recommender(), ResourceRecommender
    └── 导入: npu.acl_rt_wrapper [间接通过utils_npu]

resource_recommender.py
    ├── 读取: profiling/npu/profiling_npu_result.json
    └── 被导入: scheduler_npu.py, scaler_npu.py

utils_npu.py
    ├── 导入: npu.acl_rt_wrapper
    │       └── init_device(), set_device_res_limit()
    └── 启动: llm_inference_npu.py (子进程)

llm_inference_npu.py
    ├── 导入: npu.acl_rt_wrapper
    │       └── _apply_acl()中调用
    ├── 导入: transformers
    │       └── AutoTokenizer, AutoModelForCausalLM
    └── 启动: Flask服务器 (/health, /predict)

scaler_npu.py
    ├── 导入: resource_recommender.py
    │       └── get_recommender(), ResourceRecommender
    ├── HTTP调用: scheduler_npu.py的端点
    │       └── /schedule, /delete_instance, /reschedule_with_limits
    └── 导入: scheduler_npu_adapter.py [可选]

service_generator_npu.py
    └── 生成: 实例配置和事件
        └── 被: 仿真脚本使用

scheduler_npu_adapter.py
    ├── 导入: scheduler_npu.py
    │       └── app, allocate_instance, start_instance
    └── HTTP调用: scheduler_npu.py的端点

utils_npu_adapter.py
    └── 导入: utils_npu.py
        └── start_instance(), stop_instance()
```

---

## 四、关键调用时序图

### 4.1 完整调度流程时序

```
用户                    scheduler_npu              resource_recommender          npu_resource              utils_npu              llm_inference_npu
 │                             │                              │                         │                      │                      │
 │  POST /schedule_with_sla    │                              │                         │                      │                      │
 │────────────────────────────>│                              │                         │                      │                      │
 │                             │  generate_schedule_request() │                         │                      │                      │
 │                             │─────────────────────────────>│                         │                      │                      │
 │                             │                              │  recommend_for_latency()│                      │                      │
 │                             │                              │────────────────────────>│                      │                      │
 │                             │                              │<────────────────────────│                      │                      │
 │                             │<─────────────────────────────│                         │                      │                      │
 │                             │                                                            │                      │                      │
 │                             │  allocate_instance()                                       │                      │                      │
 │                             │─────────────────────────────────────────────────────────>│                      │                      │
 │                             │                                                            │  NPU.allocate()      │                      │
 │                             │                                                            │  (Best/Worst Fit)    │                      │
 │                             │<─────────────────────────────────────────────────────────│                      │                      │
 │                             │                                                            │                      │                      │
 │                             │  start_instance() (后台线程)                               │                      │                      │
 │                             │───────────────────────────────────────────────────────────────────────────────>│                      │
 │                             │                                                                                   │  Popen(llm_inf)      │
 │                             │                                                                                   │─────────────────────>│
 │                             │                                                                                   │                      │  _apply_acl()
 │                             │                                                                                   │                      │  加载模型
 │                             │                                                                                   │                      │  启动Flask
 │                             │                                                                                   │<─────────────────────│
 │                             │<───────────────────────────────────────────────────────────────────────────────│                      │
 │                             │                                                            │                      │                      │
 │  返回instance_id, port,     │                                                            │                      │                      │
 │  recommendation             │                                                            │                      │                      │
 │<────────────────────────────│                                                            │                      │                      │
 │                             │                                                            │                      │                      │
 │  GET /health (轮询)         │                                                            │                      │                      │
 │  (通过port直接访问)         │                                                            │                      │                      │
 │──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────>│
 │<───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────│
 │                             │                                                            │                      │                      │
 │  POST /predict              │                                                            │                      │                      │
 │  (通过port直接访问)         │                                                            │                      │                      │
 │──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────>│
 │<───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────│
```

---

### 4.2 资源推荐流程时序

```
用户                    scheduler_npu              resource_recommender          profiling_npu_result.json
 │                             │                              │                              │
 │  POST /recommend_resources  │                              │                              │
 │────────────────────────────>│                              │                              │
 │                             │  get_recommender()           │                              │
 │                             │─────────────────────────────>│                              │
 │                             │                              │  __init__()                  │
 │                             │                              │  load_profiling_data()       │
 │                             │                              │─────────────────────────────>│
 │                             │                              │<─────────────────────────────│
 │                             │<─────────────────────────────│                              │
 │                             │                              │                              │
 │                             │  recommend_for_latency()     │                              │
 │                             │  (或recommend_for_throughput)│                              │
 │                             │─────────────────────────────>│                              │
 │                             │                              │  遍历observations            │
 │                             │                              │  筛选满足SLA的配置           │
 │                             │                              │  选择资源使用最少的          │
 │                             │<─────────────────────────────│                              │
 │                             │                              │                              │
 │  返回recommendation         │                              │                              │
 │<────────────────────────────│                              │                              │
```

---

### 4.3 自动扩缩容流程时序

```
scaler_npu              scheduler_npu              resource_recommender              推理实例
    │                        │                              │                              │
    │  NPUScaler.run()       │                              │                              │
    │  (后台线程循环)         │                              │                              │
    │                        │                              │                              │
    ├──get_average_latency()─┤                              │                              │
    │<───────────────────────│                              │                              │
    │                        │                              │                              │
    ├──get_scaling_advice()────────────────────────────────>│                              │
    │                        │                              │                              │
    │                        │                              ├──get_recommender()──────────>│
    │                        │                              │<─────────────────────────────│
    │                        │                              │                              │
    │                        │                              ├──get_scaling_advice()        │
    │                        │                              │  (current_cube, vector,       │
    │                        │                              │   current_latency,            │
    │                        │                              │   target_latency)             │
    │<──────────────────────────────────────────────────────│                              │
    │                        │                              │                              │
    │  根据advice.action决策  │                              │                              │
    │                        │                              │                              │
    ├─if scale_up────────────┤                              │                              │
    │  ├─if vertical_scale──>│                              │                              │
    │  │  POST /reschedule_  │                              │                              │
    │  │      with_limits    │                              │                              │
    │  │                     ├──调整NPU资源限制─────────────>│                              │
    │  │                     │                              │                              │
    │  └─else scale_out─────>│                              │                              │
    │    POST /schedule      │                              │                              │
    │                        ├──allocate_instance()────────>│                              │
    │                        │                              │                              │
    │                        ├──start_instance()───────────────────────────────────────────>│
    │                        │                              │                              │
    │                        │                              │                              ├──启动新实例
    │                        │                              │                              │
    ├─if scale_down─────────>│                              │                              │
       POST /delete_instance │                              │                              │
                             ├──deallocate()───────────────>│                              │
                             ├──stop_instance()──────────────────────────────────────────>│
                                                            │                              ├──停止实例
```

---

## 五、配置文件和数据流

### 5.1 配置文件

| 文件 | 用途 | 读取者 |
|------|------|--------|
| `profiling/npu/profiling_npu_result.json` | 资源画像数据 | `resource_recommender.py` |
| `scheduling/npu_nodes_info.json` | NPU节点信息 | `npu_resource.py` |

### 5.2 环境变量

| 变量 | 设置者 | 使用者 | 说明 |
|------|--------|--------|------|
| `NPU_DEVICE_ID` | `utils_npu.py` | `llm_inference_npu.py` | NPU设备ID |
| `CUBE_LIMIT` | `utils_npu.py` | `llm_inference_npu.py` | Cube核心限制 |
| `VECTOR_LIMIT` | `utils_npu.py` | `llm_inference_npu.py` | Vector核心限制 |
| `MEMORY_GB` | `utils_npu.py` | `llm_inference_npu.py` | 内存限制 |
| `PORT` | `utils_npu.py` | `llm_inference_npu.py` | 服务端口号 |
| `MODEL_PATH` | `utils_npu.py` | `llm_inference_npu.py` | 模型路径 |
| `INSTANCE_ID` | `utils_npu.py` | `llm_inference_npu.py` | 实例ID |

### 5.3 日志文件

| 文件 | 生成者 | 内容 |
|------|--------|------|
| `logs/<service_name>-<instance_id>.log` | `utils_npu.py` | 推理服务输出 |
| `logs/<service_name>-<instance_id>.pid` | `utils_npu.py` | 进程ID |
| `logs/dilu-scaler-npu.log` | `scaler_npu.py` | 扩缩容日志 |

---

## 六、HTTP API端点汇总

### 6.1 scheduler_npu.py (Port 5000)

| 端点 | 方法 | 功能 | 调用者 |
|------|------|------|--------|
| `/schedule` | POST | 普通调度 | run_demo_npu.py, scaler_npu.py |
| `/schedule_with_sla` | POST | 基于SLA的智能调度 | test_resource_recommendation.py |
| `/delete_instance` | POST | 删除实例 | run_demo_npu.py, scaler_npu.py |
| `/reschedule_with_limits` | POST | 垂直缩放 | scaler_npu.py |
| `/recommend_resources` | POST | 获取资源配置建议 | test_resource_recommendation.py |
| `/scaling_advice` | POST | 获取扩缩容建议 | test_resource_recommendation.py |
| `/health` | GET | 调度器健康检查 | - |
| `/instances` | GET | 获取所有实例 | test_resource_recommendation.py |
| `/cluster_resources` | GET | 获取集群资源 | - |

### 6.2 llm_inference_npu.py (Port 15000+)

| 端点 | 方法 | 功能 | 调用者 |
|------|------|------|--------|
| `/health` | GET | 服务健康检查 | run_demo_npu.py |
| `/predict` | POST | 模型推理 | run_demo_npu.py |

### 6.3 scaler_npu.py (Port 14999)

| 端点 | 方法 | 功能 | 调用者 |
|------|------|------|--------|
| `/register_service` | POST | 注册服务 | - |
| `/delete_service` | POST | 删除服务 | - |
| `/<service_id>` | POST | 处理推理请求 | 用户 |
| `/service/<service_id>/status` | GET | 获取服务状态 | - |
| `/service/<service_id>/scale` | POST | 手动扩缩容 | - |

---

## 七、总结

整个NPU运行流程的调用链可以概括为：

1. **用户层**：通过HTTP API与调度器交互
2. **调度器层**：处理请求，调用资源推荐引擎，分配NPU资源
3. **资源推荐层**：基于资源画像数据，提供最优配置建议
4. **资源管理层**：执行实际的资源分配和实例管理
5. **推理服务层**：运行实际的模型推理服务

**核心特点：**
- 资源画像驱动：基于profiling数据做决策
- SLA感知：根据延迟/吞吐要求自动调整资源
- 2D协同缩放：支持垂直（资源调整）和水平（实例增减）缩放
- 模块化设计：各组件通过HTTP API和导入关系松耦合

---

**文档版本：** 2026-01-30
