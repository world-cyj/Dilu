# NPU 模块目录结构

本文档展示NPU模块的完整目录结构和文件清单。

## 完整目录树

```
npu/
├── README.md                          # 模块说明文档
├── STRUCTURE.md                       # 本文件
├── copy_npu_files.py                  # 文件复制脚本
│
├── scheduling/                        # 调度模块
│   ├── __init__.py
│   │
│   ├── core/                          # 核心调度组件
│   │   ├── __init__.py
│   │   ├── npu_resource.py           # NPU资源模型
│   │   ├── scheduler_npu.py          # NPU调度器（主服务）
│   │   ├── utils_npu.py              # NPU实例管理工具
│   │   ├── resource_recommender.py   # 资源推荐引擎
│   │   └── scaler_npu.py             # 扩缩容器（调度器内嵌）
│   │
│   ├── scripts/                       # 脚本工具
│   │   ├── __init__.py
│   │   └── demo/                      # Demo和测试脚本
│   │       ├── __init__.py
│   │       ├── run_demo_npu.py       # Demo运行脚本
│   │       ├── llm_inference_npu.py  # LLM推理服务
│   │       ├── npu_worker_entry.py   # NPU Worker入口
│   │       ├── verify_core_functions.py  # 核心功能验证
│   │       ├── test_resource_recommendation.py  # 资源推荐测试
│   │       └── service_generator_npu.py  # 服务生成器
│   │
│   └── utils/                         # 工具函数
│       ├── __init__.py
│       └── acl_rt_wrapper.py         # ACL运行时封装
│
├── profiling/                         # 资源画像模块
│   ├── __init__.py
│   └── inference/                     # 推理画像
│       ├── __init__.py
│       ├── profile_npu.py            # NPU资源画像主脚本
│       └── run_one_observation_npu.py # 单次观测运行
│
└── adaptive_scaling/                  # 自适应扩缩容
    ├── __init__.py
    └── horizontal/                    # 水平扩缩容
        ├── __init__.py
        ├── scaler_npu.py             # NPU扩缩容器（独立服务）
        ├── scheduler_npu_adapter.py  # 调度器适配器
        └── utils_npu_adapter.py      # 工具适配器
```

## 文件统计

### 按类别统计

| 类别 | 文件数 | 说明 |
|-----|-------|------|
| 核心组件 | 5 | 资源模型、调度器、推荐引擎等 |
| Demo脚本 | 6 | 运行、测试、验证脚本 |
| 工具函数 | 2 | ACL封装、实例管理 |
| 资源画像 | 2 | 画像脚本和观测工具 |
| 扩缩容 | 3 | 扩缩容器和适配器 |
| 初始化文件 | 8 | __init__.py |
| **总计** | **26** | Python文件 |

### 按模块统计

| 模块 | 文件数 |
|-----|-------|
| scheduling/core | 5 |
| scheduling/scripts/demo | 6 |
| scheduling/utils | 2 |
| profiling/inference | 2 |
| adaptive_scaling/horizontal | 3 |
| 根目录 | 3 |

## 核心文件说明

### 1. 调度模块 (scheduling/)

#### core/ - 核心组件

- **npu_resource.py**: NPU资源模型
  - NPU类定义（Cube/Vector/Memory）
  - Best-Fit/Worst-Fit分配算法
  - 资源统计和监控

- **scheduler_npu.py**: NPU调度器
  - Flask HTTP服务
  - /schedule, /schedule_with_sla端点
  - /recommend_resources, /scaling_advice端点
  - 资源分配和管理

- **utils_npu.py**: 实例管理工具
  - start_instance(): 启动NPU实例
  - stop_instance(): 停止NPU实例
  - 进程管理和日志记录

- **resource_recommender.py**: 资源推荐引擎
  - recommend_for_latency(): 延迟感知推荐
  - recommend_for_throughput(): 吞吐感知推荐
  - get_scaling_advice(): 扩缩容建议

- **scaler_npu.py**: 扩缩容器（调度器内嵌版本）
  - 垂直缩放支持
  - 水平缩放支持

#### scripts/demo/ - Demo脚本

- **run_demo_npu.py**: Demo运行脚本
  - 启动调度器
  - 创建和删除实例
  - 测试推理服务

- **llm_inference_npu.py**: LLM推理服务
  - ACL资源限制
  - 模型加载和推理
  - HTTP服务接口

- **npu_worker_entry.py**: NPU Worker入口
  - 占位worker实现
  - 健康检查和预测接口

- **verify_core_functions.py**: 核心功能验证
  - 验证资源模型
  - 验证调度器
  - 验证推荐引擎
  - 验证扩缩容机制

- **test_resource_recommendation.py**: 资源推荐测试
  - 测试推荐API
  - 测试调度API

- **service_generator_npu.py**: 服务生成器
  - 生成NPU服务工作负载
  - 生成生命周期事件

#### utils/ - 工具函数

- **acl_rt_wrapper.py**: ACL运行时封装
  - init_device(): 初始化设备
  - set_device_res_limit(): 设置资源限制
  - apply_device_res_limit(): 应用资源限制

### 2. 资源画像模块 (profiling/)

#### inference/ - 推理画像

- **profile_npu.py**: NPU资源画像主脚本
  - Cube/Vector网格测试
  - 延迟/吞吐采集
  - 画像数据生成

- **run_one_observation_npu.py**: 单次观测运行
  - 单次资源测试
  - 性能数据采集

### 3. 自适应扩缩容模块 (adaptive_scaling/)

#### horizontal/ - 水平扩缩容

- **scaler_npu.py**: NPU扩缩容器（独立服务版本）
  - NPUService类
  - NPUScaler类
  - 2D协同缩放

- **scheduler_npu_adapter.py**: 调度器适配器
  - 复用scheduler_npu.py核心逻辑
  - 提供兼容接口

- **utils_npu_adapter.py**: 工具适配器
  - 复用utils_npu.py功能
  - 提供兼容接口

## 与原Dilu的对应关系

```
npu/scheduling/core/npu_resource.py          -> scheduling/npu_resource.py
npu/scheduling/core/scheduler_npu.py         -> scheduling/scheduler_npu.py
npu/scheduling/core/utils_npu.py             -> scheduling/utils_npu.py
npu/scheduling/core/resource_recommender.py  -> scheduling/resource_recommender.py
npu/scheduling/utils/acl_rt_wrapper.py       -> scheduling/npu/acl_rt_wrapper.py
npu/profiling/inference/profile_npu.py       -> profiling/npu/profile_npu.py
npu/adaptive_scaling/horizontal/scaler_npu.py -> adaptive_2D_scaling/horizontal_scaling/scaler_npu.py
```

## 使用方式

### 1. 直接使用原Dilu路径（推荐）

```bash
cd /vllm-workspace/Dilu
python3 scheduling/scripts_demo/run_demo_npu.py
```

### 2. 使用NPU独立模块

```bash
cd /vllm-workspace/Dilu/npu
export PYTHONPATH=/vllm-workspace/Dilu:$PYTHONPATH
python3 scheduling/scripts/demo/run_demo_npu.py
```

## 文件复制记录

复制时间: 2026-01-30  
复制脚本: copy_npu_files.py  
成功复制: 18个文件  
创建__init__.py: 8个  
总计: 26个Python文件
