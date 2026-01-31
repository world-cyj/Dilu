# Dilu NPU 版本 - 独立模块说明

本文档说明NPU版本Dilu的代码组织结构和文件映射关系。

## 目录结构

```
npu/
├── README.md                      # 本文件
├── scheduling/                    # 调度模块（对应原Dilu/scheduling）
│   ├── core/                      # 核心调度组件
│   │   ├── npu_resource.py        # NPU资源模型
│   │   ├── scheduler_npu.py       # NPU调度器
│   │   ├── utils_npu.py           # NPU实例管理
│   │   └── resource_recommender.py # 资源推荐引擎
│   ├── scripts/                   # 脚本工具
│   │   ├── demo/                  # Demo脚本
│   │   │   ├── run_demo_npu.py
│   │   │   ├── llm_inference_npu.py
│   │   │   ├── npu_worker_entry.py
│   │   │   └── verify_core_functions.py
│   │   ├── deploy/                # 部署脚本（待实现）
│   │   └── tasks/                 # 任务脚本（待NPU化）
│   └── utils/                     # 工具函数
│       └── acl_rt_wrapper.py      # ACL运行时封装
│
├── profiling/                     # 资源画像模块（对应原Dilu/profiling/npu）
│   ├── inference/                 # 推理画像
│   │   ├── profile_npu.py         # NPU资源画像
│   │   └── run_one_observation_npu.py
│   └── training/                  # 训练画像（待实现）
│
├── adaptive_scaling/              # 自适应扩缩容（对应原Dilu/adaptive_2D_scaling）
│   ├── horizontal/                # 水平扩缩容
│   │   ├── scaler_npu.py          # NPU扩缩容器
│   │   ├── scheduler_npu_adapter.py
│   │   └── utils_npu_adapter.py
│   └── vertical/                  # 垂直扩缩容（NPU通过ACL实现）
│       └── README.md
│
├── docs/                          # 文档
│   ├── ARCHITECTURE.md            # 架构设计
│   ├── API.md                     # API文档
│   └── MIGRATION.md               # 迁移指南
│
└── tests/                         # 测试
    ├── unit/                      # 单元测试
    └── integration/               # 集成测试
```

## 与原Dilu的映射关系

### 核心文件映射

| NPU版本路径 | 原Dilu路径 | 说明 |
|------------|-----------|------|
| `scheduling/core/npu_resource.py` | `scheduling/npu_resource.py` | NPU资源模型 |
| `scheduling/core/scheduler_npu.py` | `scheduling/scheduler_npu.py` | NPU调度器 |
| `scheduling/core/utils_npu.py` | `scheduling/utils_npu.py` | NPU实例管理 |
| `scheduling/core/resource_recommender.py` | `scheduling/resource_recommender.py` | 资源推荐引擎 |
| `scheduling/utils/acl_rt_wrapper.py` | `scheduling/npu/acl_rt_wrapper.py` | ACL运行时封装 |
| `profiling/inference/profile_npu.py` | `profiling/npu/profile_npu.py` | NPU资源画像 |
| `adaptive_scaling/horizontal/scaler_npu.py` | `adaptive_2D_scaling/horizontal_scaling/scaler_npu.py` | NPU扩缩容器 |

### Demo和测试文件映射

| NPU版本路径 | 原Dilu路径 | 说明 |
|------------|-----------|------|
| `scheduling/scripts/demo/run_demo_npu.py` | `scheduling/scripts_demo/run_demo_npu.py` | Demo运行脚本 |
| `scheduling/scripts/demo/llm_inference_npu.py` | `scheduling/scripts_demo/llm_inference_npu.py` | LLM推理服务 |
| `scheduling/scripts/demo/verify_core_functions.py` | `scheduling/scripts_demo/verify_core_functions.py` | 核心功能验证 |

## 使用说明

### 1. 直接使用原Dilu路径（推荐）

当前NPU代码已与原Dilu代码集成，可以直接使用：

```bash
cd /vllm-workspace/Dilu
python3 scheduling/scripts_demo/run_demo_npu.py
```

### 2. 独立使用NPU模块

如果需要独立使用NPU模块，可以设置Python路径：

```bash
export PYTHONPATH=/vllm-workspace/Dilu/npu:$PYTHONPATH
python3 npu/scheduling/core/scheduler_npu.py
```

## 核心功能清单

### ✅ 已实现功能

1. **资源模型** (`npu_resource.py`)
   - NPU类定义（Cube/Vector/Memory）
   - Best-Fit/Worst-Fit分配算法
   - 资源统计和监控

2. **调度器** (`scheduler_npu.py`)
   - 普通调度 (`/schedule`)
   - 智能调度 (`/schedule_with_sla`)
   - 资源推荐 (`/recommend_resources`)
   - 扩缩容建议 (`/scaling_advice`)
   - 垂直缩放 (`/reschedule_with_limits`)

3. **资源推荐引擎** (`resource_recommender.py`)
   - 延迟感知推荐
   - 吞吐感知推荐
   - 性价比推荐
   - 扩缩容建议

4. **扩缩容** (`scaler_npu.py`)
   - 水平扩缩容
   - 垂直扩缩容
   - 2D协同缩放
   - SLA感知决策

5. **资源画像** (`profile_npu.py`)
   - Cube/Vector网格测试
   - 延迟/吞吐采集
   - 画像数据生成

6. **推理服务** (`llm_inference_npu.py`)
   - ACL资源限制
   - 模型加载和推理
   - HTTP服务接口

### ⏳ 待实现功能

1. **更多模型支持**
   - BERT/RoBERTa NPU化
   - ResNet/VGG NPU化
   - GPT2 NPU化

2. **部署脚本**
   - deploy_inference_npu.py
   - deploy_train_npu.py

3. **仿真基线**
   - scheduler_dilu_npu.py

4. **高级特性**
   - 多节点调度
   - 预测性扩缩容
   - 在线学习

## 设计差异说明

### 与GPU版本的主要差异

| 特性 | GPU版本 | NPU版本 |
|-----|--------|--------|
| 资源维度 | SM核心 + 显存 | Cube核心 + Vector核心 + 显存 |
| 资源隔离 | CUDA MPS | ACL Runtime |
| 运行时 | Docker容器 | 本机进程 |
| 通信方式 | SSH + HTTP | 本机HTTP |

### 架构优势

1. **更细粒度的资源控制**：Cube和Vector核心独立配置
2. **更轻量的运行时**：无需Docker，直接进程管理
3. **更智能的调度**：基于资源画像的SLA感知推荐
4. **2D协同缩放**：垂直+水平协同决策

## 验证状态

所有核心功能已通过验证：

- ✅ 资源模型和分配算法
- ✅ 调度器核心功能
- ✅ 资源推荐引擎
- ✅ 扩缩容机制

验证脚本：`scheduling/scripts_demo/verify_core_functions.py`

## 联系方式

如有问题，请参考：
- 架构文档：`docs/DILU_ARCHITECTURE.md`
- 迁移状态：`docs/NPU_MIGRATION_STATUS.md`
- 调用链文档：`docs/NPU_CALL_CHAIN.md`
