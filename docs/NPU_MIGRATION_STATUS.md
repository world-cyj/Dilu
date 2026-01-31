# NPU 迁移状态：已做 vs 待补

对照 [NPU_MIGRATION_FILE_LIST.md](NPU_MIGRATION_FILE_LIST.md) 的改造清单，当前实现状态如下。

---

## 一、已完成（P0 核心调度平面）

| 清单项 | 改造策略 | 当前实现 | 状态 |
|--------|----------|----------|------|
| **scheduling/scheduler.py** | GPU→NPU 资源模型、单机 4 卡、Best-Fit/Worst-Fit | **scheduling/scheduler_npu.py**：NPU 类（cube/vector/memory）、单机 4 卡、/schedule、/delete_instance、/reschedule_with_limits、/submit_task、/health、/instances、/cluster、/metrics | ✅ 已做 |
| **scheduling/utils_docker.py** | 本机子进程 + ACL 设限，无 Docker/SSH | **scheduling/utils_npu.py**：子进程启动、环境变量传 NPU_DEVICE_ID/CUBE_LIMIT/VECTOR_LIMIT/PORT/MODEL_PATH；MODEL_PATH 时启动 llm_inference_npu.py | ✅ 已做 |
| **scheduling/scaler.py** | 字段改为 cube/vector/memory，与 scheduler 对齐 | **scheduling/scaler_npu.py**：cube_requests/limits、vector_requests/limits、memory；2D co-scaling（垂直 reschedule_with_limits + 水平增删实例） | ✅ 已做 |

**配套已做：**

- **scheduling/npu/**：`acl_rt_wrapper.py`（ACL init_device/set_device_res_limit/init_context，无 ACL 时 stub）、`__init__.py`
- **scheduling/npu_resource.py**：910B3 常量（20 Cube、40 Vector、64GB/卡）
- **scheduling/scripts_demo/**：`npu_worker_entry.py`（ACL 设限 + 占位 /health、/predict）、`llm_inference_npu.py`（ACL 后加载 CausalLM、/health、/predict）、`run_demo_npu.py`、`run_demo_npu.sh`
- **README.md**：已增加 NPU 迁移小节与 demo 命令

---

## 二、新增核心组件（资源画像到资源分配链路）

| 组件 | 功能 | 状态 |
|------|------|------|
| **scheduling/resource_recommender.py** | 资源推荐引擎：基于资源画像数据，为调度器提供最优的 Cube/Vector/Memory 配置建议。支持根据SLA（延迟/吞吐）推荐资源、性价比推荐、扩容/缩容建议 | ✅ 已做 |
| **scheduling/scheduler_npu.py 新端点** | 新增 `/schedule_with_sla`、`/recommend_resources`、`/scaling_advice` 端点，实现基于SLA的智能调度 | ✅ 已做 |
| **adaptive_2D_scaling/horizontal_scaling/scheduler_npu_adapter.py** | 适配器模式：复用 scheduling/scheduler_npu.py 的核心逻辑，为 horizontal_scaling 组件提供兼容的接口 | ✅ 已做 |
| **adaptive_2D_scaling/horizontal_scaling/utils_npu_adapter.py** | 适配器模式：复用 scheduling/utils_npu.py 的实例管理功能 | ✅ 已做 |
| **adaptive_2D_scaling/horizontal_scaling/scaler_npu.py** | 基于NPU资源模型的自动扩缩容组件：支持2D协同缩放（垂直+水平）和基于资源画像的智能决策 | ✅ 已做 |
| **scheduling/simulations/workload/service_generator_npu.py** | NPU服务生成器：为NPU场景生成仿真服务工作负载，支持Cube/Vector资源规格和LLM推理服务 | ✅ 已做 |
| **scheduling/scripts_demo/test_resource_recommendation.py** | 测试脚本：验证资源推荐引擎API（/recommend_resources、/schedule_with_sla、/scaling_advice） | ✅ 已做 |

**资源画像到资源分配的完整链路：**

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  profile_npu.py │     │ resource_       │     │ scheduler_npu.py│
│   (资源画像)     │ ──→ │ recommender.py  │ ──→ │   (调度器)       │
│                 │     │  (资源推荐引擎)  │     │                 │
│ Cube/Vector网格 │     │ 根据SLA推荐     │     │ /schedule_with_ │
│ 测试延迟/吞吐   │     │ 最优资源配置    │     │    sla端点      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │
                              ↓
                       ┌─────────────────┐
                       │  scaler_npu.py  │
                       │   (自动缩放)     │
                       │                 │
                       │ /scaling_advice │
                       │  智能扩缩容建议   │
                       └─────────────────┘
```

---

## 三、未做 / 待补

### P1 优先级

| 清单项 | 说明 | 建议 | 状态 |
|--------|------|------|------|
| **adaptive_2D_scaling/horizontal_scaling/scheduler.py** | 仍为 GPU 类、utils_docker、多节点 | 已创建 scheduler_npu_adapter.py，horizontal_scaling 可直接使用 | ✅ 已提供适配器 |
| **adaptive_2D_scaling/horizontal_scaling/utils_docker.py** | 仍为 Docker+GPU+MPS | 已创建 utils_npu_adapter.py，horizontal_scaling 可直接使用 | ✅ 已提供适配器 |
| **adaptive_2D_scaling/horizontal_scaling/scaler.py** | 仍调用上述 GPU scheduler/utils_docker | 已创建 scaler_npu.py，支持NPU资源模型和智能缩放 | ✅ 已做 |
| **adaptive_2D_scaling/vertical_scaling/server/RCKM.c** | CUDA kernel 限流、与 client 通信 | NPU 无 kernel 拦截：方案 A 整块移除；方案 B 保留"槽位"仅与 ACL 设限联动（不拦截） | ⏳ 待补 |
| **adaptive_2D_scaling/vertical_scaling/client/**（hijack、cuda_originals、nvml_entry 等） | CUDA/NVML 劫持、LD_PRELOAD | NPU 路径不加载：删除或条件编译为 stub，NPU 部署时不使用 client | ⏳ 待补 |
| **scheduling/simulations/workload/service_generator.py** | 仍为 gpu_num、sm_requests/limits、memory | 已创建 service_generator_npu.py，支持NPU规格 | ✅ 已做 |
| **scheduling/simulations/baseline/scheduler_dilu.py** | 仍为 GPU 类、sm_*、多节点 | 与 scheduler_npu 对齐：NPU 资源模型（Cube+Vector）+ 单机 4 卡 | ⏳ 待补 |
| **scheduling/scripts_deploy/deploy_inference_funcs.py、deploy_train_funcs.py** | 仍为 Docker 镜像、num_gpus、sm_*、commands | 增加 NPU 路径：本机进程 + ACL 配置，请求字段改为 cube/vector/memory，或单独写 deploy_*_npu.py | ⏳ 待补 |
| **scheduling/scripts_tasks/run_*_INF*.py**（BERT、Roberta、GPT2、LLaMA2、Resnet152、VGG19 等） | torch.cuda、CUDA_VISIBLE_DEVICES、memory_allocated | 改为 torch_npu（或 ACL+CPU）+ NPU 设备与内存统计；或由 llm_inference_npu.py 覆盖 LLM 场景，其余脚本逐步 NPU 化 | ⏳ 待补 |
| **evaluation/scripts/run_*_INF*.py** | 与 scripts_tasks 同源 | 同上，与 scripts_tasks 保持一致 | ⏳ 待补 |

### P2 优先级

| 清单项 | 说明 | 建议 | 状态 |
|--------|------|------|------|
| **scheduling/simulations/baseline/scheduler_k8s.py** | 整卡调度、GPU/sm_* | 可选：整 NPU 卡基线或保留作 GPU 对比 | ⏳ 可选 |
| **scheduling/simulations/baseline/scheduler_infless_*.py** | InfLess 基线 | 可选：NPU 适配或仅保留 GPU 仿真 | ⏳ 可选 |
| **scheduling/scripts_tasks/dp_*.py、cv_models.py** | 部分 torch.cuda | 改为 NPU 后端（torch_npu），与 run_* 一致 | ⏳ 待补 |
| **evaluation/scripts/dp_*.py、cv_models.py** | 同上 | 同上 | ⏳ 待补 |
| **profiling/inference/methods/HGSS.sh、HGSS_simulation.sh** | MPS、SM、nvidia-cuda-mps-control | NPU 版已有 **profiling/npu/profile_npu.py、run_one_observation_npu.py**（ACL Cube/Vector 网格 + 吞吐/时延）；HGSS 可保留 GPU，或增加 NPU 分支调用 profiling/npu | ✅ 已做 |
| **profiling/inference/observations/scripts/*.py**（bert_base、gpt2_large 等） | torch + GPU | 设备与后端改为 NPU，与 profiling/npu 或 HGSS NPU 分支配合 | ⏳ 待补 |
| **profiling/training/binary-search-based-searching.sh** | 训练侧资源搜索 | 若有 GPU 假设则改为 NPU 资源与 ACL | ⏳ 待补 |
| **scheduling/simulations/workload/instances-*.txt** | 格式为 gpu_num、sm_*、memory | 可新生成 NPU 版 trace 或读取时映射 | ⏳ 待补 |
| **scheduling/simulations/comp_baselines.py** | 依赖 baseline 与 workload | 随 scheduler_dilu 与 workload NPU 化一起适配 | ⏳ 待补 |
| **README.md** | 主说明仍以 CUDA/Docker 为主 | 已加 NPU 小节；可选：单独"NPU 部署"章节（CANN/ACL、单容器 4 卡、无 Docker） | ✅ 已做 |

### 无需改

| 清单项 | 说明 |
|--------|------|
| **scheduling/data/calculate_threshold.py** | 无 GPU 依赖，可不改 |

---

## 四、按改造类型汇总

| 类型 | 已做 | 待补 |
|------|------|------|
| **替换（资源模型+运行时）** | scheduler_npu、utils_npu、scaler_npu、npu_resource、acl_rt_wrapper、resource_recommender、scheduler_npu_adapter、utils_npu_adapter、scaler_npu、service_generator_npu | scheduler_dilu；scripts_deploy；scripts_tasks/evaluation 的 run_*_INF*.py、dp_*、cv_models |
| **简化/移除** | - | RCKM、vertical client（NPU 路径移除或 stub） |
| **适配（设备/API）** | llm_inference_npu（LLM 一条龙）、profile_npu（资源画像） | profiling observations 脚本；training profiling |
| **可选/延后** | README NPU 小节 | K8s/InfLess 仿真、instances-*.txt、HGSS NPU 分支、training profiling |

---

## 五、推荐后续顺序

1. **✅ 已打通推理实例**：在真实 910B3 上确认 scheduler_npu + utils_npu + llm_inference_npu 全流程（端口、MODEL_PATH、ACL 顺序）；修复 demo 中"占位 worker 抢端口"等问题。
2. **✅ P1 horizontal**：已创建 scheduler_npu_adapter、utils_npu_adapter、scaler_npu，horizontal_scaling 可直接使用 NPU 组件。
3. **✅ P1 仿真**：已创建 service_generator_npu 支持 NPU 规格；待补 scheduler_dilu 改为 NPU 模型；可选 comp_baselines 读 NPU trace。
4. **P1 部署与任务脚本**：deploy_* 支持 NPU 注册与启动；run_*_INF*.py 增加 torch_npu/设备上报（或仅 LLM 走 llm_inference_npu）。
5. **P2**：profiling 与 observations NPU 化、K8s/InfLess 可选、README 细化。

当前 **P0 + P1核心已齐**，可在单机 4 卡 910B3 上跑"调度 + 本机任务 + ACL 限制 + 资源推荐"的完整流程；其余为功能补全与仿真/评估一致化。

---

## 六、关键新功能说明

### 6.1 资源推荐引擎 (resource_recommender.py)

**功能：**
- 基于资源画像数据，为不同模型和负载特征推荐最优资源配置
- 支持根据目标延迟推荐资源
- 支持根据目标吞吐推荐资源
- 支持根据成本预算推荐资源（性价比最优）
- 动态调整建议（基于实时性能反馈）

**使用示例：**
```python
from resource_recommender import get_recommender

recommender = get_recommender()

# 根据延迟SLA推荐
config = recommender.recommend_for_latency(target_latency=0.2, batch_size=1)
print(f"Recommended: Cube={config.cube}, Vector={config.vector}")

# 生成调度请求
request = recommender.generate_schedule_request(
    service_name='my-service',
    model_path='/vllm-workspace/models/Qwen3-4B',
    sla_latency=0.2
)
```

### 6.2 智能调度API

**新增端点：**
- `POST /schedule_with_sla` - 基于SLA的智能调度
- `POST /recommend_resources` - 获取资源配置建议
- `POST /scaling_advice` - 获取扩容/缩容建议

**使用示例：**
```bash
# 基于SLA调度
curl -X POST http://localhost:5000/schedule_with_sla \
  -H "Content-Type: application/json" \
  -d '{
    "service_name": "my-service",
    "model_path": "/vllm-workspace/models/Qwen3-4B",
    "sla_latency": 0.2
  }'

# 获取扩容建议
curl -X POST http://localhost:5000/scaling_advice \
  -H "Content-Type: application/json" \
  -d '{
    "current_cube": 8,
    "current_vector": 20,
    "current_latency": 0.5,
    "target_latency": 0.2
  }'
```

### 6.3 NPU Scaler (scaler_npu.py)

**功能：**
- 支持2D协同缩放（垂直+水平）
- 基于资源画像的智能缩放决策
- 性能监控和SLA感知
- 手动扩缩容API

**特性：**
- 垂直缩放：通过 `/reschedule_with_limits` 调整Cube/Vector限制
- 水平缩放：添加/删除实例
- 智能决策：优先垂直缩放，必要时水平缩放
- SLA感知：根据实际延迟与目标延迟的差距决策

---

**最后更新：** 2026-01-30
