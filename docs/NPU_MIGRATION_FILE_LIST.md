# Dilu GPU → 华为 910B3 NPU 迁移：文件改造清单

本文档基于对 Dilu 项目的扫描，列出所有与 GPU 相关的代码文件，并按**优先级**、**改造策略**、**难度**标注，供 NPU 迁移时使用。

---

## 一、项目目录结构概览

```
Dilu/
├── adaptive_2D_scaling/          # 2D 弹性伸缩
│   ├── horizontal_scaling/       # 水平伸缩：scheduler, scaler, utils_docker
│   └── vertical_scaling/         # 垂直伸缩：RCKM(C)、client(CUDA/NVML hijack)
├── evaluation/                   # 评估脚本（run_*_INF*.py 等，含 torch.cuda）
├── profiling/                     # 资源画像（HGSS shell 用 MPS/SM）
├── scheduling/                    # 核心调度平面
│   ├── scheduler.py               # 调度器 + GPU 资源模型 + Best-Fit/Worst-Fit
│   ├── scaler.py                  # 服务管理 + 扩缩容
│   ├── utils_docker.py            # Docker + GPU + MPS 启动/停止实例
│   ├── data/                      # 阈值计算（无 GPU 依赖）
│   ├── scripts_deploy/           # 部署推理/训练
│   ├── scripts_tasks/             # 任务脚本（run_*INF*.py 等，torch.cuda）
│   └── simulations/              # 仿真 baseline（scheduler_dilu/k8s、workload）
└── README.md
```

---

## 二、核心约束与改造原则

| 约束 | 说明 |
|------|------|
| 不创建新容器 | 单容器内 4 张 NPU，所有任务同进程/子进程，通过 ACL 做资源隔离 |
| 无 Kernel 拦截 | 无法实现 RCKM 式 CUDA kernel 拦截，用 ACL `set_device_res_limit` 做静态核心数/显存限制 |
| 资源模型变化 | GPU: SM(request/limit) + Memory → NPU: Cube Core + Vector Core + Memory（ACL 管控） |

---

## 三、文件改造清单（按优先级）

| 文件路径 | 当前功能 | GPU 依赖 | NPU 改造策略 | 难度 | 优先级 |
|----------|----------|----------|----------------|------|--------|
| `scheduling/scheduler.py` | 调度器：GPU 资源模型、Best-Fit/Worst-Fit、colocated 调度、/schedule、/delete_instance | GPU 类（total_sm, sm_req/lim, memory）、nodes_info 多机多卡 | 将 GPU 改为 NPU 资源模型（total_cube, total_vector, cube_req/lim, vector_req/lim, memory）；nodes_info 改为单机 4 卡；算法逻辑保留，仅资源维度和 can_allocate/calculate_score 适配 Cube+Vector | 中等 | P0 |
| `scheduling/utils_docker.py` | 通过 SSH 在远端执行 docker run，挂载 CUDA、传 MPS 与 sm_requests/limits 环境变量 | Docker、--gpus、CUDA_MPS_*、CUDA_VISIBLE_DEVICES、/usr/local/cuda | 删除 Docker/SSH；改为本机子进程启动 Python 任务，在子进程内先 acl.rt.set_device + set_device_res_limit(Cube/Vector) 再执行业务；环境变量改为 NPU 设备号与 core 限制 | 复杂 | P0 |
| `scheduling/scaler.py` | 服务注册、scale_out/scale_in、调用 scheduler /schedule、/delete_instance，Service 含 num_gpus/sm_* | 无直接 CUDA 调用，但字段为 num_gpus、sm_requests、sm_limits | 字段扩展或替换为 num_npus、cube_requests/limits、vector_requests/limits、memory；与 scheduler API 对齐；调度逻辑可复用 | 简单 | P0 |
| `adaptive_2D_scaling/horizontal_scaling/scheduler.py` | 与 scheduling/scheduler.py 几乎相同的调度逻辑与 GPU 模型 | 同 scheduling/scheduler.py | 与 scheduling/scheduler.py 统一改造：NPU 资源模型 + 单机 4 卡；或抽公共模块复用 | 中等 | P1 |
| `adaptive_2D_scaling/horizontal_scaling/utils_docker.py` | 与 scheduling/utils_docker.py 相同的 Docker+GPU+MPS 启动 | 同 scheduling/utils_docker.py | 与 scheduling/utils_docker.py 统一改为本机进程 + ACL 资源限制 | 复杂 | P1 |
| `adaptive_2D_scaling/horizontal_scaling/scaler.py` | 水平伸缩的 scaler，调用上述 scheduler/utils_docker | 同 scheduling/scaler | 同 scheduling/scaler：API 字段改为 NPU（cube/vector/memory） | 简单 | P1 |
| `adaptive_2D_scaling/vertical_scaling/server/RCKM.c` | RCKM 服务端：按 GPU 维度的 kernel 限流、request/limit 控制、与 client 通信 | CUDA 概念、GPU 设备、每 GPU 任务槽 | NPU 无法做 kernel 级拦截；方案 A：整块移除，仅靠 ACL 静态限制；方案 B：保留进程级“槽位”与 ACL 设限联动（仅设限，不拦截） | 复杂 | P1 |
| `adaptive_2D_scaling/vertical_scaling/client/src/container_client_hijack_call.c` | 容器内 CUDA/NVML 劫持，与 RCKM 通信做 kernel 限流 | CUDA API、NVML、LD_PRELOAD | NPU 无等价 hijack；删除或条件编译为 stub，NPU 路径不加载 | 复杂 | P1 |
| `adaptive_2D_scaling/vertical_scaling/client/src/cuda_originals.c` | CUDA 原始符号转发 | CUDA | NPU 路径不需要；保留仅用于 GPU 分支或移除 | 中等 | P1 |
| `adaptive_2D_scaling/vertical_scaling/client/src/nvml_entry.c` | NVML 符号与调用 | NVML | NPU 路径不需要；保留仅用于 GPU 分支或移除 | 中等 | P1 |
| `adaptive_2D_scaling/vertical_scaling/client/include/*.h`、`client-lib/*.so` | CUDA/NVML 头文件与库 | CUDA、NVML | NPU 构建不依赖；保留供 GPU 环境使用 | 简单 | P1 |
| `scheduling/simulations/workload/service_generator.py` | 生成实例规格：gpu_num、sm_requests/limits、memory | gpu_num、sm_* 字段 | 改为 npu_num、cube_requests/limits、vector_requests/limits、memory；或保留字段名在仿真层做“GPU 语义”映射到 NPU 数量与核心 | 简单 | P1 |
| `scheduling/simulations/baseline/scheduler_dilu.py` | 仿真 Dilu 调度逻辑（与 scheduler.py 一致） | GPU 类、sm_*、多节点 | 与 scheduling/scheduler.py 一致：NPU 资源模型（Cube+Vector）+ 单机 4 卡 | 中等 | P1 |
| `scheduling/simulations/baseline/scheduler_k8s.py` | K8s 基线（整卡调度） | GPU、sm_* | 可选：改为“整 NPU 卡”基线或保留作对比 | 简单 | P2 |
| `scheduling/simulations/baseline/scheduler_infless_l.py`、`scheduler_infless_r.py` | InfLess 基线 | GPU 资源 | 可选：适配 NPU 或仅保留 GPU 仿真 | 简单 | P2 |
| `scheduling/scripts_deploy/deploy_inference_funcs.py`、`deploy_train_funcs.py` | 部署推理/训练任务 | 可能含 Docker/镜像/GPU 假设 | 改为本机 NPU 进程启动与 ACL 配置，去掉 Docker/SSH 或保留为可选 | 中等 | P1 |
| `scheduling/scripts_tasks/run_BERT_INF_batch.py` 及同目录其他 `run_*_INF*.py` | 推理入口：torch.cuda、device、memory 上报 | torch.cuda.*、CUDA_VISIBLE_DEVICES | 改为 torch_npu 或 ACL + 裸推理；设备与内存统计改为 NPU API | 中等 | P1 |
| `evaluation/scripts/run_*_INF*.py`（同上） | 与 scripts_tasks 对应的评估脚本 | 同 run_*_INF*.py | 同 run_*_INF*.py 的 NPU 设备与内存改造 | 中等 | P1 |
| `scheduling/scripts_tasks/dp_*.py`、`cv_models.py` 等 | 模型定义与训练/推理逻辑 | 部分含 torch.cuda 注释或调用 | 改为 NPU 后端（torch_npu 或等效），保证与 run_* 一致 | 中等 | P2 |
| `evaluation/scripts/dp_*.py`、`cv_models.py` 等 | 同上 | 同上 | 同上 | 中等 | P2 |
| `profiling/inference/methods/HGSS.sh`、`HGSS_simulation.sh` | 资源画像：MPS、nvidia-cuda-mps-control、SM 率、batch 搜索 | CUDA_VISIBLE_DEVICES、MPS、SM | 改为 NPU：通过 ACL 设置不同 Cube/Vector 组合 + 测吞吐/时延，做 NPU 资源画像 | 复杂 | P2 |
| `profiling/inference/observations/scripts/*.py`（bert_base、gpt2_large 等） | 各模型 profiling 脚本 | 通常为 torch + GPU | 设备与后端改为 NPU，与 HGSS 脚本配合 | 中等 | P2 |
| `profiling/training/binary-search-based-searching.sh` | 训练侧资源搜索 | 可能含 GPU 假设 | 若有 GPU 相关逻辑则改为 NPU 资源与 ACL | 中等 | P2 |
| `scheduling/data/calculate_threshold.py` | 根据请求间隔计算 pre_warm/keep_alive | 无 GPU 依赖 | 可不改 | - | - |
| `scheduling/simulations/workload/instances-*.txt` | 预生成实例 trace（gpu_num、sm_*、memory） | 数据格式为 GPU 语义 | 可新生成 NPU 版 trace（npu_num、cube/vector、memory）或做读取时映射 | 简单 | P2 |
| `scheduling/simulations/comp_baselines.py` | 仿真对比入口 | 依赖上述 baseline scheduler 与 workload | 随 scheduler_dilu 与 workload 格式一起适配 NPU | 简单 | P2 |
| `README.md` | 项目说明、环境要求（CUDA、Docker、NVIDIA Driver） | 文档描述 GPU 栈 | 更新为 910B3 NPU、CANN/ACL、单容器 4 卡、无 Docker 部署说明 | 简单 | P2 |

---

## 四、按改造类型汇总

| 改造类型 | 文件 |
|----------|------|
| **替换**（逻辑保留、资源模型与运行时替换） | `scheduler.py`（两处）、`utils_docker.py`（两处）、`scaler.py`（两处）、`service_generator.py`、`scheduler_dilu.py` |
| **简化**（移除或降级为静态策略） | `RCKM.c`（仅保留与 ACL 设限联动或移除）、vertical_scaling/client 整套（删除或 stub） |
| **删除/条件编译** | Docker/SSH 启动路径、CUDA/NVML hijack、MPS 相关 |
| **适配**（设备与 API 替换） | 所有 `run_*_INF*.py`、`dp_*.py`、`cv_models.py`、profiling 脚本（torch.cuda → torch_npu/ACL） |
| **可选/延后** | K8s/InfLess 仿真、instances-*.txt 格式、README |

---

## 五、推荐实施顺序

1. **P0**：先做 `scheduler.py` 的 NPU 资源模型（NPU 类：cube/vector/memory）+ 单机 4 卡；再做 `utils_docker.py` 改为本机进程 + ACL 设限；最后对齐 `scaler.py` 的请求字段。
2. **P1**：统一 horizontal 与 scheduling 的 scheduler/utils_docker/scaler；处理 RCKM 与 client（删除或仅 ACL 联动）；仿真 workload 与 scheduler_dilu；部署脚本与 run_* 推理脚本改为 NPU。
3. **P2**：Profiling 脚本 NPU 化、仿真 baseline 与 trace、README 与文档。

完成第一步后，即可在单机 4 张 910B3 NPU 上跑起“调度 + 本机任务 + ACL 核心/显存限制”的简化 Dilu 流程，再逐步补齐监控与画像。
