# NPU 侧改造验证报告

以下验证均在项目根 / scheduling 下执行，**无真实 NPU 时 ACL 为 stub，调度与 HTTP 流程可照常跑通**。

---

## 1. 模块导入

| 检查项 | 命令/方式 | 结果 |
|--------|-----------|------|
| npu_resource, npu.acl_rt_wrapper, utils_npu, scheduler_npu | `python3 -c "from npu_resource import ...; from npu.acl_rt_wrapper import ...; import utils_npu; from scheduler_npu import app"` | ✅ 通过 |
| scaler_npu (ServiceManager, Service, Scaler) | `from scaler_npu import ServiceManager, Service, Scaler, app` | ✅ 通过 |
| llm_inference_npu (/health, /predict 路由) | 导入 `scripts_demo.llm_inference_npu`，检查 `app.url_map` | ✅ 通过 |

---

## 2. 调度与实例生命周期

| 检查项 | 结果 |
|--------|------|
| **run_demo_npu.py**：启动调度器 → /schedule → 等 /health → /predict → /delete_instance | ✅ 通过 |
| **run_demo_npu.sh**：同上（curl + 调度器后台） | ✅ 通过 |
| **POST /schedule**：返回 instance_id, port, selected_npus，NPU 0 分配 Cube 5/20, Vector 10/40, Mem 8/64 | ✅ 通过 |
| **POST /delete_instance**：释放资源，NPU 归零 | ✅ 通过 |
| Worker 进程：pid 写入 logs/*.pid，stop 时按 pid kill | ✅ 通过 |

---

## 3. 任务与监控 API

| 检查项 | 结果 |
|--------|------|
| **POST /submit_task**：同步调度+执行+返回 result（含 task_id, instance_id, result） | ✅ 通过 |
| **GET /instances**：返回实例列表（含 status 等） | ✅ 通过 |
| **GET /cluster**：返回 active/new NPU 快照 | ✅ 通过 |
| **GET /metrics**：返回 cluster + totals（cube/vector/memory_utilization） | ✅ 通过 |

---

## 4. IE 机制（request/limit + 2D co-scaling）

| 检查项 | 结果 |
|--------|------|
| **POST /reschedule_with_limits**：先删原实例，再按 new_cube_limits/new_vector_limits 建新实例 | ✅ 通过 |
| 新实例分配到不同 NPU、新 port，资源按新 limit 记账 | ✅ 通过 |
| **scaler_npu**：scale_vertical_up / scale_vertical_down 调用 /reschedule_with_limits | ✅ 代码就绪 |
| **Scaler.run()**：单实例先垂直扩/缩，再水平扩缩 | ✅ 代码就绪 |

---

## 5. 资源画像（Profiling）

| 检查项 | 结果 |
|--------|------|
| **run_one_observation_npu.py**：输出格式 `batch_size,cube,vector,mean_latency,throughput` | ✅ 通过（例：`1,10,20,0.010017,99.8283`） |
| **profile_npu.py**：小网格（1 cube, 1 vector, 1 batch）跑完并写入 JSON/CSV | ✅ 通过 |

---

## 6. LLM 推理链路

| 检查项 | 结果 |
|--------|------|
| **utils_npu**：当仅设 MODEL_PATH、未设 COMMAND 时，worker 使用 llm_inference_npu.py | ✅ 通过（日志含 `[llm_npu]`） |
| **llm_inference_npu.py**：无 MODEL_PATH 时 stub 模式，/health、/predict 路由存在 | ✅ 通过 |
| 有 MODEL_PATH 时：ACL 设限 → 加载 CausalLM → /health 返回 JSON，/predict 支持 text 或 input | ✅ 逻辑就绪（需真实模型路径与 torch_npu/cuda 环境） |

---

## 7. 快速自测命令汇总

```bash
# 从 scheduling 目录
cd /vllm-workspace/Dilu/scheduling
export PYTHONPATH="$PWD"

# 一键 Demo（推荐）
python3 scripts_demo/run_demo_npu.py

# 或 Shell 版
./scripts_demo/run_demo_npu.sh

# 单次观测（资源画像）
cd /vllm-workspace/Dilu && PYTHONPATH="$PWD/scheduling" python3 profiling/npu/run_one_observation_npu.py --batch_size 1 --cube 10 --vector 20 --iters 5

# 小网格画像
PYTHONPATH="$PWD/scheduling" python3 profiling/npu/profile_npu.py --device 0 --cube_list 10 --vector_list 20 --batch_sizes 1 --iters 3 --out /tmp/out.json
```

---

## 结论

当前 NPU 侧改造在**无 NPU/无 ACL** 环境下均可跑通：调度、实例启停、submit_task、监控 API、reschedule_with_limits、资源画像脚本、scaler_npu 与 llm_inference_npu 模块均验证通过或逻辑就绪。在具备 910B3 + ACL 与真实模型路径时，仅需安装 `acl` 与 `torch_npu`（或 cuda），即可启用真实核心限制与 LLM 推理。
