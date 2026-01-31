# Dilu NPU Demo 使用说明

本目录为 Dilu 从 GPU 迁移到华为 910B3 NPU 的**雏形实现**：单机 4 张 NPU、本机进程 + ACL 核心/显存管控、无 Docker。

## 环境要求

- 单容器内 4 张华为 910B3 NPU
- Python 3.7+
- Flask、requests
- 可选：ACL（华为 NPU 运行时），无 ACL 时 ACL 封装为 stub，调度与 HTTP 流程仍可跑通

## 目录与组件

| 路径 | 说明 |
|------|------|
| `scheduling/npu/` | ACL 封装：`set_device_res_limit`、`get_device_res_limit`、`reset_device_res_limit`、设备/上下文初始化 |
| `scheduling/npu_resource.py` | NPU 资源模型：单卡 20 Cube + 40 Vector + 64GB，单机 4 卡 |
| `scheduling/scheduler_npu.py` | NPU 调度器：Best-Fit/Worst-Fit，/schedule、/delete_instance、/health |
| `scheduling/utils_npu.py` | 实例生命周期：本机子进程 + 环境变量传设备与配额，写 pid 文件便于 stop |
| `scheduling/scaler_npu.py` | NPU 版 Scaler：注册服务字段为 cube/vector/memory，对接 scheduler_npu |
| `scheduling/scripts_demo/npu_worker_entry.py` | Demo worker：读环境变量 → 设置 ACL 限制 → 启动 /health、/predict HTTP 服务 |
| `scheduling/scripts_demo/llm_inference_npu.py` | **LLM 推理服务**：ACL 设限后加载 CausalLM，/health、/predict（batch 队列），参考 run_LLaMA2_INF.py |
| `scheduling/scripts_demo/run_demo_npu.sh` | 一键：启动调度器 → /schedule → 等 /health → /predict → /delete_instance |
| `profiling/npu/profile_npu.py` | **NPU 资源画像**：在 Cube/Vector 网格上跑观测，输出 JSON/CSV 曲线 |
| `profiling/npu/run_one_observation_npu.py` | 单次观测脚本（供 profile_npu 调起） |

## 快速跑 Demo

```bash
cd /vllm-workspace/Dilu/scheduling
export PYTHONPATH="$PWD:$PYTHONPATH"

# 1. 启动 NPU 调度器（默认 5000）
python scheduler_npu.py &

# 2. 提交一个推理实例（cube/vector 比例 0.25~0.75，8GB 显存）
curl -X POST http://127.0.0.1:5000/schedule -H "Content-Type: application/json" -d '{
  "num": 1,
  "cube_requests": 0.25,
  "cube_limits": 0.75,
  "vector_requests": 0.25,
  "vector_limits": 0.75,
  "memory": [8],
  "type": "inference",
  "service_name": "demo-inference",
  "image": "",
  "COMMAND": ""
}'
# 返回示例：{"instance_id":"xxx","port":15000,"selected_npus":[{"id":0,"ip":"127.0.0.1","index":0}]}

# 3. 等几秒后调 /health 和 /predict（将 PORT 换成上一步返回的 port）
curl http://127.0.0.1:15000/health
curl -X POST http://127.0.0.1:15000/predict -H "Content-Type: application/json" -d '{"input":"test"}'

# 4. 删除实例
curl -X POST http://127.0.0.1:5000/delete_instance -H "Content-Type: application/json" -d '{"instance_id":"上一步的 instance_id"}'
```

或使用脚本（会启动调度器、发一次 schedule、等 health、发 predict、删实例、停调度器）：

```bash
cd /vllm-workspace/Dilu/scheduling
chmod +x scripts_demo/run_demo_npu.sh
./scripts_demo/run_demo_npu.sh
```

## API 说明（NPU）

- **POST /schedule**  
  - 体：`num`, `cube_requests`, `cube_limits`, `vector_requests`, `vector_limits`, `memory`（列表或单值）, `type`（inference / llm-inference / training）, `service_name`, `image`, `COMMAND`  
  - 调度器将比例转为核心数（如 0.75 * 20 Cube、0.75 * 40 Vector），Best-Fit/Worst-Fit 分配单机 4 卡，并调用 `utils_npu.start_instance` 起子进程；子进程内通过环境变量得到设备与配额并设置 ACL。
  - 可选字段：`COMMAND`（启动自定义模型服务命令）、`MODEL_PATH`（模型路径）。**当仅设 `MODEL_PATH` 且未设 `COMMAND` 时，自动启动 `llm_inference_npu.py` 做真实 LLM 推理。**
- **POST /reschedule_with_limits**（IE 垂直伸缩）  
  - 体：`instance_id`, `new_cube_limits`, `new_vector_limits`, `service_name`, `type`, `memory`, `num`, `image`, `COMMAND`  
  - 先删原实例，再按新 request/limit 建新实例，实现 2D co-scaling 的垂直维。
- **POST /delete_instance**  
  - 体：`instance_id`  
  - 从资源池扣减并调用 `utils_npu.stop_instance`（按 pid 文件 kill 子进程）。
- **GET /health**  
  - 调度器自身健康检查。
- **GET /instances**  
  - 查看当前实例列表与状态（starting/running/unhealthy/stopped）。
- **GET /instance/<id>**  
  - 查看单个实例状态与资源占用。
- **GET /cluster**  
  - 查看集群资源快照（各卡 cube/vector/memory 使用）。
- **GET /metrics**  
  - 返回集群总利用率与快照。
- **POST /submit_task**  
  - 一次性：调度 + 启动实例 + 调用 `/predict` 并返回结果（同步）。
  - body: 与 `/schedule` 相同字段 + `payload`。
- **GET /task/<task_id>**  
  - 查询任务状态与结果。

## 与 GPU 版差异

- 无 Docker/SSH：实例为本机子进程，通过环境变量和 ACL 做资源隔离。
- 资源维度：GPU 的 SM request/limit 改为 NPU 的 Cube/Vector request/limit + Memory。
- 无 kernel 拦截：仅用 ACL `set_device_res_limit` 做静态核心数限制，在子进程内、业务算子前调用。

## NPU 资源画像（Profiling）

参考 Dilu `profiling/inference`（HGSS + observations），用 ACL 对 Cube/Vector 做核心限制并生成曲线：

```bash
cd /path/to/Dilu
export PYTHONPATH="$PWD/scheduling:$PYTHONPATH"
python profiling/npu/profile_npu.py --device 0 --out profiling/npu/out.json --out_csv profiling/npu/out.csv
# 指定模型做真实推理画像：--model_path /path/to/llama-2-7b-hf
```

详见 `profiling/npu/README.md`。

## IE 机制简化版（request/limit + 2D co-scaling）

- **request/limit**：调度与实例均使用 `cube_requests/limits`、`vector_requests/limits`、`memory`。
- **垂直伸缩**：`/reschedule_with_limits` 按新 limit 替换实例；Scaler 单实例时可 `scale_vertical_up` / `scale_vertical_down`（调整 limit 后 reschedule）。
- **水平伸缩**：`scale_out` / `scale_in` 增删实例。
- **2D 策略**（`scaler_npu.py`）：扩容时若仅 1 实例且请求持续偏高则先垂直扩，否则水平扩；缩容时先水平减实例，单实例且请求持续偏低则垂直缩。

## 真实 LLM 推理链路

- 调度时在 body 中设 **`MODEL_PATH`**（例如 `/vllm-workspace/models/xxx`），不设 `COMMAND` 时，`utils_npu` 会直接启动 **`llm_inference_npu.py`**，在 ACL 设限后加载 CausalLM（torch_npu / cuda / cpu），提供 `/health`、`/predict`（支持 `text` 或 `input` 字段，batch 队列同 Dilu run_LLaMA2_INF）。
- 若需自定义启动方式，可显式传 **`COMMAND`**（例如指定其他推理服务脚本）。

## 后续可做

- 将 profiling 曲线接入调度器，做 request/limit 推荐。
- 增加 Cube/Vector 利用率与显存监控，为调度与画像提供依据。
