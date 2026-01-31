# 启用真实核心限制与 LLM 推理所需条件

当前 NPU 侧代码在**无 NPU/无 ACL** 时以 stub 运行；要启用**真实 ACL 核心限制**和**真实 LLM 推理**，需要满足下面条件并在正确位置配置。

---

## 一、真实核心限制（ACL）

### 1. 环境与依赖

| 项目 | 说明 |
|------|------|
| **硬件** | 华为 910B3 NPU，单机 4 张（或至少 1 张） |
| **CANN / 驱动** | 已安装与 910B3 匹配的 CANN 及驱动，设备在系统中可见 |
| **Python ACL 包** | 能成功 `import acl`（通常由 CANN 或 Ascend 软件包提供） |

代码里 `scheduling/npu/acl_rt_wrapper.py` 会执行：

```python
try:
    import acl
    _acl_rt = acl.rt
    _acl_available = True
except ImportError:
    pass
```

只要在**运行调度器、worker、llm_inference_npu 的 Python 环境**里能 `import acl`，就会走真实 ACL 路径；否则继续用 stub（不报错，但不做真实设限）。

### 2. 你需要提供的

- **运行环境**：在带 910B3 + CANN 的机器/容器内，用已配置好 CANN 的 Python 运行：
  - `scheduler_npu.py`
  - 由 `utils_npu` 拉起的 worker（`npu_worker_entry.py` 或 `llm_inference_npu.py`）
- **无需改代码**：无需在项目里写死设备或卡号；设备号由调度器分配，通过环境变量 `NPU_DEVICE_ID`、`CUBE_LIMIT`、`VECTOR_LIMIT` 传给子进程，子进程内再调 ACL。

### 3. 如何确认“真实核心限制”已启用

- 必须在 **scheduling 目录** 或已设置 **PYTHONPATH 包含 scheduling**，否则会导入到系统其它 `npu` 包导致 `ACL available: False`。在**同一环境**里运行：
  ```bash
  cd /path/to/Dilu/scheduling && PYTHONPATH=$PWD python3 -c "
  from npu.acl_rt_wrapper import is_acl_available, init_device, set_device_res_limit, init_context, ACL_RT_DEV_RES_CUBE_CORE, ACL_RT_DEV_RES_VECTOR_CORE
  print('ACL available:', is_acl_available())
  if is_acl_available():
      ok, err = init_device(0)
      if ok: ok, err = set_device_res_limit(0, ACL_RT_DEV_RES_CUBE_CORE, 10)
      if ok: ok, err = set_device_res_limit(0, ACL_RT_DEV_RES_VECTOR_CORE, 20)
      if ok: ok, err = init_context(0)
      print('set_device_res_limit(0, CUBE, 10):', ok, err)
  "
  ```
- 若输出 `ACL available: True` 且最后为 `(True, None)`，说明真实核心限制已启用。

### 4. ACL 约束与错误 507033

- **约束**：`set_device_res_limit` 必须在 **set_device 之后**、**create_context / 算子执行之前** 调用。
- 若返回 **507033**，多为调用顺序错误（例如进程内已先调 `create_context` 或 `torch.npu` 已初始化设备）。
- **正确顺序**：`init_device(device_id)` → `set_device_res_limit(cube)` → `set_device_res_limit(vector)` → `init_context(device_id)`。也可使用封装好的 `apply_device_res_limit(device_id, cube_limit, vector_limit)`。

---

## 二、真实 LLM 推理

### 1. 环境与依赖

| 项目 | 说明 |
|------|------|
| **模型路径** | 本地目录，内含 HuggingFace CausalLM 格式（如 `config.json`、`pytorch_model.bin` 或 safetensors 等） |
| **Python 包** | `torch`、`transformers`；NPU 上建议装 `torch_npu`（与当前 CANN 版本匹配） |
| **设备** | 优先：`torch_npu` 可用 → NPU；否则 `torch.cuda` → GPU；否则 CPU |

`llm_inference_npu.py` 中设备选择逻辑为：

```python
if hasattr(torch, "npu") and torch.npu.is_available():
    device = torch.device("npu:%d" % device_id)
elif torch.cuda.is_available():
    device = torch.device("cuda:%d" % device_id)
else:
    device = torch.device("cpu")
```

### 2. 你需要提供的

| 提供项 | 在哪里提供 | 说明 |
|--------|------------|------|
| **MODEL_PATH** | 调用 `/schedule` 或 `/submit_task` 时的 **body** | 模型所在目录的绝对路径，例如 `/vllm-workspace/models/llama-2-7b-hf` |
| **不设 COMMAND** | 同上 | 不传或留空时，调度器会用 `llm_inference_npu.py` 作为 worker，并自动把 `MODEL_PATH` 通过环境变量传给该进程 |
| **memory** | 同上 | 显存需求（GB），需能装下模型，例如 7B 常用 `[16]` 或更大 |

示例（提交时带上 MODEL_PATH）：

```bash
curl -X POST http://127.0.0.1:5000/schedule -H "Content-Type: application/json" -d '{
  "num": 1,
  "cube_requests": 0.25,
  "cube_limits": 0.75,
  "vector_requests": 0.25,
  "vector_limits": 0.75,
  "memory": [16],
  "type": "inference",
  "service_name": "llm-inference",
  "MODEL_PATH": "/vllm-workspace/models/llama-2-7b-hf"
}'
```

若你的模型在 `/vllm-workspace/models` 下某子目录，把 `MODEL_PATH` 换成该子目录路径即可。

### 3. 模型目录要求

- 目录存在且可读；
- 内含 `config.json` 以及权重文件（如 `pytorch_model.bin`、`model.safetensors` 等），以便 `AutoModelForCausalLM.from_pretrained(model_path)` 能正常加载。

### 4. 如何确认“真实 LLM 推理”已启用

- 提交一个带 `MODEL_PATH` 的 schedule，看 worker 日志（`logs/<service_name>-<instance_id>.log`）：
  - 若出现 `[llm_npu] MODEL_PATH not set or invalid` → 未用上 MODEL_PATH 或路径错误；
  - 若出现 `[llm_npu] ACL limits applied` 且无 “MODEL_PATH not set” → 已用上 MODEL_PATH；
  - 若加载成功，会加载模型并监听端口；此时调该实例的 `/predict`（body 里 `text` 或 `input`）应返回真实生成结果而非 stub。

---

## 三、汇总：你需要提供什么

| 目标 | 需要提供 |
|------|-----------|
| **真实核心限制** | ① 910B3 + CANN 环境；② 能 `import acl` 的 Python 环境；无需在代码里写死配置。 |
| **真实 LLM 推理** | ① 同上（若要在 NPU 上跑）；② 已安装 `torch`、`transformers`，NPU 上建议 `torch_npu`；③ 在 **/schedule 或 /submit_task 的 body 里提供 `MODEL_PATH`**（模型目录绝对路径）和合适的 **memory**；④ 不设 COMMAND，让系统自动用 `llm_inference_npu.py`。 |

按上述提供后，无需改现有 NPU 侧业务代码即可启用真实核心限制与 LLM 推理；若你希望用自定义推理服务，可传 **COMMAND** 覆盖默认的 `llm_inference_npu.py`。
