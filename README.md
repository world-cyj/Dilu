# Dilu
Dilu is a GPU-based serverless deep-learning system that realizes **Introspective Elasticity (IE)** — a fine-grained, two-dimensional co-scaling mechanism enabling GPU resourcing-on-demand.

> This repository contains a simplified reference implementation
> for the paper  
> **“Dilu: Enabling GPU Resourcing-on-Demand for Serverless DL Serving via Introspective Elasticity.”**

## Features
* **Multi-factor profiling** with efficient pruning search  
* **Resourcing-complementary scheduling** for high GPU utilization under QoS constraints  
* **Adaptive 2D co-scaling** (vertical & horizontal) with real-time decisions

## Requirements
* PyTorch 1.11
* DeepSpeed 0.11.1
* NCCL 2.10.3
* CUDA 11.7 + NVIDIA Driver 515.105.01
* Docker 24.0.5

## Usage
### Prerequisites
1. Install Docker
2. Install CUDA and NVIDIA Driver
3. Pull the basic images, such as `lvcunchi1999/torch110cu111_ddp:cluster`, `lvcunchi1999/torch110cu111_deepspeed:latest`

### Build Images and Run
See the README.md files of each subfolder for more details. The main steps are as follows:

0. Profiling to get the resource configurations.
1. Start the RCKM server on each node (see adaptive_2D_scaling/vertical_scaling/README.md).
2. Start the scaler and scheduler (see cluster_scheduling/README.md).
3. Deploy train/inference tasks.
4. Generate inference workloads.

---

## NPU 迁移（华为 910B3）

在单机 4 张 910B3 NPU、单容器内无 Docker 的场景下，提供 NPU 版调度与 demo 雏形：

- **资源模型**：每卡 20 Cube Core + 40 Vector Core + 64GB 显存，通过 ACL `set_device_res_limit` 做核心与显存管控。
- **调度**：`scheduling/scheduler_npu.py`（Best-Fit/Worst-Fit）、`scheduling/scaler_npu.py`，API 使用 `cube_requests/limits`、`vector_requests/limits`、`memory`。
- **实例启动**：本机子进程 + 环境变量传设备与配额，子进程内先 ACL 设限再启动 HTTP 服务（`scheduling/utils_npu.py`、`scheduling/scripts_demo/npu_worker_entry.py`）。

**快速跑 NPU Demo：**

```bash
cd scheduling && export PYTHONPATH=$PWD && python3 scripts_demo/run_demo_npu.py
```

详见 [docs/README_NPU_DEMO.md](docs/README_NPU_DEMO.md) 与 [docs/NPU_MIGRATION_FILE_LIST.md](docs/NPU_MIGRATION_FILE_LIST.md)。

## Citation

```bibtex
@inproceedings{lv2025dilu,
  title={Dilu: Enabling GPU Resourcing-on-Demand for Serverless DL Serving via Introspective Elasticity},
  author={Lv, Cunchi and Shi, Xiao and Lei, Zhengyu and Huang, Jinyue and Tan, Wenting and Zheng, Xiaohui and Zhao, Xiaofang},
  booktitle={Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1},
  pages={311--325},
  year={2025}
}
```
