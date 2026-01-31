# NPU 资源画像（Profiling）

参考 Dilu 的 `profiling/inference`（HGSS + observations），用 ACL 对 Cube/Vector 做核心限制，在网格上跑观测并生成 profiling 曲线，供调度与 request/limit 推荐使用。

## 用法

```bash
# 从 Dilu 根目录运行
cd /path/to/Dilu
export PYTHONPATH="$PWD/scheduling:$PYTHONPATH"

# 仅合成负载（无模型），快速得到曲线
python profiling/npu/profile_npu.py --device 0 --out profiling/npu/out.json --out_csv profiling/npu/out.csv

# 指定模型做真实推理画像（需 NPU/GPU 与 transformers）
python profiling/npu/profile_npu.py --device 0 --model_path /path/to/llama-2-7b-hf \
  --cube_list 4,8,12,16,20 --vector_list 10,20,30,40 --batch_sizes 1,2,4 --out out.json
```

## 输出

- **JSON**：`observations` 列表，每项含 `batch_size`, `cube`, `vector`, `mean_latency`, `throughput`。
- **CSV**：同上，便于画图或给调度器做 request/limit 推荐。

## 单次观测

```bash
python profiling/npu/run_one_observation_npu.py --batch_size 2 --cube 10 --vector 20 --device 0 --iters 20
# 输出：batch_size,cube,vector,mean_latency,throughput
```
