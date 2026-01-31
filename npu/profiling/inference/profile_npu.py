# -*- coding: utf-8 -*-
"""
NPU 资源画像：在 (Cube, Vector) 网格上跑观测，生成 profiling 曲线 JSON/CSV，供调度与 request/limit 推荐使用。
参考 Dilu profiling/inference 的 HGSS 与 observations/profiling.sh，用 ACL 替代 MPS 做核心限制。
"""
import os
import sys
import json
import subprocess
import argparse

_dilu_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_script_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="NPU resource profiling: sweep Cube/Vector, output curve")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--model_path", type=str, default="", help="Optional model for real inference")
parser.add_argument("--cube_list", type=str, default="4,8,12,16,20", help="Comma-separated Cube limits")
parser.add_argument("--vector_list", type=str, default="10,20,30,40", help="Comma-separated Vector limits")
parser.add_argument("--batch_sizes", type=str, default="1,2,4", help="Comma-separated batch sizes")
parser.add_argument("--iters", type=int, default=15)
parser.add_argument("--out", type=str, default="", help="Output JSON path (default: profiling_npu_result.json)")
parser.add_argument("--out_csv", type=str, default="", help="Output CSV path")
args = parser.parse_args()

cube_list = [int(x) for x in args.cube_list.split(",")]
vector_list = [int(x) for x in args.vector_list.split(",")]
batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
iters = args.iters
device_id = args.device
model_path = args.model_path or os.environ.get("MODEL_PATH", "")
out_json = args.out or os.path.join(_script_dir, "profiling_npu_result.json")
out_csv = args.out_csv or os.path.join(_script_dir, "profiling_npu_result.csv")

env = os.environ.copy()
env["NPU_DEVICE_ID"] = str(device_id)
if model_path:
    env["MODEL_PATH"] = model_path

observations = []
lines_csv = ["batch_size,cube,vector,mean_latency,throughput"]

for batch_size in batch_sizes:
    for cube in cube_list:
        for vector in vector_list:
            cmd = [
                sys.executable,
                os.path.join(_script_dir, "run_one_observation_npu.py"),
                "--batch_size", str(batch_size),
                "--cube", str(cube),
                "--vector", str(vector),
                "--device", str(device_id),
                "--iters", str(iters),
            ]
            if model_path:
                cmd.extend(["--model_path", model_path])
            try:
                out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300, cwd=_dilu_root)
                line = (out.stdout or "").strip()
                if line and not line.startswith("#"):
                    parts = line.split(",")
                    if len(parts) >= 5:
                        observations.append({
                            "batch_size": int(parts[0]),
                            "cube": int(parts[1]),
                            "vector": int(parts[2]),
                            "mean_latency": float(parts[3]),
                            "throughput": float(parts[4]),
                        })
                        lines_csv.append(line)
            except subprocess.TimeoutExpired:
                observations.append({"batch_size": batch_size, "cube": cube, "vector": vector, "mean_latency": -1, "throughput": 0})
            except Exception as e:
                print("Run failed cube=%s vector=%s: %s" % (cube, vector, e), file=sys.stderr)

os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({"device_id": device_id, "model_path": model_path, "observations": observations}, f, indent=2)
if out_csv:
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_csv))
print("Wrote %s (%d points) and %s" % (out_json, len(observations), out_csv))
