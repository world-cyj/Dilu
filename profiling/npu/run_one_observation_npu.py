# -*- coding: utf-8 -*-
"""
单次 NPU 观测：在给定 Cube/Vector 限制下跑一轮推理或合成负载，输出 (batch_size, cube, vector, mean_latency, throughput)。
供 profile_npu.py 调起，用于绘制 Cube/Vector 资源画像曲线。
"""
import os
import sys
import time
import argparse

# 兼容从 profiling/npu 或项目根运行
_dilu_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_sched = os.path.join(_dilu_root, "scheduling")
for _p in [_dilu_root, _sched]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

parser = argparse.ArgumentParser(description="Single NPU observation for profiling")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--cube", type=int, default=10, help="Cube core limit")
parser.add_argument("--vector", type=int, default=20, help="Vector core limit")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--model_path", type=str, default="", help="Optional: model path for real inference")
parser.add_argument("--iters", type=int, default=20)
args = parser.parse_args()

batch_size = args.batch_size
cube_limit = args.cube
vector_limit = args.vector
device_id = args.device
model_path = args.model_path or os.environ.get("MODEL_PATH", "")
iters = args.iters

# 1) 设置 ACL 限制
try:
    from scheduling.npu.acl_rt_wrapper import (
        init_device,
        init_context,
        set_device_res_limit,
        ACL_RT_DEV_RES_CUBE_CORE,
        ACL_RT_DEV_RES_VECTOR_CORE,
    )
    ok, err = init_device(device_id)
    if not ok:
        print("init_device failed:", err, file=sys.stderr)
    ok, err = set_device_res_limit(device_id, ACL_RT_DEV_RES_CUBE_CORE, cube_limit)
    if not ok:
        print("set_device_res_limit CUBE failed:", err, file=sys.stderr)
    ok, err = set_device_res_limit(device_id, ACL_RT_DEV_RES_VECTOR_CORE, vector_limit)
    if not ok:
        print("set_device_res_limit VECTOR failed:", err, file=sys.stderr)
    ok, err = init_context(device_id)
    if not ok:
        print("init_context failed:", err, file=sys.stderr)
except Exception as e:
    print("ACL init exception:", e, file=sys.stderr)

# 2) 运行负载：有模型则推理，否则合成
start_time = time.time()
if model_path and os.path.isdir(model_path):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dev = "npu:%d" % device_id if hasattr(torch, "npu") and torch.npu.is_available() else "cuda:%d" % device_id if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path)
        if "npu" in dev:
            model = model.to("npu:%d" % device_id)
        elif "cuda" in dev:
            model = model.to("cuda:%d" % device_id)
        model.eval()
        texts = ["The quick brown fox jumps over the lazy dog."] * batch_size
        for _ in range(iters):
            with torch.no_grad():
                inp = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
                if "npu" in dev:
                    inp = {k: v.to("npu:%d" % device_id) for k, v in inp.items()}
                elif "cuda" in dev:
                    inp = {k: v.to("cuda:%d" % device_id) for k, v in inp.items()}
                model.generate(
                    inp["input_ids"],
                    attention_mask=inp.get("attention_mask"),
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
    except Exception as e:
        print("Model inference failed, using synthetic:", e, file=sys.stderr)
        # 合成：模拟计算时间与核心数负相关
        time.sleep(0.05 * (20.0 / max(1, cube_limit)) * (40.0 / max(1, vector_limit)) * batch_size * iters / 20.0)
else:
    # 合成负载：耗时与 cube/vector 成反比
    time.sleep(0.05 * (20.0 / max(1, cube_limit)) * (40.0 / max(1, vector_limit)) * batch_size * iters / 20.0)

end_time = time.time()
elapsed = end_time - start_time
mean_latency = elapsed / max(1, iters)
throughput = (iters * batch_size) / elapsed if elapsed > 0 else 0.0

# 与 Dilu 观测格式对齐：batch_size, cube, vector, mean_latency, throughput
print("%d,%d,%d,%.6f,%.4f" % (batch_size, cube_limit, vector_limit, mean_latency, throughput))
