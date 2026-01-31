#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPU ResNet 推理服务：图像分类任务，支持ACL资源限制
参考 Dilu run_Resnet152_INF_batch.py，适配NPU环境
"""
import os
import sys
import time
import queue
import threading
import argparse
import traceback

# 调度目录加入 path
_sched = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _sched not in sys.path:
    sys.path.insert(0, _sched)

# 1) 在加载模型前设置 ACL 限制
def _apply_acl():
    device_id = int(os.environ.get("NPU_DEVICE_ID", "0"))
    cube_limit = int(os.environ.get("CUBE_LIMIT", "20"))
    vector_limit = int(os.environ.get("VECTOR_LIMIT", "40"))
    try:
        from npu.acl_rt_wrapper import (
            init_device,
            init_context,
            set_device_res_limit,
            ACL_RT_DEV_RES_CUBE_CORE,
            ACL_RT_DEV_RES_VECTOR_CORE,
        )
        init_device(device_id)
        set_device_res_limit(device_id, ACL_RT_DEV_RES_CUBE_CORE, cube_limit)
        set_device_res_limit(device_id, ACL_RT_DEV_RES_VECTOR_CORE, vector_limit)
        init_context(device_id)
        print("[resnet_npu] ACL limits applied: device=%s cube=%s vector=%s" % (device_id, cube_limit, vector_limit), file=sys.stderr)
        return True
    except Exception as e:
        print("[resnet_npu] ACL init skip: %s" % e, file=sys.stderr)
        return False

acl_applied = _apply_acl()

# 2) 导入PyTorch和Flask
try:
    import torch
    print(f"[resnet_npu] PyTorch version: {torch.__version__}", file=sys.stderr)
    if hasattr(torch, 'npu'):
        print(f"[resnet_npu] torch.npu available: {torch.npu.is_available()}", file=sys.stderr)
    if torch.cuda.is_available():
        print(f"[resnet_npu] torch.cuda available", file=sys.stderr)
except Exception as e:
    print(f"[resnet_npu] ERROR importing torch: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

try:
    from flask import Flask, request, jsonify
except Exception as e:
    print(f"[resnet_npu] ERROR importing flask: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

# 3) 解析参数
parser = argparse.ArgumentParser(description="NPU ResNet inference server (Dilu-style)")
parser.add_argument("--model_name", type=str, default="resnet152", 
                    choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", 
                             "vgg16", "vgg19", "alexnet", "inception_v3"],
                    help="Model name")
parser.add_argument("--port", type=int, default=0, help="Port (or set PORT)")
parser.add_argument("--device", type=str, default="0", help="Device id for cuda/npu")
args = parser.parse_args()

model_name = args.model_name or os.environ.get("MODEL_NAME", "resnet152")
port = args.port or int(os.environ.get("PORT", "15000"))
# 优先使用环境变量 NPU_DEVICE_ID，其次才是命令行参数
device_id = int(os.environ.get("NPU_DEVICE_ID", args.device or "0"))

print(f"[resnet_npu] Configuration: model={model_name}, port={port}, device_id={device_id}", file=sys.stderr)

# 4) 加载模型
model = None
device = None
num_classes = 1000

print(f"[resnet_npu] Loading {model_name} model...", file=sys.stderr)
try:
    # 从本地cv_models_npu导入
    from cv_models_npu import resnet18, resnet34, resnet50, resnet101, resnet152
    from cv_models_npu import vgg16, vgg19, alexnet, inception_v3
    
    model_zoo = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152,
        "vgg16": vgg16,
        "vgg19": vgg19,
        "alexnet": alexnet,
        "inception_v3": inception_v3,
    }
    
    model_fn = model_zoo.get(model_name, resnet152)
    model = model_fn(pretrained=False, num_classes=num_classes)
    
    # 确定设备
    if hasattr(torch, "npu") and torch.npu.is_available():
        device = torch.device("npu:%d" % device_id)
        print(f"[resnet_npu] Using NPU device: {device}", file=sys.stderr)
    elif torch.cuda.is_available():
        device = torch.device("cuda:%d" % device_id)
        print(f"[resnet_npu] Using CUDA device: {device}", file=sys.stderr)
    else:
        device = torch.device("cpu")
        print(f"[resnet_npu] Using CPU device", file=sys.stderr)
    
    model = model.to(device)
    model.eval()
    print("[resnet_npu] Model loaded successfully", file=sys.stderr)
    
except Exception as e:
    print(f"[resnet_npu] ERROR loading model: {e}", file=sys.stderr)
    traceback.print_exc()
    model = None
    device = torch.device("cpu")

# 5) 创建Flask应用
app = Flask(__name__)
batch_queue = []
batch_size = 16  # CV模型通常batch size更大
wait_time = 0.02
wait_time_lock = threading.Lock()
exec_time_list = []
exec_time_list_lock = threading.Lock()
thread_lock = threading.Lock()
MONITOR_DURATION = 1
MIN_WAITING_DURATION = 0.00001


def inference(image_tensors):
    """ResNet图像分类推理"""
    if model is None:
        return [{"label": 0, "confidence": 0.5} for _ in image_tensors]
    
    try:
        # 将图像堆叠成一个batch
        batch = torch.cat(image_tensors, dim=0).to(device)
        
        with torch.no_grad():
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=-1)
            predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
            confidences = probs.max(dim=1)[0].cpu().numpy().tolist()
        
        return [{"label": label, "confidence": round(conf, 4)} 
                for label, conf in zip(predicted_labels, confidences)]
    except Exception as e:
        print(f"[resnet_npu] ERROR in inference: {e}", file=sys.stderr)
        traceback.print_exc()
        return [{"label": 0, "confidence": 0.0, "error": str(e)} for _ in image_tensors]


def batch_inference_loop():
    print("[resnet_npu] Batch inference loop started", file=sys.stderr)
    while True:
        if len(batch_queue) == 0:
            time.sleep(MIN_WAITING_DURATION)
            continue
        start_time = time.time()
        while time.time() - start_time < wait_time and len(batch_queue) < batch_size:
            time.sleep(MIN_WAITING_DURATION)
        with thread_lock:
            items = list(batch_queue)
            batch_queue.clear()
        if not items:
            continue
        image_tensors = [x["image"] for x in items]
        result_queues = [x["result_queue"] for x in items]
        print(f"[resnet_npu] Processing batch of {len(image_tensors)} requests", file=sys.stderr)
        predictions = inference(image_tensors)
        for pred, q in zip(predictions, result_queues):
            q.put(pred)


def rps_monitor():
    global wait_time
    print("[resnet_npu] RPS monitor started", file=sys.stderr)
    while True:
        with exec_time_list_lock:
            exec_time_list.clear()
        time.sleep(MONITOR_DURATION)
        with wait_time_lock:
            with exec_time_list_lock:
                if exec_time_list:
                    rps = len(exec_time_list) / MONITOR_DURATION
                    avg_et = sum(exec_time_list) / len(exec_time_list)
                    wait_time = MIN_WAITING_DURATION if avg_et < 1 / rps else 0.1 * avg_et


@app.route("/health", methods=["GET"])
def health():
    try:
        if model is None:
            return jsonify({"status": "healthy", "model_loaded": False}), 200
        
        mem = 0
        try:
            if device.type == "npu" and hasattr(torch.npu, "memory_allocated"):
                mem = torch.npu.memory_allocated(device) / (1024 ** 3)
            elif device.type == "cuda":
                mem = torch.cuda.memory_allocated(device) / (1024 ** 3)
        except Exception as e:
            print(f"[resnet_npu] Error getting memory: {e}", file=sys.stderr)
        
        return jsonify({
            "status": "healthy", 
            "model_loaded": True, 
            "device": str(device), 
            "memory_gb": round(mem, 2),
            "model_name": model_name,
            "num_classes": num_classes
        }), 200
    except Exception as e:
        print(f"[resnet_npu] ERROR in health check: {e}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or {}
        
        # 支持两种输入格式：
        # 1. 直接提供image_tensor (list格式)
        # 2. 提供image_url或base64 (简化处理，实际使用时需要解码)
        
        # 简化版本：生成随机图像用于测试
        # 实际使用时应该从请求中解析真实图像
        import torch.nn.functional as F
        
        # 默认生成随机图像 [3, 224, 224]
        image_tensor = torch.rand(1, 3, 224, 224)
        
        result_queue = queue.Queue()
        with thread_lock:
            batch_queue.append({"image": image_tensor, "result_queue": result_queue})
        
        st = time.time()
        prediction = result_queue.get(timeout=120)
        et = time.time()
        
        with exec_time_list_lock:
            exec_time_list.append(et - st)
        
        return jsonify({
            "prediction": prediction, 
            "latency": round(et - st, 4),
            "model": model_name
        }), 200
    except queue.Empty:
        print("[resnet_npu] ERROR: Prediction timeout", file=sys.stderr)
        return jsonify({"error": "Prediction timeout"}), 504
    except Exception as e:
        print(f"[resnet_npu] ERROR in predict: {e}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/shutdown", methods=["GET"])
def shutdown():
    os._exit(0)


if __name__ == "__main__":
    print(f"[resnet_npu] Starting server on port {port}", file=sys.stderr)
    
    threading.Thread(target=batch_inference_loop, daemon=True).start()
    threading.Thread(target=rps_monitor, daemon=True).start()
    
    print(f"[resnet_npu] Server ready, model_loaded={model is not None}", file=sys.stderr)
    
    try:
        app.run(debug=False, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"[resnet_npu] ERROR starting server: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
