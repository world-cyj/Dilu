#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPU BERT 推理服务：文本分类任务，支持ACL资源限制
参考 Dilu run_BERT_INF_batch.py，适配NPU环境
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
        print("[bert_npu] ACL limits applied: device=%s cube=%s vector=%s" % (device_id, cube_limit, vector_limit), file=sys.stderr)
        return True
    except Exception as e:
        print("[bert_npu] ACL init skip: %s" % e, file=sys.stderr)
        return False

acl_applied = _apply_acl()

# 2) 导入PyTorch和Flask
try:
    import torch
    print(f"[bert_npu] PyTorch version: {torch.__version__}", file=sys.stderr)
    if hasattr(torch, 'npu'):
        print(f"[bert_npu] torch.npu available: {torch.npu.is_available()}", file=sys.stderr)
    if torch.cuda.is_available():
        print(f"[bert_npu] torch.cuda available", file=sys.stderr)
except Exception as e:
    print(f"[bert_npu] ERROR importing torch: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

try:
    from flask import Flask, request, jsonify
except Exception as e:
    print(f"[bert_npu] ERROR importing flask: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

# 3) 解析参数
parser = argparse.ArgumentParser(description="NPU BERT inference server (Dilu-style)")
parser.add_argument("--model_name_or_path", type=str, default="", help="Model path (or set MODEL_PATH)")
parser.add_argument("--port", type=int, default=0, help="Port (or set PORT)")
parser.add_argument("--device", type=str, default="0", help="Device id for cuda/npu")
args = parser.parse_args()

model_path = args.model_name_or_path or os.environ.get("MODEL_PATH", "bert-base-uncased")
port = args.port or int(os.environ.get("PORT", "15000"))
# 优先使用环境变量 NPU_DEVICE_ID，其次才是命令行参数
device_id = int(os.environ.get("NPU_DEVICE_ID", args.device or "0"))

print(f"[bert_npu] Configuration: model_path={model_path}, port={port}, device_id={device_id}", file=sys.stderr)

# 4) 加载模型
model = None
tokenizer = None
device = None
num_labels = 2  # 默认二分类

print(f"[bert_npu] Loading BERT model from {model_path}...", file=sys.stderr)
try:
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    
    print("[bert_npu] Loading config and tokenizer...", file=sys.stderr)
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    num_labels = config.num_labels if hasattr(config, 'num_labels') else 2
    
    print("[bert_npu] Loading model...", file=sys.stderr)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
    
    # 确定设备
    if hasattr(torch, "npu") and torch.npu.is_available():
        device = torch.device("npu:%d" % device_id)
        print(f"[bert_npu] Using NPU device: {device}", file=sys.stderr)
    elif torch.cuda.is_available():
        device = torch.device("cuda:%d" % device_id)
        print(f"[bert_npu] Using CUDA device: {device}", file=sys.stderr)
    else:
        device = torch.device("cpu")
        print(f"[bert_npu] Using CPU device", file=sys.stderr)
    
    model = model.to(device)
    model.eval()
    print("[bert_npu] Model loaded successfully", file=sys.stderr)
    
except Exception as e:
    print(f"[bert_npu] ERROR loading model: {e}", file=sys.stderr)
    traceback.print_exc()
    model = None
    tokenizer = None
    device = torch.device("cpu")

# 5) 创建Flask应用
app = Flask(__name__)
batch_queue = []
batch_size = 4
wait_time = 0.02
wait_time_lock = threading.Lock()
exec_time_list = []
exec_time_list_lock = threading.Lock()
thread_lock = threading.Lock()
MONITOR_DURATION = 1
MIN_WAITING_DURATION = 0.00001


def inference(texts):
    """BERT文本分类推理"""
    if model is None or tokenizer is None:
        return [{"label": 0, "confidence": 0.5} for _ in texts]
    
    try:
        encoded_data = tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=256,
            truncation=True,
            return_tensors='pt',
            padding='longest'
        )
        input_ids = encoded_data['input_ids'].to(device)
        attention_mask = encoded_data['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_labels = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            confidences = probs.max(dim=1)[0].cpu().numpy().tolist()
        
        return [{"label": label, "confidence": round(conf, 4)} 
                for label, conf in zip(predicted_labels, confidences)]
    except Exception as e:
        print(f"[bert_npu] ERROR in inference: {e}", file=sys.stderr)
        traceback.print_exc()
        return [{"label": 0, "confidence": 0.0, "error": str(e)} for _ in texts]


def batch_inference_loop():
    print("[bert_npu] Batch inference loop started", file=sys.stderr)
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
        texts = [x["text"] for x in items]
        result_queues = [x["result_queue"] for x in items]
        print(f"[bert_npu] Processing batch of {len(texts)} requests", file=sys.stderr)
        predictions = inference(texts)
        for pred, q in zip(predictions, result_queues):
            q.put(pred)


def rps_monitor():
    global wait_time
    print("[bert_npu] RPS monitor started", file=sys.stderr)
    while True:
        with exec_time_list_lock:
            exec_time_list.clear()
        time.sleep(MONITOR_DURATION)
        with wait_time_lock:
            with exec_time_list_lock:
                if exec_time_list:
                    rps = len(exec_time_list) / MONITOR_DURATION
                    avg_et = sum(exec_time_list) / len(exec_time_list)
                    wait_time = MIN_WAITING_DURATION if avg_et < 1 / rps else 0.125 * avg_et


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
            print(f"[bert_npu] Error getting memory: {e}", file=sys.stderr)
        
        return jsonify({
            "status": "healthy", 
            "model_loaded": True, 
            "device": str(device), 
            "memory_gb": round(mem, 2),
            "num_labels": num_labels
        }), 200
    except Exception as e:
        print(f"[bert_npu] ERROR in health check: {e}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or {}
        text = data.get("text") or data.get("input") or ""
        
        if not text:
            return jsonify({"error": "No text or input provided"}), 400
        
        result_queue = queue.Queue()
        with thread_lock:
            batch_queue.append({"text": text, "result_queue": result_queue})
        
        st = time.time()
        prediction = result_queue.get(timeout=120)
        et = time.time()
        
        with exec_time_list_lock:
            exec_time_list.append(et - st)
        
        return jsonify({
            "prediction": prediction, 
            "latency": round(et - st, 4),
            "model": "bert"
        }), 200
    except queue.Empty:
        print("[bert_npu] ERROR: Prediction timeout", file=sys.stderr)
        return jsonify({"error": "Prediction timeout"}), 504
    except Exception as e:
        print(f"[bert_npu] ERROR in predict: {e}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/shutdown", methods=["GET"])
def shutdown():
    os._exit(0)


if __name__ == "__main__":
    print(f"[bert_npu] Starting server on port {port}", file=sys.stderr)
    
    threading.Thread(target=batch_inference_loop, daemon=True).start()
    threading.Thread(target=rps_monitor, daemon=True).start()
    
    print(f"[bert_npu] Server ready, model_loaded={model is not None}", file=sys.stderr)
    
    try:
        app.run(debug=False, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"[bert_npu] ERROR starting server: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
