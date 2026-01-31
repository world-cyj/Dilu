#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPU LLM 推理服务：参考 Dilu run_LLaMA2_INF.py，在 ACL 设限后加载模型，提供 /health、/predict。
支持环境变量：PORT, NPU_DEVICE_ID, CUBE_LIMIT, VECTOR_LIMIT, MODEL_PATH。
设备优先：torch_npu > torch.cuda > cpu。
"""
import os
import sys
import time
import queue
import threading
import argparse
import traceback

# 调度目录加入 path，便于 import npu
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
        print("[llm_npu] ACL limits applied: device=%s cube=%s vector=%s" % (device_id, cube_limit, vector_limit), file=sys.stderr)
        return True
    except Exception as e:
        print("[llm_npu] ACL init skip: %s" % e, file=sys.stderr)
        return False

# 应用ACL限制
acl_applied = _apply_acl()

# 2) 导入PyTorch和Flask
try:
    import torch
    print(f"[llm_npu] PyTorch version: {torch.__version__}", file=sys.stderr)
    if hasattr(torch, 'npu'):
        print(f"[llm_npu] torch.npu available: {torch.npu.is_available()}", file=sys.stderr)
    if torch.cuda.is_available():
        print(f"[llm_npu] torch.cuda available", file=sys.stderr)
except Exception as e:
    print(f"[llm_npu] ERROR importing torch: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

try:
    from flask import Flask, request, jsonify
except Exception as e:
    print(f"[llm_npu] ERROR importing flask: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

# 3) 解析参数
parser = argparse.ArgumentParser(description="NPU LLM inference server (Dilu-style)")
parser.add_argument("--model_name_or_path", type=str, default="", help="Model path (or set MODEL_PATH)")
parser.add_argument("--port", type=int, default=0, help="Port (or set PORT)")
parser.add_argument("--device", type=str, default="0", help="Device id for cuda/npu")
args = parser.parse_args()

model_path = args.model_name_or_path or os.environ.get("MODEL_PATH", "")
port = args.port or int(os.environ.get("PORT", "15000"))
# 优先使用环境变量 NPU_DEVICE_ID，其次才是命令行参数
device_id = int(os.environ.get("NPU_DEVICE_ID", args.device or "0"))

print(f"[llm_npu] Configuration: model_path={model_path}, port={port}, device_id={device_id}", file=sys.stderr)

# 4) 加载模型
model = None
tokenizer = None
device = None

if not model_path or not os.path.isdir(model_path):
    print("[llm_npu] MODEL_PATH not set or invalid, /predict will return stub. Set MODEL_PATH for real inference.", file=sys.stderr)
    device = torch.device("cpu")
else:
    print(f"[llm_npu] Loading model from {model_path}...", file=sys.stderr)
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("[llm_npu] Loading tokenizer...", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        print("[llm_npu] Tokenizer loaded", file=sys.stderr)
        
        # 确定设备
        if hasattr(torch, "npu") and torch.npu.is_available():
            device = torch.device("npu:%d" % device_id)
            print(f"[llm_npu] Using NPU device: {device}", file=sys.stderr)
        elif torch.cuda.is_available():
            device = torch.device("cuda:%d" % device_id)
            print(f"[llm_npu] Using CUDA device: {device}", file=sys.stderr)
        else:
            device = torch.device("cpu")
            print(f"[llm_npu] Using CPU device", file=sys.stderr)
        
        print("[llm_npu] Loading model (this may take a while)...", file=sys.stderr)
        
        # 加载模型，使用float32以确保兼容性
        # Qwen3使用bfloat16，但NPU可能不完全支持，使用float32更安全
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
        )
        print(f"[llm_npu] Moving model to {device}...", file=sys.stderr)
        model = model.to(device)
        model.eval()
        print(f"[llm_npu] Model loaded with dtype: {model.dtype}", file=sys.stderr)
        print("[llm_npu] Model loaded successfully", file=sys.stderr)
        
    except Exception as e:
        print(f"[llm_npu] ERROR loading model: {e}", file=sys.stderr)
        traceback.print_exc()
        model = None
        tokenizer = None
        device = torch.device("cpu")

# 5) 创建Flask应用
app = Flask(__name__)
batch_queue = []
batch_size = 8
wait_time = 0.02
wait_time_lock = threading.Lock()
exec_time_list = []
exec_time_list_lock = threading.Lock()
thread_lock = threading.Lock()
MONITOR_DURATION = 1
MIN_WAITING_DURATION = 0.00001


def inference(sentences):
    if model is None or tokenizer is None:
        return ["[stub] no model loaded. Set MODEL_PATH." for _ in sentences]
    try:
        # 使用 chat template 格式化输入（如果是 chat 模型）
        formatted_sentences = []
        for text in sentences:
            if hasattr(tokenizer, 'apply_chat_template'):
                # Qwen3 等 Chat 模型使用 chat template
                messages = [{"role": "user", "content": text}]
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                formatted_sentences.append(formatted)
            else:
                # 非 Chat 模型直接使用输入
                formatted_sentences.append(text)
        
        with torch.no_grad():
            inp = tokenizer(formatted_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
            # 确保有 attention_mask
            if "attention_mask" not in inp:
                inp["attention_mask"] = torch.ones_like(inp["input_ids"])
            inp = {k: v.to(device) for k, v in inp.items()}
            
            # 设置生成参数
            # 获取正确的 eos_token_id
            eos_token_id = tokenizer.eos_token_id
            if hasattr(tokenizer, 'convert_tokens_to_ids'):
                im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
                if im_end_id is not None:
                    eos_token_id = im_end_id
            
            # 设置停止词
            stop_words = ["<|im_end|>", "<|endoftext|>"]
            stopping_criteria = None
            try:
                from transformers import StoppingCriteria, StoppingCriteriaList
                
                class StopOnTokens(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        for stop_word in stop_words:
                            stop_ids = tokenizer.convert_tokens_to_ids(stop_word)
                            if input_ids[0][-1] == stop_ids:
                                return True
                        return False
                
                stopping_criteria = StoppingCriteriaList([StopOnTokens()])
            except Exception as e:
                print(f"[llm_npu] Warning: Could not create stopping criteria: {e}", file=sys.stderr)
            
            generate_kwargs = {
                "input_ids": inp["input_ids"],
                "attention_mask": inp.get("attention_mask"),
                "max_new_tokens": 128,  # 限制生成长度
                "num_return_sequences": 1,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                "eos_token_id": eos_token_id,
                "use_cache": True,
                "do_sample": False,  # 贪婪解码，确定性输出
            }
            
            if stopping_criteria:
                generate_kwargs["stopping_criteria"] = stopping_criteria
            
            generated = model.generate(**generate_kwargs)
        
        # 解码并移除输入部分
        decoded = []
        for i, g in enumerate(generated):
            # 只保留生成的部分（跳过输入）
            input_length = inp["input_ids"][i].shape[0]
            output_tokens = g[input_length:]
            text = tokenizer.decode(output_tokens, skip_special_tokens=True)
            decoded.append(text)
        
        return decoded
    except Exception as e:
        print(f"[llm_npu] ERROR in inference: {e}", file=sys.stderr)
        traceback.print_exc()
        return [f"[error] {str(e)}" for _ in sentences]


def batch_inference_loop():
    print("[llm_npu] Batch inference loop started", file=sys.stderr)
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
        print(f"[llm_npu] Processing batch of {len(texts)} requests", file=sys.stderr)
        predictions = inference(texts)
        for pred, q in zip(predictions, result_queues):
            q.put(pred)


def rps_monitor():
    global wait_time
    print("[llm_npu] RPS monitor started", file=sys.stderr)
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
            print(f"[llm_npu] Error getting memory: {e}", file=sys.stderr)
        
        return jsonify({
            "status": "healthy", 
            "model_loaded": True, 
            "device": str(device), 
            "memory_gb": round(mem, 2)
        }), 200
    except Exception as e:
        print(f"[llm_npu] ERROR in health check: {e}", file=sys.stderr)
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
        
        return jsonify({"prediction": prediction, "latency": round(et - st, 4)}), 200
    except queue.Empty:
        print("[llm_npu] ERROR: Prediction timeout", file=sys.stderr)
        return jsonify({"error": "Prediction timeout"}), 504
    except Exception as e:
        print(f"[llm_npu] ERROR in predict: {e}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"[llm_npu] Starting server on port {port}", file=sys.stderr)
    
    # 启动后台线程
    threading.Thread(target=batch_inference_loop, daemon=True).start()
    threading.Thread(target=rps_monitor, daemon=True).start()
    
    print(f"[llm_npu] Server ready, model_loaded={model is not None}", file=sys.stderr)
    
    # 启动Flask服务
    try:
        app.run(debug=False, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"[llm_npu] ERROR starting server: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
