"""
run_BERT_INF_batch.py  --  NPU版BERT推理服务（修复版）
修复点：
  1. args.device = 'npu:N' 替代 'cuda:N'
  2. /health 用 torch_npu.npu.memory_allocated
  3. thread_lock 提前定义，修复多线程竞争
  4. --simulate 模式无需model_name_or_path
"""
from flask import Flask, request, jsonify
import argparse
import torch
import threading
import queue
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', default='',
                    help='Path to BERT model (not needed in simulate mode)')
parser.add_argument('--device',   default=0,     type=int)
parser.add_argument('--port',     default=15000, type=int)
parser.add_argument('--simulate', action='store_true', default=False)
args = parser.parse_args()

NPU_AVAILABLE = False
try:
    import torch_npu
    torch_npu.npu.set_device(args.device)
    DEVICE_STR = f'npu:{args.device}'
    NPU_AVAILABLE = True
    print(f'[BERT] Using NPU: {DEVICE_STR}')
except (ImportError, Exception) as e:
    print(f'[BERT] torch_npu not available ({e}), using CPU')
    DEVICE_STR = 'cpu'

# ── 模型加载（仿真模式跳过）─────────────────────────────────────────────
model     = None
tokenizer = None
if not args.simulate:
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    if not args.model_name_or_path:
        raise ValueError('--model_name_or_path required in non-simulate mode')
    config    = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    model     = AutoModelForSequenceClassification.from_pretrained(
                    args.model_name_or_path, config=config)
    model     = model.to(DEVICE_STR)
    model.eval()
    print(f'[BERT] Model loaded on {DEVICE_STR}')
else:
    print('[BERT] Simulate mode: model not loaded')

app               = Flask(__name__)
thread_lock       = threading.Lock()
batch_queue       = []
batch_size        = 4
wait_time         = 0.02
wait_time_lock    = threading.Lock()
exec_time_list    = []
exec_time_list_lock = threading.Lock()
MONITOR_DURATION  = 1
MIN_WAITING_DURATION = 0.00001


def RPS_monitor():
    global wait_time
    while True:
        with exec_time_list_lock:
            exec_time_list.clear()
        time.sleep(MONITOR_DURATION)
        with exec_time_list_lock:
            if exec_time_list:
                rps = len(exec_time_list) / MONITOR_DURATION
                avg_et = sum(exec_time_list) / len(exec_time_list)
                with wait_time_lock:
                    wait_time = (MIN_WAITING_DURATION
                                 if avg_et < 1 / rps else avg_et / 10)


def batch_inference():
    global wait_time
    while True:
        if not batch_queue:
            time.sleep(0.001)
            continue
        t0 = time.time()
        while time.time() - t0 < wait_time and len(batch_queue) < batch_size:
            time.sleep(0.0001)
        with thread_lock:
            if not batch_queue:
                continue
            texts         = [item['text']         for item in batch_queue]
            result_queues = [item['result_queue'] for item in batch_queue]
            batch_queue.clear()
        # 编码 + 推理
        enc = tokenizer(texts, return_tensors='pt', truncation=True,
                        max_length=128, padding=True)
        enc = {k: v.to(DEVICE_STR) for k, v in enc.items()}
        with torch.no_grad():
            out   = model(**enc)
            preds = out.logits.cpu().numpy().tolist()
        for pred, rq in zip(preds, result_queues):
            rq.put({'prediction': pred})


def predict():
    if args.simulate:
        import random
        time.sleep(max(0.005, 0.022 + random.gauss(0, 0.004)))
        return jsonify({'prediction': [0.1, 0.9]})
    data = request.get_json() or {}
    text = data.get('text', 'default text for inference testing')
    st   = time.time()
    rq   = queue.Queue()
    with thread_lock:
        batch_queue.append({'text': text, 'result_queue': rq})
    pred = rq.get(timeout=10)
    et   = time.time()
    with exec_time_list_lock:
        exec_time_list.append(et - st)
    return jsonify({'prediction': pred})


@app.route('/health', methods=['GET'])
def health_check():
    if args.simulate:
        return jsonify({'status': 'healthy', 'device': 'simulate'}), 200
    if NPU_AVAILABLE:
        try:
            mem = torch_npu.npu.memory_allocated(args.device)
            if mem > 0.2 * 1024**3:
                return jsonify({'status': 'healthy', 'device': DEVICE_STR,
                                'mem_gb': round(mem/1024**3, 3)}), 200
            return jsonify({'status': 'unhealthy',
                            'reason': 'NPU memory not allocated'}), 503
        except Exception as e:
            return jsonify({'status': 'error', 'reason': str(e)}), 503
    # CPU fallback
    if model is not None:
        return jsonify({'status': 'healthy', 'device': 'cpu'}), 200
    return jsonify({'status': 'unhealthy', 'reason': 'model not loaded'}), 503


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    return predict()

@app.route('/shutdown', methods=['GET'])
def shutdown():
    os._exit(0)


if __name__ == '__main__':
    if not args.simulate:
        threading.Thread(target=batch_inference, daemon=True).start()
    threading.Thread(target=RPS_monitor, daemon=True).start()
    print(f'[BERT] Server starting on port {args.port}')
    app.run(debug=False, port=args.port, threaded=True)
