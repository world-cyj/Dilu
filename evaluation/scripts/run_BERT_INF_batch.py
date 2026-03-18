"""
run_BERT_INF_batch.py  --  BERT-base 推理服务 (torch_npu / CANN 8.3 RC1)
========================================================================
替代原 CUDA 版本，使用 torch_npu 运行在昇腾 910B3 NPU 上。
动态 batching + transformers AutoModel。
"""

from flask import Flask, request, jsonify
import argparse
import torch
import threading
import queue
import time
import os

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', required=True)
parser.add_argument('--device', default=0,     type=int)
parser.add_argument('--port',   default=15002, type=int)
args = parser.parse_args()

if NPU_AVAILABLE:
    npu_device = f'npu:{args.device}'
    torch_npu.npu.set_device(npu_device)
else:
    npu_device = 'cpu'
    print('[WARN] torch_npu not available, running on CPU')

app = Flask(__name__)

# ── 模型加载 ──────────────────────────────────────────────────────────────
config    = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
model     = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path, config=config)
model     = model.to(npu_device)
model.eval()

# ── 动态 Batching ─────────────────────────────────────────────────────────
batch_queue    = []
batch_size     = 4
wait_time      = 0.02
wait_time_lock = threading.Lock()
exec_time_list = []
exec_time_lock = threading.Lock()
thread_lock    = threading.Lock()
MONITOR_DUR    = 1
MIN_WAIT       = 0.00001


def rps_monitor():
    global wait_time
    while True:
        with exec_time_lock:
            exec_time_list.clear()
        time.sleep(MONITOR_DUR)
        with exec_time_lock:
            if exec_time_list:
                rps = len(exec_time_list) / MONITOR_DUR
                avg = sum(exec_time_list) / len(exec_time_list)
                wait_time = MIN_WAIT if avg < 1/rps else avg / 6


def inference(texts):
    enc = tokenizer.batch_encode_plus(
        texts, add_special_tokens=True,
        max_length=256, truncation=True,
        return_tensors='pt', padding='longest')
    input_ids   = enc['input_ids'].to(npu_device)
    attn_mask   = enc['attention_mask'].to(npu_device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attn_mask)
    return torch.argmax(outputs.logits, dim=1).cpu().numpy().tolist()


def batch_inference():
    while True:
        if batch_queue:
            t0 = time.time()
            while time.time()-t0 < wait_time and len(batch_queue) < batch_size:
                time.sleep(MIN_WAIT)
            with thread_lock:
                texts  = [item['text']         for item in batch_queue]
                queues = [item['result_queue']  for item in batch_queue]
                batch_queue.clear()
            preds = inference(texts)
            for pred, q in zip(preds, queues):
                q.put(pred)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data.get('text', 'This is a test sentence for BERT inference.')
    t0   = time.time()
    rq   = queue.Queue()
    with thread_lock:
        batch_queue.append({'text': text, 'result_queue': rq})
    pred = rq.get()
    with exec_time_lock:
        exec_time_list.append(time.time() - t0)
    return jsonify({'prediction': pred})


@app.route('/health', methods=['GET'])
def health_check():
    if NPU_AVAILABLE:
        try:
            mem_used  = torch_npu.npu.memory_allocated(args.device)
            mem_total = torch_npu.npu.get_device_properties(args.device).total_memory
            # BERT-base 约 420MB
            if mem_used > 200 * 1024 * 1024:
                return jsonify({'status': 'healthy',
                                'npu_mem_used_gb': round(mem_used/1024**3, 3)}), 200
            return jsonify({'status': 'unhealthy',
                            'reason': 'NPU memory too low'}), 503
        except Exception as e:
            return jsonify({'status': 'unhealthy', 'reason': str(e)}), 503
    return jsonify({'status': 'healthy', 'device': 'cpu'}), 200


@app.route('/shutdown', methods=['GET'])
def shutdown():
    os._exit(0)


if __name__ == '__main__':
    thread_lock = threading.Lock()
    threading.Thread(target=batch_inference, daemon=True).start()
    threading.Thread(target=rps_monitor,     daemon=True).start()
    print(f'[BERT-base] Serving on port {args.port}, device={npu_device}')
    app.run(host='0.0.0.0', port=args.port, debug=False)
