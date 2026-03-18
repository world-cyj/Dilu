"""
run_Resnet152_INF_batch.py  --  ResNet-152 推理服务 (torch_npu / CANN 8.3 RC1)
===============================================================================
替代原 CUDA 版本，使用 torch_npu 运行在昇腾 910B3 NPU 上。
动态 batching + RPS 自适应等待，保留原有 Flask API (/predict /health /shutdown)。
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
    from torch_npu.contrib import transfer_to_npu  # 自动算子适配
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False

import torchvision.models as tv_models

parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0,     type=int)
parser.add_argument('--port',   default=15000, type=int)
args = parser.parse_args()

# ── NPU 设备初始化 ────────────────────────────────────────────────────────
if NPU_AVAILABLE:
    npu_device = f'npu:{args.device}'
    torch_npu.npu.set_device(npu_device)
    # ACL 算力配额（由 scheduler_npu 通过 acl.rt.set_device_res_limit 统一管理）
else:
    npu_device = 'cpu'
    print('[WARN] torch_npu not available, running on CPU')

app = Flask(__name__)

# ── 模型加载 ──────────────────────────────────────────────────────────────
model = tv_models.resnet152(pretrained=False)
model = model.to(npu_device)
model.eval()
if NPU_AVAILABLE:
    model = torch_npu.npu.optimize(model) if hasattr(torch_npu.npu, 'optimize') else model

# ── 动态 Batching ─────────────────────────────────────────────────────────
batch_queue     = []
batch_size      = 16
wait_time       = 0.02
wait_time_lock  = threading.Lock()
exec_time_list  = []
exec_time_lock  = threading.Lock()
thread_lock     = threading.Lock()
MONITOR_DUR     = 1
MIN_WAIT        = 0.00001


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
                wait_time = MIN_WAIT if avg < 1/rps else avg / 10


def batch_inference():
    while True:
        if batch_queue:
            t0 = time.time()
            while time.time()-t0 < wait_time and len(batch_queue) < batch_size:
                time.sleep(0.0001)
            with thread_lock:
                imgs    = [item['image'].unsqueeze(0) for item in batch_queue]
                queues  = [item['result_queue']       for item in batch_queue]
                batch_queue.clear()
            data = torch.cat(imgs).to(npu_device)
            with torch.no_grad():
                out  = model(data)
                preds = out.cpu().numpy().tolist()
            for pred, q in zip(preds, queues):
                q.put({'prediction': pred})


@app.route('/predict', methods=['POST'])
def predict():
    # 接受 JSON {"data": [...]} 或直接生成随机输入（用于负载测试）
    img = torch.rand(3, 224, 224).to(npu_device)
    t0  = time.time()
    rq  = queue.Queue()
    with thread_lock:
        batch_queue.append({'image': img, 'result_queue': rq})
    result = rq.get()
    with exec_time_lock:
        exec_time_list.append(time.time() - t0)
    return jsonify(result)


@app.route('/health', methods=['GET'])
def health_check():
    if NPU_AVAILABLE:
        try:
            mem_used = torch_npu.npu.memory_allocated(args.device)
            mem_total = torch_npu.npu.get_device_properties(args.device).total_memory
            util = mem_used / max(mem_total, 1)
            # 模型已加载即视为健康（ResNet-152 约 240MB）
            if mem_used > 50 * 1024 * 1024:  # >50MB
                return jsonify({'status': 'healthy',
                                'npu_mem_used_gb': round(mem_used/1024**3, 3),
                                'npu_mem_util': round(util, 3)}), 200
            return jsonify({'status': 'unhealthy',
                            'reason': 'NPU memory too low'}), 503
        except Exception as e:
            return jsonify({'status': 'unhealthy', 'reason': str(e)}), 503
    # CPU fallback: 直接健康
    return jsonify({'status': 'healthy', 'device': 'cpu'}), 200


@app.route('/shutdown', methods=['GET'])
def shutdown():
    os._exit(0)


if __name__ == '__main__':
    threading.Thread(target=batch_inference, daemon=True).start()
    threading.Thread(target=rps_monitor,     daemon=True).start()
    print(f'[ResNet152] Serving on port {args.port}, device={npu_device}')
    app.run(host='0.0.0.0', port=args.port, debug=False)
