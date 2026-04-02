"""
run_VGG19_INF_batch.py  --  NPU版VGG-19推理服务（修复版）
修复点：同ResNet-152，改用torch_npu，修复thread_lock定义位置
"""
from flask import Flask, request, jsonify
import argparse
import torch
import threading
import queue
import time
import os

parser = argparse.ArgumentParser()
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
    print(f'[VGG19] Using NPU: {DEVICE_STR}')
except (ImportError, Exception) as e:
    print(f'[VGG19] torch_npu not available ({e}), using CPU')
    DEVICE_STR = 'cpu'

try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cv_models as model_zoo
    model = model_zoo.vgg19(pretrained=False)
except ImportError:
    import torchvision.models as tv
    model = tv.vgg19(pretrained=False)

model = model.to(DEVICE_STR)
model.eval()
print(f'[VGG19] Model loaded on {DEVICE_STR}')

app               = Flask(__name__)
thread_lock       = threading.Lock()
batch_queue       = []
batch_size        = 16
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
            data_Xs       = [item['image'].unsqueeze(0) for item in batch_queue]
            result_queues = [item['result_queue']        for item in batch_queue]
            data_X = torch.cat(data_Xs)
            batch_queue.clear()
        with torch.no_grad():
            data_X = data_X.to(DEVICE_STR)
            output = model(data_X)
            preds  = output.cpu().numpy().tolist()
        for pred, rq in zip(preds, result_queues):
            rq.put({'prediction': pred})


def predict():
    if args.simulate:
        import random
        time.sleep(max(0.005, 0.028 + random.gauss(0, 0.005)))
        return jsonify({'prediction': [0.1]*1000})
    image_tensor = torch.rand([3, 224, 224])
    st = time.time()
    rq = queue.Queue()
    with thread_lock:
        batch_queue.append({'image': image_tensor, 'result_queue': rq})
    pred = rq.get(timeout=10)
    et = time.time()
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
    return jsonify({'status': 'healthy', 'device': 'cpu'}), 200


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    return predict()

@app.route('/shutdown', methods=['GET'])
def shutdown():
    os._exit(0)


if __name__ == '__main__':
    threading.Thread(target=batch_inference, daemon=True).start()
    threading.Thread(target=RPS_monitor,    daemon=True).start()
    print(f'[VGG19] Server starting on port {args.port}')
    app.run(debug=False, port=args.port, threaded=True)
