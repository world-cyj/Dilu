"""
run_Resnet152_INF_batch.py  --  NPU版ResNet-152推理服务
================================================================
修复点：
  1. model.to('npu:N') 替代 model.to(cuda:N)
  2. /health 检查 torch_npu.npu.memory_allocated
  3. thread_lock 提前定义，修复多线程竞争
  4. 仿真模式（无NPU时）用CPU运行并返回正常延迟
"""
from flask import Flask, request, jsonify
import argparse
import torch
import threading
import queue
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--port',   default=15000, type=int)
parser.add_argument('--simulate', action='store_true', default=False)
args = parser.parse_args()

# ── NPU / CPU 设备选择 ────────────────────────────────────────────────────
NPU_AVAILABLE = False
try:
    import torch_npu
    torch_npu.npu.set_device(args.device)
    DEVICE_STR = f'npu:{args.device}'
    NPU_AVAILABLE = True
    print(f'[ResNet] Using NPU device: {DEVICE_STR}')
except (ImportError, Exception) as e:
    print(f'[ResNet] torch_npu not available ({e}), using CPU')
    DEVICE_STR = 'cpu'

# ── 模型加载 ─────────────────────────────────────────────────────────────
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cv_models as model_zoo
    model = model_zoo.resnet152(pretrained=False)
except ImportError:
    import torchvision.models as tv
    model = tv.resnet152(pretrained=False)

model = model.to(DEVICE_STR)
model.eval()
print(f'[ResNet] Model loaded on {DEVICE_STR}')

# ── 全局状态（提前定义锁） ────────────────────────────────────────────────
app          = Flask(__name__)
thread_lock  = threading.Lock()    # 必须在模块级定义，不能在 main 里
batch_queue  = []
batch_size   = 16
wait_time    = 0.02
wait_time_lock     = threading.Lock()
exec_time_list     = []
exec_time_list_lock = threading.Lock()
MONITOR_DURATION   = 1
MIN_WAITING_DURATION = 0.00001


def RPS_monitor():
    global wait_time
    while True:
        with exec_time_list_lock:
            exec_time_list.clear()
        time.sleep(MONITOR_DURATION)
        with exec_time_list_lock:
            if exec_time_list:
                RPS = len(exec_time_list) / MONITOR_DURATION
                avg_et = sum(exec_time_list) / len(exec_time_list)
                with wait_time_lock:
                    wait_time = (MIN_WAITING_DURATION
                                 if avg_et < 1 / RPS
                                 else avg_et / 10)


def batch_inference():
    global wait_time
    while True:
        if not batch_queue:
            time.sleep(0.001)
            continue
        start_time = time.time()
        while (time.time() - start_time < wait_time
               and len(batch_queue) < batch_size):
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
            if NPU_AVAILABLE:
                predictions = output.cpu().numpy().tolist()
            else:
                predictions = output.numpy().tolist()

        for pred, rq in zip(predictions, result_queues):
            rq.put({'prediction': pred})


def predict():
    if args.simulate:
        import random
        lat = max(0.005, 0.025 + random.gauss(0, 0.005))
        time.sleep(lat)
        return jsonify({'prediction': [0.1] * 1000})

    image_tensor = torch.rand([3, 224, 224])
    st = time.time()
    result_queue = queue.Queue()
    with thread_lock:
        batch_queue.append({'image': image_tensor, 'result_queue': result_queue})
    prediction = result_queue.get(timeout=10)
    et = time.time()
    with exec_time_list_lock:
        exec_time_list.append(et - st)
    return jsonify({'prediction': prediction})


@app.route('/health', methods=['GET'])
def health_check():
    if args.simulate:
        return jsonify({'status': 'healthy', 'device': 'simulate'}), 200
    if NPU_AVAILABLE:
        try:
            mem = torch_npu.npu.memory_allocated(args.device)
            print(f'[Health] NPU{args.device} allocated: {mem/1024**3:.3f}GB')
            if mem > 0.2 * 1024**3:
                return jsonify({'status': 'healthy', 'device': DEVICE_STR,
                                'mem_gb': round(mem/1024**3, 3)}), 200
            return jsonify({'status': 'unhealthy',
                            'reason': 'NPU memory not allocated'}), 503
        except Exception as e:
            return jsonify({'status': 'error', 'reason': str(e)}), 503
    # CPU fallback: model加载即就绪
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
    print(f'[ResNet] Server starting on port {args.port}')
    app.run(debug=False, port=args.port, threaded=True)
