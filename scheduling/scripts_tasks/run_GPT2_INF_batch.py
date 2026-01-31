from flask import Flask, request, jsonify
import argparse
import torch
import os
import sys
import threading
import time
import queue

# 尝试导入torch_npu，如果不可用则使用ACL
try:
    import torch_npu
    NPU_AVAILABLE = True
except ImportError:
    try:
        import acl
        NPU_AVAILABLE = True
    except ImportError:
        NPU_AVAILABLE = False
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description="Flask server for running a transformers model")
parser.add_argument("--model_name_or_path", required=True, help="Path to pretrained model or model identifier from huggingface.co/models")
parser.add_argument("--device", default=0, type=int, help="Device to run the model on")
parser.add_argument("--port", default=15000, type=int, help="Port to run the Flask app on")
args = parser.parse_args()

app = Flask(__name__)

config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({'eos_token': '[EOS]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id=50256

# 设置设备
if NPU_AVAILABLE:
    if 'torch_npu' in sys.modules:
        args.device = "npu:" + str(args.device)
        model = model.to(args.device)
    else:
        # 使用ACL设置设备
        device_id = args.device
        acl.init()
        acl.rt.set_device(device_id)
        args.device = device_id
else:
    args.device = "cuda:" + str(args.device) if torch.cuda.is_available() else "cpu"
    model.to(args.device)
model.eval()

batch_queue = []
batch_size = 4  


wait_time = 0.02  
wait_time_lock = threading.Lock() 
exec_time_list = []
exec_time_list_lock = threading.Lock()
MONITOR_DURATION = 1 
MIN_WAITING_DURATION = 0.00001

def RPS_monitor():
    global wait_time, wait_time_lock, exec_time_list_lock, exec_time_list
    while True:
       
        with exec_time_list_lock:
            exec_time_list.clear()
        time.sleep(MONITOR_DURATION)
        # update wait_time
        with wait_time_lock and exec_time_list_lock:
            if len(exec_time_list)!=0:
                RPS = len(exec_time_list)/MONITOR_DURATION
                average_et = 0
                for et in exec_time_list:
                    average_et += et
                average_et /= len(exec_time_list)
                wait_time = MIN_WAITING_DURATION if average_et < 1/RPS else 1/10*(average_et) 
            


def batch_inference():
    while True:
        if len(batch_queue)!=0:
            start_time = time.time()
            global wait_time
            while time.time() - start_time < wait_time and len(batch_queue) < batch_size:
                time.sleep(MIN_WAITING_DURATION)  
            with thread_lock:
                texts_to_process = [item['text'] for item in list(batch_queue)]
                result_queues = [item['result_queue'] for item in list(batch_queue)]
                batch_queue.clear()

            predictions = inference(texts_to_process)
            for pred, res_queue in zip(predictions, result_queues):
                res_queue.put(pred)

def inference(texts):
    encoded_data = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=128,
        truncation=True,
        return_tensors='pt',
        padding='longest' 
    )
    input_ids = encoded_data['input_ids'].to(args.device)
    with torch.no_grad():
        outputs = model(input_ids)
    predicted_labels = torch.argmax(outputs.logits, dim=1).cpu().numpy().tolist()
    
    return predicted_labels

@app.route('/health', methods=['GET'])
def health_check():
    if NPU_AVAILABLE:
        if 'torch_npu' in sys.modules:
            # 使用torch_npu检查
            device = torch.device(args.device)
            allocated_memory = torch.npu.memory_allocated(device) / (1024 ** 3)  
            print("allocated_memory: ", allocated_memory)
            if allocated_memory > 3.1: 
                return jsonify({'status': 'healthy'}), 200
            else:
                return jsonify({'status': 'unhealthy', 'reason': 'Not enough NPU memory allocated'}), 503
        else:
            # 使用ACL检查
            try:
                device_id = args.device
                # 检查设备是否可用
                ret = acl.rt.device_reset(device_id)
                if ret == 0:
                    return jsonify({'status': 'healthy'}), 200
                else:
                    return jsonify({'status': 'unhealthy', 'reason': 'NPU device not available'}), 503
            except Exception as e:
                return jsonify({'status': 'unhealthy', 'reason': str(e)}), 503
    else:
        # 回退到GPU检查
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            torch.cuda.set_device(device)
            allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)  
            print("allocated_memory: ", allocated_memory)
            if allocated_memory > 3.1: 
                return jsonify({'status': 'healthy'}), 200
            else:
                return jsonify({'status': 'unhealthy', 'reason': 'Not enough GPU memory allocated'}), 503
        else:
            return jsonify({'status': 'unhealthy', 'reason': 'No acceleration device available'}), 503


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        st = time.time()
        data = request.json
        text = data['text']
        result_queue = queue.Queue()
        with thread_lock:
            batch_queue.append({'text': text, 'result_queue': result_queue})
        prediction = result_queue.get() 
        et = time.time()
        with exec_time_list_lock:
            exec_time_list.append(et-st)
        return jsonify({'prediction': prediction})

@app.route('/shutdown', methods=['GET'])
def shutdown():
    os._exit(0)

if __name__ == '__main__':
    thread_lock = threading.Lock()
    threading.Thread(target=batch_inference, daemon=True).start()
    threading.Thread(target=RPS_monitor, daemon=True).start()
    app.run(debug=False, port=args.port)
