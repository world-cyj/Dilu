"""
deploy_inference_funcs_npu.py  --  NPU 推理任务部署（进程级，无 Docker）
向 scaler_npu.py 注册推理服务，触发进程部署到 910B3 NPU 卡。

Models/paths based on /mnt/caoyujia/Dilu/evaluation/models/model_list.txt
"""

import json
import time
import requests

REGISTER_URL = 'http://127.0.0.1:14999/register_service'

# 模型路径（可按需修改为实际挂载路径）
MODEL_PATHS = {
    'resnet152':     'torchvision',
    'vgg19':         'torchvision',
    'bert':          '/mnt/caoyujia/models/bert-base-uncased',
    'roberta':       '/mnt/caoyujia/models/roberta-large',
    'gpt2':          '/mnt/caoyujia/models/gpt2-large',
    'llama2':        '/mnt/caoyujia/models/llama-2-7b-hf',
    'chatglm3':      '/mnt/caoyujia/models/chatglm3-6b',
}

EVAL_DIR = '/mnt/caoyujia/Dilu/evaluation/scripts'

# torch_npu / CANN 8.3 RC1 环境
# 模型路径以本地路径为准，resnet152/vgg19 用 torchvision 无需额外路径
inference_services = [
    {
        'service_name': 'resnet152-inf',
        'num_gpus':     1,
        # 从 HGSS 画像结果取：Vector-heavy
        'vector_req':   14,
        'cube_req':     3,
        'vector_lim':   17,
        'cube_lim':     5,
        'memory':       [4],
        'task_type':    'inference',
        'throughput':   60,
        'commands':     f'python {EVAL_DIR}/run_Resnet152_INF_batch.py',
        # torch_npu: ASCEND_RT_VISIBLE_DEVICES 由 scheduler_npu 注入
    },
    {
        'service_name': 'vgg19-inf',
        'num_gpus':     1,
        'vector_req':   15,
        'cube_req':     2,
        'vector_lim':   18,
        'cube_lim':     4,
        'memory':       [4],
        'task_type':    'inference',
        'throughput':   45,
        'commands':     f'python {EVAL_DIR}/run_VGG19_INF_batch.py',
    },
    {
        'service_name': 'bert-inf',
        'num_gpus':     1,
        'vector_req':   9,
        'cube_req':     5,
        'vector_lim':   12,
        'cube_lim':     7,
        'memory':       [5],
        'task_type':    'inference',
        'throughput':   48,
        'commands':     f'python {EVAL_DIR}/run_BERT_INF_batch.py '
                        f'--model_name_or_path {MODEL_PATHS["bert"]}',
    },
    {
        'service_name': 'roberta-inf',
        'num_gpus':     1,
        'vector_req':   9,
        'cube_req':     5,
        'vector_lim':   12,
        'cube_lim':     7,
        'memory':       [6],
        'task_type':    'inference',
        'throughput':   40,
        'commands':     f'python {EVAL_DIR}/run_Roberta_INF_batch.py '
                        f'--model_name_or_path {MODEL_PATHS["roberta"]}',
    },
    {
        'service_name': 'gpt2-inf',
        'num_gpus':     1,
        'vector_req':   10,
        'cube_req':     6,
        'vector_lim':   13,
        'cube_lim':     8,
        'memory':       [14],
        'task_type':    'llm-inference',
        'throughput':   12,
        'commands':     f'python {EVAL_DIR}/run_GPT2_INF_batch.py '
                        f'--model_name_or_path {MODEL_PATHS["gpt2"]}',
    },
    {
        'service_name': 'llama2-inf',
        'num_gpus':     1,
        'vector_req':   11,
        'cube_req':     7,
        'vector_lim':   14,
        'cube_lim':     9,
        'memory':       [26],
        'task_type':    'llm-inference',
        'throughput':   8,
        'commands':     f'python {EVAL_DIR}/run_LLaMA2_INF.py '
                        f'--model_name_or_path {MODEL_PATHS["llama2"]}',
    },
]


def deploy_all():
    for svc in inference_services:
        print(f'Deploying {svc["service_name"]}...')
        try:
            resp = requests.post(
                REGISTER_URL,
                data=json.dumps(svc),
                headers={'Content-Type': 'application/json'},
                timeout=15)
            if resp.status_code == 200:
                print(f'  OK: {resp.json()["message"]}')
            else:
                print(f'  FAIL {resp.status_code}: {resp.text[:80]}')
        except Exception as e:
            print(f'  ERROR: {e}')
        time.sleep(3)


if __name__ == '__main__':
    deploy_all()
