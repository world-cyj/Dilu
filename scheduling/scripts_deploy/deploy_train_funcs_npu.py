"""
deploy_train_funcs_npu.py  --  NPU 训练任务部署（进程级，无 Docker）
向 scaler_npu.py 注册训练任务，通过 scheduler_npu 启动进程。
使用 HCCL 替代 NCCL 进行多卡通信。
"""

import json
import time
import requests

REGISTER_URL = 'http://127.0.0.1:14999/register_service'
EVAL_DIR     = '/mnt/caoyujia/Dilu/evaluation/scripts'
MODEL_DIR    = '/mnt/caoyujia/models'

training_jobs = [
    {
        'service_name': 'dp-bert-training',
        'num_gpus':     4,
        'vector_req':   12,
        'cube_req':     6,
        'vector_lim':   16,
        'cube_lim':     8,
        'memory':       [8, 8, 8, 8],
        'task_type':    'training',
        'throughput':   0,
        # MindSpore DDP 替代 PyTorch DDP; HCCL 替代 NCCL
        'commands':     f'python {EVAL_DIR}/dp_bert.py '
                        '--nodes 1 --npus 4 --nr 0 '
                        '--iteration 600 --batch_size 192',
    },
    {
        'service_name': 'dp-roberta-training',
        'num_gpus':     2,
        'vector_req':   14,
        'cube_req':     8,
        'vector_lim':   18,
        'cube_lim':     10,
        'memory':       [10, 10],
        'task_type':    'training',
        'throughput':   0,
        'commands':     f'python {EVAL_DIR}/dp_roberta.py '
                        '--nodes 1 --npus 2 --nr 0 '
                        '--iteration 1000 --batch_size 64',
    },
    {
        'service_name': 'dp-resnet152-training',
        'num_gpus':     2,
        'vector_req':   16,
        'cube_req':     4,
        'vector_lim':   20,
        'cube_lim':     6,
        'memory':       [8, 8],
        'task_type':    'training',
        'throughput':   0,
        'commands':     f'python {EVAL_DIR}/dp_resnet152.py '
                        '--nodes 1 --npus 2 --nr 0 '
                        '--iteration 100 --batch_size 128',
    },
]


def deploy_all():
    for job in training_jobs:
        print(f'Deploying training job: {job["service_name"]}...')
        try:
            resp = requests.post(
                REGISTER_URL,
                data=json.dumps(job),
                headers={'Content-Type': 'application/json'},
                timeout=15)
            if resp.status_code == 200:
                print(f'  OK: {resp.json()["message"]}')
            else:
                print(f'  FAIL {resp.status_code}: {resp.text[:80]}')
        except Exception as e:
            print(f'  ERROR: {e}')
        time.sleep(5)


if __name__ == '__main__':
    deploy_all()
