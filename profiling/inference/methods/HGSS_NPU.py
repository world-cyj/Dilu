"""
HGSS_NPU.py  --  Hierarchical Grid Search & Stepping for Ascend NPU
====================================================================
对标原 HGSS.sh，完全用 Python + ACL 接口重写，支持真实 NPU 和仿真模式。

两阶段搜索:
  Phase-1 (Init):   固定 BS=1，步进 Vector/Cube 配额，找到首个满足 QoS 的配置。
  Phase-2 (Step):   以满足QoS的配置为起点，BS 翻倍 + 动态调整配额，
                    追求 efficiency = throughput / (vector+cube) 最大。

与原 HGSS.sh 的关键差异:
  - CUDA_MPS_ACTIVE_THREAD_PERCENTAGE → acl.rt.set_device_res_limit()
  - SM% (0-100) → Vector(0-40) + Cube(0-20) 二维配额
  - nvidia-cuda-mps-control → 纯 Python ACL 调用

Usage:
  # 真实 NPU
  python HGSS_NPU.py --model resnet152 --device 0 --qos 0.04

  # 仿真
  python HGSS_NPU.py --model resnet152 --qos 0.04 --simulate
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime

parser = argparse.ArgumentParser(description='HGSS-NPU Profiling')
parser.add_argument('--model',   required=True,
                    choices=['resnet152','vgg','bert_base','gpt2_large',
                             'roberta_large','llama2'],
                    help='Model to profile')
parser.add_argument('--device',  type=int, default=0)
parser.add_argument('--qos',     type=float, default=0.05,
                    help='SLO latency (seconds)')
parser.add_argument('--iters',   type=int, default=50)
parser.add_argument('--outdir',  default=os.path.join(
                    os.path.dirname(__file__),
                    '..', 'observations', 'profiling_results_npu'))
parser.add_argument('--simulate',action='store_true', default=False)
args = parser.parse_args()

# ── 常量 ──────────────────────────────────────────────────────────────────
MAX_VECTOR = 40
MAX_CUBE   = 20
V_STEP     = 5      # Phase-1 步长
C_STEP     = 2
V_INIT     = 5
C_INIT     = 2

SCRIPT_DIR = os.path.join(os.path.dirname(__file__),
                           '..', 'observations', 'scripts')
# 模型脚本映射
MODEL_SCRIPTS = {
    'resnet152':     'resnet152.py',
    'vgg':           'vgg.py',
    'bert_base':     'bert_base.py',
    'gpt2_large':    'gpt2_large.py',
    'roberta_large': 'roberta_large.py',
    'llama2':        'llama2.py',
}
MODEL_PATHS = {
    'bert_base':     '/mnt/caoyujia/models/bert-base-uncased',
    'gpt2_large':    '/mnt/caoyujia/models/gpt2-large',
    'roberta_large': '/mnt/caoyujia/models/roberta-large',
    'llama2':        '/mnt/caoyujia/models/llama-2-7b-hf',
}

os.makedirs(args.outdir, exist_ok=True)

# ── ACL 初始化（真实模式）────────────────────────────────────────────────
ACL_AVAILABLE = False  # noqa: F841
if not args.simulate:
    try:
        import acl
        acl.init()
        acl.rt.set_device(args.device)
        ACL_AVAILABLE = True
        print(f'[HGSS] ACL initialized on device={args.device}')
    except ImportError:
        print('[HGSS] acl not available, switching to simulate mode')
        args.simulate = True


def set_quota(device_id, vector, cube):
    """调用 ACL 设置 Vector/Cube 算力配额。"""
    if ACL_AVAILABLE:
        import acl
        acl.rt.set_device_res_limit(device_id, 1, vector)  # VECTOR_CORE=1
        acl.rt.set_device_res_limit(device_id, 0, cube)    # CUBE_CORE=0
    # 模拟模式下无需真实调用


def run_model(batch_size, vector, cube):
    """
    调用模型推理脚本，返回 (elapsed_per_iter, throughput)。
    先通过 ACL 设置配额，再 subprocess 执行推理脚本。
    """
    set_quota(args.device, vector, cube)

    script = os.path.join(SCRIPT_DIR, MODEL_SCRIPTS[args.model])
    cmd = [
        sys.executable, script,
        '--device',     str(args.device),
        '--batch_size', str(batch_size),
        '--vector',     str(vector),
        '--cube',       str(cube),
        '--iters',      str(args.iters),
    ]
    if args.model in MODEL_PATHS:
        cmd += ['--model_name_or_path', MODEL_PATHS[args.model]]
    if args.simulate:
        cmd.append('--simulate')

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        line   = result.stdout.strip().split('\n')[-1]
        parts  = [float(x) for x in line.split(',')]
        # format: batch_size, vector, elapsed_per_iter, throughput
        return parts[2], parts[3]
    except Exception as e:
        print(f'  [ERR] run_model failed: {e}')
        return 9999.0, 0.0


def efficiency(throughput, vector, cube):
    return throughput / max(vector + cube, 1)


# ── Phase-1: 找到首个满足 QoS 的 (vector, cube, bs=1) ────────────────────
print(f'\n[HGSS] Phase-1: Find QoS-satisfying config '
      f'(model={args.model}, QoS={args.qos}s)')

best_v    = MAX_VECTOR
best_c    = MAX_CUBE
best_bs   = 1
best_eff  = 0.0
found_qos = False

v = V_INIT
while v <= MAX_VECTOR:
    c = max(C_INIT, v // 4)   # Cube 约为 Vector 的 1/4（NPU架构比例）
    c = min(c, MAX_CUBE)
    lat, thr = run_model(1, v, c)
    eff_val  = efficiency(thr, v, c)
    print(f'  Phase-1: V={v:2d} C={c:2d} BS=1  lat={lat:.4f}s  '
          f'thr={thr:.1f}  eff={eff_val:.3f}')

    if lat <= args.qos:
        best_v, best_c, best_bs = v, c, 1
        best_eff = eff_val
        found_qos = True
        print(f'  [HGSS] Phase-1 done: V={v} C={c} satisfies QoS')
        break
    v += V_STEP

if not found_qos:
    best_v, best_c = MAX_VECTOR, MAX_CUBE
    print(f'  [HGSS] Phase-1: Max quota still violates QoS, using V={best_v} C={best_c}')


# ── Phase-2: BS 翻倍，追求 efficiency 最大 ───────────────────────────────
print(f'\n[HGSS] Phase-2: Step up BS for max efficiency '
      f'(start V={best_v} C={best_c} BS={best_bs*2})')

results = []
batch_size  = best_bs * 2
last_eff    = best_eff
last_v      = best_v
last_c      = best_c

for _ in range(8):   # 最多翻倍8次
    lat, thr = run_model(batch_size, last_v, last_c)
    eff_val  = efficiency(thr, last_v, last_c)
    print(f'  Phase-2: V={last_v:2d} C={last_c:2d} BS={batch_size:3d}  '
          f'lat={lat:.4f}s  thr={thr:.1f}  eff={eff_val:.3f}')

    results.append({
        'model':      args.model,
        'batch_size': batch_size,
        'vector_req': last_v,
        'cube_req':   last_c,
        'vector_lim': min(last_v + 3, MAX_VECTOR),
        'cube_lim':   min(last_c + 2, MAX_CUBE),
        'latency_s':  round(lat, 6),
        'throughput': round(thr, 2),
        'efficiency': round(eff_val, 4),
        'qos_ok':     lat <= args.qos,
    })

    if lat > args.qos:
        # QoS 违约 → 尝试调大配额
        new_v = min(last_v + V_STEP, MAX_VECTOR)
        new_c = min(last_c + C_STEP, MAX_CUBE)
        if new_v == last_v and new_c == last_c:
            print('  [HGSS] Phase-2: Max quota reached, stop')
            break
        lat2, thr2 = run_model(batch_size, new_v, new_c)
        eff2 = efficiency(thr2, new_v, new_c)
        print(f'  Phase-2 retry: V={new_v} C={new_c} BS={batch_size}  '
              f'lat={lat2:.4f}s  eff={eff2:.3f}')
        if lat2 > args.qos:
            print('  [HGSS] Phase-2: Still violates after scale-up, stop')
            break
        if eff2 < last_eff - 0.05:
            print('  [HGSS] Phase-2: Efficiency dropped, stop')
            break
        last_v, last_c = new_v, new_c
        if eff2 > best_eff:
            best_v, best_c, best_bs = new_v, new_c, batch_size
            best_eff = eff2
    else:
        if eff_val > best_eff:
            best_v, best_c, best_bs = last_v, last_c, batch_size
            best_eff = eff_val
        last_eff = eff_val
        batch_size *= 2

# ── 保存结果 ──────────────────────────────────────────────────────────────
out_csv = os.path.join(args.outdir, f'{args.model}_hgss_npu.csv')
fieldnames = ['model','batch_size','vector_req','cube_req',
              'vector_lim','cube_lim','latency_s','throughput',
              'efficiency','qos_ok']
with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f'\n[HGSS] Best config: V={best_v} C={best_c} '
      f'BS={best_bs} eff={best_eff:.4f}')
print(f'[HGSS] Results: {out_csv}')
print(f'[HGSS] Total rows: {len(results)}')

# cleanup
if ACL_AVAILABLE:
    try:
        import acl
        acl.rt.reset_device(args.device)
        acl.finalize()
    except Exception:
        pass
