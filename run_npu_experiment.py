#!/usr/bin/env python3
"""
run_npu_experiment.py  --  一键启动完整真实 NPU 实验
====================================================
按顺序执行所有阶段:
  Phase 1: HGSS 多维资源画像
  Phase 3: 启动调度器 + 扩缩容引擎 + 配额控制器
  Phase 4: 部署推理/训练任务 + 生成流量
  Phase 5: 评估指标采集

用法:
  python run_npu_experiment.py --simulate          # 仿真模式
  python run_npu_experiment.py --phase 1           # 仅画像
  python run_npu_experiment.py --phase 3 4 5       # 完整实验
  python run_npu_experiment.py                     # 全流程
"""

import argparse
import os
import subprocess
import sys
import time
import threading
import signal

ROOT = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--simulate',  action='store_true', default=False)
parser.add_argument('--phase',     nargs='+', type=int, default=[1,3,4,5])
parser.add_argument('--models',    nargs='+',
                    default=['resnet152', 'bert_base', 'gpt2_large'])
parser.add_argument('--duration',  type=int, default=60,
                    help='Evaluation duration (seconds)')
parser.add_argument('--rps',       type=float, default=10.0)
args = parser.parse_args()

SIM_FLAG = ['--simulate'] if args.simulate else []

background_procs = []


def _run(cmd, cwd=None, wait=True, label=''):
    print(f'\n[RUN] {label or " ".join(cmd[:3])}')
    print(f'      CMD: {" ".join(cmd)}')
    if wait:
        r = subprocess.run(cmd, cwd=cwd)
        return r.returncode
    else:
        p = subprocess.Popen(cmd, cwd=cwd)
        background_procs.append((label, p))
        return p


def _wait_port(port, timeout=30):
    import socket
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=1):
                print(f'  Port {port} ready')
                return True
        except OSError:
            time.sleep(1)
    print(f'  [WARN] Port {port} not ready after {timeout}s')
    return False


def phase1_profiling():
    """Phase 1: HGSS 多维资源画像"""
    print('\n' + '='*60)
    print('PHASE 1: Multi-Dimensional Resource Profiling')
    print('='*60)
    methods_dir = os.path.join(ROOT, 'profiling', 'inference', 'methods')
    for model in args.models:
        print(f'\n  Profiling model: {model}')
        cmd = [
            sys.executable,
            os.path.join(methods_dir, 'HGSS_NPU.py'),
            '--model', model,
            '--qos', '0.05',
            '--device', '0',
        ] + SIM_FLAG
        ret = _run(cmd, label=f'HGSS {model}')
        if ret != 0:
            print(f'  [WARN] HGSS {model} returned {ret}')
    print('\n[Phase 1] Done. Check profiling/inference/observations/profiling_results_npu/')


def phase3_control_plane():
    """Phase 3: 启动控制平面"""
    print('\n' + '='*60)
    print('PHASE 3: Starting Control Plane')
    print('='*60)
    sched_dir = os.path.join(ROOT, 'scheduling')

    # 1. 启动 scheduler_npu.py (port 5000)
    _run([sys.executable, os.path.join(sched_dir, 'scheduler_npu.py')],
         wait=False, label='scheduler_npu')
    _wait_port(5000, timeout=15)

    # 2. 启动 scaler_npu.py (port 14999)
    _run([sys.executable, os.path.join(sched_dir, 'scaler_npu.py')],
         wait=False, label='scaler_npu')
    _wait_port(14999, timeout=15)

    # 3. 启动 npu_quota_controller.py (后台)
    ctrl_cmd = [
        sys.executable,
        os.path.join(sched_dir, 'npu_quota_controller.py'),
        '--interval', '30',
        '--log_path', 'logs/dilu-scaler.log',
    ] + SIM_FLAG
    _run(ctrl_cmd, wait=False, label='quota_controller')
    time.sleep(2)
    print('[Phase 3] Control plane started.')


def phase4_deploy_and_traffic():
    """Phase 4: 部署任务 + 生成流量"""
    print('\n' + '='*60)
    print('PHASE 4: Deploying Services & Generating Traffic')
    print('='*60)
    deploy_dir = os.path.join(ROOT, 'scheduling', 'scripts_deploy')

    # 部署推理任务
    _run([sys.executable,
          os.path.join(deploy_dir, 'deploy_inference_funcs_npu.py')],
         label='deploy_inference', wait=True)
    time.sleep(5)

    # 部署训练任务（可选）
    _run([sys.executable,
          os.path.join(deploy_dir, 'deploy_train_funcs_npu.py')],
         label='deploy_training', wait=True)
    time.sleep(3)
    print('[Phase 4] Services deployed.')

    # 生成流量（后台）
    traffic_script = os.path.join(
        ROOT, 'scheduling', 'scripts_tasks', 'workload_generator',
        'req_generator_image_bursty.py')
    if os.path.exists(traffic_script):
        _run([sys.executable, traffic_script,
              '--gateway', 'http://127.0.0.1:14999',
              '--service', 'resnet152-inf'],
             wait=False, label='traffic_gen')
        print('[Phase 4] Traffic generator started.')
    else:
        print(f'[Phase 4] Traffic script not found: {traffic_script}')
        print('         Run manually or use npu_evaluator.py --simulate')


def phase5_evaluation():
    """Phase 5: 评估指标采集"""
    print('\n' + '='*60)
    print('PHASE 5: Evaluation & Metrics Collection')
    print('='*60)
    eval_script = os.path.join(
        ROOT, 'evaluation', 'scripts', 'npu_evaluator.py')

    services = ['resnet152-inf', 'bert-inf', 'gpt2-inf']
    for svc in services:
        cmd = [
            sys.executable, eval_script,
            '--service',    svc,
            '--duration',   str(args.duration),
            '--target_rps', str(args.rps),
            '--slo_ms',     '50',
            '--outdir',     os.path.join(ROOT, 'evaluation', 'logs'),
        ]
        if args.simulate:
            cmd.append('--simulate')
        _run(cmd, label=f'eval {svc}', wait=True)
        time.sleep(2)

    # 检查 quota controller 日志
    log_path = os.path.join(ROOT, 'scheduling', 'logs', 'dilu-scaler.log')
    if os.path.exists(log_path):
        print(f'\n[Phase 5] Scaler log tail ({log_path}):')
        lines = open(log_path).readlines()
        for l in lines[-20:]:
            print('  ', l.rstrip())
    print('[Phase 5] Evaluation done. Results in evaluation/logs/')


def cleanup(sig=None, frame=None):
    print('\n[Cleanup] Terminating background processes...')
    for label, p in background_procs:
        if p.poll() is None:
            p.terminate()
            print(f'  Stopped: {label}')
    sys.exit(0)


signal.signal(signal.SIGINT,  cleanup)
signal.signal(signal.SIGTERM, cleanup)

if __name__ == '__main__':
    print('\nDilu-NPU Real Experiment Runner')
    print(f'  Phases:   {args.phase}')
    print(f'  Models:   {args.models}')
    print(f'  Simulate: {args.simulate}')

    if 1 in args.phase:
        phase1_profiling()
    if 3 in args.phase:
        phase3_control_plane()
    if 4 in args.phase:
        phase4_deploy_and_traffic()
    if 5 in args.phase:
        phase5_evaluation()

    if background_procs:
        print('\n[Main] Background services running. Press Ctrl+C to stop.')
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            cleanup()
