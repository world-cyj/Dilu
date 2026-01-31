#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整系统启动脚本
启动调度器、监控Dashboard和预测性扩缩容服务
"""
import os
import sys
import time
import threading
import subprocess
import signal

# 确保 scheduling 在 path 中
sched_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sched_dir not in sys.path:
    sys.path.insert(0, sched_dir)

# 全局进程管理
processes = {}


def start_scheduler():
    """启动调度器"""
    print("[Launcher] Starting Scheduler...")
    from scheduler_npu import app
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)


def start_dashboard():
    """启动Dashboard"""
    print("[Launcher] Starting Dashboard...")
    os.environ["SCHEDULER_URL"] = "http://127.0.0.1:5000"
    os.environ["DASHBOARD_PORT"] = "8080"
    
    from dashboard import start_dashboard
    start_dashboard()


def start_predictive_scaler():
    """启动预测性扩缩容服务"""
    print("[Launcher] Starting Predictive Scaler...")
    from predictive_scaler import get_predictive_scaler
    
    scaler = get_predictive_scaler("http://127.0.0.1:5000")
    scaler.start()
    
    # 保持运行
    while True:
        time.sleep(60)
        status = scaler.get_status()
        print(f"[Scaler] Status: {status}")


def signal_handler(sig, frame):
    """信号处理"""
    print("\n[Launcher] Shutting down...")
    for name, proc in processes.items():
        print(f"[Launcher] Stopping {name}...")
        if hasattr(proc, 'terminate'):
            proc.terminate()
    sys.exit(0)


def main():
    """主函数"""
    print("=" * 60)
    print("NPU Scheduling System Launcher")
    print("=" * 60)
    print("\nStarting components:")
    print("  1. Scheduler (port: 5000)")
    print("  2. Dashboard (port: 8080)")
    print("  3. Predictive Scaler")
    print("\nPress Ctrl+C to stop all services")
    print("=" * 60 + "\n")
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动调度器线程
    scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
    scheduler_thread.start()
    
    # 等待调度器启动
    time.sleep(3)
    
    # 启动预测性扩缩容线程
    scaler_thread = threading.Thread(target=start_predictive_scaler, daemon=True)
    scaler_thread.start()
    
    # 启动Dashboard（主线程）
    try:
        start_dashboard()
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
