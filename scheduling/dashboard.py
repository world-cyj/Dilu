# -*- coding: utf-8 -*-
"""
NPU 调度监控 Dashboard
提供Web可视化界面，实时监控集群状态、服务性能、资源使用等
"""
import os
import sys
import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque

# Flask imports
from flask import Flask, render_template_string, jsonify, request
import requests

# 确保 scheduling 在 path 中
sched_dir = os.path.dirname(os.path.abspath(__file__))
if sched_dir not in sys.path:
    sys.path.insert(0, sched_dir)

app = Flask(__name__)

# 配置
SCHEDULER_URL = os.environ.get("SCHEDULER_URL", "http://127.0.0.1:5000")
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "8080"))

# 数据缓存
_cache = {
    "cluster_status": {},
    "instances": [],
    "metrics_history": deque(maxlen=1000),
    "last_update": 0,
}
_cache_lock = threading.Lock()


def fetch_scheduler_data():
    """从调度器获取数据"""
    try:
        # 获取集群状态
        r = requests.get(f"{SCHEDULER_URL}/cluster", timeout=5)
        if r.status_code == 200:
            with _cache_lock:
                _cache["cluster_status"] = r.json()
        
        # 获取实例列表
        r = requests.get(f"{SCHEDULER_URL}/instances", timeout=5)
        if r.status_code == 200:
            with _cache_lock:
                _cache["instances"] = r.json().get("instances", [])
        
        with _cache_lock:
            _cache["last_update"] = time.time()
            
    except Exception as e:
        print(f"[Dashboard] Error fetching data: {e}")


def background_updater():
    """后台数据更新线程"""
    while True:
        fetch_scheduler_data()
        time.sleep(5)  # 每5秒更新一次


# HTML模板
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NPU Cluster Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            line-height: 1.6;
        }
        
        .header {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 20px 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        
        .header h1 {
            font-size: 28px;
            font-weight: 600;
        }
        
        .header .subtitle {
            opacity: 0.9;
            font-size: 14px;
            margin-top: 5px;
        }
        
        .container {
            padding: 20px 30px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: #1e293b;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #334155;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        }
        
        .stat-card .label {
            font-size: 12px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .stat-card .value {
            font-size: 32px;
            font-weight: 700;
            margin-top: 8px;
            color: #f8fafc;
        }
        
        .stat-card .trend {
            font-size: 12px;
            margin-top: 5px;
        }
        
        .trend.up { color: #4ade80; }
        .trend.down { color: #f87171; }
        
        .section {
            background: #1e293b;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #334155;
        }
        
        .section h2 {
            font-size: 18px;
            margin-bottom: 15px;
            color: #f8fafc;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .section h2::before {
            content: '';
            width: 4px;
            height: 20px;
            background: #3b82f6;
            border-radius: 2px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th, td {
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #334155;
        }
        
        th {
            font-weight: 600;
            color: #94a3b8;
            font-size: 12px;
            text-transform: uppercase;
        }
        
        tr:hover {
            background: #334155;
        }
        
        .status {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }
        
        .status.running {
            background: rgba(74, 222, 128, 0.2);
            color: #4ade80;
        }
        
        .status.stopped {
            background: rgba(248, 113, 113, 0.2);
            color: #f87171;
        }
        
        .status.starting {
            background: rgba(251, 191, 36, 0.2);
            color: #fbbf24;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #334155;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-bar .fill {
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #60a5fa);
            transition: width 0.3s;
        }
        
        .progress-bar .fill.high {
            background: linear-gradient(90deg, #ef4444, #f87171);
        }
        
        .progress-bar .fill.medium {
            background: linear-gradient(90deg, #fbbf24, #f59e0b);
        }
        
        .npu-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }
        
        .npu-card {
            background: #0f172a;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #334155;
        }
        
        .npu-card h3 {
            font-size: 14px;
            margin-bottom: 10px;
            color: #94a3b8;
        }
        
        .metric-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 13px;
        }
        
        .metric-row .value {
            font-family: monospace;
            color: #60a5fa;
        }
        
        .refresh-info {
            text-align: center;
            color: #64748b;
            font-size: 12px;
            margin-top: 20px;
        }
        
        .chart-container {
            height: 200px;
            background: #0f172a;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .live-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #4ade80;
            border-radius: 50%;
            animation: pulse 2s infinite;
            margin-right: 8px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 NPU Cluster Dashboard</h1>
        <div class="subtitle">实时监控与资源管理</div>
    </div>
    
    <div class="container">
        <!-- 统计卡片 -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="label">Active NPU Devices</div>
                <div class="value" id="active-npus">-</div>
                <div class="trend">Available for scheduling</div>
            </div>
            <div class="stat-card">
                <div class="label">Running Instances</div>
                <div class="value" id="running-instances">-</div>
                <div class="trend">Active services</div>
            </div>
            <div class="stat-card">
                <div class="label">Total Requests</div>
                <div class="value" id="total-requests">-</div>
                <div class="trend up">↑ 12% from last hour</div>
            </div>
            <div class="stat-card">
                <div class="label">Avg Latency</div>
                <div class="value" id="avg-latency">-</div>
                <div class="trend down">↓ 5% from last hour</div>
            </div>
        </div>
        
        <!-- NPU资源状态 -->
        <div class="section">
            <h2>NPU Resource Status</h2>
            <div class="npu-grid" id="npu-grid">
                <!-- 动态生成 -->
            </div>
        </div>
        
        <!-- 运行中的实例 -->
        <div class="section">
            <h2>Running Instances</h2>
            <table>
                <thead>
                    <tr>
                        <th>Instance ID</th>
                        <th>Service</th>
                        <th>Status</th>
                        <th>NPU</th>
                        <th>Port</th>
                        <th>Resources</th>
                        <th>Latency</th>
                    </tr>
                </thead>
                <tbody id="instances-table">
                    <!-- 动态生成 -->
                </tbody>
            </table>
        </div>
        
        <div class="refresh-info">
            <span class="live-indicator"></span>
            Auto-refresh every 5 seconds | Last update: <span id="last-update">-</span>
        </div>
    </div>
    
    <script>
        // 格式化数字
        function formatNumber(num) {
            if (num === undefined || num === null) return '-';
            return num.toLocaleString();
        }
        
        // 格式化延迟
        function formatLatency(latency) {
            if (!latency) return '-';
            return (latency * 1000).toFixed(0) + 'ms';
        }
        
        // 获取状态样式
        function getStatusClass(status) {
            const map = {
                'running': 'running',
                'healthy': 'running',
                'stopped': 'stopped',
                'starting': 'starting',
            };
            return map[status] || 'stopped';
        }
        
        // 获取进度条颜色
        function getProgressClass(usage) {
            if (usage > 80) return 'high';
            if (usage > 50) return 'medium';
            return '';
        }
        
        // 更新NPU卡片
        function updateNPUGrid(npus) {
            const grid = document.getElementById('npu-grid');
            grid.innerHTML = '';
            
            npus.forEach(npu => {
                const card = document.createElement('div');
                card.className = 'npu-card';
                
                const cubeUsage = (npu.cube_used / npu.cube_total * 100).toFixed(1);
                const vectorUsage = (npu.vector_used / npu.vector_total * 100).toFixed(1);
                const memUsage = (npu.memory_used / npu.memory_total * 100).toFixed(1);
                
                card.innerHTML = `
                    <h3>NPU ${npu.id} (${npu.status})</h3>
                    <div class="metric-row">
                        <span>Cube Cores</span>
                        <span class="value">${npu.cube_used}/${npu.cube_total}</span>
                    </div>
                    <div class="progress-bar">
                        <div class="fill ${getProgressClass(cubeUsage)}" style="width: ${cubeUsage}%"></div>
                    </div>
                    <div class="metric-row" style="margin-top: 10px;">
                        <span>Vector Cores</span>
                        <span class="value">${npu.vector_used}/${npu.vector_total}</span>
                    </div>
                    <div class="progress-bar">
                        <div class="fill ${getProgressClass(vectorUsage)}" style="width: ${vectorUsage}%"></div>
                    </div>
                    <div class="metric-row" style="margin-top: 10px;">
                        <span>Memory</span>
                        <span class="value">${npu.memory_used}/${npu.memory_total} GB</span>
                    </div>
                    <div class="progress-bar">
                        <div class="fill ${getProgressClass(memUsage)}" style="width: ${memUsage}%"></div>
                    </div>
                    <div class="metric-row" style="margin-top: 10px;">
                        <span>Instances</span>
                        <span class="value">${npu.instances}</span>
                    </div>
                `;
                
                grid.appendChild(card);
            });
        }
        
        // 更新实例表格
        function updateInstancesTable(instances) {
            const tbody = document.getElementById('instances-table');
            tbody.innerHTML = '';
            
            instances.forEach(inst => {
                const row = document.createElement('tr');
                
                const resources = `C:${inst.cube_lim || '-'}/V:${inst.vector_lim || '-'}`;
                
                row.innerHTML = `
                    <td>${inst.instance_id.substring(0, 8)}...</td>
                    <td>${inst.service_name}</td>
                    <td><span class="status ${getStatusClass(inst.status)}">${inst.status}</span></td>
                    <td>${inst.device || '-'}</td>
                    <td>${inst.port || '-'}</td>
                    <td>${resources}</td>
                    <td>${formatLatency(inst.latency)}</td>
                `;
                
                tbody.appendChild(row);
            });
        }
        
        // 更新仪表板
        async function updateDashboard() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                // 更新统计卡片
                document.getElementById('active-npus').textContent = 
                    formatNumber(data.cluster.active_npus);
                document.getElementById('running-instances').textContent = 
                    formatNumber(data.cluster.running_instances);
                document.getElementById('total-requests').textContent = 
                    formatNumber(data.cluster.total_requests);
                document.getElementById('avg-latency').textContent = 
                    formatLatency(data.cluster.avg_latency);
                
                // 更新NPU网格
                if (data.cluster.npus) {
                    updateNPUGrid(data.cluster.npus);
                }
                
                // 更新实例表格
                if (data.instances) {
                    updateInstancesTable(data.instances);
                }
                
                // 更新时间
                document.getElementById('last-update').textContent = 
                    new Date().toLocaleTimeString();
                    
            } catch (error) {
                console.error('Error updating dashboard:', error);
            }
        }
        
        // 初始更新
        updateDashboard();
        
        // 定时更新
        setInterval(updateDashboard, 5000);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    """Dashboard主页"""
    return render_template_string(DASHBOARD_TEMPLATE)


@app.route("/api/status")
def api_status():
    """API: 获取集群状态"""
    with _cache_lock:
        cluster_status = _cache.get("cluster_status", {})
        instances = _cache.get("instances", [])
    
    # 处理集群数据
    npus = []
    active_npus = cluster_status.get("active", [])
    new_npus = cluster_status.get("new", [])
    
    for npu in active_npus:
        npus.append({
            "id": npu.get("id", "-"),
            "status": "active",
            "cube_used": npu.get("cube_used", 0),
            "cube_total": npu.get("cube_total", 20),
            "vector_used": npu.get("vector_used", 0),
            "vector_total": npu.get("vector_total", 40),
            "memory_used": npu.get("memory_used", 0),
            "memory_total": npu.get("memory_total", 64),
            "instances": npu.get("instances", 0),
        })
    
    for npu in new_npus:
        npus.append({
            "id": npu.get("id", "-"),
            "status": "idle",
            "cube_used": 0,
            "cube_total": npu.get("total_cube", 20),
            "vector_used": 0,
            "vector_total": npu.get("total_vector", 40),
            "memory_used": 0,
            "memory_total": npu.get("total_memory", 64),
            "instances": 0,
        })
    
    # 计算汇总数据
    running_instances = [i for i in instances if i.get("status") == "running"]
    
    # 计算平均延迟
    latencies = [i.get("latency", 0) for i in instances if i.get("latency")]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    return jsonify({
        "cluster": {
            "active_npus": len(active_npus),
            "new_npus": len(new_npus),
            "running_instances": len(running_instances),
            "total_instances": len(instances),
            "total_requests": sum(i.get("request_count", 0) for i in instances),
            "avg_latency": avg_latency,
            "npus": npus,
        },
        "instances": instances,
        "last_update": _cache.get("last_update", 0),
    })


@app.route("/api/instances")
def api_instances():
    """API: 获取实例列表"""
    with _cache_lock:
        instances = _cache.get("instances", [])
    return jsonify({"instances": instances})


@app.route("/api/npus")
def api_npus():
    """API: 获取NPU列表"""
    with _cache_lock:
        cluster_status = _cache.get("cluster_status", {})
    
    return jsonify({
        "active": cluster_status.get("active", []),
        "new": cluster_status.get("new", []),
    })


@app.route("/api/metrics/<instance_id>")
def api_metrics(instance_id):
    """API: 获取特定实例的指标"""
    try:
        # 从调度器获取实例信息
        r = requests.get(f"{SCHEDULER_URL}/instance/{instance_id}", timeout=5)
        if r.status_code != 200:
            return jsonify({"error": "Instance not found"}), 404
        
        instance = r.json()
        port = instance.get("port")
        
        if not port:
            return jsonify({"error": "Instance port not available"}), 400
        
        # 从实例获取健康数据
        r = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
        if r.status_code != 200:
            return jsonify({"error": "Instance health check failed"}), 500
        
        return jsonify(r.json())
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/scale", methods=["POST"])
def api_scale():
    """API: 手动扩缩容"""
    data = request.get_json()
    instance_id = data.get("instance_id")
    action = data.get("action")  # 'up' or 'down'
    
    if not instance_id or not action:
        return jsonify({"error": "instance_id and action required"}), 400
    
    try:
        # 获取当前配置
        r = requests.get(f"{SCHEDULER_URL}/instance/{instance_id}", timeout=5)
        if r.status_code != 200:
            return jsonify({"error": "Instance not found"}), 404
        
        instance = r.json()
        
        # 计算新配置
        current_cube = instance.get("cube_lim", 8)
        current_vector = instance.get("vector_lim", 20)
        
        if action == "up":
            new_cube = min(current_cube + 4, 20)
            new_vector = min(current_vector + 10, 40)
        else:
            new_cube = max(current_cube - 4, 4)
            new_vector = max(current_vector - 10, 10)
        
        # 执行重新调度
        payload = {
            "instance_id": instance_id,
            "new_cube_requests": new_cube / 20,
            "new_cube_limits": min(new_cube * 1.2 / 20, 1.0),
            "new_vector_requests": new_vector / 40,
            "new_vector_limits": min(new_vector * 1.2 / 40, 1.0),
        }
        
        r = requests.post(
            f"{SCHEDULER_URL}/reschedule_with_limits",
            json=payload,
            timeout=30
        )
        
        return jsonify(r.json()), r.status_code
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def start_dashboard():
    """启动Dashboard"""
    # 启动后台更新线程
    updater = threading.Thread(target=background_updater, daemon=True)
    updater.start()
    
    # 启动Flask应用
    print(f"[Dashboard] Starting on port {DASHBOARD_PORT}")
    print(f"[Dashboard] Scheduler URL: {SCHEDULER_URL}")
    app.run(host="0.0.0.0", port=DASHBOARD_PORT, debug=False)


if __name__ == "__main__":
    start_dashboard()
