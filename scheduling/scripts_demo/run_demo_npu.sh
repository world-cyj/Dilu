#!/bin/bash
# NPU Demo：启动调度器 + 注册服务 + 提交一次调度 + 调用 /predict 自测
# 在单机 4 张 NPU 环境下，从 scheduling 目录运行：./scripts_demo/run_demo_npu.sh

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"
SCHED_PORT=${SCHED_PORT:-5000}
SCALER_PORT=${SCALER_PORT:-14999}

echo "=== 1. 启动 NPU 调度器 (port $SCHED_PORT) ==="
python3 scheduler_npu.py &
SCHED_PID=$!
sleep 2
if ! kill -0 $SCHED_PID 2>/dev/null; then
  echo "Scheduler failed to start"
  exit 1
fi
echo "Scheduler PID: $SCHED_PID"

echo "=== 2. 提交 NPU 推理实例（cube/vector 比例 0.25~0.75）==="
curl -s http://127.0.0.1:$SCHED_PORT/health || true
REG=$(curl -s -X POST http://127.0.0.1:$SCHED_PORT/schedule -H "Content-Type: application/json" -d '{
  "num": 1,
  "cube_requests": 0.25,
  "cube_limits": 0.75,
  "vector_requests": 0.25,
  "vector_limits": 0.75,
  "memory": [8],
  "type": "inference",
  "service_name": "demo-npu-inference",
  "image": "",
  "COMMAND": ""
}')
echo "$REG"
INSTANCE_ID=$(echo "$REG" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('instance_id',''))" 2>/dev/null || true)
PORT=$(echo "$REG" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('port',''))" 2>/dev/null || true)
if [ -z "$INSTANCE_ID" ]; then
  echo "Schedule failed or no instance_id in response"
  kill $SCHED_PID 2>/dev/null
  exit 1
fi
echo "Instance: $INSTANCE_ID Port: $PORT"

echo "=== 3. 等待 worker /health 就绪 ==="
for i in 1 2 3 4 5 6 7 8 9 10; do
  if curl -s "http://127.0.0.1:$PORT/health" | grep -q ok; then
    echo "Worker ready."
    break
  fi
  sleep 1
done

echo "=== 4. 调用 /predict ==="
curl -s -X POST "http://127.0.0.1:$PORT/predict" -H "Content-Type: application/json" -d '{"input":"hello"}' || true
echo ""

echo "=== 5. 删除实例 ==="
curl -s -X POST http://127.0.0.1:$SCHED_PORT/delete_instance -H "Content-Type: application/json" -d "{\"instance_id\": \"$INSTANCE_ID\"}"
echo ""

echo "=== 6. 停止调度器 ==="
kill $SCHED_PID 2>/dev/null || true
echo "Demo done."
