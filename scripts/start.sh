#!/usr/bin/env bash
# CPUEdgeInference 服务启动（Linux / macOS）
#
# 可用环境变量：
#   MODEL_DIR        模型目录（默认 ./models）
#   HOST / PORT      监听地址（默认 127.0.0.1:8080）
#   BACKEND          mock（默认，零依赖）| llama-cpp（真实推理）
#   THREADS          推理线程数（0=auto）
#   MEMORY_BUDGET_MB 内存预算（0=自动取总内存 60%）
#   PREFER_QUANT     指定量化，如 Q4_K_M
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  PY="${PYTHON:-python3}"
fi

MODEL_DIR="${MODEL_DIR:-./models}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
BACKEND="${BACKEND:-mock}"
THREADS="${THREADS:-0}"
MEMORY_BUDGET_MB="${MEMORY_BUDGET_MB:-0}"
PREFER_QUANT="${PREFER_QUANT:-auto}"

mkdir -p "$MODEL_DIR"

echo "启动 CPUEdgeInference"
echo "  backend=$BACKEND  model-dir=$MODEL_DIR  http://$HOST:$PORT"
echo "  threads=$THREADS  memory-budget=${MEMORY_BUDGET_MB}MB  prefer-quant=$PREFER_QUANT"
echo

# 注意：全局选项必须写在子命令 serve 之前
exec "$PY" -m edgeinfer.cli \
  --model-dir "$MODEL_DIR" \
  --backend "$BACKEND" \
  --threads "$THREADS" \
  --memory-budget-mb "$MEMORY_BUDGET_MB" \
  --prefer-quant "$PREFER_QUANT" \
  serve --host "$HOST" --port "$PORT"
