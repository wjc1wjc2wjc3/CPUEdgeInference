#!/usr/bin/env bash
# CPUEdgeInference 环境准备（Linux / macOS）
#
#   ./scripts/setup.sh                    # 仅创建虚拟环境（核心零依赖，无需装包）
#   INSTALL_EXTRAS=1 ./scripts/setup.sh   # 额外安装 llama-cpp-python（真实 CPU 推理）
#   NO_VENV=1 ./scripts/setup.sh          # 不建虚拟环境，直接用系统 Python
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || {
  echo "[x] 未找到 $PY，请先安装 Python 3.9+"; exit 1
}
"$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' || {
  echo "[x] 需要 Python 3.9+（当前：$("$PY" --version 2>&1)）"; exit 1
}

if [ "${NO_VENV:-0}" != "1" ]; then
  echo "[1/3] 创建虚拟环境 .venv"
  "$PY" -m venv .venv
  # shellcheck disable=SC1091
  . .venv/bin/activate
  PY=python
else
  echo "[1/3] 跳过虚拟环境（NO_VENV=1）"
fi

echo "[2/3] 升级 pip"
"$PY" -m pip install --upgrade pip -q

if [ "${INSTALL_EXTRAS:-0}" = "1" ]; then
  echo "[3/3] 安装 llama-cpp-python（真实 CPU 推理）"
  # 优先使用预编译 wheel；从源码编译时给出可用提示
  case "$(uname -s)" in
    Darwin) export CMAKE_ARGS="-DLLAMA_METAL=on" ;;   # macOS：即便只用 CPU 也建议带 Metal 构建
  esac
  if ! "$PY" -m pip install llama-cpp-python; then
    echo "[!] 预编译包安装失败，尝试从源码编译（需要 cmake / C++ 编译器）："
    echo "    CMAKE_ARGS=\"$CMAKE_ARGS\" $PY -m pip install llama-cpp-python --force-reinstall --no-cache-dir"
    exit 1
  fi
else
  echo "[3/3] 跳过可选依赖（核心零依赖，mock 后端可直接跑通链路）"
fi

mkdir -p ./models
echo
echo "完成。接下来："
echo "  ./scripts/start.sh                       # 启动服务（默认 mock 后端）"
echo "  BACKEND=llama-cpp ./scripts/start.sh     # 用真实模型启动"
echo "  ./scripts/start.sh --help"
echo
echo "把 GGUF 权重放进 ./models 即可（本项目不联网下载）。"
if [ "${NO_VENV:-0}" != "1" ]; then
  echo "进入虚拟环境： source .venv/bin/activate"
fi
