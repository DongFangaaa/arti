#!/bin/sh
# =============================================================================
# 视觉逻辑算法应用赛 — 容器入口脚本
# =============================================================================
# 默认行为：
#   1. 检查必要文件存在
#   2. 检查 best.onnx 是否就位（否则提示导出）
#   3. exec CMD（默认：python -m src.main）
# =============================================================================

set -e

echo "[entrypoint] vision-app container starting..."
echo "[entrypoint] python : $(python --version 2>&1)"
echo "[entrypoint] onnx   : $(python -c 'import onnxruntime as ort; print(ort.__version__)')"
echo "[entrypoint] opencv : $(python -c 'import cv2; print(cv2.__version__)')"

# ---- 1) 检查 ONNX 模型 ----
ONNX_PATH="${VISION_MODEL:-/app/models/best.onnx}"
if [ ! -f "${ONNX_PATH}" ]; then
    echo "[entrypoint][FATAL] model not found: ${ONNX_PATH}" >&2
    echo "[entrypoint] 在开发机导出：" >&2
    echo "    python -c \"from ultralytics import YOLO; YOLO('models/best.pt').export(format='onnx', imgsz=640, simplify=True)\"" >&2
    exit 1
fi
echo "[entrypoint] model   : ${ONNX_PATH} ($(stat -c %s "${ONNX_PATH}" 2>/dev/null || stat -f %z "${ONNX_PATH}") bytes)"

# ---- 2) 检查 config ----
if [ ! -f "${VISION_CONFIG:-/app/config/camera_config.yaml}" ]; then
    echo "[entrypoint][FATAL] VISION_CONFIG not found" >&2
    exit 1
fi
if [ ! -f "${VISION_PLC_CONFIG:-/app/config/plc_config.yaml}" ]; then
    echo "[entrypoint][FATAL] VISION_PLC_CONFIG not found" >&2
    exit 1
fi

# ---- 3) 优雅停机：转发 SIGTERM ----
trap 'echo "[entrypoint] SIGTERM, exit."; exit 0' TERM INT

# ---- 4) exec 替换当前进程 ----
echo "[entrypoint] exec $*"
exec "$@"