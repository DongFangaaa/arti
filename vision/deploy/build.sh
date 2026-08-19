#!/bin/bash
# =============================================================================
# 一键构建 + 启动
# =============================================================================
set -e

# 颜色
R() { printf '\033[31m%s\033[0m\n' "$*"; }
G() { printf '\033[32m%s\033[0m\n' "$*"; }
Y() { printf '\033[33m%s\033[0m\n' "$*"; }

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

IMAGE_NAME=${IMAGE_NAME:-vision-app}
IMAGE_TAG=${IMAGE_TAG:-2.4}

# ---- 0) 检查 ONNX 模型 ----
if [ ! -f "models/best.onnx" ]; then
    Y "[WARN] models/best.onnx 不存在，先在开发机导出："
    echo "  python -c \"from ultralytics import YOLO; YOLO('models/best.pt').export(format='onnx', imgsz=640, simplify=True)\""
    exit 1
fi

# ---- 1) 容器运行时：检测 podman / docker ----
if command -v podman >/dev/null 2>&1; then
    ENGINE=podman
elif command -v docker >/dev/null 2>&1; then
    ENGINE=docker
else
    R "[FATAL] 未检测到 podman 或 docker，请先安装其中之一"
    exit 1
fi
G "[INFO] using engine: ${ENGINE}"

# ---- 2) 构建镜像 ----
G "[INFO] 构建镜像 ${IMAGE_NAME}:${IMAGE_TAG} ..."
${ENGINE} build \
    -f Dockerfile \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    "${PROJECT_ROOT}"

# ---- 3) compose up ----
G "[INFO] 启动容器 ..."
if [ "${ENGINE}" = "podman" ]; then
    podman-compose up -d
else
    docker compose up -d
fi

G "[OK] 容器已启动，日志：${ENGINE} logs -f vision-app"