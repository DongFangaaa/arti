# =============================================================================
# 视觉逻辑算法应用赛 — Linux 运行时镜像（仅识别 + 通讯，不调相机）
# =============================================================================
# 构建：  docker build -t vision-app:2.4 .
# 运行：  docker compose up    （详见 deploy/README.md）
# 体积：  ~600 MB（headless opencv + onnxruntime，无 torch）
# =============================================================================

FROM python:3.11-slim AS base

# ---- 系统依赖 ----
# libgl1 / libglib2 / libsm6 等是 opencv-python-headless 唯一还需要的少数 so
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- 工作目录 ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

# ---- 1) 装依赖（先装，单独成层，缓存友好）----
COPY deploy/requirements.runtime.txt /app/deploy/requirements.runtime.txt
RUN pip install --no-cache-dir -r /app/deploy/requirements.runtime.txt

# ---- 2) 拷源码（注意：MvImport / Dll/ 不需要，容器不接海康相机）----
COPY src/ /app/src/
COPY config/ /app/config/
COPY models/best.onnx /app/models/best.onnx
COPY deploy/entrypoint.sh /app/deploy/entrypoint.sh

# ---- 3) 配置默认路径指向容器内 ----
#   main.py 默认从 src/ 加载 config.yaml.default；这里覆盖为容器内路径
ENV VISION_CONFIG=/app/config/camera_config.yaml \
    VISION_PLC_CONFIG=/app/config/plc_config.yaml \
    DUMMY_CAMERA=1 \
    PYTHONPATH=/app

# ---- 入口 ----
RUN chmod +x /app/deploy/entrypoint.sh
ENTRYPOINT ["/app/deploy/entrypoint.sh"]

# ---- 默认参数 ----
CMD ["python", "-m", "src.main"]