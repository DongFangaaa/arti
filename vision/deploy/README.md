# 视觉检测 — 容器化部署说明

> 适用版本：v2.4（仅 ONNX 推理 + Modbus 通讯，不连接海康相机）

---

## 目录结构

```
.
├── Dockerfile                       # 镜像构建（python:3.11-slim + headless opencv）
├── docker-compose.yml              # Podman / Docker Compose（host 网络）
├── deploy/
│   ├── requirements.runtime.txt    # 容器精简依赖（无 torch）
│   ├── entrypoint.sh               # 容器入口脚本
│   ├── build.sh                    # 一键构建 + 启动
│   └── README.md                   # 本文档
├── models/
│   ├── best.pt                     # 开发/训练用
│   └── best.onnx                   # ★ 容器内推理用
└── src/main.py                     # 程序入口
```

## 适用场景

| 场景 | 推荐部署 |
|------|----------|
| **生产工控机（现场 + 相机）** | 直接在 Windows 上跑 `python src/main.py`，装 MVS SDK |
| **服务器 / CI / 联调测试** | 用本目录的 Docker，启用 DummyCamera |

**为什么 Docker 不接相机？**

海康 MVS SDK（`MvImport/`、`Dll/`）仅支持 Windows；Linux 容器无法直接驱动 GigE/USB3 相机。本镜像定位为：
- **算法正确性验证**（用 `test_images/` 走 DummyCamera）
- **Modbus 联调测试**（host 网络直连现场 PLC）
- **CI 回归测试**

## 快速开始

### 前置：导出 ONNX（仅首次）

```bash
# 在开发机（已装 ultralytics）执行一次
python -c "
from ultralytics import YOLO
YOLO('models/best.pt').export(format='onnx', imgsz=640, simplify=True)
"
# 生成 models/best.onnx（~10 MB）
```

### 构建 + 启动

```bash
# 一键脚本（推荐）
bash deploy/build.sh

# 或手动
docker build -t vision-app:2.4 .
docker compose up -d
docker compose logs -f vision-app
```

### 停止 / 清理

```bash
docker compose down           # 停容器
docker rmi vision-app:2.4     # 删镜像
docker system prune -a        # 全清
```

## 运行模式

### DummyCamera 模式（默认）

```bash
# 容器内启动时已默认 DUMMY_CAMERA=1
# 走 src/camera.py 的 DummyCamera 分支，循环读 test_images/*.png
# 适合：算法验证、PLC 联调
```

### 接 Modbus TCP 设备

`docker-compose.yml` 已设置 `network_mode: host`：
- 容器直连宿主机网络，Modbus 默认端口 **502** 直接可用
- PLC 地址写在 `config/plc_config.yaml`

```yaml
# config/plc_config.yaml 关键项
plc:
  host: 192.168.1.100      # PLC IP（host 网络下也可写 127.0.0.1）
  port: 502
```

## 镜像内启动逻辑

```
entrypoint.sh
  ↓ 检查 best.onnx / config 存在
  ↓ 检查 python / onnx / opencv 版本
  ↓ exec CMD
  ↓
python -m src.main
  ↓ 读 DUMMY_CAMERA 环境变量决定相机分支
  ↓ 加载 ONNX 模型 → 进入 PLC 触发循环
```

## 自定义启动命令

`docker-compose.yml` 默认 `CMD=["python", "-m", "src.main"]`。可覆盖：

```bash
# 跑测试（不进 PLC 循环）
docker compose run --rm vision python -m test.test_detector

# 跑批量识别脚本
docker compose run --rm vision python scripts/batch_preprocess_test.py

# bash 调试
docker compose run --rm vision bash
```

## 常见问题

### Q1: `best.onnx not found`
需要先在开发机导出（见上面"前置：导出 ONNX"）。

### Q2: `libGL.so.1: cannot open shared object file`
镜像里 `apt-get install -y libgl1` 已装；如果用 alpine/distroless 需要额外加包。

### Q3: Modbus 连不上 PLC
- 确认 `network_mode: host`（已在 compose 里设）
- 确认 `config/plc_config.yaml` 的 `host/port`
- 宿主机 `iptables -I INPUT -p tcp --dport 502 -j ACCEPT`

### Q4: 想用 GPU 推理
1. base 改 `FROM nvidia/cuda:12.x.x-runtime-ubuntu22.04`
2. `pip install onnxruntime-gpu`
3. 加 `--gpus all` 到 compose 的 deploy.resources.reservations

### Q5: 现场需要接相机
**不要用本镜像**。直接 Windows 装 MVS SDK 后跑：
```bash
pip install -r requirements.txt    # 含 ultralytics/torch
python src/main.py
```

## 性能基线（4 核 CPU）

| 阶段 | avg | min | max |
|---|---|---|---|
| 取图（DummyCamera） | 5 ms | 3 | 8 |
| 预处理（fastNlMeans） | 380 ms | 360 | 440 |
| ONNX 推理（detect 640） | 50 ms | 40 | 65 |
| **端到端** | **~450 ms / 张** | | |

实际瓶颈在 `fastNlMeans`（CPU 计算密集）；如需更实时，换 `denoise="gaussian"` 可降至 ~80ms。