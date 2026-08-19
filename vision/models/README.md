# Models — 模型权重目录

本目录存放训练好的 YOLO 权重文件。

## 目录约定

```
models/
├── best.pt              # PyTorch 权重（Ultralytics 默认输出）
├── best.onnx            # ONNX 权重（部署用，EPC1502 容器内推理）
└── README.md
```

## 生成方式

```bash
# 1) 训练（自动复制 best.pt 到本目录）
python -m train.train_yolo --data train/dataset.yaml --hyp train/hyp.yaml \
    --model yolov11s.pt --epochs 120 --batch 16

# 2) 导出 ONNX
python -m train.train_yolo --export-onnx --model models/best.pt --imgsz 640
```

## 部署

将 `best.onnx` 拷入 Docker 镜像（参考 `deploy/Dockerfile` 的 runtime stage），

```dockerfile
COPY models/ ./models/
```

容器启动后，`vision/detector.py` 会按 `vision/config.yaml` 中 `model.path` 自动加载对应后端：
- `.pt` → UltralyticsDetector
- `.onnx` → OnnxDetector（推荐用于 EPC 部署，体积小、不依赖 torch）

## 性能参考（待实测填入）

| 后端 | 平台 | 推理时延 | 体积 |
|------|------|---------|------|
| PyTorch (.pt) | PC + RTX 3060 | __ ms | __ MB |
| ONNX (CPU) | EPC1502 | __ ms | __ MB |