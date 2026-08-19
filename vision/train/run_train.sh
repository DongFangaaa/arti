#!/usr/bin/env bash
# =============================================================================
# 一键训练脚本：data 准备 → 训练 → 导出 ONNX → 拷贝到 models/
# =============================================================================
set -e

cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-32}"
IMGSZ="${IMGSZ:-224}"
MODEL="${MODEL:-yolov11n-cls.pt}"
NAME="${NAME:-defect_v23}"

echo "[1/4] prepare dataset ..."
python data/prepare_cls_dataset.py

echo "[2/4] check consistency ..."
python tools/check_consistency.py

echo "[3/4] train ..."
python -m train.train_yolo \
    --data train/dataset.yaml \
    --hyp  train/hyp.yaml \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --imgsz "$IMGSZ" \
    --name "$NAME"

echo "[4/4] export ONNX ..."
python -m train.train_yolo --export-onnx \
    --model models/best.pt --imgsz "$IMGSZ"

echo "[DONE] models/best.pt and models/best.onnx are ready"