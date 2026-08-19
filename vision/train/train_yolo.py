"""YOLOv11-cls 训练脚本（缺陷分类版 v2.3）。

用法：
    python -m train.train_yolo \\
        --data train/dataset.yaml \\
        --hyp   train/hyp.yaml \\
        --model yolov11n-cls.pt \\
        --epochs 100 \\
        --batch 32

训练完成后导出 ONNX：
    python -m train.train_yolo --export-onnx \\
        --model ../models/best.pt --imgsz 224
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vision.utils import setup_logger  # noqa: E402

logger = setup_logger("train", log_dir="../train/logs")


def load_hyp(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        logger.error("ultralytics not installed: %s", e)
        sys.exit(1)

    hyp = load_hyp(args.hyp) if args.hyp else {}
    logger.info("hyperparams: %s", hyp)

    model = YOLO(args.model)
    model.train(
        data=args.data,
        task="classify",                 # ← 关键：cls 任务
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(PROJECT_ROOT / "runs" / "classify"),
        name=args.name,
        patience=hyp.get("patience", 25),
        optimizer=hyp.get("optimizer", "AdamW"),
        lr0=hyp.get("lr0", 0.001),
        lrf=hyp.get("lrf", 0.01),
        momentum=hyp.get("momentum", 0.937),
        weight_decay=hyp.get("weight_decay", 0.0005),
        warmup_epochs=hyp.get("warmup_epochs", 3.0),
        hsv_h=hyp.get("hsv_h", 0.015),
        hsv_s=hyp.get("hsv_s", 0.7),
        hsv_v=hyp.get("hsv_v", 0.4),
        degrees=hyp.get("degrees", 15.0),
        translate=hyp.get("translate", 0.1),
        scale=hyp.get("scale", 0.5),
        shear=hyp.get("shear", 5.0),
        perspective=hyp.get("perspective", 0.0005),
        flipud=hyp.get("flipud", 0.0),
        fliplr=hyp.get("fliplr", 0.5),
        mosaic=hyp.get("mosaic", 0.0),
        mixup=hyp.get("mixup", 0.2),
        copy_paste=hyp.get("copy_paste", 0.0),
        label_smoothing=hyp.get("label_smoothing", 0.1),
        plots=True,
        verbose=True,
    )

    # 训练完成后导出 best.pt → models/best.pt
    best_src = PROJECT_ROOT / "runs" / "classify" / args.name / "weights" / "best.pt"
    if best_src.exists():
        dst = PROJECT_ROOT / "models" / "best.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(best_src.read_bytes())
        logger.info("copied best.pt -> %s", dst)
    else:
        logger.warning("best.pt not found at %s", best_src)


def export_onnx(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        logger.error("ultralytics not installed: %s", e)
        sys.exit(1)

    model = YOLO(args.model)
    # cls 导出时 task='classify'
    out = model.export(format="onnx", imgsz=args.imgsz, opset=12,
                       simplify=True, dynamic=False)
    logger.info("exported ONNX: %s", out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train/dataset.yaml")
    parser.add_argument("--hyp", default="train/hyp.yaml")
    parser.add_argument("--model", default="yolov11n-cls.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--name", default="defect_v23")
    parser.add_argument("--export-onnx", action="store_true",
                        help="export ONNX instead of training")
    args = parser.parse_args(argv)

    if args.export_onnx:
        export_onnx(args)
    else:
        train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())