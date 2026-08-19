"""模型推理测试：单图/批量测试，输出准确率。

用法：
    python -m test.test_detector --config vision/config.yaml --img-dir data/demo_images
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import yaml  # noqa: E402

from vision.detector import DetectorConfig, build_detector  # noqa: E402
from vision.utils import save_debug_image, setup_logger  # noqa: E402

logger = setup_logger("test_detector", log_dir="logs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="vision/config.yaml")
    parser.add_argument("--img-dir", default="data/demo_images")
    parser.add_argument("--save-dir", default="logs/test_detector")
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    det_cfg = DetectorConfig(
        model_path=str(Path(args.config).parent / cfg["model"]["path"]),
        device=cfg["model"]["device"],
        imgsz=cfg["model"]["imgsz"],
        conf=cfg["model"]["conf"],
        iou=cfg["model"]["iou"],
        half=cfg["model"]["half"],
        warmup=cfg["model"]["warmup"],
        classes=cfg["classes"]["names"],
        plc_defect_id=cfg["classes"]["plc_defect_id"],
    )
    det = build_detector(det_cfg, logger)
    det.warmup()

    img_paths = sorted(Path(args.img_dir).glob("*.png")) + \
                sorted(Path(args.img_dir).glob("*.bmp")) + \
                sorted(Path(args.img_dir).glob("*.jpg"))
    if not img_paths:
        logger.error("no images found in %s", args.img_dir)
        return 1

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        dets = det.detect(img)
        overlays = [(d.xyxy[0], d.xyxy[1], d.xyxy[2], d.xyxy[3],
                     f"{d.cls_name}:{d.conf:.2f}",
                     (0, 255, 0) if d.conf >= det_cfg.conf else (0, 0, 255))
                    for d in dets]
        save_debug_image(
            img,
            f"{args.save_dir}/{p.stem}.png",
            overlays,
        )
        logger.info("%s -> %d detections", p.name, len(dets))
    elapsed = time.perf_counter() - t0
    logger.info("processed %d images in %.2fs (%.1f fps)",
                len(img_paths), elapsed, len(img_paths) / max(elapsed, 1e-6))
    return 0


if __name__ == "__main__":
    sys.exit(main())