"""相机采集测试：验证图像质量、帧率。

用法：
    python -m test.test_camera --config vision/config.yaml --frames 50 --save test/out
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

from vision.camera import CameraConfig, HikCamera, DummyCamera  # noqa: E402
from vision.utils import setup_logger  # noqa: E402

logger = setup_logger("test_camera", log_dir="logs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="vision/config.yaml")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--save", default="")
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.dummy:
        cam = DummyCamera(
            sorted(str(p) for p in Path("demo_images").glob("*.png"))[:20],
            logger,
        )
    else:
        cam = HikCamera(CameraConfig(**cfg["camera"]), logger)
    cam.open()

    if args.save:
        Path(args.save).mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    for i in range(args.frames):
        img = cam.capture(timeout_ms=1000)
        logger.info("frame %d shape=%s", i, img.shape)
        if args.save:
            cv2.imwrite(f"{args.save}/frame_{i:04d}.png", img)
    elapsed = time.perf_counter() - t0
    fps = args.frames / max(elapsed, 1e-6)
    logger.info("captured %d frames in %.2fs -> %.1f fps",
                args.frames, elapsed, fps)
    cam.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())