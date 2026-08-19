"""视觉端工具函数（日志、图像保存、结果格式化、异常处理）。"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
def setup_logger(level: str = "INFO",
                 log_dir: str = "./logs",
                 name: str = "vision") -> logging.Logger:
    """初始化全局 Logger，同时输出到控制台与文件。"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(
        Path(log_dir) / f"{name}_{datetime.now():%Y%m%d}.log",
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# 图像 / 结果
# ---------------------------------------------------------------------------
def save_debug_image(image: np.ndarray,
                      path: str,
                      overlays: list[tuple] | None = None) -> str:
    """保存调试图像，支持叠加检测框。

    overlays: [(x1, y1, x2, y2, label, color_bgr), ...]
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img = image.copy()
    if overlays:
        for x1, y1, x2, y2, label, color in overlays:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, label, (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(path, img)
    return path


def format_result(defect_id: int,
                  conf: float,
                  x: float = 0.0,
                  y: float = 0.0) -> dict[str, Any]:
    """将检测结果格式化为统一字典。"""
    return {
        "ts": int(time.time() * 1000),
        "defect_id": int(defect_id),
        "confidence": round(float(conf), 4),
        "x_mm": round(float(x), 2),
        "y_mm": round(float(y), 2),
    }


def encode_payload(result: dict[str, Any],
                   fmt: str = "json_line") -> bytes:
    """编码为 PLC 可接收的字节流。"""
    if fmt == "json_line":
        return (json.dumps(result, ensure_ascii=False) + "\n").encode("utf-8")
    if fmt == "csv":
        row = f"{result['defect_id']},{int(result['confidence']*100)}," \
              f"{result['x_mm']},{result['y_mm']}\n"
        return row.encode("utf-8")
    raise ValueError(f"unknown send_format: {fmt}")


def decode_trigger(payload: bytes) -> int:
    """解析 PLC 触发指令（'TRIG\\n' 或 '1\\n'）。"""
    txt = payload.decode("utf-8", errors="ignore").strip()
    if txt in ("1", "TRIG", "trigger", "GO"):
        return 1
    return 0


# ---------------------------------------------------------------------------
# 异常 / 重试
# ---------------------------------------------------------------------------
class VisionError(RuntimeError):
    """视觉端通用错误。"""


def retry(callable_fn,
          retries: int = 3,
          delay: float = 0.1,
          except_types: tuple = (Exception,),
          logger: logging.Logger | None = None):
    """简易重试装饰器。"""
    last_exc: Exception | None = None
    for k in range(retries):
        try:
            return callable_fn()
        except except_types as e:
            last_exc = e
            if logger:
                logger.warning("retry %d/%d after error: %s", k + 1, retries, e)
            time.sleep(delay)
    raise VisionError(f"failed after {retries} retries: {last_exc}")


def now_ms() -> int:
    return int(time.time() * 1000)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)