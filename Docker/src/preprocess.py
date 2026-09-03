"""彩色图像预处理模块（BGR → 高斯去噪 → 亮度 CLAHE）。

对应赛题"安装调试"环节可调参数：denoise_ksize / clahe_clip / clahe_tile。

依赖：opencv-python(headless) >= 4.5
"""
from __future__ import annotations

import logging

import numpy as np

try:
    from .config import AppConfig
except ImportError:  # 脚本模式（python src/xxx.py）回退
    from config import AppConfig


class Preprocessor:
    """保留颜色的预处理流水线：

      1. 统一为三通道 BGR
      2. 高斯去噪（GaussianBlur）
      3. 仅增强 LAB 亮度通道，保留颜色信息
    """

    def __init__(self, cfg: AppConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        if cfg.pre_clahe:
            import cv2
            self._clahe = cv2.createCLAHE(
                clipLimit=max(0.1, cfg.pre_clahe_clip),
                tileGridSize=(cfg.pre_clahe_tile, cfg.pre_clahe_tile),
            )
        else:
            self._clahe = None

    def run(self, image: np.ndarray) -> np.ndarray:
        """执行完整预处理流水线，输出始终为三通道 BGR。"""
        if not self.cfg.pre_enable:
            return image
        img = self._to_bgr(image)
        img = self._denoise(img)
        if self._clahe is not None:
            img = self._enhance_luminance(img)
        return img

    @staticmethod
    def _to_bgr(image: np.ndarray) -> np.ndarray:
        import cv2
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        raise ValueError(f"unsupported image shape: {image.shape}")

    def _enhance_luminance(self, image: np.ndarray) -> np.ndarray:
        import cv2
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        import cv2
        k = self.cfg.pre_denoise_ksize
        if k <= 0:
            k = 3
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(image, (k, k), 0)
