"""彩色图像预处理模块（BGR → 去噪 → 亮度增强 → 外圆分割）。

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
      4. 检测最大的外圆，将圆外背景填为纯白色
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
        circle = self._find_largest_outer_circle(img)
        img = self._denoise(img)
        if self._clahe is not None:
            img = self._enhance_luminance(img)
        if circle is not None:
            img = self._whiten_outside_circle(img, circle)
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

    def _find_largest_outer_circle(
            self, image: np.ndarray) -> tuple[int, int, int] | None:
        """检测画面中的最大外圆，排除孔洞等小圆。"""
        import cv2
        height, width = image.shape[:2]
        minimum_size = min(height, width)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=minimum_size * 0.5,
            param1=100,
            param2=40,
            minRadius=int(minimum_size * 0.25),
            maxRadius=int(minimum_size * 0.49),
        )
        if circles is None:
            self.logger.warning("未检测到物料外圆，本帧保留原背景")
            return None
        x, y, radius = max(circles[0], key=lambda item: item[2])
        circle = int(round(x)), int(round(y)), int(round(radius))
        self.logger.debug("检测到最大外圆: center=(%d,%d), radius=%d", *circle)
        return circle

    @staticmethod
    def _whiten_outside_circle(
            image: np.ndarray, circle: tuple[int, int, int]) -> np.ndarray:
        """保留圆内图像，将外圆以外的像素设为纯白。"""
        import cv2
        x, y, radius = circle
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, thickness=-1)
        result = np.full_like(image, 255)
        result[mask != 0] = image[mask != 0]
        return result

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        import cv2
        k = self.cfg.pre_denoise_ksize
        if k <= 0:
            k = 3
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(image, (k, k), 0)
