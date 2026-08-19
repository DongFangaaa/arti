"""图像预处理模块（v2.4 — method5_combined 激进升级）。

⚠️ v2.4 重大变更：
- 默认 mode 改为 "combined"（CLAHE + Gamma 合并处理）
- Gamma 默认 0.8（提亮暗部，对工业缺陷更敏感）
- USM 锐化默认开启（amount=1.0，sigma=1.2）
- 仍保留 "legacy" 模式作为 fallback（出问题时切换）

流水线（mode=combined）：
  1. 去噪（fastNlMeansDenoisingColored，h=5）
  2. CLAHE + Gamma 合并（仅 LAB-L 通道）
  3. USM 锐化（amount + sigma）

依赖：opencv-python >= 4.5
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    enable: bool = True

    # v2.4 新增：处理模式
    #   legacy   = 老的逐步骤处理（CLAHE/Gamma/USM 各自独立）
    #   combined = method5_combined（CLAHE+Gamma 合并）  ← 默认
    mode: str = "combined"

    # 去噪
    denoise: str = "fastNlMeans"     # fastNlMeans | gaussian | none
    denoise_strength: int = 5       # h 参数（fastNlMeans），建议 3~7

    # CLAHE
    clahe: bool = True
    clahe_clip: float = 4.0         # clipLimit（combined 模式下默认 4.0，更激进）
    clahe_tile: int = 8             # tileGridSize

    # Gamma
    gamma: float = 0.8              # v2.4 默认 0.8（提亮暗部）

    # USM 锐化
    sharpen: bool = True            # v2.4 默认开
    sharpen_amount: float = 1.0     # amount，v2.4 默认 1.0（中间值）
    sharpen_radius: float = 1.2     # sigma，v2.4 默认 1.2（粗一些）

    normalize: bool = True          # 仅对调试图生效


class Preprocessor:
    def __init__(self, cfg: PreprocessConfig, logger: logging.Logger | None = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger("vision.preprocess")
        # mode 兼容
        if cfg.mode not in ("legacy", "combined"):
            self.logger.warning(
                "unknown preprocess mode=%s, fallback to 'combined'", cfg.mode)
            cfg.mode = "combined"
        if cfg.clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=max(0.1, cfg.clahe_clip),
                tileGridSize=(cfg.clahe_tile, cfg.clahe_tile),
            )
        else:
            self._clahe = None

    def run(self, image: np.ndarray) -> np.ndarray:
        """执行完整预处理流水线。"""
        if not self.cfg.enable:
            return image

        if self.cfg.mode == "combined":
            return self._run_combined(image)
        return self._run_legacy(image)

    # ------------------------------------------------------------------
    # mode="combined"：v2.4 默认流水线（method5_combined 风格）
    # ------------------------------------------------------------------
    def _run_combined(self, image: np.ndarray) -> np.ndarray:
        img = image

        # 1) 去噪
        img = self._denoise(img)

        # 2) CLAHE + Gamma 合并（仅 LAB-L 通道）
        if self._clahe is not None:
            img = self._clahe_gamma(img)

        # 3) USM 锐化
        if self.cfg.sharpen and self.cfg.sharpen_amount > 1e-3:
            img = self._usm_sharpen(img)

        return img

    def _clahe_gamma(self, image: np.ndarray) -> np.ndarray:
        """CLAHE + Gamma 合并到单次 LAB-L 通道处理（method5_combined 风格）。"""
        if image.ndim == 3 and image.shape[2] == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            # 先 CLAHE
            l = self._clahe.apply(l)
            # 再 Gamma（仅当 ≠1）
            if abs(self.cfg.gamma - 1.0) > 1e-3:
                inv_gamma = 1.0 / self.cfg.gamma
                table = np.array(
                    [np.clip(((i / 255.0) ** inv_gamma) * 255, 0, 255)
                     for i in range(256)],
                    dtype=np.uint8,
                )
                l = cv2.LUT(l, table)
            return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        # 灰度分支
        out = self._clahe.apply(image)
        if abs(self.cfg.gamma - 1.0) > 1e-3:
            inv_gamma = 1.0 / self.cfg.gamma
            table = np.array(
                [np.clip(((i / 255.0) ** inv_gamma) * 255, 0, 255)
                 for i in range(256)],
                dtype=np.uint8,
            )
            out = cv2.LUT(out, table)
        return out

    # ------------------------------------------------------------------
    # mode="legacy"：v2.3 老流水线（保留作 fallback）
    # ------------------------------------------------------------------
    def _run_legacy(self, image: np.ndarray) -> np.ndarray:
        img = image

        # 去噪
        img = self._denoise(img)

        # CLAHE（仅 LAB-L）
        if self._clahe is not None:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            img = cv2.merge([l, a, b])
            img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

        # Gamma（独立 LUT 步骤）
        if abs(self.cfg.gamma - 1.0) > 1e-3:
            lut = np.array([
                np.clip(((i / 255.0) ** (1.0 / self.cfg.gamma)) * 255, 0, 255)
                for i in range(256)
            ], dtype=np.uint8)
            img = cv2.LUT(img, lut)

        # USM 锐化
        if self.cfg.sharpen and self.cfg.sharpen_amount > 1e-3:
            img = self._usm_sharpen(img)

        return img

    # ------------------------------------------------------------------
    # 通用子函数
    # ------------------------------------------------------------------
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        if self.cfg.denoise == "fastNlMeans":
            h = max(1, self.cfg.denoise_strength)
            if image.ndim == 3 and image.shape[2] == 3:
                return cv2.fastNlMeansDenoisingColored(
                    image, None, h, h, 7, 21)
            return cv2.fastNlMeansDenoising(image, None, h, 7, 21)
        if self.cfg.denoise == "gaussian":
            return cv2.GaussianBlur(image, (3, 3), 0)
        return image

    def _usm_sharpen(self, image: np.ndarray) -> np.ndarray:
        """Unsharp Mask 锐化：output = 原图 + amount * (原图 - 模糊图)。"""
        blurred = cv2.GaussianBlur(
            image,
            ksize=(0, 0),
            sigmaX=self.cfg.sharpen_radius,
        )
        return cv2.addWeighted(
            image, 1.0 + self.cfg.sharpen_amount,
            blurred, -self.cfg.sharpen_amount,
            0.0,
        )

    def crop_roi(self, image: np.ndarray,
                 x: int, y: int, w: int, h: int) -> np.ndarray:
        """裁剪 ROI；用于去除传送带背景。"""
        H, W = image.shape[:2]
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))
        return image[y:y + h, x:x + w]


# ============================================================================
# 向后兼容：旧 API
# ============================================================================
# 旧代码若直接 import method5_combined 或 denoise，提供 shim：
def method5_combined(img: np.ndarray, gamma: float = 0.8,
                     clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """兼容 model/scripts/enhance_contrast.py 的 method5_combined 接口。"""
    pre = Preprocessor(PreprocessConfig(
        mode="combined",
        clahe=True, clahe_clip=clip_limit, clahe_tile=tile_size,
        gamma=gamma,
        denoise="none", sharpen=False,
    ))
    return pre._clahe_gamma(img)


def denoise(img: np.ndarray, strength: int = 3) -> np.ndarray:
    """兼容 model/scripts/enhance_contrast.py 的 denoise 接口。"""
    pre = Preprocessor(PreprocessConfig(denoise="fastNlMeans",
                                        denoise_strength=strength))
    return pre._denoise(img)


def sharpen(img: np.ndarray, amount: float = 1.5, sigma: float = 1.5) -> np.ndarray:
    """兼容 model/scripts/enhance_contrast.py 的 sharpen 接口。"""
    pre = Preprocessor(PreprocessConfig(
        sharpen=True, sharpen_amount=amount - 1.0, sharpen_radius=sigma))
    return pre._usm_sharpen(img)