"""对单张图跑一遍预处理流水线，输出每个阶段的结果。

用法：
    python -m tools.demo_preprocess --img path/to/img.bmp --out output/dir

输出：
    0_original.png    原始图
    1_denoise.png     去噪后
    2_clahe.png       CLAHE 后
    3_gamma.png       Gamma 校正后
    4_sharpen.png     USM 锐化后（最终）
    5_compare.png     横向拼接（用于答辩展示）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocess import PreprocessConfig, Preprocessor  # noqa: E402


def stage_denoise(img: np.ndarray, method: str = "fastNlMeans") -> np.ndarray:
    if method == "fastNlMeans":
        return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    if method == "gaussian":
        return cv2.GaussianBlur(img, (3, 3), 0)
    return img.copy()


def stage_clahe(img: np.ndarray, clip: float = 4.0) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    out = cv2.merge([l, a, b])
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def stage_gamma(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    if abs(gamma - 1.0) <= 1e-3:
        return img
    lut = np.array([
        np.clip(((i / 255.0) ** (1.0 / gamma)) * 255, 0, 255)
        for i in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(img, lut)


def stage_sharpen(img: np.ndarray,
                  amount: float = 0.5,
                  radius: float = 1.0) -> np.ndarray:
    if amount <= 1e-3:
        return img
    blurred = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=radius)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0.0)


def make_compare(images: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """横向拼接多张图，每张上方写标题。"""
    pad = 12
    title_h = 24
    # 全部缩到相同高度
    h_target = 360
    resized = []
    for name, im in images:
        h, w = im.shape[:2]
        scale = h_target / h
        new = cv2.resize(im, (int(w * scale), h_target))
        resized.append((name, new))
    # 总宽度
    total_w = sum(im.shape[1] for _, im in resized) + pad * (len(resized) + 1)
    canvas = np.full((h_target + title_h + pad * 2, total_w, 3),
                     240, dtype=np.uint8)
    x = pad
    for name, im in resized:
        cv2.putText(canvas, name, (x, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        canvas[title_h + pad:title_h + pad + im.shape[0],
               x:x + im.shape[1]] = im
        x += im.shape[1] + pad
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)
    parser.add_argument("--out", default="logs/preprocess_demo")
    parser.add_argument("--sharpen-amount", type=float, default=0.5)
    parser.add_argument("--sharpen-radius", type=float, default=1.0)
    parser.add_argument("--clahe-clip", type=float, default=4.0)
    args = parser.parse_args()

    src = Path(args.img)
    if not src.exists():
        print(f"[ERROR] {src} not found", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(src))
    if img is None:
        print(f"[ERROR] cannot read {src}", file=sys.stderr)
        return 1
    print(f"[INFO] input: {src}  shape={img.shape}")

    # 各阶段
    original = img.copy()
    denoised = stage_denoise(img, "fastNlMeans")
    clahe_img = stage_clahe(denoised, clip=args.clahe_clip)
    gamma_img = stage_gamma(clahe_img, 1.0)        # 默认 1.0 跳过
    sharpened = stage_sharpen(gamma_img,
                               amount=args.sharpen_amount,
                               radius=args.sharpen_radius)

    # CLAHE 对比（不同 clipLimit）
    clahe_2 = stage_clahe(denoised, clip=2.0)
    clahe_4 = stage_clahe(denoised, clip=4.0)
    clahe_6 = stage_clahe(denoised, clip=6.0)

    # 保存
    cv2.imwrite(str(out_dir / "0_original.png"), original)
    cv2.imwrite(str(out_dir / "1_denoise.png"), denoised)
    cv2.imwrite(str(out_dir / "2_clahe.png"), clahe_img)
    cv2.imwrite(str(out_dir / "3_gamma.png"), gamma_img)
    cv2.imwrite(str(out_dir / "4_sharpen.png"), sharpened)
    print(f"[OK] saved individual stages to {out_dir}/")

    # 横向拼接
    compare = make_compare([
        ("0_original", original),
        ("1_denoise", denoised),
        ("2_clahe", clahe_img),
        ("4_sharpen", sharpened),
    ])
    cv2.imwrite(str(out_dir / "5_compare.png"), compare)
    print(f"[OK] saved comparison: {out_dir/'5_compare.png'}")

    # CLAHE clipLimit 对比
    clahe_compare = make_compare([
        ("orig", original),
        ("clip=2.0", clahe_2),
        ("clip=4.0", clahe_4),
        ("clip=6.0", clahe_6),
    ])
    cv2.imwrite(str(out_dir / "6_clahe_compare.png"), clahe_compare)
    print(f"[OK] saved CLAHE comparison: {out_dir/'6_clahe_compare.png'}")

    # 同步用 Preprocessor 类跑一次完整流水线，验证一致性
    cfg = PreprocessConfig(
        enable=True,
        denoise="fastNlMeans",
        clahe=True,
        clahe_clip=args.clahe_clip,
        gamma=1.0,
        sharpen=True,
        sharpen_amount=args.sharpen_amount,
        sharpen_radius=args.sharpen_radius,
    )
    pp = Preprocessor(cfg)
    full = pp.run(img)
    cv2.imwrite(str(out_dir / "6_full_pipeline.png"), full)
    diff = cv2.absdiff(full, sharpened)
    print(f"[INFO] pipeline vs stage-by-stage max diff = {int(diff.max())} "
          f"(应为 0 或接近 0)")

    # 客观指标
    def sharpness_score(im: np.ndarray) -> float:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    print("\n========== 客观指标（拉普拉斯方差，越大越锐利）==========")
    print(f"  原始       : {sharpness_score(original):8.2f}")
    print(f"  去噪后     : {sharpness_score(denoised):8.2f}")
    print(f"  CLAHE 2.0  : {sharpness_score(clahe_2):8.2f}")
    print(f"  CLAHE 4.0  : {sharpness_score(clahe_4):8.2f}")
    print(f"  CLAHE 6.0  : {sharpness_score(clahe_6):8.2f}")
    print(f"  USM 锐化后 : {sharpness_score(sharpened):8.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())