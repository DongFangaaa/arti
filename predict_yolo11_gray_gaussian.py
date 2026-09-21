"""对指定图片进行灰度化、高斯滤波，并调用 YOLO11 识别。

在 VS Code 中直接运行时，通常只需要修改下面两个路径常量。
也可以通过命令行的 --model 和 --image 临时覆盖。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ======================== 可修改接口 ========================
MODEL_PATH = Path(r"D:\Git\atfi\models\yolo11n_four_defects_best.pt")
IMAGE_PATH = Path(r"D:\Git\atfi\vision_color_gray_comparison.png")

CONFIDENCE = 0.80
IMAGE_SIZE = 640
GAUSSIAN_KERNEL_SIZE = 5
GAUSSIAN_SIGMA = 0.0
# None 表示由 Ultralytics 自动选择；也可以改成 "0"（第一块GPU）或 "cpu"。
DEVICE: str | None = None
# ===========================================================


def gray_gaussian_preprocess(
    image_path: str | Path,
    kernel_size: int = 5,
    sigma: float = 0.0,
) -> np.ndarray:
    """在一个函数中完成图片读取、灰度化和高斯滤波。

    YOLO通常接收三通道图片，因此最后会把处理后的单通道灰度图复制为
    三通道BGR；三个通道的数据完全相同，不会恢复彩色信息。
    """
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在：{image_path}")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Gaussian kernel_size 必须是大于0的奇数，例如3、5、7")
    if sigma < 0:
        raise ValueError("Gaussian sigma 不能小于0")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"OpenCV无法读取图片：{image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(
        gray,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
    )
    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)


def run_yolo11(
    model_path: str | Path,
    image_path: str | Path,
    *,
    conf: float = 0.80,
    imgsz: int = 640,
    kernel_size: int = 5,
    sigma: float = 0.0,
    device: str | None = None,
) -> dict[str, Any]:
    """预处理一张具体图片，调用YOLO11，并直接显示识别结果。"""
    from ultralytics import YOLO

    model_path = Path(model_path).expanduser().resolve()
    image_path = Path(image_path).expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"模型不存在：{model_path}")
    if not 0.0 <= conf <= 1.0:
        raise ValueError("conf 必须在0到1之间")

    processed_image = gray_gaussian_preprocess(
        image_path=image_path,
        kernel_size=kernel_size,
        sigma=sigma,
    )

    model = YOLO(str(model_path))
    predict_options: dict[str, Any] = {
        "source": processed_image,
        "conf": conf,
        "imgsz": imgsz,
        "verbose": False,
    }
    if device is not None:
        predict_options["device"] = device

    results = model.predict(**predict_options)
    if not results:
        raise RuntimeError("YOLO没有返回识别结果")
    result = results[0]

    response: dict[str, Any] = {
        "model": str(model_path),
        "image": str(image_path),
        "conf_threshold": conf,
        "gaussian_kernel_size": kernel_size,
        "gaussian_sigma": sigma,
    }

    # 检测模型结果：输出类别、置信度和边界框。
    detections: list[dict[str, Any]] = []
    if result.boxes is not None:
        for box, confidence, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            class_id = int(class_id)
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": str(result.names[class_id]),
                    "confidence": round(float(confidence), 6),
                    "box_xyxy": [round(float(value), 2) for value in box],
                }
            )
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        response["task"] = "detect"
        response["detections"] = detections

    # 分类模型结果：输出最高概率类别以及前5项。
    elif result.probs is not None:
        top_ids = [int(class_id) for class_id in result.probs.top5]
        top_confidences = [float(value) for value in result.probs.top5conf.cpu()]
        top5 = [
            {
                "class_id": class_id,
                "class_name": str(result.names[class_id]),
                "confidence": round(confidence, 6),
            }
            for class_id, confidence in zip(top_ids, top_confidences)
        ]
        response["task"] = "classify"
        response["top1"] = top5[0] if top5 else None
        response["top5"] = top5
    else:
        response["task"] = "unknown"

    annotated_image = result.plot()
    cv2.imshow("YOLO11 recognition result", annotated_image)
    print("识别结果窗口已打开，按任意键关闭。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="灰度化 + 高斯滤波后调用YOLO11识别一张图片"
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="YOLO11模型地址")
    parser.add_argument("--image", type=Path, default=IMAGE_PATH, help="待识别图片地址")
    parser.add_argument("--conf", type=float, default=CONFIDENCE, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE, help="YOLO输入尺寸")
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=GAUSSIAN_KERNEL_SIZE,
        help="高斯核尺寸，必须是正奇数",
    )
    parser.add_argument("--sigma", type=float, default=GAUSSIAN_SIGMA, help="高斯sigma")
    parser.add_argument("--device", default=DEVICE, help='例如"0"或"cpu"，默认自动选择')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    response = run_yolo11(
        model_path=args.model,
        image_path=args.image,
        conf=args.conf,
        imgsz=args.imgsz,
        kernel_size=args.kernel_size,
        sigma=args.sigma,
        device=args.device,
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
