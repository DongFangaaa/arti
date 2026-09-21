"""使用原始彩色图片调用 YOLO11 识别。

在 VS Code 中直接运行时，通常只需要修改下面两个路径常量。
也可以通过命令行的 --model 和 --image 临时覆盖。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2


# ======================== 可修改接口 ========================
MODEL_PATH = Path(r"D:\Git\atfi\runs_defects\yolo11n_four_defects-3\weights\best.pt")
IMAGE_PATH = Path(r"D:\Git\atfi\data.zip\data\data\split_v3\test\images\Image_20260803154209217.jpg")

CONFIDENCE = 0.80
IMAGE_SIZE = 640
# None 表示由 Ultralytics 自动选择；也可以改成 "0"（第一块GPU）或 "cpu"。
DEVICE: str | None = None
# ===========================================================


def run_yolo11(
    model_path: str | Path,
    image_path: str | Path,
    *,
    conf: float = 0.80,
    imgsz: int = 640,
    device: str | None = None,
) -> dict[str, Any]:
    """使用原始彩色图片调用YOLO11，并直接显示识别结果。"""
    from ultralytics import YOLO

    model_path = Path(model_path).expanduser().resolve()
    image_path = Path(image_path).expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"模型不存在：{model_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在：{image_path}")
    if not 0.0 <= conf <= 1.0:
        raise ValueError("conf 必须在0到1之间")

    model = YOLO(str(model_path))
    predict_options: dict[str, Any] = {
        "source": str(image_path),
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
        "input_mode": "original_color",
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
        description="使用原始彩色图片调用YOLO11识别一张图片"
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="YOLO11模型地址")
    parser.add_argument("--image", type=Path, default=IMAGE_PATH, help="待识别图片地址")
    parser.add_argument("--conf", type=float, default=CONFIDENCE, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE, help="YOLO输入尺寸")
    parser.add_argument("--device", default=DEVICE, help='例如"0"或"cpu"，默认自动选择')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    response = run_yolo11(
        model_path=args.model,
        image_path=args.image,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
