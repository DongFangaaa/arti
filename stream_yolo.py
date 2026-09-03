"""使用 best.pt 对 vision 容器的实时 MJPEG 数据流做 YOLO 识别。

默认数据流：http://192.168.3.20:8080/stream.mjpg
退出预览：在窗口中按 Q 或 Esc。
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


class LatestFrameStream:
    """后台持续读取视频流，只保留最新帧，避免推理期间积压旧画面。"""

    def __init__(self, source: str):
        self.source = int(source) if source.isdecimal() else source
        self._condition = threading.Condition()
        self._capture: cv2.VideoCapture | None = None
        self._frame = None
        self._sequence = 0
        self._stopped = False
        self._thread = threading.Thread(target=self._reader, daemon=True)

    def start(self) -> "LatestFrameStream":
        self._thread.start()
        return self

    def _open(self) -> bool:
        capture = cv2.VideoCapture(self.source)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            capture.release()
            return False
        self._capture = capture
        return True

    def _reader(self) -> None:
        while not self._stopped:
            if self._capture is None and not self._open():
                time.sleep(1.0)
                continue

            try:
                ok, frame = self._capture.read()
            except cv2.error:
                # close() 会释放底层流，使正在阻塞的 read() 退出。
                if self._stopped:
                    break
                ok, frame = False, None
            if not ok:
                if self._capture is not None:
                    self._capture.release()
                self._capture = None
                time.sleep(0.2)
                continue

            with self._condition:
                self._frame = frame
                self._sequence += 1
                self._condition.notify_all()

    def read(self, after_sequence: int, timeout: float = 5.0):
        """等待并返回比 after_sequence 更新的一帧。"""
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._stopped and self._sequence <= after_sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, after_sequence
                self._condition.wait(remaining)
            if self._frame is None:
                return None, after_sequence
            return self._frame.copy(), self._sequence

    def close(self) -> None:
        self._stopped = True
        with self._condition:
            self._condition.notify_all()
        if self._capture is not None:
            self._capture.release()
        self._thread.join(timeout=2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 best.pt 识别实时视频流")
    parser.add_argument(
        "--source",
        default="http://192.168.3.20:8080/stream.mjpg",
        help="MJPEG/RTSP 地址、视频文件或摄像头编号",
    )
    parser.add_argument(
        "--model",
        default=str(Path(__file__).with_name("best.pt")),
        help="YOLO .pt 模型路径",
    )
    parser.add_argument("--conf", type=float, default=0.55, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    parser.add_argument("--device", default="cpu", help="cpu、0、0,1 等")
    parser.add_argument("--output", help="可选：将标注画面保存为 MP4")
    parser.add_argument("--headless", action="store_true", help="不显示窗口")
    parser.add_argument("--max-frames", type=int, default=0, help="处理帧数；0 表示持续运行")
    return parser.parse_args()


def describe_result(result) -> str:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return "未检测到目标"
    items = []
    for class_id, confidence in zip(boxes.cls.tolist(), boxes.conf.tolist()):
        name = result.names[int(class_id)]
        items.append(f"{name} {confidence:.4f}")
    return ", ".join(items)


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"模型不存在：{model_path}")

    print(f"加载模型：{model_path}")
    model = YOLO(str(model_path))
    print(f"类别：{model.names}")
    print(f"连接数据流：{args.source}")

    stream = LatestFrameStream(args.source).start()
    writer = None
    sequence = 0
    processed = 0

    try:
        while args.max_frames <= 0 or processed < args.max_frames:
            frame, sequence = stream.read(sequence)
            if frame is None:
                print("等待数据流画面……")
                continue

            result = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            annotated = result.plot()
            processed += 1
            print(time.strftime("%H:%M:%S"), describe_result(result))

            if args.output:
                if writer is None:
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        args.output,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10.0,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"无法创建输出视频：{args.output}")
                writer.write(annotated)

            if not args.headless:
                cv2.imshow("YOLO live stream - Q/Esc to quit", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        stream.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
