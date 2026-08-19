"""YOLO 分类模型推理模块（缺陷分类版 v2.3 — cls 架构）。

采用 YOLOv11n-cls 分类模型（无 bbox 输出），输出为单类别 + 置信度。

支持两种后端：
  - Ultralytics (PyTorch .pt)         —— 推荐本地开发使用
  - ONNX Runtime (exported .onnx)     —— 部署/EPC1502 推理

输出统一为 `ClsResult` 数据类。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:  # noqa: BLE001
    ULTRALYTICS_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except Exception:  # noqa: BLE001
    ONNX_AVAILABLE = False


@dataclass
class ClsResult:
    cls_name: str           # scratch / stain / chip / hole
    cls_id: int             # 0..3
    conf: float             # top-1 置信度
    topk: list              # [(cls_name, cls_id, conf), ...] top-5 排序
    # ---- detect 模型专用 ----
    # [(cls_name, cls_id, conf, (x1,y1,x2,y2)), ...]  按 conf 降序
    boxes: list = field(default_factory=list)


@dataclass
class DetectorConfig:
    model_path: str = "../models/best.pt"
    device: str = "cpu"         # cpu | cuda:0
    imgsz: int = 224            # cls 任务常用 224
    conf: float = 0.55          # top-1 置信度阈值（低于此报 fail_safe）
    warmup: bool = True
    classes: list = field(default_factory=lambda: ["scratch", "stain", "chip", "hole"])
    # 与 PLC 的 defect_id 映射（+1 偏移，空出 0 作 fail_safe）：
    # scratch→3, stain→4, chip→2, hole→1
    plc_defect_id: dict = field(default_factory=lambda: {
        "scratch": 3, "stain": 4, "chip": 2, "hole": 1,
    })


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------
class BaseDetector:
    def __init__(self, cfg: DetectorConfig, logger: logging.Logger | None = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger("vision.detector")

    def warmup(self) -> None:
        """加载模型并做一次空推理。"""
        raise NotImplementedError

    def detect(self, image: np.ndarray) -> ClsResult:
        """执行一次推理，返回 ClsResult。"""
        raise NotImplementedError

    def to_plc_defect_id(self, cls_name: str) -> int:
        return int(self.cfg.plc_defect_id.get(cls_name, 0))


# ---------------------------------------------------------------------------
# Ultralytics 后端
# ---------------------------------------------------------------------------
class UltralyticsDetector(BaseDetector):
    def __init__(self, cfg: DetectorConfig, logger: logging.Logger | None = None):
        super().__init__(cfg, logger)
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("ultralytics not installed")
        self.model = YOLO(cfg.model_path)
        self.model.to(cfg.device)
        self.logger.info("ultralytics cls model loaded: %s, device=%s",
                         cfg.model_path, cfg.device)

    def warmup(self) -> None:
        try:
            self.model.predict(
                np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8),
                imgsz=self.cfg.imgsz,
                verbose=False,
            )
            self.logger.info("ultralytics warmup ok")
        except Exception as e:
            self.logger.warning("warmup failed: %s", e)

    def detect(self, image: np.ndarray) -> ClsResult:
        t0 = time.perf_counter()
        results = self.model.predict(
            image,
            imgsz=self.cfg.imgsz,
            verbose=False,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        if not results:
            return self._empty_result()

        r = results[0]
        # Ultralytics cls 输出：r.probs.top5, r.probs.top5conf, r.probs.data
        try:
            top5_idx = list(r.probs.top5)
            top5_conf = list(r.probs.top5conf)
        except Exception:
            # 旧版本兼容
            data = list(r.probs.data.cpu().numpy())
            top5_idx = sorted(range(len(data)), key=lambda i: -data[i])[:5]
            top5_conf = [data[i] for i in top5_idx]

        top1_idx = int(top5_idx[0])
        top1_conf = float(top5_conf[0])
        top1_name = self.cfg.classes[top1_idx] if 0 <= top1_idx < len(self.cfg.classes) else "?"

        topk = []
        for idx, conf in zip(top5_idx, top5_conf):
            name = self.cfg.classes[int(idx)] if 0 <= int(idx) < len(self.cfg.classes) else "?"
            topk.append((name, int(idx), float(conf)))

        self.logger.debug("cls: %s (%.3f) in %.1f ms", top1_name, top1_conf, elapsed)
        return ClsResult(cls_name=top1_name, cls_id=top1_idx,
                         conf=top1_conf, topk=topk)

    def _empty_result(self) -> ClsResult:
        return ClsResult(cls_name="unknown", cls_id=-1,
                         conf=0.0, topk=[])


# ---------------------------------------------------------------------------
# ONNX Runtime 后端
# ---------------------------------------------------------------------------
class OnnxDetector(BaseDetector):
    """ONNX Runtime 分类推理。

    假定导出时使用 ultralytics `model.export(format="onnx")`，
    输出张量形状 [1, num_classes]（softmax 概率）。
    """

    def __init__(self, cfg: DetectorConfig, logger: logging.Logger | None = None):
        super().__init__(cfg, logger)
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime not installed")
        path = Path(cfg.model_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CUDAExecutionProvider"] if cfg.device.startswith("cuda") \
            else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), sess_options=so,
                                            providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_shape = self.session.get_inputs()[0].shape  # [1,3,H,W]
        self.logger.info("ONNX cls detector ready: %s, providers=%s",
                         path.name, self.session.get_providers())

    def warmup(self) -> None:
        try:
            dummy = np.zeros(self.input_shape, dtype=np.float32)
            self.session.run(self.output_names, {self.input_name: dummy})
            self.logger.info("onnx warmup ok")
        except Exception as e:
            self.logger.warning("onnx warmup failed: %s", e)

    def detect(self, image: np.ndarray) -> ClsResult:
        # 1) letterbox
        h0, w0 = image.shape[:2]
        H, W = int(self.input_shape[2]), int(self.input_shape[3])
        import cv2
        scale = min(W / w0, H / h0)
        nw, nh = int(w0 * scale), int(h0 * scale)
        resized = np.zeros((H, W, 3), dtype=np.uint8)
        r = cv2.resize(image, (nw, nh))
        resized[:nh, :nw] = r

        # 2) 归一化（与 Ultralytics 一致：0~255 / 255）
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]   # [1,3,H,W]

        # 3) 推理
        t0 = time.perf_counter()
        outs = self.session.run(self.output_names, {self.input_name: blob})
        elapsed = (time.perf_counter() - t0) * 1000
        probs = outs[0][0]   # [num_classes]

        # 4) top-k
        k = min(5, len(probs))
        top_idx = probs.argsort()[::-1][:k]
        top_conf = probs[top_idx]

        top1_idx = int(top_idx[0])
        top1_conf = float(top_conf[0])
        top1_name = self.cfg.classes[top1_idx] if 0 <= top1_idx < len(self.cfg.classes) else "?"
        topk = [(self.cfg.classes[int(i)] if 0 <= int(i) < len(self.cfg.classes) else "?",
                 int(i), float(c))
                for i, c in zip(top_idx, top_conf)]

        self.logger.debug("onnx cls: %s (%.3f) in %.1f ms",
                          top1_name, top1_conf, elapsed)
        return ClsResult(cls_name=top1_name, cls_id=top1_idx,
                         conf=top1_conf, topk=topk)


# ---------------------------------------------------------------------------
# Ultralytics Detect（带 bbox）后端
# ---------------------------------------------------------------------------
class UltralyticsDetectDetector(BaseDetector):
    """YOLOv8/v11 detect 模型：输出带 bbox 的多个候选框。

    本类把"检测"降级为"分类"：每次取置信度最高的框，
    用其 cls_id / conf 作为最终识别结果。
    """

    def __init__(self, cfg: DetectorConfig, logger: logging.Logger | None = None):
        super().__init__(cfg, logger)
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("ultralytics not installed")
        self.model = YOLO(cfg.model_path)
        self.model.to(cfg.device)
        # 用模型自带的类别名覆盖默认 cfg.classes（防止与训练时不一致）
        try:
            model_names = list(self.model.names.values())
            if model_names:
                self.cfg.classes = model_names
                self.logger.info("using model class names: %s", model_names)
        except Exception as e:  # noqa: BLE001
            self.logger.warning("read model names failed: %s", e)
        self.logger.info("ultralytics detect model loaded: %s, device=%s",
                         cfg.model_path, cfg.device)

    def warmup(self) -> None:
        try:
            h = w = self.cfg.imgsz
            self.model.predict(
                np.zeros((h, w, 3), dtype=np.uint8),
                imgsz=self.cfg.imgsz,
                verbose=False,
            )
            self.logger.info("ultralytics detect warmup ok")
        except Exception as e:  # noqa: BLE001
            self.logger.warning("detect warmup failed: %s", e)

    def detect(self, image: np.ndarray) -> ClsResult:
        t0 = time.perf_counter()
        results = self.model.predict(
            image,
            imgsz=self.cfg.imgsz,
            verbose=False,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        if not results:
            return self._empty_result()
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            # 没有检出框
            self.logger.debug("no boxes detected (%.1f ms)", elapsed)
            return self._empty_result()

        # 取置信度最高的框
        confs = r.boxes.conf.cpu().numpy()
        clses = r.boxes.cls.cpu().numpy().astype(int)
        # xyxy (n, 4) — letterbox 下的坐标，转回原图
        xyxys = r.boxes.xyxy.cpu().numpy()
        # 如果有原图尺寸信息，可以反向映射回原始 image
        # 但 image 已是 letterbox 之后的预处理结果（preprocessor 不缩放）
        # 这里 imgsz=640 而 image 可能是 448x448，predict 会自动 letterbox
        # yolo 返回的 xyxy 已经按原图（= input image）坐标映射过
        top_i = int(np.argmax(confs))
        top_conf = float(confs[top_i])
        top_cls = int(clses[top_i])

        # 构造一个 topk + boxes（按置信度排序，全部 bbox）
        order = np.argsort(-confs)
        topk = []
        boxes = []
        for i in order:
            ci = int(clses[i])
            cn = (self.cfg.classes[ci]
                  if 0 <= ci < len(self.cfg.classes) else "?")
            xyxy = tuple(float(v) for v in xyxys[i])
            topk.append((cn, ci, float(confs[i])))
            boxes.append((cn, ci, float(confs[i]), xyxy))
        # topk 只保留前 5 个用于显示
        topk = topk[:5]

        top1_name = (self.cfg.classes[top_cls]
                     if 0 <= top_cls < len(self.cfg.classes) else "?")
        self.logger.debug("detect: %s (%.3f) [%d boxes] in %.1f ms",
                          top1_name, top_conf, len(boxes), elapsed)
        return ClsResult(cls_name=top1_name, cls_id=top_cls,
                         conf=top_conf, topk=topk, boxes=boxes)

    def _empty_result(self) -> ClsResult:
        return ClsResult(cls_name="unknown", cls_id=-1,
                         conf=0.0, topk=[], boxes=[])


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------
def build_detector(cfg: DetectorConfig,
                   logger: logging.Logger | None = None) -> BaseDetector:
    """根据模型文件后缀 + 模型 task 自动选择后端。"""
    p = Path(cfg.model_path)
    suf = p.suffix.lower()
    logger = logger or logging.getLogger("vision.detector")
    if suf == ".onnx":
        logger.info("using OnnxDetector: %s", cfg.model_path)
        return OnnxDetector(cfg, logger)
    if suf in (".pt", ".pth"):
        # 先 try：读 model.task，挑对应后端
        try:
            from ultralytics import YOLO as _Y
            probe = _Y(cfg.model_path)
            task = getattr(probe, "task", "detect")
            del probe
        except Exception as e:  # noqa: BLE001
            logger.warning("probe model task failed: %s, fallback to cls", e)
            task = "classify"
        if task == "classify":
            logger.info("using UltralyticsDetector (cls): %s", cfg.model_path)
            return UltralyticsDetector(cfg, logger)
        # detect / segment / pose 都用 detect 后端
        logger.info("using UltralyticsDetectDetector (task=%s): %s", task, cfg.model_path)
        return UltralyticsDetectDetector(cfg, logger)
    raise ValueError(f"unsupported model format: {suf}")