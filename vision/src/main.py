"""视觉主程序入口（缺陷分类版 v2.3 — cls 架构）。

采用 YOLOv11n-cls 分类模型，单张图像输出 1 个类别 + 置信度。

流程：
  1. 加载配置 → 初始化相机、预处理、检测器、通讯
  2. 循环：
       a. 等待 PLC 触发 (wait_trigger)
       b. 抓帧 (camera.capture)
       c. 预处理 (preprocess.run)
       d. 推理 (detector.detect)
       e. 转换为 PLC defect_id → 上报 (comm.send_result)
       f. 异常处理：超时/失败 → 上报 fail_safe_defect

退出：Ctrl-C 或 PLC 发送 "STOP" 指令。
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import yaml

from .camera import CameraConfig, DummyCamera, HikCamera, VisionError
from .comm import PlcComm, PlcResult, build_comm
from .detector import BaseDetector, ClsResult, DetectorConfig, build_detector
from .preprocess import PreprocessConfig, Preprocessor
from .utils import ensure_dir, format_result, now_ms, setup_logger


# ---------------------------------------------------------------------------
# 顶层封装
# ---------------------------------------------------------------------------
class VisionApp:
    def __init__(self, cfg_path: str):
        self.cfg_path = cfg_path
        # 优先使用 config.yaml.local（现场调试版，git ignored），
        # 加载不到时回退到传入的 cfg_path（默认 config.yaml）
        actual_cfg = cfg_path
        local_cfg = Path(cfg_path).parent / "config.yaml.local"
        # local 存在且有内容才用，否则回退到默认 config.yaml
        if local_cfg.exists() and local_cfg.stat().st_size > 0:
            actual_cfg = str(local_cfg)
            print(f"[main] using local config: {actual_cfg}")
        else:
            print(f"[main] using default config: {actual_cfg}")
        self.cfg_path = actual_cfg
        with open(actual_cfg, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}
        self.logger = setup_logger(
            level=self.cfg["logging"]["level"],
            log_dir=self.cfg["logging"]["dir"],
            name="vision",
        )
        self.preprocessor: Preprocessor | None = None
        self.detector: BaseDetector | None = None
        self.comm: PlcComm | None = None
        self.camera: HikCamera | None = None
        self._seq = 0  # 上报序列号（PLC 端可校验）
        self._running = False

    # ----------------------------------------------------------- init
    def _init_components(self) -> None:
        # 把嵌套的 roi: {enable, x, y, width, height} 摊平为 CameraConfig 字段
        cam_dict = dict(self.cfg["camera"])
        roi = cam_dict.pop("roi", {})
        cam_dict.setdefault("roi_enable", roi.get("enable", False))
        cam_dict.setdefault("roi_x", roi.get("x", 0))
        cam_dict.setdefault("roi_y", roi.get("y", 0))
        cam_dict.setdefault("roi_w", roi.get("width", 720))
        cam_dict.setdefault("roi_h", roi.get("height", 540))
        cam_cfg = CameraConfig(**cam_dict)
        # DummyCamera 启用条件（显式）：
        #   1) 环境变量 USE_DUMMY_CAMERA=1
        #   2) config runtime.dummy_camera: true
        # 否则 fail-fast（EPC 现场必须有 MVS SDK）
        from .camera import MVS_AVAILABLE, USE_DUMMY_CAMERA
        env_dummy = USE_DUMMY_CAMERA
        cfg_dummy = self.cfg.get("runtime", {}).get("dummy_camera", False)
        use_dummy = env_dummy or cfg_dummy
        if use_dummy:
            self.logger.warning(
                "DummyCamera ENABLED (env=%s, cfg=%s). "
                "Only for debug. Production must use HikCamera.",
                env_dummy, cfg_dummy,
            )
            dummy_dir = self.cfg.get("runtime", {}).get(
                "dummy_image_dir", "./demo_images")
            paths = sorted(str(p) for p in Path(dummy_dir).glob("*.png"))[:20]
            if not paths:
                paths = sorted(str(p) for p in Path(dummy_dir).glob("*.bmp"))[:20]
            self.camera = DummyCamera(paths, self.logger)
        else:
            # 真实相机：MVS SDK 必须可用，否则启动失败（fail-fast）
            if not MVS_AVAILABLE:
                raise RuntimeError(
                    "MVS SDK not installed! Camera initialization aborted. "
                    "Either install MvImport wheel (x86_64) into the container, "
                    "or set USE_DUMMY_CAMERA=1 for debug only."
                )
            self.camera = HikCamera(cam_cfg, self.logger)
        self.camera.open()

        pp_cfg = PreprocessConfig(**self.cfg["preprocess"])
        self.preprocessor = Preprocessor(pp_cfg, self.logger)

        det_cfg = DetectorConfig(
            model_path=str(Path(self.cfg_path).parent / self.cfg["model"]["path"]),
            device=self.cfg["model"]["device"],
            imgsz=self.cfg["model"]["imgsz"],
            conf=self.cfg["model"]["conf"],
            warmup=self.cfg["model"]["warmup"],
            classes=self.cfg["classes"]["names"],
            plc_defect_id=self.cfg["classes"]["plc_defect_id"],
        )
        self.detector = build_detector(det_cfg, self.logger)
        if det_cfg.warmup:
            self.detector.warmup()

        self.comm = build_comm(self.cfg.get("plc", {}), self.logger)
        self.comm.open()

    # ----------------------------------------------------------- main loop
    def run_once(self) -> dict:
        """执行一次完整的"触发 → 推理 → 上报"循环。"""
        # 1) 触发
        timeout = self.cfg["runtime"]["vision_timeout_ms"]
        if not self.comm.read_trigger():
            return {"status": "no_trigger"}

        t0 = now_ms()

        # 2) 抓帧（带重试）
        image: np.ndarray | None = None
        try:
            for _ in range(self.cfg["runtime"]["camera_retry"]):
                image = self.camera.capture(timeout_ms=timeout)
                if image is not None:
                    break
        except VisionError as e:
            self.logger.error("camera failed: %s", e)
            self._report_fail()
            return {"status": "camera_error"}

        # 3) 预处理
        image = self.preprocessor.run(image)

        # 4) 推理（cls 模型返回 ClsResult）
        result: ClsResult = self.detector.detect(image)

        # 5) 置信度阈值检查
        if result.conf < self.cfg["model"]["conf"]:
            self.logger.warning("low confidence %.2f, report fail_safe",
                                result.conf)
            self._report_fail()
            return {"status": "low_conf", "conf": result.conf,
                    "elapsed_ms": now_ms() - t0}

        # 6) 转换为 PLC defect_id 并上报
        defect_id = self.detector.to_plc_defect_id(result.cls_name)
        self._seq += 1
        self.comm.write_result(PlcResult(
            defect_class=defect_id,
            confidence=result.conf,
            valid=True,
            seq=self._seq,
        ))

        # 7) 可选：保存调试图
        if self.cfg["logging"].get("save_debug_image", False):
            self._save_debug(image, result)

        out = format_result(defect_id, result.conf, 0.0, 0.0)
        out["status"] = "ok"
        out["cls"] = result.cls_name
        out["topk"] = result.topk[:3]
        out["seq"] = self._seq
        out["elapsed_ms"] = now_ms() - t0
        self.logger.info("OK %s in %d ms", out, out["elapsed_ms"])
        return out

    def _report_fail(self) -> None:
        """视觉失败时上报 fail_safe_defect。"""
        try:
            self._seq += 1
            self.comm.write_result(PlcResult(
                defect_class=self.cfg["runtime"]["fail_safe_defect"],
                confidence=0.0,
                valid=False,
                seq=self._seq,
            ))
        except Exception as e:  # noqa: BLE001
            self.logger.error("report_fail failed: %s", e)

    def _save_debug(self, image: np.ndarray, result: ClsResult) -> None:
        """保存调试图，叠加 top-1 类别文本。"""
        from .utils import save_debug_image
        import cv2
        img = image.copy()
        text = f"{result.cls_name}: {result.conf:.2f}"
        cv2.putText(img, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        save_debug_image(
            img,
            Path(self.cfg["logging"]["debug_image_dir"]) /
            f"dbg_{now_ms()}.png",
        )

    # ----------------------------------------------------------- run
    def run(self) -> None:
        self._init_components()
        self._running = True
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        self.logger.info("vision main loop started")
        while self._running:
            try:
                self.run_once()
            except Exception as e:  # noqa: BLE001
                self.logger.exception("loop error: %s", e)
                time.sleep(0.5)
        self.shutdown()

    def _on_signal(self, signum, frame) -> None:  # noqa: ARG002
        self.logger.info("signal %d received, shutting down", signum)
        self._running = False

    def shutdown(self) -> None:
        try:
            if self.camera:
                self.camera.close()
        except Exception:
            pass
        try:
            if self.comm:
                self.comm.close()
        except Exception:
            pass
        self.logger.info("vision shutdown done")


# ---------------------------------------------------------------------------
# 调试：列出可见 GigE 相机
# ---------------------------------------------------------------------------
def list_ports(logger) -> int:
    """枚举所有 GigE/USB 相机，仅供现场调试。"""
    try:
        from MvImport.MvCameraControl_class import MvCamera
        from MvImport.CameraParams_header import MV_GIGE_DEVICE, MV_USB_DEVICE
    except Exception as e:
        logger.error("MVS SDK not installed: %s", e)
        return 1
    device_list = MvCamera.MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(
        MV_GIGE_DEVICE | MV_USB_DEVICE, device_list
    )
    if ret != 0:
        logger.error("EnumDevices failed, ret=%d", ret)
        return 1
    n = device_list.nDeviceNum
    logger.info("found %d camera(s)", n)
    for i in range(n):
        info = device_list.pDeviceInfo[i]
        try:
            # GigE info
            name = info.SpecialInfo.stGigEInfo.chModelName.decode(
                "utf-8", errors="ignore")
            ip = ".".join(str(b) for b in info.SpecialInfo.stGigEInfo.nCurrentIp)
            sn = info.SpecialInfo.stGigEInfo.chSerialNumber.decode(
                "utf-8", errors="ignore")
        except Exception:
            name = ip = sn = "?"
        logger.info("[%d] %s | ip=%s | sn=%s", i, name, ip, sn)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vision main (cls defect v2.3)")
    parser.add_argument(
        "-c", "--config", default="vision/config.yaml",
        help="path to config.yaml",
    )
    parser.add_argument(
        "--dummy", action="store_true",
        help="use DummyCamera (for demo without hardware)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="run once then exit (for test)",
    )
    parser.add_argument(
        "--list-ports", action="store_true",
        help="enumerate visible GigE/USB cameras and exit",
    )
    args = parser.parse_args(argv)

    if args.list_ports:
        logger = setup_logger("vision", log_dir="./logs", name="vision")
        return list_ports(logger)

    app = VisionApp(args.config)
    if args.dummy:
        app.cfg.setdefault("runtime", {})["dummy_camera"] = True
    if args.once:
        app._init_components()
        out = app.run_once()
        print(out)
        app.shutdown()
        return 0
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())