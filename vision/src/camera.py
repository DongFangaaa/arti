"""海康 MVS 工业相机封装。

提供：
  - HikCamera.open()      枚举并打开第一台 GigE 相机
  - HikCamera.set_params() 设置曝光/增益/ROI
  - HikCamera.capture()    取一帧 BGR numpy
  - HikCamera.trigger_once() 软触发一次（硬件触发模式下使用）
  - HikCamera.close()
  - DummyCamera           调试用：从本地目录循环读取 PNG/BMP

依赖：海康 MVS SDK 的 Python 绑定 `MvImport` / `MVS`
     （机器视觉部门提供的 wheel，对应版本 3.x，**EPC 1502 装 x86_64 版**）。

DummyCamera 启用方式：
    USE_DUMMY_CAMERA=1 python -m vision.main   # 环境变量开启（调试时）
    不设置环境变量 → 强制使用 HikCamera，SDK 缺失直接抛异常（fail-fast）
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

# 是否启用 DummyCamera（默认关闭，EPC 现场必须装 MVS SDK）
USE_DUMMY_CAMERA = os.getenv("USE_DUMMY_CAMERA", "0") == "1"

try:
    # 现场实际安装的模块名可能是 MvImport 或 MVS，根据 MVS 版本调整
    from MvImport.CameraParams_header import (
        MV_GIGE_DEVICE,
        MV_USB_DEVICE,
        MV_ACCESS_Exclusive,
    )
    from MvImport.MvCameraControl_class import MvCamera
    MVS_AVAILABLE = True
except Exception:  # noqa: BLE001
    MVS_AVAILABLE = False

from .utils import VisionError, retry


@dataclass
class CameraConfig:
    ip: str = "192.168.3.35"
    exposure_time: int = 8000          # μs
    gain: float = 12.0
    frame_rate: float = 10.0
    trigger_mode: str = "software"     # software | hardware
    pixel_format: str = "BayerRG8"
    roi_enable: bool = False
    roi_x: int = 0
    roi_y: int = 0
    roi_w: int = 720
    roi_h: int = 540


class HikCamera:
    """海康工业相机封装，屏蔽 SDK 细节。"""

    def __init__(self, cfg: CameraConfig, logger: logging.Logger | None = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger("vision.camera")
        self.handle: Any = None
        self._opened = False

    # ------------------------------------------------------------------ open
    def open(self) -> bool:
        """枚举并打开第一台匹配的 GigE 相机。"""
        if not MVS_AVAILABLE:
            raise VisionError("MVS SDK not installed (MvImport missing)")
        # 1) 枚举
        device_list = MvCamera.MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(
            MV_GIGE_DEVICE | MV_USB_DEVICE, device_list
        )
        if ret != 0 or device_list.nDeviceNum == 0:
            raise VisionError(f"MV_CC_EnumDevices failed, ret={ret}")

        # 2) 默认打开第一台
        cam = MvCamera()
        ret = cam.MV_CC_CreateHandle(device_list.pDeviceInfo[0])
        if ret != 0:
            raise VisionError(f"MV_CC_CreateHandle failed, ret={ret}")

        # 3) 打开
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise VisionError(f"MV_CC_OpenDevice failed, ret={ret}")

        self.handle = cam
        self._opened = True
        self.logger.info("camera opened, sn=%s",
                         self._safe_get_serial(cam))
        self.set_params()
        return True

    @staticmethod
    def _safe_get_serial(cam) -> str:
        try:
            buf = (ctypes := __import__("ctypes")).create_string_buffer(64)
            cam.MV_CC_GetDeviceSerialNumber(buf)
            return buf.value.decode("utf-8", errors="ignore")
        except Exception:
            return "unknown"

    # ------------------------------------------------------------------ params
    def set_params(self) -> None:
        """按 CameraConfig 设置曝光 / 增益 / 帧率 / 触发模式。"""
        cam = self.handle
        cam.MV_CC_SetFloatValue("ExposureTime", float(self.cfg.exposure_time))
        cam.MV_CC_SetFloatValue("Gain", float(self.cfg.gain))
        cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(self.cfg.frame_rate))

        if self.cfg.trigger_mode == "software":
            cam.MV_CC_SetEnumValue("TriggerMode", 1)         # on
            cam.MV_CC_SetEnumValue("TriggerSource", 7)        # software
        else:
            cam.MV_CC_SetEnumValue("TriggerMode", 0)         # off (continuous)

        if self.cfg.roi_enable:
            cam.MV_CC_SetIntValue("OffsetX", self.cfg.roi_x)
            cam.MV_CC_SetIntValue("OffsetY", self.cfg.roi_y)
            cam.MV_CC_SetWidth(self.cfg.roi_w)
            cam.MV_CC_SetHeight(self.cfg.roi_h)

    # ------------------------------------------------------------------ capture
    def capture(self, timeout_ms: int = 1000) -> np.ndarray:
        """抓取一帧并转换为 BGR ndarray。"""
        if not self._opened:
            raise VisionError("camera not opened")

        st_frame = None
        try:
            from MvImport.CameraParams_header import MV_FRAME_OUT_INFO_EX
            st_frame = MV_FRAME_OUT_INFO_EX()
        except Exception:
            st_frame = None

        buf_size = 720 * 540 * 3
        pdata = (ctypes := __import__("ctypes")).create_string_buffer(buf_size)
        ret = self.handle.MV_CC_GetOneFrameTimeout(
            pdata, buf_size, st_frame, timeout_ms
        )
        if ret != 0:
            raise VisionError(f"MV_CC_GetOneFrameTimeout ret={ret}")

        w = st_frame.nWidth if st_frame else 720
        h = st_frame.nHeight if st_frame else 540
        img = np.frombuffer(pdata.raw, dtype=np.uint8).reshape(h, w, 3)[:, :, :3]
        # BayerRG8 → BGR（SDK 输出时若已是 BGR 则跳过）
        if self.cfg.pixel_format == "BayerRG8":
            img = self._bayer_to_bgr(img, w, h)
        return img

    @staticmethod
    def _bayer_to_bgr(bayer: np.ndarray, w: int, h: int) -> np.ndarray:
        """BayerRG8 → BGR 转换。"""
        bgr = np.zeros((h, w, 3), dtype=np.uint8)
        # 简化处理：SDK 输出 Bayer 时用 OpenCV 解码
        try:
            import cv2
            bgr = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2BGR)
        except Exception:
            bgr[:, :, 0] = bayer[:, :, 0]
        return bgr

    # ------------------------------------------------------------------ trigger
    def trigger_once(self) -> None:
        """发送软触发命令。"""
        if not self._opened:
            raise VisionError("camera not opened")
        ret = self.handle.MV_CC_SetCommandValue("TriggerSoftware")
        if ret != 0:
            raise VisionError(f"TriggerSoftware failed, ret={ret}")

    # ------------------------------------------------------------------ close
    def close(self) -> None:
        if self._opened and self.handle is not None:
            try:
                self.handle.MV_CC_CloseDevice()
                self.handle.MV_CC_DestroyHandle()
            except Exception:
                pass
            self._opened = False
            self.logger.info("camera closed")

    # ------------------------------------------------------------------ context
    def __enter__(self) -> "HikCamera":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()


# ---------------------------------------------------------------------------
# 离线占位（无相机时用于 demo）
# ---------------------------------------------------------------------------
class DummyCamera:
    """无硬件相机时的占位实现：从文件循环读取。"""

    def __init__(self, image_paths: list[str], logger: logging.Logger | None = None):
        import cv2
        self.logger = logger or logging.getLogger("vision.dummy_cam")
        self.images = [cv2.imread(p) for p in image_paths]
        self.images = [im for im in self.images if im is not None]
        if not self.images:
            raise VisionError("DummyCamera: no valid images")
        self.idx = 0

    def open(self) -> bool:
        self.logger.info("dummy camera opened, %d frames", len(self.images))
        return True

    def set_params(self) -> None:
        return None

    def capture(self, timeout_ms: int = 1000) -> np.ndarray:
        time.sleep(0.05)
        img = self.images[self.idx % len(self.images)]
        self.idx += 1
        return img

    def trigger_once(self) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()