"""相机模块（海康 MVS SDK — GigE，TCP 协议接入）。

成像参数（曝光/增益/白平衡等）由海康 MVS 客户端手动调节并保存在相机内，
本模块不做任何成像参数覆盖，只负责：枚举 → 按 IP 匹配 → 连接 → 抓帧 → 关闭。

Podman 容器内使用方式：
  1. MVS SDK（Linux x86_64）由宿主机挂载：-v /opt/mvs_sdk:/opt/mvs_sdk:ro
     环境变量 MVCAM_COMMON_RUNENV / LD_LIBRARY_PATH / PYTHONPATH
     已在 Dockerfile 中预设。
  2. --network=host 使容器与相机处于同一网段（192.168.3.x）。
"""
from __future__ import annotations

import ctypes
import logging

import numpy as np

try:
    from .config import AppConfig, VisionError
except ImportError:  # 脚本模式回退
    from config import AppConfig, VisionError


class CameraBase:
    """相机抽象接口。"""

    def open(self) -> None:  # pragma: no cover - 抽象
        raise NotImplementedError

    def capture(self, timeout_ms: int = 1000) -> np.ndarray:
        raise NotImplementedError

    def clear_buffer(self) -> None:
        """Discard frames queued inside the camera SDK."""
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - 抽象
        raise NotImplementedError


class HikCamera(CameraBase):
    """海康 GigE 工业相机（对应赛题视觉传感器，有效像素 720x540）。"""

    def __init__(self, cfg: AppConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self._cam = None
        self._open = False
        self.last_frame_number = 0
        # 延迟导入：容器内必须装好 MVS SDK，否则 import 即失败
        try:
            from MvImport.MvCameraControl_class import MvCamera  # noqa: F401
        except Exception as e:
            raise VisionError(
                "MVS SDK 不可用（MvImport 导入失败）。容器内请安装海康 MVS SDK "
                "或挂载 /opt/mvs_sdk；仅调试时可设 USE_DUMMY_CAMERA=1。"
            ) from e

    # ---------------- 枚举与匹配 ----------------
    @staticmethod
    def _gige_ip_str(n_current_ip: int) -> str:
        """MVS 的 nCurrentIp 为 32 位整数（大端），转为点分十进制。"""
        return ".".join(str((n_current_ip >> s) & 0xFF) for s in (24, 16, 8, 0))

    @staticmethod
    def _deref_device_info(ptr):
        """把 pDeviceInfo[i] 转为 MV_CC_DEVICE_INFO 实体（MVS 3.x 官方做法）。

        MVS 3.x 的 pDeviceInfo[i] 是指针/地址，直接传给 MV_CC_CreateHandle
        会得到 0x80000004（参数错误），必须先 cast 出结构体实体。
        """
        from MvImport.CameraParams_header import MV_CC_DEVICE_INFO

        # 如果已经是结构体实体（有 SpecialInfo 属性）
        if hasattr(ptr, "SpecialInfo"):
            return ptr

        # 安全转换指针
        try:
            # 如果 ptr 是整数，直接 cast
            if isinstance(ptr, int):
                return ctypes.cast(ptr, ctypes.POINTER(MV_CC_DEVICE_INFO)).contents
            # 如果 ptr 已经是 c_void_p
            elif isinstance(ptr, ctypes.c_void_p):
                return ctypes.cast(ptr, ctypes.POINTER(MV_CC_DEVICE_INFO)).contents
            # 其他情况：尝试直接 cast
            else:
                return ctypes.cast(ptr, ctypes.POINTER(MV_CC_DEVICE_INFO)).contents
        except Exception as e:
            # 最后的兜底方案
            raise TypeError(f"无法转换指针: {ptr} (类型: {type(ptr)})") from e

    @classmethod
    def _extract_gige_info(cls, info):
        """兼容新旧版 MVS SDK：自动解引用指针或直接访问结构体。"""
        try:
            real_info = cls._deref_device_info(info)
            return real_info.SpecialInfo.stGigEInfo
        except Exception as e:
            raise VisionError(
                "无法解析相机设备信息：当前 MVS SDK 版本与代码不兼容，"
                "请检查 SDK 版本或联系技术支持。"
            ) from e

    def _enum_gige(self):
        """枚举所有 GigE 设备，返回 (found_list, dev_list)。

        返回:
            found: [(index, ip, model, sn, dev_info), ...]
            dev_list: MV_CC_DEVICE_INFO_LIST 对象（供 CreateHandle 复用，避免二次枚举）
            dev_info 为 cast 后的 MV_CC_DEVICE_INFO 实体（MVS 3.x 必须，
            直接传 pDeviceInfo[i] 指针会得到 0x80000004）

        关键修复：原版 open() 调用了两次 MV_CC_EnumDevices（一次 _enum_gige，一次手动
        dev_list = ... + EnumDevices），两次调用之间设备状态可能变化，导致
        CreateHandle 返回 0x80000004。改为只枚举一次，复用同一 dev_list。
        """
        from MvImport.CameraParams_header import (
            MV_CC_DEVICE_INFO_LIST, MV_GIGE_DEVICE)
        from MvImport.MvCameraControl_class import MvCamera
        dev_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, dev_list)
        if ret != 0:
            raise VisionError(f"枚举相机失败 ret=0x{ret:08x}")
        found = []
        for i in range(dev_list.nDeviceNum):
            try:
                info = self._deref_device_info(dev_list.pDeviceInfo[i])
                gige = self._extract_gige_info(info)
            except VisionError:
                self.logger.warning("设备 [%d] 信息解析失败，跳过", i)
                continue
            ip = self._gige_ip_str(gige.nCurrentIp)
            model = bytes(gige.chModelName).decode("utf-8", errors="ignore").strip("\x00")
            sn = bytes(gige.chSerialNumber).decode("utf-8", errors="ignore").strip("\x00")
            # 携带设备信息实体，open() 直接用它创建句柄
            found.append((i, ip, model, sn, info))
        return found, dev_list

    # ---------------- 生命周期 ----------------
    def open(self) -> None:
        from MvImport.CameraParams_header import MV_ACCESS_Exclusive
        from MvImport.MvCameraControl_class import MvCamera

        # 只枚举一次，复用 dev_list 创建句柄（关键修复：避免二次枚举导致 0x80000004）
        found, dev_list = self._enum_gige()
        for idx, ip, model, sn, _dev in found:
            self.logger.info("发现相机 [%d] %s ip=%s sn=%s", idx, model, ip, sn)

        if not found:
            raise VisionError(
                "未找到任何 GigE 相机；请检查网线、相机供电与容器网络（--network=host）。")

        # 自动搜寻模式：cfg.camera_ip 为空 / "auto" → 连第一台；
        # 否则按 IP 精确匹配；匹配失败则回退连第一台（保持之前的能力）。
        ip_cfg = (self.cfg.camera_ip or "").strip().lower()
        if ip_cfg in ("", "auto"):
            idx, ip, model, sn, dev_info = found[0]
            self.logger.warning(
                "自动搜寻模式已启用（cfg.camera_ip=%r），将连接第一台相机 %s (%s, sn=%s）",
                self.cfg.camera_ip, ip, model, sn)
        else:
            target = next((f for f in found if f[1] == ip_cfg), None)
            if target is None:
                # 指定 IP 找不到时回退：自动连第一台
                idx, ip, model, sn, dev_info = found[0]
                self.logger.warning(
                    "指定相机 ip=%s 未找到，自动回退连接第一台相机 %s (%s, sn=%s)；"
                    "在线相机: %s",
                    self.cfg.camera_ip, ip, model, sn,
                    [f[1] for f in found])
            else:
                idx, ip, model, sn, dev_info = target
        self.logger.info("目标相机匹配成功: %s (%s, sn=%s)", ip, model, sn)

        # 关键修复（MVS 3.x）：CreateHandle 必须传 cast 后的 MV_CC_DEVICE_INFO
        # 实体；直接传 pDeviceInfo[idx] 指针会得到 0x80000004（参数错误），
        # 进而导致 OpenDevice 同样失败。
        self._cam = MvCamera()
        self._dev_list = dev_list  # 保活设备列表，防止实体指针悬空
        ret = self._cam.MV_CC_CreateHandle(dev_info)
        if ret != 0:
            raise VisionError(f"CreateHandle 失败 ret=0x{ret:08x}")
        ret = self._cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise VisionError(f"OpenDevice 失败 ret=0x{ret:08x}")

        # GigE 最佳包大小（避免丢包；网络传输参数，非成像参数）
        # Select a packet size compatible with the actual NIC MTU.
        packet_size = self._cam.MV_CC_GetOptimalPacketSize()
        if packet_size <= 0:
            self.logger.warning(
                "GetOptimalPacketSize failed value=%s; fallback=1500",
                packet_size)
            packet_size = 1500
        ret = self._cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
        if ret != 0:
            self.logger.warning(
                "Set PacketSize=%d failed ret=0x%08x", packet_size, ret)
        else:
            self.logger.info("GigE PacketSize=%d", packet_size)

        # 关闭外触发，连续采集；曝光/增益等成像参数已由 MVS 手动调好，
        # 此处一律不设置，直接使用相机内部保存的参数。
        self._cam.MV_CC_SetEnumValue("TriggerMode", 0)

        ret = self._cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise VisionError(f"StartGrabbing 失败 ret=0x{ret:08x}")
        self._open = True
        self.logger.info("相机已开启: %s（成像参数使用 MVS 手动调节值）", ip)

    # ---------------- 抓帧 ----------------
    def clear_buffer(self) -> None:
        """Clear queued frames before a new detection cycle."""
        if not self._open:
            raise VisionError("camera is not open")
        ret = self._cam.MV_CC_ClearImageBuffer()
        if ret != 0:
            raise VisionError(f"clear buffer failed ret=0x{ret:08x}")
        self.logger.info("camera frame buffer cleared")

    def capture(self, timeout_ms: int = 1000) -> np.ndarray:
        """抓取一帧并转为 BGR 彩色图（np.ndarray, HxWx3）。

        MVS 3.x 新签名：MV_CC_GetImageBuffer(stFrameOut, timeout)——
        接收缓冲由 SDK 内部分配，帧信息在 stFrameOut.stFrameInfo，
        数据地址在 stFrameOut.pBufAddr；取完必须 MV_CC_FreeImageBuffer 释放。

        像素格式处理：
          - BGR8 Packed → 直接 reshape 返回
          - BayerBG8    → 显式按 BG 排列去马赛克（COLOR_BayerBG2BGR）
          - Mono8       → 灰度转 BGR
          - 其他格式    → 交 MVS SDK 按帧信息自动转换（保底）
        """
        from MvImport.CameraParams_header import (
            MV_FRAME_OUT, PixelType_Gvsp_BGR8_Packed)
        if not self._open:
            raise VisionError("相机未打开")

        st_frame = MV_FRAME_OUT()
        ret = self._cam.MV_CC_GetImageBuffer(st_frame, timeout_ms)
        if ret != 0:
            raise VisionError(f"抓帧超时/失败 ret=0x{ret:08x}")
        try:
            info = st_frame.stFrameInfo
            self.last_frame_number = int(getattr(info, "nFrameNum", 0))
            w, h = info.nWidth, info.nHeight
            frame_len = info.nFrameLen
            pixel_type = info.enPixelType

            # pBufAddr 指向 SDK 内部缓冲（ARM64 兼容处理）
            buf_addr = st_frame.pBufAddr

            # 统一转成整数地址
            if isinstance(buf_addr, int):
                addr = buf_addr
            elif isinstance(buf_addr, (bytes, bytearray)):
                addr = int.from_bytes(buf_addr, "little")
            elif isinstance(buf_addr, ctypes.Array):
                # ctypes 数组，转为字节再解析
                try:
                    addr = int.from_bytes(bytes(buf_addr), "little")
                except:
                    addr = ctypes.addressof(buf_addr)
            elif hasattr(buf_addr, "value"):
                # c_void_p 或指针类型
                addr = buf_addr.value
            elif hasattr(buf_addr, "contents"):
                # 指针指向的内容
                try:
                    addr = ctypes.addressof(buf_addr.contents)
                except:
                    addr = ctypes.cast(buf_addr, ctypes.c_void_p).value
            else:
                # 最后尝试直接转换
                try:
                    addr = int(buf_addr)
                except:
                    raise VisionError(
                        f"无法解析 pBufAddr: {type(buf_addr).__name__} {buf_addr!r}")

            raw = np.ctypeslib.as_array(
                (ctypes.c_ubyte * frame_len).from_address(addr))

            if pixel_type == PixelType_Gvsp_BGR8_Packed:
                return raw[:w * h * 3].reshape(h, w, 3).copy()

            import cv2
            from MvImport.CameraParams_header import (
                PixelType_Gvsp_BayerBG8, PixelType_Gvsp_BayerGB8,
                PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerRG8,
                PixelType_Gvsp_Mono8)
            # 现场相机为 BayerRG8。原映射会造成红蓝通道互换（画面偏青）。
            # 使用互补排列后输出真正的 OpenCV BGR，预览和 YOLO 共用同一颜色。
            bayer_codes = {
                PixelType_Gvsp_BayerBG8: cv2.COLOR_BayerRG2BGR,
                PixelType_Gvsp_BayerGB8: cv2.COLOR_BayerGR2BGR,
                PixelType_Gvsp_BayerGR8: cv2.COLOR_BayerGB2BGR,
                PixelType_Gvsp_BayerRG8: cv2.COLOR_BayerBG2BGR,
            }
            if pixel_type in bayer_codes:
                gray = raw[:w * h].reshape(h, w).copy()
                return cv2.cvtColor(gray, bayer_codes[pixel_type])
            if pixel_type == PixelType_Gvsp_Mono8:
                gray = raw[:w * h].reshape(h, w).copy()
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            self.logger.warning("非常见像素格式 0x%08x，交由 SDK 转换", pixel_type)
            from MvImport.CameraParams_header import MV_CC_PIXEL_CONVERT_PARAM
            convert = MV_CC_PIXEL_CONVERT_PARAM()
            ctypes.memset(ctypes.byref(convert), 0, ctypes.sizeof(convert))
            convert.nWidth = w
            convert.nHeight = h
            convert.enSrcPixelType = pixel_type
            convert.pSrcData = st_frame.pBufAddr
            convert.nSrcDataLen = frame_len
            convert.enDstPixelType = PixelType_Gvsp_BGR8_Packed
            dst_buf = (ctypes.c_ubyte * (w * h * 3))()
            convert.pDstBuffer = dst_buf
            convert.nDstBufferSize = w * h * 3
            ret = self._cam.MV_CC_ConvertPixelType(convert)
            if ret != 0:
                raise VisionError(f"像素格式转换失败 ret=0x{ret:08x}")
            return np.ctypeslib.as_array(dst_buf).reshape(h, w, 3).copy()
        finally:
            self._cam.MV_CC_FreeImageBuffer(st_frame)

    def close(self) -> None:
        if self._cam is not None and self._open:
            try:
                self._cam.MV_CC_StopGrabbing()
                self._cam.MV_CC_CloseDevice()
                self._cam.MV_CC_DestroyHandle()
            except Exception:  # noqa: BLE001
                pass
            self._open = False
            self.logger.info("相机已关闭")