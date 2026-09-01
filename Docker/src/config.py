"""配置模块（视觉逻辑算法应用赛 · v3.4 — TCP 帧协议 + ONNX 检测版）。

集中管理全部运行参数：默认值即比赛现场配置，
也可通过 YAML 文件覆盖（项目根目录 config.yaml 自动加载）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path


class VisionError(Exception):
    """视觉系统错误基类（相机/模型/通讯统一抛出）。"""


def now_ms() -> int:
    """当前时间戳（毫秒），用于计时与序列号日志。"""
    return int(time.time() * 1000)


@dataclass
class AppConfig:
    """全部运行参数的内置默认值。

    注：相机曝光/增益等成像参数不在此配置 —— 由海康 MVS 客户端
    手动调节并保存在相机内部，程序不覆盖相机参数。
    """

    # ---- 相机（海康 GigE，赛题视觉检测系统）----
    camera_ip: str = "192.168.3.253"    # 赛题现场相机 IP

    # ---- 模型（ONNX Runtime 推理，YOLOv11n 检测模型导出版）----
    model_path: str = ""                # 空 → 自动定位 models/best.onnx
    device: str = "cpu"                 # EPC1502 无独显，用 cpu
    imgsz: int = 640                    # 仅兜底；实际输入尺寸以 ONNX 模型为准
    conf: float = 0.55                  # 置信度阈值（低于此值不发送组号，仅记日志）
    warmup: bool = True

    # ---- 缺陷类别 → PLC 组号映射（S<组号>E 帧，组号 1~4）----
    # 类别名以训练 dataset.yaml / ONNX 元数据为准（notch = 缺角）
    class_names: list = field(
        default_factory=lambda: ["hole", "notch", "scratch", "stain"])
    plc_group_id: dict = field(
        default_factory=lambda: {"hole": 1, "notch": 2, "scratch": 3, "stain": 4})

    # ---- PLC 通讯（原始 TCP 帧协议，见 plc_comm.py）----
    # 视觉端作为 TCP 服务端监听，PLC 作为客户端主动连接。
    # 报文：PLC→视觉 S0E=停止 / S1E=启动；视觉→PLC S1E~S4E=缺陷组号
    tcp_host: str = "0.0.0.0"           # 监听地址（0.0.0.0 = 所有网卡）
    tcp_port: int = 2000                # 监听端口（需与 PLC 侧约定一致）
    tcp_cmd_timeout_s: float = 0.05     # 指令轮询超时（兼顾响应速度与 CPU 占用）

    # ---- 预处理（灰度 → 高斯去噪 → CLAHE）----
    pre_enable: bool = True
    pre_denoise_ksize: int = 5          # 高斯核大小（奇数），0 表示自动 (3)
    pre_clahe: bool = True
    pre_clahe_clip: float = 2.0         # clipLimit，建议 2.0~4.0
    pre_clahe_tile: int = 8             # tileGridSize

    # ---- 运行时 ----
    vision_timeout_ms: int = 1000       # 单次抓帧超时
    camera_retry: int = 3               # 抓帧重试次数

    # ---- 多帧投票（单次启动指令内连续采样，取多数表决结果上报）----
    vote_n_frames: int = 3              # 每次采样帧数（1 = 关闭投票，退化为单帧）
    vote_min_samples: int = 2           # 最少多少个有效样本才算"有结果"
    vote_frame_interval_ms: int = 60    # 帧间间隔（毫秒），给相机/物料稳定时间
    vote_force_send: bool = True        # 即使平均置信度低于 conf，仍按多数表决上报

    # ---- 日志 ----
    log_level: str = "INFO"
    log_dir: str = "./logs"
    save_debug_image: bool = False
    debug_image_dir: str = "./logs/debug_images"

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        """从 YAML 文件加载覆盖项（可选；文件不存在则全用默认值）。"""
        cfg = cls()
        p = Path(path)
        if not p.exists():
            return cfg
        import yaml
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

        cam = raw.get("camera", {})
        cfg.camera_ip = cam.get("ip", cfg.camera_ip)

        model = raw.get("model", {})
        cfg.model_path = model.get("path", cfg.model_path)
        cfg.device = model.get("device", cfg.device)
        cfg.imgsz = int(model.get("imgsz", cfg.imgsz))
        cfg.conf = float(model.get("conf", cfg.conf))
        cfg.warmup = bool(model.get("warmup", cfg.warmup))

        classes = raw.get("classes", {})
        if classes.get("names"):
            cfg.class_names = list(classes["names"])
        if classes.get("plc_group_id"):
            cfg.plc_group_id = dict(classes["plc_group_id"])

        tcp = raw.get("tcp", {})
        cfg.tcp_host = tcp.get("host", cfg.tcp_host)
        cfg.tcp_port = int(tcp.get("port", cfg.tcp_port))
        cfg.tcp_cmd_timeout_s = float(
            tcp.get("cmd_timeout_s", cfg.tcp_cmd_timeout_s))

        pre = raw.get("preprocess", {})
        cfg.pre_enable = bool(pre.get("enable", cfg.pre_enable))
        cfg.pre_denoise_ksize = int(pre.get("denoise_ksize", cfg.pre_denoise_ksize))
        cfg.pre_clahe = bool(pre.get("clahe", cfg.pre_clahe))
        cfg.pre_clahe_clip = float(pre.get("clahe_clip", cfg.pre_clahe_clip))
        cfg.pre_clahe_tile = int(pre.get("clahe_tile", cfg.pre_clahe_tile))

        rt = raw.get("runtime", {})
        cfg.vision_timeout_ms = int(rt.get("vision_timeout_ms", cfg.vision_timeout_ms))
        cfg.camera_retry = int(rt.get("camera_retry", cfg.camera_retry))
        cfg.vote_n_frames = max(1, int(rt.get("vote_n_frames", cfg.vote_n_frames)))
        cfg.vote_min_samples = max(
            1, min(cfg.vote_n_frames,
                   int(rt.get("vote_min_samples", cfg.vote_min_samples))))
        cfg.vote_frame_interval_ms = int(
            rt.get("vote_frame_interval_ms", cfg.vote_frame_interval_ms))
        cfg.vote_force_send = bool(
            rt.get("vote_force_send", cfg.vote_force_send))

        log = raw.get("logging", {})
        cfg.log_level = log.get("level", cfg.log_level)
        cfg.log_dir = log.get("dir", cfg.log_dir)
        cfg.save_debug_image = bool(log.get("save_debug_image", cfg.save_debug_image))
        cfg.debug_image_dir = log.get("debug_image_dir", cfg.debug_image_dir)
        return cfg
