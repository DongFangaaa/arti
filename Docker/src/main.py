"""视觉主程序（视觉逻辑算法应用赛 · 缺陷分类版 v3.4 — TCP 帧协议版）。

本文件即运行程序：VisionApp 主流程编排 + CLI 入口。
功能模块：
  config.py      运行参数配置（AppConfig）
  camera.py      海康相机连接与抓帧（HikCamera，按 IP 192.168.3.35 匹配）
  preprocess.py  图像预处理（灰度 → 高斯去噪 → CLAHE）
  detector.py    YOLOv11n 缺陷检测推理（ONNX，类别 → 组号 1~4）
  plc_comm.py    TCP 帧协议通讯（S0E/S1E 启停指令，S1E~S4E 组号上报）

通讯流程（与 plc_comm.py 协议一致，帧格式 S<数字>E\\n）：
  1. 视觉端作为 TCP 服务端监听（默认 0.0.0.0:2000），等待 PLC 连接
  2. PLC → 视觉：S1E = 启动检测 / S0E = 停止检测
  3. 启动状态下循环：抓帧 → 预处理 → 推理 → 发送组号帧 S1E~S4E
     （组号映射：1=hole 2=notch 3=scratch 4=stain）
  4. PLC 断线 → 关闭连接，回到监听状态等待重连

异常约定：
  相机故障 / 推理异常 / 低置信度 → 不发送组号，仅记日志。
  （TCP 帧协议无"结果无效"标志位，PLC 侧超时未收到组号应自行报警。）

运行方式（Podman 容器内）：
  podman run --network=host -d --restart=always \
      --name vision \
      -v /opt/mvs_sdk:/opt/mvs_sdk:ro \
      -v /opt/vision_logs:/app/logs \
      localhost/vision:3.4
  # --network=host：容器直接访问相机（192.168.3.35），
  # 并在 0.0.0.0:2000 监听 PLC 的 TCP 连接

调试：
  python src/main.py --list-cameras     # 枚举可见相机（安装调试阶段用）
  python src/main.py --once             # 不接 PLC 强制执行一次检测（测试用）
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

import numpy as np

try:
    from .camera import CameraBase, HikCamera
    from .config import AppConfig, VisionError, now_ms
    from .detector import ClsResult, DefectClassifier
    from .live_view import CameraLiveView
    from .plc_comm import TCP
    from .preprocess import Preprocessor
except ImportError:  # 脚本模式（python src/main.py）回退
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from camera import CameraBase, HikCamera
    from config import AppConfig, VisionError, now_ms
    from detector import ClsResult, DefectClassifier
    from live_view import CameraLiveView
    from plc_comm import TCP
    from preprocess import Preprocessor

# PLC 指令帧（S<数字>E）：与 plc_comm.py 模块 docstring 保持一致
CMD_STOP = 0    # S0E 停止视觉
CMD_START = 1   # S1E 启动视觉
FRAME_RESULT_TIMEOUT_S = 3.0


# =============================================================================
# 视觉应用主流程
# =============================================================================
class VisionApp:
    """视觉应用：监听 PLC 指令 → 拍照 → 预处理 → 推理 → 发送组号。"""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.logger = self._setup_logger()
        self.camera: CameraBase | None = None
        self.preprocessor: Preprocessor | None = None
        self.preview_preprocessor: Preprocessor | None = None
        self.detector: DefectClassifier | None = None
        self.comm: TCP | None = None
        self.live_view: CameraLiveView | None = None
        self._camera_lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._manual_yolo_enabled = threading.Event()
        self._shutdown_event = threading.Event()
        self._manual_yolo_thread: threading.Thread | None = None
        self._running = False
        self._detecting = False    # PLC S1E 启动 / S0E 停止

    def _setup_logger(self) -> logging.Logger:
        Path(self.cfg.log_dir).mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("vision")
        logger.setLevel(getattr(logging, self.cfg.log_level.upper(), logging.INFO))
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        fh = logging.FileHandler(
            Path(self.cfg.log_dir) / "vision.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        return logger

    # ---------------- 初始化 ----------------
    def _init_components(self, with_comm: bool = True) -> None:
        # 1) 相机：海康 GigE 相机（按 IP 匹配，成像参数由 MVS 手动调节）
        self.camera = HikCamera(self.cfg, self.logger)
        self.camera.open()

        # 识别与预览分别持有预处理器，避免两个线程并发使用同一 CLAHE 实例。
        self.preprocessor = Preprocessor(self.cfg, self.logger)
        self.preview_preprocessor = Preprocessor(self.cfg, self.logger)

        # 实时预览和识别共用一个相机句柄，所有 MVS 抓帧调用串行化。
        if with_comm:
            self.live_view = CameraLiveView(
                camera=self.camera,
                camera_lock=self._camera_lock,
                logger=self.logger,
                host="0.0.0.0",
                port=8080,
                fps=5.0,
                frame_transform=self.preview_preprocessor.run,
            )
            self.live_view.start()

        # 2) 彩色预处理已初始化：去噪、亮度增强、动态外圆分割。

        # 3) 缺陷分类模型
        self.detector = DefectClassifier(self.cfg, self.logger)

        # 网页手动 YOLO 只更新预览，不发送 PLC 分类包。
        if with_comm and self.live_view is not None:
            self.live_view.set_yolo_controls(
                self.start_manual_yolo,
                self.stop_manual_yolo,
                self.manual_yolo_running,
            )
            self._manual_yolo_thread = threading.Thread(
                target=self._manual_yolo_loop,
                name="manual-yolo-preview",
                daemon=True,
            )
            self._manual_yolo_thread.start()

        # 4) PLC 通讯（TCP 帧协议，服务端模式；--once 调试可跳过）
        if with_comm:
            self.comm = TCP(self.cfg.tcp_host, self.cfg.tcp_port)

    # ---------------- 单次检测 + 上报 ----------------
    def _recognize_frame(self, raw_image: np.ndarray, run_options):
        """预处理并识别一张图片；通讯发送由主线程完成。"""
        image = self.preprocessor.run(raw_image)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        with self._inference_lock:
            result = self.detector.detect(image, run_options=run_options)
        return image, result

    def start_manual_yolo(self) -> None:
        self._manual_yolo_enabled.set()
        self.logger.info("网页手动 YOLO 已启动（仅更新预览，不发送 PLC）")

    def stop_manual_yolo(self) -> None:
        self._manual_yolo_enabled.clear()
        if self.live_view is not None:
            self.live_view.clear_detection()
        self.logger.info("网页手动 YOLO 已关闭")

    def manual_yolo_running(self) -> bool:
        return self._manual_yolo_enabled.is_set()

    def _manual_yolo_loop(self) -> None:
        """网页手动识别循环；PLC 正式识别始终优先。"""
        while not self._shutdown_event.is_set():
            if not self._manual_yolo_enabled.wait(timeout=0.2):
                continue
            if self._detecting or self.live_view is None:
                self._shutdown_event.wait(0.1)
                continue
            frame = self.live_view.get_latest_frame()
            if frame is None:
                self._shutdown_event.wait(0.1)
                continue
            try:
                image, result = self._recognize_frame(frame, run_options=None)
                # 识别期间可能按下“关闭”，此时丢弃在途结果，保持 WAIT。
                if not self._manual_yolo_enabled.is_set():
                    self.live_view.clear_detection()
                    continue
                valid = result.cls_name != "none" and result.conf >= self.cfg.conf
                group = (self.detector.to_plc_group_id(result.cls_name)
                         if valid else None)
                self.live_view.update_detection(
                    image, result, group=group, valid=valid)
                self.logger.info(
                    "网页手动 YOLO: cls=%s conf=%.4f group=%s valid=%s",
                    result.cls_name, result.conf, group, valid)
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("网页手动 YOLO 推理失败: %s", exc)
            # 推理结束后留出短间隔，同时让关闭按钮快速生效。
            for _ in range(5):
                if (self._shutdown_event.wait(0.1)
                        or not self._manual_yolo_enabled.is_set()):
                    break

    def _detect_and_send(self) -> dict:
        """抓取一张最新图片并识别；3秒无结果时终止推理并等待重拍。"""
        t0 = now_ms()

        # 丢弃上一次推理期间积压的帧，确保本次抓到当前画面。
        raw_image = None
        last_err = None
        with self._camera_lock:
            try:
                self.camera.clear_buffer()
            except VisionError as e:
                self.logger.warning("清空相机缓存失败，继续抓取当前帧: %s", e)

            for _ in range(self.cfg.camera_retry):
                try:
                    raw_image = self.camera.capture(
                        timeout_ms=self.cfg.vision_timeout_ms)
                    break
                except VisionError as e:
                    last_err = e
                    self.logger.warning("当前帧抓取失败，重试: %s", e)

        if raw_image is None:
            self.logger.error("当前帧抓取失败: %s（下一循环重新拍摄）", last_err)
            return {"status": "camera_error", "sent": False,
                    "elapsed_ms": now_ms() - t0}

        frame_number = getattr(self.camera, "last_frame_number", 0)
        if self.live_view is not None:
            self.live_view.clear_detection()
            self.live_view.update_frame(raw_image)
        frame_hash = hashlib.sha256(raw_image.tobytes()).hexdigest()[:12]
        self.logger.info(
            "已抓取当前帧 number=%s sha256=%s", frame_number, frame_hash)

        # 预处理和ONNX推理合计最多等待3秒。超时终止旧推理，
        # 旧结果永不发送；主循环随后重新获取最新画面。
        run_options = self.detector.create_run_options()
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="vision-current-frame",
        )
        frame_started = time.monotonic()
        future = executor.submit(
            self._recognize_frame, raw_image, run_options)
        try:
            image, result = future.result(timeout=FRAME_RESULT_TIMEOUT_S)
        except FutureTimeoutError:
            self.detector.cancel(run_options)
            self.logger.warning(
                "当前帧在%.1fs内未识别完成，终止旧推理；下一循环重新拍摄当前画面",
                FRAME_RESULT_TIMEOUT_S)
            try:
                future.result(timeout=0.5)
            except Exception:
                pass
            return {"status": "timeout", "sent": False,
                    "frame_number": frame_number,
                    "frame_hash": frame_hash,
                    "elapsed_ms": now_ms() - t0}
        except Exception as e:  # noqa: BLE001
            self.logger.exception(
                "当前帧推理异常: %s（下一循环重新拍摄）", e)
            return {"status": "inference_error", "sent": False,
                    "frame_number": frame_number,
                    "frame_hash": frame_hash,
                    "elapsed_ms": now_ms() - t0}
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        frame_elapsed = time.monotonic() - frame_started
        if frame_elapsed > FRAME_RESULT_TIMEOUT_S:
            self.logger.warning(
                "当前帧结果在%.1fs后才到达，结果作废；下一循环重新拍摄",
                frame_elapsed)
            return {"status": "late_result", "sent": False,
                    "frame_number": frame_number,
                    "frame_hash": frame_hash,
                    "elapsed_ms": now_ms() - t0}

        if result.cls_name == "none" or result.conf < self.cfg.conf:
            if self.live_view is not None:
                self.live_view.update_detection(
                    image, result, group=None, valid=False)
            self.logger.warning(
                "当前帧没有有效识别结果：top-1=%s conf=%.3f < %.2f；"
                "下一循环重新拍摄",
                result.cls_name, result.conf, self.cfg.conf)
            return {"status": "low_conf", "sent": False,
                    "frame_number": frame_number,
                    "frame_hash": frame_hash,
                    "cls": result.cls_name,
                    "conf": round(result.conf, 4),
                    "elapsed_ms": now_ms() - t0}

        group = self.detector.to_plc_group_id(result.cls_name)
        if self.live_view is not None:
            self.live_view.update_detection(
                image, result, group=group, valid=True)
        self.comm.send(group)
        elapsed = now_ms() - t0
        out = {
            "status": "ok",
            "sent": True,
            "frame_number": frame_number,
            "frame_hash": frame_hash,
            "cls": result.cls_name,
            "group": group,
            "conf": round(result.conf, 4),
            "topk": result.topk,
            "elapsed_ms": elapsed,
        }
        self.logger.info("单帧检测完成 %s（%d ms）", out, elapsed)
        if self.cfg.save_debug_image:
            self._save_debug(image, result, valid=True)
        return out

    def _save_debug(self, image: np.ndarray, result: ClsResult, valid: bool) -> None:
        import cv2
        Path(self.cfg.debug_image_dir).mkdir(parents=True, exist_ok=True)
        img = image.copy()
        color = (0, 255, 0) if valid else (0, 0, 255)
        cv2.putText(img, f"{result.cls_name}: {result.conf:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.imwrite(str(Path(self.cfg.debug_image_dir) / f"dbg_{now_ms()}.png"), img)

    # ---------------- 主循环 ----------------
    def run(self) -> None:
        self._init_components()
        self._running = True
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)
        self.logger.info("=" * 50)
        self.logger.info("视觉系统已启动（TCP 帧协议，监听 %s:%d）",
                         self.cfg.tcp_host, self.cfg.tcp_port)
        self.logger.info("=" * 50)

        while self._running:
            # ---- 等待 PLC 连接（断线后回到这里重连）----
            try:
                self.logger.info("等待 PLC 连接 ...")
                addr = self.comm.start(should_stop=lambda: not self._running)
                self.logger.info("PLC 已连接: %s:%d", addr[0], addr[1])
            except ConnectionError:
                break    # should_stop 触发（容器停止 / Ctrl-C）
            except OSError as e:
                self.logger.error("TCP 监听失败 %s:%d — %s，3s 后重试",
                                  self.cfg.tcp_host, self.cfg.tcp_port, e)
                time.sleep(3.0)
                continue

            # ---- 指令循环：S1E 启动 / S0E 停止 ----
            self._detecting = False
            try:
                while self._running:
                    cmd = self.comm.receive(timeout=self.cfg.tcp_cmd_timeout_s)
                    if cmd == CMD_START:
                        if self._manual_yolo_enabled.is_set():
                            self._manual_yolo_enabled.clear()
                            self.logger.info("PLC 正式识别启动，已自动暂停网页手动 YOLO")
                            # 等待正在执行的最后一次网页推理退出，再开始 PLC
                            # 的3秒计时，避免手动模式占用模型导致正式识别超时。
                            with self._inference_lock:
                                pass
                        if not self._detecting:
                            self.logger.info("收到启动指令 S1E，开始检测")
                        self._detecting = True
                    elif cmd == CMD_STOP:
                        if self._detecting:
                            self.logger.info("收到停止指令 S0E，停止检测")
                        self._detecting = False
                    if self._detecting:
                        detect_result = self._detect_and_send()
                        # 一次S1只上报一个结果；失败则保持检测状态，
                        # 下一循环轮询PLC后重新拍摄最新画面。
                        if detect_result.get("sent", False):
                            self._detecting = False
            except ConnectionError as e:
                self.logger.error("PLC 连接中断: %s，等待重连 ...", e)
                self.comm.close()
            except Exception as e:  # noqa: BLE001
                self.logger.exception("主循环异常: %s", e)
                time.sleep(0.5)

        self.shutdown()

    def _on_signal(self, signum, frame) -> None:  # noqa: ARG002
        self.logger.info("收到信号 %d，正在退出...", signum)
        self._running = False

    def shutdown(self) -> None:
        self._manual_yolo_enabled.clear()
        self._shutdown_event.set()
        if (self._manual_yolo_thread is not None
                and self._manual_yolo_thread.is_alive()):
            self._manual_yolo_thread.join(timeout=3.5)
        if self.live_view is not None:
            try:
                self.live_view.stop()
            except Exception:  # noqa: BLE001
                pass
        for dev in (self.camera, self.comm):
            if dev is not None:
                try:
                    dev.close()
                except Exception:  # noqa: BLE001
                    pass
        self.logger.info("视觉系统已关闭")

    # ---------------- --once 测试模式（不接 PLC） ----------------
    def run_once_direct(self) -> dict:
        """不连接 PLC，强制执行一次 抓帧→预处理→推理（--once 调试用）。"""
        t0 = now_ms()
        with self._camera_lock:
            image = self.camera.capture(timeout_ms=self.cfg.vision_timeout_ms)
        image = self.preprocessor.run(image)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        result = self.detector.detect(image)
        group = self.detector.to_plc_group_id(result.cls_name)
        return {"cls": result.cls_name, "group": group,
                "conf": round(result.conf, 4),
                "valid": bool(result.conf >= self.cfg.conf),
                "elapsed_ms": now_ms() - t0}


# =============================================================================
# 调试工具：枚举可见 GigE 相机（安装调试阶段用）
# =============================================================================
def list_cameras() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("vision")
    try:
        cfg = AppConfig()
        cam = HikCamera(cfg, logger)
        found, _dev_list = cam._enum_gige()  # ← 修改：接收两个返回值
    except VisionError as e:
        logger.error("%s", e)
        return 1
    
    logger.info("共发现 %d 台 GigE 相机:", len(found))
    for idx, ip, model, sn, _dev_info in found:  # ← 修改：解包 5 个值
        mark = "  <== 目标相机" if ip == cfg.camera_ip else ""
        logger.info("  [%d] %s | ip=%s | sn=%s%s", idx, model, ip, sn, mark)
    return 0


# =============================================================================
# CLI 入口
# =============================================================================
def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="视觉逻辑算法应用赛 · 缺陷分类视觉主程序 v3.4")
    parser.add_argument("-c", "--config", default="",
                        help="指定 YAML 配置文件（默认自动加载项目根目录 config.yaml）")
    parser.add_argument("--once", action="store_true",
                        help="不接 PLC，强制执行一次检测流程后退出（测试用）")
    parser.add_argument("--list-cameras", action="store_true",
                        help="枚举可见 GigE 相机后退出（安装调试用）")
    args = parser.parse_args(argv)

    # 安装调试：枚举相机
    if args.list_cameras:
        return list_cameras()

    # 配置加载优先级：--config 指定 > 项目根目录 config.yaml > 内置默认值
    default_cfg = Path(__file__).resolve().parent.parent / "config.yaml"
    if args.config:
        cfg_path = args.config
    elif default_cfg.exists():
        cfg_path = str(default_cfg)
    else:
        cfg_path = ""
    if cfg_path:
        print(f"[main] 使用配置文件: {cfg_path}")
        cfg = AppConfig.from_yaml(cfg_path)
    else:
        print("[main] 未找到 config.yaml，使用内置默认值")
        cfg = AppConfig()
    app = VisionApp(cfg)

    # 测试：不接 PLC，强制执行一次检测
    if args.once:
        app._init_components(with_comm=False)
        print(app.run_once_direct())
        app.shutdown()
        return 0

    # 正式运行：监听 PLC 指令（S1E 启动 / S0E 停止）
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
