"""相机实时预览 HTTP 服务。

浏览器入口：
  http://192.168.3.20:8080/
  http://192.168.3.20:8080/stream.mjpg
  http://192.168.3.20:8080/snapshot.jpg
  http://192.168.3.20:8080/health
"""
from __future__ import annotations

import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np


_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>大赛设备 · 摄像头实时画面</title>
  <style>
    :root { color-scheme:dark; --cyan:#00a6ad; --panel:#202a31; }
    * { box-sizing:border-box; }
    body { margin:0; min-height:100vh; background:#11181d; color:#eef3f5;
           font-family:"Microsoft YaHei",Arial,sans-serif; }
    header { height:72px; display:flex; align-items:center; gap:18px;
             padding:0 28px; background:#172127; border-bottom:4px solid var(--cyan); }
    .mark { width:36px; height:36px; border:3px solid var(--cyan); border-radius:7px; }
    h1 { margin:0; font-size:25px; font-weight:600; }
    main { padding:24px; }
    .panel { max-width:1280px; margin:auto; background:var(--panel); padding:18px;
             border:1px solid #40505a; border-radius:8px; box-shadow:0 8px 30px #0008; }
    .status { display:flex; align-items:center; gap:10px; margin-bottom:14px; color:#b9c9cf; }
    .dot { width:11px; height:11px; border-radius:50%; background:#23c879;
           box-shadow:0 0 12px #23c879; }
    img { display:block; width:100%; max-height:calc(100vh - 180px); object-fit:contain;
          background:#050708; border:1px solid #52636d; }
    footer { margin-top:12px; color:#8fa3ac; font-size:13px; }
  </style>
</head>
<body>
  <header><div class="mark"></div><h1>摄像头实时画面</h1></header>
  <main><section class="panel">
    <div class="status"><span class="dot"></span><span>LIVE · 视觉相机 192.168.3.253</span></div>
    <img id="stream" src="/stream.mjpg" alt="摄像头实时画面">
    <footer>画面同步叠加最近一次 YOLO 检测框、类别、组号和置信度；若画面中断会自动重连。</footer>
  </section></main>
  <script>
    const img=document.getElementById('stream');
    img.onerror=()=>setTimeout(()=>{img.src='/stream.mjpg?t='+Date.now()},1000);
  </script>
</body></html>""".encode("utf-8")


class CameraLiveView:
    """持续抓取最新帧并通过 MJPEG 提供给浏览器。"""

    def __init__(self, camera, camera_lock, logger: logging.Logger,
                 host: str = "0.0.0.0", port: int = 8080,
                 fps: float = 5.0, jpeg_quality: int = 80):
        self.camera = camera
        self.camera_lock = camera_lock
        self.logger = logger
        self.host = host
        self.port = int(port)
        self.period = 1.0 / max(0.5, float(fps))
        self.jpeg_quality = max(30, min(95, int(jpeg_quality)))
        self._running = threading.Event()
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._sequence = 0
        self._detection: dict | None = None
        self._capture_thread: threading.Thread | None = None
        self._http_thread: threading.Thread | None = None
        self._server: ThreadingHTTPServer | None = None

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._server = ThreadingHTTPServer((self.host, self.port), self._make_handler())
        self._server.daemon_threads = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="camera-live-capture", daemon=True)
        self._http_thread = threading.Thread(
            target=self._server.serve_forever, name="camera-live-http", daemon=True)
        self._capture_thread.start()
        self._http_thread.start()
        self.logger.info("摄像头实时预览已启动: http://%s:%d/", self.host, self.port)

    def stop(self) -> None:
        self._running.clear()
        with self._condition:
            self._condition.notify_all()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        for thread in (self._capture_thread, self._http_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=1.5)

    def clear_detection(self) -> None:
        """开始一次新识别前清除旧框，避免旧结果附着到新物料。"""
        with self._condition:
            self._detection = None

    def update_detection(self, frame: np.ndarray, result,
                         group: int | None = None, valid: bool = True) -> None:
        """保存本次 YOLO 结果，并立即发布与该结果对应的标注帧。"""
        box = tuple(int(round(v)) for v in getattr(result, "box", ()))
        detection = {
            "cls": str(getattr(result, "cls_name", "none")),
            "conf": float(getattr(result, "conf", 0.0)),
            "group": group,
            "box": box,
            "valid": bool(valid),
            "time": time.time(),
        }
        with self._condition:
            self._detection = detection
        self.update_frame(frame)

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        image = frame.copy()
        with self._condition:
            detection = dict(self._detection) if self._detection else None
        if detection is None:
            cv2.putText(image, "YOLO: WAITING", (18, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 210, 255), 2,
                        cv2.LINE_AA)
            return image

        valid = detection["valid"] and detection["cls"] != "none"
        color = (40, 220, 40) if valid else (0, 170, 255)
        box = detection["box"]
        if len(box) == 4:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = box
            x1, x2 = sorted((max(0, min(w - 1, x1)), max(0, min(w - 1, x2))))
            y1, y2 = sorted((max(0, min(h - 1, y1)), max(0, min(h - 1, y2))))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

        group_text = "-" if detection["group"] is None else str(detection["group"])
        label = (f"YOLO: {detection['cls']}  "
                 f"conf={detection['conf']:.3f}  group={group_text}")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
        cv2.rectangle(image, (10, 8), (min(image.shape[1] - 1, tw + 30), th + 28),
                      (20, 28, 32), -1)
        cv2.putText(image, label, (18, th + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, color, 2, cv2.LINE_AA)
        return image

    def update_frame(self, frame: np.ndarray) -> None:
        display = self._draw_overlay(frame)
        ok, encoded = cv2.imencode(
            ".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return
        with self._condition:
            self._jpeg = encoded.tobytes()
            self._sequence += 1
            self._condition.notify_all()

    def _capture_loop(self) -> None:
        next_tick = time.monotonic()
        last_warning = 0.0
        while self._running.is_set():
            try:
                with self.camera_lock:
                    frame = self.camera.capture(timeout_ms=500)
                self.update_frame(frame)
            except Exception as exc:  # noqa: BLE001
                now = time.monotonic()
                if now - last_warning >= 5.0:
                    self.logger.warning("实时预览抓帧失败，将继续重试: %s", exc)
                    last_warning = now
            next_tick += self.period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()

    def _make_handler(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "VisionLive/1.0"

            def log_message(self, fmt, *args):
                owner.logger.debug("实时预览 HTTP: " + fmt, *args)

            def _headers(self, status=HTTPStatus.OK, content_type="text/plain",
                         length: int | None = None):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Access-Control-Allow-Origin", "*")
                if length is not None:
                    self.send_header("Content-Length", str(length))
                self.end_headers()

            def do_GET(self):  # noqa: N802
                path = self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    self._headers(content_type="text/html; charset=utf-8", length=len(_PAGE))
                    self.wfile.write(_PAGE)
                    return
                if path == "/health":
                    with owner._condition:
                        detection = dict(owner._detection) if owner._detection else None
                        if detection is not None:
                            detection.pop("time", None)
                        payload = json.dumps({
                            "status": "ok" if owner._jpeg else "waiting",
                            "sequence": owner._sequence,
                            "detection": detection,
                        }).encode("utf-8")
                    self._headers(content_type="application/json", length=len(payload))
                    self.wfile.write(payload)
                    return
                if path == "/snapshot.jpg":
                    with owner._condition:
                        jpeg = owner._jpeg
                    if jpeg is None:
                        self._headers(status=HTTPStatus.SERVICE_UNAVAILABLE)
                        return
                    self._headers(content_type="image/jpeg", length=len(jpeg))
                    self.wfile.write(jpeg)
                    return
                if path == "/stream.mjpg":
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    sequence = -1
                    try:
                        while owner._running.is_set():
                            with owner._condition:
                                owner._condition.wait_for(
                                    lambda: owner._sequence != sequence or not owner._running.is_set(),
                                    timeout=2.0)
                                jpeg = owner._jpeg
                                sequence = owner._sequence
                            if jpeg is None:
                                continue
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                self._headers(status=HTTPStatus.NOT_FOUND)

        return Handler
