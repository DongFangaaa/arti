"""test.py — 海康 MVS 工业相机取图 + 预处理 + 识别 + 全链路耗时日志。

执行流程：
  1. 枚举相机 → 创建句柄 → 打开设备 → 启动采集
  2. **启动后调一次**自动调参（基于亮度+标准差微调曝光/增益）——  一次性，不再调
  3. 循环 12 次取图 → 预处理 → YOLO 识别 → 保存（每步带时间戳和耗时）
  4. 优雅释放资源
"""
from __future__ import annotations

import ctypes as _ct
import datetime as _dt
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 海康 SDK 路径
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MV_IMPORT_PATH = os.path.join(BASE_DIR, 'MvImport')
DLL_PATH = os.path.join(BASE_DIR, 'Dll')

sys.path.insert(0, MV_IMPORT_PATH)

if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(DLL_PATH)
os.environ['PATH'] = DLL_PATH + os.pathsep + os.environ['PATH']

from MvCameraControl_class import (
    MV_CC_DEVICE_INFO_LIST, MV_FRAME_OUT,
    MV_GIGE_DEVICE, MV_USB_DEVICE,
    MvCamera,
)

# 让 src 子包可被 import
sys.path.insert(0, BASE_DIR)
from src.preprocess import PreprocessConfig, Preprocessor   # noqa: E402
from src.detector import DetectorConfig, build_detector     # noqa: E402
from src.auto_tune import TuneTarget, auto_tune             # noqa: E402

import numpy as np                                                       # noqa: E402
import cv2                                                              # noqa: E402


# ---------------------------------------------------------------------------
# 日志：默认 INFO；想看更细可改成 DEBUG
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vision.test")

CAPTURE_DIR = Path(BASE_DIR) / "data" / "captures"
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

CAPTURE_LOOP_TIMES = 12           # 自动调参后默认拍照次数（每次会话）
PER_PIECE_PROMPT = (
    "放好圆片后回车拍照（输入 0 / q 退出）："
)
INITIAL_EXPOSURE_US = 8000.0      # 与 camera_config.yaml 默认值保持一致
INITIAL_GAIN_DB = 12.0


def _stamp() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.") + \
        f"{_dt.datetime.now().microsecond // 1000:03d}"


def _fmt_ms(elapsed_ms: float) -> str:
    return f"{elapsed_ms:7.2f} ms"


def _bayer_to_bgr(bayer: np.ndarray, pixel_type: int) -> np.ndarray:
    """根据 PixelType 把 Bayer / Mono / BGR 统一转成 BGR 三通道。"""
    if pixel_type == 0x01080001:                # Mono8 → 灰度复制到 3 通道
        return cv2.cvtColor(bayer, cv2.COLOR_GRAY2BGR)
    if pixel_type == 0x01080009:                # BayerGR8
        return cv2.cvtColor(bayer, cv2.COLOR_BayerGR2BGR)
    if pixel_type == 0x0108000A:                # BayerGB8
        return cv2.cvtColor(bayer, cv2.COLOR_BayerGB2BGR)
    if pixel_type == 0x0108000B:                # BayerBG8
        return cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR)
    if pixel_type == 0x0108000E:                # BayerRG8
        return cv2.cvtColor(bayer, cv2.COLOR_BayerRG2BGR)
    return bayer


def _grab_one_bgr(cam, drop: int = 0) -> np.ndarray | None:
    """从相机抓一帧并直接转 BGR ndarray；失败返回 None。

    Parameters
    ----------
    drop : int
        先丢弃 SDK 内部缓存里的前 N 帧（不读也不释放，只为更新参数后取最新帧），
        确保拿到的是当前参数下的真实图像，而不是历史 buffer。
    """
    # 丢弃旧帧
    for _ in range(max(0, drop)):
        st_tmp = MV_FRAME_OUT()
        ret = cam.MV_CC_GetImageBuffer(st_tmp, 200)
        if ret == 0:
            cam.MV_CC_FreeImageBuffer(st_tmp)

    st = MV_FRAME_OUT()
    t0 = time.perf_counter()
    ret = cam.MV_CC_GetImageBuffer(st, 1000)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if ret != 0:
        log.error("[取图] 失败 0x%08X, 耗时=%s", ret, _fmt_ms(elapsed_ms))
        return None

    info = st.stFrameInfo
    w, h = int(info.nWidth), int(info.nHeight)
    pixel_type = int(info.enPixelType)
    if pixel_type in (0x01080009, 0x0108000A, 0x0108000B,
                      0x0108000E, 0x01080001):  # Bayer*/Mono
        byte_size = w * h
    else:
        byte_size = w * h * 3
    buf = bytes(_ct.string_at(st.pBufAddr, byte_size))
    cam.MV_CC_FreeImageBuffer(st)

    if pixel_type in (0x01080001, 0x01080009, 0x0108000A,
                      0x0108000B, 0x0108000E):
        img_raw = np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
    else:
        img_raw = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    return _bayer_to_bgr(img_raw, pixel_type)


def _draw_inference(img_bgr: np.ndarray, result) -> np.ndarray:
    """在图上绘制识别结果：
        - 所有候选框（boxes），top1 用实线+粗+黄，其它用细+灰
        - 顶部信息条：类别、置信度、top-k
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]

    # 字体大小根据分辨率自适应
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs_main = max(0.5, h / 480.0)
    fs_sub = max(0.35, h / 720.0)
    thickness_main = 2
    thickness_sub = 1

    # ---- 1) 画 bounding box（detect 模型专用）----
    boxes = getattr(result, "boxes", None) or []
    for i, item in enumerate(boxes):
        # item 形如 (name, cls_id, conf, (x1,y1,x2,y2))
        try:
            name, _cid, c, xyxy = item
            x1, y1, x2, y2 = [int(v) for v in xyxy]
        except Exception:
            continue
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
        if i == 0:
            color = (0, 255, 255); thickness_box = 3            # top1 亮黄粗
        else:
            color = (180, 180, 180); thickness_box = 1          # 其它候选 灰细
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness_box, cv2.LINE_AA)
        # 框上标签
        label = f"{name} {c:.2f}"
        (tw, th_lbl), _ = cv2.getTextSize(label, font, fs_sub, 1)
        ty = max(y1 - 4, th_lbl + 4)
        cv2.rectangle(out, (x1, ty - th_lbl - 2), (x1 + tw + 6, ty + 2),
                      color, -1, cv2.LINE_AA)
        cv2.putText(out, label, (x1 + 3, ty - 2),
                    font, fs_sub, (0, 0, 0), 1, cv2.LINE_AA)

    # ---- 2) 顶部信息条 ----
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, int(h * 0.22)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    # 主标题
    main = f"class={result.cls_name}  conf={result.conf:.4f}"
    cv2.putText(out, main, (10, int(h * 0.07)),
                font, fs_main, (0, 255, 255), thickness_main, cv2.LINE_AA)
    sub = f"id={result.cls_id}  top-{len(result.topk)} below"
    cv2.putText(out, sub, (10, int(h * 0.13)),
                font, fs_sub, (220, 220, 220), thickness_sub, cv2.LINE_AA)

    # top-k 列表
    if result.topk:
        line_h = int(h * 0.04)
        base_y = int(h * 0.18)
        for idx, (name, cid, conf) in enumerate(result.topk):
            text = f"{idx+1}. {name:8s} ({cid}) {conf:.3f}"
            cv2.putText(out, text, (10, base_y + idx * line_h),
                        font, fs_sub, (180, 255, 180), thickness_sub, cv2.LINE_AA)
    return out


def _apply_initial_camera_params(cam) -> None:
    """先把 CameraConfig 里的初值写到相机，给 auto_tune 一个起点。

    同时启用 **数字降噪 / 坏点校正**，用以消除传感器本身或暗电流造成的亮点。
    现场如果不需要，可在 camera_config.yaml 里关掉。
    """
    from MvImport.CameraParams_header import MVCC_FLOATVALUE  # noqa: WPS433
    st = MVCC_FLOATVALUE()

    # 1) 曝光 / 增益 / 帧率
    cam.MV_CC_SetFloatValue("ExposureTime", INITIAL_EXPOSURE_US)
    cam.MV_CC_SetFloatValue("Gain", INITIAL_GAIN_DB)
    cam.MV_CC_SetFloatValue("AcquisitionFrameRate", 10.0)
    cam.MV_CC_SetEnumValue("TriggerMode", 0)            # 连续采集

    # 2) 数字降噪（消除亮点 / 暗电流噪声）
    #    海康通用参数：NoiseReductionMode (Enum: 0=off, 1=on)
    cam.MV_CC_SetEnumValue("NoiseReductionMode", 1)

    # 3) 坏点校正（部分型号支持）
    cam.MV_CC_SetEnumValue("DefectCorrectMode", 1)

    # 4) 锐化（适度）
    cam.MV_CC_SetBoolValue("SharpnessEnabled", True)

    # 验证写入
    cam.MV_CC_GetFloatValue("ExposureTime", st)
    cam.MV_CC_GetFloatValue("Gain", st)


def main() -> int:
    t_pipeline_start = time.perf_counter()
    log.info("===== 开始取图-预处理-识别测试 =====")

    # -------------------------------------------------------------- SDK init
    ret = MvCamera.MV_CC_Initialize()
    if ret != 0:
        log.error("MV_CC_Initialize 失败: 0x%08X", ret)
        return 1

    # -------------------------------------------------------------- enumerate
    device_list = MV_CC_DEVICE_INFO_LIST()
    t0 = time.perf_counter()
    ret = MvCamera.MV_CC_EnumDevices(
        MV_GIGE_DEVICE | MV_USB_DEVICE, device_list
    )
    log.info("[枚举] 耗时=%s, 返回码=%d, 设备数=%d",
             _fmt_ms((time.perf_counter() - t0) * 1000),
             ret, device_list.nDeviceNum)
    if ret != 0 or device_list.nDeviceNum == 0:
        log.error("未找到相机")
        return 1

    # -------------------------------------------------------------- create / open / grab
    cam = MvCamera()
    t0 = time.perf_counter()
    ret = cam.MV_CC_CreateHandle(device_list.pDeviceInfo[0])
    log.info("[创建句柄] 耗时=%s, 返回码=%d",
             _fmt_ms((time.perf_counter() - t0) * 1000), ret)
    if ret != 0:
        log.error("创建句柄失败: 0x%08X", ret)
        return 1

    t0 = time.perf_counter()
    ret = cam.MV_CC_OpenDevice()
    log.info("[打开设备] 耗时=%s, 返回码=%d",
             _fmt_ms((time.perf_counter() - t0) * 1000), ret)
    if ret != 0:
        cam.MV_CC_DestroyHandle()
        return 1

    t0 = time.perf_counter()
    ret = cam.MV_CC_StartGrabbing()
    log.info("[开始采集] 耗时=%s, 返回码=%d",
             _fmt_ms((time.perf_counter() - t0) * 1000), ret)
    if ret != 0:
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        return 1

    # -------------------------------------------------------------- 应用初值参数
    _apply_initial_camera_params(cam)

    # -------------------------------------------------------------- 自动调参（仅一次）
    log.info("[自动调参] 启动时一次性，目标: 亮度[110,145] / 标准差[40,70]")
    tune_target = TuneTarget(
        mean_min=110, mean_max=145,
        std_min=40, std_max=70,
        exposure_min=100.0,
        exposure_max=60000.0,
        gain_min=0.0,
        gain_max=30.0,
        max_rounds=6, stabilize_ms=80,
    )
    t0 = time.perf_counter()
    tune_res = auto_tune(
        handle=cam,
        initial_exposure_us=INITIAL_EXPOSURE_US,
        initial_gain_db=INITIAL_GAIN_DB,
        grab_one_frame=lambda: _grab_one_bgr(cam, drop=2),
        target=tune_target,
        logger=log,
    )
    log.info("[自动调参] 总耗时=%s, 结果=%s",
             _fmt_ms((time.perf_counter() - t0) * 1000), tune_res)

    # -------------------------------------------------------------- 创建预处理器 / 检测器
    pp_cfg = PreprocessConfig(
        enable=True,
        mode="legacy",                 # 与 batch 脚本保持一致：仅去噪
        denoise="fastNlMeans",
        denoise_strength=5,
        clahe=False,
        sharpen=False,
        gamma=1.0,
    )
    preprocessor = Preprocessor(pp_cfg, logger=log)

    det_cfg = DetectorConfig(
        model_path=str(Path(BASE_DIR) / "models" / "best.pt"),
        device="cpu", imgsz=640, conf=0.25, warmup=True,
    )
    detector = build_detector(det_cfg, logger=log)
    detector.warmup()  # 预热一次，避开加载耗时

    # -------------------------------------------------------------- 拍照循环：每按一次回车拍一张、识别一次
    shot_index = 0
    all_grab_ms, all_pp_ms, all_det_ms = [], [], []
    all_cls_counter: dict[str, int] = {}

    while True:
        raw = input(PER_PIECE_PROMPT).strip()
        if raw == "0" or raw.lower() in ("q", "quit", "exit"):
            log.info("用户请求退出（已拍 %d 张）", shot_index)
            break

        shot_index += 1
        log.info("---- 第 %d 张 ----", shot_index)

        # 取图
        grab_started_at = _stamp()
        t_grab = time.perf_counter()
        img_bgr = _grab_one_bgr(cam)
        grab_ms = (time.perf_counter() - t_grab) * 1000
        if img_bgr is None:
            log.warning("取图失败，跳过本张")
            shot_index -= 1
            continue
        grab_finished_at = _stamp()
        log.info("[取图] 开始=%s, 结束=%s, 耗时=%s, shape=%s",
                 grab_started_at, grab_finished_at,
                 _fmt_ms(grab_ms), img_bgr.shape)

        # 预处理
        t0 = time.perf_counter()
        img_proc = preprocessor.run(img_bgr)
        pp_ms = (time.perf_counter() - t0) * 1000
        log.info("[预处理] 耗时=%s, mode=%s",
                 _fmt_ms(pp_ms), pp_cfg.mode)

        # 识别
        detect_started_at = _stamp()
        t0 = time.perf_counter()
        result = detector.detect(img_proc)
        det_ms = (time.perf_counter() - t0) * 1000
        detect_finished_at = _stamp()
        log.info("[识别] 开始=%s, 结束=%s, 耗时=%s, class=%s id=%d conf=%.4f",
                 detect_started_at, detect_finished_at, _fmt_ms(det_ms),
                 result.cls_name, result.cls_id, result.conf)

        # 保存：原图 + 预处理图 + 识别标注图
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base = f"{shot_index:03d}_{ts}"
        raw_path = CAPTURE_DIR / f"{base}_raw.png"
        proc_path = CAPTURE_DIR / f"{base}_preproc_{result.cls_name}.png"
        cv2.imwrite(str(raw_path), img_bgr)
        cv2.imwrite(str(proc_path), img_proc)

        vis = _draw_inference(img_proc, result)
        vis_path = CAPTURE_DIR / f"{base}_infer_{result.cls_name}.png"
        cv2.imwrite(str(vis_path), vis)

        # 统计
        all_grab_ms.append(grab_ms)
        all_pp_ms.append(pp_ms)
        all_det_ms.append(det_ms)
        all_cls_counter[result.cls_name] = all_cls_counter.get(result.cls_name, 0) + 1

    # -------------------------------------------------------------- 汇总
    total_ms = (time.perf_counter() - t_pipeline_start) * 1000
    n = len(all_grab_ms)

    log.info("===== 测试完成 =====")
    log.info("调参: in_target=%s, 最终 exp=%.1fμs gain=%.2fdB, mean=%.1f std=%.1f",
             tune_res.in_target, tune_res.exposure_us, tune_res.gain_db,
             tune_res.mean, tune_res.stddev)
    log.info("累计拍照 %d 张", n)

    if n > 0:
        log.info("拍照 %d 次统计:", n)
        log.info("  取图   avg=%s min=%s max=%s",
                 _fmt_ms(sum(all_grab_ms) / n),
                 _fmt_ms(min(all_grab_ms)),
                 _fmt_ms(max(all_grab_ms)))
        log.info("  预处理 avg=%s min=%s max=%s",
                 _fmt_ms(sum(all_pp_ms) / n),
                 _fmt_ms(min(all_pp_ms)),
                 _fmt_ms(max(all_pp_ms)))
        log.info("  推理   avg=%s min=%s max=%s",
                 _fmt_ms(sum(all_det_ms) / n),
                 _fmt_ms(min(all_det_ms)),
                 _fmt_ms(max(all_det_ms)))
        log.info("  分类分布: %s", all_cls_counter)

    log.info("管线总耗时: %s", _fmt_ms(total_ms))

    # -------------------------------------------------------------- shutdown
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    return 0


if __name__ == '__main__':
    sys.exit(main())