"""相机自动调参模块。

目标：根据当前光照条件，自动调整 **曝光时间** 与 **增益**，使抓到的画面
      平均亮度 + 标准差落在期望区间内。其它参数（帧率/触发模式等）由
      CameraConfig 预先确定，本模块不修改。

判断指标：
    mean  : 像素平均亮度（BGR → 灰度）         目标 110~145
    stddev: 像素标准差                         目标 40~70
    SNR   : stddev / (max(stddev, 1) / gain)   防止一味拉高 gain 导致噪点爆炸

算法：
    1. 先以 cfg 中的曝光/增益为初值抓一帧
    2. 若 mean 已落在目标区间 → 直接结束；否则：
       - mean 偏低 → 拉高曝光/增益
       - mean 偏高 → 拉低曝光/增益
       - stddev 过低 → 微调 gamma/对比度（这里简化为微调增益）
       - stddev 过高（>80）→ 降增益
    3. 最多 N_ROUNDS 轮；任何一轮同时落入区间即提前停止

依赖：海康 MVS SDK（c_void_p + c_float 交互）
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 调参目标 / 上下限
# ---------------------------------------------------------------------------
@dataclass
class TuneTarget:
    mean_min: float = 110.0
    mean_max: float = 145.0
    std_min: float = 40.0
    std_max: float = 70.0
    # 曝光/增益边界（按海康常规相机范围；具体型号不同）
    exposure_min: float = 100.0       # μs
    exposure_max: float = 30000.0     # μs
    gain_min: float = 0.0
    gain_max: float = 20.0
    max_rounds: int = 5
    stabilize_ms: int = 60            # 每轮参数写入后等待相机稳定


@dataclass
class TuneResult:
    """调参最终结果。"""
    rounds: int
    mean: float
    stddev: float
    exposure_us: float
    gain_db: float
    in_target: bool


def _image_stats(img_bgr: np.ndarray) -> tuple[float, float]:
    """返回 (mean, stddev) —— 灰度图上算。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) \
        if img_bgr.ndim == 3 else img_bgr
    return float(gray.mean()), float(gray.std())


def _get_float(handle, key: str, struct_factory) -> float | None:
    try:
        st = struct_factory()
        ret = handle.MV_CC_GetFloatValue(key, st)
        if ret == 0:
            return float(st.fCurValue)
    except Exception as e:  # noqa: BLE001
        logging.getLogger("vision.auto_tune").debug("read %s failed: %s", key, e)
    return None


def _set_float(handle, key: str, value: float) -> bool:
    try:
        ret = handle.MV_CC_SetFloatValue(key, float(value))
        return ret == 0
    except Exception as e:  # noqa: BLE001
        logging.getLogger("vision.auto_tune").debug("set %s failed: %s", key, e)
        return False


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def auto_tune(
    handle,
    initial_exposure_us: float,
    initial_gain_db: float,
    grab_one_frame,
    target: TuneTarget | None = None,
    logger: logging.Logger | None = None,
) -> TuneResult:
    """迭代调整曝光 + 增益。

    Parameters
    ----------
    handle : MvCamera 实例（已 open + StartGrabbing）
    initial_exposure_us / initial_gain_db : 调参起点（来自 CameraConfig）
    grab_one_frame : () -> np.ndarray BGR；抓一帧返回
    target : TuneTarget
    logger : 日志

    Returns
    -------
    TuneResult
    """
    log = logger or logging.getLogger("vision.auto_tune")
    tgt = target or TuneTarget()

    from MvImport.CameraParams_header import MVCC_FLOATVALUE  # noqa: WPS433

    exposure = _clamp(initial_exposure_us, tgt.exposure_min, tgt.exposure_max)
    gain = _clamp(initial_gain_db, tgt.gain_min, tgt.gain_max)

    cur_mean, cur_std = 0.0, 0.0
    in_target = False

    log.info("auto_tune start, exp=%.1fμs gain=%.2fdB, target mean=[%.0f,%.0f] std=[%.0f,%.0f]",
             exposure, gain, tgt.mean_min, tgt.mean_max, tgt.std_min, tgt.std_max)

    rnd = 0
    for rnd in range(1, tgt.max_rounds + 1):
        # 写入当前参数
        _set_float(handle, "ExposureTime", exposure)
        _set_float(handle, "Gain", gain)
        time.sleep(tgt.stabilize_ms / 1000.0)

        # 取一帧（grab_one_frame 内部会自动 drop 旧帧）
        img = grab_one_frame()
        if img is None:
            log.warning("auto_tune round %d: grab failed", rnd)
            continue

        cur_mean, cur_std = _image_stats(img)
        log.info("auto_tune round %d: exp=%.1fμs gain=%.2fdB -> mean=%.1f std=%.1f",
                 rnd, exposure, gain, cur_mean, cur_std)

        # 是否达标
        if (tgt.mean_min <= cur_mean <= tgt.mean_max and
                tgt.std_min <= cur_std <= tgt.std_max):
            log.info("auto_tune hit target at round %d", rnd)
            in_target = True
            break

        # ---------- 调曝光（主控亮度） ----------
        if cur_mean < tgt.mean_min:
            # 偏暗：拉曝光；若曝光已到顶，再拉 gain
            new_exp = exposure * (tgt.mean_min / max(cur_mean, 1e-3))
            new_exp = _clamp(new_exp, tgt.exposure_min, tgt.exposure_max)
            if new_exp >= tgt.exposure_max * 0.95:
                gain = _clamp(gain + 1.5, tgt.gain_min, tgt.gain_max)
            exposure = new_exp
        elif cur_mean > tgt.mean_max:
            # 偏亮：压曝光
            new_exp = exposure * (tgt.mean_max / max(cur_mean, 1e-3))
            exposure = _clamp(new_exp, tgt.exposure_min, tgt.exposure_max)

        # ---------- 调增益（控制噪声/对比度） ----------
        if cur_std > tgt.std_max + 15:                 # 标准差太高 → 噪声多
            gain = _clamp(gain - 0.5, tgt.gain_min, tgt.gain_max)
        elif cur_std < tgt.std_min - 15:               # 标准差太低 → 偏色/过曝
            gain = _clamp(gain + 0.5, tgt.gain_min, tgt.gain_max)

        exposure = _clamp(exposure, tgt.exposure_min, tgt.exposure_max)
        gain = _clamp(gain, tgt.gain_min, tgt.gain_max)

    # 收敛后再写一次，使最终值稳定生效
    _set_float(handle, "ExposureTime", exposure)
    _set_float(handle, "Gain", gain)
    time.sleep(tgt.stabilize_ms / 1000.0)

    # 物理极限识别：如果曝光/增益都已贴上限且仍未达标，
    # 标注 converged_at_limit=True（不是 in_target，但表示已尽力）。
    at_limit = (exposure >= tgt.exposure_max * 0.98 or
                gain >= tgt.gain_max * 0.98)
    if not in_target and at_limit:
        log.warning(
            "auto_tune 达到曝光/增益上限仍未完全进入目标区间，"
            "建议检查镜头前光照或降低 mean_min (当前=%.0f)", tgt.mean_min)

    result = TuneResult(
        rounds=(rnd if in_target or at_limit else tgt.max_rounds),
        mean=cur_mean,
        stddev=cur_std,
        exposure_us=exposure,
        gain_db=gain,
        in_target=in_target,
    )
    log.info("auto_tune done: %s, at_limit=%s", result, at_limit)
    return result