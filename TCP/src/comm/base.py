"""PlcComm 抽象接口。

所有 PLC 通讯实现（OPC UA / Modbus / TCP / GDS）必须遵循此接口，
使 vision/main.py 无需关心具体协议。
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PlcResult:
    """视觉 → PLC 单次上报结果。"""
    defect_class: int   # 0=hole, 1=chip, 2=scratch, 3=stain
    confidence: float   # 0.0~1.0
    valid: bool         # True=有效结果；False=视觉失败（PLC 应停机报警）
    seq: int = 0        # 序列号（PLC 端可校验，避免丢包/重发）


class PlcComm(ABC):
    """PLC 通讯抽象类。

    所有实现必须实现以下方法，OPC UA / Modbus / TCP 协议
    都被此接口封装，main.py 不感知具体协议。
    """

    def __init__(self, cfg: dict, logger: logging.Logger | None = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger("vision.comm")

    # ---- 生命周期 ----
    @abstractmethod
    def open(self) -> None:
        """建立连接（启动期调用一次）。失败应抛 VisionError。"""

    @abstractmethod
    def close(self) -> None:
        """关闭连接（退出期调用一次）。"""

    # ---- 读写接口 ----
    @abstractmethod
    def read_trigger(self) -> bool:
        """读 PLC 触发信号（True=触发一次视觉检测）。

        内部应处理超时/重连；返回 False 表示无新触发或通讯异常。
        """

    @abstractmethod
    def write_result(self, result: PlcResult) -> None:
        """把视觉结果写入 PLC。

        协议可包含多寄存器/多节点，但接口统一为一次调用。
        """

    # ---- 可选接口 ----
    def heartbeat(self) -> bool:
        """心跳检查（默认空实现；OPC UA 客户端可用读 server_state 替代）。"""
        return True

    def __enter__(self) -> "PlcComm":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()