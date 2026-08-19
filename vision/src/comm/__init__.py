"""PLC 通讯模块入口。

按 vision/config.yaml 中的 plc.protocol 字段选择实现：
    - "modbus_tcp" → ModbusTcpComm（主路径，赛题要求 TCP 协议）
    - "modbus_udp" → ModbusUdpComm（可选 fallback）

新增实现只需在 _REGISTRY 注册即可，main.py 无需改动。
"""
from __future__ import annotations

import logging

from .base import PlcComm, PlcResult
from .modbus_comm import ModbusTcpComm, ModbusUdpComm

# 实现注册表（key = 协议名小写，value = 类）
_REGISTRY: dict[str, type[PlcComm]] = {}


def build_comm(cfg: dict, logger: logging.Logger | None = None) -> PlcComm:
    """按 cfg["protocol"] 选择具体实现。

    cfg 形如：
        plc:
          protocol: "modbus_tcp"
          modbus_tcp:
            host: 127.0.0.1
            port: 502
            unit_id: 1
            registers: {trigger: 100, defect_class: 110, ...}
    """
    if not _REGISTRY:
        _load_builtins()

    protocol = cfg.get("protocol", "modbus_tcp").lower()
    impl_cfg = cfg.get(protocol, cfg.get("modbus", {}))

    cls = _REGISTRY.get(protocol)
    if cls is None:
        raise ValueError(
            f"Unknown PLC protocol: {protocol}. "
            f"Available: {list(_REGISTRY)}")
    return cls(impl_cfg, logger)


def _load_builtins() -> None:
    """加载内置实现（延迟加载避免不必要的 import）。"""
    from .modbus_comm import ModbusTcpComm, ModbusUdpComm
    _REGISTRY["modbus_tcp"] = ModbusTcpComm
    _REGISTRY["modbus_udp"] = ModbusUdpComm


__all__ = ["PlcComm", "PlcResult", "build_comm"]