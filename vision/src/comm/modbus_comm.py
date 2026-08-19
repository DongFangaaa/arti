"""Modbus TCP 通讯实现（主路径，赛题要求 TCP 协议）。

通过 pymodbus 客户端连接 PLC 端的 Modbus TCP Server（默认端口 502）。
PLCnext Engineer 中可通过 Arp.Plc.Eclr 程序 + 自定义 Modbus Server 实现。

寄存器映射（默认）：
  100  trigger        (BOOL as INT 0/1, PLC→Vision)
  110  defect_class   (INT, Vision→PLC, +1 偏移：1=hole, 2=notch, 3=scratch, 4=stain, 0=未识别)
  111  confidence     (INT×1000, Vision→PLC, 0~1000)
  112  valid          (BOOL as INT 0/1, Vision→PLC)
  113  seq            (INT, Vision→PLC, 序列号，PLC 端可校验)
"""
from __future__ import annotations

import logging
import socket
from typing import Any

from .base import PlcComm, PlcResult

# pymodbus 3.6.x: kw = {"slave": id}
# pymodbus 3.14+: kw = {"device_id": id}
try:  # noqa: E402
    from pymodbus.client import ModbusTcpClient
    import inspect
    _sig = inspect.signature(ModbusTcpClient.read_holding_registers)
    _HAS_DEVICE_ID = "device_id" in _sig.parameters
except Exception:  # noqa: BLE001
    _HAS_DEVICE_ID = False


class _BaseModbus(PlcComm):
    """Modbus TCP/UDP 公共逻辑。"""

    def __init__(self, cfg: dict, logger: logging.Logger | None = None):
        super().__init__(cfg, logger)
        self.host: str = cfg.get("host", "127.0.0.1")
        self.port: int = cfg.get("port", 502)
        self.unit_id: int = cfg.get("unit_id", 1)
        self.timeout: float = cfg.get("timeout_s", 2.0)
        # 寄存器地址
        reg = cfg.get("registers", {})
        self.reg_trigger: int = reg.get("trigger", 100)
        self.reg_defect: int = reg.get("defect_class", 110)
        self.reg_confidence: int = reg.get("confidence", 111)
        self.reg_valid: int = reg.get("valid", 112)
        self.reg_seq: int = reg.get("seq", 113)
        self._client: Any = None
        self._connected = False

    def _log_addr(self) -> None:
        self.logger.info(
            "Modbus configured host=%s port=%d unit=%d "
            "regs={trigger:%d,defect:%d,conf:%d,valid:%d,seq:%d}",
            self.host, self.port, self.unit_id,
            self.reg_trigger, self.reg_defect,
            self.reg_confidence, self.reg_valid, self.reg_seq,
        )

    def _do_connect(self) -> None:
        """子类实现：实际建立连接。"""
        raise NotImplementedError

    def open(self) -> None:
        self._log_addr()
        try:
            self._do_connect()
        except Exception as e:
            raise RuntimeError(
                f"Modbus connect failed: {self.host}:{self.port} — {e}"
            ) from e
        self._connected = True
        self.logger.info("Modbus connected to %s:%d", self.host, self.port)

    def read_trigger(self) -> bool:
        if not self._connected or self._client is None:
            return False
        try:
            # pymodbus 3.6.6: slave; pymodbus 3.14+: device_id
            kwargs = {"device_id": self.unit_id} if _HAS_DEVICE_ID else {"slave": self.unit_id}
            rr = self._client.read_holding_registers(
                self.reg_trigger, count=1, **kwargs)
            if rr.isError():
                return False
            return bool(rr.registers[0])
        except Exception as e:
            self.logger.error("Modbus read_trigger error: %s", e)
            # 一次失败不立即断开，标记 unhealthy 让 heartbeat 触发重连
            return False

    def write_result(self, result: PlcResult) -> None:
        if not self._connected or self._client is None:
            return
        try:
            conf_int = int(round(result.confidence * 1000))  # 0~1000
            values = [
                int(result.defect_class) & 0xFFFF,
                conf_int & 0xFFFF,
                int(bool(result.valid)) & 0xFFFF,
                int(result.seq) & 0xFFFF,
            ]
            kwargs = {"device_id": self.unit_id} if _HAS_DEVICE_ID else {"slave": self.unit_id}
            self._client.write_registers(
                self.reg_defect,
                values,
                **kwargs,
            )
            self.logger.info(
                "Modbus write class=%d conf=%d valid=%s seq=%d",
                result.defect_class, conf_int, result.valid, result.seq,
            )
        except Exception as e:
            self.logger.error("Modbus write_result error: %s", e)


class ModbusTcpComm(_BaseModbus):
    """Modbus TCP 客户端。"""

    def _do_connect(self) -> None:
        try:
            from pymodbus.client import ModbusTcpClient
        except ImportError as e:
            raise RuntimeError(
                "Modbus support requires the `pymodbus` package") from e
        self._client = ModbusTcpClient(
            self.host, port=self.port, timeout=self.timeout)
        if not self._client.connect():
            raise RuntimeError("connect returned False")

    def close(self) -> None:
        if self._client and self._connected:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._connected = False
            self.logger.info("Modbus TCP closed")


class ModbusUdpComm(_BaseModbus):
    """Modbus UDP 客户端（可选 fallback）。"""

    def _do_connect(self) -> None:
        try:
            from pymodbus.client import ModbusUdpClient
        except ImportError as e:
            raise RuntimeError(
                "Modbus UDP support requires the `pymodbus` package") from e
        self._client = ModbusUdpClient(
            self.host, port=self.port, timeout=self.timeout)
        # UDP 无显式 connect

    def close(self) -> None:
        if self._client and self._connected:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._connected = False
            self.logger.info("Modbus UDP closed")