"""本地 Modbus 模拟服务器（调试用，pymodbus 3.x）。

监听 127.0.0.1:502，提供 Modbus TCP Server + 200 个保持寄存器。

用法：
    python -m vision.dummy_modbus_server

配合 vision/main.py 在 PC 上做端到端调试：
    - PLC 程序视角：往寄存器 100 写 1 触发视觉
    - 视觉程序读到 1 → 抓图/推理 → 写 110~113
    - 调试终端：实时看寄存器值
"""
from __future__ import annotations

import logging
import signal
import sys
import time
from threading import Thread

# pymodbus 3.x
from pymodbus.datastore import ModbusSimulatorContext
from pymodbus.server import StartTcpServer


def _setup_logger() -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] dummy_modbus: %(message)s",
        level=logging.INFO,
    )
    return logging.getLogger("dummy_modbus")


def _watch_registers(context: ModbusSimulatorContext,
                     logger: logging.Logger) -> None:
    """每秒打印一次关键寄存器的值（调试用）。"""
    try:
        # simulator ctx 默认是单设备，addr=1
        trigger = context.getValues(0x03, 100, 1)[0]
        defect = context.getValues(0x03, 110, 1)[0]
        conf = context.getValues(0x03, 111, 1)[0]
        valid = context.getValues(0x03, 112, 1)[0]
        seq = context.getValues(0x03, 113, 1)[0]
        logger.info(
            "[reg] trigger=%d  defect=%d  conf=%d  valid=%d  seq=%d",
            trigger, defect, conf, valid, seq,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("watchdog error: %s", e)


def main() -> int:
    logger = _setup_logger()
    logger.info("starting dummy Modbus TCP server on 127.0.0.1:502")

    # ModbusSimulatorContext 已内建200+ 寄存器
    context = ModbusSimulatorContext()

    stop_flag = {"stop": False}

    def watchdog() -> None:
        while not stop_flag["stop"]:
            _watch_registers(context, logger)
            time.sleep(1.0)

    t = Thread(target=watchdog, daemon=True)
    t.start()

    def on_signal(signum, frame):  # noqa: ARG001
        logger.info("signal %d received, shutting down", signum)
        stop_flag["stop"] = True
        sys.exit(0)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        StartTcpServer(context=context, address=("127.0.0.1", 502))
    except OSError as e:
        logger.error("bind 127.0.0.1:502 failed: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())