"""与 PLC 的 Modbus TCP/UDP 通讯测试（适配 src.comm v2.3 API）。

用法：
    python -m test.test_plc_comm --mode modbus_tcp --ip 192.168.3.99 --port 502
    python -m test.test_plc_comm --mode modbus_udp --ip 127.0.0.1 --port 502
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.comm import build_comm, PlcResult  # noqa: E402


def _setup_logger() -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )
    return logging.getLogger("test_plc_comm")


def _make_cfg(protocol: str, ip: str, port: int) -> dict:
    """构造与 src/config.yaml 结构一致的配置字典。"""
    base = {
        "host": ip,
        "port": port,
        "unit_id": 1,
        "timeout_s": 2.0,
        "registers": {
            "trigger": 100,
            "defect_class": 110,
            "confidence": 111,
            "valid": 112,
            "seq": 113,
        },
    }
    return {
        "protocol": protocol,
        protocol: base,          # e.g. modbus_tcp: {...}
    }


def test_modbus(protocol: str, ip: str, port: int, logger: logging.Logger) -> bool:
    """测试 Modbus TCP 或 UDP 读写。"""
    cfg = _make_cfg(protocol, ip, port)
    comm = build_comm(cfg, logger)

    try:
        comm.open()
        logger.info("[%s] connected to %s:%d", protocol, ip, port)

        # ---- 写结果 ----
        result = PlcResult(
            defect_class=3,      # scratch
            confidence=0.92,
            valid=True,
            seq=1,
        )
        comm.write_result(result)
        logger.info(
            "[%s] write_result → class=%d conf=%.3f valid=%s seq=%d",
            protocol, result.defect_class, result.confidence,
            result.valid, result.seq,
        )
        time.sleep(0.2)

        # ---- 读触发（若连的是 dummy server，需提前往 100 写 1）----
        trigger = comm.read_trigger()
        logger.info("[%s] read_trigger → %s", protocol, trigger)

        return True
    except Exception as e:
        logger.error("[%s] test failed: %s", protocol, e)
        return False
    finally:
        comm.close()
        logger.info("[%s] connection closed", protocol)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Test Modbus TCP/UDP communication with PLC"
    )
    parser.add_argument(
        "--mode",
        choices=["modbus_tcp", "modbus_udp"],
        default="modbus_tcp",
        help="通讯协议（默认 modbus_tcp）",
    )
    parser.add_argument(
        "--ip",
        default="192.168.3.99",
        help="PLC / 模拟服务器 IP",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=502,
        help="Modbus 端口（默认 502）",
    )
    args = parser.parse_args(argv)

    logger = _setup_logger()
    ok = test_modbus(args.mode, args.ip, args.port, logger)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
