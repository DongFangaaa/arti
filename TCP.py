"""PLC 与视觉程序之间的 TCP 收发接口。

报文格式固定为：S<数字>E\n

PLC -> Python：
    S0E\n  停止视觉
    S1E\n  启动视觉

Python -> PLC：
    S1E\n ~ S4E\n  发送识别到的目标组号
"""

from __future__ import annotations

import socket
import threading


class TCP:
    """单客户端 TCP 服务端，只负责建立连接、接收和发送报文。"""

    def __init__(self, host: str = "0.0.0.0", port: int = 2000) -> None:
        self.host = host
        self.port = port
        self._server: socket.socket | None = None
        self._client: socket.socket | None = None
        self._receive_buffer = bytearray()
        self._send_lock = threading.Lock()

    def start(self) -> tuple[str, int]:
        """启动监听并等待 PLC 连接；连接成功后返回 PLC 的地址。"""
        if self._server is not None:
            raise RuntimeError("TCP 服务已经启动")

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        self._server = server

        client, address = server.accept()
        self._client = client
        self._receive_buffer.clear()
        return address

    def receive(self, timeout: float | None = None) -> int | None:
        """接收一帧并返回其中的数字；超时返回 None。

        该接口会自动处理 TCP 拆包、粘包和帧前无效字节。
        """
        client = self._require_client()
        client.settimeout(timeout)

        while True:
            value = self._extract_frame()
            if value is not None:
                return value

            try:
                data = client.recv(1024)
            except socket.timeout:
                return None

            if not data:
                raise ConnectionError("PLC 已断开 TCP 连接")

            self._receive_buffer.extend(data)

    def send(self, value: int) -> None:
        """向 PLC 发送一帧，例如 send(2) 会发送 b'S2E\\n'。"""
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 9:
            raise ValueError("发送数据必须是 0 到 9 的整数")

        frame = f"S{value}E\n".encode("ascii")
        client = self._require_client()
        with self._send_lock:
            client.sendall(frame)

    def close(self) -> None:
        """关闭 PLC 连接和监听套接字。"""
        if self._client is not None:
            try:
                self._client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._client.close()
            self._client = None

        if self._server is not None:
            self._server.close()
            self._server = None

        self._receive_buffer.clear()

    def _require_client(self) -> socket.socket:
        if self._client is None:
            raise RuntimeError("尚未调用 start()，或 PLC 尚未连接")
        return self._client

    def _extract_frame(self) -> int | None:
        while True:
            start = self._receive_buffer.find(b"S")
            if start < 0:
                self._receive_buffer.clear()
                return None

            if start > 0:
                del self._receive_buffer[:start]

            if len(self._receive_buffer) < 4:
                return None

            frame = bytes(self._receive_buffer[:4])
            if frame[2:] == b"E\n" and 48 <= frame[1] <= 57:
                del self._receive_buffer[:4]
                return frame[1] - 48

            # 当前 S 不是有效帧头，丢弃后继续寻找下一帧。
            del self._receive_buffer[0]


__all__ = ["TCP"]
