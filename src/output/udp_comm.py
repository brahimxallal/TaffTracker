from __future__ import annotations

import logging
import os
import socket
import subprocess
import threading
from collections.abc import Callable

LOGGER = logging.getLogger("udp_comm")


def _default_reachability_probe(host: str, timeout_ms: int) -> bool:
    if not host.strip():
        return False
    if os.name == "nt":
        command = ["ping", "-n", "1", "-w", str(timeout_ms), host]
    else:
        timeout_s = max(1, int((timeout_ms + 999) / 1000))
        command = ["ping", "-c", "1", "-W", str(timeout_s), host]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=max(1.0, timeout_ms / 1000.0 + 0.5),
        )
    except Exception:
        return False
    return completed.returncode == 0


class UDPComm:
    """UDP transport with health tracking.

    UDP is connectionless, so liveness is tracked by periodically probing
    reachability to the configured receiver address.
    """

    _PROBE_INTERVAL_S = 1.0
    _PROBE_TIMEOUT_MS = 200

    def __init__(
        self,
        host: str,
        port: int,
        redundancy: int = 2,
        *,
        reachability_probe: Callable[[str, int], bool] | None = None,
        start_monitor: bool = True,
    ) -> None:
        self._address = (host, port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._redundancy = redundancy
        self._packets_sent: int = 0
        self._packets_failed: int = 0
        self._connected = False
        self._reachability_probe = reachability_probe or _default_reachability_probe
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        if start_monitor:
            self._monitor_thread = threading.Thread(
                target=self._monitor_reachability,
                name="UDPReachability",
                daemon=True,
            )
            self._monitor_thread.start()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def packets_sent(self) -> int:
        return self._packets_sent

    @property
    def packets_failed(self) -> int:
        return self._packets_failed

    def send(self, packet: bytes) -> None:
        try:
            for _ in range(self._redundancy):
                self._socket.sendto(packet, self._address)
            self._packets_sent += 1
        except OSError:
            self._packets_failed += 1

    def reconnect(self) -> bool:
        return self._connected

    def health_check(self) -> dict[str, object]:
        return {
            "connected": self._connected,
            "address": f"{self._address[0]}:{self._address[1]}",
            "packets_sent": self._packets_sent,
            "packets_failed": self._packets_failed,
        }

    def close(self) -> None:
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=self._PROBE_INTERVAL_S + 0.5)
        self._socket.close()

    def _monitor_reachability(self) -> None:
        self._connected = self._probe_reachability()
        while not self._monitor_stop.wait(self._PROBE_INTERVAL_S):
            self._connected = self._probe_reachability()

    def _probe_reachability(self) -> bool:
        try:
            return bool(self._reachability_probe(self._address[0], self._PROBE_TIMEOUT_MS))
        except Exception:
            return False
