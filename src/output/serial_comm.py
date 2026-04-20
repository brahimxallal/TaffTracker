from __future__ import annotations

import logging
import time

import serial

LOGGER = logging.getLogger("serial_comm")


class SerialComm:
    """Resilient serial transport with automatic reconnection.

    On send failure the connection is marked disconnected and subsequent
    calls to ``send()`` silently skip until ``reconnect()`` succeeds.
    The caller (OutputProcess) can periodically call ``reconnect()`` to
    re-establish the link with exponential backoff.
    """

    _BACKOFF_INITIAL: float = 0.5
    _BACKOFF_MAX: float = 5.0
    _BACKOFF_FACTOR: float = 2.0

    def __init__(self, port: str, baud_rate: int = 921_600) -> None:
        self._port = port
        self._baud_rate = baud_rate
        self._serial: serial.Serial | None = None
        self._connected = False
        self._packets_sent: int = 0
        self._packets_failed: int = 0
        self._backoff: float = self._BACKOFF_INITIAL
        self._last_reconnect_attempt: float = 0.0
        self._connect()

    # --- Public API ---

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
        """Fire-and-forget send. Silently skips when disconnected."""
        if not self._connected or self._serial is None:
            self._packets_failed += 1
            return
        try:
            # write_timeout=0 ensures this doesn't block forever if hardware hangs
            self._serial.write(packet)
            self._packets_sent += 1
        except serial.SerialTimeoutException:
            # write_timeout=0 makes this expected for slow receivers — just skip
            pass
        except (serial.SerialException, OSError):
            self._packets_failed += 1
            self._mark_disconnected()

    def reconnect(self) -> bool:
        """Attempt reconnection with exponential backoff.

        Returns True if now connected, False if backoff timer hasn't
        elapsed yet or the reconnection attempt failed.
        """
        if self._connected:
            return True
        now = time.monotonic()
        if now - self._last_reconnect_attempt < self._backoff:
            return False
        self._last_reconnect_attempt = now
        self._connect()
        if self._connected:
            self._backoff = self._BACKOFF_INITIAL
            LOGGER.info("Reconnected to %s", self._port)
        else:
            self._backoff = min(self._backoff * self._BACKOFF_FACTOR, self._BACKOFF_MAX)
        return self._connected

    def health_check(self) -> dict[str, object]:
        return {
            "connected": self._connected,
            "port": self._port,
            "packets_sent": self._packets_sent,
            "packets_failed": self._packets_failed,
        }

    def close(self) -> None:
        self._connected = False
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

    # --- Internal ---

    def _connect(self) -> None:
        try:
            self._serial = serial.Serial(
                port=self._port, baudrate=self._baud_rate, timeout=0, write_timeout=0
            )
            self._connected = True
        except (serial.SerialException, OSError) as exc:
            LOGGER.debug("Serial connect failed on %s: %s", self._port, exc)
            self._serial = None
            self._connected = False

    def _mark_disconnected(self) -> None:
        self._connected = False
        LOGGER.warning("Serial connection lost on %s", self._port)
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
