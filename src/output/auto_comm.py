from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Literal, Protocol

from src.config import CommConfig
from src.output.serial_comm import SerialComm
from src.output.udp_comm import UDPComm

LOGGER = logging.getLogger("auto_comm")

CommSenderChannel = Literal["serial", "udp"]


class CommSender(Protocol):
    @property
    def is_connected(self) -> bool: ...

    @property
    def packets_sent(self) -> int: ...

    @property
    def packets_failed(self) -> int: ...

    def send(self, packet: bytes) -> None: ...

    def reconnect(self) -> bool: ...

    def health_check(self) -> dict[str, object]: ...

    def close(self) -> None: ...


class AutoCommTransport:
    _SERIAL_FAILURE_THRESHOLD = 20
    _SERIAL_PROBE_INTERVAL_S = 0.5
    _SERIAL_STABILITY_WINDOW_S = 1.0
    _RESTORE_INTERVAL_S = 0.5

    def __init__(
        self,
        comm_config: CommConfig,
        *,
        serial_factory: Callable[[str, int], CommSender] = SerialComm,
        udp_factory: Callable[[str, int, int], CommSender] = UDPComm,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._comm_config = comm_config
        self._serial_factory = serial_factory
        self._udp_factory = udp_factory
        self._clock = clock
        self._active_sender: CommSender | None = None
        self._active_channel: CommSenderChannel | None = None
        self._packets_sent = 0
        self._packets_failed = 0
        self._consecutive_serial_failures = 0
        self._serial_candidate_since: float | None = None
        self._last_serial_probe_attempt = float("-inf")
        self._last_restore_attempt = float("-inf")
        self._restore_transport(force=True)

    @property
    def is_connected(self) -> bool:
        return self._active_sender is not None and self._active_sender.is_connected

    @property
    def packets_sent(self) -> int:
        return self._packets_sent

    @property
    def packets_failed(self) -> int:
        return self._packets_failed

    @property
    def active_channel(self) -> CommSenderChannel | None:
        return self._active_channel

    def send(self, packet: bytes) -> None:
        if self._active_channel != "serial":
            self._maybe_promote_serial()
        if self._active_sender is None:
            self._restore_transport()
            if self._active_sender is None:
                return

        active_sender = self._active_sender
        prev_sent = active_sender.packets_sent
        prev_failed = active_sender.packets_failed
        active_sender.send(packet)

        sent_delta = max(0, active_sender.packets_sent - prev_sent)
        failed_delta = max(0, active_sender.packets_failed - prev_failed)
        self._packets_sent += sent_delta
        self._packets_failed += failed_delta

        if self._active_channel == "serial":
            if failed_delta > 0:
                self._consecutive_serial_failures += failed_delta
                if self._consecutive_serial_failures >= self._SERIAL_FAILURE_THRESHOLD:
                    self._switch_to_udp()
            else:
                self._consecutive_serial_failures = 0

    def reconnect(self) -> bool:
        if self._active_channel == "serial" and self._active_sender is not None:
            if self._active_sender.reconnect():
                self._consecutive_serial_failures = 0
                return True
            return False
        if self._active_channel == "udp":
            self._maybe_promote_serial()
            return self.is_connected
        self._restore_transport()
        return self.is_connected

    def health_check(self) -> dict[str, object]:
        return {
            "connected": self.is_connected,
            "active_channel": self._active_channel,
            "packets_sent": self._packets_sent,
            "packets_failed": self._packets_failed,
            "sender": (
                self._active_sender.health_check() if self._active_sender is not None else None
            ),
        }

    def close(self) -> None:
        if self._active_sender is not None:
            self._active_sender.close()
            self._active_sender = None
            self._active_channel = None
        self._serial_candidate_since = None

    def _restore_transport(self, *, force: bool = False) -> bool:
        now = self._clock()
        if not force and now - self._last_restore_attempt < self._RESTORE_INTERVAL_S:
            return self.is_connected
        self._last_restore_attempt = now

        if self._try_activate_serial():
            return True
        if self._try_activate_udp():
            return True
        return False

    def _maybe_promote_serial(self) -> None:
        now = self._clock()
        if now - self._last_serial_probe_attempt < self._SERIAL_PROBE_INTERVAL_S:
            return
        self._last_serial_probe_attempt = now

        probe_sender = self._create_serial_sender()
        if probe_sender is None or not probe_sender.is_connected:
            if probe_sender is not None:
                probe_sender.close()
            self._serial_candidate_since = None
            return

        try:
            if self._serial_candidate_since is None:
                self._serial_candidate_since = now
                LOGGER.info(
                    "Auto transport: serial detected on %s, waiting %.1fs before failback",
                    self._comm_config.serial_port,
                    self._SERIAL_STABILITY_WINDOW_S,
                )
                return
            if now - self._serial_candidate_since < self._SERIAL_STABILITY_WINDOW_S:
                return

            LOGGER.info(
                "Auto transport: promoting back to serial on %s",
                self._comm_config.serial_port,
            )
            self._set_active_sender("serial", probe_sender)
            probe_sender = None
        finally:
            if probe_sender is not None:
                probe_sender.close()

    def _switch_to_udp(self) -> None:
        udp_sender = self._create_udp_sender()
        if udp_sender is None:
            LOGGER.warning(
                "Auto transport: UDP fallback unavailable after %d serial send failures",
                self._consecutive_serial_failures,
            )
            self._consecutive_serial_failures = 0
            return
        LOGGER.warning(
            "Auto transport: switching to UDP %s:%s after %d serial send failures",
            self._comm_config.udp_host,
            self._comm_config.udp_port,
            self._consecutive_serial_failures,
        )
        self._set_active_sender("udp", udp_sender)

    def _try_activate_serial(self) -> bool:
        serial_sender = self._create_serial_sender()
        if serial_sender is None or not serial_sender.is_connected:
            if serial_sender is not None:
                serial_sender.close()
            return False
        LOGGER.info("Auto transport: using serial on %s", self._comm_config.serial_port)
        self._set_active_sender("serial", serial_sender)
        return True

    def _try_activate_udp(self) -> bool:
        udp_sender = self._create_udp_sender()
        if udp_sender is None:
            return False
        LOGGER.info(
            "Auto transport: using UDP output on %s:%s",
            self._comm_config.udp_host,
            self._comm_config.udp_port,
        )
        self._set_active_sender("udp", udp_sender)
        return True

    def _set_active_sender(self, channel: CommSenderChannel, sender: CommSender) -> None:
        if self._active_sender is not None and self._active_sender is not sender:
            self._active_sender.close()
        self._active_sender = sender
        self._active_channel = channel
        self._consecutive_serial_failures = 0
        self._serial_candidate_since = None

    def _create_serial_sender(self) -> CommSender | None:
        try:
            return self._serial_factory(self._comm_config.serial_port, self._comm_config.baud_rate)
        except Exception as exc:
            LOGGER.info(
                "Auto transport: serial unavailable on %s (%s)",
                self._comm_config.serial_port,
                exc,
            )
            return None

    def _create_udp_sender(self) -> CommSender | None:
        try:
            return self._udp_factory(
                self._comm_config.udp_host,
                self._comm_config.udp_port,
                self._comm_config.udp_redundancy,
            )
        except Exception as exc:
            LOGGER.warning(
                "Auto transport: UDP unavailable on %s:%s (%s)",
                self._comm_config.udp_host,
                self._comm_config.udp_port,
                exc,
            )
            return None
