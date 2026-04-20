from __future__ import annotations

from collections.abc import Callable

import pytest

from src.config import CommConfig
from src.output.auto_comm import AutoCommTransport


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, delta_s: float) -> None:
        self.now += delta_s


class FakeSender:
    def __init__(
        self,
        *,
        connected: bool = True,
        send_failures: list[bool] | None = None,
        reconnect_results: list[bool] | None = None,
        label: str = "sender",
    ) -> None:
        self._connected = connected
        self._send_failures = list(send_failures or [])
        self._reconnect_results = list(reconnect_results or [])
        self.label = label
        self.closed = False
        self._packets_sent = 0
        self._packets_failed = 0

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
        del packet
        if not self._connected:
            self._packets_failed += 1
            return
        should_fail = self._send_failures.pop(0) if self._send_failures else False
        if should_fail:
            self._packets_failed += 1
            self._connected = False
            return
        self._packets_sent += 1

    def reconnect(self) -> bool:
        if self._connected:
            return True
        if self._reconnect_results:
            self._connected = self._reconnect_results.pop(0)
        return self._connected

    def health_check(self) -> dict[str, object]:
        return {
            "connected": self._connected,
            "label": self.label,
            "packets_sent": self._packets_sent,
            "packets_failed": self._packets_failed,
        }

    def close(self) -> None:
        self.closed = True
        self._connected = False


class SenderFactory:
    def __init__(self, senders: list[FakeSender | Exception]) -> None:
        self._senders = list(senders)
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object) -> FakeSender:
        self.calls.append(args)
        next_sender = self._senders.pop(0)
        if isinstance(next_sender, Exception):
            raise next_sender
        return next_sender


def _make_config() -> CommConfig:
    return CommConfig(
        channel="auto",
        serial_port="COM99",
        udp_host="1.2.3.4",
        udp_port=9999,
        udp_redundancy=1,
    )


@pytest.mark.unit
def test_auto_comm_prefers_serial_at_startup() -> None:
    serial_sender = FakeSender(label="serial")
    serial_factory = SenderFactory([serial_sender])
    udp_factory = SenderFactory([])

    transport = AutoCommTransport(
        _make_config(),
        serial_factory=serial_factory,
        udp_factory=udp_factory,
        clock=FakeClock(),
    )

    assert transport.active_channel == "serial"
    assert transport.is_connected is True
    assert len(serial_factory.calls) == 1
    assert udp_factory.calls == []


@pytest.mark.unit
def test_auto_comm_falls_back_to_udp_at_startup() -> None:
    serial_sender = FakeSender(connected=False, label="serial")
    udp_sender = FakeSender(label="udp")
    transport = AutoCommTransport(
        _make_config(),
        serial_factory=SenderFactory([serial_sender]),
        udp_factory=SenderFactory([udp_sender]),
        clock=FakeClock(),
    )

    assert transport.active_channel == "udp"
    assert serial_sender.closed is True


@pytest.mark.unit
def test_auto_comm_switches_to_udp_after_repeated_serial_failures() -> None:
    serial_sender = FakeSender(connected=True, send_failures=[True], label="serial")
    udp_sender = FakeSender(label="udp")
    transport = AutoCommTransport(
        _make_config(),
        serial_factory=SenderFactory([serial_sender]),
        udp_factory=SenderFactory([udp_sender]),
        clock=FakeClock(),
    )

    for _ in range(transport._SERIAL_FAILURE_THRESHOLD):
        transport.send(b"packet")

    assert transport.active_channel == "udp"
    assert transport.packets_failed == transport._SERIAL_FAILURE_THRESHOLD
    assert serial_sender.closed is True


@pytest.mark.unit
def test_auto_comm_does_not_retry_udp_failover_every_packet_when_udp_is_down() -> None:
    serial_sender = FakeSender(connected=True, send_failures=[True], label="serial")
    udp_factory = SenderFactory([OSError("udp down")])
    transport = AutoCommTransport(
        _make_config(),
        serial_factory=SenderFactory([serial_sender]),
        udp_factory=udp_factory,
        clock=FakeClock(),
    )

    for _ in range(transport._SERIAL_FAILURE_THRESHOLD + 1):
        transport.send(b"packet")

    assert transport.active_channel == "serial"
    assert transport.packets_failed == transport._SERIAL_FAILURE_THRESHOLD + 1
    assert len(udp_factory.calls) == 1


@pytest.mark.unit
def test_auto_comm_promotes_udp_back_to_serial_after_stability_window() -> None:
    clock = FakeClock()
    startup_serial = FakeSender(connected=False, label="serial-startup")
    missed_probe = FakeSender(connected=False, label="serial-missed-probe")
    first_good_probe = FakeSender(label="serial-probe-1")
    second_good_probe = FakeSender(label="serial-probe-2")
    promoted_serial = FakeSender(label="serial-promoted")
    udp_sender = FakeSender(label="udp")
    serial_factory = SenderFactory(
        [startup_serial, missed_probe, first_good_probe, second_good_probe, promoted_serial]
    )
    transport = AutoCommTransport(
        _make_config(),
        serial_factory=serial_factory,
        udp_factory=SenderFactory([udp_sender]),
        clock=clock,
    )

    transport.send(b"packet")
    assert transport.active_channel == "udp"

    clock.advance(0.6)
    transport.send(b"packet")
    assert transport.active_channel == "udp"

    clock.advance(0.8)
    transport.send(b"packet")
    assert transport.active_channel == "udp"

    clock.advance(0.6)
    transport.send(b"packet")

    assert transport.active_channel == "serial"
    assert udp_sender.closed is True
    assert first_good_probe.closed is True
    assert second_good_probe.closed is True


@pytest.mark.unit
def test_auto_comm_recovers_when_no_transport_is_available_at_startup() -> None:
    clock = FakeClock()
    startup_serial = FakeSender(connected=False, label="serial-startup")
    retry_serial = FakeSender(connected=False, label="serial-retry")
    udp_sender = FakeSender(label="udp-recovered")
    transport = AutoCommTransport(
        _make_config(),
        serial_factory=SenderFactory([startup_serial, retry_serial]),
        udp_factory=SenderFactory([OSError("udp down"), udp_sender]),
        clock=clock,
    )

    assert transport.active_channel is None
    assert transport.is_connected is False

    clock.advance(0.6)
    assert transport.reconnect() is True
    assert transport.active_channel == "udp"
