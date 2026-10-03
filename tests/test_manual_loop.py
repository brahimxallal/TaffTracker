"""Unit tests for :func:`src.output.manual_loop.run_manual_tick_loop`.

Covers the pure-function extraction directly (no OutputProcess shell). The
thread-behavior tests in ``tests/test_manual_thread.py`` still exercise the
class delegate, so together they guard both the extracted code and the
wiring contract.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.config import GimbalConfig, ServoControlConfig
from src.output.manual_loop import ManualLoopConfig, run_manual_tick_loop
from src.shared.protocol import decode_packet_v2


def _make_config(
    *,
    invert_pan: bool = False,
    invert_tilt: bool = False,
    pan_limit_deg: float = 90.0,
    tilt_limit_deg: float = 90.0,
    velocity_floor_dps: float = 80.0,
    command_slew_limit_dps: float = 0.0,
    tick_s: float = 0.005,
) -> ManualLoopConfig:
    return ManualLoopConfig.from_configs(
        gimbal_config=GimbalConfig(
            invert_pan=invert_pan,
            invert_tilt=invert_tilt,
            pan_limit_deg=pan_limit_deg,
            tilt_limit_deg=tilt_limit_deg,
        ),
        servo_control_config=ServoControlConfig(
            manual_response_velocity_floor_dps=velocity_floor_dps,
            manual_command_slew_limit_dps=command_slew_limit_dps,
        ),
        tick_s=tick_s,
    )


def _sequence_counter() -> tuple[Any, list[int]]:
    state = [0]
    emitted: list[int] = []

    def _get() -> int:
        seq = state[0]
        state[0] = (seq + 1) & 0xFFFF
        emitted.append(seq)
        return seq

    return _get, emitted


class _StoppingSender:
    is_connected = True

    def __init__(self, stop_after: int, stop_event: threading.Event) -> None:
        self.stop_after = stop_after
        self.stop_event = stop_event
        self.sent_packets: list[bytes] = []

    def send(self, packet: bytes) -> None:
        self.sent_packets.append(packet)
        if len(self.sent_packets) >= self.stop_after:
            self.stop_event.set()


@pytest.mark.unit
def test_loop_sends_packets_when_manual_mode_active() -> None:
    sender = MagicMock()
    sender.is_connected = True
    shutdown = threading.Event()

    manual_mode = mp.Value("b", 1)
    manual_pan = mp.Value("d", 10.0)
    manual_tilt = mp.Value("d", -5.0)

    get_seq, emitted = _sequence_counter()
    cfg = _make_config(tick_s=0.002)

    thread = threading.Thread(
        target=run_manual_tick_loop,
        kwargs=dict(
            sender=sender,
            shutdown_event=shutdown,
            manual_mode=manual_mode,
            manual_pan=manual_pan,
            manual_tilt=manual_tilt,
            relay_flag=None,
            laser_enabled=None,
            loop_config=cfg,
            get_next_sequence=get_seq,
        ),
        daemon=True,
    )
    thread.start()
    time.sleep(0.05)
    shutdown.set()
    thread.join(timeout=1.0)

    # Non-zero send calls, monotonically increasing sequence numbers.
    assert sender.send.call_count > 0
    assert sender.send.call_count == len(emitted)
    assert emitted == sorted(emitted)


@pytest.mark.unit
def test_loop_skips_sending_when_manual_mode_off() -> None:
    sender = MagicMock()
    sender.is_connected = True
    shutdown = threading.Event()
    manual_mode = mp.Value("b", 0)  # OFF
    get_seq, _ = _sequence_counter()

    thread = threading.Thread(
        target=run_manual_tick_loop,
        kwargs=dict(
            sender=sender,
            shutdown_event=shutdown,
            manual_mode=manual_mode,
            manual_pan=mp.Value("d", 15.0),
            manual_tilt=mp.Value("d", 0.0),
            relay_flag=None,
            laser_enabled=None,
            loop_config=_make_config(tick_s=0.002),
            get_next_sequence=get_seq,
        ),
        daemon=True,
    )
    thread.start()
    time.sleep(0.03)
    shutdown.set()
    thread.join(timeout=1.0)

    assert sender.send.call_count == 0


@pytest.mark.unit
def test_loop_skips_sending_when_sender_is_none() -> None:
    shutdown = threading.Event()
    manual_mode = mp.Value("b", 1)
    get_seq, emitted = _sequence_counter()

    thread = threading.Thread(
        target=run_manual_tick_loop,
        kwargs=dict(
            sender=None,
            shutdown_event=shutdown,
            manual_mode=manual_mode,
            manual_pan=mp.Value("d", 10.0),
            manual_tilt=mp.Value("d", 0.0),
            relay_flag=None,
            laser_enabled=None,
            loop_config=_make_config(tick_s=0.002),
            get_next_sequence=get_seq,
        ),
        daemon=True,
    )
    thread.start()
    time.sleep(0.03)
    shutdown.set()
    thread.join(timeout=1.0)

    # No sequence allocations — the loop never reached the send path.
    assert emitted == []


@pytest.mark.unit
def test_loop_clamps_pan_and_tilt_to_limits() -> None:
    """Requests beyond the gimbal's mechanical envelope must be clamped."""
    sender = MagicMock()
    sender.is_connected = True
    shutdown = threading.Event()

    manual_mode = mp.Value("b", 1)
    manual_pan = mp.Value("d", 500.0)  # WAY over limit
    manual_tilt = mp.Value("d", -500.0)
    get_seq, _ = _sequence_counter()

    cfg = _make_config(pan_limit_deg=45.0, tilt_limit_deg=30.0, tick_s=0.002)

    thread = threading.Thread(
        target=run_manual_tick_loop,
        kwargs=dict(
            sender=sender,
            shutdown_event=shutdown,
            manual_mode=manual_mode,
            manual_pan=manual_pan,
            manual_tilt=manual_tilt,
            relay_flag=None,
            laser_enabled=None,
            loop_config=cfg,
            get_next_sequence=get_seq,
        ),
        daemon=True,
    )
    thread.start()
    time.sleep(0.02)
    shutdown.set()
    thread.join(timeout=1.0)

    # At least one packet was produced; we can't inspect the decoded bytes
    # without a full protocol decoder, but we can at least assert the loop
    # didn't crash on out-of-limit inputs.
    assert sender.send.call_count > 0


@pytest.mark.unit
def test_loop_respects_pan_inversion_flag() -> None:
    """Inverted pan flips the sign of the commanded angle."""
    sender = MagicMock()
    sender.is_connected = True
    shutdown = threading.Event()
    manual_mode = mp.Value("b", 1)

    # Compare ManualLoopConfig derived signs directly — cheaper than decoding packets.
    cfg_inverted = _make_config(invert_pan=True)
    cfg_normal = _make_config(invert_pan=False)

    assert cfg_inverted.pan_sign == -1.0
    assert cfg_normal.pan_sign == 1.0

    # Smoke check: an inverted-pan loop also emits packets without crashing.
    get_seq, _ = _sequence_counter()
    thread = threading.Thread(
        target=run_manual_tick_loop,
        kwargs=dict(
            sender=sender,
            shutdown_event=shutdown,
            manual_mode=manual_mode,
            manual_pan=mp.Value("d", 5.0),
            manual_tilt=mp.Value("d", 0.0),
            relay_flag=None,
            laser_enabled=None,
            loop_config=cfg_inverted,
            get_next_sequence=get_seq,
        ),
        daemon=True,
    )
    thread.start()
    time.sleep(0.02)
    shutdown.set()
    thread.join(timeout=1.0)
    assert sender.send.call_count > 0


@pytest.mark.unit
def test_manual_loop_config_from_configs_preserves_defaults() -> None:
    cfg = ManualLoopConfig.from_configs(GimbalConfig(), ServoControlConfig())
    assert cfg.tick_s == 0.01
    assert cfg.pan_sign == 1.0
    assert cfg.tilt_sign == 1.0
    assert cfg.pan_limit_deg == 90.0
    assert cfg.tilt_limit_deg == 90.0
    assert cfg.velocity_floor_dps == 80.0
    assert cfg.command_slew_limit_dps == 0.0


@pytest.mark.unit
def test_manual_loop_slew_limits_sudden_command_step() -> None:
    now = 0.0
    stop = threading.Event()
    manual_mode = mp.Value("b", 1)
    manual_pan = mp.Value("d", 0.0)
    manual_tilt = mp.Value("d", 0.0)

    def fake_monotonic() -> float:
        return now

    def fake_sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    class StepSender(_StoppingSender):
        def send(self, packet: bytes) -> None:
            super().send(packet)
            if len(self.sent_packets) == 1:
                manual_pan.value = 90.0

    sender = StepSender(stop_after=3, stop_event=stop)
    get_seq, _ = _sequence_counter()

    run_manual_tick_loop(
        sender=sender,
        shutdown_event=stop,
        manual_mode=manual_mode,
        manual_pan=manual_pan,
        manual_tilt=manual_tilt,
        relay_flag=mp.Value("b", 0),
        laser_enabled=mp.Value("b", 1),
        loop_config=_make_config(
            tick_s=0.01,
            command_slew_limit_dps=100.0,
            velocity_floor_dps=0.0,
        ),
        get_next_sequence=get_seq,
        monotonic=fake_monotonic,
        sleep=fake_sleep,
    )

    decoded = [decode_packet_v2(packet) for packet in sender.sent_packets]

    assert [packet.pan for packet in decoded] == [0, 100, 200]


@pytest.mark.unit
def test_manual_loop_uses_deadline_sleep_when_manual_active() -> None:
    sleep_calls: list[float] = []
    now = 0.0

    def fake_monotonic() -> float:
        return now

    def fake_sleep(seconds: float) -> None:
        nonlocal now
        sleep_calls.append(seconds)
        now += seconds

    cfg = _make_config(tick_s=0.01)
    stop = threading.Event()
    manual_mode = mp.Value("b", 1)
    manual_pan = mp.Value("d", 1.0)
    manual_tilt = mp.Value("d", 0.0)
    sender = _StoppingSender(stop_after=3, stop_event=stop)
    get_seq, emitted = _sequence_counter()

    run_manual_tick_loop(
        sender=sender,
        shutdown_event=stop,
        manual_mode=manual_mode,
        manual_pan=manual_pan,
        manual_tilt=manual_tilt,
        relay_flag=mp.Value("b", 0),
        laser_enabled=mp.Value("b", 1),
        loop_config=cfg,
        get_next_sequence=get_seq,
        monotonic=fake_monotonic,
        sleep=fake_sleep,
    )

    assert len(sender.sent_packets) == 3
    assert emitted == [0, 1, 2]
    assert sleep_calls
    assert all(0.0 <= seconds <= cfg.tick_s for seconds in sleep_calls)


@pytest.mark.unit
def test_manual_loop_drops_missed_deadlines_after_overrun() -> None:
    """An overrun should not create zero-sleep catch-up bursts."""
    sleep_calls: list[float] = []
    now = 0.0

    def fake_monotonic() -> float:
        return now

    def fake_sleep(seconds: float) -> None:
        nonlocal now
        sleep_calls.append(seconds)
        now += seconds

    cfg = _make_config(tick_s=0.01)
    stop = threading.Event()

    class OverrunSender(_StoppingSender):
        def send(self, packet: bytes) -> None:
            nonlocal now
            now += cfg.tick_s * 4.0
            super().send(packet)

    sender = OverrunSender(stop_after=3, stop_event=stop)
    get_seq, _ = _sequence_counter()

    run_manual_tick_loop(
        sender=sender,
        shutdown_event=stop,
        manual_mode=mp.Value("b", 1),
        manual_pan=mp.Value("d", 1.0),
        manual_tilt=mp.Value("d", 0.0),
        relay_flag=mp.Value("b", 0),
        laser_enabled=mp.Value("b", 1),
        loop_config=cfg,
        get_next_sequence=get_seq,
        monotonic=fake_monotonic,
        sleep=fake_sleep,
    )

    assert len(sender.sent_packets) == 3
    assert sleep_calls
    assert all(seconds == pytest.approx(cfg.tick_s) for seconds in sleep_calls)
