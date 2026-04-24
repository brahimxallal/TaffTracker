"""100 Hz manual-mode packet emitter loop.

When the user is flying the gimbal by hand (joystick / keyboard overrides),
the output process fires control packets at a fixed 100 Hz cadence
independent of inference output. That keeps the ESP32 servo loop well-fed
even if inference frames are slow, and it lets the UI feel responsive
immediately instead of waiting for the next tracking update.

Previously this lived inline in :class:`OutputProcess` as ``_manual_tick_loop``.
Extracting it lets the loop be unit-tested as a pure function and keeps the
output process shell free of threading boilerplate.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.sharedctypes import Synchronized
from typing import Protocol

from src.config import GimbalConfig, ServoControlConfig
from src.output.manual_control import (
    ManualVelocityTracker,
    boost_manual_velocity,
    build_manual_packet,
    rewrite_packet_sequence,
)


class _SenderLike(Protocol):
    """Duck-typed interface for the serial/UDP sender."""

    is_connected: bool

    def send(self, packet: bytes) -> None:  # pragma: no cover - protocol only
        ...


@dataclass(frozen=True)
class ManualLoopConfig:
    """Frozen view of the tuning knobs for the manual tick loop."""

    tick_s: float
    pan_sign: float
    tilt_sign: float
    pan_limit_deg: float
    tilt_limit_deg: float
    velocity_floor_dps: float

    @classmethod
    def from_configs(
        cls,
        gimbal_config: GimbalConfig,
        servo_control_config: ServoControlConfig,
        tick_s: float = 0.01,
    ) -> ManualLoopConfig:
        return cls(
            tick_s=tick_s,
            pan_sign=-1.0 if gimbal_config.invert_pan else 1.0,
            tilt_sign=-1.0 if gimbal_config.invert_tilt else 1.0,
            pan_limit_deg=gimbal_config.pan_limit_deg,
            tilt_limit_deg=gimbal_config.tilt_limit_deg,
            velocity_floor_dps=servo_control_config.manual_response_velocity_floor_dps,
        )


def run_manual_tick_loop(
    *,
    sender: _SenderLike | None,
    shutdown_event: threading.Event | mp.synchronize.Event,
    manual_mode: Synchronized | None,
    manual_pan: Synchronized | None,
    manual_tilt: Synchronized | None,
    relay_flag: Synchronized | None,
    laser_enabled: Synchronized | None,
    loop_config: ManualLoopConfig,
    get_next_sequence: Callable[[], int],
) -> None:
    """Run until ``shutdown_event`` is set.

    When manual mode is inactive (or when ``sender`` is None), the loop sleeps
    one tick and resets the velocity differentiator so that re-entering
    manual mode doesn't emit a stale jump.
    """
    velocity_tracker = ManualVelocityTracker()
    tick_s = loop_config.tick_s

    while not shutdown_event.is_set():
        is_manual = manual_mode is not None and bool(manual_mode.value)
        if not is_manual or sender is None:
            velocity_tracker.reset()
            time.sleep(tick_s)
            continue

        manual_pan_deg = float(manual_pan.value) if manual_pan is not None else 0.0
        manual_tilt_deg = float(manual_tilt.value) if manual_tilt is not None else 0.0
        manual_pan_deg = max(
            -loop_config.pan_limit_deg, min(loop_config.pan_limit_deg, manual_pan_deg)
        )
        manual_tilt_deg = max(
            -loop_config.tilt_limit_deg, min(loop_config.tilt_limit_deg, manual_tilt_deg)
        )

        pan_deg = manual_pan_deg * loop_config.pan_sign
        tilt_deg = manual_tilt_deg * loop_config.tilt_sign
        now_ns = time.perf_counter_ns()

        pan_vel_dps, tilt_vel_dps = velocity_tracker.compute_velocity_dps(pan_deg, tilt_deg, now_ns)
        pan_vel_dps = boost_manual_velocity(pan_vel_dps, loop_config.velocity_floor_dps)
        tilt_vel_dps = boost_manual_velocity(tilt_vel_dps, loop_config.velocity_floor_dps)

        relay_on = relay_flag is not None and bool(relay_flag.value)
        laser_on = laser_enabled is None or bool(laser_enabled.value)
        packet = build_manual_packet(
            pan_deg=pan_deg,
            tilt_deg=tilt_deg,
            pan_vel_dps=pan_vel_dps,
            tilt_vel_dps=tilt_vel_dps,
            now_ns=now_ns,
            laser_enabled=laser_on,
            relay_on=relay_on,
        )

        seq = get_next_sequence()
        packet = rewrite_packet_sequence(packet, seq)

        if sender.is_connected:
            sender.send(packet)

        time.sleep(tick_s)
