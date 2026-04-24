"""Keyboard-driven manual gimbal controller.

When the user presses ``m`` to enter manual mode, this driver takes over
the pan/tilt shared values: ZQSD jogs the gimbal at "fine" speed, and
the arrow keys jog at "coarse" speed (the two can be combined for
diagonal motion). Acceleration is ramped over ``accel_time_s`` so a
single key press doesn't slam the servos.

Previously this lived inline in :func:`src.main.main`. Extracting it
into a class keeps the velocity/timestamp state in one place where it
can be unit-tested with synthetic key sequences and a fake clock.

Key codes follow OpenCV's ``waitKeyEx`` convention: the low byte is
ASCII for letter keys, and the arrow keys come back as the constants
defined in :data:`ARROW_LEFT` / :data:`ARROW_UP` / etc.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

# OpenCV waitKeyEx return values for the arrow keys (Windows + most Linux).
ARROW_LEFT = 0x250000
ARROW_UP = 0x260000
ARROW_RIGHT = 0x270000
ARROW_DOWN = 0x280000


@dataclass(frozen=True)
class ManualKeyboardConfig:
    """Tuning knobs for the manual keyboard driver."""

    fine_speed_dps: float = 120.0
    coarse_speed_dps: float = 300.0
    accel_time_s: float = 0.05
    decel_multiplier: float = 3.0
    movement_threshold_dps: float = 0.05


class ManualKeyboardDriver:
    """Tracks per-tick velocity and updates pan/tilt mp.Values in place.

    Usage::

        driver = ManualKeyboardDriver(ManualKeyboardConfig())
        # ... per-frame, while in manual mode:
        driver.tick(
            key=key, key_low=key_low,
            manual_pan=manual_pan_mp_value,
            manual_tilt=manual_tilt_mp_value,
            pan_limit_deg=config.gimbal.pan_limit_deg,
            tilt_limit_deg=config.gimbal.tilt_limit_deg,
        )
        # ... when leaving manual mode:
        driver.reset_clock()
    """

    def __init__(
        self,
        config: ManualKeyboardConfig | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._config = config or ManualKeyboardConfig()
        self._clock = clock
        self._vel_pan: float = 0.0
        self._vel_tilt: float = 0.0
        self._last_time: float = clock()

    @property
    def velocity_pan_dps(self) -> float:
        return self._vel_pan

    @property
    def velocity_tilt_dps(self) -> float:
        return self._vel_tilt

    def reset_clock(self) -> None:
        """Re-anchor the integrator timestamp without clearing velocity.

        Call this when the user toggles into auto mode so the next manual
        re-entry doesn't see a giant ``dt`` from the gap.
        """
        self._last_time = self._clock()

    def reset(self) -> None:
        """Zero velocity and re-anchor the timestamp."""
        self._vel_pan = 0.0
        self._vel_tilt = 0.0
        self._last_time = self._clock()

    def tick(
        self,
        *,
        key: int,
        key_low: int,
        manual_pan,  # mp.sharedctypes.Synchronized[c_double]
        manual_tilt,  # mp.sharedctypes.Synchronized[c_double]
        pan_limit_deg: float,
        tilt_limit_deg: float,
    ) -> bool:
        """Apply one keyboard tick, returning True if the gimbal moved.

        ``manual_pan`` / ``manual_tilt`` are duck-typed against
        ``multiprocessing.Value`` (any object with a ``.value`` float
        attribute works — handy for tests).
        """
        now = self._clock()
        dt = min(now - self._last_time, 0.1)
        self._last_time = now

        req_pan, req_tilt = self._requested_velocity(key, key_low)

        ramp_rate = dt / self._config.accel_time_s if self._config.accel_time_s > 0 else 1.0
        if req_pan != 0.0:
            self._vel_pan += (req_pan - self._vel_pan) * min(1.0, ramp_rate)
        else:
            self._vel_pan *= max(0.0, 1.0 - ramp_rate * self._config.decel_multiplier)
        if req_tilt != 0.0:
            self._vel_tilt += (req_tilt - self._vel_tilt) * min(1.0, ramp_rate)
        else:
            self._vel_tilt *= max(0.0, 1.0 - ramp_rate * self._config.decel_multiplier)

        threshold = self._config.movement_threshold_dps
        moved = abs(self._vel_pan) > threshold or abs(self._vel_tilt) > threshold
        if moved:
            manual_pan.value = max(
                -pan_limit_deg,
                min(pan_limit_deg, manual_pan.value + self._vel_pan * dt),
            )
            manual_tilt.value = max(
                -tilt_limit_deg,
                min(tilt_limit_deg, manual_tilt.value + self._vel_tilt * dt),
            )
        else:
            self._vel_pan = 0.0
            self._vel_tilt = 0.0
            self._last_time = now
        return moved

    def _requested_velocity(self, key: int, key_low: int) -> tuple[float, float]:
        """Map the latest key event to (pan_dps, tilt_dps) request.

        ZQSD = fine speed, arrows = coarse speed. ZQSD and arrows can
        combine for diagonals because they're checked independently.
        """
        fine = self._config.fine_speed_dps
        coarse = self._config.coarse_speed_dps

        req_pan = 0.0
        req_tilt = 0.0

        if key_low == ord("q"):
            req_pan = -fine
        elif key_low == ord("d"):
            req_pan = fine
        elif key_low == ord("z"):
            req_tilt = -fine
        elif key_low == ord("s"):
            req_tilt = fine

        if key == ARROW_LEFT:
            req_pan = -coarse
        elif key == ARROW_RIGHT:
            req_pan = coarse
        if key == ARROW_UP:
            req_tilt = -coarse
        elif key == ARROW_DOWN:
            req_tilt = coarse

        return req_pan, req_tilt
