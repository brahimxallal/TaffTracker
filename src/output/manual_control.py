"""Manual-mode velocity helpers for the output process.

Extracted from ``src/output/process.py``. Manual mode bypasses the
Kalman + PID path and sends absolute pan/tilt angles directly, so it
needs its own velocity-estimation and minimum-response-speed logic.

Split out to keep the orchestrator focused on the auto-tracking
pipeline. All state lives inside :class:`ManualVelocityTracker`; the
free function :func:`boost_manual_velocity` is pure.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import copysign

# Cap dt so a long pause between manual commands does not produce a
# pathological velocity spike on the next frame.
_MAX_DT_S = 0.1


def boost_manual_velocity(velocity_dps: float, floor_dps: float) -> float:
    """Ensure every non-zero manual velocity clears the servo response floor.

    Zero stays zero (so a released stick reports no motion). Any non-zero
    velocity is lifted to at least ``floor_dps`` while preserving sign, so
    the servo reacts immediately instead of idling under static friction.
    """
    if abs(velocity_dps) <= 1e-6:
        return 0.0
    return copysign(max(abs(velocity_dps), floor_dps), velocity_dps)


@dataclass
class ManualVelocityTracker:
    """Numerical differentiator for manual-mode pan/tilt commands.

    Holds the last commanded (pan_deg, tilt_deg, timestamp_ns) triple and
    returns the instantaneous angular velocity in deg/s. ``reset()`` drops
    the history so the next sample produces zero velocity — use it when
    the user toggles manual off and on to avoid a phantom jump.
    """

    _last_pan_deg: float | None = None
    _last_tilt_deg: float | None = None
    _last_timestamp_ns: int | None = None

    def reset(self) -> None:
        """Forget the previous sample so the next call returns (0, 0)."""
        self._last_pan_deg = None
        self._last_tilt_deg = None
        self._last_timestamp_ns = None

    def compute_velocity_dps(
        self,
        pan_deg: float,
        tilt_deg: float,
        timestamp_ns: int,
    ) -> tuple[float, float]:
        """Return (pan_dps, tilt_dps) against the previous recorded sample.

        Always stores the current sample before returning, so consecutive
        calls produce a rolling derivative. Returns (0, 0) on the first
        call after construction or ``reset()``, and also when the supplied
        timestamp is not strictly newer than the stored one.
        """
        if (
            self._last_timestamp_ns is None
            or self._last_pan_deg is None
            or self._last_tilt_deg is None
            or timestamp_ns <= self._last_timestamp_ns
        ):
            pan_vel_dps = 0.0
            tilt_vel_dps = 0.0
        else:
            dt = min((timestamp_ns - self._last_timestamp_ns) / 1e9, _MAX_DT_S)
            if dt <= 0.0:
                pan_vel_dps = 0.0
                tilt_vel_dps = 0.0
            else:
                pan_vel_dps = (pan_deg - self._last_pan_deg) / dt
                tilt_vel_dps = (tilt_deg - self._last_tilt_deg) / dt
        self._last_pan_deg = pan_deg
        self._last_tilt_deg = tilt_deg
        self._last_timestamp_ns = timestamp_ns
        return pan_vel_dps, tilt_vel_dps
