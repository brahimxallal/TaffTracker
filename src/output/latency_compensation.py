"""Delay-aware helpers for the camera-on-gimbal auto controller.

Why this exists
---------------
The auto controller integrates a commanded position from the image-plane error.
That error describes where the target was one *loop delay* ago: camera exposure,
stream/driver buffering, the PC pipeline and the servo's own response all sit
between a command leaving the host and the frame that shows its effect arriving
back. For an integrating loop the usable gain is bounded by roughly
``0.5 / delay``, so with ~200 ms of delay the loop is stuck at a time constant of
0.4-1 s, and a target walking at ``v`` deg/s is trailed by ``v / kp`` degrees.

Two ideas lift that limit without touching the tracking hot path:

* **In-flight compensation** (a Smith predictor for an integrator). Part of the
  motion already commanded has not shown up in the image yet, so the controller
  would "see" an error it has already corrected and keep pushing. Subtracting the
  in-flight motion, ``e_cmd = e_meas - (c_last - c(t_meas - delay))``, removes the
  delay from the loop for target-static disturbances and allows a much higher gain.
* **Absolute-velocity feedforward.** Image-plane velocity is *relative* to the
  moving camera, so it vanishes in steady tracking and cannot remove the ramp lag.
  Adding the gimbal's own (delay-aligned) motion back gives the target's absolute
  angle, ``p = e_meas + c(t_meas - delay)``, whose derivative is the target's true
  angular velocity and is independent of the loop.

``delay`` is the time between a command being issued and the frame that first
shows it arriving back, in the same clock as ``TrackingMessage.timestamp_ns``
(camera + stream + servo response; the PC pipeline age is added separately).

It does not have to be exact. In closed-loop simulation the shipped gains stay
stable for an assumed delay from about 0.1 s below to about 0.25 s above the true
one (40-250 ms cameras, 30/60 fps, fast to slow servos). Under-estimating is the
dangerous side: the feedforward then reads the commanded acceleration as target
motion, which is positive feedback. That is why the velocity estimator is kept
deliberately slow (``velocity_tau_s``). ``delay_s == 0`` keeps the legacy
controller bit-for-bit.

A controller without lag cannot know the target stopped until it sees it, so
after a sudden stop it overshoots by roughly the loop delay times the target
speed; ``velocity_ff_gain`` below 1 trades some of the lag reduction for that.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class LatencyCompensationConfig:
    """Tuning for :func:`compensate_error`. ``delay_s == 0`` disables everything."""

    delay_s: float = 0.0
    smith_gain: float = 1.0
    velocity_ff_gain: float = 1.0
    # Time constant of the velocity estimator, in seconds (frame-rate independent).
    velocity_tau_s: float = 0.18
    velocity_deadband_dps: float = 3.0
    velocity_max_dps: float = 150.0
    # A single measurement may move the velocity estimate by at most this much angle...
    innovation_clamp_deg: float = 6.0
    # ...and a jump larger than this restarts the estimator (new target / ID switch).
    reset_jump_deg: float = 20.0
    # Upper bound on the pipeline age added to the lead (a stalled frame must not
    # inflate the extrapolation).
    max_pipeline_age_s: float = 0.05

    def __post_init__(self) -> None:
        if self.delay_s < 0.0 or self.delay_s > 1.0:
            raise ValueError(f"delay_s must be in [0, 1]; got {self.delay_s}")
        if self.smith_gain < 0.0 or self.smith_gain > 1.5:
            raise ValueError(f"smith_gain must be in [0, 1.5]; got {self.smith_gain}")
        if self.velocity_ff_gain < 0.0 or self.velocity_ff_gain > 1.5:
            raise ValueError(f"velocity_ff_gain must be in [0, 1.5]; got {self.velocity_ff_gain}")
        if not 0.0 < self.velocity_tau_s <= 2.0:
            raise ValueError(f"velocity_tau_s must be in (0, 2]; got {self.velocity_tau_s}")
        if self.velocity_deadband_dps < 0.0 or self.velocity_max_dps <= 0.0:
            raise ValueError("velocity_deadband_dps must be >= 0 and velocity_max_dps > 0")

    @property
    def enabled(self) -> bool:
        return self.delay_s > 0.0 and (self.smith_gain > 0.0 or self.velocity_ff_gain > 0.0)


class CommandHistory:
    """Bounded, time-stamped record of the commands the controller has issued.

    Timestamps are the ``TrackingMessage`` timestamps (seconds), so lookups share
    a clock with the frames the compensation is aligned against.
    """

    def __init__(self, maxlen: int = 512) -> None:
        self._t: deque[float] = deque(maxlen=maxlen)
        self._pan: deque[float] = deque(maxlen=maxlen)
        self._tilt: deque[float] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._t)

    def clear(self) -> None:
        self._t.clear()
        self._pan.clear()
        self._tilt.clear()

    def append(self, t_s: float, pan_deg: float, tilt_deg: float) -> None:
        if self._t and t_s <= self._t[-1]:
            # Clock went backwards (restart / skew): old samples would mis-align.
            self.clear()
        self._t.append(t_s)
        self._pan.append(pan_deg)
        self._tilt.append(tilt_deg)

    @property
    def latest(self) -> tuple[float, float] | None:
        if not self._t:
            return None
        return self._pan[-1], self._tilt[-1]

    def at(self, t_s: float) -> tuple[float, float] | None:
        """Command in force at ``t_s`` (linear interpolation, clamped to the record)."""
        n = len(self._t)
        if n == 0:
            return None
        times = self._t
        if t_s >= times[-1]:
            return self._pan[-1], self._tilt[-1]
        if t_s <= times[0]:
            return self._pan[0], self._tilt[0]
        # Queries are always recent, so walk back from the newest sample.
        i = n - 1
        while i > 0 and times[i - 1] > t_s:
            i -= 1
        t0 = times[i - 1]
        t1 = times[i]
        w = (t_s - t0) / (t1 - t0) if t1 > t0 else 1.0
        return (
            self._pan[i - 1] + w * (self._pan[i] - self._pan[i - 1]),
            self._tilt[i - 1] + w * (self._tilt[i] - self._tilt[i - 1]),
        )


class AxisVelocityEstimator:
    """Alpha-beta tracker on one axis of the target's *absolute* angle (degrees)."""

    def __init__(self) -> None:
        self._position = 0.0
        self._velocity = 0.0
        self._t_s: float | None = None

    @property
    def velocity_dps(self) -> float:
        return self._velocity

    def reset(self) -> None:
        self._position = 0.0
        self._velocity = 0.0
        self._t_s = None

    def _restart(self, position_deg: float, t_s: float) -> float:
        self._position = position_deg
        self._velocity = 0.0
        self._t_s = t_s
        return 0.0

    def update(self, position_deg: float, t_s: float, config: LatencyCompensationConfig) -> float:
        """Feed one absolute-angle sample; returns the velocity estimate (deg/s)."""
        if self._t_s is None:
            return self._restart(position_deg, t_s)
        dt = t_s - self._t_s
        if dt <= 0.0 or dt > 0.25:
            return self._restart(position_deg, t_s)
        predicted = self._position + self._velocity * dt
        residual = position_deg - predicted
        if abs(residual) > config.reset_jump_deg:
            return self._restart(position_deg, t_s)
        clamp = config.innovation_clamp_deg
        residual = max(-clamp, min(clamp, residual))
        # Critically damped alpha-beta gains for a pole pair of time constant tau.
        alpha = 1.0 - math.exp(-dt / config.velocity_tau_s)
        beta = alpha * alpha / (2.0 - alpha)
        self._position = predicted + alpha * residual
        limit = config.velocity_max_dps
        self._velocity = max(-limit, min(limit, self._velocity + beta * residual / dt))
        self._t_s = t_s
        return self._velocity


@dataclass(frozen=True, slots=True)
class CompensatedError:
    pan_err_deg: float
    tilt_err_deg: float
    pan_ff_dps: float
    tilt_ff_dps: float


class LatencyCompensator:
    """Per-run state for :func:`compensate_error` (history + velocity estimators)."""

    def __init__(self) -> None:
        self.history = CommandHistory()
        self.pan = AxisVelocityEstimator()
        self.tilt = AxisVelocityEstimator()

    def reset(self) -> None:
        self.history.clear()
        self.pan.reset()
        self.tilt.reset()

    def record_command(self, t_s: float, pan_deg: float, tilt_deg: float) -> None:
        self.history.append(t_s, pan_deg, tilt_deg)


def _soft_deadband(value: float, deadband: float) -> float:
    """Shrink ``value`` toward zero by ``deadband`` (continuous, unlike a hard snap)."""
    if value > deadband:
        return value - deadband
    if value < -deadband:
        return value + deadband
    return 0.0


def compensate_error(
    *,
    config: LatencyCompensationConfig,
    compensator: LatencyCompensator,
    t_s: float,
    pan_err_deg: float,
    tilt_err_deg: float,
    pipeline_age_s: float,
    fresh_measurement: bool,
) -> CompensatedError:
    """Return the delay-compensated error and velocity feedforward for one frame.

    ``pan_err_deg`` / ``tilt_err_deg`` are the controller-frame image errors
    (target minus camera axis) measured in the frame stamped ``t_s``. The
    feedforward is in deg/s and is meant to be added to the integrated rate.
    ``fresh_measurement`` is False for planned Kalman-only frames: the velocity
    estimator is not fed with predictions, it just keeps its last estimate.
    """
    history = compensator.history
    latest = history.latest
    anchor = history.at(t_s - config.delay_s)
    if latest is None or anchor is None:
        return CompensatedError(pan_err_deg, tilt_err_deg, 0.0, 0.0)

    lead_s = config.delay_s + max(0.0, min(config.max_pipeline_age_s, pipeline_age_s))
    ff_gain = config.velocity_ff_gain

    if ff_gain > 0.0:
        if fresh_measurement:
            pan_v = compensator.pan.update(pan_err_deg + anchor[0], t_s, config)
            tilt_v = compensator.tilt.update(tilt_err_deg + anchor[1], t_s, config)
        else:
            pan_v = compensator.pan.velocity_dps
            tilt_v = compensator.tilt.velocity_dps
        pan_ff = ff_gain * _soft_deadband(pan_v, config.velocity_deadband_dps)
        tilt_ff = ff_gain * _soft_deadband(tilt_v, config.velocity_deadband_dps)
    else:
        pan_ff = 0.0
        tilt_ff = 0.0

    smith = config.smith_gain
    pan_cmd_err = pan_err_deg - smith * (latest[0] - anchor[0]) + pan_ff * lead_s
    tilt_cmd_err = tilt_err_deg - smith * (latest[1] - anchor[1]) + tilt_ff * lead_s
    return CompensatedError(pan_cmd_err, tilt_cmd_err, pan_ff, tilt_ff)
