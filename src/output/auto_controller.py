"""Camera-on-gimbal incremental PD controller for auto-tracking mode.

Previously this lived inline in :meth:`OutputProcess._encode_packet`. The
extracted module owns the per-frame controller math:

    error → gain-scheduled rate → derivative damping → slew limit
    → integral accumulation → integral decay → clamp to mechanical limits

It does *not* own packet encoding, manual-mode handling, or laser
boresight offsets — those stay in ``OutputProcess`` because they couple
to multiprocessing-shared values that don't belong in pure logic.

The controller is split into:

* :class:`AutoControllerConfig` — frozen view of the tuning knobs.
* :class:`AutoControllerState` — mutable per-frame integrator/derivative
  state, owned by the caller and threaded through each tick.
* :func:`compute_auto_command` — pure step function: takes the latest
  tracking message + current state + config, mutates state in place,
  and returns the commanded ``(pan_deg, tilt_deg)``.

Keeping state out of the function signature (no closures) makes this
trivially unit-testable: build a state, call the function repeatedly
with synthetic messages, assert the integrator behaves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import degrees

from src.config import GimbalConfig, ServoControlConfig
from src.shared.types import TrackingMessage


@dataclass(frozen=True)
class AutoControllerConfig:
    """Frozen view of all gains the auto controller needs.

    Built from :class:`GimbalConfig` + :class:`ServoControlConfig` so the
    OutputProcess doesn't have to pass two configs at every call.
    """

    pan_sign: float
    tilt_sign: float
    pan_limit_deg: float
    tilt_limit_deg: float
    kd: float
    slew_limit_dps: float
    integral_decay_rate: float
    tilt_scale: float
    deadband_deg: float
    predictive_lead_s: float
    gain_schedule_threshold_deg: float
    effective_kp_far: float
    effective_kp_near: float
    velocity_feedforward_gain: float
    derivative_filter_alpha: float

    @classmethod
    def from_configs(
        cls,
        gimbal_config: GimbalConfig,
        servo_control_config: ServoControlConfig,
    ) -> AutoControllerConfig:
        return cls(
            pan_sign=-1.0 if gimbal_config.invert_pan else 1.0,
            tilt_sign=-1.0 if gimbal_config.invert_tilt else 1.0,
            pan_limit_deg=gimbal_config.pan_limit_deg,
            tilt_limit_deg=gimbal_config.tilt_limit_deg,
            kd=gimbal_config.kd,
            slew_limit_dps=gimbal_config.slew_limit_dps,
            integral_decay_rate=gimbal_config.integral_decay_rate,
            tilt_scale=gimbal_config.tilt_scale,
            deadband_deg=gimbal_config.deadband_deg,
            predictive_lead_s=gimbal_config.predictive_lead_s,
            gain_schedule_threshold_deg=gimbal_config.gain_schedule_threshold_deg,
            effective_kp_far=gimbal_config.effective_kp_far,
            effective_kp_near=gimbal_config.effective_kp_near,
            velocity_feedforward_gain=gimbal_config.velocity_feedforward_gain,
            derivative_filter_alpha=servo_control_config.derivative_filter_alpha,
        )


@dataclass
class AutoControllerState:
    """Per-frame integrator + derivative state.

    Mutated in place by :func:`compute_auto_command`. Reset via
    :meth:`reset` when entering manual mode or when the tracker reports
    ``state_source == "center"``.
    """

    pi_integral_pan: float = 0.0
    pi_integral_tilt: float = 0.0
    prev_err_pan: float = 0.0
    prev_err_tilt: float = 0.0
    prev_d_pan: float = 0.0
    prev_d_tilt: float = 0.0
    last_encode_ts_ns: int | None = field(default=None)

    def reset(self) -> None:
        """Zero all integrators + derivative history."""
        self.pi_integral_pan = 0.0
        self.pi_integral_tilt = 0.0
        self.prev_err_pan = 0.0
        self.prev_err_tilt = 0.0
        self.prev_d_pan = 0.0
        self.prev_d_tilt = 0.0
        self.last_encode_ts_ns = None


def _compute_dt_seconds(
    message: TrackingMessage,
    state: AutoControllerState,
) -> float:
    """Time delta between this and the previous tick, clamped to 100 ms.

    Falls back to ``1 / fps`` (or 1/60 s) on the first tick or when the
    timestamp goes backwards (clock skew across processes).
    """
    fallback = 1.0 / max(message.fps, 1.0) if message.fps > 0 else 1.0 / 60.0
    if state.last_encode_ts_ns is None:
        return fallback
    if message.timestamp_ns <= state.last_encode_ts_ns:
        return fallback
    return min((message.timestamp_ns - state.last_encode_ts_ns) / 1e9, 0.1)


def _compute_lead_error_deg(
    message: TrackingMessage,
    config: AutoControllerConfig,
) -> tuple[float, float]:
    """Error angle including velocity-based predictive lead, sign-corrected.

    Lead is only applied when the target is moving fast enough (>30 dps)
    to be worth predicting; static targets get no lead so we don't
    amplify low-speed noise into command jitter.
    """
    angles_rad = message.servo_angles or message.filtered_angles or (0.0, 0.0)
    vel_rad = message.servo_angular_velocity or message.angular_velocity or (0.0, 0.0)
    speed_dps = (degrees(vel_rad[0]) ** 2 + degrees(vel_rad[1]) ** 2) ** 0.5
    lead = config.predictive_lead_s if speed_dps > 30.0 else 0.0
    pan_err_deg = (degrees(angles_rad[0]) + degrees(vel_rad[0]) * lead) * config.pan_sign
    tilt_err_deg = (degrees(angles_rad[1]) + degrees(vel_rad[1]) * lead) * config.tilt_sign
    return pan_err_deg, tilt_err_deg


def _gain_scheduled_kp(error_abs_deg: float, config: AutoControllerConfig) -> float:
    """Aggressive far, gentle near."""
    if error_abs_deg > config.gain_schedule_threshold_deg:
        return config.effective_kp_far
    return config.effective_kp_near


def _slew_cap_dps(error_abs_deg: float, config: AutoControllerConfig) -> float:
    """Distance-scaled slew: larger error → higher slew cap, with a 25% floor."""
    floor = config.slew_limit_dps * 0.25
    return min(config.slew_limit_dps, max(floor, error_abs_deg * 6.0))


def _track_error_command(
    message: TrackingMessage,
    state: AutoControllerState,
    config: AutoControllerConfig,
    pan_err_deg: float,
    tilt_err_deg: float,
    dt: float,
) -> tuple[float, float]:
    """Run the controller for one tick when the target is acquired."""
    pan_abs = abs(pan_err_deg)
    tilt_abs = abs(tilt_err_deg)
    kp_pan = _gain_scheduled_kp(pan_abs, config)
    kp_tilt = _gain_scheduled_kp(tilt_abs, config)

    # Velocity feedforward (sign-corrected to controller frame).
    ang_vel = message.servo_angular_velocity or message.angular_velocity or (0.0, 0.0)
    ff_pan = config.velocity_feedforward_gain * degrees(ang_vel[0]) * config.pan_sign
    ff_tilt = config.velocity_feedforward_gain * degrees(ang_vel[1]) * config.tilt_sign

    # Approach rate (°/s) + feedforward, zero inside deadband.
    rate_pan = (kp_pan * pan_err_deg + ff_pan) if pan_abs >= config.deadband_deg else 0.0
    rate_tilt = (
        (kp_tilt * config.tilt_scale * tilt_err_deg + ff_tilt)
        if tilt_abs >= config.deadband_deg
        else 0.0
    )

    # Derivative damping from LP-filtered error finite-difference.
    if dt > 0:
        raw_d_pan = (pan_err_deg - state.prev_err_pan) / dt
        raw_d_tilt = (tilt_err_deg - state.prev_err_tilt) / dt
        a = config.derivative_filter_alpha
        d_err_pan = a * raw_d_pan + (1.0 - a) * state.prev_d_pan
        d_err_tilt = a * raw_d_tilt + (1.0 - a) * state.prev_d_tilt
        state.prev_d_pan = d_err_pan
        state.prev_d_tilt = d_err_tilt
    else:
        d_err_pan = 0.0
        d_err_tilt = 0.0
    rate_pan += config.kd * d_err_pan
    rate_tilt += config.kd * config.tilt_scale * d_err_tilt

    state.prev_err_pan = pan_err_deg
    state.prev_err_tilt = tilt_err_deg

    # Distance-scaled slew limit.
    slew_pan = _slew_cap_dps(pan_abs, config)
    slew_tilt = _slew_cap_dps(tilt_abs, config)
    rate_pan = max(-slew_pan, min(slew_pan, rate_pan))
    rate_tilt = max(-slew_tilt, min(slew_tilt, rate_tilt))

    # Accumulate commanded position.
    state.pi_integral_pan += rate_pan * dt
    state.pi_integral_tilt += rate_tilt * dt

    # Integral decay inside the deadband: bleed the accumulator toward
    # zero so a long quiescent period doesn't leave residual offset.
    if pan_abs < config.deadband_deg and config.integral_decay_rate > 0 and dt > 0:
        decay = max(0.0, 1.0 - config.integral_decay_rate * dt)
        state.pi_integral_pan *= decay
    if tilt_abs < config.deadband_deg and config.integral_decay_rate > 0 and dt > 0:
        decay = max(0.0, 1.0 - config.integral_decay_rate * dt)
        state.pi_integral_tilt *= decay

    # Clamp to mechanical limits.
    state.pi_integral_pan = max(
        -config.pan_limit_deg, min(config.pan_limit_deg, state.pi_integral_pan)
    )
    state.pi_integral_tilt = max(
        -config.tilt_limit_deg, min(config.tilt_limit_deg, state.pi_integral_tilt)
    )

    return state.pi_integral_pan, state.pi_integral_tilt


def compute_auto_command(
    *,
    message: TrackingMessage,
    state: AutoControllerState,
    config: AutoControllerConfig,
) -> tuple[float, float]:
    """Single-tick auto-mode controller step.

    Returns ``(pan_deg, tilt_deg)`` — the commanded servo position in
    degrees, before laser-boresight offsets and final clamping.

    Mutates ``state`` in place: integrator advances, derivative history
    rolls forward, and ``last_encode_ts_ns`` is updated. Behaviour:

    * ``target_acquired=True`` → run the full PD controller.
    * ``state_source == "center"`` → reset state, command (0, 0).
    * Otherwise (prediction/hold) → freeze at the current commanded
      position without integrating new error.
    """
    dt = _compute_dt_seconds(message, state)
    state.last_encode_ts_ns = message.timestamp_ns

    pan_err_deg, tilt_err_deg = _compute_lead_error_deg(message, config)

    if message.target_acquired:
        return _track_error_command(message, state, config, pan_err_deg, tilt_err_deg, dt)

    if message.state_source == "center":
        state.reset()
        return 0.0, 0.0

    # Prediction / hold: maintain current commanded position.
    return state.pi_integral_pan, state.pi_integral_tilt
