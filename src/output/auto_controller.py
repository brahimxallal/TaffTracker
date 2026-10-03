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
from math import degrees, sin

from src.config import GimbalConfig, SearchConfig, ServoControlConfig
from src.output.command_smoother import CommandSlewLimiter
from src.output.velocity_smoother import VelocitySmoother, VelocitySmootherConfig
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
    command_slew_limit_dps: float
    velocity_smoother: VelocitySmootherConfig
    micro_relock_min_loss_s: float
    micro_relock_window_s: float
    micro_relock_gain: float
    micro_relock_ramp_frames: int
    search: SearchConfig

    @classmethod
    def from_configs(
        cls,
        gimbal_config: GimbalConfig,
        servo_control_config: ServoControlConfig,
        search_config: SearchConfig | None = None,
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
            command_slew_limit_dps=servo_control_config.auto_command_slew_limit_dps,
            velocity_smoother=VelocitySmootherConfig(
                enabled=servo_control_config.velocity_smoother_enabled,
                alpha=servo_control_config.velocity_smoother_alpha,
                deadband_dps=servo_control_config.velocity_smoother_deadband_dps,
                slew_dps_per_s=servo_control_config.velocity_smoother_slew_dps_per_s,
            ),
            micro_relock_window_s=servo_control_config.micro_relock_window_s,
            micro_relock_min_loss_s=servo_control_config.micro_relock_min_loss_s,
            micro_relock_gain=servo_control_config.micro_relock_gain,
            micro_relock_ramp_frames=servo_control_config.micro_relock_ramp_frames,
            search=search_config or SearchConfig(),
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
    velocity_smoother_pan: VelocitySmoother = field(default_factory=VelocitySmoother)
    velocity_smoother_tilt: VelocitySmoother = field(default_factory=VelocitySmoother)
    command_limiter: CommandSlewLimiter = field(default_factory=CommandSlewLimiter)
    last_known_pan_deg: float | None = None
    last_known_tilt_deg: float | None = None
    last_known_pan_velocity_dps: float = 0.0
    last_known_tilt_velocity_dps: float = 0.0
    last_known_confidence: float = 0.0
    last_known_timestamp_ns: int | None = None
    search_started_timestamp_ns: int | None = None
    search_active: bool = False
    search_pan_velocity_dps: float = 0.0
    search_tilt_velocity_dps: float = 0.0
    last_target_acquired: bool = False
    last_track_id: int | None = None
    lost_track_id: int | None = None
    lost_started_timestamp_ns: int | None = None
    micro_relock_frames_remaining: int = 0
    micro_relock_total_frames: int = 0

    def reset(self) -> None:
        """Zero all integrators + derivative history."""
        self.pi_integral_pan = 0.0
        self.pi_integral_tilt = 0.0
        self.prev_err_pan = 0.0
        self.prev_err_tilt = 0.0
        self.prev_d_pan = 0.0
        self.prev_d_tilt = 0.0
        self.last_encode_ts_ns = None
        self.velocity_smoother_pan.reset()
        self.velocity_smoother_tilt.reset()
        self.command_limiter.reset()
        self.last_target_acquired = False
        self.last_track_id = None
        self.lost_track_id = None
        self.lost_started_timestamp_ns = None
        self.micro_relock_frames_remaining = 0
        self.micro_relock_total_frames = 0
        self.clear_search_memory()

    def clear_search_memory(self) -> None:
        """Forget target-lost search anchors and velocity telemetry."""
        self.last_known_pan_deg = None
        self.last_known_tilt_deg = None
        self.last_known_pan_velocity_dps = 0.0
        self.last_known_tilt_velocity_dps = 0.0
        self.last_known_confidence = 0.0
        self.last_known_timestamp_ns = None
        self.search_started_timestamp_ns = None
        self.search_active = False
        self.search_pan_velocity_dps = 0.0
        self.search_tilt_velocity_dps = 0.0

    def stop_search(self) -> None:
        """End active sweep while keeping last-known target memory."""
        self.search_started_timestamp_ns = None
        self.search_active = False
        self.search_pan_velocity_dps = 0.0
        self.search_tilt_velocity_dps = 0.0


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
    state: AutoControllerState,
    config: AutoControllerConfig,
    dt: float,
) -> tuple[float, float]:
    """Error angle including velocity-based predictive lead, sign-corrected.

    Lead is only applied when the target is moving fast enough (>30 dps)
    to be worth predicting; static targets get no lead so we don't
    amplify low-speed noise into command jitter.

    When ``config.velocity_smoother.enabled`` is True, the velocity
    components are passed through per-axis :class:`VelocitySmoother`
    instances on ``state`` before being multiplied by the lead time.
    With smoothing disabled (the default), the smoother is a pure
    pass-through so behaviour is bit-identical to pre-smoother builds.
    """
    angles_rad = message.servo_angles or message.filtered_angles or (0.0, 0.0)
    vel_rad = message.servo_angular_velocity or message.angular_velocity or (0.0, 0.0)
    pan_vel_dps = state.velocity_smoother_pan.smooth(
        degrees(vel_rad[0]), dt, config.velocity_smoother
    )
    tilt_vel_dps = state.velocity_smoother_tilt.smooth(
        degrees(vel_rad[1]), dt, config.velocity_smoother
    )
    speed_dps = (pan_vel_dps**2 + tilt_vel_dps**2) ** 0.5
    lead = config.predictive_lead_s if speed_dps > 30.0 else 0.0
    pan_err_deg = (degrees(angles_rad[0]) + pan_vel_dps * lead) * config.pan_sign
    tilt_err_deg = (degrees(angles_rad[1]) + tilt_vel_dps * lead) * config.tilt_sign
    return pan_err_deg, tilt_err_deg


def _prepare_relock_state(
    *,
    message: TrackingMessage,
    state: AutoControllerState,
    config: AutoControllerConfig,
    pan_err_deg: float,
    tilt_err_deg: float,
) -> bool:
    """Track brief same-ID losses and arm a short reacquire gain ramp."""
    if not message.target_acquired:
        if state.last_target_acquired or state.lost_started_timestamp_ns is None:
            state.lost_started_timestamp_ns = message.timestamp_ns
            state.lost_track_id = message.track_id or state.last_track_id
        elif message.track_id is not None:
            state.lost_track_id = message.track_id
        state.last_target_acquired = False
        if message.track_id is not None:
            state.last_track_id = message.track_id
        return False

    was_micro_relock = False
    if (
        not state.last_target_acquired
        and message.state_source == "measurement"
        and message.track_id is not None
        and message.track_id == state.lost_track_id
        and state.lost_started_timestamp_ns is not None
    ):
        lost_s = (message.timestamp_ns - state.lost_started_timestamp_ns) / 1e9
        if (
            max(0.0, config.micro_relock_min_loss_s)
            <= lost_s
            <= max(0.0, config.micro_relock_window_s)
        ):
            frames = max(0, int(config.micro_relock_ramp_frames))
            state.micro_relock_frames_remaining = frames
            state.micro_relock_total_frames = frames
            state.prev_err_pan = pan_err_deg
            state.prev_err_tilt = tilt_err_deg
            state.prev_d_pan = 0.0
            state.prev_d_tilt = 0.0
            state.velocity_smoother_pan.reset()
            state.velocity_smoother_tilt.reset()
            was_micro_relock = frames > 0

    state.last_target_acquired = True
    state.last_track_id = message.track_id
    state.lost_track_id = None
    state.lost_started_timestamp_ns = None
    return was_micro_relock


def _consume_micro_relock_gain(state: AutoControllerState, config: AutoControllerConfig) -> float:
    if state.micro_relock_frames_remaining <= 0:
        return 1.0

    total = max(1, state.micro_relock_total_frames)
    elapsed = total - state.micro_relock_frames_remaining
    state.micro_relock_frames_remaining -= 1
    floor = max(0.0, min(1.0, config.micro_relock_gain))
    ramp = elapsed / total
    return floor + (1.0 - floor) * ramp


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

    micro_relock_gain = _consume_micro_relock_gain(state, config)
    rate_pan *= micro_relock_gain
    rate_tilt *= micro_relock_gain

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


def _slew_toward(current: float, desired: float, max_delta: float) -> float:
    if desired > current + max_delta:
        return current + max_delta
    if desired < current - max_delta:
        return current - max_delta
    return desired


def _apply_command_slew_limit(
    *,
    command: tuple[float, float],
    state: AutoControllerState,
    config: AutoControllerConfig,
    dt: float,
) -> tuple[float, float]:
    limited = state.command_limiter.limit(
        command[0],
        command[1],
        dt_s=dt,
        limit_dps=config.command_slew_limit_dps,
    )
    state.pi_integral_pan, state.pi_integral_tilt = limited
    return limited


def _prediction_limited_anchor(
    *,
    state: AutoControllerState,
    config: AutoControllerConfig,
    elapsed_s: float,
) -> tuple[float, float] | None:
    if state.last_known_pan_deg is None or state.last_known_tilt_deg is None:
        return None
    horizon_s = min(max(0.0, elapsed_s), max(0.0, config.search.prediction_horizon_s))
    pan = state.last_known_pan_deg + state.last_known_pan_velocity_dps * horizon_s
    tilt = state.last_known_tilt_deg + state.last_known_tilt_velocity_dps * horizon_s
    return (
        max(-config.pan_limit_deg, min(config.pan_limit_deg, pan)),
        max(-config.tilt_limit_deg, min(config.tilt_limit_deg, tilt)),
    )


def _search_command(
    *,
    message: TrackingMessage,
    state: AutoControllerState,
    config: AutoControllerConfig,
    dt: float,
) -> tuple[float, float] | None:
    search = config.search
    if not search.enabled or state.last_known_timestamp_ns is None:
        state.stop_search()
        return None

    elapsed_s = max(0.0, (message.timestamp_ns - state.last_known_timestamp_ns) / 1e9)
    if elapsed_s < search.start_after_s:
        state.stop_search()
        return state.pi_integral_pan, state.pi_integral_tilt

    if search.timeout_s > 0.0 and elapsed_s >= search.timeout_s:
        state.stop_search()
        if not search.return_home:
            return state.pi_integral_pan, state.pi_integral_tilt
        max_delta = max(0.0, search.scan_speed_dps) * max(dt, 0.0)
        next_pan = _slew_toward(state.pi_integral_pan, 0.0, max_delta)
        next_tilt = _slew_toward(state.pi_integral_tilt, 0.0, max_delta)
        state.search_pan_velocity_dps = (next_pan - state.pi_integral_pan) / dt if dt > 0 else 0.0
        state.search_tilt_velocity_dps = (
            (next_tilt - state.pi_integral_tilt) / dt if dt > 0 else 0.0
        )
        state.pi_integral_pan = next_pan
        state.pi_integral_tilt = next_tilt
        return next_pan, next_tilt

    anchor = _prediction_limited_anchor(state=state, config=config, elapsed_s=elapsed_s)
    if anchor is None:
        state.stop_search()
        return None

    if state.search_started_timestamp_ns is None:
        state.search_started_timestamp_ns = message.timestamp_ns
    state.search_active = True

    search_elapsed_s = max(
        0.0, (message.timestamp_ns - state.search_started_timestamp_ns) / 1e9
    )
    radius = min(
        max(0.0, search.max_radius_deg),
        max(0.0, search.initial_radius_deg)
        + max(0.0, search.expansion_rate_dps) * search_elapsed_s,
    )
    if radius <= 1e-6:
        desired_pan, desired_tilt = anchor
    else:
        omega = max(0.0, search.scan_speed_dps) / radius
        phase = omega * search_elapsed_s
        desired_pan = anchor[0] + radius * sin(phase)
        desired_tilt = anchor[1] + radius * search.tilt_amplitude_ratio * sin(phase * 0.5)

    desired_pan = max(-config.pan_limit_deg, min(config.pan_limit_deg, desired_pan))
    desired_tilt = max(-config.tilt_limit_deg, min(config.tilt_limit_deg, desired_tilt))

    max_delta = max(0.0, search.scan_speed_dps) * max(dt, 0.0)
    prev_pan = state.pi_integral_pan
    prev_tilt = state.pi_integral_tilt
    next_pan = _slew_toward(prev_pan, desired_pan, max_delta)
    next_tilt = _slew_toward(prev_tilt, desired_tilt, max_delta)
    state.search_pan_velocity_dps = (next_pan - prev_pan) / dt if dt > 0 else 0.0
    state.search_tilt_velocity_dps = (next_tilt - prev_tilt) / dt if dt > 0 else 0.0
    state.pi_integral_pan = next_pan
    state.pi_integral_tilt = next_tilt
    return next_pan, next_tilt


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
    * lost target + search enabled → sweep smoothly around the predicted
      last-known absolute target angle, expanding over time.
    * ``state_source == "center"`` without search → reset state, command
      (0, 0).
    * Otherwise (prediction/hold) → freeze at the current commanded
      position without integrating new error.
    """
    dt = _compute_dt_seconds(message, state)
    state.last_encode_ts_ns = message.timestamp_ns

    pan_err_deg, tilt_err_deg = _compute_lead_error_deg(message, state, config, dt)
    _prepare_relock_state(
        message=message,
        state=state,
        config=config,
        pan_err_deg=pan_err_deg,
        tilt_err_deg=tilt_err_deg,
    )

    if message.target_acquired:
        if state.search_active:
            state.prev_err_pan = pan_err_deg
            state.prev_err_tilt = tilt_err_deg
            state.prev_d_pan = 0.0
            state.prev_d_tilt = 0.0
            state.velocity_smoother_pan.reset()
            state.velocity_smoother_tilt.reset()
        state.stop_search()
        command = _track_error_command(message, state, config, pan_err_deg, tilt_err_deg, dt)
        command = _apply_command_slew_limit(
            command=command,
            state=state,
            config=config,
            dt=dt,
        )
        vel_rad = message.servo_angular_velocity or message.angular_velocity or (0.0, 0.0)
        state.last_known_pan_deg = max(
            -config.pan_limit_deg,
            min(config.pan_limit_deg, command[0] + pan_err_deg),
        )
        state.last_known_tilt_deg = max(
            -config.tilt_limit_deg,
            min(config.tilt_limit_deg, command[1] + tilt_err_deg),
        )
        state.last_known_pan_velocity_dps = degrees(vel_rad[0]) * config.pan_sign
        state.last_known_tilt_velocity_dps = degrees(vel_rad[1]) * config.tilt_sign
        state.last_known_confidence = message.confidence
        state.last_known_timestamp_ns = message.timestamp_ns
        return command

    prev_command = (
        state.command_limiter.value
        if state.command_limiter.initialized
        else (state.pi_integral_pan, state.pi_integral_tilt)
    )
    search_command = _search_command(message=message, state=state, config=config, dt=dt)
    if search_command is not None:
        command = _apply_command_slew_limit(
            command=search_command,
            state=state,
            config=config,
            dt=dt,
        )
        if dt > 0:
            state.search_pan_velocity_dps = (command[0] - prev_command[0]) / dt
            state.search_tilt_velocity_dps = (command[1] - prev_command[1]) / dt
        return command

    if message.state_source == "center":
        state.reset()
        state.command_limiter.seed(0.0, 0.0)
        return 0.0, 0.0

    # Prediction / hold: maintain current commanded position.
    return state.pi_integral_pan, state.pi_integral_tilt
