"""Unit tests for :mod:`src.output.auto_controller`.

These cover the pure-function extraction directly, independently of the
OutputProcess multiprocessing shell. The class-level integration tests
in ``tests/test_control_stability.py`` and ``tests/test_encode_packet.py``
still exercise the wired-up path end-to-end.
"""

from __future__ import annotations

import math
from math import radians

import pytest

import src.output.auto_controller as auto_controller
from src.config import GimbalConfig, ServoControlConfig
from src.output.auto_controller import (
    AutoControllerConfig,
    AutoControllerState,
    _track_error_command,
    compute_auto_command,
)
from src.output.latency_compensation import LatencyCompensationConfig
from src.shared.types import TrackingMessage

_TS_BASE = 1_000_000_000  # arbitrary 1s mark


def _make_message(
    *,
    target_acquired: bool = True,
    state_source: str = "measurement",
    servo_angles: tuple[float, float] = (0.0, 0.0),
    servo_angular_velocity: tuple[float, float] = (0.0, 0.0),
    timestamp_ns: int = _TS_BASE,
    fps: float = 60.0,
    track_id: int | None = 1,
) -> TrackingMessage:
    return TrackingMessage(
        frame_id=1,
        timestamp_ns=timestamp_ns,
        target_kind="human",
        target_acquired=target_acquired,
        state_source=state_source,  # type: ignore[arg-type]
        track_id=track_id,
        confidence=0.9 if target_acquired else 0.0,
        raw_pixel=None,
        filtered_pixel=None,
        raw_angles=None,
        filtered_angles=servo_angles,
        servo_angles=servo_angles,
        servo_angular_velocity=servo_angular_velocity,
        inference_ms=5.0,
        tracking_ms=1.0,
        total_latency_ms=6.0,
        fps=fps,
    )


def _config(**overrides) -> AutoControllerConfig:
    servo_overrides = overrides.pop("servo_control", {})
    base = GimbalConfig(
        kp=1.5,
        kd=0.0,
        deadband_deg=1.0,
        slew_limit_dps=50.0,
        integral_decay_rate=2.0,
        tilt_scale=1.0,
        gain_schedule_threshold_deg=3.0,
        predictive_lead_s=0.05,
        velocity_feedforward_gain=0.0,
    )
    for key, value in overrides.items():
        base = base.__class__(**{**base.__dict__, key: value})
    servo = ServoControlConfig(**{**ServoControlConfig().__dict__, **servo_overrides})
    return AutoControllerConfig.from_configs(base, servo)


@pytest.mark.unit
def test_state_reset_zeros_all_fields() -> None:
    state = AutoControllerState(
        pi_integral_pan=4.0,
        pi_integral_tilt=-2.0,
        prev_err_pan=1.0,
        prev_err_tilt=2.0,
        prev_d_pan=3.0,
        prev_d_tilt=4.0,
        last_encode_ts_ns=12345,
    )
    state.reset()
    assert state.pi_integral_pan == 0.0
    assert state.pi_integral_tilt == 0.0
    assert state.prev_err_pan == 0.0
    assert state.prev_err_tilt == 0.0
    assert state.prev_d_pan == 0.0
    assert state.prev_d_tilt == 0.0
    assert state.last_encode_ts_ns is None


@pytest.mark.unit
def test_config_inverts_signs_when_gimbal_inverted() -> None:
    cfg_normal = AutoControllerConfig.from_configs(GimbalConfig(), ServoControlConfig())
    cfg_pan_inverted = AutoControllerConfig.from_configs(
        GimbalConfig(invert_pan=True), ServoControlConfig()
    )
    cfg_tilt_inverted = AutoControllerConfig.from_configs(
        GimbalConfig(invert_tilt=True), ServoControlConfig()
    )
    assert cfg_normal.pan_sign == 1.0
    assert cfg_normal.tilt_sign == 1.0
    assert cfg_pan_inverted.pan_sign == -1.0
    assert cfg_pan_inverted.tilt_sign == 1.0
    assert cfg_tilt_inverted.pan_sign == 1.0
    assert cfg_tilt_inverted.tilt_sign == -1.0


@pytest.mark.unit
def test_acquired_with_zero_error_holds_integral_at_zero() -> None:
    state = AutoControllerState()
    cfg = _config()
    msg = _make_message(servo_angles=(0.0, 0.0), servo_angular_velocity=(0.0, 0.0))
    pan, tilt = compute_auto_command(message=msg, state=state, config=cfg)
    assert pan == 0.0
    assert tilt == 0.0
    assert state.last_encode_ts_ns == _TS_BASE


@pytest.mark.unit
def test_acquired_with_far_error_drives_integral_in_correct_direction() -> None:
    """Positive (right-of-center) servo angle → positive commanded pan."""
    state = AutoControllerState()
    cfg = _config()
    # 10 deg error in pan, 0 in tilt
    msg = _make_message(servo_angles=(radians(10.0), 0.0))
    pan, _ = compute_auto_command(message=msg, state=state, config=cfg)
    assert pan > 0.0
    assert state.pi_integral_pan > 0.0


@pytest.mark.unit
def test_acquired_inside_deadband_does_not_grow_integral() -> None:
    state = AutoControllerState()
    cfg = _config(deadband_deg=2.0)
    # 0.5 deg error well inside the 2-deg deadband
    msg = _make_message(servo_angles=(radians(0.5), 0.0))
    compute_auto_command(message=msg, state=state, config=cfg)
    assert state.pi_integral_pan == 0.0


@pytest.mark.unit
def test_state_source_center_resets_integral() -> None:
    state = AutoControllerState(pi_integral_pan=5.0, pi_integral_tilt=-3.0)
    cfg = _config()
    msg = _make_message(
        target_acquired=False,
        state_source="center",
        servo_angles=(0.0, 0.0),
    )
    pan, tilt = compute_auto_command(message=msg, state=state, config=cfg)
    assert pan == 0.0
    assert tilt == 0.0
    assert state.pi_integral_pan == 0.0
    assert state.pi_integral_tilt == 0.0


@pytest.mark.unit
def test_prediction_state_holds_commanded_position() -> None:
    """When the tracker says 'lost' or 'prediction', the controller freezes."""
    state = AutoControllerState(pi_integral_pan=7.5, pi_integral_tilt=2.5)
    cfg = _config()
    msg = _make_message(
        target_acquired=False,
        state_source="prediction",
        servo_angles=(radians(20.0), 0.0),  # would otherwise drive a big move
    )
    pan, tilt = compute_auto_command(message=msg, state=state, config=cfg)
    assert pan == 7.5
    assert tilt == 2.5
    # Integral state must NOT advance during prediction.
    assert state.pi_integral_pan == 7.5
    assert state.pi_integral_tilt == 2.5


@pytest.mark.unit
def test_clamps_to_mechanical_limits() -> None:
    state = AutoControllerState()
    cfg = _config(pan_limit_deg=5.0)
    # Drive a huge error for many ticks; integral must saturate at +5°.
    for tick in range(50):
        msg = _make_message(
            servo_angles=(radians(45.0), 0.0),
            timestamp_ns=_TS_BASE + tick * 16_666_667,
        )
        compute_auto_command(message=msg, state=state, config=cfg)
    assert state.pi_integral_pan == pytest.approx(5.0, abs=1e-9)


@pytest.mark.unit
def test_inverted_pan_drives_integral_negative() -> None:
    state = AutoControllerState()
    cfg = AutoControllerConfig.from_configs(
        GimbalConfig(invert_pan=True, kp=1.5), ServoControlConfig()
    )
    msg = _make_message(servo_angles=(radians(10.0), 0.0))
    pan, _ = compute_auto_command(message=msg, state=state, config=cfg)
    assert pan < 0.0


@pytest.mark.unit
def test_dt_falls_back_to_fps_on_first_tick() -> None:
    """First-tick dt comes from message.fps because state.last_encode_ts_ns is None."""
    state = AutoControllerState()
    cfg = _config()
    msg = _make_message(servo_angles=(radians(10.0), 0.0), fps=60.0)
    compute_auto_command(message=msg, state=state, config=cfg)
    # After one tick at fps=60, the integral should reflect roughly 1/60 s of
    # rate. With kp=1.5 and 10° error, rate ≈ 15 dps, so integral ≈ 15/60 = 0.25.
    # We allow a wide tolerance because of slew clamping.
    assert state.pi_integral_pan > 0.1
    assert state.pi_integral_pan < 5.0


@pytest.mark.unit
def test_dt_uses_message_timestamp_delta_after_first_tick() -> None:
    state = AutoControllerState()
    cfg = _config()
    msg1 = _make_message(servo_angles=(radians(10.0), 0.0), timestamp_ns=_TS_BASE)
    compute_auto_command(message=msg1, state=state, config=cfg)
    integral_after_first = state.pi_integral_pan
    msg2 = _make_message(
        servo_angles=(radians(10.0), 0.0),
        timestamp_ns=_TS_BASE + 16_666_667,  # +1 frame @ 60fps
    )
    compute_auto_command(message=msg2, state=state, config=cfg)
    # Integral should grow further on the second tick.
    assert state.pi_integral_pan > integral_after_first


@pytest.mark.unit
def test_lead_compensation_adds_to_error_when_target_is_fast() -> None:
    """Predictive lead only kicks in for speeds > 30 dps."""
    cfg = _config(predictive_lead_s=0.1)
    # Fast target: 60 dps (well above 30 dps threshold)
    fast = AutoControllerState()
    msg_fast = _make_message(
        servo_angles=(radians(5.0), 0.0),
        servo_angular_velocity=(radians(60.0), 0.0),
    )
    compute_auto_command(message=msg_fast, state=fast, config=cfg)

    # Slow target: 5 dps (below threshold) — same starting error
    slow = AutoControllerState()
    msg_slow = _make_message(
        servo_angles=(radians(5.0), 0.0),
        servo_angular_velocity=(radians(5.0), 0.0),
    )
    compute_auto_command(message=msg_slow, state=slow, config=cfg)

    # The fast-target integral should be larger because lead added to the error.
    assert fast.pi_integral_pan > slow.pi_integral_pan


@pytest.mark.unit
def test_auto_command_slew_limit_caps_output_step() -> None:
    state = AutoControllerState()
    cfg = _config(
        kp=100.0,
        kd=0.0,
        deadband_deg=0.0,
        slew_limit_dps=1000.0,
        servo_control={"auto_command_slew_limit_dps": 30.0},
    )

    compute_auto_command(
        message=_make_message(servo_angles=(0.0, 0.0), timestamp_ns=_TS_BASE),
        state=state,
        config=cfg,
    )
    pan, _ = compute_auto_command(
        message=_make_message(
            servo_angles=(radians(45.0), 0.0),
            timestamp_ns=_TS_BASE + 100_000_000,
        ),
        state=state,
        config=cfg,
    )

    assert pan == pytest.approx(3.0)
    assert state.pi_integral_pan == pytest.approx(3.0)


@pytest.mark.unit
def test_same_track_micro_relock_softens_first_reacquired_command() -> None:
    base_cfg = dict(
        kp=5.0,
        kd=0.0,
        deadband_deg=0.0,
        slew_limit_dps=100.0,
    )
    base_servo = {
        "micro_relock_min_loss_s": 0.025,
        "micro_relock_window_s": 0.12,
        "micro_relock_ramp_frames": 4,
    }
    softened = AutoControllerState()
    normal = AutoControllerState()
    soft_cfg = _config(
        **base_cfg,
        servo_control={**base_servo, "micro_relock_gain": 0.25},
    )
    normal_cfg = _config(
        **base_cfg,
        servo_control={**base_servo, "micro_relock_gain": 1.0},
    )

    for state, cfg in ((softened, soft_cfg), (normal, normal_cfg)):
        compute_auto_command(
            message=_make_message(
                servo_angles=(0.0, 0.0),
                timestamp_ns=_TS_BASE,
                track_id=7,
            ),
            state=state,
            config=cfg,
        )
        compute_auto_command(
            message=_make_message(
                target_acquired=False,
                state_source="prediction",
                servo_angles=(0.0, 0.0),
                timestamp_ns=_TS_BASE + 30_000_000,
                track_id=7,
            ),
            state=state,
            config=cfg,
        )

    soft_pan, _ = compute_auto_command(
        message=_make_message(
            servo_angles=(radians(20.0), 0.0),
            timestamp_ns=_TS_BASE + 80_000_000,
            track_id=7,
        ),
        state=softened,
        config=soft_cfg,
    )
    normal_pan, _ = compute_auto_command(
        message=_make_message(
            servo_angles=(radians(20.0), 0.0),
            timestamp_ns=_TS_BASE + 80_000_000,
            track_id=7,
        ),
        state=normal,
        config=normal_cfg,
    )

    assert soft_pan < normal_pan * 0.6


@pytest.mark.unit
def test_long_relock_does_not_use_micro_loss_softening() -> None:
    cfg = _config(
        kp=5.0,
        kd=0.0,
        deadband_deg=0.0,
        slew_limit_dps=100.0,
        servo_control={
            "micro_relock_min_loss_s": 0.025,
            "micro_relock_window_s": 0.12,
            "micro_relock_gain": 0.25,
            "micro_relock_ramp_frames": 4,
        },
    )
    state = AutoControllerState()
    compute_auto_command(
        message=_make_message(servo_angles=(0.0, 0.0), timestamp_ns=_TS_BASE, track_id=7),
        state=state,
        config=cfg,
    )
    compute_auto_command(
        message=_make_message(
            target_acquired=False,
            state_source="prediction",
            servo_angles=(0.0, 0.0),
            timestamp_ns=_TS_BASE + 30_000_000,
            track_id=7,
        ),
        state=state,
        config=cfg,
    )
    pan, _ = compute_auto_command(
        message=_make_message(
            servo_angles=(radians(20.0), 0.0),
            timestamp_ns=_TS_BASE + 300_000_000,
            track_id=7,
        ),
        state=state,
        config=cfg,
    )

    assert pan == pytest.approx(10.0)


# --- Delay compensation (src/output/latency_compensation.py) ---------------------------------

_FRAME_NS = 16_666_667  # one frame at 60 fps


def _compensated_config(**overrides) -> AutoControllerConfig:
    servo_control = {"latency_comp_delay_s": 0.3, **overrides.pop("servo_control", {})}
    return _config(servo_control=servo_control, **overrides)


def _scripted_messages() -> list[TrackingMessage]:
    """200 frames: a sweeping target with a short prediction burst and a short loss."""
    messages = []
    for i in range(200):
        t = i / 60.0
        pan = 14.0 * math.sin(2 * math.pi * 0.4 * t) + 3.0
        tilt = 5.0 * math.cos(2 * math.pi * 0.25 * t)
        pan_rate = 14.0 * 2 * math.pi * 0.4 * math.cos(2 * math.pi * 0.4 * t)
        tilt_rate = -5.0 * 2 * math.pi * 0.25 * math.sin(2 * math.pi * 0.25 * t)
        timestamp_ns = _TS_BASE + i * _FRAME_NS
        if 80 <= i < 86:
            messages.append(
                _make_message(
                    target_acquired=False,
                    state_source="prediction",
                    servo_angles=(radians(pan), radians(tilt)),
                    servo_angular_velocity=(radians(pan_rate), radians(tilt_rate)),
                    timestamp_ns=timestamp_ns,
                )
            )
        elif 140 <= i < 143:
            messages.append(
                _make_message(
                    target_acquired=False,
                    state_source="lost",
                    servo_angles=(0.0, 0.0),
                    timestamp_ns=timestamp_ns,
                )
            )
        else:
            messages.append(
                _make_message(
                    servo_angles=(radians(pan), radians(tilt)),
                    servo_angular_velocity=(radians(pan_rate), radians(tilt_rate)),
                    timestamp_ns=timestamp_ns,
                )
            )
    return messages


# Commands of the controller as it was before delay compensation existed, for the sequence above
# (captured by running that sequence through the previous version of this module).
_PRE_COMPENSATION_COMMANDS = {
    10: (6.724492811606197, 1.1064113289763826),
    40: (29.73702847186002, 1.498741420152468),
    79: (37.98852592709729, 0.017930998486213476),
    85: (37.98852592709729, 0.017930998486213476),
    100: (31.537198032618406, -0.8778292618416784),
    139: (19.566260010465733, -3.7295153006302684),
    145: (19.98737332608763, -3.759643230868455),
    199: (56.43751544708069, -1.9634189980240329),
}


@pytest.mark.unit
def test_without_latency_compensation_the_controller_is_unchanged() -> None:
    """``latency_comp_delay_s = 0`` must reproduce the previous controller exactly."""
    config = AutoControllerConfig.from_configs(
        GimbalConfig(
            kp=3.5,
            kd=0.2,
            deadband_deg=0.65,
            slew_limit_dps=125.0,
            integral_decay_rate=0.2,
            tilt_scale=0.35,
            gain_schedule_threshold_deg=4.0,
            predictive_lead_s=0.035,
            velocity_feedforward_gain=0.3,
            kp_far=2.6,
            kp_near=1.0,
        ),
        ServoControlConfig(velocity_smoother_enabled=True, auto_command_slew_limit_dps=130.0),
    )
    assert config.latency.enabled is False
    state = AutoControllerState()

    commands = [
        compute_auto_command(message=message, state=state, config=config)
        for message in _scripted_messages()
    ]

    for frame, (pan, tilt) in _PRE_COMPENSATION_COMMANDS.items():
        assert commands[frame] == (pytest.approx(pan, rel=1e-9), pytest.approx(tilt, rel=1e-9))
    assert len(state.latency.history) == 0  # nothing is recorded while disabled


@pytest.mark.unit
def test_latency_compensation_config_is_built_from_servo_control() -> None:
    cfg = _config(
        servo_control={
            "latency_comp_delay_s": 0.25,
            "latency_comp_smith_gain": 0.9,
            "latency_comp_velocity_ff_gain": 0.8,
            "latency_comp_velocity_tau_s": 0.2,
            "latency_comp_velocity_deadband_dps": 2.5,
            "latency_comp_velocity_max_dps": 120.0,
        }
    )

    assert cfg.latency == LatencyCompensationConfig(
        delay_s=0.25,
        smith_gain=0.9,
        velocity_ff_gain=0.8,
        velocity_tau_s=0.2,
        velocity_deadband_dps=2.5,
        velocity_max_dps=120.0,
    )
    assert cfg.latency.enabled is True
    assert _config().latency.enabled is False


@pytest.mark.unit
def test_every_issued_command_is_recorded_for_delay_compensation() -> None:
    cfg = _compensated_config()
    state = AutoControllerState()

    acquired = compute_auto_command(
        message=_make_message(servo_angles=(radians(4.0), 0.0), timestamp_ns=_TS_BASE),
        state=state,
        config=cfg,
    )
    held = compute_auto_command(
        message=_make_message(
            target_acquired=False,
            state_source="prediction",
            timestamp_ns=_TS_BASE + _FRAME_NS,
        ),
        state=state,
        config=cfg,
    )

    assert held == acquired  # a prediction frame holds the position...
    assert len(state.latency.history) == 2  # ...and the hold is part of the history too
    assert state.latency.history.latest == held


@pytest.mark.unit
def test_recentering_restarts_the_command_history() -> None:
    cfg = _compensated_config()
    state = AutoControllerState()
    for frame in range(3):
        compute_auto_command(
            message=_make_message(
                servo_angles=(radians(4.0), 0.0), timestamp_ns=_TS_BASE + frame * _FRAME_NS
            ),
            state=state,
            config=cfg,
        )
    assert len(state.latency.history) == 3

    command = compute_auto_command(
        message=_make_message(
            target_acquired=False,
            state_source="center",
            timestamp_ns=_TS_BASE + 3 * _FRAME_NS,
        ),
        state=state,
        config=cfg,
    )

    assert command == (0.0, 0.0)
    assert len(state.latency.history) == 1
    assert state.latency.history.latest == (0.0, 0.0)


@pytest.mark.unit
def test_state_reset_clears_the_delay_compensation_state() -> None:
    cfg = _compensated_config()
    state = AutoControllerState()
    for frame in range(30):
        compute_auto_command(
            message=_make_message(
                servo_angles=(radians(2.0 + 0.2 * frame), 0.0),
                timestamp_ns=_TS_BASE + frame * _FRAME_NS,
            ),
            state=state,
            config=cfg,
        )
    assert len(state.latency.history) == 30
    assert state.latency.pan.velocity_dps != 0.0

    state.reset()

    assert len(state.latency.history) == 0
    assert state.latency.pan.velocity_dps == 0.0


@pytest.mark.unit
def test_compensation_gets_the_frame_time_pipeline_age_and_freshness(monkeypatch) -> None:
    calls: list[dict] = []
    real = auto_controller.compensate_error

    def spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(auto_controller, "compensate_error", spy)
    cfg = _compensated_config()
    state = AutoControllerState()

    compute_auto_command(
        message=_make_message(servo_angles=(radians(4.0), 0.0), timestamp_ns=_TS_BASE),
        state=state,
        config=cfg,
    )
    compute_auto_command(  # a planned Kalman-only frame while the lock is healthy
        message=_make_message(
            state_source="prediction",
            servo_angles=(radians(4.0), 0.0),
            timestamp_ns=_TS_BASE + _FRAME_NS,
        ),
        state=state,
        config=cfg,
    )

    assert [call["fresh_measurement"] for call in calls] == [True, False]
    assert calls[0]["t_s"] == pytest.approx(_TS_BASE / 1e9)
    assert calls[1]["t_s"] == pytest.approx((_TS_BASE + _FRAME_NS) / 1e9)
    assert calls[0]["pipeline_age_s"] == pytest.approx(0.006)  # total_latency_ms of the message
    assert calls[0]["pan_err_deg"] == pytest.approx(4.0)  # controller-frame error, in degrees


@pytest.mark.unit
def test_in_flight_motion_is_subtracted_from_the_error() -> None:
    """The same constant error keeps being 'seen' until the command's effect shows in the image."""

    def final_command(cfg: AutoControllerConfig) -> float:
        state = AutoControllerState()
        for frame in range(30):  # half a second of a constant 5 deg error
            compute_auto_command(
                message=_make_message(
                    servo_angles=(radians(5.0), 0.0), timestamp_ns=_TS_BASE + frame * _FRAME_NS
                ),
                state=state,
                config=cfg,
            )
        return state.pi_integral_pan

    plain = final_command(_config())
    compensated = final_command(
        _compensated_config(servo_control={"latency_comp_velocity_ff_gain": 0.0})
    )

    assert 0.0 < compensated < plain


@pytest.mark.unit
def test_legacy_lead_is_replaced_by_the_compensation_feedforward() -> None:
    """Relative velocity no longer leads the command once delay compensation is on."""
    cfg = _compensated_config(predictive_lead_s=0.1)
    fast = AutoControllerState()
    slow = AutoControllerState()

    compute_auto_command(
        message=_make_message(
            servo_angles=(radians(5.0), 0.0), servo_angular_velocity=(radians(60.0), 0.0)
        ),
        state=fast,
        config=cfg,
    )
    compute_auto_command(
        message=_make_message(
            servo_angles=(radians(5.0), 0.0), servo_angular_velocity=(radians(5.0), 0.0)
        ),
        state=slow,
        config=cfg,
    )

    assert fast.pi_integral_pan == pytest.approx(slow.pi_integral_pan)


@pytest.mark.unit
class TestFeedforwardInTrackErrorCommand:
    """``ff_dps`` is the absolute-velocity feedforward from delay compensation."""

    _DT = 1.0 / 60.0

    def _command(
        self,
        pan_err_deg: float,
        tilt_err_deg: float,
        ff_dps: tuple[float, float] | None,
        *,
        state: AutoControllerState | None = None,
        cfg: AutoControllerConfig | None = None,
    ) -> tuple[float, float]:
        return _track_error_command(
            _make_message(),
            state or AutoControllerState(),
            cfg or _config(),
            pan_err_deg,
            tilt_err_deg,
            self._DT,
            ff_dps=ff_dps,
        )

    def test_keeps_moving_the_command_inside_the_position_deadband(self) -> None:
        # 0.2 deg is well inside the 1 deg deadband: feedback alone would hold still.
        assert self._command(0.2, -0.2, None) == (0.0, 0.0)

        pan, tilt = self._command(0.2, -0.2, (10.0, -8.0))

        assert pan == pytest.approx(10.0 * self._DT)
        assert tilt == pytest.approx(-8.0 * self._DT)

    def test_rides_on_top_of_the_distance_scaled_slew_cap(self) -> None:
        # 1.2 deg of error: feedback rate 1.5 * 1.2 = 1.8 deg/s under a cap of 12.5 deg/s.
        pan, _ = self._command(1.2, 0.0, (30.0, 0.0))

        assert pan == pytest.approx((1.8 + 30.0) * self._DT)

    def test_is_limited_by_the_absolute_slew_limit(self) -> None:
        pan, _ = self._command(1.2, 0.0, (200.0, 0.0))

        assert pan == pytest.approx(50.0 * self._DT)  # slew_limit_dps of the test config

    def test_is_softened_with_the_micro_relock_ramp(self) -> None:
        cfg = _config(servo_control={"micro_relock_gain": 0.25, "micro_relock_ramp_frames": 4})
        state = AutoControllerState(micro_relock_frames_remaining=4, micro_relock_total_frames=4)

        pan, _ = self._command(0.2, 0.0, (20.0, 0.0), state=state, cfg=cfg)

        assert pan == pytest.approx(20.0 * 0.25 * self._DT)
