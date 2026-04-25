"""Tests for :mod:`src.output.velocity_smoother`.

The defining contract is the safe-by-default behavior: with
``enabled=False`` the smoother is a pure pass-through so wiring it into
the controller cannot change live behavior until the user opts in.
"""

from __future__ import annotations

from math import radians

import pytest

from src.config import GimbalConfig, ServoControlConfig
from src.output.auto_controller import (
    AutoControllerConfig,
    AutoControllerState,
    _compute_lead_error_deg,
    compute_auto_command,
)
from src.output.velocity_smoother import VelocitySmoother, VelocitySmootherConfig
from src.shared.types import TrackingMessage

_TS_BASE = 1_000_000_000


def _make_message(
    *,
    servo_angles: tuple[float, float] = (0.0, 0.0),
    servo_angular_velocity: tuple[float, float] = (0.0, 0.0),
    timestamp_ns: int = _TS_BASE,
    target_acquired: bool = True,
    state_source: str = "measurement",
) -> TrackingMessage:
    return TrackingMessage(
        frame_id=1,
        timestamp_ns=timestamp_ns,
        target_kind="human",
        target_acquired=target_acquired,
        state_source=state_source,  # type: ignore[arg-type]
        track_id=1,
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
        fps=60.0,
    )


@pytest.mark.unit
def test_default_config_is_disabled() -> None:
    cfg = VelocitySmootherConfig()
    assert cfg.enabled is False


@pytest.mark.unit
def test_disabled_smoother_is_pure_passthrough() -> None:
    s = VelocitySmoother()
    cfg = VelocitySmootherConfig()  # enabled=False
    assert s.smooth(0.0, 0.01, cfg) == 0.0
    assert s.smooth(123.4, 0.01, cfg) == 123.4
    assert s.smooth(-50.0, 0.01, cfg) == -50.0
    # Tiny values still pass through (no deadband when disabled).
    assert s.smooth(0.001, 0.01, cfg) == 0.001


@pytest.mark.unit
def test_enabled_smoother_warm_starts_on_first_sample() -> None:
    s = VelocitySmoother()
    cfg = VelocitySmootherConfig(enabled=True, alpha=0.4, deadband_dps=0.0)
    # First sample is the seed → no smoothing lag.
    assert s.smooth(100.0, 0.01, cfg) == 100.0


@pytest.mark.unit
def test_enabled_smoother_applies_ema() -> None:
    s = VelocitySmoother()
    cfg = VelocitySmootherConfig(enabled=True, alpha=0.5, deadband_dps=0.0)
    s.smooth(100.0, 0.01, cfg)  # seed
    # ema = 0.5 * 0 + 0.5 * 100 = 50
    assert s.smooth(0.0, 0.01, cfg) == pytest.approx(50.0)


@pytest.mark.unit
def test_enabled_smoother_deadband_snaps_to_zero() -> None:
    s = VelocitySmoother()
    cfg = VelocitySmootherConfig(enabled=True, alpha=1.0, deadband_dps=10.0)
    # alpha=1.0 means the EMA echoes the input, so deadband is the only filter.
    assert s.smooth(5.0, 0.01, cfg) == 0.0
    assert s.smooth(-9.99, 0.01, cfg) == 0.0
    assert s.smooth(10.0, 0.01, cfg) == 10.0
    assert s.smooth(-15.0, 0.01, cfg) == -15.0


@pytest.mark.unit
def test_enabled_smoother_slew_cap_limits_step() -> None:
    s = VelocitySmoother()
    cfg = VelocitySmootherConfig(enabled=True, alpha=1.0, deadband_dps=0.0, slew_dps_per_s=100.0)
    s.smooth(0.0, 0.01, cfg)  # seed at 0
    assert s.smooth(50.0, 0.01, cfg) == pytest.approx(1.0)  # capped from 50 → 1
    assert s.smooth(50.0, 0.01, cfg) == pytest.approx(2.0)  # +1 again
    # Negative direction is also capped.
    s2 = VelocitySmoother()
    s2.smooth(0.0, 0.01, cfg)
    assert s2.smooth(-50.0, 0.01, cfg) == pytest.approx(-1.0)


@pytest.mark.unit
def test_reset_clears_state() -> None:
    s = VelocitySmoother()
    cfg = VelocitySmootherConfig(enabled=True, alpha=0.5, deadband_dps=0.0)
    s.smooth(100.0, 0.01, cfg)
    s.smooth(80.0, 0.01, cfg)
    s.reset()
    assert s.value == 0.0
    # After reset, next sample warm-starts again (no lag from previous run).
    assert s.smooth(50.0, 0.01, cfg) == 50.0


@pytest.mark.unit
def test_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError):
        VelocitySmootherConfig(enabled=True, alpha=-0.1)
    with pytest.raises(ValueError):
        VelocitySmootherConfig(enabled=True, alpha=1.1)


@pytest.mark.unit
def test_invalid_deadband_raises() -> None:
    with pytest.raises(ValueError):
        VelocitySmootherConfig(enabled=True, deadband_dps=-1.0)


@pytest.mark.unit
def test_invalid_slew_raises() -> None:
    with pytest.raises(ValueError):
        VelocitySmootherConfig(enabled=True, slew_dps_per_s=-1.0)


@pytest.mark.unit
def test_servo_control_config_defaults_keep_smoother_disabled() -> None:
    """The wiring contract: the smoother defaults must keep behaviour
    bit-identical for fresh installs."""
    cfg = ServoControlConfig()
    assert cfg.velocity_smoother_enabled is False


@pytest.mark.unit
def test_auto_controller_config_carries_smoother_settings() -> None:
    cfg = AutoControllerConfig.from_configs(
        GimbalConfig(),
        ServoControlConfig(
            velocity_smoother_enabled=True,
            velocity_smoother_alpha=0.6,
            velocity_smoother_deadband_dps=12.5,
            velocity_smoother_slew_dps_per_s=200.0,
        ),
    )
    assert cfg.velocity_smoother.enabled is True
    assert cfg.velocity_smoother.alpha == 0.6
    assert cfg.velocity_smoother.deadband_dps == 12.5
    assert cfg.velocity_smoother.slew_dps_per_s == 200.0


@pytest.mark.unit
def test_compute_auto_command_unchanged_when_smoother_disabled() -> None:
    """Smoother-disabled builds must produce the same commanded output as
    a baseline build that doesn't know about the smoother at all."""
    # Baseline: smoother disabled, lead disabled → integral grows from
    # angle error only.
    gimbal = GimbalConfig(predictive_lead_s=0.0)
    servo_off = ServoControlConfig(velocity_smoother_enabled=False)
    cfg_off = AutoControllerConfig.from_configs(gimbal, servo_off)
    state_off = AutoControllerState()

    msg = _make_message(
        servo_angles=(radians(10.0), 0.0),
        servo_angular_velocity=(radians(60.0), 0.0),
    )
    pan_off, _ = compute_auto_command(message=msg, state=state_off, config=cfg_off)

    # Equivalent path with explicit defaults: still disabled, must match.
    servo_off2 = ServoControlConfig()
    cfg_off2 = AutoControllerConfig.from_configs(gimbal, servo_off2)
    state_off2 = AutoControllerState()
    pan_off2, _ = compute_auto_command(message=msg, state=state_off2, config=cfg_off2)

    assert pan_off == pan_off2


@pytest.mark.unit
def test_lead_error_smoother_attenuates_noise_spike() -> None:
    """With lead enabled and smoother enabled, a single noisy velocity
    sample must contribute less to the computed lead error than the raw
    value would have. This is tested at the lead-error layer directly so
    the controller's slew cap on the integrator doesn't mask the win."""
    gimbal = GimbalConfig(predictive_lead_s=0.1)

    # Raw path: smoother disabled.
    cfg_raw = AutoControllerConfig.from_configs(
        gimbal, ServoControlConfig(velocity_smoother_enabled=False)
    )
    state_raw = AutoControllerState()

    # Smoothed path: enabled with strong attenuation (alpha=0.3).
    cfg_smooth = AutoControllerConfig.from_configs(
        gimbal,
        ServoControlConfig(
            velocity_smoother_enabled=True,
            velocity_smoother_alpha=0.3,
            velocity_smoother_deadband_dps=0.0,
        ),
    )
    state_smooth = AutoControllerState()

    # Seed both smoothers with a moderate baseline.
    baseline = _make_message(
        servo_angles=(radians(5.0), 0.0),
        servo_angular_velocity=(radians(40.0), 0.0),
    )
    _compute_lead_error_deg(baseline, state_raw, cfg_raw, dt=1 / 60)
    _compute_lead_error_deg(baseline, state_smooth, cfg_smooth, dt=1 / 60)

    spike = _make_message(
        servo_angles=(radians(5.0), 0.0),
        servo_angular_velocity=(radians(400.0), 0.0),  # 10x noise spike
    )
    raw_pan_err, _ = _compute_lead_error_deg(spike, state_raw, cfg_raw, dt=1 / 60)
    smooth_pan_err, _ = _compute_lead_error_deg(spike, state_smooth, cfg_smooth, dt=1 / 60)

    # Both pan_err contain the position term (5°) plus velocity*lead.
    # Raw: 5 + 400*0.1 = 45°. Smooth: 5 + (0.3*400 + 0.7*40)*0.1 = ~19.8°.
    assert abs(smooth_pan_err) < abs(raw_pan_err)


@pytest.mark.unit
def test_lead_error_disabled_smoother_is_bit_identical_to_raw_velocity() -> None:
    """With smoother disabled, the lead error must equal the result
    computed from the raw velocity — the smoother becomes invisible."""
    gimbal = GimbalConfig(predictive_lead_s=0.1, kp=1.0)
    cfg = AutoControllerConfig.from_configs(gimbal, ServoControlConfig())
    state = AutoControllerState()

    msg = _make_message(
        servo_angles=(radians(5.0), 0.0),
        servo_angular_velocity=(radians(60.0), 0.0),
    )
    pan_err, _ = _compute_lead_error_deg(msg, state, cfg, dt=1 / 60)
    # Position 5° + velocity 60 dps × 0.1 s lead = 11°, then sign=+1.
    assert pan_err == pytest.approx(11.0, abs=1e-9)


@pytest.mark.unit
def test_state_reset_clears_smoother_history() -> None:
    state = AutoControllerState()
    cfg = VelocitySmootherConfig(enabled=True, alpha=0.5, deadband_dps=0.0)
    state.velocity_smoother_pan.smooth(100.0, 0.01, cfg)
    state.velocity_smoother_pan.smooth(80.0, 0.01, cfg)
    assert state.velocity_smoother_pan.value != 0.0
    state.reset()
    assert state.velocity_smoother_pan.value == 0.0
