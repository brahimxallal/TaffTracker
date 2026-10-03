from __future__ import annotations

from dataclasses import replace
from math import radians

import pytest

from src.config import GimbalConfig, SearchConfig, ServoControlConfig
from src.output.auto_controller import (
    AutoControllerConfig,
    AutoControllerState,
    compute_auto_command,
)
from src.shared.types import TrackingMessage

_TS_BASE = 1_000_000_000


def _message(
    *,
    timestamp_ns: int,
    target_acquired: bool,
    state_source: str,
    servo_angles: tuple[float, float] | None = (0.0, 0.0),
    servo_angular_velocity: tuple[float, float] | None = (0.0, 0.0),
    confidence: float = 0.9,
) -> TrackingMessage:
    return TrackingMessage(
        frame_id=1,
        timestamp_ns=timestamp_ns,
        target_kind="human",
        target_acquired=target_acquired,
        state_source=state_source,  # type: ignore[arg-type]
        track_id=1 if target_acquired else None,
        confidence=confidence if target_acquired else 0.0,
        raw_pixel=None,
        filtered_pixel=None,
        raw_angles=None,
        filtered_angles=servo_angles,
        servo_angles=servo_angles,
        servo_angular_velocity=servo_angular_velocity,
        inference_ms=5.0,
        tracking_ms=1.0,
        total_latency_ms=7.0,
        fps=60.0,
    )


def _config(**search_overrides) -> AutoControllerConfig:
    search = SearchConfig(
        enabled=True,
        start_after_s=0.0,
        timeout_s=5.0,
        initial_radius_deg=2.0,
        max_radius_deg=20.0,
        expansion_rate_dps=6.0,
        scan_speed_dps=18.0,
        tilt_amplitude_ratio=0.35,
        prediction_horizon_s=0.5,
        return_home=True,
    )
    search = replace(search, **search_overrides)
    return AutoControllerConfig.from_configs(
        GimbalConfig(
            kp=0.0,
            kd=0.0,
            deadband_deg=0.0,
            slew_limit_dps=120.0,
            pan_limit_deg=30.0,
            tilt_limit_deg=20.0,
        ),
        ServoControlConfig(),
        search,
    )


@pytest.mark.unit
def test_search_remembers_last_known_target_and_slews_toward_prediction() -> None:
    cfg = _config()
    state = AutoControllerState()

    acquired = _message(
        timestamp_ns=_TS_BASE,
        target_acquired=True,
        state_source="measurement",
        servo_angles=(radians(10.0), 0.0),
        servo_angular_velocity=(radians(20.0), 0.0),
    )
    assert compute_auto_command(message=acquired, state=state, config=cfg) == (0.0, 0.0)
    assert state.last_known_pan_deg == pytest.approx(10.0)
    assert state.last_known_pan_velocity_dps == pytest.approx(20.0)

    lost = _message(
        timestamp_ns=_TS_BASE + 500_000_000,
        target_acquired=False,
        state_source="lost",
    )
    pan, tilt = compute_auto_command(message=lost, state=state, config=cfg)

    assert state.search_active is True
    assert 0.0 < pan <= 1.8
    assert tilt == pytest.approx(0.0)
    assert abs(state.search_pan_velocity_dps) <= cfg.search.scan_speed_dps


@pytest.mark.unit
def test_search_respects_mechanical_limits() -> None:
    cfg = _config(scan_speed_dps=500.0)
    state = AutoControllerState()

    acquired = _message(
        timestamp_ns=_TS_BASE,
        target_acquired=True,
        state_source="measurement",
        servo_angles=(radians(80.0), radians(60.0)),
        servo_angular_velocity=(radians(80.0), radians(80.0)),
    )
    compute_auto_command(message=acquired, state=state, config=cfg)

    lost = _message(
        timestamp_ns=_TS_BASE + 2_000_000_000,
        target_acquired=False,
        state_source="lost",
    )
    pan, tilt = compute_auto_command(message=lost, state=state, config=cfg)

    assert -cfg.pan_limit_deg <= pan <= cfg.pan_limit_deg
    assert -cfg.tilt_limit_deg <= tilt <= cfg.tilt_limit_deg


@pytest.mark.unit
def test_search_times_out_and_slews_home_without_jumping() -> None:
    cfg = _config(timeout_s=0.5, scan_speed_dps=10.0, return_home=True)
    state = AutoControllerState()
    state.pi_integral_pan = 15.0
    state.pi_integral_tilt = -8.0
    state.last_known_pan_deg = 15.0
    state.last_known_tilt_deg = -8.0
    state.last_known_timestamp_ns = _TS_BASE

    lost_after_timeout = _message(
        timestamp_ns=_TS_BASE + 1_000_000_000,
        target_acquired=False,
        state_source="center",
    )
    pan, tilt = compute_auto_command(message=lost_after_timeout, state=state, config=cfg)

    assert 14.0 <= pan < 15.0
    assert -8.0 < tilt <= -7.0
    assert state.search_active is False


@pytest.mark.unit
def test_search_disabled_preserves_center_reset_behavior() -> None:
    cfg = AutoControllerConfig.from_configs(
        GimbalConfig(),
        ServoControlConfig(),
        SearchConfig(enabled=False),
    )
    state = AutoControllerState()
    state.pi_integral_pan = 12.0
    state.pi_integral_tilt = -4.0

    centered = _message(
        timestamp_ns=_TS_BASE,
        target_acquired=False,
        state_source="center",
    )

    assert compute_auto_command(message=centered, state=state, config=cfg) == (0.0, 0.0)
    assert state.last_known_timestamp_ns is None
