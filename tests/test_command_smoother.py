"""Tests for shared outgoing command smoothing helpers."""

from __future__ import annotations

import pytest

from src.output.command_smoother import BacklashCompensator, CommandSlewLimiter


@pytest.mark.unit
def test_command_slew_limiter_seeds_first_command_without_lag() -> None:
    limiter = CommandSlewLimiter()

    pan, tilt = limiter.limit(10.0, -5.0, dt_s=0.01, limit_dps=100.0)

    assert pan == pytest.approx(10.0)
    assert tilt == pytest.approx(-5.0)


@pytest.mark.unit
def test_command_slew_limiter_caps_subsequent_position_step_per_axis() -> None:
    limiter = CommandSlewLimiter()
    limiter.limit(0.0, 0.0, dt_s=0.01, limit_dps=100.0)

    pan, tilt = limiter.limit(90.0, -90.0, dt_s=0.01, limit_dps=100.0)

    assert pan == pytest.approx(1.0)
    assert tilt == pytest.approx(-1.0)


@pytest.mark.unit
def test_command_slew_limiter_passes_through_when_disabled() -> None:
    limiter = CommandSlewLimiter()
    limiter.limit(0.0, 0.0, dt_s=0.01, limit_dps=100.0)

    pan, tilt = limiter.limit(90.0, -90.0, dt_s=0.01, limit_dps=0.0)

    assert pan == pytest.approx(90.0)
    assert tilt == pytest.approx(-90.0)


@pytest.mark.unit
def test_backlash_compensator_adds_short_pulse_on_direction_reversal() -> None:
    comp = BacklashCompensator()

    assert comp.compensate(0.0, 0.0, dt_s=0.02, enabled=True, pan_deg=0.2, tilt_deg=0.0) == (
        0.0,
        0.0,
    )
    assert comp.compensate(1.0, 0.0, dt_s=0.02, enabled=True, pan_deg=0.2, tilt_deg=0.0) == (
        1.0,
        0.0,
    )
    pan, tilt = comp.compensate(
        0.5,
        0.0,
        dt_s=0.02,
        enabled=True,
        pan_deg=0.2,
        tilt_deg=0.0,
        pulse_decay_dps=0.0,
    )

    assert pan == pytest.approx(0.3)
    assert tilt == pytest.approx(0.0)


@pytest.mark.unit
def test_backlash_compensator_decay_removes_pulse_without_new_motion() -> None:
    comp = BacklashCompensator()
    comp.compensate(0.0, 0.0, dt_s=0.02, enabled=True, pan_deg=0.2, tilt_deg=0.0)
    comp.compensate(1.0, 0.0, dt_s=0.02, enabled=True, pan_deg=0.2, tilt_deg=0.0)
    comp.compensate(
        0.5,
        0.0,
        dt_s=0.02,
        enabled=True,
        pan_deg=0.2,
        tilt_deg=0.0,
        pulse_decay_dps=0.0,
    )

    pan, _ = comp.compensate(
        0.5,
        0.0,
        dt_s=0.05,
        enabled=True,
        pan_deg=0.2,
        tilt_deg=0.0,
        pulse_decay_dps=2.0,
    )

    assert pan == pytest.approx(0.4)


@pytest.mark.unit
def test_backlash_compensator_ignores_tiny_command_noise() -> None:
    comp = BacklashCompensator()
    comp.compensate(0.0, 0.0, dt_s=0.02, enabled=True, pan_deg=0.2, tilt_deg=0.0)
    pan, _ = comp.compensate(
        0.03,
        0.0,
        dt_s=0.02,
        enabled=True,
        pan_deg=0.2,
        tilt_deg=0.0,
        min_step_deg=0.08,
    )

    assert pan == pytest.approx(0.03)
