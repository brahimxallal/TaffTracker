"""Tests for ``src.output.manual_control``.

Covers the pure helper and the small stateful tracker that were extracted
from :class:`OutputProcess` — the behaviour is identical to the prior
private implementation, but the public surface deserves its own tests.
"""

from __future__ import annotations

import pytest

from src.output.manual_control import ManualVelocityTracker, boost_manual_velocity

# ---------------------------------------------------------------------------
# boost_manual_velocity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_boost_manual_velocity_zero_stays_zero() -> None:
    assert boost_manual_velocity(0.0, 80.0) == 0.0


@pytest.mark.unit
def test_boost_manual_velocity_below_threshold_is_zero() -> None:
    # Effectively-zero inputs should not get lifted to the floor.
    assert boost_manual_velocity(1e-9, 80.0) == 0.0
    assert boost_manual_velocity(-1e-9, 80.0) == 0.0


@pytest.mark.unit
def test_boost_manual_velocity_lifts_to_floor_preserving_sign() -> None:
    assert boost_manual_velocity(10.0, 80.0) == 80.0
    assert boost_manual_velocity(-10.0, 80.0) == -80.0


@pytest.mark.unit
def test_boost_manual_velocity_passes_through_when_above_floor() -> None:
    assert boost_manual_velocity(150.0, 80.0) == 150.0
    assert boost_manual_velocity(-150.0, 80.0) == -150.0


# ---------------------------------------------------------------------------
# ManualVelocityTracker
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tracker_first_sample_returns_zero() -> None:
    tracker = ManualVelocityTracker()
    pan_vel, tilt_vel = tracker.compute_velocity_dps(10.0, 5.0, 1_000_000_000)
    assert pan_vel == 0.0
    assert tilt_vel == 0.0


@pytest.mark.unit
def test_tracker_second_sample_computes_derivative() -> None:
    tracker = ManualVelocityTracker()
    tracker.compute_velocity_dps(10.0, 5.0, 1_000_000_000)
    # 50 ms later (well under the 100 ms dt cap), +0.5 deg pan, -0.2 deg tilt
    # -> 10 dps pan, -4 dps tilt
    pan_vel, tilt_vel = tracker.compute_velocity_dps(10.5, 4.8, 1_050_000_000)
    assert pan_vel == pytest.approx(10.0)
    assert tilt_vel == pytest.approx(-4.0)


@pytest.mark.unit
def test_tracker_caps_dt_to_prevent_spikes() -> None:
    """dt is clamped to 100 ms, so a 10-second pause still reports realistic vel."""
    tracker = ManualVelocityTracker()
    tracker.compute_velocity_dps(0.0, 0.0, 0)
    # 10 s gap, 1 deg move -> uncapped would give 0.1 dps; capped gives 10 dps
    pan_vel, _ = tracker.compute_velocity_dps(1.0, 0.0, 10_000_000_000)
    assert pan_vel == pytest.approx(10.0)


@pytest.mark.unit
def test_tracker_non_monotonic_timestamp_returns_zero() -> None:
    tracker = ManualVelocityTracker()
    tracker.compute_velocity_dps(10.0, 5.0, 2_000_000_000)
    # Older timestamp than the stored sample -> guard returns (0, 0)
    pan_vel, tilt_vel = tracker.compute_velocity_dps(20.0, 10.0, 1_000_000_000)
    assert pan_vel == 0.0
    assert tilt_vel == 0.0


@pytest.mark.unit
def test_tracker_reset_forgets_history() -> None:
    tracker = ManualVelocityTracker()
    tracker.compute_velocity_dps(10.0, 5.0, 1_000_000_000)
    tracker.reset()
    # First call after reset is treated as a new history -> zero velocity
    pan_vel, tilt_vel = tracker.compute_velocity_dps(20.0, 10.0, 2_000_000_000)
    assert pan_vel == 0.0
    assert tilt_vel == 0.0


@pytest.mark.unit
def test_tracker_stores_sample_even_when_returning_zero() -> None:
    """Even on the first call, the sample is recorded for the next derivative."""
    tracker = ManualVelocityTracker()
    tracker.compute_velocity_dps(10.0, 5.0, 1_000_000_000)
    # 50 ms gap keeps dt under the 0.1 s clamp so the derivative is exact.
    pan_vel, tilt_vel = tracker.compute_velocity_dps(11.0, 5.0, 1_050_000_000)
    assert pan_vel == pytest.approx(20.0)
    assert tilt_vel == pytest.approx(0.0)
