"""Tests for ``src.output.manual_control``.

Covers the pure helper and the small stateful tracker that were extracted
from :class:`OutputProcess` — the behaviour is identical to the prior
private implementation, but the public surface deserves its own tests.
"""

from __future__ import annotations

import pytest

from src.output.manual_control import (
    ManualVelocityTracker,
    boost_manual_velocity,
    build_manual_packet,
    rewrite_packet_sequence,
)
from src.shared.protocol import FLAG_LASER_ON, FLAG_RELAY_ON, decode_packet_v2

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


# ---------------------------------------------------------------------------
# build_manual_packet + rewrite_packet_sequence
# ---------------------------------------------------------------------------


def _build_sample_packet(
    *,
    pan_deg: float = 1.5,
    tilt_deg: float = -2.0,
    pan_vel_dps: float = 10.0,
    tilt_vel_dps: float = -5.0,
    now_ns: int = 1_234_000_000,
    laser_enabled: bool = False,
    relay_on: bool = False,
) -> bytes:
    return build_manual_packet(
        pan_deg=pan_deg,
        tilt_deg=tilt_deg,
        pan_vel_dps=pan_vel_dps,
        tilt_vel_dps=tilt_vel_dps,
        now_ns=now_ns,
        laser_enabled=laser_enabled,
        relay_on=relay_on,
    )


@pytest.mark.unit
def test_build_manual_packet_encodes_centi_degrees() -> None:
    packet = _build_sample_packet(pan_deg=1.5, tilt_deg=-2.0, pan_vel_dps=10.0, tilt_vel_dps=-5.0)
    decoded = decode_packet_v2(packet)
    assert decoded is not None
    assert decoded.pan == 150  # 1.5 deg * 100
    assert decoded.tilt == -200  # -2.0 deg * 100
    assert decoded.pan_vel == 1000
    assert decoded.tilt_vel == -500


@pytest.mark.unit
def test_build_manual_packet_uses_manual_quality_flag() -> None:
    """The firmware looks at the top bit of the quality byte to bypass EMA/deadzone."""
    packet = _build_sample_packet()
    decoded = decode_packet_v2(packet)
    assert decoded is not None
    assert decoded.quality & 0x80  # QUALITY_FLAG_MANUAL


@pytest.mark.unit
def test_build_manual_packet_laser_and_relay_flags() -> None:
    packet_on = _build_sample_packet(laser_enabled=True, relay_on=True)
    packet_off = _build_sample_packet(laser_enabled=False, relay_on=False)
    on_decoded = decode_packet_v2(packet_on)
    off_decoded = decode_packet_v2(packet_off)
    assert on_decoded is not None and off_decoded is not None
    assert on_decoded.state & FLAG_LASER_ON
    assert on_decoded.state & FLAG_RELAY_ON
    assert not (off_decoded.state & FLAG_LASER_ON)
    assert not (off_decoded.state & FLAG_RELAY_ON)


@pytest.mark.unit
def test_build_manual_packet_sets_sequence_to_zero() -> None:
    """Sequence is filled in later via rewrite_packet_sequence."""
    packet = _build_sample_packet()
    decoded = decode_packet_v2(packet)
    assert decoded is not None
    assert decoded.sequence == 0


@pytest.mark.unit
def test_rewrite_packet_sequence_updates_seq_and_keeps_packet_valid() -> None:
    packet = _build_sample_packet(pan_deg=3.0, tilt_deg=4.0)
    rewritten = rewrite_packet_sequence(packet, sequence=1234)
    decoded = decode_packet_v2(rewritten)
    # Decoding succeeded -> CRC was recomputed correctly.
    assert decoded is not None
    assert decoded.sequence == 1234
    # Payload fields unchanged.
    assert decoded.pan == 300
    assert decoded.tilt == 400


@pytest.mark.unit
def test_rewrite_packet_sequence_wraps_at_16_bits() -> None:
    packet = _build_sample_packet()
    rewritten = rewrite_packet_sequence(packet, sequence=0x1_0001)
    decoded = decode_packet_v2(rewritten)
    assert decoded is not None
    assert decoded.sequence == 1  # low 16 bits only


@pytest.mark.unit
def test_rewrite_packet_sequence_is_pure() -> None:
    """The original bytes object must not be mutated in place."""
    packet = _build_sample_packet()
    original = bytes(packet)
    rewrite_packet_sequence(packet, sequence=42)
    assert packet == original
