from __future__ import annotations

from math import radians

import pytest

from src.output.relock_diagnostics import RelockDiagnostics
from src.shared.protocol import FLAG_FAST_MOTION
from src.shared.types import TrackingMessage


def _message(**overrides) -> TrackingMessage:
    defaults: dict = dict(
        frame_id=1,
        timestamp_ns=1_000_000_000,
        target_kind="human",
        target_acquired=True,
        state_source="measurement",
        track_id=7,
        confidence=0.88,
        raw_pixel=(320.0, 240.0),
        filtered_pixel=(320.0, 240.0),
        raw_angles=(0.0, 0.0),
        filtered_angles=(radians(4.0), radians(-3.0)),
        inference_ms=6.0,
        tracking_ms=1.0,
        total_latency_ms=9.0,
    )
    defaults.update(overrides)
    return TrackingMessage(**defaults)


@pytest.mark.unit
def test_relock_diagnostics_emits_event_on_lost_to_measurement_transition() -> None:
    diag = RelockDiagnostics()
    lost = _message(
        frame_id=10,
        timestamp_ns=1_000_000_000,
        target_acquired=False,
        state_source="lost",
        track_id=7,
        confidence=0.0,
        filtered_angles=None,
    )
    reacquired = _message(frame_id=11, timestamp_ns=1_150_000_000, track_id=8)

    assert diag.observe(lost, command_deg=(1.0, 2.0), velocity_dps=(0.0, 0.0), state_flags=0) is None
    event = diag.observe(
        reacquired,
        command_deg=(5.0, -1.0),
        velocity_dps=(80.0, -20.0),
        state_flags=FLAG_FAST_MOTION,
    )

    assert event is not None
    assert event.frame_id == 11
    assert event.track_id == 8
    assert event.previous_track_id == 7
    assert event.lost_ms == pytest.approx(150.0)
    assert event.error_deg == pytest.approx(5.0)
    assert event.command_delta_deg == pytest.approx(5.0)
    assert event.command_velocity_dps == pytest.approx((80.0**2 + 20.0**2) ** 0.5)
    assert event.fast_motion is True


@pytest.mark.unit
def test_relock_diagnostics_suppresses_same_track_micro_loss() -> None:
    diag = RelockDiagnostics(micro_loss_window_s=0.12)
    acquired = _message(frame_id=9, timestamp_ns=1_000_000_000, track_id=7)
    lost = _message(
        frame_id=10,
        timestamp_ns=1_030_000_000,
        target_acquired=False,
        state_source="prediction",
        track_id=7,
        confidence=0.0,
        filtered_angles=(radians(4.0), radians(-3.0)),
    )
    reacquired = _message(frame_id=11, timestamp_ns=1_060_000_000, track_id=7)

    assert diag.observe(acquired, command_deg=(1.0, 2.0), velocity_dps=(0.0, 0.0), state_flags=0)
    assert diag.observe(lost, command_deg=(1.0, 2.0), velocity_dps=(0.0, 0.0), state_flags=0) is None
    assert (
        diag.observe(
            reacquired,
            command_deg=(5.0, -1.0),
            velocity_dps=(80.0, -20.0),
            state_flags=0,
        )
        is None
    )


@pytest.mark.unit
def test_relock_diagnostics_keeps_long_same_track_loss_visible() -> None:
    diag = RelockDiagnostics(micro_loss_window_s=0.12)
    acquired = _message(frame_id=9, timestamp_ns=1_000_000_000, track_id=7)
    lost = _message(
        frame_id=10,
        timestamp_ns=1_030_000_000,
        target_acquired=False,
        state_source="prediction",
        track_id=7,
        confidence=0.0,
        filtered_angles=(radians(4.0), radians(-3.0)),
    )
    reacquired = _message(frame_id=11, timestamp_ns=1_300_000_000, track_id=7)

    diag.observe(acquired, command_deg=(1.0, 2.0), velocity_dps=(0.0, 0.0), state_flags=0)
    diag.observe(lost, command_deg=(1.0, 2.0), velocity_dps=(0.0, 0.0), state_flags=0)
    event = diag.observe(
        reacquired,
        command_deg=(5.0, -1.0),
        velocity_dps=(80.0, -20.0),
        state_flags=0,
    )

    assert event is not None
    assert event.lost_ms == pytest.approx(270.0)
