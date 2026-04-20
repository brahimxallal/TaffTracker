from __future__ import annotations

import numpy as np
import pytest

from src.output.visualizer import FrameSmoother, draw_overlay
from src.shared.types import TrackingMessage


def _make_message(**overrides) -> TrackingMessage:
    defaults: dict = dict(
        frame_id=1,
        timestamp_ns=1_000_000_000,
        target_kind="human",
        target_acquired=True,
        state_source="measurement",
        track_id=1,
        confidence=0.9,
        raw_pixel=(100.0, 200.0),
        filtered_pixel=(101.0, 201.0),
        raw_angles=(0.1, 0.2),
        filtered_angles=(0.11, 0.21),
        inference_ms=10.0,
        tracking_ms=1.0,
        total_latency_ms=15.0,
    )
    defaults.update(overrides)
    return TrackingMessage(**defaults)


@pytest.mark.unit
def test_draw_overlay_preserves_frame_shape() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    msg = _make_message()

    result = draw_overlay(frame, msg)

    assert result.shape == frame.shape
    assert result.dtype == frame.dtype


@pytest.mark.unit
def test_draw_overlay_with_no_pixels_does_not_raise() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    msg = _make_message(raw_pixel=None, filtered_pixel=None)

    result = draw_overlay(frame, msg)

    assert result.shape == frame.shape


@pytest.mark.unit
def test_draw_overlay_draws_in_place_and_returns_same_buffer() -> None:
    """draw_overlay mutates the input frame in-place (no copy) and returns it."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    msg = _make_message()

    result = draw_overlay(frame, msg)

    # Must return the exact same buffer (identity, not just equality)
    assert result is frame
    # Frame should have been modified (HUD elements drawn)
    assert result.sum() > 0


@pytest.mark.unit
def test_draw_overlay_accepts_disabled_laser_indicator() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    msg = _make_message()

    result = draw_overlay(frame, msg, laser_enabled=False)

    assert result.shape == frame.shape


@pytest.mark.unit
def test_smoother_isolation() -> None:
    """Two FrameSmoother instances must not share state."""
    sm1 = FrameSmoother()
    sm2 = FrameSmoother()

    sm1.update(1, 100.0, 200.0)
    sm2.update(2, 500.0, 600.0)

    assert sm1.prev_track_id == 1
    assert sm2.prev_track_id == 2
    assert sm1.smooth_x != sm2.smooth_x
    assert sm1.smooth_y != sm2.smooth_y


@pytest.mark.unit
def test_smoother_clear_resets_state() -> None:
    sm = FrameSmoother()
    sm.update(1, 100.0, 200.0)
    sm.clear()

    assert sm.prev_track_id is None
    assert sm.smooth_x is None
    assert sm.smooth_y is None
