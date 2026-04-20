"""Phase I: Soft re-identification tests."""
from __future__ import annotations

import numpy as np
import pytest

from src.tracking.reid import ReIDBuffer


def _make_frame(color: tuple[int, int, int], size: int = 64) -> np.ndarray:
    """Create a solid-color BGR frame."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


@pytest.mark.unit
def test_reid_matching_same_appearance() -> None:
    """Identical frames should match above threshold."""
    buf = ReIDBuffer(correlation_threshold=0.5)
    frame = _make_frame((100, 150, 200))
    bbox = np.array([0.0, 0.0, 64.0, 64.0])

    buf.store_lost_track(42, frame, bbox, timestamp_ns=1_000_000_000)
    matched = buf.match(frame, bbox, timestamp_ns=1_500_000_000)

    assert matched == 42


@pytest.mark.unit
def test_reid_no_match_different_appearance() -> None:
    """Very different frames should not match."""
    buf = ReIDBuffer(correlation_threshold=0.9)
    red_frame = _make_frame((0, 0, 255))
    blue_frame = _make_frame((255, 0, 0))
    bbox = np.array([0.0, 0.0, 64.0, 64.0])

    buf.store_lost_track(42, red_frame, bbox, timestamp_ns=1_000_000_000)
    matched = buf.match(blue_frame, bbox, timestamp_ns=1_500_000_000)

    assert matched is None


@pytest.mark.unit
def test_reid_expired_track_not_matched() -> None:
    buf = ReIDBuffer(max_age_ns=1_000_000_000)  # 1 second
    frame = _make_frame((100, 100, 100))
    bbox = np.array([0.0, 0.0, 64.0, 64.0])

    buf.store_lost_track(42, frame, bbox, timestamp_ns=1_000_000_000)
    # 2 seconds later — expired
    matched = buf.match(frame, bbox, timestamp_ns=3_000_000_000)

    assert matched is None


@pytest.mark.unit
def test_reid_match_removes_from_buffer() -> None:
    buf = ReIDBuffer(correlation_threshold=0.5)
    frame = _make_frame((100, 150, 200))
    bbox = np.array([0.0, 0.0, 64.0, 64.0])

    buf.store_lost_track(42, frame, bbox, timestamp_ns=1_000_000_000)
    buf.match(frame, bbox, timestamp_ns=1_500_000_000)  # consumes it
    second_match = buf.match(frame, bbox, timestamp_ns=2_000_000_000)

    assert second_match is None  # already consumed


@pytest.mark.unit
def test_reid_respects_max_stored() -> None:
    buf = ReIDBuffer(max_stored=2, correlation_threshold=0.5)
    frame = _make_frame((100, 100, 100))
    bbox = np.array([0.0, 0.0, 64.0, 64.0])

    buf.store_lost_track(1, frame, bbox, timestamp_ns=1_000_000_000)
    buf.store_lost_track(2, frame, bbox, timestamp_ns=2_000_000_000)
    buf.store_lost_track(3, frame, bbox, timestamp_ns=3_000_000_000)

    # Track 1 should have been evicted (oldest)
    assert 1 not in buf._lost_tracks
    assert len(buf._lost_tracks) == 2


@pytest.mark.unit
def test_reid_empty_buffer_returns_none() -> None:
    buf = ReIDBuffer()
    frame = _make_frame((100, 100, 100))
    bbox = np.array([0.0, 0.0, 64.0, 64.0])

    assert buf.match(frame, bbox, timestamp_ns=1_000_000_000) is None


@pytest.mark.unit
def test_reid_tiny_bbox_returns_none() -> None:
    buf = ReIDBuffer()
    frame = _make_frame((100, 100, 100))
    tiny_bbox = np.array([10.0, 10.0, 12.0, 12.0])  # 2x2 — too small

    hist = buf.compute_histogram(frame, tiny_bbox)
    assert hist is None


@pytest.mark.unit
def test_reid_clear() -> None:
    buf = ReIDBuffer()
    frame = _make_frame((100, 100, 100))
    bbox = np.array([0.0, 0.0, 64.0, 64.0])
    buf.store_lost_track(1, frame, bbox, timestamp_ns=1_000_000_000)

    buf.clear()


@pytest.mark.unit
def test_reid_spatial_gate_rejects_distant_match() -> None:
    """A track that moved too far spatially should not match even if appearance matches."""
    buf = ReIDBuffer(correlation_threshold=0.5, max_spatial_distance_px=50.0)
    frame = _make_frame((100, 150, 200))
    bbox_near = np.array([0.0, 0.0, 64.0, 64.0])

    buf.store_lost_track(42, frame, bbox_near, timestamp_ns=1_000_000_000)

    # Same appearance but bbox center moved >50px away
    bbox_far = np.array([200.0, 200.0, 264.0, 264.0])
    matched = buf.match(frame, bbox_far, timestamp_ns=1_500_000_000)
    assert matched is None  # too far even though appearance matches


@pytest.mark.unit
def test_reid_spatial_gate_allows_close_match() -> None:
    """A track within spatial distance should match if appearance matches."""
    buf = ReIDBuffer(correlation_threshold=0.5, max_spatial_distance_px=100.0)
    frame = _make_frame((100, 150, 200))
    bbox_a = np.array([0.0, 0.0, 64.0, 64.0])

    buf.store_lost_track(42, frame, bbox_a, timestamp_ns=1_000_000_000)

    # Same appearance, moved only ~30px
    bbox_b = np.array([20.0, 20.0, 84.0, 84.0])
    matched = buf.match(frame, bbox_b, timestamp_ns=1_500_000_000)
    assert matched == 42

    assert len(buf._lost_tracks) == 0
