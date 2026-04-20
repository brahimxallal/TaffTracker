"""Tests for BoTSORT tracker (ByteTrack + Camera Motion Compensation)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.types import Detection
from src.tracking.botsort import BoTSORT, _warp_bbox


def _make_frame(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.mark.unit
def test_botsort_preserves_track_id_like_bytetrack() -> None:
    """BoTSORT should behave identically to ByteTracker for basic matching."""
    tracker = BoTSORT(
        track_thresh=0.5, match_thresh=0.3, max_lost=3, birth_min_hits=1, cmc_enabled=False
    )
    frame_one = [Detection(bbox=np.array([0, 0, 10, 10]), score=0.9)]
    frame_two = [Detection(bbox=np.array([1, 0, 11, 10]), score=0.88)]

    first_tracks = tracker.update(frame_one, timestamp_ns=1)
    second_tracks = tracker.update(frame_two, timestamp_ns=2)

    assert len(first_tracks) == 1
    assert len(second_tracks) == 1
    assert first_tracks[0].track_id == second_tracks[0].track_id


@pytest.mark.unit
def test_botsort_cmc_warps_tracks_with_frame() -> None:
    """When CMC is enabled and a frame is passed, tracks should be warped before matching."""
    tracker = BoTSORT(
        track_thresh=0.5, match_thresh=0.3, max_lost=3, birth_min_hits=1, cmc_enabled=True
    )
    frame1 = _make_frame(64, 64, seed=42)
    frame2 = _make_frame(64, 64, seed=42)  # same frame = identity warp

    det = [Detection(bbox=np.array([10, 10, 30, 30]), score=0.9)]
    tracker.update(det, timestamp_ns=1, frame=frame1)
    tracks = tracker.update(det, timestamp_ns=2, frame=frame2)

    assert len(tracks) == 1
    assert tracks[0].track_id == 1


@pytest.mark.unit
def test_botsort_cmc_disabled_works_like_bytetrack() -> None:
    """With cmc_enabled=False, BoTSORT is identical to ByteTracker."""
    tracker = BoTSORT(
        track_thresh=0.5, match_thresh=0.3, max_lost=2, birth_min_hits=1, cmc_enabled=False
    )
    tracker.update([Detection(bbox=np.array([0, 0, 10, 10]), score=0.9)], timestamp_ns=1)
    tracker.update([], timestamp_ns=2)
    tracker.update([], timestamp_ns=3)
    tracks = tracker.update([], timestamp_ns=4)
    assert tracks == []


@pytest.mark.unit
def test_botsort_reset_clears_cmc_state() -> None:
    tracker = BoTSORT(track_thresh=0.5, match_thresh=0.3, max_lost=10, birth_min_hits=1)
    frame = _make_frame(64, 64, seed=1)
    tracker.update(
        [Detection(bbox=np.array([0, 0, 10, 10]), score=0.9)], timestamp_ns=1, frame=frame
    )

    tracker.reset()
    assert tracker._prev_gray is None
    tracks = tracker.update([], timestamp_ns=2)
    assert tracks == []


@pytest.mark.unit
def test_botsort_update_without_frame_still_works() -> None:
    """Calling update without a frame should work (no CMC applied)."""
    tracker = BoTSORT(
        track_thresh=0.5, match_thresh=0.3, max_lost=3, birth_min_hits=1, cmc_enabled=True
    )
    det = [Detection(bbox=np.array([10, 10, 30, 30]), score=0.9)]
    tracks = tracker.update(det, timestamp_ns=1)
    assert len(tracks) == 1


@pytest.mark.unit
def test_warp_bbox_identity() -> None:
    """Identity affine should leave bbox unchanged."""
    bbox = np.array([10.0, 20.0, 30.0, 40.0])
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = _warp_bbox(bbox, identity)
    np.testing.assert_allclose(result, bbox, atol=1e-10)


@pytest.mark.unit
def test_warp_bbox_translation() -> None:
    """Translation warp should shift bbox by (tx, ty)."""
    bbox = np.array([10.0, 20.0, 30.0, 40.0])
    tx, ty = 5.0, -3.0
    warp = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]])
    result = _warp_bbox(bbox, warp)
    expected = np.array([15.0, 17.0, 35.0, 37.0])
    np.testing.assert_allclose(result, expected, atol=1e-10)


@pytest.mark.unit
def test_botsort_low_conf_matching_works() -> None:
    """Low-confidence detections should still match existing tracks."""
    tracker = BoTSORT(
        track_thresh=0.5,
        low_thresh=0.2,
        match_thresh=0.3,
        max_lost=2,
        birth_min_hits=1,
        cmc_enabled=False,
    )
    tracker.update([Detection(bbox=np.array([0, 0, 10, 10]), score=0.9)], timestamp_ns=1)
    tracks = tracker.update([Detection(bbox=np.array([1, 0, 11, 10]), score=0.3)], timestamp_ns=2)
    assert len(tracks) == 1
    assert tracks[0].lost_frames == 0
