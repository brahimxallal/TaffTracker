import numpy as np
import pytest

from src.shared.types import Detection
from src.tracking.bytetrack import ByteTracker


@pytest.mark.unit
def test_bytetrack_preserves_track_id_for_continuous_motion() -> None:
    tracker = ByteTracker(track_thresh=0.5, match_thresh=0.3, max_lost=3, birth_min_hits=1)
    frame_one = [Detection(bbox=np.array([0, 0, 10, 10]), score=0.9)]
    frame_two = [Detection(bbox=np.array([1, 0, 11, 10]), score=0.88)]

    first_tracks = tracker.update(frame_one, timestamp_ns=1)
    second_tracks = tracker.update(frame_two, timestamp_ns=2)

    assert len(first_tracks) == 1
    assert len(second_tracks) == 1
    assert first_tracks[0].track_id == second_tracks[0].track_id


@pytest.mark.unit
def test_bytetrack_keeps_track_alive_with_low_confidence_match() -> None:
    tracker = ByteTracker(
        track_thresh=0.5, low_thresh=0.2, match_thresh=0.3, max_lost=2, birth_min_hits=1
    )
    tracker.update([Detection(bbox=np.array([0, 0, 10, 10]), score=0.9)], timestamp_ns=1)

    tracks = tracker.update([Detection(bbox=np.array([1, 0, 11, 10]), score=0.3)], timestamp_ns=2)

    assert len(tracks) == 1
    assert tracks[0].lost_frames == 0


@pytest.mark.unit
def test_bytetrack_empty_detections_returns_empty() -> None:
    tracker = ByteTracker(track_thresh=0.5, match_thresh=0.3, max_lost=3)

    tracks = tracker.update([], timestamp_ns=1)

    assert tracks == []


@pytest.mark.unit
def test_bytetrack_prunes_tracks_exceeding_max_lost() -> None:
    tracker = ByteTracker(track_thresh=0.5, match_thresh=0.3, max_lost=2, birth_min_hits=1)
    tracker.update([Detection(bbox=np.array([0, 0, 10, 10]), score=0.9)], timestamp_ns=1)
    # Provide no matching detection so the track accumulates lost frames
    tracker.update([], timestamp_ns=2)
    tracker.update([], timestamp_ns=3)

    tracks = tracker.update([], timestamp_ns=4)

    assert tracks == []


@pytest.mark.unit
def test_bytetrack_reset_clears_all_state() -> None:
    tracker = ByteTracker(track_thresh=0.5, match_thresh=0.3, max_lost=10, birth_min_hits=1)
    tracker.update([Detection(bbox=np.array([0, 0, 10, 10]), score=0.9)], timestamp_ns=1)

    tracker.reset()
    tracks = tracker.update([], timestamp_ns=2)

    assert tracks == []
    # New tracks get id starting from 1 again after reset
    new_tracks = tracker.update(
        [Detection(bbox=np.array([5, 5, 15, 15]), score=0.9)], timestamp_ns=3
    )
    assert new_tracks[0].track_id == 1


@pytest.mark.unit
def test_bytetrack_set_track_threshold_updates_split() -> None:
    """set_track_threshold should change the high/low confidence split."""
    tracker = ByteTracker(track_thresh=0.5, match_thresh=0.3, max_lost=10, birth_min_hits=1)
    # Detection at 0.45 should be low-conf with threshold 0.5
    tracks = tracker.update([Detection(bbox=np.array([0, 0, 50, 50]), score=0.45)], timestamp_ns=1)
    # Lower the threshold — now 0.45 should be high-conf
    tracker.set_track_threshold(0.4)
    tracks = tracker.update([Detection(bbox=np.array([0, 0, 50, 50]), score=0.45)], timestamp_ns=2)
    assert len(tracks) > 0
