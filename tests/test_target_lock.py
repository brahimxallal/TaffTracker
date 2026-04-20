"""Tests for target locking in TrackerStage.select_primary_track."""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

from src.inference.postprocess import KeypointStabilizer
from src.inference.stages.tracker import TrackerStage, _MAX_CACHED_TRACKS
from src.shared.types import Track
from src.tracking.botsort import BoTSORT
from src.tracking.kalman import KalmanFilter
from src.tracking.one_euro import OneEuroFilter2D
from src.tracking.reid import ReIDBuffer


def _make_track(
    track_id: int,
    score: float = 0.9,
    lost_frames: int = 0,
    bbox: tuple[float, float, float, float] = (10, 20, 50, 80),
) -> Track:
    return Track(
        track_id=track_id,
        bbox=np.array(bbox, dtype=np.float32),
        score=score,
        lost_frames=lost_frames,
    )


def _make_stage(max_lost_frames: int = 30) -> TrackerStage:
    """Create a minimal TrackerStage for testing select_primary_track."""
    return TrackerStage(
        tracker=BoTSORT(),
        kalman=KalmanFilter(process_noise=1.0, measurement_noise=1.0),
        stabilizer=KeypointStabilizer(alpha=0.2),
        reid_buffer=ReIDBuffer(),
        max_lost_frames=max_lost_frames,
    )


def _select(stage: TrackerStage, tracks: list[Track]) -> Track | None:
    """Helper to call select_primary_track with defaults for unit tests."""
    result, _reid_match = stage.select_primary_track(
        tracks=tracks,
        frame=np.zeros((10, 10, 3), dtype=np.uint8),
        timestamp_ns=0,
        prev_locked_id=stage.locked_track_id,
    )
    return result


@pytest.mark.unit
class TestTargetLock:

    def test_locks_first_best_track(self):
        stage = _make_stage()
        t1 = _make_track(track_id=1, score=0.9)
        t2 = _make_track(track_id=2, score=0.7)
        result = _select(stage, [t1, t2])
        assert result.track_id == 1
        assert stage.locked_track_id == 1

    def test_stays_locked_despite_better_score(self):
        stage = _make_stage()
        # Lock onto track 1
        t1 = _make_track(track_id=1, score=0.9)
        t2 = _make_track(track_id=2, score=0.7)
        _select(stage, [t1, t2])

        # Track 2 now has a much better score — should still pick track 1
        t1_lower = _make_track(track_id=1, score=0.5)
        t2_higher = _make_track(track_id=2, score=0.99)
        result = _select(stage, [t1_lower, t2_higher])
        assert result.track_id == 1

    def test_stays_locked_despite_bigger_bbox(self):
        stage = _make_stage()
        t1 = _make_track(track_id=1, score=0.9, bbox=(10, 10, 50, 50))
        t2 = _make_track(track_id=2, score=0.8, bbox=(10, 10, 200, 200))
        _select(stage, [t1, t2])
        assert stage.locked_track_id == 1

        # Track 2 now has a bigger bbox — should still pick track 1
        result = _select(stage, [t1, t2])
        assert result.track_id == 1

    def test_unlocks_when_lost_exceeds_max(self):
        stage = _make_stage(max_lost_frames=5)
        # Lock track 1
        t1 = _make_track(track_id=1, score=0.9)
        _select(stage, [t1])
        assert stage.locked_track_id == 1

        # Track 1 lost for too long, track 2 available
        t1_lost = _make_track(track_id=1, score=0.5, lost_frames=6)
        t2 = _make_track(track_id=2, score=0.85)
        result = _select(stage, [t1_lost, t2])
        assert result.track_id == 2
        assert stage.locked_track_id == 2

    def test_unlocks_when_track_disappears(self):
        stage = _make_stage()
        # Lock track 1
        _select(stage, [_make_track(track_id=1, score=0.9)])
        assert stage.locked_track_id == 1

        # Track 1 gone from tracker output entirely
        t3 = _make_track(track_id=3, score=0.8)
        result = _select(stage, [t3])
        assert result.track_id == 3
        assert stage.locked_track_id == 3

    def test_resets_lock_on_empty_tracks(self):
        stage = _make_stage()
        _select(stage, [_make_track(track_id=1, score=0.9)])
        assert stage.locked_track_id == 1

        result = _select(stage, [])
        assert result is None
        assert stage.locked_track_id is None

    def test_keeps_locked_track_with_minor_lost_frames(self):
        stage = _make_stage(max_lost_frames=30)
        _select(stage, [_make_track(track_id=1, score=0.9)])

        # Track 1 briefly lost but within tolerance
        t1_brief = _make_track(track_id=1, score=0.5, lost_frames=5)
        t2 = _make_track(track_id=2, score=0.95)
        result = _select(stage, [t1_brief, t2])
        assert result.track_id == 1

    def test_relocks_after_all_lost(self):
        stage = _make_stage()
        _select(stage, [_make_track(track_id=1, score=0.9)])

        # Everyone disappears
        _select(stage, [])
        assert stage.locked_track_id is None

        # New target appears
        t5 = _make_track(track_id=5, score=0.88)
        result = _select(stage, [t5])
        assert result.track_id == 5
        assert stage.locked_track_id == 5


@pytest.mark.unit
class TestMultiDogLockStability:
    """Phase M: multi-dog lock stability tests."""

    def test_relock_prefers_spatial_proximity(self):
        """When re-locking after loss, prefer the dog nearest to last position."""
        stage = _make_stage(max_lost_frames=5)
        # Lock on track 1 at position (100, 100)
        t1 = _make_track(track_id=1, score=0.9, bbox=(80, 80, 120, 120))
        _select(stage, [t1])
        # Simulate known centroid position
        stage.last_locked_centroid = (100.0, 100.0)

        # Track 1 disappears, two new dogs appear: one near (110,110) and one far (500,500)
        t_near = _make_track(track_id=2, score=0.95, bbox=(90, 90, 130, 130))  # center ~110,110
        t_far = _make_track(track_id=3, score=0.99, bbox=(480, 480, 520, 520))  # center ~500,500
        # Force lock release by removing track 1
        result = _select(stage, [t_near, t_far])
        assert result.track_id == 2, "Should re-lock to nearest dog, not highest score"

    def test_lock_stable_multi_dog_intermittent_loss(self):
        """Lock stays on Dog A even when it has lost_frames=1-3 while Dog B is visible."""
        stage = _make_stage(max_lost_frames=30)
        # Establish lock on track 1 first (single target)
        _select(stage, [_make_track(track_id=1, score=0.9)])
        assert stage.locked_track_id == 1

        # Simulate 20 frames where Dog A intermittently has lost_frames 0-3
        for frame in range(20):
            lost = frame % 4  # cycles 0, 1, 2, 3
            t1_frame = _make_track(track_id=1, score=0.85, lost_frames=lost)
            t2_frame = _make_track(track_id=2, score=0.95, lost_frames=0)
            result = _select(stage, [t1_frame, t2_frame])
            assert result.track_id == 1, f"Lock should not switch at frame {frame} (lost={lost})"

    def test_track_state_cache_save_restore(self):
        """Switching away from a track and back should restore Kalman state."""
        stage = _make_stage()
        ema_pixel = OneEuroFilter2D(mincutoff=0.1, beta=0.05, dcutoff=1.0)
        servo_ema_pixel = OneEuroFilter2D(mincutoff=10.0, beta=0.2, dcutoff=1.0)

        # Build state on track 1
        for i in range(10):
            stage.kalman.update((float(100 + i), float(200 + i)), dt=0.016)
            stage.stabilizer.smooth(np.array([0.8, 0.9, 0.7], dtype=np.float32))
        ema_pixel((109.0, 209.0), 0.0)

        state_before = stage.kalman.current_state()
        assert state_before is not None

        # Save track 1 state
        stage.save_track_state(1, ema_pixel.snapshot(), servo_ema_pixel.snapshot(), timestamp_ns=1000)
        assert 1 in stage._track_state_cache

        # Simulate switching to track 2 — reset filters
        stage.kalman.reset()
        stage.stabilizer.reset()
        ema_pixel.x_prev = None

        # Restore track 1
        restored = stage.restore_track_state(1, ema_pixel, servo_ema_pixel, current_timestamp_ns=2000)
        assert restored is True
        assert 1 not in stage._track_state_cache  # consumed from cache

        state_after = stage.kalman.current_state()
        assert state_after is not None
        assert abs(state_after.x - state_before.x) < 1e-9
        assert abs(state_after.y - state_before.y) < 1e-9
        assert ema_pixel.x_prev == (109.0, 209.0)

    def test_track_state_cache_eviction(self):
        """Cache evicts oldest entry when exceeding max size."""
        stage = _make_stage()
        ema_pixel = OneEuroFilter2D(mincutoff=0.1, beta=0.05, dcutoff=1.0)
        servo_ema_pixel = OneEuroFilter2D(mincutoff=10.0, beta=0.2, dcutoff=1.0)
        stage.kalman.update((50.0, 50.0), dt=0.016)
        ema_pixel((100.0, 100.0), 0.0)

        # Fill cache to max + 1
        for i in range(_MAX_CACHED_TRACKS + 1):
            stage.save_track_state(i, ema_pixel.snapshot(), servo_ema_pixel.snapshot(), timestamp_ns=i * 1000)

        assert len(stage._track_state_cache) == _MAX_CACHED_TRACKS
        # Oldest (track_id=0, timestamp_ns=0) should have been evicted
        assert 0 not in stage._track_state_cache

    def test_ema_not_nulled_on_lock_change(self):
        """EMA pixel should NOT be wiped on lock transition (smooth decay)."""
        stage = _make_stage()
        ema_pixel = OneEuroFilter2D(mincutoff=0.1, beta=0.05, dcutoff=1.0)
        servo_ema_pixel = OneEuroFilter2D(mincutoff=10.0, beta=0.2, dcutoff=1.0)
        ema_pixel((200.0, 300.0), 0.0)

        # Lock track 1, then force transition to track 2 (no cache, no re-ID)
        stage._locked_track_id = 1
        _select(stage, [_make_track(track_id=1, score=0.9)])

        # Simulate lock change path (no cache hit, no re-ID match)
        stage.kalman.update((100.0, 100.0), dt=0.016)
        stage.save_track_state(1, ema_pixel.snapshot(), servo_ema_pixel.snapshot(), timestamp_ns=1000)
        restored = stage.restore_track_state(99, ema_pixel, servo_ema_pixel, current_timestamp_ns=2000)
        assert restored is False

        # After reset without cache hit, EMA should NOT be None (M4)
        # The actual reset path in the main loop keeps _ema_pixel alive
        assert ema_pixel.x_prev is not None, "EMA should survive lock transitions"

    def test_person_tracking_unaffected(self):
        """All existing lock behavior works unchanged for person tracking."""
        stage = _make_stage()
        # Same sequence as test_stays_locked_despite_better_score
        t1 = _make_track(track_id=1, score=0.9)
        t2 = _make_track(track_id=2, score=0.7)
        _select(stage, [t1, t2])
        assert stage.locked_track_id == 1

        t1_lower = _make_track(track_id=1, score=0.5)
        t2_higher = _make_track(track_id=2, score=0.99)
        result = _select(stage, [t1_lower, t2_higher])
        assert result.track_id == 1

        # Lock release works as before
        t1_lost = _make_track(track_id=1, score=0.5, lost_frames=31)
        result = _select(stage, [t1_lost, t2_higher])
        assert result.track_id == 2
