"""Tests for BoTSORT tracker (ByteTrack + Camera Motion Compensation)."""

from __future__ import annotations

import cv2
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


def _prime_cmc_state(tracker: BoTSORT, shape: tuple[int, int] = (16, 16)) -> np.ndarray:
    """Prime the tracker so _estimate_affine can run to its last-stage clamp.

    Seeds _prev_gray with a same-shape frame and stocks _prev_pts with enough
    points to bypass the early-exit guards.
    """
    tracker._prev_gray = np.zeros(shape, dtype=np.uint8)
    tracker._prev_pts = np.array(
        [[[1.0, 1.0]], [[5.0, 5.0]], [[10.0, 10.0]], [[14.0, 14.0]]],
        dtype=np.float32,
    )
    tracker._frames_since_feature_extract = 0
    return np.zeros(shape, dtype=np.uint8)


@pytest.mark.unit
def test_cmc_clamp_rejects_large_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A warp whose translation exceeds cmc_max_translation_px must be rejected."""
    tracker = BoTSORT(cmc_enabled=True, cmc_downscale=1.0, cmc_max_translation_px=50.0)
    gray = _prime_cmc_state(tracker)

    # Patch cv2 calls so _estimate_affine runs end-to-end with a bogus translation.
    def fake_lk(prev, curr, pts, _next, **_kwargs):
        n = len(pts)
        status = np.ones((n, 1), dtype=np.uint8)
        err = np.zeros((n, 1), dtype=np.float32)
        return pts.copy(), status, err

    def fake_affine(_src, _dst, **_kwargs):
        # Return a warp with a translation that exceeds the 50 px clamp.
        warp = np.array([[1.0, 0.0, 200.0], [0.0, 1.0, 5.0]], dtype=np.float64)
        inliers = np.ones((4, 1), dtype=np.uint8)
        return warp, inliers

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fake_lk)
    monkeypatch.setattr(cv2, "estimateAffinePartial2D", fake_affine)

    assert tracker._estimate_affine(gray) is None


@pytest.mark.unit
def test_cmc_clamp_accepts_small_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A warp within the translation threshold must pass through unchanged."""
    tracker = BoTSORT(cmc_enabled=True, cmc_downscale=1.0, cmc_max_translation_px=50.0)
    gray = _prime_cmc_state(tracker)

    def fake_lk(prev, curr, pts, _next, **_kwargs):
        n = len(pts)
        return pts.copy(), np.ones((n, 1), dtype=np.uint8), np.zeros((n, 1), dtype=np.float32)

    def fake_affine(_src, _dst, **_kwargs):
        warp = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, -8.0]], dtype=np.float64)
        return warp, np.ones((4, 1), dtype=np.uint8)

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fake_lk)
    monkeypatch.setattr(cv2, "estimateAffinePartial2D", fake_affine)

    warp = tracker._estimate_affine(gray)
    assert warp is not None
    np.testing.assert_allclose(warp[0, 2], 10.0)
    np.testing.assert_allclose(warp[1, 2], -8.0)


@pytest.mark.unit
def test_cmc_clamp_rejects_large_ty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clamp must trip on either axis, not only x."""
    tracker = BoTSORT(cmc_enabled=True, cmc_downscale=1.0, cmc_max_translation_px=50.0)
    gray = _prime_cmc_state(tracker)

    def fake_lk(prev, curr, pts, _next, **_kwargs):
        n = len(pts)
        return pts.copy(), np.ones((n, 1), dtype=np.uint8), np.zeros((n, 1), dtype=np.float32)

    def fake_affine(_src, _dst, **_kwargs):
        warp = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -300.0]], dtype=np.float64)
        return warp, np.ones((4, 1), dtype=np.uint8)

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fake_lk)
    monkeypatch.setattr(cv2, "estimateAffinePartial2D", fake_affine)

    assert tracker._estimate_affine(gray) is None


@pytest.mark.unit
def test_cmc_clamp_disabled_when_threshold_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting cmc_max_translation_px=0 turns the safety clamp off."""
    tracker = BoTSORT(cmc_enabled=True, cmc_downscale=1.0, cmc_max_translation_px=0.0)
    gray = _prime_cmc_state(tracker)

    def fake_lk(prev, curr, pts, _next, **_kwargs):
        n = len(pts)
        return pts.copy(), np.ones((n, 1), dtype=np.uint8), np.zeros((n, 1), dtype=np.float32)

    def fake_affine(_src, _dst, **_kwargs):
        warp = np.array([[1.0, 0.0, 500.0], [0.0, 1.0, 500.0]], dtype=np.float64)
        return warp, np.ones((4, 1), dtype=np.uint8)

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fake_lk)
    monkeypatch.setattr(cv2, "estimateAffinePartial2D", fake_affine)

    warp = tracker._estimate_affine(gray)
    assert warp is not None
    np.testing.assert_allclose(warp[0, 2], 500.0)
    np.testing.assert_allclose(warp[1, 2], 500.0)


@pytest.mark.unit
def test_cmc_clamp_applies_after_downscale_rescale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clamp comparison must happen in full-resolution px, after the 1/scale rescale."""
    # Threshold = 50 px full-res; downscale = 0.5; cv2 returns tx=40 at half-res,
    # which becomes tx=80 full-res → must be rejected.
    tracker = BoTSORT(cmc_enabled=True, cmc_downscale=0.5, cmc_max_translation_px=50.0)
    gray = _prime_cmc_state(tracker)

    def fake_lk(prev, curr, pts, _next, **_kwargs):
        n = len(pts)
        return pts.copy(), np.ones((n, 1), dtype=np.uint8), np.zeros((n, 1), dtype=np.float32)

    def fake_affine(_src, _dst, **_kwargs):
        # Translation at half-res is well under 50, but *2 rescale → 80 full-res.
        warp = np.array([[1.0, 0.0, 40.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        return warp, np.ones((4, 1), dtype=np.uint8)

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fake_lk)
    monkeypatch.setattr(cv2, "estimateAffinePartial2D", fake_affine)

    assert tracker._estimate_affine(gray) is None


@pytest.mark.unit
def test_cmc_clamp_default_threshold_is_positive() -> None:
    """Sanity check: default keeps the safety clamp active."""
    tracker = BoTSORT()
    assert tracker._cmc_max_translation_px > 0.0


@pytest.mark.unit
def test_cmc_clamp_negative_threshold_is_clipped_to_zero() -> None:
    """Negative thresholds are normalized to 0 (i.e., clamp disabled)."""
    tracker = BoTSORT(cmc_max_translation_px=-5.0)
    assert tracker._cmc_max_translation_px == 0.0


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
