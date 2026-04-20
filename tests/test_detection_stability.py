"""Phase 4: Detection Stability L1 tests — MAD outlier, stabilized centroid, temporal smoothing."""

from __future__ import annotations

import numpy as np
import pytest

from src.inference.postprocess import (
    KeypointStabilizer,
    _mad_filter_keypoints,
    compute_stabilized_centroid,
)
from src.shared.pose_schema import get_pose_schema
from src.shared.types import Detection

# --- MAD filter tests ---


@pytest.mark.unit
def test_mad_rejects_outlier_keypoint() -> None:
    # 4 clustered points + 1 extreme outlier
    points = np.array([[10, 10], [11, 11], [10, 11], [11, 10], [500, 500]], dtype=np.float32)
    confs = np.ones(5, dtype=np.float32)

    filtered_pts, filtered_confs = _mad_filter_keypoints(points, confs, scale=3.0)

    assert len(filtered_pts) == 4
    assert len(filtered_confs) == 4
    # Outlier (500, 500) should be removed
    assert not np.any(np.all(filtered_pts == [500, 500], axis=1))


@pytest.mark.unit
def test_mad_keeps_all_when_no_outliers() -> None:
    points = np.array([[10, 10], [11, 11], [10, 11], [11, 10]], dtype=np.float32)
    confs = np.ones(4, dtype=np.float32)

    filtered_pts, filtered_confs = _mad_filter_keypoints(points, confs)

    assert len(filtered_pts) == 4


@pytest.mark.unit
def test_mad_skips_when_too_few_points() -> None:
    points = np.array([[10, 10], [500, 500]], dtype=np.float32)
    confs = np.ones(2, dtype=np.float32)

    filtered_pts, filtered_confs = _mad_filter_keypoints(points, confs)

    assert len(filtered_pts) == 2  # no filtering with < 3 points


@pytest.mark.unit
def test_mad_handles_identical_points() -> None:
    points = np.array([[10, 10], [10, 10], [10, 10]], dtype=np.float32)
    confs = np.ones(3, dtype=np.float32)

    filtered_pts, filtered_confs = _mad_filter_keypoints(points, confs)

    assert len(filtered_pts) == 3  # MAD=0, keep all


@pytest.mark.unit
def test_mad_never_discards_everything() -> None:
    # All points are outliers relative to each other (spread out)
    points = np.array([[0, 0], [100, 0], [0, 100]], dtype=np.float32)
    confs = np.ones(3, dtype=np.float32)

    filtered_pts, filtered_confs = _mad_filter_keypoints(points, confs, scale=0.01)

    # Should keep all rather than discard everything
    assert len(filtered_pts) > 0


# --- KeypointStabilizer tests ---


@pytest.mark.unit
def test_stabilizer_first_frame_passes_through() -> None:
    stab = KeypointStabilizer(alpha=0.3)
    confs = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    smoothed = stab.smooth(confs)

    np.testing.assert_array_almost_equal(smoothed, confs)


@pytest.mark.unit
def test_stabilizer_smooths_second_frame() -> None:
    stab = KeypointStabilizer(alpha=0.3)
    frame1 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    frame2 = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    stab.smooth(frame1)
    smoothed = stab.smooth(frame2)

    # EMA: 0.3 * 0.0 + 0.7 * 1.0 = 0.7
    np.testing.assert_array_almost_equal(smoothed, [0.7, 0.7, 0.7])


@pytest.mark.unit
def test_stabilizer_converges_over_frames() -> None:
    stab = KeypointStabilizer(alpha=0.5)
    high = np.array([1.0], dtype=np.float32)
    low = np.array([0.0], dtype=np.float32)

    stab.smooth(high)
    for _ in range(20):
        result = stab.smooth(low)

    assert result[0] < 0.01  # should converge to 0


@pytest.mark.unit
def test_stabilizer_reset_clears_state() -> None:
    stab = KeypointStabilizer(alpha=0.3)
    stab.smooth(np.array([1.0, 1.0], dtype=np.float32))
    stab.smooth(np.array([0.5, 0.5], dtype=np.float32))

    stab.reset()
    fresh = stab.smooth(np.array([0.0, 0.0], dtype=np.float32))

    np.testing.assert_array_almost_equal(fresh, [0.0, 0.0])


@pytest.mark.unit
def test_stabilizer_snapshot_restore() -> None:
    """Snapshot preserves smoothed state; restore brings it back after reset."""
    stab = KeypointStabilizer(alpha=0.3)
    for _ in range(5):
        stab.smooth(np.array([0.8, 0.9, 0.7], dtype=np.float32))

    snap = stab.snapshot()
    assert snap is not None

    smoothed_before = snap.copy()

    stab.reset()
    assert stab.snapshot() is None

    stab.restore(snap)
    restored = stab.snapshot()
    np.testing.assert_array_almost_equal(restored, smoothed_before)


# --- compute_stabilized_centroid tests ---


def _make_detection_with_keypoints(
    kp_positions: list[tuple[float, float, float]],
    bbox: tuple[float, float, float, float] = (0, 0, 100, 100),
) -> Detection:
    num_kp = len(kp_positions)
    keypoints = np.zeros((num_kp, 3), dtype=np.float32)
    for i, (x, y, v) in enumerate(kp_positions):
        keypoints[i] = [x, y, v]
    return Detection(
        bbox=np.array(bbox, dtype=np.float32),
        score=0.9,
        keypoints=keypoints,
    )


@pytest.mark.unit
def test_stabilized_centroid_matches_original_without_outliers() -> None:
    pose_schema = get_pose_schema("human")
    # 5 head keypoints at consistent positions
    kps = [(0, 0, 0)] * 17
    kps[0] = (52.0, 58.0, 1.0)  # nose
    kps[1] = (50.0, 58.0, 1.0)  # l_eye
    kps[2] = (54.0, 58.0, 1.0)  # r_eye
    kps[3] = (48.0, 59.0, 1.0)  # l_ear
    kps[4] = (56.0, 59.0, 1.0)  # r_ear
    det = _make_detection_with_keypoints(kps, bbox=(40, 50, 60, 70))

    centroid = compute_stabilized_centroid(det, pose_schema)

    # Should be close to the weighted average of head keypoints
    assert 48.0 < centroid[0] < 56.0
    assert 57.0 < centroid[1] < 60.0


@pytest.mark.unit
def test_stabilized_centroid_rejects_outlier_keypoint() -> None:
    pose_schema = get_pose_schema("human")
    # 4 consistent head kps + 1 outlier ear
    kps = [(0, 0, 0)] * 17
    kps[0] = (50.0, 50.0, 1.0)  # nose
    kps[1] = (49.0, 50.0, 1.0)  # l_eye
    kps[2] = (51.0, 50.0, 1.0)  # r_eye
    kps[3] = (48.0, 51.0, 1.0)  # l_ear
    kps[4] = (300.0, 300.0, 1.0)  # r_ear — outlier!
    det = _make_detection_with_keypoints(kps, bbox=(30, 30, 70, 70))

    centroid = compute_stabilized_centroid(det, pose_schema)

    # Outlier should be rejected, centroid stays near head (50, 50)
    # With head_blend_alpha=0.85 and bbox_fallback_y_ratio=0.2, Y is blended
    # slightly toward top of bbox (38), so Y is slightly below 50
    assert 47.0 < centroid[0] < 53.0
    assert 46.0 < centroid[1] < 53.0


@pytest.mark.unit
def test_stabilized_centroid_falls_back_to_bbox_without_keypoints() -> None:
    det = Detection(
        bbox=np.array([10, 20, 30, 40], dtype=np.float32),
        score=0.9,
        keypoints=None,
    )
    pose_schema = get_pose_schema("human")

    centroid = compute_stabilized_centroid(det, pose_schema)

    # No keypoints → falls back to smart bbox position:
    # bbox_cx = (10+30)/2 = 20, bbox_ref_y = 20 + (40-20)*0.15 = 23
    assert centroid == pytest.approx((20.0, 23.0))


@pytest.mark.unit
def test_stabilized_centroid_with_stabilizer() -> None:
    pose_schema = get_pose_schema("human")
    stab = KeypointStabilizer(alpha=0.5)

    kps = [(0, 0, 0)] * 17
    kps[0] = (50.0, 50.0, 1.0)
    kps[1] = (49.0, 50.0, 1.0)
    kps[2] = (51.0, 50.0, 1.0)
    kps[3] = (48.0, 51.0, 1.0)
    kps[4] = (52.0, 51.0, 1.0)
    det = _make_detection_with_keypoints(kps, bbox=(30, 30, 70, 70))

    # First frame: stabilizer initializes
    c1 = compute_stabilized_centroid(det, pose_schema, stab)
    # Second frame: same detection, smoothed confidences
    c2 = compute_stabilized_centroid(det, pose_schema, stab)

    # Both should be similar since input is consistent
    assert abs(c1[0] - c2[0]) < 2.0
    assert abs(c1[1] - c2[1]) < 2.0
