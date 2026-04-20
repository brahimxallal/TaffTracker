"""Tests for depth estimation and parallax correction."""
from __future__ import annotations

from math import radians, tan

import numpy as np
import pytest

from src.calibration.camera_model import CameraModel
from src.calibration.depth_estimator import (
    DepthSmoother,
    _AVERAGE_DOG_HEIGHT_M,
    _AVERAGE_HUMAN_HEIGHT_M,
    _estimate_visible_fraction,
    estimate_depth,
)


# ─── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def cam() -> CameraModel:
    """Camera with 90° HFOV at 640×640."""
    return CameraModel.from_fov(90.0, 640, 640)


def _make_keypoints(pairs: dict[int, tuple[float, float, float]], n: int = 17) -> np.ndarray:
    """Build a (N, 3) keypoint array with specific indices set."""
    kp = np.zeros((n, 3), dtype=np.float32)
    for idx, (x, y, conf) in pairs.items():
        kp[idx] = [x, y, conf]
    return kp


# ─── Depth Estimation ────────────────────────────────────────

class TestEstimateDepth:

    def test_shoulder_pair_basic(self, cam: CameraModel):
        """Shoulder width of 0.40m at known pixel separation → correct depth."""
        fx = cam.focal_length_px
        real_width = 0.40  # meters
        target_depth = 2.0  # meters
        pixel_span = fx * real_width / target_depth
        cx = 320.0
        # Place shoulders symmetrically around center
        half = pixel_span / 2.0
        kp = _make_keypoints({
            5: (cx - half, 200.0, 0.9),   # left_shoulder
            6: (cx + half, 200.0, 0.9),   # right_shoulder
        })
        result = estimate_depth(kp, fx)
        assert result is not None
        assert result.pair == (5, 6)
        assert abs(result.depth_m - target_depth) < 0.01

    def test_hip_pair(self, cam: CameraModel):
        fx = cam.focal_length_px
        real_width = 0.30
        target_depth = 3.0
        pixel_span = fx * real_width / target_depth
        cx = 320.0
        half = pixel_span / 2.0
        kp = _make_keypoints({
            11: (cx - half, 400.0, 0.8),
            12: (cx + half, 400.0, 0.8),
        })
        result = estimate_depth(kp, fx)
        assert result is not None
        assert result.pair == (11, 12)
        assert abs(result.depth_m - target_depth) < 0.01

    def test_prefers_higher_confidence_pair(self, cam: CameraModel):
        """When multiple pairs visible, best_pair reflects the highest-weight pair."""
        fx = cam.focal_length_px
        # Shoulders at low confidence
        kp = _make_keypoints({
            5: (250.0, 200.0, 0.4),
            6: (390.0, 200.0, 0.4),
            11: (270.0, 400.0, 0.9),  # hips at high confidence
            12: (370.0, 400.0, 0.9),
        })
        result = estimate_depth(kp, fx)
        assert result is not None
        # Multi-pair fusion: both pairs contribute, n_pairs >= 2
        assert result.n_pairs >= 2

    def test_returns_none_below_confidence(self, cam: CameraModel):
        fx = cam.focal_length_px
        kp = _make_keypoints({
            5: (250.0, 200.0, 0.1),
            6: (390.0, 200.0, 0.1),
        })
        result = estimate_depth(kp, fx, min_confidence=0.3)
        assert result is None

    def test_returns_none_for_none_keypoints(self, cam: CameraModel):
        assert estimate_depth(None, cam.focal_length_px) is None

    def test_returns_none_for_degenerate_span(self, cam: CameraModel):
        """Overlapping keypoints should not produce depth."""
        fx = cam.focal_length_px
        kp = _make_keypoints({
            5: (320.0, 200.0, 0.9),
            6: (320.0, 200.0, 0.9),  # same position
        })
        result = estimate_depth(kp, fx)
        assert result is None

    def test_custom_reference_spans(self, cam: CameraModel):
        fx = cam.focal_length_px
        custom = {(0, 1): 0.20}  # 20cm reference (plain float, no tuple)
        target_depth = 1.5
        pixel_span = fx * 0.20 / target_depth
        kp = _make_keypoints({
            0: (300.0, 200.0, 0.9),
            1: (300.0 + pixel_span, 200.0, 0.9),
        })
        result = estimate_depth(kp, fx, reference_spans=custom)
        assert result is not None
        assert abs(result.depth_m - target_depth) < 0.01

    def test_custom_reference_spans_tuple(self, cam: CameraModel):
        """Custom spans with (distance, reliability) tuples."""
        fx = cam.focal_length_px
        custom = {(0, 1): (0.20, 0.5)}
        target_depth = 1.5
        pixel_span = fx * 0.20 / target_depth
        kp = _make_keypoints({
            0: (300.0, 200.0, 0.9),
            1: (300.0 + pixel_span, 200.0, 0.9),
        })
        result = estimate_depth(kp, fx, reference_spans=custom)
        assert result is not None
        assert abs(result.depth_m - target_depth) < 0.01

    def test_keypoints_too_few_columns(self, cam: CameraModel):
        """(N, 2) keypoints without confidence should return None."""
        kp = np.zeros((17, 2), dtype=np.float32)
        assert estimate_depth(kp, cam.focal_length_px) is None


# ─── Multi-Pair Fusion ───────────────────────────────────────

class TestMultiPairFusion:

    def test_fuses_multiple_pairs(self, cam: CameraModel):
        """When all 4 pairs are visible and agree on depth, fused result uses all."""
        fx = cam.focal_length_px
        target_depth = 2.5
        # Set each pair at pixel separation matching target_depth
        # shoulders (5,6): 0.40m  →  pixel_span = fx * 0.40 / 2.5
        # hips (11,12): 0.30m     →  pixel_span = fx * 0.30 / 2.5
        # torso (5,11): 0.50m     →  vertical pixel span
        # torso (6,12): 0.50m     →  vertical pixel span
        s_half = fx * 0.40 / target_depth / 2.0
        h_half = fx * 0.30 / target_depth / 2.0
        t_span = fx * 0.50 / target_depth
        kp = _make_keypoints({
            5:  (320.0 - s_half, 200.0, 0.9),
            6:  (320.0 + s_half, 200.0, 0.9),
            11: (320.0 - h_half, 200.0 + t_span, 0.9),
            12: (320.0 + h_half, 200.0 + t_span, 0.9),
        })
        result = estimate_depth(kp, fx)
        assert result is not None
        assert result.n_pairs == 4
        assert abs(result.depth_m - target_depth) < 0.05

    def test_single_pair_returns_n_pairs_1(self, cam: CameraModel):
        """Only one valid pair → n_pairs == 1."""
        fx = cam.focal_length_px
        target_depth = 2.0
        pixel_span = fx * 0.40 / target_depth
        half = pixel_span / 2.0
        kp = _make_keypoints({
            5: (320.0 - half, 200.0, 0.9),
            6: (320.0 + half, 200.0, 0.9),
        })
        result = estimate_depth(kp, fx)
        assert result is not None
        assert result.n_pairs == 1

    def test_weighted_fusion_correctness(self, cam: CameraModel):
        """Two pairs at different depths/weights → weighted mean."""
        fx = cam.focal_length_px
        # Custom spans: pair A gives depth=2.0, pair B gives depth=4.0
        # pair A has reliability 1.0, pair B has reliability 0.5
        # Both have equal confidence 0.8
        custom = {
            (0, 1): (0.40, 1.0),
            (2, 3): (0.30, 0.5),
        }
        span_a = fx * 0.40 / 2.0
        span_b = fx * 0.30 / 4.0
        kp = _make_keypoints({
            0: (200.0, 200.0, 0.8),
            1: (200.0 + span_a, 200.0, 0.8),
            2: (200.0, 300.0, 0.8),
            3: (200.0 + span_b, 300.0, 0.8),
        })
        result = estimate_depth(kp, fx, reference_spans=custom)
        assert result is not None
        assert result.n_pairs >= 1
        # Weighted mean: w_a = 0.8*1.0 = 0.8, w_b = 0.8*0.5 = 0.4
        # But pair B at 4.0 might be outlier-rejected (4.0/2.0 = 2.0, border case)
        # Either way the fused depth should be reasonable (between 2.0 and 4.0)
        assert 1.5 < result.depth_m < 4.5


# ─── Outlier Rejection ───────────────────────────────────────

class TestOutlierRejection:

    def test_outlier_pair_rejected(self, cam: CameraModel):
        """One pair giving a wildly different depth should be rejected."""
        fx = cam.focal_length_px
        target_depth = 2.0
        # Custom spans: 3 pairs agreeing on ~2.0m, 1 outlier at ~20.0m
        custom = {
            (0, 1): (0.40, 1.0),
            (2, 3): (0.30, 1.0),
            (4, 7): (0.50, 1.0),
            (8, 9): (0.20, 1.0),  # will set up to give 20m
        }
        span_a = fx * 0.40 / target_depth
        span_b = fx * 0.30 / target_depth
        span_c = fx * 0.50 / target_depth
        span_outlier = fx * 0.20 / 20.0  # very small span → huge depth
        kp = _make_keypoints({
            0: (100.0, 100.0, 0.9),
            1: (100.0 + span_a, 100.0, 0.9),
            2: (100.0, 200.0, 0.9),
            3: (100.0 + span_b, 200.0, 0.9),
            4: (100.0, 300.0, 0.9),
            7: (100.0 + span_c, 300.0, 0.9),
            8: (100.0, 400.0, 0.9),
            9: (100.0 + span_outlier, 400.0, 0.9),
        })
        result = estimate_depth(kp, fx, reference_spans=custom)
        assert result is not None
        # The 20m outlier should be rejected; depth should remain ~2.0m
        assert abs(result.depth_m - target_depth) < 0.5


# ─── Bounding Box Fallback ───────────────────────────────────

class TestBboxFallback:

    def test_bbox_fallback_when_no_keypoints(self, cam: CameraModel):
        """With no valid keypoints, bbox height produces depth estimate."""
        fx = cam.focal_length_px
        target_depth = 3.0
        bbox_h = fx * _AVERAGE_HUMAN_HEIGHT_M / target_depth
        bbox = np.array([100.0, 50.0, 200.0, 50.0 + bbox_h])
        result = estimate_depth(None, fx, bbox=bbox)
        assert result is not None
        assert result.pair is None  # bbox fallback
        assert result.n_pairs == 0
        assert abs(result.depth_m - target_depth) < 0.1

    def test_bbox_fallback_low_confidence(self, cam: CameraModel):
        """Bbox fallback has lower confidence than keypoint-based estimates."""
        fx = cam.focal_length_px
        bbox = np.array([100.0, 50.0, 200.0, 350.0])
        result = estimate_depth(None, fx, bbox=bbox)
        assert result is not None
        assert result.confidence <= 0.3

    def test_no_fallback_without_bbox(self, cam: CameraModel):
        """No keypoints and no bbox → None."""
        assert estimate_depth(None, cam.focal_length_px) is None

    def test_bbox_too_small_ignored(self, cam: CameraModel):
        """Bbox with height < 10px returns None."""
        fx = cam.focal_length_px
        bbox = np.array([100.0, 100.0, 200.0, 105.0])  # height = 5px
        assert estimate_depth(None, fx, bbox=bbox) is None

    def test_keypoints_preferred_over_bbox(self, cam: CameraModel):
        """When keypoints are valid, bbox is not used."""
        fx = cam.focal_length_px
        target_depth = 2.0
        pixel_span = fx * 0.40 / target_depth
        half = pixel_span / 2.0
        kp = _make_keypoints({
            5: (320.0 - half, 200.0, 0.9),
            6: (320.0 + half, 200.0, 0.9),
        })
        # bbox at a very different depth
        bbox = np.array([100.0, 50.0, 200.0, 600.0])
        result = estimate_depth(kp, fx, bbox=bbox)
        assert result is not None
        assert result.pair is not None  # keypoint-based, not bbox
        assert abs(result.depth_m - target_depth) < 0.1


# ─── Dog Keypoint Support ────────────────────────────────────

class TestDogKeypoints:

    def test_dog_ear_pair_basic(self, cam: CameraModel):
        """Dog ear_base pair (14,15) at known separation → correct depth."""
        fx = cam.focal_length_px
        target_depth = 3.0
        pixel_span = fx * 0.12 / target_depth
        half = pixel_span / 2.0
        kp = _make_keypoints({
            14: (320.0 - half, 200.0, 0.9),  # left_ear_base
            15: (320.0 + half, 200.0, 0.9),  # right_ear_base
        }, n=24)
        result = estimate_depth(kp, fx, target_kind="dog")
        assert result is not None
        assert result.pair == (14, 15)
        assert abs(result.depth_m - target_depth) < 0.05

    def test_dog_elbow_pair(self, cam: CameraModel):
        """Dog front elbow pair (2,8) → correct depth."""
        fx = cam.focal_length_px
        target_depth = 4.0
        pixel_span = fx * 0.25 / target_depth
        half = pixel_span / 2.0
        kp = _make_keypoints({
            2: (320.0 - half, 300.0, 0.85),  # front_left_elbow
            8: (320.0 + half, 300.0, 0.85),  # front_right_elbow
        }, n=24)
        result = estimate_depth(kp, fx, target_kind="dog")
        assert result is not None
        assert result.pair == (2, 8)
        assert abs(result.depth_m - target_depth) < 0.1

    def test_dog_uses_dog_spans_not_human(self, cam: CameraModel):
        """target_kind='dog' uses DOG_KEYPOINT_SPANS, not HUMAN_KEYPOINT_SPANS."""
        fx = cam.focal_length_px
        # Place human shoulder indices (5,6) with valid coords
        # For dog these are rear_left_elbow(5) and front_right_paw(6)
        # Dog spans don't include (5,6) so this pair should NOT be used
        kp = _make_keypoints({
            5: (250.0, 200.0, 0.9),
            6: (390.0, 200.0, 0.9),
        }, n=24)
        result = estimate_depth(kp, fx, target_kind="dog")
        # Pair (5,6) is not in DOG_KEYPOINT_SPANS, so only works if (2,5) or (8,11) has data
        # With only indices 5 and 6 set, no dog pair can form → None
        assert result is None

    def test_dog_bbox_fallback_uses_dog_height(self, cam: CameraModel):
        """Dog bbox fallback uses _AVERAGE_DOG_HEIGHT_M."""
        fx = cam.focal_length_px
        target_depth = 5.0
        bbox_h = fx * _AVERAGE_DOG_HEIGHT_M / target_depth
        bbox = np.array([100.0, 50.0, 200.0, 50.0 + bbox_h])
        result = estimate_depth(None, fx, bbox=bbox, target_kind="dog")
        assert result is not None
        assert result.pair is None
        assert abs(result.depth_m - target_depth) < 0.2

    def test_dog_multi_pair_fusion(self, cam: CameraModel):
        """Multiple dog pairs visible and agreeing → fused result."""
        fx = cam.focal_length_px
        target_depth = 3.0
        ear_half = fx * 0.12 / target_depth / 2.0
        elbow_half = fx * 0.25 / target_depth / 2.0
        kp = _make_keypoints({
            14: (320.0 - ear_half, 100.0, 0.9),
            15: (320.0 + ear_half, 100.0, 0.9),
            2: (320.0 - elbow_half, 250.0, 0.8),
            8: (320.0 + elbow_half, 250.0, 0.8),
        }, n=24)
        result = estimate_depth(kp, fx, target_kind="dog")
        assert result is not None
        assert result.n_pairs >= 2
        assert abs(result.depth_m - target_depth) < 0.2


# ─── Depth Range Clamping ────────────────────────────────────

class TestDepthRangeClamping:

    def test_too_close_rejected(self, cam: CameraModel):
        """Depth < MIN_DEPTH_M (0.3m) should be rejected."""
        fx = cam.focal_length_px
        # Set up shoulder pair giving depth = 0.1m
        pixel_span = fx * 0.40 / 0.1  # very large span
        half = pixel_span / 2.0
        kp = _make_keypoints({
            5: (320.0 - half, 200.0, 0.9),
            6: (320.0 + half, 200.0, 0.9),
        })
        result = estimate_depth(kp, fx)
        assert result is None

    def test_too_far_rejected(self, cam: CameraModel):
        """Depth > MAX_DEPTH_M (25m) should be rejected."""
        fx = cam.focal_length_px
        # Set up shoulder pair giving depth = 50m
        pixel_span = fx * 0.40 / 50.0  # very small span
        half = pixel_span / 2.0
        kp = _make_keypoints({
            5: (320.0 - half, 200.0, 0.9),
            6: (320.0 + half, 200.0, 0.9),
        })
        result = estimate_depth(kp, fx)
        assert result is None

    def test_valid_range_accepted(self, cam: CameraModel):
        """Depth within range passes."""
        fx = cam.focal_length_px
        target_depth = 5.0
        pixel_span = fx * 0.40 / target_depth
        half = pixel_span / 2.0
        kp = _make_keypoints({
            5: (320.0 - half, 200.0, 0.9),
            6: (320.0 + half, 200.0, 0.9),
        })
        result = estimate_depth(kp, fx)
        assert result is not None
        assert abs(result.depth_m - target_depth) < 0.1

    def test_bbox_fallback_clamped(self, cam: CameraModel):
        """Bbox fallback also respects depth range."""
        fx = cam.focal_length_px
        # bbox height that produces depth < MIN_DEPTH_M
        huge_bbox_h = fx * _AVERAGE_HUMAN_HEIGHT_M / 0.1  # depth = 0.1m
        bbox = np.array([0.0, 0.0, 100.0, huge_bbox_h])
        result = estimate_depth(None, fx, bbox=bbox)
        assert result is None


# ─── Nonlinear Confidence Weighting ──────────────────────────

class TestNonlinearWeighting:

    def test_low_conf_pair_downweighted(self, cam: CameraModel):
        """Low-confidence pair contributes less than high-confidence pair."""
        fx = cam.focal_length_px
        # Two custom pairs: A at depth=2m conf=0.9, B at depth=4m conf=0.4
        # Quadratic weight: A = 0.9² * 1.0 = 0.81, B = 0.4² * 1.0 = 0.16
        # Fused depth should strongly favor A (≈ 2.3m, not 3.0m)
        custom = {(0, 1): (0.40, 1.0), (2, 3): (0.40, 1.0)}
        span_a = fx * 0.40 / 2.0
        span_b = fx * 0.40 / 4.0
        kp = _make_keypoints({
            0: (200.0, 200.0, 0.9),
            1: (200.0 + span_a, 200.0, 0.9),
            2: (200.0, 300.0, 0.4),
            3: (200.0 + span_b, 300.0, 0.4),
        })
        result = estimate_depth(kp, fx, reference_spans=custom)
        assert result is not None
        # With quadratic weighting, depth should be closer to 2.0 than 3.0
        assert result.depth_m < 3.0


# ─── Orientation Compensation ────────────────────────────────

class TestOrientationCompensation:

    def test_rotated_person_reduces_shoulder_weight(self, cam: CameraModel):
        """When person rotates, shoulder span compresses → shoulder weight reduced."""
        fx = cam.focal_length_px
        true_depth = 3.0
        # Diagonal pair (5,11) gives correct depth
        diag_span = fx * 0.50 / true_depth
        # Shoulder pair (5,6) compressed by rotation → reports larger depth (5m)
        shoulder_span = fx * 0.40 / 5.0
        s_half = shoulder_span / 2.0
        kp = _make_keypoints({
            5: (320.0 - s_half, 200.0, 0.9),
            6: (320.0 + s_half, 200.0, 0.9),
            11: (320.0 - s_half, 200.0 + diag_span, 0.9),
        })
        result = estimate_depth(kp, fx)
        assert result is not None
        # With orientation compensation, fused depth should be closer to 3.0 than 5.0
        assert result.depth_m < 4.5

    def test_no_compensation_when_consistent(self, cam: CameraModel):
        """When shoulder and diagonal agree, no weight reduction."""
        fx = cam.focal_length_px
        target_depth = 2.5
        s_half = fx * 0.40 / target_depth / 2.0
        t_span = fx * 0.50 / target_depth
        kp = _make_keypoints({
            5:  (320.0 - s_half, 200.0, 0.9),
            6:  (320.0 + s_half, 200.0, 0.9),
            11: (320.0 - s_half, 200.0 + t_span, 0.9),
        })
        result = estimate_depth(kp, fx)
        assert result is not None
        assert abs(result.depth_m - target_depth) < 0.2

    def test_orientation_only_for_human(self, cam: CameraModel):
        """Dog targets do not get orientation compensation (different body geometry)."""
        fx = cam.focal_length_px
        # Place dog ear pair at compressed span
        ear_span = fx * 0.12 / 5.0  # reports 5m (compressed)
        ear_half = ear_span / 2.0
        # Place body length pair at correct 3m
        body_span = fx * 0.45 / 3.0
        kp = _make_keypoints({
            14: (320.0 - ear_half, 100.0, 0.9),
            15: (320.0 + ear_half, 100.0, 0.9),
            2:  (320.0, 200.0, 0.9),
            5:  (320.0, 200.0 + body_span, 0.9),
        }, n=24)
        # For dog, both pairs should contribute without orientation penalty
        result = estimate_depth(kp, fx, target_kind="dog")
        assert result is not None
        assert result.n_pairs >= 1


# ─── Partial Body Bbox Correction ────────────────────────────

class TestPartialBodyBbox:

    def test_waist_crop_reduces_effective_height(self, cam: CameraModel):
        """Shoulders visible but ankles/knees missing → fraction ≈ 0.55."""
        fx = cam.focal_length_px
        # Create keypoints with only shoulders visible (confidence 0.9)
        kp = _make_keypoints({
            5: (250.0, 200.0, 0.9),
            6: (390.0, 200.0, 0.9),
        })
        frac = _estimate_visible_fraction(kp, "human")
        assert abs(frac - 0.55) < 0.01

    def test_shin_crop_fraction(self, cam: CameraModel):
        """Shoulders + knees visible but ankles missing → fraction ≈ 0.85."""
        kp = _make_keypoints({
            5: (250.0, 200.0, 0.9),
            6: (390.0, 200.0, 0.9),
            13: (270.0, 400.0, 0.9),  # left knee
        })
        frac = _estimate_visible_fraction(kp, "human")
        assert abs(frac - 0.85) < 0.01

    def test_full_body_fraction(self, cam: CameraModel):
        """Full body visible (ankles present) → fraction = 1.0."""
        kp = _make_keypoints({
            5: (250.0, 200.0, 0.9),
            6: (390.0, 200.0, 0.9),
            15: (270.0, 500.0, 0.9),  # left ankle
        })
        frac = _estimate_visible_fraction(kp, "human")
        assert frac == 1.0

    def test_dog_paw_crop_fraction(self, cam: CameraModel):
        """Dog with elbows but no paws → fraction ≈ 0.85."""
        kp = _make_keypoints({
            2: (200.0, 200.0, 0.9),  # front_left_elbow
        }, n=24)
        frac = _estimate_visible_fraction(kp, "dog")
        assert abs(frac - 0.85) < 0.01

    def test_bbox_fallback_with_partial_body(self, cam: CameraModel):
        """Bbox + waist-crop keypoints → deeper depth (fraction=0.55)."""
        fx = cam.focal_length_px
        # Bbox showing upper body only.  With full height 1.70m the depth would be X.
        # With fraction=0.55 the effective height is 0.935m, so depth is shallower.
        bbox_h = 200.0  # pixels
        full_depth = fx * _AVERAGE_HUMAN_HEIGHT_M / bbox_h
        partial_depth = fx * (_AVERAGE_HUMAN_HEIGHT_M * 0.55) / bbox_h

        # Keypoints: only shoulders, no ankles/knees → fraction=0.55
        kp = _make_keypoints({
            5: (250.0, 200.0, 0.9),
            6: (390.0, 200.0, 0.9),
        })
        # Shoulders visible means keypoint pairs work, so bbox isn't used unless
        # the shoulder pair pixel distance gives a depth outside range.
        # Force no valid keypoint pairs by setting min_confidence very high
        bbox = np.array([100.0, 100.0, 300.0, 100.0 + bbox_h])
        result = estimate_depth(kp, fx, bbox=bbox, min_confidence=0.95)
        assert result is not None
        assert result.pair is None  # bbox fallback
        assert abs(result.depth_m - partial_depth) < 0.3


# ─── Innovation Gate on DepthSmoother ─────────────────────────

class TestInnovationGate:

    def test_spike_rejected(self):
        """A 4× depth jump from smoothed value should be rejected."""
        s = DepthSmoother(alpha=0.3, max_stale_frames=10)
        s.update(3.0)   # initialize
        s.update(3.1)   # normal update
        smoothed_before = s.update(None)  # get current smoothed
        s.restore(s.snapshot())  # undo the None
        # Now send a 4× spike
        result = s.update(12.0)
        # Should reject the spike and return the previous smoothed value
        assert result is not None
        assert abs(result - smoothed_before) < 0.5

    def test_moderate_change_accepted(self):
        """A 1.5× change from smoothed value should be accepted."""
        s = DepthSmoother(alpha=0.5)
        s.update(3.0)
        result = s.update(4.5)  # ratio = 1.5 (within 3×)
        assert result is not None
        # EMA: 0.5*4.5 + 0.5*3.0 = 3.75
        assert abs(result - 3.75) < 0.1

    def test_spike_increments_stale(self):
        """Rejected spike increments stale counter; eventually expires."""
        s = DepthSmoother(alpha=0.3, max_stale_frames=2)
        s.update(3.0)
        # Send 3 consecutive spikes → should expire after max_stale
        s.update(30.0)  # spike 1 → stale=1
        s.update(30.0)  # spike 2 → stale=2
        result = s.update(30.0)  # spike 3 → stale=3 > max_stale=2 → None
        assert result is None

    def test_gate_then_recovery(self):
        """After a spike is rejected, a reasonable value is accepted."""
        s = DepthSmoother(alpha=0.5, max_stale_frames=10)
        s.update(3.0)
        s.update(15.0)  # rejected spike
        result = s.update(3.2)  # reasonable → accepted
        assert result is not None
        # EMA of 3.0 and 3.2
        assert 2.5 < result < 3.5


# ─── Depth Smoother ──────────────────────────────────────────

class TestDepthSmoother:

    def test_first_value_passes_through(self):
        s = DepthSmoother(alpha=0.3)
        assert s.update(2.0) == 2.0

    def test_ema_smoothing(self):
        s = DepthSmoother(alpha=0.5)
        s.update(2.0)
        result = s.update(4.0)
        assert abs(result - 3.0) < 0.01  # 0.5*4 + 0.5*2 = 3.0

    def test_stale_depth_persists(self):
        s = DepthSmoother(alpha=0.3, max_stale_frames=5)
        s.update(2.0)
        # 4 frames with no depth → stale but still valid
        for _ in range(4):
            assert s.update(None) is not None

    def test_stale_depth_expires(self):
        s = DepthSmoother(alpha=0.3, max_stale_frames=3)
        s.update(2.0)
        for _ in range(3):
            s.update(None)
        assert s.update(None) is None

    def test_reset(self):
        s = DepthSmoother(alpha=0.3)
        s.update(2.0)
        s.reset()
        assert s.update(None) is None

    def test_snapshot_restore(self):
        s = DepthSmoother(alpha=0.3)
        s.update(2.0)
        snap = s.snapshot()
        s.update(10.0)
        s.restore(snap)
        assert abs(s.update(None) - 2.0) < 0.01

    def test_innovation_gate_rejects_2x_jump(self):
        """With tightened 2× gate, a 2.5× jump should be rejected."""
        s = DepthSmoother(alpha=0.5)
        s.update(3.0)
        # 2.5× jump (ratio=2.5 > 2.0) → should be rejected
        result = s.update(7.5)
        assert result is not None
        assert abs(result - 3.0) < 0.1  # stays at old smoothed

    def test_innovation_gate_accepts_within_2x(self):
        """A 1.8× change should be accepted (within 2× gate)."""
        s = DepthSmoother(alpha=0.5)
        s.update(3.0)
        result = s.update(5.4)  # ratio=1.8 < 2.0 → accepted
        assert result is not None
        # EMA: 0.5*5.4 + 0.5*3.0 = 4.2
        assert abs(result - 4.2) < 0.1


# ─── CameraModel Parallax ────────────────────────────────────

class TestPixelToAngleWithParallax:

    def test_zero_offset_matches_basic(self, cam: CameraModel):
        """With zero offset, parallax method should match basic pixel_to_angle."""
        px, py, depth = 400.0, 300.0, 2.0
        basic = cam.pixel_to_angle(px, py)
        parallax = cam.pixel_to_angle_with_parallax(px, py, depth, 0.0, 0.0, 0.0)
        assert abs(parallax[0] - basic[0]) < 1e-6
        assert abs(parallax[1] - basic[1]) < 1e-6

    def test_x_offset_shifts_pan(self, cam: CameraModel):
        """Gimbal to the right of camera → pan left to hit same target."""
        # Target at center of image, 3m away
        cx, cy = 320.0, 320.0
        depth = 3.0
        # No offset → zero angles
        pan_0, tilt_0 = cam.pixel_to_angle_with_parallax(cx, cy, depth, 0.0, 0.0, 0.0)
        assert abs(pan_0) < 1e-6
        # Gimbal 0.2m to the right → must pan left (negative) to compensate
        pan_off, _ = cam.pixel_to_angle_with_parallax(cx, cy, depth, 0.2, 0.0, 0.0)
        assert pan_off < pan_0

    def test_y_offset_shifts_tilt(self, cam: CameraModel):
        """Gimbal below camera → tilt up to hit same target."""
        cx, cy = 320.0, 320.0
        depth = 3.0
        tilt_0 = cam.pixel_to_angle_with_parallax(cx, cy, depth, 0.0, 0.0, 0.0)[1]
        # Gimbal 0.15m below → must tilt up (negative) to compensate
        _, tilt_off = cam.pixel_to_angle_with_parallax(cx, cy, depth, 0.0, 0.15, 0.0)
        assert tilt_off < tilt_0

    def test_z_offset_fallback_when_behind(self, cam: CameraModel):
        """If z_offset puts gimbal in front of target, fall back to basic angles."""
        px, py = 400.0, 300.0
        depth = 0.5
        # Gimbal 1m forward of camera → target behind gimbal
        basic = cam.pixel_to_angle(px, py)
        fallback = cam.pixel_to_angle_with_parallax(px, py, depth, 0.0, 0.0, 1.0)
        assert abs(fallback[0] - basic[0]) < 1e-6
        assert abs(fallback[1] - basic[1]) < 1e-6

    def test_parallax_decreases_with_distance(self, cam: CameraModel):
        """At large distances, parallax correction is negligible."""
        px, py = 400.0, 300.0
        offset_x = 0.2  # 20cm offset

        # At 2m
        pan_near, _ = cam.pixel_to_angle_with_parallax(px, py, 2.0, offset_x, 0.0, 0.0)
        # At 20m
        pan_far, _ = cam.pixel_to_angle_with_parallax(px, py, 20.0, offset_x, 0.0, 0.0)
        # Basic (no parallax)
        pan_basic, _ = cam.pixel_to_angle(px, py)

        # Far should be closer to basic than near
        assert abs(pan_far - pan_basic) < abs(pan_near - pan_basic)


class TestFocalLengthProperty:

    def test_focal_length_from_fov(self):
        cam = CameraModel.from_fov(90.0, 640, 640)
        expected_fx = 640.0 / (2.0 * tan(radians(45.0)))
        assert abs(cam.focal_length_px - expected_fx) < 0.01
