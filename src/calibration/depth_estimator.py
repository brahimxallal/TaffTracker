"""Estimate target depth from keypoint span ratios using pinhole geometry.

Measures pixel distance between stable skeleton node pairs (e.g., shoulder-to-shoulder)
and divides by known anthropometric widths to recover depth via:

    Z = f * d_real / d_pixel

Uses confidence-weighted multi-pair fusion with outlier rejection,
orientation compensation, and partial-body bounding-box fallback.
Supports both human (COCO-17) and dog (Enhanced-24) keypoint schemas.
Includes temporal EMA smoothing with innovation gating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Anthropometric reference spans
# ---------------------------------------------------------------------------
# Format: (idx_a, idx_b) → (real_distance_m, reliability_weight)

# COCO-17 human keypoint pairs
HUMAN_KEYPOINT_SPANS: dict[tuple[int, int], tuple[float, float]] = {
    (5, 6): (0.40, 1.0),  # left_shoulder ↔ right_shoulder (most stable)
    (11, 12): (0.30, 0.6),  # left_hip ↔ right_hip (compresses under rotation)
    (5, 11): (0.50, 0.8),  # left_shoulder ↔ left_hip (torso height)
    (6, 12): (0.50, 0.8),  # right_shoulder ↔ right_hip (torso height)
}

# Enhanced-24 dog keypoint pairs (indices from pose_schema.py DOG_KEYPOINT_NAMES)
DOG_KEYPOINT_SPANS: dict[tuple[int, int], tuple[float, float]] = {
    (14, 15): (0.12, 1.0),  # left_ear_base ↔ right_ear_base (most stable, front-facing)
    (20, 21): (0.08, 0.8),  # left_eye ↔ right_eye
    (2, 8): (0.25, 0.9),  # front_left_elbow ↔ front_right_elbow (≈ shoulder/withers width)
    (2, 5): (0.45, 0.7),  # front_left_elbow ↔ rear_left_elbow (body length proxy)
    (8, 11): (0.45, 0.7),  # front_right_elbow ↔ rear_right_elbow (body length proxy)
}

_AVERAGE_HUMAN_HEIGHT_M = 1.70
_AVERAGE_DOG_HEIGHT_M = 0.55  # medium-sized dog

# Depth range clamps — physics limits for a gimbal tracker
MIN_DEPTH_M = 0.3  # closer than 30cm is impossible for gimbal aiming
MAX_DEPTH_M = 25.0  # beyond 25m parallax is negligible

# Human horizontal pairs used for orientation compensation
_HORIZONTAL_PAIRS = {(5, 6), (11, 12)}
# Human diagonal pairs used as orientation reference
_DIAGONAL_PAIRS = {(5, 11), (6, 12)}

# Human ankle keypoint indices (COCO-17)
_HUMAN_ANKLE_INDICES = (15, 16)
_HUMAN_KNEE_INDICES = (13, 14)
_HUMAN_SHOULDER_INDICES = (5, 6)

# Dog paw keypoint indices (Enhanced-24)
_DOG_PAW_INDICES = (0, 3, 6, 9)
_DOG_ELBOW_INDICES = (2, 5, 8, 11)


@dataclass
class DepthEstimate:
    depth_m: float
    pair: tuple[int, int] | None  # None when from bbox fallback
    confidence: float
    n_pairs: int = 1  # how many pairs contributed


class DepthSmoother:
    """EMA depth smoother with innovation gating and staleness decay."""

    def __init__(self, alpha: float = 0.3, max_stale_frames: int = 60) -> None:
        self._alpha = alpha
        self._max_stale = max_stale_frames
        self._smoothed: float | None = None
        self._stale_count = 0

    def update(self, depth_m: float | None) -> float | None:
        if depth_m is not None:
            if self._smoothed is not None:
                # Innovation gate: reject >2× jumps from current estimate
                ratio = depth_m / self._smoothed
                if ratio > 2.0 or ratio < 0.5:
                    self._stale_count += 1
                    if self._stale_count > self._max_stale:
                        self._smoothed = None
                        return None
                    return self._smoothed
            if self._smoothed is None:
                self._smoothed = depth_m
            else:
                self._smoothed = self._alpha * depth_m + (1.0 - self._alpha) * self._smoothed
            self._stale_count = 0
            return self._smoothed

        # No new depth — use last value with staleness tracking
        if self._smoothed is not None:
            self._stale_count += 1
            if self._stale_count > self._max_stale:
                self._smoothed = None
                return None
            return self._smoothed
        return None

    def reset(self) -> None:
        self._smoothed = None
        self._stale_count = 0

    def snapshot(self) -> tuple[float | None, int]:
        return self._smoothed, self._stale_count

    def restore(self, state: tuple[float | None, int]) -> None:
        self._smoothed, self._stale_count = state


def estimate_depth(
    keypoints: np.ndarray,
    fx: float,
    reference_spans: dict[tuple[int, int], tuple[float, float] | float] | None = None,
    min_confidence: float = 0.3,
    bbox: np.ndarray | None = None,
    target_kind: Literal["human", "dog"] = "human",
    outlier_ratio: float = 2.0,
) -> DepthEstimate | None:
    """Estimate depth using confidence-weighted multi-pair fusion.

    All valid keypoint pairs contribute, weighted by
    ``(mean_confidence² × pair_reliability)``.  Outlier pairs whose depth
    deviates >2× from the weighted median are rejected.  Horizontal-pair
    weights are reduced when body rotation is detected (human only).

    Falls back to partial-body-corrected bounding-box height when no
    keypoint pairs are usable.

    Parameters
    ----------
    keypoints:
        (N, >=3) array with columns [x, y, confidence].
    fx:
        Camera focal length in pixels.
    reference_spans:
        Mapping of (idx_a, idx_b) → real_distance_m or (real_distance_m, reliability).
        Defaults to target-appropriate span table.
    min_confidence:
        Minimum visibility for both keypoints in a pair.
    bbox:
        Optional (4,) array [x1, y1, x2, y2] for bbox-height fallback.
    target_kind:
        ``"human"`` or ``"dog"`` — selects span table and fallback height.
    """
    if reference_spans is not None:
        spans = reference_spans
    elif target_kind == "dog":
        spans = DOG_KEYPOINT_SPANS
    else:
        spans = HUMAN_KEYPOINT_SPANS

    # --- Collect all valid pair estimates ---
    # (depth_m, weight, pair_key, is_horizontal)
    candidates: list[tuple[float, float, tuple[int, int], bool]] = []

    if keypoints is not None and len(keypoints) >= 2 and keypoints.shape[1] >= 3:
        for pair_key, span_val in spans.items():
            idx_a, idx_b = pair_key
            if idx_a >= len(keypoints) or idx_b >= len(keypoints):
                continue

            conf_a = float(keypoints[idx_a, 2])
            conf_b = float(keypoints[idx_b, 2])
            if conf_a < min_confidence or conf_b < min_confidence:
                continue

            px_a = keypoints[idx_a, :2]
            px_b = keypoints[idx_b, :2]
            pixel_dist = float(np.linalg.norm(px_a - px_b))
            if pixel_dist < 1.0:
                continue  # degenerate — keypoints overlapping

            # Unpack (real_dist, reliability) or just real_dist
            if isinstance(span_val, tuple):
                real_dist_m, reliability = span_val
            else:
                real_dist_m, reliability = float(span_val), 1.0

            depth_m = fx * real_dist_m / pixel_dist

            # Depth range clamping
            if depth_m < MIN_DEPTH_M or depth_m > MAX_DEPTH_M:
                continue

            # Nonlinear (quadratic) confidence weighting
            mean_conf = (conf_a + conf_b) * 0.5
            weight = (mean_conf**2) * reliability
            is_horiz = pair_key in _HORIZONTAL_PAIRS
            candidates.append((depth_m, weight, pair_key, is_horiz))

    if not candidates:
        return _bbox_fallback(bbox, fx, keypoints, target_kind)

    # --- Orientation compensation (human only) ---
    if target_kind == "human" and len(candidates) >= 2:
        candidates = _apply_orientation_compensation(candidates)

    if len(candidates) == 1:
        d, w, p, _ = candidates[0]
        return DepthEstimate(depth_m=d, pair=p, confidence=w, n_pairs=1)

    # --- Outlier rejection: remove estimates >2× from weighted median ---
    depths = np.array([c[0] for c in candidates])
    weights = np.array([c[1] for c in candidates])

    # Weighted median via sorted cumulative weights
    sorted_idx = np.argsort(depths)
    cum_w = np.cumsum(weights[sorted_idx])
    median_idx = np.searchsorted(cum_w, cum_w[-1] * 0.5)
    w_median = depths[sorted_idx[median_idx]]

    # Keep only pairs within outlier_ratio× of median
    filtered = []
    lo_bound = 1.0 / outlier_ratio
    for d, w, p, _ in candidates:
        ratio = d / w_median if w_median > 0 else 1.0
        if lo_bound <= ratio <= outlier_ratio:
            filtered.append((d, w, p))

    if not filtered:
        closest_idx = int(np.argmin(np.abs(depths - w_median)))
        d, w, p, _ = candidates[closest_idx]
        return DepthEstimate(depth_m=d, pair=p, confidence=w, n_pairs=1)

    # --- Confidence-weighted mean ---
    total_w = sum(w for _, w, _ in filtered)
    fused_depth = sum(d * w for d, w, _ in filtered) / total_w
    best_pair = max(filtered, key=lambda c: c[1])[2]
    avg_conf = total_w / len(filtered)

    return DepthEstimate(
        depth_m=fused_depth,
        pair=best_pair,
        confidence=avg_conf,
        n_pairs=len(filtered),
    )


def _apply_orientation_compensation(
    candidates: list[tuple[float, float, tuple[int, int], bool]],
) -> list[tuple[float, float, tuple[int, int], bool]]:
    """Reduce weight of horizontal pairs when body rotation is detected.

    If both horizontal (e.g. shoulders) and diagonal (e.g. shoulder→hip) pairs
    are visible, a rotating person compresses horizontal spans while diagonal
    spans stay more stable.  When the horizontal depth estimate significantly
    exceeds the diagonal estimate, the person is turned sideways and the
    horizontal pair is unreliable — its weight is scaled down.
    """
    horiz_depths = [d for d, _, p, h in candidates if h]
    diag_depths = [d for d, _, p, h in candidates if not h]

    if not horiz_depths or not diag_depths:
        return candidates

    median_horiz = float(np.median(horiz_depths))
    median_diag = float(np.median(diag_depths))

    if median_diag <= 0:
        return candidates

    # ratio > 1 means horizontal pairs report larger depth (= smaller pixel span = rotation)
    rotation_ratio = median_horiz / median_diag
    if rotation_ratio <= 1.1:
        return candidates  # no significant rotation detected

    # Scale down horizontal weights: clamp ratio to reduce weight to minimum 0.3
    scale = max(0.3, 1.0 / rotation_ratio)
    return [(d, w * scale, p, h) if h else (d, w, p, h) for d, w, p, h in candidates]


def _estimate_visible_fraction(
    keypoints: np.ndarray | None,
    target_kind: Literal["human", "dog"],
    min_confidence: float = 0.3,
) -> float:
    """Estimate what body fraction is visible from keypoint coverage.

    Returns 1.0 (full body), or a reduced fraction if lower extremities
    are occluded/cropped.
    """
    if keypoints is None or len(keypoints) < 2 or keypoints.shape[1] < 3:
        return 1.0  # unknown → assume full body

    def _visible(idx: int) -> bool:
        return idx < len(keypoints) and float(keypoints[idx, 2]) >= min_confidence

    if target_kind == "human":
        has_shoulders = any(_visible(i) for i in _HUMAN_SHOULDER_INDICES)
        has_ankles = any(_visible(i) for i in _HUMAN_ANKLE_INDICES)
        has_knees = any(_visible(i) for i in _HUMAN_KNEE_INDICES)

        if not has_shoulders:
            return 1.0  # can't determine crop without upper body reference
        if has_ankles:
            return 1.0  # full body visible
        if has_knees:
            return 0.85  # shin crop
        return 0.55  # waist crop
    else:
        # Dog: check paws vs elbows
        has_elbows = any(_visible(i) for i in _DOG_ELBOW_INDICES)
        has_paws = any(_visible(i) for i in _DOG_PAW_INDICES)
        if not has_elbows:
            return 1.0
        if has_paws:
            return 1.0
        return 0.85  # paws cropped


def _bbox_fallback(
    bbox: np.ndarray | None,
    fx: float,
    keypoints: np.ndarray | None = None,
    target_kind: Literal["human", "dog"] = "human",
) -> DepthEstimate | None:
    """Estimate depth from bounding-box height with partial-body correction."""
    if bbox is None:
        return None
    bb = np.asarray(bbox, dtype=np.float32).ravel()
    if bb.shape[0] < 4:
        return None
    height_px = float(bb[3] - bb[1])
    if height_px < 10.0:
        return None

    base_height = _AVERAGE_DOG_HEIGHT_M if target_kind == "dog" else _AVERAGE_HUMAN_HEIGHT_M
    visible_fraction = _estimate_visible_fraction(keypoints, target_kind)
    effective_height = base_height * visible_fraction

    depth_m = fx * effective_height / height_px

    if depth_m < MIN_DEPTH_M or depth_m > MAX_DEPTH_M:
        return None

    return DepthEstimate(depth_m=depth_m, pair=None, confidence=0.3, n_pairs=0)
