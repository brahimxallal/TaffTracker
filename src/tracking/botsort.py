"""BoTSORT tracker — ByteTrack + Camera Motion Compensation (CMC).

Extends the two-tier ByteTrack matching with sparse-optical-flow-based
affine warping so that track bboxes are compensated for camera motion
before IoU association.  This is critical for camera-on-gimbal setups
where every servo movement shifts the entire field of view.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.shared.types import Detection, Track
from src.tracking.bytetrack import ByteTracker


class BoTSORT(ByteTracker):
    """Drop-in replacement for ByteTracker with camera motion compensation.

    Optimizations over naive CMC:
    - Grayscale + downscale computed once per frame (not twice)
    - CMC skipped entirely when gimbal angular velocity is near zero
    - Reduced default max_corners for half-res frames
    """

    # Angular velocity below this (rad/s magnitude) → gimbal stationary, skip CMC
    _CMC_SKIP_ANGULAR_VEL_THRESHOLD = 0.005  # ~0.3 deg/s — effectively at rest

    def __init__(
        self,
        track_thresh: float = 0.45,
        match_thresh: float = 0.55,
        max_lost: int = 30,
        low_thresh: float = 0.10,
        birth_min_hits: int = 2,
        cmc_enabled: bool = True,
        cmc_max_corners: int = 50,
        cmc_quality_level: float = 0.01,
        cmc_min_distance: float = 5.0,
        cmc_ransac_threshold: float = 3.0,
        cmc_downscale: float = 0.25,
        cmc_feature_reuse_frames: int = 5,
    ) -> None:
        super().__init__(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            max_lost=max_lost,
            low_thresh=low_thresh,
            birth_min_hits=birth_min_hits,
        )
        self._cmc_enabled = cmc_enabled
        self._cmc_max_corners = cmc_max_corners
        self._cmc_quality_level = cmc_quality_level
        self._cmc_min_distance = cmc_min_distance
        self._cmc_ransac_threshold = cmc_ransac_threshold
        self._cmc_downscale = cmc_downscale
        self._cmc_feature_reuse_frames = max(1, cmc_feature_reuse_frames)
        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None
        self._frames_since_feature_extract: int = 0

    def reset(self) -> None:
        super().reset()
        self._prev_gray = None
        self._prev_pts = None
        self._frames_since_feature_extract = 0

    def update(
        self,
        detections: list[Detection],
        timestamp_ns: int = 0,
        frame: np.ndarray | None = None,
        angular_velocity: tuple[float, float] | None = None,
    ) -> list[Track]:
        # --- Camera Motion Compensation ---
        if self._cmc_enabled and frame is not None:
            # Compute grayscale + downscale ONCE per frame (eliminates duplicate cvtColor)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            scale = self._cmc_downscale
            if scale < 1.0:
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # Gate: skip expensive LK + RANSAC when gimbal is stationary
            gimbal_moving = True
            if angular_velocity is not None:
                av_mag = (angular_velocity[0] ** 2 + angular_velocity[1] ** 2) ** 0.5
                gimbal_moving = av_mag > self._CMC_SKIP_ANGULAR_VEL_THRESHOLD

            if self._tracks and gimbal_moving:
                warp = self._estimate_affine(gray)
                if warp is not None:
                    self._warp_tracks(warp)

            # Cache for next iteration (always update even when skipping CMC)
            self._prev_gray = gray

        # Delegate to ByteTracker's two-tier matching
        return super().update(detections, timestamp_ns=timestamp_ns)

    def _estimate_affine(self, gray: np.ndarray) -> np.ndarray | None:
        """Estimate 2×3 affine from sparse optical flow between prev and current frame.

        Expects a pre-computed grayscale frame (already downscaled if applicable).
        The returned warp translation is scaled back to full resolution.
        """
        if self._prev_gray is None:
            return None
        if self._prev_gray.shape != gray.shape:
            return None

        # Feature extraction: reuse cached points when possible
        self._frames_since_feature_extract += 1
        if (
            self._prev_pts is None
            or self._frames_since_feature_extract >= self._cmc_feature_reuse_frames
        ):
            scale = self._cmc_downscale
            min_dist = self._cmc_min_distance * scale if scale < 1.0 else self._cmc_min_distance
            self._prev_pts = cv2.goodFeaturesToTrack(
                self._prev_gray,
                maxCorners=self._cmc_max_corners,
                qualityLevel=self._cmc_quality_level,
                minDistance=min_dist,
            )
            self._frames_since_feature_extract = 0

        prev_pts = self._prev_pts
        if prev_pts is None or len(prev_pts) < 4:
            return None

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            gray,
            prev_pts,
            None,
            winSize=(15, 15),
            maxLevel=2,
        )
        if curr_pts is None:
            return None

        mask = status.ravel().astype(bool)
        if mask.sum() < 4:
            return None

        src = prev_pts[mask].reshape(-1, 2)
        dst = curr_pts[mask].reshape(-1, 2)
        warp, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=self._cmc_ransac_threshold
        )
        if warp is None or inliers is None or inliers.sum() < 4:
            return None

        # Cache tracked points for reuse (only inlier destinations)
        self._prev_pts = dst[inliers.ravel().astype(bool)].reshape(-1, 1, 2).astype(np.float32)

        # Scale warp translation back to full resolution
        scale = self._cmc_downscale
        if scale < 1.0:
            inv_scale = 1.0 / scale
            warp[0, 2] *= inv_scale
            warp[1, 2] *= inv_scale
        return warp

    def _warp_tracks(self, warp: np.ndarray) -> None:
        """Apply affine warp to all active track bboxes and tentative tracks."""
        for track in self._tracks:
            track.bbox = _warp_bbox(track.bbox, warp)
        for track in self._tentative:
            track.bbox = _warp_bbox(track.bbox, warp)


def _warp_bbox(bbox: np.ndarray, warp: np.ndarray) -> np.ndarray:
    """Warp an [x1,y1,x2,y2] bbox through a 2×3 affine transform."""
    corners = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float64)
    ones = np.ones((2, 1), dtype=np.float64)
    homog = np.hstack([corners, ones])  # (2, 3)
    warped = homog @ warp.T  # (2, 2)
    return np.array([warped[0, 0], warped[0, 1], warped[1, 0], warped[1, 1]], dtype=bbox.dtype)
