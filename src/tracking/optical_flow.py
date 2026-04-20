from __future__ import annotations

import cv2
import numpy as np


class OpticalFlowRefiner:
    """Refine YOLO keypoints using sparse Lucas-Kanade optical flow.

    Between detections the neural network produces per-frame pixel jitter.
    LK flow tracks the same physical point across frames with sub-pixel
    accuracy, then blends with the NN output so the centroid is both
    stable *and* drift-free.

    The blend ratio adapts per-keypoint: when flow tracking succeeds and
    the keypoint confidence is high the refiner trusts flow more; when
    the NN confidence is low or flow fails it falls back entirely to
    the raw detection.
    """

    def __init__(
        self,
        win_size: int = 15,
        max_level: int = 2,
        flow_weight: float = 0.7,
        min_confidence: float = 0.3,
        fb_threshold_px: float = 2.0,
    ) -> None:
        self._win_size = (win_size, win_size)
        self._max_level = max_level
        self._flow_weight = flow_weight
        self._min_confidence = min_confidence
        self._fb_threshold_px = fb_threshold_px
        self._lk_params = dict(
            winSize=self._win_size,
            maxLevel=self._max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        self._prev_gray: np.ndarray | None = None
        self._prev_points: np.ndarray | None = None

    def refine(
        self,
        frame: np.ndarray,
        points: np.ndarray,
        confidences: np.ndarray | None = None,
    ) -> np.ndarray:
        """Refine keypoints using optical flow with sub-pixel accuracy.

        Args:
            frame: Current BGR frame.
            points: ``(N, 2)`` float32 keypoint coordinates from YOLO.
            confidences: Optional ``(N,)`` float32 per-keypoint confidence.

        Returns:
            ``(N, 2)`` refined points.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pts = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 1, 2)

        if (
            self._prev_gray is None
            or self._prev_points is None
            or len(pts) != len(self._prev_points)
        ):
            self._prev_gray = gray
            self._prev_points = pts.copy()
            return pts.reshape(-1, 2)

        # Forward LK flow
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_points, None, **self._lk_params
        )

        # Backward verification: track back and check consistency
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._prev_gray, next_pts, None, **self._lk_params
        )

        # Sub-pixel corner refinement on flow-predicted positions
        good_mask = (status.ravel() == 1) & (back_status.ravel() == 1)
        if good_mask.any():
            good_idx = np.where(good_mask)[0]
            subpix_pts = next_pts[good_idx].copy()
            # Clamp to frame bounds to prevent cornerSubPix assertion failure
            h, w = gray.shape[:2]
            margin = 4.0  # cornerSubPix winSize(3,3) needs ≥3px margin
            subpix_pts[..., 0] = np.clip(subpix_pts[..., 0], margin, w - 1 - margin)
            subpix_pts[..., 1] = np.clip(subpix_pts[..., 1], margin, h - 1 - margin)
            cv2.cornerSubPix(
                gray,
                subpix_pts,
                winSize=(3, 3),
                zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.01),
            )
            next_pts[good_idx] = subpix_pts

        # Forward-backward consistency check
        fb_dist = np.linalg.norm(self._prev_points.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1)
        fb_ok = fb_dist < self._fb_threshold_px

        # Adaptive blend per keypoint
        refined = np.empty_like(pts)
        n_pts = len(pts)
        for i in range(n_pts):
            if good_mask[i] and fb_ok[i]:
                w = self._flow_weight
                if confidences is not None and i < len(confidences):
                    conf = confidences[i]
                    if conf < self._min_confidence:
                        w = 0.0  # trust NN only for low-confidence keypoints
                    else:
                        # Scale flow weight by confidence
                        w *= min(conf / 0.8, 1.0)
                refined[i] = w * next_pts[i] + (1.0 - w) * pts[i]
            else:
                refined[i] = pts[i]

        self._prev_gray = gray
        self._prev_points = refined.copy()
        return refined.reshape(-1, 2)

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_points = None
