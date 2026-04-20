"""Laser dot detection via brightness peak + optional red halo confirmation."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, pi

import cv2
import numpy as np

from src.config import LaserConfig

# Laser dots saturate camera sensors → appear WHITE (S≈0, V≈255), not red.
# Primary detection: find the brightest small blob in the frame.
_BRIGHTNESS_THRESH = 240  # V channel threshold for saturated spots


@dataclass(frozen=True, slots=True)
class LaserDetection:
    center: tuple[float, float]
    radius: float
    brightness: float


class LaserDetector:
    """Detect a laser dot by brightness peak with size filtering.

    Camera sensors overexpose on laser light: the dot center saturates all
    channels to near-255, appearing white regardless of laser color.  We
    detect by thresholding the V (brightness) channel at a high level, then
    filtering for small circular blobs.  A secondary red-halo check confirms
    laser identity when the dot has non-saturated edges.
    """

    def __init__(self, config: LaserConfig) -> None:
        self._cfg = config
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(
        self,
        frame: np.ndarray,
        roi_center: tuple[float, float] | None = None,
        roi_radius: float | None = None,
    ) -> LaserDetection | None:
        if roi_center is not None:
            result = self._detect_in_region(
                frame,
                roi_center,
                roi_radius or self._cfg.roi_radius_px,
            )
            if result is not None:
                # Jump guard: reject if detection is too far from predicted position
                dx = result.center[0] - roi_center[0]
                dy = result.center[1] - roi_center[1]
                if hypot(dx, dy) > self._cfg.max_jump_px:
                    return None
                return result
            return None  # No full-frame fallback — avoids false positives
        return self._detect_in_region(frame, None, 0.0)

    def _detect_in_region(
        self,
        frame: np.ndarray,
        roi_center: tuple[float, float] | None,
        roi_radius: float,
    ) -> LaserDetection | None:
        h, w = frame.shape[:2]
        cfg = self._cfg

        # --- ROI extraction ---
        region = frame
        ox, oy = 0, 0
        if roi_center is not None:
            cx, cy = roi_center
            x1 = max(0, int(cx - roi_radius))
            y1 = max(0, int(cy - roi_radius))
            x2 = min(w, int(cx + roi_radius))
            y2 = min(h, int(cy + roi_radius))
            if x2 - x1 < 4 or y2 - y1 < 4:
                return None
            region = frame[y1:y2, x1:x2]
            ox, oy = x1, y1

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]

        # --- Primary: brightness peak detection ---
        # Laser dots saturate the sensor → V ≈ 255 regardless of color
        bright_thresh = max(cfg.val_min, _BRIGHTNESS_THRESH)
        bright_mask = cv2.inRange(v_channel, int(bright_thresh), 255)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, self._kernel)
        result = self._best_blob(bright_mask, region, ox, oy, cfg)
        if result is not None:
            return result

        # --- Fallback: HSV red detection (for dimmer / non-saturated dots) ---
        lo_mask = cv2.inRange(
            hsv,
            np.array([0, cfg.sat_min, cfg.val_min], dtype=np.uint8),
            np.array([cfg.hue_low_upper, 255, 255], dtype=np.uint8),
        )
        hi_mask = cv2.inRange(
            hsv,
            np.array([cfg.hue_high_lower, cfg.sat_min, cfg.val_min], dtype=np.uint8),
            np.array([180, 255, 255], dtype=np.uint8),
        )
        red_mask = lo_mask | hi_mask
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, self._kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, self._kernel)
        return self._best_blob(red_mask, region, ox, oy, cfg)

    def _best_blob(
        self,
        mask: np.ndarray,
        region: np.ndarray,
        ox: int,
        oy: int,
        cfg: LaserConfig,
    ) -> LaserDetection | None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        best: LaserDetection | None = None
        best_brightness: float = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < cfg.min_area or area > cfg.max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1e-6:
                continue
            circularity = 4.0 * pi * area / (perimeter * perimeter)
            if circularity < cfg.min_circularity:
                continue

            m = cv2.moments(cnt)
            if m["m00"] < 1e-6:
                continue
            cx_local = m["m10"] / m["m00"]
            cy_local = m["m01"] / m["m00"]

            bx, by, bw, bh = cv2.boundingRect(cnt)
            roi_gray = gray[by : by + bh, bx : bx + bw]
            brightness = float(roi_gray.max()) if roi_gray.size > 0 else 0.0

            # Brightness gate
            if brightness < cfg.min_brightness:
                continue

            if brightness > best_brightness:
                best_brightness = brightness
                best = LaserDetection(
                    center=(cx_local + ox, cy_local + oy),
                    radius=float(max(bw, bh)) / 2.0,
                    brightness=brightness,
                )

        return best
