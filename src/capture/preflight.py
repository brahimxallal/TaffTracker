"""Frame health monitor for detecting AE/AWB/AF drift on phone cameras.

Maintains rolling statistics of brightness and sharpness (Laplacian variance)
and emits debounced warnings when thresholds are crossed.
"""

from __future__ import annotations

import logging
import time
from collections import deque

import cv2
import numpy as np

from src.config import PreflightConfig

LOGGER = logging.getLogger("preflight")


class FrameHealthMonitor:
    """Detects phone camera AE/AWB/AF drift via rolling frame statistics."""

    def __init__(self, config: PreflightConfig) -> None:
        self._config = config
        self._brightness_buf: deque[float] = deque(maxlen=config.window_size)
        self._sharpness_buf: deque[float] = deque(maxlen=config.window_size)
        self._sharpness_max: float = 0.0
        self._last_warn_time: float = 0.0

    def check(self, frame: np.ndarray) -> list[str]:
        """Analyse a frame and return warning messages (empty if healthy)."""
        if not self._config.enabled:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        brightness = float(gray.mean())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        self._brightness_buf.append(brightness)
        self._sharpness_buf.append(sharpness)
        self._sharpness_max = max(self._sharpness_max, sharpness)

        if len(self._brightness_buf) < self._config.window_size:
            return []  # not enough data yet

        now = time.monotonic()
        if now - self._last_warn_time < self._config.warn_cooldown_s:
            return []

        warnings: list[str] = []
        brightness_std = float(np.std(self._brightness_buf))
        if brightness_std > self._config.brightness_stddev_warn:
            warnings.append(
                f"preflight: brightness hunting (stddev={brightness_std:.1f} "
                f"> {self._config.brightness_stddev_warn})"
            )

        if self._sharpness_max > 0:
            current_sharpness = float(np.mean(list(self._sharpness_buf)[-5:]))
            drop = 1.0 - current_sharpness / self._sharpness_max
            if drop > self._config.sharpness_drop_warn:
                warnings.append(
                    f"preflight: sharpness drop ({drop:.0%} "
                    f"> {self._config.sharpness_drop_warn:.0%})"
                )

        if warnings:
            self._last_warn_time = now

        return warnings
