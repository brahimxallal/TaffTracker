"""Tests for src.laser.detector — laser dot detection by brightness peak."""
from __future__ import annotations

import numpy as np
import pytest
import cv2

from src.config import LaserConfig
from src.laser.detector import LaserDetector, LaserDetection


def _make_frame(w: int = 640, h: int = 480, color: tuple[int, int, int] = (40, 40, 40)) -> np.ndarray:
    frame = np.full((h, w, 3), color, dtype=np.uint8)
    return frame


def _draw_dot(frame: np.ndarray, center: tuple[int, int], radius: int, bgr: tuple[int, int, int]) -> None:
    cv2.circle(frame, center, radius, bgr, -1)


@pytest.fixture
def config() -> LaserConfig:
    return LaserConfig()


@pytest.fixture
def detector(config: LaserConfig) -> LaserDetector:
    return LaserDetector(config)


class TestBrightnessDetection:
    """Primary detection: brightest small blob."""

    def test_detects_bright_white_dot(self, detector: LaserDetector) -> None:
        """Laser dots saturate the sensor and appear white."""
        frame = _make_frame()
        _draw_dot(frame, (300, 200), 5, (255, 255, 255))
        det = detector.detect(frame)
        assert det is not None
        assert abs(det.center[0] - 300) < 5
        assert abs(det.center[1] - 200) < 5
        assert det.brightness >= 240

    def test_detects_bright_red_dot(self) -> None:
        config = LaserConfig(min_brightness=0)
        detector = LaserDetector(config)
        frame = _make_frame()
        _draw_dot(frame, (300, 200), 5, (0, 0, 255))
        det = detector.detect(frame)
        assert det is not None
        assert abs(det.center[0] - 300) < 5

    def test_returns_none_on_empty_frame(self, detector: LaserDetector) -> None:
        assert detector.detect(_make_frame()) is None

    def test_returns_none_on_dim_frame(self, detector: LaserDetector) -> None:
        frame = _make_frame(color=(100, 100, 100))
        assert detector.detect(frame) is None

    def test_rejects_oversized_blob(self) -> None:
        config = LaserConfig(max_area=50.0)
        detector = LaserDetector(config)
        frame = _make_frame()
        _draw_dot(frame, (300, 200), 20, (255, 255, 255))
        assert detector.detect(frame) is None

    def test_rejects_undersized_blob(self) -> None:
        config = LaserConfig(min_area=100.0)
        detector = LaserDetector(config)
        frame = _make_frame()
        _draw_dot(frame, (300, 200), 3, (255, 255, 255))
        assert detector.detect(frame) is None

    def test_picks_brightest_blob(self) -> None:
        frame = _make_frame()
        _draw_dot(frame, (100, 200), 5, (200, 200, 200))  # dimmer
        _draw_dot(frame, (400, 200), 5, (255, 255, 255))  # brighter
        config = LaserConfig(val_min=150)
        detector = LaserDetector(config)
        det = detector.detect(frame)
        assert det is not None
        assert det.center[0] > 300


class TestRedHaloFallback:
    """Secondary detection: HSV red for non-saturated dots."""

    def test_detects_dim_red_dot(self) -> None:
        config = LaserConfig(val_min=100, sat_min=50, min_brightness=0)
        detector = LaserDetector(config)
        frame = _make_frame()
        _draw_dot(frame, (300, 200), 5, (0, 0, 180))  # dim red, below brightness thresh
        det = detector.detect(frame)
        assert det is not None
        assert abs(det.center[0] - 300) < 5


class TestROI:
    def test_roi_finds_dot_inside(self, detector: LaserDetector) -> None:
        frame = _make_frame()
        _draw_dot(frame, (300, 200), 5, (255, 255, 255))
        det = detector.detect(frame, roi_center=(300.0, 200.0), roi_radius=50.0)
        assert det is not None
        assert abs(det.center[0] - 300) < 5

    def test_roi_prefers_inside_dot(self, detector: LaserDetector) -> None:
        frame = _make_frame()
        _draw_dot(frame, (100, 100), 5, (255, 255, 255))
        _draw_dot(frame, (500, 400), 5, (255, 255, 255))
        det = detector.detect(frame, roi_center=(100.0, 100.0), roi_radius=50.0)
        assert det is not None
        assert abs(det.center[0] - 100) < 10

    def test_no_fallback_to_full_frame(self, detector: LaserDetector) -> None:
        """When ROI is provided but dot is outside, return None (no fallback)."""
        frame = _make_frame()
        _draw_dot(frame, (500, 400), 5, (255, 255, 255))
        det = detector.detect(frame, roi_center=(100.0, 100.0), roi_radius=50.0)
        assert det is None

    def test_tiny_roi_returns_none(self, detector: LaserDetector) -> None:
        """Tiny ROI that can't contain the dot returns None."""
        frame = _make_frame()
        _draw_dot(frame, (300, 200), 5, (255, 255, 255))
        det = detector.detect(frame, roi_center=(300.0, 200.0), roi_radius=1.0)
        assert det is None


class TestLaserDetectionDataclass:
    def test_fields(self) -> None:
        d = LaserDetection(center=(1.5, 2.5), radius=3.0, brightness=250.0)
        assert d.center == (1.5, 2.5)
        assert d.radius == 3.0
        assert d.brightness == 250.0

    def test_frozen(self) -> None:
        d = LaserDetection(center=(1.0, 2.0), radius=3.0, brightness=200.0)
        with pytest.raises(AttributeError):
            d.center = (0.0, 0.0)  # type: ignore[misc]
