"""Tests for FrameHealthMonitor (preflight AE/AWB/AF drift detection)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.config import PreflightConfig
from src.shared.preflight import FrameHealthMonitor


def _constant_frame(brightness: int = 128, shape: tuple[int, ...] = (64, 64, 3)) -> np.ndarray:
    return np.full(shape, brightness, dtype=np.uint8)


def _sharp_frame(shape: tuple[int, ...] = (64, 64, 3)) -> np.ndarray:
    """Frame with high-frequency edges → high Laplacian variance."""
    frame = np.zeros(shape, dtype=np.uint8)
    frame[::2, :, :] = 255  # horizontal stripes
    return frame


def _blurry_frame(shape: tuple[int, ...] = (64, 64, 3)) -> np.ndarray:
    """Uniform gray → near-zero Laplacian variance."""
    return np.full(shape, 128, dtype=np.uint8)


@pytest.mark.unit
def test_healthy_feed_no_warnings() -> None:
    """Constant-brightness frames produce no warnings."""
    config = PreflightConfig(enabled=True, window_size=10)
    monitor = FrameHealthMonitor(config)

    frame = _constant_frame(128)
    for _ in range(20):
        warnings = monitor.check(frame)

    assert warnings == []


@pytest.mark.unit
def test_brightness_hunting_warns() -> None:
    """Linearly ramping brightness triggers brightness hunting warning."""
    config = PreflightConfig(
        enabled=True,
        window_size=10,
        brightness_stddev_warn=5.0,  # low threshold for test
        warn_cooldown_s=0.0,  # no cooldown
    )
    monitor = FrameHealthMonitor(config)

    # Feed 10 frames with widely varying brightness to fill the window
    warnings = []
    for i in range(10):
        brightness = 50 + i * 20  # 50, 70, 90, ..., 230
        frame = _constant_frame(brightness)
        warnings = monitor.check(frame)

    assert len(warnings) >= 1
    assert any("brightness hunting" in w for w in warnings)


@pytest.mark.unit
def test_sharpness_drop_warns() -> None:
    """Sudden sharpness drop triggers AF drift warning."""
    config = PreflightConfig(
        enabled=True,
        window_size=10,
        sharpness_drop_warn=0.3,  # 30% drop threshold
        warn_cooldown_s=0.0,
    )
    monitor = FrameHealthMonitor(config)

    # Fill window with sharp frames
    sharp = _sharp_frame()
    for _ in range(10):
        monitor.check(sharp)

    # Now feed blurry frames
    blurry = _blurry_frame()
    warnings = []
    for _ in range(10):
        warnings = monitor.check(blurry)

    assert len(warnings) >= 1
    assert any("sharpness drop" in w for w in warnings)


@pytest.mark.unit
def test_cooldown_prevents_spam() -> None:
    """Two consecutive warning triggers within cooldown → only first fires."""
    config = PreflightConfig(
        enabled=True,
        window_size=5,
        brightness_stddev_warn=1.0,  # very low threshold
        warn_cooldown_s=100.0,  # very long cooldown
    )
    monitor = FrameHealthMonitor(config)

    # Generate warning-triggering frames with controlled time
    warn_count = 0
    base_time = 1000.0  # start at a high base time

    def fake_monotonic() -> float:
        # Each call returns the same base_time so cooldown blocks subsequent warnings
        return base_time

    with patch("src.shared.preflight.time.monotonic", side_effect=fake_monotonic):
        for i in range(20):
            brightness = 50 + (i % 5) * 40  # oscillating brightness
            frame = _constant_frame(brightness)
            warnings = monitor.check(frame)
            if warnings:
                warn_count += 1

    # Should only fire once due to long cooldown
    assert warn_count == 1


@pytest.mark.unit
def test_disabled_returns_empty() -> None:
    """Disabled monitor always returns empty warnings."""
    config = PreflightConfig(enabled=False)
    monitor = FrameHealthMonitor(config)

    for i in range(20):
        frame = _constant_frame(50 + i * 10)
        warnings = monitor.check(frame)
        assert warnings == []


@pytest.mark.unit
def test_insufficient_data_no_warnings() -> None:
    """Monitor doesn't warn until window is full."""
    config = PreflightConfig(enabled=True, window_size=20, warn_cooldown_s=0.0)
    monitor = FrameHealthMonitor(config)

    for i in range(19):
        # Wide brightness variation but not enough samples
        frame = _constant_frame(50 + i * 10)
        warnings = monitor.check(frame)
        assert warnings == []


@pytest.mark.unit
def test_frame_drop_sample_window_has_sane_default() -> None:
    """The inference loop flushes a drop-rate sample every N frames.

    120 frames @ 60 fps ≈ 2 s — fine-grained enough to see sustained drops
    but coarse enough that a single hiccup doesn't spam the profiler.
    """
    config = PreflightConfig()
    assert config.frame_drop_sample_window == 120


@pytest.mark.unit
def test_frame_drop_sample_window_is_tunable() -> None:
    """The knob must be overridable so ops can widen or tighten the window."""
    config = PreflightConfig(frame_drop_sample_window=600)
    assert config.frame_drop_sample_window == 600
