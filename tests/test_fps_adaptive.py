"""FPS-adaptive parameters and unified adaptive mode."""
from __future__ import annotations

import pytest

from src.config import (
    TrackingConfig,
    adapt_tracking_for_fps,
)
from src.tracking.adaptive import AdaptiveController


# --- FPS adaptation ---


@pytest.mark.unit
def test_adapt_60fps_is_identity() -> None:
    base = TrackingConfig()
    adapted = adapt_tracking_for_fps(base, 60.0)
    assert adapted.process_noise == pytest.approx(base.process_noise)
    assert adapted.max_lost_frames == base.max_lost_frames


@pytest.mark.unit
def test_adapt_30fps_doubles_process_noise() -> None:
    base = TrackingConfig(process_noise=3.0, max_lost_frames=30)
    adapted = adapt_tracking_for_fps(base, 30.0)
    # ratio = 30/60 = 0.5, Q = 3.0 / 0.5 = 6.0
    assert adapted.process_noise == pytest.approx(6.0)
    # lost frames = 30 * 0.5 = 15
    assert adapted.max_lost_frames == 15


@pytest.mark.unit
def test_adapt_120fps_halves_process_noise() -> None:
    base = TrackingConfig(process_noise=4.0, max_lost_frames=30)
    adapted = adapt_tracking_for_fps(base, 120.0)
    # ratio = 120/60 = 2.0, Q = 4.0 / 2.0 = 2.0
    assert adapted.process_noise == pytest.approx(2.0)
    # lost frames = 30 * 2.0 = 60
    assert adapted.max_lost_frames == 60


@pytest.mark.unit
def test_adapt_preserves_hold_time() -> None:
    base = TrackingConfig(hold_time_s=0.75)
    adapted = adapt_tracking_for_fps(base, 30.0)
    assert adapted.hold_time_s == 0.75  # hold time is absolute, not frame-relative


@pytest.mark.unit
def test_adapt_zero_fps_returns_unchanged() -> None:
    base = TrackingConfig()
    adapted = adapt_tracking_for_fps(base, 0.0)
    assert adapted == base


@pytest.mark.unit
def test_adapt_very_low_fps_clamps() -> None:
    base = TrackingConfig(max_lost_frames=30)
    adapted = adapt_tracking_for_fps(base, 1.0)
    # min 5 lost frames
    assert adapted.max_lost_frames >= 5


@pytest.mark.unit
def test_unified_defaults_are_balanced() -> None:
    """Unified TrackingConfig defaults sit between old preset extremes."""
    cfg = TrackingConfig()
    assert cfg.confidence_threshold == 0.45
    assert cfg.hold_time_s == 0.65
    assert cfg.process_noise == 2.5
    assert cfg.measurement_noise == 5.0
    assert cfg.max_lost_frames == 30


@pytest.mark.unit
def test_fps_adaptation_composes_with_defaults() -> None:
    """FPS adaptation works with unified defaults."""
    cfg = TrackingConfig()
    adapted = adapt_tracking_for_fps(cfg, 30.0)
    assert adapted.process_noise > cfg.process_noise  # lower FPS → higher Q
    assert adapted.max_lost_frames < cfg.max_lost_frames  # fewer frames at lower FPS


# --- Adaptive controller ---


@pytest.mark.unit
def test_adaptive_controller_initializes_to_base() -> None:
    base = TrackingConfig()
    ctrl = AdaptiveController(base)
    assert ctrl.confidence_threshold == base.confidence_threshold
    assert ctrl.hold_time_s == base.hold_time_s


@pytest.mark.unit
def test_adaptive_controller_lowers_conf_on_low_reliability() -> None:
    base = TrackingConfig(confidence_threshold=0.45)
    ctrl = AdaptiveController(base)
    # Simulate 60 frames with 50% detection rate (below 70% threshold)
    for i in range(60):
        ctrl.update(detected=(i % 2 == 0), speed=100.0)
    assert ctrl.confidence_threshold < base.confidence_threshold


@pytest.mark.unit
def test_adaptive_controller_raises_conf_on_high_reliability() -> None:
    base = TrackingConfig(confidence_threshold=0.45)
    ctrl = AdaptiveController(base)
    # Simulate 60 frames with 100% detection rate
    for _ in range(60):
        ctrl.update(detected=True, speed=100.0)
    assert ctrl.confidence_threshold > base.confidence_threshold


@pytest.mark.unit
def test_adaptive_controller_shortens_hold_on_fast_motion() -> None:
    base = TrackingConfig(hold_time_s=0.65)
    ctrl = AdaptiveController(base)
    for _ in range(30):
        ctrl.update(detected=True, speed=300.0)  # fast
    assert ctrl.hold_time_s < base.hold_time_s


@pytest.mark.unit
def test_adaptive_controller_extends_hold_on_slow_motion() -> None:
    base = TrackingConfig(hold_time_s=0.65)
    ctrl = AdaptiveController(base)
    for _ in range(30):
        ctrl.update(detected=True, speed=20.0)  # slow
    assert ctrl.hold_time_s > base.hold_time_s


@pytest.mark.unit
def test_adaptive_controller_reset_restores_defaults() -> None:
    base = TrackingConfig()
    ctrl = AdaptiveController(base)
    for _ in range(60):
        ctrl.update(detected=False, speed=500.0)
    ctrl.reset()
    assert ctrl.confidence_threshold == base.confidence_threshold
    assert ctrl.hold_time_s == base.hold_time_s


# --- Phase D: Linear interpolation tests ---


@pytest.mark.unit
def test_adaptive_conf_interpolates_at_midpoint() -> None:
    """At 80% reliability (midpoint of 70-90%), conf should be between extremes."""
    base = TrackingConfig(confidence_threshold=0.45)
    ctrl = AdaptiveController(base)
    # 80% detection rate = midpoint
    for i in range(100):
        ctrl.update(detected=(i % 5 != 0), speed=100.0)  # 80% hit rate
    low_conf = 0.40  # 0.45 - 0.05
    high_conf = 0.47  # 0.45 + 0.02
    assert low_conf < ctrl.confidence_threshold < high_conf


@pytest.mark.unit
def test_adaptive_hold_interpolates_at_mid_speed() -> None:
    """At 125 px/s (midpoint of 50-200), hold should be between slow and fast."""
    base = TrackingConfig(hold_time_s=0.65)
    ctrl = AdaptiveController(base)
    for _ in range(30):
        ctrl.update(detected=True, speed=125.0)  # midpoint speed
    slow_hold = 0.80  # 0.65 + 0.15
    fast_hold = 0.45  # 0.65 - 0.20
    assert fast_hold < ctrl.hold_time_s < slow_hold


@pytest.mark.unit
def test_adaptive_no_discontinuity_at_boundary() -> None:
    """Values near the boundary should be close to the endpoint."""
    base = TrackingConfig(confidence_threshold=0.45)
    ctrl = AdaptiveController(base)
    # Just above reliability_low (0.71)
    hits = [True] * 71 + [False] * 29  # 71%
    for h in hits:
        ctrl.update(detected=h, speed=100.0)
    conf_just_above = ctrl.confidence_threshold
    # Should be very close to the low-reliability conf (0.40), not jumping to base
    assert abs(conf_just_above - 0.40) < 0.02
