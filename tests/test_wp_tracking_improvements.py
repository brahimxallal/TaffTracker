"""Tests for WP1-WP8 tracking improvements."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import (
    CommConfig,
    KalmanConfig,
    SmoothingConfig,
    default_tracking_config,
)
from src.shared.types import Detection
from src.tracking.bytetrack import ByteTracker

# ── WP1: Birth hysteresis ───────────────────────────────────────────────


@pytest.mark.unit
class TestBirthHysteresis:

    def test_default_birth_min_hits_is_2(self):
        tracker = ByteTracker()
        assert tracker._config.birth_min_hits == 2

    def test_single_detection_not_promoted_immediately(self):
        tracker = ByteTracker(track_thresh=0.5, match_thresh=0.3, birth_min_hits=2)
        detections = [Detection(bbox=np.array([0, 0, 10, 10]), score=0.9)]
        tracks = tracker.update(detections, timestamp_ns=1)
        # Track should be tentative, not yet in output
        assert len(tracks) == 0
        assert len(tracker._tentative) == 1

    def test_two_consecutive_detections_promote_track(self):
        tracker = ByteTracker(track_thresh=0.5, match_thresh=0.3, birth_min_hits=2)
        det1 = [Detection(bbox=np.array([0, 0, 10, 10]), score=0.9)]
        det2 = [Detection(bbox=np.array([1, 0, 11, 10]), score=0.88)]
        tracker.update(det1, timestamp_ns=1)
        tracks = tracker.update(det2, timestamp_ns=2)
        assert len(tracks) == 1
        assert tracks[0].age == 2

    def test_no_match_discards_tentative(self):
        tracker = ByteTracker(track_thresh=0.5, match_thresh=0.3, birth_min_hits=2)
        det1 = [Detection(bbox=np.array([0, 0, 10, 10]), score=0.9)]
        # Second frame has detection far away — no match
        det2 = [Detection(bbox=np.array([500, 500, 510, 510]), score=0.9)]
        tracker.update(det1, timestamp_ns=1)
        tracks = tracker.update(det2, timestamp_ns=2)
        # Neither should be promoted (old tentative discarded, new one just started)
        assert len(tracks) == 0

    def test_birth_min_hits_1_promotes_immediately(self):
        tracker = ByteTracker(track_thresh=0.5, match_thresh=0.3, birth_min_hits=1)
        detections = [Detection(bbox=np.array([0, 0, 10, 10]), score=0.9)]
        tracks = tracker.update(detections, timestamp_ns=1)
        assert len(tracks) == 1

    def test_birth_min_hits_3_requires_three_frames(self):
        tracker = ByteTracker(track_thresh=0.5, match_thresh=0.3, birth_min_hits=3)
        det = [Detection(bbox=np.array([0, 0, 10, 10]), score=0.9)]
        tracker.update(det, timestamp_ns=1)
        assert len(tracker._tentative) == 1
        tracks = tracker.update(
            [Detection(bbox=np.array([1, 0, 11, 10]), score=0.9)], timestamp_ns=2
        )
        assert len(tracks) == 0  # age=2, need 3
        tracks = tracker.update(
            [Detection(bbox=np.array([2, 0, 12, 10]), score=0.9)], timestamp_ns=3
        )
        assert len(tracks) == 1  # age=3, promoted!

    def test_reset_clears_tentative(self):
        tracker = ByteTracker(track_thresh=0.5, match_thresh=0.3, birth_min_hits=2)
        tracker.update([Detection(bbox=np.array([0, 0, 10, 10]), score=0.9)], timestamp_ns=1)
        assert len(tracker._tentative) == 1
        tracker.reset()
        assert len(tracker._tentative) == 0


# ── WP2: Servo smoothing config ─────────────────────────────────────────


@pytest.mark.unit
class TestServoSmoothingConfig:

    def test_smoothing_config_has_servo_fields(self):
        cfg = SmoothingConfig()
        assert cfg.servo_mincutoff == 10.0
        assert cfg.servo_beta == 0.2
        assert cfg.display_mincutoff == 0.1
        assert cfg.display_beta == 0.05
        assert cfg.dcutoff == 1.0

    def test_dog_config_has_higher_servo_responsiveness(self):
        dog = default_tracking_config("dog")
        human = default_tracking_config("human")
        assert dog.smoothing.servo_mincutoff > human.smoothing.servo_mincutoff
        assert dog.smoothing.servo_beta > human.smoothing.servo_beta

    def test_smoothing_config_frozen(self):
        cfg = SmoothingConfig()
        with pytest.raises(AttributeError):
            cfg.servo_mincutoff = 99.0  # type: ignore[misc]


# ── WP4: Auto-connect config ────────────────────────────────────────────


@pytest.mark.unit
class TestAutoConnect:

    def test_comm_channel_accepts_auto(self):
        cfg = CommConfig(channel="auto")
        assert cfg.channel == "auto"

    def test_comm_channel_serial_still_works(self):
        cfg = CommConfig(channel="serial")
        assert cfg.channel == "serial"

    def test_comm_channel_udp_still_works(self):
        cfg = CommConfig(channel="udp")
        assert cfg.channel == "udp"


# ── WP1: Kalman config tuning ───────────────────────────────────────────


@pytest.mark.unit
class TestKalmanTuning:

    def test_human_defaults_tuned(self):
        cfg = KalmanConfig()
        assert cfg.innovation_gate_sigma == 5.0
        assert cfg.adaptive_r_speed_thresh == 100.0
        assert cfg.prediction_decay_start == 8
        assert cfg.prediction_velocity_decay == 0.85

    def test_dog_kalman_differs(self):
        dog = default_tracking_config("dog")
        human = default_tracking_config("human")
        assert dog.kalman.innovation_gate_sigma > human.kalman.innovation_gate_sigma
        assert dog.kalman.prediction_decay_start < human.kalman.prediction_decay_start
        assert dog.kalman.prediction_velocity_decay < human.kalman.prediction_velocity_decay

    def test_dog_hold_time_shorter(self):
        dog = default_tracking_config("dog")
        human = default_tracking_config("human")
        assert dog.hold_time_s < human.hold_time_s


# ── S4: Slew-aware adaptive freeze ──────────────────────────────────────

from src.config import AdaptiveConfig, TrackingConfig
from src.tracking.adaptive import AdaptiveController


@pytest.mark.unit
class TestSlewFreeze:

    def _make_controller(self, **adaptive_kwargs) -> AdaptiveController:
        defaults = dict(
            min_detection_samples=5,
            slew_freeze_threshold_deg_s=80.0,
            slew_freeze_hold_frames=5,
        )
        defaults.update(adaptive_kwargs)
        adaptive_cfg = AdaptiveConfig(**defaults)
        tracking = TrackingConfig(adaptive=adaptive_cfg, confidence_threshold=0.45)
        return AdaptiveController(tracking)

    def test_slew_freezes_reliability_update(self):
        """During high-speed slew, detection misses should NOT decay confidence."""
        ctrl = self._make_controller()
        # Fill initial detection window with hits
        for _ in range(10):
            ctrl.update(True, 0.0)
        conf_before = ctrl.confidence_threshold

        # Now simulate high-speed slew with detections failing
        ctrl.notify_camera_motion(150.0)  # above threshold
        for _ in range(5):
            ctrl.update(False, 0.0)

        # Confidence should NOT have dropped (detection misses were frozen)
        assert ctrl.confidence_threshold == conf_before

    def test_slow_motion_does_not_freeze(self):
        """Below slew threshold, detection misses decay confidence normally."""
        ctrl = self._make_controller(
            reliability_low=0.5,
            conf_floor=0.20,
            conf_reduction=0.10,
        )
        # Fill detection window with hits
        for _ in range(10):
            ctrl.update(True, 0.0)
        conf_before = ctrl.confidence_threshold

        # Slow camera motion — no freeze
        ctrl.notify_camera_motion(30.0)
        # Feed many misses
        for _ in range(20):
            ctrl.update(False, 0.0)

        # Confidence should have decreased
        assert ctrl.confidence_threshold < conf_before

    def test_freeze_expires_after_hold_frames(self):
        """After freeze hold frames expire, updates resume normally."""
        ctrl = self._make_controller(
            slew_freeze_hold_frames=3,
            reliability_low=0.5,
            conf_floor=0.20,
            conf_reduction=0.10,
        )
        for _ in range(10):
            ctrl.update(True, 0.0)
        conf_before = ctrl.confidence_threshold

        # Trigger freeze
        ctrl.notify_camera_motion(150.0)
        # 3 frozen frames
        for _ in range(3):
            ctrl.update(False, 0.0)
        assert ctrl.confidence_threshold == conf_before

        # After 3 more frames (freeze expired), misses should take effect
        for _ in range(20):
            ctrl.update(False, 0.0)
        assert ctrl.confidence_threshold < conf_before
