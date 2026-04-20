from __future__ import annotations

from collections import deque
from statistics import mean

from src.config import AdaptiveConfig, TrackingConfig


class AdaptiveController:
    """Runtime adaptive tracking parameter controller.

    Adjusts confidence_threshold and hold_time_s based on rolling
    observations of detection reliability and target speed. Designed to
    replace static scene presets (indoor/outdoor/sport) with a single
    unified mode that auto-tunes to conditions.
    """

    def __init__(self, base_config: TrackingConfig) -> None:
        self._base = base_config
        self._cfg: AdaptiveConfig = base_config.adaptive
        self._detection_hits: deque[bool] = deque(maxlen=self._cfg.detection_window)
        self._speed_window: deque[float] = deque(maxlen=self._cfg.speed_window)
        self.confidence_threshold: float = base_config.confidence_threshold
        self.hold_time_s: float = base_config.hold_time_s
        self._slew_freeze_remaining: int = 0

    def notify_camera_motion(self, angular_speed_deg_s: float) -> None:
        """Freeze reliability updates during high-speed camera slews.

        When the camera is rotating fast, pose confidence drops due to
        motion blur and rolling shutter. Without this freeze, the
        adaptive controller would lower the confidence threshold,
        admitting junk detections.
        """
        if angular_speed_deg_s >= self._cfg.slew_freeze_threshold_deg_s:
            self._slew_freeze_remaining = self._cfg.slew_freeze_hold_frames

    def update(self, detected: bool, speed: float) -> None:
        """Call once per frame with detection result and target speed (px/s)."""
        # During slew freeze, skip detection window updates
        if self._slew_freeze_remaining > 0:
            self._slew_freeze_remaining -= 1
            self._speed_window.append(speed)
            return

        self._detection_hits.append(detected)
        self._speed_window.append(speed)
        cfg = self._cfg

        # --- Confidence auto-tune (linear interpolation) ---
        if len(self._detection_hits) >= cfg.min_detection_samples:
            reliability = sum(self._detection_hits) / len(self._detection_hits)
            base_conf = self._base.confidence_threshold
            if reliability < cfg.reliability_low:
                self.confidence_threshold = max(cfg.conf_floor, base_conf - cfg.conf_reduction)
            elif reliability > cfg.reliability_high:
                self.confidence_threshold = min(cfg.conf_ceiling, base_conf + cfg.conf_raise)
            else:
                # Smooth interpolation between low and high thresholds
                denom = cfg.reliability_high - cfg.reliability_low
                t = (reliability - cfg.reliability_low) / denom if denom > 0 else 0.5
                low_conf = max(cfg.conf_floor, base_conf - cfg.conf_reduction)
                high_conf = min(cfg.conf_ceiling, base_conf + cfg.conf_raise)
                self.confidence_threshold = low_conf + t * (high_conf - low_conf)

        # --- Hold time auto-tune (linear interpolation) ---
        if len(self._speed_window) >= cfg.min_speed_samples:
            avg_speed = mean(self._speed_window)
            base_hold = self._base.hold_time_s
            if avg_speed > cfg.fast_speed_thresh:
                self.hold_time_s = max(
                    cfg.hold_time_min_s, base_hold - cfg.hold_time_fast_reduction
                )
            elif avg_speed < cfg.slow_speed_thresh:
                self.hold_time_s = min(cfg.hold_time_max_s, base_hold + cfg.hold_time_slow_increase)
            else:
                # Smooth interpolation: slow → base → fast
                denom = cfg.fast_speed_thresh - cfg.slow_speed_thresh
                t = (avg_speed - cfg.slow_speed_thresh) / denom if denom > 0 else 0.5
                slow_hold = min(cfg.hold_time_max_s, base_hold + cfg.hold_time_slow_increase)
                fast_hold = max(cfg.hold_time_min_s, base_hold - cfg.hold_time_fast_reduction)
                self.hold_time_s = slow_hold + t * (fast_hold - slow_hold)

    def reset(self) -> None:
        """Reset rolling windows (e.g. on target lock change)."""
        self._detection_hits.clear()
        self._speed_window.clear()
        self.confidence_threshold = self._base.confidence_threshold
        self.hold_time_s = self._base.hold_time_s
