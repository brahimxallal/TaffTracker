from __future__ import annotations

from dataclasses import dataclass

from src.config import InferenceScheduleConfig


@dataclass(frozen=True)
class DetectorScheduleDecision:
    run_detector: bool
    reason: str


class DetectorScheduler:
    """Decide when to run the heavy detector during an active lock.

    The detector still runs continuously while searching. Once a reliable
    lock exists, planned prediction frames are allowed between detector frames.
    """

    def __init__(self, config: InferenceScheduleConfig) -> None:
        self._config = config
        self._last_detector_frame_id: int | None = None
        self._last_detector_timestamp_ns: int | None = None
        self._last_detector_confidence: float = 0.0

    @property
    def last_detector_confidence(self) -> float:
        return self._last_detector_confidence

    def mark_detector_result(
        self,
        *,
        frame_id: int,
        timestamp_ns: int,
        confidence: float,
    ) -> None:
        self._last_detector_frame_id = int(frame_id)
        self._last_detector_timestamp_ns = int(timestamp_ns)
        self._last_detector_confidence = max(0.0, min(1.0, float(confidence)))

    def should_run_detector(
        self,
        *,
        frame_id: int,
        timestamp_ns: int,
        has_predictable_lock: bool,
        force_detector: bool,
    ) -> DetectorScheduleDecision:
        cfg = self._config
        if not cfg.detector_skip_enabled:
            return DetectorScheduleDecision(True, "disabled")
        if force_detector:
            return DetectorScheduleDecision(True, "forced")
        if not has_predictable_lock:
            return DetectorScheduleDecision(True, "search")
        if self._last_detector_frame_id is None or self._last_detector_timestamp_ns is None:
            return DetectorScheduleDecision(True, "cold_start")
        if self._last_detector_confidence < cfg.detector_min_skip_confidence:
            return DetectorScheduleDecision(True, "low_confidence")

        frames_since_detector = int(frame_id) - self._last_detector_frame_id
        if frames_since_detector >= max(1, int(cfg.detector_locked_stride)):
            return DetectorScheduleDecision(True, "stride")

        age_ms = (int(timestamp_ns) - self._last_detector_timestamp_ns) / 1_000_000.0
        if age_ms >= cfg.detector_max_prediction_age_ms:
            return DetectorScheduleDecision(True, "prediction_age")

        return DetectorScheduleDecision(False, "predict")
