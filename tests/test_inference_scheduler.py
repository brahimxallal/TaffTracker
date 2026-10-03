from __future__ import annotations

import pytest

from src.config import InferenceScheduleConfig
from src.inference.scheduler import DetectorScheduler


@pytest.mark.unit
def test_scheduler_detects_when_no_predictable_lock() -> None:
    scheduler = DetectorScheduler(InferenceScheduleConfig(detector_locked_stride=3))

    decision = scheduler.should_run_detector(
        frame_id=10,
        timestamp_ns=10_000_000,
        has_predictable_lock=False,
        force_detector=False,
    )

    assert decision.run_detector is True
    assert decision.reason == "search"


@pytest.mark.unit
def test_scheduler_skips_between_high_confidence_detector_frames() -> None:
    scheduler = DetectorScheduler(
        InferenceScheduleConfig(
            detector_locked_stride=3,
            detector_min_skip_confidence=0.60,
            detector_max_prediction_age_ms=120.0,
        )
    )
    scheduler.mark_detector_result(frame_id=1, timestamp_ns=1_000_000, confidence=0.82)

    first_skip = scheduler.should_run_detector(
        frame_id=2,
        timestamp_ns=18_000_000,
        has_predictable_lock=True,
        force_detector=False,
    )
    second_skip = scheduler.should_run_detector(
        frame_id=3,
        timestamp_ns=35_000_000,
        has_predictable_lock=True,
        force_detector=False,
    )
    next_detect = scheduler.should_run_detector(
        frame_id=4,
        timestamp_ns=52_000_000,
        has_predictable_lock=True,
        force_detector=False,
    )

    assert first_skip.run_detector is False
    assert second_skip.run_detector is False
    assert next_detect.run_detector is True
    assert next_detect.reason == "stride"


@pytest.mark.unit
def test_scheduler_forces_detector_when_last_confidence_is_too_low() -> None:
    scheduler = DetectorScheduler(
        InferenceScheduleConfig(detector_locked_stride=3, detector_min_skip_confidence=0.60)
    )
    scheduler.mark_detector_result(frame_id=1, timestamp_ns=1_000_000, confidence=0.50)

    decision = scheduler.should_run_detector(
        frame_id=2,
        timestamp_ns=18_000_000,
        has_predictable_lock=True,
        force_detector=False,
    )

    assert decision.run_detector is True
    assert decision.reason == "low_confidence"


@pytest.mark.unit
def test_scheduler_limits_prediction_age_even_inside_stride() -> None:
    scheduler = DetectorScheduler(
        InferenceScheduleConfig(
            detector_locked_stride=10,
            detector_min_skip_confidence=0.60,
            detector_max_prediction_age_ms=30.0,
        )
    )
    scheduler.mark_detector_result(frame_id=1, timestamp_ns=1_000_000, confidence=0.90)

    decision = scheduler.should_run_detector(
        frame_id=2,
        timestamp_ns=40_000_000,
        has_predictable_lock=True,
        force_detector=False,
    )

    assert decision.run_detector is True
    assert decision.reason == "prediction_age"
