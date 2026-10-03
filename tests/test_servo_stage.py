from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

from src.inference.stages.servo import ServoStage


class _Detector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame, *, roi_center, roi_radius):
        del frame, roi_center, roi_radius
        self.calls += 1
        return None


@pytest.mark.unit
def test_servo_stage_skips_laser_detector_when_runtime_laser_is_disabled() -> None:
    detector = _Detector()
    laser_enabled = mp.Value("b", 0)
    stage = ServoStage(laser_detector=detector, laser_enabled=laser_enabled)

    laser_pixel, servo_angles = stage.process(
        np.zeros((32, 32, 3), dtype=np.uint8),
        filtered_pixel=(16.0, 16.0),
        servo_angles=(0.1, 0.2),
        target_acquired=True,
        dt=1.0 / 60.0,
    )

    assert laser_pixel is None
    assert servo_angles == (0.1, 0.2)
    assert detector.calls == 0


@pytest.mark.unit
def test_servo_stage_runs_laser_detector_when_runtime_laser_is_enabled() -> None:
    detector = _Detector()
    laser_enabled = mp.Value("b", 1)
    stage = ServoStage(laser_detector=detector, laser_enabled=laser_enabled)

    stage.process(
        np.zeros((32, 32, 3), dtype=np.uint8),
        filtered_pixel=(16.0, 16.0),
        servo_angles=(0.1, 0.2),
        target_acquired=True,
        dt=1.0 / 60.0,
    )

    assert detector.calls == 1
