from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.calibration.camera_model import CameraModel
from src.config import TrackingConfig
from src.inference.pipeline import TrackingPipeline
from src.inference.postprocess import KeypointStabilizer
from src.inference.stages.centroid import CentroidStage
from src.inference.stages.servo import ServoStage
from src.inference.stages.tracker import TrackerStage
from src.shared.pose_schema import get_pose_schema
from src.tracking.adaptive import AdaptiveController
from src.tracking.botsort import BoTSORT
from src.tracking.kalman import KalmanFilter
from src.tracking.one_euro import OneEuroFilter2D
from src.tracking.reid import ReIDBuffer


@pytest.mark.unit
def test_planned_prediction_frame_keeps_target_acquired_for_control() -> None:
    tracking_config = TrackingConfig()
    kalman = KalmanFilter(process_noise=1.0, measurement_noise=1.0)
    kalman.update((320.0, 300.0), dt=1.0 / 60.0)
    tracker_stage = TrackerStage(
        tracker=BoTSORT(track_thresh=0.3, match_thresh=0.3, max_lost=30),
        kalman=kalman,
        stabilizer=KeypointStabilizer(),
        reid_buffer=ReIDBuffer(),
        max_lost_frames=30,
    )
    tracker_stage.force_lock(7)
    centroid_stage = CentroidStage(
        camera_model=CameraModel.from_fov(66.2, 640, 640),
        target="human",
    )
    pipeline = TrackingPipeline(
        tracker_stage=tracker_stage,
        centroid_stage=centroid_stage,
        servo_stage=ServoStage(laser_detector=None, laser_enabled=None),
        adaptive=AdaptiveController(tracking_config),
        tracking_config=tracking_config,
        pose_schema=get_pose_schema("human"),
        ema_pixel=OneEuroFilter2D(),
        servo_ema_pixel=OneEuroFilter2D(),
    )
    record = SimpleNamespace(frame_id=2, timestamp_ns=1_016_000_000)
    frame = np.zeros((640, 640, 3), dtype=np.uint8)

    message, next_was_lost, current_locked, _ = pipeline.process_prediction_frame(
        record=record,
        frame=frame,
        prev_locked_id=7,
        was_lost=False,
        dt=1.0 / 60.0,
        fps=60.0,
        wait_ms=0.1,
        prediction_confidence=0.76,
    )

    assert message.state_source == "prediction"
    assert message.target_acquired is True
    assert message.confidence == pytest.approx(0.76)
    assert message.track_id == 7
    assert message.servo_angles is not None
    assert next_was_lost is False
    assert current_locked == 7
