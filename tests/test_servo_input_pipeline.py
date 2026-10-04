"""``tracking.servo_input`` decides what position the gimbal controller steers from."""

from __future__ import annotations

from dataclasses import replace
from math import radians
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
from src.shared.types import TrackingMessage
from src.tracking.adaptive import AdaptiveController
from src.tracking.botsort import BoTSORT
from src.tracking.kalman import KalmanFilter
from src.tracking.one_euro import OneEuroFilter2D
from src.tracking.reid import ReIDBuffer

_FPS = 60.0
_REST = (320.0, 320.0)
# 200 px sideways in a single frame: a target that suddenly outruns the Kalman estimate.
_JUMP = (520.0, 320.0)
_CAMERA = CameraModel.from_fov(66.2, 640, 640)


def _make_pipeline(servo_input: str) -> TrackingPipeline:
    tracking_config = replace(
        TrackingConfig(),
        servo_input=servo_input,
        egomotion_compensation_enabled=False,
    )
    kalman = KalmanFilter(
        process_noise=tracking_config.process_noise,
        measurement_noise=tracking_config.measurement_noise,
        config=tracking_config.kalman,
    )
    tracker_stage = TrackerStage(
        tracker=BoTSORT(track_thresh=0.3, match_thresh=0.3, max_lost=30),
        kalman=kalman,
        stabilizer=KeypointStabilizer(),
        reid_buffer=ReIDBuffer(),
        max_lost_frames=30,
    )
    tracker_stage.force_lock(7)
    return TrackingPipeline(
        tracker_stage=tracker_stage,
        centroid_stage=CentroidStage(camera_model=_CAMERA, target="human"),
        servo_stage=ServoStage(laser_detector=None, laser_enabled=None),
        adaptive=AdaptiveController(tracking_config),
        tracking_config=tracking_config,
        pose_schema=get_pose_schema("human"),
        # Near-transparent filters, so the test compares the sources rather than the smoothing.
        ema_pixel=OneEuroFilter2D(mincutoff=1e6),
        servo_ema_pixel=OneEuroFilter2D(mincutoff=1e6),
    )


def _last_message(
    servo_input: str, monkeypatch: pytest.MonkeyPatch
) -> tuple[TrackingMessage, TrackingPipeline]:
    """Hold a target at rest for a second, then move it 200 px in one frame."""
    pipeline = _make_pipeline(servo_input)
    centroid = {"pixel": _REST}
    monkeypatch.setattr(
        "src.inference.pipeline.compute_stabilized_centroid",
        lambda *args, **kwargs: centroid["pixel"],
    )
    track = SimpleNamespace(
        bbox=(300.0, 300.0, 340.0, 340.0),
        score=0.95,
        track_id=7,
        lost_frames=0,
        keypoints=None,
        age=30,
    )
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    message = None
    for frame_id in range(61):
        centroid["pixel"] = _JUMP if frame_id == 60 else _REST
        record = SimpleNamespace(
            frame_id=frame_id, timestamp_ns=1_000_000_000 + frame_id * 16_666_667
        )
        message, _ = pipeline._build_message(
            primary_track=track,
            all_tracks=[track],
            record=record,
            frame=frame,
            was_lost=False,
            dt=1.0 / _FPS,
            fps=_FPS,
            wait_ms=0.1,
            inference_ms=5.0,
            postprocess_ms=1.0,
        )
    assert message is not None
    return message, pipeline


@pytest.mark.unit
def test_measurement_mode_steers_from_the_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    message, pipeline = _last_message("measurement", monkeypatch)

    expected = pipeline.centroid_stage.compute_angles(_JUMP)
    assert message.state_source == "measurement"
    assert message.servo_angles is not None
    assert message.servo_angles == pytest.approx(expected, abs=1e-4)


@pytest.mark.unit
def test_kalman_mode_steers_from_the_filtered_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    message, pipeline = _last_message("kalman", monkeypatch)

    detection_angles = pipeline.centroid_stage.compute_angles(_JUMP)
    assert message.servo_angles is not None
    # The Kalman position has only started to follow the 200 px jump (about 20 deg of pan).
    assert detection_angles[0] - message.servo_angles[0] > radians(5.0)


@pytest.mark.unit
def test_display_estimate_is_the_same_in_both_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    kalman_message, _ = _last_message("kalman", monkeypatch)
    measurement_message, _ = _last_message("measurement", monkeypatch)

    assert measurement_message.filtered_pixel == kalman_message.filtered_pixel
    assert measurement_message.filtered_angles == kalman_message.filtered_angles
    assert measurement_message.raw_pixel == kalman_message.raw_pixel == _JUMP
