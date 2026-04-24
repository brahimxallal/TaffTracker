from __future__ import annotations

import multiprocessing as mp
from math import degrees
from types import SimpleNamespace

import numpy as np
import pytest

from src.calibration.camera_model import CameraModel
from src.config import TrackingConfig
from src.inference.stages.centroid import CentroidStage
from src.shared.types import Track


def _make_centroid_stage(**overrides) -> tuple[CentroidStage, mp.Value, mp.Value]:
    command_pan = mp.Value("d", 0.0)
    command_tilt = mp.Value("d", 0.0)
    camera_model = CameraModel.from_fov(72.0, 640, 640)
    stage = CentroidStage(
        camera_model=camera_model,
        target="human",
    )
    return stage, command_pan, command_tilt


def _make_track(cx: float, cy: float) -> Track:
    return Track(
        track_id=1,
        bbox=np.array([cx - 10.0, cy - 10.0, cx + 10.0, cy + 10.0], dtype=np.float32),
        score=0.95,
        keypoints=None,
        lost_frames=0,
    )


@pytest.mark.unit
def test_compensate_measurement_for_egomotion_applies_inverse_image_shift() -> None:
    stage, command_pan, command_tilt = _make_centroid_stage()
    stage.update_commanded_camera_motion(1_000_000_000, command_pan, command_tilt)
    command_pan.value = degrees(0.1 * 0.02)
    stage.update_commanded_camera_motion(1_020_000_000, command_pan, command_tilt)

    compensation = stage.compensate_measurement_for_egomotion(
        (300.0, 320.0),
        0.02,
        1_020_000_000,
    )

    assert compensation is not None
    assert compensation.applied_delta_px[0] > 0.0
    assert compensation.compensated_pixel[0] > 300.0


@pytest.mark.unit
def test_compensate_measurement_for_egomotion_skips_stale_velocity_sample() -> None:
    stage, command_pan, command_tilt = _make_centroid_stage()
    stage.update_commanded_camera_motion(1_000_000_000, command_pan, command_tilt)
    command_pan.value = degrees(0.1 * 0.02)
    stage.update_commanded_camera_motion(1_020_000_000, command_pan, command_tilt)

    compensation = stage.compensate_measurement_for_egomotion(
        (300.0, 320.0),
        0.02,
        1_100_000_000,
    )

    assert compensation is None


@pytest.mark.unit
def test_build_message_reports_egomotion_applied_px_and_limits_false_velocity() -> None:
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

    camera_model = CameraModel.from_fov(72.0, 640, 640)
    kalman = KalmanFilter(process_noise=1.0, measurement_noise=1.0)
    stabilizer = KeypointStabilizer(alpha=0.2)
    pose_schema = get_pose_schema("human")
    tracking_config = TrackingConfig()
    command_pan = mp.Value("d", 0.0)
    command_tilt = mp.Value("d", 0.0)

    tracker_stage = TrackerStage(
        tracker=BoTSORT(),
        kalman=kalman,
        stabilizer=stabilizer,
        reid_buffer=ReIDBuffer(),
        max_lost_frames=30,
    )
    centroid_stage = CentroidStage(
        camera_model=camera_model,
        target="human",
    )
    servo_stage = ServoStage()
    ema_pixel = OneEuroFilter2D(mincutoff=0.1, beta=0.05, dcutoff=1.0)
    servo_ema_pixel = OneEuroFilter2D(mincutoff=10.0, beta=0.2, dcutoff=1.0)
    adaptive = AdaptiveController(tracking_config)

    pipeline = TrackingPipeline(
        tracker_stage=tracker_stage,
        centroid_stage=centroid_stage,
        servo_stage=servo_stage,
        adaptive=adaptive,
        tracking_config=tracking_config,
        pose_schema=pose_schema,
        ema_pixel=ema_pixel,
        servo_ema_pixel=servo_ema_pixel,
    )

    dt = 1.0 / 60.0
    track0 = _make_track(320.0, 320.0)
    track1 = _make_track(315.0, 320.0)
    record0 = SimpleNamespace(
        frame_id=0, timestamp_ns=1_000_000_000, frame=np.zeros((640, 640, 3), dtype=np.uint8)
    )
    record1 = SimpleNamespace(
        frame_id=1, timestamp_ns=1_016_666_667, frame=np.zeros((640, 640, 3), dtype=np.uint8)
    )

    centroid_stage.update_commanded_camera_motion(record0.timestamp_ns, command_pan, command_tilt)

    message0, was_lost0, _, _ = pipeline.process_frame(
        record=record0,
        undistorted=record0.frame,
        tracks=[track0],
        prev_locked_id=None,
        was_lost=False,
        dt=dt,
        fps=60.0,
        wait_ms=0.0,
        inference_ms=0.0,
        postprocess_ms=0.0,
    )
    assert was_lost0 is False

    command_pan.value = degrees(0.46 * dt)
    centroid_stage.update_commanded_camera_motion(record1.timestamp_ns, command_pan, command_tilt)

    message1, _, _, _ = pipeline.process_frame(
        record=record1,
        undistorted=record1.frame,
        tracks=[track1],
        prev_locked_id=tracker_stage.locked_track_id,
        was_lost=False,
        dt=dt,
        fps=60.0,
        wait_ms=0.0,
        inference_ms=0.0,
        postprocess_ms=0.0,
    )

    assert message0.egomotion_applied_px is None
    assert message1.egomotion_applied_px is not None
    assert abs(message1.filtered_velocity[0]) < 20.0


@pytest.mark.unit
def test_compensation_drops_to_none_when_command_holds_after_motion() -> None:
    stage, command_pan, command_tilt = _make_centroid_stage()

    stage.update_commanded_camera_motion(1_000_000_000, command_pan, command_tilt)
    command_pan.value = degrees(0.2 * 0.02)
    stage.update_commanded_camera_motion(1_020_000_000, command_pan, command_tilt)
    assert stage._last_camera_angular_velocity is not None
    assert stage._last_camera_angular_velocity[0] > 0.0

    stage.update_commanded_camera_motion(1_040_000_000, command_pan, command_tilt)

    compensation = stage.compensate_measurement_for_egomotion(
        (300.0, 320.0),
        0.02,
        1_040_000_000,
    )

    assert compensation is None
