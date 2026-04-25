"""Tests for InferenceProcess helper methods that don't require TensorRT."""

from __future__ import annotations

import multiprocessing as mp
from math import degrees, isclose

import pytest

from src.config import (
    CameraConfig,
    ModelConfig,
    RuntimePaths,
    TrackingConfig,
)
from src.inference.process import InferenceProcess
from src.shared.ring_buffer import SharedRingBuffer


def _build_inference_proc(**overrides) -> InferenceProcess:
    ring_buffer, write_index = SharedRingBuffer.create((4, 4, 3), num_slots=2)
    ring_buffer.cleanup()
    command_pan = mp.Value("d", 0.0)
    command_tilt = mp.Value("d", 0.0)
    defaults = dict(
        layout=ring_buffer.layout,
        write_index=write_index,
        result_queue=mp.Queue(),
        capture_done_event=mp.Event(),
        shutdown_event=mp.Event(),
        error_queue=mp.Queue(),
        mode="camera",
        target="human",
        camera_config=CameraConfig(width=640, height=640, fps=60, fov=72.0),
        tracking_config=TrackingConfig(),
        model_config=ModelConfig(),
        runtime_paths=RuntimePaths(),
        command_pan=command_pan,
        command_tilt=command_tilt,
    )
    defaults.update(overrides)
    return InferenceProcess(**defaults)


# --- _load_camera_model ---


@pytest.mark.unit
def test_load_camera_model_from_fov() -> None:
    proc = _build_inference_proc(camera_config=CameraConfig(width=640, height=640, fov=72.0))
    model = proc._load_camera_model()
    assert model.image_size == (640, 640)
    assert model.focal_length_px > 0


@pytest.mark.unit
def test_load_camera_model_rejects_missing_intrinsics_in_camera_mode() -> None:
    proc = _build_inference_proc(camera_config=CameraConfig(width=640, height=640, fov=None))

    with pytest.raises(ValueError, match="camera.fov must be set"):
        proc._load_camera_model()


@pytest.mark.unit
def test_load_camera_model_identity_fallback_in_video_mode() -> None:
    proc = _build_inference_proc(
        mode="video",
        camera_config=CameraConfig(width=640, height=640, fov=None),
    )

    model = proc._load_camera_model()

    assert model.image_size == (640, 640)
    # Identity model has focal_length = width
    assert isclose(model.focal_length_px, 640.0, rel_tol=1e-5)


# --- _compute_dt ---


@pytest.mark.unit
def test_compute_dt_first_call() -> None:
    proc = _build_inference_proc(camera_config=CameraConfig(fps=60))
    dt = proc._compute_dt(1_000_000_000, None)
    assert isclose(dt, 1.0 / 60.0, rel_tol=1e-5)


@pytest.mark.unit
def test_compute_dt_normal() -> None:
    proc = _build_inference_proc()
    # 10ms apart
    dt = proc._compute_dt(1_010_000_000, 1_000_000_000)
    assert isclose(dt, 0.01, rel_tol=1e-5)


@pytest.mark.unit
def test_compute_dt_clamps_minimum() -> None:
    proc = _build_inference_proc()
    # Same timestamp → clamps to 1e-3
    dt = proc._compute_dt(1_000_000_000, 1_000_000_000)
    assert dt >= 1e-3


@pytest.mark.unit
def test_update_commanded_camera_motion_tracks_shared_command_velocity() -> None:
    from src.calibration.camera_model import CameraModel
    from src.inference.stages.centroid import CentroidStage

    stage = CentroidStage(
        camera_model=CameraModel.from_fov(72.0, 640, 640),
        target="human",
    )
    command_pan = mp.Value("d", 0.0)
    command_tilt = mp.Value("d", 0.0)

    stage.update_commanded_camera_motion(1_000_000_000, command_pan, command_tilt)
    command_pan.value = degrees(0.03)
    command_tilt.value = degrees(-0.01)
    stage.update_commanded_camera_motion(1_100_000_000, command_pan, command_tilt)

    assert stage._last_camera_angular_velocity is not None
    assert isclose(stage._last_camera_angular_velocity[0], 0.3, rel_tol=1e-5)
    assert isclose(stage._last_camera_angular_velocity[1], -0.1, rel_tol=1e-5)


# --- _load_pose_schema ---


@pytest.mark.unit
def test_load_pose_schema_human() -> None:
    proc = _build_inference_proc(target="human")
    schema = proc._load_pose_schema()
    assert schema.target_kind == "human"
    assert schema.keypoint_count == 17


@pytest.mark.unit
def test_load_pose_schema_dog() -> None:
    proc = _build_inference_proc(target="dog")
    schema = proc._load_pose_schema()
    assert schema.target_kind == "dog"


# --- _resolve_engine_path ---


@pytest.mark.unit
def test_resolve_engine_path_human() -> None:
    proc = _build_inference_proc(target="human")
    path = proc._resolve_engine_path()
    assert "person" in str(path).lower() or "human" in str(path).lower()


@pytest.mark.unit
def test_resolve_engine_path_dog() -> None:
    proc = _build_inference_proc(target="dog")
    path = proc._resolve_engine_path()
    assert "dog" in str(path).lower()
