"""Unit tests for the pure helpers in :mod:`src.inference.bootstrap`.

These cover the same behavior as the thin delegates in ``InferenceProcess``
but avoid spinning up a process shell — useful as a faster smoke layer and
as a regression gate for the delegation contract.
"""

from __future__ import annotations

import logging
from math import isclose

import pytest

from src.config import CameraConfig, ModelConfig, RuntimePaths
from src.inference.bootstrap import (
    compute_dt,
    load_camera_model,
    load_pose_schema,
    resolve_engine_path,
)

_LOGGER = logging.getLogger("test.inference.bootstrap")


# --- resolve_engine_path ---


@pytest.mark.unit
def test_resolve_engine_path_human_points_to_person_engine() -> None:
    path = resolve_engine_path("human", ModelConfig())
    assert "person" in str(path).lower() or "human" in str(path).lower()


@pytest.mark.unit
def test_resolve_engine_path_dog_points_to_dog_engine() -> None:
    path = resolve_engine_path("dog", ModelConfig())
    assert "dog" in str(path).lower()


# --- load_camera_model ---


@pytest.mark.unit
def test_load_camera_model_from_configured_fov() -> None:
    model = load_camera_model(
        mode="camera",
        camera_config=CameraConfig(width=640, height=640, fov=72.0),
        runtime_paths=RuntimePaths(),
        logger=_LOGGER,
    )
    assert model.image_size == (640, 640)
    assert model.focal_length_px > 0


@pytest.mark.unit
def test_load_camera_model_camera_mode_without_fov_raises() -> None:
    with pytest.raises(ValueError, match="camera.fov must be set"):
        load_camera_model(
            mode="camera",
            camera_config=CameraConfig(width=640, height=640, fov=None),
            runtime_paths=RuntimePaths(),
            logger=_LOGGER,
        )


@pytest.mark.unit
def test_load_camera_model_video_mode_falls_back_to_identity() -> None:
    model = load_camera_model(
        mode="video",
        camera_config=CameraConfig(width=640, height=640, fov=None),
        runtime_paths=RuntimePaths(),
        logger=_LOGGER,
    )
    assert isclose(model.focal_length_px, 640.0, rel_tol=1e-5)


# --- load_pose_schema ---


@pytest.mark.unit
def test_load_pose_schema_human_has_17_keypoints() -> None:
    schema = load_pose_schema("human", RuntimePaths(), _LOGGER)
    assert schema.target_kind == "human"
    assert schema.keypoint_count == 17


@pytest.mark.unit
def test_load_pose_schema_dog_is_dog_kind() -> None:
    schema = load_pose_schema("dog", RuntimePaths(), _LOGGER)
    assert schema.target_kind == "dog"


# --- compute_dt ---


@pytest.mark.unit
def test_compute_dt_first_frame_uses_nominal_fps() -> None:
    dt = compute_dt(current_timestamp_ns=1_000_000_000, last_timestamp_ns=None, camera_fps=60)
    assert isclose(dt, 1.0 / 60.0, rel_tol=1e-5)


@pytest.mark.unit
def test_compute_dt_normal_10ms_gap() -> None:
    dt = compute_dt(
        current_timestamp_ns=1_010_000_000,
        last_timestamp_ns=1_000_000_000,
        camera_fps=60,
    )
    assert isclose(dt, 0.01, rel_tol=1e-5)


@pytest.mark.unit
def test_compute_dt_clamps_to_1ms_minimum() -> None:
    dt = compute_dt(
        current_timestamp_ns=1_000_000_000,
        last_timestamp_ns=1_000_000_000,
        camera_fps=60,
    )
    assert dt >= 1e-3


@pytest.mark.unit
def test_compute_dt_handles_zero_fps() -> None:
    # camera_fps <= 0 must not divide-by-zero on the first-frame path.
    dt = compute_dt(current_timestamp_ns=0, last_timestamp_ns=None, camera_fps=0)
    assert dt == 1.0
