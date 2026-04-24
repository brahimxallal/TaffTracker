from __future__ import annotations

import math

import pytest

from src.calibration.camera_model import CameraModel

# ── _compute_angles is a method on InferenceProcess, but we can test
#    the underlying CameraModel.pixel_to_angle directly since
#    _compute_angles just dispatches to it. ─────────


@pytest.fixture
def model_640x480() -> CameraModel:
    return CameraModel.from_fov(hfov_degrees=90.0, width=640, height=480)


# ── Center pixel → zero angles ───────────────────────────────────


@pytest.mark.unit
def test_center_pixel_zero_angles(model_640x480: CameraModel) -> None:
    pan, tilt = model_640x480.pixel_to_angle(320.0, 240.0)
    assert abs(pan) < 1e-6
    assert abs(tilt) < 1e-6


# ── Right of center → positive pan ───────────────────────────────


@pytest.mark.unit
def test_right_of_center_positive_pan(model_640x480: CameraModel) -> None:
    pan, _ = model_640x480.pixel_to_angle(500.0, 240.0)
    assert pan > 0


# ── Below center → positive tilt ─────────────────────────────────


@pytest.mark.unit
def test_below_center_positive_tilt(model_640x480: CameraModel) -> None:
    _, tilt = model_640x480.pixel_to_angle(320.0, 400.0)
    assert tilt > 0


# ── Edge pixel → ~45° for 90° HFOV ───────────────────────────────


@pytest.mark.unit
def test_edge_pixel_45_degrees(model_640x480: CameraModel) -> None:
    pan, _ = model_640x480.pixel_to_angle(640.0, 240.0)
    assert abs(math.degrees(pan) - 45.0) < 0.5


# ── Symmetry: left/right are equal magnitude ─────────────────────


@pytest.mark.unit
def test_pixel_to_angle_symmetry(model_640x480: CameraModel) -> None:
    pan_left, tilt_left = model_640x480.pixel_to_angle(200.0, 240.0)
    pan_right, tilt_right = model_640x480.pixel_to_angle(440.0, 240.0)
    assert abs(pan_left + pan_right) < 1e-6
    assert abs(tilt_left - tilt_right) < 1e-6


# ── pixel_velocity_to_angular ────────────────────────────────────


@pytest.mark.unit
def test_pixel_velocity_to_angular(model_640x480: CameraModel) -> None:
    vx, vy = model_640x480.pixel_velocity_to_angular(320.0, 240.0)
    # For 90° HFOV: fx = 320, so 320 px/s → 1 rad/s
    assert abs(vx - 1.0) < 0.01
    # fy = fx = 320 (from_fov uses square pixels), 240/320 = 0.75
    assert abs(vy - 0.75) < 0.01


# ── from_fov identity check ─────────────────────────────────────


@pytest.mark.unit
def test_from_fov_focal_length() -> None:
    model = CameraModel.from_fov(90.0, 640, 480)
    # For 90° HFOV: fx = (640/2) / tan(45°) = 320
    assert abs(model.focal_length_px - 320.0) < 0.1


# ── identity model: focal length equals width ────────────────────


@pytest.mark.unit
def test_identity_model_focal_length() -> None:
    model = CameraModel.identity(640, 480)
    assert abs(model.focal_length_px - 640.0) < 0.1
