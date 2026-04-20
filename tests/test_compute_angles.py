from __future__ import annotations

import math

import pytest

from src.calibration.camera_model import CameraModel


# ── _compute_angles is a method on InferenceProcess, but we can test
#    the underlying CameraModel.pixel_to_angle + pixel_to_angle_with_parallax
#    directly since _compute_angles just dispatches to them. ─────────


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


# ── Parallax with zero offset matches direct angles ──────────────


@pytest.mark.unit
def test_parallax_zero_offset_matches_direct(model_640x480: CameraModel) -> None:
    px, py = 400.0, 300.0
    depth = 3.0
    pan_direct, tilt_direct = model_640x480.pixel_to_angle(px, py)
    pan_par, tilt_par = model_640x480.pixel_to_angle_with_parallax(
        px, py, depth, 0.0, 0.0, 0.0,
    )
    assert abs(pan_direct - pan_par) < 1e-6
    assert abs(tilt_direct - tilt_par) < 1e-6


# ── Parallax with offset shifts angles ───────────────────────────


@pytest.mark.unit
def test_parallax_with_offset_shifts_angles(model_640x480: CameraModel) -> None:
    px, py = 320.0, 240.0  # center pixel → zero direct angles
    depth = 3.0
    pan_par, tilt_par = model_640x480.pixel_to_angle_with_parallax(
        px, py, depth, 0.05, 0.0, 0.0,  # camera 5cm right of gimbal
    )
    # Gimbal should aim slightly left (negative pan) to compensate
    assert pan_par < 0


# ── Parallax: further depth → smaller correction ─────────────────


@pytest.mark.unit
def test_parallax_decreases_with_depth(model_640x480: CameraModel) -> None:
    px, py = 320.0, 240.0
    _, _, offset_x = 0.0, 0.0, 0.05
    pan_near, _ = model_640x480.pixel_to_angle_with_parallax(
        px, py, 1.0, offset_x, 0.0, 0.0,
    )
    pan_far, _ = model_640x480.pixel_to_angle_with_parallax(
        px, py, 10.0, offset_x, 0.0, 0.0,
    )
    # At greater depth, parallax correction should be smaller
    assert abs(pan_far) < abs(pan_near)


# ── Parallax: target behind gimbal falls back to direct ──────────


@pytest.mark.unit
def test_parallax_fallback_when_behind(model_640x480: CameraModel) -> None:
    px, py = 400.0, 300.0
    # z_cam = depth, offset_z > depth → z_g < 0
    pan_par, tilt_par = model_640x480.pixel_to_angle_with_parallax(
        px, py, depth_m=1.0, offset_x=0.0, offset_y=0.0, offset_z=2.0,
    )
    pan_direct, tilt_direct = model_640x480.pixel_to_angle(px, py)
    assert abs(pan_par - pan_direct) < 1e-6
    assert abs(tilt_par - tilt_direct) < 1e-6


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
