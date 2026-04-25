from math import degrees, isclose, radians, tan

import pytest

from src.calibration.camera_model import CameraModel


@pytest.mark.unit
def test_identity_camera_model_maps_center_to_zero_angle() -> None:
    model = CameraModel.identity(640, 640)

    angle_x, angle_y = model.pixel_to_angle(320.0, 320.0)

    assert isclose(angle_x, 0.0, abs_tol=1e-9)
    assert isclose(angle_y, 0.0, abs_tol=1e-9)


@pytest.mark.unit
def test_identity_camera_model_maps_right_edge_to_positive_angle() -> None:
    model = CameraModel.identity(640, 640)

    angle_x, _ = model.pixel_to_angle(640.0, 320.0)

    assert angle_x > 0.0


# --- from_fov tests ---


@pytest.mark.unit
def test_from_fov_center_maps_to_zero_angle() -> None:
    model = CameraModel.from_fov(66.2, 640, 640)

    angle_x, angle_y = model.pixel_to_angle(320.0, 320.0)

    assert isclose(angle_x, 0.0, abs_tol=1e-9)
    assert isclose(angle_y, 0.0, abs_tol=1e-9)


@pytest.mark.unit
def test_from_fov_computes_correct_focal_length() -> None:
    """For 66.2 deg HFOV at 640px, fx should be ~491.7."""
    model = CameraModel.from_fov(66.2, 640, 640)
    fx = float(model._calibration.camera_matrix[0, 0])
    expected_fx = 320.0 / tan(radians(66.2 / 2.0))

    assert isclose(fx, expected_fx, rel_tol=1e-6)
    assert isclose(fx, 491.7, rel_tol=0.01)


@pytest.mark.unit
def test_from_fov_edge_angle_matches_half_fov() -> None:
    """Right edge of frame should map to approximately half the FOV."""
    hfov = 66.2
    model = CameraModel.from_fov(hfov, 640, 640)

    angle_x, _ = model.pixel_to_angle(640.0, 320.0)

    assert isclose(degrees(angle_x), hfov / 2.0, rel_tol=0.01)


@pytest.mark.unit
def test_from_fov_wider_fov_gives_larger_angles() -> None:
    """Wider FOV → same pixel offset produces larger angle."""
    narrow = CameraModel.from_fov(50.0, 640, 640)
    wide = CameraModel.from_fov(90.0, 640, 640)

    narrow_angle, _ = narrow.pixel_to_angle(480.0, 320.0)
    wide_angle, _ = wide.pixel_to_angle(480.0, 320.0)

    assert wide_angle > narrow_angle


@pytest.mark.unit
def test_from_fov_velocity_scaling() -> None:
    """Wider FOV → same px/s velocity produces larger angular velocity."""
    narrow = CameraModel.from_fov(50.0, 640, 640)
    wide = CameraModel.from_fov(90.0, 640, 640)

    narrow_vx, _ = narrow.pixel_velocity_to_angular(100.0, 0.0)
    wide_vx, _ = wide.pixel_velocity_to_angular(100.0, 0.0)

    assert wide_vx > narrow_vx


@pytest.mark.unit
def test_angular_velocity_to_pixel_velocity_round_trip() -> None:
    model = CameraModel.from_fov(72.0, 640, 640)

    dx_px, dy_px = model.angular_velocity_to_pixel_velocity(0.2, -0.1, 0.5)

    vx_rad_s, vy_rad_s = model.pixel_velocity_to_angular(dx_px / 0.5, dy_px / 0.5)

    assert isclose(vx_rad_s, -0.2, rel_tol=1e-6)
    assert isclose(vy_rad_s, 0.1, rel_tol=1e-6)


@pytest.mark.unit
def test_positive_pan_velocity_moves_stationary_target_left_in_image() -> None:
    model = CameraModel.from_fov(72.0, 640, 640)

    dx_px, dy_px = model.angular_velocity_to_pixel_velocity(0.1, 0.0, 0.2)

    assert dx_px < 0.0
    assert isclose(dy_px, 0.0, abs_tol=1e-9)


# --- load() tests ---


# --- image_size and focal_length_px ---


@pytest.mark.unit
def test_image_size_property() -> None:
    model = CameraModel.from_fov(72.0, 1280, 720)
    assert model.image_size == (1280, 720)


@pytest.mark.unit
def test_focal_length_px_from_fov() -> None:
    model = CameraModel.from_fov(90.0, 640, 640)
    expected_fx = (640 / 2.0) / tan(radians(45.0))
    assert isclose(model.focal_length_px, expected_fx, rel_tol=1e-5)


@pytest.mark.unit
def test_focal_lengths_px_returns_both_axes() -> None:
    model = CameraModel.from_fov(90.0, 640, 480)

    assert model.focal_lengths_px == pytest.approx((320.0, 320.0))
