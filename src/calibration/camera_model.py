from __future__ import annotations

from dataclasses import dataclass
from math import atan2, radians, tan
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    image_size: tuple[int, int]


class CameraModel:
    def __init__(self, calibration: CameraCalibration) -> None:
        self._calibration = calibration
        width, height = calibration.image_size
        self._is_identity = not np.any(calibration.distortion_coefficients)
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            calibration.camera_matrix,
            calibration.distortion_coefficients,
            None,
            calibration.camera_matrix,
            (width, height),
            cv2.CV_32FC1,
        )

    @classmethod
    def identity(cls, width: int, height: int) -> "CameraModel":
        camera_matrix = np.array(
            [[float(width), 0.0, width / 2.0], [0.0, float(height), height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        distortion = np.zeros((5, 1), dtype=np.float32)
        return cls(CameraCalibration(camera_matrix, distortion, (width, height)))

    @classmethod
    def from_fov(cls, hfov_degrees: float, width: int, height: int) -> "CameraModel":
        """Create camera model from horizontal field-of-view in degrees."""
        fx = (width / 2.0) / tan(radians(hfov_degrees / 2.0))
        fy = fx  # square pixels assumed
        camera_matrix = np.array(
            [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        distortion = np.zeros((5, 1), dtype=np.float32)
        return cls(CameraCalibration(camera_matrix, distortion, (width, height)))

    @classmethod
    def load(cls, path: str | Path) -> "CameraModel":
        calibration_file = Path(path)
        if not calibration_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
        payload = np.load(calibration_file)
        return cls(
            CameraCalibration(
                camera_matrix=payload["camera_matrix"],
                distortion_coefficients=payload["distortion_coefficients"],
                image_size=(int(payload["image_width"]), int(payload["image_height"])),
            )
        )

    @property
    def image_size(self) -> tuple[int, int]:
        return self._calibration.image_size

    @property
    def focal_length_px(self) -> float:
        """Return horizontal focal length in pixels."""
        return float(self._calibration.camera_matrix[0, 0])

    @property
    def focal_lengths_px(self) -> tuple[float, float]:
        """Return horizontal and vertical focal lengths in pixels."""
        return (
            float(self._calibration.camera_matrix[0, 0]),
            float(self._calibration.camera_matrix[1, 1]),
        )

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        if self._is_identity:
            return frame
        return cv2.remap(frame, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)

    def pixel_to_angle(self, px: float, py: float) -> tuple[float, float]:
        # Pinhole projection maps image offsets directly into angular
        # offsets; homography would be incorrect for pan/tilt motion.
        fx, fy = self.focal_lengths_px
        cx = float(self._calibration.camera_matrix[0, 2])
        cy = float(self._calibration.camera_matrix[1, 2])
        return atan2(px - cx, fx), atan2(py - cy, fy)

    def pixel_to_angle_with_parallax(
        self,
        px: float,
        py: float,
        depth_m: float,
        offset_x: float,
        offset_y: float,
        offset_z: float,
    ) -> tuple[float, float]:
        """Compute gimbal angles corrected for camera-to-gimbal physical offset.

        Projects the pixel into 3D at the estimated depth, then re-projects
        from the gimbal pivot position to get parallax-corrected aim angles.
        """
        fx, fy = self.focal_lengths_px
        cx = float(self._calibration.camera_matrix[0, 2])
        cy = float(self._calibration.camera_matrix[1, 2])
        # Target position in camera frame (meters)
        x_cam = depth_m * (px - cx) / fx
        y_cam = depth_m * (py - cy) / fy
        z_cam = depth_m
        # Translate to gimbal pivot frame
        x_g = x_cam - offset_x
        y_g = y_cam - offset_y
        z_g = z_cam - offset_z
        if z_g <= 0.0:
            # Target behind gimbal — fall back to direct angles
            return atan2(px - cx, fx), atan2(py - cy, fy)
        return atan2(x_g, z_g), atan2(y_g, z_g)

    def pixel_velocity_to_angular(self, vx_pps: float, vy_pps: float) -> tuple[float, float]:
        """Convert pixel velocity (px/sec) to angular velocity (rad/sec) using focal length."""
        fx, fy = self.focal_lengths_px
        return vx_pps / fx, vy_pps / fy

    def angular_velocity_to_pixel_velocity(
        self,
        omega_pan_rad_s: float,
        omega_tilt_rad_s: float,
        dt_s: float,
    ) -> tuple[float, float]:
        """Convert camera angular velocity into image-plane pixel delta over ``dt_s``.

        Positive camera pan/tilt rotates the camera in the same positive sense
        as ``pixel_to_angle``. A world-stationary target therefore appears to
        move in the opposite image direction, so the returned delta is the raw
        image motion induced by camera ego-motion.
        """
        if dt_s <= 0.0:
            return 0.0, 0.0
        fx, fy = self.focal_lengths_px
        return -omega_pan_rad_s * fx * dt_s, -omega_tilt_rad_s * fy * dt_s
