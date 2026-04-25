from __future__ import annotations

from dataclasses import dataclass
from math import atan2, radians, tan

import numpy as np


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    camera_matrix: np.ndarray
    image_size: tuple[int, int]


class CameraModel:
    """Pinhole camera model used for pixel <-> angle conversion.

    Built from a single FOV value (preferred) or as an identity fallback
    for video-mode tests. Lens distortion correction has been removed —
    the deployed phone-camera-on-gimbal mount does not produce visible
    distortion at the working distances, and the prior checkerboard
    calibration path was never reliably reproducible.
    """

    def __init__(self, calibration: CameraCalibration) -> None:
        self._calibration = calibration

    @classmethod
    def identity(cls, width: int, height: int) -> CameraModel:
        camera_matrix = np.array(
            [
                [float(width), 0.0, width / 2.0],
                [0.0, float(height), height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return cls(CameraCalibration(camera_matrix, (width, height)))

    @classmethod
    def from_fov(cls, hfov_degrees: float, width: int, height: int) -> CameraModel:
        """Create camera model from horizontal field-of-view in degrees."""
        fx = (width / 2.0) / tan(radians(hfov_degrees / 2.0))
        fy = fx  # square pixels assumed
        camera_matrix = np.array(
            [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return cls(CameraCalibration(camera_matrix, (width, height)))

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

    def pixel_to_angle(self, px: float, py: float) -> tuple[float, float]:
        # Pinhole projection maps image offsets directly into angular
        # offsets; homography would be incorrect for pan/tilt motion.
        fx, fy = self.focal_lengths_px
        cx = float(self._calibration.camera_matrix[0, 2])
        cy = float(self._calibration.camera_matrix[1, 2])
        return atan2(px - cx, fx), atan2(py - cy, fy)

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
