from __future__ import annotations

import logging
from dataclasses import dataclass

from src.calibration.camera_model import CameraModel
from src.config import TargetKind

LOGGER = logging.getLogger("inference.centroid")


@dataclass(frozen=True, slots=True)
class EgomotionCompensation:
    compensated_pixel: tuple[float, float]
    applied_delta_px: tuple[float, float]


class CentroidStage:
    """Egomotion compensation and pixel→angle conversion.

    Depth-based parallax correction has been removed — the camera is
    assumed to be effectively on-axis with the gimbal pivot (measured
    parallax was negligible on this rig). Angles are produced directly
    from the pinhole pixel→angle mapping.
    """

    def __init__(
        self,
        camera_model: CameraModel,
        target: TargetKind,
    ) -> None:
        self._camera_model = camera_model
        self._target = target
        self._last_camera_angular_velocity: tuple[float, float] | None = None
        self._last_camera_angular_velocity_timestamp_ns: int | None = None
        self._last_command_pan_rad: float | None = None
        self._last_command_tilt_rad: float | None = None
        self._last_command_timestamp_ns: int | None = None

    @property
    def camera_model(self) -> CameraModel:
        return self._camera_model

    @property
    def last_camera_angular_velocity(self) -> tuple[float, float] | None:
        return self._last_camera_angular_velocity

    def compensate_measurement_for_egomotion(
        self,
        pixel: tuple[float, float],
        dt: float,
        timestamp_ns: int,
    ) -> EgomotionCompensation | None:
        if (
            self._last_camera_angular_velocity is None
            or self._last_camera_angular_velocity_timestamp_ns is None
            or dt <= 0.0
        ):
            return None
        max_age_ns = int(max(dt * 2.0, 1e-3) * 1_000_000_000)
        age_ns = timestamp_ns - self._last_camera_angular_velocity_timestamp_ns
        if age_ns < 0 or age_ns > max_age_ns:
            return None
        ego_delta_x, ego_delta_y = self._camera_model.angular_velocity_to_pixel_velocity(
            self._last_camera_angular_velocity[0],
            self._last_camera_angular_velocity[1],
            dt,
        )
        if abs(ego_delta_x) < 1e-6 and abs(ego_delta_y) < 1e-6:
            return None
        applied_delta = (-ego_delta_x, -ego_delta_y)
        return EgomotionCompensation(
            compensated_pixel=(pixel[0] + applied_delta[0], pixel[1] + applied_delta[1]),
            applied_delta_px=applied_delta,
        )

    def update_commanded_camera_motion(
        self,
        timestamp_ns: int,
        command_pan,
        command_tilt,
    ) -> None:
        if command_pan is None or command_tilt is None:
            self._last_camera_angular_velocity = None
            self._last_camera_angular_velocity_timestamp_ns = None
            return

        from math import radians

        pan_rad = radians(float(command_pan.value))
        tilt_rad = radians(float(command_tilt.value))
        if (
            self._last_command_timestamp_ns is None
            or self._last_command_pan_rad is None
            or self._last_command_tilt_rad is None
            or timestamp_ns <= self._last_command_timestamp_ns
        ):
            self._last_camera_angular_velocity = None
            self._last_camera_angular_velocity_timestamp_ns = None
        else:
            dt = min((timestamp_ns - self._last_command_timestamp_ns) / 1e9, 0.1)
            if dt <= 0.0:
                self._last_camera_angular_velocity = None
                self._last_camera_angular_velocity_timestamp_ns = None
            else:
                self._last_camera_angular_velocity = (
                    (pan_rad - self._last_command_pan_rad) / dt,
                    (tilt_rad - self._last_command_tilt_rad) / dt,
                )
                self._last_camera_angular_velocity_timestamp_ns = timestamp_ns

        self._last_command_pan_rad = pan_rad
        self._last_command_tilt_rad = tilt_rad
        self._last_command_timestamp_ns = timestamp_ns

    def compute_angles(
        self,
        pixel: tuple[float, float],
    ) -> tuple[float, float]:
        return self._camera_model.pixel_to_angle(*pixel)

    def compute_angular_velocity(
        self,
        pixel: tuple[float, float] | None,
        vx_pps: float,
        vy_pps: float,
    ) -> tuple[float, float]:
        return self._camera_model.pixel_velocity_to_angular(vx_pps, vy_pps)
