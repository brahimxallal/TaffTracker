from __future__ import annotations

import logging
from math import radians

from src.tracking.visual_servo import VisualServoController, ServoMode

LOGGER = logging.getLogger("inference.servo")


class ServoStage:
    """Wraps laser detection + visual servo PID correction."""

    def __init__(
        self,
        laser_detector=None,
        servo_controller: VisualServoController | None = None,
        laser_roi_radius: float = 150.0,
    ) -> None:
        self.laser_detector = laser_detector
        self.servo_controller = servo_controller
        self._laser_roi_radius = laser_roi_radius

    def process(
        self,
        frame,
        filtered_pixel: tuple[float, float] | None,
        servo_angles: tuple[float, float] | None,
        target_acquired: bool,
        dt: float,
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None, str]:
        """Returns (laser_pixel, servo_angles, servo_mode)."""
        laser_pixel = None
        servo_mode = "acquisition"

        if self.laser_detector is not None and filtered_pixel is not None:
            det = self.laser_detector.detect(
                frame,
                roi_center=filtered_pixel,
                roi_radius=self._laser_roi_radius,
            )
            if det is not None:
                laser_pixel = det.center

        if self.servo_controller is not None:
            vs_state = self.servo_controller.update(
                target_pixel=filtered_pixel,
                laser_pixel=laser_pixel,
                open_loop_angles_rad=servo_angles,
                target_acquired=target_acquired,
                dt=dt,
            )
            servo_mode = vs_state.mode.value
            if vs_state.mode == ServoMode.VISUAL_SERVO:
                servo_angles = (
                    radians(vs_state.commanded_pan_deg),
                    radians(vs_state.commanded_tilt_deg),
                )

        return laser_pixel, servo_angles, servo_mode
