from __future__ import annotations

import logging

LOGGER = logging.getLogger("inference.servo")


class ServoStage:
    """Laser-dot overlay detection.

    Historically this stage also ran a PID visual servo that closed the loop
    on the detected laser pixel. That closed-loop path has been removed; the
    stage now only forwards the open-loop ``servo_angles`` produced upstream
    and optionally returns the detected laser centroid for the visualizer.
    """

    def __init__(
        self,
        laser_detector=None,
        laser_roi_radius: float = 150.0,
    ) -> None:
        self.laser_detector = laser_detector
        self._laser_roi_radius = laser_roi_radius

    def process(
        self,
        frame,
        filtered_pixel: tuple[float, float] | None,
        servo_angles: tuple[float, float] | None,
        target_acquired: bool,
        dt: float,
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        """Returns ``(laser_pixel, servo_angles)``.

        ``servo_angles`` is passed through unchanged — the visual-servo PID
        branch that previously overrode it has been removed.
        """
        del target_acquired, dt  # retained for call-site compatibility
        laser_pixel = None

        if self.laser_detector is not None and filtered_pixel is not None:
            det = self.laser_detector.detect(
                frame,
                roi_center=filtered_pixel,
                roi_radius=self._laser_roi_radius,
            )
            if det is not None:
                laser_pixel = det.center

        return laser_pixel, servo_angles
