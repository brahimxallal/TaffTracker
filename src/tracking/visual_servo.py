"""Closed-loop visual servoing: PID controller + state machine.

Compares target pixel position with laser dot pixel position and drives
incremental corrections via PID to minimize the error. Falls back to
open-loop (ACQUISITION mode) when the laser dot is not visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import degrees


class ServoMode(str, Enum):
    ACQUISITION = "acquisition"
    VISUAL_SERVO = "visual_servo"
    LOST = "lost"


@dataclass(frozen=True, slots=True)
class VisualServoState:
    mode: ServoMode
    commanded_pan_deg: float
    commanded_tilt_deg: float
    error_px: tuple[float, float]  # (ex, ey) in pixels


class PIDAxis:
    """Single-axis PID with anti-windup (integral clamp + conditional integration)."""

    __slots__ = (
        "_kp",
        "_ki",
        "_kd",
        "_integral_limit",
        "_output_limit",
        "_integral",
        "_prev_error",
        "_initialized",
    )

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        integral_limit: float,
        output_limit: float,
    ) -> None:
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._integral_limit = integral_limit
        self._output_limit = output_limit
        self._integral = 0.0
        self._prev_error = 0.0
        self._initialized = False

    def update(self, error: float, dt: float) -> float:
        p = self._kp * error

        # Conditional integration: only accumulate when output is not saturated
        candidate_i = self._integral + error * dt
        candidate_i = max(-self._integral_limit, min(self._integral_limit, candidate_i))
        i = self._ki * candidate_i

        dt = max(dt, 1e-3)
        if self._initialized:
            d = self._kd * (error - self._prev_error) / dt
        else:
            d = 0.0
            self._initialized = True

        output = p + i + d
        # Only commit integral if output is within limits (anti-windup)
        if abs(output) < self._output_limit:
            self._integral = candidate_i

        output = max(-self._output_limit, min(self._output_limit, output))
        self._prev_error = error
        return output

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._initialized = False


class VisualServoController:
    """State machine + PID for closed-loop laser-guided tracking.

    States:
        ACQUISITION:   Open-loop. Waiting for laser to appear near target.
        VISUAL_SERVO:  Closed-loop PID on (target - laser) pixel error.
        LOST:          Target lost. Reset PID, wait for re-acquisition.

    Transitions:
        ACQUISITION → VISUAL_SERVO: laser near target for
            `entry_threshold_frames` consecutive frames.
        VISUAL_SERVO → ACQUISITION: laser lost for
            `exit_threshold_frames` consecutive frames.
        Any → LOST:                 target_acquired=False.
        LOST → ACQUISITION:         target_acquired=True.
    """

    def __init__(
        self,
        *,
        kp: float,
        ki: float,
        kd: float,
        integral_limit_deg: float,
        max_correction_deg: float,
        entry_threshold_frames: int,
        exit_threshold_frames: int,
        association_radius_px: float,
        deg_per_pixel_x: float,
        deg_per_pixel_y: float,
    ) -> None:
        self._pid_pan = PIDAxis(kp, ki, kd, integral_limit_deg, max_correction_deg)
        self._pid_tilt = PIDAxis(kp, ki, kd, integral_limit_deg, max_correction_deg)
        self._entry_threshold = entry_threshold_frames
        self._exit_threshold = exit_threshold_frames
        self._assoc_radius = association_radius_px
        self._dpx = deg_per_pixel_x
        self._dpy = deg_per_pixel_y
        self._max_correction = max_correction_deg

        self._mode = ServoMode.ACQUISITION
        self._consecutive_laser = 0
        self._consecutive_no_laser = 0
        self._commanded_pan = 0.0  # degrees
        self._commanded_tilt = 0.0  # degrees

    @property
    def mode(self) -> ServoMode:
        return self._mode

    def update(
        self,
        *,
        target_pixel: tuple[float, float] | None,
        laser_pixel: tuple[float, float] | None,
        open_loop_angles_rad: tuple[float, float] | None,
        target_acquired: bool,
        dt: float,
    ) -> VisualServoState:
        # --- Handle target lost ---
        if not target_acquired:
            if self._mode != ServoMode.LOST:
                self._mode = ServoMode.LOST
                self._pid_pan.reset()
                self._pid_tilt.reset()
                self._consecutive_laser = 0
                self._consecutive_no_laser = 0
            return VisualServoState(
                mode=ServoMode.LOST,
                commanded_pan_deg=self._commanded_pan,
                commanded_tilt_deg=self._commanded_tilt,
                error_px=(0.0, 0.0),
            )

        # --- Target acquired: check laser association ---
        laser_near = False
        error_px = (0.0, 0.0)
        if target_pixel is not None and laser_pixel is not None:
            ex = target_pixel[0] - laser_pixel[0]
            ey = target_pixel[1] - laser_pixel[1]
            dist = (ex * ex + ey * ey) ** 0.5
            if dist <= self._assoc_radius:
                laser_near = True
                error_px = (ex, ey)

        # --- LOST → ACQUISITION on target re-acquisition ---
        if self._mode == ServoMode.LOST:
            self._mode = ServoMode.ACQUISITION
            self._consecutive_laser = 0
            self._consecutive_no_laser = 0
            # Seed commanded position from open-loop
            if open_loop_angles_rad is not None:
                self._commanded_pan = degrees(open_loop_angles_rad[0])
                self._commanded_tilt = degrees(open_loop_angles_rad[1])

        # --- State transitions ---
        if self._mode == ServoMode.ACQUISITION:
            # Seed commanded position from open-loop every frame in acquisition
            if open_loop_angles_rad is not None:
                self._commanded_pan = degrees(open_loop_angles_rad[0])
                self._commanded_tilt = degrees(open_loop_angles_rad[1])

            if laser_near:
                self._consecutive_laser += 1
                self._consecutive_no_laser = 0
            else:
                self._consecutive_laser = 0

            if self._consecutive_laser >= self._entry_threshold:
                self._mode = ServoMode.VISUAL_SERVO
                self._pid_pan.reset()
                self._pid_tilt.reset()

        elif self._mode == ServoMode.VISUAL_SERVO:
            if laser_near:
                self._consecutive_no_laser = 0
                # PID correction in degree space
                error_deg_x = error_px[0] * self._dpx
                error_deg_y = error_px[1] * self._dpy
                correction_pan = self._pid_pan.update(error_deg_x, dt)
                correction_tilt = self._pid_tilt.update(error_deg_y, dt)
                self._commanded_pan += correction_pan
                self._commanded_tilt += correction_tilt
                # Clamp total correction range
                if open_loop_angles_rad is not None:
                    base_pan = degrees(open_loop_angles_rad[0])
                    base_tilt = degrees(open_loop_angles_rad[1])
                    self._commanded_pan = max(
                        base_pan - self._max_correction,
                        min(base_pan + self._max_correction, self._commanded_pan),
                    )
                    self._commanded_tilt = max(
                        base_tilt - self._max_correction,
                        min(base_tilt + self._max_correction, self._commanded_tilt),
                    )
            else:
                self._consecutive_no_laser += 1
                if self._consecutive_no_laser >= self._exit_threshold:
                    self._mode = ServoMode.ACQUISITION
                    self._consecutive_laser = 0
                    self._pid_pan.reset()
                    self._pid_tilt.reset()
                    # Reset commanded position so acquisition re-seeds from open-loop
                    self._commanded_pan = 0.0
                    self._commanded_tilt = 0.0

        return VisualServoState(
            mode=self._mode,
            commanded_pan_deg=self._commanded_pan,
            commanded_tilt_deg=self._commanded_tilt,
            error_px=error_px,
        )
