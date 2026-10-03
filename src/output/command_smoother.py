"""Outgoing servo command smoothing helpers.

The PC controller already limits many internal rates, but final outgoing
pan/tilt positions can still jump when manual input or relock/search state
changes abruptly. This module provides a tiny reusable slew limiter for the
last step before packet encoding.
"""

from __future__ import annotations


def _slew_toward(current: float, desired: float, max_delta: float) -> float:
    if desired > current + max_delta:
        return current + max_delta
    if desired < current - max_delta:
        return current - max_delta
    return desired


class CommandSlewLimiter:
    """Stateful per-axis position slew limiter for pan/tilt degrees."""

    def __init__(self) -> None:
        self._pan_deg: float = 0.0
        self._tilt_deg: float = 0.0
        self._initialized: bool = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def value(self) -> tuple[float, float]:
        return self._pan_deg, self._tilt_deg

    def reset(self) -> None:
        self._pan_deg = 0.0
        self._tilt_deg = 0.0
        self._initialized = False

    def seed(self, pan_deg: float, tilt_deg: float) -> None:
        self._pan_deg = pan_deg
        self._tilt_deg = tilt_deg
        self._initialized = True

    def limit(
        self,
        pan_deg: float,
        tilt_deg: float,
        *,
        dt_s: float,
        limit_dps: float,
    ) -> tuple[float, float]:
        """Return a command capped to ``limit_dps`` from the previous output.

        ``limit_dps <= 0`` disables smoothing. The first command seeds the
        limiter so enabling it does not create startup lag.
        """
        if not self._initialized or limit_dps <= 0.0 or dt_s <= 0.0:
            self.seed(pan_deg, tilt_deg)
            return pan_deg, tilt_deg

        max_delta = max(0.0, limit_dps) * dt_s
        self._pan_deg = _slew_toward(self._pan_deg, pan_deg, max_delta)
        self._tilt_deg = _slew_toward(self._tilt_deg, tilt_deg, max_delta)
        return self._pan_deg, self._tilt_deg


class _BacklashAxisState:
    def __init__(self) -> None:
        self.previous_command_deg: float | None = None
        self.direction: int = 0
        self.offset_deg: float = 0.0

    def reset(self) -> None:
        self.previous_command_deg = None
        self.direction = 0
        self.offset_deg = 0.0


def _direction_from_delta(delta: float, min_step_deg: float) -> int:
    if delta > min_step_deg:
        return 1
    if delta < -min_step_deg:
        return -1
    return 0


class BacklashCompensator:
    """Short reversal pulse to take up servo/gear slack without a steady bias."""

    def __init__(self) -> None:
        self._pan = _BacklashAxisState()
        self._tilt = _BacklashAxisState()

    def reset(self) -> None:
        self._pan.reset()
        self._tilt.reset()

    def compensate(
        self,
        pan_command_deg: float,
        tilt_command_deg: float,
        *,
        dt_s: float,
        enabled: bool,
        pan_deg_amount: float | None = None,
        tilt_deg_amount: float | None = None,
        pan_deg: float | None = None,
        tilt_deg: float | None = None,
        min_step_deg: float = 0.08,
        pulse_decay_dps: float = 3.0,
    ) -> tuple[float, float]:
        """Return pan/tilt with a tiny pulse when commanded direction reverses.

        ``pan_deg``/``tilt_deg`` are accepted as keyword aliases for legacy
        tests/config vocabulary; internally they mean compensation amount.
        """
        pan_amount = pan_deg_amount if pan_deg_amount is not None else (pan_deg or 0.0)
        tilt_amount = tilt_deg_amount if tilt_deg_amount is not None else (tilt_deg or 0.0)
        if not enabled:
            self.reset()
            self._pan.previous_command_deg = pan_command_deg
            self._tilt.previous_command_deg = tilt_command_deg
            return pan_command_deg, tilt_command_deg

        return (
            self._compensate_axis(
                self._pan,
                pan_command_deg,
                backlash_deg=pan_amount,
                dt_s=dt_s,
                min_step_deg=min_step_deg,
                pulse_decay_dps=pulse_decay_dps,
            ),
            self._compensate_axis(
                self._tilt,
                tilt_command_deg,
                backlash_deg=tilt_amount,
                dt_s=dt_s,
                min_step_deg=min_step_deg,
                pulse_decay_dps=pulse_decay_dps,
            ),
        )

    def _compensate_axis(
        self,
        axis: _BacklashAxisState,
        command_deg: float,
        *,
        backlash_deg: float,
        dt_s: float,
        min_step_deg: float,
        pulse_decay_dps: float,
    ) -> float:
        if axis.previous_command_deg is None:
            axis.previous_command_deg = command_deg
            return command_deg

        delta = command_deg - axis.previous_command_deg
        axis.previous_command_deg = command_deg
        direction = _direction_from_delta(delta, max(0.0, min_step_deg))
        if backlash_deg > 0.0 and direction != 0:
            if axis.direction != 0 and direction != axis.direction:
                axis.offset_deg = direction * backlash_deg
            axis.direction = direction

        if axis.offset_deg != 0.0 and pulse_decay_dps > 0.0 and dt_s > 0.0:
            axis.offset_deg = _slew_toward(
                axis.offset_deg,
                0.0,
                max(0.0, pulse_decay_dps) * dt_s,
            )
        return command_deg + axis.offset_deg
