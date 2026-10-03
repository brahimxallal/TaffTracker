from __future__ import annotations

import logging
from dataclasses import dataclass
from math import degrees

from src.shared.protocol import FLAG_FAST_MOTION
from src.shared.types import TrackingMessage

LOGGER = logging.getLogger("output.relock")


@dataclass(frozen=True, slots=True)
class RelockEvent:
    frame_id: int
    track_id: int | None
    previous_track_id: int | None
    lost_ms: float
    error_deg: float
    command_delta_deg: float
    command_velocity_dps: float
    confidence: float
    fast_motion: bool


class RelockDiagnostics:
    """Detect and log lost -> measurement transitions on the auto path."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        micro_loss_window_s: float = 0.0,
    ) -> None:
        self._logger = logger or LOGGER
        self._micro_loss_window_ms = max(0.0, micro_loss_window_s) * 1000.0
        self._was_acquired = False
        self._lost_since_ns: int | None = None
        self._last_track_id: int | None = None
        self._last_command_deg: tuple[float, float] | None = None

    def observe(
        self,
        message: TrackingMessage,
        *,
        command_deg: tuple[float, float],
        velocity_dps: tuple[float, float],
        state_flags: int,
    ) -> RelockEvent | None:
        if not message.target_acquired:
            if self._was_acquired or self._lost_since_ns is None:
                self._lost_since_ns = message.timestamp_ns
            if message.track_id is not None:
                self._last_track_id = message.track_id
            self._last_command_deg = command_deg
            self._was_acquired = False
            return None

        event: RelockEvent | None = None
        if not self._was_acquired and message.state_source == "measurement":
            event = self._build_event(
                message,
                command_deg=command_deg,
                velocity_dps=velocity_dps,
                state_flags=state_flags,
            )
            if self._is_same_track_micro_loss(event):
                event = None
            else:
                self._log_event(event)

        self._was_acquired = True
        self._lost_since_ns = None
        self._last_track_id = message.track_id
        self._last_command_deg = command_deg
        return event

    def _is_same_track_micro_loss(self, event: RelockEvent) -> bool:
        return (
            self._micro_loss_window_ms > 0.0
            and event.track_id is not None
            and event.track_id == event.previous_track_id
            and event.lost_ms <= self._micro_loss_window_ms
        )

    def _build_event(
        self,
        message: TrackingMessage,
        *,
        command_deg: tuple[float, float],
        velocity_dps: tuple[float, float],
        state_flags: int,
    ) -> RelockEvent:
        if message.filtered_angles is None:
            error_deg = 0.0
        else:
            pan_rad, tilt_rad = message.filtered_angles
            error_deg = (degrees(pan_rad) ** 2 + degrees(tilt_rad) ** 2) ** 0.5

        if self._last_command_deg is None:
            command_delta_deg = 0.0
        else:
            command_delta_deg = (
                (command_deg[0] - self._last_command_deg[0]) ** 2
                + (command_deg[1] - self._last_command_deg[1]) ** 2
            ) ** 0.5

        lost_ms = (
            (message.timestamp_ns - self._lost_since_ns) / 1_000_000.0
            if self._lost_since_ns is not None
            else 0.0
        )
        command_velocity_dps = (velocity_dps[0] ** 2 + velocity_dps[1] ** 2) ** 0.5
        return RelockEvent(
            frame_id=message.frame_id,
            track_id=message.track_id,
            previous_track_id=self._last_track_id,
            lost_ms=lost_ms,
            error_deg=error_deg,
            command_delta_deg=command_delta_deg,
            command_velocity_dps=command_velocity_dps,
            confidence=message.confidence,
            fast_motion=bool(state_flags & FLAG_FAST_MOTION),
        )

    def _log_event(self, event: RelockEvent) -> None:
        self._logger.info(
            "Relock event: frame=%d track=%s prev=%s lost=%.1fms err=%.2fdeg "
            "cmd_delta=%.2fdeg cmd_vel=%.1fdps conf=%.2f fast=%s",
            event.frame_id,
            event.track_id,
            event.previous_track_id,
            event.lost_ms,
            event.error_deg,
            event.command_delta_deg,
            event.command_velocity_dps,
            event.confidence,
            event.fast_motion,
        )
