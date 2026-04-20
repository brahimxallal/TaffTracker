"""Interactive laser-to-camera boresight calibration state machine.

Runs inside the live tracker.  The user presses ``k`` to enter calibration
mode, uses arrow keys to nudge the pan/tilt offset in real time until the
laser dot overlays the tracked centroid, then presses Enter to persist the
offset to JSON (and memory for immediate use).

This module owns only the state machine and key parsing; it does not touch
OpenCV, serial, or the servo output pipeline.  The host process applies
``current_offset()`` at the output stage each frame so nudges are visible
instantly.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.calibration.laser_boresight import LaserBoresight, save_boresight

LOGGER = logging.getLogger(__name__)

# OpenCV extended key codes (cv2.waitKeyEx) — match scripts/calibrate.py.
_KEY_LEFT = 0x250000
_KEY_UP = 0x260000
_KEY_RIGHT = 0x270000
_KEY_DOWN = 0x280000
_KEY_ENTER = 13
_KEY_ESC = 27

# Shift + arrow on Windows (cv2.waitKeyEx raises bit 16).
_SHIFT_MASK = 0x010000

FINE_STEP_DEG = 0.1
COARSE_STEP_DEG = 1.0

_SaveCallback = Callable[[LaserBoresight], None]


class CalibrationState(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"


@dataclass
class _NudgeResult:
    consumed: bool
    saved: bool = False
    aborted: bool = False


class LaserCalibrator:
    """Owns the boresight-calibration mode toggled by the ``k`` hotkey.

    The host renders ``current_offset()`` into the live servo command each
    frame so the laser dot moves as the user nudges.  On ``Enter`` the offset
    is persisted via ``save_path`` (and the host keeps using it).  On ``Esc``
    the prior offset is restored.
    """

    def __init__(self, save_path: Path, initial: LaserBoresight | None = None) -> None:
        self._save_path = save_path
        self._current = initial or LaserBoresight.zero()
        self._snapshot = self._current
        self._state = CalibrationState.IDLE

    # --- state queries -------------------------------------------------
    @property
    def state(self) -> CalibrationState:
        return self._state

    @property
    def active(self) -> bool:
        return self._state is CalibrationState.ACTIVE

    def current_offset(self) -> LaserBoresight:
        return self._current

    # --- lifecycle -----------------------------------------------------
    def start(self) -> None:
        if self._state is CalibrationState.ACTIVE:
            return
        self._snapshot = self._current
        self._state = CalibrationState.ACTIVE
        LOGGER.info(
            "Laser calibration started (current pan=%+.3f° tilt=%+.3f°)",
            self._current.pan_offset_deg,
            self._current.tilt_offset_deg,
        )

    def abort(self) -> None:
        if self._state is CalibrationState.IDLE:
            return
        restored = self._snapshot
        self._current = restored
        self._state = CalibrationState.IDLE
        LOGGER.info(
            "Laser calibration aborted; restored pan=%+.3f° tilt=%+.3f°",
            restored.pan_offset_deg,
            restored.tilt_offset_deg,
        )

    def commit(self) -> LaserBoresight:
        """Persist the current offset and exit calibration mode."""
        if self._state is CalibrationState.IDLE:
            return self._current
        save_boresight(self._save_path, self._current)
        self._snapshot = self._current
        self._state = CalibrationState.IDLE
        LOGGER.info(
            "Laser calibration saved to %s (pan=%+.3f° tilt=%+.3f°)",
            self._save_path,
            self._current.pan_offset_deg,
            self._current.tilt_offset_deg,
        )
        return self._current

    # --- input ---------------------------------------------------------
    def handle_key(self, raw_key: int) -> bool:
        """Consume a key while active. Returns True if the key was handled.

        Host processes should call this before dispatching to other key
        handlers so calibration arrow-nudges do not leak into manual-jog.
        """
        if self._state is CalibrationState.IDLE:
            return False
        if raw_key in (-1, 0):
            return False

        # Check exact (unshifted) arrow keys first — fine step.
        # Must be checked before shift-mask stripping because bit 16
        # (_SHIFT_MASK) is already set inside _KEY_LEFT and _KEY_RIGHT.
        if raw_key == _KEY_LEFT:
            self._apply_delta(-FINE_STEP_DEG, 0.0)
            return True
        if raw_key == _KEY_RIGHT:
            self._apply_delta(+FINE_STEP_DEG, 0.0)
            return True
        if raw_key == _KEY_UP:
            self._apply_delta(0.0, +FINE_STEP_DEG)
            return True
        if raw_key == _KEY_DOWN:
            self._apply_delta(0.0, -FINE_STEP_DEG)
            return True

        # Shifted arrows — coarse step (mask-stripped).
        base = raw_key & ~_SHIFT_MASK
        if base in (_KEY_LEFT, _KEY_RIGHT, _KEY_UP, _KEY_DOWN):
            if base == _KEY_LEFT:
                self._apply_delta(-COARSE_STEP_DEG, 0.0)
            elif base == _KEY_RIGHT:
                self._apply_delta(+COARSE_STEP_DEG, 0.0)
            elif base == _KEY_UP:
                self._apply_delta(0.0, +COARSE_STEP_DEG)
            else:
                self._apply_delta(0.0, -COARSE_STEP_DEG)
            return True

        low = raw_key & 0xFF
        if low == _KEY_ENTER:
            self.commit()
            return True
        if low == _KEY_ESC:
            self.abort()
            return True
        return False

    def _apply_delta(self, d_pan: float, d_tilt: float) -> None:
        self._current = LaserBoresight(
            pan_offset_deg=self._current.pan_offset_deg + d_pan,
            tilt_offset_deg=self._current.tilt_offset_deg + d_tilt,
        )

    # --- HUD helpers ---------------------------------------------------
    def hud_line(self) -> str:
        b = self._current
        return (
            f"LASER CAL  pan={b.pan_offset_deg:+.3f}°  "
            f"tilt={b.tilt_offset_deg:+.3f}°  "
            "<-/->/up/dn=0.1°  SHIFT=1.0°  ENTER=save  ESC=cancel"
        )
