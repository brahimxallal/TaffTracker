"""Laser-to-camera boresight offset persistence.

The laser emitter is rigidly mounted on the gimbal but offset from the camera
optical axis.  This module persists the constant angular offset learned by the
interactive calibrator so the servo output stage can shift commanded angles by
that amount, putting the laser dot (not the camera center) on the target.

Persistence lives in ``calibration_data/servo_limits.json`` alongside the
existing servo-center offsets; we only add three keys rather than introducing a
new file.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

_PAN_KEY = "laser_pan_offset_deg"
_TILT_KEY = "laser_tilt_offset_deg"
_TIMESTAMP_KEY = "laser_calibrated_at"


@dataclass(frozen=True)
class LaserBoresight:
    """Static pan/tilt offset applied to servo commands so the laser hits the target."""

    pan_offset_deg: float = 0.0
    tilt_offset_deg: float = 0.0

    @classmethod
    def zero(cls) -> "LaserBoresight":
        return cls()


def load_boresight(path: Path) -> LaserBoresight:
    """Load boresight offsets from the shared servo-limits JSON.

    Returns a zero offset if the file is missing or the keys are absent (the
    system has not been calibrated yet).  Any malformed value is logged and
    reset to zero so a bad file cannot brick the tracker.
    """
    if not path.exists():
        return LaserBoresight.zero()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Could not read laser boresight from %s: %s", path, exc)
        return LaserBoresight.zero()
    try:
        pan = float(data.get(_PAN_KEY, 0.0))
        tilt = float(data.get(_TILT_KEY, 0.0))
    except (TypeError, ValueError) as exc:
        LOGGER.warning("Malformed laser boresight values in %s: %s", path, exc)
        return LaserBoresight.zero()
    return LaserBoresight(pan_offset_deg=pan, tilt_offset_deg=tilt)


def save_boresight(path: Path, boresight: LaserBoresight) -> None:
    """Merge boresight offsets into the servo-limits JSON, creating it if needed.

    Existing keys (servo center offsets, mechanical limits) are preserved; only
    the three laser-related keys are written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, object] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except (json.JSONDecodeError, OSError) as exc:
            LOGGER.warning("Replacing unreadable %s: %s", path, exc)
    existing[_PAN_KEY] = float(boresight.pan_offset_deg)
    existing[_TILT_KEY] = float(boresight.tilt_offset_deg)
    existing[_TIMESTAMP_KEY] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
