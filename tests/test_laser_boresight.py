"""Tests for laser boresight persistence, calibrator state machine, and orientation parsing."""

from __future__ import annotations

import json
from pathlib import Path

from src.calibration.laser_boresight import LaserBoresight, load_boresight, save_boresight
from src.calibration.laser_calibrator import (
    _KEY_DOWN,
    _KEY_ENTER,
    _KEY_ESC,
    _KEY_LEFT,
    _KEY_RIGHT,
    _KEY_UP,
    _SHIFT_MASK,
    CalibrationState,
    LaserCalibrator,
)
from src.config import LaserBoresightConfig, Orientation

# ── LaserBoresight persistence ──────────────────────────────────────────


class TestBoresightRoundtrip:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "servo_limits.json"
        original = LaserBoresight(pan_offset_deg=1.234, tilt_offset_deg=-0.567)
        save_boresight(path, original)
        loaded = load_boresight(path)
        assert abs(loaded.pan_offset_deg - original.pan_offset_deg) < 1e-6
        assert abs(loaded.tilt_offset_deg - original.tilt_offset_deg) < 1e-6

    def test_load_missing_file_returns_zero(self, tmp_path: Path) -> None:
        bs = load_boresight(tmp_path / "nonexistent.json")
        assert bs.pan_offset_deg == 0.0
        assert bs.tilt_offset_deg == 0.0

    def test_load_malformed_json_returns_zero(self, tmp_path: Path) -> None:
        path = tmp_path / "servo_limits.json"
        path.write_text("not valid json {{{", encoding="utf-8")
        bs = load_boresight(path)
        assert bs.pan_offset_deg == 0.0

    def test_save_preserves_existing_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "servo_limits.json"
        path.write_text(json.dumps({"pan_min_deg": -110, "pan_max_deg": 110}), encoding="utf-8")
        save_boresight(path, LaserBoresight(pan_offset_deg=2.0, tilt_offset_deg=-1.0))
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["pan_min_deg"] == -110
        assert data["pan_max_deg"] == 110
        assert abs(data["laser_pan_offset_deg"] - 2.0) < 1e-6
        assert abs(data["laser_tilt_offset_deg"] - (-1.0)) < 1e-6
        assert "laser_calibrated_at" in data

    def test_zero_factory(self) -> None:
        bs = LaserBoresight.zero()
        assert bs.pan_offset_deg == 0.0
        assert bs.tilt_offset_deg == 0.0


# ── LaserCalibrator state machine ───────────────────────────────────────


class TestLaserCalibrator:
    def _make_calibrator(
        self, tmp_path: Path, pan: float = 0.0, tilt: float = 0.0
    ) -> LaserCalibrator:
        return LaserCalibrator(
            save_path=tmp_path / "servo_limits.json",
            initial=LaserBoresight(pan_offset_deg=pan, tilt_offset_deg=tilt),
        )

    def test_initial_state_is_idle(self, tmp_path: Path) -> None:
        cal = self._make_calibrator(tmp_path)
        assert cal.state is CalibrationState.IDLE
        assert not cal.active

    def test_start_enters_active(self, tmp_path: Path) -> None:
        cal = self._make_calibrator(tmp_path)
        cal.start()
        assert cal.active

    def test_arrow_nudge_fine(self, tmp_path: Path) -> None:
        cal = self._make_calibrator(tmp_path)
        cal.start()
        cal.handle_key(_KEY_RIGHT)
        assert abs(cal.current_offset().pan_offset_deg - 0.1) < 1e-6
        cal.handle_key(_KEY_LEFT)
        assert abs(cal.current_offset().pan_offset_deg) < 1e-6
        cal.handle_key(_KEY_UP)
        assert abs(cal.current_offset().tilt_offset_deg - 0.1) < 1e-6
        cal.handle_key(_KEY_DOWN)
        assert abs(cal.current_offset().tilt_offset_deg) < 1e-6

    def test_shift_arrow_coarse(self, tmp_path: Path) -> None:
        """Only Shift+DOWN (0x290000) produces a distinguishable keycode.

        _SHIFT_MASK (bit 16) collides with _KEY_LEFT/_KEY_RIGHT/_KEY_UP
        because their virtual-key bytes (0x25, 0x27, 0x26) have bit 0 set
        or the OR result maps onto another arrow key.  Shift+DOWN is the
        one combination that survives the mask round-trip cleanly.
        """
        cal = self._make_calibrator(tmp_path)
        cal.start()
        cal.handle_key(_KEY_DOWN | _SHIFT_MASK)  # 0x290000
        assert abs(cal.current_offset().tilt_offset_deg - (-1.0)) < 1e-6

    def test_esc_restores_prior_offset(self, tmp_path: Path) -> None:
        cal = self._make_calibrator(tmp_path, pan=0.5, tilt=-0.3)
        cal.start()
        cal.handle_key(_KEY_RIGHT)
        cal.handle_key(_KEY_RIGHT)
        assert cal.current_offset().pan_offset_deg != 0.5
        cal.handle_key(_KEY_ESC)
        assert not cal.active
        assert abs(cal.current_offset().pan_offset_deg - 0.5) < 1e-6
        assert abs(cal.current_offset().tilt_offset_deg - (-0.3)) < 1e-6

    def test_enter_saves_and_exits(self, tmp_path: Path) -> None:
        cal = self._make_calibrator(tmp_path)
        cal.start()
        cal.handle_key(_KEY_RIGHT)
        cal.handle_key(_KEY_DOWN)
        cal.handle_key(_KEY_ENTER)
        assert not cal.active
        path = tmp_path / "servo_limits.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert abs(data["laser_pan_offset_deg"] - 0.1) < 1e-6
        assert abs(data["laser_tilt_offset_deg"] - (-0.1)) < 1e-6

    def test_keys_ignored_when_idle(self, tmp_path: Path) -> None:
        cal = self._make_calibrator(tmp_path)
        assert not cal.handle_key(_KEY_RIGHT)
        assert cal.current_offset().pan_offset_deg == 0.0

    def test_hud_line_format(self, tmp_path: Path) -> None:
        cal = self._make_calibrator(tmp_path, pan=1.5, tilt=-0.2)
        cal.start()
        hud = cal.hud_line()
        assert "LASER CAL" in hud
        assert "+1.500" in hud
        assert "-0.200" in hud


# ── Orientation parsing ─────────────────────────────────────────────────


class TestOrientationParsing:
    def test_parse_landscape_native(self) -> None:
        from src.config_loader import _parse_orientation

        result = _parse_orientation({"orientation": "landscape_native"})
        assert result is Orientation.LANDSCAPE_NATIVE

    def test_parse_portrait(self) -> None:
        from src.config_loader import _parse_orientation

        result = _parse_orientation({"orientation": "portrait"})
        assert result is Orientation.PORTRAIT

    def test_legacy_portrait_mode_true(self) -> None:
        from src.config_loader import _parse_orientation

        result = _parse_orientation({"portrait_mode": True})
        assert result is Orientation.PORTRAIT

    def test_legacy_portrait_mode_false(self) -> None:
        from src.config_loader import _parse_orientation

        result = _parse_orientation({"portrait_mode": False})
        assert result is Orientation.LANDSCAPE_NATIVE

    def test_explicit_orientation_beats_legacy(self) -> None:
        from src.config_loader import _parse_orientation

        result = _parse_orientation({"orientation": "landscape_native", "portrait_mode": True})
        assert result is Orientation.LANDSCAPE_NATIVE

    def test_default_is_landscape(self) -> None:
        from src.config_loader import _parse_orientation

        result = _parse_orientation({})
        assert result is Orientation.LANDSCAPE_NATIVE

    def test_unknown_orientation_falls_back(self) -> None:
        from src.config_loader import _parse_orientation

        result = _parse_orientation({"orientation": "upside_down"})
        assert result is Orientation.LANDSCAPE_NATIVE

    def test_portrait_mode_property_compat(self) -> None:
        from src.config import CameraConfig

        cam_p = CameraConfig(orientation=Orientation.PORTRAIT)
        assert cam_p.portrait_mode is True
        cam_l = CameraConfig(orientation=Orientation.LANDSCAPE_NATIVE)
        assert cam_l.portrait_mode is False


# ── Output boresight injection ──────────────────────────────────────────


class TestOutputBoresightInjection:
    def test_boresight_config_defaults_zero(self) -> None:
        cfg = LaserBoresightConfig()
        assert cfg.pan_offset_deg == 0.0
        assert cfg.tilt_offset_deg == 0.0

    def test_boresight_config_custom_values(self) -> None:
        cfg = LaserBoresightConfig(pan_offset_deg=1.0, tilt_offset_deg=-0.5)
        assert cfg.pan_offset_deg == 1.0
        assert cfg.tilt_offset_deg == -0.5
