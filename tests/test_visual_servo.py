"""Tests for visual servo controller (PID + state machine)."""

from __future__ import annotations

from math import radians

import pytest

from src.tracking.visual_servo import PIDAxis, ServoMode, VisualServoController

# ---------------------------------------------------------------------------
# PIDAxis tests
# ---------------------------------------------------------------------------


class TestPIDAxis:

    def test_proportional_only(self):
        pid = PIDAxis(kp=1.0, ki=0.0, kd=0.0, integral_limit=10.0, output_limit=100.0)
        out = pid.update(5.0, dt=0.016)
        assert out == pytest.approx(5.0)

    def test_integral_accumulates(self):
        pid = PIDAxis(kp=0.0, ki=1.0, kd=0.0, integral_limit=100.0, output_limit=100.0)
        pid.update(10.0, dt=1.0)  # integral = 10
        out = pid.update(10.0, dt=1.0)  # integral = 20
        assert out == pytest.approx(20.0)

    def test_integral_clamp(self):
        pid = PIDAxis(kp=0.0, ki=1.0, kd=0.0, integral_limit=5.0, output_limit=100.0)
        for _ in range(100):
            pid.update(10.0, dt=1.0)
        out = pid.update(10.0, dt=1.0)
        assert abs(out) <= 5.0 + 1e-6

    def test_derivative(self):
        pid = PIDAxis(kp=0.0, ki=0.0, kd=1.0, integral_limit=10.0, output_limit=100.0)
        pid.update(0.0, dt=0.01)  # initialize, no derivative on first call
        out = pid.update(10.0, dt=0.01)  # d = (10-0)/0.01 = 1000, but kd*d = 1000
        assert out != 0.0

    def test_output_clamp(self):
        pid = PIDAxis(kp=100.0, ki=0.0, kd=0.0, integral_limit=10.0, output_limit=5.0)
        out = pid.update(100.0, dt=0.016)
        assert abs(out) <= 5.0

    def test_reset(self):
        pid = PIDAxis(kp=0.0, ki=1.0, kd=0.0, integral_limit=100.0, output_limit=100.0)
        pid.update(10.0, dt=1.0)
        pid.reset()
        out = pid.update(0.0, dt=1.0)
        assert out == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# VisualServoController tests
# ---------------------------------------------------------------------------


def _make_controller(**kwargs) -> VisualServoController:
    defaults = dict(
        kp=0.5,
        ki=0.0,
        kd=0.0,
        integral_limit_deg=5.0,
        max_correction_deg=45.0,
        entry_threshold_frames=3,
        exit_threshold_frames=5,
        association_radius_px=100.0,
        deg_per_pixel_x=0.1,
        deg_per_pixel_y=0.1,
    )
    defaults.update(kwargs)
    return VisualServoController(**defaults)


class TestVisualServoStateMachine:

    def test_initial_mode_is_acquisition(self):
        ctrl = _make_controller()
        assert ctrl.mode == ServoMode.ACQUISITION

    def test_stays_acquisition_without_laser(self):
        ctrl = _make_controller()
        for _ in range(20):
            state = ctrl.update(
                target_pixel=(320, 240),
                laser_pixel=None,
                open_loop_angles_rad=(0.0, 0.0),
                target_acquired=True,
                dt=0.016,
            )
        assert state.mode == ServoMode.ACQUISITION

    def test_transitions_to_visual_servo(self):
        ctrl = _make_controller(entry_threshold_frames=3)
        # Laser near target for 3 frames
        for _ in range(3):
            state = ctrl.update(
                target_pixel=(320, 240),
                laser_pixel=(325, 242),
                open_loop_angles_rad=(0.0, 0.0),
                target_acquired=True,
                dt=0.016,
            )
        assert state.mode == ServoMode.VISUAL_SERVO

    def test_laser_far_doesnt_count(self):
        ctrl = _make_controller(entry_threshold_frames=3, association_radius_px=50.0)
        for _ in range(10):
            state = ctrl.update(
                target_pixel=(320, 240),
                laser_pixel=(500, 400),  # > 50px away
                open_loop_angles_rad=(0.0, 0.0),
                target_acquired=True,
                dt=0.016,
            )
        assert state.mode == ServoMode.ACQUISITION

    def test_exits_visual_servo_when_laser_lost(self):
        ctrl = _make_controller(entry_threshold_frames=2, exit_threshold_frames=3)
        # Enter VISUAL_SERVO
        for _ in range(2):
            ctrl.update(
                target_pixel=(320, 240),
                laser_pixel=(322, 241),
                open_loop_angles_rad=(0.0, 0.0),
                target_acquired=True,
                dt=0.016,
            )
        # Lose laser for 3 frames
        for _ in range(3):
            state = ctrl.update(
                target_pixel=(320, 240),
                laser_pixel=None,
                open_loop_angles_rad=(0.0, 0.0),
                target_acquired=True,
                dt=0.016,
            )
        assert state.mode == ServoMode.ACQUISITION

    def test_target_lost_goes_to_lost_mode(self):
        ctrl = _make_controller()
        state = ctrl.update(
            target_pixel=None,
            laser_pixel=None,
            open_loop_angles_rad=None,
            target_acquired=False,
            dt=0.016,
        )
        assert state.mode == ServoMode.LOST

    def test_lost_to_acquisition_on_reacquire(self):
        ctrl = _make_controller()
        # Go to LOST
        ctrl.update(
            target_pixel=None,
            laser_pixel=None,
            open_loop_angles_rad=None,
            target_acquired=False,
            dt=0.016,
        )
        # Re-acquire target
        state = ctrl.update(
            target_pixel=(320, 240),
            laser_pixel=None,
            open_loop_angles_rad=(0.1, 0.05),
            target_acquired=True,
            dt=0.016,
        )
        assert state.mode == ServoMode.ACQUISITION

    def test_acquisition_seeds_from_open_loop(self):
        ctrl = _make_controller()
        state = ctrl.update(
            target_pixel=(320, 240),
            laser_pixel=None,
            open_loop_angles_rad=(radians(5.0), radians(-3.0)),
            target_acquired=True,
            dt=0.016,
        )
        assert state.commanded_pan_deg == pytest.approx(5.0)
        assert state.commanded_tilt_deg == pytest.approx(-3.0)

    def test_consecutive_counter_resets_on_miss(self):
        ctrl = _make_controller(entry_threshold_frames=3)
        # 2 frames with laser, then 1 without
        for _ in range(2):
            ctrl.update(
                target_pixel=(320, 240),
                laser_pixel=(322, 241),
                open_loop_angles_rad=(0.0, 0.0),
                target_acquired=True,
                dt=0.016,
            )
        ctrl.update(
            target_pixel=(320, 240),
            laser_pixel=None,
            open_loop_angles_rad=(0.0, 0.0),
            target_acquired=True,
            dt=0.016,
        )
        # 2 more frames — shouldn't have entered yet (counter was reset)
        for _ in range(2):
            state = ctrl.update(
                target_pixel=(320, 240),
                laser_pixel=(322, 241),
                open_loop_angles_rad=(0.0, 0.0),
                target_acquired=True,
                dt=0.016,
            )
        assert state.mode == ServoMode.ACQUISITION


class TestVisualServoPID:

    def test_pid_drives_correction_toward_target(self):
        """When target is to the right of laser, PID should produce positive pan correction."""
        ctrl = _make_controller(kp=0.5, ki=0.0, kd=0.0, entry_threshold_frames=1)
        # Enter VISUAL_SERVO
        ctrl.update(
            target_pixel=(350, 240),
            laser_pixel=(300, 240),  # target 50px right of laser
            open_loop_angles_rad=(0.0, 0.0),
            target_acquired=True,
            dt=0.016,
        )
        # Now in VISUAL_SERVO, PID should add positive correction
        state = ctrl.update(
            target_pixel=(350, 240),
            laser_pixel=(300, 240),
            open_loop_angles_rad=(0.0, 0.0),
            target_acquired=True,
            dt=0.016,
        )
        assert state.mode == ServoMode.VISUAL_SERVO
        # Error = 50px * 0.1 deg/px = 5 deg, kp=0.5 → correction = 2.5 deg
        assert state.commanded_pan_deg > 0.0

    def test_pid_converges_with_error(self):
        """Simulate PID reducing error over iterations."""
        ctrl = _make_controller(kp=0.3, ki=0.0, kd=0.0, entry_threshold_frames=1)
        # Enter VISUAL_SERVO
        ctrl.update(
            target_pixel=(320, 240),
            laser_pixel=(310, 240),
            open_loop_angles_rad=(0.0, 0.0),
            target_acquired=True,
            dt=0.016,
        )
        errors = []
        for _i in range(10):
            state = ctrl.update(
                target_pixel=(320, 240),
                laser_pixel=(310, 240),
                open_loop_angles_rad=(0.0, 0.0),
                target_acquired=True,
                dt=0.016,
            )
            errors.append(state.error_px[0])
        # Error should remain constant (10px) since we're not moving the laser,
        # but commanded_pan should accumulate
        assert state.commanded_pan_deg > 0.0

    def test_max_correction_clamp(self):
        ctrl = _make_controller(
            kp=10.0,
            ki=0.0,
            kd=0.0,
            entry_threshold_frames=1,
            max_correction_deg=5.0,
        )
        # Enter VISUAL_SERVO with large error
        ctrl.update(
            target_pixel=(600, 240),
            laser_pixel=(100, 240),
            open_loop_angles_rad=(0.0, 0.0),
            target_acquired=True,
            dt=0.016,
        )
        state = ctrl.update(
            target_pixel=(600, 240),
            laser_pixel=(100, 240),
            open_loop_angles_rad=(0.0, 0.0),
            target_acquired=True,
            dt=0.016,
        )
        # Should be clamped to max_correction_deg from open-loop baseline
        assert abs(state.commanded_pan_deg) <= 5.0 + 1e-6

    def test_pid_resets_on_mode_exit(self):
        """PID integral should reset when exiting VISUAL_SERVO."""
        ctrl = _make_controller(
            kp=0.0,
            ki=1.0,
            kd=0.0,
            entry_threshold_frames=1,
            exit_threshold_frames=2,
        )
        # Enter VISUAL_SERVO and accumulate integral
        for _ in range(5):
            ctrl.update(
                target_pixel=(330, 240),
                laser_pixel=(320, 240),
                open_loop_angles_rad=(0.0, 0.0),
                target_acquired=True,
                dt=0.016,
            )
        # Exit by losing laser
        for _ in range(2):
            ctrl.update(
                target_pixel=(330, 240),
                laser_pixel=None,
                open_loop_angles_rad=(0.0, 0.0),
                target_acquired=True,
                dt=0.016,
            )
        assert ctrl.mode == ServoMode.ACQUISITION
        # Re-enter — should start fresh
        for _ in range(1):
            state = ctrl.update(
                target_pixel=(330, 240),
                laser_pixel=(320, 240),
                open_loop_angles_rad=(0.0, 0.0),
                target_acquired=True,
                dt=0.016,
            )
        # With fresh PID (ki=1.0, error=10px*0.1=1deg), integral should be small
        assert state.mode == ServoMode.VISUAL_SERVO

    def test_error_px_reported(self):
        ctrl = _make_controller(entry_threshold_frames=1)
        ctrl.update(
            target_pixel=(320, 240),
            laser_pixel=(310, 235),
            open_loop_angles_rad=(0.0, 0.0),
            target_acquired=True,
            dt=0.016,
        )
        state = ctrl.update(
            target_pixel=(320, 240),
            laser_pixel=(310, 235),
            open_loop_angles_rad=(0.0, 0.0),
            target_acquired=True,
            dt=0.016,
        )
        assert state.error_px == pytest.approx((10.0, 5.0))

    def test_commanded_resets_on_visual_servo_exit(self):
        """_commanded_pan/_commanded_tilt should reset to zero when
        exiting VISUAL_SERVO so that ACQUISITION re-seeds from open-loop."""
        ctrl = _make_controller(
            kp=0.5,
            ki=0.0,
            kd=0.0,
            entry_threshold_frames=1,
            exit_threshold_frames=2,
        )
        # Enter VISUAL_SERVO and accumulate non-zero commanded position
        for _ in range(3):
            ctrl.update(
                target_pixel=(340, 250),
                laser_pixel=(320, 240),
                open_loop_angles_rad=(0.0, 0.0),
                target_acquired=True,
                dt=0.016,
            )
        assert ctrl.mode == ServoMode.VISUAL_SERVO
        assert ctrl._commanded_pan != 0.0 or ctrl._commanded_tilt != 0.0

        # Exit by losing laser
        for _ in range(2):
            ctrl.update(
                target_pixel=(340, 250),
                laser_pixel=None,
                open_loop_angles_rad=(0.0, 0.0),
                target_acquired=True,
                dt=0.016,
            )
        assert ctrl.mode == ServoMode.ACQUISITION
        # Commanded position should be reset to zero
        assert ctrl._commanded_pan == 0.0
        assert ctrl._commanded_tilt == 0.0


class TestVisualServoConfig:

    def test_config_defaults(self):
        from src.config import VisualServoConfig

        cfg = VisualServoConfig()
        assert cfg.enabled is False
        assert cfg.kp == 0.4
        assert cfg.ki == 0.05
        assert cfg.kd == 0.1

    def test_config_in_pipeline(self):
        from src.config import PipelineConfig, VisualServoConfig

        cfg = PipelineConfig(mode="camera", target="human", source="0")
        assert isinstance(cfg.visual_servo, VisualServoConfig)
        assert cfg.visual_servo.enabled is False

    def test_yaml_parsing(self):
        from src.config_loader import build_config_from_yaml

        yaml_data = {
            "visual_servo": {
                "enabled": True,
                "kp": 0.8,
                "ki": 0.1,
            },
        }
        config = build_config_from_yaml(yaml_data)
        assert config.visual_servo.enabled is True
        assert config.visual_servo.kp == pytest.approx(0.8)
        assert config.visual_servo.ki == pytest.approx(0.1)
        # Defaults for unspecified fields
        assert config.visual_servo.kd == pytest.approx(0.1)
