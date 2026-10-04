"""Closed-loop regression tests for the delay-compensated auto controller.

The real ``compute_auto_command`` runs against a deliberately small plant:

* the camera frame that reaches the controller shows the scene ``camera_delay_s``
  earlier (exposure + stream + driver buffering), stamped with its arrival time;
* the gimbal follows the commanded position through a first-order servo response.

There is no Kalman filter, pixel noise, firmware smoothing or output-side slew
limiter in this loop, so it checks the controller alone. It is the CI-sized
subset of the larger rig simulation used to tune ``config.yaml``; it cannot prove
the numbers on real hardware, but it fails when the shipped preset stops giving
the lag, stability and overshoot trade-off it was tuned for.

The reference plant needs about 0.15-0.2 s from a command to its effect showing
up in a frame, and the preset assumes ``latency_comp_delay_s`` = 0.30 s.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from src.config import GimbalConfig, ServoControlConfig
from src.config_loader import build_config_from_yaml, load_yaml_config
from src.output.auto_controller import (
    AutoControllerConfig,
    AutoControllerState,
    compute_auto_command,
)
from src.shared.types import TrackingMessage

_LIVE = build_config_from_yaml(load_yaml_config(Path(__file__).parents[1] / "config.yaml"))
_BASE_NS = 1_000_000_000_000

Vec = tuple[float, float]
Target = Callable[[float], Vec]


@dataclass(frozen=True)
class _Rig:
    camera_delay_s: float = 0.10
    servo_tau_s: float = 0.05
    fps: float = 60.0


_DEFAULT_RIG = _Rig()


@dataclass(frozen=True)
class _Run:
    t: list[float]
    gimbal: list[Vec]
    target: list[Vec]

    def error(self, axis: int) -> list[float]:
        return [
            target[axis] - gimbal[axis]
            for target, gimbal in zip(self.target, self.gimbal, strict=True)
        ]

    def mean_error(self, axis: int, t_from: float) -> float:
        errors = [e for t, e in zip(self.t, self.error(axis), strict=True) if t >= t_from]
        return sum(errors) / len(errors)

    def peak_to_peak(self, axis: int, t_from: float) -> float:
        positions = [g[axis] for t, g in zip(self.t, self.gimbal, strict=True) if t >= t_from]
        return max(positions) - min(positions)

    def overshoot(self, axis: int, final: float) -> float:
        """Peak excursion beyond ``final`` for a target that ends up at a larger position."""
        return max(0.0, max(g[axis] for g in self.gimbal) - final)

    def settle_time(self, axis: int, *, band_deg: float = 1.0, t_from: float = 0.0) -> float:
        """Seconds after ``t_from`` from which |error| stays within ``band_deg`` for good."""
        last_outside = None
        for t, e in zip(self.t, self.error(axis), strict=True):
            if t >= t_from and abs(e) > band_deg:
                last_outside = t
        if last_outside is None:
            return 0.0
        if last_outside >= self.t[-1]:
            return math.inf
        return last_outside + (self.t[1] - self.t[0]) - t_from


def _preset(
    *, invert_pan: bool = False, invert_tilt: bool = False, **servo: float
) -> AutoControllerConfig:
    """The shipped ``config.yaml`` controller, optionally with other axis signs or delay."""
    gimbal: GimbalConfig = replace(_LIVE.gimbal, invert_pan=invert_pan, invert_tilt=invert_tilt)
    servo_control: ServoControlConfig = replace(_LIVE.servo_control, **servo)
    return AutoControllerConfig.from_configs(gimbal, servo_control, _LIVE.search)


def _legacy() -> AutoControllerConfig:
    """The controller as it was before delay compensation (same file, old gains)."""
    gimbal = replace(_LIVE.gimbal, kp_far=2.6, kp_near=1.0, kd=0.2, tilt_scale=0.35)
    servo_control = replace(_LIVE.servo_control, latency_comp_delay_s=0.0)
    return AutoControllerConfig.from_configs(gimbal, servo_control, _LIVE.search)


def _lookup(series: list[Vec], dt: float, t: float) -> Vec:
    """Linear interpolation of a series sampled every ``dt`` seconds from t = 0."""
    x = t / dt
    if x <= 0.0:
        return series[0]
    i = int(x)
    if i >= len(series) - 1:
        return series[-1]
    w = x - i
    return (
        series[i][0] * (1.0 - w) + series[i + 1][0] * w,
        series[i][1] * (1.0 - w) + series[i + 1][1] * w,
    )


def _simulate(
    config: AutoControllerConfig,
    target: Target,
    seconds: float,
    rig: _Rig = _DEFAULT_RIG,
    start: Vec = (0.0, 0.0),
) -> _Run:
    dt = 1.0 / rig.fps
    keep = math.exp(-dt / rig.servo_tau_s)
    state = AutoControllerState()
    state.pi_integral_pan, state.pi_integral_tilt = start
    gimbal: list[Vec] = [start]  # gimbal[k] is the position at t = k * dt
    run = _Run(t=[], gimbal=[], target=[])
    for k in range(1, int(round(seconds * rig.fps)) + 1):
        t = k * dt
        scene_target = target(t - rig.camera_delay_s)
        scene_gimbal = _lookup(gimbal, dt, t - rig.camera_delay_s)
        # Image angles carry the axis inversion; the controller undoes it.
        angles = (
            math.radians(config.pan_sign * (scene_target[0] - scene_gimbal[0])),
            math.radians(config.tilt_sign * (scene_target[1] - scene_gimbal[1])),
        )
        message = TrackingMessage(
            frame_id=k,
            timestamp_ns=_BASE_NS + int(round(t * 1e9)),
            target_kind="human",
            target_acquired=True,
            state_source="measurement",
            track_id=1,
            confidence=0.9,
            raw_pixel=None,
            filtered_pixel=None,
            raw_angles=None,
            filtered_angles=angles,
            servo_angles=angles,
            servo_angular_velocity=(0.0, 0.0),
            inference_ms=5.0,
            tracking_ms=1.0,
            total_latency_ms=8.0,
            fps=rig.fps,
        )
        pan, tilt = compute_auto_command(message=message, state=state, config=config)
        previous = gimbal[-1]
        gimbal.append((pan + (previous[0] - pan) * keep, tilt + (previous[1] - tilt) * keep))
        run.t.append(t)
        run.gimbal.append(gimbal[-1])
        run.target.append(target(t))
    return run


def _step(size: Vec) -> Target:
    return lambda t: size


def _ramp(speed: Vec, *, t0: float, start: Vec) -> Target:
    return lambda t: (
        start[0] + speed[0] * max(0.0, t - t0),
        start[1] + speed[1] * max(0.0, t - t0),
    )


def _move_then_stop(speed: Vec, *, t0: float, moving_s: float, start: Vec) -> Target:
    def target(t: float) -> Vec:
        moved_s = max(0.0, min(t - t0, moving_s))
        return start[0] + speed[0] * moved_s, start[1] + speed[1] * moved_s

    return target


_RIGS = [
    pytest.param(_Rig(camera_delay_s=0.04), id="cam40ms"),
    pytest.param(_Rig(camera_delay_s=0.10), id="cam100ms"),
    pytest.param(_Rig(camera_delay_s=0.20), id="cam200ms"),
    pytest.param(_Rig(camera_delay_s=0.10, fps=30.0), id="cam100ms-30fps"),
    pytest.param(_Rig(camera_delay_s=0.15, servo_tau_s=0.10), id="cam150ms-slow-servo"),
]

_AXIS_SIGNS = [
    pytest.param(False, False, id="normal"),
    pytest.param(True, True, id="both-inverted"),
    pytest.param(False, True, id="tilt-inverted"),
]


_WALK_START: Vec = (-45.0, 0.0)
_WALK_TARGET = _ramp((30.0, 0.0), t0=0.5, start=_WALK_START)


@pytest.mark.integration
class TestWalkingTarget:
    """A target walking at 30 deg/s, constant for 3 s after a 0.5 s head start."""

    @staticmethod
    def _lag(config: AutoControllerConfig, rig: _Rig) -> float:
        run = _simulate(config, _WALK_TARGET, 3.5, rig, start=_WALK_START)
        return run.mean_error(0, t_from=2.5)

    @pytest.mark.parametrize("rig", _RIGS)
    def test_lag_is_a_fraction_of_the_old_controllers(self, rig: _Rig) -> None:
        legacy_lag = self._lag(_legacy(), rig)
        preset_lag = self._lag(_preset(), rig)

        # The old integrating loop trails by speed / kp = 30 / 2.6 = 11.5 deg.
        assert legacy_lag > 9.0
        assert 0.0 <= preset_lag < 3.0
        assert preset_lag < legacy_lag / 3.0

    @pytest.mark.parametrize(("invert_pan", "invert_tilt"), _AXIS_SIGNS)
    def test_pan_and_tilt_both_keep_up_with_either_axis_inversion(
        self, invert_pan: bool, invert_tilt: bool
    ) -> None:
        config = _preset(invert_pan=invert_pan, invert_tilt=invert_tilt)
        target = _ramp((-30.0, 15.0), t0=0.5, start=(45.0, -20.0))

        run = _simulate(config, target, 3.5, start=(45.0, -20.0))

        # The target moves toward negative pan / positive tilt, so the lag is negative / positive.
        # Tilt runs at tilt_scale of the pan gain, so its lag is a little larger; a flipped or
        # missing feedforward would leave 7-11 deg here.
        assert -3.0 < run.mean_error(0, t_from=2.5) <= 0.0
        assert 0.0 <= run.mean_error(1, t_from=2.5) < 5.0


@pytest.mark.integration
class TestStep:
    """A sudden 10 deg pan / 6 deg tilt offset, held for 7 s."""

    @pytest.mark.parametrize("rig", _RIGS)
    def test_settles_without_overshoot_or_hunting(self, rig: _Rig) -> None:
        run = _simulate(_preset(), _step((10.0, 6.0)), 7.0, rig)

        for axis, size in ((0, 10.0), (1, 6.0)):
            assert run.settle_time(axis) < 2.5
            assert run.overshoot(axis, size) < 1.0
            assert run.peak_to_peak(axis, t_from=5.0) < 0.5

    @pytest.mark.parametrize(("invert_pan", "invert_tilt"), _AXIS_SIGNS)
    def test_settles_with_either_axis_inversion(self, invert_pan: bool, invert_tilt: bool) -> None:
        config = _preset(invert_pan=invert_pan, invert_tilt=invert_tilt)

        run = _simulate(config, _step((-10.0, 6.0)), 7.0)

        assert run.settle_time(0) < 2.5
        assert run.settle_time(1) < 2.5
        assert run.peak_to_peak(0, t_from=5.0) < 0.5
        assert run.peak_to_peak(1, t_from=5.0) < 0.5


@pytest.mark.integration
class TestAssumedDelayTolerance:
    """``latency_comp_delay_s`` only has to be roughly right.

    Under-estimating is the dangerous side (the feedforward then reads command
    acceleration as target motion), over-estimating makes stops ring, so the
    preset sits above the loop delay and is checked on both sides of it.
    """

    @pytest.mark.parametrize("assumed_s", [0.15, 0.20, 0.30, 0.40])
    def test_stays_stable_when_the_assumed_delay_is_off(self, assumed_s: float) -> None:
        config = _preset(latency_comp_delay_s=assumed_s)

        walking = _simulate(
            config,
            _ramp((30.0, 0.0), t0=0.5, start=(-45.0, 0.0)),
            3.5,
            start=(-45.0, 0.0),
        )
        step = _simulate(config, _step((10.0, 6.0)), 7.0)
        stop = _simulate(
            config,
            _move_then_stop((30.0, 0.0), t0=0.5, moving_s=1.5, start=(-20.0, 0.0)),
            6.0,
            start=(-20.0, 0.0),
        )

        assert 0.0 <= walking.mean_error(0, t_from=2.5) < 3.5
        assert step.settle_time(0) < 3.0
        assert step.peak_to_peak(0, t_from=5.0) < 0.5
        assert step.peak_to_peak(1, t_from=5.0) < 0.5
        assert stop.peak_to_peak(0, t_from=4.5) < 2.0


@pytest.mark.integration
class TestSuddenStop:
    """What zero lag costs: the gimbal cannot know the target stopped until it sees it."""

    @pytest.mark.parametrize(
        "rig",
        [
            pytest.param(_Rig(camera_delay_s=0.10), id="cam100ms"),
            pytest.param(_Rig(camera_delay_s=0.20), id="cam200ms"),
        ],
    )
    def test_overshoot_is_bounded_by_the_loop_delay_and_dies_out(self, rig: _Rig) -> None:
        speed = 30.0
        start = (-20.0, 0.0)
        end = start[0] + speed * 1.5
        target = _move_then_stop((speed, 0.0), t0=0.5, moving_s=1.5, start=start)

        run = _simulate(_preset(), target, 6.0, rig, start=start)

        assert run.overshoot(0, end) < 0.35 * speed  # about 0.25 s of target motion
        assert run.settle_time(0, t_from=2.0) < 3.0
        assert run.peak_to_peak(0, t_from=4.5) < 1.5

    def test_the_old_controller_does_not_overshoot_but_trails(self) -> None:
        start = (-20.0, 0.0)
        target = _move_then_stop((30.0, 0.0), t0=0.5, moving_s=1.5, start=start)

        run = _simulate(_legacy(), target, 6.0, start=start)

        assert run.overshoot(0, start[0] + 30.0 * 1.5) < 0.5
