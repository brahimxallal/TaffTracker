"""Unit tests for the delay-compensation helpers (``src/output/latency_compensation.py``)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.output.latency_compensation import (
    AxisVelocityEstimator,
    CommandHistory,
    CompensatedError,
    LatencyCompensationConfig,
    LatencyCompensator,
    compensate_error,
)

_FPS = 60.0
_CFG = LatencyCompensationConfig(delay_s=0.3)


def _history(
    pan: Callable[[float], float],
    tilt: Callable[[float], float] = lambda t: 0.0,
    *,
    seconds: float = 1.0,
) -> LatencyCompensator:
    """A compensator whose command history follows the given pan/tilt curves."""
    compensator = LatencyCompensator()
    for k in range(int(round(seconds * _FPS)) + 1):
        t = k / _FPS
        compensator.record_command(t, pan(t), tilt(t))
    return compensator


def _prime(
    compensator: LatencyCompensator,
    config: LatencyCompensationConfig,
    *,
    pan_dps: float = 0.0,
    tilt_dps: float = 0.0,
    seconds: float = 3.0,
    pipeline_age_s: float = 0.008,
) -> CompensatedError:
    """Move a target at constant speed past a gimbal that stays at 0 deg.

    The measured error is then the target's absolute angle, so the velocity
    estimator sees a clean ramp. Returns the compensation for the last frame.
    """
    out = CompensatedError(0.0, 0.0, 0.0, 0.0)
    for k in range(int(round(seconds * _FPS)) + 1):
        t = k / _FPS
        compensator.record_command(t, 0.0, 0.0)
        out = compensate_error(
            config=config,
            compensator=compensator,
            t_s=t,
            pan_err_deg=pan_dps * t,
            tilt_err_deg=tilt_dps * t,
            pipeline_age_s=pipeline_age_s,
            fresh_measurement=True,
        )
    return out


@pytest.mark.unit
class TestLatencyCompensationConfig:
    def test_default_is_disabled(self) -> None:
        assert LatencyCompensationConfig().enabled is False

    @pytest.mark.parametrize(
        ("kwargs", "enabled"),
        [
            ({"delay_s": 0.3}, True),
            ({"delay_s": 0.3, "smith_gain": 0.0}, True),
            ({"delay_s": 0.3, "velocity_ff_gain": 0.0}, True),
            ({"delay_s": 0.3, "smith_gain": 0.0, "velocity_ff_gain": 0.0}, False),
            ({"delay_s": 0.0}, False),
        ],
    )
    def test_enabled_needs_a_delay_and_at_least_one_gain(
        self, kwargs: dict[str, float], enabled: bool
    ) -> None:
        assert LatencyCompensationConfig(**kwargs).enabled is enabled

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"delay_s": -0.1},
            {"delay_s": 1.1},
            {"smith_gain": -0.1},
            {"smith_gain": 1.6},
            {"velocity_ff_gain": -0.1},
            {"velocity_ff_gain": 1.6},
            {"velocity_tau_s": 0.0},
            {"velocity_tau_s": 2.5},
            {"velocity_deadband_dps": -1.0},
            {"velocity_max_dps": 0.0},
        ],
    )
    def test_rejects_out_of_range_values(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            LatencyCompensationConfig(**kwargs)


@pytest.mark.unit
class TestCommandHistory:
    def test_empty_history_has_no_commands(self) -> None:
        history = CommandHistory()

        assert len(history) == 0
        assert history.latest is None
        assert history.at(1.0) is None

    def test_latest_is_the_most_recent_command(self) -> None:
        history = CommandHistory()
        history.append(0.0, 1.0, -1.0)
        history.append(0.1, 2.0, -2.0)

        assert history.latest == (2.0, -2.0)
        assert len(history) == 2

    def test_at_interpolates_linearly_between_samples(self) -> None:
        history = CommandHistory()
        history.append(0.0, 0.0, 0.0)
        history.append(1.0, 10.0, -20.0)
        history.append(2.0, 10.0, -20.0)

        first = history.at(0.25)
        second = history.at(1.5)

        assert first == (pytest.approx(2.5), pytest.approx(-5.0))
        assert second == (pytest.approx(10.0), pytest.approx(-20.0))

    def test_at_returns_the_stored_value_on_a_sample_time(self) -> None:
        history = CommandHistory()
        history.append(0.0, 1.0, 2.0)
        history.append(0.5, 3.0, 4.0)
        history.append(1.0, 5.0, 6.0)

        assert history.at(0.5) == (3.0, 4.0)

    def test_at_clamps_outside_the_recorded_range(self) -> None:
        history = CommandHistory()
        history.append(1.0, 5.0, 6.0)
        history.append(2.0, 7.0, 8.0)

        assert history.at(0.0) == (5.0, 6.0)
        assert history.at(3.0) == (7.0, 8.0)

    def test_a_non_increasing_timestamp_restarts_the_history(self) -> None:
        history = CommandHistory()
        history.append(1.0, 1.0, 1.0)
        history.append(2.0, 2.0, 2.0)

        history.append(1.5, 9.0, 9.0)  # the clock went backwards

        assert len(history) == 1
        assert history.latest == (9.0, 9.0)

        history.append(1.5, 5.0, 5.0)  # a repeated timestamp cannot be interpolated either

        assert len(history) == 1
        assert history.latest == (5.0, 5.0)

    def test_history_is_bounded(self) -> None:
        history = CommandHistory(maxlen=4)
        for i in range(10):
            history.append(float(i), float(i), 0.0)

        assert len(history) == 4
        assert history.at(0.0) == (6.0, 0.0)  # oldest retained sample
        assert history.latest == (9.0, 0.0)

    def test_clear_empties_the_history(self) -> None:
        history = CommandHistory()
        history.append(0.0, 1.0, 1.0)

        history.clear()

        assert len(history) == 0
        assert history.latest is None


@pytest.mark.unit
class TestAxisVelocityEstimator:
    @staticmethod
    def _follow_ramp(
        estimator: AxisVelocityEstimator,
        config: LatencyCompensationConfig,
        velocity_dps: float,
        *,
        fps: float = _FPS,
        seconds: float = 3.0,
    ) -> float:
        velocity = 0.0
        for k in range(int(round(seconds * fps)) + 1):
            t = k / fps
            velocity = estimator.update(velocity_dps * t, t, config)
        return velocity

    def test_first_sample_has_no_velocity(self) -> None:
        estimator = AxisVelocityEstimator()

        assert estimator.update(5.0, 0.0, _CFG) == 0.0
        assert estimator.velocity_dps == 0.0

    @pytest.mark.parametrize("velocity_dps", [20.0, -35.0])
    def test_converges_to_the_ramp_velocity(self, velocity_dps: float) -> None:
        estimator = AxisVelocityEstimator()

        estimate = self._follow_ramp(estimator, _CFG, velocity_dps)

        assert estimate == pytest.approx(velocity_dps, abs=0.3)
        assert estimator.velocity_dps == estimate

    def test_time_constant_does_not_depend_on_the_frame_rate(self) -> None:
        estimates = []
        for fps in (30.0, 60.0, 120.0):
            estimator = AxisVelocityEstimator()
            estimates.append(self._follow_ramp(estimator, _CFG, 20.0, fps=fps, seconds=0.6))

        assert max(estimates) - min(estimates) < 1.0
        assert all(5.0 < estimate < 20.0 for estimate in estimates)  # still converging

    def test_restarts_after_a_gap_in_the_measurements(self) -> None:
        estimator = AxisVelocityEstimator()
        self._follow_ramp(estimator, _CFG, 20.0)
        assert estimator.velocity_dps > 10.0

        assert estimator.update(500.0, 3.0 + 0.3, _CFG) == 0.0  # longer than 0.25 s
        assert estimator.velocity_dps == 0.0

    def test_restarts_on_a_repeated_timestamp(self) -> None:
        estimator = AxisVelocityEstimator()
        estimator.update(0.0, 1.0, _CFG)
        estimator.update(0.5, 1.0 + 1.0 / _FPS, _CFG)

        assert estimator.update(0.6, 1.0 + 1.0 / _FPS, _CFG) == 0.0

    def test_restarts_on_a_jump_beyond_the_reset_threshold(self) -> None:
        estimator = AxisVelocityEstimator()
        self._follow_ramp(estimator, _CFG, 20.0)
        assert estimator.velocity_dps > 10.0
        t_next = 3.0 + 1.0 / _FPS

        jump = _CFG.reset_jump_deg + 10.0
        assert estimator.update(20.0 * t_next + jump, t_next, _CFG) == 0.0

    def test_a_single_sample_moves_the_estimate_by_at_most_the_innovation_clamp(self) -> None:
        clamp = _CFG.innovation_clamp_deg
        assert clamp < 10.0 < _CFG.reset_jump_deg
        big = AxisVelocityEstimator()
        at_clamp = AxisVelocityEstimator()
        for estimator, jump in ((big, 10.0), (at_clamp, clamp)):
            estimator.update(0.0, 0.0, _CFG)
            estimator.update(jump, 1.0 / _FPS, _CFG)

        assert big.velocity_dps > 0.0
        assert big.velocity_dps == pytest.approx(at_clamp.velocity_dps)

    def test_velocity_is_limited(self) -> None:
        # The ramp is faster than the limit but slow enough for the position estimate
        # to stay within reset_jump_deg of the target, so the estimator keeps running.
        config = LatencyCompensationConfig(delay_s=0.3, velocity_max_dps=5.0)

        assert self._follow_ramp(AxisVelocityEstimator(), config, 10.0) == pytest.approx(5.0)
        assert self._follow_ramp(AxisVelocityEstimator(), config, -10.0) == pytest.approx(-5.0)

    def test_reset_forgets_the_estimate(self) -> None:
        estimator = AxisVelocityEstimator()
        self._follow_ramp(estimator, _CFG, 20.0)

        estimator.reset()

        assert estimator.velocity_dps == 0.0
        assert estimator.update(10.0, 5.0, _CFG) == 0.0  # first sample again


@pytest.mark.unit
class TestCompensateError:
    @staticmethod
    def _call(
        compensator: LatencyCompensator,
        config: LatencyCompensationConfig = _CFG,
        *,
        t_s: float = 1.0,
        pan_err_deg: float = 0.0,
        tilt_err_deg: float = 0.0,
        pipeline_age_s: float = 0.0,
        fresh_measurement: bool = True,
    ) -> CompensatedError:
        return compensate_error(
            config=config,
            compensator=compensator,
            t_s=t_s,
            pan_err_deg=pan_err_deg,
            tilt_err_deg=tilt_err_deg,
            pipeline_age_s=pipeline_age_s,
            fresh_measurement=fresh_measurement,
        )

    def test_without_history_the_measured_error_passes_through(self) -> None:
        out = self._call(LatencyCompensator(), pan_err_deg=4.0, tilt_err_deg=-2.0)

        assert out == CompensatedError(4.0, -2.0, 0.0, 0.0)

    def test_a_single_recorded_command_has_nothing_in_flight(self) -> None:
        compensator = LatencyCompensator()
        compensator.record_command(1.0, 3.0, 4.0)

        out = self._call(compensator, pan_err_deg=2.0, tilt_err_deg=-1.0, pipeline_age_s=0.01)

        assert out == CompensatedError(2.0, -1.0, 0.0, 0.0)

    def test_subtracts_the_motion_that_is_still_in_flight(self) -> None:
        config = LatencyCompensationConfig(delay_s=0.3, velocity_ff_gain=0.0)
        compensator = _history(lambda t: 2.0 * t, lambda t: -t)

        out = self._call(compensator, config, pan_err_deg=5.0, tilt_err_deg=1.0)

        # In the last 0.3 s the commands moved pan by +0.6 deg and tilt by -0.3 deg;
        # that motion has not reached the image yet.
        assert out.pan_err_deg == pytest.approx(5.0 - 0.6)
        assert out.tilt_err_deg == pytest.approx(1.0 + 0.3)
        assert (out.pan_ff_dps, out.tilt_ff_dps) == (0.0, 0.0)

    def test_smith_gain_scales_the_in_flight_term(self) -> None:
        config = LatencyCompensationConfig(delay_s=0.3, smith_gain=0.5, velocity_ff_gain=0.0)
        compensator = _history(lambda t: 2.0 * t)

        out = self._call(compensator, config, pan_err_deg=5.0)

        assert out.pan_err_deg == pytest.approx(5.0 - 0.5 * 0.6)

    def test_with_both_gains_at_zero_the_error_passes_through(self) -> None:
        config = LatencyCompensationConfig(delay_s=0.3, smith_gain=0.0, velocity_ff_gain=0.0)
        compensator = _history(lambda t: 2.0 * t)

        out = self._call(compensator, config, pan_err_deg=5.0, tilt_err_deg=-3.0)

        assert out == CompensatedError(5.0, -3.0, 0.0, 0.0)

    def test_feedforward_leads_by_the_delay_plus_the_pipeline_age(self) -> None:
        compensator = LatencyCompensator()

        out = _prime(compensator, _CFG, pan_dps=20.0, pipeline_age_s=0.008)

        # The soft deadband removes 3 deg/s from the estimated 20 deg/s.
        assert out.pan_ff_dps == pytest.approx(20.0 - _CFG.velocity_deadband_dps, abs=0.5)
        assert out.pan_err_deg == pytest.approx(20.0 * 3.0 + out.pan_ff_dps * (0.3 + 0.008))
        assert (out.tilt_err_deg, out.tilt_ff_dps) == (0.0, 0.0)

    def test_tilt_is_estimated_independently_of_pan(self) -> None:
        compensator = LatencyCompensator()

        out = _prime(compensator, _CFG, tilt_dps=-15.0)

        assert out.tilt_ff_dps == pytest.approx(-(15.0 - _CFG.velocity_deadband_dps), abs=0.5)
        assert out.pan_ff_dps == 0.0

    @pytest.mark.parametrize("velocity_dps", [10.0, -10.0])
    def test_the_feedforward_deadband_shrinks_the_estimate_toward_zero(
        self, velocity_dps: float
    ) -> None:
        compensator = LatencyCompensator()

        out = _prime(compensator, _CFG, pan_dps=velocity_dps)

        shrunk = compensator.pan.velocity_dps - _CFG.velocity_deadband_dps * (
            1.0 if velocity_dps > 0 else -1.0
        )
        assert out.pan_ff_dps == pytest.approx(_CFG.velocity_ff_gain * shrunk)

    def test_targets_slower_than_the_deadband_get_no_feedforward(self) -> None:
        compensator = LatencyCompensator()

        out = _prime(compensator, _CFG, pan_dps=2.0)

        assert out.pan_ff_dps == 0.0
        assert out.pan_err_deg == pytest.approx(2.0 * 3.0)  # nothing added to the error

    def test_feedforward_gain_scales_the_feedforward(self) -> None:
        full = _prime(LatencyCompensator(), _CFG, pan_dps=20.0)
        half_cfg = LatencyCompensationConfig(delay_s=0.3, velocity_ff_gain=0.5)

        half = _prime(LatencyCompensator(), half_cfg, pan_dps=20.0)

        assert half.pan_ff_dps == pytest.approx(0.5 * full.pan_ff_dps)

    def test_pipeline_age_is_clamped(self) -> None:
        compensator = LatencyCompensator()
        _prime(compensator, _CFG, pan_dps=20.0)
        t_next = 3.0 + 1.0 / _FPS

        def call(age_s: float) -> CompensatedError:
            return self._call(
                compensator, t_s=t_next, pipeline_age_s=age_s, fresh_measurement=False
            )

        assert call(5.0) == call(_CFG.max_pipeline_age_s)  # a stalled frame cannot inflate it
        assert call(-1.0) == call(0.0)
        assert call(_CFG.max_pipeline_age_s).pan_err_deg > call(0.0).pan_err_deg

    def test_prediction_frames_do_not_feed_the_velocity_estimator(self) -> None:
        compensator = LatencyCompensator()
        _prime(compensator, _CFG, pan_dps=20.0)
        before = compensator.pan.velocity_dps

        out = self._call(
            compensator, t_s=3.0 + 1.0 / _FPS, pan_err_deg=45.0, fresh_measurement=False
        )

        assert compensator.pan.velocity_dps == before
        assert out.pan_ff_dps == pytest.approx(
            _CFG.velocity_ff_gain * (before - _CFG.velocity_deadband_dps)
        )


@pytest.mark.unit
class TestLatencyCompensator:
    def test_reset_clears_history_and_estimators(self) -> None:
        compensator = LatencyCompensator()
        _prime(compensator, _CFG, pan_dps=20.0, tilt_dps=10.0)
        assert len(compensator.history) > 0
        assert compensator.pan.velocity_dps != 0.0

        compensator.reset()

        assert len(compensator.history) == 0
        assert compensator.pan.velocity_dps == 0.0
        assert compensator.tilt.velocity_dps == 0.0
        assert compensator.pan.update(10.0, 5.0, _CFG) == 0.0  # estimator starts over
