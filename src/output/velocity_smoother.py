"""Per-axis angular-velocity smoother used to gate predictive lead safely.

Background
----------
Predictive lead in the auto controller (see :func:`_compute_lead_error_deg`
in :mod:`src.output.auto_controller`) multiplies the target's angular
velocity by a small lookahead time. The win is real on fast targets, but
the velocity signal we get from the Kalman filter has high-frequency
noise that, when multiplied by the lead time, becomes high-frequency
*command* noise to the gimbal. Live tuning (Phase Q in project memory)
disabled lead for that reason.

This module is the missing link: it provides a stateful per-axis smoother
that:

1. Applies an exponential moving average (low-pass) to attenuate
   measurement noise.
2. Snaps the output to zero inside a configurable deadband so static
   targets don't crawl from residual noise.
3. Optionally caps the slew rate of the smoothed velocity so a single
   noisy outlier can't swing the lead term across multiple frames.

The smoother is **safe by default**: with :attr:`VelocitySmootherConfig.enabled`
set to ``False`` it is a pure pass-through. The auto controller wires it
in unconditionally, but only routes velocity through it when the config
is enabled. That keeps current-deployment behavior bit-identical to the
pre-smoother build.

Activation policy
-----------------
1. Tune ``servo_control.velocity_smoother.*`` in ``config.yaml``.
2. Set ``servo_control.velocity_smoother.enabled: true``.
3. Set ``gimbal.predictive_lead_s`` to a non-zero value (it is 0.0 by
   default — the controller skips lead entirely when it is zero, so the
   smoother never gets exercised in the lead path either).

Step 3 is the actual gate for live behavior change. Steps 1–2 just
prepare the smoother to be useful when lead is re-enabled.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VelocitySmootherConfig:
    """Tuning for :class:`VelocitySmoother`.

    Defaults are conservative — ``enabled=False`` means the smoother is a
    pure pass-through. Once enabled, ``alpha=0.4`` gives a moderate
    low-pass and ``deadband_dps=5.0`` snaps to zero for slow-moving or
    static targets.
    """

    enabled: bool = False
    alpha: float = 0.4
    deadband_dps: float = 5.0
    slew_dps_per_s: float = 0.0  # 0 = no slew cap

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {self.alpha}")
        if self.deadband_dps < 0.0:
            raise ValueError(f"deadband_dps must be >= 0; got {self.deadband_dps}")
        if self.slew_dps_per_s < 0.0:
            raise ValueError(f"slew_dps_per_s must be >= 0; got {self.slew_dps_per_s}")


class VelocitySmoother:
    """Stateful per-axis EMA + deadband + optional slew cap.

    One instance per axis. ``smooth(v, dt, config)`` returns the
    filtered velocity in the same units as the input (typically
    degrees/second). ``reset()`` clears the internal state — call this
    on lock changes or when the upstream tracker reports a center
    reset, to avoid carrying stale velocity across track identities.

    The config is passed at call time (rather than stored on the
    instance) so the controller config can change between ticks without
    the state object needing to know — useful when the same
    ``AutoControllerState`` is reused across runs with different tuning.
    """

    def __init__(self) -> None:
        self._prev: float = 0.0
        self._initialized: bool = False

    @property
    def value(self) -> float:
        """Last smoothed value emitted (0.0 before any input)."""
        return self._prev

    def reset(self) -> None:
        self._prev = 0.0
        self._initialized = False

    def smooth(
        self,
        velocity_dps: float,
        dt_s: float,
        config: VelocitySmootherConfig,
    ) -> float:
        """Run one tick of the filter.

        With ``config.enabled=False`` this is a pure pass-through so the
        call site can be unconditional without changing behavior.
        ``dt_s`` is only consulted by the slew cap; it is ignored when
        ``slew_dps_per_s`` is 0.
        """
        if not config.enabled:
            self._prev = velocity_dps
            self._initialized = True
            return velocity_dps

        if not self._initialized:
            # Seed with the first sample so we don't start from 0 and
            # bleed in over the first ~1/(1-alpha) ticks. This matches
            # the conventional "warm-start the EMA on first sample" rule.
            ema = velocity_dps
            self._initialized = True
        else:
            ema = config.alpha * velocity_dps + (1.0 - config.alpha) * self._prev

        # Optional slew cap: limits how fast the smoothed velocity can
        # change between ticks. Useful as a hard guarantee against single
        # outlier samples (still respected even if alpha=1.0).
        if config.slew_dps_per_s > 0.0 and dt_s > 0.0:
            max_step = config.slew_dps_per_s * dt_s
            delta = ema - self._prev
            if delta > max_step:
                ema = self._prev + max_step
            elif delta < -max_step:
                ema = self._prev - max_step

        # Deadband: snap small magnitudes to zero so static targets don't
        # crawl on residual noise. Applied AFTER the slew cap so that a
        # large genuine deceleration into the deadband still reads zero.
        if abs(ema) < config.deadband_dps:
            ema = 0.0

        self._prev = ema
        return ema
