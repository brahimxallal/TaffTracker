"""Sweep VelocitySmoother parameters against synthetic noisy velocity.

Use this to pick a starting ``alpha`` and ``deadband_dps`` for
``servo_control.velocity_smoother_*`` in ``config.yaml`` BEFORE flipping
``velocity_smoother_enabled: true`` on the live system. The script never
touches the live pipeline — it just simulates a typical Kalman velocity
trace (slow drift + HF measurement noise + occasional spikes) and reports
how each candidate setting attenuates the noise vs. lags the signal.

Usage::

    python scripts/analyze_velocity_smoother.py
    python scripts/analyze_velocity_smoother.py --alpha-grid 0.2,0.3,0.4,0.5
    python scripts/analyze_velocity_smoother.py --noise-std 25 --spike-amp 200

The recommended alpha is the one that gives the largest noise reduction
without the lag exceeding a single inference frame (~16 ms @ 60 fps).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.output.velocity_smoother import (  # noqa: E402
    VelocitySmoother,
    VelocitySmootherConfig,
)


def _generate_velocity_trace(
    *,
    n_samples: int,
    fps: float,
    drift_amplitude_dps: float,
    drift_period_s: float,
    noise_std_dps: float,
    spike_amp_dps: float,
    spike_every_n: int,
    seed: int,
) -> np.ndarray:
    """Build a velocity-vs-time trace that resembles real Kalman output.

    Components:
      - Slow sinusoidal drift (target moving across frame)
      - Gaussian high-frequency measurement noise
      - Periodic outlier spikes (single-sample noise bursts)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fps
    drift = drift_amplitude_dps * np.sin(2 * np.pi * t / drift_period_s)
    noise = rng.normal(0.0, noise_std_dps, size=n_samples)
    trace = drift + noise
    if spike_every_n > 0:
        for i in range(spike_every_n - 1, n_samples, spike_every_n):
            sign = 1.0 if (i // spike_every_n) % 2 == 0 else -1.0
            trace[i] += sign * spike_amp_dps
    return trace, drift


def _apply_smoother(trace: np.ndarray, dt_s: float, config: VelocitySmootherConfig) -> np.ndarray:
    smoother = VelocitySmoother()
    out = np.empty_like(trace)
    for i, v in enumerate(trace):
        out[i] = smoother.smooth(float(v), dt_s, config)
    return out


def _measure_lag_samples(reference: np.ndarray, smoothed: np.ndarray) -> int:
    """Estimate lag (in samples) between the smoothed signal and a clean reference.

    Uses cross-correlation; positive return means smoothed lags reference.
    """
    ref = reference - reference.mean()
    sm = smoothed - smoothed.mean()
    corr = np.correlate(sm, ref, mode="full")
    return int(corr.argmax() - (len(ref) - 1))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--alpha-grid", default="0.2,0.3,0.4,0.5,0.7,1.0")
    parser.add_argument("--deadband-grid", default="0.0,5.0,10.0")
    parser.add_argument("--n-samples", type=int, default=600)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--drift-amp", type=float, default=80.0, help="dps; slow target speed")
    parser.add_argument("--drift-period", type=float, default=2.0, help="seconds")
    parser.add_argument("--noise-std", type=float, default=15.0, help="dps; HF measurement noise")
    parser.add_argument("--spike-amp", type=float, default=120.0, help="dps; outlier amplitude")
    parser.add_argument("--spike-every", type=int, default=30, help="0 to disable spikes")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    alphas = [float(x) for x in args.alpha_grid.split(",")]
    deadbands = [float(x) for x in args.deadband_grid.split(",")]
    dt_s = 1.0 / args.fps

    trace, drift = _generate_velocity_trace(
        n_samples=args.n_samples,
        fps=args.fps,
        drift_amplitude_dps=args.drift_amp,
        drift_period_s=args.drift_period,
        noise_std_dps=args.noise_std,
        spike_amp_dps=args.spike_amp,
        spike_every_n=args.spike_every,
        seed=args.seed,
    )

    raw_residual_std = float(np.std(trace - drift))
    print(
        f"Synthetic velocity trace: {args.n_samples} samples @ {args.fps:.0f} fps, "
        f"drift {args.drift_amp:.0f} dps p-p over {args.drift_period:.1f} s, "
        f"noise std {args.noise_std:.1f} dps"
    )
    print(f"Raw signal noise std (vs clean drift): {raw_residual_std:.2f} dps")
    print()
    print(
        f"{'alpha':>7} {'deadband':>9} "
        f"{'noise_std':>10} {'reduction':>10} {'lag_ms':>8} {'verdict':>10}"
    )
    print("-" * 60)

    best: tuple[float, float, float] | None = None  # (reduction, alpha, deadband)
    for deadband in deadbands:
        for alpha in alphas:
            cfg = VelocitySmootherConfig(enabled=True, alpha=alpha, deadband_dps=deadband)
            smoothed = _apply_smoother(trace, dt_s, cfg)
            residual = smoothed - drift
            std = float(np.std(residual))
            reduction = 1.0 - std / raw_residual_std if raw_residual_std > 0 else 0.0
            lag_samples = _measure_lag_samples(drift, smoothed)
            lag_ms = lag_samples * dt_s * 1000.0
            verdict = "BAD lag" if abs(lag_ms) > 1000.0 / args.fps else "ok"
            print(
                f"{alpha:>7.2f} {deadband:>9.1f} "
                f"{std:>10.2f} {reduction*100:>9.1f}% {lag_ms:>8.1f} {verdict:>10}"
            )
            if verdict == "ok" and (best is None or reduction > best[0]):
                best = (reduction, alpha, deadband)

    if best is not None:
        print()
        print("Recommended starting point:")
        print(f"  velocity_smoother_alpha:        {best[1]}")
        print(f"  velocity_smoother_deadband_dps: {best[2]}")
        print(
            f"  -> noise reduction {best[0] * 100:.0f}% with sub-frame lag. "
            f"Set in config.yaml under servo_control:, then enable on the live system."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
