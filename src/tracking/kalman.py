from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from src.config import KalmanConfig

LOGGER = logging.getLogger("kalman")


@dataclass(frozen=True, slots=True)
class KalmanState:
    x: float
    y: float
    vx: float
    vy: float

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y

    @property
    def velocity(self) -> tuple[float, float]:
        return self.vx, self.vy


# Backward-compatible aliases for tests that import these constants.
MAX_CONSECUTIVE_PREDICTIONS: int = KalmanConfig().max_consecutive_predictions


class KalmanFilter:
    def __init__(
        self,
        process_noise: float = 3.0,
        measurement_noise: float = 8.0,
        config: KalmanConfig | None = None,
        fps_ratio: float = 1.0,
    ) -> None:
        self._config = config or KalmanConfig()
        self._fps_ratio = max(fps_ratio, 0.1)
        self._base_process_noise = float(process_noise)
        self._base_measurement_noise = float(measurement_noise)
        self._process_noise = float(process_noise)
        self._measurement_noise = float(measurement_noise)
        self._state = np.zeros((4, 1), dtype=np.float64)
        self._covariance = np.eye(4, dtype=np.float64) * 500.0
        self._measurement_matrix = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        self._measurement_covariance = np.eye(2, dtype=np.float64) * self._measurement_noise
        self._identity4 = np.eye(4, dtype=np.float64)
        self._initialized = False
        self._last_dt: float = -1.0
        self._cached_transition = np.eye(4, dtype=np.float64)
        self._process_cov_buffer = np.zeros((4, 4), dtype=np.float64)
        # Phase 5 additions
        self._consecutive_predictions: int = 0
        self._last_innovation_gated: bool = False
        self._oru_cache: deque[tuple[tuple[float, float], float]] = deque(
            maxlen=self._config.oru_cache_size
        )

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def consecutive_predictions(self) -> int:
        return self._consecutive_predictions

    @property
    def last_innovation_gated(self) -> bool:
        return self._last_innovation_gated

    def reset(self) -> None:
        self._state.fill(0.0)
        self._covariance = np.eye(4, dtype=np.float64) * 500.0
        self._initialized = False
        self._last_dt = -1.0
        self._consecutive_predictions = 0
        self._last_innovation_gated = False
        self._oru_cache.clear()

    def _reset_to_measurement(self, z: np.ndarray) -> None:
        """Hard-reset state to the given measurement, preserving initialization."""
        self._state = np.array(
            [[float(z[0, 0])], [float(z[1, 0])], [0.0], [0.0]], dtype=np.float64
        )
        self._covariance = np.eye(4, dtype=np.float64) * 500.0
        self._consecutive_predictions = 0
        self._last_innovation_gated = False

    def snapshot(self) -> dict:
        """Return an independent copy of the filter state for later restoration."""
        return {
            "state": self._state.copy(),
            "covariance": self._covariance.copy(),
            "initialized": self._initialized,
            "consecutive_predictions": self._consecutive_predictions,
        }

    def restore(self, snap: dict) -> None:
        """Restore filter state from a previous snapshot."""
        self._state = snap["state"].copy()
        self._covariance = snap["covariance"].copy()
        self._initialized = snap["initialized"]
        self._consecutive_predictions = snap["consecutive_predictions"]
        self._last_innovation_gated = False
        self._oru_cache.clear()

    def predict(self, dt: float) -> KalmanState | None:
        if not self._initialized:
            return None
        self._consecutive_predictions += 1
        if self._consecutive_predictions > self._config.max_consecutive_predictions:
            self.reset()
            return None
        transition = self._transition_matrix(dt)
        process_covariance = self._process_covariance(dt)
        self._state = transition @ self._state
        self._covariance = transition @ self._covariance @ transition.T + process_covariance
        # Decay velocity during extended occlusion to prevent overshoot on re-acquisition.
        decay_start = self._config.prediction_decay_start
        if self._consecutive_predictions > decay_start:
            decay = self._config.prediction_velocity_decay ** (
                self._consecutive_predictions - decay_start
            )
            self._state[2, 0] *= decay
            self._state[3, 0] *= decay
        return self.current_state()

    def update(
        self,
        measurement: tuple[float, float],
        dt: float,
    ) -> KalmanState:
        measurement_vector = np.array([[measurement[0]], [measurement[1]]], dtype=np.float64)
        if not self._initialized:
            self._state = np.array(
                [[measurement[0]], [measurement[1]], [0.0], [0.0]], dtype=np.float64
            )
            self._covariance = np.eye(4, dtype=np.float64) * 500.0
            self._initialized = True
            self._consecutive_predictions = 0
            self._oru_cache.append((measurement, dt))
            return self.current_state()

        # Adaptive R/Q based on current speed estimate
        speed = np.sqrt(float(self._state[2, 0]) ** 2 + float(self._state[3, 0]) ** 2)
        self._adapt_noise(speed)

        transition = self._transition_matrix(dt)
        process_covariance = self._process_covariance(dt)
        self._state = transition @ self._state
        self._covariance = transition @ self._covariance @ transition.T + process_covariance

        innovation = measurement_vector - (self._measurement_matrix @ self._state)
        innovation_covariance = (
            self._measurement_matrix @ self._covariance @ self._measurement_matrix.T
            + self._measurement_covariance
        )

        # Innovation gating: reject wild measurements
        # Scale gate wider at lower FPS (larger inter-frame displacements)
        effective_gate = self._config.innovation_gate_sigma * max(1.0, 1.0 / (self._fps_ratio**0.5))
        mahal_sq = float((innovation.T @ np.linalg.solve(innovation_covariance, innovation)).item())
        if mahal_sq > effective_gate**2:
            self._last_innovation_gated = True
            self._consecutive_predictions += 1
            if self._consecutive_predictions > self._config.max_consecutive_predictions:
                self.reset()
            elif self._consecutive_predictions >= self._config.max_consecutive_gated:
                # Too many consecutive rejections — the filter is stuck.
                # Soft reset: inflate covariance to accept future measurements
                # but keep the predicted state to avoid injecting a wild measurement.
                self._covariance *= 10.0
                self._consecutive_predictions = 0
                self._last_innovation_gated = False
            return self.current_state()

        self._last_innovation_gated = False
        self._consecutive_predictions = 0

        # Inline 2x2 matrix inversion (faster than np.linalg.inv for fixed size)
        s00 = innovation_covariance[0, 0]
        s01 = innovation_covariance[0, 1]
        s10 = innovation_covariance[1, 0]
        s11 = innovation_covariance[1, 1]
        det = s00 * s11 - s01 * s10
        if abs(det) < 1e-12:
            self._last_innovation_gated = True
            return self.current_state()
        inv_det = 1.0 / det
        s_inv = np.array([[s11, -s01], [-s10, s00]], dtype=np.float64) * inv_det
        kalman_gain = self._covariance @ self._measurement_matrix.T @ s_inv
        self._state = self._state + (kalman_gain @ innovation)
        self._covariance = (
            self._identity4 - kalman_gain @ self._measurement_matrix
        ) @ self._covariance

        # NaN/inf guard: reset to measurement if state became non-finite
        if not np.all(np.isfinite(self._state)):
            LOGGER.warning("Kalman state non-finite after update, resetting to measurement")
            self._reset_to_measurement(measurement_vector)
            return self.current_state()

        # OC-SORT ORU: cache observation for re-update after occlusion
        self._oru_cache.append((measurement, dt))

        return self.current_state()

    def oru_re_update(self) -> KalmanState | None:
        """Re-apply cached observations (OC-SORT Observation-centric Re-Update).

        Call after re-acquiring a track that was lost. Replays cached
        measurements to smooth the re-acquisition transition.
        """
        if not self._oru_cache or not self._initialized:
            return self.current_state()
        cached = list(self._oru_cache)
        self._oru_cache.clear()
        for measurement, dt in cached:
            self.update(measurement, dt)
        return self.current_state()

    def current_state(self) -> KalmanState | None:
        if not self._initialized:
            return None
        return KalmanState(
            x=float(self._state[0, 0]),
            y=float(self._state[1, 0]),
            vx=float(self._state[2, 0]),
            vy=float(self._state[3, 0]),
        )

    def _adapt_noise(self, speed: float) -> None:
        """Adjust R and Q based on target speed."""
        cfg = self._config
        # Adaptive R: low speed → high R (trust model, suppress jitter)
        #             high speed → low R (trust measurements, model lags)
        speed_frac = min(speed / cfg.adaptive_r_speed_thresh, 1.0)
        adaptive_r = cfg.adaptive_r_max - speed_frac * (cfg.adaptive_r_max - cfg.adaptive_r_min)
        self._measurement_noise = max(adaptive_r, self._base_measurement_noise)
        self._measurement_covariance[0, 0] = self._measurement_noise
        self._measurement_covariance[1, 1] = self._measurement_noise
        self._measurement_covariance[0, 1] = 0.0
        self._measurement_covariance[1, 0] = 0.0

        # Adaptive Q: smooth ramp from 1× to max multiplier between thresholds
        if speed <= cfg.adaptive_q_speed_thresh:
            self._process_noise = self._base_process_noise
        elif speed >= cfg.adaptive_q_speed_ceil:
            self._process_noise = self._base_process_noise * cfg.adaptive_q_max_multiplier
        else:
            frac = (speed - cfg.adaptive_q_speed_thresh) / (
                cfg.adaptive_q_speed_ceil - cfg.adaptive_q_speed_thresh
            )
            self._process_noise = self._base_process_noise * (
                1.0 + frac * (cfg.adaptive_q_max_multiplier - 1.0)
            )

    def _transition_matrix(self, dt: float) -> np.ndarray:
        delta_t = max(dt, 1e-3)
        if math.isclose(delta_t, self._last_dt, rel_tol=1e-9):
            return self._cached_transition
        self._last_dt = delta_t
        self._cached_transition = np.array(
            [
                [1.0, 0.0, delta_t, 0.0],
                [0.0, 1.0, 0.0, delta_t],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return self._cached_transition

    def _process_covariance(self, dt: float) -> np.ndarray:
        delta_t = max(dt, 1e-3)
        dt2 = delta_t * delta_t
        dt3 = dt2 * delta_t
        dt4 = dt3 * delta_t
        buf = self._process_cov_buffer
        buf[0, 0] = dt4 / 4.0
        buf[0, 2] = dt3 / 2.0
        buf[1, 1] = dt4 / 4.0
        buf[1, 3] = dt3 / 2.0
        buf[2, 0] = dt3 / 2.0
        buf[2, 2] = dt2
        buf[3, 1] = dt3 / 2.0
        buf[3, 3] = dt2
        return buf * self._process_noise
