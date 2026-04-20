import numpy as np
import pytest

from src.config import KalmanConfig
from src.tracking.kalman import KalmanFilter


@pytest.mark.unit
def test_kalman_initializes_from_first_measurement() -> None:
    kalman = KalmanFilter(process_noise=1.0, measurement_noise=1.0)

    state = kalman.update((10.0, 5.0), dt=0.1)

    assert state.x == 10.0
    assert state.y == 5.0
    assert state.vx == 0.0
    assert state.vy == 0.0


@pytest.mark.unit
def test_kalman_predicts_forward_motion() -> None:
    kalman = KalmanFilter(process_noise=0.01, measurement_noise=0.01)
    kalman.update((0.0, 0.0), dt=0.1)
    kalman.update((10.0, 0.0), dt=1.0)

    prediction = kalman.predict(1.0)

    assert prediction is not None
    assert prediction.x > 10.0
    assert prediction.vx > 0.0


@pytest.mark.unit
def test_kalman_predict_before_init_returns_none() -> None:
    kalman = KalmanFilter()

    result = kalman.predict(0.1)

    assert result is None


@pytest.mark.unit
def test_kalman_reset_clears_state() -> None:
    kalman = KalmanFilter()
    kalman.update((50.0, 75.0), dt=0.1)
    kalman.update((55.0, 80.0), dt=0.1)

    kalman.reset()

    assert not kalman.initialized
    assert kalman.current_state() is None


@pytest.mark.unit
def test_kalman_negative_dt_clamps_to_minimum() -> None:
    kalman = KalmanFilter()
    kalman.update((0.0, 0.0), dt=0.1)

    # Negative dt should not raise — it is clamped to 1e-3 internally
    state = kalman.update((1.0, 1.0), dt=-0.5)

    assert state is not None


@pytest.mark.unit
def test_kalman_snapshot_restore_roundtrip() -> None:
    """Snapshot after multiple updates, restore, verify state matches exactly."""
    kalman = KalmanFilter(process_noise=1.0, measurement_noise=1.0)
    for i in range(10):
        kalman.update((float(i * 10), float(i * 5)), dt=0.016)

    state_before = kalman.current_state()
    snap = kalman.snapshot()

    # Mutate the filter
    kalman.update((999.0, 999.0), dt=0.016)
    assert kalman.current_state().x != state_before.x

    # Restore
    kalman.restore(snap)
    state_after = kalman.current_state()

    assert abs(state_after.x - state_before.x) < 1e-9
    assert abs(state_after.y - state_before.y) < 1e-9
    assert abs(state_after.vx - state_before.vx) < 1e-9
    assert abs(state_after.vy - state_before.vy) < 1e-9


@pytest.mark.unit
def test_kalman_snapshot_is_independent_copy() -> None:
    """Snapshot arrays are independent — mutating filter doesn't affect snapshot."""
    kalman = KalmanFilter(process_noise=1.0, measurement_noise=1.0)
    kalman.update((100.0, 200.0), dt=0.016)

    snap = kalman.snapshot()
    original_state = snap["state"].copy()

    # Mutate the filter's internal state
    kalman.update((500.0, 500.0), dt=0.016)
    kalman.update((600.0, 600.0), dt=0.016)

    # Snapshot should be untouched
    np.testing.assert_array_equal(snap["state"], original_state)


@pytest.mark.unit
def test_kalman_soft_reset_on_consecutive_gated() -> None:
    """When max_consecutive_gated is exceeded, the filter should soft-reset
    (inflate covariance) rather than reinit from the rejected measurement."""
    cfg = KalmanConfig(
        innovation_gate_sigma=2.0,
        max_consecutive_gated=2,
        max_consecutive_predictions=30,
    )
    kalman = KalmanFilter(process_noise=1.0, measurement_noise=1.0, config=cfg)
    # Initialize at (100, 100) moving slowly
    kalman.update((100.0, 100.0), dt=0.016)
    kalman.update((101.0, 101.0), dt=0.016)

    # Inject wild measurements that should be gated
    for _ in range(3):
        kalman.update((900.0, 900.0), dt=0.016)

    state_after = kalman.current_state()
    # After soft reset: the state should NOT have jumped to (900, 900).
    # It should remain near the predicted position, not the rejected measurement.
    assert state_after is not None
    assert abs(state_after.x - 900.0) > 100.0, "State should NOT jump to rejected measurement"
    assert abs(state_after.y - 900.0) > 100.0, "State should NOT jump to rejected measurement"


@pytest.mark.unit
def test_kalman_adaptive_r_inverted_speed_relationship() -> None:
    """At low speed, R should be high (trust model). At high speed, R should be low (trust measurement)."""
    cfg = KalmanConfig(adaptive_r_min=8.0, adaptive_r_max=20.0, adaptive_r_speed_thresh=200.0)
    kalman = KalmanFilter(process_noise=1.0, measurement_noise=1.0, config=cfg)
    kalman.update((100.0, 100.0), dt=0.016)

    # At low speed: R should be close to adaptive_r_max
    kalman._adapt_noise(0.0)
    r_at_rest = kalman._measurement_noise
    assert r_at_rest == pytest.approx(20.0), f"At rest, R should be max (20.0), got {r_at_rest}"

    # At high speed: R should be close to adaptive_r_min
    kalman._adapt_noise(200.0)
    r_at_speed = kalman._measurement_noise
    assert r_at_speed == pytest.approx(
        8.0
    ), f"At full speed, R should be min (8.0), got {r_at_speed}"


@pytest.mark.unit
def test_kalman_nan_guard_resets_to_measurement() -> None:
    """If the state becomes non-finite after update, reset to measurement instead of propagating NaN."""
    kalman = KalmanFilter(process_noise=1.0, measurement_noise=1.0)
    kalman.update((100.0, 100.0), dt=0.016)

    # Force a near-singular covariance to trigger NaN
    kalman._covariance = np.full((4, 4), 1e308, dtype=np.float64)

    state = kalman.update((200.0, 200.0), dt=0.016)

    # State must be finite — NaN should have been caught and reset
    assert state is not None
    assert np.isfinite(state.x)
    assert np.isfinite(state.y)
    assert np.isfinite(state.vx)
    assert np.isfinite(state.vy)
