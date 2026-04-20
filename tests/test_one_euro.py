from __future__ import annotations

import math

import pytest

from src.tracking.one_euro import OneEuroFilter2D


@pytest.fixture
def filt() -> OneEuroFilter2D:
    return OneEuroFilter2D(mincutoff=1.0, beta=0.0, dcutoff=1.0)


# ── First sample passes through unchanged ─────────────────────────


@pytest.mark.unit
def test_first_sample_passthrough(filt: OneEuroFilter2D) -> None:
    result = filt((100.0, 200.0), t=0.0)
    assert result == (100.0, 200.0)


# ── Stationary input converges to the constant value ──────────────


@pytest.mark.unit
def test_stationary_converges(filt: OneEuroFilter2D) -> None:
    target = (300.0, 150.0)
    dt = 1 / 60.0
    for i in range(200):
        result = filt(target, t=i * dt)
    assert abs(result[0] - target[0]) < 0.5
    assert abs(result[1] - target[1]) < 0.5


# ── Zero dt returns input unchanged ──────────────────────────────


@pytest.mark.unit
def test_zero_dt_returns_input(filt: OneEuroFilter2D) -> None:
    filt((10.0, 20.0), t=1.0)
    result = filt((50.0, 60.0), t=1.0)  # same timestamp → dt=0
    assert result == (50.0, 60.0)


# ── Higher beta reduces lag on fast motion ────────────────────────


@pytest.mark.unit
def test_higher_beta_reduces_lag() -> None:
    low_beta = OneEuroFilter2D(mincutoff=1.0, beta=0.0, dcutoff=1.0)
    high_beta = OneEuroFilter2D(mincutoff=1.0, beta=1.0, dcutoff=1.0)

    dt = 1 / 60.0
    # Feed same step input to both
    for i in range(10):
        low_beta((0.0, 0.0), t=i * dt)
        high_beta((0.0, 0.0), t=i * dt)

    # Step to (100, 100)
    for i in range(10, 30):
        r_low = low_beta((100.0, 100.0), t=i * dt)
        r_high = high_beta((100.0, 100.0), t=i * dt)

    # High beta should be closer to target (less lag)
    err_low = math.hypot(r_low[0] - 100, r_low[1] - 100)
    err_high = math.hypot(r_high[0] - 100, r_high[1] - 100)
    assert err_high < err_low, f"High beta error {err_high:.2f} >= low beta {err_low:.2f}"


# ── Lower mincutoff reduces jitter ───────────────────────────────


@pytest.mark.unit
def test_lower_mincutoff_reduces_jitter() -> None:
    smooth = OneEuroFilter2D(mincutoff=0.1, beta=0.0, dcutoff=1.0)
    rough = OneEuroFilter2D(mincutoff=10.0, beta=0.0, dcutoff=1.0)

    dt = 1 / 60.0
    # Random jitter around 100 (not perfectly alternating)
    import random

    rng = random.Random(42)
    rng_vals = [100 + rng.uniform(-5, 5) for _ in range(200)]

    outputs_smooth = []
    outputs_rough = []
    for i, v in enumerate(rng_vals):
        outputs_smooth.append(smooth((float(v), float(v)), t=i * dt))
        outputs_rough.append(rough((float(v), float(v)), t=i * dt))

    # Lower mincutoff → lower cutoff frequency → heavier smoothing → smaller variance
    var_smooth = sum((o[0] - 100) ** 2 for o in outputs_smooth[-100:]) / 100
    var_rough = sum((o[0] - 100) ** 2 for o in outputs_rough[-100:]) / 100
    assert var_smooth < var_rough, (
        f"Filter with mincutoff=0.1 should be smoother: var={var_smooth:.2f}, "
        f"rough var={var_rough:.2f}"
    )


# ── _alpha returns correct value ─────────────────────────────────


@pytest.mark.unit
def test_alpha_computation(filt: OneEuroFilter2D) -> None:
    dt = 1 / 60.0
    cutoff = 1.0
    tau = 1.0 / (2 * math.pi * cutoff)
    expected = 1.0 / (1.0 + tau / dt)
    assert abs(filt._alpha(dt, cutoff) - expected) < 1e-10


# ── snapshot / restore roundtrip ─────────────────────────────────


@pytest.mark.unit
def test_snapshot_restore_roundtrip() -> None:
    f1 = OneEuroFilter2D(mincutoff=1.0, beta=0.5, dcutoff=1.0)
    dt = 1 / 60.0
    for i in range(20):
        f1((float(i * 5), float(i * 3)), t=i * dt)

    snap = f1.snapshot()

    f2 = OneEuroFilter2D(mincutoff=1.0, beta=0.5, dcutoff=1.0)
    f2.restore(snap)

    # Both should produce identical output on next input
    next_input = (200.0, 100.0)
    next_t = 20 * dt
    assert f1(next_input, next_t) == f2(next_input, next_t)


# ── Negative dt returns input (safety) ───────────────────────────


@pytest.mark.unit
def test_negative_dt_returns_input(filt: OneEuroFilter2D) -> None:
    filt((10.0, 20.0), t=1.0)
    result = filt((50.0, 60.0), t=0.5)  # earlier timestamp → dt<0
    assert result == (50.0, 60.0)


# ── 2D follows each axis independently ───────────────────────────


@pytest.mark.unit
def test_independent_axes() -> None:
    f = OneEuroFilter2D(mincutoff=1.0, beta=0.0, dcutoff=1.0)
    dt = 1 / 60.0
    # Move only in X
    for i in range(50):
        result = f((float(i * 2), 0.0), t=i * dt)
    # X should have moved significantly, Y stays near 0
    assert result[0] > 50.0
    assert abs(result[1]) < 1.0


@pytest.mark.unit
def test_dx_dy_before_any_sample() -> None:
    """dx/dy should be 0.0 before any samples are fed."""
    f = OneEuroFilter2D()
    assert f.dx == 0.0
    assert f.dy == 0.0


@pytest.mark.unit
def test_dx_dy_reflect_motion() -> None:
    """dx/dy should reflect the smoothed velocity in pixels/second."""
    f = OneEuroFilter2D(mincutoff=100.0, beta=0.0, dcutoff=100.0)
    dt = 1 / 60.0
    speed_px_per_s = 120.0  # 2 px/frame at 60fps
    for i in range(60):
        f((i * speed_px_per_s * dt, 0.0), t=i * dt)
    # After convergence, dx should approximate the input speed
    assert f.dx == pytest.approx(speed_px_per_s, rel=0.2)
    assert f.dy == pytest.approx(0.0, abs=5.0)
