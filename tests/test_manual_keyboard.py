"""Unit tests for :class:`src.ui.manual_keyboard.ManualKeyboardDriver`."""

from __future__ import annotations

import pytest

from src.ui.manual_keyboard import (
    ARROW_DOWN,
    ARROW_LEFT,
    ARROW_RIGHT,
    ARROW_UP,
    ManualKeyboardConfig,
    ManualKeyboardDriver,
)


class _FakeValue:
    """Duck-typed mp.Value stand-in (just .value)."""

    def __init__(self, value: float) -> None:
        self.value = value


class _FakeClock:
    """Monotonic clock for deterministic dt under test."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.mark.unit
def test_initial_state_has_zero_velocity() -> None:
    driver = ManualKeyboardDriver(clock=_FakeClock())
    assert driver.velocity_pan_dps == 0.0
    assert driver.velocity_tilt_dps == 0.0


@pytest.mark.unit
def test_reset_zeroes_velocity() -> None:
    clock = _FakeClock()
    driver = ManualKeyboardDriver(clock=clock)
    pan = _FakeValue(0.0)
    tilt = _FakeValue(0.0)

    # Drive some velocity in.
    for _ in range(10):
        clock.advance(0.05)
        driver.tick(
            key=0,
            key_low=ord("d"),
            manual_pan=pan,
            manual_tilt=tilt,
            pan_limit_deg=90.0,
            tilt_limit_deg=90.0,
        )
    assert driver.velocity_pan_dps > 0.0

    driver.reset()
    assert driver.velocity_pan_dps == 0.0
    assert driver.velocity_tilt_dps == 0.0


@pytest.mark.unit
def test_d_key_drives_pan_positive() -> None:
    clock = _FakeClock()
    driver = ManualKeyboardDriver(clock=clock)
    pan = _FakeValue(0.0)
    tilt = _FakeValue(0.0)

    for _ in range(20):
        clock.advance(0.05)
        driver.tick(
            key=0,
            key_low=ord("d"),
            manual_pan=pan,
            manual_tilt=tilt,
            pan_limit_deg=90.0,
            tilt_limit_deg=90.0,
        )

    assert pan.value > 0.0


@pytest.mark.unit
def test_q_key_drives_pan_negative() -> None:
    clock = _FakeClock()
    driver = ManualKeyboardDriver(clock=clock)
    pan = _FakeValue(0.0)
    tilt = _FakeValue(0.0)

    for _ in range(20):
        clock.advance(0.05)
        driver.tick(
            key=0,
            key_low=ord("q"),
            manual_pan=pan,
            manual_tilt=tilt,
            pan_limit_deg=90.0,
            tilt_limit_deg=90.0,
        )

    assert pan.value < 0.0


@pytest.mark.unit
def test_z_and_s_keys_drive_tilt() -> None:
    clock = _FakeClock()
    driver = ManualKeyboardDriver(clock=clock)
    pan = _FakeValue(0.0)
    tilt_up = _FakeValue(0.0)

    for _ in range(20):
        clock.advance(0.05)
        driver.tick(
            key=0,
            key_low=ord("z"),
            manual_pan=pan,
            manual_tilt=tilt_up,
            pan_limit_deg=90.0,
            tilt_limit_deg=90.0,
        )
    assert tilt_up.value < 0.0

    driver.reset()
    pan = _FakeValue(0.0)
    tilt_down = _FakeValue(0.0)
    for _ in range(20):
        clock.advance(0.05)
        driver.tick(
            key=0,
            key_low=ord("s"),
            manual_pan=pan,
            manual_tilt=tilt_down,
            pan_limit_deg=90.0,
            tilt_limit_deg=90.0,
        )
    assert tilt_down.value > 0.0


@pytest.mark.unit
def test_arrow_keys_use_coarse_speed() -> None:
    """Arrow keys should drive faster than ZQSD over the same time window.

    Use a small enough total dt so neither driver saturates at the limit.
    """
    clock_arrow = _FakeClock()
    driver_arrow = ManualKeyboardDriver(clock=clock_arrow)
    pan_arrow = _FakeValue(0.0)
    tilt_arrow = _FakeValue(0.0)
    for _ in range(3):
        clock_arrow.advance(0.05)
        driver_arrow.tick(
            key=ARROW_RIGHT,
            key_low=0,
            manual_pan=pan_arrow,
            manual_tilt=tilt_arrow,
            pan_limit_deg=90.0,
            tilt_limit_deg=90.0,
        )

    clock_zqsd = _FakeClock()
    driver_zqsd = ManualKeyboardDriver(clock=clock_zqsd)
    pan_zqsd = _FakeValue(0.0)
    tilt_zqsd = _FakeValue(0.0)
    for _ in range(3):
        clock_zqsd.advance(0.05)
        driver_zqsd.tick(
            key=0,
            key_low=ord("d"),
            manual_pan=pan_zqsd,
            manual_tilt=tilt_zqsd,
            pan_limit_deg=90.0,
            tilt_limit_deg=90.0,
        )

    # Coarse (300 dps) over 3×50ms ≈ 45° before saturation.
    # Fine (120 dps) over 3×50ms ≈ 18° — no saturation.
    assert pan_arrow.value < 90.0
    assert pan_zqsd.value < 90.0
    assert pan_arrow.value > pan_zqsd.value


@pytest.mark.unit
def test_pan_clamps_to_limit() -> None:
    clock = _FakeClock()
    driver = ManualKeyboardDriver(clock=clock)
    pan = _FakeValue(0.0)
    tilt = _FakeValue(0.0)
    pan_limit = 5.0  # very tight limit

    for _ in range(200):
        clock.advance(0.05)
        driver.tick(
            key=ARROW_RIGHT,
            key_low=0,
            manual_pan=pan,
            manual_tilt=tilt,
            pan_limit_deg=pan_limit,
            tilt_limit_deg=90.0,
        )

    assert pan.value == pytest.approx(pan_limit, abs=1e-9)


@pytest.mark.unit
def test_tilt_clamps_to_negative_limit() -> None:
    clock = _FakeClock()
    driver = ManualKeyboardDriver(clock=clock)
    pan = _FakeValue(0.0)
    tilt = _FakeValue(0.0)
    tilt_limit = 3.0

    for _ in range(200):
        clock.advance(0.05)
        driver.tick(
            key=ARROW_UP,
            key_low=0,
            manual_pan=pan,
            manual_tilt=tilt,
            pan_limit_deg=90.0,
            tilt_limit_deg=tilt_limit,
        )

    assert tilt.value == pytest.approx(-tilt_limit, abs=1e-9)


@pytest.mark.unit
def test_no_key_decays_velocity_to_zero() -> None:
    clock = _FakeClock()
    driver = ManualKeyboardDriver(clock=clock)
    pan = _FakeValue(0.0)
    tilt = _FakeValue(0.0)

    # Build up some velocity
    for _ in range(10):
        clock.advance(0.05)
        driver.tick(
            key=0,
            key_low=ord("d"),
            manual_pan=pan,
            manual_tilt=tilt,
            pan_limit_deg=90.0,
            tilt_limit_deg=90.0,
        )
    assert driver.velocity_pan_dps > 0.0

    # Release the key — many ticks with no input
    for _ in range(100):
        clock.advance(0.05)
        driver.tick(
            key=0,
            key_low=0,
            manual_pan=pan,
            manual_tilt=tilt,
            pan_limit_deg=90.0,
            tilt_limit_deg=90.0,
        )

    assert driver.velocity_pan_dps == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_diagonal_arrow_combo_moves_both_axes() -> None:
    """ARROW_LEFT alone moves pan; ARROW_DOWN alone moves tilt; the driver
    treats them as two independent axes inside one tick."""
    clock = _FakeClock()
    driver = ManualKeyboardDriver(clock=clock)
    pan = _FakeValue(0.0)
    tilt = _FakeValue(0.0)

    # Two consecutive ticks, one with LEFT, one with DOWN.
    clock.advance(0.05)
    driver.tick(
        key=ARROW_LEFT,
        key_low=0,
        manual_pan=pan,
        manual_tilt=tilt,
        pan_limit_deg=90.0,
        tilt_limit_deg=90.0,
    )
    clock.advance(0.05)
    driver.tick(
        key=ARROW_DOWN,
        key_low=0,
        manual_pan=pan,
        manual_tilt=tilt,
        pan_limit_deg=90.0,
        tilt_limit_deg=90.0,
    )

    assert pan.value < 0.0
    assert tilt.value > 0.0


@pytest.mark.unit
def test_movement_threshold_clamps_residual_velocity_to_zero() -> None:
    """Below threshold, the driver snaps velocity exactly to zero rather
    than letting it crawl forever."""
    clock = _FakeClock()
    cfg = ManualKeyboardConfig(movement_threshold_dps=10.0, decel_multiplier=10.0)
    driver = ManualKeyboardDriver(cfg, clock=clock)
    pan = _FakeValue(0.0)
    tilt = _FakeValue(0.0)

    # One quick burst
    clock.advance(0.05)
    driver.tick(
        key=0,
        key_low=ord("d"),
        manual_pan=pan,
        manual_tilt=tilt,
        pan_limit_deg=90.0,
        tilt_limit_deg=90.0,
    )

    # Many idle ticks to decay below threshold
    for _ in range(50):
        clock.advance(0.05)
        driver.tick(
            key=0,
            key_low=0,
            manual_pan=pan,
            manual_tilt=tilt,
            pan_limit_deg=90.0,
            tilt_limit_deg=90.0,
        )

    assert driver.velocity_pan_dps == 0.0
