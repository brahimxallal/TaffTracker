from __future__ import annotations

import pytest

from scripts.calibrate import (
    MANUAL_JOG_FAST_STEP_DEG,
    MANUAL_JOG_STEP_DEG,
    _KEY_LEFT,
    _KEY_UP,
    parse_jog_key,
)


@pytest.mark.unit
def test_parse_jog_key_uses_faster_default_step() -> None:
    assert parse_jog_key(_KEY_LEFT) == (-MANUAL_JOG_STEP_DEG, 0.0)
    assert parse_jog_key(_KEY_UP) == (0.0, MANUAL_JOG_STEP_DEG)
    assert parse_jog_key(ord("q")) == (-MANUAL_JOG_STEP_DEG, 0.0)
    assert parse_jog_key(ord("d")) == (MANUAL_JOG_STEP_DEG, 0.0)


@pytest.mark.unit
def test_parse_jog_key_uses_faster_shift_step() -> None:
    assert parse_jog_key(ord("Q")) == (-MANUAL_JOG_FAST_STEP_DEG, 0.0)
    assert parse_jog_key(ord("Z")) == (0.0, MANUAL_JOG_FAST_STEP_DEG)