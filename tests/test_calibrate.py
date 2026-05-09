from __future__ import annotations

import pytest

from scripts.calibrate import (
    _KEY_LEFT,
    _KEY_UP,
    MANUAL_JOG_FAST_STEP_DEG,
    MANUAL_JOG_STEP_DEG,
    parse_jog_key,
    read_current_offsets,
    write_offsets_and_confirm,
)
from src.shared.protocol import encode_cal_get_offsets, encode_cal_set_offsets


class _FakeTransport:
    def __init__(self, response: bytes) -> None:
        self._response = bytearray(response)
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    def read(self, size: int, timeout_s: float = 0.2) -> bytes:
        if not self._response:
            return b""
        chunk = self._response[:size]
        del self._response[:size]
        return bytes(chunk)

    def reset_input_buffer(self) -> None:
        return None


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


@pytest.mark.unit
def test_read_current_offsets_resyncs_past_firmware_logs() -> None:
    transport = _FakeTransport(b"I (123) comm: saved offsets\r\n" + encode_cal_get_offsets())

    assert read_current_offsets(transport) == (0.0, 0.0)


@pytest.mark.unit
def test_read_current_offsets_skips_false_cal_header() -> None:
    false_header = b"\xccnot-a-pkt!"
    transport = _FakeTransport(false_header + encode_cal_get_offsets())

    assert read_current_offsets(transport) == (0.0, 0.0)


@pytest.mark.unit
def test_write_offsets_resyncs_past_firmware_logs() -> None:
    transport = _FakeTransport(
        b"W (123) comm: calibration\r\n" + encode_cal_set_offsets(1.25, -2.5)
    )

    assert write_offsets_and_confirm(transport, 1.25, -2.5) is True


@pytest.mark.unit
def test_write_offsets_requires_set_ack() -> None:
    transport = _FakeTransport(encode_cal_get_offsets() + encode_cal_set_offsets(1.25, -2.5))

    assert write_offsets_and_confirm(transport, 1.25, -2.5) is True
