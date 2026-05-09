from __future__ import annotations

import pytest

from scripts.nudge_offset import _read_response
from src.shared.protocol import (
    CAL_CMD_GET_OFFSETS,
    CAL_CMD_SET_OFFSETS,
    encode_cal_get_offsets,
    encode_cal_set_offsets,
)


class _FakeSerial:
    def __init__(self, response: bytes) -> None:
        self._response = bytearray(response)

    def read(self, size: int) -> bytes:
        if not self._response:
            return b""
        chunk = self._response[:size]
        del self._response[:size]
        return bytes(chunk)


@pytest.mark.unit
def test_read_response_skips_false_cal_header() -> None:
    fake = _FakeSerial(b"\xccnot-a-pkt!" + encode_cal_get_offsets())

    raw = _read_response(fake, CAL_CMD_GET_OFFSETS)

    assert raw == encode_cal_get_offsets()


@pytest.mark.unit
def test_read_response_requires_expected_command() -> None:
    fake = _FakeSerial(encode_cal_get_offsets() + encode_cal_set_offsets(1.25, -2.5))

    raw = _read_response(fake, CAL_CMD_SET_OFFSETS)

    assert raw == encode_cal_set_offsets(1.25, -2.5)
