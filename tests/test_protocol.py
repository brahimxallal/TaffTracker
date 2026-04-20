import pytest

from src.shared.protocol import (
    PACKET_V2_SIZE,
    FLAG_FAST_MOTION,
    FLAG_HIGH_CONFIDENCE,
    FLAG_OCCLUSION_RECOVERY,
    FLAG_TARGET_ACQUIRED,
    STATE_CENTER,
    STATE_LOST,
    STATE_MEASUREMENT,
    STATE_PREDICTION,
    build_state_flags,
    decode_packet_v2,
    encode_packet_v2,
    CAL_PACKET_SIZE,
    HEADER_CAL,
    CAL_CMD_SET_OFFSETS,
    CAL_CMD_GET_OFFSETS,
    CAL_CMD_RESET_DEFAULTS,
    encode_cal_set_offsets,
    encode_cal_get_offsets,
    encode_cal_reset_defaults,
    decode_cal_response,
)


# --- v2 protocol tests ---


@pytest.mark.unit
def test_v2_packet_size() -> None:
    assert PACKET_V2_SIZE == 21

@pytest.mark.unit
def test_v2_round_trip() -> None:
    packet = encode_packet_v2(
        sequence=42,
        timestamp_ms=987654,
        pan=1500,
        tilt=-2000,
        pan_vel=500,
        tilt_vel=-300,
        confidence=200,
        state=0b00100110,
        quality=180,
        latency=12,
    )

    assert len(packet) == PACKET_V2_SIZE

    decoded = decode_packet_v2(packet)
    assert decoded is not None
    assert decoded.sequence == 42
    assert decoded.timestamp_ms == 987654
    assert decoded.pan == 1500
    assert decoded.tilt == -2000
    assert decoded.pan_vel == 500
    assert decoded.tilt_vel == -300
    assert decoded.confidence == 200
    assert decoded.state == 0b00100110
    assert decoded.quality == 180
    assert decoded.latency == 12


@pytest.mark.unit
def test_v2_rejects_corruption() -> None:
    packet = bytearray(encode_packet_v2(
        sequence=1, timestamp_ms=2, pan=3, tilt=4,
        pan_vel=5, tilt_vel=6, confidence=7, state=8, quality=9, latency=10,
    ))
    packet[-1] ^= 0xFF
    assert decode_packet_v2(bytes(packet)) is None


@pytest.mark.unit
def test_v2_rejects_wrong_header() -> None:
    packet = bytearray(encode_packet_v2(
        sequence=1, timestamp_ms=2, pan=3, tilt=4,
        pan_vel=5, tilt_vel=6, confidence=7, state=8, quality=9, latency=10,
    ))
    packet[0] = 0xAA  # v1 header
    assert decode_packet_v2(bytes(packet)) is None


@pytest.mark.unit
def test_v2_rejects_wrong_size() -> None:
    assert decode_packet_v2(b"\xBB" * 10) is None
    assert decode_packet_v2(b"\xBB" * 21) is None


@pytest.mark.unit
def test_v2_clamps_values() -> None:
    packet = encode_packet_v2(
        sequence=0,
        timestamp_ms=0,
        pan=40000,       # exceeds int16 max
        tilt=-40000,     # exceeds int16 min
        pan_vel=0,
        tilt_vel=0,
        confidence=300,  # exceeds uint8
        state=0,
        quality=0,
        latency=999,     # exceeds uint8
    )
    decoded = decode_packet_v2(packet)
    assert decoded is not None
    assert decoded.pan == 32767
    assert decoded.tilt == -32768
    assert decoded.confidence == 255
    assert decoded.latency == 255


@pytest.mark.unit
def test_v2_negative_velocity() -> None:
    packet = encode_packet_v2(
        sequence=0, timestamp_ms=0, pan=0, tilt=0,
        pan_vel=-15000, tilt_vel=-20000,
        confidence=0, state=0, quality=0, latency=0,
    )
    decoded = decode_packet_v2(packet)
    assert decoded is not None
    assert decoded.pan_vel == -15000
    assert decoded.tilt_vel == -20000


# --- state flags tests ---


@pytest.mark.unit
def test_build_state_flags_measurement() -> None:
    flags = build_state_flags(
        state_source="measurement",
        target_acquired=True,
        confidence=0.9,
        velocity_magnitude_dps=10.0,
        is_occlusion_recovery=False,
    )
    assert (flags & 0b11) == STATE_MEASUREMENT
    assert flags & FLAG_TARGET_ACQUIRED
    assert flags & FLAG_HIGH_CONFIDENCE
    assert not (flags & FLAG_FAST_MOTION)
    assert not (flags & FLAG_OCCLUSION_RECOVERY)


@pytest.mark.unit
def test_build_state_flags_prediction_fast() -> None:
    flags = build_state_flags(
        state_source="prediction",
        target_acquired=False,
        confidence=0.3,
        velocity_magnitude_dps=90.0,
        is_occlusion_recovery=False,
    )
    assert (flags & 0b11) == STATE_PREDICTION
    assert not (flags & FLAG_TARGET_ACQUIRED)
    assert not (flags & FLAG_HIGH_CONFIDENCE)
    assert flags & FLAG_FAST_MOTION


@pytest.mark.unit
def test_build_state_flags_lost() -> None:
    flags = build_state_flags(
        state_source="lost",
        target_acquired=False,
        confidence=0.0,
        velocity_magnitude_dps=0.0,
        is_occlusion_recovery=False,
    )
    assert (flags & 0b11) == STATE_LOST
    assert flags == 0


@pytest.mark.unit
def test_build_state_flags_center() -> None:
    flags = build_state_flags(
        state_source="center",
        target_acquired=False,
        confidence=0.0,
        velocity_magnitude_dps=0.0,
        is_occlusion_recovery=False,
    )
    assert (flags & 0b11) == STATE_CENTER


@pytest.mark.unit
def test_build_state_flags_occlusion_recovery() -> None:
    flags = build_state_flags(
        state_source="measurement",
        target_acquired=True,
        confidence=0.8,
        velocity_magnitude_dps=5.0,
        is_occlusion_recovery=True,
    )
    assert flags & FLAG_OCCLUSION_RECOVERY
    assert flags & FLAG_HIGH_CONFIDENCE
    assert flags & FLAG_TARGET_ACQUIRED


# --- calibration protocol tests ---


@pytest.mark.unit
def test_cal_packet_size() -> None:
    assert CAL_PACKET_SIZE == 10


@pytest.mark.unit
def test_cal_set_offsets_round_trip() -> None:
    packet = encode_cal_set_offsets(-65.0, 7.0)
    assert len(packet) == CAL_PACKET_SIZE
    assert packet[0] == HEADER_CAL
    resp = decode_cal_response(packet)
    assert resp is not None
    assert resp.command == CAL_CMD_SET_OFFSETS
    assert resp.pan_offset_cd == -6500
    assert resp.tilt_offset_cd == 700
    assert resp.pan_offset_deg == pytest.approx(-65.0)
    assert resp.tilt_offset_deg == pytest.approx(7.0)


@pytest.mark.unit
def test_cal_get_offsets_round_trip() -> None:
    packet = encode_cal_get_offsets()
    assert len(packet) == CAL_PACKET_SIZE
    resp = decode_cal_response(packet)
    assert resp is not None
    assert resp.command == CAL_CMD_GET_OFFSETS
    assert resp.pan_offset_cd == 0
    assert resp.tilt_offset_cd == 0


@pytest.mark.unit
def test_cal_reset_defaults_round_trip() -> None:
    packet = encode_cal_reset_defaults()
    assert len(packet) == CAL_PACKET_SIZE
    resp = decode_cal_response(packet)
    assert resp is not None
    assert resp.command == CAL_CMD_RESET_DEFAULTS


@pytest.mark.unit
def test_cal_rejects_corruption() -> None:
    packet = bytearray(encode_cal_set_offsets(10.0, -5.0))
    packet[-1] ^= 0xFF
    assert decode_cal_response(bytes(packet)) is None


@pytest.mark.unit
def test_cal_rejects_wrong_header() -> None:
    packet = bytearray(encode_cal_set_offsets(1.0, 2.0))
    packet[0] = 0xAA
    assert decode_cal_response(bytes(packet)) is None


@pytest.mark.unit
def test_cal_rejects_wrong_size() -> None:
    assert decode_cal_response(b"\xCC" * 5) is None
    assert decode_cal_response(b"\xCC" * 15) is None


@pytest.mark.unit
def test_cal_fractional_offsets() -> None:
    packet = encode_cal_set_offsets(-12.34, 5.67)
    resp = decode_cal_response(packet)
    assert resp is not None
    assert resp.pan_offset_cd == -1234
    assert resp.tilt_offset_cd == 567
    assert resp.pan_offset_deg == pytest.approx(-12.34)
    assert resp.tilt_offset_deg == pytest.approx(5.67)
