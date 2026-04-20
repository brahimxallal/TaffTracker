from __future__ import annotations

import struct
from dataclasses import dataclass

# --- Protocol v2 (21 bytes, header 0xBB) ---

HEADER_V2 = 0xBB
# B(1) H(2) I(4) h(2)*4 B(1)*4 H(2) = 1 + 2 + 4 + 8 + 4 + 2 = 21 bytes
PACKET_V2_STRUCT = struct.Struct("<BHIhhhhBBBBH")
PACKET_V2_SIZE = PACKET_V2_STRUCT.size  # 21 bytes

# --- Calibration protocol (10 bytes, header 0xCC, USB only) ---

HEADER_CAL = 0xCC
CAL_CMD_SET_OFFSETS = 0x01
CAL_CMD_GET_OFFSETS = 0x02
CAL_CMD_RESET_DEFAULTS = 0x03
# B(1) B(1) h(2) h(2) H(2) H(2) = 10 bytes
# header | cmd | pan_centideg | tilt_centideg | reserved | crc16
CAL_PACKET_STRUCT = struct.Struct("<BBhhHH")
CAL_PACKET_SIZE = CAL_PACKET_STRUCT.size  # 10 bytes


# State flag bit positions (byte 16 of v2 packet).
STATE_LOST = 0b00
STATE_PREDICTION = 0b01
STATE_MEASUREMENT = 0b10
STATE_CENTER = 0b11
FLAG_TARGET_ACQUIRED = 1 << 2
FLAG_HIGH_CONFIDENCE = 1 << 3
FLAG_FAST_MOTION = 1 << 4
FLAG_OCCLUSION_RECOVERY = 1 << 5
FLAG_LASER_ON = 1 << 6
FLAG_RELAY_ON = 1 << 7

# Quality byte flags (MSB used for manual mode signalling to firmware)
QUALITY_FLAG_MANUAL = 1 << 7  # Bit 7 of quality byte: bypass firmware EMA/deadzone


@dataclass(frozen=True)
class TrackingPacketV2:
    sequence: int
    timestamp_ms: int
    pan: int  # centidegrees
    tilt: int  # centidegrees
    pan_vel: int  # centidegrees/sec
    tilt_vel: int  # centidegrees/sec
    confidence: int  # 0-255
    state: int  # state flags byte
    quality: int  # 0-255
    latency: int  # PC pipeline latency in ms, 0-255


# --- Shared helpers ---


def _clamp_int16(value: int) -> int:
    return max(-32768, min(32767, value))


def _clamp_uint8(value: int) -> int:
    return max(0, min(255, value))


def checksum_bytes(payload: bytes) -> int:
    # CRC16-CCITT (poly: 0x1021, init: 0xFFFF)
    crc = 0xFFFF
    for byte in payload:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc = crc << 1
            crc &= 0xFFFF
    return crc


# --- v2 encode/decode ---


def encode_packet_v2(
    sequence: int,
    timestamp_ms: int,
    pan: int,
    tilt: int,
    pan_vel: int,
    tilt_vel: int,
    confidence: int,
    state: int,
    quality: int,
    latency: int,
) -> bytes:
    seq = sequence & 0xFFFF
    ts = timestamp_ms & 0xFFFFFFFF
    pan_val = _clamp_int16(pan)
    tilt_val = _clamp_int16(tilt)
    pan_vel_val = _clamp_int16(pan_vel)
    tilt_vel_val = _clamp_int16(tilt_vel)
    conf = _clamp_uint8(confidence)
    st = _clamp_uint8(state)
    qual = _clamp_uint8(quality)
    lat = _clamp_uint8(latency)
    # Payload = bytes 1-18 (everything except header and checksum)
    payload = struct.pack(
        "<HIhhhhBBBB",
        seq,
        ts,
        pan_val,
        tilt_val,
        pan_vel_val,
        tilt_vel_val,
        conf,
        st,
        qual,
        lat,
    )
    checksum = checksum_bytes(payload)
    return PACKET_V2_STRUCT.pack(
        HEADER_V2,
        seq,
        ts,
        pan_val,
        tilt_val,
        pan_vel_val,
        tilt_vel_val,
        conf,
        st,
        qual,
        lat,
        checksum,
    )


def decode_packet_v2(data: bytes) -> TrackingPacketV2 | None:
    if len(data) != PACKET_V2_SIZE:
        return None

    (
        header,
        sequence,
        timestamp_ms,
        pan,
        tilt,
        pan_vel,
        tilt_vel,
        confidence,
        state,
        quality,
        latency,
        checksum,
    ) = PACKET_V2_STRUCT.unpack(data)

    if header != HEADER_V2:
        return None

    payload = struct.pack(
        "<HIhhhhBBBB",
        sequence,
        timestamp_ms,
        pan,
        tilt,
        pan_vel,
        tilt_vel,
        confidence,
        state,
        quality,
        latency,
    )
    if checksum != checksum_bytes(payload):
        return None

    return TrackingPacketV2(
        sequence=sequence,
        timestamp_ms=timestamp_ms,
        pan=pan,
        tilt=tilt,
        pan_vel=pan_vel,
        tilt_vel=tilt_vel,
        confidence=confidence,
        state=state,
        quality=quality,
        latency=latency,
    )


def build_state_flags(
    state_source: str,
    target_acquired: bool,
    confidence: float,
    velocity_magnitude_dps: float,
    is_occlusion_recovery: bool,
) -> int:
    if state_source == "measurement":
        flags = STATE_MEASUREMENT
    elif state_source == "prediction":
        flags = STATE_PREDICTION
    elif state_source == "center":
        flags = STATE_CENTER
    else:
        flags = STATE_LOST
    if target_acquired:
        flags |= FLAG_TARGET_ACQUIRED
    if confidence > 0.7:
        flags |= FLAG_HIGH_CONFIDENCE
    if velocity_magnitude_dps > 60.0:
        flags |= FLAG_FAST_MOTION
    if is_occlusion_recovery:
        flags |= FLAG_OCCLUSION_RECOVERY
    return flags


# --- Calibration packet encode/decode (0xCC, USB only) ---


def encode_cal_set_offsets(pan_offset_deg: float, tilt_offset_deg: float) -> bytes:
    """Encode a calibration packet to set servo offsets on the ESP32 NVS."""
    pan_cd = _clamp_int16(int(round(pan_offset_deg * 100.0)))
    tilt_cd = _clamp_int16(int(round(tilt_offset_deg * 100.0)))
    payload = struct.pack("<BhhH", CAL_CMD_SET_OFFSETS, pan_cd, tilt_cd, 0)
    crc = checksum_bytes(payload)
    return CAL_PACKET_STRUCT.pack(HEADER_CAL, CAL_CMD_SET_OFFSETS, pan_cd, tilt_cd, 0, crc)


def encode_cal_get_offsets() -> bytes:
    """Encode a calibration packet to request current offsets from the ESP32."""
    payload = struct.pack("<BhhH", CAL_CMD_GET_OFFSETS, 0, 0, 0)
    crc = checksum_bytes(payload)
    return CAL_PACKET_STRUCT.pack(HEADER_CAL, CAL_CMD_GET_OFFSETS, 0, 0, 0, crc)


def encode_cal_reset_defaults() -> bytes:
    """Encode a calibration packet to reset offsets to compile-time defaults."""
    payload = struct.pack("<BhhH", CAL_CMD_RESET_DEFAULTS, 0, 0, 0)
    crc = checksum_bytes(payload)
    return CAL_PACKET_STRUCT.pack(HEADER_CAL, CAL_CMD_RESET_DEFAULTS, 0, 0, 0, crc)


@dataclass(frozen=True)
class CalibrationResponse:
    command: int
    pan_offset_cd: int  # centidegrees
    tilt_offset_cd: int  # centidegrees

    @property
    def pan_offset_deg(self) -> float:
        return self.pan_offset_cd / 100.0

    @property
    def tilt_offset_deg(self) -> float:
        return self.tilt_offset_cd / 100.0


def decode_cal_response(data: bytes) -> CalibrationResponse | None:
    """Decode a calibration response packet from the ESP32."""
    if len(data) != CAL_PACKET_SIZE:
        return None
    header, cmd, pan_cd, tilt_cd, reserved, crc = CAL_PACKET_STRUCT.unpack(data)
    if header != HEADER_CAL:
        return None
    payload = struct.pack("<BhhH", cmd, pan_cd, tilt_cd, reserved)
    if crc != checksum_bytes(payload):
        return None
    return CalibrationResponse(command=cmd, pan_offset_cd=pan_cd, tilt_offset_cd=tilt_cd)
