from __future__ import annotations

import pytest

from src.sensors.phone_imu_protocol import (
    HEADER_SIZE,
    SENSOR_TYPE_ACCELEROMETER,
    SENSOR_TYPE_GAME_ROTATION_VECTOR,
    SENSOR_TYPE_GYROSCOPE,
    SENSOR_TYPE_GYROSCOPE_UNCALIBRATED,
    ImuSample,
    pack_imu_sample,
    parse_imu_sample,
)


@pytest.mark.unit
def test_imu_sample_packet_round_trip() -> None:
    sample = ImuSample(
        version=1,
        header_size=HEADER_SIZE,
        sequence=42,
        timestamp_ns=123_456_789,
        sensor_type=SENSOR_TYPE_GYROSCOPE,
        accuracy=3,
        values=(0.1, -0.2, 0.3),
        values_count=3,
        flags=0,
    )

    packet = pack_imu_sample(sample)
    parsed = parse_imu_sample(packet)

    assert parsed.version == sample.version
    assert parsed.header_size == sample.header_size
    assert parsed.sequence == sample.sequence
    assert parsed.timestamp_ns == sample.timestamp_ns
    assert parsed.sensor_type == sample.sensor_type
    assert parsed.accuracy == sample.accuracy
    assert parsed.values_count == sample.values_count
    assert parsed.flags == sample.flags
    assert parsed.values == pytest.approx(sample.values)
    assert parsed.sensor_name == "gyroscope"


@pytest.mark.unit
def test_imu_packet_rejects_bad_magic() -> None:
    sample = ImuSample(
        version=1,
        header_size=HEADER_SIZE,
        sequence=1,
        timestamp_ns=2,
        sensor_type=SENSOR_TYPE_ACCELEROMETER,
        accuracy=3,
        values=(1.0, 2.0, 3.0),
        values_count=3,
        flags=0,
    )
    packet = bytearray(pack_imu_sample(sample))
    packet[:8] = b"BADIMU!!"

    with pytest.raises(ValueError, match="magic"):
        parse_imu_sample(bytes(packet))


@pytest.mark.unit
def test_imu_packet_rejects_invalid_values_count() -> None:
    sample = ImuSample(
        version=1,
        header_size=HEADER_SIZE,
        sequence=1,
        timestamp_ns=2,
        sensor_type=SENSOR_TYPE_GAME_ROTATION_VECTOR,
        accuracy=3,
        values=(0.0, 0.0, 0.0, 1.0),
        values_count=17,
        flags=0,
    )

    with pytest.raises(ValueError, match="values_count"):
        pack_imu_sample(sample)


@pytest.mark.unit
def test_imu_packet_supports_uncalibrated_six_axis_values() -> None:
    sample = ImuSample(
        version=1,
        header_size=HEADER_SIZE,
        sequence=7,
        timestamp_ns=70,
        sensor_type=SENSOR_TYPE_GYROSCOPE_UNCALIBRATED,
        accuracy=0,
        values=(1.0, 2.0, 3.0, 0.01, 0.02, 0.03),
        values_count=6,
        flags=0,
    )

    parsed = parse_imu_sample(pack_imu_sample(sample))

    assert parsed.sensor_name == "gyroscope_uncalibrated"
    assert parsed.active_values == pytest.approx((1.0, 2.0, 3.0, 0.01, 0.02, 0.03))
