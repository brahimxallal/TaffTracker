from __future__ import annotations

import socket
import subprocess

import pytest

from src.sensors.phone_imu import PhoneImuReceiver, PhoneImuRuntimeConfig
from src.sensors.phone_imu_protocol import (
    HEADER_SIZE,
    SENSOR_TYPE_ACCELEROMETER,
    SENSOR_TYPE_GYROSCOPE,
    ImuSample,
    pack_imu_sample,
)


def _socket_pair() -> tuple[socket.socket, socket.socket]:
    return socket.socketpair()


def _sample(sequence: int, sensor_type: int = SENSOR_TYPE_ACCELEROMETER) -> ImuSample:
    return ImuSample(
        version=1,
        header_size=HEADER_SIZE,
        sequence=sequence,
        timestamp_ns=sequence * 1_000_000,
        sensor_type=sensor_type,
        accuracy=3,
        values=(float(sequence), float(sequence + 1), float(sequence + 2)),
        values_count=3,
        flags=0,
    )


@pytest.mark.unit
def test_phone_imu_receiver_reads_latest_sample_per_sensor() -> None:
    sender, receiver_sock = _socket_pair()
    receiver = PhoneImuReceiver.from_socket(
        receiver_sock,
        PhoneImuRuntimeConfig(read_timeout_s=0.1),
    )
    sender.sendall(
        pack_imu_sample(_sample(1, SENSOR_TYPE_ACCELEROMETER))
        + pack_imu_sample(_sample(2, SENSOR_TYPE_GYROSCOPE))
        + pack_imu_sample(_sample(3, SENSOR_TYPE_ACCELEROMETER))
    )

    try:
        snapshot = receiver.read_latest()
    finally:
        receiver.release()
        sender.close()

    assert snapshot is not None
    assert snapshot.latest_by_sensor["accelerometer"].sequence == 3
    assert snapshot.latest_by_sensor["gyroscope"].sequence == 2
    assert snapshot.samples_read == 3


@pytest.mark.unit
def test_phone_imu_starts_android_service_after_listener_is_ready(monkeypatch) -> None:
    launches = []

    def fake_start_sensor_service(config, controls):
        launches.append((config, controls))

    monkeypatch.setattr("src.sensors.phone_imu.start_taffcam_sensor_service", fake_start_sensor_service)

    receiver = PhoneImuReceiver(
        PhoneImuRuntimeConfig(
            sensor_port=0,
            requested_rate_hz=200,
            start_android_service=True,
        )
    )

    try:
        assert receiver.is_opened()
    finally:
        receiver.release()

    assert len(launches) == 1
    launch_config, controls = launches[0]
    assert launch_config.app_package == "com.tafftracker.taffcam"
    assert controls["imu_enabled"] is True
    assert controls["sensor_only"] is True
    assert controls["imu_rate_hz"] == 200


@pytest.mark.unit
def test_phone_imu_adb_reverse_timeout_is_not_retried(monkeypatch) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        raise subprocess.TimeoutExpired(cmd, timeout=kwargs["timeout"])

    monkeypatch.setattr("src.sensors.phone_imu.subprocess.run", fake_run)

    receiver = PhoneImuReceiver(
        PhoneImuRuntimeConfig(
            sensor_port=0,
            adb_reverse=True,
            adb_reverse_timeout_s=4.0,
        )
    )

    try:
        receiver._ensure_listening()
    finally:
        receiver.release()

    assert len(calls) == 1
    assert calls[0][0][-3:] == ["reverse", "tcp:0", "tcp:0"]
    assert calls[0][1]["timeout"] == 4.0


@pytest.mark.unit
def test_phone_imu_listener_is_loopback_only_by_default() -> None:
    with pytest.raises(ValueError, match="loopback"):
        PhoneImuReceiver(
            PhoneImuRuntimeConfig(
                sensor_host="0.0.0.0",
                sensor_port=0,
            )
        )
