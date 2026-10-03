from __future__ import annotations

from src.config import PhoneImuConfig
from src.sensors.process import build_phone_imu_runtime_config


def test_build_phone_imu_runtime_config_maps_project_config() -> None:
    runtime = build_phone_imu_runtime_config(
        PhoneImuConfig(
            bind_host="127.0.0.1",
            sensor_port=28003,
            requested_rate_hz=250,
            adb_reverse_enabled=True,
            adb_path="adb.exe",
            adb_serial="DEVICE",
            adb_reverse_timeout_s=3.5,
            allow_remote_clients=False,
            start_service_on_open=True,
            force_stop_app_on_open=True,
            read_timeout_s=0.05,
            app_start_delay_s=0.1,
        )
    )

    assert runtime.sensor_host == "127.0.0.1"
    assert runtime.sensor_port == 28003
    assert runtime.requested_rate_hz == 250
    assert runtime.adb_reverse is True
    assert runtime.adb_path == "adb.exe"
    assert runtime.adb_serial == "DEVICE"
    assert runtime.adb_reverse_timeout_s == 3.5
    assert runtime.allow_remote_clients is False
    assert runtime.start_android_service is True
    assert runtime.force_stop_app is True
    assert runtime.read_timeout_s == 0.05
    assert runtime.app_start_delay_s == 0.1
