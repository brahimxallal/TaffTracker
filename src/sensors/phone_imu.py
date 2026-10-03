from __future__ import annotations

import ipaddress
import logging
import select
import socket
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from src.capture.taffcam_adb import TaffCamLaunchConfig, start_taffcam_sensor_service
from src.sensors.phone_imu_protocol import HEADER_SIZE, ImuSample, parse_imu_sample

LOGGER = logging.getLogger(__name__)


def _validate_bind_host(host: str, *, allow_remote_clients: bool) -> None:
    if allow_remote_clients:
        return
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Invalid phone IMU bind host {host!r}") from exc
    if not infos:
        raise ValueError(f"Invalid phone IMU bind host {host!r}")
    for info in infos:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise ValueError(f"Invalid phone IMU bind host {host!r}") from exc
        if not ip.is_loopback:
            raise ValueError(
                "Phone IMU listeners bind to loopback by default; set "
                "allow_remote_clients=True only for an explicitly trusted lab network"
            )


@dataclass(frozen=True, slots=True)
class PhoneImuRuntimeConfig:
    sensor_host: str = "127.0.0.1"
    sensor_port: int = 27185
    requested_rate_hz: int = 200
    accept_timeout_s: float = 0.02
    read_timeout_s: float = 0.03
    listen_backlog: int = 1
    max_buffer_bytes: int = 512 * 1024
    adb_reverse: bool = False
    adb_path: str = "adb"
    adb_serial: str | None = None
    adb_reverse_timeout_s: float = 4.0
    allow_remote_clients: bool = False
    remove_adb_reverse_on_release: bool = False
    start_android_service: bool = False
    force_stop_app: bool = False
    app_package: str = "com.tafftracker.taffcam"
    app_activity: str = ".MainActivity"
    app_receiver: str = ".TaffCommandReceiver"
    app_service: str = ".TaffSensorService"
    app_start_action: str = "com.tafftracker.taffcam.START_SENSORS"
    app_start_delay_s: float = 0.0
    startup_controls: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PhoneImuSnapshot:
    latest_by_sensor: dict[str, ImuSample]
    samples_read: int
    host_received_ns: int


class PhoneImuReceiver:
    """Loopback TCP receiver for timestamped TaffCam motion-sensor samples."""

    def __init__(
        self,
        config: PhoneImuRuntimeConfig | None = None,
        *,
        start_listening: bool = True,
    ) -> None:
        self._config = config or PhoneImuRuntimeConfig()
        self._listener: socket.socket | None = None
        self._sock: socket.socket | None = None
        self._recv_buffer = bytearray()
        self._released = False
        self._adb_reverse_started = False
        self._adb_reverse_attempted = False
        self._android_service_started = False
        self._latest_by_sensor: dict[str, ImuSample] = {}
        self._samples_read = 0
        if start_listening:
            self._ensure_listening()

    @classmethod
    def from_socket(
        cls,
        sock: socket.socket,
        config: PhoneImuRuntimeConfig | None = None,
    ) -> PhoneImuReceiver:
        receiver = cls(config, start_listening=False)
        receiver._sock = sock
        receiver._sock.settimeout(receiver._config.read_timeout_s)
        return receiver

    def is_opened(self) -> bool:
        if self._released:
            return False
        return self._sock is not None or self._listener is not None

    def release(self) -> None:
        self._released = True
        self._close_socket_attr("_sock")
        self._close_socket_attr("_listener")
        if self._adb_reverse_started and self._config.remove_adb_reverse_on_release:
            self._remove_adb_reverse()

    def read_latest(self) -> PhoneImuSnapshot | None:
        if self._released:
            return None
        if not self._ensure_sensor_socket():
            return None
        try:
            samples = self._read_available_samples()
        except (EOFError, OSError, ValueError) as exc:
            LOGGER.warning("Phone IMU stream failed; waiting for reconnect: %s", exc)
            self._close_socket_attr("_sock")
            self._recv_buffer.clear()
            return None
        if samples == 0:
            return None
        return PhoneImuSnapshot(
            latest_by_sensor=dict(self._latest_by_sensor),
            samples_read=samples,
            host_received_ns=time.perf_counter_ns(),
        )

    def _ensure_listening(self) -> None:
        if self._released:
            return
        if self._listener is None:
            self._listener = self._open_listener(self._config.sensor_host, self._config.sensor_port)
        if self._config.adb_reverse and not self._adb_reverse_started:
            self._start_adb_reverse()
        self._start_android_service_once()

    def _open_listener(self, host: str, port: int) -> socket.socket:
        _validate_bind_host(host, allow_remote_clients=self._config.allow_remote_clients)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, port))
            listener.listen(self._config.listen_backlog)
            listener.settimeout(self._config.accept_timeout_s)
        except Exception:
            listener.close()
            raise
        return listener

    def _ensure_sensor_socket(self) -> bool:
        if self._sock is not None:
            return True
        self._ensure_listening()
        if self._listener is None:
            return False
        try:
            sock, addr = self._listener.accept()
        except TimeoutError:
            return False
        except BlockingIOError:
            return False
        sock.settimeout(self._config.read_timeout_s)
        self._sock = sock
        self._recv_buffer.clear()
        LOGGER.info("Phone IMU stream connected from %s", addr)
        return True

    def _read_available_samples(self) -> int:
        deadline = time.monotonic() + self._config.read_timeout_s
        samples = 0
        while samples == 0:
            samples += self._pop_complete_samples()
            if samples:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return 0
            if not self._recv_into_buffer(timeout_s=remaining):
                return 0

        while self._recv_into_buffer(timeout_s=0.0):
            samples += self._pop_complete_samples()
        samples += self._pop_complete_samples()
        return samples

    def _recv_into_buffer(self, *, timeout_s: float) -> bool:
        if self._sock is None:
            return False
        if timeout_s <= 0.0:
            readable, _, _ = select.select([self._sock], [], [], 0.0)
            if not readable:
                return False
            self._sock.settimeout(0.0)
        else:
            self._sock.settimeout(timeout_s)
        try:
            data = self._sock.recv(4096)
        except (TimeoutError, BlockingIOError):
            return False
        if not data:
            raise EOFError("phone IMU socket closed")
        self._recv_buffer.extend(data)
        if len(self._recv_buffer) > self._config.max_buffer_bytes:
            raise ValueError(f"phone IMU receive buffer too large: {len(self._recv_buffer)}")
        return True

    def _pop_complete_samples(self) -> int:
        samples = 0
        while len(self._recv_buffer) >= HEADER_SIZE:
            sample = parse_imu_sample(bytes(self._recv_buffer[:HEADER_SIZE]))
            del self._recv_buffer[:HEADER_SIZE]
            self._latest_by_sensor[sample.sensor_name] = sample
            self._samples_read += 1
            samples += 1
        return samples

    def _start_adb_reverse(self) -> None:
        if self._adb_reverse_attempted:
            return
        self._adb_reverse_attempted = True
        cmd = [self._config.adb_path]
        if self._config.adb_serial:
            cmd.extend(["-s", self._config.adb_serial])
        port = self._config.sensor_port
        cmd.extend(["reverse", f"tcp:{port}", f"tcp:{port}"])
        try:
            subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                timeout=self._config.adb_reverse_timeout_s,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            LOGGER.warning("Failed to run adb reverse for phone IMU port %d: %s", port, exc)
            return
        self._adb_reverse_started = True

    def _start_android_service_once(self) -> None:
        if not self._config.start_android_service or self._android_service_started:
            return
        self._android_service_started = True
        controls = dict(self._config.startup_controls)
        controls.setdefault("imu_enabled", True)
        controls.setdefault("sensor_only", True)
        controls.setdefault("imu_rate_hz", self._config.requested_rate_hz)
        start_taffcam_sensor_service(
            TaffCamLaunchConfig(
                adb_path=self._config.adb_path,
                adb_serial=self._config.adb_serial,
                app_package=self._config.app_package,
                app_activity=self._config.app_activity,
                app_receiver=self._config.app_receiver,
                app_service=self._config.app_service,
                start_action=self._config.app_start_action,
                start_delay_s=self._config.app_start_delay_s,
                force_stop_before_start=self._config.force_stop_app,
            ),
            controls,
        )

    def _remove_adb_reverse(self) -> None:
        cmd = [self._config.adb_path]
        if self._config.adb_serial:
            cmd.extend(["-s", self._config.adb_serial])
        port = self._config.sensor_port
        cmd.extend(["reverse", "--remove", f"tcp:{port}"])
        try:
            subprocess.run(cmd, check=False, capture_output=True, timeout=2.0)
        except (OSError, subprocess.SubprocessError):
            LOGGER.debug("Failed to remove adb reverse for phone IMU port %d", port)

    def _close_socket_attr(self, attr: str) -> None:
        sock = getattr(self, attr)
        if sock is None:
            return
        setattr(self, attr, None)
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()
