from __future__ import annotations

import ipaddress
import json
import logging
import select
import socket
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from src.capture.phone_protocol import (
    HEADER_SIZE,
    PIXEL_FORMAT_I420,
    PIXEL_FORMAT_NV12,
    PIXEL_FORMAT_NV21,
    FrameHeader,
    pack_frame_header,
    parse_frame_header,
)
from src.capture.taffcam_adb import TaffCamLaunchConfig, start_taffcam_app

LOGGER = logging.getLogger(__name__)

_BACKEND_NAME = "TAFF_PHONE_YUV"
_STATS_LOG_INTERVAL_FRAMES = 120


def _validate_bind_host(host: str, *, allow_remote_clients: bool) -> None:
    if allow_remote_clients:
        return
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Invalid phone camera bind host {host!r}") from exc
    if not infos:
        raise ValueError(f"Invalid phone camera bind host {host!r}")
    for info in infos:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise ValueError(f"Invalid phone camera bind host {host!r}") from exc
        if not ip.is_loopback:
            raise ValueError(
                "Phone camera listeners bind to loopback by default; set "
                "allow_remote_clients=True only for an explicitly trusted lab network"
            )


@dataclass(frozen=True, slots=True)
class PhoneCameraRuntimeConfig:
    frame_host: str = "127.0.0.1"
    frame_port: int = 27183
    control_host: str = "127.0.0.1"
    control_port: int = 27184
    requested_width: int = 0
    requested_height: int = 0
    requested_fps: float = 0.0
    accept_timeout_s: float = 0.02
    read_timeout_s: float = 0.03
    control_timeout_s: float = 0.05
    listen_backlog: int = 1
    recv_chunk_bytes: int = 1024 * 1024
    max_payload_bytes: int = 64 * 1024 * 1024
    max_buffer_bytes: int = 96 * 1024 * 1024
    adb_reverse: bool = False
    adb_path: str = "adb"
    adb_serial: str | None = None
    allow_remote_clients: bool = False
    remove_adb_reverse_on_release: bool = False
    start_phone_app: bool = False
    app_package: str = "com.tafftracker.taffcam"
    app_activity: str = ".MainActivity"
    app_receiver: str = ".TaffCommandReceiver"
    app_start_action: str = "com.tafftracker.taffcam.START"
    app_start_delay_s: float = 1.0
    startup_controls: Mapping[str, Any] = field(default_factory=dict)


class PhoneYuvCaptureSource:
    """OpenCV-like capture source for the TaffCam raw phone YUV stream."""

    def __init__(
        self,
        config: PhoneCameraRuntimeConfig | None = None,
        *,
        start_listening: bool = True,
    ) -> None:
        self._config = config or PhoneCameraRuntimeConfig()
        self._frame_listener: socket.socket | None = None
        self._control_listener: socket.socket | None = None
        self._frame_sock: socket.socket | None = None
        self._control_sock: socket.socket | None = None
        self._recv_buffer = bytearray()
        self._released = False
        self._adb_reverse_started = False
        self._phone_app_started = False
        self._startup_controls_sent = False
        self._last_header: FrameHeader | None = None
        self._last_frame_shape: tuple[int, int] | None = None
        self._stats_window_start_host_s: float | None = None
        self._stats_window_start_android_ns: int | None = None
        self._stats_window_start_sequence: int | None = None
        self._stats_frames = 0
        if start_listening:
            self._ensure_listening()

    @classmethod
    def from_socket(
        cls,
        sock: socket.socket,
        config: PhoneCameraRuntimeConfig | None = None,
    ) -> PhoneYuvCaptureSource:
        source = cls(config, start_listening=False)
        source._frame_sock = sock
        source._frame_sock.settimeout(source._config.read_timeout_s)
        return source

    @property
    def last_header(self) -> FrameHeader | None:
        return self._last_header

    def isOpened(self) -> bool:
        if self._released:
            return False
        return self._frame_sock is not None or self._frame_listener is not None

    def getBackendName(self) -> str:
        return _BACKEND_NAME

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            if self._last_header is not None:
                return float(self._last_header.width)
            return float(self._config.requested_width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            if self._last_header is not None:
                return float(self._last_header.height)
            return float(self._config.requested_height)
        if prop == cv2.CAP_PROP_FPS:
            return float(self._config.requested_fps)
        if prop == cv2.CAP_PROP_BUFFERSIZE:
            return 1.0
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._released:
            return False, None
        if not self._ensure_frame_socket():
            return False, None

        try:
            frame = self._read_latest_frame()
        except (EOFError, OSError, ValueError) as exc:
            LOGGER.warning("Phone YUV stream failed; waiting for reconnect: %s", exc)
            self._close_frame_socket()
            return False, None

        if frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        self._released = True
        self._close_frame_socket()
        self._close_control_socket()
        self._close_socket_attr("_frame_listener")
        self._close_socket_attr("_control_listener")
        if self._adb_reverse_started and self._config.remove_adb_reverse_on_release:
            self._remove_adb_reverse()

    def _ensure_listening(self) -> None:
        if self._released:
            return
        if self._frame_listener is None:
            self._frame_listener = self._open_listener(
                self._config.frame_host,
                self._config.frame_port,
            )
        if self._control_listener is None:
            self._control_listener = self._open_listener(
                self._config.control_host,
                self._config.control_port,
            )
        if self._config.adb_reverse and not self._adb_reverse_started:
            self._start_adb_reverse()
        self._start_phone_app_once()

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

    def _ensure_frame_socket(self) -> bool:
        if self._frame_sock is not None:
            self._accept_control_socket()
            return True

        self._ensure_listening()
        self._accept_control_socket()
        if self._frame_listener is None:
            return False

        try:
            sock, addr = self._frame_listener.accept()
        except TimeoutError:
            return False
        except BlockingIOError:
            return False

        sock.settimeout(self._config.read_timeout_s)
        self._frame_sock = sock
        self._recv_buffer.clear()
        LOGGER.info("Phone YUV frame stream connected from %s", addr)
        self._accept_control_socket()
        self._send_startup_controls()
        return True

    def _accept_control_socket(self) -> None:
        if self._control_sock is not None or self._control_listener is None:
            self._send_startup_controls()
            return

        readable, _, _ = select.select([self._control_listener], [], [], 0.0)
        if not readable:
            return

        try:
            sock, addr = self._control_listener.accept()
        except TimeoutError:
            return
        except BlockingIOError:
            return

        sock.settimeout(self._config.control_timeout_s)
        self._control_sock = sock
        self._startup_controls_sent = False
        LOGGER.info("Phone YUV control stream connected from %s", addr)
        self._send_startup_controls()

    def _send_startup_controls(self) -> None:
        if self._startup_controls_sent or self._control_sock is None:
            return

        payload = b"".join(
            (json.dumps(command, separators=(",", ":")) + "\n").encode("utf-8")
            for command in self._startup_commands()
        )
        if not payload:
            self._startup_controls_sent = True
            return

        try:
            self._control_sock.sendall(payload)
        except OSError as exc:
            LOGGER.warning("Failed to send phone YUV startup controls: %s", exc)
            self._close_control_socket()
            return
        self._startup_controls_sent = True

    def _mode_controls(self) -> dict[str, Any]:
        controls = dict(self._config.startup_controls)
        if self._config.requested_width > 0:
            controls.setdefault("width", self._config.requested_width)
        if self._config.requested_height > 0:
            controls.setdefault("height", self._config.requested_height)
        if self._config.requested_fps > 0:
            controls.setdefault("fps", self._config.requested_fps)
        controls.setdefault("stream_format", "yuv")
        return controls

    def _startup_commands(self) -> list[dict[str, Any]]:
        controls = self._mode_controls()
        commands: list[dict[str, Any]] = [
            {
                "cmd": "set_mode",
                "camera_id": controls.get("camera_id"),
                "width": controls.get("width"),
                "height": controls.get("height"),
                "fps": controls.get("fps"),
                "stream_format": controls.get("stream_format"),
                "capture_mode": controls.get("capture_mode"),
            }
        ]
        if controls.get("focus_diopters") is not None:
            commands.append(
                {
                    "cmd": "set_focus",
                    "auto": False,
                    "focus_diopters": controls["focus_diopters"],
                }
            )
        if controls.get("exposure_ns") is not None or controls.get("iso") is not None:
            commands.append(
                {
                    "cmd": "set_exposure",
                    "auto": False,
                    "exposure_ns": controls.get("exposure_ns"),
                    "iso": controls.get("iso"),
                }
            )
        if "awb_enabled" in controls or "white_balance_kelvin" in controls:
            commands.append(
                {
                    "cmd": "set_wb",
                    "auto": bool(controls.get("awb_enabled", False)),
                    "mode": "auto" if controls.get("awb_enabled", False) else "off",
                    "white_balance_kelvin": controls.get("white_balance_kelvin"),
                }
            )
        if "awb_lock" in controls:
            commands.append({"cmd": "set_auto_locks", "awb_lock": controls["awb_lock"]})
        if controls.get("torch_enabled") is not None:
            commands.append({"cmd": "set_torch", "enabled": controls["torch_enabled"]})
        if controls.get("zoom_ratio") is not None:
            commands.append({"cmd": "set_zoom", "ratio": controls["zoom_ratio"]})
        return [
            {key: value for key, value in command.items() if value is not None}
            for command in commands
        ]

    def _read_latest_frame(self) -> np.ndarray | None:
        latest: tuple[FrameHeader, bytes] | None = None
        deadline = time.monotonic() + self._config.read_timeout_s

        while latest is None:
            latest = self._pop_latest_complete_frame()
            if latest is not None:
                break

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None
            if not self._recv_into_buffer(timeout_s=remaining):
                return None

        while self._recv_into_buffer(timeout_s=0.0):
            newer = self._pop_latest_complete_frame()
            if newer is not None:
                latest = newer

        newer = self._pop_latest_complete_frame()
        if newer is not None:
            latest = newer

        header, payload = latest
        frame = self._payload_to_bgr(header, payload)
        self._last_header = header
        self._last_frame_shape = (header.height, header.width)
        self._record_stream_stats(header)
        return frame

    def _record_stream_stats(self, header: FrameHeader) -> None:
        now_s = time.perf_counter()
        if (
            self._stats_window_start_host_s is None
            or self._stats_window_start_android_ns is None
            or self._stats_window_start_sequence is None
        ):
            self._reset_stream_stats(header, now_s)
            return

        self._stats_frames += 1
        if self._stats_frames < _STATS_LOG_INTERVAL_FRAMES:
            return

        delivered_intervals = max(1, self._stats_frames - 1)
        host_elapsed_s = max(now_s - self._stats_window_start_host_s, 1e-9)
        android_elapsed_s = max(
            (header.timestamp_ns - self._stats_window_start_android_ns) / 1_000_000_000.0,
            0.0,
        )
        seq_span = max(0, header.sequence - self._stats_window_start_sequence)
        skipped_frames = max(0, seq_span - delivered_intervals)
        delivered_fps = delivered_intervals / host_elapsed_s
        source_fps = seq_span / android_elapsed_s if android_elapsed_s > 0.0 else 0.0
        LOGGER.info(
            "Phone YUV stream: delivered_fps=%.1f source_fps=%.1f skipped=%d seq=%d size=%dx%d fmt=%d",
            delivered_fps,
            source_fps,
            skipped_frames,
            header.sequence,
            header.width,
            header.height,
            header.pixel_format,
        )
        self._reset_stream_stats(header, now_s)

    def _reset_stream_stats(self, header: FrameHeader, now_s: float) -> None:
        self._stats_window_start_host_s = now_s
        self._stats_window_start_android_ns = header.timestamp_ns
        self._stats_window_start_sequence = header.sequence
        self._stats_frames = 1

    def _recv_into_buffer(self, *, timeout_s: float) -> bool:
        if self._frame_sock is None:
            return False

        if timeout_s <= 0.0:
            readable, _, _ = select.select([self._frame_sock], [], [], 0.0)
            if not readable:
                return False
            self._frame_sock.settimeout(0.0)
        else:
            self._frame_sock.settimeout(timeout_s)

        try:
            data = self._frame_sock.recv(self._config.recv_chunk_bytes)
        except (TimeoutError, BlockingIOError):
            return False

        if not data:
            raise EOFError("phone YUV frame socket closed")
        self._recv_buffer.extend(data)
        if len(self._recv_buffer) > self._config.max_buffer_bytes:
            raise ValueError(f"phone YUV receive buffer too large: {len(self._recv_buffer)}")
        return True

    def _pop_latest_complete_frame(self) -> tuple[FrameHeader, bytes] | None:
        latest: tuple[FrameHeader, bytes] | None = None
        while True:
            frame = self._pop_one_complete_frame()
            if frame is None:
                return latest
            latest = frame

    def _pop_one_complete_frame(self) -> tuple[FrameHeader, bytes] | None:
        if len(self._recv_buffer) < HEADER_SIZE:
            return None

        header = parse_frame_header(bytes(self._recv_buffer[:HEADER_SIZE]))
        if header.payload_len > self._config.max_payload_bytes:
            raise ValueError(f"phone YUV payload too large: {header.payload_len}")

        total_len = header.header_size + header.payload_len
        if len(self._recv_buffer) < total_len:
            return None

        payload_start = header.header_size
        payload_end = total_len
        payload = bytes(self._recv_buffer[payload_start:payload_end])
        del self._recv_buffer[:total_len]
        return header, payload

    def _payload_to_bgr(self, header: FrameHeader, payload: bytes) -> np.ndarray:
        width = int(header.width)
        height = int(header.height)
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid phone YUV frame size: {width}x{height}")
        if width % 2 != 0 or height % 2 != 0:
            raise ValueError(f"phone YUV 4:2:0 frame size must be even: {width}x{height}")

        expected_len = width * height * 3 // 2
        if len(payload) != expected_len:
            raise ValueError(
                f"invalid phone YUV payload length: {len(payload)} != {expected_len}"
            )

        yuv = np.frombuffer(payload, dtype=np.uint8).reshape((height * 3 // 2, width))
        if header.pixel_format == PIXEL_FORMAT_I420:
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        if header.pixel_format == PIXEL_FORMAT_NV12:
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        if header.pixel_format == PIXEL_FORMAT_NV21:
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
        raise ValueError(f"unsupported phone YUV pixel format: {header.pixel_format}")

    def _start_adb_reverse(self) -> None:
        for port in (self._config.frame_port, self._config.control_port):
            cmd = [self._config.adb_path]
            if self._config.adb_serial:
                cmd.extend(["-s", self._config.adb_serial])
            cmd.extend(["reverse", f"tcp:{port}", f"tcp:{port}"])
            try:
                subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    timeout=2.0,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                LOGGER.warning("Failed to run adb reverse for phone YUV port %d: %s", port, exc)
                return
        self._adb_reverse_started = True

    def _start_phone_app_once(self) -> None:
        if not self._config.start_phone_app or self._phone_app_started:
            return
        self._phone_app_started = True
        LOGGER.info("Starting TaffCam Android app for phone YUV stream")
        start_taffcam_app(
            TaffCamLaunchConfig(
                adb_path=self._config.adb_path,
                adb_serial=self._config.adb_serial,
                app_package=self._config.app_package,
                app_activity=self._config.app_activity,
                app_receiver=self._config.app_receiver,
                start_action=self._config.app_start_action,
                start_delay_s=self._config.app_start_delay_s,
            ),
            self._mode_controls(),
        )

    def _remove_adb_reverse(self) -> None:
        for port in (self._config.frame_port, self._config.control_port):
            cmd = [self._config.adb_path]
            if self._config.adb_serial:
                cmd.extend(["-s", self._config.adb_serial])
            cmd.extend(["reverse", "--remove", f"tcp:{port}"])
            try:
                subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    timeout=2.0,
                )
            except (OSError, subprocess.SubprocessError):
                LOGGER.debug("Failed to remove adb reverse for phone YUV port %d", port)

    def _close_frame_socket(self) -> None:
        self._close_socket_attr("_frame_sock")
        self._recv_buffer.clear()

    def _close_control_socket(self) -> None:
        self._startup_controls_sent = False
        self._close_socket_attr("_control_sock")

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


def make_frame_packet(header: FrameHeader, payload: bytes) -> bytes:
    return pack_frame_header(header) + payload
