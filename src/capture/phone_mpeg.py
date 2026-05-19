from __future__ import annotations

import importlib
import ipaddress
import json
import logging
import select
import socket
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import cv2
import numpy as np

from src.capture.encoded_protocol import (
    FLAG_CODEC_CONFIG,
    FLAG_KEYFRAME,
    HEADER_SIZE,
    LENGTH_PREFIX_SIZE,
    LENGTH_PREFIX_STRUCT,
    EncodedAccessUnitHeader,
    codec_name_to_id,
    pack_access_unit_packet,
    parse_access_unit_header,
)
from src.capture.taffcam_adb import TaffCamLaunchConfig, start_taffcam_app

LOGGER = logging.getLogger(__name__)

_BACKEND_NAME = "TAFF_PHONE_MPEG"
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
class PhoneMpegRuntimeConfig:
    frame_host: str = "127.0.0.1"
    frame_port: int = 27183
    control_host: str = "127.0.0.1"
    control_port: int = 27184
    requested_width: int = 0
    requested_height: int = 0
    requested_fps: float = 0.0
    codec: str = "h264"
    bitrate_bps: int = 8_000_000
    keyframe_interval_s: float = 1.0
    decode_backend: str = "pyav"
    accept_timeout_s: float = 0.02
    read_timeout_s: float = 0.03
    control_timeout_s: float = 0.05
    listen_backlog: int = 1
    recv_chunk_bytes: int = 256 * 1024
    max_payload_bytes: int = 16 * 1024 * 1024
    max_buffer_bytes: int = 32 * 1024 * 1024
    max_decode_backlog_packets: int = 8
    adb_reverse: bool = False
    adb_path: str = "adb"
    adb_serial: str | None = None
    adb_reverse_timeout_s: float = 4.0
    allow_remote_clients: bool = False
    remove_adb_reverse_on_release: bool = False
    start_phone_app: bool = False
    force_stop_app: bool = False
    app_package: str = "com.tafftracker.taffcam"
    app_activity: str = ".MainActivity"
    app_receiver: str = ".TaffCommandReceiver"
    app_start_action: str = "com.tafftracker.taffcam.START"
    app_start_delay_s: float = 1.0
    startup_controls: Mapping[str, Any] = field(default_factory=dict)


class EncodedVideoDecoder(Protocol):
    def decode(
        self,
        header: EncodedAccessUnitHeader,
        payload: bytes,
    ) -> np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...] | None:
        ...

    def close(self) -> None:
        ...


class PyAvVideoDecoder:
    """Small PyAV adapter kept injectable so tests do not need PyAV installed."""

    def __init__(self, codec: str) -> None:
        try:
            self._av = importlib.import_module("av")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "phone_mpeg source_backend requires PyAV (`pip install av`) "
                "unless a decoder is injected"
            ) from exc
        self._codec = self._av.CodecContext.create(codec, "r")
        self._codec_config: bytes | None = None

    def decode(
        self,
        header: EncodedAccessUnitHeader,
        payload: bytes,
    ) -> list[np.ndarray]:
        if (header.flags & FLAG_CODEC_CONFIG) != 0:
            self._codec_config = bytes(payload)
            try:
                self._codec.extradata = self._codec_config
            except (AttributeError, ValueError):
                LOGGER.debug("PyAV decoder rejected codec config extradata", exc_info=True)
            return []

        packet = self._av.Packet(payload)
        packet.pts = int(header.timestamp_ns)
        return [frame.to_ndarray(format="bgr24") for frame in self._codec.decode(packet)]

    def close(self) -> None:
        close = getattr(self._codec, "close", None)
        if close is not None:
            close()


class PhoneMpegCaptureSource:
    """OpenCV-like capture source for the TaffCam encoded phone stream."""

    def __init__(
        self,
        config: PhoneMpegRuntimeConfig | None = None,
        *,
        decoder: EncodedVideoDecoder | None = None,
        start_listening: bool = True,
    ) -> None:
        self._config = config or PhoneMpegRuntimeConfig()
        self._expected_codec_id = codec_name_to_id(self._config.codec)
        self._decoder = decoder or PyAvVideoDecoder(self._config.codec)
        self._frame_listener: socket.socket | None = None
        self._control_listener: socket.socket | None = None
        self._frame_sock: socket.socket | None = None
        self._control_sock: socket.socket | None = None
        self._recv_buffer = bytearray()
        self._released = False
        self._adb_reverse_started = False
        self._adb_reverse_attempted = False
        self._phone_app_started = False
        self._startup_controls_sent = False
        self._last_header: EncodedAccessUnitHeader | None = None
        self._last_frame_shape: tuple[int, int] | None = None
        self._stats_window_start_host_s: float | None = None
        self._stats_window_start_android_ns: int | None = None
        self._stats_window_start_sequence: int | None = None
        self._stats_frames = 0
        self._codec_mismatch_skips = 0
        self._catchup_drops = 0
        if start_listening:
            self._ensure_listening()

    @classmethod
    def from_socket(
        cls,
        sock: socket.socket,
        config: PhoneMpegRuntimeConfig | None = None,
        *,
        decoder: EncodedVideoDecoder,
    ) -> PhoneMpegCaptureSource:
        source = cls(config, decoder=decoder, start_listening=False)
        source._frame_sock = sock
        source._frame_sock.settimeout(source._config.read_timeout_s)
        return source

    @property
    def last_header(self) -> EncodedAccessUnitHeader | None:
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
            LOGGER.warning("Phone MPEG stream failed; waiting for reconnect: %s", exc)
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
        self._decoder.close()
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
        LOGGER.info("Phone MPEG frame stream connected from %s", addr)
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
        LOGGER.info("Phone MPEG control stream connected from %s", addr)
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
            LOGGER.warning("Failed to send phone MPEG startup controls: %s", exc)
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
        controls.setdefault("stream_format", "mpeg")
        controls.setdefault("codec", self._config.codec)
        controls.setdefault("bitrate_bps", self._config.bitrate_bps)
        controls.setdefault("keyframe_interval_s", self._config.keyframe_interval_s)
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
                "codec": controls.get("codec"),
                "bitrate_bps": controls.get("bitrate_bps"),
                "keyframe_interval_s": controls.get("keyframe_interval_s"),
                "focus_diopters": controls.get("focus_diopters"),
                "exposure_ns": controls.get("exposure_ns"),
                "iso": controls.get("iso"),
                "awb_enabled": controls.get("awb_enabled"),
                "awb_lock": controls.get("awb_lock"),
                "torch_enabled": controls.get("torch_enabled"),
                "zoom_ratio": controls.get("zoom_ratio"),
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
        latest_frame: np.ndarray | None = None
        deadline = time.monotonic() + self._config.read_timeout_s

        while latest_frame is None:
            latest_frame = self._decode_available_access_units()
            if latest_frame is not None:
                break

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None
            if not self._recv_into_buffer(timeout_s=remaining):
                return None

        while self._recv_into_buffer(timeout_s=0.0):
            newer = self._decode_available_access_units()
            if newer is not None:
                latest_frame = newer

        newer = self._decode_available_access_units()
        if newer is not None:
            latest_frame = newer

        return latest_frame

    def _decode_available_access_units(self) -> np.ndarray | None:
        latest_frame: np.ndarray | None = None
        access_units: list[tuple[EncodedAccessUnitHeader, bytes]] = []
        while True:
            item = self._pop_one_complete_access_unit()
            if item is None:
                break
            access_units.append(item)

        if not access_units:
            return latest_frame

        for header, payload in self._trim_decode_backlog(access_units):
            if header.codec_id != self._expected_codec_id:
                self._codec_mismatch_skips += 1
                if self._codec_mismatch_skips == 1 or self._codec_mismatch_skips % 120 == 0:
                    LOGGER.warning(
                        "Skipping phone video codec mismatch: stream=%s configured=%s skipped=%d",
                        header.codec_name,
                        self._config.codec,
                        self._codec_mismatch_skips,
                    )
                continue
            decoded = self._decoder.decode(header, payload)
            frames = self._normalize_decoded_frames(decoded)
            if not frames:
                continue
            latest_frame = self._coerce_bgr_frame(frames[-1])
            self._last_header = header
            self._last_frame_shape = latest_frame.shape[:2]
            self._record_stream_stats(header)

        return latest_frame

    def _trim_decode_backlog(
        self,
        access_units: list[tuple[EncodedAccessUnitHeader, bytes]],
    ) -> list[tuple[EncodedAccessUnitHeader, bytes]]:
        limit = self._config.max_decode_backlog_packets
        if limit <= 0 or len(access_units) <= limit:
            return access_units

        if self._config.codec.lower() not in {"h264", "hevc"}:
            return access_units

        config_index: int | None = None
        keyframe_index: int | None = None
        for index, (header, _) in enumerate(access_units):
            if header.codec_id != self._expected_codec_id:
                continue
            if header.flags & FLAG_CODEC_CONFIG:
                config_index = index
            elif header.flags & FLAG_KEYFRAME:
                keyframe_index = index

        if keyframe_index is None:
            return access_units

        trimmed: list[tuple[EncodedAccessUnitHeader, bytes]] = []
        if config_index is not None and config_index < keyframe_index:
            trimmed.append(access_units[config_index])
        trimmed.extend(access_units[keyframe_index:])

        dropped = len(access_units) - len(trimmed)
        self._catchup_drops += dropped
        if dropped > 0 and (self._catchup_drops == dropped or self._catchup_drops % 120 == 0):
            LOGGER.warning(
                "Phone MPEG decoder catch-up: dropped %d stale access units "
                "(total=%d, resume_seq=%d)",
                dropped,
                self._catchup_drops,
                access_units[keyframe_index][0].sequence,
            )
        return trimmed

    def _normalize_decoded_frames(
        self,
        decoded: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...] | None,
    ) -> list[np.ndarray]:
        if decoded is None:
            return []
        if isinstance(decoded, np.ndarray):
            return [decoded]
        return list(decoded)

    def _coerce_bgr_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "phone MPEG decoder returned non-BGR uint8 frame: "
                f"shape={frame.shape} dtype={frame.dtype}"
            )
        if not frame.flags.c_contiguous:
            return np.ascontiguousarray(frame)
        return frame

    def _record_stream_stats(self, header: EncodedAccessUnitHeader) -> None:
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
            "Phone MPEG stream: delivered_fps=%.1f source_fps=%.1f skipped=%d "
            "seq=%d size=%dx%d codec=%s",
            delivered_fps,
            source_fps,
            skipped_frames,
            header.sequence,
            header.width,
            header.height,
            header.codec_name,
        )
        self._reset_stream_stats(header, now_s)

    def _reset_stream_stats(self, header: EncodedAccessUnitHeader, now_s: float) -> None:
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
            raise EOFError("phone MPEG frame socket closed")
        self._recv_buffer.extend(data)
        if len(self._recv_buffer) > self._config.max_buffer_bytes:
            raise ValueError(f"phone MPEG receive buffer too large: {len(self._recv_buffer)}")
        return True

    def _pop_one_complete_access_unit(
        self,
    ) -> tuple[EncodedAccessUnitHeader, bytes] | None:
        if len(self._recv_buffer) < LENGTH_PREFIX_SIZE + HEADER_SIZE:
            return None

        packet_len = LENGTH_PREFIX_STRUCT.unpack_from(self._recv_buffer)[0]
        if packet_len < HEADER_SIZE:
            raise ValueError(f"encoded phone packet too small: {packet_len}")
        if packet_len > self._config.max_buffer_bytes - LENGTH_PREFIX_SIZE:
            raise ValueError(f"phone MPEG packet too large: {packet_len}")

        total_len = LENGTH_PREFIX_SIZE + packet_len
        if len(self._recv_buffer) < total_len:
            return None

        header_start = LENGTH_PREFIX_SIZE
        header = parse_access_unit_header(
            bytes(self._recv_buffer[header_start : header_start + HEADER_SIZE])
        )
        if header.header_size > packet_len:
            raise ValueError(f"encoded phone header larger than packet: {header.header_size}")
        payload_len = packet_len - header.header_size
        if header.payload_len != payload_len:
            raise ValueError(
                f"encoded phone payload length mismatch: {header.payload_len} != {payload_len}"
            )
        if payload_len > self._config.max_payload_bytes:
            raise ValueError(f"phone MPEG payload too large: {payload_len}")

        payload_start = LENGTH_PREFIX_SIZE + header.header_size
        payload_end = total_len
        payload = bytes(self._recv_buffer[payload_start:payload_end])
        del self._recv_buffer[:total_len]
        return header, payload

    def _start_adb_reverse(self) -> None:
        if self._adb_reverse_attempted:
            return
        self._adb_reverse_attempted = True
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
                    timeout=self._config.adb_reverse_timeout_s,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                LOGGER.warning("Failed to run adb reverse for phone MPEG port %d: %s", port, exc)
                return
        self._adb_reverse_started = True

    def _start_phone_app_once(self) -> None:
        if not self._config.start_phone_app or self._phone_app_started:
            return
        self._phone_app_started = True
        LOGGER.info("Starting TaffCam Android app for phone MPEG stream")
        start_taffcam_app(
            TaffCamLaunchConfig(
                adb_path=self._config.adb_path,
                adb_serial=self._config.adb_serial,
                app_package=self._config.app_package,
                app_activity=self._config.app_activity,
                app_receiver=self._config.app_receiver,
                start_action=self._config.app_start_action,
                start_delay_s=self._config.app_start_delay_s,
                force_stop_before_start=self._config.force_stop_app,
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
                LOGGER.debug("Failed to remove adb reverse for phone MPEG port %d", port)

    def _close_frame_socket(self) -> None:
        self._close_socket_attr("_frame_sock")
        self._recv_buffer.clear()
        self._startup_controls_sent = False
        self._close_control_socket()

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


def make_access_unit_packet(header: EncodedAccessUnitHeader, payload: bytes) -> bytes:
    return pack_access_unit_packet(header, payload)
