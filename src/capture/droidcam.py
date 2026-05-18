from __future__ import annotations

import json
import logging
import platform
import select
import socket
import struct
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

from src.config import DroidCamConfig

LOGGER = logging.getLogger(__name__)

_BACKEND_NAME = "DROIDCAM_DIRECT"
_HEADER_SIZE = 12
_NO_PTS = (1 << 64) - 1
_STOP_OR_ERROR_PAYLOAD_LEN = (1 << 32) - 1
_MAX_FRAME_SIZE = 16 * 1024 * 1024
_MAX_BUFFER_SIZE = 32 * 1024 * 1024
_RECV_CHUNK_BYTES = 256 * 1024
_STATS_LOG_INTERVAL_FRAMES = 120
_DROIDCAM_OBS_VERSION = "7.0.0"
_DROIDCAM_OBS_PLUGIN_VERSION = "243"
_DROIDCAM_NONCE = 5912


@dataclass(frozen=True, slots=True)
class DroidCamRuntimeConfig:
    host: str = "192.168.1.16"
    port: int = 4747
    width: int = 640
    height: int = 480
    fps: float = 60.0
    video_format: str = "avc"
    connect_timeout_s: float = 2.0
    read_timeout_s: float = 0.03
    max_frame_bytes: int = _MAX_FRAME_SIZE
    max_buffer_bytes: int = _MAX_BUFFER_SIZE
    recv_chunk_bytes: int = _RECV_CHUNK_BYTES

    @classmethod
    def from_config(cls, config: DroidCamConfig) -> DroidCamRuntimeConfig:
        return cls(
            host=config.host,
            port=config.port,
            width=config.width,
            height=config.height,
            fps=float(config.fps),
            video_format=config.video_format,
            connect_timeout_s=config.connect_timeout_s,
            read_timeout_s=config.read_timeout_s,
        )


@dataclass(frozen=True)
class DroidCamControlResult:
    applied: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    info: dict[str, object] = field(default_factory=dict)


class DroidCamDecodeAdapter(Protocol):
    def decode(self, *, pts: int, payload: bytes, is_config: bool) -> list[np.ndarray]:
        ...

    def close(self) -> None:
        ...


class PyAvDroidCamDecoder:
    """Decode DroidCam AVC/HEVC packets with PyAV."""

    def __init__(self, video_format: str) -> None:
        import importlib

        try:
            av = importlib.import_module("av")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "droidcam source_backend requires PyAV (`pip install av`) "
                "for avc/hevc streams"
            ) from exc
        codec_name = {"avc": "h264", "hevc": "hevc"}[video_format]
        self._av = av
        self._codec = av.CodecContext.create(codec_name, "r")
        self._pending_config = b""

    def decode(self, *, pts: int, payload: bytes, is_config: bool) -> list[np.ndarray]:
        if is_config:
            self._pending_config = payload
            try:
                self._codec.extradata = payload
            except (AttributeError, ValueError):
                LOGGER.debug("PyAV rejected DroidCam codec config", exc_info=True)
            return []
        if self._pending_config:
            payload = self._pending_config + payload
            self._pending_config = b""
        packet = self._av.Packet(_normalize_nal_payload(payload))
        packet.pts = int(pts)
        return [frame.to_ndarray(format="bgr24") for frame in self._codec.decode(packet)]

    def close(self) -> None:
        close = getattr(self._codec, "close", None)
        if close is not None:
            close()


class JpegDroidCamDecoder:
    """Decode DroidCam MJPG payloads with OpenCV."""

    def decode(self, *, pts: int, payload: bytes, is_config: bool) -> list[np.ndarray]:
        if is_config:
            return []
        encoded = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return [] if frame is None else [frame]

    def close(self) -> None:
        return None


class DroidCamRemoteControl:
    """Tiny client for the DroidCam Classic remote-control HTTP API."""

    def __init__(self, config: DroidCamConfig) -> None:
        self._config = config
        self._base_url = f"http://{config.host}:{config.port}"

    def apply_startup_controls(self) -> DroidCamControlResult:
        if not self._config.remote_enabled:
            return DroidCamControlResult()

        applied: list[str] = []
        warnings: list[str] = []
        info: dict[str, object] = {}

        try:
            info = self._get_json("/v1/camera/info")
        except DroidCamRemoteError as exc:
            return DroidCamControlResult(
                warnings=(f"DroidCam remote controls unavailable: {exc}",)
            )

        def put(path: str, label: str) -> None:
            try:
                self._put(path)
            except DroidCamRemoteError as exc:
                warnings.append(f"{label}: {exc}")
            else:
                applied.append(label)

        if self._config.active_camera is not None:
            current = _coerce_int(info.get("active"))
            if current != self._config.active_camera:
                put(f"/v1/camera/active/{self._config.active_camera}", "active_camera")

        if self._config.autofocus_mode is not None:
            current = _coerce_int(info.get("focusMode"))
            if current != self._config.autofocus_mode:
                put(f"/v1/camera/autofocus_mode/{self._config.autofocus_mode}", "focus_mode")

        if self._config.autofocus_once:
            put("/v1/camera/autofocus", "autofocus_once")

        if self._config.manual_focus is not None:
            put(f"/v3/camera/mf/{self._config.manual_focus:g}", "manual_focus")

        if self._config.wb_mode is not None:
            current = _coerce_int(info.get("wbMode"))
            if current != self._config.wb_mode:
                put(f"/v1/camera/wb_mode/{self._config.wb_mode}", "wb_mode")

        if self._config.wb_lock is not None:
            current = _coerce_bool(info.get("wbLock"))
            if current is not None and current != self._config.wb_lock:
                put("/v1/camera/wbl_toggle", "wb_lock")

        if self._config.wb_level is not None:
            put(f"/v2/camera/wb_level/{self._config.wb_level}", "wb_level")

        if self._config.exposure_lock is not None:
            current = _coerce_bool(info.get("exposure_lock"))
            if current is not None and current != self._config.exposure_lock:
                put("/v1/camera/el_toggle", "exposure_lock")

        if self._config.ev is not None:
            put(f"/v3/camera/ev/{self._config.ev:g}", "exposure_ev")

        if self._config.torch_enabled is not None:
            current = _coerce_bool(info.get("led_on"))
            if current is not None and current != self._config.torch_enabled:
                put("/v1/camera/torch_toggle", "torch")

        if self._config.zoom is not None:
            put(f"/v3/camera/zoom/{self._config.zoom:g}", "zoom")

        return DroidCamControlResult(
            applied=tuple(applied),
            warnings=tuple(warnings),
            info=info,
        )

    def _get_json(self, path: str) -> dict[str, object]:
        status, body = self._request("GET", path)
        if not 200 <= status < 300:
            raise DroidCamRemoteError(f"GET {path} returned HTTP {status}")
        try:
            decoded = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise DroidCamRemoteError(f"GET {path} returned non-JSON") from exc
        if not isinstance(decoded, dict):
            raise DroidCamRemoteError(f"GET {path} returned {type(decoded).__name__}")
        return decoded

    def _put(self, path: str) -> None:
        status, _ = self._request("PUT", path)
        if not 200 <= status < 300:
            raise DroidCamRemoteError(f"PUT {path} returned HTTP {status}")

    def _request(self, method: str, path: str) -> tuple[int, bytes]:
        request = urllib.request.Request(
            self._base_url + path,
            method=method,
            headers={"Connection": "close"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._config.remote_timeout_s) as resp:
                return int(resp.status), resp.read()
        except urllib.error.HTTPError as exc:
            return int(exc.code), exc.read()
        except (OSError, TimeoutError) as exc:
            raise DroidCamRemoteError(str(exc)) from exc


class DroidCamRemoteError(RuntimeError):
    pass


class DroidCamStreamError(OSError):
    """Base class for DroidCam direct-stream connection failures."""


class DroidCamDirectCaptureSource:
    """OpenCV-like capture source for DroidCam Classic's direct TCP video stream."""

    def __init__(
        self,
        config: DroidCamRuntimeConfig | None = None,
        *,
        decoder: DroidCamDecodeAdapter | None = None,
        connector: Callable[[tuple[str, int], float], socket.socket] | None = None,
        connect_now: bool = False,
    ) -> None:
        self._config = config or DroidCamRuntimeConfig()
        self._decoder = decoder or self._build_decoder()
        self._connector = connector or socket.create_connection
        self._sock: socket.socket | None = None
        self._recv_buffer = bytearray()
        self._released = False
        self._last_frame_shape: tuple[int, int] | None = None
        self._stats_start_s: float | None = None
        self._stats_frames = 0
        self._stream_failures = 0
        self._last_failure_log_s = 0.0
        if connect_now:
            self._connect()

    @classmethod
    def from_socket(
        cls,
        sock: socket.socket,
        config: DroidCamRuntimeConfig | None = None,
        *,
        decoder: DroidCamDecodeAdapter,
    ) -> DroidCamDirectCaptureSource:
        source = cls(config, decoder=decoder)
        source._sock = sock
        source._sock.settimeout(source._config.read_timeout_s)
        return source

    def isOpened(self) -> bool:
        return not self._released

    def getBackendName(self) -> str:
        return _BACKEND_NAME

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            if self._last_frame_shape is not None:
                return float(self._last_frame_shape[1])
            return float(self._config.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            if self._last_frame_shape is not None:
                return float(self._last_frame_shape[0])
            return float(self._config.height)
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self._config.fps)
        if prop_id == cv2.CAP_PROP_BUFFERSIZE:
            return 1.0
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        return prop_id in (
            cv2.CAP_PROP_BUFFERSIZE,
            cv2.CAP_PROP_FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT,
            cv2.CAP_PROP_FPS,
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._released:
            return False, None
        try:
            if self._sock is None:
                self._connect()
            frame = self._read_latest_frame()
        except (EOFError, OSError, DroidCamStreamError, ValueError) as exc:
            self._record_stream_failure(exc)
            self._close_socket()
            return False, None
        if frame is None:
            return False, None
        if self._stream_failures:
            LOGGER.info("DroidCam direct stream recovered after %d failures", self._stream_failures)
            self._stream_failures = 0
        return True, frame

    def release(self) -> None:
        self._released = True
        self._close_socket()
        self._decoder.close()

    def _build_decoder(self) -> DroidCamDecodeAdapter:
        video_format = _droidcam_protocol_format(self._config.video_format)
        if video_format == "jpg":
            return JpegDroidCamDecoder()
        if video_format in ("avc", "hevc"):
            return PyAvDroidCamDecoder(video_format)
        raise ValueError(f"unsupported DroidCam format: {self._config.video_format}")

    def _connect(self) -> None:
        sock = self._connector(
            (self._config.host, self._config.port),
            self._config.connect_timeout_s,
        )
        try:
            sock.settimeout(self._config.read_timeout_s)
            sock.sendall(_build_video_request(self._config))
        except Exception:
            sock.close()
            raise
        self._sock = sock
        self._recv_buffer.clear()
        LOGGER.info(
            "DroidCam direct stream connected to %s:%d format=%s size=%dx%d",
            self._config.host,
            self._config.port,
            self._config.video_format,
            self._config.width,
            self._config.height,
        )

    def _read_latest_frame(self) -> np.ndarray | None:
        latest_frame: np.ndarray | None = None
        while latest_frame is None:
            decoded = self._decode_available_frames()
            if decoded is not None:
                latest_frame = decoded
                break
            if not self._recv_into_buffer(timeout_s=self._config.read_timeout_s):
                return None

        while self._recv_into_buffer(timeout_s=0.0):
            newer = self._decode_available_frames()
            if newer is not None:
                latest_frame = newer

        newer = self._decode_available_frames()
        if newer is not None:
            latest_frame = newer
        return latest_frame

    def _decode_available_frames(self) -> np.ndarray | None:
        latest: np.ndarray | None = None
        while True:
            item = self._pop_one_packet()
            if item is None:
                return latest
            pts, payload = item
            frames = self._decoder.decode(
                pts=pts,
                payload=payload,
                is_config=(pts == _NO_PTS),
            )
            if not frames:
                continue
            latest = _coerce_bgr(frames[-1])
            self._last_frame_shape = latest.shape[:2]
            self._record_stats()

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
            data = self._sock.recv(self._config.recv_chunk_bytes)
        except (TimeoutError, BlockingIOError):
            return False
        if not data:
            raise EOFError("DroidCam socket closed")
        self._recv_buffer.extend(data)
        if len(self._recv_buffer) > self._config.max_buffer_bytes:
            raise ValueError(f"DroidCam receive buffer too large: {len(self._recv_buffer)}")
        return True

    def _pop_one_packet(self) -> tuple[int, bytes] | None:
        if len(self._recv_buffer) < _HEADER_SIZE:
            return None
        pts, payload_len = struct.unpack_from(">QI", self._recv_buffer, 0)
        if payload_len == _STOP_OR_ERROR_PAYLOAD_LEN:
            raise EOFError("DroidCam app signaled stream stop/error")
        if payload_len <= 0:
            raise ValueError("DroidCam sent empty packet")
        if payload_len > self._config.max_frame_bytes:
            raise ValueError(f"DroidCam packet too large: {payload_len}")
        total_len = _HEADER_SIZE + payload_len
        if len(self._recv_buffer) < total_len:
            return None
        payload = bytes(self._recv_buffer[_HEADER_SIZE:total_len])
        del self._recv_buffer[:total_len]
        return pts, payload

    def _record_stats(self) -> None:
        now_s = cv2.getTickCount() / cv2.getTickFrequency()
        if self._stats_start_s is None:
            self._stats_start_s = now_s
            self._stats_frames = 1
            return
        self._stats_frames += 1
        if self._stats_frames < _STATS_LOG_INTERVAL_FRAMES:
            return
        elapsed = max(now_s - self._stats_start_s, 1e-9)
        LOGGER.info("DroidCam direct stream: delivered_fps=%.1f", self._stats_frames / elapsed)
        self._stats_start_s = now_s
        self._stats_frames = 0

    def _record_stream_failure(self, exc: Exception) -> None:
        self._stream_failures += 1
        now_s = time.monotonic()
        if self._stream_failures == 1 or now_s - self._last_failure_log_s >= 2.0:
            LOGGER.warning(
                "DroidCam direct stream failed; reconnecting later: %s "
                "(failures=%d)",
                exc,
                self._stream_failures,
            )
            self._last_failure_log_s = now_s

    def _close_socket(self) -> None:
        sock = self._sock
        self._sock = None
        self._recv_buffer.clear()
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()


def apply_droidcam_startup_controls(config: DroidCamConfig) -> DroidCamControlResult:
    result = DroidCamRemoteControl(config).apply_startup_controls()
    for warning in result.warnings:
        LOGGER.warning("%s", warning)
    if result.applied:
        LOGGER.info("Applied DroidCam startup controls: %s", ", ".join(result.applied))
    return result


def _build_video_request(config: DroidCamRuntimeConfig) -> bytes:
    video_format = _droidcam_protocol_format(config.video_format)
    forwarded_port = _droidcam_forwarded_port(config)
    path = (
        f"/v5/video/{video_format}/{config.width}x{config.height}/port/{forwarded_port}/"
        f"os/{_droidcam_os_name()}/obs/{_DROIDCAM_OBS_VERSION}/"
        f"client/{_DROIDCAM_OBS_PLUGIN_VERSION}/"
        f"nonce/{_DROIDCAM_NONCE}/"
    )
    return f"GET {path}".encode("ascii")


def _droidcam_forwarded_port(config: DroidCamRuntimeConfig) -> int:
    host = config.host.strip().lower()
    if host in {"127.0.0.1", "localhost", "::1"}:
        return int(config.port)
    return 0


def _droidcam_os_name() -> str:
    if platform.system().lower() != "windows":
        return platform.system().lower() or "linux"
    version = platform.version() or "10.0.0"
    return f"win{version}"


def _droidcam_protocol_format(video_format: str) -> str:
    normalized = video_format.lower()
    if normalized in ("mjpg", "mjpeg"):
        return "jpg"
    if normalized in ("avc", "jpg", "hevc"):
        return normalized
    raise ValueError(f"unsupported DroidCam format: {video_format}")


def _normalize_nal_payload(data: bytes) -> bytes:
    if _has_start_code(data) or _looks_length_prefixed(data):
        return data
    return struct.pack(">I", len(data)) + data


def _has_start_code(data: bytes) -> bool:
    return data.startswith(b"\x00\x00\x00\x01") or data.startswith(b"\x00\x00\x01")


def _looks_length_prefixed(data: bytes) -> bool:
    if len(data) < 5 or _has_start_code(data):
        return False
    nal_len = struct.unpack(">I", data[:4])[0]
    return 0 < nal_len <= len(data) - 4


def _coerce_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"DroidCam decoder returned invalid frame {frame.shape} {frame.dtype}")
    if not frame.flags.c_contiguous:
        return np.ascontiguousarray(frame)
    return frame


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: object) -> bool | None:
    number = _coerce_int(value)
    if number is None or number < 0:
        return None
    return bool(number)
