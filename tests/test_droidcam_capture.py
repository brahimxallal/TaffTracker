from __future__ import annotations

import socket
import struct

import numpy as np
import pytest

from src.capture.droidcam import (
    _NO_PTS,
    DroidCamDirectCaptureSource,
    DroidCamRemoteControl,
    DroidCamRuntimeConfig,
    _build_video_request,
    _normalize_nal_payload,
)
from src.config import DroidCamConfig


class _FakeDecoder:
    def __init__(self) -> None:
        self.calls: list[tuple[int, bytes, bool]] = []

    def decode(self, *, pts: int, payload: bytes, is_config: bool):
        self.calls.append((pts, payload, is_config))
        if is_config:
            return []
        value = payload[0] if payload else 0
        return [np.full((2, 3, 3), value, dtype=np.uint8)]

    def close(self) -> None:
        return None


@pytest.mark.unit
def test_droidcam_video_request_uses_v5_encoded_stream_path() -> None:
    request = _build_video_request(
        DroidCamRuntimeConfig(
            host="192.168.1.16",
            port=4747,
            width=640,
            height=480,
            video_format="avc",
        )
    )

    assert request.startswith(
        b"GET /v5/video/avc/640x480/port/0/os/win"
    ) or request.startswith(
        b"GET /v5/video/avc/640x480/port/0/os/linux"
    )
    assert b"/obs/7.0.0/client/243/nonce/5912/" in request
    assert b"HTTP/1.1" not in request


@pytest.mark.unit
def test_droidcam_mjpg_alias_uses_official_jpg_path() -> None:
    request = _build_video_request(
        DroidCamRuntimeConfig(width=640, height=480, video_format="mjpg")
    )

    assert b"GET /v5/video/jpg/640x480/port/0/" in request


@pytest.mark.unit
def test_droidcam_localhost_request_reports_adb_forward_port() -> None:
    request = _build_video_request(
        DroidCamRuntimeConfig(host="127.0.0.1", port=4747, width=640, height=480)
    )

    assert b"/port/4747/" in request

@pytest.mark.unit
def test_normalize_nal_payload_prefixes_single_raw_nal() -> None:
    payload = b"\x65abc"

    normalized = _normalize_nal_payload(payload)

    assert normalized == struct.pack(">I", len(payload)) + payload


@pytest.mark.unit
def test_direct_capture_decodes_latest_droidcam_packet() -> None:
    left, right = socket.socketpair()
    decoder = _FakeDecoder()
    try:
        packet_a = struct.pack(">QI", 1, 1) + b"\x01"
        packet_b = struct.pack(">QI", 2, 1) + b"\x02"
        right.sendall(packet_a + packet_b)
        source = DroidCamDirectCaptureSource.from_socket(
            left,
            DroidCamRuntimeConfig(read_timeout_s=0.001),
            decoder=decoder,
        )

        ok, frame = source.read()

        assert ok is True
        assert frame is not None
        assert frame.shape == (2, 3, 3)
        assert int(frame[0, 0, 0]) == 2
        assert decoder.calls == [(1, b"\x01", False), (2, b"\x02", False)]
        source.release()
    finally:
        right.close()


@pytest.mark.unit
def test_direct_capture_handles_official_config_packet_marker() -> None:
    left, right = socket.socketpair()
    decoder = _FakeDecoder()
    try:
        config_packet = struct.pack(">QI", _NO_PTS, 3) + b"cfg"
        frame_packet = struct.pack(">QI", 10, 1) + b"\x07"
        right.sendall(config_packet + frame_packet)
        source = DroidCamDirectCaptureSource.from_socket(
            left,
            DroidCamRuntimeConfig(read_timeout_s=0.001),
            decoder=decoder,
        )

        ok, frame = source.read()

        assert ok is True
        assert frame is not None
        assert int(frame[0, 0, 0]) == 7
        assert decoder.calls == [(_NO_PTS, b"cfg", True), (10, b"\x07", False)]
        source.release()
    finally:
        right.close()


@pytest.mark.unit
def test_direct_capture_decodes_jpeg_packet_with_zero_pts() -> None:
    left, right = socket.socketpair()
    decoder = _FakeDecoder()
    try:
        frame_packet = struct.pack(">QI", 0, 1) + b"\x09"
        right.sendall(frame_packet)
        source = DroidCamDirectCaptureSource.from_socket(
            left,
            DroidCamRuntimeConfig(video_format="jpg", read_timeout_s=0.001),
            decoder=decoder,
        )

        ok, frame = source.read()

        assert ok is True
        assert frame is not None
        assert int(frame[0, 0, 0]) == 9
        assert decoder.calls == [(0, b"\x09", False)]
        source.release()
    finally:
        right.close()


@pytest.mark.unit
def test_direct_capture_decodes_avc_packet_with_zero_pts_as_frame() -> None:
    left, right = socket.socketpair()
    decoder = _FakeDecoder()
    try:
        frame_packet = struct.pack(">QI", 0, 1) + b"\x08"
        right.sendall(frame_packet)
        source = DroidCamDirectCaptureSource.from_socket(
            left,
            DroidCamRuntimeConfig(video_format="avc", read_timeout_s=0.001),
            decoder=decoder,
        )

        ok, frame = source.read()

        assert ok is True
        assert frame is not None
        assert int(frame[0, 0, 0]) == 8
        assert decoder.calls == [(0, b"\x08", False)]
        source.release()
    finally:
        right.close()


@pytest.mark.unit
def test_remote_control_applies_only_needed_toggle_commands(monkeypatch) -> None:
    config = DroidCamConfig(
        remote_enabled=True,
        active_camera=0,
        autofocus_once=True,
        autofocus_mode=3,
        manual_focus=12.5,
        exposure_lock=True,
        ev=-1.0,
        wb_mode=5,
        wb_lock=True,
        wb_level=5200,
        torch_enabled=True,
        zoom=1.2,
    )
    control = DroidCamRemoteControl(config)
    puts: list[str] = []

    monkeypatch.setattr(
        control,
        "_get_json",
        lambda path: {
            "active": 1,
            "focusMode": 2,
            "exposure_lock": 0,
            "wbMode": 0,
            "wbLock": 0,
            "led_on": 0,
        },
    )
    monkeypatch.setattr(control, "_put", lambda path: puts.append(path))

    result = control.apply_startup_controls()

    assert result.warnings == ()
    assert result.applied == (
        "active_camera",
        "focus_mode",
        "autofocus_once",
        "manual_focus",
        "wb_mode",
        "wb_lock",
        "wb_level",
        "exposure_lock",
        "exposure_ev",
        "torch",
        "zoom",
    )
    assert puts == [
        "/v1/camera/active/0",
        "/v1/camera/autofocus_mode/3",
        "/v1/camera/autofocus",
        "/v3/camera/mf/12.5",
        "/v1/camera/wb_mode/5",
        "/v1/camera/wbl_toggle",
        "/v2/camera/wb_level/5200",
        "/v1/camera/el_toggle",
        "/v3/camera/ev/-1",
        "/v1/camera/torch_toggle",
        "/v3/camera/zoom/1.2",
    ]


@pytest.mark.unit
def test_remote_control_disabled_is_noop() -> None:
    result = DroidCamRemoteControl(DroidCamConfig(remote_enabled=False)).apply_startup_controls()

    assert result.applied == ()
    assert result.warnings == ()
