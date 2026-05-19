from __future__ import annotations

import socket
import subprocess

import numpy as np
import pytest

from src.capture.phone_protocol import (
    HEADER_SIZE,
    PIXEL_FORMAT_I420,
    FrameHeader,
    pack_frame_header,
    parse_frame_header,
)
from src.capture.phone_yuv import PhoneCameraRuntimeConfig, PhoneYuvCaptureSource


def _header(
    *,
    sequence: int = 1,
    width: int = 2,
    height: int = 2,
    payload_len: int = 6,
) -> FrameHeader:
    return FrameHeader(
        version=1,
        header_size=HEADER_SIZE,
        sequence=sequence,
        timestamp_ns=sequence * 1_000,
        width=width,
        height=height,
        pixel_format=PIXEL_FORMAT_I420,
        flags=0,
        exposure_ns=8_000_000,
        iso=100,
        focus_diopters=0.0,
        payload_len=payload_len,
    )


def _i420_payload(y_value: int, *, width: int = 2, height: int = 2) -> bytes:
    y = bytes([y_value]) * (width * height)
    u = bytes([128]) * (width * height // 4)
    v = bytes([128]) * (width * height // 4)
    return y + u + v


def _socket_pair() -> tuple[socket.socket, socket.socket]:
    return socket.socketpair()


@pytest.mark.unit
def test_frame_header_pack_parse_round_trip():
    header = _header(sequence=42)

    parsed = parse_frame_header(pack_frame_header(header))

    assert parsed == header


@pytest.mark.unit
def test_malformed_magic_rejection():
    packet = bytearray(pack_frame_header(_header()))
    packet[:8] = b"BADYUV!!"

    with pytest.raises(ValueError, match="magic"):
        parse_frame_header(bytes(packet))


@pytest.mark.unit
def test_i420_conversion_shape_dtype():
    sender, receiver = _socket_pair()
    source = PhoneYuvCaptureSource.from_socket(
        receiver,
        PhoneCameraRuntimeConfig(read_timeout_s=0.1),
    )
    payload = _i420_payload(90)
    sender.sendall(pack_frame_header(_header(payload_len=len(payload))) + payload)

    try:
        success, frame = source.read()
    finally:
        source.release()
        sender.close()

    assert success
    assert frame is not None
    assert frame.shape == (2, 2, 3)
    assert frame.dtype == np.uint8


@pytest.mark.unit
def test_latest_frame_drain_returns_newest_sequence():
    sender, receiver = _socket_pair()
    source = PhoneYuvCaptureSource.from_socket(
        receiver,
        PhoneCameraRuntimeConfig(read_timeout_s=0.1),
    )
    old_payload = _i420_payload(20)
    new_payload = _i420_payload(210)
    sender.sendall(
        pack_frame_header(_header(sequence=1, payload_len=len(old_payload)))
        + old_payload
        + pack_frame_header(_header(sequence=2, payload_len=len(new_payload)))
        + new_payload
    )

    try:
        success, frame = source.read()
    finally:
        source.release()
        sender.close()

    assert success
    assert frame is not None
    assert source.last_header is not None
    assert source.last_header.sequence == 2
    assert float(frame.mean()) > 150.0


@pytest.mark.unit
def test_startup_controls_are_sent_as_command_sequence():
    source = PhoneYuvCaptureSource(
        PhoneCameraRuntimeConfig(
            requested_width=640,
            requested_height=480,
            requested_fps=60.0,
            startup_controls={
                "capture_mode": "auto",
                "focus_diopters": 0.8,
                "exposure_ns": 4_000_000,
                "iso": 320,
                "awb_enabled": False,
                "awb_lock": True,
                "torch_enabled": True,
                "zoom_ratio": 1.4,
            },
        ),
        start_listening=False,
    )

    try:
        commands = source._startup_commands()
    finally:
        source.release()

    assert [command["cmd"] for command in commands] == [
        "set_mode",
        "set_focus",
        "set_exposure",
        "set_wb",
        "set_auto_locks",
        "set_torch",
        "set_zoom",
    ]
    assert commands[0]["width"] == 640
    assert commands[0]["height"] == 480
    assert commands[0]["fps"] == 60.0
    assert commands[0]["stream_format"] == "yuv"
    assert commands[0]["capture_mode"] == "auto"
    assert commands[1]["focus_diopters"] == 0.8
    assert commands[2]["exposure_ns"] == 4_000_000
    assert commands[5]["enabled"] is True


@pytest.mark.unit
def test_start_phone_app_runs_after_yuv_listeners_are_ready(monkeypatch) -> None:
    launches = []

    def fake_start_taffcam_app(config, controls):
        launches.append((config, controls))

    monkeypatch.setattr("src.capture.phone_yuv.start_taffcam_app", fake_start_taffcam_app)

    source = PhoneYuvCaptureSource(
        PhoneCameraRuntimeConfig(
            frame_port=0,
            control_port=0,
            requested_width=640,
            requested_height=480,
            requested_fps=60.0,
            startup_controls={"capture_mode": "auto"},
            start_phone_app=True,
        ),
    )

    try:
        assert source.isOpened()
    finally:
        source.release()

    assert len(launches) == 1
    launch_config, controls = launches[0]
    assert launch_config.app_package == "com.tafftracker.taffcam"
    assert controls["width"] == 640
    assert controls["height"] == 480
    assert controls["fps"] == 60.0
    assert controls["stream_format"] == "yuv"
    assert controls["capture_mode"] == "auto"


@pytest.mark.unit
def test_adb_reverse_timeout_is_not_retried_on_every_yuv_listen(monkeypatch) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        raise subprocess.TimeoutExpired(cmd, timeout=kwargs["timeout"])

    monkeypatch.setattr("src.capture.phone_yuv.subprocess.run", fake_run)

    source = PhoneYuvCaptureSource(
        PhoneCameraRuntimeConfig(
            frame_port=0,
            control_port=0,
            adb_reverse=True,
            adb_reverse_timeout_s=4.0,
        ),
    )

    try:
        source._ensure_listening()
    finally:
        source.release()

    assert len(calls) == 1
    assert calls[0][0][-3:] == ["reverse", "tcp:0", "tcp:0"]
    assert calls[0][1]["timeout"] == 4.0


@pytest.mark.unit
def test_release_closes_socket_and_marks_source_closed():
    sender, receiver = _socket_pair()
    source = PhoneYuvCaptureSource.from_socket(receiver)

    source.release()
    sender.close()

    assert not source.isOpened()
    assert source.read() == (False, None)


@pytest.mark.unit
def test_default_listener_hosts_are_loopback_only() -> None:
    config = PhoneCameraRuntimeConfig(frame_port=0, control_port=0)
    source = PhoneYuvCaptureSource(config)

    try:
        assert config.frame_host == "127.0.0.1"
        assert config.control_host == "127.0.0.1"
    finally:
        source.release()


@pytest.mark.unit
def test_non_loopback_listener_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="loopback"):
        PhoneYuvCaptureSource(
            PhoneCameraRuntimeConfig(
                frame_host="0.0.0.0",
                frame_port=0,
                control_port=0,
            )
        )
