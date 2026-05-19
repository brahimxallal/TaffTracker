from __future__ import annotations

import socket
import subprocess

import numpy as np
import pytest

from src.capture.encoded_protocol import (
    CODEC_H264,
    CODEC_MPEG4,
    FLAG_CODEC_CONFIG,
    FLAG_KEYFRAME,
    HEADER_SIZE,
    LENGTH_PREFIX_SIZE,
    EncodedAccessUnitHeader,
    codec_name_to_id,
    pack_access_unit_header,
    pack_access_unit_packet,
    parse_access_unit_packet,
)
from src.capture.phone_mpeg import PhoneMpegCaptureSource, PhoneMpegRuntimeConfig


def _header(
    *,
    sequence: int = 1,
    width: int = 2,
    height: int = 2,
    payload_len: int = 3,
    codec_id: int = CODEC_MPEG4,
    flags: int = FLAG_KEYFRAME,
) -> EncodedAccessUnitHeader:
    return EncodedAccessUnitHeader(
        version=1,
        header_size=HEADER_SIZE,
        sequence=sequence,
        timestamp_ns=sequence * 1_000,
        width=width,
        height=height,
        codec_id=codec_id,
        flags=flags,
        exposure_ns=8_000_000,
        iso=100,
        focus_diopters=0.0,
        payload_len=payload_len,
    )


def _socket_pair() -> tuple[socket.socket, socket.socket]:
    return socket.socketpair()


class _FakeDecoder:
    def __init__(self) -> None:
        self.sequences: list[int] = []
        self.closed = False

    def decode(
        self,
        header: EncodedAccessUnitHeader,
        payload: bytes,
    ) -> np.ndarray:
        self.sequences.append(header.sequence)
        value = payload[0] if payload else header.sequence
        return np.full((header.height, header.width, 3), value, dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


@pytest.mark.unit
def test_encoded_access_unit_packet_round_trip() -> None:
    payload = b"abc123"
    header = _header(sequence=42, payload_len=len(payload))

    packet = pack_access_unit_packet(header, payload)
    parsed_header, parsed_payload = parse_access_unit_packet(packet)

    assert len(packet) == LENGTH_PREFIX_SIZE + header.header_size + len(payload)
    assert parsed_header == header
    assert parsed_header.codec_name == "mpeg4"
    assert parsed_payload == payload


@pytest.mark.unit
def test_malformed_encoded_magic_rejection() -> None:
    header = bytearray(pack_access_unit_header(_header()))
    header[:8] = b"BADENC!!"
    packet = len(header).to_bytes(LENGTH_PREFIX_SIZE, "little") + bytes(header)

    with pytest.raises(ValueError, match="magic"):
        parse_access_unit_packet(packet)


@pytest.mark.unit
def test_codec_name_validation() -> None:
    assert codec_name_to_id("mpeg4") == CODEC_MPEG4

    with pytest.raises(ValueError, match="codec"):
        codec_name_to_id("vp9")


@pytest.mark.unit
def test_decoder_injection_returns_bgr_uint8_frame() -> None:
    sender, receiver = _socket_pair()
    decoder = _FakeDecoder()
    source = PhoneMpegCaptureSource.from_socket(
        receiver,
        PhoneMpegRuntimeConfig(read_timeout_s=0.1, codec="mpeg4"),
        decoder=decoder,
    )
    payload = b"\x5a\x00\x00"
    sender.sendall(pack_access_unit_packet(_header(payload_len=len(payload)), payload))

    try:
        success, frame = source.read()
    finally:
        source.release()
        sender.close()

    assert success
    assert frame is not None
    assert frame.shape == (2, 2, 3)
    assert frame.dtype == np.uint8
    assert int(frame[0, 0, 0]) == 0x5A
    assert decoder.closed is True


@pytest.mark.unit
def test_decodes_queued_access_units_in_order_and_returns_newest_frame() -> None:
    sender, receiver = _socket_pair()
    decoder = _FakeDecoder()
    source = PhoneMpegCaptureSource.from_socket(
        receiver,
        PhoneMpegRuntimeConfig(read_timeout_s=0.1, codec="mpeg4"),
        decoder=decoder,
    )
    old_payload = b"\x10"
    new_payload = b"\xf0"
    sender.sendall(
        pack_access_unit_packet(
            _header(sequence=1, payload_len=len(old_payload)),
            old_payload,
        )
        + pack_access_unit_packet(
            _header(sequence=2, payload_len=len(new_payload)),
            new_payload,
        )
    )

    try:
        success, frame = source.read()
    finally:
        source.release()
        sender.close()

    assert success
    assert frame is not None
    assert decoder.sequences == [1, 2]
    assert source.last_header is not None
    assert source.last_header.sequence == 2
    assert int(frame.mean()) == 0xF0


@pytest.mark.unit
def test_skips_stream_codec_mismatch() -> None:
    sender, receiver = _socket_pair()
    decoder = _FakeDecoder()
    source = PhoneMpegCaptureSource.from_socket(
        receiver,
        PhoneMpegRuntimeConfig(read_timeout_s=0.1, codec="mpeg4"),
        decoder=decoder,
    )
    payload = b"\x01"
    sender.sendall(
        pack_access_unit_packet(
            _header(payload_len=len(payload), codec_id=CODEC_H264),
            payload,
        )
    )

    try:
        success, frame = source.read()
    finally:
        source.release()
        sender.close()

    assert not success
    assert frame is None
    assert decoder.sequences == []


@pytest.mark.unit
def test_h264_backlog_catches_up_at_latest_keyframe() -> None:
    sender, receiver = _socket_pair()
    decoder = _FakeDecoder()
    source = PhoneMpegCaptureSource.from_socket(
        receiver,
        PhoneMpegRuntimeConfig(
            read_timeout_s=0.1,
            codec="h264",
            max_decode_backlog_packets=2,
        ),
        decoder=decoder,
    )
    config_payload = b"cfg"
    sender.sendall(
        pack_access_unit_packet(
            _header(
                sequence=0,
                payload_len=len(config_payload),
                codec_id=CODEC_H264,
                flags=FLAG_CODEC_CONFIG,
            ),
            config_payload,
        )
        + pack_access_unit_packet(
            _header(sequence=1, payload_len=1, codec_id=CODEC_H264, flags=0),
            b"\x01",
        )
        + pack_access_unit_packet(
            _header(sequence=2, payload_len=1, codec_id=CODEC_H264, flags=FLAG_KEYFRAME),
            b"\x02",
        )
        + pack_access_unit_packet(
            _header(sequence=3, payload_len=1, codec_id=CODEC_H264, flags=0),
            b"\x03",
        )
    )

    try:
        success, frame = source.read()
    finally:
        source.release()
        sender.close()

    assert success
    assert frame is not None
    assert decoder.sequences == [0, 2, 3]
    assert source.last_header is not None
    assert source.last_header.sequence == 3
    assert int(frame.mean()) == 3


@pytest.mark.unit
def test_startup_controls_include_codec() -> None:
    source = PhoneMpegCaptureSource(
        PhoneMpegRuntimeConfig(
            requested_width=640,
            requested_height=480,
            requested_fps=60.0,
            codec="mpeg4",
            bitrate_bps=6_000_000,
            keyframe_interval_s=0.0,
            startup_controls={"capture_mode": "auto", "torch_enabled": True},
        ),
        decoder=_FakeDecoder(),
        start_listening=False,
    )

    try:
        commands = source._startup_commands()
    finally:
        source.release()

    assert commands[0]["cmd"] == "set_mode"
    assert commands[0]["stream_format"] == "mpeg"
    assert commands[0]["codec"] == "mpeg4"
    assert commands[0]["bitrate_bps"] == 6_000_000
    assert commands[0]["keyframe_interval_s"] == 0.0
    assert commands[0]["width"] == 640
    assert commands[0]["height"] == 480
    assert commands[0]["fps"] == 60.0
    assert commands[0]["capture_mode"] == "auto"
    assert commands[1] == {"cmd": "set_torch", "enabled": True}


@pytest.mark.unit
def test_start_phone_app_runs_after_mpeg_listeners_are_ready(monkeypatch) -> None:
    launches = []

    def fake_start_taffcam_app(config, controls):
        launches.append((config, controls))

    monkeypatch.setattr("src.capture.phone_mpeg.start_taffcam_app", fake_start_taffcam_app)

    source = PhoneMpegCaptureSource(
        PhoneMpegRuntimeConfig(
            frame_port=0,
            control_port=0,
            requested_width=640,
            requested_height=480,
            requested_fps=60.0,
            codec="h264",
            startup_controls={"capture_mode": "auto"},
            start_phone_app=True,
        ),
        decoder=_FakeDecoder(),
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
    assert controls["stream_format"] == "mpeg"
    assert controls["codec"] == "h264"
    assert controls["capture_mode"] == "auto"


@pytest.mark.unit
def test_adb_reverse_timeout_is_not_retried_on_every_mpeg_listen(monkeypatch) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        raise subprocess.TimeoutExpired(cmd, timeout=kwargs["timeout"])

    monkeypatch.setattr("src.capture.phone_mpeg.subprocess.run", fake_run)

    source = PhoneMpegCaptureSource(
        PhoneMpegRuntimeConfig(
            frame_port=0,
            control_port=0,
            adb_reverse=True,
            adb_reverse_timeout_s=4.0,
        ),
        decoder=_FakeDecoder(),
    )

    try:
        source._ensure_listening()
    finally:
        source.release()

    assert len(calls) == 1
    assert calls[0][0][-3:] == ["reverse", "tcp:0", "tcp:0"]
    assert calls[0][1]["timeout"] == 4.0


@pytest.mark.unit
def test_default_listener_hosts_are_loopback_only() -> None:
    config = PhoneMpegRuntimeConfig(frame_port=0, control_port=0)
    source = PhoneMpegCaptureSource(config, decoder=_FakeDecoder())

    try:
        assert config.frame_host == "127.0.0.1"
        assert config.control_host == "127.0.0.1"
    finally:
        source.release()


@pytest.mark.unit
def test_non_loopback_listener_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="loopback"):
        PhoneMpegCaptureSource(
            PhoneMpegRuntimeConfig(
                frame_host="0.0.0.0",
                frame_port=0,
                control_port=0,
            ),
            decoder=_FakeDecoder(),
        )
