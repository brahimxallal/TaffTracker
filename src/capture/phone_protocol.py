from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

MAGIC = b"TAFFYUV1"
VERSION = 1

PIXEL_FORMAT_I420 = 1
PIXEL_FORMAT_NV12 = 2
PIXEL_FORMAT_NV21 = 3
SUPPORTED_PIXEL_FORMATS = {
    PIXEL_FORMAT_I420,
    PIXEL_FORMAT_NV12,
    PIXEL_FORMAT_NV21,
}

HEADER_STRUCT = struct.Struct("<8sHHQqHHHHqifI")
HEADER_SIZE = HEADER_STRUCT.size


@dataclass(frozen=True, slots=True)
class FrameHeader:
    version: int
    header_size: int
    sequence: int
    timestamp_ns: int
    width: int
    height: int
    pixel_format: int
    flags: int
    exposure_ns: int
    iso: int
    focus_diopters: float
    payload_len: int


def parse_frame_header(data: bytes) -> FrameHeader:
    if len(data) < HEADER_SIZE:
        raise ValueError(f"frame header too short: {len(data)} < {HEADER_SIZE}")

    (
        magic,
        version,
        header_size,
        sequence,
        timestamp_ns,
        width,
        height,
        pixel_format,
        flags,
        exposure_ns,
        iso,
        focus_diopters,
        payload_len,
    ) = HEADER_STRUCT.unpack_from(data)

    if magic != MAGIC:
        raise ValueError(f"invalid phone YUV magic: {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported phone YUV version: {version}")
    if header_size < HEADER_SIZE:
        raise ValueError(f"invalid phone YUV header size: {header_size}")
    if pixel_format not in SUPPORTED_PIXEL_FORMATS:
        raise ValueError(f"unsupported phone YUV pixel format: {pixel_format}")

    return FrameHeader(
        version=version,
        header_size=header_size,
        sequence=sequence,
        timestamp_ns=timestamp_ns,
        width=width,
        height=height,
        pixel_format=pixel_format,
        flags=flags,
        exposure_ns=exposure_ns,
        iso=iso,
        focus_diopters=focus_diopters,
        payload_len=payload_len,
    )


def pack_frame_header(header: FrameHeader) -> bytes:
    if header.version != VERSION:
        raise ValueError(f"unsupported phone YUV version: {header.version}")
    if header.header_size < HEADER_SIZE:
        raise ValueError(f"invalid phone YUV header size: {header.header_size}")
    if header.pixel_format not in SUPPORTED_PIXEL_FORMATS:
        raise ValueError(f"unsupported phone YUV pixel format: {header.pixel_format}")

    fixed_header = HEADER_STRUCT.pack(
        MAGIC,
        header.version,
        header.header_size,
        header.sequence,
        header.timestamp_ns,
        header.width,
        header.height,
        header.pixel_format,
        header.flags,
        header.exposure_ns,
        header.iso,
        header.focus_diopters,
        header.payload_len,
    )
    if header.header_size == HEADER_SIZE:
        return fixed_header
    return fixed_header + bytes(header.header_size - HEADER_SIZE)


def read_exact(sock: socket.socket, n: int) -> bytes:
    if n < 0:
        raise ValueError("read size must be non-negative")

    chunks: list[bytes] = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("socket closed while reading phone YUV frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(
    sock: socket.socket,
    *,
    max_payload_len: int = 64 * 1024 * 1024,
) -> tuple[FrameHeader, bytes]:
    header_prefix = read_exact(sock, HEADER_SIZE)
    header = parse_frame_header(header_prefix)
    if header.payload_len > max_payload_len:
        raise ValueError(f"phone YUV payload too large: {header.payload_len}")
    if header.header_size > HEADER_SIZE:
        read_exact(sock, header.header_size - HEADER_SIZE)
    payload = read_exact(sock, header.payload_len)
    return header, payload
