from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

MAGIC = b"TAFFENC1"
VERSION = 1

CODEC_MPEG4 = 1
CODEC_H264 = 2
CODEC_HEVC = 3

CODEC_NAME_TO_ID = {
    "mpeg4": CODEC_MPEG4,
    "h264": CODEC_H264,
    "hevc": CODEC_HEVC,
}
CODEC_ID_TO_NAME = {codec_id: name for name, codec_id in CODEC_NAME_TO_ID.items()}
SUPPORTED_CODEC_IDS = frozenset(CODEC_ID_TO_NAME)

FLAG_KEYFRAME = 1 << 0
FLAG_CODEC_CONFIG = 1 << 1

LENGTH_PREFIX_STRUCT = struct.Struct("<I")
LENGTH_PREFIX_SIZE = LENGTH_PREFIX_STRUCT.size
HEADER_STRUCT = struct.Struct("<8sHHQqHHHHqifI")
HEADER_SIZE = HEADER_STRUCT.size


@dataclass(frozen=True, slots=True)
class EncodedAccessUnitHeader:
    version: int
    header_size: int
    sequence: int
    timestamp_ns: int
    width: int
    height: int
    codec_id: int
    flags: int
    exposure_ns: int
    iso: int
    focus_diopters: float
    payload_len: int

    @property
    def codec_name(self) -> str:
        return CODEC_ID_TO_NAME[self.codec_id]


def codec_name_to_id(codec: str) -> int:
    try:
        return CODEC_NAME_TO_ID[codec.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported encoded phone codec: {codec!r}") from exc


def parse_access_unit_header(data: bytes) -> EncodedAccessUnitHeader:
    if len(data) < HEADER_SIZE:
        raise ValueError(f"encoded access unit header too short: {len(data)} < {HEADER_SIZE}")

    (
        magic,
        version,
        header_size,
        sequence,
        timestamp_ns,
        width,
        height,
        codec_id,
        flags,
        exposure_ns,
        iso,
        focus_diopters,
        payload_len,
    ) = HEADER_STRUCT.unpack_from(data)

    if magic != MAGIC:
        raise ValueError(f"invalid encoded phone magic: {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported encoded phone version: {version}")
    if header_size < HEADER_SIZE:
        raise ValueError(f"invalid encoded phone header size: {header_size}")
    if codec_id not in SUPPORTED_CODEC_IDS:
        raise ValueError(f"unsupported encoded phone codec id: {codec_id}")

    return EncodedAccessUnitHeader(
        version=version,
        header_size=header_size,
        sequence=sequence,
        timestamp_ns=timestamp_ns,
        width=width,
        height=height,
        codec_id=codec_id,
        flags=flags,
        exposure_ns=exposure_ns,
        iso=iso,
        focus_diopters=focus_diopters,
        payload_len=payload_len,
    )


def pack_access_unit_header(header: EncodedAccessUnitHeader) -> bytes:
    if header.version != VERSION:
        raise ValueError(f"unsupported encoded phone version: {header.version}")
    if header.header_size < HEADER_SIZE:
        raise ValueError(f"invalid encoded phone header size: {header.header_size}")
    if header.codec_id not in SUPPORTED_CODEC_IDS:
        raise ValueError(f"unsupported encoded phone codec id: {header.codec_id}")

    fixed_header = HEADER_STRUCT.pack(
        MAGIC,
        header.version,
        header.header_size,
        header.sequence,
        header.timestamp_ns,
        header.width,
        header.height,
        header.codec_id,
        header.flags,
        header.exposure_ns,
        header.iso,
        header.focus_diopters,
        header.payload_len,
    )
    if header.header_size == HEADER_SIZE:
        return fixed_header
    return fixed_header + bytes(header.header_size - HEADER_SIZE)


def pack_access_unit_packet(header: EncodedAccessUnitHeader, payload: bytes) -> bytes:
    if header.payload_len != len(payload):
        raise ValueError(
            f"encoded phone payload length mismatch: {header.payload_len} != {len(payload)}"
        )
    header_bytes = pack_access_unit_header(header)
    packet_len = len(header_bytes) + len(payload)
    return LENGTH_PREFIX_STRUCT.pack(packet_len) + header_bytes + payload


def parse_access_unit_packet(
    data: bytes,
    *,
    max_payload_len: int = 64 * 1024 * 1024,
) -> tuple[EncodedAccessUnitHeader, bytes]:
    if len(data) < LENGTH_PREFIX_SIZE:
        raise ValueError(
            f"encoded access unit packet too short: {len(data)} < {LENGTH_PREFIX_SIZE}"
        )

    packet_len = LENGTH_PREFIX_STRUCT.unpack_from(data)[0]
    if packet_len < HEADER_SIZE:
        raise ValueError(f"encoded phone packet too small: {packet_len}")
    if len(data) < LENGTH_PREFIX_SIZE + packet_len:
        raise ValueError(
            f"encoded access unit packet incomplete: {len(data)} < "
            f"{LENGTH_PREFIX_SIZE + packet_len}"
        )

    packet_start = LENGTH_PREFIX_SIZE
    packet_end = packet_start + packet_len
    header = parse_access_unit_header(data[packet_start : packet_start + HEADER_SIZE])
    if header.header_size > packet_len:
        raise ValueError(f"encoded phone header larger than packet: {header.header_size}")
    payload_len = packet_len - header.header_size
    if header.payload_len != payload_len:
        raise ValueError(
            f"encoded phone payload length mismatch: {header.payload_len} != {payload_len}"
        )
    if payload_len > max_payload_len:
        raise ValueError(f"encoded phone payload too large: {payload_len}")

    payload_start = packet_start + header.header_size
    return header, data[payload_start:packet_end]


def read_exact(sock: socket.socket, n: int) -> bytes:
    if n < 0:
        raise ValueError("read size must be non-negative")

    chunks: list[bytes] = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("socket closed while reading encoded phone access unit")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_access_unit(
    sock: socket.socket,
    *,
    max_payload_len: int = 64 * 1024 * 1024,
) -> tuple[EncodedAccessUnitHeader, bytes]:
    length_prefix = read_exact(sock, LENGTH_PREFIX_SIZE)
    packet_len = LENGTH_PREFIX_STRUCT.unpack(length_prefix)[0]
    if packet_len < HEADER_SIZE:
        raise ValueError(f"encoded phone packet too small: {packet_len}")
    packet = read_exact(sock, packet_len)
    return parse_access_unit_packet(
        length_prefix + packet,
        max_payload_len=max_payload_len,
    )
