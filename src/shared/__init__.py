"""Shared data structures and transport primitives."""

from src.shared.protocol import (
    HEADER_V2,
    TrackingPacketV2,
    build_state_flags,
    decode_packet_v2,
    encode_packet_v2,
)
from src.shared.ring_buffer import SharedRingBuffer
from src.shared.types import Detection, ProcessErrorReport, Track, TrackingMessage

__all__ = [
    "HEADER_V2",
    "Detection",
    "ProcessErrorReport",
    "SharedRingBuffer",
    "Track",
    "TrackingMessage",
    "TrackingPacketV2",
    "build_state_flags",
    "decode_packet_v2",
    "encode_packet_v2",
]
