from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimeTelemetry:
    frame_id: int
    timestamp_ns: int
    target_kind: str
    target_acquired: bool
    state_source: str
    track_id: int | None
    confidence: float
    filtered_pixel: tuple[float, float] | None
    filtered_angles: tuple[float, float] | None
    fps: float
    inference_ms: float
    tracking_ms: float
    postprocess_ms: float
    wait_ms: float
    total_latency_ms: float
    transport_status: str
    packet_sequence: int
    lock_frames: int
    total_frames: int
    display_drops: int
    display_total: int


def put_latest(queue: Any | None, item: Any) -> None:
    """Publish only the newest telemetry item without blocking the hot path."""
    if queue is None:
        return
    try:
        queue.put_nowait(item)
        return
    except Full:
        pass

    try:
        queue.get_nowait()
    except Empty:
        pass

    try:
        queue.put_nowait(item)
    except Full:
        pass
