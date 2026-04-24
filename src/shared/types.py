from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


def _coerce_bbox(bbox: np.ndarray | list[float] | tuple[float, float, float, float]) -> np.ndarray:
    array = np.asarray(bbox, dtype=np.float32)
    if array.shape != (4,):
        raise ValueError(f"expected bbox shape (4,), got {array.shape}")
    return array


def _coerce_keypoints(keypoints: np.ndarray | None) -> np.ndarray | None:
    if keypoints is None:
        return None
    array = np.asarray(keypoints, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("expected keypoints with shape (N, >=2)")
    return array


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: np.ndarray
    score: float
    keypoints: np.ndarray | None = None
    class_id: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", _coerce_bbox(self.bbox))
        object.__setattr__(self, "keypoints", _coerce_keypoints(self.keypoints))

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)


@dataclass(slots=True)
class Track:
    track_id: int
    bbox: np.ndarray
    score: float
    keypoints: np.ndarray | None = None
    age: int = 1
    lost_frames: int = 0
    last_timestamp_ns: int = 0

    def __post_init__(self) -> None:
        self.bbox = _coerce_bbox(self.bbox)
        self.keypoints = _coerce_keypoints(self.keypoints)

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)


@dataclass(frozen=True, slots=True)
class TrackingMessage:
    frame_id: int
    timestamp_ns: int
    target_kind: Literal["human", "dog"]
    target_acquired: bool
    state_source: Literal["measurement", "prediction", "lost", "center"]
    track_id: int | None
    confidence: float
    raw_pixel: tuple[float, float] | None
    filtered_pixel: tuple[float, float] | None
    raw_angles: tuple[float, float] | None
    filtered_angles: tuple[float, float] | None
    inference_ms: float
    tracking_ms: float
    total_latency_ms: float
    fps: float = 0.0
    wait_ms: float = 0.0
    postprocess_ms: float = 0.0
    servo_angles: tuple[float, float] | None = None  # raw Kalman angles (no EMA) for servo output
    servo_angular_velocity: tuple[float, float] | None = (
        None  # raw Kalman velocity for servo output
    )
    filtered_velocity: tuple[float, float] | None = None  # (vx, vy) in pixels/sec from Kalman
    angular_velocity: tuple[float, float] | None = None  # (pan_vel, tilt_vel) in rad/s
    is_occlusion_recovery: bool = False
    hold_time_s: float | None = None  # adaptive hold time override
    laser_pixel: tuple[float, float] | None = None
    other_targets: tuple[tuple[float, float, int, float], ...] = ()  # (cx, cy, track_id, score)
    egomotion_applied_px: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class ProcessErrorReport:
    process_name: str
    summary: str
    traceback_text: str
    timestamp_ns: int
    severity: Literal["debug", "info", "warning", "error", "critical"] = "error"
    context: dict[str, str] | None = None
