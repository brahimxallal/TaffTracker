from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Stub hardware-only modules so tests collect on GPU-less machines
for _mod in ("cuda", "cuda.bindings", "cuda.bindings.runtime", "tensorrt"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import numpy as np
import pytest

from src.config import CameraConfig, TrackingConfig
from src.shared.ring_buffer import SharedRingBuffer
from src.shared.types import Detection, TrackingMessage


@pytest.fixture
def ring_buffer_pair():
    """Shared ring buffer with owner and write_index."""
    buffer, write_index = SharedRingBuffer.create((480, 640, 3), num_slots=3)
    yield buffer, write_index
    buffer.cleanup()


@pytest.fixture
def sample_frame() -> np.ndarray:
    """A 480x640 BGR test frame."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections() -> list[Detection]:
    return [
        Detection(bbox=np.array([10.0, 20.0, 50.0, 80.0]), score=0.92),
        Detection(bbox=np.array([200.0, 150.0, 280.0, 310.0]), score=0.75),
    ]


@pytest.fixture
def sample_tracking_message() -> TrackingMessage:
    return TrackingMessage(
        frame_id=1,
        timestamp_ns=1_000_000_000,
        target_kind="human",
        target_acquired=True,
        state_source="measurement",
        track_id=1,
        confidence=0.9,
        raw_pixel=(100.0, 200.0),
        filtered_pixel=(101.0, 201.0),
        raw_angles=(0.1, 0.2),
        filtered_angles=(0.11, 0.21),
        inference_ms=10.0,
        tracking_ms=1.0,
        total_latency_ms=15.0,
    )


@pytest.fixture
def camera_config() -> CameraConfig:
    return CameraConfig(width=640, height=480, fps=30)


@pytest.fixture
def tracking_config() -> TrackingConfig:
    return TrackingConfig(hold_time_s=0.5, confidence_threshold=0.5)
