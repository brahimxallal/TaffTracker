from __future__ import annotations

import time

import numpy as np
import pytest

from src.inference.postprocess import parse_yolo_output
from src.shared.types import Detection
from src.tracking.bytetrack import ByteTracker
from src.tracking.kalman import KalmanFilter


def _perf_mean_ms(fn, iterations: int = 200) -> float:
    # Warm up
    for _ in range(10):
        fn()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1000.0


@pytest.mark.perf
def test_postprocess_under_1ms() -> None:
    num_keypoints = 17
    raw = np.zeros((1, 5 + num_keypoints * 3, 30), dtype=np.float32)
    raw[0, 4, :] = 0.9
    for i in range(30):
        raw[0, 0:4, i] = [float(i * 25), float(i * 15), 20.0, 20.0]

    mean_ms = _perf_mean_ms(
        lambda: parse_yolo_output(raw, conf_threshold=0.5, num_keypoints=num_keypoints)
    )

    assert (
        mean_ms < 10.0
    ), f"postprocess mean {mean_ms:.3f}ms exceeds 10ms budget (target: <1ms on GPU machine)"


@pytest.mark.perf
def test_kalman_update_under_10us() -> None:
    kalman = KalmanFilter(process_noise=1.0, measurement_noise=2.0)
    kalman.update((100.0, 200.0), dt=0.016)

    mean_ms = _perf_mean_ms(lambda: kalman.update((101.0, 200.0), dt=0.016))
    mean_us = mean_ms * 1000.0

    assert (
        mean_us < 500.0
    ), f"Kalman update mean {mean_us:.2f}us exceeds 500us budget (target: <10us on GPU machine)"


@pytest.mark.perf
def test_bytetrack_single_target_under_500us() -> None:
    tracker = ByteTracker(track_thresh=0.5, match_thresh=0.5, max_lost=5)
    det = Detection(bbox=np.array([50.0, 50.0, 100.0, 150.0]), score=0.9)
    tracker.update([det], timestamp_ns=0)

    counter = [0]

    def _step() -> None:
        offset = float(counter[0])
        counter[0] += 1
        d = Detection(bbox=np.array([50.0 + offset, 50.0, 100.0 + offset, 150.0]), score=0.9)
        tracker.update([d], timestamp_ns=counter[0] * 16_000_000)

    mean_ms = _perf_mean_ms(_step)
    mean_us = mean_ms * 1000.0

    assert (
        mean_us < 5000.0
    ), f"ByteTracker single-target mean {mean_us:.1f}us exceeds 5ms budget (target: <500us on GPU machine)"
