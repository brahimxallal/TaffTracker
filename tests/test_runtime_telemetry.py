from __future__ import annotations

from queue import Queue

from src.shared.telemetry import RuntimeTelemetry, put_latest


def _telemetry(frame_id: int) -> RuntimeTelemetry:
    return RuntimeTelemetry(
        frame_id=frame_id,
        timestamp_ns=frame_id * 1_000_000,
        target_kind="human",
        target_acquired=True,
        state_source="measurement",
        track_id=1,
        confidence=0.9,
        filtered_pixel=(10.0, 20.0),
        filtered_angles=(0.1, 0.2),
        fps=60.0,
        inference_ms=4.0,
        tracking_ms=1.0,
        postprocess_ms=0.5,
        wait_ms=0.1,
        total_latency_ms=8.0,
        transport_status="LINK OFFLINE",
        packet_sequence=frame_id,
        lock_frames=frame_id,
        total_frames=frame_id,
        display_drops=0,
        display_total=frame_id,
    )


def test_put_latest_replaces_stale_telemetry_without_blocking():
    queue: Queue[RuntimeTelemetry] = Queue(maxsize=1)

    put_latest(queue, _telemetry(1))
    put_latest(queue, _telemetry(2))

    assert queue.get_nowait().frame_id == 2
    assert queue.empty()
