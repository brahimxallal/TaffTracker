from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

from src.shared.ring_buffer import SharedRingBuffer


def _read_from_attached_buffer(layout, write_index, result_queue) -> None:
    buffer = SharedRingBuffer.attach(layout, write_index)
    try:
        record = buffer.read_latest(copy=True)
        if record is None:
            result_queue.put(None)
            return
        result_queue.put((record.frame_id, record.timestamp_ns, int(record.frame[0, 0, 0])))
    finally:
        buffer.close()


@pytest.mark.unit
def test_ring_buffer_round_trip() -> None:
    buffer, write_index = SharedRingBuffer.create((4, 4, 3), num_slots=3)
    frame = np.full((4, 4, 3), 17, dtype=np.uint8)

    try:
        frame_id = buffer.write(frame, timestamp_ns=42)
        record = buffer.read_latest(copy=True)

        assert frame_id == 1
        assert record is not None
        assert record.frame_id == 1
        assert record.timestamp_ns == 42
        assert np.array_equal(record.frame, frame)
        assert record.frame is not frame
    finally:
        buffer.cleanup()


@pytest.mark.integration
def test_ring_buffer_attach_in_spawned_process() -> None:
    context = mp.get_context("spawn")
    buffer, write_index = SharedRingBuffer.create((2, 2, 3), num_slots=3, context=context)
    frame = np.full((2, 2, 3), 9, dtype=np.uint8)
    result_queue = context.Queue()

    try:
        buffer.write(frame, timestamp_ns=99)
        process = context.Process(
            target=_read_from_attached_buffer,
            args=(buffer.layout, write_index, result_queue),
        )
        process.start()
        process.join(10)

        assert process.exitcode == 0
        assert result_queue.get(timeout=1) == (1, 99, 9)
    finally:
        buffer.cleanup()


@pytest.mark.unit
def test_ring_buffer_reads_specific_frame_before_overwrite() -> None:
    buffer, _ = SharedRingBuffer.create((2, 2, 3), num_slots=3)

    try:
        for frame_id, value in enumerate((11, 22, 33), start=1):
            buffer.write(np.full((2, 2, 3), value, dtype=np.uint8), timestamp_ns=frame_id)

        record = buffer.read_frame(2, copy=True)

        assert record is not None
        assert record.frame_id == 2
        assert record.timestamp_ns == 2
        assert int(record.frame[0, 0, 0]) == 22
    finally:
        buffer.cleanup()


@pytest.mark.unit
def test_ring_buffer_returns_none_for_overwritten_frame() -> None:
    buffer, _ = SharedRingBuffer.create((2, 2, 3), num_slots=3)

    try:
        for frame_id, value in enumerate((1, 2, 3, 4), start=1):
            buffer.write(np.full((2, 2, 3), value, dtype=np.uint8), timestamp_ns=frame_id)

        assert buffer.read_frame(1, copy=True) is None
    finally:
        buffer.cleanup()
