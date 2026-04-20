"""Tests for SharedDisplayBuffer — single-slot shared memory display transport."""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

from src.shared.display_buffer import SharedDisplayBuffer


def _write_then_signal(layout, frame_value: int, ready_event, done_event) -> None:
    """Helper target: attach, write a frame, signal done."""
    buf = SharedDisplayBuffer.attach(layout)
    frame = np.full((4, 4, 3), frame_value, dtype=np.uint8)
    ready_event.wait(timeout=5)
    buf.write(frame)
    done_event.set()
    buf.close()


@pytest.mark.unit
def test_display_buffer_round_trip() -> None:
    buf = SharedDisplayBuffer.create((4, 4, 3))
    try:
        frame = np.full((4, 4, 3), 42, dtype=np.uint8)
        buf.write(frame)
        result = buf.read()
        assert result is not None
        assert np.array_equal(result, frame)
        assert result is not frame  # must be a copy
    finally:
        buf.cleanup()


@pytest.mark.unit
def test_display_buffer_returns_none_when_no_new_frame() -> None:
    buf = SharedDisplayBuffer.create((4, 4, 3))
    try:
        # Never written → None
        assert buf.read() is None

        # Write then read → frame
        buf.write(np.full((4, 4, 3), 1, dtype=np.uint8))
        assert buf.read() is not None

        # Second read without new write → None
        assert buf.read() is None
    finally:
        buf.cleanup()


@pytest.mark.unit
def test_display_buffer_overwrite() -> None:
    buf = SharedDisplayBuffer.create((4, 4, 3))
    try:
        buf.write(np.full((4, 4, 3), 10, dtype=np.uint8))
        buf.write(np.full((4, 4, 3), 20, dtype=np.uint8))
        result = buf.read()
        assert result is not None
        assert int(result[0, 0, 0]) == 20  # latest wins
    finally:
        buf.cleanup()


@pytest.mark.unit
def test_display_buffer_rejects_wrong_shape() -> None:
    buf = SharedDisplayBuffer.create((4, 4, 3))
    try:
        with pytest.raises(ValueError, match="expected"):
            buf.write(np.full((8, 8, 3), 1, dtype=np.uint8))
    finally:
        buf.cleanup()


@pytest.mark.integration
def test_display_buffer_cross_process() -> None:
    ctx = mp.get_context("spawn")
    buf = SharedDisplayBuffer.create((4, 4, 3))
    ready_event = ctx.Event()
    done_event = ctx.Event()

    try:
        proc = ctx.Process(
            target=_write_then_signal,
            args=(buf.layout, 77, ready_event, done_event),
        )
        proc.start()
        ready_event.set()
        done_event.wait(timeout=10)
        proc.join(timeout=10)

        assert proc.exitcode == 0
        result = buf.read()
        assert result is not None
        assert int(result[0, 0, 0]) == 77
    finally:
        buf.cleanup()
