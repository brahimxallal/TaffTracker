"""Tests for the 100 Hz manual tick thread in OutputProcess."""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from unittest.mock import MagicMock

import pytest

from src.config import CameraConfig, CommConfig, GimbalConfig, TrackingConfig
from src.output.process import OutputProcess
from src.shared.ring_buffer import SharedRingBuffer


def _make_manual_proc():
    """Create an OutputProcess with manual mode shared values."""
    layout, write_index = SharedRingBuffer.create((480, 640, 3), num_slots=2)
    manual_mode = mp.Value("b", 0)
    manual_pan = mp.Value("d", 0.0)
    manual_tilt = mp.Value("d", 0.0)
    proc = OutputProcess(
        layout=layout.layout,
        write_index=write_index,
        result_queue=mp.Queue(),
        shutdown_event=mp.Event(),
        error_queue=mp.Queue(),
        mode="camera",
        camera_config=CameraConfig(width=640, height=480),
        comm_config=CommConfig(),
        tracking_config=TrackingConfig(),
        gimbal_config=GimbalConfig(tilt_scale=1.0),
        manual_mode=manual_mode,
        manual_pan=manual_pan,
        manual_tilt=manual_tilt,
    )
    return proc, layout, manual_mode, manual_pan, manual_tilt


@pytest.mark.unit
def test_manual_emits_at_100hz():
    """Manual tick thread emits packets at ~100 Hz when manual mode is active."""
    proc, layout, manual_mode, manual_pan, manual_tilt = _make_manual_proc()
    try:
        sender = MagicMock()
        sender.is_connected = True
        send_lock = threading.Lock()
        shutdown = mp.Event()
        proc._manual_sequence = 0

        # Activate manual mode with a non-zero pan position
        manual_mode.value = 1
        manual_pan.value = 10.0
        manual_tilt.value = 5.0

        thread = threading.Thread(
            target=proc._manual_tick_loop,
            args=(sender, send_lock, shutdown),
            daemon=True,
        )
        thread.start()

        # Let it run for ~0.5s
        time.sleep(0.5)
        shutdown.set()
        thread.join(timeout=1.0)

        # At 100 Hz for 0.5s → expect ~50 packets (allow ±30% for scheduling)
        call_count = sender.send.call_count
        assert (
            call_count >= 35
        ), f"Manual thread too slow: {call_count} packets in 0.5s (expected ≥35)"
        assert (
            call_count <= 70
        ), f"Manual thread too fast: {call_count} packets in 0.5s (expected ≤70)"
    finally:
        layout.cleanup()


@pytest.mark.unit
def test_auto_mode_thread_idle():
    """Manual tick thread does NOT emit packets when manual_mode is off."""
    proc, layout, manual_mode, manual_pan, manual_tilt = _make_manual_proc()
    try:
        sender = MagicMock()
        sender.is_connected = True
        send_lock = threading.Lock()
        shutdown = mp.Event()
        proc._manual_sequence = 0

        # Manual mode OFF
        manual_mode.value = 0
        manual_pan.value = 10.0

        thread = threading.Thread(
            target=proc._manual_tick_loop,
            args=(sender, send_lock, shutdown),
            daemon=True,
        )
        thread.start()

        time.sleep(0.3)
        shutdown.set()
        thread.join(timeout=1.0)

        assert (
            sender.send.call_count == 0
        ), f"Thread emitted {sender.send.call_count} packets while manual_mode=0"
    finally:
        layout.cleanup()


@pytest.mark.unit
def test_manual_thread_stops_on_mode_switch():
    """Thread stops emitting promptly when manual_mode switches off."""
    proc, layout, manual_mode, manual_pan, manual_tilt = _make_manual_proc()
    try:
        sender = MagicMock()
        sender.is_connected = True
        send_lock = threading.Lock()
        shutdown = mp.Event()
        proc._manual_sequence = 0

        manual_mode.value = 1
        manual_pan.value = 10.0

        thread = threading.Thread(
            target=proc._manual_tick_loop,
            args=(sender, send_lock, shutdown),
            daemon=True,
        )
        thread.start()

        # Let it emit for 0.2s
        time.sleep(0.2)
        count_active = sender.send.call_count
        assert count_active > 0

        # Switch to auto — thread should stop emitting
        manual_mode.value = 0
        sender.send.reset_mock()
        time.sleep(0.2)

        # Should have emitted 0 or very few (one straggler at most)
        assert (
            sender.send.call_count <= 1
        ), f"Thread still emitting after mode switch: {sender.send.call_count} packets"

        shutdown.set()
        thread.join(timeout=1.0)
    finally:
        layout.cleanup()
