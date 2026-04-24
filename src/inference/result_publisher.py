"""Backpressure-aware publisher for inference → output messages.

The output process can briefly stall (serial write blocking, visualizer
frame pace) while inference keeps running at camera rate. When the result
queue fills up, blindly calling ``put_nowait`` raises ``queue.Full`` every
frame and we'd silently deadlock on shutdown ``put(None)``.

:class:`ResultPublisher` handles the stall by dropping the oldest message
in the queue and retrying, then rate-limits "dropped N messages" warnings
to at most once per second so the log stays readable during sustained
backpressure.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from queue import Empty, Full

from src.shared.types import TrackingMessage


class ResultPublisher:
    """Non-blocking publisher with drop-oldest on queue-full and rate-limited telemetry."""

    def __init__(
        self,
        result_queue: mp.Queue,
        *,
        logger: logging.Logger,
        warning_window_s: float = 1.0,
    ) -> None:
        self._queue = result_queue
        self._logger = logger
        self._warning_window_s = max(0.1, float(warning_window_s))
        self._total_drop_count = 0
        self._window_drop_count = 0
        self._window_start = time.perf_counter()

    @property
    def total_drop_count(self) -> int:
        """Total messages dropped since construction — for end-of-run telemetry."""
        return self._total_drop_count

    def publish(self, message: TrackingMessage | None) -> None:
        """Enqueue ``message`` (or None for shutdown sentinel), non-blocking.

        On queue-full: increments drop counters, pops the oldest message,
        retries, and emits a rate-limited warning summarizing drops in the
        window.
        """
        try:
            self._queue.put_nowait(message)
            return
        except Full:
            pass

        self._total_drop_count += 1
        self._window_drop_count += 1
        now = time.perf_counter()
        if now - self._window_start >= self._warning_window_s:
            self._logger.warning(
                "Inference result queue backpressure: dropped %d messages in the last %.1fs",
                self._window_drop_count,
                now - self._window_start,
            )
            self._window_drop_count = 0
            self._window_start = now

        try:
            self._queue.get_nowait()
        except Empty:
            pass
        try:
            self._queue.put_nowait(message)
        except Full:
            # Consumer is still stuck; drop this message entirely rather than
            # blocking the inference loop.
            pass
