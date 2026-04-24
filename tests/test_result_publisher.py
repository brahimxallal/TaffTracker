"""Unit tests for :class:`src.inference.result_publisher.ResultPublisher`.

Uses a small in-memory stub queue with a configurable capacity so the
drop-oldest + rate-limited-warning behavior is exercised deterministically
without relying on Python's multiprocessing queue timings.
"""

from __future__ import annotations

import logging
from queue import Empty, Full

import pytest

from src.inference.result_publisher import ResultPublisher


class _BoundedStubQueue:
    """Minimal in-memory queue that mimics mp.Queue's put_nowait / get_nowait."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._items: list[object] = []

    def put_nowait(self, item: object) -> None:
        if len(self._items) >= self._capacity:
            raise Full
        self._items.append(item)

    def get_nowait(self) -> object:
        if not self._items:
            raise Empty
        return self._items.pop(0)

    def __len__(self) -> int:
        return len(self._items)


@pytest.mark.unit
def test_publish_enqueues_when_capacity_available() -> None:
    q = _BoundedStubQueue(capacity=3)
    pub = ResultPublisher(q, logger=logging.getLogger("test.pub"))

    pub.publish("a")
    pub.publish("b")

    assert len(q) == 2
    assert pub.total_drop_count == 0


@pytest.mark.unit
def test_publish_drops_oldest_when_queue_full() -> None:
    q = _BoundedStubQueue(capacity=2)
    pub = ResultPublisher(q, logger=logging.getLogger("test.pub"))

    pub.publish("a")
    pub.publish("b")
    pub.publish("c")  # triggers drop-oldest then retry

    # Queue should still hold 2 items — oldest ("a") evicted, newest ("c") seated.
    assert len(q) == 2
    assert q.get_nowait() == "b"
    assert q.get_nowait() == "c"
    assert pub.total_drop_count == 1


@pytest.mark.unit
def test_publish_sentinel_none_also_enqueues() -> None:
    q = _BoundedStubQueue(capacity=2)
    pub = ResultPublisher(q, logger=logging.getLogger("test.pub"))

    pub.publish(None)

    assert len(q) == 1
    assert q.get_nowait() is None


@pytest.mark.unit
def test_publish_warning_is_rate_limited(caplog: pytest.LogCaptureFixture) -> None:
    """Flooding the queue must not produce one warning per dropped message."""
    q = _BoundedStubQueue(capacity=1)
    pub = ResultPublisher(
        q,
        logger=logging.getLogger("test.pub.ratelim"),
        warning_window_s=0.5,
    )
    caplog.set_level(logging.WARNING, logger="test.pub.ratelim")

    pub.publish("seed")
    for i in range(20):
        pub.publish(f"x{i}")

    # 20 drops in a tight loop share a single window, so at most one warning
    # should have been emitted.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) <= 1
    assert pub.total_drop_count == 20


@pytest.mark.unit
def test_publish_keeps_running_when_second_put_also_full() -> None:
    """If the consumer is fully stuck, publish must not raise."""

    class _AlwaysFull:
        def put_nowait(self, _: object) -> None:
            raise Full

        def get_nowait(self) -> object:
            raise Empty

    pub = ResultPublisher(_AlwaysFull(), logger=logging.getLogger("test.pub"))

    # Must not raise — the frame is simply dropped.
    pub.publish("x")
    assert pub.total_drop_count == 1


@pytest.mark.unit
def test_total_drop_count_is_cumulative() -> None:
    q = _BoundedStubQueue(capacity=1)
    pub = ResultPublisher(q, logger=logging.getLogger("test.pub"))

    pub.publish("seed")
    for _ in range(5):
        pub.publish("x")

    assert pub.total_drop_count == 5
