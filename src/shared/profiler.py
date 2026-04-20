from __future__ import annotations

import csv
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
from time import perf_counter_ns
from typing import Iterator
import threading


@dataclass(frozen=True)
class StageSnapshot:
    count: int
    last_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


class StageProfiler:
    def __init__(self, window_size: int = 100, enabled: bool = True) -> None:
        self._window_size = window_size
        self._enabled = enabled
        self._samples: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=window_size))
        self._lock = threading.Lock()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self._enabled:
            yield
            return
        start_ns = perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ns = perf_counter_ns() - start_ns
            self.add_sample(name, elapsed_ns)

    def add_sample(self, name: str, elapsed_ns: int) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._samples[name].append(elapsed_ns)

    def get_snapshot(self, name: str) -> StageSnapshot | None:
        with self._lock:
            samples = tuple(self._samples.get(name, ()))
        return self._snapshot_from_samples(samples)

    def flush_p50_p95_p99(self, name: str) -> tuple[float, float, float] | None:
        if not self._enabled:
            return None
        with self._lock:
            samples = tuple(self._samples.get(name, ()))
            if not samples:
                return None
            self._samples[name].clear()
        snapshot = self._snapshot_from_samples(samples)
        if snapshot is None:
            return None
        return snapshot.p50_ms, snapshot.p95_ms, snapshot.p99_ms

    def get_percentiles(self, name: str) -> tuple[float, float, float] | None:
        snapshot = self.get_snapshot(name)
        if snapshot is None:
            return None
        return snapshot.p50_ms, snapshot.p95_ms, snapshot.p99_ms

    def _snapshot_from_samples(self, samples: tuple[int, ...]) -> StageSnapshot | None:
        if not samples:
            return None

        ordered = sorted(samples)
        p50_index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.50) - 1))
        p95_index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))
        p99_index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.99) - 1))
        last_ms = samples[-1] / 1_000_000.0
        mean_ms = (sum(samples) / len(samples)) / 1_000_000.0
        p50_ms = ordered[p50_index] / 1_000_000.0
        p95_ms = ordered[p95_index] / 1_000_000.0
        p99_ms = ordered[p99_index] / 1_000_000.0
        return StageSnapshot(
            count=len(samples),
            last_ms=last_ms,
            mean_ms=mean_ms,
            p50_ms=p50_ms,
            p95_ms=p95_ms,
            p99_ms=p99_ms,
        )

    def get_stats(self) -> dict[str, StageSnapshot]:
        with self._lock:
            stage_names = tuple(self._samples.keys())
            all_samples = {name: tuple(self._samples[name]) for name in stage_names}
        result: dict[str, StageSnapshot] = {}
        for name, samples in all_samples.items():
            snapshot = self._snapshot_from_samples(samples)
            if snapshot is not None:
                result[name] = snapshot
        return result

    def write_csv(self, output_path: str | Path) -> Path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["stage", "count", "last_ms", "mean_ms", "p50_ms", "p95_ms", "p99_ms"])
            for stage_name, snapshot in sorted(self.get_stats().items()):
                writer.writerow(
                    [
                        stage_name,
                        snapshot.count,
                        snapshot.last_ms,
                        snapshot.mean_ms,
                        snapshot.p50_ms,
                        snapshot.p95_ms,
                        snapshot.p99_ms,
                    ]
                )
        return destination
