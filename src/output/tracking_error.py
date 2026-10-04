"""Whole-run statistics for the aiming error the camera measures while locked.

The error is the angle between the camera axis and the tracked target in the
frame being controlled, i.e. how far the gimbal is *behind* (or ahead of) the
target at that instant. It is the quantity a lower-latency control path is meant
to shrink, so the output process keeps a histogram of it and reports percentiles
in the periodic log and in ``output_metrics.json`` for before/after comparisons.

A fixed-bin histogram keeps the per-frame cost to one integer increment and the
memory bounded for arbitrarily long runs (a rolling window would only describe
the last few seconds).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrackingErrorSummary:
    count: int
    mean_deg: float
    p50_deg: float
    p95_deg: float
    p99_deg: float
    max_deg: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean_deg": self.mean_deg,
            "p50_deg": self.p50_deg,
            "p95_deg": self.p95_deg,
            "p99_deg": self.p99_deg,
            "max_deg": self.max_deg,
        }


class _Histogram:
    def __init__(self, bins: int, bin_deg: float) -> None:
        self._bin_deg = bin_deg
        self._counts = [0] * (bins + 1)  # last bin collects everything above range
        self.count = 0
        self._sum = 0.0
        self._max = 0.0

    def add(self, error_deg: float) -> None:
        index = min(len(self._counts) - 1, int(error_deg / self._bin_deg))
        self._counts[index] += 1
        self.count += 1
        self._sum += error_deg
        if error_deg > self._max:
            self._max = error_deg

    def _percentile(self, fraction: float) -> float:
        target = max(1, math.ceil(self.count * fraction))
        seen = 0
        for index, bucket in enumerate(self._counts):
            seen += bucket
            if seen >= target:
                return min((index + 0.5) * self._bin_deg, self._max)
        return self._max

    def summary(self) -> TrackingErrorSummary | None:
        if self.count == 0:
            return None
        return TrackingErrorSummary(
            count=self.count,
            mean_deg=self._sum / self.count,
            p50_deg=self._percentile(0.50),
            p95_deg=self._percentile(0.95),
            p99_deg=self._percentile(0.99),
            max_deg=self._max,
        )


class TrackingErrorStats:
    """Histogram of per-frame aiming error with a whole-run and a per-interval view."""

    def __init__(self, max_deg: float = 60.0, bin_deg: float = 0.1) -> None:
        if max_deg <= 0.0 or bin_deg <= 0.0:
            raise ValueError("max_deg and bin_deg must be positive")
        self._bins = max(1, int(math.ceil(max_deg / bin_deg)))
        self._bin_deg = bin_deg
        self._run = _Histogram(self._bins, bin_deg)
        self._interval = _Histogram(self._bins, bin_deg)

    def observe(self, error_deg: float) -> None:
        """Record one frame's error (degrees); non-finite values are ignored."""
        if not math.isfinite(error_deg):
            return
        error_deg = abs(error_deg)
        self._run.add(error_deg)
        self._interval.add(error_deg)

    def summary(self) -> TrackingErrorSummary | None:
        """Statistics over every frame observed so far (``None`` before the first)."""
        return self._run.summary()

    def flush_interval(self) -> TrackingErrorSummary | None:
        """Statistics since the previous flush; starts a new interval."""
        result = self._interval.summary()
        self._interval = _Histogram(self._bins, self._bin_deg)
        return result
