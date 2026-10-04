"""Tests for ``src.output.tracking_error`` and its entry in ``output_metrics.json``."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest

from src.output.telemetry import write_metrics_summary
from src.output.tracking_error import TrackingErrorStats, TrackingErrorSummary
from src.shared.profiler import StageProfiler


def _bin_centres(count: int, bin_deg: float = 0.1) -> list[float]:
    """One error value in the middle of each of the first ``count`` bins."""
    return [(i + 0.5) * bin_deg for i in range(count)]


@pytest.mark.unit
class TestTrackingErrorStats:
    def test_nothing_is_reported_before_the_first_observation(self) -> None:
        stats = TrackingErrorStats()

        assert stats.summary() is None
        assert stats.flush_interval() is None

    def test_percentiles_of_a_uniform_distribution(self) -> None:
        stats = TrackingErrorStats()
        for error in _bin_centres(100):  # 0.05 ... 9.95 deg
            stats.observe(error)

        summary = stats.summary()

        assert summary is not None
        assert summary.count == 100
        assert summary.mean_deg == pytest.approx(5.0)
        assert summary.p50_deg == pytest.approx(5.0, abs=0.1)
        assert summary.p95_deg == pytest.approx(9.5, abs=0.1)
        assert summary.p99_deg == pytest.approx(9.9, abs=0.1)
        assert summary.max_deg == pytest.approx(9.95)

    def test_the_sign_of_the_error_does_not_matter(self) -> None:
        stats = TrackingErrorStats()
        stats.observe(-3.0)
        stats.observe(3.0)

        summary = stats.summary()

        assert summary is not None
        assert summary.mean_deg == pytest.approx(3.0)
        assert summary.max_deg == pytest.approx(3.0)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_errors_are_ignored(self, bad: float) -> None:
        stats = TrackingErrorStats()

        stats.observe(bad)

        assert stats.summary() is None

    def test_a_percentile_never_exceeds_the_largest_error_seen(self) -> None:
        stats = TrackingErrorStats()
        stats.observe(0.01)  # far below the middle of its 0.1 deg bin

        summary = stats.summary()

        assert summary is not None
        assert summary.p50_deg == pytest.approx(0.01)
        assert summary.p99_deg == pytest.approx(0.01)

    def test_errors_beyond_the_range_saturate_the_percentiles_but_not_the_maximum(self) -> None:
        stats = TrackingErrorStats(max_deg=10.0, bin_deg=0.5)
        for _ in range(10):
            stats.observe(50.0)

        summary = stats.summary()

        assert summary is not None
        assert summary.count == 10
        assert summary.p50_deg <= 10.5
        assert summary.max_deg == 50.0

    def test_flush_interval_covers_only_the_samples_since_the_last_flush(self) -> None:
        stats = TrackingErrorStats()
        for _ in range(10):
            stats.observe(1.0)
        first = stats.flush_interval()
        for _ in range(4):
            stats.observe(5.0)
        second = stats.flush_interval()

        assert first is not None
        assert (first.count, first.p50_deg) == (10, pytest.approx(1.0, abs=0.1))
        assert second is not None
        assert (second.count, second.p50_deg) == (4, pytest.approx(5.0, abs=0.1))
        assert stats.flush_interval() is None  # nothing new since the last flush
        whole_run = stats.summary()
        assert whole_run is not None
        assert whole_run.count == 14  # flushing does not forget the whole run

    @pytest.mark.parametrize("kwargs", [{"max_deg": 0.0}, {"bin_deg": 0.0}, {"bin_deg": -1.0}])
    def test_rejects_a_non_positive_range_or_bin_width(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            TrackingErrorStats(**kwargs)

    def test_summary_serialises_every_field(self) -> None:
        summary = TrackingErrorSummary(
            count=3, mean_deg=1.0, p50_deg=1.1, p95_deg=2.0, p99_deg=2.5, max_deg=3.0
        )

        assert summary.as_dict() == {
            "count": 3,
            "mean_deg": 1.0,
            "p50_deg": 1.1,
            "p95_deg": 2.0,
            "p99_deg": 2.5,
            "max_deg": 3.0,
        }


@pytest.mark.unit
class TestMetricsSummary:
    @staticmethod
    def _write(tmp_path: Path, tracking_error: TrackingErrorStats | None) -> dict:
        write_metrics_summary(
            StageProfiler(window_size=8, enabled=True),
            None,
            time.perf_counter() - 2.0,
            1,
            10,
            log_dir=tmp_path,
            tracking_error=tracking_error,
        )
        return json.loads((tmp_path / "output_metrics.json").read_text(encoding="utf-8"))

    def test_includes_the_whole_run_tracking_error(self, tmp_path: Path) -> None:
        stats = TrackingErrorStats()
        for error in (1.0, 2.0, 3.0):
            stats.observe(error)

        payload = self._write(tmp_path, stats)

        assert payload["tracking_error_deg"]["count"] == 3
        assert payload["tracking_error_deg"]["mean_deg"] == pytest.approx(2.0)
        assert payload["tracking_error_deg"]["max_deg"] == pytest.approx(3.0)
        assert set(payload["tracking_error_deg"]) == {
            "count",
            "mean_deg",
            "p50_deg",
            "p95_deg",
            "p99_deg",
            "max_deg",
        }

    def test_is_null_when_nothing_was_tracked(self, tmp_path: Path) -> None:
        assert self._write(tmp_path, TrackingErrorStats())["tracking_error_deg"] is None
        assert self._write(tmp_path, None)["tracking_error_deg"] is None
