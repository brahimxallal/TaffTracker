"""Tests for ``src.inference.telemetry``.

Covers the two pure helpers extracted from ``InferenceProcess``:
:func:`format_profiler_summary` and :func:`write_profiler_summary`.
"""

from __future__ import annotations

import json

import pytest

from src.inference.telemetry import format_profiler_summary, write_profiler_summary
from src.shared.profiler import StageProfiler


class _StubPipeline:
    """Minimal stand-in for :class:`TrackingPipeline`.

    The telemetry helpers only read ``measurement_update_count`` and
    ``measurement_gated_count``; constructing the real pipeline would
    require a full stack of stages and is unnecessary here.
    """

    def __init__(self, update_count: int = 0, gated_count: int = 0) -> None:
        self.measurement_update_count = update_count
        self.measurement_gated_count = gated_count


def _make_profiler_with_samples(**stage_samples_ns: tuple[int, ...]) -> StageProfiler:
    profiler = StageProfiler(window_size=32, enabled=True)
    for stage_name, samples in stage_samples_ns.items():
        for sample_ns in samples:
            profiler.add_sample(stage_name, sample_ns)
    return profiler


# ---------------------------------------------------------------------------
# format_profiler_summary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_format_summary_returns_none_when_no_inference_samples() -> None:
    profiler = _make_profiler_with_samples(wait=(1_000_000,))  # wait only, no inference
    pipeline = _StubPipeline()
    assert format_profiler_summary(profiler, pipeline, fps=60.0, publish_drop_count=0) is None


@pytest.mark.unit
def test_format_summary_minimal_includes_fps_and_inference() -> None:
    profiler = _make_profiler_with_samples(inference=(5_000_000,))  # 5 ms
    pipeline = _StubPipeline()
    line = format_profiler_summary(profiler, pipeline, fps=59.4, publish_drop_count=0)
    assert line is not None
    assert line.startswith("FPS 59.4")
    assert "inf p50/p95/p99 5.0/5.0/5.0 ms" in line
    # With no wait/postprocess/total/gate/drops samples, none of those appear.
    for banned in ("wait", "post", "total", "gate_reject", "publish_drops"):
        assert banned not in line


@pytest.mark.unit
def test_format_summary_includes_all_stage_segments_when_available() -> None:
    profiler = _make_profiler_with_samples(
        wait=(2_000_000,),
        inference=(8_000_000,),
        postprocess=(1_500_000,),
        total_latency=(12_000_000,),
    )
    pipeline = _StubPipeline()
    line = format_profiler_summary(profiler, pipeline, fps=60.0, publish_drop_count=0)
    assert line is not None
    assert "wait p50/p95/p99 2.0/2.0/2.0 ms" in line
    assert "inf p50/p95/p99 8.0/8.0/8.0 ms" in line
    assert "post p50/p95/p99 1.5/1.5/1.5 ms" in line
    assert "total p50/p95/p99 12.0/12.0/12.0 ms" in line


@pytest.mark.unit
def test_format_summary_includes_gate_rejection_rate() -> None:
    profiler = _make_profiler_with_samples(inference=(5_000_000,))
    pipeline = _StubPipeline(update_count=200, gated_count=10)
    line = format_profiler_summary(profiler, pipeline, fps=60.0, publish_drop_count=0)
    assert line is not None
    # 10/200 = 5.0 %
    assert "gate_reject 5.0%" in line


@pytest.mark.unit
def test_format_summary_omits_gate_rejection_when_no_updates() -> None:
    profiler = _make_profiler_with_samples(inference=(5_000_000,))
    pipeline = _StubPipeline(update_count=0, gated_count=0)
    line = format_profiler_summary(profiler, pipeline, fps=60.0, publish_drop_count=0)
    assert line is not None
    assert "gate_reject" not in line


@pytest.mark.unit
def test_format_summary_includes_publish_drops_when_positive() -> None:
    profiler = _make_profiler_with_samples(inference=(5_000_000,))
    pipeline = _StubPipeline()
    line = format_profiler_summary(profiler, pipeline, fps=60.0, publish_drop_count=7)
    assert line is not None
    assert "publish_drops 7" in line


@pytest.mark.unit
def test_format_summary_omits_publish_drops_when_zero() -> None:
    profiler = _make_profiler_with_samples(inference=(5_000_000,))
    pipeline = _StubPipeline()
    line = format_profiler_summary(profiler, pipeline, fps=60.0, publish_drop_count=0)
    assert line is not None
    assert "publish_drops" not in line


# ---------------------------------------------------------------------------
# write_profiler_summary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_write_summary_creates_csvs_and_json(tmp_path) -> None:
    profiler = _make_profiler_with_samples(
        inference=(4_000_000, 5_000_000, 6_000_000),
        postprocess=(1_000_000, 1_500_000),
    )
    pipeline = _StubPipeline(update_count=100, gated_count=4)

    csv_path = tmp_path / "profiler.csv"
    summary_path = tmp_path / "profiler_summary.csv"
    metrics_path = tmp_path / "metrics" / "inference_metrics.json"

    write_profiler_summary(
        profiler,
        pipeline,
        csv_path=csv_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        publish_drop_count=2,
    )

    assert csv_path.exists()
    assert summary_path.exists()
    assert metrics_path.exists()

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["publish_drop_count"] == 2
    assert payload["measurement_update_count"] == 100
    assert payload["measurement_gated_count"] == 4
    assert payload["gate_rejection_rate"] == pytest.approx(0.04)
    assert "inference" in payload["stages"]
    assert payload["stages"]["inference"]["count"] == 3


@pytest.mark.unit
def test_write_summary_handles_zero_measurement_updates(tmp_path) -> None:
    profiler = _make_profiler_with_samples(inference=(5_000_000,))
    pipeline = _StubPipeline(update_count=0, gated_count=0)

    metrics_path = tmp_path / "inference_metrics.json"
    write_profiler_summary(
        profiler,
        pipeline,
        csv_path=tmp_path / "profiler.csv",
        summary_path=tmp_path / "summary.csv",
        metrics_path=metrics_path,
        publish_drop_count=0,
    )

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    # Avoids division by zero and reports zero rate explicitly.
    assert payload["gate_rejection_rate"] == 0.0


@pytest.mark.unit
def test_write_summary_creates_metrics_parent_directory(tmp_path) -> None:
    profiler = _make_profiler_with_samples(inference=(5_000_000,))
    pipeline = _StubPipeline()

    nested = tmp_path / "does" / "not" / "exist" / "metrics.json"
    write_profiler_summary(
        profiler,
        pipeline,
        csv_path=tmp_path / "profiler.csv",
        summary_path=tmp_path / "summary.csv",
        metrics_path=nested,
        publish_drop_count=0,
    )

    assert nested.exists()
    assert nested.parent.is_dir()
