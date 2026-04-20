import pytest

from src.shared.profiler import StageProfiler


@pytest.mark.unit
def test_profiler_collects_stage_stats() -> None:
    profiler = StageProfiler(window_size=4)

    profiler.add_sample("capture", 1_000_000)
    profiler.add_sample("capture", 2_000_000)
    profiler.add_sample("capture", 3_000_000)

    snapshot = profiler.get_snapshot("capture")

    assert snapshot is not None
    assert snapshot.count == 3
    assert snapshot.last_ms == 3.0
    assert snapshot.mean_ms == 2.0
    assert snapshot.p50_ms == 2.0
    assert snapshot.p95_ms == 3.0
    assert snapshot.p99_ms == 3.0


@pytest.mark.unit
def test_stage_context_manager_records_sample() -> None:
    profiler = StageProfiler(window_size=10)

    with profiler.stage("test_stage"):
        _ = sum(range(100))

    snapshot = profiler.get_snapshot("test_stage")
    assert snapshot is not None
    assert snapshot.count == 1
    assert snapshot.last_ms >= 0.0


@pytest.mark.unit
def test_stage_context_manager_records_on_exception() -> None:
    profiler = StageProfiler(window_size=10)

    with pytest.raises(ValueError):
        with profiler.stage("err_stage"):
            raise ValueError("boom")

    snapshot = profiler.get_snapshot("err_stage")
    assert snapshot is not None
    assert snapshot.count == 1


@pytest.mark.unit
def test_disabled_profiler_stage_skips_recording() -> None:
    profiler = StageProfiler(enabled=False)

    with profiler.stage("noop"):
        _ = 1 + 1

    assert profiler.get_snapshot("noop") is None


@pytest.mark.unit
def test_disabled_profiler_add_sample_skips() -> None:
    profiler = StageProfiler(enabled=False)

    profiler.add_sample("noop", 1_000_000)

    assert profiler.get_snapshot("noop") is None


@pytest.mark.unit
def test_get_stats_returns_all_stages() -> None:
    profiler = StageProfiler(window_size=4)
    profiler.add_sample("a", 1_000_000)
    profiler.add_sample("b", 2_000_000)

    stats = profiler.get_stats()

    assert "a" in stats
    assert "b" in stats
    assert stats["a"].count == 1
    assert stats["b"].count == 1


@pytest.mark.unit
def test_get_snapshot_missing_stage_returns_none() -> None:
    profiler = StageProfiler()

    assert profiler.get_snapshot("nonexistent") is None


@pytest.mark.unit
def test_flush_p50_p95_p99_returns_percentiles_and_clears_stage() -> None:
    profiler = StageProfiler(window_size=8)
    profiler.add_sample("capture", 1_000_000)
    profiler.add_sample("capture", 3_000_000)
    profiler.add_sample("capture", 5_000_000)

    percentiles = profiler.flush_p50_p95_p99("capture")

    assert percentiles == (3.0, 5.0, 5.0)
    assert profiler.get_snapshot("capture") is None
