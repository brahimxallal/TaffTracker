from pathlib import Path

import pytest

from src.shared.profiler import StageProfiler


@pytest.mark.unit
def test_profiler_writes_csv_summary(tmp_path: Path) -> None:
    profiler = StageProfiler(window_size=4)
    profiler.add_sample("capture", 1_000_000)
    profiler.add_sample("capture", 2_000_000)
    profiler.add_sample("inference", 4_000_000)

    output_path = tmp_path / "profile.csv"
    profiler.write_csv(output_path)

    content = output_path.read_text(encoding="utf-8")
    assert "stage,count,last_ms,mean_ms,p50_ms,p95_ms,p99_ms" in content
    assert "capture,2,2.0,1.5,1.0,2.0,2.0" in content
    assert "inference,1,4.0,4.0,4.0,4.0,4.0" in content


@pytest.mark.unit
def test_profiler_get_percentiles_returns_tuple() -> None:
    profiler = StageProfiler(window_size=4)
    profiler.add_sample("latency", 2_000_000)
    profiler.add_sample("latency", 4_000_000)

    percentiles = profiler.get_percentiles("latency")

    assert percentiles == (2.0, 4.0, 4.0)
