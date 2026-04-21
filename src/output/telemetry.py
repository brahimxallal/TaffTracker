"""Metrics snapshot writer for the output process.

Extracted from ``src/output/process.py`` so that the shutdown-only
telemetry I/O does not bloat the runtime-path module. Pure function —
no multiprocessing, no hardware, no mutable state.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.shared.profiler import StageProfiler


def write_metrics_summary(
    profiler: StageProfiler,
    sender: Any,
    run_start: float,
    display_drops: int,
    display_total: int,
    *,
    log_dir: Path = Path("logs"),
) -> None:
    """Persist a JSON summary of the output-process run to ``logs/output_metrics.json``.

    Captures packet counters, packet-send latency percentiles, and display
    drop rate. Called once during graceful shutdown; never on the hot path.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "output_metrics.json"
    send_stats = profiler.get_snapshot("packet_send")
    duration_s = max(time.perf_counter() - run_start, 1e-6)
    packets_sent = int(getattr(sender, "packets_sent", 0)) if sender is not None else 0
    packets_failed = int(getattr(sender, "packets_failed", 0)) if sender is not None else 0
    payload = {
        "duration_s": duration_s,
        "display_drops": display_drops,
        "display_total": display_total,
        "display_drop_rate": (
            (display_drops / max(1, display_total)) if display_total > 0 else 0.0
        ),
        "packets_sent": packets_sent,
        "packets_failed": packets_failed,
        "packet_send_rate_hz": packets_sent / duration_s,
        "packet_send_latency_ms": (
            {
                "count": send_stats.count,
                "last_ms": send_stats.last_ms,
                "mean_ms": send_stats.mean_ms,
                "p50_ms": send_stats.p50_ms,
                "p95_ms": send_stats.p95_ms,
                "p99_ms": send_stats.p99_ms,
            }
            if send_stats is not None
            else None
        ),
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
