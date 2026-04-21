"""Profiler summary helpers for the inference process.

Extracted from ``src/inference/process.py`` so the orchestrator doesn't
have to carry the formatting + JSON/CSV dumping logic inline.

Two public entry points:

* :func:`format_profiler_summary` — build the one-line ``Profiler summary``
  log string from a :class:`StageProfiler` plus a :class:`TrackingPipeline`'s
  gate counters. Pure; returns ``None`` when no inference samples have been
  recorded yet (the caller should not log an empty summary).
* :func:`write_profiler_summary` — dump the per-stage CSVs and the
  shutdown-time JSON metrics snapshot. Has filesystem side effects only.

Kept deliberately small: the orchestrator keeps the schedule (every 300
frames for the log line, once at shutdown for the files).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.inference.pipeline import TrackingPipeline
    from src.shared.profiler import StageProfiler


def format_profiler_summary(
    profiler: StageProfiler,
    pipeline: TrackingPipeline,
    fps: float,
    publish_drop_count: int,
) -> str | None:
    """Build the one-line profiler log string, or ``None`` if nothing to log.

    The caller normally gates logging on a frame counter (every N frames),
    and skips emission when this returns ``None`` — no inference samples
    have been recorded yet, so any line would be misleading.
    """
    inference_stats = profiler.get_percentiles("inference")
    if inference_stats is None:
        return None

    wait_stats = profiler.get_percentiles("wait")
    postprocess_stats = profiler.get_percentiles("postprocess")
    total_stats = profiler.get_percentiles("total_latency")

    segments = [f"FPS {fps:.1f}"]
    if wait_stats is not None:
        segments.append(
            f"wait p50/p95/p99 {wait_stats[0]:.1f}/{wait_stats[1]:.1f}/{wait_stats[2]:.1f} ms"
        )
    segments.append(
        f"inf p50/p95/p99 {inference_stats[0]:.1f}/{inference_stats[1]:.1f}/{inference_stats[2]:.1f} ms"
    )
    if postprocess_stats is not None:
        segments.append(
            f"post p50/p95/p99 {postprocess_stats[0]:.1f}/{postprocess_stats[1]:.1f}/{postprocess_stats[2]:.1f} ms"
        )
    if total_stats is not None:
        segments.append(
            f"total p50/p95/p99 {total_stats[0]:.1f}/{total_stats[1]:.1f}/{total_stats[2]:.1f} ms"
        )
    if pipeline.measurement_update_count > 0:
        gate_pct = (pipeline.measurement_gated_count / pipeline.measurement_update_count) * 100.0
        segments.append(f"gate_reject {gate_pct:.1f}%")
    if publish_drop_count > 0:
        segments.append(f"publish_drops {publish_drop_count}")

    return " | ".join(segments)


def write_profiler_summary(
    profiler: StageProfiler,
    pipeline: TrackingPipeline,
    *,
    csv_path: Path,
    summary_path: Path,
    metrics_path: Path,
    publish_drop_count: int,
) -> None:
    """Persist the profiler stats: per-run CSV, rolling summary CSV, JSON metrics.

    The CSVs are written via :meth:`StageProfiler.write_csv` (the profiler
    itself owns the schema). The JSON file captures gate-rejection rate and
    per-stage percentile stats for shutdown-time inspection.
    """
    profiler.write_csv(csv_path)
    profiler.write_csv(summary_path)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    gate_rate = (
        pipeline.measurement_gated_count / pipeline.measurement_update_count
        if pipeline.measurement_update_count > 0
        else 0.0
    )
    payload = {
        "publish_drop_count": publish_drop_count,
        "measurement_update_count": pipeline.measurement_update_count,
        "measurement_gated_count": pipeline.measurement_gated_count,
        "gate_rejection_rate": gate_rate,
        "stages": {
            name: {
                "count": snapshot.count,
                "last_ms": snapshot.last_ms,
                "mean_ms": snapshot.mean_ms,
                "p50_ms": snapshot.p50_ms,
                "p95_ms": snapshot.p95_ms,
                "p99_ms": snapshot.p99_ms,
            }
            for name, snapshot in profiler.get_stats().items()
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
