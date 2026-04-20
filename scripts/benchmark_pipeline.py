from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a headless tracker benchmark and export summary metrics.")
    parser.add_argument("--duration", type=float, default=60.0, help="Benchmark duration in seconds")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--config", default="config.yaml", help="Path to runtime config")
    parser.add_argument("--mode", choices=("camera", "video"), default=None)
    parser.add_argument("--target", choices=("human", "dog"), default=None)
    parser.add_argument("--source", default=None)
    return parser.parse_args()


def _read_profiler_summary(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stage = row.pop("stage")
            result[stage] = {
                key: float(value) if key != "count" else int(value)
                for key, value in row.items()
            }
    return result


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "src.main",
        "--config",
        args.config,
        "--headless",
        "--profile",
    ]
    if args.mode is not None:
        cmd.extend(["--mode", args.mode])
    if args.target is not None:
        cmd.extend(["--target", args.target])
    if args.source is not None:
        cmd.extend(["--source", args.source])
    return cmd


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5.0)


def main() -> None:
    args = _parse_args()
    workspace_root = Path(__file__).resolve().parents[1]
    logs_dir = workspace_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or logs_dir / f"bench_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    profiler_summary_path = logs_dir / "profiler_summary.csv"
    inference_metrics_path = logs_dir / "inference_metrics.json"
    output_metrics_path = logs_dir / "output_metrics.json"

    for path in (profiler_summary_path, inference_metrics_path, output_metrics_path):
        path.unlink(missing_ok=True)

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        _build_command(args),
        cwd=workspace_root,
        creationflags=creationflags,
    )
    started_at = time.perf_counter()
    try:
        deadline = started_at + args.duration
        while time.perf_counter() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Tracker exited early with code {process.returncode}")
            time.sleep(0.25)
    finally:
        _stop_process(process)

    elapsed_s = time.perf_counter() - started_at
    profiler_summary = _read_profiler_summary(profiler_summary_path)
    inference_metrics = _read_json(inference_metrics_path)
    output_metrics = _read_json(output_metrics_path)
    payload = {
        "duration_s": elapsed_s,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "command": _build_command(args),
        "stages": profiler_summary,
        "kalman_gate_rejection_rate": inference_metrics.get("gate_rejection_rate"),
        "inference_publish_drops": inference_metrics.get("publish_drop_count"),
        "display_drops": output_metrics.get("display_drops"),
        "display_drop_rate": output_metrics.get("display_drop_rate"),
        "servo_packet_rate_hz": output_metrics.get("packet_send_rate_hz"),
        "packet_send_latency_ms": output_metrics.get("packet_send_latency_ms"),
        "packets_sent": output_metrics.get("packets_sent"),
        "packets_failed": output_metrics.get("packets_failed"),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()