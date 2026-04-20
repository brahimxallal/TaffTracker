"""Command-line argument parsing, config assembly, and environment checks.

Extracted from ``src/main.py`` so the orchestrator can focus on process
lifecycle. The entry-point (``src.main.main``) re-exports the key names so
existing tests that import them from ``src.main`` keep working.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import PipelineConfig
from src.config_loader import build_config_from_yaml, load_yaml_config
from src.shared.pose_schema import get_pose_schema

LOGGER = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    """Build the argparse namespace used by ``src.main.main``."""
    parser = argparse.ArgumentParser(description="Ultra-low-latency vision-guided gimbal tracker")
    parser.add_argument("--mode", choices=("camera", "video"), default=None)
    parser.add_argument("--target", choices=("human", "dog"), default=None)
    parser.add_argument("--source", default=None, help="Camera index, video path, or IP camera URL")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--comm", choices=("serial", "udp", "auto"), default=None)
    parser.add_argument(
        "--no-comm", action="store_true", help="Disable serial/UDP output (visualization only)"
    )
    parser.add_argument("--serial-port", default=None)
    parser.add_argument("--baud-rate", type=int, default=None)
    parser.add_argument("--udp-host", default=None)
    parser.add_argument("--udp-port", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--backend", default=None, choices=("auto", "dshow", "msmf", "ffmpeg"))
    parser.add_argument(
        "--confidence", type=float, default=None, help="Override default confidence threshold"
    )
    parser.add_argument(
        "--hold-time", type=float, default=None, help="Override default hold time (seconds)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose DEBUG logging")
    parser.add_argument("--headless", action="store_true", help="Disable display window")
    parser.add_argument("--profile", action="store_true", help="Enable frequent profiler output")
    parser.add_argument("--log-file", type=str, default=None, help="Write logs to file")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Set log level",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=None,
        help="Camera horizontal FOV in degrees (overrides config.yaml)",
    )
    parser.add_argument("--invert-pan", action="store_true", help="Invert pan servo direction")
    parser.add_argument("--invert-tilt", action="store_true", help="Invert tilt servo direction")
    parser.add_argument(
        "--no-quit",
        action="store_true",
        help="Ignore ESC key - only Ctrl+C or process signal can stop the tracker",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Load YAML defaults, apply CLI overrides, and emit a summary log line."""
    yaml_data = load_yaml_config(Path(args.config))
    cli_overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    config = build_config_from_yaml(yaml_data, cli_overrides)
    LOGGER.info(
        "Adaptive mode (Q=%.1f, R=%.1f, max_lost=%d) at %d FPS | controller=PID",
        config.tracking.process_noise,
        config.tracking.measurement_noise,
        config.tracking.max_lost_frames,
        config.camera.fps,
    )
    return config


def validate_environment(config: PipelineConfig) -> None:
    """Fail fast when engines, sources, or comm settings would break startup."""
    engine_path = (
        config.models.person_engine_path
        if config.target == "human"
        else config.models.dog_engine_path
    )
    if not engine_path.exists():
        raise FileNotFoundError(
            f"Required TensorRT engine missing: {engine_path}. Run scripts/export_engines.py first."
        )

    source = config.source.strip()
    is_stream_source = source.lower().startswith(("http://", "https://", "rtsp://"))
    if config.mode == "video" and not is_stream_source and not Path(source).exists():
        raise FileNotFoundError(f"Video source not found: {source}")

    if config.mode == "camera":
        if config.camera.fov is None:
            raise ValueError("camera.fov must be set in config.yaml (horizontal FOV in degrees)")
        if config.comms.enabled:
            if config.comms.channel == "serial" and not config.comms.serial_port.strip():
                raise ValueError("Serial communication requires a non-empty COM port")
            if config.comms.channel == "udp":
                if not config.comms.udp_host.strip():
                    raise ValueError("UDP communication requires a non-empty host")
                if config.comms.udp_port <= 0:
                    raise ValueError("UDP communication requires a positive port number")

    if config.target == "dog":
        get_pose_schema("dog", config.paths.resolved_dog_pose_schema_path())
