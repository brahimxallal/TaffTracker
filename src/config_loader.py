"""Load runtime configuration from config.yaml with CLI overrides."""

from __future__ import annotations

import argparse
import logging
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, get_args, get_type_hints

import yaml

from src.calibration.laser_boresight import load_boresight
from src.config import (
    CameraConfig,
    CommConfig,
    GimbalConfig,
    LaserBoresightConfig,
    LaserConfig,
    ModelConfig,
    Orientation,
    PipelineConfig,
    PreflightConfig,
    RelayConfig,
    RuntimeFlags,
    RuntimePaths,
    ServoControlConfig,
    adapt_tracking_for_fps,
    default_tracking_config,
)

LOGGER = logging.getLogger("config")


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load config.yaml and return as a dict. Returns empty dict if missing."""
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def _parse_orientation(cam_cfg: dict[str, Any]) -> Orientation:
    """Resolve camera.orientation from YAML, honouring the legacy portrait_mode bool.

    Precedence: explicit ``orientation`` string > legacy ``portrait_mode`` bool >
    default (LANDSCAPE_NATIVE).  Unknown orientation strings fall back to the
    default with a warning so a typo cannot silently break capture.
    """
    raw = cam_cfg.get("orientation")
    if raw is not None:
        try:
            return Orientation(str(raw).lower())
        except ValueError:
            valid = ", ".join(o.value for o in Orientation)
            LOGGER.warning(
                "Unknown camera.orientation '%s' (valid: %s); using landscape_native",
                raw,
                valid,
            )
            return Orientation.LANDSCAPE_NATIVE
    if "portrait_mode" in cam_cfg:
        return (
            Orientation.PORTRAIT if bool(cam_cfg["portrait_mode"]) else Orientation.LANDSCAPE_NATIVE
        )
    return Orientation.LANDSCAPE_NATIVE


def _overlay_dataclass(base, yaml_dict: dict[str, Any]):
    """Override fields on a frozen dataclass from a YAML dict, casting types."""
    if not yaml_dict:
        return base
    valid_keys = {f.name for f in fields(base)}
    overrides = {}
    for key, val in yaml_dict.items():
        if key not in valid_keys:
            LOGGER.warning("Unknown config key '%s' (ignored)", key)
            continue
        current = getattr(base, key)
        if current is None:
            overrides[key] = val
        else:
            overrides[key] = type(current)(val)
    return replace(base, **overrides) if overrides else base


_LITERAL_CONSTRAINTS: dict[str, tuple[str, ...]] = {
    "mode": get_args(get_type_hints(PipelineConfig)["mode"]),
    "target": get_args(get_type_hints(PipelineConfig)["target"]),
}


def _validate_config(config: PipelineConfig) -> None:
    """Validate Literal-constrained fields at the system boundary (YAML + CLI)."""
    for field_name, allowed in _LITERAL_CONSTRAINTS.items():
        value = getattr(config, field_name)
        if value not in allowed:
            raise ValueError(f"Invalid {field_name}={value!r}; expected one of {allowed}")
    if config.camera.backend not in get_args(get_type_hints(CameraConfig)["backend"]):
        raise ValueError(
            f"Invalid camera.backend={config.camera.backend!r}; "
            f"expected one of {get_args(get_type_hints(CameraConfig)['backend'])}"
        )
    if config.models.precision not in get_args(get_type_hints(ModelConfig)["precision"]):
        raise ValueError(
            f"Invalid models.precision={config.models.precision!r}; "
            f"expected one of {get_args(get_type_hints(ModelConfig)['precision'])}"
        )
    if config.comms.channel not in get_args(get_type_hints(CommConfig)["channel"]):
        raise ValueError(
            f"Invalid comms.channel={config.comms.channel!r}; "
            f"expected one of {get_args(get_type_hints(CommConfig)['channel'])}"
        )
    if config.flags.log_level not in get_args(get_type_hints(RuntimeFlags)["log_level"]):
        raise ValueError(
            f"Invalid flags.log_level={config.flags.log_level!r}; "
            f"expected one of {get_args(get_type_hints(RuntimeFlags)['log_level'])}"
        )


def build_launcher_env(
    config_path: Path,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, str]:
    config = build_config_from_yaml(load_yaml_config(config_path), cli_overrides)
    if config.comms.channel == "serial":
        comm_detail = config.comms.serial_port
    elif config.comms.channel == "udp":
        comm_detail = f"{config.comms.udp_host}:{config.comms.udp_port}"
    else:
        comm_detail = "serial->udp fallback"

    return {
        "CFG_MODE": config.mode,
        "CFG_TARGET": config.target,
        "CFG_SOURCE": config.source,
        "CFG_CHANNEL": config.comms.channel,
        "CFG_SERIAL_PORT": config.comms.serial_port,
        "CFG_BAUD_RATE": str(config.comms.baud_rate),
        "CFG_BACKEND": config.camera.backend,
        "CFG_UDP_HOST": config.comms.udp_host,
        "CFG_UDP_PORT": str(config.comms.udp_port),
        "CFG_FOV": "?" if config.camera.fov is None else f"{config.camera.fov:g}",
        "CFG_COMM_DETAIL": comm_detail,
    }


def print_launcher_env(config_path: Path, cli_overrides: dict[str, Any] | None = None) -> None:
    for key, value in build_launcher_env(config_path, cli_overrides).items():
        print(f"{key}={value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Config loader utilities")
    parser.add_argument(
        "--launcher-env",
        action="store_true",
        help="Print launch.bat environment variables derived from config.yaml",
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default="config.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()
    if args.launcher_env:
        print_launcher_env(Path(args.config_path))
        return
    parser.error("No action requested")


def build_config_from_yaml(
    yaml_data: dict[str, Any],
    cli_overrides: dict[str, Any] | None = None,
) -> PipelineConfig:
    """Build PipelineConfig from YAML data, applying CLI overrides on top."""
    cli = cli_overrides or {}

    mode = cli.get("mode") or yaml_data.get("mode", "camera")
    target = cli.get("target") or yaml_data.get("target", "human")
    source = cli.get("source") or str(yaml_data.get("source", "0"))

    cam_cfg = yaml_data.get("camera", {})
    fov_raw = cli.get("fov") if cli.get("fov") is not None else cam_cfg.get("fov")
    orientation = _parse_orientation(cam_cfg)
    camera = CameraConfig(
        width=cli.get("width") if cli.get("width") is not None else cam_cfg.get("width", 640),
        height=cli.get("height") if cli.get("height") is not None else cam_cfg.get("height", 640),
        fps=cli.get("fps") if cli.get("fps") is not None else cam_cfg.get("fps", 60),
        backend=cli.get("backend") or cam_cfg.get("backend", "auto"),
        buffer_size=int(cam_cfg.get("buffer_size", 1)),
        orientation=orientation,
        fov=float(fov_raw) if fov_raw is not None else None,
        capture_width=int(cam_cfg.get("capture_width", 1920)),
        capture_height=int(cam_cfg.get("capture_height", 1080)),
    )

    comm_cfg = yaml_data.get("comms", {})
    cli_comm_selected = any(
        cli.get(key) is not None
        for key in ("comm", "serial_port", "baud_rate", "udp_host", "udp_port")
    )
    comms = CommConfig(
        channel=cli.get("comm") or comm_cfg.get("channel", "serial"),
        serial_port=cli.get("serial_port") or comm_cfg.get("serial_port", "COM4"),
        baud_rate=(
            cli.get("baud_rate")
            if cli.get("baud_rate") is not None
            else comm_cfg.get("baud_rate", 921600)
        ),
        udp_host=cli.get("udp_host") or comm_cfg.get("udp_host", "192.168.4.1"),
        udp_port=(
            cli.get("udp_port")
            if cli.get("udp_port") is not None
            else comm_cfg.get("udp_port", 6000)
        ),
        udp_redundancy=int(comm_cfg.get("udp_redundancy", 2)),
        enabled=(
            False
            if cli.get("no_comm", False)
            else (True if cli_comm_selected else comm_cfg.get("enabled", True))
        ),
    )

    mdl_cfg = yaml_data.get("models", {})
    models = ModelConfig(
        person_model_path=Path(mdl_cfg.get("person_model", "models/yolo26n-person-17pose.pt")),
        dog_model_path=Path(mdl_cfg.get("dog_model", "models/Enhanceddog/best.pt")),
        engine_dir=Path(mdl_cfg.get("engine_dir", "engines")),
        image_size=mdl_cfg.get("image_size", 640),
        precision=mdl_cfg.get("precision", "fp16"),
    )

    tracking = default_tracking_config(target)
    trk_cfg = yaml_data.get("tracking", {})
    # Dog-specific YAML overrides: merge tracking_dog on top of tracking for dog target
    if target == "dog":
        dog_cfg = yaml_data.get("tracking_dog", {})
        if dog_cfg:
            # Merge dog overrides into tracking section (dog values win)
            merged = dict(trk_cfg)
            for key, val in dog_cfg.items():
                if isinstance(val, dict) and isinstance(merged.get(key), dict):
                    merged[key] = {**merged[key], **val}
                else:
                    merged[key] = val
            trk_cfg = merged
    if trk_cfg:
        # Generic sub-config loader: reads all fields from the dataclass defaults
        kalman = _overlay_dataclass(tracking.kalman, trk_cfg.get("kalman", {}))
        smoothing = _overlay_dataclass(tracking.smoothing, trk_cfg.get("smoothing", {}))
        postprocess = _overlay_dataclass(tracking.postprocess, trk_cfg.get("postprocess", {}))
        adaptive = _overlay_dataclass(tracking.adaptive, trk_cfg.get("adaptive", {}))

        # Top-level tracking overrides
        tracking_overrides = {
            "kalman": kalman,
            "smoothing": smoothing,
            "postprocess": postprocess,
            "adaptive": adaptive,
        }
        for key in (
            "confidence_threshold",
            "hold_time_s",
            "max_lost_frames",
            "process_noise",
            "measurement_noise",
            "tracker_match_threshold",
            "tracker_track_threshold",
            "tracker_birth_min_hits",
        ):
            if key in trk_cfg:
                tracking_overrides[key] = type(getattr(tracking, key))(trk_cfg[key])
        tracking = replace(tracking, **tracking_overrides)

    if cli.get("confidence") is not None:
        tracking = replace(tracking, confidence_threshold=cli["confidence"])
    if cli.get("hold_time") is not None:
        tracking = replace(tracking, hold_time_s=cli["hold_time"])
    tracking = adapt_tracking_for_fps(tracking, camera.fps)

    rt_cfg = yaml_data.get("runtime", {})
    debug = cli.get("debug", False) or rt_cfg.get("debug", False)
    flags = RuntimeFlags(
        debug=debug,
        headless=cli.get("headless", False) or rt_cfg.get("headless", False),
        profile=cli.get("profile", False) or rt_cfg.get("profile", False),
        log_file=(
            Path(cli["log_file"])
            if cli.get("log_file")
            else (Path(rt_cfg["log_file"]) if rt_cfg.get("log_file") else None)
        ),
        log_level="DEBUG" if debug else (cli.get("log_level") or rt_cfg.get("log_level", "INFO")),
    )

    gimbal_cfg = yaml_data.get("gimbal", {})

    gimbal = GimbalConfig(
        invert_pan=cli.get("invert_pan", False) or gimbal_cfg.get("invert_pan", False),
        invert_tilt=cli.get("invert_tilt", False) or gimbal_cfg.get("invert_tilt", False),
        pan_limit_deg=float(gimbal_cfg.get("pan_limit_deg", 90.0)),
        tilt_limit_deg=float(gimbal_cfg.get("tilt_limit_deg", 90.0)),
        kp=float(gimbal_cfg.get("kp", 1.2)),
        ki=float(gimbal_cfg.get("ki", 0.0)),
        kd=float(gimbal_cfg.get("kd", 0.6)),
        deadband_deg=float(gimbal_cfg.get("deadband_deg", 1.2)),
        integral_decay_rate=float(gimbal_cfg.get("integral_decay_rate", 1.0)),
        slew_limit_dps=float(gimbal_cfg.get("slew_limit_dps", 25.0)),
        tilt_scale=float(gimbal_cfg.get("tilt_scale", 0.45)),
    )

    paths = RuntimePaths(workspace_root=Path.cwd())

    laser_cfg = yaml_data.get("laser", {})
    laser = LaserConfig(
        enabled=laser_cfg.get("enabled", True),
        hue_low_upper=laser_cfg.get("hue_low_upper", 5),
        hue_high_lower=laser_cfg.get("hue_high_lower", 175),
        sat_min=laser_cfg.get("sat_min", 120),
        val_min=laser_cfg.get("val_min", 160),
        min_area=float(laser_cfg.get("min_area", 2.0)),
        max_area=float(laser_cfg.get("max_area", 800.0)),
        min_circularity=float(laser_cfg.get("min_circularity", 0.1)),
        roi_radius_px=float(laser_cfg.get("roi_radius_px", 150.0)),
        max_jump_px=float(laser_cfg.get("max_jump_px", 50.0)),
        min_brightness=int(laser_cfg.get("min_brightness", 200)),
    )

    relay_cfg = yaml_data.get("relay", {})
    relay = RelayConfig(
        pulse_ms=int(relay_cfg.get("pulse_ms", 500)),
    )

    servo_ctrl_cfg = yaml_data.get("servo_control", {})
    servo_control = _overlay_dataclass(ServoControlConfig(), servo_ctrl_cfg)

    pf_cfg = yaml_data.get("preflight", {})
    preflight = _overlay_dataclass(PreflightConfig(), pf_cfg)

    laser_boresight = _build_laser_boresight(yaml_data.get("laser_boresight", {}), paths)

    config = PipelineConfig(
        mode=mode,
        target=target,
        source=source,
        camera=camera,
        comms=comms,
        tracking=tracking,
        models=models,
        paths=paths,
        flags=flags,
        gimbal=gimbal,
        laser=laser,
        laser_boresight=laser_boresight,
        preflight=preflight,
        relay=relay,
        servo_control=servo_control,
    )
    _validate_config(config)
    return config


def _build_laser_boresight(
    yaml_section: dict[str, Any],
    paths: RuntimePaths,
) -> LaserBoresightConfig:
    """Resolve the laser boresight offset from YAML + servo_limits.json.

    ``source: auto`` (default) reads ``calibration_data/servo_limits.json`` and
    falls back to inline values on miss.  ``source: inline`` always uses the
    YAML values (useful for reproducible tests).
    """
    source = str(yaml_section.get("source", "auto")).lower()
    inline = LaserBoresightConfig(
        pan_offset_deg=float(yaml_section.get("pan_offset_deg", 0.0)),
        tilt_offset_deg=float(yaml_section.get("tilt_offset_deg", 0.0)),
    )
    if source == "inline":
        return inline
    if source != "auto":
        LOGGER.warning("Unknown laser_boresight.source '%s' (using 'auto')", source)
    json_path = paths.resolve_path(paths.calibration_dir) / "servo_limits.json"
    persisted = load_boresight(json_path)
    if persisted.pan_offset_deg == 0.0 and persisted.tilt_offset_deg == 0.0:
        return inline
    return LaserBoresightConfig(
        pan_offset_deg=persisted.pan_offset_deg,
        tilt_offset_deg=persisted.tilt_offset_deg,
    )


if __name__ == "__main__":
    main()
