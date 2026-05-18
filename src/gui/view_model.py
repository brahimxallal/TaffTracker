from __future__ import annotations

from dataclasses import dataclass, replace
from math import degrees
from typing import Literal

from src.config import (
    Backend,
    Mode,
    PipelineConfig,
    Precision,
    SourceBackend,
    TargetKind,
    adapt_tracking_for_fps,
    default_tracking_config,
)
from src.shared.telemetry import RuntimeTelemetry

PanelId = Literal[
    "live",
    "tracking",
    "gimbal",
    "calibration",
    "diagnostics",
    "settings",
]
ControlScope = Literal["startup", "runtime", "safety", "calibration"]


@dataclass(frozen=True, slots=True)
class PanelSpec:
    panel_id: PanelId
    title: str
    purpose: str


@dataclass(frozen=True, slots=True)
class RuntimeControlSpec:
    label: str
    path: str
    scope: ControlScope
    requires_restart: bool
    transaction_required: bool
    safety_critical: bool = False


@dataclass(frozen=True, slots=True)
class DashboardShellSpec:
    title: str
    panels: tuple[PanelSpec, ...]
    status_fields: tuple[str, ...]
    quick_actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DashboardConfigSummary:
    mode: str
    target: str
    source: str
    camera: str
    comms: str
    laser: str
    profile: str


@dataclass(frozen=True, slots=True)
class DiagnosticRow:
    metric: str
    value: str


@dataclass(frozen=True, slots=True)
class LaunchSettings:
    mode: Mode
    target: TargetKind
    source: str
    comms_enabled: bool
    camera_width: int
    camera_height: int
    camera_fps: int
    camera_source_backend: SourceBackend
    camera_backend: Backend
    camera_fov: float | None
    model_precision: Precision
    model_image_size: int


@dataclass(frozen=True, slots=True)
class TuningSettings:
    tracking_confidence_threshold: float
    tracking_hold_time_s: float
    gimbal_kp: float
    gimbal_ki: float
    gimbal_kd: float
    gimbal_deadband_deg: float
    gimbal_slew_limit_dps: float
    gimbal_kp_near: float | None
    gimbal_kp_far: float | None
    gimbal_predictive_lead_s: float
    laser_startup_enabled: bool
    relay_pulse_ms: int
    boresight_pan_offset_deg: float
    boresight_tilt_offset_deg: float


def build_dashboard_shell_spec() -> DashboardShellSpec:
    return DashboardShellSpec(
        title="TaffTracker Control Plane",
        panels=(
            PanelSpec("live", "Live", "Video observer and process controls"),
            PanelSpec("tracking", "Tracking", "Target lock, centroid, angle, and filter state"),
            PanelSpec("gimbal", "Gimbal", "Servo limits and runtime control tuning"),
            PanelSpec(
                "calibration", "Calibration", "Camera, mount offset, and boresight workflows"
            ),
            PanelSpec(
                "diagnostics", "Diagnostics", "Latency, packet, watchdog, and transport health"
            ),
            PanelSpec("settings", "Settings", "Launch configuration and environment validation"),
        ),
        status_fields=(
            "mode",
            "target",
            "source",
            "fps",
            "latency_ms",
            "transport",
            "process_state",
            "safety",
        ),
        quick_actions=(
            "start",
            "stop",
            "emergency_stop",
            "relock",
            "cycle_target",
            "manual_mode",
            "laser_toggle",
            "relay_pulse",
        ),
    )


def runtime_control_specs() -> tuple[RuntimeControlSpec, ...]:
    return (
        RuntimeControlSpec("Mode", "mode", "startup", True, False),
        RuntimeControlSpec("Target", "target", "startup", True, False),
        RuntimeControlSpec("Source", "source", "startup", True, False),
        RuntimeControlSpec("Camera width", "camera.width", "startup", True, False),
        RuntimeControlSpec("Camera height", "camera.height", "startup", True, False),
        RuntimeControlSpec(
            "Camera source backend", "camera.source_backend", "startup", True, False
        ),
        RuntimeControlSpec("Camera backend", "camera.backend", "startup", True, False),
        RuntimeControlSpec("Model precision", "models.precision", "startup", True, False),
        RuntimeControlSpec("Model image size", "models.image_size", "startup", True, False),
        RuntimeControlSpec(
            "Tracking confidence", "tracking.confidence_threshold", "runtime", False, True
        ),
        RuntimeControlSpec("Hold time", "tracking.hold_time_s", "runtime", False, True),
        RuntimeControlSpec("Gimbal kp", "gimbal.kp", "runtime", False, True),
        RuntimeControlSpec("Gimbal ki", "gimbal.ki", "runtime", False, True),
        RuntimeControlSpec("Gimbal kd", "gimbal.kd", "runtime", False, True),
        RuntimeControlSpec("Deadband", "gimbal.deadband_deg", "runtime", False, True),
        RuntimeControlSpec("Slew limit", "gimbal.slew_limit_dps", "runtime", False, True),
        RuntimeControlSpec("Near kp", "gimbal.kp_near", "runtime", False, True),
        RuntimeControlSpec("Far kp", "gimbal.kp_far", "runtime", False, True),
        RuntimeControlSpec("Predictive lead", "gimbal.predictive_lead_s", "runtime", False, True),
        RuntimeControlSpec(
            "Velocity smoothing",
            "servo_control.velocity_smoother_enabled",
            "runtime",
            False,
            True,
        ),
        RuntimeControlSpec("Laser enable", "laser.enabled", "safety", False, True, True),
        RuntimeControlSpec("Relay pulse", "relay.pulse_ms", "safety", False, True, True),
        RuntimeControlSpec(
            "Mount pan offset",
            "laser_boresight.pan_offset_deg",
            "calibration",
            False,
            True,
        ),
        RuntimeControlSpec(
            "Mount tilt offset",
            "laser_boresight.tilt_offset_deg",
            "calibration",
            False,
            True,
        ),
    )


def summarize_config(config: PipelineConfig) -> DashboardConfigSummary:
    comms = _format_comms(config)
    camera = (
        f"{config.camera.width}x{config.camera.height}@{config.camera.fps}\n"
        f"{config.camera.source_backend}/{config.camera.backend}"
    )
    if config.camera.fov is not None:
        camera = f"{camera} fov={config.camera.fov:.1f}"
    return DashboardConfigSummary(
        mode=config.mode,
        target=config.target,
        source=config.source,
        camera=camera,
        comms=comms,
        laser="enabled" if config.laser.enabled else "disabled",
        profile="enabled" if config.flags.profile else "disabled",
    )


def launch_settings_from_config(config: PipelineConfig) -> LaunchSettings:
    return LaunchSettings(
        mode=config.mode,
        target=config.target,
        source=config.source,
        comms_enabled=config.comms.enabled,
        camera_width=config.camera.width,
        camera_height=config.camera.height,
        camera_fps=config.camera.fps,
        camera_source_backend=config.camera.source_backend,
        camera_backend=config.camera.backend,
        camera_fov=config.camera.fov,
        model_precision=config.models.precision,
        model_image_size=config.models.image_size,
    )


def apply_launch_settings(
    config: PipelineConfig,
    settings: LaunchSettings,
) -> PipelineConfig:
    source = settings.source.strip() or suggest_source_for_mode(settings.mode, config.source)
    camera = replace(
        config.camera,
        width=settings.camera_width,
        height=settings.camera_height,
        fps=settings.camera_fps,
        source_backend=settings.camera_source_backend,
        backend=settings.camera_backend,
        fov=settings.camera_fov,
    )
    tracking = config.tracking
    if settings.target != config.target:
        tracking = adapt_tracking_for_fps(
            default_tracking_config(settings.target),
            settings.camera_fps,
        )
    return replace(
        config,
        mode=settings.mode,
        target=settings.target,
        source=source,
        camera=camera,
        comms=replace(config.comms, enabled=settings.comms_enabled),
        models=replace(
            config.models,
            precision=settings.model_precision,
            image_size=settings.model_image_size,
        ),
        tracking=tracking,
    )


def validate_launch_settings(settings: LaunchSettings) -> tuple[str, ...]:
    errors: list[str] = []
    if not settings.source.strip():
        errors.append("source is required")
    if settings.mode == "camera" and settings.camera_fov is None:
        errors.append("camera mode requires FOV > 0 deg")
    if settings.camera_width < 160 or settings.camera_height < 160:
        errors.append("camera width/height must be at least 160 px")
    if settings.camera_fps <= 0:
        errors.append("camera FPS must be greater than 0")
    if settings.model_image_size <= 0 or settings.model_image_size % 32 != 0:
        errors.append("model image size must be a positive multiple of 32")
    return tuple(errors)


def tuning_settings_from_config(config: PipelineConfig) -> TuningSettings:
    return TuningSettings(
        tracking_confidence_threshold=config.tracking.confidence_threshold,
        tracking_hold_time_s=config.tracking.hold_time_s,
        gimbal_kp=config.gimbal.kp,
        gimbal_ki=config.gimbal.ki,
        gimbal_kd=config.gimbal.kd,
        gimbal_deadband_deg=config.gimbal.deadband_deg,
        gimbal_slew_limit_dps=config.gimbal.slew_limit_dps,
        gimbal_kp_near=config.gimbal.kp_near,
        gimbal_kp_far=config.gimbal.kp_far,
        gimbal_predictive_lead_s=config.gimbal.predictive_lead_s,
        laser_startup_enabled=config.laser.enabled,
        relay_pulse_ms=config.relay.pulse_ms,
        boresight_pan_offset_deg=config.laser_boresight.pan_offset_deg,
        boresight_tilt_offset_deg=config.laser_boresight.tilt_offset_deg,
    )


def apply_tuning_settings(
    config: PipelineConfig,
    settings: TuningSettings,
) -> PipelineConfig:
    return replace(
        config,
        tracking=replace(
            config.tracking,
            confidence_threshold=settings.tracking_confidence_threshold,
            hold_time_s=settings.tracking_hold_time_s,
        ),
        gimbal=replace(
            config.gimbal,
            kp=settings.gimbal_kp,
            ki=settings.gimbal_ki,
            kd=settings.gimbal_kd,
            deadband_deg=settings.gimbal_deadband_deg,
            slew_limit_dps=settings.gimbal_slew_limit_dps,
            kp_near=settings.gimbal_kp_near,
            kp_far=settings.gimbal_kp_far,
            predictive_lead_s=settings.gimbal_predictive_lead_s,
        ),
        laser=replace(config.laser, enabled=settings.laser_startup_enabled),
        relay=replace(config.relay, pulse_ms=settings.relay_pulse_ms),
        laser_boresight=replace(
            config.laser_boresight,
            pan_offset_deg=settings.boresight_pan_offset_deg,
            tilt_offset_deg=settings.boresight_tilt_offset_deg,
        ),
    )


def suggest_source_for_mode(mode: Mode, current_source: str) -> str:
    source = current_source.strip()
    if mode == "camera":
        return "0" if not source or _looks_like_file_video_source(source) else source
    if not source or source == "0":
        return "videos/person.mp4"
    return source


def runtime_transaction_paths() -> tuple[str, ...]:
    return tuple(spec.path for spec in runtime_control_specs() if spec.transaction_required)


def safety_action_names() -> tuple[str, ...]:
    spec = build_dashboard_shell_spec()
    return tuple(
        action for action in spec.quick_actions if "emergency" in action or "laser" in action
    )


def build_runtime_diagnostic_rows(
    telemetry: RuntimeTelemetry | None,
) -> tuple[DiagnosticRow, ...]:
    if telemetry is None:
        return (
            DiagnosticRow("FPS", "n/a"),
            DiagnosticRow("Total latency", "n/a"),
            DiagnosticRow("Inference", "n/a"),
            DiagnosticRow("Tracking", "n/a"),
            DiagnosticRow("Postprocess", "n/a"),
            DiagnosticRow("Wait", "n/a"),
            DiagnosticRow("Lock", "n/a"),
            DiagnosticRow("Link", "n/a"),
        )

    lock_pct = (telemetry.lock_frames / max(1, telemetry.total_frames)) * 100.0
    drop_pct = (telemetry.display_drops / max(1, telemetry.display_total)) * 100.0
    state = "locked" if telemetry.target_acquired else telemetry.state_source
    return (
        DiagnosticRow("Frame", str(telemetry.frame_id)),
        DiagnosticRow("Target", telemetry.target_kind),
        DiagnosticRow("State", state),
        DiagnosticRow("Track ID", "n/a" if telemetry.track_id is None else str(telemetry.track_id)),
        DiagnosticRow("Confidence", f"{telemetry.confidence:.2f}"),
        DiagnosticRow("Pixel", _format_pair(telemetry.filtered_pixel, suffix=" px")),
        DiagnosticRow("Angles", _format_angles_deg(telemetry.filtered_angles)),
        DiagnosticRow("FPS", f"{telemetry.fps:.1f}"),
        DiagnosticRow("Total latency", f"{telemetry.total_latency_ms:.1f} ms"),
        DiagnosticRow("Inference", f"{telemetry.inference_ms:.1f} ms"),
        DiagnosticRow("Tracking", f"{telemetry.tracking_ms:.1f} ms"),
        DiagnosticRow("Postprocess", f"{telemetry.postprocess_ms:.1f} ms"),
        DiagnosticRow("Wait", f"{telemetry.wait_ms:.1f} ms"),
        DiagnosticRow(
            "Lock", f"{lock_pct:.0f}% ({telemetry.lock_frames}/{telemetry.total_frames})"
        ),
        DiagnosticRow("Link", telemetry.transport_status),
        DiagnosticRow("Packet seq", str(telemetry.packet_sequence)),
        DiagnosticRow(
            "Display drops",
            f"{telemetry.display_drops}/{max(1, telemetry.display_total)} ({drop_pct:.1f}%)",
        ),
    )


def _format_comms(config: PipelineConfig) -> str:
    if not config.comms.enabled:
        return "disabled"
    if config.comms.channel == "udp":
        return f"udp {config.comms.udp_host}:{config.comms.udp_port}"
    if config.comms.channel == "auto":
        return "auto"
    return f"serial {config.comms.serial_port}@{config.comms.baud_rate}"


def _format_pair(pair: tuple[float, float] | None, *, suffix: str = "") -> str:
    if pair is None:
        return "n/a"
    return f"{pair[0]:.1f}, {pair[1]:.1f}{suffix}"


def _format_angles_deg(angles_rad: tuple[float, float] | None) -> str:
    if angles_rad is None:
        return "n/a"
    return f"{degrees(angles_rad[0]):.2f}, {degrees(angles_rad[1]):.2f} deg"


def _looks_like_file_video_source(source: str) -> bool:
    return source.lower().endswith((".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm", ".wmv"))
