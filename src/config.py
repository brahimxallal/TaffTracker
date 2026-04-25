from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Literal

Mode = Literal["camera", "video"]
TargetKind = Literal["human", "dog"]
CommChannel = Literal["serial", "udp", "auto"]
Precision = Literal["fp16", "int8"]
Backend = Literal["auto", "msmf", "dshow"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Orientation(str, Enum):
    PORTRAIT = "portrait"  # Portrait stream letterboxed in landscape feed (old DroidCam portrait)
    LANDSCAPE_NATIVE = (
        "landscape_native"  # Phone physically horizontal, USB right; full-frame landscape feed
    )


@dataclass(frozen=True)
class CameraConfig:
    width: int = 640
    height: int = 640
    fps: int = 60
    backend: Backend = "auto"
    buffer_size: int = 1
    orientation: Orientation = Orientation.LANDSCAPE_NATIVE
    fov: float | None = None
    capture_width: int = 1920  # Native camera resolution to request
    capture_height: int = 1080  # (letterbox rescales to width x height)

    @property
    def portrait_mode(self) -> bool:
        """Legacy shim: some capture code still reads this boolean."""
        return self.orientation == Orientation.PORTRAIT


@dataclass(frozen=True)
class LaserBoresightConfig:
    """Static angular offset between the laser emitter and the camera optical axis.

    Applied at the servo output stage so that commanded servo angles place the
    laser dot (not the camera center) onto the tracked target pixel.
    """

    pan_offset_deg: float = 0.0
    tilt_offset_deg: float = 0.0


@dataclass(frozen=True)
class CommConfig:
    channel: CommChannel = "serial"
    serial_port: str = "COM4"
    baud_rate: int = 921600
    udp_host: str = "192.168.4.1"
    udp_port: int = 6000
    udp_redundancy: int = 2
    enabled: bool = True


@dataclass(frozen=True)
class PostprocessConfig:
    mad_scale: float = 3.0
    conf_ema_alpha: float = 0.2
    mad_min_points: int = 3
    high_quality_min_keypoints: int = 3
    high_quality_min_conf: float = 0.5
    medium_quality_alpha_reduction: float = 0.10
    nose_priority_threshold: float = 0.5
    nose_priority_blend: float = 0.45
    body_fallback_head_weight: float = 0.6


@dataclass(frozen=True)
class KalmanConfig:
    innovation_gate_sigma: float = 5.0
    adaptive_r_min: float = 8.0
    adaptive_r_max: float = 20.0
    adaptive_r_speed_thresh: float = 100.0
    adaptive_q_max_multiplier: float = 2.0
    adaptive_q_speed_thresh: float = 300.0
    adaptive_q_speed_ceil: float = 600.0
    max_consecutive_predictions: int = 30
    max_consecutive_gated: int = 3
    oru_cache_size: int = 5
    prediction_decay_start: int = 8
    prediction_velocity_decay: float = 0.85


@dataclass(frozen=True)
class AdaptiveConfig:
    detection_window: int = 120
    speed_window: int = 60
    min_detection_samples: int = 30
    min_speed_samples: int = 10
    reliability_low: float = 0.70
    reliability_high: float = 0.90
    conf_floor: float = 0.30
    conf_ceiling: float = 0.55
    conf_reduction: float = 0.05
    conf_raise: float = 0.02
    fast_speed_thresh: float = 200.0
    slow_speed_thresh: float = 50.0
    hold_time_min_s: float = 0.3
    hold_time_max_s: float = 1.0
    hold_time_fast_reduction: float = 0.20
    hold_time_slow_increase: float = 0.15
    slew_freeze_threshold_deg_s: float = 80.0
    slew_freeze_hold_frames: int = 10


@dataclass(frozen=True)
class SmoothingConfig:
    display_mincutoff: float = 0.1
    display_beta: float = 0.05
    servo_mincutoff: float = 10.0
    servo_beta: float = 0.2
    dcutoff: float = 1.0


@dataclass(frozen=True)
class TrackingConfig:
    confidence_threshold: float = 0.45
    hold_time_s: float = 0.65
    max_lost_frames: int = 30
    process_noise: float = 2.5
    measurement_noise: float = 5.0
    tracker_match_threshold: float = 0.55
    tracker_track_threshold: float = 0.40
    tracker_birth_min_hits: int = 2
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)


@dataclass(frozen=True)
class ModelConfig:
    person_model_path: Path = Path("models/yolo26n-person-17pose.pt")
    dog_model_path: Path = Path("models/Enhanceddog/best.pt")
    engine_dir: Path = Path("engines")
    image_size: int = 640
    precision: Precision = "fp16"

    @property
    def person_engine_path(self) -> Path:
        return self._resolve_engine("yolo26n-person-17pose")

    @property
    def dog_engine_path(self) -> Path:
        return self._resolve_engine("enhanced-dog-24pose")

    def _resolve_engine(self, stem: str) -> Path:
        """Resolve engine path with INT8 preference and FP16 fallback."""
        if self.precision == "int8":
            int8_path = self.engine_dir / f"{stem}-int8.engine"
            if int8_path.exists():
                return int8_path
        return self.engine_dir / f"{stem}.engine"


@dataclass(frozen=True)
class RuntimePaths:
    workspace_root: Path = Path(".")
    calibration_dir: Path = Path("calibration_data")
    log_dir: Path = Path("logs")
    dog_pose_schema_path: Path = Path("models/dog-pose.yaml")

    def resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else self.workspace_root / path

    def resolved_dog_pose_schema_path(self) -> Path:
        return self.resolve_path(self.dog_pose_schema_path)

    def resolved_log_dir(self) -> Path:
        return self.resolve_path(self.log_dir)

    def profiler_csv_path(self, mode: Mode, target: TargetKind) -> Path:
        return self.resolved_log_dir() / f"profile_{mode}_{target}.csv"

    def profiler_summary_path(self) -> Path:
        return self.resolved_log_dir() / "profiler_summary.csv"

    def inference_metrics_path(self) -> Path:
        return self.resolved_log_dir() / "inference_metrics.json"

    def output_metrics_path(self) -> Path:
        return self.resolved_log_dir() / "output_metrics.json"


@dataclass(frozen=True)
class GimbalConfig:
    invert_pan: bool = False
    invert_tilt: bool = False
    pan_limit_deg: float = 90.0  # Max absolute angle PC will send for pan (MG996R ±90°)
    tilt_limit_deg: float = 90.0  # Max absolute angle PC will send for tilt (MG996R ±90°)
    kp: float = 1.2  # Approach gain (°/s per ° error; integral-type accumulator)
    ki: float = 0.0  # Reserved (unused in incremental controller)
    kd: float = 0.6  # Derivative damping using LP-filtered error finite-diff
    deadband_deg: float = 1.2  # Suppress corrections below this error (degrees)
    integral_decay_rate: float = 1.0  # Decay rate (/s) for accumulated command inside deadband
    slew_limit_dps: float = 25.0  # Max command change rate (°/s)
    tilt_scale: float = (
        0.45  # Tilt gain multiplier (camera-on-gimbal tilt has stronger ego-motion feedback)
    )
    velocity_feedforward_gain: float = 0.0  # FF gain on Kalman target velocity (0 = disabled)
    kp_far: float | None = None  # kp when error > gain_schedule_threshold_deg (None → kp)
    kp_near: float | None = None  # kp when error <= gain_schedule_threshold_deg (None → kp)
    gain_schedule_threshold_deg: float = 3.0  # Boundary between far/near kp
    predictive_lead_s: float = 0.0  # Velocity-based lead time (0 = disabled)

    @property
    def effective_kp_far(self) -> float:
        return self.kp_far if self.kp_far is not None else self.kp

    @property
    def effective_kp_near(self) -> float:
        return self.kp_near if self.kp_near is not None else self.kp


@dataclass(frozen=True)
class LaserConfig:
    enabled: bool = True
    hue_low_upper: int = 5
    hue_high_lower: int = 175
    sat_min: int = 120
    val_min: int = 160
    min_area: float = 2.0
    max_area: float = 800.0
    min_circularity: float = 0.1
    roi_radius_px: float = 150.0
    max_jump_px: float = 50.0
    min_brightness: int = 200


@dataclass(frozen=True)
class PreflightConfig:
    enabled: bool = True
    window_size: int = 60
    brightness_stddev_warn: float = 18.0
    sharpness_drop_warn: float = 0.4
    warn_cooldown_s: float = 10.0
    # Every N processed frames, the inference loop flushes a rolling
    # "frame_drops" sample into the profiler. Larger values mean less
    # profiler chatter; smaller values mean tighter drop-rate telemetry.
    frame_drop_sample_window: int = 120


@dataclass(frozen=True)
class RelayConfig:
    pulse_ms: int = 500


@dataclass(frozen=True)
class ServoControlConfig:
    """Output-stage servo controller tuning constants."""

    # Minimum |response velocity| emitted when the user is driving the
    # gimbal manually — keeps the firmware from quantizing a small stick
    # deflection to zero.
    manual_response_velocity_floor_dps: float = 80.0
    # Low-pass filter on the error-derivative term (0 = heavy smoothing,
    # 1 = raw finite difference). Derivative is taken on the angular
    # error, NOT Kalman velocity, because Kalman velocity is polluted by
    # ego-motion on camera-on-gimbal mounts.
    derivative_filter_alpha: float = 0.35
    # Velocity smoother for the predictive-lead path. Disabled by default;
    # enabling is meaningful only when ``gimbal.predictive_lead_s > 0`` —
    # see :mod:`src.output.velocity_smoother`.
    velocity_smoother_enabled: bool = False
    velocity_smoother_alpha: float = 0.4
    velocity_smoother_deadband_dps: float = 5.0
    velocity_smoother_slew_dps_per_s: float = 0.0


@dataclass(frozen=True)
class RuntimeFlags:
    debug: bool = False
    headless: bool = False
    profile: bool = False
    log_file: Path | None = None
    log_level: LogLevel = "INFO"
    # When true, prefer the GPU letterbox path
    # (:func:`src.inference.gpu_preprocess.gpu_letterbox`) over the inline
    # CPU resize+pad in the capture process. Off by default until a
    # benchmark on the deployed hardware shows it wins; flipping this
    # without measuring can REGRESS latency on small frames where the
    # H2D transfer cost outweighs the kernel speedup.
    gpu_preprocess: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    mode: Mode
    target: TargetKind
    source: str
    camera: CameraConfig = field(default_factory=CameraConfig)
    comms: CommConfig = field(default_factory=CommConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    paths: RuntimePaths = field(default_factory=RuntimePaths)
    flags: RuntimeFlags = field(default_factory=RuntimeFlags)
    gimbal: GimbalConfig = field(default_factory=GimbalConfig)
    laser: LaserConfig = field(default_factory=LaserConfig)
    laser_boresight: LaserBoresightConfig = field(default_factory=LaserBoresightConfig)
    preflight: PreflightConfig = field(default_factory=PreflightConfig)
    relay: RelayConfig = field(default_factory=RelayConfig)
    servo_control: ServoControlConfig = field(default_factory=ServoControlConfig)


def default_tracking_config(target: TargetKind) -> TrackingConfig:
    """Return target-specific tracking defaults.

    Dogs have different motion profiles from humans: lower center of gravity,
    more abrupt acceleration, different typical speeds. This factory provides
    tuned defaults for each target while keeping the same structure.
    """
    if target == "dog":
        return TrackingConfig(
            confidence_threshold=0.40,
            hold_time_s=0.5,
            max_lost_frames=120,
            kalman=KalmanConfig(
                innovation_gate_sigma=6.0,
                adaptive_r_speed_thresh=120.0,
                adaptive_q_speed_ceil=500.0,
                max_consecutive_predictions=60,
                prediction_decay_start=6,
                prediction_velocity_decay=0.80,
            ),
            adaptive=AdaptiveConfig(
                fast_speed_thresh=150.0,
                slow_speed_thresh=30.0,
                conf_floor=0.25,
            ),
            smoothing=SmoothingConfig(
                servo_mincutoff=12.0,
                servo_beta=0.3,
            ),
        )
    # Human defaults
    return TrackingConfig()


def adapt_tracking_for_fps(config: TrackingConfig, actual_fps: float) -> TrackingConfig:
    """Scale tracking parameters relative to a nominal 60 FPS baseline.

    Higher FPS → tighter Kalman (lower Q), more lost frames allowed.
    Lower FPS → looser Kalman (higher Q), fewer lost frames.
    """
    if actual_fps <= 0:
        return config
    ratio = actual_fps / 60.0
    return replace(
        config,
        process_noise=config.process_noise / max(ratio, 0.1),
        max_lost_frames=max(5, int(config.max_lost_frames * ratio)),
        hold_time_s=config.hold_time_s,  # keep hold time absolute (seconds)
    )
