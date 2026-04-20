"""Phase A: Configuration consolidation tests."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.config import (
    AdaptiveConfig,
    CameraConfig,
    GimbalConfig,
    KalmanConfig,
    ModelConfig,
    PipelineConfig,
    PostprocessConfig,
    TrackingConfig,
    adapt_tracking_for_fps,
    default_tracking_config,
)


# --- PostprocessConfig defaults match former hardcoded constants ---


@pytest.mark.unit
def test_postprocess_config_defaults() -> None:
    cfg = PostprocessConfig()
    assert cfg.mad_scale == 3.0
    assert cfg.conf_ema_alpha == 0.2
    assert cfg.mad_min_points == 3
    assert cfg.high_quality_min_keypoints == 3
    assert cfg.high_quality_min_conf == 0.5
    assert cfg.medium_quality_alpha_reduction == 0.10
    assert cfg.nose_priority_threshold == 0.5
    assert cfg.nose_priority_blend == 0.45
    assert cfg.body_fallback_head_weight == 0.6


# --- KalmanConfig defaults match former hardcoded constants ---


@pytest.mark.unit
def test_kalman_config_defaults() -> None:
    cfg = KalmanConfig()
    assert cfg.innovation_gate_sigma == 5.0
    assert cfg.adaptive_r_min == 8.0
    assert cfg.adaptive_r_max == 20.0
    assert cfg.adaptive_r_speed_thresh == 100.0
    assert cfg.adaptive_q_max_multiplier == 2.0
    assert cfg.adaptive_q_speed_thresh == 300.0
    assert cfg.adaptive_q_speed_ceil == 600.0
    assert cfg.max_consecutive_predictions == 30
    assert cfg.max_consecutive_gated == 3
    assert cfg.oru_cache_size == 5
    assert cfg.prediction_decay_start == 8
    assert cfg.prediction_velocity_decay == 0.85


# --- AdaptiveConfig defaults match former hardcoded constants ---


@pytest.mark.unit
def test_adaptive_config_defaults() -> None:
    cfg = AdaptiveConfig()
    assert cfg.detection_window == 120
    assert cfg.speed_window == 60
    assert cfg.min_detection_samples == 30
    assert cfg.min_speed_samples == 10
    assert cfg.reliability_low == 0.70
    assert cfg.reliability_high == 0.90
    assert cfg.conf_floor == 0.30
    assert cfg.conf_ceiling == 0.55
    assert cfg.fast_speed_thresh == 200.0
    assert cfg.slow_speed_thresh == 50.0
    assert cfg.hold_time_min_s == 0.3
    assert cfg.hold_time_max_s == 1.0


# --- TrackingConfig nests all sub-configs ---


@pytest.mark.unit
def test_tracking_config_nests_sub_configs() -> None:
    tc = TrackingConfig()
    assert isinstance(tc.postprocess, PostprocessConfig)
    assert isinstance(tc.kalman, KalmanConfig)
    assert isinstance(tc.adaptive, AdaptiveConfig)


@pytest.mark.unit
def test_tracking_config_replace_preserves_nested() -> None:
    tc = TrackingConfig()
    modified = replace(tc, process_noise=5.0)
    assert modified.process_noise == 5.0
    assert modified.kalman == tc.kalman  # nested unchanged


@pytest.mark.unit
def test_tracking_config_frozen() -> None:
    tc = TrackingConfig()
    with pytest.raises(AttributeError):
        tc.process_noise = 99.0  # type: ignore[misc]


# --- Config immutability ---


@pytest.mark.unit
def test_all_configs_frozen() -> None:
    pp = PostprocessConfig()
    with pytest.raises(AttributeError):
        pp.mad_scale = 99.0  # type: ignore[misc]
    kc = KalmanConfig()
    with pytest.raises(AttributeError):
        kc.innovation_gate_sigma = 99.0  # type: ignore[misc]
    ac = AdaptiveConfig()
    with pytest.raises(AttributeError):
        ac.reliability_low = 99.0  # type: ignore[misc]


# --- adapt_tracking_for_fps preserves nested configs ---


@pytest.mark.unit
def test_adapt_fps_preserves_nested_configs() -> None:
    base = TrackingConfig()
    adapted = adapt_tracking_for_fps(base, 30.0)
    assert adapted.postprocess == base.postprocess
    assert adapted.kalman == base.kalman
    assert adapted.adaptive == base.adaptive
    assert adapted.smoothing == base.smoothing


@pytest.mark.unit
def test_adapt_fps_with_custom_nested() -> None:
    custom_kalman = KalmanConfig(innovation_gate_sigma=6.0)
    base = TrackingConfig(kalman=custom_kalman)
    adapted = adapt_tracking_for_fps(base, 30.0)
    assert adapted.kalman.innovation_gate_sigma == 6.0  # preserved


# --- PipelineConfig composes correctly ---


@pytest.mark.unit
def test_pipeline_config_carries_tracking_sub_configs() -> None:
    cfg = PipelineConfig(mode="camera", target="human", source="0")
    assert cfg.tracking.kalman.innovation_gate_sigma == 5.0
    assert cfg.tracking.postprocess.mad_scale == 3.0
    assert cfg.tracking.adaptive.reliability_low == 0.70


# --- Per-target config factory ---


@pytest.mark.unit
def test_default_human_config_matches_base() -> None:
    human = default_tracking_config("human")
    # Human defaults are plain TrackingConfig()
    assert human == TrackingConfig()


@pytest.mark.unit
def test_default_dog_config_differs_from_human() -> None:
    human = default_tracking_config("human")
    dog = default_tracking_config("dog")
    assert dog != human
    assert dog.confidence_threshold < human.confidence_threshold
    assert dog.kalman.innovation_gate_sigma > human.kalman.innovation_gate_sigma
    assert dog.adaptive.fast_speed_thresh < human.adaptive.fast_speed_thresh
    assert dog.adaptive.conf_floor < human.adaptive.conf_floor


@pytest.mark.unit
def test_default_dog_config_composes_with_fps_adapt() -> None:
    dog = default_tracking_config("dog")
    adapted = adapt_tracking_for_fps(dog, 30.0)
    assert adapted.process_noise > dog.process_noise
    assert adapted.kalman == dog.kalman  # nested preserved


@pytest.mark.unit
def test_default_config_respects_cli_override() -> None:
    dog = default_tracking_config("dog")
    overridden = replace(dog, confidence_threshold=0.60)
    assert overridden.confidence_threshold == 0.60
    assert overridden.kalman == dog.kalman  # nested unchanged


# --- GimbalConfig ---


@pytest.mark.unit
def test_gimbal_config_defaults() -> None:
    gc = GimbalConfig()
    assert gc.invert_pan is False
    assert gc.invert_tilt is False


@pytest.mark.unit
def test_gimbal_config_frozen() -> None:
    gc = GimbalConfig()
    with pytest.raises(AttributeError):
        gc.invert_pan = True  # type: ignore[misc]


@pytest.mark.unit
def test_pipeline_config_carries_gimbal() -> None:
    cfg = PipelineConfig(mode="camera", target="human", source="0")
    assert isinstance(cfg.gimbal, GimbalConfig)
    assert cfg.gimbal.invert_pan is False


@pytest.mark.unit
def test_pipeline_config_custom_gimbal() -> None:
    cfg = PipelineConfig(
        mode="camera", target="human", source="0",
        gimbal=GimbalConfig(invert_tilt=True),
    )
    assert cfg.gimbal.invert_tilt is True


# --- CameraConfig.fov ---


@pytest.mark.unit
def test_camera_config_fov_default_none() -> None:
    cc = CameraConfig()
    assert cc.fov is None


@pytest.mark.unit
def test_camera_config_fov_set() -> None:
    cc = CameraConfig(fov=66.2)
    assert cc.fov == 66.2


# --- ModelConfig precision & engine path resolution ---


@pytest.mark.unit
def test_model_config_default_precision() -> None:
    mc = ModelConfig()
    assert mc.precision == "fp16"


@pytest.mark.unit
def test_model_config_fp16_engine_path() -> None:
    mc = ModelConfig(engine_dir=Path("engines"), precision="fp16")
    assert mc.person_engine_path == Path("engines/yolo26n-person-17pose.engine")
    assert mc.dog_engine_path == Path("engines/enhanced-dog-24pose.engine")


@pytest.mark.unit
def test_model_config_int8_engine_path_fallback(tmp_path: Path) -> None:
    """INT8 falls back to FP16 engine when INT8 engine doesn't exist."""
    mc = ModelConfig(engine_dir=tmp_path, precision="int8")
    # No INT8 file on disk → falls back to plain .engine
    assert mc.person_engine_path == tmp_path / "yolo26n-person-17pose.engine"


@pytest.mark.unit
def test_model_config_int8_engine_path_prefers_int8(tmp_path: Path) -> None:
    """INT8 uses -int8.engine when the file exists."""
    int8_engine = tmp_path / "yolo26n-person-17pose-int8.engine"
    int8_engine.write_bytes(b"fake")
    mc = ModelConfig(engine_dir=tmp_path, precision="int8")
    assert mc.person_engine_path == int8_engine
