from __future__ import annotations

import importlib

from src.config_loader import build_config_from_yaml
from src.gui.view_model import (
    LaunchSettings,
    TuningSettings,
    apply_launch_settings,
    apply_tuning_settings,
    build_dashboard_shell_spec,
    build_runtime_diagnostic_rows,
    launch_settings_from_config,
    runtime_control_specs,
    runtime_transaction_paths,
    safety_action_names,
    suggest_source_for_mode,
    summarize_config,
    tuning_settings_from_config,
    validate_launch_settings,
)
from src.shared.telemetry import RuntimeTelemetry


def test_dashboard_shell_exposes_required_gui_regions():
    shell = build_dashboard_shell_spec()

    assert shell.title == "TaffTracker Control Plane"
    assert [panel.panel_id for panel in shell.panels] == [
        "live",
        "tracking",
        "gimbal",
        "calibration",
        "diagnostics",
        "settings",
    ]
    assert "emergency_stop" in shell.quick_actions
    assert "process_state" in shell.status_fields


def test_runtime_control_specs_separate_startup_from_transactions():
    specs = runtime_control_specs()
    startup_paths = {spec.path for spec in specs if spec.requires_restart}
    transaction_paths = set(runtime_transaction_paths())

    assert {"mode", "target", "source", "camera.source_backend", "camera.backend"} <= startup_paths
    assert {
        "tracking.confidence_threshold",
        "gimbal.kp",
        "gimbal.kd",
        "gimbal.slew_limit_dps",
        "laser.enabled",
    } <= transaction_paths


def test_safety_actions_include_laser_and_emergency_controls():
    actions = safety_action_names()

    assert "emergency_stop" in actions
    assert "laser_toggle" in actions


def test_summarize_config_uses_loaded_pipeline_config():
    config = build_config_from_yaml(
        {
            "mode": "video",
            "target": "dog",
            "source": "videos/sample.mp4",
            "camera": {"width": 800, "height": 600, "fps": 30, "backend": "ffmpeg"},
            "comms": {"channel": "udp", "udp_host": "10.0.0.5", "udp_port": 7000},
            "laser": {"enabled": False},
        }
    )

    summary = summarize_config(config)

    assert summary.mode == "video"
    assert summary.target == "dog"
    assert summary.camera == "800x600@30\nopencv/ffmpeg"
    assert summary.comms == "udp 10.0.0.5:7000"
    assert summary.laser == "disabled"


def test_summarize_config_includes_camera_fov_when_configured():
    config = build_config_from_yaml({"camera": {"fov": 66.2}})

    summary = summarize_config(config)

    assert summary.camera == "640x640@60\nopencv/auto fov=66.2"


def test_launch_settings_update_startup_config_without_runtime_side_effects():
    config = build_config_from_yaml(
        {
            "mode": "video",
            "source": "videos/person.mp4",
            "camera": {"width": 640, "height": 640, "fps": 60, "backend": "ffmpeg"},
            "comms": {"enabled": False},
        }
    )

    settings = LaunchSettings(
        mode="camera",
        target="human",
        source="0",
        comms_enabled=True,
        camera_width=800,
        camera_height=600,
        camera_fps=30,
        camera_source_backend="phone_h264",
        camera_backend="dshow",
        camera_fov=66.2,
        model_precision="int8",
        model_image_size=640,
    )
    updated = apply_launch_settings(config, settings)

    assert launch_settings_from_config(updated) == settings
    assert updated.tracking == config.tracking


def test_suggest_source_for_mode_uses_safe_defaults_when_switching_modes():
    assert suggest_source_for_mode("camera", "videos/person.mp4") == "0"
    assert suggest_source_for_mode("camera", "1") == "1"
    assert suggest_source_for_mode("video", "0") == "videos/person.mp4"


def test_validate_launch_settings_flags_camera_fov_and_model_size():
    settings = LaunchSettings(
        mode="camera",
        target="human",
        source="0",
        comms_enabled=False,
        camera_width=640,
        camera_height=640,
        camera_fps=60,
        camera_source_backend="opencv",
        camera_backend="auto",
        camera_fov=None,
        model_precision="fp16",
        model_image_size=650,
    )

    errors = validate_launch_settings(settings)

    assert "camera mode requires FOV > 0 deg" in errors
    assert "model image size must be a positive multiple of 32" in errors


def test_apply_tuning_settings_updates_control_surfaces_only():
    config = build_config_from_yaml({})
    settings = TuningSettings(
        tracking_confidence_threshold=0.51,
        tracking_hold_time_s=0.75,
        gimbal_kp=1.5,
        gimbal_ki=0.1,
        gimbal_kd=0.7,
        gimbal_deadband_deg=1.0,
        gimbal_slew_limit_dps=40.0,
        gimbal_kp_near=0.9,
        gimbal_kp_far=1.7,
        gimbal_predictive_lead_s=0.02,
        laser_startup_enabled=False,
        relay_pulse_ms=250,
        boresight_pan_offset_deg=1.25,
        boresight_tilt_offset_deg=-0.5,
    )

    updated = apply_tuning_settings(config, settings)

    assert tuning_settings_from_config(updated) == settings
    assert updated.mode == config.mode
    assert updated.camera == config.camera
    assert updated.models == config.models


def test_gui_app_import_does_not_require_qt():
    module = importlib.import_module("src.gui.app")

    assert callable(module.main)


def test_build_runtime_diagnostic_rows_formats_latency_breakdown():
    telemetry = RuntimeTelemetry(
        frame_id=42,
        timestamp_ns=1_000_000,
        target_kind="human",
        target_acquired=True,
        state_source="measurement",
        track_id=7,
        confidence=0.91,
        filtered_pixel=(320.0, 240.0),
        filtered_angles=(0.1, -0.2),
        fps=59.7,
        inference_ms=3.9,
        tracking_ms=0.6,
        postprocess_ms=0.4,
        wait_ms=1.2,
        total_latency_ms=8.8,
        transport_status="LINK OFFLINE",
        packet_sequence=123,
        lock_frames=99,
        total_frames=100,
        display_drops=1,
        display_total=120,
    )

    rows = {row.metric: row.value for row in build_runtime_diagnostic_rows(telemetry)}

    assert rows["Inference"] == "3.9 ms"
    assert rows["Tracking"] == "0.6 ms"
    assert rows["Postprocess"] == "0.4 ms"
    assert rows["Wait"] == "1.2 ms"
    assert rows["Lock"] == "99% (99/100)"
    assert rows["Display drops"] == "1/120 (0.8%)"
