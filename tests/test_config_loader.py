"""Tests for config.yaml loading and CLI override logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config_loader import build_config_from_yaml, build_launcher_env, load_yaml_config


@pytest.mark.unit
class TestConfigLoader:
    def test_load_yaml_returns_empty_when_missing(self, tmp_path: Path):
        data = load_yaml_config(tmp_path / "nonexistent.yaml")
        assert data == {}

    def test_load_yaml_reads_file(self, tmp_path: Path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text("mode: video\ntarget: dog\nsource: test.mp4\n")
        data = load_yaml_config(cfg)
        assert data["mode"] == "video"
        assert data["target"] == "dog"

    def test_build_config_defaults(self):
        config = build_config_from_yaml({})
        assert config.mode == "camera"
        assert config.target == "human"
        assert config.source == "0"

    def test_live_config_uses_auto_transport_for_usb_wifi_failover(self):
        repo_config = Path(__file__).parents[1] / "config.yaml"
        config = build_config_from_yaml(load_yaml_config(repo_config))

        assert config.comms.channel == "auto"
        assert config.comms.serial_port == "COM4"
        assert config.comms.udp_host == "192.168.1.12"
        assert config.comms.udp_port == 6000

    def test_live_config_uses_precision_aim_preset(self):
        repo_config = Path(__file__).parents[1] / "config.yaml"
        config = build_config_from_yaml(load_yaml_config(repo_config))

        assert config.tracking.inference_schedule.detector_locked_stride == 1
        assert config.gimbal.predictive_lead_s == 0.035
        assert config.gimbal.slew_limit_dps == 125.0
        assert config.gimbal.deadband_deg == 0.65
        assert config.gimbal.kp_far == 4.0
        assert config.gimbal.kp_near == 2.5
        assert config.gimbal.kd == 0.0
        assert config.gimbal.tilt_scale == 0.6
        assert config.servo_control.auto_command_slew_limit_dps == 130.0
        assert config.servo_control.velocity_smoother_enabled is True
        assert config.servo_control.velocity_smoother_alpha == 0.45
        assert config.servo_control.velocity_smoother_deadband_dps == 6.0
        assert config.servo_control.velocity_smoother_slew_dps_per_s == 1800.0

    def test_live_config_enables_delay_compensated_servo_path(self):
        repo_config = Path(__file__).parents[1] / "config.yaml"
        config = build_config_from_yaml(load_yaml_config(repo_config))

        assert config.tracking.servo_input == "measurement"
        servo = config.servo_control
        assert servo.latency_comp_delay_s == 0.30
        assert servo.latency_comp_smith_gain == 1.0
        assert servo.latency_comp_velocity_ff_gain == 1.0
        assert servo.latency_comp_velocity_tau_s == 0.18
        assert servo.latency_comp_velocity_deadband_dps == 3.0
        assert servo.latency_comp_velocity_max_dps == 150.0

    def test_live_config_keeps_relay_auto_fire_disabled(self):
        repo_config = Path(__file__).parents[1] / "config.yaml"
        config = build_config_from_yaml(load_yaml_config(repo_config))

        assert config.relay.auto_fire_enabled is False

    def test_latency_compensation_is_off_by_default(self):
        config = build_config_from_yaml({})

        assert config.servo_control.latency_comp_delay_s == 0.0
        assert config.tracking.servo_input == "kalman"

    def test_latency_compensation_from_yaml(self):
        yaml_data = {
            "servo_control": {
                "latency_comp_delay_s": 0.25,
                "latency_comp_smith_gain": 0.9,
                "latency_comp_velocity_ff_gain": 0.8,
                "latency_comp_velocity_tau_s": 0.2,
                "latency_comp_velocity_deadband_dps": 2.5,
                "latency_comp_velocity_max_dps": 120.0,
            },
            "tracking": {"servo_input": "measurement"},
        }

        config = build_config_from_yaml(yaml_data)

        servo = config.servo_control
        assert servo.latency_comp_delay_s == 0.25
        assert servo.latency_comp_smith_gain == 0.9
        assert servo.latency_comp_velocity_ff_gain == 0.8
        assert servo.latency_comp_velocity_tau_s == 0.2
        assert servo.latency_comp_velocity_deadband_dps == 2.5
        assert servo.latency_comp_velocity_max_dps == 120.0
        assert config.tracking.servo_input == "measurement"

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("latency_comp_delay_s", -0.01),
            ("latency_comp_delay_s", 1.5),
            ("latency_comp_smith_gain", -0.1),
            ("latency_comp_smith_gain", 2.0),
            ("latency_comp_velocity_ff_gain", -0.1),
            ("latency_comp_velocity_ff_gain", 2.0),
            ("latency_comp_velocity_tau_s", 0.0),
            ("latency_comp_velocity_tau_s", 3.0),
            ("latency_comp_velocity_deadband_dps", -1.0),
            ("latency_comp_velocity_max_dps", 0.0),
        ],
    )
    def test_invalid_latency_compensation_rejected(self, key: str, value: float):
        with pytest.raises(ValueError, match="latency_comp"):
            build_config_from_yaml({"servo_control": {key: value}})

    def test_invalid_servo_input_rejected(self):
        with pytest.raises(ValueError, match="tracking.servo_input"):
            build_config_from_yaml({"tracking": {"servo_input": "magic"}})

    def test_build_config_from_yaml_data(self):
        yaml_data = {
            "mode": "video",
            "target": "dog",
            "source": "videos/doghom.mp4",
            "camera": {"fps": 30, "width": 640, "height": 640},
            "comms": {"serial_port": "COM7", "enabled": False},
        }
        config = build_config_from_yaml(yaml_data)
        assert config.mode == "video"
        assert config.target == "dog"
        assert config.camera.fps == 30
        assert config.comms.serial_port == "COM7"
        assert config.comms.enabled is False

    def test_relay_auto_fire_from_yaml(self):
        yaml_data = {"relay": {"pulse_ms": 250, "auto_fire_enabled": True}}

        config = build_config_from_yaml(yaml_data)

        assert config.relay.pulse_ms == 250
        assert config.relay.auto_fire_enabled is True
        assert config.relay.auto_fire_pulse_ms == 100

    def test_relay_auto_fire_enabled_rejects_non_bool(self):
        yaml_data = {"relay": {"auto_fire_enabled": "true"}}

        with pytest.raises(ValueError, match="relay.auto_fire_enabled"):
            build_config_from_yaml(yaml_data)

    def test_cli_overrides_yaml(self):
        yaml_data = {"mode": "camera", "target": "human", "source": "0"}
        cli = {"mode": "video", "target": "dog", "source": "test.mp4"}
        config = build_config_from_yaml(yaml_data, cli)
        assert config.mode == "video"
        assert config.target == "dog"
        assert config.source == "test.mp4"

    def test_dog_config_has_higher_max_lost(self):
        config = build_config_from_yaml({"target": "dog"})
        assert config.tracking.max_lost_frames > 30

    def test_human_config_uses_plain_defaults(self):
        config = build_config_from_yaml({"target": "human"})
        from src.config import TrackingConfig

        # After FPS adaptation, check nested configs are preserved
        assert config.tracking.kalman == TrackingConfig().kalman

    def test_fov_from_yaml(self):
        yaml_data = {"camera": {"fov": 66.2}}
        config = build_config_from_yaml(yaml_data)
        assert config.camera.fov == 66.2

    def test_fov_from_cli_overrides_yaml(self):
        yaml_data = {"camera": {"fov": 66.2}}
        cli = {"fov": 90.0}
        config = build_config_from_yaml(yaml_data, cli)
        assert config.camera.fov == 90.0

    def test_fov_none_when_not_set(self):
        config = build_config_from_yaml({})
        assert config.camera.fov is None

    def test_gimbal_from_yaml(self):
        yaml_data = {"gimbal": {"invert_pan": False, "invert_tilt": True}}
        config = build_config_from_yaml(yaml_data)
        assert config.gimbal.invert_pan is False
        assert config.gimbal.invert_tilt is True

    def test_gimbal_extended_tuning_from_yaml(self):
        yaml_data = {
            "gimbal": {
                "velocity_feedforward_gain": 0.5,
                "kp_far": 5.0,
                "kp_near": 1.5,
                "gain_schedule_threshold_deg": 3.0,
                "predictive_lead_s": 0.08,
            }
        }

        config = build_config_from_yaml(yaml_data)

        assert config.gimbal.velocity_feedforward_gain == 0.5
        assert config.gimbal.kp_far == 5.0
        assert config.gimbal.kp_near == 1.5
        assert config.gimbal.gain_schedule_threshold_deg == 3.0
        assert config.gimbal.predictive_lead_s == 0.08

    def test_search_config_from_yaml(self):
        yaml_data = {
            "search": {
                "enabled": True,
                "start_after_s": 0.15,
                "timeout_s": 4.0,
                "scan_speed_dps": 22.0,
                "return_home": False,
            }
        }

        config = build_config_from_yaml(yaml_data)

        assert config.search.enabled is True
        assert config.search.start_after_s == 0.15
        assert config.search.timeout_s == 4.0
        assert config.search.scan_speed_dps == 22.0
        assert config.search.return_home is False

    def test_inference_schedule_from_yaml(self):
        yaml_data = {
            "tracking": {
                "inference_schedule": {
                    "detector_skip_enabled": True,
                    "detector_locked_stride": 4,
                    "detector_min_skip_confidence": 0.72,
                    "detector_max_prediction_age_ms": 80.0,
                }
            }
        }

        config = build_config_from_yaml(yaml_data)

        schedule = config.tracking.inference_schedule
        assert schedule.detector_skip_enabled is True
        assert schedule.detector_locked_stride == 4
        assert schedule.detector_min_skip_confidence == 0.72
        assert schedule.detector_max_prediction_age_ms == 80.0

    def test_invalid_inference_schedule_rejected(self):
        yaml_data = {"tracking": {"inference_schedule": {"detector_locked_stride": 0}}}

        with pytest.raises(ValueError, match="detector_locked_stride"):
            build_config_from_yaml(yaml_data)

    def test_servo_control_relock_and_backlash_from_yaml(self):
        yaml_data = {
            "servo_control": {
                "micro_relock_min_loss_s": 0.03,
                "micro_relock_window_s": 0.09,
                "micro_relock_gain": 0.4,
                "micro_relock_ramp_frames": 3,
                "backlash_compensation_enabled": True,
                "backlash_pan_deg": 0.25,
                "backlash_tilt_deg": 0.15,
                "backlash_min_step_deg": 0.06,
                "backlash_pulse_decay_dps": 4.0,
            }
        }

        config = build_config_from_yaml(yaml_data)

        assert config.servo_control.micro_relock_min_loss_s == 0.03
        assert config.servo_control.micro_relock_window_s == 0.09
        assert config.servo_control.micro_relock_gain == 0.4
        assert config.servo_control.micro_relock_ramp_frames == 3
        assert config.servo_control.backlash_compensation_enabled is True
        assert config.servo_control.backlash_pan_deg == 0.25
        assert config.servo_control.backlash_tilt_deg == 0.15
        assert config.servo_control.backlash_min_step_deg == 0.06
        assert config.servo_control.backlash_pulse_decay_dps == 4.0

    def test_ffmpeg_backend_from_yaml_is_valid(self):
        config = build_config_from_yaml({"camera": {"backend": "ffmpeg"}})

        assert config.camera.backend == "ffmpeg"

    def test_phone_yuv_source_backend_from_yaml_is_valid(self):
        config = build_config_from_yaml({"camera": {"source_backend": "phone_yuv"}})

        assert config.camera.source_backend == "phone_yuv"

    def test_phone_mpeg_source_backend_from_yaml_is_valid(self):
        config = build_config_from_yaml({"camera": {"source_backend": "phone_mpeg"}})

        assert config.camera.source_backend == "phone_mpeg"

    def test_phone_h264_source_backend_from_yaml_is_valid(self):
        config = build_config_from_yaml({"camera": {"source_backend": "phone_h264"}})

        assert config.camera.source_backend == "phone_h264"

    def test_droidcam_source_backend_from_yaml_is_valid(self):
        config = build_config_from_yaml({"camera": {"source_backend": "droidcam"}})

        assert config.camera.source_backend == "droidcam"

    def test_phone_yuv_source_backend_from_cli_overrides_yaml(self):
        yaml_data = {"camera": {"source_backend": "opencv"}}
        cli = {"source_backend": "phone_yuv"}

        config = build_config_from_yaml(yaml_data, cli)

        assert config.camera.source_backend == "phone_yuv"

    def test_phone_mpeg_source_backend_from_cli_overrides_yaml(self):
        yaml_data = {"camera": {"source_backend": "opencv"}}
        cli = {"source_backend": "phone_mpeg"}

        config = build_config_from_yaml(yaml_data, cli)

        assert config.camera.source_backend == "phone_mpeg"

    def test_phone_h264_source_backend_from_cli_overrides_yaml(self):
        yaml_data = {"camera": {"source_backend": "opencv"}}
        cli = {"source_backend": "phone_h264"}

        config = build_config_from_yaml(yaml_data, cli)

        assert config.camera.source_backend == "phone_h264"

    def test_phone_camera_config_from_yaml(self):
        yaml_data = {
            "phone_camera": {
                "bind_host": "127.0.0.1",
                "allow_remote_clients": True,
                "frame_port": 28001,
                "control_port": 28002,
                "adb_reverse_enabled": False,
                "stream_format": "mpeg",
                "pixel_format": "nv12",
                "codec": "h264",
                "bitrate_bps": 6_000_000,
                "keyframe_interval_s": 0.25,
                "decode_backend": "pyav",
                "capture_mode": "high_speed",
                "focus_diopters": 0.75,
                "exposure_ns": 4_000_000,
                "iso": 320,
                "torch_enabled": True,
            }
        }

        config = build_config_from_yaml(yaml_data)

        assert config.phone_camera.bind_host == "127.0.0.1"
        assert config.phone_camera.allow_remote_clients is True
        assert config.phone_camera.frame_port == 28001
        assert config.phone_camera.control_port == 28002
        assert config.phone_camera.adb_reverse_enabled is False
        assert config.phone_camera.stream_format == "mpeg"
        assert config.phone_camera.pixel_format == "nv12"
        assert config.phone_camera.codec == "h264"
        assert config.phone_camera.bitrate_bps == 6_000_000
        assert config.phone_camera.keyframe_interval_s == 0.25
        assert config.phone_camera.decode_backend == "pyav"
        assert config.phone_camera.capture_mode == "high_speed"
        assert config.phone_camera.focus_diopters == 0.75
        assert config.phone_camera.exposure_ns == 4_000_000
        assert config.phone_camera.iso == 320
        assert config.phone_camera.torch_enabled is True

    def test_phone_imu_config_from_yaml(self):
        yaml_data = {
            "phone_imu": {
                "enabled": True,
                "bind_host": "127.0.0.1",
                "sensor_port": 28003,
                "requested_rate_hz": 100,
                "adb_reverse_enabled": False,
                "start_service_on_open": False,
                "read_timeout_s": 0.2,
            }
        }

        config = build_config_from_yaml(yaml_data)

        assert config.phone_imu.enabled is True
        assert config.phone_imu.bind_host == "127.0.0.1"
        assert config.phone_imu.sensor_port == 28003
        assert config.phone_imu.requested_rate_hz == 100
        assert config.phone_imu.adb_reverse_enabled is False
        assert config.phone_imu.start_service_on_open is False
        assert config.phone_imu.read_timeout_s == 0.2

    def test_servo_command_slew_limits_from_yaml(self):
        yaml_data = {
            "servo_control": {
                "manual_command_slew_limit_dps": 160.0,
                "auto_command_slew_limit_dps": 45.0,
                "manual_fine_speed_dps": 180.0,
                "manual_coarse_speed_dps": 420.0,
                "manual_accel_time_s": 0.03,
            }
        }

        config = build_config_from_yaml(yaml_data)

        assert config.servo_control.manual_command_slew_limit_dps == 160.0
        assert config.servo_control.auto_command_slew_limit_dps == 45.0
        assert config.servo_control.manual_fine_speed_dps == 180.0
        assert config.servo_control.manual_coarse_speed_dps == 420.0
        assert config.servo_control.manual_accel_time_s == 0.03

    def test_droidcam_config_from_yaml(self):
        yaml_data = {
            "droidcam": {
                "host": "10.0.0.20",
                "port": 4748,
                "width": 1280,
                "height": 720,
                "fps": 60,
                "video_format": "mjpg",
                "remote_enabled": True,
                "active_camera": 0,
                "autofocus_once": True,
                "autofocus_mode": 3,
                "manual_focus": 12.5,
                "exposure_lock": True,
                "ev": -1.0,
                "wb_mode": 5,
                "wb_lock": True,
                "wb_level": 5200,
                "torch_enabled": False,
                "zoom": 1.2,
            }
        }

        config = build_config_from_yaml(yaml_data)

        assert config.droidcam.host == "10.0.0.20"
        assert config.droidcam.port == 4748
        assert config.droidcam.width == 1280
        assert config.droidcam.height == 720
        assert config.droidcam.video_format == "mjpg"
        assert config.droidcam.remote_enabled is True
        assert config.droidcam.active_camera == 0
        assert config.droidcam.autofocus_once is True
        assert config.droidcam.autofocus_mode == 3
        assert config.droidcam.manual_focus == 12.5
        assert config.droidcam.exposure_lock is True
        assert config.droidcam.ev == -1.0
        assert config.droidcam.wb_mode == 5
        assert config.droidcam.wb_lock is True
        assert config.droidcam.wb_level == 5200
        assert config.droidcam.torch_enabled is False
        assert config.droidcam.zoom == 1.2

    def test_invalid_phone_codec_raises(self):
        with pytest.raises(ValueError, match="phone_camera.codec"):
            build_config_from_yaml({"phone_camera": {"codec": "vp9"}})

    def test_invalid_phone_stream_format_raises(self):
        with pytest.raises(ValueError, match="phone_camera.stream_format"):
            build_config_from_yaml({"phone_camera": {"stream_format": "raw"}})

    def test_invalid_phone_decode_backend_raises(self):
        with pytest.raises(ValueError, match="phone_camera.decode_backend"):
            build_config_from_yaml({"phone_camera": {"decode_backend": "opencv"}})

    def test_invalid_phone_capture_mode_raises(self):
        with pytest.raises(ValueError, match="phone_camera.capture_mode"):
            build_config_from_yaml({"phone_camera": {"capture_mode": "magic"}})

    def test_phone_capture_mode_normal_is_valid(self):
        config = build_config_from_yaml({"phone_camera": {"capture_mode": "normal"}})

        assert config.phone_camera.capture_mode == "normal"

    def test_invalid_droidcam_video_format_raises(self):
        with pytest.raises(ValueError, match="droidcam.video_format"):
            build_config_from_yaml({"droidcam": {"video_format": "mpeg4"}})

    def test_gimbal_defaults_when_missing(self):
        config = build_config_from_yaml({})
        assert config.gimbal.invert_pan is False
        assert config.gimbal.invert_tilt is False

    def test_gimbal_cli_override(self):
        yaml_data = {"gimbal": {"invert_tilt": False}}
        cli = {"invert_tilt": True}
        config = build_config_from_yaml(yaml_data, cli)
        assert config.gimbal.invert_tilt is True

    def test_tracking_yaml_overrides_defaults(self):
        yaml_data = {
            "tracking": {
                "confidence_threshold": 0.55,
                "lock_acquire_confidence": 0.70,
                "lock_keep_confidence": 0.50,
                "low_confidence_grace_frames": 3,
                "hold_time_s": 2.0,
                "max_lost_frames": 99,
                "tracker_match_threshold": 0.6,
                "kalman": {"innovation_gate_sigma": 5.0},
                "smoothing": {"display_mincutoff": 0.5},
                "postprocess": {"body_fallback_head_weight": 0.5},
                "adaptive": {"fast_speed_thresh": 300.0},
            }
        }
        config = build_config_from_yaml(yaml_data)
        assert config.tracking.confidence_threshold == 0.55
        assert config.tracking.lock_acquire_confidence == 0.70
        assert config.tracking.lock_keep_confidence == 0.50
        assert config.tracking.low_confidence_grace_frames == 3
        assert config.tracking.hold_time_s == 2.0
        assert config.tracking.max_lost_frames == 99
        assert config.tracking.tracker_match_threshold == 0.6
        assert config.tracking.kalman.innovation_gate_sigma == 5.0
        assert config.tracking.smoothing.display_mincutoff == 0.5
        assert config.tracking.postprocess.body_fallback_head_weight == 0.5
        assert config.tracking.adaptive.fast_speed_thresh == 300.0

    def test_tracking_lock_confidence_hysteresis_is_validated(self):
        yaml_data = {
            "tracking": {
                "lock_acquire_confidence": 0.40,
                "lock_keep_confidence": 0.50,
            }
        }

        with pytest.raises(ValueError, match="lock_keep_confidence"):
            build_config_from_yaml(yaml_data)

    def test_cli_comm_override_reenables_disabled_yaml_output(self):
        yaml_data = {"comms": {"enabled": False, "channel": "serial", "serial_port": "COM7"}}
        cli = {"comm": "udp", "udp_host": "10.0.0.8", "udp_port": 7001}

        config = build_config_from_yaml(yaml_data, cli)

        assert config.comms.enabled is True
        assert config.comms.channel == "udp"
        assert config.comms.udp_host == "10.0.0.8"
        assert config.comms.udp_port == 7001

    def test_cli_confidence_overrides_yaml(self):
        yaml_data = {"tracking": {"confidence_threshold": 0.7}}
        cli = {"confidence": 0.3}
        config = build_config_from_yaml(yaml_data, cli)
        assert config.tracking.confidence_threshold == 0.3

    def test_cli_hold_time_overrides_yaml(self):
        yaml_data = {"tracking": {"hold_time_s": 1.0}}
        cli = {"hold_time": 3.0}
        config = build_config_from_yaml(yaml_data, cli)
        assert config.tracking.hold_time_s == 3.0

    def test_runtime_flags_from_yaml(self):
        yaml_data = {
            "runtime": {"debug": True, "headless": True, "profile": True, "log_level": "DEBUG"}
        }
        config = build_config_from_yaml(yaml_data)
        assert config.flags.debug is True
        assert config.flags.headless is True
        assert config.flags.profile is True

    def test_runtime_ring_buffer_slots_from_yaml(self):
        config = build_config_from_yaml({"runtime": {"ring_buffer_slots": 8}})

        assert config.flags.ring_buffer_slots == 8

    def test_invalid_runtime_ring_buffer_slots_raises(self):
        with pytest.raises(ValueError, match="runtime.ring_buffer_slots"):
            build_config_from_yaml({"runtime": {"ring_buffer_slots": 2}})

    def test_preflight_check_interval_from_yaml(self):
        config = build_config_from_yaml({"preflight": {"check_interval_frames": 30}})

        assert config.preflight.check_interval_frames == 30

    def test_invalid_preflight_check_interval_raises(self):
        with pytest.raises(ValueError, match="preflight.check_interval_frames"):
            build_config_from_yaml({"preflight": {"check_interval_frames": 0}})

    def test_laser_config_from_yaml(self):
        yaml_data = {"laser": {"enabled": False, "roi_radius_px": 200.0}}
        config = build_config_from_yaml(yaml_data)
        assert config.laser.enabled is False
        assert config.laser.roi_radius_px == 200.0

    def test_launcher_env_uses_real_config_loader(self, tmp_path: Path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "\n".join(
                [
                    "target: dog",
                    'source: "7"',
                    "camera:",
                    "  backend: dshow",
                    "  fov: 66.2",
                    "comms:",
                    "  channel: udp",
                    "  baud_rate: 115200",
                    "  udp_host: 10.0.0.7",
                    "  udp_port: 7000",
                ]
            ),
            encoding="utf-8",
        )

        env = build_launcher_env(cfg)

        assert env["CFG_TARGET"] == "dog"
        assert env["CFG_SOURCE"] == "7"
        assert env["CFG_BACKEND"] == "dshow"
        assert env["CFG_BAUD_RATE"] == "115200"
        assert env["CFG_FOV"] == "66.2"
        assert env["CFG_COMM_DETAIL"] == "10.0.0.7:7000"
        assert "CFG_CAMERA_MOUNT" not in env

    def test_legacy_ground_plane_yaml_is_ignored(self):
        config = build_config_from_yaml({"ground_plane": {"enabled": True}})
        assert not hasattr(config, "ground_plane")

    def test_model_precision_from_yaml(self):
        yaml_data = {"models": {"precision": "int8"}}
        config = build_config_from_yaml(yaml_data)
        assert config.models.precision == "int8"

    def test_model_precision_defaults_to_fp16(self):
        config = build_config_from_yaml({})
        assert config.models.precision == "fp16"
