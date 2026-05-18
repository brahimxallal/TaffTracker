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
        assert config.tracking.hold_time_s == 2.0
        assert config.tracking.max_lost_frames == 99
        assert config.tracking.tracker_match_threshold == 0.6
        assert config.tracking.kalman.innovation_gate_sigma == 5.0
        assert config.tracking.smoothing.display_mincutoff == 0.5
        assert config.tracking.postprocess.body_fallback_head_weight == 0.5
        assert config.tracking.adaptive.fast_speed_thresh == 300.0

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
