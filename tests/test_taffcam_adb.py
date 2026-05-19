from __future__ import annotations

import subprocess

import pytest

from src.capture.taffcam_adb import (
    TaffCamLaunchConfig,
    build_adb_start_extras,
    start_taffcam_app,
)


@pytest.mark.unit
def test_build_adb_start_extras_uses_android_extra_types() -> None:
    extras = build_adb_start_extras(
        {
            "width": 640,
            "height": 480,
            "fps": 60.0,
            "stream_format": "mpeg",
            "capture_mode": "auto",
            "codec": "h264",
            "bitrate_bps": 8_000_000,
            "keyframe_interval_s": 1.0,
            "exposure_ns": 8_000_000,
            "iso": 800,
            "awb_enabled": False,
            "awb_lock": True,
            "torch_enabled": False,
            "focus_diopters": 0.8,
            "zoom_ratio": 1.4,
            "unused": None,
        }
    )

    assert extras == [
        "--ez",
        "taff_start",
        "true",
        "--ei",
        "width",
        "640",
        "--ei",
        "height",
        "480",
        "--ei",
        "fps",
        "60",
        "--es",
        "stream_format",
        "mpeg",
        "--es",
        "capture_mode",
        "auto",
        "--es",
        "codec",
        "h264",
        "--ei",
        "bitrate_bps",
        "8000000",
        "--ef",
        "keyframe_interval_s",
        "1.0",
        "--el",
        "exposure_ns",
        "8000000",
        "--ei",
        "iso",
        "800",
        "--ez",
        "awb_enabled",
        "false",
        "--ez",
        "awb_lock",
        "true",
        "--ez",
        "torch_enabled",
        "false",
        "--ef",
        "focus_diopters",
        "0.8",
    ]


@pytest.mark.unit
def test_start_taffcam_app_starts_activity_and_broadcasts_controls(monkeypatch) -> None:
    calls: list[list[str]] = []
    sleeps: list[float] = []

    def fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("src.capture.taffcam_adb.subprocess.run", fake_run)
    monkeypatch.setattr("src.capture.taffcam_adb.time.sleep", lambda delay: sleeps.append(delay))

    start_taffcam_app(
        TaffCamLaunchConfig(adb_serial="DEVICE", start_delay_s=0.25),
        {
            "width": 640,
            "height": 480,
            "fps": 60,
            "stream_format": "mpeg",
            "codec": "h264",
        },
    )

    assert calls[0] == ["adb", "-s", "DEVICE", "shell", "input", "keyevent", "224"]
    assert calls[1] == ["adb", "-s", "DEVICE", "shell", "wm", "dismiss-keyguard"]
    assert calls[2][:8] == [
        "adb",
        "-s",
        "DEVICE",
        "shell",
        "am",
        "start",
        "-n",
        "com.tafftracker.taffcam/.MainActivity",
    ]
    assert calls[2][8:10] == ["-a", "com.tafftracker.taffcam.START"]
    assert calls[3][:8] == [
        "adb",
        "-s",
        "DEVICE",
        "shell",
        "am",
        "broadcast",
        "-n",
        "com.tafftracker.taffcam/.TaffCommandReceiver",
    ]
    assert calls[3][8:10] == ["-a", "com.tafftracker.taffcam.START"]
    assert calls[2][10:] == calls[3][10:]
    assert sleeps == [0.25]
