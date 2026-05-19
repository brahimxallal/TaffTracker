from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)

APP_ID = "com.tafftracker.taffcam"
START_ACTION = f"{APP_ID}.START"

_EXTRA_ORDER = (
    ("--ei", "width", "int"),
    ("--ei", "height", "int"),
    ("--ei", "fps", "int"),
    ("--es", "stream_format", "str"),
    ("--es", "capture_mode", "str"),
    ("--es", "codec", "str"),
    ("--ei", "bitrate_bps", "int"),
    ("--ef", "keyframe_interval_s", "float"),
    ("--el", "exposure_ns", "int"),
    ("--ei", "iso", "int"),
    ("--ez", "awb_enabled", "bool"),
    ("--ez", "awb_lock", "bool"),
    ("--ez", "torch_enabled", "bool"),
    ("--ef", "focus_diopters", "float"),
)


@dataclass(frozen=True, slots=True)
class TaffCamLaunchConfig:
    adb_path: str = "adb"
    adb_serial: str | None = None
    app_package: str = APP_ID
    app_activity: str = ".MainActivity"
    app_receiver: str = ".TaffCommandReceiver"
    start_action: str = START_ACTION
    start_delay_s: float = 1.0
    command_timeout_s: float = 6.0
    wake_screen: bool = True
    dismiss_keyguard: bool = True


def build_adb_start_extras(controls: Mapping[str, Any]) -> list[str]:
    extras = ["--ez", "taff_start", "true"]
    for flag, key, value_type in _EXTRA_ORDER:
        value = controls.get(key)
        if value is None:
            continue
        extras.extend([flag, key, _format_extra(value, value_type)])
    return extras


def start_taffcam_app(config: TaffCamLaunchConfig, controls: Mapping[str, Any]) -> None:
    if config.wake_screen:
        _run_adb(config, ["shell", "input", "keyevent", "224"], timeout_s=2.0)
    if config.dismiss_keyguard:
        _run_adb(config, ["shell", "wm", "dismiss-keyguard"], timeout_s=2.0)

    extras = build_adb_start_extras(controls)
    activity = _component(config.app_package, config.app_activity)
    start = _run_adb(
        config,
        ["shell", "am", "start", "-n", activity, "-a", config.start_action, *extras],
    )
    _log_adb_failure("start TaffCam activity", start)
    if config.start_delay_s > 0.0:
        time.sleep(config.start_delay_s)

    receiver = _component(config.app_package, config.app_receiver)
    broadcast = _run_adb(
        config,
        ["shell", "am", "broadcast", "-n", receiver, "-a", config.start_action, *extras],
    )
    _log_adb_failure("broadcast TaffCam start", broadcast)


def _format_extra(value: Any, value_type: str) -> str:
    if value_type == "int":
        return _format_int(value)
    if value_type == "bool":
        return _format_bool(value)
    return str(value)


def _format_int(value: Any) -> str:
    return str(int(round(float(value))))


def _format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _component(package: str, name: str) -> str:
    if "/" in name:
        return name
    if name.startswith("."):
        return f"{package}/{name}"
    return f"{package}/{name}"


def _run_adb(
    config: TaffCamLaunchConfig,
    args: Sequence[str],
    *,
    timeout_s: float | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = [config.adb_path]
    if config.adb_serial:
        cmd.extend(["-s", config.adb_serial])
    cmd.extend(args)
    try:
        return subprocess.run(
            cmd,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_s or config.command_timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            cmd,
            124,
            stdout=exc.stdout or "",
            stderr=f"ADB command timed out after {(timeout_s or config.command_timeout_s):.1f}s",
        )
    except OSError as exc:
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=str(exc))


def _log_adb_failure(action: str, result: subprocess.CompletedProcess[str]) -> None:
    if result.returncode == 0:
        return
    message = result.stderr.strip() or result.stdout.strip()
    LOGGER.warning("Failed to %s: %s", action, message or f"exit {result.returncode}")
