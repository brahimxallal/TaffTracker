"""Host-side regression tests for safety-critical ESP32-S3 firmware code.

The real ``comm_task.c`` and ``control_task.c`` are compiled for the host against tiny ESP-IDF
stubs (``firmware/esp32s3_gimbal/host_tests/stubs``) and run under AddressSanitizer/UBSan:

* USB frame parser - crafted and random garbled byte streams must never overrun the frame buffer,
  and the parser must resynchronise on the next valid packet.
* Relay fail-safe - the relay must be released when the host stops sending.

The tests are skipped when no C compiler (or no sanitizer runtime) is available, so they never
block a CPU-only or Windows development machine. On CI (``CI=true``) that is an error instead, so
these safety checks cannot disappear silently when a runner image changes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import NoReturn

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIRMWARE = ROOT / "firmware" / "esp32s3_gimbal"
HOST_TESTS = FIRMWARE / "host_tests"
CC = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
# GitHub Actions and most other CI systems set CI=true.
IN_CI = os.environ.get("CI", "").strip().lower() in {"1", "true", "yes"}

pytestmark = pytest.mark.unit


def _unavailable(reason: str) -> NoReturn:
    """Skip on a developer machine, fail on CI (the safety checks must not vanish silently)."""
    if IN_CI:
        pytest.fail(f"{reason}; required on CI")
    pytest.skip(reason)


def _compile(out_dir: Path, harness: str, define: str, source: str) -> Path:
    if CC is None:
        _unavailable("no C compiler available")
    exe = out_dir / Path(harness).stem
    base = [
        str(CC),
        "-std=gnu11",
        "-g",
        "-O1",
        "-fno-omit-frame-pointer",
        "-Wall",
        "-Wextra",
        "-Wno-unused-function",
        "-Wno-unused-parameter",
        "-I",
        str(HOST_TESTS / "stubs"),
        "-I",
        str(FIRMWARE / "include"),
        f'-D{define}="{(FIRMWARE / "src" / source).as_posix()}"',
        str(HOST_TESTS / harness),
        "-o",
        str(exe),
        "-lm",
    ]
    sanitized = [*base, "-fsanitize=address,undefined", "-fno-sanitize-recover=all"]
    result = subprocess.run(sanitized, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return exe
    plain = subprocess.run(base, capture_output=True, text=True, check=False)
    if plain.returncode == 0:
        _unavailable("C compiler has no AddressSanitizer/UBSan runtime")
    pytest.fail(f"firmware host harness failed to compile:\n{plain.stderr}")


def _run(exe: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(exe), *args], capture_output=True, text=True, timeout=120, check=False
    )


@pytest.fixture(scope="module")
def usb_harness(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _compile(
        tmp_path_factory.mktemp("fw_usb"), "harness_usb.c", "COMM_TASK_C", "comm_task.c"
    )


@pytest.fixture(scope="module")
def control_harness(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _compile(
        tmp_path_factory.mktemp("fw_ctl"), "harness_ctl.c", "CONTROL_TASK_C", "control_task.c"
    )


def test_usb_parser_decodes_valid_frames_and_recovers_from_noise(usb_harness: Path) -> None:
    result = _run(usb_harness, "valid")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout


def test_usb_parser_resync_into_cal_frame_does_not_overrun_buffer(usb_harness: Path) -> None:
    result = _run(usb_harness, "crafted")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout


def test_usb_parser_survives_random_garbled_streams(usb_harness: Path) -> None:
    result = _run(usb_harness, "fuzz", "50000", "7")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "no memory error" in result.stdout


def test_relay_is_released_when_host_goes_silent(control_harness: Path) -> None:
    result = _run(control_harness)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT: PASS" in result.stdout
