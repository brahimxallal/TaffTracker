from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "scripts" / "benchmark_taffcam.py"


_INTERVAL_RE = re.compile(
    r"^(?P<label>Android timestamp interval|Host receive interval): "
    r"median=(?P<median>[0-9.]+) ms p95=(?P<p95>[0-9.]+) ms approx_fps=(?P<fps>[0-9.]+)"
)
_CAPTURED_RE = re.compile(
    r"^Captured (?P<count>\d+) frames via (?P<backend>\w+), "
    r"seq (?P<first>\d+)->(?P<last>\d+), skipped=(?P<skipped>\d+), "
    r"size=(?P<width>\d+)x(?P<height>\d+), (?P<stream>.+)$"
)


@dataclass(frozen=True, slots=True)
class SweepMode:
    name: str
    backend: str
    width: int
    height: int
    fps: int
    codec: str = "h264"
    bitrate_bps: int = 8_000_000
    capture_mode: str = "auto"


@dataclass(frozen=True, slots=True)
class SweepResult:
    mode: SweepMode
    returncode: int
    captured: int = 0
    skipped: int = 0
    android_fps: float = 0.0
    android_median_ms: float = 0.0
    android_p95_ms: float = 0.0
    host_fps: float = 0.0
    host_median_ms: float = 0.0
    host_p95_ms: float = 0.0
    output: str = ""


def _default_modes() -> list[SweepMode]:
    return [
        SweepMode("h264_320x240_60", "phone_mpeg", 320, 240, 60, bitrate_bps=4_000_000),
        SweepMode("h264_640x360_60", "phone_mpeg", 640, 360, 60, bitrate_bps=6_000_000),
        SweepMode("h264_640x480_60", "phone_mpeg", 640, 480, 60, bitrate_bps=8_000_000),
        SweepMode("h264_1280x720_60", "phone_mpeg", 1280, 720, 60, bitrate_bps=12_000_000),
        SweepMode("h264_640x480_60_normal", "phone_mpeg", 640, 480, 60, bitrate_bps=8_000_000, capture_mode="normal"),
        SweepMode("h264_1280x720_60_normal", "phone_mpeg", 1280, 720, 60, bitrate_bps=12_000_000, capture_mode="normal"),
        SweepMode("h264_640x480_30_normal", "phone_mpeg", 640, 480, 30, bitrate_bps=6_000_000, capture_mode="normal"),
        SweepMode("yuv_640x480_60", "phone_yuv", 640, 480, 60, bitrate_bps=0),
    ]


def _run(
    args: Sequence[str],
    *,
    timeout_s: float = 8.0,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            args,
            124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or f"Command timed out after {timeout_s:.1f}s",
        )


def _adb_device_state(adb: str, serial: str | None) -> str:
    result = _run([adb, "devices"], timeout_s=5.0)
    if result.returncode != 0:
        return "adb_error"
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("List of devices"):
            continue
        parts = line.split()
        if len(parts) >= 2 and (serial is None or parts[0] == serial):
            return parts[1]
    return "missing"


def _parse_result(mode: SweepMode, returncode: int, output: str) -> SweepResult:
    captured = 0
    skipped = 0
    android_fps = android_median_ms = android_p95_ms = 0.0
    host_fps = host_median_ms = host_p95_ms = 0.0

    for line in output.splitlines():
        captured_match = _CAPTURED_RE.match(line.strip())
        if captured_match:
            captured = int(captured_match.group("count"))
            skipped = int(captured_match.group("skipped"))
            continue

        interval_match = _INTERVAL_RE.match(line.strip())
        if not interval_match:
            continue
        median = float(interval_match.group("median"))
        p95 = float(interval_match.group("p95"))
        fps = float(interval_match.group("fps"))
        if interval_match.group("label").startswith("Android"):
            android_median_ms = median
            android_p95_ms = p95
            android_fps = fps
        else:
            host_median_ms = median
            host_p95_ms = p95
            host_fps = fps

    return SweepResult(
        mode=mode,
        returncode=returncode,
        captured=captured,
        skipped=skipped,
        android_fps=android_fps,
        android_median_ms=android_median_ms,
        android_p95_ms=android_p95_ms,
        host_fps=host_fps,
        host_median_ms=host_median_ms,
        host_p95_ms=host_p95_ms,
        output=output,
    )


def _benchmark_mode(args: argparse.Namespace, mode: SweepMode) -> SweepResult:
    command = [
        sys.executable,
        str(BENCHMARK),
        "--source-backend",
        mode.backend,
        "--capture-mode",
        mode.capture_mode,
        "--width",
        str(mode.width),
        "--height",
        str(mode.height),
        "--fps",
        str(mode.fps),
        "--frames",
        str(args.frames),
        "--timeout-s",
        str(args.timeout_s),
        "--adb",
        args.adb,
        "--exposure-ns",
        str(args.exposure_ns),
        "--iso",
        str(args.iso),
        "--start-delay-s",
        str(args.start_delay_s),
    ]
    if args.serial:
        command.extend(["--serial", args.serial])
    if args.no_adb_reverse:
        command.append("--no-adb-reverse")
    if args.torch:
        command.append("--torch")
    if mode.backend == "phone_mpeg":
        command.extend(
            [
                "--codec",
                mode.codec,
                "--bitrate-bps",
                str(mode.bitrate_bps),
                "--keyframe-interval-s",
                str(args.keyframe_interval_s),
            ]
        )

    result = _run(command, timeout_s=args.timeout_s + 8.0)
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    return _parse_result(mode, result.returncode, output)


def _print_summary(results: Sequence[SweepResult]) -> None:
    print("\nSummary")
    print(
        "mode                 rc frames skip android_fps android_med android_p95 "
        "host_fps host_med host_p95"
    )
    for result in results:
        print(
            f"{result.mode.name:20s} {result.returncode:2d} "
            f"{result.captured:6d} {result.skipped:4d} "
            f"{result.android_fps:11.1f} {result.android_median_ms:11.2f} "
            f"{result.android_p95_ms:11.2f} {result.host_fps:8.1f} "
            f"{result.host_median_ms:8.2f} {result.host_p95_ms:8.2f}"
        )

    good = [item for item in results if item.returncode == 0 and item.android_fps > 0.0]
    if not good:
        print("\nNo successful modes.")
        return
    best = max(good, key=lambda item: (item.android_fps, -item.host_p95_ms, item.captured))
    print(
        "\nBest by source FPS: "
        f"{best.mode.name} android_fps={best.android_fps:.1f} "
        f"host_p95={best.host_p95_ms:.2f}ms skipped={best.skipped}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep TaffCam modes and rank them by real Android timestamp FPS."
    )
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--start-delay-s", type=float, default=1.0)
    parser.add_argument("--exposure-ns", type=int, default=8_000_000)
    parser.add_argument("--iso", type=int, default=800)
    parser.add_argument("--keyframe-interval-s", type=float, default=1.0)
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--no-adb-reverse", action="store_true")
    parser.add_argument("--include-yuv", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = _adb_device_state(args.adb, args.serial)
    if state != "device":
        print(f"ADB device is {state}; authorize USB debugging before sweeping.")
        return 2

    modes = _default_modes()
    if not args.include_yuv:
        modes = [mode for mode in modes if mode.backend != "phone_yuv"]

    results: list[SweepResult] = []
    for mode in modes:
        print(f"\n=== {mode.name} ===")
        result = _benchmark_mode(args, mode)
        print(result.output.strip())
        results.append(result)
        if args.stop_on_failure and result.returncode != 0:
            break

    _print_summary(results)
    return 0 if all(result.returncode == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
