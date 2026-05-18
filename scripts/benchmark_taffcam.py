from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capture.phone_mpeg import PhoneMpegCaptureSource, PhoneMpegRuntimeConfig  # noqa: E402
from src.capture.phone_yuv import PhoneCameraRuntimeConfig, PhoneYuvCaptureSource  # noqa: E402

APP_ID = "com.tafftracker.taffcam"
START_RECEIVER = f"{APP_ID}/.TaffCommandReceiver"
START_ACTION = f"{APP_ID}.START"
START_ACTIVITY = f"{APP_ID}/.MainActivity"


def _run_adb(
    adb: str,
    serial: str | None,
    args: Sequence[str],
    *,
    check: bool = False,
    timeout_s: float = 6.0,
) -> subprocess.CompletedProcess[str]:
    cmd = [adb]
    if serial:
        cmd.extend(["-s", serial])
    cmd.extend(args)
    try:
        return subprocess.run(
            cmd,
            check=check,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            cmd,
            124,
            stdout=exc.stdout or "",
            stderr=f"ADB command timed out after {timeout_s:.1f}s: {' '.join(cmd)}",
        )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percentile / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def _intervals(values: Sequence[int | float], scale: float) -> list[float]:
    return [(b - a) * scale for a, b in zip(values, values[1:], strict=False)]


def _print_stats(name: str, intervals_ms: Sequence[float]) -> None:
    if not intervals_ms:
        print(f"{name}: not enough frames")
        return
    median_ms = statistics.median(intervals_ms)
    p95_ms = _percentile(intervals_ms, 95)
    fps = 1000.0 / median_ms if median_ms > 0.0 else 0.0
    print(f"{name}: median={median_ms:.2f} ms p95={p95_ms:.2f} ms approx_fps={fps:.1f}")


def _adb_start_extras(args: argparse.Namespace) -> list[str]:
    encoded_backend = args.source_backend in ("phone_mpeg", "phone_h264")
    stream_format = "mpeg" if encoded_backend else "yuv"
    extras = [
        "--ez",
        "taff_start",
        "true",
        "--ei",
        "width",
        str(args.width),
        "--ei",
        "height",
        str(args.height),
        "--ei",
        "fps",
        str(round(args.fps)),
        "--es",
        "stream_format",
        stream_format,
        "--es",
        "capture_mode",
        args.capture_mode,
        "--el",
        "exposure_ns",
        str(args.exposure_ns),
        "--ei",
        "iso",
        str(args.iso),
        "--ez",
        "awb_enabled",
        "false",
        "--ez",
        "awb_lock",
        "true",
        "--ez",
        "torch_enabled",
        "true" if args.torch else "false",
    ]
    if args.source_backend in ("phone_mpeg", "phone_h264"):
        extras.extend(
            [
                "--es",
                "codec",
                args.codec,
                "--ei",
                "bitrate_bps",
                str(args.bitrate_bps),
                "--ef",
                "keyframe_interval_s",
                str(args.keyframe_interval_s),
            ]
        )
    if args.focus_diopters is not None:
        extras.extend(["--ef", "focus_diopters", str(args.focus_diopters)])
    return extras


def _start_phone_app(args: argparse.Namespace) -> None:
    if not args.start_app:
        return
    print("Starting TaffCam on phone...")
    _run_adb(args.adb, args.serial, ["shell", "input", "keyevent", "224"])
    _run_adb(args.adb, args.serial, ["shell", "wm", "dismiss-keyguard"])
    start = _run_adb(
        args.adb,
        args.serial,
        ["shell", "am", "start", "-n", START_ACTIVITY, "-a", START_ACTION, *_adb_start_extras(args)],
    )
    if start.returncode != 0:
        print(start.stderr.strip() or start.stdout.strip())
    time.sleep(args.start_delay_s)
    broadcast = _run_adb(
        args.adb,
        args.serial,
        [
            "shell",
            "am",
            "broadcast",
            "-n",
            START_RECEIVER,
            "-a",
            START_ACTION,
            *_adb_start_extras(args),
        ],
    )
    if broadcast.returncode != 0:
        print(broadcast.stderr.strip() or broadcast.stdout.strip())


def _start_adb_reverse(args: argparse.Namespace) -> None:
    if args.no_adb_reverse:
        return
    for port in (args.frame_port, args.control_port):
        result = _run_adb(
            args.adb,
            args.serial,
            ["reverse", f"tcp:{port}", f"tcp:{port}"],
            timeout_s=4.0,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip()
            print(f"adb reverse tcp:{port} failed: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TaffCam phone capture latency/FPS.")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--timeout-s", type=float, default=12.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--source-backend",
        choices=("phone_h264", "phone_mpeg", "phone_yuv"),
        default="phone_h264",
        help="Capture backend to benchmark. phone_h264 is the low-latency encoder path.",
    )
    parser.add_argument("--capture-mode", choices=("auto", "normal", "yuv", "high_speed"), default="auto")
    parser.add_argument("--codec", choices=("h264", "mpeg4"), default="h264")
    parser.add_argument("--bitrate-bps", type=int, default=8_000_000)
    parser.add_argument("--keyframe-interval-s", type=float, default=1.0)
    parser.add_argument("--exposure-ns", type=int, default=8_000_000)
    parser.add_argument("--iso", type=int, default=800)
    parser.add_argument("--focus-diopters", type=float, default=None)
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--frame-port", type=int, default=27183)
    parser.add_argument("--control-port", type=int, default=27184)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--no-adb-reverse", action="store_true")
    parser.add_argument("--no-start-app", dest="start_app", action="store_false")
    parser.add_argument("--start-delay-s", type=float, default=1.0)
    parser.set_defaults(start_app=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start_app:
        _run_adb(args.adb, args.serial, ["shell", "am", "force-stop", APP_ID])
        time.sleep(0.4)
        _start_adb_reverse(args)
        _start_phone_app(args)

    startup_controls = {
        "capture_mode": args.capture_mode,
        "stream_format": "mpeg"
        if args.source_backend in ("phone_mpeg", "phone_h264")
        else "yuv",
        "exposure_ns": args.exposure_ns,
        "iso": args.iso,
        "awb_enabled": False,
        "awb_lock": True,
        "torch_enabled": args.torch,
    }
    if args.source_backend in ("phone_mpeg", "phone_h264"):
        startup_controls.update(
            {
                "codec": args.codec,
                "bitrate_bps": args.bitrate_bps,
                "keyframe_interval_s": args.keyframe_interval_s,
            }
        )
    if args.focus_diopters is not None:
        startup_controls["focus_diopters"] = args.focus_diopters

    common_config = {
        "frame_host": args.bind_host,
        "frame_port": args.frame_port,
        "control_host": args.bind_host,
        "control_port": args.control_port,
        "requested_width": args.width,
        "requested_height": args.height,
        "requested_fps": args.fps,
        "read_timeout_s": 0.08,
        "control_timeout_s": 2.0,
        "adb_reverse": not args.no_adb_reverse,
        "adb_path": args.adb,
        "adb_serial": args.serial,
        "startup_controls": startup_controls,
    }
    if args.source_backend in ("phone_mpeg", "phone_h264"):
        config = PhoneMpegRuntimeConfig(
            **common_config,
            codec=args.codec,
            bitrate_bps=args.bitrate_bps,
            keyframe_interval_s=args.keyframe_interval_s,
        )
        source = PhoneMpegCaptureSource(config)
    else:
        config = PhoneCameraRuntimeConfig(**common_config)
        source = PhoneYuvCaptureSource(config)
    try:
        headers = []
        host_ns = []
        deadline = time.monotonic() + args.timeout_s
        while len(headers) < args.frames and time.monotonic() < deadline:
            ok, _frame = source.read()
            if not ok:
                continue
            header = source.last_header
            if header is None:
                continue
            if headers and header.sequence <= headers[-1].sequence:
                headers.clear()
                host_ns.clear()
            headers.append(header)
            host_ns.append(time.perf_counter_ns())

        if len(headers) < 2:
            print(f"Captured {len(headers)} frame(s); no benchmark possible.")
            return 2

        first = headers[0]
        last = headers[-1]
        seq_span = max(0, last.sequence - first.sequence)
        skipped = max(0, seq_span - (len(headers) - 1))
        stream_label = (
            f"codec={last.codec_name}"
            if hasattr(last, "codec_name")
            else f"fmt={last.pixel_format}"
        )
        print(
            f"Captured {len(headers)} frames via {args.source_backend}, "
            f"seq {first.sequence}->{last.sequence}, skipped={skipped}, "
            f"size={last.width}x{last.height}, {stream_label}"
        )
        _print_stats("Android timestamp interval", _intervals([h.timestamp_ns for h in headers], 1e-6))
        _print_stats("Host receive interval", _intervals(host_ns, 1e-6))
        if last.exposure_ns:
            print(
                f"Last metadata: exposure={last.exposure_ns} ns iso={last.iso} "
                f"focus={last.focus_diopters:.3f} flags=0x{last.flags:x}"
            )
    finally:
        source.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
