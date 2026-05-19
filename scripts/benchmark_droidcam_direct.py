from __future__ import annotations

import argparse
import hashlib
import statistics
import time

import cv2

from src.capture.droidcam import DroidCamDirectCaptureSource, DroidCamRuntimeConfig


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((percentile / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark DroidCam Classic's direct /v5/video encoded TCP stream. "
            "Close the Windows DroidCam client first; DroidCam usually allows one video client."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4747)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument(
        "--format",
        choices=("avc", "jpg", "mjpg", "hevc"),
        default="jpg",
        help="DroidCam direct stream format; mjpg is accepted as an alias for jpg",
    )
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--read-timeout-s", type=float, default=0.03)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = DroidCamDirectCaptureSource(
        DroidCamRuntimeConfig(
            host=args.host,
            port=args.port,
            width=args.width,
            height=args.height,
            fps=args.fps,
            video_format=args.format,
            read_timeout_s=args.read_timeout_s,
        )
    )

    host_ns: list[int] = []
    hashes: list[str] = []
    diffs: list[float] = []
    previous = None
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline and len(host_ns) < args.max_frames:
            ok, frame = source.read()
            if not ok or frame is None:
                time.sleep(0.002)
                continue
            now_ns = time.perf_counter_ns()
            small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hashes.append(hashlib.blake2b(gray.tobytes(), digest_size=8).hexdigest())
            if previous is not None:
                diffs.append(float(cv2.mean(cv2.absdiff(gray, previous))[0]))
            previous = gray
            host_ns.append(now_ns)
    finally:
        source.release()

    if len(host_ns) < 2:
        print(
            f"captured={len(host_ns)}; no direct stream. "
            "Close the DroidCam PC client, then rerun this benchmark."
        )
        return 1

    intervals_ms = [(b - a) / 1_000_000.0 for a, b in zip(host_ns, host_ns[1:], strict=False)]
    median_ms = statistics.median(intervals_ms)
    unique_hashes = len(set(hashes))
    exact_duplicate_ratio = 1.0 - unique_hashes / max(1, len(hashes))
    print(
        f"captured={len(host_ns)} unique_hashes={unique_hashes} "
        f"exact_duplicate_ratio={exact_duplicate_ratio:.3f}"
    )
    print(
        f"host_interval median={median_ms:.2f}ms p95={_percentile(intervals_ms, 95):.2f}ms "
        f"approx_fps={1000.0 / median_ms:.1f}"
    )
    if diffs:
        print(f"pixel_diff median={statistics.median(diffs):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
