from __future__ import annotations

import argparse
import hashlib
import statistics
import time

import cv2
import numpy as np


def _backend_id(name: str) -> int:
    return {
        "any": cv2.CAP_ANY,
        "msmf": cv2.CAP_MSMF,
        "dshow": cv2.CAP_DSHOW,
    }[name]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((percentile / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether DroidCam/OpenCV delivers unique pixel frames or only "
            "a 60 Hz virtual-device clock. Move a hand in front of the phone during the run."
        )
    )
    parser.add_argument("--source", default="0")
    parser.add_argument("--backend", choices=("msmf", "dshow", "any"), default="msmf")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--diff-threshold", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source: int | str = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source, _backend_id(args.backend))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("open_failed")
        return 2

    print(
        "actual "
        f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} "
        f"fps={cap.get(cv2.CAP_PROP_FPS):.1f} backend={args.backend}"
    )
    print("Move your hand/face during the test so unique-frame detection is meaningful.")

    host_ns: list[int] = []
    hashes: list[str] = []
    diffs: list[float] = []
    previous: np.ndarray | None = None
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline and len(host_ns) < args.max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        now_ns = time.perf_counter_ns()
        small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hashes.append(hashlib.blake2b(gray.tobytes(), digest_size=8).hexdigest())
        if previous is not None:
            diffs.append(float(np.mean(cv2.absdiff(gray, previous))))
        previous = gray
        host_ns.append(now_ns)
    cap.release()

    if len(host_ns) < 2:
        print(f"captured={len(host_ns)}; not enough frames")
        return 1

    intervals_ms = [(b - a) / 1_000_000.0 for a, b in zip(host_ns, host_ns[1:], strict=False)]
    median_ms = statistics.median(intervals_ms)
    unique_hashes = len(set(hashes))
    exact_duplicate_ratio = 1.0 - unique_hashes / max(1, len(hashes))
    near_duplicates = sum(1 for value in diffs if value < args.diff_threshold)
    near_duplicate_ratio = near_duplicates / max(1, len(diffs))
    print(
        f"captured={len(host_ns)} unique_hashes={unique_hashes} "
        f"exact_duplicate_ratio={exact_duplicate_ratio:.3f}"
    )
    print(
        f"host_interval median={median_ms:.2f}ms p95={_percentile(intervals_ms, 95):.2f}ms "
        f"approx_fps={1000.0 / median_ms:.1f}"
    )
    print(
        f"pixel_diff median={statistics.median(diffs):.3f} "
        f"near_duplicate_ratio_lt_{args.diff_threshold:g}={near_duplicate_ratio:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
