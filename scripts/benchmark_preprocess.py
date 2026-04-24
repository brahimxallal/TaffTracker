"""Benchmark CPU vs GPU letterbox preprocessing.

Usage::

    python scripts/benchmark_preprocess.py
    python scripts/benchmark_preprocess.py --src-h 1080 --src-w 1920 --iters 500
    python scripts/benchmark_preprocess.py --no-gpu                  # CPU only

The point of this script is to *prove* whether the GPU letterbox path is
worth turning on (``runtime.gpu_preprocess: true``) on the deployed
hardware. The GPU path beats CPU on big frames where the H2D upload is
amortized; on small frames the upload dominates and CPU wins. Don't
flip the production flag without re-running this on the target host.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Allow running as a script without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.gpu_preprocess import cpu_letterbox  # noqa: E402


def _make_random_frame(h: int, w: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _bench(fn, frame: np.ndarray, dst_h: int, dst_w: int, *, iters: int, warmup: int) -> dict:
    # Warmup
    for _ in range(warmup):
        fn(frame, dst_h, dst_w)

    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(frame, dst_h, dst_w)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "iters": iters,
        "p50_ms": statistics.median(samples),
        "p95_ms": samples[int(0.95 * iters) - 1] if iters >= 20 else samples[-1],
        "p99_ms": samples[int(0.99 * iters) - 1] if iters >= 100 else samples[-1],
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "mean_ms": statistics.fmean(samples),
    }


def _format_row(name: str, stats: dict) -> str:
    return (
        f"  {name:<10} "
        f"p50={stats['p50_ms']:6.3f} ms  "
        f"p95={stats['p95_ms']:6.3f} ms  "
        f"p99={stats['p99_ms']:6.3f} ms  "
        f"mean={stats['mean_ms']:6.3f} ms  "
        f"({stats['iters']} iters)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--src-h", type=int, default=1080)
    parser.add_argument("--src-w", type=int, default=1920)
    parser.add_argument("--dst-h", type=int, default=640)
    parser.add_argument("--dst-w", type=int, default=640)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--no-gpu", action="store_true", help="Skip the GPU benchmark")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frame = _make_random_frame(args.src_h, args.src_w, seed=args.seed)

    print(
        f"benchmark: src={args.src_w}x{args.src_h} -> dst={args.dst_w}x{args.dst_h}, "
        f"iters={args.iters}, warmup={args.warmup}"
    )

    cpu_stats = _bench(
        cpu_letterbox, frame, args.dst_h, args.dst_w, iters=args.iters, warmup=args.warmup
    )
    print(_format_row("CPU", cpu_stats))

    if args.no_gpu:
        return 0

    try:
        import torch

        if not torch.cuda.is_available():
            print("  GPU        skipped (no CUDA device)")
            return 0
    except ImportError:
        print("  GPU        skipped (PyTorch not installed)")
        return 0

    from src.inference.gpu_preprocess import gpu_letterbox

    gpu_stats = _bench(
        gpu_letterbox, frame, args.dst_h, args.dst_w, iters=args.iters, warmup=args.warmup
    )
    print(_format_row("GPU", gpu_stats))

    speedup = cpu_stats["p50_ms"] / gpu_stats["p50_ms"]
    verdict = "GPU faster" if speedup > 1.0 else "CPU faster"
    print(f"\np50 speedup: {speedup:.2f}x ({verdict})")
    if speedup < 1.0:
        print("  → leave runtime.gpu_preprocess=False on this host.")
    elif speedup < 1.2:
        print("  → marginal win; benchmark variance may be larger than the gain.")
    else:
        print("  → consider runtime.gpu_preprocess=true on this host.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
