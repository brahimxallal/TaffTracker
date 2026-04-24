"""Optional GPU-side letterbox preprocessing.

The default capture path letterboxes on the CPU using ``cv2.resize`` plus a
pre-allocated padded buffer (see :mod:`src.capture.process`). That path is
already fast enough for 1080p60, but on weaker CPUs or when the camera is
configured for very high resolutions it can become the dominant cost ahead
of TensorRT inference.

This module provides a **second, opt-in** letterbox implementation that
runs entirely on the GPU using PyTorch tensor ops. It is gated by
:attr:`src.config.RuntimeFlags.gpu_preprocess` and is **not wired into the
live pipeline** — it ships only as a benchmark-validated utility plus a
NumPy reference (:func:`cpu_letterbox`) that lets us assert numerical
parity in tests, regardless of whether CUDA is available on the test
machine.

The GPU path is intentionally pure (no I/O, no logging, no global state)
so it can be benchmarked head-to-head with the CPU path via
``scripts/benchmark_preprocess.py``.

Activation policy:
    1. Set ``runtime.gpu_preprocess: true`` in ``config.yaml`` (or pass
       ``--gpu-preprocess`` once the CLI flag is wired up).
    2. The capture process can then call :func:`gpu_letterbox` instead of
       its inline CPU letterbox. This wiring is deliberately deferred —
       the pipeline already meets its latency target on CPU letterbox,
       and we want a benchmark to *prove* the GPU path is faster on the
       deployed hardware before flipping the default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LetterboxParams:
    """Geometry of a letterbox transform.

    All values are integer pixel offsets; ``scale`` is unitless. Useful as
    a return value when the caller needs to map detection coordinates
    back to original-frame coordinates.
    """

    new_w: int
    new_h: int
    pad_left: int
    pad_top: int
    scale: float


def compute_letterbox_params(
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
) -> LetterboxParams:
    """Compute the resize + pad geometry for a letterbox transform.

    The output preserves aspect ratio and fits the source image inside the
    destination canvas with symmetric padding (off-by-one biases toward
    top-left, matching ``cv2.resize`` + manual pad behavior).
    """
    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        raise ValueError(
            f"Letterbox dimensions must be positive: src={src_w}x{src_h}, dst={dst_w}x{dst_h}"
        )
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    pad_left = (dst_w - new_w) // 2
    pad_top = (dst_h - new_h) // 2
    return LetterboxParams(
        new_w=new_w, new_h=new_h, pad_left=pad_left, pad_top=pad_top, scale=scale
    )


def cpu_letterbox(
    frame: np.ndarray,
    dst_h: int,
    dst_w: int,
    *,
    pad_value: int = 114,
) -> tuple[np.ndarray, LetterboxParams]:
    """Reference CPU letterbox using ``cv2.resize``.

    Mirrors the live capture-path behavior so test outputs and the
    GPU path can be compared against the same baseline.
    """
    import cv2

    src_h, src_w = frame.shape[:2]
    params = compute_letterbox_params(src_h, src_w, dst_h, dst_w)

    out = np.full((dst_h, dst_w, 3), pad_value, dtype=np.uint8)
    if params.new_w == src_w and params.new_h == src_h:
        scaled = frame
    else:
        scaled = cv2.resize(
            frame,
            (params.new_w, params.new_h),
            interpolation=cv2.INTER_LINEAR,
        )
    out[
        params.pad_top : params.pad_top + params.new_h,
        params.pad_left : params.pad_left + params.new_w,
    ] = scaled
    return out, params


def gpu_letterbox(
    frame: np.ndarray,
    dst_h: int,
    dst_w: int,
    *,
    pad_value: int = 114,
):
    """Letterbox a frame on the GPU using PyTorch tensor ops.

    Returns a ``(tensor, params)`` pair where ``tensor`` is a CUDA tensor
    in HWC uint8 layout (matching the existing capture-path output shape,
    so it remains a drop-in replacement for the CPU buffer when wired up).

    Raises :class:`RuntimeError` if PyTorch with CUDA is not available.
    The benchmark and live wiring both gate on
    :attr:`src.config.RuntimeFlags.gpu_preprocess`, so this function is
    never reached in environments that lack CUDA.

    The implementation uses ``torch.nn.functional.interpolate`` with
    bilinear interpolation, which matches ``cv2.INTER_LINEAR`` to within a
    few quantization steps. Tests assert parity within a small tolerance
    (see ``tests/test_gpu_preprocess.py``).
    """
    try:
        import torch
        import torch.nn.functional as F  # noqa: N812
    except ImportError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "gpu_letterbox requires PyTorch. Install with `pip install torch`."
        ) from exc

    if not torch.cuda.is_available():  # pragma: no cover - hardware-dependent
        raise RuntimeError("gpu_letterbox requires CUDA. No CUDA device detected.")

    src_h, src_w = frame.shape[:2]
    params = compute_letterbox_params(src_h, src_w, dst_h, dst_w)

    # Upload: HWC uint8 → CHW float32 on GPU
    src_tensor = torch.from_numpy(frame).to(device="cuda", non_blocking=True)
    src_chw = src_tensor.permute(2, 0, 1).unsqueeze(0).float()

    if params.new_w != src_w or params.new_h != src_h:
        scaled = F.interpolate(
            src_chw,
            size=(params.new_h, params.new_w),
            mode="bilinear",
            align_corners=False,
        )
    else:
        scaled = src_chw

    # Build the output canvas pre-filled with pad_value.
    out = torch.full(
        (1, 3, dst_h, dst_w),
        float(pad_value),
        dtype=torch.float32,
        device="cuda",
    )
    out[
        :,
        :,
        params.pad_top : params.pad_top + params.new_h,
        params.pad_left : params.pad_left + params.new_w,
    ] = scaled

    # Return as HWC uint8 tensor for shape parity with the capture buffer.
    out_hwc = out.squeeze(0).clamp(0, 255).byte().permute(1, 2, 0).contiguous()
    return out_hwc, params
