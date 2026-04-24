"""Deterministic tests for :mod:`src.inference.gpu_preprocess`.

These cover the math and the CPU reference path. The actual GPU kernel
is exercised by ``scripts/benchmark_preprocess.py`` on hardware; the
production tests must pass on a CUDA-less CI runner.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.inference.gpu_preprocess import (
    LetterboxParams,
    compute_letterbox_params,
    cpu_letterbox,
)


@pytest.mark.unit
def test_compute_letterbox_params_square_to_square_is_identity() -> None:
    p = compute_letterbox_params(640, 640, 640, 640)
    assert p == LetterboxParams(new_w=640, new_h=640, pad_left=0, pad_top=0, scale=1.0)


@pytest.mark.unit
def test_compute_letterbox_params_landscape_into_square_pads_top_bottom() -> None:
    """1920x1080 → 640x640 should resize to 640x360 and pad 140 top/bottom."""
    p = compute_letterbox_params(src_h=1080, src_w=1920, dst_h=640, dst_w=640)
    assert p.new_w == 640
    assert p.new_h == 360
    assert p.pad_left == 0
    assert p.pad_top == 140  # (640 - 360) // 2
    assert p.scale == pytest.approx(640 / 1920)


@pytest.mark.unit
def test_compute_letterbox_params_portrait_into_square_pads_left_right() -> None:
    p = compute_letterbox_params(src_h=1920, src_w=1080, dst_h=640, dst_w=640)
    assert p.new_w == 360
    assert p.new_h == 640
    assert p.pad_top == 0
    assert p.pad_left == 140


@pytest.mark.unit
def test_compute_letterbox_params_rejects_zero_dimensions() -> None:
    for args in [
        (0, 100, 640, 640),
        (100, 0, 640, 640),
        (100, 100, 0, 640),
        (100, 100, 640, 0),
    ]:
        with pytest.raises(ValueError):
            compute_letterbox_params(*args)


@pytest.mark.unit
def test_cpu_letterbox_output_has_target_shape_and_dtype() -> None:
    src = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    out, params = cpu_letterbox(src, dst_h=640, dst_w=640)
    assert out.shape == (640, 640, 3)
    assert out.dtype == np.uint8
    assert params.new_h == 360
    assert params.new_w == 640


@pytest.mark.unit
def test_cpu_letterbox_pads_with_default_gray_value() -> None:
    """Top and bottom pad strips must be the YOLO default 114 gray."""
    src = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    out, params = cpu_letterbox(src, dst_h=640, dst_w=640)
    # Top pad strip
    assert np.all(out[: params.pad_top, :, :] == 114)
    # Bottom pad strip
    assert np.all(out[params.pad_top + params.new_h :, :, :] == 114)


@pytest.mark.unit
def test_cpu_letterbox_pads_with_custom_value() -> None:
    src = np.full((1080, 1920, 3), 50, dtype=np.uint8)
    out, params = cpu_letterbox(src, dst_h=640, dst_w=640, pad_value=0)
    assert np.all(out[: params.pad_top, :, :] == 0)


@pytest.mark.unit
def test_cpu_letterbox_preserves_content_in_active_region() -> None:
    """A constant-color source should land inside the active region intact."""
    src = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    out, p = cpu_letterbox(src, dst_h=640, dst_w=640)
    active = out[p.pad_top : p.pad_top + p.new_h, p.pad_left : p.pad_left + p.new_w]
    # cv2.INTER_LINEAR of a constant image is exactly that constant.
    assert np.all(active == 200)


@pytest.mark.unit
def test_cpu_letterbox_identity_when_source_matches_dest() -> None:
    rng = np.random.default_rng(0)
    src = rng.integers(0, 256, size=(640, 640, 3), dtype=np.uint8)
    out, params = cpu_letterbox(src, dst_h=640, dst_w=640)
    assert params.scale == 1.0
    np.testing.assert_array_equal(out, src)


@pytest.mark.unit
def test_runtime_flags_exposes_gpu_preprocess_off_by_default() -> None:
    """The wiring contract: gpu_preprocess must default to False so a fresh
    install never silently changes behavior."""
    from src.config import RuntimeFlags

    flags = RuntimeFlags()
    assert flags.gpu_preprocess is False


@pytest.mark.unit
def test_runtime_flags_gpu_preprocess_is_settable() -> None:
    from src.config import RuntimeFlags

    flags = RuntimeFlags(gpu_preprocess=True)
    assert flags.gpu_preprocess is True
