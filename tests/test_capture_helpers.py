from __future__ import annotations

import multiprocessing as mp

import cv2
import pytest

from src.config import CameraConfig
from src.capture.process import CaptureProcess


def _make_capture_process(**cam_overrides) -> CaptureProcess:
    """Build a CaptureProcess with defaults for unit-testing helpers."""
    cam_kw = dict(width=640, height=480, fps=30, backend="auto", buffer_size=1)
    cam_kw.update(cam_overrides)
    return CaptureProcess(
        layout=None,  # not used by helpers
        write_index=None,
        source="0",
        camera_config=CameraConfig(**cam_kw),
        capture_done_event=mp.Event(),
        shutdown_event=mp.Event(),
        error_queue=mp.Queue(),
    )


# ── _resolve_source: digit string → int ─────────────────────────


@pytest.mark.unit
def test_resolve_source_digit():
    proc = _make_capture_process()
    proc._source = "0"
    assert proc._resolve_source() == 0


@pytest.mark.unit
def test_resolve_source_digit_large():
    proc = _make_capture_process()
    proc._source = "2"
    assert proc._resolve_source() == 2


# ── _resolve_source: URL stays string ────────────────────────────


@pytest.mark.unit
def test_resolve_source_url():
    proc = _make_capture_process()
    proc._source = "http://192.168.1.10:8080/video"
    result = proc._resolve_source()
    assert isinstance(result, str)
    assert result == "http://192.168.1.10:8080/video"


# ── _resolve_source: whitespace stripped ─────────────────────────


@pytest.mark.unit
def test_resolve_source_strips_whitespace():
    proc = _make_capture_process()
    proc._source = "  0  "
    assert proc._resolve_source() == 0


# ── _resolve_backends: camera int gives MSMF first ──────────────


@pytest.mark.unit
def test_resolve_backends_camera_int():
    proc = _make_capture_process(backend="auto")
    backends = proc._resolve_backends(0)
    assert cv2.CAP_MSMF in backends
    assert backends[0] == cv2.CAP_MSMF


# ── _resolve_backends: explicit dshow ────────────────────────────


@pytest.mark.unit
def test_resolve_backends_explicit_dshow():
    proc = _make_capture_process(backend="dshow")
    backends = proc._resolve_backends(0)
    assert backends == [cv2.CAP_DSHOW]


@pytest.mark.unit
def test_resolve_backends_explicit_msmf():
    proc = _make_capture_process(backend="msmf")
    backends = proc._resolve_backends(0)
    assert backends == [cv2.CAP_MSMF]


@pytest.mark.unit
def test_resolve_backends_explicit_ffmpeg():
    proc = _make_capture_process(backend="ffmpeg")
    backends = proc._resolve_backends("http://example.com/stream")
    assert backends == [cv2.CAP_FFMPEG]


# ── _resolve_backends: string source gives FFMPEG first ──────────


@pytest.mark.unit
def test_resolve_backends_string_source():
    proc = _make_capture_process(backend="auto")
    backends = proc._resolve_backends("http://192.168.1.10/stream")
    assert backends[0] == cv2.CAP_FFMPEG


# ── _resolve_playback_interval: non-file → None ─────────────────


@pytest.mark.unit
def test_resolve_playback_interval_non_file():
    proc = _make_capture_process()
    cap = cv2.VideoCapture()  # dummy
    assert proc._resolve_playback_interval(cap, source_is_file=False) is None


# ── _resolve_playback_interval: file → reciprocal of FPS ────────


@pytest.mark.unit
def test_resolve_playback_interval_positive_fps():
    from unittest.mock import MagicMock
    proc = _make_capture_process()
    cap = MagicMock()
    cap.get.side_effect = lambda prop: 30.0 if prop == cv2.CAP_PROP_FPS else 0.0
    interval = proc._resolve_playback_interval(cap, source_is_file=True)
    assert interval is not None
    assert abs(interval - 1.0 / 30.0) < 1e-6


@pytest.mark.unit
def test_resolve_playback_interval_zero_fps_uses_config():
    from unittest.mock import MagicMock
    proc = _make_capture_process(fps=60)
    cap = MagicMock()
    cap.get.return_value = 0.0
    interval = proc._resolve_playback_interval(cap, source_is_file=True)
    assert interval is not None
    assert abs(interval - 1.0 / 60.0) < 1e-6


# ── Letterbox math verification ──────────────────────────────────


@pytest.mark.unit
def test_letterbox_math_wider_frame():
    """A 1920x1080 frame into 640x480 target should scale down and pad vertically."""
    target_w, target_h = 640, 480
    w_orig, h_orig = 1920, 1080
    scale = min(target_w / w_orig, target_h / h_orig)
    new_w = int(round(w_orig * scale))
    new_h = int(round(h_orig * scale))
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2

    # 1920:1080 = 16:9, target 640:480 = 4:3
    # Scale = min(640/1920, 480/1080) = min(0.333, 0.444) = 0.333
    assert abs(scale - 1 / 3) < 0.01
    assert new_w == 640
    assert new_h == 360
    assert pad_left == 0
    assert pad_top == 60  # (480-360)/2


@pytest.mark.unit
def test_letterbox_math_taller_frame():
    """A 480x640 (portrait) frame into 640x640 target should pad horizontally."""
    target_w, target_h = 640, 640
    w_orig, h_orig = 480, 640
    scale = min(target_w / w_orig, target_h / h_orig)
    new_w = int(round(w_orig * scale))
    new_h = int(round(h_orig * scale))
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2

    assert scale == 1.0
    assert new_w == 480
    assert new_h == 640
    assert pad_left == 80   # (640-480)/2
    assert pad_top == 0


@pytest.mark.unit
def test_letterbox_math_same_aspect():
    """Same aspect ratio → no padding."""
    target_w, target_h = 640, 480
    w_orig, h_orig = 1280, 960
    scale = min(target_w / w_orig, target_h / h_orig)
    new_w = int(round(w_orig * scale))
    new_h = int(round(h_orig * scale))
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2

    assert new_w == 640
    assert new_h == 480
    assert pad_left == 0
    assert pad_top == 0
