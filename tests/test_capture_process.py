from __future__ import annotations

import multiprocessing as mp

import cv2
import numpy as np
import pytest

from src.capture.process import CaptureProcess
from src.config import CameraConfig
from src.shared.ring_buffer import SharedRingBuffer


class _SeekableCapture:
    def __init__(self) -> None:
        self.positions: list[tuple[int, int]] = []

    def isOpened(self) -> bool:
        return True

    def set(self, prop_id: int, value: int) -> bool:
        self.positions.append((prop_id, value))
        return True

    def release(self) -> None:
        return None


class _UnseekableCapture:
    def __init__(self) -> None:
        self.released = False

    def set(self, prop_id: int, value: int) -> bool:
        return False

    def release(self) -> None:
        self.released = True


def _build_capture_process(**overrides) -> CaptureProcess:
    ring_buffer, write_index = SharedRingBuffer.create((2, 2, 3), num_slots=3)
    ring_buffer.cleanup()
    defaults = dict(
        layout=ring_buffer.layout,
        write_index=write_index,
        source="c:\\AAA\\videos\\Dog_Moving_Camera_Still.mp4",
        camera_config=CameraConfig(width=640, height=640, fps=60, backend="auto", buffer_size=1),
        capture_done_event=mp.Event(),
        shutdown_event=mp.Event(),
        error_queue=mp.Queue(),
    )
    defaults.update(overrides)
    return CaptureProcess(**defaults)


@pytest.mark.unit
def test_capture_process_defaults_gpu_preprocess_off() -> None:
    """Wiring contract: default constructor keeps the CPU letterbox path."""
    process = _build_capture_process()
    assert process._gpu_preprocess is False


@pytest.mark.unit
def test_capture_process_accepts_gpu_preprocess_flag() -> None:
    process = _build_capture_process(gpu_preprocess=True)
    assert process._gpu_preprocess is True


@pytest.mark.unit
def test_rewind_file_capture_seeks_to_frame_zero() -> None:
    process = _build_capture_process()
    capture = _SeekableCapture()

    result = process._rewind_file_capture(capture)

    assert result is capture
    assert capture.positions == [(cv2.CAP_PROP_POS_FRAMES, 0)]


@pytest.mark.unit
def test_rewind_file_capture_reopens_when_seek_fails(monkeypatch) -> None:
    process = _build_capture_process()
    capture = _UnseekableCapture()
    replacement = _SeekableCapture()
    monkeypatch.setattr(process, "_open_capture", lambda: replacement)

    result = process._rewind_file_capture(capture)

    assert capture.released is True
    assert result is replacement


# --- _resolve_source tests ---


@pytest.mark.unit
def test_resolve_source_integer_string() -> None:
    process = _build_capture_process(source="0")
    assert process._resolve_source() == 0


@pytest.mark.unit
def test_resolve_source_url_string() -> None:
    process = _build_capture_process(source="http://192.168.1.100:4747/video")
    assert process._resolve_source() == "http://192.168.1.100:4747/video"


@pytest.mark.unit
def test_resolve_source_file_path(tmp_path) -> None:
    video = tmp_path / "test.mp4"
    video.write_bytes(b"\x00" * 10)
    process = _build_capture_process(source=str(video))
    result = process._resolve_source()
    assert str(result) == str(video)


# --- _resolve_backends tests ---


@pytest.mark.unit
def test_resolve_backends_explicit_dshow() -> None:
    process = _build_capture_process(
        camera_config=CameraConfig(width=640, height=640, fps=60, backend="dshow"),
    )
    assert process._resolve_backends(0) == [cv2.CAP_DSHOW]


@pytest.mark.unit
def test_resolve_backends_explicit_ffmpeg() -> None:
    process = _build_capture_process(
        camera_config=CameraConfig(width=640, height=640, fps=60, backend="ffmpeg"),
    )
    assert process._resolve_backends("video.mp4") == [cv2.CAP_FFMPEG]


@pytest.mark.unit
def test_resolve_backends_auto_integer_source() -> None:
    process = _build_capture_process()
    backends = process._resolve_backends(0)
    assert cv2.CAP_MSMF in backends
    assert cv2.CAP_DSHOW in backends
    assert cv2.CAP_ANY in backends


@pytest.mark.unit
def test_resolve_backends_auto_string_source() -> None:
    process = _build_capture_process()
    backends = process._resolve_backends("video.mp4")
    assert cv2.CAP_FFMPEG in backends
    assert cv2.CAP_ANY in backends


# --- _resolve_playback_interval tests ---


class _FakeCapture:
    def __init__(self, fps: float = 30.0):
        self._fps = fps

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS:
            return self._fps
        return 0.0

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def release(self):
        pass


@pytest.mark.unit
def test_playback_interval_for_file() -> None:
    process = _build_capture_process()
    cap = _FakeCapture(fps=25.0)
    interval = process._resolve_playback_interval(cap, source_is_file=True)
    assert interval is not None
    assert abs(interval - 1.0 / 25.0) < 1e-6


@pytest.mark.unit
def test_playback_interval_for_camera_is_none() -> None:
    process = _build_capture_process()
    cap = _FakeCapture(fps=60.0)
    interval = process._resolve_playback_interval(cap, source_is_file=False)
    assert interval is None


@pytest.mark.unit
def test_playback_interval_zero_fps_falls_back() -> None:
    process = _build_capture_process(
        camera_config=CameraConfig(width=640, height=640, fps=30),
    )
    cap = _FakeCapture(fps=0.0)
    interval = process._resolve_playback_interval(cap, source_is_file=True)
    assert interval is not None
    assert abs(interval - 1.0 / 30.0) < 1e-6


# --- _report_error tests ---


@pytest.mark.unit
def test_capture_report_error_succeeds() -> None:
    error_queue = mp.Queue()
    process = _build_capture_process(error_queue=error_queue)

    process._report_error(RuntimeError("capture boom"))

    report = error_queue.get(timeout=1.0)
    assert "capture boom" in report.summary
    assert report.process_name == "CaptureProcess"


@pytest.mark.unit
def test_rewind_returns_none_when_both_fail(monkeypatch) -> None:
    process = _build_capture_process()
    capture = _UnseekableCapture()

    class _FailedCapture:
        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(process, "_open_capture", lambda: _FailedCapture())
    result = process._rewind_file_capture(capture)
    assert result is None


# --- _open_capture tests ---


class _MockVideoCapture:
    """Mock cv2.VideoCapture that can be configured to open or fail."""

    def __init__(
        self,
        opens: bool = True,
        native_w: int = 640,
        native_h: int = 480,
        fps: float = 30.0,
        backend_name: str = "MSMF",
    ):
        self._opens = opens
        self._props = {
            cv2.CAP_PROP_FRAME_WIDTH: float(native_w),
            cv2.CAP_PROP_FRAME_HEIGHT: float(native_h),
            cv2.CAP_PROP_FPS: fps,
        }
        self._backend_name = backend_name
        self.released = False
        self.set_calls: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return self._opens

    def get(self, prop_id: int) -> float:
        return self._props.get(prop_id, 0.0)

    def set(self, prop_id: int, value: float) -> bool:
        self.set_calls.append((prop_id, value))
        return True

    def getBackendName(self) -> str:
        return self._backend_name

    def release(self) -> None:
        self.released = True


@pytest.mark.unit
def test_open_capture_succeeds(monkeypatch) -> None:
    mock_cap = _MockVideoCapture(opens=True, native_w=640, native_h=640)
    monkeypatch.setattr(cv2, "VideoCapture", lambda src, backend: mock_cap)
    process = _build_capture_process(source="0")

    result = process._open_capture()

    assert result is mock_cap
    # Should have set buffer_size and fps
    props_set = {prop for prop, _ in mock_cap.set_calls}
    assert cv2.CAP_PROP_BUFFERSIZE in props_set
    assert cv2.CAP_PROP_FPS in props_set


@pytest.mark.unit
def test_open_capture_logs_actual_resolution(monkeypatch, caplog) -> None:
    import logging

    mock_cap = _MockVideoCapture(opens=True, native_w=1920, native_h=1080, fps=60.0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda src, backend: mock_cap)
    process = _build_capture_process(source="0")

    with caplog.at_level(logging.INFO, logger="capture"):
        result = process._open_capture()

    assert result is mock_cap
    assert any("camera: actual 1920x1080 @ 60.0 fps" in r.message for r in caplog.records)


@pytest.mark.unit
def test_open_capture_warns_on_mismatch(monkeypatch, caplog) -> None:
    import logging

    # 800x600 vs capture_width=1920 capture_height=1080: >5% deviation, plus fps mismatch
    mock_cap = _MockVideoCapture(opens=True, native_w=800, native_h=600, fps=25.0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda src, backend: mock_cap)
    process = _build_capture_process(
        source="0",
        camera_config=CameraConfig(width=640, height=640, fps=60),
    )

    with caplog.at_level(logging.WARNING, logger="capture"):
        result = process._open_capture()

    assert result is mock_cap
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) >= 1
    msg = warnings[0].message
    assert "Camera property mismatch" in msg
    assert "width" in msg
    assert "height" in msg
    assert "fps" in msg


@pytest.mark.unit
def test_open_capture_no_warn_within_5pct(monkeypatch, caplog) -> None:
    import logging

    # 1920x1080 @ 59fps vs capture 1920x1080 @ 60fps → fps deviation ~1.7% < 5%
    mock_cap = _MockVideoCapture(opens=True, native_w=1920, native_h=1080, fps=59.0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda src, backend: mock_cap)
    process = _build_capture_process(
        source="0",
        camera_config=CameraConfig(
            width=640,
            height=640,
            fps=60,
            capture_width=1920,
            capture_height=1080,
        ),
    )

    with caplog.at_level(logging.WARNING, logger="capture"):
        result = process._open_capture()

    assert result is mock_cap
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 0


@pytest.mark.unit
def test_open_capture_fails_all_backends(monkeypatch) -> None:
    mock_cap = _MockVideoCapture(opens=False)
    monkeypatch.setattr(cv2, "VideoCapture", lambda src, backend: mock_cap)
    process = _build_capture_process(source="0")

    with pytest.raises(RuntimeError, match="Failed to open capture source"):
        process._open_capture()


# --- run() loop tests ---


class _FrameCapture:
    """Mock capture that yields a fixed number of frames then stops."""

    def __init__(self, frames: list[np.ndarray]):
        self._frames = list(frames)
        self._idx = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return 640.0
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return 480.0
        if prop_id == cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        return True

    def read(self):
        if self._idx < len(self._frames):
            frame = self._frames[self._idx]
            self._idx += 1
            return True, frame
        return False, None

    def release(self) -> None:
        self.released = True


@pytest.mark.unit
def test_run_captures_frames_with_letterbox(monkeypatch) -> None:
    """run() reads frames, applies letterbox, writes to ring buffer."""
    ring_buffer, write_index = SharedRingBuffer.create((640, 640, 3), num_slots=3)
    shutdown_event = mp.Event()
    capture_done = mp.Event()

    proc = CaptureProcess(
        layout=ring_buffer.layout,
        write_index=write_index,
        source="0",
        camera_config=CameraConfig(width=640, height=640, fps=30),
        capture_done_event=capture_done,
        shutdown_event=shutdown_event,
        error_queue=mp.Queue(),
    )

    # 480p frame → letterboxed into 640×640
    frames = [np.full((480, 640, 3), 200, dtype=np.uint8) for _ in range(3)]
    mock_cap = _FrameCapture(frames)
    monkeypatch.setattr(proc, "_open_capture", lambda: mock_cap)
    # After frames exhausted, source is integer (stream), so it'll sleep+continue.
    # Set shutdown after 3 frames via frame count check.
    original_write = ring_buffer.write
    write_count = [0]

    def counting_write(frame, ts):
        original_write(frame, ts)
        write_count[0] += 1
        if write_count[0] >= 3:
            shutdown_event.set()

    mock_rb = SharedRingBuffer.attach(ring_buffer.layout, write_index)
    mock_rb.write = counting_write
    monkeypatch.setattr("src.capture.process.SharedRingBuffer.attach", lambda layout, wi: mock_rb)

    proc.run()

    assert write_count[0] == 3
    assert capture_done.is_set()
    assert mock_cap.released
    ring_buffer.cleanup()


@pytest.mark.unit
def test_run_stream_retries_on_read_failure(monkeypatch) -> None:
    """Integer/stream source: read failure → sleep + continue, not shutdown."""
    ring_buffer, write_index = SharedRingBuffer.create((8, 8, 3), num_slots=3)
    shutdown_event = mp.Event()

    proc = CaptureProcess(
        layout=ring_buffer.layout,
        write_index=write_index,
        source="0",  # integer source
        camera_config=CameraConfig(width=8, height=8, fps=30),
        capture_done_event=mp.Event(),
        shutdown_event=shutdown_event,
        error_queue=mp.Queue(),
    )

    # Sequence: fail, fail, success, then shutdown
    read_results = [
        (False, None),  # retry
        (False, None),  # retry
        (True, np.full((8, 8, 3), 55, dtype=np.uint8)),
    ]
    read_idx = [0]

    class _RetryCapture:
        def isOpened(self):
            return True

        def get(self, prop_id):
            return 8.0 if prop_id != cv2.CAP_PROP_FPS else 30.0

        def set(self, *args):
            return True

        def read(self):
            if read_idx[0] < len(read_results):
                result = read_results[read_idx[0]]
                read_idx[0] += 1
                return result
            shutdown_event.set()
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(proc, "_open_capture", lambda: _RetryCapture())
    # Stub time.sleep to avoid actual delay
    monkeypatch.setattr("src.capture.process.time.sleep", lambda s: None)

    write_count = [0]
    mock_rb = SharedRingBuffer.attach(ring_buffer.layout, write_index)
    original_write = mock_rb.write

    def counting_write(frame, ts):
        original_write(frame, ts)
        write_count[0] += 1
        shutdown_event.set()

    mock_rb.write = counting_write
    monkeypatch.setattr("src.capture.process.SharedRingBuffer.attach", lambda layout, wi: mock_rb)

    proc.run()

    assert write_count[0] == 1  # only the successful read was written
    ring_buffer.cleanup()


@pytest.mark.unit
def test_run_fps_logging_triggers_at_300(monkeypatch) -> None:
    """FPS logging triggers every 300 frames."""
    ring_buffer, write_index = SharedRingBuffer.create((4, 4, 3), num_slots=3)
    shutdown_event = mp.Event()

    proc = CaptureProcess(
        layout=ring_buffer.layout,
        write_index=write_index,
        source="0",
        camera_config=CameraConfig(width=4, height=4, fps=30),
        capture_done_event=mp.Event(),
        shutdown_event=shutdown_event,
        error_queue=mp.Queue(),
    )

    # Generate 301 frames to trigger the FPS logging at frame 300
    frame = np.full((4, 4, 3), 99, dtype=np.uint8)
    frames = [frame.copy() for _ in range(301)]
    mock_cap = _FrameCapture(frames)

    monkeypatch.setattr(proc, "_open_capture", lambda: mock_cap)

    write_count = [0]
    mock_rb = SharedRingBuffer.attach(ring_buffer.layout, write_index)
    original_write = mock_rb.write

    def counting_write(f, ts):
        original_write(f, ts)
        write_count[0] += 1
        if write_count[0] >= 301:
            shutdown_event.set()

    mock_rb.write = counting_write
    monkeypatch.setattr("src.capture.process.SharedRingBuffer.attach", lambda layout, wi: mock_rb)

    proc.run()

    # Should have processed 301 frames, FPS counter was reset at 300
    assert proc._frame_count == 1  # reset to 0 at 300, then +1
    ring_buffer.cleanup()


@pytest.mark.unit
def test_run_file_source_rewinds_on_exhaustion(monkeypatch, tmp_path) -> None:
    """File source: when frames run out, capture rewinds and keeps going."""
    ring_buffer, write_index = SharedRingBuffer.create((8, 8, 3), num_slots=3)
    shutdown_event = mp.Event()

    video_path = tmp_path / "dummy.mp4"
    video_path.write_bytes(b"\x00" * 10)

    proc = CaptureProcess(
        layout=ring_buffer.layout,
        write_index=write_index,
        source=str(video_path),
        camera_config=CameraConfig(width=8, height=8, fps=30),
        capture_done_event=mp.Event(),
        shutdown_event=shutdown_event,
        error_queue=mp.Queue(),
    )

    # First capture: 2 frames then exhausted → rewind → 1 more frame → shutdown
    read_results = [
        (True, np.full((8, 8, 3), 100, dtype=np.uint8)),
        (True, np.full((8, 8, 3), 100, dtype=np.uint8)),
        (False, None),  # triggers rewind
        (True, np.full((8, 8, 3), 100, dtype=np.uint8)),
        (False, None),  # triggers second rewind → we shut down
    ]
    call_idx = [0]

    class _RewindCapture:
        def __init__(self):
            self.released = False

        def isOpened(self):
            return True

        def get(self, prop_id):
            if prop_id == cv2.CAP_PROP_FPS:
                return 30.0
            return 8.0

        def set(self, *args):
            return True

        def read(self):
            if call_idx[0] < len(read_results):
                result = read_results[call_idx[0]]
                call_idx[0] += 1
                return result
            shutdown_event.set()
            return False, None

        def release(self):
            self.released = True

    mock_cap = _RewindCapture()
    monkeypatch.setattr(proc, "_open_capture", lambda: mock_cap)

    # Rewind succeeds via seek
    monkeypatch.setattr(proc, "_rewind_file_capture", lambda cap: cap)

    write_count = [0]
    mock_rb = SharedRingBuffer.attach(ring_buffer.layout, write_index)
    original_write = mock_rb.write

    def counting_write(frame, ts):
        original_write(frame, ts)
        write_count[0] += 1

    mock_rb.write = counting_write
    monkeypatch.setattr("src.capture.process.SharedRingBuffer.attach", lambda layout, wi: mock_rb)

    proc.run()

    assert write_count[0] == 3
    ring_buffer.cleanup()


@pytest.mark.unit
def test_run_file_source_stops_on_failed_rewind(monkeypatch, tmp_path) -> None:
    """File source: if rewind fails, capture stops."""
    ring_buffer, write_index = SharedRingBuffer.create((8, 8, 3), num_slots=3)
    shutdown_event = mp.Event()

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"\x00" * 10)

    proc = CaptureProcess(
        layout=ring_buffer.layout,
        write_index=write_index,
        source=str(video_path),
        camera_config=CameraConfig(width=8, height=8, fps=30),
        capture_done_event=mp.Event(),
        shutdown_event=shutdown_event,
        error_queue=mp.Queue(),
    )

    class _ExhaustCapture:
        def isOpened(self):
            return True

        def get(self, prop_id):
            if prop_id == cv2.CAP_PROP_FPS:
                return 30.0
            return 8.0

        def set(self, *args):
            return True

        def read(self):
            return False, None  # immediately exhausted

        def release(self):
            pass

    monkeypatch.setattr(proc, "_open_capture", lambda: _ExhaustCapture())
    monkeypatch.setattr(proc, "_rewind_file_capture", lambda cap: None)  # rewind fails
    mock_rb_cls = type("MockRB", (), {"close": lambda self: None})
    monkeypatch.setattr(
        "src.capture.process.SharedRingBuffer.attach", lambda layout, wi: mock_rb_cls()
    )

    proc.run()

    assert shutdown_event.is_set()
    ring_buffer.cleanup()


@pytest.mark.unit
def test_run_padding_only_path(monkeypatch) -> None:
    """When frame needs padding but no resize (same scale), padding path is used."""
    # Target: 10x8, source: 8x8 → scale=1.0 on height, pad left/right
    ring_buffer, write_index = SharedRingBuffer.create((8, 10, 3), num_slots=3)
    shutdown_event = mp.Event()

    proc = CaptureProcess(
        layout=ring_buffer.layout,
        write_index=write_index,
        source="0",
        camera_config=CameraConfig(width=10, height=8, fps=30),
        capture_done_event=mp.Event(),
        shutdown_event=shutdown_event,
        error_queue=mp.Queue(),
    )

    # Source frame: 8x8 → target 10x8 → scale=min(10/8, 8/8)=1.0
    # lb_new_w=8, lb_new_h=8, pad_left=1, pad_top=0
    frames = [np.full((8, 8, 3), 77, dtype=np.uint8)]

    mock_cap = _FrameCapture(frames)
    monkeypatch.setattr(proc, "_open_capture", lambda: mock_cap)

    written = []
    mock_rb = SharedRingBuffer.attach(ring_buffer.layout, write_index)
    original_write = mock_rb.write

    def capture_write(frame, ts):
        written.append(frame.copy())
        original_write(frame, ts)
        shutdown_event.set()

    mock_rb.write = capture_write
    monkeypatch.setattr("src.capture.process.SharedRingBuffer.attach", lambda layout, wi: mock_rb)

    proc.run()

    assert len(written) == 1
    # Center pixels should have our value (77), edge padding should be 114
    assert written[0][0, 0, 0] == 114  # left pad
    assert written[0][0, 1, 0] == 77  # content starts at pad_left=1
    ring_buffer.cleanup()


@pytest.mark.unit
def test_run_no_resize_when_same_size(monkeypatch) -> None:
    """When source matches target, np.copyto path is used (no resize)."""
    ring_buffer, write_index = SharedRingBuffer.create((8, 8, 3), num_slots=3)
    shutdown_event = mp.Event()

    proc = CaptureProcess(
        layout=ring_buffer.layout,
        write_index=write_index,
        source="0",
        camera_config=CameraConfig(width=8, height=8, fps=30),
        capture_done_event=mp.Event(),
        shutdown_event=shutdown_event,
        error_queue=mp.Queue(),
    )

    frames = [np.full((8, 8, 3), 42, dtype=np.uint8)]
    mock_cap = _FrameCapture(frames)
    monkeypatch.setattr(proc, "_open_capture", lambda: mock_cap)

    mock_rb = SharedRingBuffer.attach(ring_buffer.layout, write_index)
    original_write = mock_rb.write
    written = []

    def capture_write(frame, ts):
        written.append(frame.copy())
        original_write(frame, ts)
        shutdown_event.set()

    mock_rb.write = capture_write
    monkeypatch.setattr("src.capture.process.SharedRingBuffer.attach", lambda layout, wi: mock_rb)

    proc.run()

    assert len(written) == 1
    assert written[0][0, 0, 0] == 42  # pixel value preserved
    ring_buffer.cleanup()


@pytest.mark.unit
def test_run_exception_reports_error(monkeypatch) -> None:
    """When _open_capture raises, error gets reported."""
    ring_buffer, write_index = SharedRingBuffer.create((4, 4, 3), num_slots=2)
    error_queue = mp.Queue()
    shutdown_event = mp.Event()
    capture_done = mp.Event()

    proc = CaptureProcess(
        layout=ring_buffer.layout,
        write_index=write_index,
        source="0",
        camera_config=CameraConfig(width=4, height=4, fps=30),
        capture_done_event=capture_done,
        shutdown_event=shutdown_event,
        error_queue=error_queue,
    )

    monkeypatch.setattr(
        proc, "_open_capture", lambda: (_ for _ in ()).throw(RuntimeError("no camera"))
    )
    mock_rb_cls = type("MockRB", (), {"close": lambda self: None})
    monkeypatch.setattr(
        "src.capture.process.SharedRingBuffer.attach", lambda layout, wi: mock_rb_cls()
    )

    proc.run()

    assert shutdown_event.is_set()
    assert capture_done.is_set()
    report = error_queue.get(timeout=1.0)
    assert "no camera" in report.summary
    ring_buffer.cleanup()


@pytest.mark.unit
def test_run_shutdown_event_exits_immediately(monkeypatch) -> None:
    """run() exits when shutdown_event is pre-set."""
    ring_buffer, write_index = SharedRingBuffer.create((4, 4, 3), num_slots=2)
    shutdown_event = mp.Event()
    shutdown_event.set()
    capture_done = mp.Event()

    proc = CaptureProcess(
        layout=ring_buffer.layout,
        write_index=write_index,
        source="0",
        camera_config=CameraConfig(width=4, height=4, fps=30),
        capture_done_event=capture_done,
        shutdown_event=shutdown_event,
        error_queue=mp.Queue(),
    )

    mock_cap = _FrameCapture([])
    monkeypatch.setattr(proc, "_open_capture", lambda: mock_cap)
    mock_rb_cls = type("MockRB", (), {"close": lambda self: None})
    monkeypatch.setattr(
        "src.capture.process.SharedRingBuffer.attach", lambda layout, wi: mock_rb_cls()
    )

    proc.run()

    assert capture_done.is_set()
    assert mock_cap.released
    ring_buffer.cleanup()
