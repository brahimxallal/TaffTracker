from __future__ import annotations

from pathlib import Path
import logging
import multiprocessing as mp
import time
from traceback import format_exc

import cv2
import numpy as np

from src.config import CameraConfig
from src.shared.ring_buffer import RingBufferLayout, SharedRingBuffer
from src.shared.types import ProcessErrorReport


LOGGER = logging.getLogger("capture")


class CaptureProcess(mp.Process):
    def __init__(
        self,
        layout: RingBufferLayout,
        write_index,
        source: str,
        camera_config: CameraConfig,
        capture_done_event: mp.synchronize.Event,
        shutdown_event: mp.synchronize.Event,
        error_queue: mp.Queue,
    ) -> None:
        super().__init__(name="CaptureProcess")
        self._layout = layout
        self._write_index = write_index
        self._source = source
        self._camera_config = camera_config
        self._capture_done_event = capture_done_event
        self._shutdown_event = shutdown_event
        self._error_queue = error_queue
        self._frame_count = 0
        self._last_log_time = time.perf_counter()

    def run(self) -> None:
        import sys

        def _excepthook(etype, value, tb):
            import traceback

            LOGGER.error(
                "Uncaught exception in CaptureProcess: %s",
                "".join(traceback.format_exception(etype, value, tb)),
            )

        sys.excepthook = _excepthook
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(processName)s %(levelname)s %(message)s",
        )
        ring_buffer = SharedRingBuffer.attach(self._layout, self._write_index)
        capture = None
        try:
            source = self._resolve_source()
            capture = self._open_capture()
            source_is_file = isinstance(source, str) and Path(source).exists()
            source_is_stream = isinstance(source, str) and not source_is_file
            playback_interval_s = self._resolve_playback_interval(capture, source_is_file)
            next_frame_deadline_ns = time.perf_counter_ns()

            # Pre-allocate frame buffer with YOLO letterbox gray (114)
            target_h = self._camera_config.height
            target_w = self._camera_config.width
            resized_frame = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
            lb_ready = False
            lb_pad_top = 0
            lb_pad_left = 0
            lb_new_w = 0
            lb_new_h = 0
            lb_needs_resize = False
            portrait_crop_x = 0
            portrait_crop_w = 0
            last_frame_shape: tuple[int, int] | None = None
            frames_since_size_check = 0

            while not self._shutdown_event.is_set():
                if playback_interval_s is not None:
                    now_ns = time.perf_counter_ns()
                    if next_frame_deadline_ns > now_ns:
                        remaining_s = (next_frame_deadline_ns - now_ns) / 1_000_000_000.0
                        self._shutdown_event.wait(remaining_s)
                    next_frame_deadline_ns += int(playback_interval_s * 1_000_000_000.0)

                success, frame = capture.read()
                if not success:
                    if isinstance(source, int) or source_is_stream:
                        time.sleep(0.005)
                        continue
                    if source_is_file:
                        replacement = self._rewind_file_capture(capture)
                        if replacement is None:
                            LOGGER.error("Failed to rewind file source; stopping capture loop")
                            self._shutdown_event.set()
                            break
                        capture = replacement
                        next_frame_deadline_ns = time.perf_counter_ns()
                        LOGGER.info("Looping file source from beginning")
                        continue
                    LOGGER.info("Capture source exhausted; stopping capture loop")
                    self._shutdown_event.set()
                    break

                # Resolution watcher: if the DroidCam user changes resolution
                # mid-session the shape changes, so we force a letterbox recompute.
                frames_since_size_check += 1
                if frames_since_size_check >= 60:
                    frames_since_size_check = 0
                    current_shape = frame.shape[:2]
                    if last_frame_shape is not None and current_shape != last_frame_shape:
                        LOGGER.info(
                            "Capture resolution changed %s -> %s; recomputing letterbox",
                            last_frame_shape,
                            current_shape,
                        )
                        lb_ready = False
                        portrait_crop_w = 0
                    last_frame_shape = current_shape

                # Letterbox: scale to fit + pad (preserves aspect ratio)
                if not lb_ready:
                    h_orig, w_orig = frame.shape[:2]
                    last_frame_shape = (h_orig, w_orig)

                    if self._camera_config.portrait_mode and w_orig > h_orig:
                        # Extract the inner vertical slice from the landscape padding
                        actual_aspect = w_orig / h_orig
                        active_w = int(h_orig * (1.0 / actual_aspect))
                        portrait_crop_x = (w_orig - active_w) // 2
                        portrait_crop_w = active_w
                        w_orig = active_w
                    else:
                        portrait_crop_x = 0
                        portrait_crop_w = 0

                    scale = min(target_w / w_orig, target_h / h_orig)
                    lb_new_w = int(round(w_orig * scale))
                    lb_new_h = int(round(h_orig * scale))
                    lb_pad_left = (target_w - lb_new_w) // 2
                    lb_pad_top = (target_h - lb_new_h) // 2
                    lb_needs_resize = lb_new_w != w_orig or lb_new_h != h_orig
                    lb_ready = True
                    LOGGER.info(
                        "Letterbox: %dx%d -> %dx%d in %dx%d (pad_top=%d pad_left=%d scale=%.3f)",
                        w_orig,
                        h_orig,
                        lb_new_w,
                        lb_new_h,
                        target_w,
                        target_h,
                        lb_pad_top,
                        lb_pad_left,
                        scale,
                    )

                if portrait_crop_w > 0:
                    frame = frame[:, portrait_crop_x : portrait_crop_x + portrait_crop_w]

                if lb_needs_resize:
                    scaled = cv2.resize(frame, (lb_new_w, lb_new_h), interpolation=cv2.INTER_LINEAR)
                    resized_frame[
                        lb_pad_top : lb_pad_top + lb_new_h, lb_pad_left : lb_pad_left + lb_new_w
                    ] = scaled
                elif lb_pad_top > 0 or lb_pad_left > 0:
                    resized_frame[
                        lb_pad_top : lb_pad_top + lb_new_h, lb_pad_left : lb_pad_left + lb_new_w
                    ] = frame
                else:
                    np.copyto(resized_frame, frame)
                ring_buffer.write(resized_frame, time.perf_counter_ns())

                # Performance monitoring
                self._frame_count += 1
                if self._frame_count % 300 == 0:  # Log every 300 frames (~5 seconds at 60fps)
                    current_time = time.perf_counter()
                    elapsed = current_time - self._last_log_time
                    fps = self._frame_count / elapsed if elapsed > 0 else 0
                    LOGGER.info("Capture FPS: %.1f (frames: %d)", fps, self._frame_count)
                    self._frame_count = 0
                    self._last_log_time = current_time
        except BaseException as exc:
            LOGGER.exception("Capture process failed")
            self._report_error(exc)
            self._shutdown_event.set()
        finally:
            self._capture_done_event.set()
            if capture is not None:
                capture.release()
            ring_buffer.close()

    def _open_capture(self) -> cv2.VideoCapture:
        source = self._resolve_source()
        backends = self._resolve_backends(source)
        last_error: str | None = None
        for backend in backends:
            capture = cv2.VideoCapture(source, backend)
            if capture.isOpened():
                LOGGER.info("Opened source %s with backend %s", self._source, backend)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, self._camera_config.buffer_size)
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._camera_config.capture_width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._camera_config.capture_height)
                capture.set(cv2.CAP_PROP_FPS, self._camera_config.fps)

                # Read back actual camera properties
                actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = capture.get(cv2.CAP_PROP_FPS)
                backend_name = capture.getBackendName()
                LOGGER.info(
                    "camera: actual %dx%d @ %.1f fps (backend=%s)",
                    actual_w,
                    actual_h,
                    actual_fps,
                    backend_name,
                )

                # Warn if actual resolution/fps deviates >5% from requested
                req_w = self._camera_config.capture_width
                req_h = self._camera_config.capture_height
                req_fps = self._camera_config.fps
                deviations: list[str] = []
                if req_w > 0 and abs(actual_w - req_w) / req_w > 0.05:
                    deviations.append(f"width {actual_w} vs requested {req_w}")
                if req_h > 0 and abs(actual_h - req_h) / req_h > 0.05:
                    deviations.append(f"height {actual_h} vs requested {req_h}")
                if req_fps > 0 and abs(actual_fps - req_fps) / req_fps > 0.05:
                    deviations.append(f"fps {actual_fps:.1f} vs requested {req_fps}")
                if deviations:
                    LOGGER.warning(
                        "Camera property mismatch (>5%% from request): %s; letterbox will adapt",
                        "; ".join(deviations),
                    )

                return capture
            capture.release()
            last_error = f"backend {backend} failed"
            LOGGER.debug("Backend %s could not open source %s", backend, self._source)
        raise RuntimeError(f"Failed to open capture source: {self._source} ({last_error})")

    def _resolve_playback_interval(
        self, capture: cv2.VideoCapture, source_is_file: bool
    ) -> float | None:
        if not source_is_file:
            return None
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if source_fps <= 0.0:
            source_fps = float(self._camera_config.fps)
        return 1.0 / max(source_fps, 1.0)

    def _rewind_file_capture(self, capture: cv2.VideoCapture) -> cv2.VideoCapture | None:
        if capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
            return capture

        capture.release()
        replacement = self._open_capture()
        if replacement.isOpened():
            return replacement
        replacement.release()
        return None

    def _resolve_source(self) -> int | str:
        source = self._source.strip()
        if source.isdigit():
            return int(source)
        return str(Path(source)) if Path(source).exists() else source

    def _report_error(self, exc: BaseException) -> None:
        report = ProcessErrorReport(
            process_name=self.name,
            summary=str(exc) or exc.__class__.__name__,
            traceback_text=format_exc(),
            timestamp_ns=time.perf_counter_ns(),
            severity="error",
        )
        try:
            self._error_queue.put_nowait(report)
        except Exception:
            LOGGER.error("Failed to publish capture error report")

    def _resolve_backends(self, source: int | str) -> list[int]:
        explicit = {
            "dshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF,
            "ffmpeg": cv2.CAP_FFMPEG,
        }
        if self._camera_config.backend in explicit:
            return [explicit[self._camera_config.backend]]
        if isinstance(source, int):
            return [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
        return [cv2.CAP_FFMPEG, cv2.CAP_ANY]
