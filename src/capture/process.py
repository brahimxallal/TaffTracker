from __future__ import annotations

import logging
import multiprocessing as mp
import time
from pathlib import Path
from traceback import format_exc

import cv2
import numpy as np

from src.capture.droidcam import (
    DroidCamDirectCaptureSource,
    DroidCamRuntimeConfig,
    apply_droidcam_startup_controls,
)
from src.capture.phone_mpeg import PhoneMpegCaptureSource, PhoneMpegRuntimeConfig
from src.capture.phone_yuv import PhoneCameraRuntimeConfig, PhoneYuvCaptureSource
from src.config import CameraConfig, DroidCamConfig, PhoneCameraConfig
from src.shared.ring_buffer import RingBufferLayout, SharedRingBuffer
from src.shared.types import ProcessErrorReport

LOGGER = logging.getLogger("capture")

_LIVE_READ_RETRY_SLEEP_S = 0.005
_LIVE_READ_REOPEN_THRESHOLD = 30
_LIVE_REOPEN_BACKOFF_S = 0.25
_OPEN_VALIDATION_ATTEMPTS = 8
_OPEN_VALIDATION_SLEEP_S = 0.02
_ENCODED_PHONE_BACKENDS = ("phone_mpeg", "phone_h264")


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
        gpu_preprocess: bool = False,
        phone_camera_config: PhoneCameraConfig | None = None,
        droidcam_config: DroidCamConfig | None = None,
    ) -> None:
        super().__init__(name="CaptureProcess")
        self._layout = layout
        self._write_index = write_index
        self._source = source
        self._camera_config = camera_config
        self._capture_done_event = capture_done_event
        self._shutdown_event = shutdown_event
        self._error_queue = error_queue
        self._gpu_preprocess = gpu_preprocess
        self._phone_camera_config = phone_camera_config or PhoneCameraConfig()
        self._droidcam_config = droidcam_config or DroidCamConfig()
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
            phone_source = self._camera_config.source_backend in (
                "phone_yuv",
                *_ENCODED_PHONE_BACKENDS,
                "droidcam",
            )
            source_is_file = (
                not phone_source and isinstance(source, str) and Path(source).exists()
            )
            source_is_stream = isinstance(source, str) and not source_is_file
            live_source = phone_source or isinstance(source, int) or source_is_stream
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
            consecutive_read_failures = 0
            first_read_failure_ns: int | None = None

            while not self._shutdown_event.is_set():
                if capture is None:
                    try:
                        capture = self._open_capture()
                        lb_ready = False
                        portrait_crop_w = 0
                        last_frame_shape = None
                        frames_since_size_check = 0
                        consecutive_read_failures = 0
                        first_read_failure_ns = None
                        LOGGER.info("Live capture recovered after reopening source")
                    except RuntimeError as exc:
                        LOGGER.warning("Live capture reopen failed: %s", exc)
                        self._shutdown_event.wait(_LIVE_REOPEN_BACKOFF_S)
                    continue

                if playback_interval_s is not None:
                    now_ns = time.perf_counter_ns()
                    if next_frame_deadline_ns > now_ns:
                        remaining_s = (next_frame_deadline_ns - now_ns) / 1_000_000_000.0
                        self._shutdown_event.wait(remaining_s)
                    next_frame_deadline_ns += int(playback_interval_s * 1_000_000_000.0)

                success, frame = capture.read()
                if not success or frame is None:
                    if live_source:
                        consecutive_read_failures += 1
                        if first_read_failure_ns is None:
                            first_read_failure_ns = time.perf_counter_ns()
                        if consecutive_read_failures >= _LIVE_READ_REOPEN_THRESHOLD:
                            stall_ms = (
                                time.perf_counter_ns() - first_read_failure_ns
                            ) / 1_000_000.0
                            LOGGER.warning(
                                "Live capture stalled for %.0f ms after %d failed reads; reopening source",
                                stall_ms,
                                consecutive_read_failures,
                            )
                            capture.release()
                            capture = None
                            self._shutdown_event.wait(_LIVE_REOPEN_BACKOFF_S)
                            continue
                        time.sleep(_LIVE_READ_RETRY_SLEEP_S)
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
                if consecutive_read_failures:
                    LOGGER.info(
                        "Live capture recovered after %d failed reads",
                        consecutive_read_failures,
                    )
                    consecutive_read_failures = 0
                    first_read_failure_ns = None

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

                if self._gpu_preprocess:
                    # GPU path: hand the raw frame to the letterbox kernel
                    # and copy the result back into the shared buffer. The
                    # CPU pre-allocated `resized_frame` is reused as the
                    # destination so the ring-buffer contract is unchanged.
                    from src.inference.gpu_preprocess import gpu_letterbox

                    out_tensor, _ = gpu_letterbox(frame, target_h, target_w)
                    np.copyto(resized_frame, out_tensor.cpu().numpy())
                elif lb_needs_resize:
                    scaled = cv2.resize(frame, (lb_new_w, lb_new_h), interpolation=cv2.INTER_LINEAR)
                    resized_frame[
                        lb_pad_top : lb_pad_top + lb_new_h,
                        lb_pad_left : lb_pad_left + lb_new_w,
                    ] = scaled
                elif lb_pad_top > 0 or lb_pad_left > 0:
                    resized_frame[
                        lb_pad_top : lb_pad_top + lb_new_h,
                        lb_pad_left : lb_pad_left + lb_new_w,
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

    def _open_capture(self):
        if self._camera_config.source_backend in ("opencv", "droidcam"):
            apply_droidcam_startup_controls(self._droidcam_config)

        if self._camera_config.source_backend == "droidcam":
            capture = DroidCamDirectCaptureSource(
                DroidCamRuntimeConfig.from_config(self._droidcam_config)
            )
            if not capture.isOpened():
                capture.release()
                raise RuntimeError("Failed to open DroidCam direct source")
            LOGGER.info(
                "Opened DroidCam direct source %s:%d format=%s",
                self._droidcam_config.host,
                self._droidcam_config.port,
                self._droidcam_config.video_format,
            )
            return capture

        if self._camera_config.source_backend == "phone_yuv":
            capture = PhoneYuvCaptureSource(self._build_phone_runtime_config())
            if not capture.isOpened():
                capture.release()
                raise RuntimeError("Failed to open TaffCam phone_yuv listener")
            LOGGER.info(
                "Opened TaffCam phone_yuv source on %s:%d",
                self._phone_camera_config.bind_host,
                self._phone_camera_config.frame_port,
            )
            return capture
        if self._camera_config.source_backend in _ENCODED_PHONE_BACKENDS:
            capture = PhoneMpegCaptureSource(self._build_phone_mpeg_runtime_config())
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(
                    f"Failed to open TaffCam {self._camera_config.source_backend} listener"
                )
            LOGGER.info(
                "Opened TaffCam %s source on %s:%d codec=%s",
                self._camera_config.source_backend,
                self._phone_camera_config.bind_host,
                self._phone_camera_config.frame_port,
                self._phone_camera_config.codec,
            )
            return capture

        source = self._resolve_source()
        backends = self._resolve_backends(source)
        last_error: str | None = None
        for backend in backends:
            capture = cv2.VideoCapture(source, backend)
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_BUFFERSIZE, self._camera_config.buffer_size)
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._camera_config.capture_width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._camera_config.capture_height)
                capture.set(cv2.CAP_PROP_FPS, self._camera_config.fps)

                if not self._capture_produces_frames(capture, source):
                    last_error = f"backend {backend} opened but produced no frames"
                    LOGGER.warning(
                        "Backend %s opened source %s but produced no frames; trying next backend",
                        backend,
                        self._source,
                    )
                    capture.release()
                    continue

                LOGGER.info("Opened source %s with backend %s", self._source, backend)

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

    def _build_phone_startup_controls(self) -> dict[str, object]:
        phone = self._phone_camera_config
        codec = "h264" if self._camera_config.source_backend == "phone_h264" else phone.codec
        startup_controls = {
            "camera_id": phone.camera_id,
            "width": phone.width,
            "height": phone.height,
            "fps": phone.fps,
            "stream_format": (
                "mpeg" if self._camera_config.source_backend in _ENCODED_PHONE_BACKENDS else "yuv"
            ),
            "pixel_format": phone.pixel_format,
            "codec": codec,
            "bitrate_bps": phone.bitrate_bps,
            "keyframe_interval_s": phone.keyframe_interval_s,
            "capture_mode": phone.capture_mode,
            "focus_diopters": phone.focus_diopters,
            "exposure_ns": phone.exposure_ns,
            "iso": phone.iso,
            "awb_enabled": phone.awb_enabled,
            "awb_lock": phone.awb_lock,
            "white_balance_kelvin": phone.white_balance_kelvin,
            "torch_enabled": phone.torch_enabled,
            "zoom_ratio": phone.zoom_ratio,
        }
        startup_controls = {
            key: value for key, value in startup_controls.items() if value is not None
        }
        return startup_controls

    def _build_phone_runtime_config(self) -> PhoneCameraRuntimeConfig:
        phone = self._phone_camera_config
        return PhoneCameraRuntimeConfig(
            frame_host=phone.bind_host,
            frame_port=phone.frame_port,
            control_host=phone.bind_host,
            control_port=phone.control_port,
            requested_width=phone.width,
            requested_height=phone.height,
            requested_fps=float(phone.fps),
            read_timeout_s=phone.read_timeout_s,
            control_timeout_s=phone.control_timeout_s,
            adb_reverse=phone.adb_reverse_enabled,
            adb_path=phone.adb_path,
            allow_remote_clients=phone.allow_remote_clients,
            startup_controls=self._build_phone_startup_controls(),
        )

    def _build_phone_mpeg_runtime_config(self) -> PhoneMpegRuntimeConfig:
        phone = self._phone_camera_config
        codec = "h264" if self._camera_config.source_backend == "phone_h264" else phone.codec
        return PhoneMpegRuntimeConfig(
            frame_host=phone.bind_host,
            frame_port=phone.frame_port,
            control_host=phone.bind_host,
            control_port=phone.control_port,
            requested_width=phone.width,
            requested_height=phone.height,
            requested_fps=float(phone.fps),
            codec=codec,
            bitrate_bps=phone.bitrate_bps,
            keyframe_interval_s=phone.keyframe_interval_s,
            decode_backend=phone.decode_backend,
            read_timeout_s=phone.read_timeout_s,
            control_timeout_s=phone.control_timeout_s,
            adb_reverse=phone.adb_reverse_enabled,
            adb_path=phone.adb_path,
            allow_remote_clients=phone.allow_remote_clients,
            startup_controls=self._build_phone_startup_controls(),
        )

    def _capture_produces_frames(self, capture, source: int | str) -> bool:
        if (
            self._camera_config.source_backend == "opencv"
            and isinstance(source, str)
            and Path(source).exists()
        ):
            return True
        for _ in range(_OPEN_VALIDATION_ATTEMPTS):
            success, frame = capture.read()
            if success and frame is not None:
                return True
            time.sleep(_OPEN_VALIDATION_SLEEP_S)
        return False

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
