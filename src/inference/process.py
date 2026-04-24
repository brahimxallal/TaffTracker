from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import time
from collections import deque
from multiprocessing.sharedctypes import Synchronized
from queue import Empty, Full
from time import perf_counter_ns
from traceback import format_exc

from src.calibration.camera_model import CameraModel
from src.config import (
    CameraConfig,
    LaserConfig,
    Mode,
    ModelConfig,
    PreflightConfig,
    RuntimePaths,
    TargetKind,
    TrackingConfig,
)
from src.inference.pipeline import TrackingPipeline
from src.inference.postprocess import (
    KeypointStabilizer,
    parse_yolo_output,
)
from src.inference.stages.centroid import CentroidStage
from src.inference.stages.servo import ServoStage
from src.inference.stages.tracker import TrackerStage
from src.inference.telemetry import format_profiler_summary, write_profiler_summary
from src.inference.trt_engine import TRTEngine
from src.shared.pose_schema import PoseSchema, get_pose_schema
from src.shared.profiler import StageProfiler
from src.shared.ring_buffer import RingBufferLayout, SharedRingBuffer
from src.shared.types import ProcessErrorReport, TrackingMessage
from src.tracking.adaptive import AdaptiveController
from src.tracking.botsort import BoTSORT
from src.tracking.kalman import KalmanFilter
from src.tracking.reid import ReIDBuffer

LOGGER = logging.getLogger("inference")


class InferenceProcess(mp.Process):
    def __init__(
        self,
        layout: RingBufferLayout,
        write_index,
        result_queue: mp.Queue,
        capture_done_event: mp.synchronize.Event,
        shutdown_event: mp.synchronize.Event,
        error_queue: mp.Queue,
        mode: Mode,
        target: TargetKind,
        camera_config: CameraConfig,
        tracking_config: TrackingConfig,
        model_config: ModelConfig,
        runtime_paths: RuntimePaths,
        laser_config: LaserConfig | None = None,
        preflight_config: PreflightConfig | None = None,
        profile: bool = False,
        relock_event: mp.synchronize.Event | None = None,
        cycle_target_event: mp.synchronize.Event | None = None,
        command_pan: Synchronized | None = None,
        command_tilt: Synchronized | None = None,
    ) -> None:
        super().__init__(name="InferenceProcess")
        self._layout = layout
        self._write_index = write_index
        self._result_queue = result_queue
        self._capture_done_event = capture_done_event
        self._shutdown_event = shutdown_event
        self._error_queue = error_queue
        self._mode = mode
        self._target = target
        self._camera_config = camera_config
        self._tracking_config = tracking_config
        self._model_config = model_config
        self._runtime_paths = runtime_paths
        self._laser_config = laser_config
        self._preflight_config = preflight_config or PreflightConfig()
        self._profile = profile
        self._relock_event = relock_event
        self._cycle_target_event = cycle_target_event
        self._command_pan = command_pan
        self._command_tilt = command_tilt
        self._frames_processed = 0
        self._last_log_time = time.perf_counter()
        self._fps_window: deque[float] = deque(maxlen=30)
        self._publish_drop_count = 0
        self._publish_drop_window_count = 0
        self._publish_drop_window_start = time.perf_counter()

    def run(self) -> None:
        import sys

        def _excepthook(etype, value, tb):
            import traceback

            LOGGER.error(
                "Uncaught exception in InferenceProcess: %s",
                "".join(traceback.format_exception(etype, value, tb)),
            )

        sys.excepthook = _excepthook
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(processName)s %(levelname)s %(message)s",
        )

        # ── Component construction (child process) ──
        ring_buffer = SharedRingBuffer.attach(self._layout, self._write_index)
        engine = TRTEngine(self._resolve_engine_path())
        camera_model = self._load_camera_model()
        pose_schema = self._load_pose_schema()
        num_keypoints = pose_schema.keypoint_count
        profiler = StageProfiler(window_size=128, enabled=self._profile)

        tracker = BoTSORT(
            track_thresh=self._tracking_config.tracker_track_threshold,
            match_thresh=self._tracking_config.tracker_match_threshold,
            max_lost=self._tracking_config.max_lost_frames,
            birth_min_hits=self._tracking_config.tracker_birth_min_hits,
        )
        kalman = KalmanFilter(
            process_noise=self._tracking_config.process_noise,
            measurement_noise=self._tracking_config.measurement_noise,
            config=self._tracking_config.kalman,
            fps_ratio=self._camera_config.fps / 60.0,
        )
        stabilizer = KeypointStabilizer(alpha=self._tracking_config.postprocess.conf_ema_alpha)
        adaptive = AdaptiveController(self._tracking_config)
        reid_spatial = 150.0 if self._target == "dog" else 200.0
        reid_buffer = ReIDBuffer(max_spatial_distance_px=reid_spatial)

        from src.tracking.one_euro import OneEuroFilter2D

        sm = self._tracking_config.smoothing
        ema_pixel = OneEuroFilter2D(
            mincutoff=sm.display_mincutoff, beta=sm.display_beta, dcutoff=sm.dcutoff
        )
        servo_ema_pixel = OneEuroFilter2D(
            mincutoff=sm.servo_mincutoff, beta=sm.servo_beta, dcutoff=sm.dcutoff
        )

        # Laser detector (overlay only — no closed-loop PID)
        laser_detector = None
        if self._laser_config is not None and self._laser_config.enabled:
            from src.laser.detector import LaserDetector

            laser_detector = LaserDetector(self._laser_config)
            LOGGER.info(
                "Laser detector enabled (HSV red, ROI=%.0fpx)", self._laser_config.roi_radius_px
            )

        # ── Assemble stages and pipeline ──
        tracker_stage = TrackerStage(
            tracker=tracker,
            kalman=kalman,
            stabilizer=stabilizer,
            reid_buffer=reid_buffer,
            max_lost_frames=self._tracking_config.max_lost_frames,
        )
        centroid_stage = CentroidStage(
            camera_model=camera_model,
            target=self._target,
        )
        servo_stage = ServoStage(
            laser_detector=laser_detector,
            laser_roi_radius=self._laser_config.roi_radius_px if self._laser_config else 150.0,
        )
        pipeline = TrackingPipeline(
            tracker_stage=tracker_stage,
            centroid_stage=centroid_stage,
            servo_stage=servo_stage,
            adaptive=adaptive,
            tracking_config=self._tracking_config,
            pose_schema=pose_schema,
            ema_pixel=ema_pixel,
            servo_ema_pixel=servo_ema_pixel,
        )

        # ── Loop state ──
        last_frame_id = 0
        last_timestamp_ns: int | None = None
        was_lost = False
        prev_locked_id: int | None = None
        prev_locked_bbox: tuple[float, ...] | None = None

        from src.shared.preflight import FrameHealthMonitor

        health_monitor = FrameHealthMonitor(self._preflight_config)
        if self._preflight_config.enabled:
            LOGGER.info(
                "Preflight health monitor enabled (window=%d)", self._preflight_config.window_size
            )

        frame_drop_count = 0
        frame_total_count = 0
        frames_since_gc = 0
        # Rolling drop-rate telemetry: flushes every _DROP_WINDOW frames.
        _DROP_WINDOW = 120
        _FRAME_TIMEOUT_NS = int(3.0 / max(self._camera_config.fps, 1) * 1_000_000_000)
        _last_frame_received_ns = perf_counter_ns()

        gc.collect()
        gc.disable()

        try:
            while not self._shutdown_event.is_set():
                record = ring_buffer.read_latest(after_frame_id=last_frame_id, copy=False)
                if record is None:
                    if self._capture_done_event.is_set():
                        break
                    if (perf_counter_ns() - _last_frame_received_ns) > _FRAME_TIMEOUT_NS:
                        LOGGER.warning(
                            "No frame received for %.1f ms — capture may be stalled",
                            _FRAME_TIMEOUT_NS / 1_000_000.0,
                        )
                        if self._capture_done_event.wait(timeout=1.0):
                            break
                        _last_frame_received_ns = perf_counter_ns()
                    time.sleep(0.0005)
                    continue
                _last_frame_received_ns = perf_counter_ns()

                # Frame drop tracking
                if last_frame_id > 0:
                    skipped = record.frame_id - last_frame_id
                    if skipped > 1:
                        frame_drop_count += skipped - 1
                frame_total_count += 1
                if frame_total_count % _DROP_WINDOW == 0:
                    profiler.add_sample("frame_drops", frame_drop_count)
                    frame_drop_count = 0
                    frame_total_count = 0

                last_frame_id = record.frame_id
                dt = self._compute_dt(record.timestamp_ns, last_timestamp_ns)
                last_timestamp_ns = record.timestamp_ns
                centroid_stage.update_commanded_camera_motion(
                    record.timestamp_ns,
                    self._command_pan,
                    self._command_tilt,
                )

                # Check relock signal
                if self._relock_event is not None and self._relock_event.is_set():
                    self._relock_event.clear()
                    tracker_stage.request_relock()
                    prev_locked_id = None
                    LOGGER.info("Relock requested by user — releasing target lock")

                # Check cycle target signal
                if self._cycle_target_event is not None and self._cycle_target_event.is_set():
                    self._cycle_target_event.clear()
                    tracker_stage.request_cycle()
                    LOGGER.info("Cycle target requested by user")

                undistorted = camera_model.undistort(record.frame)

                for warning in health_monitor.check(undistorted):
                    LOGGER.warning(warning)

                wait_ms = (perf_counter_ns() - record.timestamp_ns) / 1_000_000.0
                profiler.add_sample("wait", int(wait_ms * 1_000_000))

                inference_start_ns = perf_counter_ns()
                with profiler.stage("inference"):
                    raw_output = engine.infer(undistorted)
                inference_ms = (perf_counter_ns() - inference_start_ns) / 1_000_000.0

                # Rolling FPS
                now_s = time.perf_counter()
                self._fps_window.append(now_s)
                fps_val = 0.0
                if len(self._fps_window) > 1:
                    fps_val = (len(self._fps_window) - 1) / (
                        self._fps_window[-1] - self._fps_window[0]
                    )

                with profiler.stage("tracking"):
                    postprocess_start_ns = perf_counter_ns()
                    detections = parse_yolo_output(
                        raw_output,
                        conf_threshold=adaptive.confidence_threshold,
                        num_keypoints=num_keypoints,
                    )
                    tracks = tracker.update(
                        detections,
                        timestamp_ns=record.timestamp_ns,
                        frame=record.frame,
                        angular_velocity=centroid_stage.last_camera_angular_velocity,
                    )
                    postprocess_ms = (perf_counter_ns() - postprocess_start_ns) / 1_000_000.0
                    profiler.add_sample("postprocess", int(postprocess_ms * 1_000_000))

                    message, was_lost, current_locked, prev_locked_bbox = pipeline.process_frame(
                        record=record,
                        undistorted=undistorted,
                        tracks=tracks,
                        prev_locked_id=prev_locked_id,
                        was_lost=was_lost,
                        dt=dt,
                        fps=fps_val,
                        wait_ms=wait_ms,
                        inference_ms=inference_ms,
                        postprocess_ms=postprocess_ms,
                    )
                    prev_locked_id = current_locked

                self._publish(message)
                profiler.add_sample("total_latency", int(message.total_latency_ms * 1_000_000))
                self._frames_processed += 1
                frames_since_gc += 1
                if self._frames_processed % 300 == 0:
                    current_time = time.perf_counter()
                    elapsed = current_time - self._last_log_time
                    fps = self._frames_processed / elapsed if elapsed > 0 else 0
                    self._log_profiler_summary(profiler, pipeline, fps)
                    self._frames_processed = 0
                    self._last_log_time = current_time
                if frames_since_gc >= 5000:
                    gc.collect()
                    frames_since_gc = 0
        except BaseException as exc:
            LOGGER.exception("Inference process failed")
            self._report_error(exc)
            self._shutdown_event.set()
        finally:
            gc.enable()
            self._publish(None)
            self._write_profiler_summary(profiler, pipeline)
            engine.close()
            ring_buffer.close()

    # ── Helpers (kept in process shell) ──

    def _resolve_engine_path(self):
        return (
            self._model_config.person_engine_path
            if self._target == "human"
            else self._model_config.dog_engine_path
        )

    def _load_camera_model(self) -> CameraModel:
        calibration_path = self._runtime_paths.calibration_file_path()
        if calibration_path.exists():
            try:
                model = CameraModel.load(calibration_path)
            except Exception as exc:
                LOGGER.warning(
                    "Failed to load camera calibration from %s: %s",
                    calibration_path,
                    exc,
                )
            else:
                if model.image_size == (self._camera_config.width, self._camera_config.height):
                    LOGGER.info("Using camera calibration from %s", calibration_path)
                    return model
                LOGGER.warning(
                    "Calibration image size %s does not match runtime size (%d, %d); falling back",
                    model.image_size,
                    self._camera_config.width,
                    self._camera_config.height,
                )
        if self._camera_config.fov is not None:
            LOGGER.info("Using configured FOV %.1f deg", self._camera_config.fov)
            return CameraModel.from_fov(
                self._camera_config.fov, self._camera_config.width, self._camera_config.height
            )
        if self._mode == "camera":
            raise ValueError(
                "camera.fov must be set or calibration_data/intrinsics.npz must exist for camera mode"
            )
        LOGGER.warning("No FOV configured, falling back to identity camera model")
        return CameraModel.identity(self._camera_config.width, self._camera_config.height)

    def _load_pose_schema(self) -> PoseSchema:
        schema_path = (
            self._runtime_paths.resolved_dog_pose_schema_path() if self._target == "dog" else None
        )
        pose_schema = get_pose_schema(self._target, schema_path)
        LOGGER.info(
            "Using %s pose schema from %s with %s keypoints",
            pose_schema.target_kind,
            pose_schema.source,
            pose_schema.keypoint_count,
        )
        return pose_schema

    def _compute_dt(self, current_timestamp_ns: int, last_timestamp_ns: int | None) -> float:
        if last_timestamp_ns is None:
            return 1.0 / max(self._camera_config.fps, 1)
        return max((current_timestamp_ns - last_timestamp_ns) / 1_000_000_000.0, 1e-3)

    def _publish(self, message: TrackingMessage | None) -> None:
        try:
            self._result_queue.put_nowait(message)
        except Full:
            self._publish_drop_count += 1
            self._publish_drop_window_count += 1
            now = time.perf_counter()
            if now - self._publish_drop_window_start >= 1.0:
                LOGGER.warning(
                    "Inference result queue backpressure: dropped %d messages in the last %.1fs",
                    self._publish_drop_window_count,
                    now - self._publish_drop_window_start,
                )
                self._publish_drop_window_count = 0
                self._publish_drop_window_start = now
            try:
                self._result_queue.get_nowait()
            except Empty:
                pass
            try:
                self._result_queue.put_nowait(message)
            except Full:
                pass

    def _log_profiler_summary(
        self, profiler: StageProfiler, pipeline: TrackingPipeline, fps: float
    ) -> None:
        line = format_profiler_summary(
            profiler,
            pipeline,
            fps=fps,
            publish_drop_count=self._publish_drop_count,
        )
        if line is not None:
            LOGGER.info("Profiler summary | %s", line)

    def _write_profiler_summary(self, profiler: StageProfiler, pipeline: TrackingPipeline) -> None:
        csv_path = self._runtime_paths.profiler_csv_path(self._mode, self._target)
        summary_path = self._runtime_paths.profiler_summary_path()
        metrics_path = self._runtime_paths.inference_metrics_path()
        write_profiler_summary(
            profiler,
            pipeline,
            csv_path=csv_path,
            summary_path=summary_path,
            metrics_path=metrics_path,
            publish_drop_count=self._publish_drop_count,
        )
        LOGGER.info("Profiler summaries saved to %s and %s", csv_path, summary_path)

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
            LOGGER.error("Failed to publish inference error report")
